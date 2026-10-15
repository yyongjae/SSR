"""CK2 raw-anchor sampler tests (CPU): determinism, group sizes / disjoint strata, no duplicates, exact inclusion
probabilities (Sum w = 256, Monte-Carlo inclusion frequency, unbiased HT), fallback, v2 WTA metric parity,
label-source fallback (EP masked), split priors / offsets on a fake split; optional real-data checks.

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_anchor_sampler.py
"""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from navsim.agents.para_ssr.ck import anchor_sampler as AS
from navsim.agents.para_ssr.ck import constants as C


# ----------------------------------------------------------------------------------------------- fixtures
def _anchors(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(1, 9) * 0.5
    v = rng.uniform(0, 15, size=(256, 1))
    k = rng.normal(scale=0.03, size=(256, 1))
    x = v * t
    y = 0.5 * k * x ** 2
    h = np.arctan(k * x)
    return np.stack([x, y, h], -1).astype(np.float32)


def _labels(rng, p_pass=0.25, n=256):
    lab = np.ones((5, n), np.float32)
    fail = rng.uniform(size=n) > p_pass
    key = rng.choice([0, 1, 3, 4], size=n)
    lab[key[fail], np.flatnonzero(fail)] = 0.0
    lab[0, rng.uniform(size=n) < 0.02] = 0.5          # NC soft label -> fail
    lab[2] = rng.uniform(size=n)                       # EP
    return lab


def _gt(rng):
    return (_anchors(1)[rng.integers(256)] + rng.normal(scale=0.3, size=(8, 3))).astype(np.float32)


@pytest.fixture
def sampler():
    return AS.AnchorSampler(anchors=_anchors(), seed=7)


# ----------------------------------------------------------------------------------------------- determinism
def test_determinism(sampler):
    rng = np.random.default_rng(0)
    lab, gt = _labels(rng), _gt(rng)
    a = sampler.sample("tokA", gt, lab, epoch=3)
    b = AS.AnchorSampler(anchors=_anchors(), seed=7).sample("tokA", gt.copy(), lab.copy(), epoch=3)
    for k in ("idx", "group", "pass_mask", "pi", "w", "rank", "dist"):
        assert np.array_equal(a[k], b[k]), k
    assert a["info"] == b["info"]
    c = sampler.sample("tokA", gt, lab, epoch=4)
    d = AS.AnchorSampler(anchors=_anchors(), seed=8).sample("tokA", gt, lab, epoch=3)
    e = sampler.sample("tokB", gt, lab, epoch=3)
    near_a, near_c = a["idx"][a["group"] == AS.GROUP_NEAR], c["idx"][c["group"] == AS.GROUP_NEAR]
    assert np.array_equal(np.sort(near_a), np.sort(near_c))          # near is epoch independent
    for o in (c, d, e):
        assert not np.array_equal(np.sort(a["idx"]), np.sort(o["idx"]))
    # stream is keyed by (seed, epoch, token) only: not by call order
    s2 = AS.AnchorSampler(anchors=_anchors(), seed=7)
    s2.sample("other", gt, lab, epoch=0)
    assert np.array_equal(s2.sample("tokA", gt, lab, epoch=3)["idx"], a["idx"])


def test_shuffle_slot_order(sampler):
    """shuffle=True (default): same drawn set / pi / w as the canonical order, all per-slot arrays permuted together,
    deterministic, and the slot position does not encode GT rank or the strat pass / fail label."""
    rng = np.random.default_rng(10)
    lab, gt = _labels(rng, p_pass=0.5), _gt(rng)
    keys = ("idx", "group", "pass_mask", "pi", "w", "rank", "dist")
    pos_near0, pos_pass = np.zeros(32), np.zeros(32)
    for ep in range(400):
        a = sampler.sample("sh", gt, lab, epoch=ep)
        b = sampler.sample("sh", gt, lab, epoch=ep, shuffle=False)
        o = np.argsort(a["idx"])
        q = np.argsort(b["idx"])
        for k in keys:
            assert np.array_equal(a[k][o], b[k][q]), k                  # same rows, only the slot order differs
        assert a["info"] == b["info"]
        assert np.array_equal(a["idx"], sampler.sample("sh", gt, lab, epoch=ep)["idx"])   # deterministic
        pos_near0[np.flatnonzero(a["rank"] == 0)[0]] += 1
        pos_pass += (a["group"] == AS.GROUP_STRAT) & a["pass_mask"]
    assert pos_near0.max() < 40 and (pos_near0 > 0).sum() > 25        # WTA winner spread over the 32 slots
    tot = pos_pass.sum()                                              # canonical order: all in slots 16..23
    assert pos_pass[:16].sum() > 0.35 * tot and pos_pass[24:].sum() > 0.15 * tot


# ----------------------------------------------------------------------------------------------- structure
def test_group_sizes_no_duplicates_and_strata(sampler):
    rng = np.random.default_rng(1)
    for i in range(50):
        lab, gt = _labels(rng, p_pass=0.5), _gt(rng)
        s = sampler.sample(f"t{i}", gt, lab, epoch=i, shuffle=False)
        assert s["idx"].shape == (32,) and len(np.unique(s["idx"])) == 32
        assert [(s["group"] == g).sum() for g in range(3)] == [8, 8, 16]
        dist = AS.gt_distance(sampler.anchors, gt)
        order = np.argsort(dist, kind="stable")
        assert np.array_equal(s["idx"][:8], order[:8])                  # near = 8 nearest, rank order
        assert np.array_equal(s["rank"][:8], np.arange(8))
        r = s["rank"]
        assert ((r[s["group"] == 1] >= 8) & (r[s["group"] == 1] < 72)).all()
        assert (r[s["group"] == 2] >= 72).all()                         # strat only from the far region
        assert np.allclose(s["dist"], dist[s["idx"]], atol=1e-5)
        assert np.array_equal(s["pass_mask"], AS.pass_mask(lab)[s["idx"]])
        if s["info"]["fallback"] == "none":
            st = s["group"] == 2
            assert s["pass_mask"][st].sum() == 8 and (~s["pass_mask"][st]).sum() == 8


def test_pass_definition():
    lab = np.ones((5, 4), np.float32)
    lab[0, 1] = 0.5            # NC 0.5 -> fail
    lab[2, 2] = 0.0            # EP 0 does not matter
    lab[4, 3] = 0.0            # comfort 0 -> fail
    assert AS.pass_mask(lab).tolist() == [True, False, True, False]
    lab[1, 0] = np.nan
    with pytest.raises(ValueError):
        AS.pass_mask(lab)


# ----------------------------------------------------------------------------------------------- weights
def test_weights_sum_and_exact_pi(sampler):
    rng = np.random.default_rng(2)
    for i in range(30):
        lab, gt = _labels(rng, p_pass=rng.uniform(0.02, 0.98)), _gt(rng)
        d = sampler.design(gt, lab)
        assert np.isclose(d["pi"].sum(), 32.0) and (d["pi"] > 0).all() and (d["pi"] <= 1).all()
        s = sampler.sample(f"t{i}", gt, lab, epoch=0)
        assert np.isclose(s["w"].astype(np.float64).sum(), 256.0, atol=1e-3)
        assert np.allclose(s["w"], 1.0 / s["pi"]) and np.array_equal(s["pi"], d["pi"][s["idx"]])


def test_monte_carlo_inclusion_and_unbiased_ht(sampler):
    rng = np.random.default_rng(3)
    lab, gt = _labels(rng, p_pass=0.3), _gt(rng)
    d = sampler.design(gt, lab)
    n_ep = 3000
    cnt = np.zeros(256)
    ht = []
    for ep in range(n_ep):
        s = sampler.sample("mc", gt, lab, epoch=ep)
        cnt[s["idx"]] += 1
        ht.append(float((s["w"].astype(np.float64) * s["pass_mask"]).sum()))
    freq = cnt / n_ep
    se = np.sqrt(d["pi"] * (1 - d["pi"]) / n_ep) + 1e-9
    assert (np.abs(freq - d["pi"]) <= 5 * se + 1e-12).all()
    true = d["pass"].sum()
    assert abs(np.mean(ht) - true) < 5 * np.std(ht) / np.sqrt(n_ep) + 1e-9


# ----------------------------------------------------------------------------------------------- fallback
@pytest.mark.parametrize("n_far_pass,expect", [(0, "pass_short"), (3, "pass_short"), (8, "none"), (181, "fail_short"),
                                               (184, "fail_short")])
def test_fallback(sampler, n_far_pass, expect):
    rng = np.random.default_rng(4)
    gt = _gt(rng)
    order = np.argsort(AS.gt_distance(sampler.anchors, gt), kind="stable")
    lab = np.ones((5, 256), np.float32)
    far = order[72:]
    lab[1, far[n_far_pass:]] = 0.0                     # DAC fail on all far anchors but n_far_pass
    s = sampler.sample("fb", gt, lab, epoch=0)
    inf = s["info"]
    assert inf["fallback"] == expect and inf["n_pass_far"] == n_far_pass and inf["n_fail_far"] == 184 - n_far_pass
    st = s["group"] == 2
    assert st.sum() == 16 and len(np.unique(s["idx"])) == 32
    tp, tf = int(s["pass_mask"][st].sum()), int((~s["pass_mask"][st]).sum())
    assert (tp, tf) == (inf["n_pass_taken"], inf["n_fail_taken"])
    if expect == "pass_short":
        assert tp == n_far_pass and tf == 16 - n_far_pass
        if n_far_pass:
            assert np.allclose(s["pi"][st & s["pass_mask"]], 1.0)
    if expect == "fail_short":
        assert tf == 184 - n_far_pass and tp == 16 - tf
    assert np.isclose(s["w"].astype(np.float64).sum(), 256.0, atol=1e-3)


def test_config_validation():
    with pytest.raises(ValueError):
        AS.AnchorSampler(anchors=_anchors(), K=30)
    with pytest.raises(ValueError):
        AS.AnchorSampler(anchors=_anchors(), K=32, n_near=8, n_mid=8, mid_pool=240, n_strat=16)
    s = AS.AnchorSampler(anchors=_anchors(), K=17, n_near=4, n_mid=4, mid_pool=32, n_strat=9)
    assert (s.n_strat_pass, s.n_strat_fail) == (4, 5)
    rng = np.random.default_rng(5)
    out = s.sample("odd", _gt(rng), _labels(rng, 0.5))
    assert len(np.unique(out["idx"])) == 17 and np.isclose(out["w"].sum(), 256.0, atol=1e-3)


def test_errors(sampler):
    rng = np.random.default_rng(6)
    lab, gt = _labels(rng), _gt(rng)
    with pytest.raises(ValueError):
        sampler.sample("x", np.full((8, 3), np.nan, np.float32), lab)
    with pytest.raises(ValueError):
        sampler.sample("x", gt, None)                  # no label source
    with pytest.raises(ValueError):
        sampler.sample("x", gt, lab[:, :100])


# ----------------------------------------------------------------------------------------------- v2 parity
def _v2_winner(anchors: np.ndarray, gt: np.ndarray) -> int:
    """Winner of v2's anchor_plan_losses WTA, read back through the loss itself: with offset_k = (gt - a_k) + k the
    winner's L1 is exactly k."""
    from navsim.agents.para_ssr.modules.anchor_planner import anchor_plan_losses
    A = torch.from_numpy(anchors)
    g = torch.from_numpy(gt)[None]
    k = torch.arange(256, dtype=torch.float32).view(1, 256, 1, 1)
    off = (g[:, None] - A[None]) + k
    pred = {"trajectory_anchors": A, "trajectory_offset": off, "im_rewards": torch.full((1, 256), 1 / 256),
            "sim_rewards": torch.full((1, 5, 256), 0.5)}
    loss = anchor_plan_losses(pred, g, torch.full((1, 5, 256), 0.5), torch.ones(1))["traj_offset_loss"]
    return int(round(float(loss)))


def test_wta_metric_matches_v2(sampler):
    rng = np.random.default_rng(7)
    for i in range(40):
        gt = _gt(rng)
        s = sampler.sample(f"w{i}", gt, _labels(rng), shuffle=False)
        assert int(s["idx"][0]) == _v2_winner(sampler.anchors, gt)
        tdist = torch.norm(torch.from_numpy(sampler.anchors).reshape(1, 256, -1) - torch.from_numpy(gt).reshape(1, 1, -1),
                           dim=2)[0].numpy()
        assert np.allclose(AS.gt_distance(sampler.anchors, gt), tdist, rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------------------------------- label source / priors
def _fake_label_files(d: Path, tokens, rng, nan_row=None):
    d.mkdir(parents=True, exist_ok=True)
    arr = np.stack([_labels(rng, p_pass=rng.uniform(0.1, 0.6)) for _ in tokens]).astype(np.float16)
    if nan_row is not None:
        arr[nan_row] = np.nan
    np.save(d / "s.npy", arr)
    json.dump(list(tokens), open(d / "s.tokens.json", "w"))
    return arr


def test_label_source_official_and_fallback(tmp_path, caplog):
    rng = np.random.default_rng(8)
    toks = [f"k{i}" for i in range(6)]
    off = _fake_label_files(tmp_path / "off", toks, rng, nan_row=5)
    _fake_label_files(tmp_path / "wote", toks, rng)
    src = AS.LabelSource(official=tmp_path / "off/s", fallback=tmp_path / "wote/s")
    assert src.kind == "official_ep" and src.ep_valid
    assert np.array_equal(src.get("k1"), off[1].astype(np.float32))
    assert src.get("k5") is None and src.get("nope") is None
    rows = src.get_rows(["k3", "nope", "k0"])
    assert np.array_equal(rows[0], off[3].astype(np.float32)) and np.isnan(rows[1]).all()
    src2 = pickle.loads(pickle.dumps(src))
    assert src2._scores is None and np.array_equal(src2.get("k2"), src.get("k2"))
    with caplog.at_level(logging.WARNING, logger=AS.__name__):
        fb = AS.LabelSource(official=tmp_path / "missing/s", fallback=tmp_path / "wote/s")
    assert fb.kind == "wote_ep_masked" and not fb.ep_valid and "FALLBACK" in caplog.text
    lab = fb.get("k0")
    assert np.isnan(lab[AS.EP]).all() and np.isfinite(np.delete(lab, AS.EP, 0)).all()
    assert np.isnan(fb.get_rows(["k0"])[:, AS.EP]).all()
    assert AS.LabelSource(prefer_official=False, official=tmp_path / "off/s",
                          fallback=tmp_path / "wote/s").kind == "wote_ep_masked"
    with pytest.raises(FileNotFoundError):
        AS.LabelSource(official=tmp_path / "a/s", fallback=tmp_path / "b/s")


def _fake_split(root: Path, split: str, n: int, rng):
    p = root / "packed" / split
    p.mkdir(parents=True)
    toks = [f"tok{i:04d}" for i in range(n)]
    pd.DataFrame({"token": toks, "log": "l", "city": "c", "row": np.arange(n)}).to_parquet(p / "tokens.parquet")
    gt = np.stack([_gt(rng) for _ in range(n)])
    gt[n - 1] = np.nan                                  # one token without GT
    np.save(p / "gt_traj.npy", gt)
    lc = root / "labels" / split / "cand"
    lc.mkdir(parents=True)
    lab = np.ones((n, 16, len(C.LABEL_COLS)), np.float32)
    lab[..., C.LBL["ep"]] = rng.uniform(0.3, 1.0, size=(n, 16))
    lab[..., C.LBL["dac"]] = (rng.uniform(size=(n, 16)) > 0.05)
    lab[..., C.LBL["nc"]] = np.where(rng.uniform(size=(n, 16)) > 0.03, 1.0, 0.5)
    ok = np.ones((n, 16), bool)
    ok[0, 3] = False
    np.save(lc / "labels.npy", lab)
    np.save(lc / "ok.npy", ok)
    return toks, lab, ok


@pytest.mark.parametrize("official", [True, False])
def test_split_priors_offsets(tmp_path, official):
    rng = np.random.default_rng(9)
    toks, lab, ok = _fake_split(tmp_path, "navtrain_val", 40, rng)
    _fake_label_files(tmp_path / "off", toks[:-2], rng)   # tok0038 has no labels
    _fake_label_files(tmp_path / "wote", toks[:-2], rng)
    src = AS.LabelSource(prefer_official=official, official=tmp_path / "off/s", fallback=tmp_path / "wote/s")
    sm = AS.AnchorSampler(anchors=_anchors(), label_source=src)
    r = AS.split_priors("navtrain_val", sm, root=tmp_path, chunk=7)
    assert r["n_tokens"] == 38 and r["skipped"] == {"no_labels": 2, "no_gt": 0}
    assert r["label_source"]["kind"] == ("official_ep" if official else "wote_ep_masked")
    keys = [k for k in AS.PRIOR_KEYS if official or k != "ep"]
    for name in ("offset_sampled", "offset_population", "offset_sampled_to_population"):
        assert all(np.isfinite(r[name][k]) for k in keys), r[name]
        assert official or np.isnan(r[name]["ep"])
    # deploy prior on the same 38 rows, ok-masked
    y = lab[:38][ok[:38]]
    assert np.isclose(r["deploy"]["dac"], y[:, C.LBL["dac"]].mean())
    assert r["deploy_n_cands"] == int(ok[:38].sum())
    # sampled prior (exact design expectation) == brute force over designs; population == plain mean
    num, den, pop = 0.0, 0.0, []
    for t in toks[:38]:
        d = sm.design(np.load(tmp_path / "packed/navtrain_val/gt_traj.npy")[toks.index(t)], src.get(t))
        num += (d["pi"] * d["pass"]).sum()
        den += d["pi"].sum()
        pop.append(d["pass"].mean())
    assert np.isclose(r["sampled"]["pass"], num / den) and np.isclose(r["population"]["pass"], np.mean(pop))
    assert np.isclose(r["offset_sampled"]["pass"], AS.logit(r["deploy"]["pass"]) - AS.logit(r["sampled"]["pass"]))
    assert np.isclose(r["offset_sampled_to_population"]["pass"],
                      AS.logit(r["population"]["pass"]) - AS.logit(r["sampled"]["pass"]))
    wp, wn = r["lsw_sampled"]["pass"]
    p, q = r["sampled"]["pass"], r["deploy"]["pass"]
    assert np.isclose(wp * p + wn * (1 - p), 1.0) and np.isclose(wp * p, q)   # reweighted prior == deploy prior
    assert np.isclose(sum(r["fallback"].values()), 1.0)


def test_logit_helpers():
    assert AS.logit(0.5) == 0.0 and np.isfinite(AS.logit(0.0)) and np.isfinite(AS.logit(1.0))
    assert np.isnan(AS.logit(float("nan")))
    assert np.isclose(AS.log_odds_offset(0.9, 0.5), np.log(9.0))


# ----------------------------------------------------------------------------------------------- real data (optional)
REAL_VAL = C.CK_DATA / "packed" / "navtrain_val" / "gt_traj.npy"
REAL_OK = REAL_VAL.is_file() and Path(str(AS.OFFICIAL_LABELS) + ".npy").is_file() and AS.ANCHORS_FILE.is_file()


@pytest.mark.skipif(not REAL_OK, reason="no packed navtrain_val / official raw256 labels / anchors")
def test_real_navtrain_val():
    src = AS.LabelSource()
    assert src.kind == "official_ep"
    sm = AS.AnchorSampler(label_source=src, seed=0)            # default anchors file, sha checked
    gt = AS.PackedGT("navtrain_val")
    for t in gt.tokens[:100]:
        g = gt.get(t)
        s = sm.sample(t, g, epoch=1, shuffle=False)
        assert int(s["idx"][0]) == _v2_winner(sm.anchors, g)
        assert len(np.unique(s["idx"])) == 32 and np.isclose(s["w"].astype(np.float64).sum(), 256.0, atol=1e-3)
    r = AS.split_priors("navtrain_val", sm, labels=src, gt=gt, max_tokens=300)
    for name in ("offset_sampled", "offset_population"):
        assert all(np.isfinite(v) for v in r[name].values()), r[name]


FIX = Path("/workspace/yongjae/ssd/yongjae_refiner/ck/phase2/impl-model/fixtures/real_b2.pt")


@pytest.mark.skipif(not (FIX.is_file() and REAL_OK), reason="no real-batch fixture")
def test_packed_gt_equals_v2_target_trajectory():
    d = torch.load(FIX, map_location="cpu", weights_only=False)
    gt = AS.PackedGT("navtrain_train")
    src = AS.LabelSource()
    for i, t in enumerate(d["tokens"]):
        assert np.array_equal(gt.get(t), d["targets"]["trajectory"][i].numpy())
        # NC / DAC / TTC / C of the official file == v2's sim_reward target (WoTE pdm_score_256)
        lab = src.get(t)
        sr = d["targets"]["sim_reward"][i].numpy()
        assert np.array_equal(lab[list(AS.PASS_IDX)], sr[list(AS.PASS_IDX)])
