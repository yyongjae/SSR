"""Tests for tools/refiner/make_draft_bank.py (draft bank, IMPL_SPEC §3.6).  CPU, < 2 min.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_make_draft_bank.py

Data: tests/fixtures/decoder_navtest_sample.npz (301 navtest humans: 8 poses, path to 8 s, v0, a0) for the sampler
tests; human/train.npz + the stage-T / E metric caches (skipped when absent) for the centerline and scoring tests.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_draft_bank as M  # noqa: E402
import score_trajectories as ST  # noqa: E402
from navsim.agents.para_ssr.refiner import decoder as D  # noqa: E402

FIX = HERE / "fixtures" / "decoder_navtest_sample.npz"
N_TOK = 30


@pytest.fixture(scope="module")
def fx():
    z = np.load(FIX, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _kw(fx, i):
    return dict(path_long=fx["path_long"][i], n_valid=int(fx["n_valid"][i]), v0=float(fx["v0"][i]),
                a0=float(fx["a0"][i]))


@pytest.fixture(scope="module")
def banks(fx):
    return [M.make_bank(fx["tau_h"][i], str(fx["token"][i]), cfg=M.BankConfig(lconst_ap=M.DECODER_AP), **_kw(fx, i))
            for i in range(N_TOK)]


def test_default_config_is_decoder_sample_bank(fx, banks):
    """lconst_ap = decoder's range -> make_bank == decoder.sample_bank bitwise (same flow, same seed)."""
    for i in range(N_TOK):
        a = D.sample_bank(fx["tau_h"][i], str(fx["token"][i]), **_kw(fx, i))
        b = banks[i]
        for k in ("drafts", "family", "params", "z_lon", "w_lat", "path_src"):
            assert np.array_equal(a[k], b[k]), (i, k)
        # valid can only be lowered by the post-check (hdrift / cv); decoded drafts are checked inside the sampler
        assert not (b["valid"] & ~a["valid"]).any()
        assert b["drafts"].dtype == np.float32 and b["drafts"].shape == (13, 8, 3)
        assert np.array_equal(b["drafts"][0], fx["tau_h"][i])          # identity = human bytes
        assert list(b["slot"]) == [D.FAMILY[f] for f in D.BANK_LAYOUT]


def test_remap_rng_passthrough_and_hits():
    g1, g2 = np.random.default_rng(5), M.RemapRNG(np.random.default_rng(5), {(0.2, 1.3): (0.5, 0.5)})
    a = [g1.uniform(0.0, 0.2), g1.choice([0.0, 0.5, 1.0]), g1.random(), g1.standard_normal(), g1.uniform(0.2, 1.3),
         g1.uniform(0.3, 1.5)]
    b = [g2.uniform(0.0, 0.2), g2.choice([0.0, 0.5, 1.0]), g2.random(), g2.standard_normal(), g2.uniform(0.2, 1.3),
         g2.uniform(0.3, 1.5)]
    assert a[:4] == b[:4] and a[5] == b[5]      # the remapped draw consumes one variate: the stream stays aligned
    assert b[4] == 0.5 and g2.hits == 1


def test_ap_range_remap_changes_only_lconst(fx, banks):
    cfg = M.BankConfig(lconst_ap=(0.7, 0.7))
    assert cfg.hash != M.BankConfig(lconst_ap=M.DECODER_AP).hash
    n_chk = 0
    for i in range(N_TOK):
        b = M.make_bank(fx["tau_h"][i], str(fx["token"][i]), cfg=cfg, **_kw(fx, i))
        base = banks[i]
        assert b["rng_hits"] >= 1
        # slots before the first L-const draw use the same stream -> identical
        assert np.array_equal(b["drafts"][:4], base["drafts"][:4])
        beta = b["flags"][:, M.FLAG_COLS.index("beta")]
        for k in range(13):
            if b["valid"][k] and D.FAMILY_NAME[int(b["family"][k])] in ("lconst", "combined") and beta[k] == 1.0:
                assert b["params"][k, 0] == np.float32(0.7)
                n_chk += 1
    assert n_chk > 20


def test_lat_range_remap_and_centerline_rule(fx, banks):
    """lat_dp remaps only the lat / combined |D_p| draw (slots 0-8 unchanged); needs_centerline follows the config."""
    cfg = M.BankConfig(lconst_ap=M.DECODER_AP, lat_dp=(1.0, 1.0))
    dps = []
    for i in range(N_TOK):
        b = M.make_bank(fx["tau_h"][i], str(fx["token"][i]), cfg=cfg, **_kw(fx, i))
        assert np.array_equal(b["drafts"][:9], banks[i]["drafts"][:9])
        for k in (9, 10):
            if b["valid"][k] and b["family"][k] == D.FAMILY["lat"]:
                dps.append(abs(float(b["params"][k, 2])))
    # realised end offset = LS spline fit of the smoothstep (small overshoot) after the curvature projection
    assert len(dps) > 20 and max(dps) <= 1.02 and np.median(dps) > 0.9
    i = int(np.argmax([np.hypot(*fx["tau_h"][j][-1, :2]) for j in range(N_TOK)]))
    ctx = D.HumanContext(fx["tau_h"][i], fx["path_long"][i], int(fx["n_valid"][i]), float(fx["v0"][i]))
    assert not ctx.creep_ok
    assert not M.needs_centerline(ctx, M.BankConfig())
    extra = M.BankConfig(centerline_families=("creep", "lconst"))
    assert M.needs_centerline(ctx, extra) == (ctx.S_avail < ctx.S8 + M.CL_NEED_M)


def test_checks_hold_on_valid_drafts(fx, banks):
    """Independent numpy re-computation of the §3.6 limits on every valid draft (human-relative, as documented)."""
    def kin(tr, v0):
        P = np.vstack([[0.0, 0.0, 0.0], np.asarray(tr, np.float64)])
        ell = np.hypot(*np.diff(P[:, :2], axis=0).T)
        u = ell / 0.5
        acc = np.concatenate([[(u[0] - v0) / 0.25], np.diff(u) / 0.5])
        dh = np.diff(np.unwrap(P[:, 2]))
        lat = np.max(u * np.abs(dh) / 0.5)
        kap = np.max(np.where(ell >= 1.0, np.abs(dh) / np.maximum(ell, 1e-9), 0.0))
        return u[0] - v0, acc.min(), acc.max(), lat, kap

    n = 0
    for i in range(N_TOK):
        b = banks[i]
        v0 = float(fx["v0"][i])
        h = kin(fx["tau_h"][i], v0)
        for k in range(13):
            if not b["valid"][k]:
                continue
            f = kin(b["drafts"][k], v0)
            tol = 1e-5
            assert min(-0.8, h[0]) - tol <= f[0] <= max(0.6, h[0]) + tol, (i, k, f[0])
            assert f[1] >= min(-4.05, h[1]) - tol and f[2] <= max(2.40, h[2]) + tol, (i, k)
            assert f[3] <= max(3.8, h[3]) + tol and f[4] <= max(0.213, h[4]) + tol, (i, k)
            assert np.allclose(b["checks"][k, :3], np.float32(f[:3]), atol=1e-4)
            n += 1
        cv = b["family"] == D.FAMILY["cv"]
        if cv.any():
            d = b["drafts"][cv][0]
            t = np.arange(1, 9) * 0.5
            assert np.allclose(d[:, 0], np.float32(v0 * t)) and (d[:, 1:] == 0).all()
    assert n > 300


def test_deterministic_per_token(fx):
    i = 3
    a = M.make_bank(fx["tau_h"][i], str(fx["token"][i]), **_kw(fx, i))
    b = M.make_bank(fx["tau_h"][i], str(fx["token"][i]), **_kw(fx, i))
    c = M.make_bank(fx["tau_h"][i], "ffffffffffffffff", **_kw(fx, i))
    assert np.array_equal(a["drafts"], b["drafts"]) and np.array_equal(a["params"], b["params"])
    assert not np.array_equal(a["params"], c["params"])


def test_npz_roundtrip_and_draftsource(fx, banks, tmp_path):
    cfg = M.BankConfig(lconst_ap=M.DECODER_AP)
    for i in range(2):
        arr = M.bank_arrays(banks[i], str(fx["token"][i]), "somelog", "train", cfg)
        M.save_bank(tmp_path / f"{fx['token'][i]}.npz", arr)
    assert not list(tmp_path.glob("*.part"))
    z = M.load_bank(tmp_path / f"{fx['token'][0]}.npz")          # allow_pickle=False
    assert z["drafts"].dtype == np.float32 and z["family"].dtype == np.int8 and z["params"].shape == (13, 6)
    assert str(z["cfg_hash"]) == cfg.hash and M.BankConfig.from_json(str(z["cfg_json"])) == cfg
    assert np.array_equal(z["drafts"], banks[0]["drafts"])
    src = ST.DraftSource(str(tmp_path))
    d, fam = src.get(str(fx["token"][0]))
    assert np.array_equal(d, banks[0]["drafts"]) and np.array_equal(fam, banks[0]["family"])


def _fake_human(fx, rows, path):
    toks = np.array([str(fx["token"][i]) for i in rows])
    n = len(rows)
    np.savez(path, tokens=toks, logs=np.array([f"log{j}" for j in range(n)]), traj=fx["tau_h"][rows],
             path=fx["path_long"][rows], n_reg=fx["n_valid"][rows].astype(np.int8),
             v0=fx["v0"][rows].astype(np.float32), a0=fx["a0"][rows].astype(np.float32),
             frame_gap=np.array([j == 1 for j in range(n)]))
    return toks


def test_generate_resume_skips_and_guard(fx, tmp_path):
    """frame-gap / no-cache / fresh-cache tokens are skipped, a rerun builds nothing, a config change refuses."""
    S8 = np.array([np.hypot(*np.diff(np.vstack([[0, 0], fx["tau_h"][i][:, :2]]), axis=0).T).sum() for i in range(60)])
    rows = [i for i in range(60) if S8[i] > 5][:6]
    hroot = tmp_path / "human"
    hroot.mkdir()
    toks = _fake_human(fx, rows, hroot / "train.npz")
    mroot = tmp_path / "mc"
    old = time.time() - 3600
    for j, t in enumerate(toks):
        if j == 2:
            continue                                   # no cache
        p = mroot / f"log{j}" / "unknown" / t / "metric_cache.pkl"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        if j != 3:
            os.utime(p, (old, old))                    # j = 3 stays fresh
    out = tmp_path / "drafts"
    cfg = M.BankConfig(lconst_ap=M.DECODER_AP)
    r = M.generate("train", None, out, cfg, workers=1, min_age=120, roots=[mroot], human_root=hroot)
    assert r["counts"] == {"todo": 3, "frame_gap": 1, "no_mc": 1, "mc_fresh": 1} and r["built"] == 3, r
    built = sorted(p.stem for p in out.glob("*.npz"))
    assert built == sorted([toks[0], toks[4], toks[5]])
    r2 = M.generate("train", None, out, cfg, workers=1, min_age=120, roots=[mroot], human_root=hroot)
    assert r2["counts"].get("exists") == 3 and r2["built"] == 0
    with pytest.raises(SystemExit):
        M.generate("train", None, out, M.BankConfig(lconst_ap=(0.3, 1.0)), workers=1, roots=[mroot],
                   human_root=hroot)


def test_summarize_counts(fx, banks, tmp_path):
    import pandas as pd
    cfg = M.BankConfig(lconst_ap=M.DECODER_AP)
    rows = []
    for i in range(3):
        t = str(fx["token"][i])
        M.save_bank(tmp_path / f"{t}.npz", M.bank_arrays(banks[i], t, "log", "train", cfg))
        for k in range(13):
            fail_nc = k in (4, 12) and i < 2           # two lconst / cv failures per token on 2 tokens
            rows.append(dict(token=t, k=k, nc=0.0 if fail_nc else 1.0, dac=0.0 if (k == 9 and i == 0) else 1.0,
                             ddc=1.0, ep=1.0, ttc=0.0 if k == 5 else 1.0, comfort=1.0, pdms=0.5, family=0,
                             pdm_progress_eff=10.0, raw_progress=10.0, sec_load=0.1, sec_score=0.1))
    rows.append(dict(token="zzz", k=-1))
    S = pd.DataFrame(rows)
    r = M.summarize(tmp_path, S)
    V = [(i, k) for i in range(3) for k in range(13) if banks[i]["valid"][k]]
    exp_fail = sum(((k in (4, 12) and i < 2) or (k == 9 and i == 0)) for i, k in V) / len(V)
    assert r["n_tokens"] == 3 and r["n_error_tokens"] == 1
    assert abs(r["overall"]["valid_rows"]["fail"] - round(exp_fail, 4)) < 1e-4
    exp_ttc = sum(((k in (4, 12) and i < 2) or (k == 9 and i == 0) or k == 5) for i, k in V) / len(V)
    assert abs(r["overall"]["valid_rows"]["fail_ttc"] - round(exp_ttc, 4)) < 1e-4
    assert r["overall"]["valid_perturbed_rows"]["n"] == len(V) - 3
    assert r["per_family"]["identity"]["fail"] == 0.0


def _creep_token_with_mc():
    hp = M.HUMAN / "train.npz"
    if not hp.exists():
        return None
    H = M.load_human("train")
    tr = H["traj"].astype(np.float64)
    P = np.concatenate([np.zeros((len(tr), 1, 2)), tr[:, :, :2]], 1)
    S8 = np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)).sum(1)
    for i in np.nonzero((S8 < 0.5) & ~H["frame_gap"])[0][:400]:
        st, p = M.mc_ready(str(H["tokens"][i]), str(H["logs"][i]), ST.MC_ROOTS, 120)
        if st == "ok":
            return H, int(i), p
    return None


def test_centerline_and_creep_on_real_cache():
    found = _creep_token_with_mc()
    if found is None:
        pytest.skip("no human/train.npz or no stopped token with a metric cache yet")
    H, i, p = found
    mc = ST.load_metric_cache(p)
    cl = M.centerline_n(mc)
    assert cl.ndim == 2 and cl.shape[1] == 2 and len(cl) >= 2
    # the ego (origin of N) stands on / next to its route centerline
    seg = np.diff(cl, axis=0)
    L2 = np.maximum((seg ** 2).sum(1), 1e-12)
    tt = np.clip((-(cl[:-1] * seg).sum(1)) / L2, 0, 1)
    dist = np.hypot(*(cl[:-1] + tt[:, None] * seg).T).min()
    assert dist < 5.0, dist
    calls = []

    def cl_fn():
        calls.append(1)
        return cl

    b = M.make_bank(H["traj"][i], str(H["tokens"][i]), path_long=H["path"][i], n_valid=int(H["n_reg"][i]),
                    v0=float(H["v0"][i]), a0=float(H["a0"][i]), centerline=cl_fn)
    assert calls == [1]
    assert b["family"][8] == D.FAMILY["creep"] or not b["valid"][8] or "fallback" in b["reason"][8]
    assert np.isfinite(b["drafts"]).all()
    if b["path_src"][8] == "centerline" and b["valid"][8]:
        # the creep draft ends near the centerline continuation (offset decays to 0 over 10 m)
        e = b["drafts"][8][-1, :2].astype(np.float64)
        d_end = np.hypot(*(cl - e).T).min()
        assert d_end < 3.0


def test_score_two_tokens_end_to_end(tmp_path):
    """generate + official scoring for 2 real tokens (1 worker): 26 rows, family copied, identity passes."""
    found = None
    if (M.HUMAN / "train.npz").exists():
        H = M.load_human("train")
        for i in range(len(H["tokens"])):
            if not H["frame_gap"][i] and M.mc_ready(str(H["tokens"][i]), str(H["logs"][i]), ST.MC_ROOTS, 120)[0] == "ok":
                found = (H, i)
                break
    if found is None:
        pytest.skip("no metric cache")
    H, i = found
    toks = [str(H["tokens"][i])]
    for j in range(i + 1, len(H["tokens"])):
        if not H["frame_gap"][j] and M.mc_ready(str(H["tokens"][j]), str(H["logs"][j]), ST.MC_ROOTS, 120)[0] == "ok":
            toks.append(str(H["tokens"][j]))
            break
    ddir = tmp_path / "drafts"
    g = M.generate("train", toks, ddir, M.BankConfig(), workers=1)
    assert g["built"] == len(toks)
    out = tmp_path / "scores" / "t.parquet"
    s = M.score(ddir, out, workers=1)
    assert s["complete"] and s["rows"] == 13 * len(toks)
    import pandas as pd
    df = pd.read_parquet(out)
    assert (df.groupby("token").k.count() == 13).all()
    z = M.load_bank(ddir / f"{toks[0]}.npz")
    assert df[df.token == toks[0]].sort_values("k").family.tolist() == z["family"].tolist()
    r = M.summarize(ddir, out)
    assert r["n_tokens"] == len(toks) and r["n_rows"] == 13 * len(toks)
