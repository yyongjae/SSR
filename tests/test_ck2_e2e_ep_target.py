"""CK2 e2e student: ck_e2e2.ep_target (user decision 2026-10-08 ~20:10 KST).  CPU only.

Every place where the student turns the 9 official label columns into CK targets goes through
ep_target.ck_targets(labels, ck_e2e2.ep_target): warm-up raw / variant targets (CK2E2ETargetBuilder -> wu_row), the
on-policy generation labels (LabelStore2.lookup via CKE2E2.label_set) and the score prior (ck2_warmup_prior).  Default
'official' since 2026-10-09 00:30 KST (was 'decoupled'; dataclass == ck_e2e2.yaml).  The warm-up anchor draw (pass mask NC / DAC / TTC / C) never depends on it.

  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_ep_target.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.ep_target import EP_TARGETS, ck_targets, decoupled_ep  # noqa: E402

CKI = list(Cn.CK_LABEL_IDX)
LB = Cn.LBL
W = D2.WARMUP_DEFAULT
PK, RL, VD = Path(W["packed"]), Path(W["raw_labels"]), Path(W["var_dir"])
HAVE = all((p / f).is_file() for p, f in ((PK, "tokens.parquet"), (RL, "labels.npy"), (VD, "labels.npy"),
                                         (VD, "index.npz")))
need = pytest.mark.skipif(not HAVE, reason="packed / raw256 / variant label files absent")
torch.set_num_threads(1)


def test_config_default_and_validation():
    from tools.ck.e2e2 import launch_util2 as L2
    c = O.CKE2E2Config.from_any(None)
    assert c.ep_target == "official" == D2.EP_TARGET_DEFAULT
    assert L2.load_ck2_yaml()["ep_target"] == "official"
    assert O.CKE2E2Config.from_any({"ep_target": "decoupled"}).ep_target == "decoupled"
    with pytest.raises(ValueError, match="ep_target"):
        O.CKE2E2Config.from_any({"ep_target": "bogus"})
    assert D2.ep_target_of(None) == "official" and D2.ep_target_of(c) == "official"
    assert D2.ep_target_of({"ep_target": "decoupled"}) == "decoupled"
    assert D2.ep_target_of(O.CKE2E2Config.from_any({"ep_target": "decoupled"})) == "decoupled"


# ----------------------------------------------------------------------------------------------- warm-up targets
def test_targets_wu_zeroed_row_is_never_ok():
    """_targets_wu: a row zeroed because an OFFICIAL CK column is non-finite is flagged not-ok for every ep_target, even
    when its decoupled EP is finite (no zero target with ok True); a non-finite decoupled EP only flags 'decoupled'."""
    lab = np.array([[1, 1, 0.8, 1, 1, 1, 0.9, 20.0, 25.0],
                    [1, 1, np.nan, 1, 1, 1, 0.9, 20.0, 25.0],         # official EP NaN, r / p finite
                    [1, 1, 0.8, 1, 1, 1, 0.9, np.nan, 25.0]], np.float32)   # official finite, decoupled EP NaN
    for ep in EP_TARGETS:
        y, fin = D2._targets_wu(lab, ep)
        assert fin.tolist() == ([True, False, True] if ep == "official" else [True, False, False]), ep
        assert np.isfinite(y).all() and (y[1] == 0).all() and y.dtype == np.float32
        assert np.array_equal(y[0], ck_targets(lab[0], ep))
    y, _ = D2._targets_wu(lab, "decoupled")
    assert y[2, 2] == 0.0 and np.array_equal(np.delete(y[2], 2), lab[2, [0, 1, 3, 4]])


def _synthetic_wu(tmp_path, n=2):
    """a 2-row warm-up source with real raw256 / variant labels (rows 0, 40000) and one non-finite progress each."""
    pk, rl, vd = tmp_path / "pk", tmp_path / "rl", tmp_path / "vd"
    for d in (pk, rl, vd):
        d.mkdir()
    rows = [0, 40000]
    pd.DataFrame({"token": [f"t{i:015d}" for i in range(n)], "log": ["L"] * n, "row": np.arange(n)}).to_parquet(
        pk / "tokens.parquet")
    np.save(pk / "ok.npy", np.ones(n, bool))
    np.save(pk / "gt_traj.npy", np.asarray(np.load(PK / "gt_traj.npy", mmap_mode="r")[rows], np.float32))
    raw = np.array(np.load(RL / "labels.npy", mmap_mode="r")[rows], np.float32)
    vl = np.array(np.load(VD / "labels.npy", mmap_mode="r")[rows], np.float32)
    raw[0, 7, LB["raw_progress"]] = np.nan                  # official columns finite, decoupled EP NaN
    vl[0, 13, LB["pdm_progress_eff"]] = np.inf
    np.save(rl / "labels.npy", raw)
    np.save(rl / "ok.npy", np.ones((n, 256), bool))
    np.save(vd / "traj.npy", np.asarray(np.load(VD / "traj.npy", mmap_mode="r")[rows], np.float32))
    np.save(vd / "labels.npy", vl)
    np.save(vd / "ok.npy", np.ones((n, 96), bool))
    z = np.load(VD / "index.npz")
    np.savez(vd / "index.npz", valid=z["valid"][rows], anchor_idx=z["anchor_idx"][rows], built=np.ones(n, bool))
    toks = pd.read_parquet(PK / "tokens.parquet").token.astype(str).to_numpy()[rows]
    return D2.warmup_cfg({"packed": str(pk), "raw_labels": str(rl), "var_dir": str(vd)}), raw, vl, list(toks)


@need
def test_warmup_targets_use_helper(tmp_path):
    wcfg, raw, vl, toks = _synthetic_wu(tmp_path)
    W_ = {ep: [D2.wu_row(D2._WUArrays(wcfg, ep), r) for r in range(2)] for ep in EP_TARGETS}
    assert D2._WUArrays(wcfg).ep_target == "official"
    for r in range(2):
        off, dec = W_["official"][r], W_["decoupled"][r]
        assert bool(off["ck2_wu_ok"]) and bool(dec["ck2_wu_ok"])
        # official: the old raw columns; decoupled: ck_targets (EP only differs) on every finite row
        y_off, y_dec = ck_targets(raw[r], "official"), ck_targets(raw[r], "decoupled")
        ok = np.isfinite(y_dec).all(-1)
        assert np.array_equal(off["ck2_raw_y"], np.where(np.isfinite(y_off).all(-1)[:, None], y_off, 0))
        assert np.array_equal(dec["ck2_raw_y"][ok], y_dec[ok])
        assert np.array_equal(np.delete(dec["ck2_raw_y"], 2, -1), np.delete(off["ck2_raw_y"], 2, -1))
        assert np.array_equal(dec["ck2_raw_ok"], off["ck2_raw_ok"] & ok)
        v_dec = ck_targets(vl[r], "decoupled")
        vok = np.isfinite(v_dec).all(-1)
        assert np.array_equal(dec["ck2_var_y"][vok], v_dec[vok])
        assert np.array_equal(off["ck2_var_y"], vl[r][:, CKI])
        assert np.array_equal(dec["ck2_var_ok"], off["ck2_var_ok"] & vok)
        n_ep = int((dec["ck2_raw_y"][:, 2] != off["ck2_raw_y"][:, 2]).sum())
        assert n_ep > 0                                         # real rows: some candidate fails NC / DAC / DDC
    # the injected non-finite progress: not ok and EP zeroed only for 'decoupled'; NC / DAC / TTC / C kept official
    d0, o0 = W_["decoupled"][0], W_["official"][0]
    assert bool(o0["ck2_raw_ok"][7]) and not bool(d0["ck2_raw_ok"][7]) and d0["ck2_raw_y"][7, 2] == 0.0
    assert np.array_equal(np.delete(d0["ck2_raw_y"][7], 2), raw[0, 7, [0, 1, 3, 4]])
    assert bool(o0["ck2_var_ok"][13]) and not bool(d0["ck2_var_ok"][13])
    # the warm-up anchor draw / variant draw / fallback anchors are identical for both EP targets
    ws = D2.WarmupSampler(n_var=32)
    arr = {ep: {k: np.stack([W_[ep][r][k] for r in range(2)]) for k in ("ck2_wu_ok", "ck2_gt", "ck2_raw_y",
                                                                         "ck2_var_valid")} for ep in EP_TARGETS}
    for e in (0, 3):
        for stream in (D2.STREAM_WU, D2.STREAM_LAB):
            sa, sb = ws.sample(toks, arr["official"], e, stream), ws.sample(toks, arr["decoupled"], e, stream)
            assert sa["ok"].all()
            for k in sa:
                assert np.array_equal(sa[k], sb[k]), (e, stream, k)


@need
def test_builder_and_warmup_cands_follow_config(tmp_path):
    """CK2E2ETargetBuilder(ck_e2e2) picks ep_target from the config (ParaSSRConfig-like namespace too); warmup_cands /
    fill_fallback just gather the builder's targets."""
    from types import SimpleNamespace
    from navsim.agents.para_ssr.ck import cands2 as C2
    tdf = pd.read_parquet(PK / "tokens.parquet")
    rows = [5, 40000]
    toks = [str(tdf.token.iloc[r]) for r in rows]
    raw = np.array(np.load(RL / "labels.npy", mmap_mode="r")[rows], np.float32)
    out = {}
    for ep in EP_TARGETS:
        b = D2.CK2E2ETargetBuilder(SimpleNamespace(ck_e2e2={"ep_target": ep}))
        assert b.ep_target == ep and b.src.ep_target == ep
        t = [b.warmup_row(r) for r in rows]
        t = {k: torch.stack([x[k] for x in t]) for k in t[0]}
        for i in range(2):
            assert np.array_equal(t["ck2_raw_y"][i].numpy(), ck_targets(raw[i], ep))
        ws = D2.WarmupSampler(n_var=32)
        wc = C2.warmup_cands(t, toks, 1, ws)
        S = ws.sample(toks, {k: t[k] for k in ("ck2_wu_ok", "ck2_gt", "ck2_raw_y", "ck2_var_valid")}, 1)
        for i in range(2):
            assert np.array_equal(wc["y"][i, :32].numpy(), ck_targets(raw[i][S["anc_idx"][i]], ep))
        out[ep] = (wc, S)
    (wa, sa), (wb, sb) = out["official"], out["decoupled"]
    assert np.array_equal(sa["anc_idx"], sb["anc_idx"]) and np.array_equal(sa["var_cols"], sb["var_cols"])
    assert torch.equal(wa["cand"], wb["cand"]) and torch.equal(wa["ok"], wb["ok"])
    assert torch.equal(wa["y"][..., [0, 1, 3, 4]], wb["y"][..., [0, 1, 3, 4]])
    assert not torch.equal(wa["y"][..., 2], wb["y"][..., 2])


# ----------------------------------------------------------------------------------------------- generation targets
def _gen_with_real_labels(io_dir, epoch, gen_rows, n_rows):
    """generation rows with REAL 9-column labels (navtrain_train variant labels: progress in metres, NC / DAC / DDC
    failures), so the two EP targets differ; one non-finite raw_progress (row 0, identity column 6 = rank 1)."""
    m = len(gen_rows)
    lab = np.array(np.load(VD / "labels.npy", mmap_mode="r")[1000:1000 + m], np.float32)
    lab[0, 6, LB["raw_progress"]] = np.nan
    a = np.load(TL.ANCHORS).astype(np.float32)
    rng = np.random.default_rng(0)
    traj = np.stack([a[rng.choice(256, 96, replace=False)] for _ in range(m)]).astype(np.float32)
    gen = D2.open_generation2(io_dir, epoch, n_rows=n_rows, mode="w+")
    D2.write_generation_rows2(gen, np.asarray(gen_rows, np.int64), traj, lab, np.ones((m, 96), bool),
                              np.ones((m, 96), bool))
    return lab


@need
@pytest.mark.skipif(not TL.have_inputs(), reason="CK2 fixture / smoke teachers absent")
def test_generation_targets_use_helper(tmp_path):
    got = {}
    for ep in EP_TARGETS:
        c = TL.cfg2(tmp_path / ep, ep_target=ep)
        ck = O.CKE2E2(c, max_epochs=30)
        lab = _gen_with_real_labels(c.io_dir, 4, [3, 9], 16)
        ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"], var_sampling=c.var_sampling,
                                    ep_target=c.ep_target)
        assert ck._labels.ep_target == ep
        ck.epoch = 5
        f, t = TL.fresh(2, rows=[3, 9])
        toks = ck.tokens_of(t["ck_row"])
        G = ck.label_set(t["ck_row"], toks, t, torch.device("cpu"))
        assert G["has"].all() and (G["G_src"] == O.SRC_PREV).all()
        for b in range(2):
            c_ = G["G_col"][b].numpy().astype(np.int64)
            y = ck_targets(lab[b][c_], ep)
            fin = np.isfinite(y).all(-1)
            assert np.array_equal(G["G_y"][b].numpy(), np.where(fin[:, None], y, 0.0))
            assert np.array_equal(G["G_ok"][b].numpy(), fin)
        got[ep] = (G, lab)
    (Ga, lab), (Gb, _) = got["official"], got["decoupled"]
    assert torch.equal(Ga["G_col"], Gb["G_col"]) and torch.equal(Ga["G_traj"], Gb["G_traj"])
    ya, yb = Ga["G_y"].numpy(), Gb["G_y"].numpy()
    ok = Gb["G_ok"].numpy()
    assert np.array_equal(np.delete(ya, 2, -1)[ok], np.delete(yb, 2, -1)[ok])
    assert (yb[..., 2][ok] >= ya[..., 2][ok] - 1e-6).all() and (yb[..., 2][ok] > ya[..., 2][ok]).any()
    # the NaN raw_progress of row 3 (identity column 6 = slot 1) is not ok only with 'decoupled'
    assert int(Ga["G_col"][0, 1]) == 6 and bool(Ga["G_ok"][0, 1]) and not bool(Gb["G_ok"][0, 1])
    # CKE2E2.labels() builds its store with cfg.ep_target
    for ep in EP_TARGETS:
        ck = O.CKE2E2(TL.cfg2(tmp_path / f"s_{ep}", ep_target=ep))
        assert ck.labels().ep_target == ep


@need
@pytest.mark.skipif(not TL.have_inputs(), reason="CK2 fixture / smoke teachers absent")
def test_onpolicy_bce_targets_through_loss(tmp_path):
    """the student BCE set of an on-policy step (CKE2E2.loss -> current_batch -> label_set) carries the configured EP"""
    for ep in EP_TARGETS:
        c = TL.cfg2(tmp_path / ep, ep_target=ep, lat_kd={"start_mb": 0})
        ck = O.CKE2E2(c, max_epochs=30)
        lab = _gen_with_real_labels(c.io_dir, 4, [3], 16)
        ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"], var_sampling=c.var_sampling,
                                    ep_target=c.ep_target)
        ck.epoch, ck.epoch_frac = 5, 5.5
        seen = {}
        fn = ck.current_batch

        def spy(*a, **k):
            seen["S"] = fn(*a, **k)
            return seen["S"]
        ck.current_batch = spy
        torch.manual_seed(0)
        st = O.build_student_ck2(c)
        f, t = TL.fresh(2, rows=[3, 7])
        pred = TL.fake_predictions(2, seed=3)
        loss, logs = ck.loss(st, f, t, pred)
        assert torch.isfinite(loss)
        S = seen["S"]
        c0 = np.arange(16) * 6                                   # identity columns of the generation row
        y = ck_targets(lab[0][c0], ep)
        assert np.array_equal(S["y"][0, :16].numpy(), np.where(np.isfinite(y).all(-1)[:, None], y, 0.0))
        if ep == "decoupled":
            assert np.array_equal(S["y"][0, :16, 2].numpy(), np.nan_to_num(decoupled_ep(lab[0][c0]), nan=0.0))
            assert not bool(S["ok_lab"][0, 1])                   # identity column 6: non-finite progress


@need
def test_score_prior_uses_ep_target(tmp_path):
    p = {}
    for ep in EP_TARGETS:
        c = O.CKE2E2Config.from_any({"ep_target": ep, "score_prior_rows": 60})
        p[ep], src = O.score_prior_of(c)
        assert src == "ck2_warmup"
        np.testing.assert_array_equal(p[ep], D2.ck2_warmup_prior(None, n_max=60, ep_target=ep))
    np.testing.assert_array_equal(np.delete(p["official"], 2), np.delete(p["decoupled"], 2))
    assert p["decoupled"][2] > p["official"][2]
