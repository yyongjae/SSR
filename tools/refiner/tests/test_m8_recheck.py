"""Tests for tools/refiner/m8_recheck.py (train-split M8 margin / mask re-check).  CPU, ~1-2 min.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest \
          -q tools/refiner/tests/test_m8_recheck.py

- surrogate defaults are untouched (the grid is passed through SurrogateConfig arguments only);
- batched per-draft margin tensors == separate scalar-config runs == validate_surrogate._guided (default setting);
- the optimisation-only object prefilter leaves guided outputs bitwise unchanged (real train tokens);
- the two mask rules coincide when no human-overlap pair exists;
- metric definitions (A1, A2, A3, DAC) on a hand-built table; the pre-stated selection rule; the train-only guard.
Real-data tests are skipped when the stage-T train data are missing.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))
import m8_recheck as R  # noqa: E402

DATA = Path("/home/external-user/ssd/yongjae_refiner")


def test_defaults_untouched_and_grid():
    from navsim.agents.para_ssr.refiner import surrogate as SU
    c = SU.SurrogateConfig()
    assert c.m_col == 0.3 and c.m_dac == 0.2 and c.use_human_mask is True
    assert SU.M_COL == 0.3 and SU.M_DAC == 0.2
    assert len(R.SETTINGS) == 48 and len({s["sid"] for s in R.SETTINGS}) == 48
    assert {(s["mask"], s["m_col"], s["m_dac"]) for s in R.SETTINGS} == {
        (mk, mc, md) for mk in ("sat", "none") for mc in R.M_COLS for md in R.M_DACS}


def test_train_only_guard():
    with pytest.raises(SystemExit):
        R._train_only(DATA / "drafts/dev/x.npz")
    with pytest.raises(SystemExit):
        R._train_only(Path("/x/sdf/navtest/t.npz"))
    R._train_only(DATA / "drafts/train/x.npz")


def test_pick_one_per_log():
    s = pd.DataFrame({"token": [f"t{i}" for i in range(30)], "log": [f"L{i % 7}" for i in range(30)]})
    p = R.pick_one_per_log(s, 100, 0)
    assert len(p) == 7 and p.log.is_unique
    p5 = R.pick_one_per_log(s, 5, 0)
    assert len(p5) == 5 and p5.log.is_unique
    assert p.equals(R.pick_one_per_log(s, 100, 0))


def test_mask_rules_coincide_without_human_overlap():
    import torch
    from navsim.agents.para_ssr.refiner import surrogate as SU
    from navsim.agents.para_ssr.refiner.geometry import dense_reference
    t = 0.5 * np.arange(1, 9)
    tau = torch.as_tensor(np.stack([5 * t, np.zeros(8), np.zeros(8)], -1)[None], dtype=torch.float64)
    boxes = torch.zeros(1, 1, 41, 5, dtype=torch.float64)
    boxes[..., 0], boxes[..., 3], boxes[..., 4] = 12.0, 4.0, 2.0
    sc = SU.SceneBatch(boxes=boxes, obs=torch.ones(1, 1, 41, dtype=torch.bool),
                       is_agent=torch.ones(1, 1, dtype=torch.bool), human_overlap=torch.zeros(1, 1, 41, dtype=torch.bool))
    d = dense_reference(tau)
    a = SU.collision_cost(d, sc, None, SU.SurrogateConfig(use_human_mask=True))
    b = SU.collision_cost(d, sc, None, SU.SurrogateConfig(use_human_mask=False))
    assert torch.equal(a["cost"], b["cost"]) and torch.equal(a["gmin"], b["gmin"])
    sc.human_overlap[..., 20:] = True
    a = SU.collision_cost(d, sc, None, SU.SurrogateConfig(use_human_mask=True))
    assert not torch.equal(a["cost"], b["cost"])


# ----------------------------------------------------------------------------------------------- metrics / selection
def _toy_df():
    """one token per log; bank k=0 human + k=1..3; guided for one setting (sid 0: sat, m_col 0, m_dac 0)."""
    rows = []
    for i in range(4):
        tok, log = f"t{i}", f"L{i}"
        # human: passes; g_s 0.2 (flag only at m_col > 0.2), dmin 0.1
        rows.append(dict(token=tok, log=log, pool="bank", k=0, sid=-1, modified=False, nc=1.0, dac=1.0, ttc=1.0,
                         g_s_sat=0.2, g_s_none=0.2, dmin=0.1))
        # k=1: NC failure, g_s -0.5 (flag at every margin); k=2: pass g_s 0.12; k=3: DAC fail dmin -0.3
        rows.append(dict(token=tok, log=log, pool="bank", k=1, sid=-1, modified=False, nc=0.0, dac=1.0, ttc=0.0,
                         g_s_sat=-0.5, g_s_none=-0.5, dmin=1.0))
        rows.append(dict(token=tok, log=log, pool="bank", k=2, sid=-1, modified=False, nc=1.0, dac=1.0, ttc=0.0,
                         g_s_sat=0.12, g_s_none=0.12, dmin=1.0))
        rows.append(dict(token=tok, log=log, pool="bank", k=3, sid=-1, modified=False, nc=1.0, dac=0.0, ttc=1.0,
                         g_s_sat=2.0, g_s_none=2.0, dmin=-0.3))
        # guided (sid 0): k=1 -> g_s 0.01 (clean at m_col 0), officially fixed only for i < 3;
        # k=2 unmodified; k=3 -> dmin 0.02, DAC fixed for i < 2
        rows.append(dict(token=tok, log=log, pool="guided", k=1, sid=0, modified=True, nc=1.0 if i < 3 else 0.0,
                         dac=1.0, ttc=1.0, g_s_sat=0.01, g_s_none=0.01, dmin=1.0))
        rows.append(dict(token=tok, log=log, pool="guided", k=2, sid=0, modified=False, nc=1.0, dac=1.0, ttc=0.0,
                         g_s_sat=0.12, g_s_none=0.12, dmin=1.0))
        rows.append(dict(token=tok, log=log, pool="guided", k=3, sid=0, modified=True, nc=1.0,
                         dac=1.0 if i < 2 else 0.0, ttc=1.0, g_s_sat=2.0, g_s_none=2.0, dmin=0.02))
    df = pd.DataFrame(rows)
    for c in ("g_h_sat", "g_h_none"):
        df[c] = df.g_s_sat + 0.05
    for c in ("t_g_s_sat", "t_g_s_none", "t_g_h_sat", "t_g_h_none", "t_dmin"):
        df[c] = df[c[2:]]
    df["pdms"] = df.nc * df.dac
    df["ep"] = 1.0
    df["mult"] = df.nc * df.dac
    return df


def test_setting_metrics_toy():
    df = _toy_df()
    s = dict(sid=0, m_col=0.0, m_dac=0.0, mask="sat")
    m = R.setting_metrics(df, s)
    assert m["A1"]["k"] == 4 and m["A1"]["n"] == 4                  # every NC failure flagged
    assert m["A2"]["k"] == 0 and m["A2"]["n"] == 4                  # human g_s 0.2 >= 0
    assert m["NC_fpr"]["k"] == 0
    # A3: sur fixed = 4 (k=1 pairs: flag0 at 0, g_s1 0.01 clean); officially fixed 3
    assert m["A3"]["n"] == 4 and m["A3"]["k"] == 3 and m["n_pairs"] == 8
    assert m["A3_decomp"]["tau0 nc fail"] == 4
    assert m["DAC_recall"]["k"] == 4 and m["DAC_human_FA"]["k"] == 0
    assert m["DAC_A3"]["n"] == 4 and m["DAC_A3"]["k"] == 2
    e = m["effect_all_sources"]
    assert e["nc fixed"] == 3 and e["nc new fail"] == 0 and e["dac fixed"] == 2
    r = m["residual_nc_gap"]
    assert r["residual nc failures (all sources)"] == 1 and r["... raw reference clean at m_col (invisible)"] == 1
    # at m_col 0.15 the human (0.2) is still clean but the bank k=2 (0.12) and the guided k=1 (0.01) are flagged
    s2 = dict(sid=0, m_col=0.15, m_dac=0.0, mask="sat")
    m2 = R.setting_metrics(df, s2)
    assert m2["NC_fpr"]["k"] == 4 and m2["A3"]["n"] == 0
    # at m_col 0.3 the human is flagged, and at m_dac 0.2 the human DAC too
    m3 = R.setting_metrics(df, dict(sid=0, m_col=0.3, m_dac=0.2, mask="sat"))
    assert m3["A2"]["k"] == 4 and m3["DAC_human_FA"]["k"] == 4


def test_select_setting_rule():
    rows = []
    for s in R.SETTINGS:
        ok = s["m_col"] <= 0.15 and s["m_dac"] <= 0.05
        rows.append(dict(sid=s["sid"], m_col=s["m_col"], m_dac=s["m_dac"], mask=s["mask"],
                         A1=0.9, A2=0.01 if ok else 0.05, A3=0.7, DAC_hFA=0.01, feasible=ok,
                         n_failed=0 if ok else 1, shortfall=0.0 if ok else 0.03))
    t = pd.DataFrame(rows)
    sel = R.select_setting(t)
    s = R.SETTINGS[sel["sid"]]
    assert (s["m_col"], s["m_dac"], s["mask"]) == (0.15, 0.05, "sat")
    # none feasible: closest = fewest failed, then smallest shortfall, then larger margins
    t2 = t.copy()
    t2["feasible"] = False
    t2["n_failed"] = 1
    t2["shortfall"] = np.where((t2.m_col == 0.1) & (t2.m_dac == 0.0), 0.001, 0.02)
    sel2 = R.select_setting(t2)
    s2 = R.SETTINGS[sel2["sid"]]
    assert (s2["m_col"], s2["m_dac"], s2["mask"]) == (0.1, 0.0, "sat") and sel2["n_feasible"] == 0
    assert isinstance(sel2["pareto_front_sids"], list)


def test_feasible_thresholds():
    m = {"A1": {"p": 0.70}, "A2": {"p": 0.02}, "A3": {"p": 0.60}, "DAC_human_FA": {"p": 0.02}}
    f = R.feasible(m)
    assert f["all"] and f["n_failed"] == 0 and f["shortfall"] == 0.0
    m["A3"] = {"p": 0.5}
    f = R.feasible(m)
    assert not f["all"] and f["n_failed"] == 1 and abs(f["shortfall"] - 0.1) < 1e-12
    m["A3"] = {"p": float("nan")}
    assert not R.feasible(m)["criteria"]["A3"]


# ----------------------------------------------------------------------------------------------- real train tokens
def _real_scene(i: int = 0):
    import torch
    import score_trajectories as ST
    from navsim.agents.para_ssr.refiner import gt_future as GF
    from navsim.agents.para_ssr.refiner import sdf as S
    from navsim.agents.para_ssr.refiner import surrogate as SU
    toks = pd.read_parquet(R.OUT / "tokens.parquet")
    tok, log = toks.token.iloc[i], toks.log.iloc[i]
    with np.load(R.DRAFT_DIR / f"{tok}.npz") as z:
        drafts, valid = np.asarray(z["drafts"], np.float32), np.asarray(z["valid"], bool)
    h = np.load(R.HUMAN)
    j = h["tokens"].tolist().index(tok)
    mc = ST.load_metric_cache(ST.locate_metric_cache(tok, log))
    sc = ST.score_token(mc, drafts[:1])
    scene = SU.collate_scenes([SU.scene_from_numpy(GF.load_objects(R.OBJ_DIR / f"{tok}.npz"),
                                                   S.load_sdf(S.sdf_path(tok, "navtrain")),
                                                   SU.centerline_from_metric_cache(mc), h["traj"][j],
                                                   sc[0]["pdm_progress_eff"], float(h["v0"][j]), float(h["a0"][j]))])
    ks = [k for k in range(1, 13) if valid[k]]
    return scene, torch.as_tensor(drafts[ks].astype(np.float64)), float(h["v0"][j])


def _guided_scalar(src, scene, v0, cfg, steps=60, lr=0.05):
    """reference: validate_surrogate._guided's optimisation loop with a SCALAR config."""
    import torch
    from navsim.agents.para_ssr.refiner import decoder as D
    from navsim.agents.para_ssr.refiner import surrogate as SU
    M = src.shape[0]
    idx = torch.zeros(M, dtype=torch.long)
    v0t = torch.full((M,), float(v0), dtype=src.dtype)
    z = torch.zeros(M, 6, dtype=src.dtype, requires_grad=True)
    w = torch.zeros(M, 6, dtype=src.dtype, requires_grad=True)
    opt = torch.optim.Adam([z, w], lr=lr)
    best = torch.full((M,), float("inf"), dtype=src.dtype)
    bz, bw = torch.zeros(M, 6, dtype=src.dtype), torch.zeros(M, 6, dtype=src.dtype)
    for it in range(steps + 1):
        dec = D.decode(src, z, w, v0t, "A")
        tot, terms = SU.surrogate_loss(dec, src, scene, idx, cfg)
        per = terms["per_draft"].detach()
        better = per < best
        best = torch.where(better, per, best)
        bz[better], bw[better] = z.detach()[better], w.detach()[better]
        if it == steps:
            break
        opt.zero_grad()
        (tot * M).backward()
        opt.step()
    with torch.no_grad():
        return D.decode(src, bz, bw, v0t, "A")["traj"]


real = pytest.mark.skipif(not (R.OUT / "tokens.parquet").exists() or not R.DRAFT_DIR.exists(),
                          reason="stage-T train data / m8_recheck pool missing")


@real
def test_guided_multi_equals_scalar_runs_and_validate_surrogate():
    import torch
    import validate_surrogate as V
    from navsim.agents.para_ssr.refiner import surrogate as SU
    torch.set_num_threads(1)
    scene, src, v0 = _real_scene(1)
    src = src[:4]
    M = len(src)
    # default setting == validate_surrogate._guided (trajectory output)
    ref, _, _ = V._guided(src, scene, v0)
    out, _ = R.guided_multi(src, scene, v0, np.full(M, 0.3), np.full(M, 0.2), True)
    assert torch.abs(out - ref).max().item() <= 1e-12
    # two settings batched == each run alone with a scalar config
    sets = [(0.05, 0.0, True), (0.3, 0.1, False)]
    srcb = src.repeat(len(sets), 1, 1)
    for mk in (True, False):
        ss = [s for s in sets if s[2] == mk] * 2          # duplicate: batching the same setting twice is harmless
        srcb = src.repeat(len(ss), 1, 1)
        outb, _ = R.guided_multi(srcb, scene, v0, np.repeat([s[0] for s in ss], M), np.repeat([s[1] for s in ss], M), mk)
        for q, (mc, md, _) in enumerate(ss):
            r = _guided_scalar(src, scene, v0, SU.SurrogateConfig(m_col=mc, m_dac=md, use_human_mask=mk))
            assert torch.abs(outb[q * M:(q + 1) * M] - r).max().item() <= 1e-9


@real
def test_near_object_prefilter_is_exact():
    import torch
    torch.set_num_threads(1)
    for i in (0, 2):
        scene, src, v0 = _real_scene(i)
        keep = R.near_objects(scene, src)
        assert 0 < int(keep.sum()) <= scene.boxes.shape[1]
        sub = R.sub_scene(scene, keep)
        assert sub.boxes.shape[1] == int(keep.sum()) and sub.human_overlap.shape[1] == int(keep.sum())
        M = len(src)
        mc = np.concatenate([np.full(M, 0.0), np.full(M, 0.3)])
        md = np.concatenate([np.full(M, 0.2), np.full(M, 0.0)])
        a, _ = R.guided_multi(src.repeat(2, 1, 1), scene, v0, mc, md, True)
        b, _ = R.guided_multi(src.repeat(2, 1, 1), sub, v0, mc, md, True)
        assert np.array_equal(a.numpy().astype(np.float32), b.numpy().astype(np.float32))
