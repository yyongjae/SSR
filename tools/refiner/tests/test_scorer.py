"""Tests for tools/refiner/score_trajectories.py (CPU, < 2 min).

Batched score_token must equal the single-trajectory official pdm_score (plain navsim PDMSimulator / PDMScorer
from default_scoring_parameters.yaml) with float equality on nc, dac, ddc, ep, ttc, comfort, pdms.  Tokens include the
official-csv DDC<1 token and NC=0.5 tokens of para_ssr_interaction_final so that the DDC / static-object branches and
the EP threshold branch (stop draft) are exercised.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_scorer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import score_trajectories as ST  # noqa: E402
import check_scorer_equivalence as CE  # noqa: E402

METRICS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms")
DDC_TOKEN = "59ea5c8c067e57e7"  # official csv DDC < 1 (model trajectory)
K13_TOKEN = "43311faf65e25505"  # navtest, PDM-Closed progress*mult 0.90 m -> EP threshold branch for slow/stop drafts
E_STOP_TOKEN = "16994a6affc45d81"  # E navtrain cache, PDM-Closed stationary (progress 0 m)


@pytest.fixture(scope="module")
def env():
    import pickle
    import pandas as pd

    t = np.load(CE.HUMAN_TABLE, allow_pickle=True)
    human = {tok: h for tok, h in zip(t["tokens"], t["human"])}
    logs = {tok: lg for tok, lg in zip(t["tokens"], t["logs"])}
    model = pickle.load(open(CE.MODEL_PKL, "rb"))["trajectories"]
    off = pd.read_csv(CE.OFFICIAL_CSV, index_col=0)
    off = off[off.token != "average"]
    nc_half = off[off.no_at_fault_collisions == 0.5].token.tolist()[:2]
    rng = np.random.default_rng(0)
    rand = [str(x) for x in rng.choice(off.token.values, 5, replace=False)]
    toks = [DDC_TOKEN] + nc_half + rand
    CE.W["sim_r"], CE.W["sc_r"] = ST.build_simulator_scorer(record=False)
    sim, sc = ST.build_simulator_scorer(record=True)
    return dict(human=human, logs=logs, model=model, toks=toks, sim=sim, sc=sc)


def _mc_path(env, tok):
    p = ST.locate_metric_cache(tok, env["logs"][tok], [CE.NAVTEST_MC])
    assert p is not None, tok
    return p


def test_scoring_config_in_effect():
    """The official run composes default_scoring_parameters.yaml: progress_distance_threshold 5.0 (dataclass
    default 0.1 is NOT in effect); the archived eval config used by cf_common/rescore_attr is identical."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorerConfig

    _, sc = ST.build_simulator_scorer(record=False)
    assert sc._config.progress_distance_threshold == 5.0
    assert PDMScorerConfig().progress_distance_threshold == 0.1
    arch = OmegaConf.load(ST.ARCHIVED_EVAL_CFG)
    arch_sc = instantiate(OmegaConf.create({"proposal_sampling": arch.proposal_sampling, "scorer": arch.scorer}).scorer)
    assert repr(arch_sc._config) == repr(sc._config)
    assert sc.proposal_sampling == arch_sc.proposal_sampling
    assert (sc.proposal_sampling.num_poses, sc.proposal_sampling.interval_length) == (40, 0.1)
    assert "default_scoring_parameters" in (ST.ROOT / "navsim/planning/script/config/pdm_scoring/"
                                            "default_run_pdm_score.yaml").read_text()


def test_batched_equals_single_navtest(env):
    """8 navtest tokens x 6 trajectories (model, human, retimed, lateral): exact equality; the naive batched EP / DDC
    must differ somewhere (otherwise the test would not exercise the fix)."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

    n, naive_diff, ddc_fail, nc_half = 0, 0, 0, 0
    for tok in env["toks"]:
        mc = ST.load_metric_cache(_mc_path(env, tok))
        named = CE.make_set("k6", np.asarray(env["model"][tok], np.float32), env["human"][tok].astype(np.float32), mc)
        trajs = np.stack([t for _, t in named])
        res = ST.score_token(mc, trajs, env["sim"], env["sc"])
        naive_ep = env["sc"]._weighted_metrics[WeightedMetricIndex.PROGRESS, 1:].copy()
        naive_ddc = env["sc"]._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, 1:].copy()
        for k, ((name, tr), b) in enumerate(zip(named, res)):
            r = CE.ref_single(mc, tr)
            for m in METRICS:
                assert b[m] == r[m], (tok, name, m, b[m], r[m])
            assert b["rec_ok"]
            assert b["mult"] == b["nc"] * b["dac"] * b["ddc"]
            assert (b["nc_time_idx"] >= 0) == (b["nc"] < 1) and (b["nc_track"] != "") == (b["nc"] < 1)
            assert (b["dac_time_idx"] >= 0) == (b["dac"] < 1)
            naive_diff += int(naive_ep[k] != r["ep"] or naive_ddc[k] != r["ddc"])
            ddc_fail += int(b["ddc"] < 1)
            nc_half += int(b["nc"] == 0.5)
            n += 1
    assert n == 6 * len(env["toks"])
    assert naive_diff > 0 and ddc_fail > 0 and nc_half > 0, (naive_diff, ddc_fail, nc_half)


def test_k13_and_e_navtrain_cache(env):
    """K = 13 (> DDC horizon + 1 proposals) on navtest and PDM-Closed-derived drafts on E's navtrain cache."""
    tok = K13_TOKEN
    mc = ST.load_metric_cache(_mc_path(env, tok))
    named = CE.make_set("k13", np.asarray(env["model"][tok], np.float32), env["human"][tok].astype(np.float32), mc)
    e_idx = ST.index_metric_cache([CE.E_MC], None)
    e_tok = [E_STOP_TOKEN, sorted(e_idx)[0]]
    cases = [(mc, named)] + [(m, CE.make_set("e6", None, None, m))
                             for m in (ST.load_metric_cache(e_idx[t]) for t in e_tok)]
    thr_branch = 0
    for mc_, nm in cases:
        res = ST.score_token(mc_, np.stack([t for _, t in nm]), env["sim"], env["sc"])
        assert len(res) == len(nm)
        for (name, tr), b in zip(nm, res):
            r = CE.ref_single(mc_, tr)
            for m in METRICS:
                assert b[m] == r[m], (name, m, b[m], r[m])
            thr_branch += int(max(b["pdm_progress_eff"], b["raw_progress"] * b["mult"]) <= 5.0)
    assert thr_branch > 0  # the stop draft hits the progress_distance_threshold branch


def test_single_trajectory_input_and_pdm_fields(env):
    tok = env["toks"][4]
    mc = ST.load_metric_cache(_mc_path(env, tok))
    h = env["human"][tok].astype(np.float64)  # float64 input is cast to float32 like the official Trajectory
    (a,) = ST.score_token(mc, h, env["sim"], env["sc"], return_states=True)
    (b,) = ST.score_token(mc, h.astype(np.float32)[None], env["sim"], env["sc"])
    assert all(a[m] == b[m] for m in METRICS)
    assert a["states"].shape == (41, 11)
    assert a["pdm_progress_eff"] == a["pdm_raw_progress"] * a["pdm_nc"] * a["pdm_dac"] * a["pdm_ddc"]
    assert set(ST.OUT_KEYS) <= set(a)


def test_nc_record_matches_cf_common(env):
    """nc_track / nc_time_idx equal cf_common.primary_nc of cf_common.score (recorded official events)."""
    sys.path.insert(0, str(ST.ROOT / "report/collision_counterfactual/counterfactual"))
    import cf_common

    cf_common.init_worker()
    checked = 0
    for tok in env["toks"][:3]:
        mc = ST.load_metric_cache(_mc_path(env, tok))
        tr = np.asarray(env["model"][tok], np.float32)
        (b,) = ST.score_token(mc, tr, env["sim"], env["sc"])
        c, nc_ev, _, _ = cf_common.score(mc, tr)
        for m in METRICS:
            assert b[m] == c[m]
        pn = cf_common.primary_nc(nc_ev)
        assert b["nc_time_idx"] == (pn["time_idx"] if pn else -1)
        assert b["nc_track"] == (pn["track"] if pn else "")
        checked += int(pn is not None)
    assert checked > 0


def test_cli_resume_and_packed(env, tmp_path):
    import pandas as pd

    toks = env["toks"][4:7]
    ddir = tmp_path / "drafts"
    ddir.mkdir()
    for i, tok in enumerate(toks):
        h = env["human"][tok].astype(np.float32)
        d = np.stack([h, CE.retime(h, 0.8), CE.lateral(h, 1.0)])
        np.savez(ddir / f"{tok}.npz", drafts=d, family=np.array([0, 1, 9], np.int8))
    tp = tmp_path / "tokens.parquet"
    pd.DataFrame(dict(token=toks + ["ffffffffffffffff"], log=[env["logs"][t] for t in toks] + ["nolog"])).to_parquet(tp)
    out = tmp_path / "scores.parquet"
    args = ["--drafts", str(ddir), "--tokens", str(tp), "--out", str(out), "--workers", "1", "--shard-size", "2",
            "--mc-roots", str(CE.NAVTEST_MC)]
    ST.main(args)
    assert not out.exists()  # the unknown token has no metric cache -> incomplete, shards kept
    shards = sorted((tmp_path / "scores.shards").glob("part-*.parquet"))
    assert len(shards) == 2
    pd.DataFrame(dict(token=toks, log=[env["logs"][t] for t in toks])).to_parquet(tp)
    ST.main(args)  # resume: nothing left to score, merge
    assert sorted((tmp_path / "scores.shards").glob("part-*.parquet")) == shards
    df = pd.read_parquet(out)
    assert len(df) == 9 and (df.k >= 0).all() and list(df.family[:3]) == [0, 1, 9]
    for tok in toks:
        with np.load(ddir / f"{tok}.npz") as z:
            ref = ST.score_token(ST.load_metric_cache(_mc_path(env, tok)), z["drafts"], env["sim"], env["sc"])
        got = df[df.token == tok].sort_values("k")
        for k, r in enumerate(ref):
            for m in METRICS + ("raw_progress", "pdm_progress_eff"):
                assert got[m].iloc[k] == r[m]
    # packed npz source gives the same drafts
    pk = tmp_path / "packed.npz"
    np.savez(pk, tokens=np.array(toks), drafts=np.stack([np.load(ddir / f"{t}.npz")["drafts"] for t in toks]))
    src = ST.DraftSource(str(pk))
    assert src.tokens() == sorted(toks)
    assert np.array_equal(src.get(toks[1])[0], np.load(ddir / f"{toks[1]}.npz")["drafts"])
    gsrc = ST.DraftSource(str(ddir / "*.npz"))
    assert gsrc.tokens() == sorted(toks)
