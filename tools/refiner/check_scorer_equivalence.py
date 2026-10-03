#!/usr/bin/env python
"""Equivalence check: batched score_trajectories.score_token vs the single-trajectory official scorer (IMPL_SPEC §3.3).

For every token, K trajectories are scored (a) together by score_token ([PDM-Closed, traj_1..K] in one call) and
(b) one at a time by
    ref  = navsim.evaluate.pdm_score.pdm_score with the PLAIN PDMSimulator / PDMScorer from
           default_scoring_parameters.yaml (separate instances), and
    cf   = cf_common.score (rescore_attr recording classes from the archived eval hydra config; navtest paths),
and nc, dac, ddc, ep, ttc, comfort, pdms are compared with float equality (==, i.e. bitwise up to +-0).
Also recorded: the naive-batching values (the scorer's own EP / DDC rows of the K+1 call) to show what the fix
changes, nc_time_idx / nc_track vs cf_common.primary_nc, and for the model trajectory on navtest the values in the
official evaluation csv of para_ssr_interaction_final (2026.09.17.00.09.41.csv).

Trajectory sets (N frame poses [8, 3] at t = 0.5 .. 4 s; see retime / lateral / pdm_closed_poses):
  navtest K=6  : model, human, model x0.7 (arc-length retime), human x1.25, human lateral +1.5 m, model lateral -1.5 m
  navtest K=13 : model, human, model x0.5, human x0.8, model x1.1, human x1.4, human lat +0.75, model lat -0.75,
                 human lat +2.0, model lat -2.0, human x0.8 + lat +1.0, stop (all zeros), PDM-Closed poses
  E navtrain K=6 (metric cache of E, no model output): PDM-Closed poses, x0.7, x1.25, lat +1.5, lat -1.5, stop

  python tools/refiner/check_scorer_equivalence.py --n-navtest 1000 --n-k13 100 --n-e 200 --workers 2
Outputs: /home/external-user/ssd/yongjae_refiner/scores/equivalence/rows.parquet (per trajectory) and
report/refiner_T/scorer_equivalence.json (summary).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_trajectories as ST  # noqa: E402

ROOT = ST.ROOT
NAVTEST_MC = ROOT / "data/exp/metric_cache"
E_MC = ROOT / "report/cause_and_correction_tests/E_train_split_feasibility/metric_cache"
MODEL_PKL = ROOT / "work_dirs/eval/para_ssr_interaction_final_navtest_trajectories.pkl"
OFFICIAL_CSV = ROOT / "work_dirs/eval/para_ssr_interaction_final/2026.09.17.00.09.41.csv"
HUMAN_TABLE = ROOT / "report/head_ablation_scenes/table.npz"
OUT_ROWS = ST.DATA / "scores/equivalence/rows.parquet"
OUT_JSON = ROOT / "report/refiner_T/scorer_equivalence.json"
METRICS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms")
T_KF = 0.5 * np.arange(1, 9)


# ----------------------------------------------------------------------------------------------- perturbations
def _knots(traj):
    """(9, 3) float64 [origin; poses] with unwrapped heading."""
    p = np.concatenate([np.zeros((1, 3)), np.asarray(traj, np.float64)], 0)
    p[:, 2] = np.unwrap(p[:, 2])
    return p


def retime(traj, f):
    """Same path, arc length scaled by f at every keyframe (s_new(t) = f * s(t)); beyond the path end the last
    segment is extended straight (heading kept)."""
    p = _knots(traj)
    seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_mono = s + 1e-9 * np.arange(len(s))  # strictly increasing for np.interp (zero-length segments)
    out = np.empty((8, 3))
    nz = np.flatnonzero(seg > 1e-3)
    u = (p[nz[-1] + 1, :2] - p[nz[-1], :2]) / seg[nz[-1]] if len(nz) else np.array([1.0, 0.0])
    for k in range(8):
        tgt = f * s[k + 1]
        if tgt <= s[-1]:
            out[k, 0] = np.interp(tgt, s_mono, p[:, 0])
            out[k, 1] = np.interp(tgt, s_mono, p[:, 1])
            out[k, 2] = np.interp(tgt, s_mono, p[:, 2])
        else:
            out[k, :2] = p[-1, :2] + (tgt - s[-1]) * u
            out[k, 2] = p[-1, 2]
    return out.astype(np.float32)


def lateral(traj, d, t_ramp=2.0):
    """Offset d [m] (+ = left) along each pose's left normal, ramped in with a smoothstep over t_ramp s; heading
    rotated by atan2(lateral speed, path speed)."""
    p = _knots(traj)
    u = np.clip(T_KF / t_ramp, 0, 1)
    r = 3 * u ** 2 - 2 * u ** 3
    dr = np.where(T_KF < t_ramp, (6 * u - 6 * u ** 2) / t_ramp, 0.0)
    h = p[1:, 2]
    v = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) / 0.5
    out = np.empty((8, 3))
    out[:, 0] = p[1:, 0] - d * r * np.sin(h)
    out[:, 1] = p[1:, 1] + d * r * np.cos(h)
    out[:, 2] = h + np.arctan2(d * dr, np.maximum(v, 0.5))
    return out.astype(np.float32)


def pdm_closed_poses(mc):
    """PDM-Closed (metric_cache.trajectory) rear-axle poses at t0 + 0.5 k s in the N frame, [8, 3] float32."""
    from nuplan.common.actor_state.state_representation import TimePoint
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    t0 = mc.ego_state.time_point.time_us
    tr = mc.trajectory
    ts = [min(max(int(t0 + 5e5 * k), tr.start_time.time_us), tr.end_time.time_us) for k in range(1, 9)]
    abs_ = np.array([[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading]
                     for s in tr.get_state_at_times([TimePoint(t) for t in ts])], np.float64)
    rel = convert_absolute_to_relative_se2_array(mc.ego_state.rear_axle, abs_)
    return rel.astype(np.float32)


def make_set(kind, model, human, mc):
    if kind == "k6":
        return [("model", model), ("human", human), ("model_x0.7", retime(model, 0.7)),
                ("human_x1.25", retime(human, 1.25)), ("human_lat+1.5", lateral(human, 1.5)),
                ("model_lat-1.5", lateral(model, -1.5))]
    if kind == "k13":
        return [("model", model), ("human", human), ("model_x0.5", retime(model, 0.5)),
                ("human_x0.8", retime(human, 0.8)), ("model_x1.1", retime(model, 1.1)),
                ("human_x1.4", retime(human, 1.4)), ("human_lat+0.75", lateral(human, 0.75)),
                ("model_lat-0.75", lateral(model, -0.75)), ("human_lat+2.0", lateral(human, 2.0)),
                ("model_lat-2.0", lateral(model, -2.0)), ("human_x0.8_lat+1.0", lateral(retime(human, 0.8), 1.0)),
                ("stop", np.zeros((8, 3), np.float32)), ("pdm_closed", pdm_closed_poses(mc))]
    if kind == "e6":
        pc = pdm_closed_poses(mc)
        return [("pdm_closed", pc), ("pdm_x0.7", retime(pc, 0.7)), ("pdm_x1.25", retime(pc, 1.25)),
                ("pdm_lat+1.5", lateral(pc, 1.5)), ("pdm_lat-1.5", lateral(pc, -1.5)),
                ("stop", np.zeros((8, 3), np.float32))]
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------- worker
W = {}


def init(use_cf):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pickle
    W["sim_b"], W["sc_b"] = ST.build_simulator_scorer(record=True)
    W["sim_r"], W["sc_r"] = ST.build_simulator_scorer(record=False)
    W["model"] = pickle.load(open(MODEL_PKL, "rb"))["trajectories"]
    t = np.load(HUMAN_TABLE, allow_pickle=True)
    W["human"] = {tok: h for tok, h in zip(t["tokens"], t["human"])}
    W["use_cf"] = use_cf
    if use_cf:
        sys.path.insert(0, str(ROOT / "report/collision_counterfactual/counterfactual"))
        import cf_common
        cf_common.init_worker()
        W["cf"] = cf_common


def ref_single(mc, poses):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    sim, sc = W["sim_r"], W["sc_r"]
    r = pdm_score(mc, Trajectory(np.asarray(poses, np.float32), TrajectorySampling(num_poses=8, interval_length=0.5)),
                  sim.proposal_sampling, sim, sc)
    return dict(nc=float(r.no_at_fault_collisions), dac=float(r.drivable_area_compliance),
                ddc=float(r.driving_direction_compliance), ep=float(r.ego_progress),
                ttc=float(r.time_to_collision_within_bound), comfort=float(r.comfort), pdms=float(r.score))


def run(task):
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

    kind, tok, mcp = task
    try:
        t0 = time.time()
        mc = ST.load_metric_cache(mcp)
        t_load = time.time() - t0
        model = np.asarray(W["model"][tok], np.float32) if kind != "e6" else None
        human = np.asarray(W["human"][tok], np.float32) if kind != "e6" else None
        named = make_set(kind, model, human, mc)
        trajs = np.stack([t for _, t in named]).astype(np.float32)
        t1 = time.time()
        res = ST.score_token(mc, trajs, W["sim_b"], W["sc_b"])
        t_batch = time.time() - t1
        naive_ep = W["sc_b"]._weighted_metrics[WeightedMetricIndex.PROGRESS, 1:].copy()
        naive_ddc = W["sc_b"]._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, 1:].copy()
        rows = []
        t_ref = t_cf = 0.0
        for k, ((name, tr), b) in enumerate(zip(named, res)):
            t2 = time.time()
            r = ref_single(mc, tr)
            t_ref += time.time() - t2
            row = dict(kind=kind, token=tok, k=k, name=name, K=len(named), naive_ep=float(naive_ep[k]),
                       naive_ddc=float(naive_ddc[k]), t_load=t_load, t_batch=t_batch)
            row.update({f"b_{m}": b[m] for m in ST.OUT_KEYS})
            row.update({f"r_{m}": r[m] for m in METRICS})
            if W["use_cf"]:
                cf = W["cf"]
                t3 = time.time()
                c, nc_ev, _, _ = cf.score(mc, tr)
                t_cf += time.time() - t3
                row.update({f"c_{m}": c[m] for m in METRICS})
                pn = cf.primary_nc(nc_ev)
                row["c_nc_time_idx"] = pn["time_idx"] if pn else -1
                row["c_nc_track"] = pn["track"] if pn else ""
                row["c_rec_ok"] = bool(c["rec_ok"])
            rows.append(row)
        for row in rows:
            row["t_ref_total"] = t_ref
            row["t_cf_total"] = t_cf
        return rows, ""
    except Exception:
        import traceback
        return [], f"{kind} {tok}: " + traceback.format_exc()[-800:]


# ----------------------------------------------------------------------------------------------- main
def summarize(df, official):
    out = {}
    for kind, g in df.groupby("kind"):
        s = dict(n_tokens=int(g.token.nunique()), n_traj=int(len(g)), K=int(g.K.iloc[0]))
        for ref in ("r", "c"):
            if f"{ref}_nc" not in g.columns or g[f"{ref}_nc"].isna().all():
                continue
            mm = {m: int((g[f"b_{m}"] != g[f"{ref}_{m}"]).sum()) for m in METRICS}
            mx = {m: float((g[f"b_{m}"] - g[f"{ref}_{m}"]).abs().max()) for m in METRICS}
            s[f"mismatch_vs_{'pdm_score' if ref == 'r' else 'cf_common'}"] = mm
            s[f"maxabs_vs_{'pdm_score' if ref == 'r' else 'cf_common'}"] = mx
            s[f"traj_with_any_mismatch_vs_{'pdm_score' if ref == 'r' else 'cf_common'}"] = int(
                np.any([g[f"b_{m}"] != g[f"{ref}_{m}"] for m in METRICS], axis=0).sum())
        s["naive_batch_mismatch_vs_pdm_score"] = dict(
            ep=int((g.naive_ep != g.r_ep).sum()), ddc=int((g.naive_ddc != g.r_ddc).sum()))
        if "c_nc_time_idx" in g.columns:
            s["nc_time_idx_mismatch_vs_cf_primary"] = int((g.b_nc_time_idx != g.c_nc_time_idx).sum())
            s["nc_track_mismatch_vs_cf_primary"] = int((g.b_nc_track != g.c_nc_track).sum())
        s["rec_ok_false"] = int((~g.b_rec_ok.astype(bool)).sum())
        s["fail_rates"] = {m: float((g[f"b_{m}"] < 1).mean()) for m in ("nc", "dac", "ddc", "ttc", "comfort")}
        s["ep_threshold_branch"] = int((np.maximum(g.b_pdm_progress_eff, g.b_raw_progress * g.b_mult) <= 5.0).sum())
        s["ddc_half"] = int((g.b_ddc == 0.5).sum())
        s["nc_half"] = int((g.b_nc == 0.5).sum())
        tok = g.groupby("token").first()
        s["sec_per_token"] = dict(load=float(tok.t_load.mean()), batched_score=float(tok.t_batch.mean()),
                                  single_ref_total=float(tok.t_ref_total.mean()),
                                  batched_score_per_traj=float((tok.t_batch / tok.K).mean()))
        by_name = {}
        for name, h in g.groupby("name"):
            by_name[name] = dict(n=int(len(h)), fail_any=float(((h.b_nc < 1) | (h.b_dac < 1) | (h.b_ddc < 1)).mean()),
                                 pdms=float(h.b_pdms.mean()))
        s["by_name"] = by_name
        out[kind] = s
    if official is not None:
        g = df[(df.kind != "e6") & (df.name == "model")].drop_duplicates("token").merge(official, on="token")
        cols = dict(nc="no_at_fault_collisions", dac="drivable_area_compliance", ddc="driving_direction_compliance",
                    ep="ego_progress", ttc="time_to_collision_within_bound", comfort="comfort", pdms="score")
        out["model_vs_official_csv"] = dict(
            n=int(len(g)),
            mismatch_exact={m: int((g[f"b_{m}"] != g[c]).sum()) for m, c in cols.items()},
            maxabs={m: float((g[f"b_{m}"] - g[c]).abs().max()) for m, c in cols.items()},
            mismatch_gt_1e12={m: int(((g[f"b_{m}"] - g[c]).abs() > 1e-12).sum()) for m, c in cols.items()})
    return out


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-navtest", type=int, default=1000)
    ap.add_argument("--n-k13", type=int, default=100)
    ap.add_argument("--n-e", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-cf", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-rows", default=str(OUT_ROWS))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    a = ap.parse_args()
    assert a.workers <= 2, "validation runs use at most 2 workers (shared machine)"

    rng = np.random.default_rng(a.seed)
    official = pd.read_csv(OFFICIAL_CSV, index_col=0)
    official = official[official.token != "average"]
    nav_idx = ST.index_metric_cache([NAVTEST_MC])
    nav = sorted(set(nav_idx) & set(official.token))
    # stratify: all DDC<1 / NC=0.5 tokens of the official csv + random sample
    special = official[(official.driving_direction_compliance < 1) | (official.no_at_fault_collisions == 0.5)].token
    special = sorted(set(special) & set(nav))
    pick = list(special) + [t for t in rng.permutation(nav) if t not in set(special)]
    k6 = pick[: a.n_navtest]
    k13 = [t for t in rng.permutation(nav)][: a.n_k13]
    e_idx = ST.index_metric_cache([E_MC])
    e_tok = [t for t in rng.permutation(sorted(e_idx))][: a.n_e]
    tasks = [("k6", t, nav_idx[t]) for t in k6] + [("k13", t, nav_idx[t]) for t in k13] + \
            [("e6", t, e_idx[t]) for t in e_tok]
    print(f"tasks: k6 {len(k6)} (special {len(special)}), k13 {len(k13)}, e6 {len(e_tok)}; workers {a.workers}",
          flush=True)
    rows, errs = [], []
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(not a.no_cf,)) as p:
        for j, (r, e) in enumerate(p.imap_unordered(run, tasks, chunksize=4)):
            rows += r
            if e:
                errs.append(e)
                print("ERR", e[-300:], flush=True)
            if (j + 1) % 100 == 0:
                print(f"{j + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    Path(a.out_rows).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out_rows, index=False)
    summ = summarize(df, official)
    summ["errors"] = errs
    summ["wall_seconds"] = time.time() - t0
    summ["workers"] = a.workers
    summ["scorer_config"] = repr(ST.build_simulator_scorer(record=False)[1]._config)
    summ["rows"] = a.out_rows
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(a.out_json, "w"), indent=1)
    print(json.dumps(summ, indent=1)[:6000], flush=True)


if __name__ == "__main__":
    main()
