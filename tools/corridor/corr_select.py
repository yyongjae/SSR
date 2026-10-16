#!/usr/bin/env python
"""km 10/15: corridor selection over ALL 256 candidates of a dump + OFFICIAL PDMS of the chosen trajectories.

Score of candidate k (v2's weighted reward, w = 0.1 / 0.5 / 0.5 / 1.0, with NC / DAC from the corridor verdict of
the TRACKED candidate instead of the anchor-trained heads; EP / TTC / comfort stay the learned heads):
  replace : 0.1 log im + 0.5 log p_nc + 0.5 log p_dac + log(5 ttc + 2 c + 5 ep)
  multiply: v2 final + 0.5 log p_nc + 0.5 log p_dac
  p_dac = sigmoid(min corner SDF / tau), p_nc = 1 - max footprint occupancy (pred) ; gt_* = the same with the GT
  corridor (oracle, navtrain splits only, never deployed).
Pre-declared variants: v2 | {replace, multiply} x tau {0.25, 0.5} (pred) | gt_replace, gt_multiply (tau 0.25)
  | v3 (km 10/15 (4), verdict with comfort/TTC): v3_veto_ttc{head,corridor} and the gt_ oracles (scores_v3).
Official label of each chosen trajectory: tools/refiner/score_trajectories.score_token (pinned official scorer).

  $PY tools/corridor/corr_select.py --dump <dir> --split navtrain_val --n 3000 --workers 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

ROOT = os.environ.get("KM_CORR_ROOT", "/home/external-user/ssd/kyungmin_ck/km_runs/ck2e2e_bevkd/eval/root")  # parent of packed/<split>/tokens.parquet
ANCHOR_FILE = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy"
EPS = 1e-6


def scores(D, V, rows, src, mode, tau):
    sim = np.asarray(D["sim"][rows], np.float32)                                         # [n, 5, 256]
    im = np.asarray(D["im"][rows], np.float32)
    fin = np.asarray(D["final"][rows], np.float32)
    nc, dac, ep, ttc, cf = (sim[:, i] for i in range(5))
    smin = V[f"{src}_sdf_min"][rows]
    occ = V["pred_occ_max"][rows] if src == "pred" else V["gt_occ_hit"][rows]
    p_dac = 1 / (1 + np.exp(-np.clip(smin / tau, -50, 50)))
    p_nc = np.clip(1 - occ, 0, 1)
    if mode == "replace":
        return (0.1 * np.log(im + EPS) + 0.5 * np.log(p_nc + EPS) + 0.5 * np.log(p_dac + EPS)
                + np.log(5 * ttc + 2 * cf + 5 * ep + EPS))
    return fin +0.5 * np.log(p_nc + EPS) + 0.5 * np.log(p_dac + EPS)


def scores_v3(D, V, rows, src, ttc_src, ep_src="head", nc_mode="hard"):
    """km 10/15 (4) v3: hard veto by the corridor (DAC: min corner SDF > 0, NC: max footprint occupancy < 0.5; no
    candidate passes -> no veto), comfort = exact (tracked states, official thresholds), then v2's weighted reward
    0.1 log im + 0.5 log NC_head + 0.5 log DAC_head + log(5 EP + 5 TTC + 2 C).
    ttc_src: head | corridor (head x [corridor TTC verdict passes]).
    nc_mode (km 10/16): hard | soft — soft = veto by DAC only, NC as + 0.5 log(1 - occupancy) (corr30_v2 ep30 val:
    the NC hard veto alone cost -1.1 PDMS, the DAC hard veto alone +0.8)."""
    sim = np.asarray(D["sim"][rows], np.float32)
    im = np.asarray(D["im"][rows], np.float32)
    nc, dac, ep, ttc = (sim[:, i] for i in range(4))
    cf = V["comfort"][rows]
    if src == "pred":
        ok, thit = (V["pred_sdf_min"][rows] > 0) & (V["pred_occ_max"][rows] < 0.5), V["pred_ttc_max"][rows]
    else:
        ok, thit = (V["gt_sdf_min"][rows] > 0) & (V["gt_occ_hit01"][rows] < 0.5), V["gt_ttc_hit"][rows]
    occ = V["pred_occ_max"][rows] if src == "pred" else V["gt_occ_hit01"][rows]
    if nc_mode == "soft":
        ok = V[f"{src}_sdf_min"][rows] > 0
    ok = np.where(ok.any(1, keepdims=True), ok, True)
    if ttc_src == "corridor":
        ttc = ttc * (thit < 0.5)
    elif ttc_src == "corridorsoft":                 # km 10/16: TTC head x (1 - corridor TTC hit probability)
        ttc = ttc * np.clip(1 - thit, 0, 1)
    if ep_src == "transport":
        # EP of the anchor (head) carried to the submitted trajectory by the predicted route progress:
        # EP(anchor + offset) = EP(anchor) * progress(anchor + offset) / progress(anchor)  (no normaliser needed)
        pc, pa = V["prog_cand"][rows], V["prog_anchor"][rows]
        ratio = np.where(pa > 1.0, np.clip(pc, 0, None) / np.maximum(pa, 1.0), 1.0)
        ep = np.clip(ep * ratio, 0, 1)
    base = (0.1 * np.log(im + EPS) + 0.5 * np.log(nc + EPS) + 0.5 * np.log(dac + EPS)
            + np.log(5 * ttc + 2 * cf + 5 * ep + EPS))
    if nc_mode == "soft":
        base = base + 0.5 * np.log(np.clip(1 - occ, 0, 1) + EPS)
    return np.where(ok, base, -1e9)


def work(job):
    from tools.ck.data import label_cands as LC
    ST = LC.ST
    ST.get_simulator_scorer()
    out = []
    for path, trajs in job:
        try:
            res = ST.score_token(ST.load_metric_cache(path), trajs.astype(np.float32))
            out.append([[d[c] for c in ("pdms", "nc", "dac", "ep", "ttc", "comfort")] for d in res])
        except Exception as e:  # noqa: BLE001
            out.append(repr(e)[:200])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--split", default="navtrain_val")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--verdict", default="verdict.npz", help="verdict file (km 10/16: verdict_<occ>.npz for test A)")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd
    from tools.ck.data.common import MC_ROOT
    d = Path(a.dump) / a.split
    D = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("traj", "sim", "im", "final")}
    V = dict(np.load(d / a.verdict))
    rows = V["rows"]
    if a.n and a.n < len(rows):
        rows = np.sort(np.random.default_rng(0).choice(rows, a.n, replace=False))
    has_gt = "gt_sdf_min" in V
    if has_gt:
        rows = rows[np.isfinite(V["gt_sdf_min"][rows]).all(1)]
    var = {"v2": np.asarray(D["final"][rows]).argmax(1)}
    for mode in ("replace", "multiply"):
        for tau in (0.25, 0.5):
            var[f"pred_{mode}_t{tau}"] = scores(D, V, rows, "pred", mode, tau).argmax(1)
        if has_gt:
            var[f"gt_{mode}"] = scores(D, V, rows, "gt", mode, 0.25).argmax(1)
    if "comfort" in V:
        for src in ("pred",) + (("gt",) if has_gt else ()):
            for ts in ("head", "corridor", "corridorsoft"):
                var[f"{'gt_' if src == 'gt' else ''}v3_veto_ttc{ts}"] = scores_v3(D, V, rows, src, ts).argmax(1)
                if "prog_cand" in V:
                    var[f"{'gt_' if src == 'gt' else ''}v3_veto_ttc{ts}_eptr"] = \
                        scores_v3(D, V, rows, src, ts, "transport").argmax(1)
                pre = f"{'gt_' if src == 'gt' else ''}v3_dacveto_ncsoft_ttc{ts}"
                var[pre] = scores_v3(D, V, rows, src, ts, nc_mode="soft").argmax(1)
                if "prog_cand" in V:
                    var[pre + "_eptr"] = scores_v3(D, V, rows, src, ts, "transport", "soft").argmax(1)
    toks = pd.read_parquet(f"{ROOT}/packed/{a.split}/tokens.parquet").token.astype(str).to_numpy()
    where = {}
    for log in os.scandir(MC_ROOT[a.split]):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = str(Path(t.path) / "metric_cache.pkl")
    names = list(var)
    C = np.stack([var[k] for k in names], 1)                                              # [n, V]
    uniq = [np.unique(c) for c in C]
    items = [(where[toks[r]], np.asarray(D["traj"][r], np.float32)[u]) for r, u in zip(rows, uniq)]
    jobs = [items[i:i + 20] for i in range(0, len(items), 20)]
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for x in ex.map(work, jobs):
            res += x
    ok = np.array([not isinstance(x, str) for x in res])
    print(f"{a.split}: {len(rows)} tokens, scored {ok.sum()} (errors {(~ok).sum()}{': ' + [x for x in res if isinstance(x, str)][0] if (~ok).any() else ''})")
    summ = {}
    for j, nm in enumerate(names):
        vals = np.array([res[i][list(uniq[i]).index(C[i, j])] for i in range(len(rows)) if ok[i]], np.float64)
        summ[nm] = dict(zip(("pdms", "nc", "dac", "ep", "ttc", "comfort"), np.round(100 * vals.mean(0), 2).tolist()))
        summ[nm]["same_as_v2"] = round(100 * float((C[ok, j] == C[ok, 0]).mean()), 1)
        print(f"  {nm:22s} PDMS {summ[nm]['pdms']:6.2f}  NC {summ[nm]['nc']:6.2f}  DAC {summ[nm]['dac']:6.2f}  "
              f"EP {summ[nm]['ep']:6.2f}  TTC {summ[nm]['ttc']:6.2f}  C {summ[nm]['comfort']:6.2f}  "
              f"(same choice as v2 {summ[nm]['same_as_v2']}%)")
    tag = "" if a.verdict == "verdict.npz" else "_" + Path(a.verdict).stem.replace("verdict_", "")
    json.dump(summ, open(d / f"select_{a.n}{tag}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
