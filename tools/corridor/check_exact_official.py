#!/usr/bin/env python
"""km 10/19 (C): exact_verdict.py vs the official PDMScorer (SSR-v2 navsim v1 code in this repo) on the same
official-simulator states, for raw anchors (+ noise) and for dumped model candidates.
  $PY tools/corridor/check_exact_official.py <exact dir> <n tokens> [dump dir]"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)
from navsim.agents.para_ssr.modules import exact_verdict as EV  # noqa: E402
from navsim.agents.para_ssr.modules.torch_tracker import track  # noqa: E402


def main():
    import pandas as pd
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from tools.corridor.build_corridor_dt01 import load_mc
    from tools.ck.data.common import MC_ROOT
    X, n = sys.argv[1], int(sys.argv[2])
    dump = sys.argv[3] if len(sys.argv) > 3 else None
    torch.set_num_threads(4)
    toks = pd.read_parquet(f"{CK}/resources/km_corridor/packed/navtest/tokens.parquet").token.astype(str).to_numpy()
    where = {}
    for log in os.scandir(MC_ROOT["navtest"]):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = Path(t.path) / "metric_cache.pkl"
    tab = {k: (np.load(f"{X}/navtest/{k}.npy", mmap_mode="r"), np.load(f"{X}/navtest/{k}_off.npy")) for k in EV.TABLES}
    ok = np.load(f"{X}/navtest/ok.npy")
    anc = np.load(f"{CK}/resources/km_corridor/trajectory_anchors_256.npy").astype(np.float64)
    T = np.load(f"{dump}/navtest/traj.npy", mmap_mode="r") if dump else None
    fs = TrajectorySampling(num_poses=40, interval_length=0.1)
    rng = np.random.default_rng(0)
    res = {}
    rows = np.flatnonzero(ok)[:n]
    for j, r in enumerate(rows):
        mc = load_mc(where[toks[r]])
        ies = mc.ego_state
        sets = {"anchor": anc, "anchor+noise": anc + rng.normal(0, 1, anc.shape) * np.array([1.0, 0.5, 0.05])}
        if T is not None:
            sets["model"] = np.asarray(T[r], np.float64)
        t = {k: v[None] for k, v in EV.pack_token(*(np.asarray(v[o[r]:o[r + 1]], np.float32) for v, o in tab.values())).items()}
        d = ies.dynamic_car_state
        init = torch.tensor([d.rear_axle_velocity_2d.x, d.rear_axle_velocity_2d.y, d.rear_axle_acceleration_2d.x,
                             d.rear_axle_acceleration_2d.y, 0, 0, 0], dtype=torch.float64)
        for name, tr in sets.items():
            arr = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(x.astype(np.float32)), ies), fs, ies.time_point) for x in tr])
            S = PDMSimulator(fs).simulate_proposals(arr, ies)
            sc = PDMScorer(fs)
            sc._reset(S, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
            sc._calculate_ego_area(); sc._calculate_no_at_fault_collision(); sc._calculate_drivable_area_compliance()
            sc._calculate_ttc(); sc._calculate_progress()
            st = track(torch.tensor(tr), init[None].repeat(len(tr), 1), full=True).float()[None]
            ex = EV.exact_verdicts(st, t)
            off = {"nc": sc._multi_metrics[MultiMetricIndex.NO_COLLISION], "dac": sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA],
                   "ttc": sc._weighted_metrics[WeightedMetricIndex.TTC], "comfort": ego_is_comfortable(S, np.arange(41) * 0.1).all(-1)}
            for k, o in off.items():
                res.setdefault((name, k), []).append(ex[k][0].numpy() == o)
            res.setdefault((name, "prog_err"), []).append(np.abs(ex["progress"][0].numpy() - sc._progress_raw))
        if j % 20 == 0:
            print(f"{j}/{len(rows)}", flush=True)
    for (name, k), v in sorted(res.items()):
        v = np.concatenate(v)
        print(f"{name:<13} {k:<8} " + (f"exact value agreement {v.mean() * 100:8.4f}%  (n {v.size})" if k != "prog_err"
                                       else f"|err| p99 {np.percentile(v, 99):.4f} m max {v.max():.4f} m"))


if __name__ == "__main__":
    main()
