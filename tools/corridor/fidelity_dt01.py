#!/usr/bin/env python
"""km 10/15: GT corridor v2 (agent occupancy every 0.1 s, build_corridor_dt01.py) -> NC verdict fidelity on tracked states.
Same 600 navtrain_val tokens / 256 raw anchors / order as fidelity_pred.py; official labels are re-used from its npz.
NC verdict at every 0.1 s step (t = 0.1 .. 4.0 s): footprint sample points (5 x 3) in an occupied cell -> collision.
Also a swept variant: 8 planes, each the union of the five 0.1 s planes in (t-0.5, t], checked at every 0.1 s step.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

from tools.corridor.fidelity import ANCH, to_ego  # noqa: E402
from tools.corridor.fidelity_pred import K, PRED, ROOT, SPLIT, body_pts, nearest  # noqa: E402

D2 = f"{K}/corridor_dt01/{SPLIT}"


def work(job):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from tools.corridor.build_corridor import load_mc

    rows, paths = job
    A = np.load(ANCH).astype(np.float64)
    GD = np.load(f"{D2}/dyn_occ.npy", mmap_mode="r")
    fs = TrajectorySampling(num_poses=40, interval_length=0.1)
    sim = PDMSimulator(fs)
    out = []
    for r, p in zip(rows, paths):
        mc = load_mc(p)
        ego = mc.ego_state
        st = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(t.astype(np.float32)), ego), fs,
                                               ego.time_point) for t in A], 0)
        tr = sim.simulate_proposals(st, ego)
        bx, by = to_ego(ego, tr[:, 1:, 0], tr[:, 1:, 1])
        bp = body_pts(bx, by, tr[:, 1:, 2] - ego.rear_axle.heading)                  # [256, 40, 15, 2]
        o = np.unpackbits(np.asarray(GD[r]), axis=-1)                                  # [40, 100, 200]
        sw = np.stack([o[5 * j: 5 * j + 5].max(0) for j in range(8)])  # (t-0.5, t]
        hit, hsw = np.zeros(len(A), bool), np.zeros(len(A), bool)
        for j in range(40):
            hit |= (nearest(o[j], bp[:, j]) > 0).any(-1)
            hsw |= (nearest(sw[j // 5], bp[:, j]) > 0).any(-1)
        out.append((~hit, ~hsw))
    return out


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd
    from tools.ck.data.common import MC_ROOT
    toks = pd.read_parquet(f"{ROOT}/packed/{SPLIT}/tokens.parquet").token.astype(str).to_numpy()
    ok = np.load(f"{K}/corridor/{SPLIT}/ok.npy")
    for v in PRED.values():
        ok &= np.load(f"{v}/ok.npy")
    rows = np.sort(np.random.default_rng(0).choice(np.where(ok)[0], 600, replace=False))
    where = {}
    for log in os.scandir(MC_ROOT[SPLIT]):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = str(Path(t.path) / "metric_cache.pkl")
    jobs = [(rows[i:i + 10], [where[toks[r]] for r in rows[i:i + 10]]) for i in range(0, len(rows), 10)]
    agg = []
    with ProcessPoolExecutor(8) as ex:
        for res in ex.map(work, jobs):
            agg += res
    F = np.load(f"{K}/km_runs/corridor_head/fidelity_pred_{SPLIT}.npz")
    nc, inr = F["nc"], F["in"]
    R = {"NG_dt05": F["NG"], "NG_dt01": np.concatenate([x[0] for x in agg]), "NG_swept": np.concatenate([x[1] for x in agg])}
    for k, p in R.items():
        for nm, m in (("all", np.ones_like(inr)), ("in-range", inr)):
            print(f"  {k:9s} {nm:8s} agree {100 * (p[m] == nc[m]).mean():6.2f}%  false-pass {100 * (p[m] & ~nc[m]).mean():5.2f}%  "
                  f"false-fail {100 * (~p[m] & nc[m]).mean():5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
