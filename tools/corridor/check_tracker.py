#!/usr/bin/env python
"""km 10/15: torch tracker (navsim/agents/para_ssr/modules/torch_tracker.py) vs the official PDMSimulator.
navtrain_val tokens, 256 raw anchors.  Reports the position / heading error of the tracked states, and the DAC
verdict (exact drivable polygons on tracked footprint corners) of both against the official label of fidelity_pred.
Two initial states for the torch tracker: full (with the logged steering angle / rate) and 'status' (vx, vy, ax, ay
only, as the model's status_feature; steering 0)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)


def main() -> int:
    import pandas as pd
    import shapely
    import torch
    from navsim.agents.para_ssr.modules.torch_tracker import track
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from tools.ck.data.common import MC_ROOT
    from tools.corridor.build_corridor import load_mc
    from tools.corridor.fidelity import ANCH, corners, to_ego
    from tools.corridor.fidelity_pred import K, PRED, ROOT, SPLIT

    n_tok = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    toks = pd.read_parquet(f"{ROOT}/packed/{SPLIT}/tokens.parquet").token.astype(str).to_numpy()
    ok = np.load(f"{K}/corridor/{SPLIT}/ok.npy")
    for v in PRED.values():
        ok &= np.load(f"{v}/ok.npy")
    rows = np.sort(np.random.default_rng(0).choice(np.where(ok)[0], 600, replace=False))
    F = np.load(f"{K}/km_runs/corridor_head/fidelity_pred_{SPLIT}.npz")
    dac_off = F["dac"].reshape(600, 256)
    where = {}
    for log in os.scandir(MC_ROOT[SPLIT]):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = str(Path(t.path) / "metric_cache.pkl")
    A = np.load(ANCH).astype(np.float64)
    fs = TrajectorySampling(num_poses=40, interval_length=0.1)
    sim = PDMSimulator(fs)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    err_p, err_h, agree = [], [], {"official_sim": [], "torch_full": [], "torch_status": []}
    for j, r in enumerate(rows[:n_tok]):
        mc = load_mc(where[toks[r]])
        ego = mc.ego_state
        st = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(t.astype(np.float32)), ego), fs,
                                               ego.time_point) for t in A], 0)
        tr = sim.simulate_proposals(st, ego)
        ox, oy = to_ego(ego, tr[..., 0], tr[..., 1])
        oh = tr[..., 2] - ego.rear_axle.heading
        dcs = ego.dynamic_car_state
        full = [dcs.rear_axle_velocity_2d.x, dcs.rear_axle_velocity_2d.y, dcs.rear_axle_acceleration_2d.x,
                dcs.rear_axle_acceleration_2d.y, ego.tire_steering_angle, dcs.tire_steering_rate, dcs.angular_velocity]
        stat = full[:4] + [0.0, 0.0, 0.0]
        res = {}
        for nm, init in (("torch_full", full), ("torch_status", stat)):
            P = torch.tensor(A, device=dev)
            I = torch.tensor([init] * len(A), device=dev, dtype=torch.float64)
            res[nm] = track(P, I).cpu().numpy()                                     # [256, 41, 3] ego frame
        tf = res["torch_full"]
        err_p.append(np.hypot(tf[..., 0] - ox, tf[..., 1] - oy).max(1))
        err_h.append(np.abs(np.arctan2(np.sin(tf[..., 2] - oh), np.cos(tf[..., 2] - oh))).max(1))
        dm = mc.drivable_area_map
        box = shapely.box(tr[..., 0].min() - 10, tr[..., 1].min() - 10, tr[..., 0].max() + 10, tr[..., 1].max() + 10)
        u = shapely.union_all([dm._geometries[i] for i in dm.query(box)])
        shapely.prepare(u)
        ra = ego.rear_axle
        c, s = np.cos(ra.heading), np.sin(ra.heading)
        for nm, arr in (("official_sim", np.stack([ox, oy, oh], -1)), ("torch_full", res["torch_full"]),
                        ("torch_status", res["torch_status"])):
            ce = corners(arr[..., 0], arr[..., 1], arr[..., 2])
            gx, gy = ra.x + c * ce[..., 0] - s * ce[..., 1], ra.y + s * ce[..., 0] + c * ce[..., 1]
            v = shapely.contains_xy(u, gx.ravel(), gy.ravel()).reshape(gx.shape).all((1, 2))
            agree[nm].append(v == dac_off[j])
    ep, eh = np.concatenate(err_p), np.concatenate(err_h)
    print(f"tokens {n_tok}, trajectories {len(ep)}")
    print(f"torch(full) vs official tracked: max pos err  median {np.median(ep):.4f} m  p99 {np.percentile(ep, 99):.4f}  "
          f"max {ep.max():.4f};  heading median {np.median(eh):.5f} p99 {np.percentile(eh, 99):.5f} rad")
    for k, v in agree.items():
        print(f"  DAC verdict agreement with the official label ({k}): {100 * np.concatenate(v).mean():.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
