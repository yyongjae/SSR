#!/usr/bin/env python
"""km 10/14 (2): how faithfully does a corridor verdict reproduce the OFFICIAL label?  Model-free trajectories only:
the 256 raw anchors (labels raw256) and the 96 rule variants (var96) of sampled navtrain_train tokens.

Ladder for DAC (each step changes ONE thing):
  L1  GT raster corridor 0.32 m (build_corridor.py), 8 planned waypoints, footprint corners, bilinear SDF >= 0
  L2  same raster, the planned trajectory interpolated to 10 Hz exactly as the official pipeline does
  L3  exact drivable polygons (metric cache), 10 Hz planned
  L4  exact drivable polygons, the LQR-tracked states of the official PDMSimulator  (should reproduce the label)
NC (collision, at-fault logic ignored): ego polygon vs agent polygons of the metric-cache observation,
  N3 on 10 Hz planned, N4 on tracked states.
Reported: agreement with the official label, false-pass / false-fail rates, also restricted to trajectories whose
footprint stays inside the raster (forward <= 32 m, |lateral| <= 32 m).

  $PY tools/corridor/fidelity.py --n 600 --workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

K = "/home/external-user/ssd/kyungmin_ck"
ROOT = f"{K}/km_runs/ck2e2e_bevkd/eval/root"
RAW = f"{K}/ck/labels/navtrain_train/raw256"
VAR = f"{K}/ck/ck2/labels/navtrain_train/var_separate_sampler16_accstraight"
ANCH = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy"
FRONT, REAR, HW = 4.049, 1.127, 2.297 / 2
RES, FWD0, LAT0 = 0.32, 0.0, -32.0


def corners(x, y, h):
    """arrays [...] -> [..., 4, 2]"""
    c = np.array([[FRONT, HW], [FRONT, -HW], [-REAR, -HW], [-REAR, HW]])
    cs, sn = np.cos(h)[..., None], np.sin(h)[..., None]
    return np.stack([x[..., None] + cs * c[:, 0] - sn * c[:, 1], y[..., None] + sn * c[:, 0] + cs * c[:, 1]], -1)


def raster_sample(sdf, pts):
    """sdf [H, W] (row fwd, col SSR x right); pts [..., 2] NAVSIM ego (x fwd, y left) -> (val, inside)"""
    h, w = sdf.shape
    r = (pts[..., 0] - FWD0) / RES - 0.5
    c = (-pts[..., 1] - LAT0) / RES - 0.5
    inside = (r <= h - 0.5) & (c >= -0.5) & (c <= w - 0.5)      # behind the ego (x < 0) is not 'out of range'
    r, c = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    r1, c1 = np.minimum(r0 + 1, h - 1), np.minimum(c0 + 1, w - 1)
    fr, fc = r - r0, c - c0
    v = (sdf[r0, c0] * (1 - fr) * (1 - fc) + sdf[r1, c0] * fr * (1 - fc)
         + sdf[r0, c1] * (1 - fr) * fc + sdf[r1, c1] * fr * fc)
    return v, inside


def to_ego(ego, gx, gy):
    ra = ego.rear_axle
    dx, dy = gx - ra.x, gy - ra.y
    c, s = np.cos(ra.heading), np.sin(ra.heading)
    return c * dx + s * dy, -s * dx + c * dy


def work(job):
    import shapely
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from tools.corridor.build_corridor import load_mc

    rows, paths = job
    A = np.load(ANCH).astype(np.float64)
    raw_y = np.load(f"{RAW}/labels.npy", mmap_mode="r"); raw_ok = np.load(f"{RAW}/ok.npy", mmap_mode="r")
    vt = np.load(f"{VAR}/traj.npy", mmap_mode="r"); vy = np.load(f"{VAR}/labels.npy", mmap_mode="r")
    vok = np.load(f"{VAR}/ok.npy", mmap_mode="r")
    S = np.load(f"{K}/corridor/navtrain_train/static_sdf.npy", mmap_mode="r")
    fs = TrajectorySampling(num_poses=40, interval_length=0.1)
    sim = PDMSimulator(fs)
    out = []
    for r, p in zip(rows, paths):
        try:
            mc = load_mc(p)
            T = np.concatenate([A, np.asarray(vt[r], np.float64)], 0)                 # [352, 8, 3]
            Y = np.concatenate([np.asarray(raw_y[r]), np.asarray(vy[r])], 0)            # [352, 9]
            okm = np.concatenate([np.asarray(raw_ok[r]).reshape(-1) if np.ndim(raw_ok[r]) else
                                  np.full(256, bool(raw_ok[r])), np.asarray(vok[r]).reshape(-1)], 0)
            okm &= np.isfinite(T).all((1, 2)) & np.isfinite(Y[:, :2]).all(1)
            sdf = np.asarray(S[r], np.float32)
            ego = mc.ego_state
            # L1: 8 planned waypoints on the raster
            c8 = corners(T[..., 0], T[..., 1], T[..., 2])                                # [K, 8, 4, 2]
            v1, in1 = raster_sample(sdf, c8)
            # 10 Hz planned, official interpolation (global frame)
            st = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(t.astype(np.float32)), ego), fs,
                                                   ego.time_point) for t in T], 0)       # [K, 41, 11]
            sim_st = sim.simulate_proposals(st, ego)                                    # [K, 41, 11]
            res = {}
            dm = mc.drivable_area_map
            gx0, gy0 = st[..., 0], st[..., 1]
            box = shapely.box(gx0.min() - 10, gy0.min() - 10, gx0.max() + 10, gy0.max() + 10)
            u = shapely.union_all([dm._geometries[i] for i in dm.query(box)])
            shapely.prepare(u)
            for tag, arr in (("plan", st), ("track", sim_st)):
                cg = corners(arr[..., 0], arr[..., 1], arr[..., 2])                     # global [K, 41, 4, 2]
                ex, ey = to_ego(ego, cg[..., 0], cg[..., 1])
                ce = np.stack([ex, ey], -1)
                if tag == "plan":
                    v2, in2 = raster_sample(sdf, ce)
                    res["L2"] = (v2 >= 0).all((1, 2)); res["in"] = in1.all((1, 2)) & in2.all((1, 2))
                inside = shapely.contains_xy(u, cg[..., 0].ravel(), cg[..., 1].ravel()).reshape(cg.shape[:-1])
                res["L3" if tag == "plan" else "L4"] = inside.all((1, 2))
                # NC: ego polygon vs agent polygons at the same 0.1 s index
                obs = mc.observation
                coll = np.zeros(len(T), bool)
                for ti in range(0, arr.shape[1]):
                    if ti >= len(obs._global_to_local_idcs):
                        break
                    om = obs[ti]
                    polys = shapely.polygons(cg[:, ti])
                    idx = om._str_tree.query(polys, predicate="intersects")
                    for k, g in zip(*idx):
                        if obs.red_light_token not in om._tokens[g]:
                            coll[k] = True
                res["N3" if tag == "plan" else "N4"] = ~coll
            res["L1"] = (v1 >= 0).all((1, 2))
            out.append((okm, Y[:, 1] > 0.5, Y[:, 0] > 0.75, res))
        except Exception as e:  # noqa: BLE001
            out.append(repr(e)[:200])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd
    from tools.ck.data.common import MC_ROOT
    toks = pd.read_parquet(f"{ROOT}/packed/navtrain_train/tokens.parquet").token.astype(str).to_numpy()
    ok = np.load(f"{K}/corridor/navtrain_train/ok.npy")
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(np.where(ok)[0], a.n, replace=False))
    where = {}
    for log in os.scandir(MC_ROOT["navtrain_train"]):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = str(Path(t.path) / "metric_cache.pkl")
    jobs = [(rows[i:i + 10], [where[toks[r]] for r in rows[i:i + 10]]) for i in range(0, len(rows), 10)]
    agg = []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(work, jobs):
            agg += res
    errs = [x for x in agg if isinstance(x, str)]
    agg = [x for x in agg if not isinstance(x, str)]
    print(f"tokens {len(agg)} (errors {len(errs)}{': ' + errs[0] if errs else ''})")
    okm = np.concatenate([x[0] for x in agg]); dac = np.concatenate([x[1] for x in agg])
    nc = np.concatenate([x[2] for x in agg])
    R = {k: np.concatenate([x[3][k] for x in agg]) for k in agg[0][3]}
    inr = R["in"] & okm
    print(f"trajectories {okm.sum()} (in raster range {100 * inr.sum() / okm.sum():.1f}%), official DAC pass "
          f"{100 * dac[okm].mean():.2f}%, NC pass {100 * nc[okm].mean():.2f}%")
    for k, lab in (("L1", dac), ("L2", dac), ("L3", dac), ("L4", dac), ("N3", nc), ("N4", nc)):
        for nm, m in (("all", okm), ("in-range", inr)):
            p, t = R[k][m], lab[m]
            print(f"  {k} {nm:8s} agree {100 * (p == t).mean():6.2f}%  false-pass {100 * (p & ~t).mean():5.2f}%  "
                  f"false-fail {100 * (~p & t).mean():5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
