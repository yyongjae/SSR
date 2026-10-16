#!/usr/bin/env python
"""km 10/14 (3): verdict fidelity of PREDICTED corridors vs the OFFICIAL label, on navtrain_val (never trained on).
Trajectories: the 256 raw anchors only (model-free).  Official label: tools/refiner/score_trajectories.score_token
(the pinned official PDM scorer).  Every corridor verdict is taken on the LQR-tracked states (PDMSimulator), 41 x 0.1 s.

DAC verdict = all footprint corners at all 41 steps have SDF >= thr, using
   G  GT raster SDF (build_corridor.py)      S  student head (frozen CK2 BEV)      T  teacher head (BEVFusion BEV)
NC verdict  = no footprint sample point (5 x 3 grid) at t = 0.5 .. 4.0 s lies in an occupied cell
   G  GT occupancy bits                      S / T  predicted occupancy probability >= 0.5
Reported for all trajectories and for those whose footprint stays in the raster (fwd <= 32 m, |lat| <= 32 m).

  $PY tools/corridor/fidelity_pred.py --n 600 --workers 7
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

from tools.corridor.fidelity import ANCH, FRONT, HW, REAR, corners, raster_sample, to_ego  # noqa: E402

K = "/home/external-user/ssd/kyungmin_ck"
ROOT = f"{K}/km_runs/ck2e2e_bevkd/eval/root"
SPLIT = "navtrain_val"
PRED = {"S": f"{K}/km_runs/corridor_head/student/pred/{SPLIT}", "T": f"{K}/km_runs/corridor_head/teacher/pred/{SPLIT}"}
PX, PY_ = np.meshgrid(np.linspace(-REAR, FRONT, 5), np.linspace(-HW, HW, 3), indexing="ij")
FP = np.stack([PX.ravel(), PY_.ravel()], -1)                                              # [15, 2] body frame


def body_pts(x, y, h):
    cs, sn = np.cos(h)[..., None], np.sin(h)[..., None]
    return np.stack([x[..., None] + cs * FP[:, 0] - sn * FP[:, 1], y[..., None] + sn * FP[:, 0] + cs * FP[:, 1]], -1)


def nearest(occ, pts):
    """occ [H, W]; pts [..., 2] ego (x fwd, y left) -> value at the containing cell (0 outside)."""
    h, w = occ.shape
    r = np.floor(pts[..., 0] / 0.32).astype(int)
    c = np.floor((-pts[..., 1] + 32.0) / 0.32).astype(int)
    ins = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    return np.where(ins, occ[np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)], 0)


def work(job):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from tools.ck.data import label_cands as LC

    ST = LC.ST
    ST.get_simulator_scorer()
    rows, paths = job
    A = np.load(ANCH).astype(np.float64)
    GS = np.load(f"{K}/corridor/{SPLIT}/static_sdf.npy", mmap_mode="r")
    GD = np.load(f"{K}/corridor/{SPLIT}/dyn_occ.npy", mmap_mode="r")
    PS = {k: np.load(f"{v}/pred_sdf.npy", mmap_mode="r") for k, v in PRED.items()}
    PO = {k: np.load(f"{v}/pred_occ.npy", mmap_mode="r") for k, v in PRED.items()}
    fs = TrajectorySampling(num_poses=40, interval_length=0.1)
    sim = PDMSimulator(fs)
    out = []
    for r, p in zip(rows, paths):
        try:
            mc = ST.load_metric_cache(p)
            lab = ST.score_token(mc, A.astype(np.float32))
            nc = np.array([d["nc"] for d in lab]) > 0.75
            dac = np.array([d["dac"] for d in lab]) > 0.5
            pdms = np.array([d["pdms"] for d in lab])
            ego = mc.ego_state
            st = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(t.astype(np.float32)), ego), fs,
                                                   ego.time_point) for t in A], 0)
            tr = sim.simulate_proposals(st, ego)                                          # [256, 41, 11] global
            cg = corners(tr[..., 0], tr[..., 1], tr[..., 2])
            ex, ey = to_ego(ego, cg[..., 0], cg[..., 1])
            ce = np.stack([ex, ey], -1)                                                   # [256, 41, 4, 2]
            sdfs = {"G": np.asarray(GS[r], np.float32), **{k: np.asarray(v[r], np.float32) for k, v in PS.items()}}
            res = {}
            for k, s in sdfs.items():
                v, ins = raster_sample(s, ce)
                res["D" + k] = (v >= 0).all((1, 2))
                if k == "G":
                    res["in"] = ins.all((1, 2))
            # NC: tracked pose at 0.5 s steps
            idx = np.arange(5, 41, 5)                                                     # 8 steps
            bx, by = to_ego(ego, tr[:, idx, 0], tr[:, idx, 1])
            bh = tr[:, idx, 2] - ego.rear_axle.heading
            bp = body_pts(bx, by, bh)                                                     # [256, 8, 15, 2]
            occs = {"G": np.unpackbits(np.asarray(GD[r]), axis=-1).astype(np.float32),
                    **{k: np.asarray(v[r], np.float32) / 255.0 for k, v in PO.items()}}
            for k, o in occs.items():
                hit = np.zeros(len(A), bool)
                for j in range(8):
                    hit |= (nearest(o[j], bp[:, j]) >= 0.5).any(-1)
                res["N" + k] = ~hit
            out.append((nc, dac, pdms, res))
        except Exception as e:  # noqa: BLE001
            out.append(repr(e)[:300])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd
    from tools.ck.data.common import MC_ROOT
    toks = pd.read_parquet(f"{ROOT}/packed/{SPLIT}/tokens.parquet").token.astype(str).to_numpy()
    ok = np.load(f"{K}/corridor/{SPLIT}/ok.npy")
    for v in PRED.values():
        ok &= np.load(f"{v}/ok.npy")
    rows = np.sort(np.random.default_rng(0).choice(np.where(ok)[0], a.n, replace=False))
    where = {}
    for log in os.scandir(MC_ROOT[SPLIT]):
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
    print(f"{SPLIT} tokens {len(agg)} (errors {len(errs)}{': ' + errs[0] if errs else ''})")
    nc = np.concatenate([x[0] for x in agg]); dac = np.concatenate([x[1] for x in agg])
    R = {k: np.concatenate([x[3][k] for x in agg]) for k in agg[0][3]}
    inr = R["in"]
    print(f"anchors {len(nc)} (footprint in raster {100 * inr.mean():.1f}%), official DAC pass {100 * dac.mean():.2f}%, "
          f"NC pass {100 * nc.mean():.2f}%")
    for k, lab in [("D" + s, dac) for s in "GST"] + [("N" + s, nc) for s in "GST"]:
        for nm, m in (("all", np.ones_like(inr)), ("in-range", inr)):
            p, t = R[k][m], lab[m]
            print(f"  {k} {nm:8s} agree {100 * (p == t).mean():6.2f}%  false-pass {100 * (p & ~t).mean():5.2f}%  "
                  f"false-fail {100 * (~p & t).mean():5.2f}%")
    np.savez_compressed(f"{K}/km_runs/corridor_head/fidelity_pred_{SPLIT}.npz", nc=nc, dac=dac,
                        pdms=np.concatenate([x[2] for x in agg]), **R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
