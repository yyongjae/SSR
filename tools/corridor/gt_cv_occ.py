#!/usr/bin/env python
"""km 10/16: diagnostic GT variants of the dynamic corridor (perception vs forecasting), on the corr_select rows.

From the metric cache observation (what the official scorer reads), red-light polygons excluded:
  gtstatic : every agent polygon of t = 0.1 s held fixed for 0.1 .. 4.0 s   (perfect perception, no motion)
  gtcv     : the t = 0.1 s polygon translated at its constant velocity (centroid displacement 0.1 s -> 0.6 s of the
             same track token; 0 if the track is not seen at 0.6 s)            (perfect perception, constant velocity)
Written as mocc_<tag>.npy u8 [N, 40, 100, 200] (x 255) next to a dump, so corr_verdict.py --occ motion_<tag> reads it.

  $PY tools/corridor/gt_cv_occ.py --dump <dump dir> --split navtrain_val [--workers 8]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

from tools.corridor.build_corridor_dt01 import load_mc  # noqa: E402

ROOT = os.environ.get("KM_CORR_ROOT", "/home/external-user/ssd/kyungmin_ck/km_runs/ck2e2e_bevkd/eval/root")  # parent of packed/<split>/tokens.parquet


def one(path):
    import shapely
    from navsim.agents.para_ssr.modules.corridor_head import raster_agents
    mc = load_mc(path)
    ra = mc.ego_state.rear_axle
    c, s = np.cos(ra.heading), np.sin(ra.heading)
    box = shapely.box(ra.x - 50, ra.y - 50, ra.x + 50, ra.y + 50)
    obs = mc.observation
    red = obs.red_light_token

    def polys(t):
        om = obs[t]
        out = {}
        for i in om.query(box):
            tok = om._tokens[i]
            if red in tok:
                continue
            xy = shapely.get_coordinates(om._geometries[i])[:4]
            dx, dy = xy[:, 0] - ra.x, xy[:, 1] - ra.y
            out[tok] = np.stack([c * dx + s * dy, -s * dx + c * dy], -1)          # ego frame (fwd, left)
        return out

    p1, p6 = polys(1), polys(6)
    rows_s, rows_c = [], []
    for tok, q in p1.items():
        v = (p6[tok].mean(0) - q.mean(0)) / 0.5 if tok in p6 else np.zeros(2)
        for t in range(1, 41):
            rows_s.append(np.concatenate([[t], q.ravel()]))
            rows_c.append(np.concatenate([[t], (q + v * (t - 1) * 0.1).ravel()]))
    f = lambda r: raster_agents(np.asarray(r, np.float32).reshape(-1, 9), planes=40, h=100, w=200)
    return f(rows_s), f(rows_c)


def work(job):
    out = []
    for r, p in job:
        try:
            out.append((r,) + one(p))
        except Exception as e:  # noqa: BLE001
            out.append((r, None, repr(e)[:200]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--split", default="navtrain_val")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd
    from tools.ck.data.common import MC_ROOT
    d = Path(a.dump) / a.split
    done = np.zeros(0, bool)
    for q in d.glob("done_*.npy"):
        done = np.load(q) if not len(done) else done | np.load(q)
    n = len(done)
    rows = np.where(done)[0]
    toks = pd.read_parquet(f"{ROOT}/packed/{a.split}/tokens.parquet").token.astype(str).to_numpy()
    where = {}
    for log in os.scandir(MC_ROOT[a.split]):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = str(Path(t.path) / "metric_cache.pkl")
    rows = np.array([r for r in rows if toks[r] in where])
    O = {}
    for tag in ("gtstatic", "gtcv"):
        f = d / f"mocc_{tag}.npy"
        if not f.exists():
            np.lib.format.open_memmap(f, "w+", np.uint8, (n, 40, 100, 200)).flush()
        O[tag] = np.load(f, mmap_mode="r+")
    jobs = [[(r, where[toks[r]]) for r in rows[i:i + 16]] for i in range(0, len(rows), 16)]
    t0, c, err = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(work, jobs):
            for x in res:
                if x[1] is None:
                    err += 1
                    print("err", x, flush=True)
                    continue
                O["gtstatic"][x[0]] = x[1].astype(np.uint8) * 255
                O["gtcv"][x[0]] = x[2].astype(np.uint8) * 255
                c += 1
            if c % 320 < 16:
                print(f"{c}/{len(rows)} {c / (time.time() - t0):.1f} tok/s err {err}", flush=True)
    for v in O.values():
        v.flush()
    print(f"done {c} in {time.time() - t0:.0f}s err {err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
