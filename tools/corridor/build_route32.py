#!/usr/bin/env python
"""km 10/16: route progress / distance channels for the 32 m corridor (v3-32), the same fields as build_corridor64.py
(route_s = progress of the cell along metric_cache.centerline (the official EP path) minus the ego's, clipped to
[-20, 100]; route_d = distance to it, clipped to 20), 0.64 m cells over forward 0..32 m x lateral -32..32 m (50 x 100,
row = forward, col = SSR x right) = the first 50 rows of build_corridor64's 100 x 100 grid.

  $PY tools/corridor/build_route32.py --split navtrain_val --out <dir> [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

from tools.corridor.build_corridor_dt01 import load_mc, to_global  # noqa: E402

H, W, RES = 50, 100, 0.64


def one(path):
    import shapely
    fwd = (np.arange(H) + 0.5) * RES
    ssr_x = -32.0 + (np.arange(W) + 0.5) * RES
    gf, gx = np.meshgrid(fwd, ssr_x, indexing="ij")
    mc = load_mc(path)
    rx, ry = to_global(mc.ego_state, gf.ravel(), (-gx).ravel())
    ls = mc.centerline.linestring
    ra = mc.ego_state.rear_axle
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s0 = shapely.line_locate_point(ls, shapely.points(ra.x, ra.y))
        pts = shapely.points(rx, ry)
        rs = np.nan_to_num(np.clip(shapely.line_locate_point(ls, pts) - s0, -20, 100), nan=-20.0)
        rd = np.nan_to_num(np.clip(shapely.distance(ls, pts), 0, 20), nan=20.0)
    return rs.reshape(H, W).astype(np.float16), rd.reshape(H, W).astype(np.float16)


def work(args):
    out = []
    for r, p in zip(*args):
        try:
            out.append((r,) + one(p) + (True,))
        except Exception as e:  # noqa: BLE001
            out.append((r, None, None, repr(e)[:200]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", default="/home/external-user/ssd/kyungmin_ck/km_runs/ck2e2e_bevkd/eval/root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args(argv)
    import pandas as pd
    from tools.ck.data.common import MC_ROOT
    toks = pd.read_parquet(Path(a.root) / "packed" / a.split / "tokens.parquet").token.astype(str).to_numpy()
    n = len(toks)
    where = {}
    for log in os.scandir(MC_ROOT[a.split]):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = Path(t.path) / "metric_cache.pkl"
    d = Path(a.out) / a.split
    d.mkdir(parents=True, exist_ok=True)
    RS = np.lib.format.open_memmap(d / "route_s.npy", "w+", np.float16, (n, H, W))
    RD = np.lib.format.open_memmap(d / "route_d.npy", "w+", np.float16, (n, H, W))
    ok = np.zeros(n, bool)
    rows = np.array([r for r in range(n) if toks[r] in where])
    jobs = [(rows[i:i + a.chunk], [str(where[toks[r]]) for r in rows[i:i + a.chunk]]) for i in range(0, len(rows), a.chunk)]
    t0, c = time.time(), 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(work, jobs)):
            for r, rs, rd, st in res:
                if st is True:
                    RS[r], RD[r], ok[r] = rs, rd, True
                c += 1
            if k % 20 == 0:
                print(f"{c}/{len(rows)} | {c / (time.time() - t0):.1f} tok/s", flush=True)
    RS.flush(); RD.flush(); np.save(d / "ok.npy", ok)
    print(f"done {c} in {time.time() - t0:.0f}s, ok {int(ok.sum())}/{n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
