#!/usr/bin/env python
"""km 10/16: ground-truth corridor for the 64 m BEV (v3-64), from the METRIC CACHE (what the official scorer reads).

Per token (rows = <root>/packed/<split>/tokens.parquet order):
  static_sdf  f16 [N, 200, 200]   signed distance (m, + inside) to the drivable area, 0.32 m cells, forward 0..64 m,
                                  lateral (SSR x, right) -32..32 m, row = forward, col = SSR x (as build_corridor.py)
  route_s     f16 [N, 100, 100]   progress (m) of the cell along the route centerline (metric_cache.centerline, the
                                  official EP path) minus the ego's, 0.64 m cells; clipped to [-20, 100]
  route_d     f16 [N, 100, 100]   distance (m) of the cell to the route centerline, clipped to 20
  agents      ragged: agent_rows f16 [M, 9] = (t index 1..49, 4 corners x (fwd, left) in the ego frame) of every
              observation polygon at t * 0.1 s (red-light polygons excluded) that touches the grid; agent_off i64 [N + 1]
              (rows of token r = agent_rows[agent_off[r]:agent_off[r + 1]]).  Rasterised in the target builder
              (corridor_head.Corridor64TargetBuilder), so 49 planes x 200 x 200 never hit the disk.
  ok          bool [N]

  $PY tools/corridor/build_corridor64.py --split navtrain_val --out <dir> [--workers 6]
Not resumable (the ragged agent table is written at the end); val ~ 15 min, train ~ 1.5 h at 6 workers.
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

from tools.corridor.build_corridor_dt01 import load_mc, sdf_from_mask, to_global  # noqa: E402

FWD = (0.0, 64.0)
LAT = (-32.0, 32.0)     # SSR x (right)
RES, RES_R = 0.32, 0.64
T_DYN = 49              # 0.1 .. 4.9 s (TTC projects 0.9 s past the 4.0 s horizon)


def grid(res: float):
    h, w = int(round((FWD[1] - FWD[0]) / res)), int(round((LAT[1] - LAT[0]) / res))
    fwd = FWD[0] + (np.arange(h) + 0.5) * res
    ssr_x = LAT[0] + (np.arange(w) + 0.5) * res
    gf, gx = np.meshgrid(fwd, ssr_x, indexing="ij")
    return h, w, gf.ravel(), (-gx).ravel()          # NAVSIM ego x (fwd), y (left)


def one(path):
    import shapely
    h, w, fx, ly = grid(RES)
    mc = load_mc(path)
    gx, gy = to_global(mc.ego_state, fx, ly)
    box = shapely.box(gx.min() - 1, gy.min() - 1, gx.max() + 1, gy.max() + 1)
    dm = mc.drivable_area_map
    geoms = [dm._geometries[i] for i in dm.query(box)]
    if geoms:
        u = shapely.union_all(geoms)
        shapely.prepare(u)
        mask = shapely.contains_xy(u, gx, gy).reshape(h, w).astype(np.uint8)
    else:
        mask = np.zeros((h, w), np.uint8)
    sdf = sdf_from_mask(mask, RES).astype(np.float16)
    # route progress / distance (0.64 m)
    hr, wr, rfx, rly = grid(RES_R)
    rx, ry = to_global(mc.ego_state, rfx, rly)
    ls = mc.centerline.linestring
    pts = shapely.points(rx, ry)
    ra = mc.ego_state.rear_axle
    s0 = shapely.line_locate_point(ls, shapely.points(ra.x, ra.y))
    import warnings
    with warnings.catch_warnings():             # as PDMPath.project: GEOS warns, values stay finite
        warnings.simplefilter("ignore", RuntimeWarning)
        rs = np.nan_to_num(np.clip(shapely.line_locate_point(ls, pts) - s0, -20, 100), nan=-20.0)
    rs = rs.reshape(hr, wr).astype(np.float16)
    rd = np.nan_to_num(np.clip(shapely.distance(ls, pts), 0, 20), nan=20.0).reshape(hr, wr).astype(np.float16)
    # agents: polygon corners in the ego frame
    c, s = np.cos(ra.heading), np.sin(ra.heading)
    obs = mc.observation
    red = obs.red_light_token
    rows = []
    for t in range(1, min(T_DYN, len(obs._global_to_local_idcs) - 1) + 1):
        om = obs[t]
        for i in om.query(box):
            if red in om._tokens[i]:
                continue
            xy = shapely.get_coordinates(om._geometries[i])[:4]
            dx, dy = xy[:, 0] - ra.x, xy[:, 1] - ra.y
            ex, ey = c * dx + s * dy, -s * dx + c * dy
            rows.append(np.concatenate([[t], np.stack([ex, ey], -1).ravel()]))
    ag = np.asarray(rows, np.float16).reshape(-1, 9)
    return sdf, rs, rd, ag


def work(args):
    rows, paths = args
    out = []
    for r, p in zip(rows, paths):
        try:
            out.append((r,) + one(p) + (True,))
        except Exception as e:  # noqa: BLE001
            out.append((r, None, None, None, None, repr(e)[:200]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", default=os.environ.get("KM_CORR_ROOT", "/home/external-user/ssd/kyungmin_ck/km_runs/ck2e2e_bevkd/eval/root"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd
    from tools.ck.data.common import MC_ROOT

    toks = pd.read_parquet(Path(a.root) / "packed" / a.split / "tokens.parquet").token.astype(str).to_numpy()
    n = len(toks)
    mc_root = MC_ROOT[a.split]
    where = {}
    for log in os.scandir(mc_root):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = Path(t.path) / "metric_cache.pkl"
    h, w, _, _ = grid(RES)
    hr, wr, _, _ = grid(RES_R)
    d = Path(a.out) / a.split
    d.mkdir(parents=True, exist_ok=True)
    S = np.lib.format.open_memmap(d / "static_sdf.npy", "w+", np.float16, (n, h, w))
    RS = np.lib.format.open_memmap(d / "route_s.npy", "w+", np.float16, (n, hr, wr))
    RD = np.lib.format.open_memmap(d / "route_d.npy", "w+", np.float16, (n, hr, wr))
    ok = np.zeros(n, bool)
    (d / "meta.txt").write_text(f"split {a.split} n {n} res {RES} grid {h}x{w} fwd {FWD} lat(SSR x) {LAT} route res "
                                f"{RES_R} ({hr}x{wr}) T_dyn {T_DYN} (t=k*0.1s, k=1..49) agents ragged ego-frame corners "
                                f"mc_root {mc_root}\n")
    rows = np.array([r for r in range(n) if toks[r] in where])
    if a.limit:
        rows = rows[: a.limit]
    print(f"{a.split}: {n} tokens, {len(rows)} to build, {n - len(rows)} without metric cache", flush=True)
    jobs = [(rows[i:i + a.chunk], [str(where[toks[r]]) for r in rows[i:i + a.chunk]])
            for i in range(0, len(rows), a.chunk)]
    ags = [np.zeros((0, 9), np.float16)] * n
    t0, c, errs = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(work, jobs)):
            for r, s, rs, rd, ag, st in res:
                if st is True:
                    S[r], RS[r], RD[r], ags[r], ok[r] = s, rs, rd, ag, True
                else:
                    errs += 1
                    if errs <= 5:
                        print(f"err row {r}: {st}", flush=True)
                c += 1
            if k % 50 == 0:
                print(f"{c}/{len(rows)} | {c / (time.time() - t0):.1f} tok/s | err {errs}", flush=True)
    S.flush(); RS.flush(); RD.flush()
    off = np.zeros(n + 1, np.int64)
    off[1:] = np.cumsum([len(x) for x in ags])
    np.save(d / "agent_off.npy", off)
    np.save(d / "agent_rows.npy", np.concatenate(ags, 0))
    np.save(d / "ok.npy", ok)
    print(f"done {c} in {time.time() - t0:.0f}s, ok {int(ok.sum())}/{n}, err {errs}, agent rows {off[-1]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
