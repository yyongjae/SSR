#!/usr/bin/env python
"""km 10/14: ground-truth corridor targets from the METRIC CACHE (the objects the official PDM scorer reads).

Per token (rows = <root>/packed/<split>/tokens.parquet order, the BEV-cache order):
  static_sdf  f16 [N, H, W]          signed distance (m, + inside) to the drivable area = union of
                                     metric_cache.drivable_area_map geometries (what DAC tests corners against)
  dyn_occ     u8  [N, T, H, W/8]     packed bits: cell centre inside any agent polygon of the observation
                                     occupancy map at t = (k+1) * 0.5 s, k < T (red-light polygons excluded)
  ok          bool [N]
Grid: the BEV range (forward 0..32 m, lateral -32..32 m), SSR axes like feasibility.rasterize_bev_sdf
(row = forward, col = SSR x = right = -NAVSIM y), cell centres, default 0.32 m (100 x 200).

  $PY tools/corridor/build_corridor.py --split navtrain_val --out <dir> [--workers 10] [--limit N]
Resumable: chunks already marked in done.npy are skipped.
"""
from __future__ import annotations

import argparse
import lzma
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

FWD = (0.0, 32.0)
LAT = (-32.0, 32.0)     # SSR x (right)
CLIP = 8.0
T_DYN = 8


def grid(res: float):
    h, w = int(round((FWD[1] - FWD[0]) / res)), int(round((LAT[1] - LAT[0]) / res))
    fwd = FWD[0] + (np.arange(h) + 0.5) * res
    ssr_x = LAT[0] + (np.arange(w) + 0.5) * res
    gf, gx = np.meshgrid(fwd, ssr_x, indexing="ij")
    return h, w, gf.ravel(), (-gx).ravel()          # NAVSIM ego x (fwd), y (left)


def to_global(ego, fx, ly):
    ra = ego.rear_axle
    c, s = np.cos(ra.heading), np.sin(ra.heading)
    return ra.x + c * fx - s * ly, ra.y + s * fx + c * ly


def sdf_from_mask(mask, res):
    import cv2
    if mask.all() or not mask.any():
        return np.full(mask.shape, CLIP if mask.all() else -CLIP, np.float32)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.clip(np.where(mask > 0, inside - 0.5, -(outside - 0.5)) * res, -CLIP, CLIP).astype(np.float32)


def load_mc(path):
    raw = open(path, "rb").read()
    try:
        return pickle.loads(lzma.decompress(raw))
    except lzma.LZMAError:
        return pickle.loads(raw)


def one(path, res):
    import shapely
    h, w, fx, ly = grid(res)
    mc = load_mc(path)
    gx, gy = to_global(mc.ego_state, fx, ly)
    # static: drivable union, restricted to geometries near the grid
    box = shapely.box(gx.min() - 1, gy.min() - 1, gx.max() + 1, gy.max() + 1)
    dm = mc.drivable_area_map
    idx = dm.query(box)
    geoms = [dm._geometries[i] for i in idx]
    if geoms:
        u = shapely.union_all(geoms)
        shapely.prepare(u)
        mask = shapely.contains_xy(u, gx, gy).reshape(h, w).astype(np.uint8)
    else:
        mask = np.zeros((h, w), np.uint8)
    sdf = sdf_from_mask(mask, res).astype(np.float16)
    # dynamic: agent polygons at t = (k+1) * 0.5 s
    obs = mc.observation
    red = obs.red_light_token
    dyn = np.zeros((T_DYN, h, w), np.uint8)
    for k in range(T_DYN):
        ti = (k + 1) * 5
        if ti >= len(obs._global_to_local_idcs):
            dyn[k] = dyn[k - 1]
            continue
        om = obs[ti]
        ii = om.query(box)
        polys = [om._geometries[i] for i in ii if red not in om._tokens[i]]
        if polys:
            u = shapely.union_all(polys)
            shapely.prepare(u)
            dyn[k] = shapely.contains_xy(u, gx, gy).reshape(h, w)
    return sdf, np.packbits(dyn, axis=-1)


def work(args):
    rows, paths, res = args
    out = []
    for r, p in zip(rows, paths):
        try:
            s, d = one(p, res)
            out.append((r, s, d, True))
        except Exception as e:  # noqa: BLE001
            out.append((r, None, None, repr(e)[:200]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", default="/home/external-user/ssd/kyungmin_ck/km_runs/ck2e2e_bevkd/eval/root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=0.32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=64)
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
    h, w, _, _ = grid(a.res)
    d = Path(a.out) / a.split
    d.mkdir(parents=True, exist_ok=True)
    fs, fd = d / "static_sdf.npy", d / "dyn_occ.npy"
    if not fs.exists():
        np.lib.format.open_memmap(fs, "w+", np.float16, (n, h, w)).flush()
        np.lib.format.open_memmap(fd, "w+", np.uint8, (n, T_DYN, h, w // 8)).flush()
        np.save(d / "ok.npy", np.zeros(n, bool))
        np.save(d / "done.npy", np.zeros(n, bool))
        (d / "meta.txt").write_text(f"split {a.split} n {n} res {a.res} grid {h}x{w} fwd {FWD} lat(SSR x) {LAT} "
                                    f"T_dyn {T_DYN} (t=(k+1)*0.5s) clip {CLIP} mc_root {mc_root}\n")
    S = np.load(fs, mmap_mode="r+")
    D = np.load(fd, mmap_mode="r+")
    ok, done = np.load(d / "ok.npy"), np.load(d / "done.npy")
    rows = np.arange(n)[~done]
    if a.limit:
        rows = rows[: a.limit]
    missing = [r for r in rows if toks[r] not in where]
    rows = np.array([r for r in rows if toks[r] in where])
    print(f"{a.split}: {n} tokens, {len(rows)} to build, {len(missing)} without metric cache", flush=True)
    jobs = [(rows[i:i + a.chunk], [str(where[toks[r]]) for r in rows[i:i + a.chunk]], a.res)
            for i in range(0, len(rows), a.chunk)]
    t0, c, errs = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, f in enumerate(as_completed([ex.submit(work, j) for j in jobs])):
            for r, s, dd, st in f.result():
                done[r] = True
                if st is True:
                    S[r], D[r], ok[r] = s, dd, True
                else:
                    errs += 1
                    if errs <= 5:
                        print(f"err row {r}: {st}", flush=True)
                c += 1
            if k % 20 == 0:
                S.flush(); D.flush(); np.save(d / "ok.npy", ok); np.save(d / "done.npy", done)
                print(f"{c}/{len(rows)} | {c / (time.time() - t0):.1f} tok/s | err {errs}", flush=True)
    S.flush(); D.flush(); np.save(d / "ok.npy", ok); np.save(d / "done.npy", done)
    print(f"done {c} in {time.time() - t0:.0f}s, ok {int(ok.sum())}/{n}, err {errs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
