#!/usr/bin/env python
"""km 10/18: 32 m corridor with 49 occupancy planes (0.1 .. 4.9 s; TTC projects 0.9 s past the 4.0 s horizon).

The 32 m grid is the first 100 rows of the 64 m grid (same 0.32 m cells, lateral -32..32 m, row = forward), so the
dynamic planes are rasterised from the 64 m cache's ego-frame agent polygons (build_corridor64.py) with the same
cell-centre rule (corridor_head.raster_agents); the static SDF is the 32 m one (build_corridor_dt01.py), linked.

  <out>/<split>/dyn_occ.npy     u8 [N, 49, 100, 25]  packed bits (np.packbits axis -1, as build_corridor_dt01.py)
                static_sdf.npy  -> <dt01>/<split>/static_sdf.npy
                ok.npy          bool [N] = dt01 ok & 64 m ok
  $PY tools/corridor/build_corridor_p49.py --split navtrain_val --out <dir> [--workers 6]
Resumable (done_p49.npy).  --check N compares planes 1..40 with the dt01 dyn_occ on N rows instead of building.
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

K = "/home/external-user/ssd/kyungmin_ck"
P, H, W = 49, 100, 200
_G = {}


def _init(c64, out):
    from navsim.agents.para_ssr.modules.corridor_head import raster_agents
    _G.update(raster=raster_agents, off=np.load(f"{c64}/agent_off.npy"),
              rows=np.load(f"{c64}/agent_rows.npy", mmap_mode="r"), D=np.load(f"{out}/dyn_occ.npy", mmap_mode="r+"))


def _occ(r):
    return _G["raster"](np.asarray(_G["rows"][_G["off"][r]:_G["off"][r + 1]]), planes=P, h=H, w=W)


def work(rows):
    for r in rows:
        _G["D"][r] = np.packbits(_occ(r).astype(np.uint8), axis=-1)
    _G["D"].flush()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default=f"{K}/corridor32_p49")
    ap.add_argument("--dt01", default=f"{K}/corridor_dt01")
    ap.add_argument("--c64", default=f"{K}/corridor64")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    d32, d64, d = Path(a.dt01) / a.split, Path(a.c64) / a.split, Path(a.out) / a.split
    ok32, ok64 = np.load(d32 / "ok.npy"), np.load(d64 / "ok.npy")
    assert len(ok32) == len(ok64), (len(ok32), len(ok64))
    n = len(ok32)
    if a.check:
        from navsim.agents.para_ssr.modules.corridor_head import raster_agents
        G = np.load(d32 / "dyn_occ.npy", mmap_mode="r")
        off, rows = np.load(d64 / "agent_off.npy"), np.load(d64 / "agent_rows.npy", mmap_mode="r")
        sel = np.random.default_rng(0).choice(np.flatnonzero(ok32 & ok64), a.check, replace=False)
        inter = uni = 0
        per = []
        for r in sel:
            o = raster_agents(np.asarray(rows[off[r]:off[r + 1]]), planes=P, h=H, w=W)[:40]
            g = np.unpackbits(np.asarray(G[r]), axis=-1).astype(bool)
            i, u = (o & g).sum(), (o | g).sum()
            inter, uni = inter + i, uni + u
            per.append(i / u if u else 1.0)
        per = np.array(per)
        print(f"{a.split}: planes 1..40 vs dt01 on {len(sel)} rows: IoU {inter / max(uni, 1):.4f} | "
              f"per-row IoU median {np.median(per):.4f}, p5 {np.percentile(per, 5):.4f}, min {per.min():.4f}")
        return 0
    d.mkdir(parents=True, exist_ok=True)
    f = d / "dyn_occ.npy"
    if not f.exists():
        np.lib.format.open_memmap(f, "w+", np.uint8, (n, P, H, W // 8)).flush()
    s = d / "static_sdf.npy"
    if not s.exists():
        s.symlink_to(d32 / "static_sdf.npy")
    ok = ok32 & ok64
    fd = d / "done_p49.npy"
    done = np.load(fd) if fd.exists() else np.zeros(n, bool)
    todo = np.flatnonzero(ok & ~done)
    (d / "meta.txt").write_text(f"split {a.split} n {n} res 0.32 grid {H}x{W} fwd (0.0, 32.0) lat(SSR x) (-32.0, 32.0) "
                                f"T_dyn {P} (t=(k+1)*0.1s) dyn from {d64} agents, static_sdf -> {d32}\n")
    print(f"{a.split}: {n} rows, ok {int(ok.sum())} (dt01 {int(ok32.sum())}, 64 m {int(ok64.sum())}), "
          f"todo {len(todo)}", flush=True)
    jobs = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
    t0, c = time.time(), 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(d64), str(d))) as ex:
        for k, rows in enumerate(ex.map(work, jobs)):
            done[rows] = True
            c += len(rows)
            if k % 20 == 0:
                np.save(fd, done)
                el = time.time() - t0
                print(f"{c}/{len(todo)} | {c / el:.1f} rows/s | eta {(len(todo) - c) / max(c / el, 1e-6) / 60:.0f} min",
                      flush=True)
    np.save(fd, done)
    np.save(d / "ok.npy", ok & done)
    print(f"done {c} in {time.time() - t0:.0f}s, ok {int((ok & done).sum())}/{n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
