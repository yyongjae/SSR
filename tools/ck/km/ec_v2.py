#!/usr/bin/env python
"""km: official NAVSIM v2 two-frame extended comfort (EC) between every candidate pair of adjacent frames.

Run with the NAVSIM v2 worktree on PYTHONPATH (km_runs/v2_env.sh), after label_v2.py --merge:
  $PY tools/ck/km/ec_v2.py --lab <labv2>/<split>

Adjacency = official run_pdm_score_one_stage.infer_start_adjacent_mapping on (token, log, start_time, frame_type) of
the split's tokens (the frames the one-stage EPDMS pairs).  For a frame t with previous frame p:
  ec[t, i, j] = official ego_is_two_frame_extended_comfort(states[t, i] overlap, states[p, j] overlap)
with the overlap / time points of SceneAggregator._compute_two_frame_comfort (interval 0.1 s).
Writes <lab>/prev_row.npy [N] (-1: no previous frame -> EC weight 0, as the official aggregation) and
<lab>/ec.npy [N, 96, 96] uint8 (255 where a column is invalid).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_G = {}


def _block(t):
    st, start, prev, f = _G["st"], _G["start"], _G["prev"], _G["f"]
    interval = 0.1
    p = prev[t]
    obs = start[t] - start[p]
    assert 0 < obs < 0.55, obs
    k = round(obs / interval)
    cur, pre = np.asarray(st[t], np.float64)[:, :-k], np.asarray(st[p], np.float64)[:, k:]
    ci, pj = np.flatnonzero(np.isfinite(cur).all((1, 2))), np.flatnonzero(np.isfinite(pre).all((1, 2)))
    blk = np.full((96, 96), 255, np.uint8)
    if len(ci) and len(pj):
        I, J = np.meshgrid(ci, pj, indexing="ij")
        r = f(cur[I.ravel()], pre[J.ravel()], np.arange(cur.shape[1]) * interval)
        blk[I.ravel(), J.ravel()] = r.astype(np.uint8)
    return t, blk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    from navsim.planning.script.run_pdm_score_one_stage import infer_start_adjacent_mapping
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )
    d = Path(a.lab)
    meta = pd.read_parquet(d / "meta.parquet")
    st = np.load(d / "states.npy", mmap_mode="r")
    N = len(meta)
    ok = meta.done.to_numpy() & (meta.err.to_numpy() == 0)
    df = pd.DataFrame({"token": meta.token, "log_name": meta.log, "start_time": meta.start_time,
                       "frame_type": meta.frame_type})[ok]
    mp = infer_start_adjacent_mapping(df)
    row = {t: i for i, t in enumerate(meta.token)}
    prev = np.full(N, -1, np.int64)
    for cur, p in mp.items():
        prev[row[cur]] = row[p]
    ec = np.lib.format.open_memmap(d / "ec.npy", "w+", np.uint8, (N, 96, 96))
    ec[:] = 255
    todo = np.flatnonzero(prev >= 0)
    _G.update(st=st, start=meta.start_time.to_numpy(), prev=prev, f=ego_is_two_frame_extended_comfort)
    import multiprocessing as mproc
    with mproc.get_context("fork").Pool(a.workers) as pool:
        for n, (t, blk) in enumerate(pool.imap_unordered(_block, todo, chunksize=16)):
            ec[t] = blk
            if n % 2000 == 0:
                print(f"{n} / {len(todo)}", flush=True)
    ec.flush()
    np.save(d / "prev_row.npy", prev)
    print(f"{N} frames, {(prev >= 0).sum()} with a previous frame", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
