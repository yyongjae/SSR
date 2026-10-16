#!/usr/bin/env python
"""km: cache the frozen CK2 v2 bev_embed [5000, 256] f16 of every token of a split (rows = packed tokens.parquet order),
so critic training on top of CK2 skips the camera loading / v2 forward each epoch.

  CUDA_VISIBLE_DEVICES=0 $PY tools/ck/km/dump_bev.py --run-dir <RUN> --split navtrain_train --out <dir> --part 0/2
Writes <out>/<split>/bev.npy (open_memmap, created by part 0, NaN-free rows marked in done_<part>.npy).
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.ck.km.critic_train import make_loader, to_dev  # noqa: E402
from tools.ck.km.ft_trunk import Pool, build, log  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run", default="ck2e2e_bevkd")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    k, n = map(int, a.part.split("/"))
    dev = torch.device("cuda:0")
    run_dir = Path(a.run_dir)
    agent, _ck, _ = build(run_dir, dev)
    del _ck
    P = Pool(run_dir / "eval" / "root", a.run, a.split, need_pool=False)
    d = Path(a.out) / a.split
    d.mkdir(parents=True, exist_ok=True)
    N = len(P.tokens)
    f = d / "bev.npy"
    if k == 0 and not f.exists():
        np.lib.format.open_memmap(f, "w+", np.float16, (N, 5000, 256)).flush()
    while not f.exists():
        time.sleep(5)
    time.sleep(5)
    bev_mm = np.load(f, mmap_mode="r+")
    rows = P.rows[k::n]
    dp = d / f"done_{k}.npy"
    done = np.load(dp) if dp.exists() else np.zeros(N, bool)
    rows = rows[~done[rows]]
    toks = P.tokens.token.astype(str).to_numpy()[rows]
    log(f"{a.split} part {a.part}: {len(rows)} tokens to dump")
    dl = make_loader(a.split, toks, agent, a.bs, a.workers, prev=False)
    t0, c = time.time(), 0
    with torch.no_grad():
        for b in dl:
            bev = agent(to_dev(b["f"], dev))["bev_embed"].half().cpu().numpy()
            r = np.array([P.row_of[t] for t in b["tok"]])
            bev_mm[r] = bev
            done[r] = True
            c += len(r)
            if c % 4000 < a.bs:
                bev_mm.flush()
                np.save(dp, done)
                log(f"{c}/{len(rows)} | {c / (time.time() - t0):.1f} tok/s")
    bev_mm.flush()
    np.save(dp, done)
    log(f"part {a.part} done: {c} tokens in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
