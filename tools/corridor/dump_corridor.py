#!/usr/bin/env python
"""km 10/14: run a trained corridor head (train_corridor.py) over a split and store its predictions.

  <out>/<split>/pred_sdf.npy  f16 [N, 100, 200]   predicted static SDF (m, + inside)
  <out>/<split>/pred_occ.npy  u8  [N, 8, 100, 200] occupancy probability * 255 at t = 0.5 .. 4.0 s
  <out>/<split>/ok.npy        bool [N]           rows that had an input
Rows = <root>/packed/<split>/tokens.parquet order (same as the GT corridor).

  CUDA_VISIBLE_DEVICES=0 $PY tools/corridor/dump_corridor.py --src student --ckpt <dir>/ckpt_last.pt --split navtrain_val --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

from tools.corridor import train_corridor as tc  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["student", "teacher"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="navtrain_val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    import pandas as pd
    if a.src == "teacher" and a.split == "navtest":         # cache_val_50x100 = navtest; navtrain_* live in cache_train
        tc.TEACHER = tc.TEACHER.replace("cache_train_50x100", "cache_val_50x100")
    toks = pd.read_parquet(f"{tc.ROOT}/packed/{a.split}/tokens.parquet").token.astype(str).to_numpy()
    ok = np.load(f"{tc.K}/corridor/{a.split}/ok.npy")
    if a.src == "student":
        dn = np.zeros_like(ok)
        for p in Path(f"{tc.K}/bevcache/{a.split}").glob("done_*.npy"):
            dn |= np.load(p)
        ok &= dn
    else:
        ok &= np.array([Path(f"{tc.TEACHER}/{t[:2]}/{t}.npz").exists() for t in toks])
    rows = np.where(ok)[0]
    print(f"{a.split} {a.src}: {len(rows)}/{len(toks)} rows", flush=True)
    dev = torch.device("cuda:0")
    model = tc.Head().to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"])
    model.eval()
    d = Path(a.out) / a.split
    d.mkdir(parents=True, exist_ok=True)
    n = len(toks)
    S = np.lib.format.open_memmap(d / "pred_sdf.npy", "w+", np.float16, (n, 100, 200))
    O = np.lib.format.open_memmap(d / "pred_occ.npy", "w+", np.uint8, (n, 8, 100, 200))
    dl = torch.utils.data.DataLoader(tc.DS(a.split, a.src, rows), batch_size=32, shuffle=False,
                                     num_workers=a.workers, pin_memory=True)
    i = 0
    with torch.no_grad():
        for x, _, _ in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                sp, op = model(x.to(dev, non_blocking=True))
            r = rows[i:i + len(x)]
            S[r] = sp.float().cpu().numpy().astype(np.float16)
            O[r] = (torch.sigmoid(op.float()) * 255).round().byte().cpu().numpy()
            i += len(x)
    S.flush(); O.flush(); np.save(d / "ok.npy", ok)
    print(f"done {i}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
