#!/usr/bin/env python
"""Trajectory file of the raw256 labels: the 256 fixed anchors repeated for every token, f32 [N, 256, 8, 3].

  make_raw256_traj.py --split {navtrain_train,navtrain_val,trainval_simscale} --out FILE [--packed DIR] [--skip-done]

label_cands.py scores this file.  N = the rows of packed/<split>/tokens.parquet.  The hash of the array is compared with its pin
before the file appears.  An existing file is never overwritten (--skip-done: one of the right shape is kept).  CPU only, a
few seconds, 2.0 GiB (train) + 0.4 GiB (val).  trainval_simscale has no pin: N is read from --packed (default
$SSR_DATA_ROOT/ck/packed/trainval_simscale), 24.6 KB per row (8.4 GB for 340k rows; only label_cands reads it).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
ANCHORS = ROOT / "resources/planning_vb/trajectory_anchors_256.npy"
PINS = {"navtrain_train": (85109, "72767c6c437f1132"), "navtrain_val": (18179, "701c66d6ca564798")}
UNPINNED = ("trainval_simscale",)            # N from its packed tokens.parquet


def tiled_sha16(anchors: np.ndarray, n: int) -> str:
    """sha16 of anchors repeated n times, without building the array."""
    h = hashlib.sha256()
    b = np.ascontiguousarray(anchors).tobytes()
    for _ in range(n):
        h.update(b)
    return h.hexdigest()[:16]


def write_tiled(out: Path, anchors: np.ndarray, n: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp{os.getpid()}")
    a = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n,) + anchors.shape)
    a[:] = anchors
    a.flush()
    del a
    os.replace(tmp, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(PINS) + list(UNPINNED))
    ap.add_argument("--out", required=True)
    ap.add_argument("--packed", default="", help="packed directory of an unpinned split (default $SSR_DATA_ROOT/ck/packed/<split>)")
    ap.add_argument("--skip-done", action="store_true", help="an existing file of the right shape is kept")
    a = ap.parse_args(argv)
    out = Path(a.out)
    anchors = np.load(ANCHORS, allow_pickle=False).astype(np.float32)
    if a.split in PINS:
        n, pin = PINS[a.split]
    else:
        import pandas as pd
        sys.path.insert(0, str(ROOT))
        from navsim.common import ssr_paths as SP
        pk = Path(a.packed or Path(SP.SSR_DATA_ROOT) / "ck/packed" / a.split) / "tokens.parquet"
        n, pin = len(pd.read_parquet(pk, columns=["token"])), None
    if out.exists():
        if a.skip_done and np.load(out, mmap_mode="r").shape == (n,) + anchors.shape:
            print(f"{out}: [{n}, 256, 8, 3] exists, kept")
            return 0
        raise SystemExit(f"refusing to overwrite {out}")
    got = tiled_sha16(anchors, n)
    if pin is not None and got != pin:
        raise SystemExit(f"{a.split}: tiled anchors sha16 {got} != pin {pin}")
    write_tiled(out, anchors, n)
    print(f"{a.split}: [{n}, 256, 8, 3] sha16 {got} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
