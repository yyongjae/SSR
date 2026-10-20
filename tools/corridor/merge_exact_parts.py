#!/usr/bin/env python
"""km 10/19 (C v3): join build_exact.py --part outputs <dir>/<split>/parts/*/ into <dir>/<split>/ (every part covers
disjoint rows of the same token list; a row's tables come from the part whose ok is set).  Writes a complete version
<dir>/<split>/v<time>/ and then swaps the symlink <dir>/<split>/current to it (ExactFullTargetBuilder reads current),
so a training job never sees a half-written merge."""
import os
import time
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
out = d / f"v{time.strftime('%m%d_%H%M%S')}"
out.mkdir()
parts = sorted(q for q in (d / "parts").glob("*/") if (q / "ok.npy").exists())
oks = [np.load(q / "ok.npy") for q in parts]
n = len(oks[0])
assert not (np.sum(oks, 0) > 1).any(), "overlapping parts"
names = [f.name[:-8] for f in parts[0].glob("*_off.npy")]
ok = np.any(oks, 0)
for nm in names:
    rows, offs = [], np.zeros(n + 1, np.int64)
    cnt = np.zeros(n, np.int64)
    tabs = [(np.load(q / f"{nm}.npy", mmap_mode="r"), np.load(q / f"{nm}_off.npy")) for q in parts]
    width = tabs[0][0].shape[1]
    for r in range(n):
        for (v, o), okp in zip(tabs, oks):
            if okp[r]:
                rows.append(np.asarray(v[o[r]:o[r + 1]]))
                cnt[r] = o[r + 1] - o[r]
                break
    offs[1:] = np.cumsum(cnt)
    np.save(out / f"{nm}_off.npy", offs)
    np.save(out / f"{nm}.npy", np.concatenate(rows, 0).astype(np.float32) if rows else np.zeros((0, width), np.float32))
np.save(out / "ok.npy", ok)
(out / "meta.txt").write_text("merged parts " + " ".join(q.name for q in parts) + "\n" + (parts[0] / "meta.txt").read_text())
tmp = d / "current.tmp"
if tmp.is_symlink():
    tmp.unlink()
tmp.symlink_to(out.name)
os.replace(tmp, d / "current")
# the previous versions go (a loader worker that still maps one keeps reading it: the inode lives until unmapped)
import shutil
for v in d.glob("v[0-9]*_[0-9]*"):
    if v.is_dir() and v.name != out.name:
        shutil.rmtree(v)
print(f"merged {len(parts)} parts -> {out}: ok {ok.sum()}/{n}, tables {names}")
