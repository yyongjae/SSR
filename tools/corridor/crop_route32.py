#!/usr/bin/env python
"""km 10/16: route32 (v3-32 route channels, 50 x 100 at 0.64 m, forward 0..32 m) = the first 50 rows of
build_corridor64.py's route_s / route_d (same cells: centres (i + 0.5) * 0.64 m forward, same lateral grid)."""
import sys
from pathlib import Path
import numpy as np
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for sp in ("navtrain_val", "navtrain_train"):
    (dst / sp).mkdir(parents=True, exist_ok=True)
    for k in ("route_s", "route_d"):
        a = np.load(src / sp / f"{k}.npy", mmap_mode="r")
        o = np.lib.format.open_memmap(dst / sp / f"{k}.npy", "w+", np.float16, (a.shape[0], 50, 100))
        for i in range(0, a.shape[0], 4096):
            o[i:i + 4096] = a[i:i + 4096, :50]
        o.flush()
    np.save(dst / sp / "ok.npy", np.load(src / sp / "ok.npy"))
    print(sp, "ok", int(np.load(dst / sp / "ok.npy").sum()), flush=True)
(dst / "DONE").write_text("ok\n")
