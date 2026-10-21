"""Synthetic factorized vocabulary for pipeline smoke tests (NOT for training).

Writes paths_1024.npy [P, 51, 2] (arcs of various curvature) and speeds_256.npy [V, 8]
(constant-speed profiles) with the same file names the real vocabulary uses, so
scripts/training/smoke_fact.sh can point VOCAB at this directory.
"""
import argparse
import json
import pathlib

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--paths", type=int, default=64)
ap.add_argument("--speeds", type=int, default=16)
a = ap.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
S = 50
s = np.arange(S + 1, dtype=np.float64)
paths = np.zeros((a.paths, S + 1, 2), np.float32)
for i, k in enumerate(np.linspace(-0.03, 0.03, a.paths)):
    if abs(k) < 1e-6:
        paths[i, :, 0] = s
    else:
        paths[i, :, 0] = np.sin(k * s) / k
        paths[i, :, 1] = (1 - np.cos(k * s)) / k
speeds = np.stack([np.arange(1, 9) * 0.5 * v for v in np.linspace(0.0, 16.0, a.speeds)]).astype(np.float32)
np.save(out / "paths_1024.npy", paths)
np.save(out / "speeds_256.npy", speeds)
json.dump({"synthetic": True, "paths": a.paths, "speeds": a.speeds}, open(out / "meta.json", "w"))
print("wrote", out, paths.shape, speeds.shape)
