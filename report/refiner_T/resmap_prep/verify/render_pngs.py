"""Render S-grid seg road / centerline (chosen transform, lateral mirror, forward flip) vs independent GT.
usage: render_pngs.py <gt_dir> <out_dir>   (needs per_token_seg.parquet from verify_axes.py in out_dir)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = Path("/home/external-user/datasets/teacher_cache/resmap")
IDX = {"navtrain": json.load(open(R / "index.json")), "navtest": json.load(open(R / "navtest/index.json"))}
ROOT = {"navtrain": R, "navtest": R / "navtest"}


def seg_raw(sub, t):
    shard, row = IDX[sub][t]
    return np.asarray(np.load(ROOT[sub] / "seg" / f"{shard}.npy", mmap_mode="r")[row], np.float32)


def gt(t):
    z = np.load(GT / f"{t}.npz")
    dac = np.unpackbits(z["dac"])[: 124 * 212].reshape(124, 212).astype(bool)[12:112, 6:206]
    return dac, z["cl"]


df = pd.read_parquet(OUT / "per_token_seg.parquet")
rng = np.random.default_rng(7)
pick = []
for sub, n in (("navtrain", 3), ("navtest", 2)):
    for k in ("left", "right", "intersection", "other"):
        c = df[(df.subset == sub) & (df.kind == k)]
        pick += [(r.token, sub, k) for r in c.sample(n, random_state=int(rng.integers(1e6))).itertuples()]
# plus the worst chosen-transform tokens (diagnostic)
for sub, n in (("navtrain", 1), ("navtest", 2)):
    c = df[df.subset == sub].nsmallest(n, "chosen(transpose):iou")
    pick += [(r.token, sub, "worst") for r in c.itertuples()]

sig = lambda x: 1 / (1 + np.exp(-x))
EXT = (32, -32, 0, 32)          # S orientation: col 0 = +32 m (left) drawn on the left; row 0 = x 0 at the bottom
hyps = {"chosen (transpose)": lambda x: np.swapaxes(x, -1, -2),
        "lateral mirror": lambda x: np.swapaxes(x, -1, -2)[..., ::-1],
        "forward flip": lambda x: np.swapaxes(x, -1, -2)[..., ::-1, :]}
thumbs = []
for i, (t, sub, kind) in enumerate(pick):
    seg = seg_raw(sub, t)
    dac, cl = gt(t)
    row = df[df.token == t].iloc[0]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.6), constrained_layout=True)
    panels = [("chosen (transpose)", 0, "road"), ("chosen (transpose)", 2, "centerline"),
              ("lateral mirror", 0, "road"), ("forward flip", 0, "road")]
    for a, (h, ch, nm) in zip(ax, panels):
        S = hyps[h](seg)[ch]
        a.imshow(sig(S), origin="lower", extent=EXT, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        a.contour(np.linspace(32 - 0.16, -32 + 0.16, 200), np.linspace(0.16, 32 - 0.16, 100), dac.astype(float),
                  levels=[0.5], colors=["#d95f02"], linewidths=1.4)
        a.plot(cl[:, 1], cl[:, 0], color="#1b9e77", lw=2.0)
        a.plot([0], [0], marker="^", color="#7570b3", ms=12)
        a.set_xlim(32, -32); a.set_ylim(0, 32); a.set_aspect("equal")
        a.set_xlabel("y_left [m]  (left of ego on the left)"); a.set_ylabel("x forward [m]")
        key = {"chosen (transpose)": "chosen(transpose)", "lateral mirror": "lateral_mirror",
               "forward flip": "forward_flip"}[h]
        a.set_title(f"{h}: seg {nm} sigmoid\nroad IoU {row[key + ':iou']:.3f}, CL logit@GT {row[key + ':cl_logit']:+.2f}",
                    fontsize=10)
    fig.suptitle(f"{t}  [{sub}, {kind}, v={row.speed:.1f} m/s, route-CL heading change {row.cl_head:+.2f} rad]  "
                 f"orange = GT DAC drivable boundary (metric cache), green = GT route centerline, triangle = rear axle",
                 fontsize=10)
    fn = OUT / f"{i:02d}_{sub}_{kind}_{t}.png"
    fig.savefig(fn, dpi=80); plt.close(fig)
    thumbs.append((t, sub, kind, seg, dac, cl, row))
    print(fn.name, round(row["chosen(transpose):iou"], 3), round(row["lateral_mirror:iou"], 3),
          round(row["forward_flip:iou"], 3))

# contact sheet: chosen road for all picked tokens
n = len(thumbs); nc = 6; nr = int(np.ceil(n / nc))
fig, ax = plt.subplots(nr, nc, figsize=(nc * 4, nr * 2.4), constrained_layout=True)
for a in ax.flat:
    a.axis("off")
for a, (t, sub, kind, seg, dac, cl, row) in zip(ax.flat, thumbs):
    a.imshow(sig(np.swapaxes(seg, -1, -2)[0]), origin="lower", extent=EXT, cmap="Greys", vmin=0, vmax=1)
    a.contour(np.linspace(32 - 0.16, -32 + 0.16, 200), np.linspace(0.16, 32 - 0.16, 100), dac.astype(float),
              levels=[0.5], colors=["#d95f02"], linewidths=0.8)
    a.plot(cl[:, 1], cl[:, 0], color="#1b9e77", lw=1.2)
    a.set_xlim(32, -32); a.set_ylim(0, 32); a.set_aspect("equal")
    a.set_title(f"{sub[:5]} {kind} IoU {row['chosen(transpose):iou']:.2f}\n{t}", fontsize=8)
fig.savefig(OUT / "contact_sheet_chosen.png", dpi=80)
