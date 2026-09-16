"""Measure (do not assume) how the ReSMap cache maps onto PARA-SSR's grid.

Two independent checks against PARA-SSR's own map GT (ParaSSRTargetBuilder):

1. BEV axes: the teacher's segmentation logits share its BEV layout (the seg
   decoder is convolutional).  Correlate them with rasterised GT under the
   four transpose/flip variants.  BevCache uses plain transpose.
2. Vector convention: chamfer (m) between the teacher's confident vectors and
   GT under the eight swap/flip variants.  BevCache uses (1 - v, u).

Measured on 60 navtrain frames (2026-09-16):

    transpose              corr 0.365    <- BevCache.bev
    transpose + flip L/R   corr 0.051
    transpose + flip F/B   corr 0.142
    transpose + both       corr 0.016

    (1 - v, u)             chamfer 0.43 m <- teacher_vectors_to_student
    next best              chamfer 5.43 m

    python tools/readout/verify_teacher_alignment.py --cache /data3/kyungmin/kd_teacher_resmap
"""
import argparse
import itertools
import pickle
import random

import numpy as np

import _env  # noqa: F401
from _env import scene_filter, split_dirs


def raster(pts, valid, h, w):
    img = np.zeros((h, w), np.float32)
    for p, v in zip(pts, valid):
        if not v:
            continue
        q = p[0]
        dense = np.concatenate([np.linspace(q[k], q[k + 1], 20) for k in range(len(q) - 1)])
        img[np.clip((dense[:, 1] * h).astype(int), 0, h - 1),
            np.clip((dense[:, 0] * w).astype(int), 0, w - 1)] = 1
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--logs", type=int, default=20)
    ap.add_argument("--per-log", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.para_ssr_targets import ParaSSRTargetBuilder
    from navsim.agents.para_ssr.readout.bev_cache import BevCache, teacher_vectors_to_student
    from navsim.common.dataloader import SceneLoader

    cache = BevCache(args.cache)
    logs_dir, blobs_dir = split_dirs(args.split)
    rng = random.Random(args.seed)
    found = {}
    for lp in rng.sample(sorted(logs_dir.glob("*.pkl")), min(20 * args.logs, 2000)):
        toks = [f["token"] for f in pickle.load(open(lp, "rb")) if f["token"] in cache]
        if toks:
            found[lp.stem] = rng.sample(toks, min(args.per_log, len(toks)))
        if len(found) >= args.logs:
            break
    cfg = ParaSSRConfig()
    tb = ParaSSRTargetBuilder(cfg, cfg.trajectory_sampling)
    sf = scene_filter("navtrain", log_names=list(found), tokens=sum(found.values(), []))
    loader = SceneLoader(logs_dir, blobs_dir, sf)

    grids = {
        "transpose": lambda t: t.transpose(0, 2, 1),
        "transpose+flipLR": lambda t: t.transpose(0, 2, 1)[:, :, ::-1],
        "transpose+flipFB": lambda t: t.transpose(0, 2, 1)[:, ::-1, :],
        "transpose+both": lambda t: t.transpose(0, 2, 1)[:, ::-1, ::-1],
    }
    vecs = {}
    for swap, fa, fb in itertools.product((0, 1), repeat=3):
        def f(v, swap=swap, fa=fa, fb=fb):
            a, b = (v[..., 1], v[..., 0]) if swap else (v[..., 0], v[..., 1])
            return np.stack([1 - a if fa else a, 1 - b if fb else b], -1)
        vecs[f"swap{swap}_flipA{fa}_flipB{fb}"] = f
    probe = np.array([[0.1, 0.7], [0.3, 0.2]])
    reference = teacher_vectors_to_student(probe)
    corr = {k: [] for k in grids}
    cham = {k: [] for k in vecs}
    H, W = cfg.bev_h, cfg.bev_w
    for tok in loader.tokens:
        scene = loader.get_scene_from_token(tok)
        tg = tb._compute_map_targets(scene, scene.scene_metadata.num_history_frames - 1)
        pts, lab, val = (tg[k].numpy() for k in ("gt_map_pts", "gt_map_labels", "gt_map_valid"))
        seg = cache.raw(tok, "seg").astype(np.float32)
        prob = 1 / (1 + np.exp(-seg.reshape(seg.shape[0], seg.shape[1] // 2, 2, seg.shape[2] // 2, 2).mean((2, 4))))
        tv = cache.raw(tok, "vectors").astype(np.float32)
        ts = cache.raw(tok, "scores").astype(np.float32)
        tl = cache.raw(tok, "labels")
        for c in range(cfg.map_num_classes):
            sel = (lab == c) & val
            g = raster(pts[sel], val[sel], H, W)
            if g.sum() >= 5:
                for k, fn in grids.items():
                    corr[k].append(np.corrcoef(np.ascontiguousarray(fn(prob))[c].ravel(), g.ravel())[0, 1])
            gp = pts[sel][:, 0].reshape(-1, 2)
            keep = (tl == c) & (ts > 0.4)
            if len(gp) >= 5 and keep.any():
                for k, fn in vecs.items():
                    p = fn(tv[keep]).reshape(-1, 2)
                    d = np.linalg.norm((p[:, None] - gp[None]) * [64.0, 32.0], axis=-1)
                    cham[k].append(0.5 * (d.min(1).mean() + d.min(0).mean()))
    print(f"{len(loader.tokens)} frames")
    for k, v in corr.items():
        print(f"  BEV {k:20s} corr {np.nanmean(v):.3f}  (n={len(v)})")
    for k, v in sorted(cham.items(), key=lambda kv: np.mean(kv[1])):
        mark = "  <- BevCache" if np.allclose(vecs[k](probe), reference) else ""
        print(f"  vec {k:20s} chamfer {np.mean(v):5.2f} m  (n={len(v)}){mark}")


if __name__ == "__main__":
    main()
