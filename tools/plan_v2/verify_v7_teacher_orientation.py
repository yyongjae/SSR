"""Measure (do not assume) the student-layout orientation of both v7 teacher caches.

    BEVFusion  its dense_heatmap shares the BEV layout; correlate the vehicle heatmap with GT agent
               centres rasterised in the student grid, under {as stored, W flip, H flip, both}.
               The loader (distill/teacher_store.py) applies the manifest's W flip.
    ReSMap     the loader's output must equal BevCache.bev (plain transpose), which
               tools/readout/verify_teacher_alignment.py measured against map GT.

    python tools/plan_v2/verify_v7_teacher_orientation.py --frames 60
"""
import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT / "tools/readout"))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.distill.teacher_store import v7_stores
    from navsim.agents.para_ssr.readout.bev_cache import BevCache
    from navsim.common.dataloader import SceneLoader

    cfg = ParaSSRConfig()
    bf, rs = v7_stores(cfg)
    bf.validate_manifest(cfg); rs.validate_manifest(cfg)
    rng = random.Random(a.seed)
    logs_dir, blobs_dir = split_dirs("trainval")
    found = {}
    for lp in rng.sample(sorted(logs_dir.glob("*.pkl")), 400):
        toks = [f["token"] for f in pickle.load(open(lp, "rb")) if bf.has_token(f["token"]) and rs.has_token(f["token"])]
        if toks:
            found[lp.stem] = rng.sample(toks, min(3, len(toks)))
        if sum(map(len, found.values())) >= a.frames:
            break
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter("navtrain", log_names=list(found),
                                                           tokens=sum(found.values(), [])))
    H, W = cfg.bev_h, cfg.bev_w
    cell = (cfg.pc_range[4] - cfg.pc_range[1]) / H
    variants = {"as stored": lambda t: t, "W flip (loader)": lambda t: t[:, :, ::-1],
                "H flip": lambda t: t[:, ::-1, :], "both": lambda t: t[:, ::-1, ::-1]}
    corr = {k: [] for k in variants}
    same = []
    bc = BevCache(str(rs.teacher_root()))
    for tok in loader.tokens:
        scene = loader.get_scene_from_token(tok)
        ann = scene.frames[scene.scene_metadata.num_history_frames - 1].annotations
        g = np.zeros((H, W), np.float32)
        for (x, y), name in zip(ann.boxes[:, :2], ann.names):
            if name != "vehicle":
                continue
            r, c = int(x / cell), int((-y - cfg.pc_range[0]) / cell)      # student: row = forward, col = right
            if 0 <= r < H and 0 <= c < W:
                g[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = 1
        if g.sum() < 9:
            continue
        with np.load(bf.path_for(tok)) as z:
            hm = z["dense_heatmap"].astype(np.float32)
        for k, fn in variants.items():
            corr[k].append(np.corrcoef(np.ascontiguousarray(fn(hm))[0].ravel(), g.ravel())[0, 1])
        same.append(np.array_equal(rs.load(tok), bc.bev(tok).astype(np.float16)))
    print(f"{len(corr['as stored'])} frames with >= 1 vehicle in the grid")
    for k, v in corr.items():
        print(f"  BEVFusion heatmap {k:16s} corr {np.nanmean(v):.3f}")
    print(f"  ReSMap loader == BevCache.bev (plain transpose): {np.mean(same) * 100:.0f}% of {len(same)}")


if __name__ == "__main__":
    main()
