"""Synthetic stage-T refiner sources for the CPU tests (not a test module).

make_sources(tmp, n) writes, for n tokens of a fake 'train' split, every source that data.pack_split reads:
  human/train.npz, drafts/train/<token>.npz, scores/train.parquet, objects/train/<token>.npz, sdf/navtrain/<token>.npz
and returns (tokens DataFrame, data.Sources with a synthetic straight centerline_fn).
make_fake_teacher(tmp, tokens) writes a manifest-valid fake teacher cache (random post-ReLU bev_feature).
Geometry: every token drives straight along +x at v_t m/s; road = |y| < 5 m (SDF = 5 - |y|); one static object on the
path at x = 25 m and one vehicle moving in the left lane; route centerline = the x axis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.agents.para_ssr.refiner import data as RD
from navsim.agents.para_ssr.refiner.gt_future import save_objects
from navsim.agents.para_ssr.refiner.sdf import grid_mesh, save_sdf

K = RD.K_DRAFTS


def straight(v: float, y: float = 0.0) -> np.ndarray:
    t = 0.5 * np.arange(1, 9)
    return np.stack([v * t, np.full(8, y), np.zeros(8)], -1).astype(np.float32)


def tokens_frame(n: int) -> pd.DataFrame:
    toks = [f"{i:02x}{'ab' * 7}" for i in range(n)]
    return pd.DataFrame({"token": toks, "log": [f"log_{i // 2}" for i in range(n)], "frame_idx": np.arange(n) * 3,
                         "fold": [i % 5 for i in range(n)], "split": "train"})


def make_sources(tmp: Path, n: int = 6, missing_drafts=(), error_tokens=(), frame_gap_tokens=()):
    tmp = Path(tmp)
    df = tokens_frame(n)
    rng = np.random.default_rng(0)
    v = 3.0 + np.arange(n, dtype=np.float32)
    # human
    (tmp / "human").mkdir(parents=True, exist_ok=True)
    traj = np.stack([straight(x) for x in v])
    np.savez(tmp / "human" / "train.npz", tokens=df.token.values.astype("U16"), traj=traj, v0=v, a0=np.zeros(n, np.float32),
             eds=np.stack([v, np.zeros(n), np.zeros(n), np.zeros(n)], -1).astype(np.float32),
             cmd=(np.arange(n) % 4).astype(np.int8), frame_gap=np.isin(np.arange(n), frame_gap_tokens))
    # drafts: identity + slower + lateral shifts
    (tmp / "drafts" / "train").mkdir(parents=True, exist_ok=True)
    for i, tk in enumerate(df.token):
        if i in missing_drafts:
            continue
        d = np.stack([straight(v[i] * (1.0 - 0.04 * k), 0.15 * (k % 3 - 1) * (k > 0)) for k in range(K)])
        np.savez(tmp / "drafts" / "train" / f"{tk}.npz", drafts=d, family=np.arange(K, dtype=np.int8) % 9,
                 params=rng.normal(size=(K, 6)).astype(np.float32), valid=np.array([True] * (K - 1) + [i % 2 == 0]))
    # scores
    rows = []
    for i, tk in enumerate(df.token):
        for k in range(K):
            nc = float(rng.random() > 0.2)
            dac = float(rng.random() > 0.1)
            ep = float(rng.uniform(0.5, 1.0))
            rows.append(dict(token=tk, log=df.log[i], k=k, nc=nc, dac=dac, ddc=1.0, ep=ep, ttc=float(rng.random() > 0.1),
                             comfort=1.0, pdms=nc * dac * (5 * ep + 7) / 12, raw_progress=float(4 * v[i]),
                             pdm_progress_eff=float(4 * v[i]), error="bad" if (i in error_tokens and k == 0) else None))
    (tmp / "scores").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(tmp / "scores" / "train.parquet", index=False)
    # objects: static on path at x=25, vehicle in the left lane moving +x at 4 m/s
    (tmp / "objects" / "train").mkdir(parents=True, exist_ok=True)
    for i, tk in enumerate(df.token):
        A = 2 + i
        kf = np.zeros((A, 11, 6), np.float32)
        first = np.zeros((A, 6), np.float32)
        meta = np.zeros((A, 5), np.int16)
        kf[0, :, 0], kf[0, :, 5] = 25.0, 1.0
        first[0] = (0.5, 0.5, 0, 0, 0, 0)
        meta[0] = (3, 0, 0, 10, 0)                       # traffic cone, static
        kf[1, :, 0], kf[1, :, 1], kf[1, :, 3], kf[1, :, 5] = 10 + 2.0 * np.arange(11), 3.5, 4.0, 1.0
        first[1] = (4.5, 2.0, 0, 4.0, 0, 0)
        meta[1] = (0, 1, 0, 10, 0)                       # vehicle
        for j in range(2, A):                            # far objects
            kf[j, :, 0], kf[j, :, 1], kf[j, :, 5] = -30.0 - 5 * j, 20.0, 1.0
            first[j] = (4.0, 2.0, 0, 0, 0, 0)
            meta[j] = (0, 1, 0, 10, 0)
        ego_kf = np.stack([v[i] * 0.5 * np.arange(11), np.zeros(11), np.zeros(11)], -1).astype(np.float32)
        save_objects(tmp / "objects" / "train" / f"{tk}.npz",
                     dict(kf=kf, first=first, meta=meta, R=np.float32(80.0), n_kf=np.int32(11), ego_kf=ego_kf,
                          track=np.array([f"trk{j:03d}" for j in range(A)])))
    # sdf: road |y| < 5
    (tmp / "sdf" / "navtrain").mkdir(parents=True, exist_ok=True)
    X, Y = grid_mesh()
    field = np.clip(5.0 - np.abs(Y), -10, 10)
    for tk in df.token:
        save_sdf(tmp / "sdf" / "navtrain" / f"{tk}.npz", field, tk)

    def cl_fn(token, log):
        n = 700
        xy = np.zeros((RD.CL_MAX, 2), np.float32)
        xy[:n, 0] = -30.0 + 0.25 * np.arange(n)                 # straight route along +x, 0.25 m vertices
        return {"cl_xy": xy, "cl_valid": np.arange(RD.CL_MAX) < n, "cl_n": np.int32(n)}

    src = RD.Sources("train", root=tmp, centerline_fn=cl_fn)
    return df, src


def make_fake_teacher(tmp: Path, tokens) -> Path:
    root = Path(tmp) / "teacher_fake_50x100"
    (root / "samples").mkdir(parents=True, exist_ok=True)
    man = {"checkpoint_sha256_head": RD.TEACHER_SHA_HEAD, "layout": RD.TEACHER_LAYOUT, "target_bev_shape": [50, 100],
           "bev_channels": 256}
    (root / "manifest.json").write_text(json.dumps(man))
    rng = np.random.default_rng(1)
    for tk in tokens:
        (root / "samples" / tk[:2]).mkdir(exist_ok=True)
        bev = np.maximum(rng.normal(0.2, 0.6, size=(256, 50, 100)), 0).astype(np.float16)
        np.savez(root / "samples" / tk[:2] / f"{tk}.npz", bev_feature=bev,
                 dense_heatmap=np.zeros((7, 50, 100), np.float16), pred_boxes_3d=np.zeros((200, 9), np.float32),
                 pred_scores_3d=np.zeros(200, np.float32), pred_labels_3d=np.zeros(200, np.int16))
    return root


def fake_resmap_bev(token: str) -> np.ndarray:
    """Deterministic fake ReSMap bev [256, 100, 50] (raw cache layout) of a token (seeded by the token string)."""
    import hashlib

    rng = np.random.default_rng(int(hashlib.sha256(token.encode()).hexdigest()[:8], 16))
    return rng.normal(0.0, 1.0, size=(256, 100, 50)).astype(np.float16)


def make_fake_resmap(tmp: Path, tokens, name: str = "resmap_fake", source=None) -> Path:
    """A meta-valid fake ReSMap cache (resmap_cache.ResmapCache) holding only the 'bev' field for `tokens`, 2 shards.
    source: optional {token: source token}; token t then holds fake_resmap_bev(source[t]) (a physically permuted cache)."""
    from navsim.agents.para_ssr.refiner import resmap_cache as RM

    root = Path(tmp) / name
    (root / "bev").mkdir(parents=True, exist_ok=True)
    tokens = [str(t) for t in tokens]
    meta = dict(checkpoint_sha256=RM.RESMAP_SHA256, split="train", num_frames=len(tokens),
                classes=list(RM.RESMAP_CLASSES), pc_range=RM.RESMAP_PC_RANGE, roi_size_m=RM.RESMAP_ROI_SIZE,
                tensors={f: {"shape": list(t), "dtype": d} for f, (t, d) in RM.RESMAP_FIELDS.items()})
    (root / "meta.json").write_text(json.dumps(meta))
    half = (len(tokens) + 1) // 2
    index = {}
    for s, part in enumerate((tokens[:half], tokens[half:])):
        if not part:
            continue
        shard = f"r0_s{s:04d}"
        arr = np.stack([fake_resmap_bev((source or {}).get(t, t)) for t in part])
        np.save(root / "bev" / f"{shard}.npy", arr)
        index.update({t: [shard, i] for i, t in enumerate(part)})
    (root / "index.json").write_text(json.dumps(index))
    return root
