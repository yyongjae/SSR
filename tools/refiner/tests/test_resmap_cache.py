"""Tests for navsim/agents/para_ssr/refiner/resmap_cache.py (ReSMap map-teacher loader + S-grid transform).  CPU, < 1 min.

Synthetic: a fake sharded cache (meta.json, index.json, <field>/<shard>.npy) with coordinate-encoded values checks the
guards, has/load, per-process memmap caching, pickling, and the axis transform cell by cell.
Real: a few train_logs tokens (incl. curves) + navtest tokens: loader == raw transpose; seg road logit (S orientation)
vs the stage-T drivable SDF and vectors vs the route centerline beat every flipped / swapped alternative.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_resmap_cache.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.refiner import adapters as AD  # noqa: E402
from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner import resmap_cache as RM  # noqa: E402
from navsim.agents.para_ssr.refiner.corridor import sample_s_grid  # noqa: E402
from navsim.agents.para_ssr.refiner.sdf import sample_sdf_np  # noqa: E402

REAL = RM.RESMAP_ROOT.is_dir() and (RD.DATA_ROOT / "packed" / "train" / "sdf.npy").is_file()
needs_real = pytest.mark.skipif(not REAL, reason="ReSMap cache / packed train split not available")


# ----------------------------------------------------------------------------------------------- synthetic cache
def _make_cache(root: Path, n_per_shard=(3, 2), sha=RM.RESMAP_SHA256, split="train", bev_tail=(256, 100, 50)):
    root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((RM.RESMAP_ROOT / "meta.json").read_text()) if RM.RESMAP_ROOT.is_dir() else dict(
        classes=list(RM.RESMAP_CLASSES), pc_range=RM.RESMAP_PC_RANGE, roi_size_m=RM.RESMAP_ROI_SIZE,
        tensors={f: {"shape": list(t), "dtype": d} for f, (t, d) in RM.RESMAP_FIELDS.items()})
    meta.update(checkpoint_sha256=sha, split=split, num_frames=int(sum(n_per_shard)))
    (root / "meta.json").write_text(json.dumps(meta))
    index, k = {}, 0
    A, B = 100, 50
    aa, bb = np.meshgrid(np.arange(A), np.arange(B), indexing="ij")
    for s, n in enumerate(n_per_shard):
        name = f"r0_s{s:04d}"
        arrs = {}
        for f, (tail, dt) in RM.RESMAP_FIELDS.items():
            if f == "bev":
                tail = bev_tail
            arrs[f] = np.zeros((n,) + tail, dt)
        for r in range(n):
            tok = f"tok{k:04d}"
            index[tok] = [name, r]
            # bev[c, a, b] = k + a / 1000 (c = 0), b / 100 (c = 1), c (c >= 2)  (float16-exact enough for the checks)
            if bev_tail == (256, 100, 50):
                arrs["bev"][r, 0] = aa
                arrs["bev"][r, 1] = bb
                arrs["bev"][r, 2] = k
            arrs["seg"][r, 0] = np.repeat(np.repeat(aa, 2, 0), 2, 1) * 2      # lateral index at 0.32 m, doubled
            arrs["vectors"][r, 0, 0] = [0.25, 0.75]
            k += 1
        for f, a in arrs.items():
            (root / f).mkdir(exist_ok=True)
            np.save(root / f / f"{name}.npy", a)
    (root / "index.json").write_text(json.dumps(index))
    return index


def test_guards(tmp_path):
    with pytest.raises(FileNotFoundError):
        RM.ResmapCache(tmp_path / "nothing")
    _make_cache(tmp_path / "bad_sha", sha="0" * 64)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        RM.ResmapCache(tmp_path / "bad_sha")
    _make_cache(tmp_path / "ok")
    RM.ResmapCache(tmp_path / "ok", expect_split="train")
    with pytest.raises(ValueError, match="split"):
        RM.ResmapCache(tmp_path / "ok", expect_split="none")
    _make_cache(tmp_path / "bad_shape", bev_tail=(256, 50, 100))       # meta fine, shard layout wrong
    c = RM.ResmapCache(tmp_path / "bad_shape")
    with pytest.raises(ValueError, match="bev"):
        c.load_bev("tok0000")


def test_synthetic_load_and_transform(tmp_path):
    idx = _make_cache(tmp_path / "c")
    c = RM.ResmapCache(tmp_path / "c")
    assert c.has("tok0003") and not c.has("nope") and len(c.tokens()) == len(idx) == 5
    raw = c.load_bev("tok0003", s_grid=False)
    s = c.load_bev("tok0003")
    assert raw.shape == (256, 100, 50) and s.shape == (256, AD.BEV_H, AD.BEV_W) and s.dtype == np.float16
    assert s.flags["C_CONTIGUOUS"]
    assert float(s[2, 0, 0]) == 3.0                                            # row 0 of shard 1 is token 3
    # S[:, r, col] = raw[:, a = col, b = r]
    np.testing.assert_array_equal(s[0], np.broadcast_to(np.arange(100)[None, :], (50, 100)))   # col = lateral a
    np.testing.assert_array_equal(s[1], np.broadcast_to(np.arange(50)[:, None], (50, 100)))    # row = forward b
    # metric: S cell (r, col) centre == ReSMap raw cell (a = col, b = r) centre
    sxy = RM.s_grid_xy()
    rxy = RM.resmap_cell_xy((100, 50), RM.BEV_CELL)
    np.testing.assert_allclose(sxy, np.swapaxes(rxy, 0, 1))
    r, col = AD.s_grid_cell(sxy[..., 0], sxy[..., 1])
    np.testing.assert_allclose(r, np.arange(50)[:, None] + 0 * col, atol=1e-9)
    np.testing.assert_allclose(col, np.arange(100)[None, :] + 0 * r, atol=1e-9)
    # torch path identical
    np.testing.assert_array_equal(RM.resmap_to_s_grid(torch.from_numpy(raw)).numpy(), s)
    # load(dtype) and seg / vectors
    assert c.load("tok0001", dtype=np.float32).dtype == np.float32
    seg = c.load_seg("tok0001", s_grid=True)
    assert seg.shape == (4, 100, 200)
    np.testing.assert_array_equal(seg[0, 0, :4], [0, 0, 2, 2])                  # lateral index along the columns
    v = c.load_vectors("tok0001")
    np.testing.assert_allclose(v["vectors"][0, 0], [8.0, 16.0])                 # u 0.25 -> x 8 m; v 0.75 -> y +16 m
    # corridor reader convention: a point at an S-cell centre reads that cell
    t = torch.from_numpy(s[:2].astype(np.float32))[None]
    pts = torch.tensor([[[[sxy[7, 30, 0], sxy[7, 30, 1]]]]], dtype=torch.float32)
    val = sample_s_grid(t, pts)[0, :, 0, 0]
    assert torch.allclose(val, torch.tensor([30.0, 7.0]))


def test_memmap_cache_per_process_and_pickle(tmp_path):
    _make_cache(tmp_path / "c")
    c = RM.ResmapCache(tmp_path / "c")
    c.load_bev("tok0000")
    c.load_bev("tok0001")
    assert list(c._mm) == [("bev", "r0_s0000")]                                # one memmap per (field, shard)
    c.load_bev("tok0004")
    assert len(c._mm) == 2
    c2 = pickle.loads(pickle.dumps(c))
    assert c2._mm == {} and c2.has("tok0004")
    np.testing.assert_array_equal(c2.load_bev("tok0004"), c.load_bev("tok0004"))
    c._pid = -1                                                                 # as in a forked worker
    c.load_bev("tok0000")
    assert c._pid == os.getpid() and list(c._mm) == [("bev", "r0_s0000")]


def test_adapter_and_norm_accept_resmap(tmp_path):
    _make_cache(tmp_path / "c")
    c = RM.ResmapCache(tmp_path / "c")
    mean, std, info = RD.compute_teacher_norm(c, ["tok0000", "tok0001", "tok0004"])
    assert mean.shape == std.shape == (256,) and info["sha_head"] == RM.RESMAP_SHA_HEAD
    ad = AD.AdapterT(mean, std)
    y = ad(torch.from_numpy(np.stack([c.load_bev("tok0000"), c.load_bev("tok0004")])))
    assert y.shape == (2, AD.OUT_CH, AD.BEV_H, AD.BEV_W) and torch.isfinite(y).all()


# ----------------------------------------------------------------------------------------------- real data
def _real_rows(n_straight=6, n_curve=6):
    pdir = RD.DATA_ROOT / "packed" / "train"
    idx = pd.read_parquet(pdir / "index.parquet")
    done = np.load(pdir / "done.npy", mmap_mode="r")
    tl = RM.train_logs()
    ok = (np.asarray(done[:, RD.PART_ID["sdf"]]) == 1) & (np.asarray(done[:, RD.PART_ID["centerline"]]) == 1)
    ok &= idx.log.isin(tl).values
    yaw = np.abs(np.asarray(np.load(pdir / "human_traj.npy", mmap_mode="r")[:, -1, 2]))
    rng = np.random.default_rng(5)
    c = rng.choice(np.flatnonzero(ok & (yaw > 0.5)), n_curve, replace=False)
    s = rng.choice(np.flatnonzero(ok & (yaw < 0.1)), n_straight, replace=False)
    return idx, np.r_[s, c]


@needs_real
def test_real_manifest_and_subsets():
    tr, nt = RM.ResmapCache.for_subset("navtrain"), RM.ResmapCache.for_subset("navtest")
    assert tr.sha256 == nt.sha256 == RM.RESMAP_SHA256
    assert len(nt.index) == 12146
    tl = RM.train_logs()
    for split, n in (("train", 19732), ("dev", 6634)):
        p = RD.DATA_ROOT / "splits" / f"{split}_trainlogs.parquet"
        if not p.is_file():
            pytest.skip("run tools/refiner/make_trainlogs_splits.py")
        d = pd.read_parquet(p)
        assert len(d) == n and d.log.isin(tl).all() and d.token.map(tr.has).all()
        full = pd.read_parquet(RD.DATA_ROOT / "splits" / f"{split}.parquet")
        pd.testing.assert_frame_equal(d, RM.restrict_to_train_logs(full, tl))
    navtest = pd.read_parquet(RD.DATA_ROOT / "splits" / "navtest.parquet")
    assert navtest.token.map(nt.has).all() and not navtest.token.head(50).map(tr.has).any()


@needs_real
def test_real_axis_transform_vs_ground_truth():
    """Road logit (seg ch 0, S orientation) vs the drivable SDF; centerline vectors vs the route centerline."""
    from scipy.spatial import cKDTree

    c = RM.ResmapCache.for_subset("navtrain")
    idx, rows = _real_rows()
    pdir = RD.DATA_ROOT / "packed" / "train"
    SDF = np.load(pdir / "sdf.npy", mmap_mode="r")
    CLX, CLN = np.load(pdir / "cl_xy.npy", mmap_mode="r"), np.load(pdir / "cl_n.npy", mmap_mode="r")
    xy = RM.s_grid_xy(RM.SEG_H, RM.SEG_W, RM.SEG_CELL)                     # [100, 200, 2]
    variants = {"ours": lambda s: s, "flip_fwd": lambda s: s[:, ::-1], "flip_lat": lambda s: s[:, :, ::-1],
                "both": lambda s: s[:, ::-1, ::-1]}
    I = {k: 0 for k in variants}
    U = {k: 0 for k in variants}
    cl_d = []
    for r in rows:
        tok = str(idx.token.values[r])
        sdf = np.asarray(SDF[r], np.float32)
        val, ok = sample_sdf_np(sdf, xy)
        ins = (val > 0) & ok
        seg = c.load_seg(tok, s_grid=True).astype(np.float32)
        np.testing.assert_array_equal(seg, np.swapaxes(c.load_seg(tok), 1, 2))
        for k, f in variants.items():
            pr = (f(seg)[0] > 0) & ok
            I[k] += (pr & ins).sum()
            U[k] += (pr | ins).sum()
        bev = c.load_bev(tok)
        assert bev.shape == (256, 50, 100) and bev.dtype == np.float16
        np.testing.assert_array_equal(bev, np.swapaxes(c.load_bev(tok, s_grid=False), 1, 2))
        v = c.load_vectors(tok)
        sel = (v["labels"] == 2) & (v["scores"] > 0.5)
        if sel.any():
            cl = np.asarray(CLX[r][: min(int(CLN[r]), RD.CL_MAX)], np.float64)
            g = cl[(cl[:, 0] > 2) & (cl[:, 0] < 30) & (np.abs(cl[:, 1]) < 30)]
            if len(g):
                cl_d.append(cKDTree(v["vectors"][sel].reshape(-1, 2)).query(g)[0])
    iou = {k: I[k] / U[k] for k in variants}
    assert iou["ours"] > 0.85, iou
    assert all(iou["ours"] > iou[k] + 0.2 for k in variants if k != "ours"), iou
    d = np.concatenate(cl_d)
    assert np.median(d) < 0.5, np.median(d)                                  # GT route centerline on a predicted one


@needs_real
def test_real_bev_and_seg_share_layout():
    """Ridge probe bev (native layout) -> 2x2-pooled road logit: the unflipped target is predicted far better."""
    c = RM.ResmapCache.for_subset("navtrain")
    idx, rows = _real_rows(4, 4)
    toks = [str(idx.token.values[r]) for r in rows]
    X = {t: c.load_bev(t, s_grid=False).astype(np.float32).reshape(256, -1).T for t in toks}
    Y = {t: c.load_seg(t).astype(np.float32)[0].reshape(100, 2, 50, 2).mean((1, 3)) for t in toks}
    fit, ev = toks[::2], toks[1::2]
    sub = np.random.default_rng(0).choice(5000, 1500, replace=False)
    out = {}
    for name, f in {"none": lambda y: y, "flipA": lambda y: y[::-1], "flipB": lambda y: y[:, ::-1]}.items():
        Xf = np.concatenate([X[t][sub] for t in fit])
        Yf = np.concatenate([f(Y[t]).reshape(-1)[sub] for t in fit])
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        Z = np.c_[(Xf - mu) / sd, np.ones(len(Xf))]
        W = np.linalg.solve(Z.T @ Z + np.eye(Z.shape[1]), Z.T @ Yf)
        Xe = np.concatenate([X[t] for t in ev])
        Ye = np.concatenate([f(Y[t]).reshape(-1) for t in ev])
        P = np.c_[(Xe - mu) / sd, np.ones(len(Xe))] @ W
        out[name] = 1 - ((Ye - P) ** 2).sum() / ((Ye - Ye.mean()) ** 2).sum()
    assert out["none"] > 0.8 and out["none"] > max(out["flipA"], out["flipB"]) + 0.2, out
