"""Tests for navsim/agents/para_ssr/refiner/adapters.py and the teacher-cache guard / S-grid flip (CPU, < 1 min).

The flip test reads REAL teacher npz (cache_train_50x100 on 10 train-split tokens and cache_val_50x100 on 10 navtest
tokens): for every predicted box with score > 0.5 inside the grid, the class heatmap in the S grid (teacher_to_s_grid)
must be higher at the box cell than at the laterally mirrored cell, the box cell must usually be the 3x3 local peak,
and corridor.sample_s_grid (the refiner's grid_sample convention) read at the box centre must beat +-1 m shifts.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_adapters.py
"""
from __future__ import annotations

import json
import os
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
from navsim.agents.para_ssr.refiner.corridor import sample_s_grid  # noqa: E402

FUTURE = Path("/home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100_future")
TRAIN_SPLIT = RD.DATA_ROOT / "splits" / "train.parquet"
NAVTEST_TOKENS = RD.DATA_ROOT / "objects" / "_validation" / "navtest_all_token_log.parquet"


# ----------------------------------------------------------------------------------------------- guard
def test_teacher_manifest_guard(tmp_path):
    tc = RD.TeacherCache.for_subset("navtrain")
    assert tc.sha_head == RD.TEACHER_SHA_HEAD == "cddf943ffec8d6a8"
    assert RD.TeacherCache.for_subset("navtest").sha_head == RD.TEACHER_SHA_HEAD
    with pytest.raises(ValueError, match="_future"):
        RD.TeacherCache(FUTURE)
    link = tmp_path / "innocent_name"
    link.symlink_to(FUTURE)                      # a renamed path to the future cache is still refused
    with pytest.raises(ValueError, match="_future"):
        RD.TeacherCache(link)
    bad = tmp_path / "bad_cache"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps({"checkpoint_sha256_head": "0000000000000000",
                                                   "layout": RD.TEACHER_LAYOUT}))
    with pytest.raises(ValueError, match="checkpoint_sha256_head"):
        RD.TeacherCache(bad)
    with pytest.raises(FileNotFoundError):
        RD.TeacherCache(tmp_path)


def test_load_bev_is_flipped_copy():
    tc = RD.TeacherCache.for_subset("navtrain")
    tok = str(pd.read_parquet(TRAIN_SPLIT).token.iloc[0])
    raw = tc.load_bev(tok, s_grid=False)
    s = tc.load_bev(tok)
    assert raw.shape == s.shape == (256, 50, 100) and s.dtype == np.float16
    assert np.array_equal(s, raw[:, :, ::-1]) and s.flags["C_CONTIGUOUS"]
    assert torch.equal(AD.teacher_to_s_grid(torch.as_tensor(raw)), torch.as_tensor(s))


# ----------------------------------------------------------------------------------------------- flip vs boxes
def _flip_stats(tc, tokens):
    a, b, peak, shift, row = [], [], [], [], []
    for tk in tokens:
        det = tc.load_det(tk)                                      # heatmap already on the S grid
        hm = det["dense_heatmap"].astype(np.float32)
        B, s, L = det["pred_boxes_3d"], det["pred_scores_3d"], det["pred_labels_3d"]
        sel = (s > 0.5) & (B[:, 0] > 1) & (B[:, 0] < 31) & (np.abs(B[:, 1]) < 31)
        for k in np.flatnonzero(sel):
            x, y = float(B[k, 0]), float(B[k, 1])
            rr, cc = AD.s_grid_cell(x, y)
            r, c = int(round(float(rr))), int(round(float(cc)))
            h = hm[L[k]]
            a.append(h[r, c])
            b.append(h[r, 99 - c])
            row.append(h[r, c] > h[49 - r, c])
            peak.append(h[r, c] >= h[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].max() - 1e-6)
            pts = torch.tensor([[[[x, y], [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]]], dtype=torch.float32)
            v = sample_s_grid(torch.as_tensor(hm[None]), pts)[0, L[k], 0].numpy()
            shift.append(v[0] > v[1:].max())
    return np.array(a), np.array(b), np.array(peak), np.array(shift), np.array(row)


@pytest.mark.parametrize("subset", ["navtrain", "navtest"])
def test_s_grid_flip_heatmap_vs_boxes(subset):
    tc = RD.TeacherCache.for_subset(subset)
    if subset == "navtrain":
        toks = pd.read_parquet(TRAIN_SPLIT).token.values[::2400][:10]
    else:
        toks = pd.read_parquet(NAVTEST_TOKENS).token.values[::1000][:10]
    a, b, peak, shift, row = _flip_stats(tc, toks)
    assert len(a) >= 40, len(a)
    assert (a > b).mean() == 1.0                     # lateral: box cell beats the mirrored cell for every box
    assert np.median(a) > 0 > np.median(b)
    assert row.mean() > 0.9                          # longitudinal direction is not flipped
    assert peak.mean() > 0.75                        # the box cell is the local 3x3 peak (no half-cell offset)
    assert shift.mean() > 0.85                       # grid_sample convention: centre beats +-1 m in x and y


# ----------------------------------------------------------------------------------------------- stats / modules
def test_channel_stats_matches_numpy():
    rng = np.random.default_rng(0)
    maps = [np.maximum(rng.normal(0.2, 0.6, size=(5, 4, 6)), 0) for _ in range(7)]
    maps[3][2] = 0.0
    mean, std, info = AD.compute_norm(iter(maps), n_ch=5)
    X = np.stack(maps).transpose(1, 0, 2, 3).reshape(5, -1)
    np.testing.assert_allclose(mean, X.mean(1), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(std, np.maximum(X.std(1), AD.STD_FLOOR), rtol=1e-5)
    assert info["n_maps"] == 7 and info["n_cells"] == 7 * 24
    zero = AD.compute_norm(iter([np.zeros((5, 2, 2))]), n_ch=5)[1]
    assert np.all(zero == AD.STD_FLOOR)


def test_norm_roundtrip(tmp_path):
    m, s = np.arange(256, dtype=np.float32), np.linspace(0.5, 2, 256).astype(np.float32)
    AD.save_norm(tmp_path / "norm.npz", m, s, {"n_tokens": 3})
    m2, s2, meta = AD.load_norm(tmp_path / "norm.npz")
    assert np.array_equal(m, m2) and np.array_equal(s, s2) and meta == {"n_tokens": 3}


def test_adapter_modules():
    rng = np.random.default_rng(0)
    mean, std = rng.uniform(0.1, 0.4, 256).astype(np.float32), rng.uniform(0.3, 0.9, 256).astype(np.float32)
    ad = AD.build_adapter("T", mean, std)
    assert sum(p.numel() for p in ad.parameters()) == 256 * 128 + 128 + 128 * 64 + 64 == 41152
    assert sum(b.numel() for n, b in ad.named_buffers() if n.startswith("norm.")) == 256 * 2 + 1
    x = torch.as_tensor(rng.uniform(0, 3, size=(2, 256, 50, 100)).astype(np.float16))
    z = ad.norm(x.float())
    np.testing.assert_allclose(z.numpy(), (x.float().numpy() - mean[:, None, None]) / std[:, None, None], rtol=1e-5, atol=1e-5)
    y = ad(x)
    assert y.shape == (2, 64, 50, 100) and torch.isfinite(y).all()
    with pytest.raises(RuntimeError, match="normalisation"):
        AD.build_adapter("T")(x)
    with pytest.raises(ValueError):
        ad(x[:, :100])
    none = AD.build_adapter("none")
    assert sum(p.numel() for p in none.parameters()) == 0
    zz = none(None, n_tokens=3)
    assert zz.shape == (3, 64, 50, 100) and not zz.any()
    with pytest.raises(ValueError):
        AD.build_adapter("S")
