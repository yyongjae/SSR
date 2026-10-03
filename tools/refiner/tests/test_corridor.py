"""Tests for navsim/agents/para_ssr/refiner/corridor.py (M3a corridor + M3b global tokens; CPU, < 30 s).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_corridor.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.refiner import corridor as C  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import kappa_limit  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "decoder_navtest_sample.npz"


def straight(v, y=0.0, n=1):
    t = 0.5 * torch.arange(1, 9, dtype=torch.float32)
    return torch.stack([v * t, torch.full_like(t, y), torch.zeros_like(t)], -1)[None].repeat(n, 1, 1)


def coord_map():
    """S-grid map with channel 0 = x and channel 1 = y_left of each cell centre."""
    r = (torch.arange(50, dtype=torch.float32) + 0.5) * 0.64
    c = 32 - (torch.arange(100, dtype=torch.float32) + 0.5) * 0.64
    return torch.stack([r[:, None].expand(50, 100), c[None, :].expand(50, 100)])[None]   # [1, 2, 50, 100]


def test_grid_convention_reproduces_coordinates():
    rng = np.random.default_rng(0)
    xy = torch.as_tensor(np.stack([rng.uniform(0.32, 31.68, 500), rng.uniform(-31.68, 31.68, 500)], -1),
                         dtype=torch.float32).reshape(1, 1, 500, 2)
    v = C.sample_s_grid(coord_map(), xy)[0, :, 0]                                          # [2, 500]
    torch.testing.assert_close(v[0], xy[0, 0, :, 0], atol=2e-5, rtol=0)
    torch.testing.assert_close(v[1], xy[0, 0, :, 1], atol=2e-5, rtol=0)
    # normalised coordinates of the grid corners / centre
    g = C.s_grid_normalized(torch.tensor([[0.0, 32.0], [32.0, -32.0], [16.0, 0.0]]))
    torch.testing.assert_close(g, torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]))


def test_off_grid_reads_zero():
    f = torch.ones(1, 3, 50, 100)
    xy = torch.tensor([[[[-1.0, 0.0], [33.0, 0.0], [5.0, 40.0], [5.0, -40.0], [5.0, 0.0], [0.1, 0.0]]]])
    v = C.sample_s_grid(f, xy)[0, 0, 0]
    assert torch.equal(v[:4], torch.zeros(4)) and v[4] == 1.0
    assert 0.0 < v[5] < 1.0                                    # outer half cell: blended with the zero padding
    assert C.in_s_grid(xy[0, 0]).tolist() == [False, False, False, False, True, True]


def test_straight_draft_geometry():
    g = C.corridor_geometry(straight(5.0))
    S8, v_end = 20.0, 5.0
    assert float(g.S_look[0]) == pytest.approx(S8 + 8.0)
    s_exp = (torch.arange(48) + 0.5) * 1.0                    # ds = max(1, 28 / 48) = 1
    torch.testing.assert_close(g.s[0], s_exp)
    d = C.lateral_offsets()
    torch.testing.assert_close(g.points[0, :, :, 0], s_exp[:, None].expand(48, 17), atol=1e-5, rtol=0)
    torch.testing.assert_close(g.points[0, :, :, 1], d[None].expand(48, 17), atol=1e-5, rtol=0)
    geo = g.geo[0]
    assert geo.shape == (6, 48, 17)
    assert torch.equal(geo[1, :, 0], (s_exp <= S8).float())
    torch.testing.assert_close(geo[2, :, 0], s_exp / 48)
    torch.testing.assert_close(geo[3, 0], d / 4.8)
    torch.testing.assert_close(geo[4, :, 0], torch.clamp(s_exp / v_end / 4, 0, 1), atol=1e-6, rtol=0)
    torch.testing.assert_close(geo[5, :, 0], torch.full((48,), v_end / 15))
    inside = (s_exp <= 32)[:, None] & (d.abs() <= 32)[None]
    assert torch.equal(geo[0].bool(), inside)
    assert not bool(g.near_stop[0]) and not bool(g.ext_clipped[0])


def test_long_draft_spacing_and_extension():
    g = C.corridor_geometry(straight(15.0))                   # S8 = 60, S_look = 75, ds = 75/48
    ds = 75.0 / 48
    torch.testing.assert_close(g.s[0], (torch.arange(48) + 0.5) * ds)
    assert float(g.s[0, -1]) + ds / 2 == pytest.approx(75.0, abs=1e-4)
    torch.testing.assert_close(g.points[0, :, 8, 0], g.s[0], atol=1e-4, rtol=0)   # straight extension continues


def test_arrival_time_first_arrival_and_standstill():
    # knots S = [0, 2, 4, 4, 4, 6, 8, 10, 12]: stop between t = 1.0 and 2.0 s
    S = torch.tensor([[0, 2, 4, 4, 4, 6, 8, 10, 12]], dtype=torch.float32)
    s = torch.tensor([[0.0, 1.0, 4.0, 4.5, 12.0, 14.0]])
    t, v = C.arrival_time_speed(S, s, torch.tensor([4.0]))
    torch.testing.assert_close(t[0], torch.tensor([0.0, 0.25, 1.0, 2.125, 4.0, 4.5]))
    torch.testing.assert_close(v[0], torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0, 4.0]))
    # monotone in s on a real draft
    z = np.load(FIX)
    tau = torch.as_tensor(z["tau_h"][:64])
    g = C.corridor_geometry(tau)
    t, _ = C.arrival_time_speed(g.S, g.s, g.v_end)
    assert bool((t[:, 1:] >= t[:, :-1] - 1e-5).all())
    # exact at the knots (first arrival)
    tk, _ = C.arrival_time_speed(g.S, g.S, g.v_end)
    Su = g.S
    first = torch.ones_like(Su, dtype=torch.bool)
    first[:, 1:] = Su[:, 1:] > Su[:, :-1]
    torch.testing.assert_close(tk[first], (0.5 * torch.arange(9.0))[None].expand_as(Su)[first], atol=1e-5, rtol=0)


def test_curved_draft_and_clipped_extension():
    R, v = 20.0, 5.0                                           # arc of radius 20 m, 5 m/s: kappa_lim(5) = 0.19 > 0.05
    t = 0.5 * torch.arange(1, 9, dtype=torch.float64)
    th = v * t / R
    tau = torch.stack([R * torch.sin(th), R * (1 - torch.cos(th)), th], -1)[None].float()
    g = C.corridor_geometry(tau)
    rad = torch.linalg.norm(g.points[0] - torch.tensor([0.0, R]), dim=-1)                  # [48, 17]
    d = C.lateral_offsets()
    err = (rad - (R - d)[None]).abs().max(1).values                                         # [48]
    inside = g.s[0] <= g.S[0, -1]
    assert float(err[inside].max()) < 1e-3                     # on the draft: exact circle offsets
    assert float(err.max()) < 0.25                             # 28 m of extension with the spline end curvature
    assert float(g.path.kappa_ext[0]) == pytest.approx(1 / R, rel=0.02)
    assert not bool(g.ext_clipped[0])
    # a tight fast turn: the extension curvature is clipped
    R2, v2 = 25.0, 14.0                                        # kappa 0.04 > kappa_lim(14) = 4.89/196 = 0.025
    th2 = v2 * t / R2
    tau2 = torch.stack([R2 * torch.sin(th2), R2 * (1 - torch.cos(th2)), th2], -1)[None].float()
    g2 = C.corridor_geometry(tau2)
    assert bool(g2.ext_clipped[0])
    assert float(g2.path.kappa_ext[0]) == pytest.approx(float(kappa_limit(torch.tensor(float(g2.v_end[0])))), rel=1e-4)
    # near stop: straight extension, flagged
    g3 = C.corridor_geometry(straight(0.3))
    assert bool(g3.near_stop[0]) and float(g3.path.kappa_ext[0]) == 0.0


def test_sample_corridor_equals_replicated_features():
    torch.manual_seed(0)
    T, K = 3, 4
    feat = torch.randn(T, 5, 50, 100)
    z = np.load(FIX)
    tau = torch.as_tensor(z["tau_h"][:T * K])
    g = C.corridor_geometry(tau)
    X = C.corridor_tensor(feat, g, T, K)                                                   # [T*K, 11, 48, 17]
    rep = feat.repeat_interleave(K, 0)
    ref = F.grid_sample(rep, C.s_grid_normalized(g.points), mode="bilinear", padding_mode="zeros", align_corners=False)
    torch.testing.assert_close(X[:, :5], ref, atol=1e-6, rtol=0)
    torch.testing.assert_close(X[:, 5:], g.geo)
    assert X.shape == (T * K, 5 + C.N_GEO, C.N_ST, C.N_LAT)


def test_global_tokens():
    torch.manual_seed(0)
    f = torch.randn(2, 3, 50, 100)
    g = C.global_tokens(f)
    assert g.shape == (2, 200, 3) and C.N_GLOBAL == 200
    # token m = 20 * row + col is the mean of cells [5 row : 5 row + 5, 5 col : 5 col + 5]
    for m in (0, 19, 20, 137, 199):
        r, c = divmod(m, 20)
        torch.testing.assert_close(g[:, m], f[:, :, 5 * r:5 * r + 5, 5 * c:5 * c + 5].mean((-1, -2)))


def test_real_drafts_finite():
    z = np.load(FIX)
    tau = torch.as_tensor(np.concatenate([z["tau_h"], z["tau_student"]]))
    g = C.corridor_geometry(tau)
    assert torch.isfinite(g.points).all() and torch.isfinite(g.geo).all()
    assert float(g.geo[:, 4].max()) <= 1.0 and float(g.geo[:, 4].min()) >= 0.0
    # station spacing rule
    ds = g.s[:, 1] - g.s[:, 0]
    torch.testing.assert_close(ds, torch.clamp(g.S_look / 48, min=1.0), atol=1e-4, rtol=1e-5)
