"""Unit tests for navsim/agents/para_ssr/refiner/geometry.py (DraftPath, dense reference).  CPU, < 1 min.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest tools/refiner/tests/test_geometry.py -q
Data: tools/refiner/tests/fixtures/decoder_navtest_sample.npz (301 navtest tokens, provenance in the .json next to
it) and, when present, the full navtest student/human trajectory pickles (12,146 tokens).
"""
import math
import os
import pickle
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "report", "planner_vs_perception_tests", "safety_filter"))

from navsim.agents.para_ssr.refiner.geometry import (DraftPath, KAPPA_MAX, dense_reference, kappa_limit,  # noqa: E402
                                                     unwrap_anchored, wrap_angle)

torch.set_num_threads(1)
FIX = os.path.join(os.path.dirname(__file__), "fixtures", "decoder_navtest_sample.npz")
STUDENT_PKL = os.path.join(REPO, "work_dirs", "eval", "para_ssr_interaction_final_navtest_trajectories.pkl")
HUMAN_PKL = os.path.join(REPO, "report", "navsim_version_audit", "adversarial_verify", "human_navtest_trajectories.pkl")


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIX, allow_pickle=True))


@pytest.fixture(scope="module")
def real_trajs(fx):
    """All navtest student + human trajectories when available, else the fixture ones."""
    out = []
    for p in (STUDENT_PKL, HUMAN_PKL):
        if os.path.exists(p):
            d = pickle.load(open(p, "rb"))["trajectories"]
            out.append(np.stack([d[k] for k in sorted(d)]).astype(np.float32))
    if not out:
        out = [fx["tau_student"], fx["tau_h"]]
    return np.concatenate(out, 0)


def _circle(R, n=8, ds=4.0, left=True):
    """Poses on a circle of radius R through the origin with heading 0 at the origin."""
    s = np.arange(1, n + 1) * ds
    th = s / R * (1 if left else -1)
    x = R * np.sin(np.abs(th))
    y = (R - R * np.cos(th)) * (1 if left else -1)
    return np.stack([x, y, th], -1)


# ---------------------------------------------------------------------------------------------------- identity
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_knots_exact_on_real_navtest(real_trajs, dtype):
    tau = torch.as_tensor(real_trajs).to(dtype)
    p = DraftPath(tau)
    e = p.eval(p.S[:, 1:])
    # Gamma_eff(S_k) == V_k bitwise except where two vertices share the same arc length (stand-still duplicates:
    # searchsorted picks the later one; the decoder anchors to the pose itself, see test_decoder)
    uniq = torch.cat([p.S[:, 1:-1] < p.S[:, 2:], torch.ones_like(p.S[:, :1], dtype=torch.bool)], 1)
    assert torch.equal(e.xy[uniq], tau[..., :2][uniq])
    assert torch.equal(e.heading[uniq], unwrap_anchored(torch.cat([tau.new_zeros(len(tau), 1), tau[..., 2]], 1))[:, 1:][uniq])
    assert (~uniq).float().mean() < 0.01
    assert torch.equal(p.knots()[:, 0], torch.zeros(len(tau), dtype=dtype))


def test_start_tangent_and_origin(fx):
    p = DraftPath(torch.as_tensor(fx["tau_h"]).double())
    e = p.eval(torch.zeros(p.B, 1, dtype=torch.float64))
    assert torch.allclose(e.xy, torch.zeros_like(e.xy))
    assert torch.allclose(e.tangent[:, 0], torch.tensor([1.0, 0.0], dtype=torch.float64).expand(p.B, 2), atol=1e-12)
    assert torch.allclose(e.normal[:, 0], torch.tensor([0.0, 1.0], dtype=torch.float64).expand(p.B, 2), atol=1e-12)


# ---------------------------------------------------------------------------------------------------- scorer schedule
def test_dense_reference_matches_sf_common(fx):
    from sf_common import Path as SFPath
    for tr in list(fx["tau_student"][:60]) + list(fx["tau_h"][:60]):
        ref = np.stack(SFPath(tr.astype(np.float64)).ref_poses_time(), -1)
        ours = dense_reference(torch.as_tensor(tr.astype(np.float64))).numpy()
        assert np.abs(ours - ref).max() < 1e-12
        # knots of the dense reference are the poses themselves
        assert np.array_equal(ours[5::5], np.concatenate([tr[:, :2], np.unwrap(np.r_[0.0, tr[:, 2]])[1:, None]], 1))


def test_s0_is_scorer_schedule(fx):
    from sf_common import Path as SFPath
    tau = torch.as_tensor(fx["tau_h"]).double()
    p = DraftPath(tau)
    s0 = p.s0_dense()
    S = p.knots()
    assert torch.equal(s0[:, ::5], S)
    for i in range(0, len(tau), 10):
        sp = SFPath(fx["tau_h"][i].astype(np.float64))
        assert np.abs(s0[i].numpy() - sp.s_orig).max() < 1e-10
    t = torch.linspace(0, 4, 81, dtype=torch.float64)
    assert torch.allclose(p.s0(t)[:, ::10], s0[:, ::5], atol=1e-12)


# ---------------------------------------------------------------------------------------------------- spline
def test_spline_c2_and_interpolating(fx):
    tau = torch.as_tensor(fx["tau_h"]).double()
    p = DraftPath(tau)
    moving = p.keep.all(1)
    assert moving.float().mean() > 0.7
    assert p.smooth_residual_max()[moving].max() < 1e-9          # interpolating where no knot was merged
    # C2 continuity at interior knots: tangent / curvature from both sides
    S = p.S[moving][:, 1:-1]
    q = DraftPath(tau[moving])
    for eps in (1e-6,):
        a, b = q.eval(S - eps), q.eval(S + eps)
        assert (a.tangent - b.tangent).abs().max() < 1e-4
        assert (a.kappa - b.kappa).abs().max() < 1e-3


def test_frenet_quantities_by_finite_differences(fx):
    tau = torch.as_tensor(fx["tau_h"][:80]).double()
    p = DraftPath(tau)
    S8 = p.S[:, -1]
    s = S8[:, None] * torch.linspace(0.05, 0.95, 37, dtype=torch.float64)[None]
    h = 1e-5
    e0, ep, em = p.eval(s), p.eval(s + h), p.eval(s - h)
    # tangent = normalised dGamma/ds, g = |dGamma/ds| (residual interpolation is rounding-level where no merge)
    mv = p.keep.all(1)
    d1 = (ep.xy - em.xy) / (2 * h)
    g_fd = d1.norm(dim=-1)
    assert torch.allclose(g_fd[mv], e0.g[mv], atol=1e-5)
    assert torch.allclose((d1 / g_fd[..., None])[mv], e0.tangent[mv], atol=1e-5)
    # kappa = d(theta)/ds / g, kappa_s = d(kappa)/ds, g_s = dg/ds
    th = torch.atan2(e0.tangent[..., 1], e0.tangent[..., 0])
    dth = wrap_angle(torch.atan2(ep.tangent[..., 1], ep.tangent[..., 0]) - torch.atan2(em.tangent[..., 1], em.tangent[..., 0])) / (2 * h)
    assert torch.allclose((dth / e0.g)[mv], e0.kappa[mv], atol=1e-4)
    assert torch.allclose(((ep.kappa - em.kappa) / (2 * h))[mv], e0.kappa_s[mv], atol=1e-3)
    assert torch.allclose(((ep.g - em.g) / (2 * h))[mv], e0.g_s[mv], atol=1e-4)
    assert torch.isfinite(th).all()


def test_circle_curvature_and_normal():
    for R, left in ((30.0, True), (12.0, False), (200.0, True)):
        tr = _circle(R, ds=3.0, left=left)
        p = DraftPath(torch.as_tensor(tr)[None])
        s = torch.linspace(1.0, float(p.S_end[0]) - 1.0, 50, dtype=torch.float64)[None]
        e = p.eval(s)
        k = (1.0 / R) * (1 if left else -1)
        # chord parameter of a C2 spline through circle points (not-a-knot end): curvature within 3 % away from
        # the clamped start (the start tangent is exact, but the start curvature is free)
        assert (e.kappa[0, 5:] - k).abs().max() < 0.03 / R
        centre = torch.tensor([0.0, R if left else -R], dtype=torch.float64)
        to_c = centre - e.xy[0]
        to_c = to_c / to_c.norm(dim=-1, keepdim=True)
        sign = 1 if left else -1
        assert (e.normal[0] * to_c).sum(-1).min() * sign > 0.999


def test_stand_still_jitter_is_merged():
    rng = np.random.default_rng(0)
    tr = np.zeros((8, 3))
    tr[:4, 0] = [2.0, 4.0, 5.5, 6.0]
    tr[4:, 0] = 6.0 + rng.normal(0, 0.01, 4)
    tr[4:, 1] = rng.normal(0, 0.01, 4)
    tr[4:, 2] = rng.normal(0, 0.002, 4)
    p = DraftPath(torch.as_tensor(tr)[None], merge_eps=0.2)
    assert not bool(p.keep[0, 5:].any())
    s = torch.linspace(0, float(p.S_end[0]), 400, dtype=torch.float64)[None]
    e = p.eval(s)
    assert e.kappa.abs().max() < 0.5                          # no jitter loops
    assert (e.normal[0, :, 1] > 0.99).all()                  # normals stay "left" of +x
    # the unmerged spline would loop through the jitter
    q = DraftPath(torch.as_tensor(tr)[None], merge_eps=0.0)
    assert q.eval(s).kappa.abs().max() > 5.0
    # knots still exact
    ek = p.eval(p.S[:, 1:])
    assert torch.equal(ek.xy[0], torch.as_tensor(tr)[:, :2])


def test_nan_tail_equals_truncated(fx):
    pl = fx["path_long"].astype(np.float64)
    nv = fx["n_valid"].astype(int)
    i = int(np.nonzero(nv < 16)[0][0])
    n = nv[i]
    a = DraftPath(torch.as_tensor(pl[i])[None])                    # NaN rows
    b = DraftPath(torch.as_tensor(pl[i, :n])[None])                # truncated
    assert int(a.n_valid[0]) == n
    assert torch.allclose(a.S_end, b.S_end, atol=1e-12)
    s = torch.linspace(0, float(b.S_end[0]), 50, dtype=torch.float64)[None]
    assert torch.allclose(a.eval(s).xy, b.eval(s).xy, atol=1e-9)
    c = DraftPath(torch.as_tensor(np.nan_to_num(pl[i]))[None], n_valid=torch.tensor([n]))
    assert torch.allclose(c.eval(s).xy, b.eval(s).xy, atol=1e-9)


def test_extend_const_curvature():
    R = 40.0
    tr = _circle(R, ds=2.0)                                        # v = 4 m/s -> kappa_lim = 0.059 > 1/40
    p = DraftPath(torch.as_tensor(tr)[None]).extend(5.0, "const_curv")
    assert abs(float(p.kappa_ext[0]) - 1.0 / R) < 1e-3
    s = torch.tensor([[p.S_end.item() + 1.0, p.S_end.item() + 5.0, p.S_end.item() + 7.0]], dtype=torch.float64)
    e = p.eval(s)
    centre = torch.tensor([0.0, R], dtype=torch.float64)
    assert ((e.xy[0] - centre).norm(dim=-1) - R).abs().max() < 0.02
    assert e.beyond.all() and e.beyond_ext.tolist() == [[False, False, True]]
    # speed clip: at ~20 m/s kappa_lim = 4.89/v^2 ~ 0.0122 < 1/40 (v = last chord / 0.5 s)
    q = DraftPath(torch.as_tensor(_circle(R, ds=10.0))[None]).extend(5.0, "const_curv")
    v_end = q.ell[:, -1] / 0.5
    assert abs(float(q.kappa_ext[0]) - float(kappa_limit(v_end)[0])) < 1e-12 and float(q.kappa_ext[0]) < 1.0 / R
    # continuity at S_end, straight policy
    r = DraftPath(torch.as_tensor(tr)[None]).extend(3.0, "straight")
    e2 = r.eval(torch.tensor([[r.S_end.item(), r.S_end.item() + 1e-9]], dtype=torch.float64))
    assert (e2.xy[0, 0] - e2.xy[0, 1]).norm() < 1e-8 and float(r.kappa_ext[0]) == 0.0


def test_kappa_limit_values():
    v = torch.tensor([0.0, 1.0, 4.4, 10.0, 30.0], dtype=torch.float64)
    k = kappa_limit(v)
    assert float(k[0]) == KAPPA_MAX and float(k[1]) == KAPPA_MAX
    assert math.isclose(float(k[3]), min(0.095, 0.0489), rel_tol=1e-12)
    assert math.isclose(float(k[4]), 4.89 / 900, rel_tol=1e-12)


# ---------------------------------------------------------------------------------------------------- autograd / batching
def test_gradcheck_eval(fx):
    tau = torch.as_tensor(fx["tau_h"][[3, 50, 120]]).double()
    S8 = DraftPath(tau).S[:, -1]
    s = (S8[:, None] * torch.tensor([[0.13, 0.41, 0.77]], dtype=torch.float64)).requires_grad_(True)

    def f_s(s_):
        e = DraftPath(tau).eval(s_)
        return e.xy, e.normal, e.kappa, e.heading

    assert torch.autograd.gradcheck(f_s, (s,), eps=1e-6, atol=1e-5)
    poses = tau.clone().requires_grad_(True)
    s_fix = s.detach()

    def f_p(P):
        e = DraftPath(P).eval(s_fix)
        return e.xy, e.kappa

    assert torch.autograd.gradcheck(f_p, (poses,), eps=1e-6, atol=1e-4)


def test_batched_vs_single(fx):
    tau = torch.as_tensor(np.concatenate([fx["tau_h"][:20], fx["tau_student"][:20]])).double()
    p = DraftPath(tau)
    s = p.S[:, -1:] * torch.linspace(0, 1.1, 30, dtype=torch.float64)[None]
    eb = p.eval(s)
    for i in range(len(tau)):
        ei = DraftPath(tau[i:i + 1]).eval(s[i:i + 1])
        for name in ("xy", "heading", "normal", "kappa", "g"):
            assert torch.allclose(getattr(eb, name)[i:i + 1], getattr(ei, name), atol=1e-12, rtol=0), name
