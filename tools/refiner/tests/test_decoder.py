"""Unit tests for navsim/agents/para_ssr/refiner/decoder.py (M4 decoder + perturbation sampler).  CPU, < 2 min.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest tools/refiner/tests/test_decoder.py -q
Data: tools/refiner/tests/fixtures/decoder_navtest_sample.npz (301 navtest tokens: human 8 poses, human path up to
8 s, v0, a0, student draft; provenance in the .json) and, when present, the full navtest student / human trajectory
pickles (12,146 tokens) for the identity test.
"""
import os
import pickle
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from navsim.agents.para_ssr.refiner import decoder as D  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import DraftPath, dense_reference  # noqa: E402

torch.set_num_threads(1)
FIX = os.path.join(os.path.dirname(__file__), "fixtures", "decoder_navtest_sample.npz")
STUDENT_PKL = os.path.join(REPO, "work_dirs", "eval", "para_ssr_interaction_final_navtest_trajectories.pkl")
HUMAN_PKL = os.path.join(REPO, "report", "navsim_version_audit", "adversarial_verify", "human_navtest_trajectories.pkl")


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIX, allow_pickle=True))


@pytest.fixture(scope="module")
def real_trajs(fx):
    out = []
    for p in (STUDENT_PKL, HUMAN_PKL):
        if os.path.exists(p):
            d = pickle.load(open(p, "rb"))["trajectories"]
            out.append(np.stack([d[k] for k in sorted(d)]).astype(np.float32))
    if not out:
        out = [fx["tau_student"], fx["tau_h"]]
    return np.concatenate(out, 0)


def _moving(fx, n=40, seed=0):
    tau = np.concatenate([fx["tau_h"], fx["tau_student"]])
    S8 = np.hypot(*np.diff(np.concatenate([np.zeros((len(tau), 1, 2)), tau[:, :, :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)
    idx = np.random.default_rng(seed).choice(np.nonzero(S8 > 5)[0], n, replace=False)
    return torch.as_tensor(tau[idx].astype(np.float64))


# ---------------------------------------------------------------------------------------------------- identity
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["A", "B"])
def test_identity_exact_real_navtest(real_trajs, dtype, mode):
    tau = torch.as_tensor(real_trajs).to(dtype)
    z = torch.zeros(len(tau), 6, dtype=dtype)
    o = D.decode(tau, z, z.clone(), None, mode)
    assert (o["traj"] - tau).abs().max().item() == 0.0
    assert torch.equal(o["traj"], tau)                         # bitwise
    assert torch.equal(o["dense"], dense_reference(tau))
    assert torch.equal(o["s"], o["s0"])
    assert (o["d"] == 0).all() and (o["flags"]["alpha"] == 1).all()


def test_identity_exact_with_long_path(fx):
    tau = torch.as_tensor(fx["tau_h"]).double()
    pl = torch.as_tensor(fx["path_long"]).double()
    pl[:, :8] = tau
    path = DraftPath(pl, n_valid=torch.as_tensor(fx["n_valid"]).long())
    z = torch.zeros(len(tau), 6, dtype=torch.float64)
    o = D.decode(tau, z, z, torch.as_tensor(fx["v0"]), "P", path=path, lat_len=path.knots()[:, -1])
    assert torch.equal(o["traj"], tau)


def test_apply_gate_returns_original_bytes(fx):
    tau = torch.as_tensor(fx["tau_student"][:10])
    o = D.decode(tau, torch.randn(10, 6), torch.randn(10, 6))
    keep = torch.tensor([True, False] * 5)
    g = D.apply_gate(tau, o["traj"], keep)
    assert torch.equal(g[~keep], tau[~keep]) and torch.equal(g[keep], o["traj"][keep])


# ---------------------------------------------------------------------------------------------------- longitudinal bounds
def test_lon_accel_and_jerk_bounds_analytic():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4000, 6, generator=g, dtype=torch.float64) * 3.0
    z[0], z[1] = -50.0, 50.0                                              # saturated
    c = D.lon_c_from_q(D.lon_q_from_z(z))
    t = torch.linspace(0, 4, 4001, dtype=torch.float64)
    pr = D.lon_profile(c, t)
    da, dv, jerk = pr["da"], pr["dv"], pr["jerk"]
    assert da.min() >= -D.A_DEC - 1e-12 and da.max() <= D.A_UP + 1e-12
    assert abs(float(da[0].min()) + D.A_DEC) < 1e-9 and abs(float(da[1].max()) - D.A_UP) < 1e-9   # bound attained
    # continuity at t0
    assert dv[:, 0].abs().max() == 0 and da[:, 0].abs().max() == 0
    # da = d(dv)/dt and jerk = d(da)/dt (finite differences), |jerk| <= max|R|
    dt = float(t[1] - t[0])
    assert torch.allclose((dv[:, 2:] - dv[:, :-2]) / (2 * dt), da[:, 1:-1], atol=1e-5)
    knots = torch.tensor([0.8, 1.6, 2.4, 3.2])
    near = ((t[1:-1, None] - knots[None]).abs() < 2 * dt).any(1)          # jerk jumps at the knots
    fd = (da[:, 2:] - da[:, :-2]) / (2 * dt)
    assert torch.allclose(fd[:, ~near], jerk[:, 1:-1][:, ~near], atol=1e-6)
    assert (jerk.abs().max(1).values <= pr["r"].abs().max(1).values + 1e-9).all()
    # the uncorrected increment rule of architecture_draft_v0 exceeds the bound (critic_impl issue 1), ours does not
    q = -D.A_DEC * torch.ones(1, 6, dtype=torch.float64)
    c_bad = torch.cat([torch.zeros(1, 2, dtype=torch.float64), torch.cumsum(0.8 * q, 1)], 1)
    assert D.lon_profile(c_bad, t)["da"].abs().max() > 11.9
    assert D.lon_profile(D.lon_c_from_q(q), t)["da"].abs().max() <= D.A_DEC + 1e-12


def test_mode_a_clamp_keeps_bounds():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(4000, 6, generator=g, dtype=torch.float64) * 2.0
    q = D.lon_q_from_z(z)
    c = D.lon_c_from_q(q)
    ca = torch.clamp(c, max=0.0)
    qa = D.lon_q_from_c(ca)[:, 1:]
    assert (qa.abs() <= q.abs() + 1e-12).all()
    assert ((qa == 0) | (torch.sign(qa) == torch.sign(q))).all()
    t = torch.linspace(0, 4, 2001, dtype=torch.float64)
    pr = D.lon_profile(ca, t)
    assert pr["dv"].max() <= 1e-12
    assert pr["da"].min() >= -D.A_DEC - 1e-12 and pr["da"].max() <= D.A_UP + 1e-12


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mode_a_never_faster(real_trajs, dtype):
    rng = np.random.default_rng(2)
    idx = rng.choice(len(real_trajs), 3000, replace=False)
    tau = torch.as_tensor(real_trajs[idx]).to(dtype)
    z = torch.as_tensor(rng.normal(0, 2, (len(tau), 6))).to(dtype)
    w = torch.as_tensor(rng.normal(0, 1, (len(tau), 6))).to(dtype)
    o = D.decode(tau, z, w, None, "A")
    assert (o["s"] <= o["s0"]).all()                            # exact, also in float32
    u = DraftPath(tau).seg_speed()
    kpt = torch.clamp(torch.arange(41) // 5, max=7)
    assert (o["v"] <= u[:, kpt]).all()
    assert (o["dv"] <= 0).all()
    assert torch.isfinite(o["traj"]).all()


def test_mode_b_limits(real_trajs):
    rng = np.random.default_rng(3)
    tau = torch.as_tensor(real_trajs[rng.choice(len(real_trajs), 2000, replace=False)]).double()
    z = torch.as_tensor(rng.normal(1.0, 2, (len(tau), 6)))
    o = D.decode(tau, z, torch.zeros_like(z), None, "B")
    S = DraftPath(tau).knots()
    u_end = (S[:, -1] - S[:, -2]) / 0.5
    L_ext = torch.clamp(u_end * D.T_EXT, max=D.L_EXT_MAX)
    assert (o["dv"] <= D.DV_ACC + 1e-9).all()
    assert (o["s"][:, -1] <= S[:, -1] + L_ext + 1e-9).all()
    assert (o["flags"]["ext_m"] <= L_ext + 1e-9).all()
    assert (o["flags"]["beta"] < 1).any() and (o["flags"]["ext_m"] > 0).any()


# ---------------------------------------------------------------------------------------------------- continuity, lateral
def test_continuity_at_t0(fx):
    tau = torch.as_tensor(np.concatenate([fx["tau_h"], fx["tau_student"]])).double()
    rng = np.random.default_rng(4)
    z = torch.as_tensor(rng.normal(0, 2, (len(tau), 6)))
    w = torch.as_tensor(rng.normal(0, 2, (len(tau), 6)))
    for mode in ("A", "B"):
        o = D.decode(tau, z, w, None, mode)
        assert (o["dv"][:, 0] == 0).all() and (o["da"][:, 0] == 0).all()
        assert (o["s"][:, 0] == 0).all() and (o["d"][:, 0] == 0).all()
    # d'(0) = 0 analytically
    e = D.lat_e_from_w(w)
    lp = D.lat_profile(e, torch.zeros(len(tau), 1, dtype=torch.float64))
    assert (lp["d"] == 0).all() and (lp["d_x"] == 0).all()
    # near t0 the offset is second order in s: |d(s)| <= C s^2
    x = torch.full((len(tau), 1), 1e-3, dtype=torch.float64)
    assert D.lat_profile(e, x)["d"].abs().max() < 1e-4


def test_lateral_bounds_and_short_paths(fx):
    tau = torch.as_tensor(np.concatenate([fx["tau_h"], fx["tau_student"]])).double()
    rng = np.random.default_rng(5)
    w = torch.as_tensor(rng.normal(0, 3, (len(tau), 6)))
    o = D.decode(tau, torch.zeros_like(w), w, None, "A", n_proj=0)
    assert o["d"].abs().max() <= D.D_MAX + 1e-12
    S8 = DraftPath(tau).knots()[:, -1]
    short = S8 < D.S_LAT_MIN
    assert short.any() and (o["d"][short] == 0).all() and (o["traj"][short] == tau[short]).all()


def test_offset_geometry_straight_and_circle():
    # straight draft along x: d(s) is literally the y coordinate, heading = atan(d')
    t = np.arange(1, 9) * 0.5
    tau = torch.as_tensor(np.stack([10 * t, 0 * t, 0 * t], -1))[None]
    w = torch.tensor([[0.3, 0.5, 0.7, 0.9, 1.0, 1.0]], dtype=torch.float64)
    o = D.decode(tau, torch.zeros(1, 6, dtype=torch.float64), w, None, "A")
    e = o["e_lat"]
    lp = D.lat_profile(e, o["s"][:, 5::5] / 40.0)
    assert torch.allclose(o["traj"][0, :, 1], lp["d"][0], atol=1e-12)
    assert torch.allclose(o["traj"][0, :, 0], tau[0, :, 0], atol=1e-12)
    assert torch.allclose(o["traj"][0, :7, 2], torch.atan(lp["d_x"][0, :7] / 40.0), atol=1e-12)
    # circle of radius R, left turn: a constant offset d at the end sits on radius R - d
    R = 60.0
    s = np.arange(1, 9) * 4.0
    th = s / R
    circ = torch.as_tensor(np.stack([R * np.sin(th), R - R * np.cos(th), th], -1))[None]
    w = torch.full((1, 6), float(np.arctanh(0.5 / D.D_MAX)), dtype=torch.float64)   # e_2..e_7 = 0.5 m
    o = D.decode(circ, torch.zeros(1, 6, dtype=torch.float64), w, None, "A")
    r = (o["traj"][0, -1, :2] - torch.tensor([0.0, R], dtype=torch.float64)).norm()
    assert abs(float(r) - (R - 0.5)) < 0.01
    assert abs(float(o["traj"][0, -1, 2]) - th[-1]) < 1e-3       # d' = 0 at the end -> tangent heading


def test_offset_curvature_formula_by_finite_differences(fx):
    tau = _moving(fx, 20, seed=6)
    p = DraftPath(tau)
    S8 = p.knots()[:, -1:]
    s = S8 * torch.linspace(0.1, 0.9, 30, dtype=torch.float64)[None]
    e = D.lat_e_from_w(torch.as_tensor(np.random.default_rng(6).normal(0, 0.5, (len(tau), 6))))

    def curve(ss):
        pe = p.eval(ss)
        lp = D.lat_profile(e, ss / S8)
        return pe.xy + lp["d"][..., None] * pe.normal, pe, lp

    h = 1e-4
    c0, pe, lp = curve(s)
    cp, cm = curve(s + h)[0], curve(s - h)[0]
    d1 = (cp - cm) / (2 * h)
    d2 = (cp - 2 * c0 + cm) / (h * h)
    k_fd = (d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]) / d1.norm(dim=-1) ** 3
    k_an = D.offset_curvature(pe, lp["d"], lp["d_x"] / S8, lp["d_xx"] / S8 ** 2)
    ok = p.keep.all(1)
    assert (k_fd - k_an)[ok].abs().max() < 2e-3


def test_curvature_projection(fx):
    tau = torch.as_tensor(np.concatenate([fx["tau_h"], fx["tau_student"]])).double()
    rng = np.random.default_rng(7)
    w = torch.as_tensor(rng.normal(0, 3, (len(tau), 6)))
    o = D.decode(tau, torch.zeros_like(w), w, torch.as_tensor(np.r_[fx["v0"], fx["v0"]]), "A")
    fl = o["flags"]
    on = fl["lat_on"]
    assert (fl["alpha"] <= 1).all() and (fl["alpha"] >= 0).all()
    assert (fl["alpha"][on] < 1).float().mean() > 0.3             # large random offsets do get projected
    # residual after n_proj = 6 first-order passes: bound met to within 2 % for all rows (p99 1.00 measured)
    assert (fl["kappa_ratio"][on] <= 1.02).float().mean() > 0.99 and fl["kappa_ratio"][on].max() < 1.1
    o0 = D.decode(tau, torch.zeros_like(w), w, None, "A", n_proj=0)
    assert (o0["flags"]["kappa_ratio"][on] > 1.1).float().mean() > 0.3
    # small smooth offsets (10 cm ramp over >= 10 m) are untouched
    S8 = DraftPath(tau).knots()[:, -1]
    w_s = torch.full_like(w, float(np.arctanh(0.1 / D.D_MAX)))
    o_s = D.decode(tau, torch.zeros_like(w), w_s, None, "A")
    assert (o_s["flags"]["alpha"][S8 >= 10] == 1).float().mean() > 0.97


def test_gradient_alive_at_zero(fx):
    """Refiner heads initialised at 0 must still receive gradients (mode-A clamp inclusive, alpha = 1 at w = 0)."""
    for dtype in (torch.float32, torch.float64):
        tau = torch.as_tensor(np.concatenate([fx["tau_h"], fx["tau_student"]])).to(dtype)
        z = torch.zeros(len(tau), 6, dtype=dtype, requires_grad=True)
        w = torch.zeros(len(tau), 6, dtype=dtype, requires_grad=True)
        o = D.decode(tau, z, w, None, "A")
        (o["traj"][..., :2] * torch.randn_like(o["traj"][..., :2])).sum().backward()
        S = DraftPath(tau).knots()
        moving = (S[:, 1:] - S[:, :-1]).min(1).values > 0.5
        lat_on = S[:, -1] >= D.S_LAT_MIN
        assert (z.grad[moving].abs().sum(1) > 0).all()
        assert (w.grad[lat_on].abs().sum(1) > 0).all()
        assert (o["flags"]["alpha"] == 1).all()


def test_gradients_finite_and_bounded_on_real_drafts(real_trajs):
    """All outputs and gradients finite on real drafts (stand-still, creeping, reversing ones included) and no
    outliers from spline cusps (path curvature / metric clips)."""
    tau = torch.as_tensor(real_trajs)
    g = torch.Generator().manual_seed(14)
    for mode in ("A", "B"):
        for scale in (0.0, 2.0):
            z = (scale * torch.randn(len(tau), 6, generator=g)).requires_grad_(True)
            w = (scale * torch.randn(len(tau), 6, generator=g)).requires_grad_(True)
            o = D.decode(tau, z, w, torch.full((len(tau),), 5.0), mode)
            keys = ("traj", "dense", "s", "v", "d", "kappa", "da", "jerk")
            assert all(torch.isfinite(o[k]).all() for k in keys)
            loss = sum((o[k] * torch.randn(o[k].shape, generator=g)).sum() for k in keys)
            loss.backward()
            assert torch.isfinite(z.grad).all() and torch.isfinite(w.grad).all()
            assert z.grad.abs().max() < 1e4 and w.grad.abs().max() < 1e4


# ---------------------------------------------------------------------------------------------------- autograd / batching
@pytest.mark.parametrize("mode", ["A", "B", "P"])
def test_gradcheck_decode(fx, mode):
    tau = _moving(fx, 3, seed=8)
    g = torch.Generator().manual_seed(9)
    z = (0.3 * torch.randn(3, 6, generator=g, dtype=torch.float64) - (0.3 if mode == "A" else -0.2)).requires_grad_(True)
    w = (0.2 * torch.randn(3, 6, generator=g, dtype=torch.float64)).requires_grad_(True)
    path = None
    if mode == "P":
        path = DraftPath(tau).extend(15.0, "const_curv")

    def f(z_, w_):
        o = D.decode(tau, z_, w_, None, mode, path=path)
        return o["traj"], o["s"], o["d"], o["kappa"]

    assert torch.autograd.gradcheck(f, (z, w), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_batched_vs_single(fx):
    tau = torch.as_tensor(np.concatenate([fx["tau_h"][:25], fx["tau_student"][:25]])).double()
    rng = np.random.default_rng(10)
    z = torch.as_tensor(rng.normal(0, 1.5, (len(tau), 6)))
    w = torch.as_tensor(rng.normal(0, 1.0, (len(tau), 6)))
    v0 = torch.as_tensor(np.r_[fx["v0"][:25], fx["v0"][:25]])
    for mode in ("A", "B"):
        ob = D.decode(tau, z, w, v0, mode)
        for i in range(len(tau)):
            oi = D.decode(tau[i], z[i], w[i], v0[i], mode)
            for k in ("traj", "dense", "s", "v", "d", "kappa"):
                assert torch.allclose(ob[k][i], oi[k], atol=1e-10, rtol=0), (mode, k)
    # float32 batch vs float64 batch
    o32 = D.decode(tau.float(), z.float(), w.float(), v0.float(), "A")
    o64 = D.decode(tau, z, w, v0, "A")
    assert (o32["traj"].double() - o64["traj"]).abs().max() < 1e-3


def test_dense_is_scorer_reference_of_traj(fx):
    tau = _moving(fx, 30, seed=11)
    rng = np.random.default_rng(11)
    o = D.decode(tau, torch.as_tensor(rng.normal(0, 1, (30, 6))), torch.as_tensor(rng.normal(0, 1, (30, 6))))
    assert torch.equal(o["dense"], dense_reference(o["traj"]))
    assert torch.equal(o["dense"][:, 5::5], torch.cat([o["traj"][..., :2], o["dense"][:, 5::5, 2:]], -1))


# ---------------------------------------------------------------------------------------------------- perturbations
@pytest.fixture(scope="module")
def banks(fx):
    out = []
    for i in range(0, len(fx["token"]), 3):
        b = D.sample_bank(fx["tau_h"][i], str(fx["token"][i]), path_long=fx["path_long"][i],
                          n_valid=int(fx["n_valid"][i]), v0=float(fx["v0"][i]), a0=float(fx["a0"][i]))
        out.append((i, b))
    return out


def test_bank_structure_and_constraints(fx, banks):
    for i, b in banks:
        assert b["drafts"].shape == (13, 8, 3) and b["drafts"].dtype == np.float32
        assert b["family"].dtype == np.int8 and b["params"].shape == (13, 6)
        assert np.array_equal(b["drafts"][0], fx["tau_h"][i])          # identity = human bytes
        assert b["family"][0] == D.FAMILY["identity"]
        ctx = D.HumanContext(fx["tau_h"][i], fx["path_long"][i], int(fx["n_valid"][i]), float(fx["v0"][i]))
        for k in range(13):
            f = D.FAMILY_NAME[int(b["family"][k])]
            if not b["valid"][k]:
                continue
            assert D.check_draft(ctx, b["drafts"][k]) == "" or f in ("identity", "cv", "hdrift")
            A_p, t_on, D_p, s_on, aux, ds4 = b["params"][k]
            if f in ("lconst", "combined"):
                assert 0.2 - 1e-6 <= A_p <= 1.3 + 1e-6 and t_on in D.T_ON_SET and ds4 > 0
                assert ds4 <= ctx.S_avail - ctx.S8 + 1e-4                 # follows the human path, no extrapolation
                assert b["path_src"][k] == "human_long"
            if f == "small":
                assert 0 <= A_p <= 0.2 + 1e-6 and abs(D_p) <= 0.3 + 1e-6
            if f in ("lat", "combined"):                                    # realised end offset (LS projection)
                assert 0.3 - 1e-6 <= abs(D_p) <= 1.5 * 1.01
            if f == "cv":
                assert np.allclose(b["drafts"][k][:, 1:], 0) and np.allclose(np.diff(b["drafts"][k][:, 0]), 0.5 * fx["v0"][i], atol=1e-5)
            if f == "hdrift":
                assert np.array_equal(b["drafts"][k][:, :2], fx["tau_h"][i][:, :2])
            if f == "ignore_brake":
                assert ctx.decel >= 1.0 and 0.3 - 1e-6 <= aux <= 1.0
            if f == "creep":
                assert ctx.S8 < 2.0 and ds4 > 0
            # t0 continuity window (human-relative)
            fdv = np.hypot(*b["drafts"][k][0, :2]) / 0.5 - fx["v0"][i]
            hdv = np.hypot(*fx["tau_h"][i][0, :2]) / 0.5 - fx["v0"][i]
            if f not in ("cv",):
                assert min(D.CONT_LO, hdv) - 1e-4 <= fdv <= max(D.CONT_HI, hdv) + 1e-4
    valid = np.stack([b["valid"] for _, b in banks])
    assert valid.mean() > 0.8


def test_bank_deterministic(fx):
    i = 5
    kw = dict(path_long=fx["path_long"][i], n_valid=int(fx["n_valid"][i]), v0=float(fx["v0"][i]))
    a = D.sample_bank(fx["tau_h"][i], str(fx["token"][i]), **kw)
    b = D.sample_bank(fx["tau_h"][i], str(fx["token"][i]), **kw)
    c = D.sample_bank(fx["tau_h"][i], "another_token", **kw)
    assert np.array_equal(a["drafts"], b["drafts"]) and np.array_equal(a["params"], b["params"])
    assert not np.array_equal(a["drafts"], c["drafts"])


def test_perturbation_reproducible_in_basis(fx):
    rng = np.random.default_rng(12)
    for i in range(0, 60, 4):
        ctx = D.HumanContext(fx["tau_h"][i], fx["path_long"][i], int(fx["n_valid"][i]), float(fx["v0"][i]))
        for fam in ("small", "lconst", "lat", "combined"):
            r = D.sample_perturbation(fam, ctx, rng)
            if not r["valid"]:
                continue
            o = D.decode(torch.as_tensor(ctx.tau_h.astype(np.float64))[None],
                         torch.as_tensor(r["z_lon"].astype(np.float64))[None],
                         torch.as_tensor(r["w_lat"].astype(np.float64))[None], ctx.v0, "P", path=r["path"],
                         lat_len=ctx.S8)
            assert np.abs(o["traj"][0].numpy() - r["draft"]).max() < 1e-4       # z/w stored as float32


def test_round_trip_reach_back(fx):
    """perturb (sample_perturbation) -> mode-A decoder on the perturbed draft fitted back to the human: within 0.1 m
    at every knot where the DOF allows (initialised at the analytic inverse -q_eff, -e_eff, then LM)."""
    rng = np.random.default_rng(13)
    rows = []
    for i in range(0, len(fx["token"]), 4):
        ctx = D.HumanContext(fx["tau_h"][i], fx["path_long"][i], int(fx["n_valid"][i]), float(fx["v0"][i]))
        fams = ["small", "lconst", "lat", "combined"] + (["ignore_brake"] if ctx.decel_ok else []) + \
               (["creep"] if ctx.creep_ok else [])
        for fam in fams:
            r = D.sample_perturbation(fam, ctx, rng)
            if r["valid"]:
                rows.append((fam, r, fx["tau_h"][i]))
    tau = torch.as_tensor(np.stack([r["draft"] for _, r, _ in rows]).astype(np.float64))
    tgt = torch.as_tensor(np.stack([h for _, _, h in rows]).astype(np.float64))
    q0 = torch.as_tensor(np.stack([-r["q_eff"] for _, r, _ in rows]))
    e0 = torch.as_tensor(np.stack([-r["e_eff"] for _, r, _ in rows]))
    z0 = D.lon_z_from_q(q0.clamp(-0.999 * D.A_DEC, 0.999 * D.A_UP))
    w0 = torch.atanh((e0 / D.D_MAX).clamp(-0.999, 0.999))
    # best of two starts (analytic inverse; zero = the draft itself): "can the decoder reach back"
    r1 = D.fit_controls(tau, tgt[..., :2], None, "A", iters=15, target_h=tgt[..., 2], z_init=z0, w_init=w0)
    r2 = D.fit_controls(tau, tgt[..., :2], None, "A", iters=15, target_h=tgt[..., 2])
    err = np.minimum(r1["err"].numpy(), r2["err"].numpy())
    herr = np.where(r1["err"].numpy() <= r2["err"].numpy(), r1["herr"].numpy(), r2["herr"].numpy())
    fam = np.array([f for f, _, _ in rows])
    thr = {"small": 0.97, "lconst": 0.95, "ignore_brake": 0.9, "creep": 0.85, "combined": 0.95, "lat": 0.85}
    for f in set(fam):
        assert np.mean(err[fam == f] < 0.1) >= thr[f], (f, np.mean(err[fam == f] < 0.1))
    assert np.mean(err < 0.1) >= 0.95
    assert np.percentile(np.degrees(herr), 90) < 2.0
    # lateral-only: the mode-A shortfall is a DOF limit.  Undoing the offset along the offset path's rotated normal
    # moves the knot along-track by ~ d sin(atan d'); when that needs a (small) speed-up mode A cannot follow.
    # With mode B (limited acceleration) the same drafts reach back (measured 99.2 % within 0.1 m, p99 0.036 m).
    m = fam == "lat"
    assert err[m].max() < 0.25
    rB = D.fit_controls(tau[m], tgt[m][..., :2], None, "B", iters=15, target_h=tgt[m][..., 2], z_init=z0[m],
                        w_init=w0[m])
    assert np.mean(rB["err"].numpy() < 0.1) >= 0.95
