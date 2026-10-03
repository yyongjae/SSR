"""M4 correction decoder (no learned parameters) + draft-bank perturbation sampler (IMPL_SPEC §3.2, §3.6).

decode(tau0 [B,8,3], z_lon [B,6], w_lat [B,6], v0 [B] | None, mode='A') -> dict (see decode()).

Conventions (N frame, see geometry.py): poses (x, y, heading) at t = 0.5..4.0 s, implicit origin pose (0,0,0) at t = 0;
dense grid n = 0..40, t = 0.1 n; s = chord arc length along the draft path (DraftPath); metres, seconds, radians.

Longitudinal (speed offset on the draft's own path)
  dv(t) = sum_{i=0..7} c_i B_{i,3}(t), clamped cubic B-spline, knots LON_KNOTS = [0,0,0,0,.8,1.6,2.4,3.2,4,4,4,4].
  Derivative control points Q_i = 3 (c_{i+1} - c_i) / (u_{i+4} - u_{i+1}) define da(t) = dv'(t) = sum Q_i B_{i,2}(t).
  Q_0 = 0 and c_0 = c_1 = 0; the free Q_1..Q_6 = A tanh(z) with A = A_DEC = 4.0 (z < 0) / A_UP = 2.0 (z >= 0), and
  c_{i+1} = c_i + Q_i (u_{i+4} - u_{i+1}) / 3.  Convex hull => -A_DEC <= da(t) <= A_UP EXACTLY; dv(0) = da(0) = 0.
  jerk(t) = sum R_i B_{i,1}(t), R_i = 2 (Q_{i+1} - Q_i) / (w_{i+3} - w_{i+1}) (w = LON_KNOTS[1:-1]), |jerk| <= max|R|.
  Mode A (default, deceleration only): c <- min(c, 0)  => dv <= 0 everywhere => v1 <= v0, s1 <= s0 pointwise.
      (Clamping c elementwise to <= 0 never increases |Q_i| = 3 |c_{i+1} - c_i| / span nor flips its sign: moving a
      positive c toward 0 only shrinks its gap to any other c; so the |da| bounds survive.  Checked numerically in
      tests/test_decoder.py::test_mode_a_clamp_keeps_bounds.)
      Optional straight-through backward (lon_st_slope = lam > 0, training only; PRESTATED_DECISION_RULE AMENDMENT 3):
      the forward is still torch.clamp(c, max=0.0) (bit-identical), the backward is d out / d c = 1 for c <= 0 and lam
      for c > 0 (lon_clamp_a).  lam = 0 (default) is the plain torch.clamp (run 1).  Liveness of a draft = any
      pre-clamp c_i < 0 (<=> any decoded c_lon < 0): lon_live(z).
  Mode B (limited acceleration): c <- min(c, DV_ACC = 2 m/s); positive control points scaled by beta in [0, 1] so that
      s1(4) <= S_8 + L_ext, L_ext = min(5 m, v_end * 1 s), path beyond S_8 = constant-curvature extension (flagged).
  Mode P (perturbation generator only): no upper clamp; beta so that s1(4) <= path.S_end + path.ext_len (the human
      path up to 8 s, no extrapolation unless the caller extended the path).
  beta: s1(4) is convex in beta, so beta = (S_max - s1(4; 0)) / (s1(4; 1) - s1(4; 0)) guarantees s1(4) <= S_max.
  v1(t) = relu(v0(t) + dv(t)) (NO softplus), v0(t) = piecewise-constant draft speed (S_{k+1} - S_k) / 0.5.
  s1 = s0 + cumulative trapezoid (0.1 s) of (v1 - v0), using the speed of the containing 0.5 s interval at both
  ends of each 0.1 s step.  s1[5k] - S_k is therefore EXACTLY 0 when dv == 0 (identity).

Lateral (Frenet offset along the draft path)
  d(s) = sum_{i=0..7} e_i B_{i,3}(s / S_L), knots LAT_KNOTS = [0,0,0,0,.2,.4,.6,.8,1,1,1,1], e_0 = e_1 = 0 (d(0) = d'(0) = 0),
  e_i = D_MAX tanh(w_i), D_MAX = 2 m (|d| <= D_MAX), S_L = S_8 (mode A/P) or S_8 + L_ext (mode B); d is held at e_7
  for s > S_L.  d == 0 if S_8 < S_LAT_MIN = 3 m.
  Offset-curve curvature (exact, for the general parameter s with metric g = |Gamma'|, see geometry.py):
      kappa_new = [g^3 k q^2 + g q d'' - g' d' q + g d' k' d + 2 g k d'^2] / (g^2 q^2 + d'^2)^{3/2},  q = 1 - k d.
  Curvature projection (training AND inference, differentiable, n_proj = 6 fixed-point passes): e <- alpha e with
      alpha = min(1, min_n (K_n - sgn(dk_n) k_n) / |dk_n|),  dk = kappa_new - k (curvature ADDED by the offset),
      K_n = max(kappa_lim(v1_n), 1.1 |k_n|), kappa_lim(v) = min(0.95/v, 4.89/v^2, 0.213), n over the 41 dense points.
      (The 10 % slack on the draft's own curvature keeps alpha = 1 and gradients alive at w = 0 where the draft sits
      at its own bound; without it float32 rounding of dk alone gives alpha = 0 there.)
  Heading H1: h = h0(s1) + atan2(d', g (1 - k d)), h0 = draft heading interpolated in arc length (unwrapped);
      (1 - k d) is floored at KD_MIN = 0.05 (flagged).  The path curvature k used here, in kappa_new and in the
      projection is clipped to |k| <= KAPPA_PATH_MAX = 1 /m (flag kappa_path_clipped): C2-spline cusps through
      creeping stand-still knots reach |k| ~ 1e3 /m and gave |d kappa_new / dz| up to 1e6 on real drafts; likewise
      g = |Gamma'| is floored at G_MIN = 0.2 (cusp of drafts that reverse at the start: 4 of 48k real drafts have
      g < 0.1).  After both clips max |dL/dz|, |dL/dw| over 48k real drafts are O(1e2-1e3) (validate_decoder.py).
  Positions / headings at the knots are ANCHORED to the draft pose:
      xy_k = tau0_k + (Gamma(s1_k) - Gamma(S_k)) + d_k N(s1_k),   h_k = h_{0,k} + (h0(s1_k) - h0(S_k)) + dh_k,
  which equals Gamma(s1_k) + d_k N_k (Gamma(S_k) == tau0_k exactly) and makes z = w = 0 return tau0 BITWISE (also at
  stand-still vertices whose arc length coincides with the next one).

Dense output: [origin; traj] linearly interpolated in time at 0.1 s (heading unwrapped), i.e. the scorer reference.

Deviations from IMPL_SPEC §3.2 (documented; interfaces unchanged, extra keyword arguments only)
  * Curvature projection: the spec formula alpha = min(1, kappa_lim / max|kappa_new|) is used in its per-point,
    draft-aware form above.  For a straight draft (k = 0) the two are identical.  The literal form shrinks the lateral
    correction to ~0 whenever the DRAFT itself exceeds kappa_lim somewhere (e.g. any curve driven faster than
    sqrt(4.89/k), or the stand-still end of a path), which is not what the projection is for; the draft-aware form
    only limits the curvature the offset adds, and never lets it grow beyond max(kappa_lim, 1.1 |k|).  It is a
    first-order scaling (kappa_new is not linear in alpha), iterated n_proj times (alpha_{i+1} = alpha_i * a(alpha_i));
    on random large offsets the residual max|kappa_new| / K has p99 3.2 / 1.25 / 1.03 / 1.00 after 1 / 2 / 4 / 6
    passes; it is returned in flags['kappa_ratio'].
  * v0 [B] (ego speed at t0) is optional: it is used for the t0-continuity flag (first-segment speed - v0 in
    [-0.8, +0.6]) and as the speed at n = 0 in kappa_lim.  The draft's own speed profile is v0(t) above.
  * Extra keyword arguments: path (a DraftPath over a LONGER pose sequence whose first 8 poses are tau0; used by the
    perturbation generator so that speed-ups follow the human path up to 8 s), lat_len (override S_L), n_proj,
    lon_st_slope (mode-A straight-through slope, see Longitudinal; 0 = off).
  * Mode 'P' (perturbation generator) in addition to 'A' / 'B'.

Known limitations (measured, see tools/refiner/validate_decoder.py -> report/refiner_T/decoder_validation.json)
  * Mode-A dead zone: control points pushed above 0 are clamped, so z directions that only raise c above 0 get zero
    gradient.  z = 0 itself is live (torch.clamp passes the gradient at the boundary; test_gradient_alive_at_zero).
    Stage-T run 1 hit it: the trained nets put z_lon > 0 on 100 % of drafts (no deceleration at all, zero gradient;
    report/refiner_T/DECISION_RUN1.md).  Option: decode(..., lon_st_slope=lam > 0) keeps the forward and passes a
    straight-through gradient of slope lam through the clamped part (tests/test_dead_zone.py).
  * Mode A cannot exactly undo a steep lateral step: returning along the offset path's (rotated) normal moves the knot
    along-track by ~ d sin(atan d'), which sometimes needs a small speed-up (mode B reaches back; mode A misses by
    <= ~0.17 m).  Other families reach back within 0.1 m (round-trip numbers in the JSON).
  * fit_controls (inverse fit) from z = w = 0 can stop in a local minimum (mode-A clamp / asymmetric tanh); start it
    from the analytic inverse (-q_eff, -e_eff of the perturbation) when the perturbation is known.

sample_perturbation / sample_bank: see their docstrings (families of IMPL_SPEC §3.6, generated in the SAME basis by
least-squares projection of the family's profile onto the constrained control points, A_p etc. recorded as derived
statistics).
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .geometry import (DT, NK, N_POSE, SUB, T_POSE, T_HORIZON, KAPPA_MAX, LON_ACC_MAX, LON_ACC_MIN,
                       DraftPath, PathEval, as_tensor, dense_reference, kappa_limit, safe_norm,
                       wrap_angle)

# ----------------------------------------------------------------------------------------------- constants
A_DEC = 4.0              # |da| bound on the deceleration side [m/s^2]
A_UP = 2.0               # da bound on the acceleration side [m/s^2]
D_MAX = 2.0              # |d| bound [m]
S_LAT_MIN = 3.0          # lateral offsets disabled below this draft length [m]
DV_ACC = 2.0             # mode B: dv <= DV_ACC [m/s]
L_EXT_MAX = 5.0          # mode B: extension <= min(L_EXT_MAX, v_end * T_EXT)
T_EXT = 1.0
KD_MIN = 0.05            # floor of (1 - kappa d) in the H1 heading term
KAPPA_SLACK = 0.1        # projection bound K = max(kappa_lim, (1 + KAPPA_SLACK) |kappa_draft|)
KAPPA_PATH_MAX = 1.0     # [1/m] clip of the draft-path curvature used inside the decoder (spline cusps at stand-still)
G_MIN = 0.2              # floor of the path metric g = |Gamma'| inside the decoder (cusps of reversing drafts)
CONT_LO, CONT_HI = -0.8, 0.6   # t0 continuity window: first-segment speed - v0 [m/s]

LON_KNOTS = (0.0, 0.0, 0.0, 0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.0, 4.0, 4.0)
LAT_KNOTS = (0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0)
N_CTRL = 8
N_FREE = 6
# c_{i+1} = c_i + Q_i * LON_CSPAN[i] / 3, i = 0..6 (Q_0 = 0)
LON_CSPAN = tuple(LON_KNOTS[i + 4] - LON_KNOTS[i + 1] for i in range(7))   # (.8, 1.6, 2.4, 2.4, 2.4, 1.6, .8)
_KNOT_IDX = tuple(SUB * k for k in range(1, N_POSE + 1))                   # dense index of pose k (5, 10, ..., 40)


# ----------------------------------------------------------------------------------------------- B-splines
def bspline_basis(x: torch.Tensor, knots: Sequence[float], degree: int) -> List[torch.Tensor]:
    """Cox-de Boor.  Returns [N_0, ..., N_degree], N_d of shape [..., len(knots) - d - 1].

    x must lie in [knots[0], knots[-1]] (clamp before); the right end is closed (x = knots[-1] belongs to the last
    non-degenerate interval).  Differentiable in x.
    """
    t = list(map(float, knots))
    nk = len(t)
    xs = x[..., None]
    tt = torch.tensor(t, dtype=x.dtype, device=x.device)
    N = ((xs >= tt[:-1]) & (xs < tt[1:])).to(x.dtype)
    last = max(i for i in range(nk - 1) if t[i] < t[i + 1])
    at_end = (x >= t[-1])
    col = torch.where(at_end, torch.ones_like(x), N[..., last])
    N = torch.cat([N[..., :last], col[..., None], N[..., last + 1:]], -1)
    out = [N]
    for d in range(1, degree + 1):
        cols = []
        for i in range(nk - d - 1):
            term = torch.zeros_like(x)
            den1 = t[i + d] - t[i]
            if den1 > 0:
                term = term + (x - t[i]) / den1 * N[..., i]
            den2 = t[i + d + 1] - t[i + 1]
            if den2 > 0:
                term = term + (t[i + d + 1] - x) / den2 * N[..., i + 1]
            cols.append(term)
        N = torch.stack(cols, -1)
        out.append(N)
    return out


def deriv_ctrl(ctrl: torch.Tensor, knots: Sequence[float], degree: int) -> torch.Tensor:
    """Control points of the derivative spline (degree - 1, knots[1:-1]): p (C_{i+1} - C_i) / (t_{i+p+1} - t_{i+1})."""
    t = knots
    span = torch.tensor([t[i + degree + 1] - t[i + 1] for i in range(ctrl.shape[-1] - 1)], dtype=ctrl.dtype,
                        device=ctrl.device)
    return degree * (ctrl[..., 1:] - ctrl[..., :-1]) / span


_LON_CACHE: Dict = {}


def _lon_basis(dtype, device):
    """Dense-grid (t = n / 10) longitudinal basis matrices (B3 [41,8], B2 [41,7], B1 [41,6])."""
    key = (dtype, str(device))
    if key not in _LON_CACHE:
        t = torch.arange(NK, dtype=torch.float64) / 10.0
        N = bspline_basis(t, LON_KNOTS, 3)
        N2 = bspline_basis(t, LON_KNOTS[1:-1], 2)[2]
        N1 = bspline_basis(t, LON_KNOTS[2:-2], 1)[1]
        _LON_CACHE[key] = tuple(m.to(dtype=dtype, device=device) for m in (N[3], N2, N1))
    return _LON_CACHE[key]


def lon_q_from_z(z: torch.Tensor) -> torch.Tensor:
    """Free derivative control points Q_1..Q_6 = A tanh(z), A = A_DEC (z < 0) / A_UP (z >= 0)."""
    tz = torch.tanh(z)
    return torch.where(z < 0, A_DEC * tz, A_UP * tz)


def lon_z_from_q(q) -> torch.Tensor:
    """Inverse of lon_q_from_z (|q| must be < A of its side)."""
    q = as_tensor(q)
    return torch.where(q < 0, torch.atanh(q / A_DEC), torch.atanh(q / A_UP))


def lon_c_from_q(q: torch.Tensor) -> torch.Tensor:
    """[.., 6] free Q -> [.., 8] control points c (c_0 = c_1 = 0)."""
    span = torch.tensor(LON_CSPAN[1:], dtype=q.dtype, device=q.device)
    inc = torch.cumsum(q * span / 3.0, -1)
    z = torch.zeros(q.shape[:-1] + (2,), dtype=q.dtype, device=q.device)
    return torch.cat([z, inc], -1)


class _ClampNegST(torch.autograd.Function):
    """out = torch.clamp(c, max=0.0) (the SAME op, so bit-identical, -0.0 included); backward: d out / d c = 1 where
    c <= 0 (as torch.clamp: the boundary passes), = slope where c > 0, 0 where c is NaN (as torch.clamp)."""

    @staticmethod
    def forward(ctx, c, slope):
        ctx.save_for_backward(c)
        ctx.slope = slope
        return torch.clamp(c, max=0.0)

    @staticmethod
    def backward(ctx, g):
        (c,) = ctx.saved_tensors
        gc = torch.where(c > 0, g * ctx.slope, torch.where(c <= 0, g, torch.zeros_like(g)))
        return gc, None


def lon_clamp_a(c: torch.Tensor, st_slope: float = 0.0) -> torch.Tensor:
    """Mode-A clamp c <- min(c, 0).  st_slope = 0: plain torch.clamp (run 1, dead zone for c > 0); st_slope = lam > 0:
    same forward, straight-through backward of slope lam on c > 0 (_ClampNegST).

    A custom Function rather than c - relu(c) + lam (relu(c) - relu(c).detach()): that form returns +0.0 for c = -0.0
    (torch.clamp keeps -0.0), i.e. it is not bitwise the clamp."""
    st_slope = float(st_slope)
    if not (math.isfinite(st_slope) and st_slope >= 0.0):
        raise ValueError(f"lon_st_slope must be finite and >= 0, got {st_slope}")
    if st_slope == 0.0:
        return torch.clamp(c, max=0.0)
    return _ClampNegST.apply(c, st_slope)


def lon_live(z: torch.Tensor) -> torch.Tensor:
    """Mode-A liveness of raw longitudinal controls z [..., 6] -> bool [...]: any c_i < 0 of c = lon_c_from_q(
    lon_q_from_z(z)) (the pre-clamp control points decode() computes; c < 0 before the clamp <=> after it).  A draft
    that is not live decodes with dv == 0 (no deceleration).  Evaluate in the dtype decode() ran in (float32 in
    training / eval_refiner) for exact agreement with (decode(...)['c_lon'] < 0).any(-1)."""
    z = as_tensor(z)
    if not z.is_floating_point():
        z = z.float()
    return (lon_c_from_q(lon_q_from_z(z)) < 0).any(-1)


def lon_q_from_c(c: torch.Tensor) -> torch.Tensor:
    """[.., 8] c -> [.., 7] derivative control points Q_0..Q_6 (Q_0 = 0 when c_0 = c_1)."""
    return deriv_ctrl(c, LON_KNOTS, 3)


def lon_profile(c: torch.Tensor, t) -> Dict[str, torch.Tensor]:
    """Analytic dv, da, jerk of control points c [B, 8] at times t [M] (or [B, M]) in [0, 4] s."""
    t = as_tensor(t, dtype=c.dtype, device=c.device).clamp(0.0, T_HORIZON)
    q = lon_q_from_c(c)
    r = deriv_ctrl(q, LON_KNOTS[1:-1], 2)
    N3 = bspline_basis(t, LON_KNOTS, 3)[3]
    N2 = bspline_basis(t, LON_KNOTS[1:-1], 2)[2]
    N1 = bspline_basis(t, LON_KNOTS[2:-2], 1)[1]
    if t.dim() == 1:
        f = lambda Nm, ctrl: ctrl @ Nm.T
    else:
        f = lambda Nm, ctrl: (Nm * ctrl[:, None, :]).sum(-1)
    return {"dv": f(N3, c), "da": f(N2, q), "jerk": f(N1, r), "q": q, "r": r}


def lat_profile(e: torch.Tensor, x) -> Dict[str, torch.Tensor]:
    """d, dd/dx, d2d/dx2 of lateral control points e [B, 8] at normalised x = s / S_L [B, M] (clamped to [0,1])."""
    x = as_tensor(x, dtype=e.dtype, device=e.device)
    xc = x.clamp(0.0, 1.0)
    Ns = bspline_basis(xc, LAT_KNOTS, 3)
    qe = deriv_ctrl(e, LAT_KNOTS, 3)
    re = deriv_ctrl(qe, LAT_KNOTS[1:-1], 2)
    N2 = bspline_basis(xc, LAT_KNOTS[1:-1], 2)[2]
    N1 = bspline_basis(xc, LAT_KNOTS[2:-2], 1)[1]
    if x.dim() == 1:
        f = lambda Nm, ctrl: ctrl @ Nm.T
    else:
        f = lambda Nm, ctrl: (Nm * ctrl[:, None, :]).sum(-1)
    return {"d": f(Ns[3], e), "d_x": f(N2, qe), "d_xx": f(N1, re)}


def lat_e_from_w(w: torch.Tensor) -> torch.Tensor:
    """[.., 6] w -> [.., 8] e (e_0 = e_1 = 0, e_i = D_MAX tanh w)."""
    z = torch.zeros(w.shape[:-1] + (2,), dtype=w.dtype, device=w.device)
    return torch.cat([z, D_MAX * torch.tanh(w)], -1)


def offset_curvature(pe: PathEval, d, d_s, d_ss) -> torch.Tensor:
    """Exact curvature of Gamma(s) + d(s) N(s) for a general path parameter s (metric g = pe.g)."""
    g, k, gs, ks = pe.g, pe.kappa, pe.g_s, pe.kappa_s
    q = 1.0 - k * d
    num = g * g * g * k * q * q + g * q * d_ss - gs * d_s * q + g * d_s * ks * d + 2.0 * g * k * d_s * d_s
    den = (g * g * q * q + d_s * d_s) ** 1.5
    return num / den


# ----------------------------------------------------------------------------------------------- decode
def _integrate(S: torch.Tensor, c: torch.Tensor, s0d: torch.Tensor, B3: torch.Tensor):
    """s1 [B,41], point speeds v1 [B,41], dv [B,41] for control points c [B,8] on knots S [B,9]."""
    u = (S[:, 1:] - S[:, :-1]) / T_POSE                                   # [B, 8]
    dv = c @ B3.T                                                         # [B, 41]
    kint = torch.arange(NK - 1, device=S.device) // SUB                   # interval of step n -> n+1
    un = u[:, kint]
    step = 0.5 * DT * ((torch.relu(un + dv[:, :-1]) - un) + (torch.relu(un + dv[:, 1:]) - un))
    ds = torch.cat([torch.zeros_like(S[:, :1]), torch.cumsum(step, 1)], 1)
    s1 = s0d + ds
    kpt = torch.clamp(torch.arange(NK, device=S.device) // SUB, max=N_POSE - 1)
    v1 = torch.relu(u[:, kpt] + dv)
    return s1, v1, dv, u


def decode(tau0, z_lon, w_lat, v0=None, mode: str = "A", *, path: Optional[DraftPath] = None,
           lat_len=None, n_proj: int = 6, lon_st_slope: float = 0.0) -> Dict[str, torch.Tensor]:
    """Decode correction controls into a trajectory.

    Args:
      tau0 : [B, 8, 3] draft (N frame).  Output dtype follows tau0 (float32 or float64).
      z_lon: [B, 6] raw longitudinal controls (unbounded; Q = A tanh z).
      w_lat: [B, 6] raw lateral controls (unbounded; e = D_MAX tanh w).
      v0   : [B] ego speed at t0 [m/s] or None (flags / kappa_lim at n = 0 only).
      mode : 'A' deceleration only (default), 'B' limited acceleration, 'P' perturbation generator.
      path : optional DraftPath whose first 8 poses are tau0 (e.g. the human path up to 8 s); default DraftPath(tau0)
             (extended by the constant-curvature rule in mode B).
      lat_len: optional [B] S_L override.
      lon_st_slope: mode A only (ignored in modes B / P): backward slope lam >= 0 of the clamp c <- min(c, 0) on the
             clamped part c > 0 (lon_clamp_a).  The forward, hence every returned value, does not depend on it
             (bit-identical); 0 (default) = plain torch.clamp (zero gradient on c > 0).  For training losses only.
    Returns dict:
      traj [B,8,3], dense [B,41,3] (scorer reference of traj), s [B,41] (= s1), v [B,41] (v1 at t_n; interval speed
      of the interval starting at t_n, the last interval for n = 40), d [B,41], kappa [B,41] (kappa_new),
      s0 [B,41], dv / da / jerk [B,41] (analytic spline terms), c_lon [B,8] (after the mode clamp / beta; mode-A
      liveness = (c_lon < 0).any(-1), see lon_live), q_lon [B,7], r_lon [B,6], e_lat [B,8]
      (after projection), dh [B,8] (H1 heading correction at the knots), flags (dict of [B] tensors: alpha, beta,
      lat_on, ext_m, extrap_m, kappa_ratio, kd_min, kd_floor_hit, kappa_path_clipped, cont_ok, first_seg_dv),
      path (the DraftPath).
    """
    tau0 = as_tensor(tau0)
    if not tau0.is_floating_point():
        tau0 = tau0.float()
    single = tau0.dim() == 2
    if single:
        tau0 = tau0[None]
    dt, dev = tau0.dtype, tau0.device
    B = tau0.shape[0]
    z = as_tensor(z_lon, dtype=dt, device=dev).reshape(B, N_FREE)
    w = as_tensor(w_lat, dtype=dt, device=dev).reshape(B, N_FREE)
    if mode not in ("A", "B", "P"):
        raise ValueError(f"mode {mode!r}")
    B3, B2, B1 = _lon_basis(dt, dev)

    own_path = path is None
    if own_path:
        path = DraftPath(tau0)
    S = path.knots()
    S8 = S[:, -1]
    u_end = (S[:, -1] - S[:, -2]) / T_POSE
    L_ext = torch.minimum(torch.full_like(S8, L_EXT_MAX), u_end * T_EXT)
    if mode == "B" and own_path:
        path.extend(L_ext, "const_curv", v=u_end)
    s0d = path.s0_dense()

    # ------------------------------------------------------------------ longitudinal
    c = lon_c_from_q(lon_q_from_z(z))
    if mode == "A":
        c = lon_clamp_a(c, lon_st_slope)
    elif mode == "B":
        c = torch.clamp(c, max=DV_ACC)
    s1, v1, dv, u = _integrate(S, c, s0d, B3)
    beta = torch.ones_like(S8)
    if mode in ("B", "P"):
        s_max = S8 + L_ext if mode == "B" else path.S_end + path.ext_len
        c_neg = torch.clamp(c, max=0.0)
        s1_neg = _integrate(S, c_neg, s0d, B3)[0][:, -1]
        over = s1[:, -1] > s_max
        den = s1[:, -1] - s1_neg
        den_safe = torch.where(over & (den > 0), den, torch.ones_like(den))
        beta = torch.where(over, torch.clamp((s_max - s1_neg) / den_safe, 0.0, 1.0), beta)
        c = c_neg + beta[:, None] * (c - c_neg)
        s1, v1, dv, u = _integrate(S, c, s0d, B3)
    q = lon_q_from_c(c)
    r = deriv_ctrl(q, LON_KNOTS[1:-1], 2)
    da = q @ B2.T
    jerk = r @ B1.T

    # ------------------------------------------------------------------ lateral
    lat_on = S8 >= S_LAT_MIN
    if lat_len is not None:
        S_L = torch.broadcast_to(as_tensor(lat_len, dtype=dt, device=dev), (B,))
    else:
        S_L = S8 + L_ext if mode == "B" else S8
    S_Ls = torch.clamp(S_L, min=1e-3)
    e = lat_e_from_w(w) * lat_on[:, None].to(dt)
    pe = path.eval(s1)
    # clip spline-artefact curvature (cusps through creeping stand-still knots reach |kappa| ~ 1e3 /m and give
    # |d kappa / dz| ~ 1e5); the ego's minimum turning radius is ~5.6 m, so |kappa| <= 1 /m never binds on a real path
    k_clip = pe.kappa.abs() > KAPPA_PATH_MAX
    pe.kappa = torch.clamp(pe.kappa, -KAPPA_PATH_MAX, KAPPA_PATH_MAX)
    pe.kappa_s = torch.where(k_clip, torch.zeros_like(pe.kappa_s), pe.kappa_s)
    # same for the metric: a draft that reverses (hairpin through the start) gives a spline cusp with g -> 0
    g_clip = pe.g < G_MIN
    pe.g = torch.clamp(pe.g, min=G_MIN)
    pe.g_s = torch.where(g_clip, torch.zeros_like(pe.g_s), pe.g_s)
    x = s1 / S_Ls[:, None]
    inside = x < 1.0
    lp = lat_profile(e, x)
    d_u = lp["d"]
    ds_u = torch.where(inside, lp["d_x"] / S_Ls[:, None], torch.zeros_like(d_u))
    dss_u = torch.where(inside, lp["d_xx"] / (S_Ls * S_Ls)[:, None], torch.zeros_like(d_u))
    v_k = v1
    if v0 is not None:
        v0t = torch.broadcast_to(as_tensor(v0, dtype=dt, device=dev), (B,))
        v_k = torch.cat([v0t[:, None], v1[:, 1:]], 1)
    Kn = torch.maximum(kappa_limit(v_k), (1.0 + KAPPA_SLACK) * pe.kappa.abs())
    alpha = torch.ones_like(S8)
    for _ in range(n_proj):
        kn = offset_curvature(pe, alpha[:, None] * d_u, alpha[:, None] * ds_u, alpha[:, None] * dss_u)
        dk = kn - pe.kappa
        adk = dk.abs()
        act = adk > 1e-9
        a_n = torch.where(act, (Kn - torch.sign(dk) * pe.kappa) / torch.where(act, adk, torch.ones_like(adk)),
                          torch.full_like(adk, float("inf")))
        alpha = alpha * torch.clamp(a_n.min(1).values, max=1.0)
    d = alpha[:, None] * d_u
    d_s = alpha[:, None] * ds_u
    d_ss = alpha[:, None] * dss_u
    kappa_new = offset_curvature(pe, d, d_s, d_ss)
    e = alpha[:, None] * e

    # ------------------------------------------------------------------ poses at the knots (anchored)
    ki = torch.tensor(_KNOT_IDX, device=dev)
    pS = path.eval(S[:, 1:])
    xy_k = pe.xy[:, ki]
    nrm_k = pe.normal[:, ki]
    d_k, ds_k = d[:, ki], d_s[:, ki]
    q_k = 1.0 - pe.kappa[:, ki] * d_k
    dh = torch.atan2(ds_k, pe.g[:, ki] * torch.clamp(q_k, min=KD_MIN))
    xy = tau0[..., :2] + (xy_k - pS.xy) + d_k[..., None] * nrm_k
    hd = tau0[..., 2] + (pe.heading[:, ki] - pS.heading) + dh
    traj = torch.cat([xy, hd[..., None]], -1)
    dense = dense_reference(traj)

    # ------------------------------------------------------------------ flags
    qd = 1.0 - pe.kappa * d
    first = safe_norm(traj[:, 0, :2]) / T_POSE
    flags = {
        "alpha": alpha, "beta": beta, "lat_on": lat_on,
        "ext_m": torch.relu(s1[:, -1] - S8),
        "extrap_m": torch.relu(s1[:, -1] - path.S_end),
        "kappa_ratio": (kappa_new.abs() / Kn).max(1).values,
        "kd_min": qd.min(1).values,
        "kd_floor_hit": (q_k < KD_MIN).any(1),
        "kappa_path_clipped": k_clip.any(1) | g_clip.any(1),
    }
    if v0 is not None:
        fdv = first - v0t
        flags["first_seg_dv"] = fdv
        flags["cont_ok"] = (fdv >= CONT_LO) & (fdv <= CONT_HI)
    out = {"traj": traj, "dense": dense, "s": s1, "v": v1, "d": d, "kappa": kappa_new, "s0": s0d,
           "dv": dv, "da": da, "jerk": jerk, "c_lon": c, "q_lon": q, "r_lon": r, "e_lat": e, "dh": dh,
           "flags": flags, "path": path}
    if single:
        out = {k: (v[0] if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
        out["flags"] = {k: v[0] for k, v in flags.items()}
    return out


def apply_gate(tau0: torch.Tensor, traj: torch.Tensor, modify: torch.Tensor) -> torch.Tensor:
    """Gate rule: rows with modify == False return the ORIGINAL tau0 values (no float round trip)."""
    return torch.where(modify.reshape(-1, *([1] * (tau0.dim() - 1))).bool(), traj.to(tau0.dtype), tau0)


# ----------------------------------------------------------------------------------------------- inverse fit
@torch.no_grad()
def fit_controls(tau0, target_xy, v0=None, mode: str = "A", *, path=None, iters: int = 25, lon: bool = True,
                 lat: bool = True, z_init=None, w_init=None, fd_eps: float = 1e-6, target_h=None,
                 w_head: float = 2.0) -> Dict:
    """Least-squares controls (z, w) so that decode(tau0, z, w, v0, mode).traj[..., :2] matches target_xy [B,8,2]
    (batched Levenberg-Marquardt; central-difference Jacobian from ONE stacked decode of 25 B rows per iteration;
    central because at z = 0 in mode A the +z side is clamped flat).
    Used for round-trip checks (perturbed draft -> how close can the refiner's decoder get back to the human).
    target_h [B,8] (optional) adds heading residuals w_head * wrap(h - target_h) [m per rad]; without it the
    position-only problem is not unique on curves (along-track timing can be traded for a lateral offset).

    Returns dict(z [B,6], w [B,6], err [B] = max knot position error [m], herr [B] = max knot heading error [rad]
    (nan without target_h), traj [B,8,3]).
    """
    tau0 = as_tensor(tau0, dtype=torch.float64)
    if tau0.dim() == 2:
        tau0 = tau0[None]
    B = tau0.shape[0]
    P = 2 * N_FREE
    tgt = as_tensor(target_xy, dtype=torch.float64).reshape(B, N_POSE, 2)
    tgh = None if target_h is None else as_tensor(target_h, dtype=torch.float64).reshape(B, N_POSE)
    v0r = None if v0 is None else torch.broadcast_to(as_tensor(v0, dtype=torch.float64), (B,))
    z = torch.zeros(B, N_FREE, dtype=torch.float64) if z_init is None else as_tensor(z_init, dtype=torch.float64).clone()
    w = torch.zeros(B, N_FREE, dtype=torch.float64) if w_init is None else as_tensor(w_init, dtype=torch.float64).clone()
    mask = torch.tensor([1.0 if lon else 0.0] * N_FREE + [1.0 if lat else 0.0] * N_FREE, dtype=torch.float64)
    # stacked copies: block 0 = base, block 1 + j = parameter j + eps, block 1 + P + j = parameter j - eps
    NB = 2 * P + 1
    tau_rep = tau0.repeat(NB, 1, 1)
    tgt_rep = tgt.repeat(NB, 1, 1)
    v0_rep = None if v0r is None else v0r.repeat(NB)
    path_rep = DraftPath(tau_rep) if path is None else None
    path_one = DraftPath(tau0) if path is None else path
    eye = torch.eye(P, dtype=torch.float64)

    def res_of(o, tg, th):
        r = (o["traj"][..., :2] - tg).reshape(tg.shape[0], -1)
        if th is not None:
            r = torch.cat([r, w_head * wrap_angle(o["traj"][..., 2] - th)], 1)
        return r

    tgh_rep = None if tgh is None else tgh.repeat(NB, 1)

    def resid_one(p):
        o = decode(tau0, p[:, :N_FREE], p[:, N_FREE:], v0r, mode, path=path_one)
        return res_of(o, tgt, tgh), o

    p = torch.cat([z, w], 1)
    lam = torch.full((B,), 1e-2, dtype=torch.float64)
    r0, _ = resid_one(p)
    cost = (r0 ** 2).sum(1)
    for _ in range(iters):
        dp = torch.cat([torch.zeros(1, P, dtype=torch.float64), eye, -eye], 0)
        pr = p.repeat(NB, 1) + fd_eps * dp.repeat_interleave(B, 0)
        if path is None:
            o = decode(tau_rep, pr[:, :N_FREE], pr[:, N_FREE:], v0_rep, mode, path=path_rep)
            rr = res_of(o, tgt_rep, tgh_rep).reshape(NB, B, -1)
        else:  # user path: decode block by block (path rows are per-token)
            rr = torch.stack([resid_one(pr[i * B:(i + 1) * B])[0] for i in range(NB)], 0)
        r = rr[0]
        J = ((rr[1:P + 1] - rr[P + 1:]) / (2 * fd_eps)).permute(1, 2, 0) * mask    # [B, R, P]
        JtJ = J.transpose(1, 2) @ J
        g = (J.transpose(1, 2) @ r[..., None])[..., 0]
        A = JtJ + lam[:, None, None] * (torch.diag_embed(torch.diagonal(JtJ, dim1=1, dim2=2)) + 1e-6 * eye)
        step = -torch.linalg.solve(A, g[..., None])[..., 0] * mask
        rn, _ = resid_one(p + step)
        cn = (rn ** 2).sum(1)
        better = cn < cost
        p = torch.where(better[:, None], p + step, p)
        cost = torch.where(better, cn, cost)
        lam = torch.where(better, lam / 3.0, lam * 4.0).clamp(1e-9, 1e9)
    _, o = resid_one(p)
    err = safe_norm(o["traj"][..., :2] - tgt).max(1).values
    herr = (wrap_angle(o["traj"][..., 2] - tgh).abs().max(1).values if tgh is not None
            else torch.full_like(err, float("nan")))
    return {"z": p[:, :N_FREE], "w": p[:, N_FREE:], "err": err, "herr": herr, "traj": o["traj"], "out": o}


# ----------------------------------------------------------------------------------------------- perturbations
FAMILY = {"identity": 0, "small": 1, "lconst": 2, "ignore_brake": 3, "creep": 4, "lat": 5, "combined": 6,
          "cv": 7, "hdrift": 8}
FAMILY_NAME = {v: k for k, v in FAMILY.items()}
PARAM_COLS = ("A_p", "t_on", "D_p", "s_on", "aux", "ds4")
BANK_LAYOUT = ("identity", "small", "small", "small", "lconst", "lconst", "lconst", "ignore_brake", "creep",
               "lat", "lat", "combined", "cv")
T_ON_SET = (0.0, 0.5, 1.0, 1.5, 2.0)
J_PERT = 2.0                        # jerk bound of perturbation profiles [m/s^3]
LAT_ACC_PERT = 3.8                  # human p99.9 lateral accel [m/s^2] (fact_drafts.md (c))
HDRIFT_MED_DEG, HDRIFT_SIG = 0.97, 1.105   # |dh_8| ~ LogNormal: median 0.97 deg, p90 4.0 deg [navtest-derived]
_T_FIT = np.arange(0, 81) * 0.05
_X_FIT = np.linspace(0.0, 1.0, 101)


def rng_for_token(token: str, salt: str = "refiner_T_bank_v1") -> np.random.Generator:
    """Deterministic per-token generator (sha256, not Python's salted hash())."""
    h = hashlib.sha256(f"{salt}:{token}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def _lon_design():
    """LS design over _T_FIT: dv = Adv @ Q_free, da = Ada @ Q_free (Q_free = Q_1..Q_6)."""
    t = torch.as_tensor(_T_FIT, dtype=torch.float64)
    N3 = bspline_basis(t, LON_KNOTS, 3)[3].numpy()
    N2 = bspline_basis(t, LON_KNOTS[1:-1], 2)[2].numpy()
    Mc = np.zeros((N_CTRL, N_FREE))
    for i in range(N_FREE):
        Mc[i + 2:, i] = LON_CSPAN[i + 1] / 3.0
    return N3 @ Mc, N2[:, 1:]


def _lat_design():
    x = torch.as_tensor(_X_FIT, dtype=torch.float64)
    N3 = bspline_basis(x, LAT_KNOTS, 3)[3].numpy()
    return N3[:, 2:]                                   # d = A @ e_2..e_7


_ADV, _ADA = _lon_design()
_ALAT = _lat_design()
_JSPAN = tuple(LON_KNOTS[1:-1][i + 3] - LON_KNOTS[1:-1][i + 1] for i in range(6))   # (.8,1.6,1.6,1.6,1.6,.8)


def project_lon(dv_target: np.ndarray, da_target: np.ndarray, j_max: Optional[float] = J_PERT,
                w_da: float = 0.5) -> np.ndarray:
    """LS projection of a (dv, da) profile on _T_FIT onto Q_1..Q_6 (Q_0 = 0), then jerk clip |R_i| <= j_max
    (sequential, from Q_0 = 0) and |Q| <= 0.98 A.  Returns Q_free [6] (float64)."""
    A = np.concatenate([_ADV, w_da * _ADA], 0)
    y = np.concatenate([dv_target, w_da * da_target], 0)
    Q = np.linalg.lstsq(A, y, rcond=None)[0]
    if j_max is not None:
        full = np.concatenate([[0.0], Q])
        for i in range(6):
            lim = j_max * _JSPAN[i] / 2.0
            full[i + 1] = full[i] + np.clip(full[i + 1] - full[i], -lim, lim)
        Q = full[1:]
    return np.clip(Q, -0.98 * A_DEC, 0.98 * A_UP)


def project_lat(d_target: np.ndarray, flat_end: bool = True) -> np.ndarray:
    """LS projection of d(x) on _X_FIT (x = s / S_L) onto e_2..e_7 (flat_end: e_6 = e_7 so d'(S_L) = 0).
    Returns e_free [6] clipped to |e| <= 0.98 D_MAX."""
    A = _ALAT
    if flat_end:
        A = np.concatenate([A[:, :4], A[:, 4:5] + A[:, 5:6]], 1)
        e = np.linalg.lstsq(A, d_target, rcond=None)[0]
        e = np.concatenate([e, e[-1:]])
    else:
        e = np.linalg.lstsq(A, d_target, rcond=None)[0]
    return np.clip(e, -0.98 * D_MAX, 0.98 * D_MAX)


def lconst_profile(A_p: float, t_on: float, j: float = J_PERT):
    """L-const: da(t) = min(A_p, j (t - t_on)) for t >= t_on (0 before); returns (dv, da) on _T_FIT."""
    da = np.clip(j * (_T_FIT - t_on), 0.0, None)
    da = np.minimum(da, A_p) if A_p >= 0 else np.maximum(-da, A_p)
    dv = np.concatenate([[0.0], np.cumsum(0.5 * (da[1:] + da[:-1]) * np.diff(_T_FIT))])
    return dv, da


def smoothstep_lat(D_p: float, s_on: float, S_L: float) -> np.ndarray:
    """Lat: d(s) = D_p * smoothstep((s - s_on) / (S_L - s_on)) on x = _X_FIT."""
    s = _X_FIT * S_L
    u = np.clip((s - s_on) / max(S_L - s_on, 1e-6), 0.0, 1.0)
    return D_p * u * u * (3.0 - 2.0 * u)


def keyframe_kinematics(traj: np.ndarray, v0: Optional[float] = None) -> Dict[str, float]:
    """0.5 s keyframe kinematics of one trajectory [8,3] (+ ego v0): lon accel min/max (segment-speed differences,
    first one (u_0 - v0) / 0.25 s), max lateral accel u_k |dh_k| / 0.5, max |kappa| = |dh_k| / l_k (l_k >= 1 m),
    max yaw rate."""
    P = np.vstack([[0.0, 0.0, 0.0], np.asarray(traj, np.float64)])
    ell = np.hypot(*np.diff(P[:, :2], axis=0).T)
    u = ell / T_POSE
    acc = list(np.diff(u) / T_POSE)
    if v0 is not None and np.isfinite(v0):
        acc = [(u[0] - v0) / (T_POSE / 2)] + acc
    h = np.unwrap(P[:, 2])
    dh = np.diff(h)
    yaw = np.abs(dh) / T_POSE
    kap = np.where(ell >= 1.0, np.abs(dh) / np.maximum(ell, 1e-9), 0.0)
    return {"acc_min": float(np.min(acc)), "acc_max": float(np.max(acc)), "lat_max": float(np.max(u * yaw)),
            "kappa_max": float(np.max(kap)), "yaw_max": float(np.max(yaw))}


class HumanContext:
    """Per-token inputs of the perturbation generator.

    tau_h    : [8,3] human trajectory (N frame; returned bytes for the identity draft).
    path_long: optional [M,3] human poses at t = 0.5 .. 0.5 M (M <= 16, i.e. up to 8 s from the log; NaN or rows
               >= n_valid = unavailable).  Its first 8 poses are REPLACED by tau_h so that the draft knots are exactly
               the human's (extract_human.py writes the same values; differences are float round-off).
    v0, a0   : ego speed / longitudinal accel at t0 (metric cache ego_state).
    centerline: optional [L,2] route centerline in N (used only by L-creep when the human path is too short).
    """

    def __init__(self, tau_h, path_long=None, n_valid=None, v0=None, a0=None, centerline=None):
        self.tau_h = np.asarray(tau_h, np.float32).reshape(N_POSE, 3)
        self.v0 = None if v0 is None or not np.isfinite(v0) else float(v0)
        self.a0 = None if a0 is None or not np.isfinite(a0) else float(a0)
        th64 = torch.as_tensor(self.tau_h.astype(np.float64))[None]
        self.draft_path = DraftPath(th64)
        self.S = self.draft_path.knots()[0].numpy()
        self.S8 = float(self.S[-1])
        self.u = np.diff(self.S) / T_POSE
        if path_long is not None:
            pl = np.array(path_long, np.float64).reshape(-1, 3)
            if n_valid is not None:
                pl[int(n_valid):] = np.nan
            pl[:N_POSE] = self.tau_h.astype(np.float64)
            self.long_path = DraftPath(torch.as_tensor(pl)[None])
        else:
            self.long_path = self.draft_path
        self.S_avail = float(self.long_path.S_end[0])
        self.centerline = None if centerline is None else np.asarray(centerline, np.float64).reshape(-1, 2)
        vh0 = self.v0 if self.v0 is not None else float(self.u[0])
        self.v_h0 = vh0
        self.decel = vh0 - float(self.u.min())
        self.kin_h = keyframe_kinematics(self.tau_h, self.v0)
        self.first_dv_h = None if self.v0 is None else float(self.u[0] - self.v0)

    @property
    def decel_ok(self):
        return self.decel >= 1.0

    @property
    def creep_ok(self):
        return self.S8 < 2.0

    @property
    def lat_ok(self):
        return self.S8 >= S_LAT_MIN


def centerline_continuation(end_pose: np.ndarray, centerline: np.ndarray, length: float, step: float = 1.0,
                            blend: float = 10.0) -> np.ndarray:
    """Poses continuing from end_pose (x, y, h) that blend onto the centerline: Frenet offset from the centerline
    with initial value and slope equal to the ego's (at end_pose), decaying to 0 over `blend` m
    (critic_logic issue 8).  Returns [ceil(length/step), 3]."""
    C = np.asarray(centerline, np.float64)
    seg = np.diff(C, axis=0)
    sl = np.hypot(*seg.T)
    keep = sl > 1e-6
    C = np.vstack([C[:1], C[1:][keep]])
    seg = np.diff(C, axis=0)
    sl = np.hypot(*seg.T)
    Sc = np.concatenate([[0.0], np.cumsum(sl)])
    p = end_pose[:2]
    # projection of p
    tt = np.clip(((p - C[:-1]) * seg).sum(1) / sl ** 2, 0, 1)
    proj = C[:-1] + tt[:, None] * seg
    i = int(np.argmin(np.hypot(*(proj - p).T)))
    s_p = Sc[i] + tt[i] * sl[i]
    th_c = np.arctan2(seg[i, 1], seg[i, 0])
    n_c = np.array([-np.sin(th_c), np.cos(th_c)])
    d0 = float((p - proj[i]) @ n_c)
    psi = float(wrap_angle(torch.tensor(end_pose[2] - th_c)).item())
    slope = math.tan(np.clip(psi, -1.2, 1.2))
    k1 = slope + 2.0 * d0 / blend
    us = np.arange(1, int(np.ceil(length / step)) + 1) * step
    out = []
    for u in us:
        sc = s_p + u
        dd = (d0 + k1 * u) * (1 - u / blend) ** 2 if u < blend else 0.0
        if sc <= Sc[-1]:
            j = min(int(np.searchsorted(Sc, sc, side="right") - 1), len(seg) - 1)
            base = C[j] + (sc - Sc[j]) / sl[j] * seg[j]
            th = np.arctan2(seg[j, 1], seg[j, 0])
        else:
            th = np.arctan2(seg[-1, 1], seg[-1, 0])
            base = C[-1] + (sc - Sc[-1]) * np.array([np.cos(th), np.sin(th)])
        nn = np.array([-np.sin(th), np.cos(th)])
        out.append(np.array([*(base + dd * nn), th]))
    out = np.array(out)
    # headings from the polyline direction (blend changes them slightly)
    xy = np.vstack([p, out[:, :2]])
    d = np.diff(xy, axis=0)
    out[:, 2] = np.arctan2(d[:, 1], d[:, 0])
    return out


def _decode_np(ctx: HumanContext, Q, e, path: DraftPath, mode="P", lat_len=None):
    th = torch.as_tensor(ctx.tau_h.astype(np.float64))[None]
    z = lon_z_from_q(torch.as_tensor(np.asarray(Q, np.float64))[None])
    w = torch.atanh(torch.as_tensor(np.asarray(e, np.float64))[None] / D_MAX)
    o = decode(th, z, w, ctx.v0, mode, path=path, lat_len=lat_len)
    return o


def _result(ctx, family, o=None, Q=None, e=None, draft=None, params=None, valid=True, reason="", path_src="draft",
            path=None):
    """Pack one sample.  z_lon / w_lat are the RAW controls given to decode (Q before beta, e before the curvature
    projection), so decode(tau_h, z, w, v0, 'P', path=<path_src path>, lat_len=S_8) reproduces the draft exactly."""
    fam = FAMILY[family]
    p = np.zeros(6, np.float32) if params is None else np.asarray(params, np.float32)
    z = np.zeros(N_FREE, np.float32)
    w = np.zeros(N_FREE, np.float32)
    fl = {}
    if o is not None:
        z = lon_z_from_q(torch.as_tensor(np.asarray(Q, np.float64))).numpy().astype(np.float32)
        w = np.arctanh(np.asarray(e, np.float64) / D_MAX).astype(np.float32)
        draft = o["traj"][0].numpy().astype(np.float32)
        p[5] = float(o["s"][0, -1] - ctx.S8)
        fl = {k: float(v[0]) for k, v in o["flags"].items()}
    return {"family": family, "code": fam, "draft": draft, "z_lon": z, "w_lat": w, "params": p, "valid": bool(valid),
            "reason": reason, "path_src": path_src, "flags": fl, "path": path,
            "q_eff": None if o is None else o["q_lon"][0, 1:].numpy(),
            "e_eff": None if o is None else o["e_lat"][0, 2:].numpy()}


def check_draft(ctx: HumanContext, draft: np.ndarray, tol: float = 1e-6) -> str:
    """'' if the draft satisfies the bank constraints, else the reason.  Each limit is max(limit, human's own value):
    t0 continuity (first-segment speed - v0 in [-0.8, +0.6]), keyframe lon accel in [-4.05, 2.40], lateral accel
    <= 3.8, |kappa| <= 0.213 (segments >= 1 m)."""
    if not np.isfinite(draft).all():
        return "nonfinite"
    k = keyframe_kinematics(draft, ctx.v0)
    kh = ctx.kin_h
    if ctx.v0 is not None:
        fdv = float(np.hypot(*draft[0, :2]) / T_POSE - ctx.v0)
        lo, hi = min(CONT_LO, ctx.first_dv_h), max(CONT_HI, ctx.first_dv_h)
        if not (lo - tol <= fdv <= hi + tol):
            return "continuity"
    if k["acc_max"] > max(LON_ACC_MAX, kh["acc_max"]) + tol or k["acc_min"] < min(LON_ACC_MIN, kh["acc_min"]) - tol:
        return "lon_acc"
    if k["lat_max"] > max(LAT_ACC_PERT, kh["lat_max"]) + tol:
        return "lat_acc"
    if k["kappa_max"] > max(KAPPA_MAX, kh["kappa_max"]) + tol:
        return "kappa"
    return ""


def _speedup_path(ctx: HumanContext, need: float, allow_centerline: bool):
    """Path source for a speed-up needing s(4) up to S8 + need: ('human_long'|'centerline'|'extrap', DraftPath)."""
    if ctx.S_avail >= ctx.S8 + need - 1e-9 or not allow_centerline:
        return "human_long", ctx.long_path
    ext = need + 2.0
    if ctx.centerline is not None and len(ctx.centerline) >= 2:
        base = ctx.tau_h.astype(np.float64)
        cont = centerline_continuation(base[-1], ctx.centerline, ext)
        P = torch.as_tensor(np.vstack([base, cont]))[None]
        return "centerline", DraftPath(P)
    p = DraftPath(torch.as_tensor(ctx.tau_h.astype(np.float64))[None]).extend(ext, "straight")
    return "extrap", p


def sample_perturbation(family: str, ctx: HumanContext, rng: np.random.Generator, *, mode: str = "A",
                        max_tries: int = 8, centerline_families: Sequence[str] = ("creep",)) -> Dict:
    """Sample one draft of a §3.6 family in the decoder basis.

    Families (A_p = accel offset [m/s^2], D_p = end lateral offset [m], both signed; S = human S_8):
      identity     : z = w = 0, draft = tau_h BYTES.
      small        : A_p ~ U[0, 0.2] (mode A; U[-0.2, 0.2] in mode B), t_on ~ T_ON_SET; D_p ~ U[-0.3, 0.3],
                     s_on ~ U[.2,.6] S (lateral only if S >= 3 m).
      lconst       : A_p ~ U[0.2, 1.3], t_on ~ {0,.5,1,1.5,2}, jerk <= 2; follows the human path up to 8 s;
                     reduced (beta) to the available path, rejected if the reduced A_p < 0.2; a rejection for
                     path length removes that t_on and all earlier ones from the retry set (a later onset needs
                     less path: ds(4) ~ A_p (4 - t_on)^2 / 2).
      ignore_brake : dv(t) = alpha max(0, v_h(0) - v_h(t)), alpha ~ U[0.3, 1] (human decel >= 1 m/s), v_h piecewise
                     linear through v0 and the mean speeds around each knot; reduced by beta, rejected if
                     alpha beta < 0.3 (params 'aux' = alpha beta).
      creep        : A_p ~ U[0.3, 1.0] from t = 0 (human S_8 < 2 m); path = human path up to 8 s, else the
                     centerline continuation (if given), else straight extrapolation (path_src flags which).
      lat          : |D_p| ~ U[0.3, 1.5] (random sign), s_on ~ U[.2,.6] S, smoothstep from s_on to S (flat end).
      combined     : lconst + lat.
      cv           : x = v0 t, y = 0, h = 0 (not in the basis; z = w = 0).  Invalid if v0 is unknown or the CV
                     draft is within 0.3 m of the human at every knot (redundant with identity).
      hdrift       : positions = human, headings + dh_8 k/8 with |dh_8| ~ LogNormal(median 0.97 deg, sigma 1.105),
                     random sign, clipped to 10 deg (student step-8 heading error, [navtest-derived]).
    Every decoded draft passes check_draft (t0 continuity + kinematic limits, human-relative), otherwise the family is
    re-sampled up to max_tries times; after that the result has valid = False and reason set.
    centerline_families: families allowed to continue on the centerline continuation when the human path is too short
    (IMPL_SPEC: only L-creep; adding 'lconst', 'ignore_brake', 'combined', 'small' is an open design option).
    Returns dict(family, code, draft [8,3] f32, z_lon [6] f32, w_lat [6] f32 (reproduce the draft with
    decode(tau_h, z, w, v0, 'P', path=path, lat_len=S)), params [6] f32 (PARAM_COLS), valid, reason, path_src,
    flags, path (the DraftPath used, in memory only), q_eff [6] / e_eff [6] (effective control points after beta and
    the curvature projection; -q_eff, -e_eff is the analytic inverse used as round-trip initialisation)).
    """
    if family not in FAMILY:
        raise ValueError(family)
    if family == "identity":
        return _result(ctx, "identity", draft=ctx.tau_h.copy())
    if family == "cv":
        if ctx.v0 is None:
            return _result(ctx, "cv", draft=ctx.tau_h.copy(), valid=False, reason="no_v0")
        t = np.arange(1, N_POSE + 1) * T_POSE
        dr = np.stack([ctx.v0 * t, np.zeros(N_POSE), np.zeros(N_POSE)], -1).astype(np.float32)
        if np.hypot(*(dr[:, :2] - ctx.tau_h[:, :2]).T).max() < 0.3:
            return _result(ctx, "cv", draft=dr, valid=False, reason="cv_redundant")
        return _result(ctx, "cv", draft=dr, params=[0, 0, 0, 0, ctx.v0, 0])
    if family == "hdrift":
        mag = math.radians(min(10.0, HDRIFT_MED_DEG * math.exp(HDRIFT_SIG * rng.standard_normal())))
        dh8 = mag * (1.0 if rng.random() < 0.5 else -1.0)
        dr = ctx.tau_h.astype(np.float64).copy()
        dr[:, 2] += dh8 * np.arange(1, N_POSE + 1) / N_POSE
        return _result(ctx, "hdrift", draft=dr.astype(np.float32), params=[0, 0, 0, 0, dh8, 0])

    reason = ""
    t_on_set = list(T_ON_SET)
    for _ in range(max_tries):
        if not t_on_set:
            break
        A_p = t_on = D_p = s_on = aux = 0.0
        Q = np.zeros(N_FREE)
        e = np.zeros(N_FREE)
        need = 0.0
        a_min = 0.0
        if family in ("small", "lconst", "combined", "creep"):
            if family == "small":
                A_p = rng.uniform(0.0, 0.2) if mode == "A" else rng.uniform(-0.2, 0.2)
                t_on = float(rng.choice(t_on_set))
            elif family == "creep":
                A_p, t_on, a_min = rng.uniform(0.3, 1.0), 0.0, 0.3
            else:
                A_p, t_on, a_min = rng.uniform(0.2, 1.3), float(rng.choice(t_on_set)), 0.2
            dvt, dat = lconst_profile(A_p, t_on)
            Q = project_lon(dvt, dat)
            need = float(np.trapz(dvt, _T_FIT))
        elif family == "ignore_brake":
            aux = rng.uniform(0.3, 1.0)
            tk = np.arange(N_POSE + 1) * T_POSE
            vk = np.concatenate([[ctx.v_h0], 0.5 * (ctx.u[1:] + ctx.u[:-1]), [ctx.u[-1]]])
            vh = np.interp(_T_FIT, tk, vk)
            dvt = aux * np.clip(ctx.v_h0 - vh, 0.0, None)
            dat = np.gradient(dvt, _T_FIT)
            Q = project_lon(dvt, dat)
            need = float(np.trapz(dvt, _T_FIT))
        if family in ("small", "lat", "combined") and ctx.lat_ok:
            if family == "small":
                D_p = rng.uniform(-0.3, 0.3)
            else:
                D_p = rng.uniform(0.3, 1.5) * (1.0 if rng.random() < 0.5 else -1.0)
            s_on = rng.uniform(0.2, 0.6) * ctx.S8
            e = project_lat(smoothstep_lat(D_p, s_on, ctx.S8))
        elif family == "lat":
            return _result(ctx, family, draft=ctx.tau_h.copy(), valid=False, reason="lat_short_path")
        src, path = _speedup_path(ctx, need, allow_centerline=(family in centerline_families))
        o = _decode_np(ctx, Q, np.asarray(e), path, "P", lat_len=ctx.S8)
        beta = float(o["flags"]["beta"][0])
        A_eff = A_p * beta
        if family in ("lconst", "combined", "creep") and A_eff < a_min - 1e-9:
            reason = "path_too_short"
            t_on_set = [t for t in t_on_set if t > t_on]
            continue
        if family == "ignore_brake":
            if aux * beta < 0.3 - 1e-9:
                reason = "path_too_short"
                continue
            aux = aux * beta
        if family in ("lat", "combined"):
            d_end = float(o["d"][0, -1])
            if abs(d_end) < 0.3 - 1e-6:
                reason = "lat_projected_small"
                continue
        dr = o["traj"][0].numpy().astype(np.float32)
        reason = check_draft(ctx, dr)
        if reason:
            continue
        d_real = float(o["d"][0, -1])
        return _result(ctx, family, o=o, Q=Q, e=e, params=[A_eff, t_on, d_real, s_on, aux, 0.0], path_src=src,
                       path=path)
    return _result(ctx, family, draft=ctx.tau_h.copy(), valid=False, reason=reason or "rejected")


def sample_bank(tau_h, token: str, *, path_long=None, n_valid=None, v0=None, a0=None, centerline=None,
                mode: str = "A", layout: Sequence[str] = BANK_LAYOUT, max_tries: int = 8,
                centerline_families: Sequence[str] = ("creep",)) -> Dict:
    """K = len(layout) (13) drafts for one token, seeded by rng_for_token(token).

    Fallbacks (IMPL_SPEC §3.6): ignore_brake -> lconst if the human decelerates < 1 m/s; creep -> lconst if human
    S_8 >= 2 m; cv -> hdrift if CV is invalid; lat / combined -> lconst if S_8 < 3 m or after max_tries rejections;
    a remaining failure keeps the human draft with valid = False.
    Returns dict: drafts [K,8,3] f32, family [K] i8, params [K,6] f32, z_lon [K,6] f32, w_lat [K,6] f32, valid [K]
    bool, path_src [K] str, reason [K] str.
    """
    rng = rng_for_token(token)
    ctx = HumanContext(tau_h, path_long, n_valid, v0, a0, centerline)
    out = []
    for fam in layout:
        f = fam
        if f == "ignore_brake" and not ctx.decel_ok:
            f = "lconst"
        if f == "creep" and not ctx.creep_ok:
            f = "lconst"
        if f in ("lat", "combined") and not ctx.lat_ok:
            f = "lconst"
        r = sample_perturbation(f, ctx, rng, mode=mode, max_tries=max_tries, centerline_families=centerline_families)
        if not r["valid"] and f == "cv":
            r = sample_perturbation("hdrift", ctx, rng, mode=mode)
        elif not r["valid"] and f in ("lat", "combined", "ignore_brake", "creep"):
            r2 = sample_perturbation("lconst", ctx, rng, mode=mode, max_tries=max_tries,
                                     centerline_families=centerline_families)
            r2["reason"] = f"fallback_from_{f}:{r['reason']}" if r2["valid"] else r2["reason"]
            r = r2
        out.append(r)
    return {
        "drafts": np.stack([r["draft"] for r in out]).astype(np.float32),
        "family": np.array([r["code"] for r in out], np.int8),
        "params": np.stack([r["params"] for r in out]).astype(np.float32),
        "z_lon": np.stack([r["z_lon"] for r in out]).astype(np.float32),
        "w_lat": np.stack([r["w_lat"] for r in out]).astype(np.float32),
        "valid": np.array([r["valid"] for r in out], bool),
        "path_src": np.array([r["path_src"] for r in out]),
        "reason": np.array([r["reason"] for r in out]),
    }
