"""Draft path geometry for the stage-T refiner decoder (IMPL_SPEC §2, §3.1).  torch, batched, differentiable.

Conventions
  Frame   : N frame = NAVSIM ego frame at t0 (rear-axle origin, x forward, y left, heading CCW from +x, rad; metres).
  Poses   : [B, N, 3] (x, y, heading) at t = 0.5, 1.0, ... (0.5 s apart).  The implicit pose 0 = (0, 0, 0) at t = 0 is
            prepended internally, so a path has K = N + 1 vertices V_0 = origin, V_k = poses[k - 1].  A draft has
            N = 8 (t = 0.5 .. 4.0 s); a "long" path (human path up to 8 s from the log) has N <= 16.
  Arc len : s is the POLYLINE (chord) arc length: S_k = sum_{j<k} |V_{j+1} - V_j| (S_0 = 0).  This is exactly the
            distance the scorer's reference travels (it moves linearly in time between poses), so the draft timing
            s0(t) = piecewise linear through (0.5 k, S_k) is the scorer's own arc-length schedule.
  Dense   : n = 0 .. 40, t = 0.1 n.  Dense index n = 5 k + j (j = 0..4) lies in pose interval k.
  Heading : unwrapped along the vertex sequence exactly like np.unwrap (a vertex heading is only changed by
            multiples of 2 pi, so un-wrapped values are the ORIGINAL bytes whenever no wrap occurs).

Path model (DraftPath)
  Gamma_sm : C2 cubic interpolating spline x(s), y(s) in the chord parameter s, start derivative (1, 0) (= ego
             heading at t0, |Gamma'| = 1 there), not-a-knot end (END_BC; natural if < 3 knots) and linear
             continuation after the last knot.  All Frenet quantities are computed for a GENERAL parameter:
             g = |Gamma'(s)| (= 1 only approximately for a chord parameter), T = Gamma'/g, N = rot90(T) (left normal),
             kappa = (x'y'' - y'x'') / g^3, g_s = dg/ds, kappa_s = dkappa/ds.
             End condition [navtest-derived, re-checked on the train split in report/refiner_T/decoder_validation.json]:
             on 4,000 navtest human paths the spline vs polyline max deviation is p50/p90/p99/max
             0.011/0.055/0.083/0.137 m for either end condition, but a natural end forces kappa = 0 at S_end and bends
             the end tangent, so its constant-curvature extension misses the real human path by p90 0.73 m at +1 s,
             vs 0.27 m with not-a-knot + the spline's end curvature (+2 s: 1.24 m; the chord-based rule of the draft
             gives 0.27 / 1.28 m).
  Knot merge: vertices closer than MERGE_EPS (0.2 m) to the previously retained vertex are NOT spline knots.
             [navtest-derived] Measured on 12,146 navtest human paths: consecutive segments shorter than 0.05 m have
             essentially random directions (p90 140-166 deg; localisation jitter at stand-still), 0.1-0.2 m segments
             p99 14 deg, 0.2-0.5 m p99 5 deg.  A C2 spline forced through jitter creates cm-scale loops with
             |kappa| ~ 1e2 /m, i.e. random normals, which would throw Frenet offsets of up to D_MAX = 2 m around at a
             stop.  (The decoder additionally clips |kappa| <= 1 /m and g >= 0.2 for the remaining cusps.)
  Gamma_eff: the path actually used for positions = Gamma_sm + piecewise-linear (in s) interpolation of the
             residuals r_k = V_k - Gamma_sm(S_k), anchored so that Gamma_eff(S_k) = V_k EXACTLY (bitwise):
                 Gamma_eff(s) = V_j + (Gamma_sm(s) - Gamma_sm(S_j)) + lam * (r_{j+1} - r_j),  lam = (s - S_j)/l_j,
             for s in original segment j (a zero-length virtual segment after the last vertex makes s = S_end an
             anchor too).  With no merged knot all r_k are rounding-level and Gamma_eff = the C2 interpolating spline;
             at merged (stand-still) knots it is C0 with small kinks.  Tangent, normal and curvature always come from
             Gamma_sm.
  h0(s)    : the DRAFT heading interpolated linearly in arc length between vertices (unwrapped).  Equals the pose
             headings at the knots (this is the H1 heading base of the decoder).
  Beyond S_end (last vertex): circular-arc extension from V_end with the tangent of Gamma_sm at S_end and curvature
             kappa_ext (0 = straight unless extend() was called; 'const_curv' = the spline's end curvature, i.e. the
             last segment's, clipped to kappa_limit(v_end); C2 junction).  eval() flags s > S_end ('beyond') and
             s > S_end + ext_len ('beyond_ext').

Deviations from IMPL_SPEC §3.1 (documented, interfaces unchanged)
  * "parameterised by arc length": the parameter is the chord arc length (the scorer's), not the spline's own arc
    length; the spline's true arc length is longer by a relative O(l^2 kappa^2 / 24).  Every Frenet formula uses the
    exact metric g = |Gamma'(s)|, so offsets are exact metres and headings exact.
  * "interpolating through the 8 points": exact at every vertex (Gamma_eff), but C2 only between retained knots;
    vertices within MERGE_EPS of the previous retained one (stand-still jitter) are not spline knots (see above).
  * eval() returns both the draft heading h0(s) ('heading', H1 base, identity at knots) and the geometric tangent
    heading ('tangent_heading').
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch

# ----------------------------------------------------------------------------------------------- constants
DT = 0.1                 # dense step [s]
NK = 41                  # dense samples 0 .. 4.0 s
T_POSE = 0.5             # pose spacing [s]
N_POSE = 8               # poses per draft
SUB = 5                  # dense steps per pose interval
T_HORIZON = 4.0

# nuPlan Pacifica (== metric-cache ego parameters == sf_common constants)
HALF_LEN = 2.588
HALF_WID = 1.1485
RA2C = 1.461             # rear axle -> box centre along heading

# PDMS v1 comfort limits (pdm_comfort_metrics.py)
LON_ACC_MIN, LON_ACC_MAX = -4.05, 2.40
LAT_ACC_MAX = 4.89
YAW_RATE_MAX = 0.95
YAW_ACC_MAX = 1.93
LON_JERK_MAX = 4.13
JERK_MAX = 8.37
KAPPA_MAX = 0.213        # human p99.9 path curvature [1/m] (fact_drafts.md (c))

MERGE_EPS = 0.2          # [m] knot merge threshold for the smooth frame (see module docstring)
END_BC = "not_a_knot"    # spline end condition (see module docstring / validate_decoder.py)
TWO_PI = 2.0 * math.pi
_START_TANGENT = (1.0, 0.0)

TensorLike = Union[torch.Tensor, "np.ndarray", float]  # noqa: F821


# ----------------------------------------------------------------------------------------------- helpers
def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap to [-pi, pi].  Exact (returns the input bytes) when |a| < pi."""
    return a - TWO_PI * torch.round(a / TWO_PI)


def unwrap_anchored(h: torch.Tensor) -> torch.Tensor:
    """np.unwrap along the last dim, adding only multiples of 2 pi (values unchanged where no wrap is needed)."""
    out = [h[..., 0]]
    for k in range(1, h.shape[-1]):
        m = torch.round((out[-1] - h[..., k]) / TWO_PI)
        out.append(h[..., k] + TWO_PI * m)
    return torch.stack(out, -1)


def kappa_limit(v: torch.Tensor, v_eps: float = 1e-3) -> torch.Tensor:
    """Speed-dependent curvature limit min(0.95/v, 4.89/v^2, 0.213) [1/m] (yaw rate, lateral accel, human p99.9)."""
    v = torch.clamp(v, min=v_eps)
    return torch.minimum(torch.minimum(YAW_RATE_MAX / v, LAT_ACC_MAX / (v * v)), torch.full_like(v, KAPPA_MAX))


def safe_norm(d: torch.Tensor) -> torch.Tensor:
    """|d| over the last dim with a finite (zero) gradient at d = 0."""
    sq = (d * d).sum(-1)
    pos = sq > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, sq, torch.ones_like(sq))), torch.zeros_like(sq))


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a / b, 0 where b == 0 (finite gradients)."""
    nz = b != 0
    return torch.where(nz, a / torch.where(nz, b, torch.ones_like(b)), torch.zeros_like(a))


def as_tensor(x, dtype=None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t
    t = torch.as_tensor(x, device=device)
    if dtype is not None:
        t = t.to(dtype)
    elif not t.is_floating_point():
        t = t.to(torch.get_default_dtype())
    return t


def dense_reference(poses: torch.Tensor) -> torch.Tensor:
    """Scorer reference of an 8-pose trajectory: [B, 8, 3] -> [B, 41, 3] at t = 0.1 n.

    Positions linear in time between [origin; poses]; heading np.unwrap-ed along [0; h_1..h_8] then linear
    (nuplan InterpolatedTrajectory / sf_common.Path.ref_poses_time).  Dense index 5k is the pose k itself.
    """
    single = poses.dim() == 2
    if single:
        poses = poses[None]
    B = poses.shape[0]
    P = torch.cat([poses.new_zeros(B, 1, 3), poses], 1)                       # [B, 9, 3]
    P = torch.cat([P[..., :2], unwrap_anchored(P[..., 2])[..., None]], -1)
    frac = torch.arange(SUB, dtype=poses.dtype, device=poses.device) / SUB     # 0, .2, .4, .6, .8
    A, Bn = P[:, :-1], P[:, 1:]
    dense = A[:, :, None, :] + (Bn - A)[:, :, None, :] * frac[None, None, :, None]
    dense = torch.cat([dense.reshape(B, -1, 3), P[:, -1:]], 1)
    return dense[0] if single else dense


def dense_index_split():
    """(k, j) of dense index n = 5k + j for n = 0..40; n = 40 -> (8, 0)."""
    n = torch.arange(NK)
    return n // SUB, n % SUB


# ----------------------------------------------------------------------------------------------- path
@dataclass
class PathEval:
    """Path quantities at query arc lengths s [B, M] (all tensors [B, M] or [B, M, 2])."""
    xy: torch.Tensor              # Gamma_eff(s) (exact vertex at s = S_k)
    heading: torch.Tensor         # draft heading h0(s), unwrapped, interpolated in arc length (H1 base)
    normal: torch.Tensor          # unit left normal of Gamma_sm
    kappa: torch.Tensor           # signed curvature of Gamma_sm [1/m] (left turn > 0)
    tangent: torch.Tensor         # unit tangent of Gamma_sm
    g: torch.Tensor               # |dGamma_sm/ds| (metric of the chord parameter)
    g_s: torch.Tensor             # dg/ds
    kappa_s: torch.Tensor         # dkappa/ds
    beyond: torch.Tensor          # s > S_end (extrapolated), bool
    beyond_ext: torch.Tensor      # s > S_end + ext_len, bool

    @property
    def tangent_heading(self) -> torch.Tensor:
        return torch.atan2(self.tangent[..., 1], self.tangent[..., 0])


class DraftPath:
    """C2 draft path through the origin and the poses, chord-arc-length parameter (see module docstring).

    Args:
      poses    : [B, N, 3] or [N, 3] tensor/array (x, y, heading) at t = 0.5 k.  NaN rows (unavailable frames)
                 and rows at index >= n_valid are replaced by the last valid pose (zero-length segments).
      n_valid  : optional [B] number of valid poses.
      merge_eps: knot merge threshold [m] for the smooth frame.
    Attributes (B = batch, K = N + 1 vertices):
      V [B,K,2], H [B,K] (unwrapped vertex headings, H[:,0] = 0), S [B,K], ell [B,K-1], keep [B,K] (spline knots),
      S_end [B], kappa_ext [B], ext_len [B].
    """

    def __init__(self, poses, n_valid=None, merge_eps: float = MERGE_EPS, dtype=None, end_bc: str = END_BC):
        P = as_tensor(poses, dtype=dtype)
        if not P.is_floating_point():
            P = P.float()
        if P.dim() == 2:
            P = P[None]
        B, N, _ = P.shape
        dev, dt = P.device, P.dtype
        # ---- invalid tail -> repeat the last valid pose
        finite = torch.isfinite(P).all(-1)                                              # [B, N]
        first_bad = torch.where(~finite, torch.arange(N, device=dev)[None].expand(B, N), torch.full((B, N), N, device=dev))
        nv = first_bad.min(1).values
        if n_valid is not None:
            nv = torch.minimum(nv, as_tensor(n_valid, device=dev).long())
        self.n_valid = nv
        if bool((nv < N).any()):
            idx = torch.arange(N, device=dev)[None].expand(B, N)
            src = torch.minimum(idx, (nv - 1).clamp(min=0)[:, None])
            Pv = torch.gather(torch.nan_to_num(P), 1, src[..., None].expand(B, N, 3))
            Pv = torch.where((nv == 0)[:, None, None], torch.zeros_like(Pv), Pv)
            P = torch.where((idx < nv[:, None])[..., None], P, Pv)
        self.poses = P
        self.B, self.N, self.K = B, N, N + 1
        self.merge_eps = float(merge_eps)
        if end_bc not in ("not_a_knot", "natural"):
            raise ValueError(end_bc)
        self.end_bc = end_bc
        V = torch.cat([P.new_zeros(B, 1, 2), P[..., :2]], 1)                            # [B, K, 2]
        Hraw = torch.cat([P.new_zeros(B, 1), P[..., 2]], 1)
        self.V, self.H = V, unwrap_anchored(Hraw)
        self.ell = safe_norm(V[:, 1:] - V[:, :-1])                                     # [B, K-1]
        self.S = torch.cat([P.new_zeros(B, 1), torch.cumsum(self.ell, 1)], 1)           # [B, K]
        self.S_end = self.S[:, -1]
        self._fit()
        self.kappa_ext = P.new_zeros(B)
        self.ext_len = P.new_zeros(B)

    # ------------------------------------------------------------------------------------------- fitting
    def _fit(self):
        B, K, V, S = self.B, self.K, self.V, self.S
        dev, dt = V.device, V.dtype
        with torch.no_grad():
            keep = [torch.ones(B, dtype=torch.bool, device=dev)]
            last = V[:, 0]
            for k in range(1, K):
                kk = safe_norm(V[:, k] - last) >= self.merge_eps
                last = torch.where(kk[:, None], V[:, k], last)
                keep.append(kk)
            keep = torch.stack(keep, 1)                                                 # [B, K]
            ar = torch.arange(K, device=dev)[None].expand(B, K)
            order = torch.argsort(torch.where(keep, ar, ar + K), dim=1)
            m = keep.sum(1)                                                             # [B] >= 1
        self.keep, self.m = keep, m
        sig = torch.gather(S, 1, order)
        X = torch.gather(V, 1, order[..., None].expand(B, K, 2))
        j = torch.arange(K, device=dev)[None].expand(B, K)
        is_pad = j >= m[:, None]
        # padding: strictly increasing dummy knots after the last retained one (masked out of the system)
        sig_last = torch.gather(sig, 1, (m - 1)[:, None])
        sig = torch.where(is_pad, sig_last + (j - m[:, None] + 1).to(dt), sig)
        X_last = torch.gather(X, 1, (m - 1)[:, None, None].expand(B, 1, 2))
        X = torch.where(is_pad[..., None], X_last.expand(B, K, 2), X)
        h = sig[:, 1:] - sig[:, :-1]                                                    # [B, K-1] > 0
        slope = (X[:, 1:] - X[:, :-1]) / h[..., None]                                   # [B, K-1, 2]
        # clamped-start system for the second derivatives M_j; end: not-a-knot (m >= 3) or natural
        is_last_or_pad = j >= (m - 1)[:, None]
        is_first = (j == 0) & ~is_last_or_pad
        interior = (j > 0) & ~is_last_or_pad
        hp = torch.cat([h, h.new_ones(B, 1)], 1)             # h_j   (j = 0..K-1, dummy at end)
        hm = torch.cat([h.new_ones(B, 1), h], 1)             # h_{j-1}
        sp = torch.cat([slope, slope.new_zeros(B, 1, 2)], 1)  # slope_j
        sm = torch.cat([slope.new_zeros(B, 1, 2), slope], 1)  # slope_{j-1}
        D0 = torch.tensor(_START_TANGENT, dtype=dt, device=dev)
        one, zero = torch.ones_like(hp), torch.zeros_like(hp)
        diag = torch.where(is_last_or_pad, one, torch.where(is_first, 2 * hp, 2 * (hm + hp)))
        lower = torch.where(interior, hm, zero)               # coefficient of M_{j-1} in row j
        upper = torch.where(is_first | interior, hp, zero)    # coefficient of M_{j+1} in row j
        rhs = torch.where(is_last_or_pad[..., None], torch.zeros_like(sp),
                          torch.where(is_first[..., None], 6 * (sp - D0), 6 * (sp - sm)))
        # not-a-knot: continuous third derivative at knot m-2:
        #   h_{m-3} M_{m-1} - (h_{m-3} + h_{m-2}) M_{m-2} + h_{m-2} M_{m-3} = 0
        hm2 = torch.cat([h.new_ones(B, 2), h[:, :K - 2]], 1)  # h_{j-2}
        nak = (j == (m - 1)[:, None]) & (m >= 3)[:, None] & (self.end_bc == "not_a_knot")
        diag = torch.where(nak, hm2, diag)
        lower = torch.where(nak, -(hm2 + hm), lower)
        lower2 = torch.where(nak, hm, zero)                   # coefficient of M_{j-2} in row j
        A = (torch.diag_embed(diag) + torch.diag_embed(lower[:, 1:], -1) + torch.diag_embed(upper[:, :-1], 1)
             + torch.diag_embed(lower2[:, 2:], -2))
        M2 = torch.linalg.solve(A, rhs)                                                 # [B, K, 2]
        self.sig, self.X, self.M2, self.h_sig = sig.contiguous(), X, M2, h
        # derivative at the last retained knot (linear continuation after it)
        mi = (m - 2).clamp(min=0)
        g1 = lambda T: torch.gather(T, 1, mi[:, None, None].expand(B, 1, T.shape[-1]))[:, 0]
        hl = torch.gather(h, 1, mi[:, None])
        Dl = g1(slope) + hl * (2 * g1(M2[:, 1:]) + g1(M2[:, :-1])) / 6
        self.D_last = torch.where((m >= 2)[:, None], Dl, D0.expand(B, 2))
        self.sig_last = torch.gather(sig, 1, (m - 1)[:, None])[:, 0]
        self.X_last = X_last[:, 0]
        # anchors: Gamma_sm at the original knots, residuals, end tangent
        G, d1, _, _ = self._smooth(S)
        self.G_S = G
        self.r = V - G
        # anchor arrays with a zero-length virtual segment after the last vertex (s = S_end is an exact anchor)
        self._Vp = torch.cat([V, V[:, -1:]], 1)
        self._rp = torch.cat([self.r, self.r[:, -1:]], 1)
        self._Hp = torch.cat([self.H, self.H[:, -1:]], 1)
        self._Gp = torch.cat([G, G[:, -1:]], 1)
        self._ellp = torch.cat([self.ell, self.ell.new_zeros(B, 1)], 1)
        t_end = d1[:, -1] / safe_norm(d1[:, -1])[:, None]
        self.T_end = t_end
        self.theta_end = torch.atan2(t_end[:, 1], t_end[:, 0])
        _, d1e, d2e, _ = self._smooth(self.S_end[:, None])
        ge = safe_norm(d1e[:, 0])
        self.kappa_end = (d1e[:, 0, 0] * d2e[:, 0, 1] - d1e[:, 0, 1] * d2e[:, 0, 0]) / (ge * ge * ge)

    def _smooth(self, s: torch.Tensor):
        """Gamma_sm and its first three derivatives at s [B, M] -> ([B,M,2],)*4."""
        B = self.B
        M = s.shape[1]
        s = s.contiguous()
        j = torch.searchsorted(self.sig, s, right=True) - 1
        # the last retained knot itself belongs to the last cubic segment (its end curvature), beyond it: linear
        inside = (s <= self.sig_last[:, None]) & (self.m >= 2)[:, None]
        jc = torch.minimum(j.clamp(min=0), (self.m - 2).clamp(min=0)[:, None])
        g1 = lambda T: torch.gather(T, 1, jc)
        g2 = lambda T: torch.gather(T, 1, jc[..., None].expand(B, M, 2))
        s0, s1 = g1(self.sig), g1(torch.cat([self.sig[:, 1:], self.sig[:, -1:] + 1], 1))
        X0, X1 = g2(self.X), g2(torch.cat([self.X[:, 1:], self.X[:, -1:]], 1))
        M0, M1 = g2(self.M2), g2(torch.cat([self.M2[:, 1:], self.M2[:, -1:]], 1))
        h = (s1 - s0)[..., None]
        a = (s1[..., None] - s[..., None])
        b = (s[..., None] - s0[..., None])
        pos = M0 * a * a * a / (6 * h) + M1 * b * b * b / (6 * h) + (X0 / h - M0 * h / 6) * a + (X1 / h - M1 * h / 6) * b
        d1 = -M0 * a * a / (2 * h) + M1 * b * b / (2 * h) + (X1 - X0) / h - (M1 - M0) * h / 6
        d2 = M0 * a / h + M1 * b / h
        d3 = (M1 - M0) / h
        # linear continuation after the last retained knot (natural end)
        u = (s - self.sig_last[:, None])[..., None]
        pos_l = self.X_last[:, None] + self.D_last[:, None] * u
        d1_l = self.D_last[:, None].expand(B, M, 2)
        z = torch.zeros_like(pos)
        ins = inside[..., None]
        return (torch.where(ins, pos, pos_l), torch.where(ins, d1, d1_l),
                torch.where(ins, d2, z), torch.where(ins, d3, z))

    # ------------------------------------------------------------------------------------------- API
    def knots(self) -> torch.Tensor:
        """Cumulative chord arc length at the draft knots t_k = 0.5 k, k = 0..8 -> [B, 9] (S[:, 0] = 0)."""
        return self.S[:, :N_POSE + 1]

    def seg_speed(self) -> torch.Tensor:
        """Draft speed per 0.5 s pose interval (scorer reference speed) -> [B, 8]."""
        S = self.knots()
        return (S[:, 1:] - S[:, :-1]) / T_POSE

    def s0_dense(self) -> torch.Tensor:
        """Draft arc length at t = 0.1 n -> [B, 41]; exact S_k at n = 5k (piecewise linear, scorer schedule)."""
        S = self.knots()
        k, j = dense_index_split()
        k, j = k.to(S.device), j.to(S.device)
        Sp = torch.cat([S, S[:, -1:]], 1)
        frac = (j.to(S.dtype) / SUB)[None]
        return torch.gather(Sp, 1, k[None].expand(self.B, NK)) + \
            (torch.gather(Sp, 1, (k + 1)[None].expand(self.B, NK)) - torch.gather(Sp, 1, k[None].expand(self.B, NK))) * frac

    def s0(self, t: torch.Tensor) -> torch.Tensor:
        """Draft arc length at arbitrary times t [B, M] or [M] (piecewise linear through (0.5k, S_k); linear
        continuation with the last segment speed after 4 s, 0 before t = 0)."""
        S = self.knots()
        t = as_tensor(t, dtype=S.dtype, device=S.device)
        if t.dim() == 1:
            t = t[None].expand(self.B, -1)
        k = torch.clamp(torch.floor(t / T_POSE), 0, N_POSE - 1).long()
        Sk = torch.gather(S, 1, k)
        Sk1 = torch.gather(S, 1, k + 1)
        return Sk + (Sk1 - Sk) * (torch.clamp(t, min=0.0) - k.to(S.dtype) * T_POSE) / T_POSE

    def extend(self, length, policy: str = "const_curv", v=None) -> "DraftPath":
        """Set the extension beyond S_end (in place, returns self).

        policy 'const_curv': circular arc starting with the tangent of Gamma_sm at S_end and the curvature of the
            last spline segment at S_end (not-a-knot end => the curvature trend of the last two segments; C2 junction),
            clipped to kappa_limit(v) (v = last segment speed ell[-1] / 0.5 unless given); straight if S_end < 2 m.
        policy 'chord': as 'const_curv' but kappa_end = wrap(theta_2 - theta_1) / mean(chord) of the last two retained
            chords (architecture_draft_v0 M4 rule; kept for comparison).
        policy 'straight': kappa = 0.
        length: [B] or float, the extension length that counts as "allowed" (eval flags s > S_end + length).
        """
        B, dt, dev = self.B, self.S.dtype, self.S.device
        self.ext_len = torch.broadcast_to(as_tensor(length, dtype=dt, device=dev), (B,)).clone()
        if policy == "straight":
            self.kappa_ext = torch.zeros(B, dtype=dt, device=dev)
            return self
        if policy == "const_curv":
            kap = self.kappa_end
        elif policy == "chord":
            m = self.m
            i2, i1, i0 = (m - 1).clamp(min=0), (m - 2).clamp(min=0), (m - 3).clamp(min=0)
            gx = lambda i: torch.gather(self.X, 1, i[:, None, None].expand(B, 1, 2))[:, 0]
            c1, c2 = gx(i1) - gx(i0), gx(i2) - gx(i1)
            th1, th2 = torch.atan2(c1[:, 1], c1[:, 0]), torch.atan2(c2[:, 1], c2[:, 0])
            L = 0.5 * (safe_norm(c1) + safe_norm(c2))
            kap = torch.where(m >= 3, _safe_div(wrap_angle(th2 - th1), L), torch.zeros_like(L))
        else:
            raise ValueError(policy)
        if v is None:
            v = self.ell[:, -1] / T_POSE
        v = torch.broadcast_to(as_tensor(v, dtype=dt, device=dev), (B,))
        kap = torch.sign(kap) * torch.minimum(kap.abs(), kappa_limit(v))
        self.kappa_ext = torch.where(self.S_end >= 2.0, kap, torch.zeros_like(kap))
        return self

    def eval(self, s) -> PathEval:
        """Path quantities at arc lengths s [B, M] (or [M], broadcast over the batch)."""
        S = self.S
        s = as_tensor(s, dtype=S.dtype, device=S.device)
        if s.dim() == 1:
            s = s[None].expand(self.B, -1)
        s = s.contiguous()
        B, M = s.shape
        pos_sm, d1, d2, d3 = self._smooth(s)
        g = safe_norm(d1)
        T = d1 / g[..., None]
        cross12 = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
        cross13 = d1[..., 0] * d3[..., 1] - d1[..., 1] * d3[..., 0]
        g3 = g * g * g
        kappa = cross12 / g3
        g_s = (d1 * d2).sum(-1) / g
        kappa_s = cross13 / g3 - 3.0 * kappa * g_s / g
        # original-segment anchors (residual interpolation and draft heading)
        K = self.K
        jo = (torch.searchsorted(S.contiguous(), s, right=True) - 1).clamp(0, K - 1)
        gg = lambda T_: torch.gather(T_, 1, jo)
        gg1 = lambda T_: torch.gather(T_, 1, jo + 1)
        gv = lambda T_: torch.gather(T_, 1, jo[..., None].expand(B, M, 2))
        gv1 = lambda T_: torch.gather(T_, 1, (jo + 1)[..., None].expand(B, M, 2))
        lam = torch.clamp(_safe_div(s - gg(S), gg(self._ellp)), 0.0, 1.0)
        xy = gv(self._Vp) + (pos_sm - gv(self._Gp)) + lam[..., None] * (gv1(self._rp) - gv(self._rp))
        head = gg(self._Hp) + lam * (gg1(self._Hp) - gg(self._Hp))
        # extension beyond the last vertex
        beyond = s > self.S_end[:, None]
        u = torch.clamp(s - self.S_end[:, None], min=0.0)
        ke = self.kappa_ext[:, None]
        th = self.theta_end[:, None] + 0.5 * ke * u
        sc = u * torch.sinc(ke * u / TWO_PI)
        xy_e = self.V[:, -1][:, None] + torch.stack([sc * torch.cos(th), sc * torch.sin(th)], -1)
        th_t = self.theta_end[:, None] + ke * u
        T_e = torch.stack([torch.cos(th_t), torch.sin(th_t)], -1)
        head_e = self.H[:, -1][:, None] + ke * u
        b2 = beyond[..., None]
        xy = torch.where(b2, xy_e, xy)
        T = torch.where(b2, T_e, T)
        head = torch.where(beyond, head_e, head)
        kappa = torch.where(beyond, ke.expand(B, M), kappa)
        g = torch.where(beyond, torch.ones_like(g), g)
        g_s = torch.where(beyond, torch.zeros_like(g_s), g_s)
        kappa_s = torch.where(beyond, torch.zeros_like(kappa_s), kappa_s)
        normal = torch.stack([-T[..., 1], T[..., 0]], -1)
        beyond_ext = s > (self.S_end + self.ext_len)[:, None]
        return PathEval(xy=xy, heading=head, normal=normal, kappa=kappa, tangent=T, g=g, g_s=g_s, kappa_s=kappa_s,
                        beyond=beyond, beyond_ext=beyond_ext)

    def smooth_residual_max(self) -> torch.Tensor:
        """max_k |V_k - Gamma_sm(S_k)| per row [B] (0 up to rounding unless knots were merged)."""
        return safe_norm(self.r).max(1).values
