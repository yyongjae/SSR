"""CK2 candidate variants: constant-da speed profiles along the candidate's own path + rule lateral shifts (decoder basis).

Reference: /home/external-user/ssd/yongjae_refiner/ck/analysis/variants_cf/cf_variants.py (build_variants, z_speed,
W_RULE).  For every shared definition this module returns the same trajectories (tests/test_ck2_variants.py checks
<= 1e-5 m / rad on real candidates and the official labels of a scored subset).

make_variants(tau [T,K,8,3], v0 [T], speeds=(-1.0,-0.5,0.5), lats=(-0.5,0.5), combine='separate', s_on_frac=0.2)
  -> dict(traj f32 [T,K,V,8,3], speed f32 [V], lat f32 [V], valid bool [T,K,V], meta dict)

Variant definitions (refiner/decoder.py decode, no learned parameters)
  speed a [m/s^2] (along the SAME path): free decoder controls Q_1..Q_6 = a (Q_0 = 0, so da ramps 0 -> a over the
      first 0.8 s B-spline span, then stays a; dv(0) = 0).  a < 0: mode 'A', z = atanh(a / A_DEC) (dv <= 0 and
      v = relu(v_draft + dv): a stopped / slow draft cannot go below 0, the variant then saturates toward the draft).
      a > 0: mode 'B', z = atanh(a / A_UP) (dv <= DV_ACC = 2 m/s; positive control points scaled by beta so that
      s(4 s) <= S_8 + L_ext, L_ext = min(5 m, v_end * 1 s); path beyond S_8 = constant-curvature extension).
  lateral D_p [m] (+ = left): e = project_lat(smoothstep_lat(D_p, s_on_frac, 1)) (flat end, e_6 = e_7), w = atanh(e /
      D_MAX), S_L = S_8 (lat_len = S_8 also in mode B), i.e. the offset ramps from s_on = s_on_frac * S_8 to D_p at S_8
      and is held there.  The decoder's curvature projection (alpha <= 1) applies; mode 'A' with z = 0 when a = 0.
  combo (a, D_p): one decode with both controls (mode by the sign of a), exactly as cf_variants' (a, r+-D_p).
  identity (index 0): tau itself (float32 bytes; NOT a decode round trip; decode(z = w = 0) is bitwise tau anyway).

Variant order (V axis); 'separate' is a prefix of 'cross':
  separate: [id] + [(a, 0) for a in speeds] + [(0, D) for D in lats]                       1 + 3 + 2 = 6
  cross   : separate + [(a, D) for a in speeds for D in lats]                              6 + 6 = 12
  names: 'id', 'a-1.0', 'a+0.5', 'l-0.5', 'a-1.0|l+0.5', ...

valid [T,K,V] (decision)
  The decoder disables lateral offsets when the draft is shorter than S_LAT_MIN = 3 m (S_8 < 3 m; e = 0).  A variant
  with a lateral part on such a candidate is returned AS DECODED (= its speed-only parent: the identity for a
  lateral-only variant, the (a, 0) variant for a combo; bitwise up to the sign of zero) and marked valid = False:
  it is a duplicate, so a trainer should mask it (its official label equals the parent's).  Speed variants are always
  valid = True; a speed variant can still be (near-)identical to the identity, e.g. decelerating a stopped draft, a
  mode-B variant with beta = 0, see meta['dev_xy'] (max knot xy distance to the identity) to detect those.

Numerics: decoded in float64 by default (as the reference and as the labels were computed), output float32.
Deterministic (no randomness); torch.no_grad; CPU or GPU (the device of tau).  v0 = ego speed at t0 [m/s]
(= hypot(status[4], status[5])); NaN -> 0 (as the reference worker).  v0 only enters kappa_lim at n = 0 of the
lateral curvature projection and the t0-continuity flag, i.e. lateral variants depend on it.

Path extension of the accelerating (mode-B) variants: ext=... (added 2026-10-08, analysis/accel_ext2)
  A mode-B variant may run past the candidate's end S_8 (s(4 s) <= S_8 + L_ext, L_ext = min(5 m, v_end 1 s)); the path
  there is not the candidate's.  ext selects it (mode-A / identity / lateral-only variants never pass S_8 and are
  bitwise independent of ext; a mode-B pose with s1 <= S_8 is bitwise independent of ext too):
  'const_curv'     (default) the decoder's own rule: circular arc with the candidate's end tangent and end curvature
                   (clipped to kappa_limit, straight if S_8 < 2 m).  Bit-exact with the code before this option.
  'straight'       zero-curvature tangent extension (DraftPath.extend 'straight').
  'centerline_gt'  follow the metric-cache PDM route centerline (ext_lines = [route_centerline_local(mc)] per token).
                   PRIVILEGED: an agent has no metric cache / GT map at inference -> analysis upper bound only.
  'centerline_pred' follow the centerline polylines v2 itself predicts (map head class 'centerline', vector map
                   output; ext_lines = pred_centerlines_local(map_cls, map_pts) per token): deployable.
  Centerline rule (ext_cfg, see EXT_CFG): pick the line nearest to the candidate end V_8 (undirected predicted lines
  are oriented along the candidate end tangent theta_8 and denoised first: spike vertices turning > 60 deg dropped,
  0.5 m resample, 2 m moving average; successors starting within chain_gap of a line's end are chained; straight
  continuation before the first / after the last point), gate |d0| <= max_dist, |psi| <= max_dpsi, V_8 at most 1 m
  before the start / after the end (d0 = signed distance of V_8 from the line, + left; psi = theta_8 - smoothed line
  heading); lines are tried best-first (cost |d0| + 2 |psi|) and one whose curve starts more than max_res from V_8
  (smoothed-normal mismatch, before the exact shift), turns more than max_turn_step per 0.1 m or more than
  max_turn_total over `length` m is skipped (chained successors must start ahead of the line end and within 0.5 m
  laterally: a sideways / backward join made curves double back); then in the line's Frenet frame
      'keep'  (default) d(u) = d0 + tan(psi) (u - u^2 / 2b) for u < b, held after (stay in the lane the candidate
              is in, parallel to the line; the heading difference decays linearly to 0 over b = blend m),
      'decay' d(u) = (d0 + k1 u)(1 - u/b)^2, k1 = tan(psi) + 2 d0 / b (merge onto the line, = decoder
              centerline_continuation used by the refiner draft bank),
  u = arc length along the line from V_8's projection; the curve is shifted to start exactly at V_8, re-parameterised
  by its own chord arc length (step du) and evaluated beyond S_8 with heading H_8 + (curve tangent - its start
  tangent).  Candidates that fail the gate, have S_8 < 2 m or no line use the fallback policy (ext_cfg['fallback'],
  default 'const_curv' -> those rows are bitwise the default).  meta['ext_on'] [T, K] marks rows on a centerline.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from navsim.agents.para_ssr.refiner.decoder import (A_DEC, A_UP, D_MAX, L_EXT_MAX, N_FREE, S_LAT_MIN, T_EXT, decode,
                                                    project_lat, smoothstep_lat)
from navsim.agents.para_ssr.refiner.geometry import T_POSE, DraftPath, PathEval, as_tensor

VERSION = "ck2_variants_v1"
SPEEDS = (-1.0, -0.5, 0.5)
LATS = (-0.5, 0.5)
COMBINES = ("separate", "cross")
S_ON_FRAC = 0.2
MAX_ROWS = 16384          # decoder rows per call (memory bound; results do not depend on it)
EXT_MODES = ("const_curv", "straight", "centerline_gt", "centerline_pred")
EXT_CFG = dict(offset="keep", blend=10.0, du=0.1, length=6.0, fallback="const_curv", min_s_end=2.0, max_res=0.3,
               max_turn_step=0.35, max_turn_total=0.8,      # curve sanity: per 0.1 m step / over `length` [rad]
               centerline_gt=dict(max_dist=4.0, max_dpsi=0.7854, chain_gap=1.5),        # route line: lanes beside it ok
               centerline_pred=dict(max_dist=1.8, max_dpsi=0.5236, chain_gap=1.5))     # per-lane lines: own lane only
PRED_MAP_PC_RANGE = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)   # v2 map head (x_right, y_forward), ParaSSRConfig.map_pc_range
PRED_MAP_CENTERLINE = 2                                    # MAP_CLASS_NAMES = (road, walkway, centerline, crosswalk)


# ----------------------------------------------------------------------------------------------- table / controls
def _check(speeds: Sequence[float], lats: Sequence[float], combine: str, s_on_frac: float):
    if combine not in COMBINES:
        raise ValueError(f"combine must be one of {COMBINES}, got {combine!r}")
    for a in speeds:
        if not (np.isfinite(a) and a != 0.0 and -A_DEC < a < A_UP):
            raise ValueError(f"speed offset {a} must be finite, != 0 and in (-{A_DEC}, {A_UP})")
    for d in lats:
        if not (np.isfinite(d) and d != 0.0 and abs(d) < D_MAX):
            raise ValueError(f"lateral shift {d} must be finite, != 0 and |D_p| < {D_MAX}")
    if len(set(speeds)) != len(speeds) or len(set(lats)) != len(lats):
        raise ValueError("duplicate speed / lateral values")
    if not (0.0 <= s_on_frac < 1.0):
        raise ValueError(f"s_on_frac {s_on_frac} must be in [0, 1)")


def variant_table(speeds: Sequence[float] = SPEEDS, lats: Sequence[float] = LATS,
                  combine: str = "separate") -> List[Tuple[float, float]]:
    """[(a, D_p)] in V order (index 0 = identity (0, 0)); see the module docstring."""
    speeds, lats = tuple(float(a) for a in speeds), tuple(float(d) for d in lats)
    _check(speeds, lats, combine, S_ON_FRAC)
    tab = [(0.0, 0.0)] + [(a, 0.0) for a in speeds] + [(0.0, d) for d in lats]
    if combine == "cross":
        tab += [(a, d) for a in speeds for d in lats]
    return tab


def variant_name(a: float, d: float) -> str:
    parts = ([f"a{a:+.1f}"] if a != 0 else []) + ([f"l{d:+.1f}"] if d != 0 else [])
    return "|".join(parts) if parts else "id"


def z_speed(a: float) -> np.ndarray:
    """[6] float64 raw longitudinal controls with Q_1..Q_6 = a (= cf_variants.z_speed)."""
    if a < 0:
        return np.full(N_FREE, np.arctanh(a / A_DEC))
    if a > 0:
        return np.full(N_FREE, np.arctanh(a / A_UP))
    return np.zeros(N_FREE)


@lru_cache(maxsize=None)
def _w_rule(d: float, s_on_frac: float) -> Tuple[float, ...]:
    e = project_lat(smoothstep_lat(d, s_on_frac, 1.0))
    return tuple(np.arctanh(e / D_MAX).tolist())


def w_rule(d: float, s_on_frac: float = S_ON_FRAC) -> np.ndarray:
    """[6] float64 raw lateral controls of the rule shift D_p (S_L-normalised; = cf_variants.W_RULE)."""
    if d == 0:
        return np.zeros(N_FREE)
    return np.array(_w_rule(float(d), float(s_on_frac)), np.float64)


# ----------------------------------------------------------------------------------------------- extension: lines
def route_centerline_local(metric_cache, back: float = 20.0, ahead: float = 160.0) -> np.ndarray:
    """metric_cache.centerline (PDMPath, global, route direction) -> [P, 2] float64 in the N frame (rear axle at t0),
    cropped to arc lengths [s_ego - back, s_ego + ahead] around the vertex nearest to the ego.  PRIVILEGED (GT)."""
    arr = np.asarray(metric_cache.centerline._states_se2_array, np.float64)[:, :2]
    ra = metric_cache.ego_state.rear_axle
    c, s = np.cos(ra.heading), np.sin(ra.heading)
    d = arr - np.array([ra.x, ra.y])
    loc = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
    prog = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(loc, axis=0).T))])
    se = prog[int(np.argmin(np.hypot(*loc.T)))]
    keep = (prog >= se - back) & (prog <= se + ahead)
    return loc[keep]


def pred_centerlines_local(map_cls, map_pts, thr: float = 0.3, pc_range=PRED_MAP_PC_RANGE,
                           cls: int = PRED_MAP_CENTERLINE) -> List[np.ndarray]:
    """v2 vector-map output (final layer; map_cls [Q, 4] logits, map_pts [Q, P, 2] normalised to pc_range in the
    (x_right, y_forward) convention) -> list of [P, 2] float64 N-frame (x forward, y left) centerline polylines with
    sigmoid(logit_centerline) >= thr (undirected: the head matches either point order)."""
    lg = np.asarray(map_cls, np.float64)[:, cls]
    pts = np.asarray(map_pts, np.float64)
    x0, y0, x1, y1 = float(pc_range[0]), float(pc_range[1]), float(pc_range[3]), float(pc_range[4])
    out = []
    for q in np.flatnonzero(1.0 / (1.0 + np.exp(-lg)) >= thr):
        xr = pts[q, :, 0] * (x1 - x0) + x0
        yf = pts[q, :, 1] * (y1 - y0) + y0
        out.append(np.stack([yf, -xr], -1))
    return out


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _clean(L: np.ndarray, denoise: bool = False, step: float = 0.5, win: float = 2.0,
           max_turn: float = 1.0472) -> np.ndarray:
    """Finite points, consecutive duplicates removed.  denoise (predicted lines): repeatedly drop the vertex with the
    largest turn while it exceeds max_turn (spikes / a reversed first point), then resample at `step` m by arc length
    and moving-average the interior points over `win` m (end points kept)."""
    L = np.asarray(L, np.float64)[:, :2]
    L = L[np.isfinite(L).all(1)]
    if len(L) < 2:
        return L
    L = L[np.concatenate([[True], np.hypot(*np.diff(L, axis=0).T) > 1e-6])]
    if not denoise or len(L) < 3:
        return L
    while len(L) >= 3:
        h = np.arctan2(*np.diff(L, axis=0).T[::-1])
        turn = np.abs(_wrap(h[1:] - h[:-1]))
        i = int(np.argmax(turn))
        if turn[i] <= max_turn:
            break
        L = np.delete(L, i + 1, axis=0)
    A = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(L, axis=0).T))])
    if A[-1] < 2 * step:
        return L
    a = np.linspace(0.0, A[-1], int(np.ceil(A[-1] / step)) + 1)
    R = np.stack([np.interp(a, A, L[:, 0]), np.interp(a, A, L[:, 1])], -1)
    w = int(round(win / step))
    if w >= 2 and len(R) > w + 1:
        h = w // 2
        Rp = np.pad(R, ((h, w - 1 - h), (0, 0)), mode="edge")
        ker = np.ones(w) / w
        Rs = np.stack([np.convolve(Rp[:, k], ker, mode="valid") for k in (0, 1)], -1)
        Rs[0], Rs[-1] = R[0], R[-1]
        R = Rs
    return R[np.concatenate([[True], np.hypot(*np.diff(R, axis=0).T) > 1e-6])]


def _project(L: np.ndarray, p: np.ndarray):
    """p onto polyline L [P, 2] (P >= 2), straight continuation before the start / after the end ->
    (s_p (may be < 0 or > length), d0 [m] (+ left; the true signed distance when p is alongside the line),
    seg idx, metres before the start, metres after the end)."""
    seg = np.diff(L, axis=0)
    sl = np.hypot(*seg.T)
    Sc = np.concatenate([[0.0], np.cumsum(sl)])
    tr = ((p - L[:-1]) * seg).sum(1) / sl ** 2
    t = np.clip(tr, 0.0, 1.0)
    proj = L[:-1] + t[:, None] * seg
    dist = np.hypot(*(proj - p).T)
    i = int(np.argmin(dist))
    r = p - L[i]
    perp = float((seg[i, 0] * r[1] - seg[i, 1] * r[0]) / sl[i])
    if i == len(seg) - 1 and tr[i] > 1.0:
        s_p, d0 = float(Sc[-1] + (tr[i] - 1.0) * sl[i]), perp
    elif i == 0 and tr[0] < 0.0:
        s_p, d0 = float(tr[0] * sl[0]), perp
    else:
        s_p, d0 = float(Sc[i] + t[i] * sl[i]), float(np.copysign(dist[i], perp))
    return s_p, d0, i, max(0.0, -s_p), max(0.0, s_p - float(Sc[-1]))


def _vertex_heading(L: np.ndarray) -> np.ndarray:
    """Unwrapped smoothed heading at the vertices (circular mean of the adjacent segment directions)."""
    seg = np.diff(L, axis=0)
    th = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    hv = np.concatenate([th[:1], 0.5 * (th[:-1] + th[1:]), th[-1:]])
    return hv


def _line_candidates(lines, directed: bool, denoise: bool):
    """[(line index, oriented cleaned polyline)] (both orientations for undirected lines)."""
    out = []
    for j, L in enumerate(lines):
        L = _clean(L, denoise=denoise)
        if len(L) < 2:
            continue
        for Lo in ((L,) if directed else (L, L[::-1])):
            out.append((j, Lo))
    return out


def _select_lines(cand, p: np.ndarray, theta: float, max_dist: float, max_dpsi: float):
    """Lines passing the gate for an end point p with tangent theta, best first -> [(cost, j, Lo, d0, psi)]."""
    ok = []
    for j, Lo in cand:
        s_p, d0, _, before, after = _project(Lo, p)
        if before > 1.0 or after > 1.0:              # end point not alongside this line
            continue
        Sc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Lo, axis=0).T))])
        th_c = float(np.interp(s_p, Sc, _vertex_heading(Lo)))
        psi = float(_wrap(theta - th_c))
        if abs(d0) > max_dist or abs(psi) > max_dpsi:
            continue
        ok.append((abs(d0) + 2.0 * abs(psi), j, Lo, d0, psi))
    ok.sort(key=lambda x: (x[0], x[1]))
    return ok


def _chain(cand, j: int, Lo: np.ndarray, s_p: float, chain_gap: float, need: float,
           max_lat: float = 0.5) -> np.ndarray:
    """Append successor lines while less than `need` m of line remain after s_p (at most 3).  A successor must start
    ahead of the current end (along the end heading in [-0.2, chain_gap] m), within max_lat m laterally, with its
    heading within 30 deg; its points not ahead of the current end (along <= 0.05 m) are dropped."""
    used = {j}
    rem = float(np.hypot(*np.diff(Lo, axis=0).T).sum()) - s_p
    for _ in range(3):
        if rem >= need:
            break
        end, hd = Lo[-1], np.arctan2(*(Lo[-1] - Lo[-2])[::-1])
        tv = np.array([np.cos(hd), np.sin(hd)])
        nxt = None
        for k, Lk in cand:
            if k in used:
                continue
            rel = Lk[0] - end
            along, lat = float(rel @ tv), float(tv[0] * rel[1] - tv[1] * rel[0])
            if not (-0.2 <= along <= chain_gap) or abs(lat) > max_lat:
                continue
            if abs(_wrap(np.arctan2(*(Lk[1] - Lk[0])[::-1]) - hd)) > 0.5236:
                continue
            gap = float(np.hypot(*rel))
            if nxt is None or gap < nxt[0]:
                nxt = (gap, k, Lk)
        if nxt is None:
            break
        _, k, Lk = nxt
        used.add(k)
        Lk = Lk[(Lk - end) @ tv > 0.05]
        if len(Lk) == 0:
            break
        rem += float(np.hypot(*np.diff(np.vstack([end, Lk]), axis=0).T).sum())
        Lo = np.vstack([Lo, Lk])
    return Lo


def ext_curve(L: np.ndarray, p: np.ndarray, theta: float, d0: float, psi: float, offset: str = "keep",
              blend: float = 10.0, du: float = 0.1, length: float = 6.0, fine: float = 0.05):
    """Extension curve from p (tangent theta) along the oriented polyline L (see the module docstring) ->
    (xy [Q, 2], th [Q] (unwrapped tangent angle, th[0] within pi of theta), kappa [Q], junction residual [m]),
    Q = round(length / du) + 1 samples at chord arc length a = 0, du, .., length (float64)."""
    L = _clean(L)
    seg = np.diff(L, axis=0)
    sl = np.hypot(*seg.T)
    Sc = np.concatenate([[0.0], np.cumsum(sl)])
    hv = _vertex_heading(L)
    s_p = _project(L, p)[0]
    ug = np.arange(0.0, length + 3.0 + 1e-9, fine)
    sq = s_p + ug
    # base point: polyline position, straight continuation before the first / after the last vertex
    xb = np.interp(sq, Sc, L[:, 0])
    yb = np.interp(sq, Sc, L[:, 1])
    over = np.clip(sq - Sc[-1], 0.0, None)
    under = np.clip(sq, None, 0.0)
    xb = xb + over * np.cos(hv[-1]) + under * np.cos(hv[0])
    yb = yb + over * np.sin(hv[-1]) + under * np.sin(hv[0])
    th = np.interp(sq, Sc, hv)
    m0 = float(np.tan(psi))
    b = float(blend)
    if offset == "keep":
        d = np.where(ug < b, d0 + m0 * (ug - ug * ug / (2 * b)), d0 + m0 * b / 2)
    elif offset == "decay":
        k1 = m0 + 2.0 * d0 / b
        d = np.where(ug < b, (d0 + k1 * ug) * (1 - ug / b) ** 2, 0.0)
    else:
        raise ValueError(f"offset {offset!r}")
    P = np.stack([xb - d * np.sin(th), yb + d * np.cos(th)], -1)
    res = float(np.hypot(*(P[0] - p)))
    P = P - P[0] + p
    A = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    a = np.arange(int(round(length / du)) + 1) * du
    xy = np.stack([np.interp(a, A, P[:, 0]), np.interp(a, A, P[:, 1])], -1)
    dP = np.diff(P, axis=0)
    thm = np.unwrap(np.arctan2(dP[:, 1], dP[:, 0]))                  # at segment midpoints
    Am = 0.5 * (A[:-1] + A[1:])
    tha = np.interp(a, Am, thm)
    tha = tha + 2 * np.pi * np.round((theta - tha[0]) / (2 * np.pi))
    kap = np.gradient(tha, du)
    return xy, tha, kap, res


class CurveExtPath(DraftPath):
    """DraftPath whose extension beyond S_end follows a sampled curve on rows `on` (set_curve); other rows keep the
    policy of extend().  Only eval() differs: for s <= S_end it is DraftPath.eval bitwise."""

    _c_on = None

    def set_curve(self, on, xy, th, kappa, du: float) -> "CurveExtPath":
        dt, dev = self.S.dtype, self.S.device
        self._c_on = as_tensor(on, device=dev).bool().reshape(self.B)
        self._c_xy = as_tensor(xy, dtype=dt, device=dev)                   # [B, Q, 2]
        self._c_th = as_tensor(th, dtype=dt, device=dev)                   # [B, Q]
        self._c_k = as_tensor(kappa, dtype=dt, device=dev)                 # [B, Q]
        self._c_du = float(du)
        return self

    def eval(self, s) -> PathEval:
        pe = super().eval(s)
        if self._c_on is None or not bool(self._c_on.any()):
            return pe
        s = as_tensor(s, dtype=self.S.dtype, device=self.S.device)
        if s.dim() == 1:
            s = s[None].expand(self.B, -1)
        B, M = s.shape
        Q = self._c_xy.shape[1]
        q = torch.clamp(s - self.S_end[:, None], min=0.0) / self._c_du
        i0 = torch.clamp(torch.floor(q), 0, Q - 2).long()
        lam = q - i0.to(q.dtype)                                           # > 1 only past the last sample
        g2 = lambda T_, i: torch.gather(T_, 1, i[..., None].expand(B, M, 2))
        xy0, xy1 = g2(self._c_xy, i0), g2(self._c_xy, i0 + 1)
        xy_c = xy0 + lam[..., None] * (xy1 - xy0)
        lc = torch.clamp(lam, max=1.0)
        th0, th1 = torch.gather(self._c_th, 1, i0), torch.gather(self._c_th, 1, i0 + 1)
        th_c = th0 + lc * (th1 - th0)
        k0, k1 = torch.gather(self._c_k, 1, i0), torch.gather(self._c_k, 1, i0 + 1)
        k_c = torch.where(lam > 1.0, torch.zeros_like(lam), k0 + lc * (k1 - k0))
        use = pe.beyond & self._c_on[:, None]
        u2 = use[..., None]
        T_c = torch.stack([torch.cos(th_c), torch.sin(th_c)], -1)
        T = torch.where(u2, T_c, pe.tangent)
        head = torch.where(use, self.H[:, -1][:, None] + (th_c - self._c_th[:, :1]), pe.heading)
        return PathEval(xy=torch.where(u2, xy_c, pe.xy), heading=head, normal=torch.stack([-T[..., 1], T[..., 0]], -1),
                        kappa=torch.where(use, k_c, pe.kappa), tangent=T, g=pe.g, g_s=pe.g_s, kappa_s=pe.kappa_s,
                        beyond=pe.beyond, beyond_ext=pe.beyond_ext)


def _ext_cfg(ext: str, ext_cfg) -> Dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in EXT_CFG.items()}
    for k, v in (ext_cfg or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    if cfg["fallback"] not in ("const_curv", "straight"):
        raise ValueError(f"fallback must be const_curv / straight, got {cfg['fallback']!r}")
    if cfg["offset"] not in ("keep", "decay"):
        raise ValueError(f"offset must be keep / decay, got {cfg['offset']!r}")
    return cfg


def centerline_curves(bp: DraftPath, ext_lines, T: int, K: int, ext: str, cfg: Dict) -> Dict[str, np.ndarray]:
    """Per candidate (N = T K rows of the base path bp) extension curves for ext 'centerline_*' (module docstring).
    Lines are tried best-first; one whose curve does not start within max_res of V_8 (before the exact shift) is
    skipped."""
    N = T * K
    Q = int(round(cfg["length"] / cfg["du"])) + 1
    gc = cfg[ext]
    out = dict(on=np.zeros(N, bool), xy=np.zeros((N, Q, 2)), th=np.zeros((N, Q)), kappa=np.zeros((N, Q)),
               d0=np.full(N, np.nan), psi=np.full(N, np.nan), res=np.full(N, np.nan), line=np.full(N, -1, np.int32))
    if ext_lines is None or len(ext_lines) != T:
        raise ValueError(f"ext={ext!r} needs ext_lines: a list of T={T} line lists")
    V8 = bp.V[:, -1].double().cpu().numpy()
    th8 = bp.theta_end.double().cpu().numpy()
    Se = bp.S_end.double().cpu().numpy()
    directed = ext == "centerline_gt"
    for t in range(T):
        lines = ext_lines[t]
        if lines is None:
            continue
        if isinstance(lines, np.ndarray) and lines.ndim == 2:
            lines = [lines]
        cand = _line_candidates(lines, directed, denoise=not directed)
        if not cand:
            continue
        for n in range(t * K, (t + 1) * K):
            if Se[n] < cfg["min_s_end"]:
                continue
            for _, j, Lo, d0, psi in _select_lines(cand, V8[n], float(th8[n]), gc["max_dist"], gc["max_dpsi"]):
                s_p = _project(Lo, V8[n])[0]
                Lc = _chain(cand, j, Lo, s_p, gc["chain_gap"], need=cfg["length"] + 1.0)
                xy, th, kap, res = ext_curve(Lc, V8[n], float(th8[n]), d0, psi, cfg["offset"], cfg["blend"],
                                             cfg["du"], cfg["length"])
                dth = np.abs(np.diff(th))
                if res > cfg["max_res"] or dth.max() > cfg["max_turn_step"] or \
                        np.abs(th - th[0]).max() > cfg["max_turn_total"]:
                    continue
                out["on"][n] = True
                out["xy"][n], out["th"][n], out["kappa"][n] = xy, th, kap
                out["d0"][n], out["psi"][n], out["res"][n], out["line"][n] = d0, psi, res, j
                break
    return out


def _ext_path(rows: torch.Tensor, ext: str, cfg: Dict, curves: Optional[Dict] = None) -> DraftPath:
    """Path object for decode(path=...) of mode-B rows; reproduces decode's own L_ext / extend call exactly."""
    p = CurveExtPath(rows) if curves is not None else DraftPath(rows)
    S = p.knots()
    S8 = S[:, -1]
    u_end = (S[:, -1] - S[:, -2]) / T_POSE
    L_ext = torch.minimum(torch.full_like(S8, L_EXT_MAX), u_end * T_EXT)
    p.extend(L_ext, "straight" if ext == "straight" else cfg["fallback"], v=u_end)
    if curves is not None:
        p.set_curve(curves["on"], curves["xy"], curves["th"], curves["kappa"], cfg["du"])
    return p


# ----------------------------------------------------------------------------------------------- main
@torch.no_grad()
def make_variants(tau: torch.Tensor, v0, speeds: Sequence[float] = SPEEDS, lats: Sequence[float] = LATS,
                  combine: str = "separate", s_on_frac: float = S_ON_FRAC, *,
                  compute_dtype: torch.dtype = torch.float64, max_rows: int = MAX_ROWS, ext: str = "const_curv",
                  ext_lines=None, ext_cfg: Optional[Dict] = None) -> Dict:
    """Speed / lateral variants of every candidate.

    Args:
      tau  : [T, K, 8, 3] candidate poses (N frame, t = 0.5 .. 4 s), floating tensor (float32 expected).
      v0   : [T] ego speed at t0 [m/s] (also accepted: scalar, [T, K]); NaN -> 0.
      speeds, lats, combine, s_on_frac: see the module docstring.
      compute_dtype: decoder dtype (float64 = the reference / label computation; float32 is faster on GPU but only
             approximately equal).
      max_rows: decoder rows per call (memory only).
      ext  : path beyond S_8 for the mode-B (a > 0) variants, one of EXT_MODES (module docstring); default
             'const_curv' = the decoder's own extension (bit-exact with the code before this option).
      ext_lines: 'centerline_*' only: list of T entries, each a list of [P, 2] N-frame polylines (or None)
             (route_centerline_local / pred_centerlines_local).
      ext_cfg: overrides of EXT_CFG (offset, blend, du, length, fallback, min_s_end, per-mode gates).
    Returns dict:
      traj  f32 [T, K, V, 8, 3]   (traj[:, :, 0] == tau.float() bitwise)
      speed f32 [V], lat f32 [V]  (a [m/s^2], D_p [m]; 0 = none)
      valid bool [T, K, V]        (False: lateral part requested on a candidate with S_8 < 3 m -> duplicate)
      meta  dict: version, names [V], combine, speeds, lats, s_on_frac, mode [V] ('id' / 'A' / 'B'), z [V, 6], w [V, 6]
            (float64 numpy), S8 [T, K], lat_on [T, K] (bool), and per variant [T, K, V] (identity: 1 / 1 / 0 / 0 / 0 / 0):
            alpha (lateral projection), beta (mode-B scale), ds4 (s(4 s) - S_8 [m]), d_end (lateral offset at 4 s [m]),
            ext_m (path length used beyond S_8 [m]), dev_xy (max knot xy distance to the identity [m]);
            ext (str), ext_on bool [T, K] (row on a centerline curve; False for const_curv / straight), and for
            'centerline_*' ext_d0 / ext_psi / ext_res f32 [T, K] (offset [m], heading difference [rad], junction
            residual [m] of the chosen line; NaN when not on a line), ext_cfg (dict).
    """
    if not torch.is_tensor(tau):
        tau = torch.as_tensor(tau)
    if not tau.is_floating_point() or tau.dim() != 4 or tuple(tau.shape[2:]) != (8, 3):
        raise ValueError(f"tau must be a floating [T, K, 8, 3] tensor, got {tuple(tau.shape)} {tau.dtype}")
    speeds, lats = tuple(float(a) for a in speeds), tuple(float(d) for d in lats)
    _check(speeds, lats, combine, s_on_frac)
    if ext not in EXT_MODES:
        raise ValueError(f"ext must be one of {EXT_MODES}, got {ext!r}")
    tab = variant_table(speeds, lats, combine)
    T, K = tau.shape[:2]
    V, N = len(tab), T * K
    dev = tau.device
    cd = compute_dtype

    v0 = torch.as_tensor(v0, dtype=cd, device=dev)
    if v0.dim() == 1:
        v0 = v0[:, None]
    v0 = torch.nan_to_num(torch.broadcast_to(v0, (T, K)), nan=0.0).reshape(N)

    flat = tau.reshape(N, 8, 3).to(cd)
    bp = DraftPath(flat) if N else None
    S8 = bp.knots()[:, -1] if N else torch.zeros(0, dtype=cd, device=dev)
    lat_on = S8 >= S_LAT_MIN
    cfg = _ext_cfg(ext, ext_cfg)
    curves = None
    if ext.startswith("centerline") and N and any(a > 0 for a, _ in tab):
        curves = centerline_curves(bp, ext_lines, T, K, ext, cfg)

    traj = torch.empty((N, V, 8, 3), dtype=torch.float32, device=dev)
    traj[:, 0] = tau.reshape(N, 8, 3).to(torch.float32)
    fl = {k: torch.zeros((N, V), dtype=torch.float32, device=dev) for k in ("alpha", "beta", "ds4", "d_end", "ext_m")}
    fl["alpha"][:, 0] = 1.0
    fl["beta"][:, 0] = 1.0
    zt = np.stack([z_speed(a) for a, _ in tab])                        # [V, 6] float64
    wt = np.stack([w_rule(d, s_on_frac) for _, d in tab])
    modes = ["id"] + ["B" if a > 0 else "A" for a, _ in tab[1:]]

    for mode in ("A", "B"):
        vs = [i for i in range(1, V) if modes[i] == mode]
        if not vs or N == 0:
            continue
        nv = len(vs)
        z_g = torch.as_tensor(zt[vs], dtype=cd, device=dev)               # [nv, 6]
        w_g = torch.as_tensor(wt[vs], dtype=cd, device=dev)
        vs_t = torch.as_tensor(vs, device=dev)
        rows_per = max(1, max_rows // nv)                                 # candidates per decode call
        for n0 in range(0, N, rows_per):
            n1 = min(N, n0 + rows_per)
            m = n1 - n0
            rows = flat[n0:n1].repeat_interleave(nv, 0)
            path = None
            if mode == "B" and ext != "const_curv":
                cv = None if curves is None else {
                    k: np.repeat(curves[k][n0:n1], nv, axis=0) for k in ("on", "xy", "th", "kappa")}
                path = _ext_path(rows, ext, cfg, cv)
            o = decode(rows, z_g.repeat(m, 1), w_g.repeat(m, 1),
                       v0=v0[n0:n1].repeat_interleave(nv, 0), mode=mode,
                       lat_len=S8[n0:n1].repeat_interleave(nv, 0), path=path)
            traj[n0:n1, vs_t] = o["traj"].reshape(m, nv, 8, 3).to(torch.float32)
            f = o["flags"]
            vals = {"alpha": f["alpha"], "beta": f["beta"], "ds4": o["s"][:, -1] - o["s0"][:, -1],
                    "d_end": o["d"][:, -1], "ext_m": f["ext_m"]}
            for k, x in vals.items():
                fl[k][n0:n1, vs_t] = x.reshape(m, nv).to(torch.float32)

    has_lat = torch.as_tensor([d != 0 for _, d in tab], device=dev)
    valid = ~(has_lat[None, :] & ~lat_on[:, None])
    dev_xy = (traj[..., :2] - traj[:, :1, :, :2]).norm(dim=-1).amax(-1)   # [N, V]

    meta = dict(version=VERSION, names=[variant_name(a, d) for a, d in tab], combine=combine, speeds=speeds,
                lats=lats, s_on_frac=float(s_on_frac), mode=modes, z=zt, w=wt, compute_dtype=str(cd),
                S8=S8.reshape(T, K).to(torch.float32), lat_on=lat_on.reshape(T, K),
                dev_xy=dev_xy.reshape(T, K, V), **{k: x.reshape(T, K, V) for k, x in fl.items()})
    meta["ext"] = ext
    meta["ext_on"] = torch.zeros((T, K), dtype=torch.bool, device=dev)
    if ext.startswith("centerline"):
        meta["ext_cfg"] = cfg
        nanf = torch.full((T, K), float("nan"), dtype=torch.float32, device=dev)
        meta.update(ext_d0=nanf.clone(), ext_psi=nanf.clone(), ext_res=nanf.clone())
        if curves is not None:
            meta["ext_on"] = torch.as_tensor(curves["on"], device=dev).reshape(T, K)
            for k in ("d0", "psi", "res"):
                meta[f"ext_{k}"] = torch.as_tensor(curves[k], dtype=torch.float32, device=dev).reshape(T, K)
    return dict(traj=traj.reshape(T, K, V, 8, 3),
                speed=torch.as_tensor([a for a, _ in tab], dtype=torch.float32, device=dev),
                lat=torch.as_tensor([d for _, d in tab], dtype=torch.float32, device=dev),
                valid=valid.reshape(T, K, V), meta=meta)


def make_variants_np(tau: np.ndarray, v0, **kw) -> Dict:
    """numpy convenience wrapper (CPU).  tau [T, K, 8, 3] with v0 [T] (or scalar / [T, K]), or one token [K, 8, 3]
    with a scalar v0 (the T axis is then dropped from every output).  Same keys as make_variants, numpy arrays."""
    tau = np.asarray(tau)
    v0 = np.asarray(v0, np.float64)
    single = tau.ndim == 3
    if single:
        tau = tau[None]
        v0 = v0.reshape(1, -1)          # scalar / [1] -> [1, 1]; per-candidate [K] -> [1, K]
        if kw.get("ext_lines") is not None:
            kw = dict(kw, ext_lines=[kw["ext_lines"]])   # one token: its line list
    t = torch.from_numpy(np.ascontiguousarray(tau, dtype=np.float64 if tau.dtype == np.float64 else np.float32))
    out = make_variants(t, torch.from_numpy(v0), **kw)

    def cv(x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
            return x[0] if single and x.ndim >= 2 and x.shape[0] == 1 else x
        return x

    meta = {k: (cv(v) if torch.is_tensor(v) else v) for k, v in out["meta"].items()}
    res = {k: cv(v) for k, v in out.items() if k != "meta"}
    res["speed"], res["lat"] = out["speed"].cpu().numpy(), out["lat"].cpu().numpy()
    res["meta"] = meta
    return res
