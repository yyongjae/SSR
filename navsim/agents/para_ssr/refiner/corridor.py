"""M3 scene reading for the stage-T refiner (IMPL_SPEC §3.8): corridor sampler (M3a) + global tokens (M3b).  torch.

Frames / grids (IMPL_SPEC §2)
  N frame: NAVSIM ego frame at t0 (rear-axle origin, x forward, y left, heading CCW from +x; metres, seconds).
  S grid : feature maps [T, C, 50, 100]; row r -> x = (r + 0.5) * 0.64 m, col c -> y_left = 32 - (c + 0.5) * 0.64 m.
           grid_sample coordinates (align_corners=False, grid[..., 0] = column, grid[..., 1] = row):
               gx = -y_left / 32,   gy = x / 16 - 1      (critic_impl.md issue 12)
           bilinear, padding_mode='zeros' (points off the grid read 0 -- NOT the border value -- and in_grid = 0).

M3a corridor (per draft, no gradient: the draft is an input)
  Path      : DraftPath(tau0) (geometry.py: the same C2 path as the M4 decoder, chord-arc-length parameter s).
  Look-ahead: S_look = S_8 + max(8 m, v_end * 1 s), v_end = (S_8 - S_7) / 0.5 s (last-segment draft speed).
              Beyond S_8 the path is extended by DraftPath.extend(S_look - S_8, 'const_curv', v=v_end): a circular arc
              with the spline's end tangent and end curvature clipped to kappa_limit(v_end) = min(0.95/v, 4.89/v^2,
              0.213); straight if S_8 < 2 m (flag near_stop); flag ext_clipped if the clip was active.
  Stations  : N_ST = 48, spacing ds = max(1 m, S_look / 48), s_j = (j + 0.5) ds, j = 0..47 (station j is the centre of
              the arc-length bin [j ds, (j + 1) ds]; the bins cover [0, 48 ds] >= [0, S_look]).
  Lateral   : N_LAT = 17 offsets d_i = -4.8 + 0.6 i m, i = 0..16 (i = 0 is 4.8 m to the RIGHT, i = 8 on the path,
              i = 16 is 4.8 m to the LEFT), along the left unit normal of the smooth path:
              q_ij = Gamma(s_j) + d_i n(s_j).
  Channels  : [64 feature channels | 6 geometric channels] -> X [B, 70, 48, 17] (B = drafts, dims (channel, station,
              lateral)); geometric channels, in this order (GEO_CHANNELS):
                0 in_grid     1 if q_ij lies in the S grid extent (0 <= x <= 32, |y| <= 32) else 0
                1 path_valid  1 if s_j <= S_8 (inside the draft) else 0 (extrapolated)
                2 s_j / 48 m
                3 d_i / 4.8 m
                4 t_d(s_j) / 4 s clipped to [0, 1]; t_d = FIRST time the draft reaches arc length s (inverse of the
                  piecewise-linear schedule through (0.5 k, S_k); stand-still segments are skipped); beyond S_8:
                  4 + (s - S_8) / max(v_end, 0.1) -> 1 after the clip
                5 v_d(s_j) / 15 m/s; v_d = speed (S_k - S_{k-1}) / 0.5 of the draft segment containing that first
                  arrival (s <= 0: first segment; beyond S_8: v_end)
M3b global tokens
  avg_pool2d(F, 5) of the 64-channel map [T, 64, 50, 100] -> [T, 64, 10, 20] -> 200 tokens [T, 200, 64] in row-major
  order (token m = 20 * row + col, row = x bin of 3.2 m from the ego, col = y_left bin from +32 m).  The learned
  positional embedding is added in refiner_net.py.

Token batching: features are per TOKEN [T, ...]; drafts are [T, K, 8, 3] (K drafts share the token's features).
Sampling is done per token with the K drafts' points concatenated, so the feature map is never replicated.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .adapters import BEV_H, BEV_W, CELL
from .geometry import N_POSE, T_POSE, DraftPath, kappa_limit

# ----------------------------------------------------------------------------------------------- constants
N_ST = 48                 # stations along the path
N_LAT = 17                # lateral offsets
LAT_MAX = 4.8             # [m]
DS_MIN = 1.0              # minimum station spacing [m]
LOOK_MIN = 8.0            # minimum look-ahead beyond S_8 [m]
T_LOOK = 1.0              # look-ahead time beyond S_8 [s]
NEAR_STOP_S = 2.0         # straight extension below this draft length [m] (DraftPath.extend rule)
S_SCALE, T_SCALE, V_SCALE = 48.0, 4.0, 15.0
V_EPS = 0.1               # [m/s] floor of v_end in the arrival time beyond S_8
POOL = 5                  # global token pooling (5 x 5 cells = 3.2 m)
N_GLOBAL = (BEV_H // POOL) * (BEV_W // POOL)    # 200
X_MAX = BEV_H * CELL      # 32 m
Y_MAX = BEV_W * CELL / 2  # 32 m
GEO_CHANNELS = ("in_grid", "path_valid", "s_over_48", "d_over_4p8", "t_arr_over_4", "v_arr_over_15")
N_GEO = len(GEO_CHANNELS)


def lateral_offsets(device=None, dtype=torch.float32) -> torch.Tensor:
    """d_i = -4.8 + 0.6 i, i = 0..16 [m] (left positive)."""
    return torch.linspace(-LAT_MAX, LAT_MAX, N_LAT, device=device, dtype=dtype)


def s_grid_normalized(xy: torch.Tensor) -> torch.Tensor:
    """N-frame points [..., 2] (x, y_left) -> grid_sample coordinates [..., 2] (gx = column, gy = row) of the S grid."""
    return torch.stack([-xy[..., 1] / Y_MAX, xy[..., 0] / (X_MAX / 2) - 1.0], -1)


def in_s_grid(xy: torch.Tensor) -> torch.Tensor:
    """True where the point lies inside the S-grid extent (0 <= x <= 32, -32 <= y <= 32)."""
    x, y = xy[..., 0], xy[..., 1]
    return (x >= 0) & (x <= X_MAX) & (y >= -Y_MAX) & (y <= Y_MAX) & torch.isfinite(x) & torch.isfinite(y)


def sample_s_grid(feat: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Bilinear read of feat [T, C, 50, 100] at N-frame points xy [T, P, Q, 2] -> [T, C, P, Q] (zeros padding)."""
    grid = s_grid_normalized(xy).to(feat.dtype)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# ----------------------------------------------------------------------------------------------- draft timing
def arrival_time_speed(S: torch.Tensor, s: torch.Tensor, v_end: torch.Tensor):
    """First-arrival time t_d(s) [s] and segment speed v_d(s) [m/s] of the draft schedule through (0.5 k, S_k).

    S [B, 9] knot arc lengths (non-decreasing), s [B, M], v_end [B].  Returns (t [B, M], v [B, M]).
    """
    B, K = S.shape
    u = (S[:, 1:] - S[:, :-1]) / T_POSE                                           # [B, 8]
    i = torch.searchsorted(S.contiguous(), s.contiguous(), right=False)           # S[i-1] < s <= S[i]
    inside = (i >= 1) & (i <= K - 1)
    ic = i.clamp(1, K - 1)
    S0 = torch.gather(S, 1, ic - 1)
    S1 = torch.gather(S, 1, ic)
    den = torch.where(S1 > S0, S1 - S0, torch.ones_like(S1))
    t_in = T_POSE * (ic - 1).to(S.dtype) + T_POSE * (s - S0) / den
    v_in = torch.gather(u, 1, ic - 1)
    S8 = S[:, -1:]
    beyond = s > S8
    t_out = N_POSE * T_POSE + (s - S8) / torch.clamp(v_end, min=V_EPS)[:, None]
    t = torch.where(inside, t_in, torch.where(beyond, t_out, torch.zeros_like(s)))
    v = torch.where(inside, v_in, torch.where(beyond, v_end[:, None].expand_as(s), u[:, :1].expand_as(s)))
    return t, v


# ----------------------------------------------------------------------------------------------- corridor geometry
@dataclass
class CorridorGeom:
    points: torch.Tensor       # [B, 48, 17, 2] N-frame sample points q_ij
    geo: torch.Tensor          # [B, 6, 48, 17] geometric channels (GEO_CHANNELS order)
    s: torch.Tensor            # [B, 48] station arc lengths [m]
    S: torch.Tensor            # [B, 9] draft knot arc lengths
    S_look: torch.Tensor       # [B]
    v_end: torch.Tensor        # [B]
    near_stop: torch.Tensor    # [B] bool (S_8 < 2 m: straight extension)
    ext_clipped: torch.Tensor  # [B] bool (extension curvature clipped by kappa_limit(v_end))
    kappa_knots: torch.Tensor  # [B, 8] smooth-path curvature at the draft poses k = 1..8 [1/m]
    path: DraftPath


@torch.no_grad()
def corridor_geometry(tau0: torch.Tensor) -> CorridorGeom:
    """Corridor points and geometric channels for drafts tau0 [B, 8, 3] (N frame)."""
    tau0 = tau0.reshape(-1, N_POSE, 3)
    B = tau0.shape[0]
    dev, dt = tau0.device, tau0.dtype
    path = DraftPath(tau0)
    S = path.knots()
    S8 = S[:, -1]
    v_end = (S8 - S[:, -2]) / T_POSE
    S_look = S8 + torch.clamp(v_end * T_LOOK, min=LOOK_MIN)
    path.extend(S_look - S8, "const_curv", v=v_end)
    ext_clipped = (path.kappa_end.abs() > kappa_limit(v_end)) & (S8 >= NEAR_STOP_S)
    ds = torch.clamp(S_look / N_ST, min=DS_MIN)
    s = (torch.arange(N_ST, device=dev, dtype=dt)[None] + 0.5) * ds[:, None]      # [B, 48]
    pe = path.eval(s)
    d = lateral_offsets(dev, dt)                                                   # [17]
    pts = pe.xy[:, :, None, :] + d[None, None, :, None] * pe.normal[:, :, None, :]  # [B, 48, 17, 2]
    t_arr, v_arr = arrival_time_speed(S, s, v_end)
    ones = torch.ones(B, N_ST, N_LAT, device=dev, dtype=dt)
    geo = torch.stack([
        in_s_grid(pts).to(dt),
        (s <= S8[:, None]).to(dt)[:, :, None] * ones,
        (s / S_SCALE)[:, :, None] * ones,
        (d / LAT_MAX)[None, None, :] * ones,
        torch.clamp(t_arr / T_SCALE, 0.0, 1.0)[:, :, None] * ones,
        (v_arr / V_SCALE)[:, :, None] * ones,
    ], 1)                                                                          # [B, 6, 48, 17]
    kk = path.eval(S[:, 1:]).kappa
    return CorridorGeom(points=pts, geo=geo, s=s, S=S, S_look=S_look, v_end=v_end, near_stop=S8 < NEAR_STOP_S,
                        ext_clipped=ext_clipped, kappa_knots=kk, path=path)


def sample_corridor(feat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """feat [T, C, 50, 100], points [T, K, 48, 17, 2] -> [T, K, C, 48, 17] (per-token grid_sample, no replication)."""
    T, K, Ns, Nl, _ = points.shape
    out = sample_s_grid(feat, points.reshape(T, K * Ns, Nl, 2))                    # [T, C, K*Ns, Nl]
    C = out.shape[1]
    return out.reshape(T, C, K, Ns, Nl).permute(0, 2, 1, 3, 4)


def corridor_tensor(feat: torch.Tensor, geom: CorridorGeom, n_tokens: int, n_drafts: int) -> torch.Tensor:
    """X [T*K, C + 6, 48, 17]: sampled features (C) then GEO_CHANNELS; draft b = t * K + k."""
    T, K = n_tokens, n_drafts
    pts = geom.points.reshape(T, K, N_ST, N_LAT, 2)
    f = sample_corridor(feat, pts).reshape(T * K, -1, N_ST, N_LAT)
    return torch.cat([f, geom.geo.to(f.dtype)], 1)


def global_tokens(feat: torch.Tensor, pool: int = POOL) -> torch.Tensor:
    """feat [T, C, 50, 100] -> [T, 200, C] (5 x 5 average pool, row-major tokens)."""
    g = F.avg_pool2d(feat, pool)                                                  # [T, C, 10, 20]
    return g.flatten(2).transpose(1, 2)
