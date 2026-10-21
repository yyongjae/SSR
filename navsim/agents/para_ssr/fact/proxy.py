"""Proxy PDM sub-scores of arbitrary candidate trajectories, on the GPU, from GT rasters.

The official PDM scorer (navsim/planning/simulation/planner/pdm_planner/scoring)
tracks the trajectory with an LQR controller, builds the ego footprint at 10 Hz
and checks it against the GT drivable map and the GT future objects.  The
proxies here evaluate the raw candidate poses at 2 Hz against the same two GT
sources rasterised by fact/targets.py:

    DAC   every footprint corner on the drivable area (SDF >= 0) at every step
    NC    the footprint never overlaps an object while the ego moves
    TTC   NC also holds for the footprint pushed 0.3 / 0.6 / 0.9 s ahead at the
          current speed and heading (PDM's constant-velocity projection)
    EP    progress along the PDM-Closed route centerline, normalised by the best
          safe candidate of the scene (WoTE's label convention)
    C     the comfort thresholds of pdm_comfort_metrics.py on 0.5 s differences

On GT anchors these proxies predict the official DAC with AUC 0.975 and NC/TTC
with 0.908 (report/42 section 7); they are labels for candidates that have no
official score, not a replacement for it.  Exact official labels of the 256
WoTE anchors are used alongside (fact/losses.py).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from navsim.agents.para_ssr.plan_map import footprint_corners

# pdm_comfort_metrics.py
MAX_ABS_JERK = 8.37
MAX_ABS_LAT_ACCEL = 4.89
MAX_LON_ACCEL = 2.40
MIN_LON_ACCEL = -4.05
MAX_ABS_YAW_ACCEL = 1.93
MAX_ABS_LON_JERK = 4.13
MAX_ABS_YAW_RATE = 0.95
STOPPED_SPEED = 5e-3          # PDMScorerConfig.stopped_speed_threshold
PROGRESS_THRESHOLD = 0.1      # PDMScorerConfig.progress_distance_threshold
DT = 0.5
TTC_HORIZONS = (0.3, 0.6, 0.9)


def field_grid(pts: torch.Tensor, extent) -> Tuple[torch.Tensor, torch.Tensor]:
    """nav (x, y) ``[B, N, 2]`` -> grid_sample coordinates ``[B, 1, N, 2]`` on a field grid
    (rows along x, columns along y) and the inside mask ``[B, N]``."""
    x0, x1, y0, y1 = extent
    u = (pts[..., 1] - y0) / (y1 - y0) * 2 - 1
    v = (pts[..., 0] - x0) / (x1 - x0) * 2 - 1
    inside = (u.abs() <= 1) & (v.abs() <= 1)
    return torch.stack([u, v], dim=-1).unsqueeze(1), inside


def bev_grid(pts: torch.Tensor, pc_range) -> Tuple[torch.Tensor, torch.Tensor]:
    """nav (x, y) ``[B, N, 2]`` -> grid_sample coordinates on the SSR BEV grid
    (rows = forward = nav x, columns = right = -nav y) and the inside mask."""
    xr0, yf0, _, xr1, yf1, _ = pc_range
    u = (-pts[..., 1] - xr0) / (xr1 - xr0) * 2 - 1
    v = (pts[..., 0] - yf0) / (yf1 - yf0) * 2 - 1
    inside = (u.abs() <= 1) & (v.abs() <= 1)
    return torch.stack([u, v], dim=-1).unsqueeze(1), inside


def sample(field: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """``field [B, C, H, W]``, ``grid [B, 1, N, 2]`` -> ``[B, N, C]`` (border padding)."""
    out = F.grid_sample(field.to(grid.dtype), grid, mode=mode, padding_mode="border", align_corners=False)
    return out[:, :, 0].transpose(1, 2)


def _footprint_points(poses: torch.Tensor) -> torch.Tensor:
    """``[..., 3]`` -> 9 probe points ``[..., 9, 2]``: 4 corners, 4 edge midpoints, centre."""
    c = footprint_corners(poses)                                   # [..., 4, 2] FL, FR, RL, RR
    mids = torch.stack([(c[..., 0, :] + c[..., 1, :]) / 2, (c[..., 2, :] + c[..., 3, :]) / 2,
                        (c[..., 0, :] + c[..., 2, :]) / 2, (c[..., 1, :] + c[..., 3, :]) / 2], dim=-2)
    centre = c.mean(dim=-2, keepdim=True)
    return torch.cat([c, mids, centre], dim=-2)


def _occupied(occ: torch.Tensor, pts: torch.Tensor, step: torch.Tensor, extent) -> torch.Tensor:
    """occ ``[B, T, nx, ny]``; pts ``[B, N, Q, 2]``; step ``[N, Q]`` (0..T-1) -> hit ``[B, N, Q]``."""
    B, N, Q, _ = pts.shape
    T = occ.shape[1]
    grid, inside = field_grid(pts.reshape(B, N * Q, 2), extent)
    vals = sample(occ, grid, mode="nearest")                        # [B, NQ, T]
    idx = step.reshape(1, N * Q, 1).expand(B, -1, 1).clamp(0, T - 1)
    hit = vals.gather(2, idx)[..., 0] > 0.5
    return (hit & inside).reshape(B, N, Q)


def _speeds(poses: torch.Tensor) -> torch.Tensor:
    """``[B, N, T, 3]`` -> speed at each pose ``[B, N, T]`` (backward difference)."""
    xy = poses[..., :2]
    prev = torch.cat([torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], dim=-2)
    return (xy - prev).norm(dim=-1) / DT


def route_progress(poses: torch.Tensor, route: torch.Tensor, route_valid: torch.Tensor,
                   route_ok: torch.Tensor) -> torch.Tensor:
    """Progress of ``[B, N, T, 3]`` along the route ``[B, R, 2]`` (stations at 1 m from the ego
    projection) -> ``[B, N]`` metres.  Falls back to forward displacement without a route."""
    B, N, T, _ = poses.shape
    end = poses[..., -1, :2]                                                     # [B, N, 2]
    a = route[:, :-1]                                                            # [B, R-1, 2]
    seg = route[:, 1:] - a
    seg_len2 = (seg * seg).sum(-1).clamp(min=1e-6)
    seg_valid = (route_valid[:, 1:] > 0)                                         # [B, R-1]
    rel = end.unsqueeze(2) - a.unsqueeze(1)                                      # [B, N, R-1, 2]
    t = ((rel * seg.unsqueeze(1)).sum(-1) / seg_len2.unsqueeze(1)).clamp(0, 1)   # [B, N, R-1]
    proj = a.unsqueeze(1) + t.unsqueeze(-1) * seg.unsqueeze(1)
    d = (end.unsqueeze(2) - proj).norm(dim=-1)
    d = torch.where(seg_valid.unsqueeze(1), d, torch.full_like(d, 1e9))
    k = d.argmin(dim=-1)                                                         # [B, N]
    spacing = seg_len2.sqrt()                                                    # [B, R-1]
    cum = torch.cat([torch.zeros_like(spacing[:, :1]), spacing.cumsum(dim=1)], dim=1)   # [B, R]
    s_end = cum.gather(1, k) + t.gather(2, k.unsqueeze(-1))[..., 0] * spacing.gather(1, k)
    ok = (route_ok > 0).view(B, 1)
    return torch.where(ok, s_end, end[..., 0]).clamp(min=0.0)


def comfort(poses: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """Comfort flag ``[B, N]`` from 0.5 s finite differences (approximate PDM thresholds)."""
    B, N, T, _ = poses.shape
    xy = torch.cat([poses.new_zeros(B, N, 1, 2), poses[..., :2]], dim=-2)                 # [B, N, T+1, 2]
    vel = (xy[..., 1:, :] - xy[..., :-1, :]) / DT                                           # [B, N, T, 2]
    speed = vel.norm(dim=-1)
    speed = torch.cat([v0.view(B, 1, 1).expand(B, N, 1).to(speed.dtype), speed], dim=-1)   # [B, N, T+1]
    lon_acc = (speed[..., 1:] - speed[..., :-1]) / DT
    lon_jerk = (lon_acc[..., 1:] - lon_acc[..., :-1]) / DT
    acc_vec = (vel[..., 1:, :] - vel[..., :-1, :]) / DT
    jerk_vec = (acc_vec[..., 1:, :] - acc_vec[..., :-1, :]) / DT
    heading = torch.cat([poses.new_zeros(B, N, 1), poses[..., 2]], dim=-1)
    dh = heading[..., 1:] - heading[..., :-1]
    yaw_rate = torch.atan2(torch.sin(dh), torch.cos(dh)) / DT
    yaw_acc = (yaw_rate[..., 1:] - yaw_rate[..., :-1]) / DT
    lat_acc = speed[..., 1:] * yaw_rate
    ok = (
        (lon_acc.amax(-1) <= MAX_LON_ACCEL) & (lon_acc.amin(-1) >= MIN_LON_ACCEL)
        & (lon_jerk.abs().amax(-1) <= MAX_ABS_LON_JERK)
        & (jerk_vec.norm(dim=-1).amax(-1) <= MAX_ABS_JERK)
        & (lat_acc.abs().amax(-1) <= MAX_ABS_LAT_ACCEL)
        & (yaw_rate.abs().amax(-1) <= MAX_ABS_YAW_RATE)
        & (yaw_acc.abs().amax(-1) <= MAX_ABS_YAW_ACCEL)
    )
    return ok.to(poses.dtype)


@torch.no_grad()
def proxy_labels(
    cand: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    extent,
    v0: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """``cand [B, N, T, 3]`` (nav frame) -> dict of ``[B, N]`` labels in [0, 1] and ``pdms``.

    targets: ``drivable_sdf [B, nx, ny]``, ``fact_occ [B, T, nx, ny]``, ``fact_route``,
    ``fact_route_valid``, ``fact_route_ok``.  ``v0 [B]`` is the current speed.
    """
    B, N, T, _ = cand.shape
    cand = cand.float()
    sdf = targets["drivable_sdf"].float()
    occ = targets["fact_occ"].float()
    dev = cand.device

    # ---- DAC: footprint corners inside the drivable area at every step
    corners = footprint_corners(cand).reshape(B, N * T * 4, 2)
    grid, inside = field_grid(corners, extent)
    s = sample(sdf.unsqueeze(1), grid)[..., 0]
    off = (s < 0) & inside
    dac = (~off.reshape(B, N, T * 4).any(-1)).float()

    # ---- NC / TTC: probe points of the footprint against the objects of that step
    speed = _speeds(cand)                                                         # [B, N, T]
    moving = speed > STOPPED_SPEED
    probes = _footprint_points(cand)                                              # [B, N, T, 9, 2]
    step = torch.arange(T, device=dev).view(T, 1).expand(T, 9)                    # [T, 9]
    hit0 = _occupied(occ, probes.reshape(B, N, T * 9, 2), step.reshape(1, T * 9).expand(N, -1), extent)
    hit0 = hit0.reshape(B, N, T, 9).any(-1) & moving
    nc = (~hit0.any(-1)).float()

    ttc_hit = hit0.any(-1)
    heading = cand[..., 2]
    direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)    # [B, N, T, 2]
    for tau in TTC_HORIZONS:
        shifted = cand.clone()
        shifted[..., :2] = cand[..., :2] + direction * (speed * tau).unsqueeze(-1)
        probes_tau = _footprint_points(shifted)
        step_tau = (torch.arange(T, device=dev).float() + tau / DT).round().long().clamp(max=T - 1)
        step_tau = step_tau.view(T, 1).expand(T, 9)
        hit = _occupied(occ, probes_tau.reshape(B, N, T * 9, 2), step_tau.reshape(1, T * 9).expand(N, -1), extent)
        ttc_hit = ttc_hit | (hit.reshape(B, N, T, 9).any(-1) & moving).any(-1)
    ttc = (~ttc_hit).float()

    # ---- EP: route progress, normalised by the best safe candidate of the scene
    ep_raw = route_progress(cand, targets["fact_route"].float(), targets["fact_route_valid"].float(),
                            targets["fact_route_ok"].float())
    safe = (nc > 0) & (dac > 0)
    ref = torch.where(safe, ep_raw, torch.zeros_like(ep_raw)).amax(dim=1, keepdim=True)     # [B, 1]
    ep = torch.where(ref > PROGRESS_THRESHOLD, (ep_raw / ref.clamp(min=1e-6)).clamp(0, 1),
                     torch.ones_like(ep_raw))

    # ---- comfort
    cf = comfort(cand, v0)

    pdms = nc * dac * (5 * ttc + 5 * ep + 2 * cf) / 12.0
    return {"nc": nc, "dac": dac, "ttc": ttc, "ep": ep, "ep_raw": ep_raw, "comfort": cf, "pdms": pdms}


def composite(nc: torch.Tensor, dac: torch.Tensor, ep: torch.Tensor, ttc: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """PDMS of probabilities/values in [0, 1] (differentiable)."""
    return nc * dac * (5 * ttc + 5 * ep + 2 * c) / 12.0
