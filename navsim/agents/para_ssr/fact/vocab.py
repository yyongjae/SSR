"""Factorized trajectory vocabulary: path anchors x speed profiles (SparseDriveV2-style).

A *path* is a polyline in the NAVSIM ego frame (x forward, y left, rear axle)
sampled at ``S`` arc-length stations of ``station_m`` metres, station 0 at the
origin: ``paths [P, S+1, 2]``.  A *speed profile* is the cumulative distance
``s(t)`` travelled at t = 0.5, 1.0, ..., 4.0 s: ``speeds [V, T]``, non-decreasing.

A candidate trajectory is the composition ``traj(t) = path(s(t))``: the point at
arc length ``s(t)`` along the path (linear interpolation between stations), with
the heading of the path tangent there.  ``tools/fact/build_vocab.py`` builds the
two files from navtrain human trajectories with k-means.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def path_headings(paths: torch.Tensor) -> torch.Tensor:
    """``[P, S+1, 2]`` -> tangent heading per station ``[P, S+1]`` (central differences)."""
    d = torch.zeros_like(paths)
    d[:, 1:-1] = paths[:, 2:] - paths[:, :-2]
    d[:, 0] = paths[:, 1] - paths[:, 0]
    d[:, -1] = paths[:, -1] - paths[:, -2]
    h = torch.atan2(d[..., 1], d[..., 0])
    # a degenerate (zero-length) segment keeps the previous heading
    bad = d.norm(dim=-1) < 1e-6
    h = torch.where(bad, torch.zeros_like(h), h)
    return h


def compose(
    paths: torch.Tensor,
    path_heading: torch.Tensor,
    speeds: torch.Tensor,
    station_m: float,
) -> torch.Tensor:
    """Compose every (path, speed) pair of the given sets.

    paths ``[..., S+1, 2]``, path_heading ``[..., S+1]``, speeds ``[..., T]`` broadcast on
    their leading dims (e.g. paths ``[B, Kp, 1, S+1, 2]`` with speeds ``[1, 1, V, T]``).
    Returns poses ``[..., T, 3]`` (x, y, heading).
    """
    S = paths.shape[-2] - 1
    s = (speeds / station_m).clamp(0.0, float(S) - 1e-4)           # fractional station index
    i0 = s.floor().long()                                           # [..., T]
    w = (s - i0.to(s.dtype)).unsqueeze(-1)                          # [..., T, 1]
    i1 = i0 + 1
    lead = torch.broadcast_shapes(paths.shape[:-2], s.shape[:-1])
    pts = paths.expand(*lead, *paths.shape[-2:])
    hd = path_heading.expand(*lead, path_heading.shape[-1])
    i0e = i0.expand(*lead, i0.shape[-1])
    i1e = i1.expand(*lead, i1.shape[-1])
    p0 = pts.gather(-2, i0e.unsqueeze(-1).expand(*i0e.shape, 2))
    p1 = pts.gather(-2, i1e.unsqueeze(-1).expand(*i1e.shape, 2))
    xy = p0 + w * (p1 - p0)
    h0 = hd.gather(-1, i0e)
    h1 = hd.gather(-1, i1e)
    # interpolate the heading on the circle
    dh = torch.atan2(torch.sin(h1 - h0), torch.cos(h1 - h0))
    h = h0 + w.squeeze(-1) * dh
    return torch.cat([xy, h.unsqueeze(-1)], dim=-1)


def resample_path(xy: torch.Tensor, num_stations: int, station_m: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Human trajectory xy ``[B, T, 2]`` -> stations ``[B, S+1, 2]`` and real length ``[B]``.

    The polyline (origin + T points) is resampled at ``station_m`` spacing; beyond its
    end it is extrapolated straight along the last segment (same rule as the builder).
    """
    B, T, _ = xy.shape
    pts = torch.cat([xy.new_zeros(B, 1, 2), xy], dim=1)                       # [B, T+1, 2]
    seg = pts[:, 1:] - pts[:, :-1]
    seg_len = seg.norm(dim=-1)                                                 # [B, T]
    cum = torch.cat([xy.new_zeros(B, 1), seg_len.cumsum(dim=1)], dim=1)        # [B, T+1]
    length = cum[:, -1]
    # direction of the last non-degenerate segment, +x if none
    last_dir = xy.new_zeros(B, 2)
    last_dir[:, 0] = 1.0
    for t in range(T):
        ok = seg_len[:, t] > 1e-3
        last_dir = torch.where(ok.unsqueeze(-1), seg[:, t] / seg_len[:, t].clamp(min=1e-6).unsqueeze(-1), last_dir)
    targets = torch.arange(num_stations + 1, device=xy.device, dtype=xy.dtype) * station_m   # [S+1]
    out = xy.new_zeros(B, num_stations + 1, 2)
    for b in range(B):
        c = cum[b]
        tb = targets.clamp(max=c[-1])
        idx = torch.searchsorted(c, tb, right=True).clamp(1, T) - 1               # segment index
        c0, c1 = c[idx], c[idx + 1]
        w = ((tb - c0) / (c1 - c0).clamp(min=1e-6)).unsqueeze(-1)
        p = pts[b, idx] + w * (pts[b, idx + 1] - pts[b, idx])
        extra = (targets - c[-1]).clamp(min=0).unsqueeze(-1) * last_dir[b]
        out[b] = p + extra
    return out, length


def cumulative_distance(xy: torch.Tensor) -> torch.Tensor:
    """``[B, T, 2]`` -> cumulative arc length at each pose ``[B, T]``."""
    pts = torch.cat([xy.new_zeros(xy.shape[0], 1, 2), xy], dim=1)
    return (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).cumsum(dim=1)


def load_vocab(path_file: str, speed_file: str):
    paths = np.load(path_file).astype(np.float32)
    speeds = np.load(speed_file).astype(np.float32)
    if paths.ndim != 3 or paths.shape[-1] != 2:
        raise ValueError(f"paths must be [P, S+1, 2], got {paths.shape}")
    if speeds.ndim != 2:
        raise ValueError(f"speeds must be [V, T], got {speeds.shape}")
    speeds = np.maximum.accumulate(np.clip(speeds, 0.0, None), axis=1)
    return torch.from_numpy(paths), torch.from_numpy(speeds)
