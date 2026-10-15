"""A candidate's schedule on its PARENT's station grid (refiner v4 hazard readout, V4Cfg.readout 'hazard').

Frames / units: NAVSIM N frame at t0 (x forward, y left, metres, seconds); candidate poses at t = 0.5 k s, k = 1..8,
the origin (0, 0) at t = 0.  The parent's corridor (refiner.corridor.corridor_geometry) has 48 stations s_j on its
smooth path Gamma_p.

on_parent(pgeom, cand, is_self)
  * knots: the 9 points (origin + 8 poses) projected on the parent's station frame (temporal.raster.frenet: nearest
    station, chord interpolation to the neighbour) -> arc length s_k and signed lateral offset d_k (left +) [B, 9];
    s_k made non-decreasing (cummax).  is_self rows (the candidate IS its parent) use the parent's own knot arc lengths
    pgeom.S and d = 0 exactly, so a parent reads its own grid without projection error.
  * t_c(j), v_c(j): corridor.arrival_time_speed(s_k, s_j, v_end) with v_end = (s_8 - s_7) / 0.5 (the same piecewise
    linear first-arrival schedule the corridor's geo channels use; beyond s_8: 4 s + (s_j - s_8) / max(v_end, 0.1)).
  * t_in(j), t_leave(j): the same schedule at the station bin's edges s_j -+ ds / 2 (ds = s_1 - s_0): the time the
    candidate enters / leaves station j's bin (t_in(0) = 0: every candidate starts in bin 0; a candidate stopping
    inside bin j leaves it after 4 s).
  * d_c(j): piecewise-linear interpolation of d_k over s_k at s_j (held constant before s_0 / after s_8).
  * ok [B]: finite projection.
Speed siblings (rule variants a +-x) lie on the parent's path (d ~ 0) up to the parent's S_8 with a different t_c(j)
(beyond it the CK2 a+0.5 variant is extended straight while the parent corridor keeps its end curvature, so d grows
there on curved parents: the report 72 I4 extension mismatch, measured up to ~0.4 m in tests/test_v4_model.py);
lateral siblings keep t_c(j) ~ t_p(j) with d ~ +-0.5 m; any other edit (learned corrector) is handled by the same
projection.
Everything under no_grad (geometry is an input, like the corridor geometry)."""
from __future__ import annotations

from typing import Dict

import torch

from ...refiner.corridor import N_ST, arrival_time_speed
from ...refiner.geometry import T_POSE
from ..temporal.raster import frenet, station_frame


def _interp(xk: torch.Tensor, yk: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Row-wise piecewise-linear interpolation: xk [B, K] non-decreasing, yk [B, K], x [B, M] -> [B, M] (held outside)."""
    K = xk.shape[1]
    i = torch.searchsorted(xk.contiguous(), x.contiguous(), right=True).clamp(1, K - 1)
    x0, x1 = torch.gather(xk, 1, i - 1), torch.gather(xk, 1, i)
    y0, y1 = torch.gather(yk, 1, i - 1), torch.gather(yk, 1, i)
    w = torch.where(x1 > x0, (x - x0) / torch.where(x1 > x0, x1 - x0, torch.ones_like(x1)), torch.zeros_like(x))
    y = y0 + w.clamp(0.0, 1.0) * (y1 - y0)
    y = torch.where(x <= xk[:, :1], yk[:, :1].expand_as(y), y)
    return torch.where(x >= xk[:, -1:], yk[:, -1:].expand_as(y), y)


@torch.no_grad()
def on_parent(pgeom, cand: torch.Tensor, is_self: torch.Tensor) -> Dict[str, torch.Tensor]:
    """pgeom = corridor_geometry(parent [B, 8, 3]) (row b = candidate b's parent), cand [B, 8, 3], is_self [B] bool ->
    t, v, d, t_in, t_leave [B, 48] (float32), s_knots, d_knots [B, 9], ok [B]."""
    B = cand.shape[0]
    c = cand.float()
    xy = torch.cat([torch.zeros(B, 1, 2, dtype=c.dtype, device=c.device), c[..., :2]], 1)          # [B, 9, 2]
    fr = frenet(xy, station_frame(pgeom))
    s_k = torch.cummax(fr["s"], 1).values
    d_k = fr["d"]
    S_p = pgeom.S.float()
    self_ = is_self.reshape(B, 1).bool()
    s_k = torch.where(self_, S_p, s_k)
    d_k = torch.where(self_, torch.zeros_like(d_k), d_k)
    ok = fr["ok"].all(1) | is_self.reshape(B).bool()
    s_k = torch.where(torch.isfinite(s_k), s_k, torch.zeros_like(s_k))
    d_k = torch.where(torch.isfinite(d_k), d_k, torch.zeros_like(d_k))
    v_end = torch.clamp((s_k[:, -1] - s_k[:, -2]) / T_POSE, min=0.0)
    s_j = pgeom.s.float()
    t, v = arrival_time_speed(s_k, s_j, v_end)
    half = 0.5 * (s_j[:, 1:2] - s_j[:, :1])
    t_in, _ = arrival_time_speed(s_k, (s_j - half).clamp(min=0.0), v_end)
    t_leave, _ = arrival_time_speed(s_k, s_j + half, v_end)
    d = _interp(s_k, d_k, s_j)
    assert t.shape == (B, N_ST)
    return {"t": t.float(), "v": v.float(), "d": d.float(), "t_in": t_in.float(), "t_leave": t_leave.float(),
            "s_knots": s_k, "d_knots": d_k, "ok": ok}
