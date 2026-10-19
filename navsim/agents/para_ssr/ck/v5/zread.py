"""refiner v5 per-state reads Z of the range fields (V5Cfg on; report 97 §2-8, add.md (2) Z_k,j).

For every executed LQR state n of a candidate (the range readout's read_geometry, same cells / bins / channels as the
NC / TTC / DAC evidence, before the soft max over the states):
  b_nc   [B, Q]  B logit at the NC read (rear axle, bin(t_n)); 'any4' = max over the distinct 2 x 2 cells; FLOOR where the
                 state is not read (stopped ego, off grid, bin out of range);
  b_ttc  [B, Q]  max over delta (0, 0.3, 0.6, 0.9 s) of the TTC read (channel rule of the readout: U where |v| delta
                 >= 2 m, else B / Bp); FLOOR where not read;
  d_min  [B, Q]  min over the 4 official corners of the bilinear margin D (m), clipped to +-D_CAP; D_CAP where no corner
                 is readable;
  unknown [B, Q] the NC point or any corner is off the parent grid, or outside the BEV extent (in_s_grid: 0 <= x <= 32 m, |y| <= 32 m):
                 no local evidence ("unknown", never "safe").
zq = [clip(b_nc / 5, +-4), clip(b_ttc / 5, +-4), d_min / D_CAP, unknown] at every stride-th state -> [B, Q', 4];
differentiable in the fields (the corrector's loss and a later Z-KD reach the field heads through it).
"""
from __future__ import annotations

from typing import Dict

import torch

from ..v4.range_readout import any4_cells, nearest_cells, read_cells, read_nearest

Tensor = torch.Tensor
FLOOR = -20.0          # logit of a state that is not read (no evidence; the unknown flag says why)
D_CAP = 5.0            # [m]


def _read_points(G: Tensor, rows: Tensor, gc: Dict[str, Tensor], ch: Tensor, read: str) -> Tensor:
    B, P = gc["s"].shape
    if read == "any4":
        jn, in_, distinct = any4_cells(gc)
        v = read_nearest(G, rows, jn.reshape(B, P * 4), in_.reshape(B, P * 4),
                         ch[..., None].expand(B, P, 4).reshape(B, P * 4)).reshape(B, P, 4)
        return torch.where(distinct, v, torch.full_like(v, -1e4)).amax(-1)
    return read_nearest(G, rows, *nearest_cells(gc), ch)


def bev_support(states: Tensor) -> Tensor:
    """states [B, Q, >=3] (N frame, rear axle) -> [B, Q] bool: the rear axle and the 4 official corners lie inside the
    S-grid / BEV extent (refiner.corridor.in_s_grid: 0 <= x <= 32 m, |y| <= 32 m)."""
    from ...refiner.corridor import in_s_grid
    from ...refiner.sdf import ego_corners
    st = states[..., :3].float()
    pts = torch.cat([st[..., None, :2], ego_corners(st)], -2)                          # [B, Q, 5, 2]
    return in_s_grid(pts).all(-1)


def state_reads(blk: Tensor, D: Tensor, rows: Tensor, geo: Dict[str, Tensor], M: int, read: str, *,
                states: Tensor) -> Dict[str, Tensor]:
    """blk [R, 48, 17, G M] logits, D [R, 48, 17] (m), rows [B], geo = range_readout.read_geometry(...), states
    [B, Q, 6] (the same executed states) -> b_nc, b_ttc, d_min, unknown [B, Q] (module docstring).  unknown = the NC
    point or a corner off the parent grid OR outside the BEV window (bev_support; review 98 R2: the parent grid can
    run past the BEV, where the fields are extrapolation, not evidence)."""
    gn, gt, gd = geo["nc"], geo["ttc"], geo["dac"]
    B, Q = gn["s"].shape
    ch_nc = geo["m_nc"][None].expand(B, Q)
    v_nc = _read_points(blk, rows, gn, ch_nc, read)
    b_nc = torch.where(geo["rd_nc"], v_nc, torch.full_like(v_nc, FLOOR))
    ch_tt = geo["m_ttc"][None] + M * geo["chan"]
    v_tt = _read_points(blk, rows, gt, ch_tt, read)
    b_tt = torch.where(geo["rd_ttc"], v_tt, torch.full_like(v_tt, FLOOR)).reshape(B, Q, -1).amax(-1)
    Dc = read_cells(D[..., None].contiguous(), rows, gd["us"], gd["ud"], 0).reshape(B, Q, 4)
    okc = gd["ok"].reshape(B, Q, 4)
    dmin = torch.where(okc, Dc, torch.full_like(Dc, float("inf"))).amin(-1)
    dmin = torch.where(torch.isfinite(dmin), dmin, torch.full_like(dmin, D_CAP)).clamp(-D_CAP, D_CAP)
    unknown = (~gn["ok"]) | (~okc.all(-1))
    unknown = unknown | ~bev_support(states)
    return {"b_nc": b_nc, "b_ttc": b_tt, "d_min": dmin, "unknown": unknown}


def zq_of(r: Dict[str, Tensor], stride: int) -> Tensor:
    """state_reads dict -> zq [B, Q', 4] at every stride-th state (float32)."""
    z = torch.stack([(r["b_nc"].float() / 5.0).clamp(-4.0, 4.0), (r["b_ttc"].float() / 5.0).clamp(-4.0, 4.0),
                     r["d_min"].float() / D_CAP, r["unknown"].float()], -1)
    return z[:, ::int(stride)]
