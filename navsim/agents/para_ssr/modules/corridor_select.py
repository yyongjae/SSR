"""km 10/17: corridor selection inside the forward (inference) -- the same computation as the offline
tools/corridor/corr_verdict.py (pred_sdf_min / pred_occ_max) + corr_select.scores (pred_* rules), so the agent's
submitted trajectory is the corridor-selected candidate and the standard NAVSIM scripts score it directly.

Every one of the 256 candidates (anchor + offset) is tracked by the torch port of the official LQR + bicycle
simulation from the status-only initial state, then judged on the PREDICTED corridor:
  sdf_min = min over 40 steps x 4 footprint corners of the predicted static SDF (out of range -> CLIP)
  occ_max = max over 40 steps x 15 footprint points of the predicted occupancy probability (out of range -> 0)
  p_dac = sigmoid(sdf_min / tau), p_nc = 1 - occ_max
Rules (``corridor_select_rule``):
  v2                   : argmax of v2's weighted reward (no corridor), the anchor planner's own choice
  pred_multiply_t<tau> : v2 final + 0.5 log p_nc + 0.5 log p_dac
  pred_replace_t<tau>  : 0.1 log im + 0.5 log p_nc + 0.5 log p_dac + log(5 TTC + 2 C + 5 EP)  (NC / DAC heads replaced)
The offline path stores the predicted occupancy as uint8 and the SDF as float16, so a candidate whose score ties
within that rounding may differ; the in-forward values are the unrounded ones.
"""
from __future__ import annotations

from typing import Dict

import torch

from . import corridor_head as CH
from .torch_tracker import track

EPS = 1e-6
RULES = ("v2", "pred_multiply", "pred_replace")


def parse_rule(rule: str):
    """'v2' -> ('v2', None); 'pred_multiply_t0.25' -> ('multiply', 0.25)."""
    if rule == "v2":
        return "v2", None
    parts = rule.split("_")
    if len(parts) != 3 or parts[0] != "pred" or parts[1] not in ("multiply", "replace") or not parts[2].startswith("t"):
        raise ValueError(f"corridor_select_rule {rule!r}: expected v2 | pred_multiply_t<tau> | pred_replace_t<tau>")
    return parts[1], float(parts[2][1:])


@torch.no_grad()
def corridor_verdict(traj: torch.Tensor, status: torch.Tensor, sdf: torch.Tensor, occ: torch.Tensor):
    """traj [B, K, 8, 3] candidates (NAVSIM ego frame); status [B, 4] = (vx, vy, ax, ay) of the rear axle;
    sdf [B, H, 200] predicted static SDF (m); occ [B, P, H, 200] predicted occupancy probability (P = 8 swept or 40).
    -> (sdf_min [B, K], occ_max [B, K]), as tools/corridor/corr_verdict.py."""
    b, k = traj.shape[:2]
    dev = traj.device
    init = torch.cat([status.float(), torch.zeros(b, 3, device=dev)], -1).repeat_interleave(k, 0)
    st = track(traj.reshape(-1, 8, 3).float(), init, full=True).reshape(b, k, 41, 7)[:, :, 1:]
    x, y, h = st[..., 0], st[..., 1], st[..., 2]
    cx = torch.tensor([CH.FRONT, CH.FRONT, -CH.REAR, -CH.REAR], device=dev)
    cy = torch.tensor([CH.HALF_W, -CH.HALF_W, -CH.HALF_W, CH.HALF_W], device=dev)
    fx, fy = torch.meshgrid(torch.linspace(-CH.REAR, CH.FRONT, 5, device=dev),
                            torch.linspace(-CH.HALF_W, CH.HALF_W, 3, device=dev), indexing="ij")
    fx, fy = fx.reshape(-1), fy.reshape(-1)
    c, s = torch.cos(h)[..., None], torch.sin(h)[..., None]
    pc = torch.stack([x[..., None] + c * cx - s * cy, y[..., None] + s * cx + c * cy], -1)       # [b, k, 40, 4, 2]
    pf = torch.stack([x[..., None] + c * fx - s * fy, y[..., None] + s * fx + c * fy], -1)       # [b, k, 40, 15, 2]
    v, inr = CH._sample(sdf.float()[:, None], pc.reshape(b, 1, -1, 2))
    sdf_min = torch.where(inr, v, torch.full_like(v, CH.CLIP)).reshape(b, k, -1).amin(-1)
    plane = torch.arange(40, device=dev) // (40 // occ.shape[1])
    v, inr = CH._sample(occ.float()[:, plane], pf.permute(0, 2, 1, 3, 4).reshape(b, 40, -1, 2))
    occ_max = torch.where(inr, v, torch.zeros_like(v)).reshape(b, 40, k, 15).amax(-1).amax(1)
    return sdf_min, occ_max


@torch.no_grad()
def corridor_select(predictions: Dict[str, torch.Tensor], status: torch.Tensor, rule: str) -> Dict[str, torch.Tensor]:
    """predictions of ParaSSRModel (anchor planner + corridor head) -> {trajectory [B, 8, 3], index [B]} of ``rule``."""
    mode, tau = parse_rule(rule)
    offset = predictions["trajectory_offset"].float()
    traj = predictions["trajectory_anchors"].float()[None] + offset                              # [B, K, 8, 3]
    fin = predictions["plan_final_rewards"].float()
    if mode == "v2":
        score = fin
    else:
        sdf_min, occ_max = corridor_verdict(traj, status, predictions["corridor_sdf"],
                                            torch.sigmoid(predictions["corridor_occ"].float()))
        p_dac = torch.sigmoid((sdf_min / tau).clamp(-50, 50))
        p_nc = (1 - occ_max).clamp(0, 1)
        corr = 0.5 * torch.log(p_nc + EPS) + 0.5 * torch.log(p_dac + EPS)
        if mode == "multiply":
            score = fin + corr
        else:
            im, sim = predictions["im_rewards"].float(), predictions["sim_rewards"].float()
            ep, ttc, cf = sim[:, 2], sim[:, 3], sim[:, 4]
            score = 0.1 * torch.log(im + EPS) + corr + torch.log(5 * ttc + 2 * cf + 5 * ep + EPS)
    idx = score.argmax(1)
    return {"trajectory": traj[torch.arange(traj.shape[0], device=traj.device), idx], "index": idx}
