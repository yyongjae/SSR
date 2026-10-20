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
km 10/18 (v3.2b):
  pred_<mode>_sigma    : p_dac = Phi(min over corners / steps of sdf / sigma), sigma = the corridor_sdf_sigma head
  suffix _ttc          : + 0.5 log p_ttc, p_ttc = 1 - ttc_max, ttc_max = max over the 40 steps and dt in {0.3, 0.6,
                         0.9} s of the predicted occupancy at t + dt under the front half of the footprint shifted along
                         the heading by speed * dt (official _calculate_ttc; a stopped ego (< 5e-3 m/s) is exempt;
                         49 planes reach 4.9 s, 40 planes hold the 4.0 s one)
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


def parse_rule(rule: str, full: bool = False):
    """'v2' -> ('v2', None); 'pred_multiply_t0.25' -> ('multiply', 0.25).
    full=True -> (mode, tau, ttc); 'pred_multiply_sigma_ttc' -> ('multiply', 'sigma', True)."""
    if rule == "v2":
        return ("v2", None, False) if full else ("v2", None)
    parts = rule.split("_")
    ttc = len(parts) == 4 and parts[3] == "ttc"
    if ttc:
        parts = parts[:3]
    if (len(parts) != 3 or parts[0] != "pred" or parts[1] not in ("multiply", "replace")
            or not (parts[2].startswith("t") or parts[2] == "sigma")):
        raise ValueError(f"corridor_select_rule {rule!r}: expected v2 | pred_<multiply|replace>_<t<tau>|sigma>[_ttc]")
    tau = "sigma" if parts[2] == "sigma" else float(parts[2][1:])
    return (parts[1], tau, ttc) if full else (parts[1], tau)


@torch.no_grad()
def corridor_verdict(traj: torch.Tensor, status: torch.Tensor, sdf: torch.Tensor, occ: torch.Tensor,
                     sigma: torch.Tensor = None, ttc: bool = False):
    """traj [B, K, 8, 3] candidates (NAVSIM ego frame); status [B, 4] = (vx, vy, ax, ay) of the rear axle;
    sdf [B, H, 200] predicted static SDF (m); occ [B, P, H, 200] predicted occupancy probability (P = 8 swept, 40 or 49).
    -> (sdf_min [B, K], occ_max [B, K]), as tools/corridor/corr_verdict.py.
    km 10/18: sigma [B, H, 200] -> sdf_min is min of sdf / sigma (z units); ttc -> also ttc_max [B, K] (3rd output)."""
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
    big = CH.CLIP
    if sigma is not None:
        sv, _ = CH._sample(sigma.float()[:, None], pc.reshape(b, 1, -1, 2))
        v, big = v / sv, CH.CLIP / CH.SIGMA_MIN
    sdf_min = torch.where(inr, v, torch.full_like(v, big)).reshape(b, k, -1).amin(-1)
    npl = occ.shape[1]
    plane = CH.occ_plane_index(npl, dev)
    v, inr = CH._sample(occ.float()[:, plane], pf.permute(0, 2, 1, 3, 4).reshape(b, 40, -1, 2))
    occ_max = torch.where(inr, v, torch.zeros_like(v)).reshape(b, 40, k, 15).amax(-1).amax(1)
    if not ttc:
        return sdf_min, occ_max
    # TTC as corridor_head.corridor_verdict_loss: front half (4 x 3 points), shifted by speed * dt, agents at t + dt
    spd = st[..., 3]
    gx, gy = torch.meshgrid(torch.linspace(0.0, CH.FRONT, 4, device=dev),
                            torch.linspace(-CH.HALF_W, CH.HALF_W, 3, device=dev), indexing="ij")
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    mov = (spd >= 5e-3).float()                                                                 # [b, k, 40]
    steps = torch.arange(40, device=dev)
    ttc_max = torch.zeros(b, k, device=dev)
    for dd in (3, 6, 9):
        sh = (spd * dd * 0.1)[..., None]
        tq = torch.stack([x[..., None] + c * (gx + sh) - s * gy, y[..., None] + s * (gx + sh) + c * gy], -1)
        tq = tq.permute(0, 2, 1, 3, 4).reshape(b, 40, -1, 2)
        tv, ti = CH._sample(occ.float()[:, CH.ttc_plane_index(npl, steps + dd)], tq)
        tv = torch.where(ti, tv, torch.zeros_like(tv)).reshape(b, 40, k, 12).amax(-1).permute(0, 2, 1)   # [b, k, 40]
        ttc_max = torch.maximum(ttc_max, (tv * mov).amax(-1))
    return sdf_min, occ_max, ttc_max


@torch.no_grad()
def corridor_select(predictions: Dict[str, torch.Tensor], status: torch.Tensor, rule: str) -> Dict[str, torch.Tensor]:
    """predictions of ParaSSRModel (anchor planner + corridor head) -> {trajectory [B, 8, 3], index [B]} of ``rule``."""
    mode, tau, use_ttc = parse_rule(rule, full=True)
    offset = predictions["trajectory_offset"].float()
    traj = predictions["trajectory_anchors"].float()[None] + offset                              # [B, K, 8, 3]
    fin = predictions["plan_final_rewards"].float()
    if mode == "v2":
        score = fin
    else:
        sigma = None
        if tau == "sigma":
            assert "corridor_sdf_sigma" in predictions, f"{rule}: the run has no sigma head (corridor_sdf_sigma)"
            sigma = predictions["corridor_sdf_sigma"]
        out = corridor_verdict(traj, status, predictions["corridor_sdf"],
                               torch.sigmoid(predictions["corridor_occ"].float()), sigma=sigma, ttc=use_ttc)
        sdf_min, occ_max = out[:2]
        p_dac = CH.phi(sdf_min) if sigma is not None else torch.sigmoid((sdf_min / tau).clamp(-50, 50))
        p_nc = (1 - occ_max).clamp(0, 1)
        corr = 0.5 * torch.log(p_nc + EPS) + 0.5 * torch.log(p_dac + EPS)
        if use_ttc:
            corr = corr + 0.5 * torch.log((1 - out[2]).clamp(0, 1) + EPS)
        if mode == "multiply":
            score = fin + corr
        else:
            im, sim = predictions["im_rewards"].float(), predictions["sim_rewards"].float()
            ep, ttc, cf = sim[:, 2], sim[:, 3], sim[:, 4]
            score = 0.1 * torch.log(im + EPS) + corr + torch.log(5 * ttc + 2 * cf + 5 * ep + EPS)
    idx = score.argmax(1)
    return {"trajectory": traj[torch.arange(traj.shape[0], device=traj.device), idx], "index": idx}
