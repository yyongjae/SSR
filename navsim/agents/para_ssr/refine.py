"""Select, THEN refine: a second stage that refines only the candidate(s) the selector picked.

WHY.  Measured on navtest with the scorer's own per-candidate table (tools/plan_v2/diagnose_arms.py):

    the pool is not the problem   best-of-256 is 98.6-99.5 PDMS on every arm, the deployed pick 87-88
    the pick is ~ its anchor      |offset| of the picked candidate is 0.4 m on every arm -- 0.44 m even
                                  on feas_gt, whose corridor cost moved the average candidate 7.7 m
    set-level refinement costs    the corridor cost on many candidates collapses them (feas_gt: end
    diversity                     spread 10.6 -> 2.7 m, scenes holding a PDMS-1 candidate 88% -> 53%)
                                  and, read from a predicted field, roughens them (offset 2nd diff
                                  0.103 -> 0.119 m; EC 83.3 -> 81.4 even when the anchor is kept)

So refining the SET before selection spends the corridor where nothing is emitted, and pays for it in
diversity.  Diversity is needed BEFORE selection (the right mode must be in the pool); refinement is
needed AFTER it (on what is emitted).  This module is the after: stage 1 (AnchorPlanner) keeps its
diverse pool and picks; stage 2 refines the pick alone, against the same objects the design asks
for -- the merged corridor on the model's own field, the pushing force along the route, lane keeping,
and continuity -- now applied to the trajectory that is actually emitted.

    input   the picked candidate's planner query h_k, its trajectory (DETACHED: stage 2 must not move
            stage 1), the corridor clearance of its footprint at every step on the static and the
            per-step agent field (detached), and the ego status (v, a)
    output  xy + delta (delta from an MLP whose last layer starts at zero), heading from the path
            (``path_poses``) -- never a free variable

Training refines two candidates per scene: the selector's own pick (the deployment distribution) and
the WTA winner (the anchor nearest the human, which also gets imitation).  At inference the pick is
refined and the corridor stays ABSOLUTE: a refinement that leaves the corridor where the unrefined
pick was inside it is discarded (``refine_fallback``).

``refine_objective`` is the objective both the learned head (training loss) and the test-time
optimiser (tools/plan_v2/refine_upper_bound.py) minimise, so the two measure the same thing.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feasibility import REAR_TO_CENTER, project_on_polyline, sample_field, ttc_samples
from .modules.kinematics import bezier_xyyaw
from .plan_map import footprint_corners

DT = 0.5


def path_poses(xy: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """xy [..., T, 2] -> poses [..., T, 3] with the heading FROM THE PATH (bezier tangent).

    The heading is never a free variable of stage 2.  Left free, the corridor gradient rotates the
    footprint instead of moving it -- measured with test-time descent on D+'s picks: headings off by
    up to 2.0 rad, and PDMS 88.11 -> 78.64 (NC 93.7, TTC 85.1) because the tracker follows a heading
    the path does not have.  Stage 1 runs heading_from_xy too, so the identity start still holds.
    """
    return bezier_xyyaw(xy, base[..., 2])


def merged_clearance(traj: torch.Tensor, static: torch.Tensor, dyn: Optional[torch.Tensor], pc_range,
                     margin: float, dyn_margin: float) -> torch.Tensor:
    """``traj`` [B, R, T, 3] -> ``[B, R, T, 4]``: min(Phi_static - margin, Phi_dyn(t) - dyn_margin) per corner.

    The user's merged corridor.  ``dyn`` [B, T', H, W] is the per-step agent field; steps beyond T'
    reuse its last frame.  Without it, the static term alone.
    """
    corners = footprint_corners(traj)                                                  # [B, R, T, 4, 2]
    phi = sample_field(static, corners, pc_range) - margin
    if dyn is not None:
        t = traj.shape[2]
        per_t = [sample_field(dyn[:, min(i, dyn.shape[1] - 1)], corners[:, :, i], pc_range) for i in range(t)]
        phi = torch.minimum(phi, torch.stack(per_t, dim=2) - dyn_margin)
    return phi


def _route_terms(traj, route, human):
    """End-centre progress along the route and distance to it.  -> (s [B,R], s0 [B,1], ds_h [B,1] or None, d [B,R])."""
    b = traj.shape[0]
    end = traj[:, :, -1]
    cand = end[..., :2] + REAR_TO_CENTER * torch.stack([end[..., 2].cos(), end[..., 2].sin()], -1)
    s = project_on_polyline(cand, route)
    start = torch.tensor([REAR_TO_CENTER, 0.0], device=traj.device, dtype=traj.dtype).expand(b, 1, 2)
    s0 = project_on_polyline(start.contiguous(), route)
    ds_h = None
    if human is not None:
        hc = human[:, -1, :2] + REAR_TO_CENTER * torch.stack([human[:, -1, 2].cos(), human[:, -1, 2].sin()], -1)
        ds_h = project_on_polyline(hc[:, None], route) - s0
    valid = ~torch.isnan(route[..., 0])
    ln = torch.nan_to_num(route, nan=0.0)
    a_, b_ = ln[:, :-1], ln[:, 1:]
    ok = valid[:, :-1] & valid[:, 1:]
    d = b_ - a_
    tt = (((cand[:, :, None] - a_[:, None]) * d[:, None]).sum(-1) / ((d ** 2).sum(-1)[:, None] + 1e-9)).clamp(0, 1)
    dist = (cand[:, :, None] - (a_[:, None] + tt[..., None] * d[:, None])).norm(dim=-1)
    dist = dist.masked_fill(~ok[:, None], float("inf")).amin(-1)
    dist = torch.where(torch.isfinite(dist), dist, torch.zeros_like(dist))
    return s, s0, ds_h, dist


def refine_objective(traj: torch.Tensor, base: torch.Tensor, static: torch.Tensor, dyn: Optional[torch.Tensor],
                     cfg, route: Optional[torch.Tensor] = None, human: Optional[torch.Tensor] = None,
                     status: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """``traj`` [B, R, T, 3] (refined, with grad), ``base`` [B, R, T, 3] (the picked candidate) -> (cost [B, R], logs).

    corridor     softplus barrier on the merged clearance, margins as the corridor cost trains;
                 ``refine_corridor_hinge``: relu(-phi)^2 instead -- zero force once the footprint
                 clears the margin.  The softplus tail keeps pushing a pick that is already inside toward
                 the middle of the drivable area: on E it moved 95% of the lane-keeping losses off the
                 route centerline (0.38 -> 0.75 m) while the pick was already in the corridor
    progress     relu(1 - ds / ds_human) along the route; needs ``human`` (training) -- the test-time
                 optimiser has no human and leaves it out rather than invent a target
    lateral      relu(d - free)^2, end centre to the route centerline
    smooth       the refinement must not add roughness: |second difference of delta|^2 over time
    continuity   first-step speed vs v + a dt/2 measured, dead zone ``refine_continuity_tol``
    trust        |delta|^2, a weak trust region around the pick
    """
    g = lambda k, d: float(getattr(cfg, k, d))                                    # noqa: E731
    beta = g("feas_beta", 0.2)
    logs: Dict[str, torch.Tensor] = {}
    phi = merged_clearance(traj, static, dyn, cfg.map_pc_range, g("refine_margin", 0.4), g("refine_dyn_margin", 0.2))
    if bool(getattr(cfg, "refine_corridor_hinge", False)):
        corridor = (F.relu(-phi) ** 2).sum(dim=(2, 3))                                     # [B, R]
    else:
        corridor = (F.softplus(-phi / beta) * beta).sum(dim=(2, 3))                        # [B, R]
    cost = g("refine_corridor_weight", 1.0) * corridor
    logs["refine/corridor"] = corridor.detach().mean()

    if route is not None and (g("refine_progress_weight", 0.0) > 0 or g("refine_lateral_weight", 0.0) > 0):
        has = ~torch.isnan(route[:, :1, 0])                                                 # [B, 1]
        s, s0, ds_h, dist = _route_terms(traj, route, human)
        if ds_h is not None and g("refine_progress_weight", 0.0) > 0:
            moving = (has & (torch.nan_to_num(ds_h) >= 2.0)).float()
            short = F.relu(1.0 - torch.nan_to_num((s - s0) / torch.nan_to_num(ds_h, nan=1.0).clamp_min(2.0), nan=1.0))
            prog = short * moving
            cost = cost + g("refine_progress_weight", 0.0) * prog
            logs["refine/progress"] = prog.detach().mean()
        if g("refine_lateral_weight", 0.0) > 0:
            lat = F.relu(dist - g("refine_lateral_free", 1.0)) ** 2 * has.float()
            cost = cost + g("refine_lateral_weight", 0.0) * lat
            logs["refine/lateral"] = lat.detach().mean()

    # Do-no-harm toward the other agents.  Test-time descent on D+'s picks with the corridor alone
    # bought DAC (+0.6) and paid more in NC (-1.0) and TTC (-2.6): moving away from the road edge moves
    # toward the agents, and the per-step clearance (margin 0.2) does not see what the scorer's TTC
    # sees -- the footprint pushed forward at its own velocity.  So stage 2 may not make the TTC
    # clearance (feasibility.ttc_samples, both samples) of the pick WORSE where it is tight.
    if dyn is not None and g("refine_ttc_weight", 0.0) > 0:
        m_t = g("refine_ttc_margin", 1.0)
        c_new = ttc_samples(traj, dyn, cfg.map_pc_range)
        with torch.no_grad():
            c_old = ttc_samples(base.detach(), dyn, cfg.map_pc_range)
        harm = F.relu(torch.clamp(c_old, max=m_t) - c_new).pow(2).sum(dim=(2, 3, 4))      # [B, R]
        cost = cost + g("refine_ttc_weight", 0.0) * harm
        logs["refine/ttc_harm"] = harm.detach().mean()

    delta = traj[..., :2] - base[..., :2].detach()
    rough = (torch.diff(delta, n=2, dim=2) ** 2).sum(-1).sum(-1)                               # [B, R]
    cost = cost + g("refine_smooth_weight", 1.0) * rough
    logs["refine/rough"] = rough.detach().mean()
    trust = (delta ** 2).sum(-1).mean(-1)
    cost = cost + g("refine_trust_weight", 0.05) * trust
    logs["refine/delta_m"] = delta.detach().norm(dim=-1).mean()

    if status is not None and g("refine_continuity_weight", 0.0) > 0:
        v, a = status[:, 0:2].float(), status[:, 2:4].float()
        v_meas = (v + 0.5 * DT * a).norm(dim=-1)[:, None]                                    # [B, 1]
        v_plan = traj[:, :, 0, :2].norm(dim=-1) / DT
        mis = F.softplus(((v_plan - v_meas).abs() - g("refine_continuity_tol", 0.5)) / beta) * beta
        cost = cost + g("refine_continuity_weight", 0.0) * mis
        logs["refine/start_speed_err"] = (v_plan - v_meas).detach().abs().mean()
    return cost, logs


class SelectedRefiner(nn.Module):
    """Stage 2: refine the picked candidate(s).  See the module docstring."""

    def __init__(self, embed_dims: int, pc_range, fut_ts: int = 8, hidden: int = 256,
                 margin: float = 0.4, dyn_margin: float = 0.2):
        super().__init__()
        self.fut_ts, self.pc_range = fut_ts, tuple(pc_range)
        self.margin, self.dyn_margin = float(margin), float(dyn_margin)
        d_in = embed_dims + fut_ts * 3 + fut_ts * 4 * 2 + 4
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, fut_ts * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, traj: torch.Tensor, static: torch.Tensor, dyn: Optional[torch.Tensor],
                status: torch.Tensor) -> torch.Tensor:
        """h [B, R, C], traj [B, R, T, 3], static [B, H, W], dyn [B, T', H, W] | None, status [B, 4] -> [B, R, T, 3]."""
        b, r = traj.shape[:2]
        base = traj.detach().float()
        with torch.no_grad():
            st = sample_field(static.detach().float(), footprint_corners(base), self.pc_range)          # [B,R,T,4]
            if dyn is not None:
                corners = footprint_corners(base)
                dy = torch.stack([sample_field(dyn[:, min(i, dyn.shape[1] - 1)].detach().float(), corners[:, :, i],
                                               self.pc_range) for i in range(self.fut_ts)], dim=2)
            else:
                dy = torch.full_like(st, 10.0)
        x = torch.cat([h.float(), base.reshape(b, r, -1), st.clamp(-10, 10).reshape(b, r, -1),
                       dy.clamp(-10, 10).reshape(b, r, -1), status.float()[:, None].expand(b, r, 4)], dim=-1)
        delta = self.mlp(x.to(self.mlp[0].weight.dtype)).float().view(b, r, self.fut_ts, 2)
        return path_poses(base[..., :2] + delta, base)


def refine_select(model_refiner: SelectedRefiner, predictions: Dict[str, torch.Tensor], h: torch.Tensor,
                  status: torch.Tensor, cfg, gt_trajectory: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Run stage 2 on the predictions of stage 1 (after any field gate).

    Training: refine [the pick, the WTA winner]; inference: the pick, with the absolute-corridor fallback.
    """
    anchors = predictions["trajectory_anchors"].to(predictions["trajectory_offset"].dtype)
    all_traj = anchors.unsqueeze(0) + predictions["trajectory_offset"]                     # [B, K, T, 3]
    b, k = all_traj.shape[:2]
    pick = predictions["plan_topk_index"][:, :1]                                            # [B, 1]
    idx = pick
    if gt_trajectory is not None:
        flat = anchors.reshape(1, k, -1)
        wta = torch.norm(flat - gt_trajectory.reshape(b, 1, -1).to(flat.dtype), dim=2).argmin(1, keepdim=True)
        idx = torch.cat([pick, wta], dim=1)                                                 # [B, 2]
    r = idx.shape[1]
    base = all_traj.gather(1, idx[:, :, None, None].expand(-1, -1, *all_traj.shape[2:]))
    hk = h.gather(1, idx[:, :, None].expand(-1, -1, h.shape[-1]))
    static = predictions["drivable_sdf_pred"]
    dyn = predictions.get("agent_sdf_pred")
    refined = model_refiner(hk, base, static, dyn, status)
    out = {"refine_index": idx, "refine_base": base, "refine_trajectory": refined}
    if gt_trajectory is None:
        final = refined[:, 0]
        if bool(getattr(cfg, "refine_fallback", True)):
            with torch.no_grad():
                pc = cfg.map_pc_range
                m, dm = float(getattr(cfg, "feas_select_margin", 0.1)), float(getattr(cfg, "feas_select_dyn_margin", -1e9))
                c_new = merged_clearance(refined[:, :1].float(), static.float(), dyn.float() if dyn is not None and dm > -1e8 else None,
                                         pc, m, dm).amin(dim=(2, 3))[:, 0]
                c_old = merged_clearance(base[:, :1].float(), static.float(), dyn.float() if dyn is not None and dm > -1e8 else None,
                                         pc, m, dm).amin(dim=(2, 3))[:, 0]
                bad = (c_new < 0) & (c_new < c_old)
            final = torch.where(bad[:, None, None], base[:, 0].to(final.dtype), final)
            out["refine_fallback_used"] = bad
        out["trajectory_unrefined"] = predictions["trajectory"]
        out["trajectory"] = final.to(predictions["trajectory"].dtype)
    return out


def refine_losses(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                  features: Dict[str, torch.Tensor], cfg) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Objective on the refined [pick, WTA winner] + imitation of the human on the WTA winner."""
    if "refine_trajectory" not in predictions:
        return None, {}
    refined, base = predictions["refine_trajectory"], predictions["refine_base"]
    pred = str(getattr(cfg, "feas_field", "gt")) == "pred"
    static = (predictions["drivable_sdf_pred"] if pred else targets["drivable_sdf_bev"]).detach().float()
    dyn = predictions.get("agent_sdf_pred") if pred else targets.get("agent_sdf_bev")
    dyn = dyn.detach().float() if dyn is not None else None
    route = targets.get("route_centerline")
    route = route.float() if route is not None else None
    human = targets["trajectory"].float()
    nc = int(getattr(cfg, "num_navi_cmd", 4))
    status = features["status_feature"][:, nc:nc + 4].float()
    cost, logs = refine_objective(refined, base, static, dyn, cfg, route, human, status)
    loss = cost.mean()
    imit = F.l1_loss(refined[:, -1], human)
    loss = loss + float(getattr(cfg, "refine_imitation_weight", 1.0)) * imit
    logs["refine/imitation_l1"] = imit.detach()
    with torch.no_grad():
        c_b = merged_clearance(base[:, :1].float(), static, dyn, cfg.map_pc_range, 0.0, 0.0).amin(dim=(2, 3))
        c_r = merged_clearance(refined[:, :1].float(), static, dyn, cfg.map_pc_range, 0.0, 0.0).amin(dim=(2, 3))
        logs["refine/pick_in_corridor_before"] = (c_b >= 0).float().mean()
        logs["refine/pick_in_corridor_after"] = (c_r >= 0).float().mean()
    return float(getattr(cfg, "refine_weight", 1.0)) * loss, logs
