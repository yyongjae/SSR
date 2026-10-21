"""One differentiable feasibility cost, wired to the offset head, the score head and the field.

The planner emits ``anchor_k + offset_k`` for K anchors and picks one by a weighted rule.  Nothing in
training says the emitted footprint has to stay on the road: the offset head is supervised by a
Winner-Take-All L1 to the human trajectory, and the score heads by the precomputed PDM scores OF THE
ANCHORS -- a label for a trajectory the model does not output.  Measured on navtest, 59.7% of the
scenes where the emitted trajectory leaves the drivable area had an anchor that was inside it: the
refinement step creates the failure, and the median penetration is 0.32 m at a single corner.

This module adds a cost on a signed-distance field ``Phi``

    d_{k,t,c} = Phi(corner c of (anchor_k + offset_k) at step t)
    c_k       = sum_{t,c} softplus((margin - d) / beta) * beta

and sends it three ways:

    offset head   sum_k w_k c_k,  w_k = softmax(selection score), so high-ranked candidates are
                  pulled inside hardest                                        [cfg.feas_to_offset]
    score head    the drivable-area head's target becomes exp(-c0_k / tau), with c0_k the SAME cost
                  at margin 0 -- DAC is "no corner outside", not "no corner within the buffer".  The
                  target is computed on the REFINED trajectory, so target and output finally
                  describe the same thing                                       [cfg.feas_to_score]
    field         with a predicted Phi the same gradient reaches the perception path, so it is
                  shaped by what planning needs                        [cfg.feas_field = "pred"]

Two fields, because DAC and NC are different metrics with different failure profiles (measured:
59.7% of DAC failures are created by the refinement step, but 67.6% of NC failures were already in
the anchor).  ``Phi_static`` is the drivable area, constant over the horizon.  ``Phi_dyn`` is one
field per ego step: the free space left by the other agents where they will BE at that step, so the
plan is bound by the model's own prediction of the scene rather than by its snapshot of it.  Each
field owns one score head -- static the drivable-area column, dynamic the collision column -- and
``para_ssr_loss`` drops both from the anchor-label BCE, since those labels describe the anchor.

``cfg.feas_field = "gt"`` takes Phi from the target instead and isolates the mechanism from the
quality of the field.  Either way the cost under the OTHER field is logged (detached): predicted
cost falling while ground-truth cost does not is the field collapsing to "everything is drivable",
not learning.  That diagnostic caught exactly this in the first attempt.

RESOLUTION, and a deliberate asymmetry.  Bilinear sampling of an exact SDF has a median error of
0.142 m on the 0.64 m BEV grid, 0.071 m at 0.32 m, 0.035 m at 0.16 m; the phenomenon is a 0.32 m
penetration.  The TARGET is therefore rasterised on a ``feas_target_upsample``-times finer grid
(default 4x -> 0.16 m, 11% noise): it is a label, and labels carry no fairness cost -- SV-WAM reads
the official PDMDrivableMap, PLUTO rasterises at 0.25 m.  The PREDICTED field stays on the BEV grid,
because a head that outputs finer than every other head in the model would buy accuracy with
resolution rather than with the mechanism under test.  So the predicted field has a 0.142 m
representational floor, 44% of the phenomenon, and the "pred" arm is expected to trail the "gt" arm
by whatever that floor costs.  That gap is a result, not something to engineer away; a teacher with
a finer field (or richer sensors) is the follow-up it points to.

GROUNDING.  A plain L1 over the whole field spends its budget where the answer is trivial: 37% of
cells are saturated and 75% are more than 2 m from the boundary, and the first attempt converged to
a 0.92 m mean error -- three times the phenomenon.  The cost's sensitivity to the field is
``d c / d Phi = -sigmoid((margin - Phi) / beta)``, so the grounding loss is weighted by that same
quantity evaluated at the ground truth.  Deep-interior cells keep a floor weight, because a field
that drifts DOWN there would invent obstacles.

GEOMETRY.  Rows run along SSR y (forward, pc_range[1]..pc_range[4]), columns along SSR x (lateral,
pc_range[0]..pc_range[3]).  Trajectories are NAVSIM (x forward, y left), so corners are swapped
exactly as the vector targets are (``para_ssr_targets.py:583``: SSR x = -navsim y, SSR y = navsim x).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.para_ssr.cache_key import cache_key
from navsim.agents.para_ssr.plan_map import footprint_corners
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder

# indices into anchor_planner.SIM_KEYS  (head_mode = "wote")
NC_INDEX = 0        # no_at_fault_collisions   <- owned by the dynamic field
DAC_INDEX = 1       # drivable_area_compliance <- owned by the static field
# index into anchor_planner.CORRIDOR_KEYS (head_mode = "corridor"): ONE head, the merged corridor
FEAS_INDEX = 0


def target_shape(cfg) -> Tuple[int, int]:
    """Grid of the ground-truth field.  Finer than the BEV on purpose -- see the module docstring."""
    m = int(getattr(cfg, "feas_target_upsample", 4))
    return int(cfg.bev_h) * m, int(cfg.bev_w) * m


# --------------------------------------------------------------------------- target

def rasterize_bev_sdf(polygon, pc_range: Sequence[float], h: int, w: int,
                      clip: float = 8.0) -> np.ndarray:
    """Drivable polygon in NAVSIM ego axes -> float32 SDF ``[h, w]``, positive inside.

    A cell is drivable when its CENTRE is inside, matching ``plan_map.rasterize_sdf``; cv2.fillPoly
    would also take every cell an edge merely clips, a half-cell outward bias.
    """
    import cv2
    import shapely

    y0, y1 = float(pc_range[1]), float(pc_range[4])          # SSR y = forward
    x0, x1 = float(pc_range[0]), float(pc_range[3])          # SSR x = lateral
    res_y, res_x = (y1 - y0) / h, (x1 - x0) / w
    if polygon is None or polygon.is_empty:
        return np.full((h, w), -clip, np.float32)
    ssr_y = y0 + (np.arange(h) + 0.5) * res_y
    ssr_x = x0 + (np.arange(w) + 0.5) * res_x
    gy, gx = np.meshgrid(ssr_y, ssr_x, indexing="ij")
    shapely.prepare(polygon)
    mask = shapely.contains_xy(polygon, gy.ravel(), (-gx).ravel())     # back to NAVSIM axes
    mask = mask.reshape(h, w).astype(np.uint8)
    if mask.all() or not mask.any():
        return np.full((h, w), clip if mask.all() else -clip, np.float32)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    res = 0.5 * (abs(res_x) + abs(res_y))
    sdf = np.where(mask > 0, inside - 0.5, -(outside - 0.5)) * res
    return np.clip(sdf, -clip, clip).astype(np.float32)


def agent_free_sdf(scene, cur_idx: int, cfg, h: int, w: int, clip: float = 8.0) -> np.ndarray:
    """Free space left by the other agents at each ego step -> ``[T, h, w]``, positive = free.

    Step t is the annotation frame ``cur_idx + 1 + t``, which is where the ego trajectory's step t
    also lands (both are 0.5 s apart), so the cost compares the plan against the scene at the same
    instant.  Boxes are taken to the CURRENT ego frame; an agent that leaves the log simply stops
    contributing, which is the same convention ``_track_future`` uses for the motion targets.
    """
    import cv2

    T = int(cfg.trajectory_sampling.num_poses)
    y0, y1 = float(cfg.map_pc_range[1]), float(cfg.map_pc_range[4])
    x0, x1 = float(cfg.map_pc_range[0]), float(cfg.map_pc_range[3])
    res = 0.5 * (abs(x1 - x0) / w + abs(y1 - y0) / h)
    poses = [np.asarray(f.ego_status.ego_pose, dtype=np.float64) for f in scene.frames]
    cp = poses[cur_idx]
    c, s_ = np.cos(cp[2]), np.sin(cp[2])

    out = np.full((T, h, w), clip, np.float32)
    for t in range(T):
        f_idx = cur_idx + 1 + t
        if f_idx >= len(scene.frames):
            out[t] = out[t - 1] if t else clip
            continue
        ann = scene.frames[f_idx].annotations
        boxes = np.asarray(ann.boxes, dtype=np.float64)
        mask = np.zeros((h, w), np.uint8)
        fp = poses[f_idx]
        fc, fs = np.cos(fp[2]), np.sin(fp[2])
        for b in boxes:
            # box frame -> global -> current ego (NAVSIM axes: x forward, y left)
            gx, gy = fc * b[0] - fs * b[1] + fp[0], fs * b[0] + fc * b[1] + fp[1]
            dx, dy = gx - cp[0], gy - cp[1]
            lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy
            yaw = float(b[6]) + fp[2] - cp[2]
            hl, hw_ = 0.5 * float(b[3]), 0.5 * float(b[4])
            cc, ss = np.cos(yaw), np.sin(yaw)
            corner = np.array([[hl, hw_], [hl, -hw_], [-hl, -hw_], [-hl, hw_]])
            pts = corner @ np.array([[cc, ss], [-ss, cc]]) + [lx, ly]        # [4, 2] NAVSIM
            # NAVSIM -> SSR -> pixel (col = lateral, row = forward)
            col = (-pts[:, 1] - x0) / (x1 - x0) * w
            row = (pts[:, 0] - y0) / (y1 - y0) * h
            cv2.fillConvexPoly(mask, np.stack([col, row], -1).round().astype(np.int32), 1)
        if not mask.any():
            continue
        free = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        occ = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        out[t] = np.clip(np.where(mask > 0, -(occ - 0.5), free - 0.5) * res, -clip, clip)
    return out


class DrivableFieldTargetBuilder(AbstractTargetBuilder):
    """``drivable_sdf_bev`` float16 ``[H, W]`` -- metres, positive inside, on the upsampled grid."""

    def __init__(self, config):
        self._config = config

    def get_unique_name(self) -> str:
        cfg = self._config
        h, w = target_shape(cfg)
        return cache_key("para_ssr_drivable_sdf_bev", (
            ("pc_range", tuple(float(v) for v in cfg.map_pc_range)),
            ("grid", (h, w)),
            ("clip", float(getattr(cfg, "feas_clip", 8.0))),
            ("layers", "pdm_drivable_v1"),
        ))

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        # the polygon query is plan_map's, reused so the field cannot drift from the DAC metric
        from navsim.agents.para_ssr.plan_map import DrivableAreaTargetBuilder

        cfg = self._config
        cur = scene.scene_metadata.num_history_frames - 1
        poly = DrivableAreaTargetBuilder(cfg).drivable_polygon(scene, cur)
        h, w = target_shape(cfg)
        clip = float(getattr(cfg, "feas_clip", 8.0))
        out = {"drivable_sdf_bev": torch.from_numpy(
            rasterize_bev_sdf(poly, cfg.map_pc_range, h, w, clip)).half()}
        if float(getattr(cfg, "feas_dyn_weight", 0.0)) > 0:
            out["agent_sdf_bev"] = torch.from_numpy(
                agent_free_sdf(scene, cur, cfg, h, w, clip)).half()          # [T, h, w]
        return out


# --------------------------------------------------------------------------- field head

class DrivableFieldHead(nn.Module):
    """``bev_embed`` [B, bev_h*bev_w, C] -> signed distance field [B, bev_h, bev_w] in metres.

    Same grid as every other head.  Deliberately shallow: the question is whether the BEV already
    carries the region and the model simply never had to say so, and a deep decoder would turn that
    into "can a decoder learn it".
    """

    def __init__(self, embed_dims: int, bev_h: int, bev_w: int, hidden: int = 128,
                 clip: float = 8.0, dyn_steps: int = 0, prior_channels: int = 0):
        super().__init__()
        self.bev_h, self.bev_w, self.clip = int(bev_h), int(bev_w), float(clip)
        self.dyn_steps = int(dyn_steps)
        # ``prior_channels`` > 0: the head also reads what the OTHER heads already say about the same
        # space (field_priors: the seg head's drivable probability, the det/motion head's forecast
        # agent occupancy per step), so the corridor is built from perception rather than beside it.
        self.prior_channels = int(prior_channels)
        self.trunk = nn.Sequential(
            nn.Conv2d(embed_dims + self.prior_channels, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
        )
        # one plane for the drivable area, one per ego step for the agents.  A shared trunk, because
        # both read the same scene; only the last 1x1 separates them, so the dynamic planes cost
        # almost nothing beyond their own output channels.
        self.out = nn.Conv2d(hidden, 1 + self.dyn_steps, 1)

    def forward(self, bev_embed: torch.Tensor, priors: Optional[torch.Tensor] = None):
        bs, n, c = bev_embed.shape
        if n != self.bev_h * self.bev_w:
            raise ValueError(f"bev_embed has {n} cells, expected {self.bev_h}x{self.bev_w}")
        x = bev_embed.transpose(1, 2).reshape(bs, c, self.bev_h, self.bev_w).float()
        if self.prior_channels:
            if priors is None or priors.shape[1] != self.prior_channels:
                raise ValueError(f"field head expects {self.prior_channels} prior channels, got "
                                 f"{None if priors is None else tuple(priors.shape)}")
            x = torch.cat([x, priors.float()], dim=1)
        # tanh keeps the field in +-clip metres; an unbounded head drifts early in training
        y = self.clip * torch.tanh(self.out(self.trunk(x)) / self.clip)
        return (y[:, 0], y[:, 1:]) if self.dyn_steps else (y[:, 0], None)


# --------------------------------------------------------------------------- priors from perception

def field_priors(det_out: Optional[Dict[str, torch.Tensor]], seg_logits: Optional[torch.Tensor], cfg,
                 bev_h: int, bev_w: int, steps: int, topk: int = 40, soft: float = 0.3) -> Optional[torch.Tensor]:
    """What the other heads already say about the space the corridor describes -> ``[B, P, H, W]``.

    channel 0 (if ``seg_logits``)    the seg head's drivable probability -- the same region the static
                                      field measures the distance to
    channels 1..steps (if ``det_out``) forecast agent occupancy at each ego step: the det head's boxes,
                                      moved along EVERY motion mode of the motion head and weighted by
                                      detection score x mode probability, so a car the motion head
                                      thinks may turn occupies both futures in proportion

    Without these the field head had to re-infer from the BEV alone, with two conv layers, the region
    the seg head segments and the futures the motion head forecasts -- and the dynamic planes are 4 s
    of other agents' motion.  Everything is detached: the priors inform the field, the field's own
    losses must not reshape detection or segmentation (the same reason feas_cost_grad_to_field is off).
    Geometry follows agent_free_sdf: boxes in the current ego frame, SSR axes, step t = t+1 frames ahead.
    """
    chans = []
    pc = cfg.map_pc_range
    x0, y0, x1, y1 = float(pc[0]), float(pc[1]), float(pc[3]), float(pc[4])
    if seg_logits is not None:
        chans.append(torch.sigmoid(seg_logits.detach().float())[:, None])
    if det_out is not None and steps > 0:
        box = det_out["all_bbox_preds"][-1].detach().float()                   # [B, Q, 10] SSR physical code
        score = det_out["all_cls_scores"][-1].detach().float().sigmoid().amax(-1)          # [B, Q]
        traj = det_out["traj_preds"].detach().float()                            # [B, Q, M, Tm, 2] per-step offsets
        pmode = det_out["traj_cls_preds"].detach().float().softmax(-1)           # [B, Q, M]
        b, q = score.shape
        k = min(int(topk), q)
        top = score.topk(k, dim=-1).indices                                      # [B, K]
        g = lambda t: t.gather(1, top.view(b, k, *([1] * (t.dim() - 2))).expand(b, k, *t.shape[2:]))  # noqa: E731
        box, score, traj, pmode = g(box), g(score), g(traj), g(pmode)
        m, tm = traj.shape[2], traj.shape[3]
        ctr = box[..., 0:2]                                                      # [B, K, 2] (x right, y forward)
        hw, hl = 0.5 * box[..., 2].exp(), 0.5 * box[..., 3].exp()                # half width / half length
        axis0 = F.normalize(torch.stack([box[..., 6], box[..., 7]], -1), dim=-1)          # (sin, cos) of SSR yaw
        fut = ctr[:, :, None, None] + traj.cumsum(dim=3)                          # [B, K, M, Tm, 2]
        step_len = traj.norm(dim=-1, keepdim=True)
        heading = torch.where(step_len > 0.25, traj / step_len.clamp_min(1e-6), axis0[:, :, None, None].expand_as(traj))
        w = (score[:, :, None] * pmode)                                          # [B, K, M]
        ys = torch.linspace(y0, y1, bev_h + 1, device=box.device)[:-1] + 0.5 * (y1 - y0) / bev_h
        xs = torch.linspace(x0, x1, bev_w + 1, device=box.device)[:-1] + 0.5 * (x1 - x0) / bev_w
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        cells = torch.stack([gx, gy], -1).view(1, 1, 1, -1, 2)                   # [1, 1, 1, HW, 2]
        for t in range(steps):
            tt = min(t, tm - 1)
            d = cells - fut[:, :, :, tt, None, :]                                # [B, K, M, HW, 2]
            u_ax = heading[:, :, :, tt, None, :]                                 # length axis
            u = (d * u_ax).sum(-1).abs()
            v = (d[..., 0] * u_ax[..., 1] - d[..., 1] * u_ax[..., 0]).abs()
            inside = torch.sigmoid((hl[:, :, None, None] - u) / soft) * torch.sigmoid((hw[:, :, None, None] - v) / soft)
            occ = 1.0 - torch.exp(torch.log1p(-(w[..., None] * inside).clamp(max=1 - 1e-4)).sum(dim=(1, 2)))
            chans.append(occ.view(b, 1, bev_h, bev_w))
    if not chans:
        return None
    return torch.cat(chans, dim=1)


# --------------------------------------------------------------------------- cost

def sample_field(field: torch.Tensor, pts_nav: torch.Tensor, pc_range: Sequence[float]) -> torch.Tensor:
    """``field`` [B, H, W] and NAVSIM points ``[B, ..., 2]`` -> sampled values ``[B, ...]``.

    Points outside the grid are clamped to the border by ``grid_sample(padding_mode="border")``:
    beyond the BEV there is no evidence either way, and a large negative would invent an obstacle.
    """
    b, h, w = field.shape
    y0, y1 = float(pc_range[1]), float(pc_range[4])
    x0, x1 = float(pc_range[0]), float(pc_range[3])
    ssr_x = -pts_nav[..., 1]                       # navsim y (left) -> SSR x (right)
    ssr_y = pts_nav[..., 0]                        # navsim x (forward) -> SSR y
    gx = 2.0 * (ssr_x - x0) / (x1 - x0) - 1.0      # grid_sample: last dim indexes W
    gy = 2.0 * (ssr_y - y0) / (y1 - y0) - 1.0
    shape = gx.shape[1:]
    grid = torch.stack([gx.reshape(b, -1), gy.reshape(b, -1)], dim=-1)[:, None]      # [B, 1, N, 2]
    out = F.grid_sample(field[:, None], grid.to(field.dtype), mode="bilinear",
                        padding_mode="border", align_corners=False)                  # [B, 1, 1, N]
    return out.reshape(b, *shape)


def corner_sdf(field: torch.Tensor, traj: torch.Tensor, pc_range: Sequence[float]) -> torch.Tensor:
    """``traj`` [B, K, T, 3] rear-axle poses -> sampled signed distance ``[B, K, T, 4]``."""
    return sample_field(field, footprint_corners(traj), pc_range)


def cost_from_sdf(sdf: torch.Tensor, margin: float, beta: float) -> torch.Tensor:
    """``[B, K, T, 4]`` -> ``[B, K]``.  softplus, not relu: a corner already outside must keep a
    gradient, and relu's kink makes the pull depend on where the corner happens to land."""
    return (F.softplus((margin - sdf) / beta) * beta).sum(dim=(2, 3))


def feasibility_losses(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                       cfg, iteration: int = 0) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """-> (loss, logs).  Returns (None, {}) when the term is off or its inputs are missing."""
    w_total = float(getattr(cfg, "feas_weight", 0.0))
    if w_total <= 0 or "trajectory_offset" not in predictions:
        return None, {}
    if "drivable_sdf_bev" not in targets:
        raise KeyError("feas_weight > 0 needs drivable_sdf_bev (DrivableFieldTargetBuilder)")

    # the planner's own gradient must not be fed a random field; ramp in after the grounding loss
    # has had a chance to converge.  Same shape as the kd schedule (readout/distill.py).
    warm = int(getattr(cfg, "feas_warmup_iters", 0))
    ramp = int(getattr(cfg, "feas_ramp_iters", 0))
    coef = 1.0 if iteration >= warm + ramp else (
        0.0 if iteration < warm else (iteration - warm) / max(ramp, 1))

    anchors = predictions["trajectory_anchors"].to(predictions["trajectory_offset"].dtype)
    traj = (anchors.unsqueeze(0) + predictions["trajectory_offset"]).float()      # [B, K, T, 3]
    gt_field = targets["drivable_sdf_bev"].float()
    pred_field = predictions.get("drivable_sdf_pred")
    source = str(getattr(cfg, "feas_field", "gt"))
    if source == "pred" and pred_field is None:
        raise KeyError("feas_field='pred' needs the DrivableFieldHead output")
    # The cost is monotone decreasing in Phi, so letting its gradient reach the field gives the model
    # a way to make the constraint cheap without moving a waypoint: lift the field.  Measured on 60
    # navtest scenes with the ground-truth field (tools/plan_v2/feas_objective_gate.py), a global
    # lift of +8 m takes mean_k c_k from 16.04 to 0.20 -- and the first "pred" arm found exactly that
    # (cost_used 0.100 while the same trajectories cost 15.85 under the ground truth, field_mae_near
    # rising 1.01 -> 1.42 m).  No scalar objective on the candidate costs repairs it: the candidate
    # ORDERING is nearly invariant to the field's level (AUC 0.923 / 0.937 / 0.932 at delta =
    # -2 / 0 / +8 m), so an ordering loss has no gradient to resist the lift, while a magnitude loss
    # is minimised by it.  Listwise CE, raw pairwise and standardised pairwise were all measured and
    # all put their optimum away from the true field.  The field is therefore identified by its
    # grounding loss alone, and the cost trains the trajectory.  Setting this flag reproduces the
    # collapse; it is kept because that ablation is the evidence.
    grad_to_field = bool(getattr(cfg, "feas_cost_grad_to_field", False))
    field = gt_field if source != "pred" else (pred_field if grad_to_field else pred_field.detach())

    pc_range = cfg.map_pc_range
    margin = float(getattr(cfg, "feas_margin", 0.4))
    beta = float(getattr(cfg, "feas_beta", 0.2))
    sdf = corner_sdf(field, traj, pc_range)                             # [B, K, T, 4]
    c = cost_from_sdf(sdf, margin, beta)                                # [B, K] buffer cost
    c0 = cost_from_sdf(sdf, 0.0, beta)                                  # [B, K] actual violation

    # dynamic field: one plane per ego step, so step t is scored against where the agents will be
    # at step t.  Same corners, same softplus; only the margin differs, because NC is an overlap
    # test and a lane-width buffer around every vehicle would make the plan unable to pass anything.
    dyn_w = float(getattr(cfg, "feas_dyn_weight", 0.0))
    c_dyn = c_dyn0 = sdf_dyn = None
    if dyn_w > 0:
        gt_dyn = targets.get("agent_sdf_bev")
        if gt_dyn is None:
            raise KeyError("feas_dyn_weight > 0 needs agent_sdf_bev (DrivableFieldTargetBuilder)")
        pred_dyn = predictions.get("agent_sdf_pred")
        if source == "pred" and pred_dyn is None:
            raise KeyError("feas_field='pred' with feas_dyn_weight > 0 needs the dynamic head output")
        dyn = (gt_dyn if source != "pred" else
               (pred_dyn if grad_to_field else pred_dyn.detach())).float()   # [B, T, H, W]
        dm = float(getattr(cfg, "feas_dyn_margin", 0.2))
        corners = footprint_corners(traj)                               # [B, K, T, 4, 2]
        per_t = [sample_field(dyn[:, t], corners[:, :, t], pc_range) for t in range(dyn.shape[1])]
        sdf_dyn = torch.stack(per_t, dim=2)                             # [B, K, T, 4]
        c_dyn = cost_from_sdf(sdf_dyn, dm, beta)
        c_dyn0 = cost_from_sdf(sdf_dyn, 0.0, beta)

    # ---- the merged corridor, for JUDGING -------------------------------------------------
    # Phi_corridor = min(Phi_static - m_static, Phi_dyn - m_dyn): one field, each obstacle class
    # keeping its own margin.  The margins differ for a reason -- a 0.4 m buffer around every
    # vehicle would make normal passing infeasible in a 3.5 m lane -- and subtracting them BEFORE
    # the min is what lets a single field carry both.  ``c_merge0`` is the same at margin 0: the
    # violation itself, which is what the feasible head has to predict.
    merged = str(getattr(cfg, "plan_head_mode", "wote")) == "corridor"
    c_merge = c_merge0 = None
    if merged:
        ms = float(getattr(cfg, "feas_margin", 0.4))
        md = float(getattr(cfg, "feas_dyn_margin", 0.2))
        phi = sdf - ms if sdf_dyn is None else torch.minimum(sdf - ms, sdf_dyn - md)
        hard = sdf if sdf_dyn is None else torch.minimum(sdf, sdf_dyn)
        c_merge = cost_from_sdf(phi, 0.0, beta)
        c_merge0 = cost_from_sdf(hard, 0.0, beta)

    # weighting over candidates.  detach: the weights say WHERE to spend the constraint; they must
    # not become a way for the model to make the constraint cheap by re-ranking.
    scheme = str(getattr(cfg, "feas_weighting", "softmax"))
    # rank by the UNGATED score where one exists: with corridor_first the gate would drive w_k to
    # zero for exactly the candidates the constraint should pull back in.
    rank = predictions.get("plan_rank_rewards")
    logits = (rank if rank is not None else predictions["plan_final_rewards"]).detach().float()
    if scheme == "softmax":
        w = torch.softmax(logits / float(getattr(cfg, "feas_temp", 1.0)), dim=-1)
    elif scheme == "topk":
        k = min(int(getattr(cfg, "feas_topk", 32)), logits.shape[-1])
        w = torch.zeros_like(logits).scatter_(1, logits.topk(k, dim=-1).indices, 1.0 / k)
    elif scheme == "uniform":
        w = torch.full_like(logits, 1.0 / logits.shape[-1])
    elif scheme == "select":
        w = select_weights(logits, predictions, targets, cfg)
    else:
        raise ValueError(f"unknown feas_weighting {scheme!r}")

    loss = c.new_zeros(())
    logs: Dict[str, torch.Tensor] = {"feas/coef": c.new_tensor(coef)}
    logs["feas/weight_support"] = (w > 0).float().sum(-1).mean()
    logs["feas/weight_eff_n"] = torch.exp(-(w * (w + 1e-12).log()).sum(-1)).mean()

    # Candidates OUTSIDE the refined set: hold them near their anchor.  Zeroing their geometric
    # gradient (select) is not enough -- the offset head is ONE MLP shared by all 256 queries, so
    # refining S moves the rest through the shared weights.  Measured on B2 (select, no hold) at
    # matched epochs against v2_r34: candidate spread 9.01 -> 8.36 m from epoch 5 to 10 (v2_r34
    # 15.21 -> 14.95), mean |offset| 5.64 -> 6.97 m (v2_r34 0.97 -> 1.17).  The hold is a hinge at
    # ``feas_offset_anchor_radius`` (2 m): below it the refinement is free -- v2_r34's own offsets
    # live there -- and 2 m is also where the anchor's TTC label stops describing the output
    # (ttc_label, band mode), so a held candidate keeps a label that is right for it.
    off = predictions["trajectory_offset"][..., :2].float().norm(dim=-1)          # [B, K, T]
    outside = (w == 0).float()                                                   # [B, K]
    inside = 1.0 - outside
    logs["feas/offset_mag_in"] = ((off.mean(-1) * inside).sum() / inside.sum().clamp_min(1.0)).detach()
    logs["feas/offset_mag_out"] = ((off.mean(-1) * outside).sum() / outside.sum().clamp_min(1.0)).detach()
    hold_w = float(getattr(cfg, "feas_offset_anchor_weight", 0.0))
    if hold_w > 0:
        r0 = float(getattr(cfg, "feas_offset_anchor_radius", 2.0))
        excess = F.relu(off - r0) ** 2                                           # [B, K, T]
        hold = (excess * outside[..., None]).sum() / (outside.sum() * off.shape[-1]).clamp_min(1.0)
        loss = loss + hold_w * hold
        logs["feas/offset_hold"] = hold.detach()
    if bool(getattr(cfg, "feas_to_offset", True)):
        off = (w * c).sum(dim=-1).mean()
        loss = loss + coef * float(getattr(cfg, "feas_offset_weight", 1.0)) * off
        logs["feas/offset"] = off.detach()
        if c_dyn is not None:
            offd = (w * c_dyn).sum(dim=-1).mean()
            loss = loss + coef * dyn_w * float(getattr(cfg, "feas_offset_weight", 1.0)) * offd
            logs["feas/offset_dyn"] = offd.detach()
    if bool(getattr(cfg, "feas_to_score", False)):
        tau = float(getattr(cfg, "feas_score_tau", 0.5))
        sw = float(getattr(cfg, "feas_score_weight", 1.0))
        rw = predictions["sim_rewards"]
        if merged:
            # ONE corridor, one verdict.  The road and the agents are the same question -- may the
            # ego BE here -- so the judgement takes the WORSE of the two at each corner and step,
            # each measured against its own margin, and a single head predicts it.  Summing the two
            # costs (what the offset path still does) is right for PUSHING, because both obstacles
            # should push at once; it is wrong for JUDGING, because a candidate is infeasible if
            # either one is violated, not twice as infeasible if both are.
            tgt = torch.exp(-c_merge0.detach() / tau)
            feas = rw[:, FEAS_INDEX].float().clamp(1e-6, 1 - 1e-6)
            sc = F.binary_cross_entropy(feas, tgt)
            loss = loss + coef * sw * sc
            logs["feas/score"] = sc.detach()
            logs["feas/score_target"] = tgt.mean()
            logs["feas/score_pred"] = feas.mean().detach()
        else:
            # DAC is "no corner outside", so the target uses the margin-0 cost.  anchor_plan_losses
            # must drop its own DAC column or the head gets two conflicting targets -- see
            # para_ssr_loss.py, which passes feas_skip_sim_index.
            tgt = torch.exp(-c0.detach() / tau)
            dac = rw[:, DAC_INDEX].float().clamp(1e-6, 1 - 1e-6)
            sc = F.binary_cross_entropy(dac, tgt)
            loss = loss + coef * sw * sc
            logs["feas/score"] = sc.detach()
            logs["feas/score_target"] = tgt.mean()
            if c_dyn0 is not None:
                tgt_n = torch.exp(-c_dyn0.detach() / tau)
                nc = rw[:, NC_INDEX].float().clamp(1e-6, 1 - 1e-6)
                scn = F.binary_cross_entropy(nc, tgt_n)
                loss = loss + coef * dyn_w * sw * scn
                logs["feas/score_nc"] = scn.detach()
                logs["feas/score_nc_target"] = tgt_n.mean()
    if pred_field is not None:
        # Ground the field wherever the cost is sensitive to it.  dc/dPhi = -sigmoid((margin-Phi)/beta),
        # so weight by that at the ground truth: cells the cost cannot feel get a floor, not zero,
        # because a field drifting DOWN deep inside would invent obstacles.
        band = float(getattr(cfg, "feas_ground_band", 1.0))
        floor = float(getattr(cfg, "feas_ground_floor", 0.05))
        # the target is finer than the head's grid; average it down to compare like with like.
        # An SDF is smooth away from the medial axis, so the cell mean is a fair coarse target.
        r = gt_field.shape[-1] // pred_field.shape[-1]
        gt_coarse = F.avg_pool2d(gt_field[:, None], r).squeeze(1) if r > 1 else gt_field
        gw = torch.sigmoid((margin + band - gt_coarse) / band).clamp_min(floor)
        elem = F.smooth_l1_loss(pred_field, gt_coarse, beta=0.5, reduction="none")
        fl = (gw * elem).sum() / gw.sum().clamp_min(1.0)
        loss = loss + float(getattr(cfg, "feas_field_weight", 1.0)) * fl
        if dyn_w > 0 and predictions.get("agent_sdf_pred") is not None:
            gd = targets["agent_sdf_bev"].float()
            rd = gd.shape[-1] // predictions["agent_sdf_pred"].shape[-1]
            gd_c = F.avg_pool2d(gd, rd) if rd > 1 else gd
            dmv = float(getattr(cfg, "feas_dyn_margin", 0.2))
            gwd = torch.sigmoid((dmv + band - gd_c) / band).clamp_min(floor)
            fld = (gwd * F.smooth_l1_loss(predictions["agent_sdf_pred"], gd_c, beta=0.5,
                                          reduction="none")).sum() / gwd.sum().clamp_min(1.0)
            loss = loss + float(getattr(cfg, "feas_field_weight", 1.0)) * dyn_w * fld
            logs["feas/field_l1_dyn"] = fld.detach()
        with torch.no_grad():
            near = (gt_coarse.abs() < 1.0)
            logs["feas/field_l1"] = fl.detach()
            logs["feas/field_mae_near"] = ((pred_field - gt_coarse).abs() * near).sum() / near.sum().clamp_min(1)

    with torch.no_grad():
        # the diagnostic that costs nothing: the same cost under the OTHER field.  Predicted cost
        # falling while the ground-truth cost does not is the model cheating, not learning.
        other = gt_field if source == "pred" else pred_field
        logs["feas/cost_used"] = c.mean()
        logs["feas/cost0_used"] = c0.mean()
        if other is not None:
            logs["feas/cost_other"] = cost_from_sdf(corner_sdf(other.float(), traj, pc_range),
                                                    margin, beta).mean()
        picked = logits.argmax(-1)
        rows = torch.arange(len(c), device=c.device)
        min_sdf = sdf.amin(dim=(2, 3))                                   # [B, K]
        logs["feas/cost_picked"] = c[rows, picked]. mean()
        logs["feas/picked_min_sdf"] = min_sdf[rows, picked].mean()
        # a candidate violates when a corner is truly outside, not merely inside the buffer
        logs["feas/frac_violating"] = (min_sdf < 0).float().mean()
        if c_dyn is not None:
            logs["feas/cost_dyn"] = c_dyn.mean()
            logs["feas/cost_dyn_picked"] = c_dyn[rows, picked].mean()
    return w_total * loss, logs


def select_weights(logits: torch.Tensor, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor], cfg) -> torch.Tensor:
    """``[B, K]`` offset-path weights: uniform over a selected set S, exactly zero outside it.

        S = (top-k by the anchor's own PDM label) U (top-k by the model's ungated score) U (WTA winner)

    WHY.  ``softmax(score / 1.0)`` was meant to spend the corridor cost where selection looks, and
    measured on `feas_gt` it does not: its effective support is 198 of 256 candidates (median; the
    top candidate gets 0.019), because the candidates' log-scores are close at temperature 1.  So
    the cost pushed almost every candidate inward, and the 256 outputs collapsed onto one path --
    endpoint spread 14.6 m (v2_r34) -> 2.8 m, scenes holding a PDMS-1.0 candidate 88% -> 52%,
    candidate ceiling 99.50 -> 97.04 (tools/plan_v2/pdms_all_candidates.py).  Imitation cannot hold
    the others: WTA L1 reaches one candidate of 256.

    The LABEL top-k is the stable half: from the first iteration it names the anchors worth
    refining in THIS scene (a hard left on a straight road has DAC 0 and never enters), while
    keeping the modes that matter -- label top-16 anchors still span 7.9 m at the endpoint and
    20.8 m of progress (median, 20k navtrain scenes).  The MODEL top-k is the half that tracks
    deployment: whatever the model ranks high is what it will emit, so a bad candidate it favours
    must be corrected, which is what the corridor cost was for.  The WTA winner is in S because it
    is already being refined toward the human.

    The anchor label only CHOOSES who is refined.  The cost that refines them is still computed on
    the emitted trajectory, so supervision stays on what the model outputs.  Candidates outside S
    get no geometric gradient and stay near their anchor, as in v2_r34 (mean |offset| 1.3 m).
    """
    b, k = logits.shape
    sel = torch.zeros_like(logits)
    km = min(int(getattr(cfg, "feas_select_model_k", 16)), k)
    if km > 0:
        sel.scatter_(1, logits.topk(km, dim=-1).indices, 1.0)
    kl = min(int(getattr(cfg, "feas_select_label_k", 16)), k)
    lab = targets.get("sim_reward")
    if kl > 0 and lab is not None:
        nc, dac, ep, ttc, cf = lab.float().unbind(1)                   # WoTE column order
        pdm = nc * dac * (5 * ep + 5 * ttc + 2 * cf) / 12
        lsel = torch.zeros_like(logits).scatter_(1, pdm.topk(kl, dim=-1).indices, 1.0)
        valid = targets.get("sim_reward_valid")
        if valid is not None:                                         # navtest has no anchor labels
            lsel = lsel * valid.float().view(-1, 1)
        sel = torch.maximum(sel, lsel)
    gt = targets.get("trajectory")
    if gt is not None:
        flat = predictions["trajectory_anchors"].reshape(1, k, -1).to(logits.dtype)
        d = torch.norm(flat - gt.reshape(b, 1, -1).to(logits.dtype), dim=2)
        sel.scatter_(1, d.argmin(dim=1, keepdim=True), 1.0)
    return sel / sel.sum(dim=-1, keepdim=True).clamp_min(1.0)


# --------------------------------------------------------------------------- time-to-collision label

TTC_COL = 3        # WoTE label column of time_to_collision_within_bound (SIM_KEYS order)


def ttc_samples(traj: torch.Tensor, dyn: torch.Tensor, pc_range: Sequence[float],
                dt: float = 0.5, min_speed: float = 0.05, stopped: float = 1e3,
                outside: str = "border") -> torch.Tensor:
    """``traj`` [B, K, T, 3] and the per-step agent field ``dyn`` [B, T, H, W] -> ``[B, K, T-1, 2, 4]``.

    The scorer's TTC (pdm_scorer._calculate_ttc) asks, at every step, whether the ego footprint
    pushed forward at its CURRENT velocity would hit an agent where that agent will be a moment
    later.  This asks the same on the planner's own grid: for each step t < T the footprint now,
    against plane t, and the footprint advanced by v_t * dt, against plane t+1 (the planes are one
    ego step = dt apart).  A step at which the ego is (nearly) stopped is exempt, as in the scorer,
    and gets a large clearance.

    Measured against the v1 scorer's TTC of the emitted trajectory (tools/plan_v2/ttc_label_fidelity.py,
    3000 navtest scenes x 256, anchor labels also from the v1 scorer -- the one the training labels
    come from): on `feas_gt` candidates (mean |offset| 7.8 m) this label is right 81.1% of the time
    where the anchor's is right 69.5%; on v2_r34 (|offset| 1.3 m) the anchor's wins, 87.9% vs 82.8%.
    Hence ttc_label's "band" mode.  Reading the BEV border beyond the grid (grid_sample padding)
    measured slightly BETTER than exempting those samples (AUC 0.790 vs 0.783 on feas_gt), so
    ``outside`` stays "border".  It is a better label where offsets move, not an exact one: the scorer uses
    0.1 s steps, the tracked trajectory, and drops tracks already counted as collisions.
    """
    b, k, t, _ = traj.shape
    prev = torch.cat([traj.new_zeros(b, k, 1, 3), traj[:, :, :-1]], dim=2)
    vel = (traj[..., :2] - prev[..., :2]) / dt                               # [B, K, T, 2]
    moving = vel.norm(dim=-1)[:, :, : t - 1] > min_speed                     # [B, K, T-1]
    now = traj[:, :, : t - 1]
    ahead = torch.cat([now[..., :2] + vel[:, :, : t - 1] * dt, now[..., 2:]], dim=-1)
    c_now, c_ahead = footprint_corners(now), footprint_corners(ahead)       # [B, K, T-1, 4, 2]
    s_now = torch.stack([sample_field(dyn[:, i], c_now[:, :, i], pc_range) for i in range(t - 1)], dim=2)
    s_ahead = torch.stack([sample_field(dyn[:, i + 1], c_ahead[:, :, i], pc_range) for i in range(t - 1)], dim=2)
    if outside == "exempt":
        # A corner beyond the BEV reads the border row under grid_sample's "border" padding, so an
        # agent sitting on the far edge becomes a wall for every fast candidate, and anything past
        # the edge is invisible.  38-48% of candidates put a corner outside at some step (navtest,
        # v2_r34 / feas_gt).  The field has no evidence there either way, so such a sample does not
        # judge -- the same exemption a stopped ego gets.
        def inside(c):
            x, y = c[..., 0], -c[..., 1]                                     # NAVSIM x, SSR x
            return ((x >= pc_range[1]) & (x <= pc_range[4]) & (y >= pc_range[0]) & (y <= pc_range[3]))
        s_now = s_now.masked_fill(~inside(c_now), stopped)
        s_ahead = s_ahead.masked_fill(~inside(c_ahead), stopped)
    elif outside != "border":
        raise ValueError(f"unknown outside {outside!r}")
    s = torch.stack([s_now, s_ahead], dim=3)                                 # [B, K, T-1, 2, 4]
    return s.masked_fill(~moving[..., None, None], stopped)


def ttc_label(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], cfg,
              sim_reward: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Replace the TTC column of the anchor labels, for the candidates where the anchor no longer
    describes the output.  ``cfg.feas_ttc_target``:

        anchor     unchanged (WoTE)
        geometric  every candidate's TTC target is computed on its EMITTED trajectory
        band       geometric only for candidates whose mean |offset| is in [lo, hi) m (default 2-12),
                   the anchor's label elsewhere -- the rule the measurement supports, see above
        hybrid     only the candidates in the refined set S (feasibility.select_weights) -- those
                   are the ones whose offsets move, so the anchor's label goes stale exactly there;
                   outside S the offset stays near the anchor and the anchor label is the more
                   accurate of the two (see ttc_samples)

    The target is ``exp(-c / tau)`` with ``c`` the softplus cost of the samples at margin
    ``feas_ttc_margin`` -- the same form, and the same tau, as the feasible head's target.  It is a
    LABEL: computed without gradient, from the ground-truth agent field.
    """
    mode = str(getattr(cfg, "feas_ttc_target", "anchor"))
    if mode == "anchor":
        return sim_reward, {}
    dyn = targets.get("agent_sdf_bev")
    if dyn is None:
        raise KeyError("feas_ttc_target needs agent_sdf_bev (DrivableFieldTargetBuilder with feas_dyn_weight > 0)")
    with torch.no_grad():
        anchors = predictions["trajectory_anchors"].to(predictions["trajectory_offset"].dtype)
        traj = (anchors.unsqueeze(0) + predictions["trajectory_offset"]).float()
        s = ttc_samples(traj, dyn.float(), cfg.map_pc_range,
                        outside=str(getattr(cfg, "feas_ttc_outside", "border")))
        margin = float(getattr(cfg, "feas_ttc_margin", 0.0))
        beta = float(getattr(cfg, "feas_beta", 0.2))
        c = (F.softplus((margin - s) / beta) * beta).sum(dim=(2, 3, 4))       # [B, K]
        geo = torch.exp(-c / float(getattr(cfg, "feas_score_tau", 0.5)))
        if mode == "geometric":
            use = torch.ones_like(geo, dtype=torch.bool)
        elif mode == "band":
            # choose, per candidate, the label that is more accurate for it -- which is decided by
            # how far its offset moved it from the anchor.  Measured against the v1 scorer's TTC of
            # the emitted trajectory (tools/plan_v2/ttc_label_fidelity.py, 1.5M navtest candidates):
            # the anchor's label wins below 2 m (by 3-6 points: the anchor still IS the output) and
            # above 12 m (by 3.8: the plan leaves the BEV the field covers), the geometric label wins
            # from 2 to 12 m by 7-19 points.  Overall this is within 0.5 point of the better single
            # label in both regimes -- 87.74% vs 87.86% (v2_r34, small offsets) and 80.66% vs 81.12%
            # (feas_gt, large) -- where either single label loses 5-11 points in one of them.
            lo = float(getattr(cfg, "feas_ttc_band_lo", 2.0))
            hi = float(getattr(cfg, "feas_ttc_band_hi", 12.0))
            mag = predictions["trajectory_offset"][..., :2].detach().float().norm(dim=-1).mean(dim=-1)
            use = (mag >= lo) & (mag < hi)
        elif mode == "hybrid":
            rank = predictions.get("plan_rank_rewards")
            rank = (rank if rank is not None else predictions["plan_final_rewards"]).detach().float()
            use = select_weights(rank, predictions, targets, cfg) > 0
        else:
            raise ValueError(f"unknown feas_ttc_target {mode!r}")
        new = sim_reward.clone()
        # clone: on a float32 label .float() is the SAME storage, so without it `old` would read the
        # replaced values in the logs below (the target itself is computed before the write and was
        # never affected -- the first B2 run logged ttc/disagree = 0 exactly because of this)
        old = new[:, TTC_COL].float().clone()
        new[:, TTC_COL] = torch.where(use, geo, old).to(new.dtype)
        u = use.float()
        logs = {
            "ttc/frac_replaced": u.mean(),
            "ttc/target_geo": (geo * u).sum() / u.sum().clamp_min(1.0),
            "ttc/target_anchor_replaced": (old * u).sum() / u.sum().clamp_min(1.0),
            # how often the two labels disagree on pass/fail where one replaces the other
            "ttc/disagree": (((geo >= 0.5) != (old >= 0.5)).float() * u).sum() / u.sum().clamp_min(1.0),
        }
    return new, logs


# --------------------------------------------------------------------------- ego-progress label

EP_COL = 2                       # WoTE label column of ego_progress (SIM_KEYS order)
REAR_TO_CENTER = (4.049 - 1.127) / 2   # pacifica rear axle -> bbox centre; the scorer projects CENTRES


class RouteCenterlineTargetBuilder(AbstractTargetBuilder):
    """``route_centerline`` [P, 2] float32: the scorer's route centerline in the current ego frame
    (NaN padded), read by token from the packed array tools/plan_v2/dump_route_centerline.py wrote.
    Same lazy memory-mapped pattern as plan_score_targets.AnchorScoreTargetBuilder."""

    def __init__(self, config):
        if not getattr(config, "feas_route_file", ""):
            raise ValueError("feas_ep_target needs feas_route_file (packed route centerlines, no .npy suffix)")
        self._path = str(config.feas_route_file)
        self._max_offset = float(getattr(config, "feas_route_max_offset", 5.0))
        self._arr = None
        self._index = None

    def get_unique_name(self) -> str:
        return cache_key("para_ssr_route_centerline", (("file", self._path.rsplit("/", 1)[-1]),))

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        import json
        if self._arr is None:                                       # lazily: pickled into workers
            self._arr = np.load(self._path + ".npy", mmap_mode="r")
            self._index = {t: i for i, t in enumerate(json.load(open(self._path + ".tokens.json")))}
        token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
        i = self._index.get(token)
        if i is None:
            return {"route_centerline": torch.full(self._arr.shape[1:], float("nan"))}
        line = np.array(self._arr[i], dtype=np.float32)          # a copy: the mmap slice is read-only
        # A route that does not pass the ego is a failed extraction, not a route: measured on navtrain
        # val logs, one sg-one-north log alone put the human's end 30-150 m off its "route", and the
        # top 1% of such scenes carried 98% of the val lateral term.  0.20% of navtrain is > 5 m.
        far = float(self._max_offset) > 0 and (
            not np.isfinite(line[:, 0]).any()
            or np.nanmin(np.linalg.norm(line - np.array([REAR_TO_CENTER, 0.0], np.float32), axis=-1)) > self._max_offset)
        if far:
            return {"route_centerline": torch.full(self._arr.shape[1:], float("nan"))}
        return {"route_centerline": torch.from_numpy(line)}


def project_on_polyline(pts: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    """Arc length of the nearest point on a polyline, like shapely's ``LineString.project``.

    ``pts`` [B, M, 2], ``line`` [B, P, 2] NaN padded at the end -> [B, M].  A row with fewer than two
    valid points returns NaN.
    """
    valid = ~torch.isnan(line[..., 0])                                         # [B, P]
    ln = torch.nan_to_num(line, nan=0.0)
    a, b = ln[:, :-1], ln[:, 1:]                                                # [B, P-1, 2]
    seg_ok = valid[:, :-1] & valid[:, 1:]
    d = b - a
    L = d.norm(dim=-1) * seg_ok                                                 # [B, P-1]
    cum = torch.cumsum(L, dim=-1) - L                                           # arc length at segment start
    rel = pts[:, :, None, :] - a[:, None]                                       # [B, M, P-1, 2]
    t = ((rel * d[:, None]).sum(-1) / (L[:, None] ** 2 + 1e-9)).clamp(0, 1)
    proj = a[:, None] + t[..., None] * d[:, None]
    dist2 = ((pts[:, :, None] - proj) ** 2).sum(-1).masked_fill(~seg_ok[:, None], float("inf"))
    j = dist2.argmin(dim=-1, keepdim=True)                                      # [B, M, 1]
    s = (cum[:, None].expand_as(dist2) + t * L[:, None]).gather(-1, j).squeeze(-1)
    return s.masked_fill(~seg_ok.any(-1, keepdim=True), float("nan"))


def ep_label(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], cfg,
             sim_reward: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Replace the ego-progress column of the anchor labels with one computed on the EMITTED
    trajectory.  ``cfg.feas_ep_target``: anchor (unchanged) | geometric.

        raw_k  = project(centre of candidate k's last pose) - project(ego centre) on the route centerline
        EP_k   = raw_k / max(raw_human, raw_k)  *  1[merged corridor clearance >= 0 on the GT fields]

    Why this form, measured on navtest (tools/plan_v2/ep_label_fidelity.py, 3000 scenes x 256):
    among candidates the scorer does not zero out, the anchor's EP label ranks progress at a
    within-scene rank correlation of 0.19-0.30, this label at 0.89-0.95; the anchor label mostly
    encodes whether the ANCHOR is admissible.  The admissibility factor here is the same corridor
    verdict the feasible head is trained on.  Normalising by the human's progress beat both the
    best candidate's and (surprisingly) the PDM reference's when the label was used for selection,
    and the human trajectory is available at training time where the PDM reference is not.

    Honest caveat from the same measurement: used as a selection signal this label beat the anchor
    label on v2_r34 (+0.73 PDMS) and feas_gt (+0.45) but not on B3 at epoch 23 (-0.10).
    """
    mode = str(getattr(cfg, "feas_ep_target", "anchor"))
    if mode == "anchor":
        return sim_reward, {}
    if mode != "geometric":
        raise ValueError(f"unknown feas_ep_target {mode!r}")
    line = targets.get("route_centerline")
    if line is None:
        raise KeyError("feas_ep_target=geometric needs route_centerline (RouteCenterlineTargetBuilder)")
    with torch.no_grad():
        anchors = predictions["trajectory_anchors"].to(predictions["trajectory_offset"].dtype)
        traj = (anchors.unsqueeze(0) + predictions["trajectory_offset"]).float()          # [B, K, T, 3]
        end = traj[:, :, -1]
        cand = end[..., :2] + REAR_TO_CENTER * torch.stack([end[..., 2].cos(), end[..., 2].sin()], -1)
        hum = targets["trajectory"].float()[:, -1]                                      # [B, 3]
        hum_c = hum[:, :2] + REAR_TO_CENTER * torch.stack([hum[:, 2].cos(), hum[:, 2].sin()], -1)
        start = torch.tensor([REAR_TO_CENTER, 0.0], device=traj.device).expand(traj.shape[0], 1, 2)
        pts = torch.cat([start, hum_c[:, None], cand], dim=1)                            # [B, 2+K, 2]
        s = project_on_polyline(pts, line.float().to(traj.device))
        s0, sh, sk = s[:, :1], s[:, 1:2], s[:, 2:]
        raw = (sk - s0).clamp_min(0.0)
        raw_h = (sh - s0).clamp_min(0.0)
        geo = raw / torch.maximum(torch.maximum(raw_h, raw), torch.full_like(raw, 1e-6))
        # admissibility: the merged corridor on the ground-truth fields at margin 0
        p = {"trajectory_anchors": predictions["trajectory_anchors"],
             "trajectory_offset": predictions["trajectory_offset"].detach(),
             "drivable_sdf_pred": targets["drivable_sdf_bev"].float()}
        if targets.get("agent_sdf_bev") is not None:
            p["agent_sdf_pred"] = targets["agent_sdf_bev"].float()

        class _G:
            map_pc_range = cfg.map_pc_range
            feas_dyn_weight = 1.0 if "agent_sdf_pred" in p else 0.0
        gate = (corridor_clearance(p, _G, 0.0, 0.0) >= 0).float()
        lab = geo * gate
        ok = ~torch.isnan(lab).any(dim=-1, keepdim=True)                                # scenes with a route
        new = sim_reward.clone()
        old = new[:, EP_COL].float().clone()
        new[:, EP_COL] = torch.where(ok, torch.nan_to_num(lab), old).to(new.dtype)
        okf = ok.float()
        logs = {"ep/frac_scenes_with_route": okf.mean(),
                "ep/target_geo": (torch.nan_to_num(lab) * okf).sum() / (okf.sum() * lab.shape[1]).clamp_min(1.0),
                "ep/target_anchor": old.mean(),
                "ep/gate_pass": (gate * okf).sum() / (okf.sum() * gate.shape[1]).clamp_min(1.0)}
    return new, logs


# --------------------------------------------------------------------------- ranking inside the corridor

def admissible_rank_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                         sim_reward: torch.Tensor, cfg) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Listwise top-1 ranking, restricted to the candidates the CORRIDOR admits.

    WHY.  Measured on navtest (tools/plan_v2/diagnose_arms.py): with the ground-truth corridor gate the
    deployed pick scores 89.7 PDMS and the best admissible candidate 99.1 -- 6.5 of the 9.3 points lost
    are ranking among admissible candidates, and the pick's EP is 12 points below the best admissible
    one.  The score heads are trained pointwise (one BCE per metric per candidate), which ranks the
    whole set reasonably (D+: EP head Spearman 0.19 -> 0.60) yet leaves top-1 unchanged: nothing asks
    "which admissible candidate is FIRST".

    WHAT.  The corridor decides feasibility absolutely (field gate at inference); ranking decides only
    the preference among what it admits.  So the ranking is trained exactly on the set the deployed
    argmax ranges over (``rank_set``="gate": the inference field gate on the model's OWN predicted
    field) -- and nowhere else.  Because that gate is imperfect (D+: ~8% of drivable candidates wrongly
    rejected, ~15% of non-drivable ones wrongly admitted), the label keeps NC * DAC: a candidate the
    gate wrongly admits ranks low, so the ranker learns to cover the gate's misses instead of being
    made more aggressive into them.  ("corridor" = ground-truth corridor set + preference label only.)

        p = softmax_A( score_k / T )                score = the selection rule's ungated preference
        q = softmax_A( label / tau )      label = NC * DAC * (5 EP + 5 TTC + 2 C) / 12 on the emitted traj
        L = - sum_{k in A} q_k log p_k

    Inadmissible candidates are absent, not pushed down: the corridor, not the ranker, owns them, so
    the ranker's capacity is spent on the distinction the deployment actually makes.  ``rank_set``
    = "all" trains the same loss over all 256 (the usual listwise ranking) as the control.
    """
    w = float(getattr(cfg, "rank_weight", 0.0))
    if w <= 0 or "im_rewards" not in predictions:
        return None, {}
    eps = 1e-6
    im = predictions["im_rewards"].float()
    sim = predictions["sim_rewards"].float()                                    # [B, H, K]
    if str(getattr(cfg, "plan_head_mode", "wote")) == "corridor":
        _, ep_h, ttc_h, c_h = sim.unbind(1)
    else:
        _, _, ep_h, ttc_h, c_h = sim.unbind(1)
    wr = tuple(float(v) for v in getattr(cfg, "plan_reward_weights", (0.1, 0.5, 0.5, 1.0)))
    score = wr[0] * torch.log(im + eps) + wr[3] * torch.log(5 * ttc_h + 2 * c_h + 5 * ep_h + eps)   # [B, K]
    lab = sim_reward.float()
    pref = (5 * lab[:, EP_COL] + 5 * lab[:, TTC_COL] + 2 * lab[:, 4]) / 12.0                        # [B, K]
    full = lab[:, 0] * lab[:, 1] * pref                                                              # NC * DAC * pref
    valid = targets.get("sim_reward_valid")
    valid = (valid.float().view(-1) > 0) if valid is not None else torch.ones(score.shape[0], dtype=torch.bool, device=score.device)
    mode = str(getattr(cfg, "rank_set", "gate"))
    # gate      THE DEPLOYMENT SET: candidates the inference field gate admits (the model's own predicted
    #           field, static, margin feas_select_margin) -- the set the deployed argmax actually ranges
    #           over -- with the FULL label, so what that gate wrongly admits ranks low by its NC/DAC
    # corridor  the ground-truth merged corridor, preference label only (NC/DAC are the corridor's job)
    # all       control: the usual listwise ranking over all K with the full label
    label = pref if mode == "corridor" else full
    if mode == "gate":
        with torch.no_grad():
            p = {"trajectory_anchors": predictions["trajectory_anchors"],
                 "trajectory_offset": predictions["trajectory_offset"].detach(),
                 "drivable_sdf_pred": predictions["drivable_sdf_pred"].detach().float()}

            class _P:
                map_pc_range = cfg.map_pc_range
                feas_dyn_weight = 0.0
            adm = corridor_clearance(p, _P, float(getattr(cfg, "feas_select_margin", 0.1)), -1e9) >= 0
            # a scene the gate empties is deployed ungated (field_gate_select), so it ranks over all
            adm = torch.where(adm.any(-1, keepdim=True), adm, torch.ones_like(adm))
    elif mode == "corridor":
        with torch.no_grad():
            p = {"trajectory_anchors": predictions["trajectory_anchors"],
                 "trajectory_offset": predictions["trajectory_offset"].detach(),
                 "drivable_sdf_pred": targets["drivable_sdf_bev"].float()}
            if targets.get("agent_sdf_bev") is not None:
                p["agent_sdf_pred"] = targets["agent_sdf_bev"].float()

            class _G:
                map_pc_range = cfg.map_pc_range
                feas_dyn_weight = 1.0 if "agent_sdf_pred" in p else 0.0
            adm = corridor_clearance(p, _G, 0.0, 0.0) >= 0                                          # [B, K]
    else:
        adm = torch.ones_like(score, dtype=torch.bool)
    use = valid & (adm.sum(-1) >= 2)
    if not use.any():
        return score.sum() * 0.0, {"rank/scenes": torch.zeros((), device=score.device)}
    neg = torch.finfo(score.dtype).min / 4
    T = float(getattr(cfg, "rank_temp", 1.0))
    tau = float(getattr(cfg, "rank_tau", 0.05))
    logp = torch.log_softmax(torch.where(adm, score / T, torch.full_like(score, neg)), dim=-1)
    q = torch.softmax(torch.where(adm, label / tau, torch.full_like(label, neg)), dim=-1).detach()
    per = -(q * torch.where(adm, logp, torch.zeros_like(logp))).sum(-1)                              # [B]
    loss = (per * use.float()).sum() / use.float().sum()
    with torch.no_grad():
        top_model = torch.where(adm, score, torch.full_like(score, neg)).argmax(-1)
        top_label = torch.where(adm, label, torch.full_like(label, neg)).argmax(-1)
        regret = (label.gather(1, top_label[:, None]) - label.gather(1, top_model[:, None]))[:, 0]
    logs = {"rank/loss": loss.detach(), "rank/admissible_n": adm.float().sum(-1)[use].mean(),
            "rank/top1_hit": (top_model == top_label).float()[use].mean(),
            "rank/top1_regret": regret[use].mean(), "rank/scenes": use.float().mean()}
    return w * loss, logs


def selection_curse_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                         sim_reward: torch.Tensor, cfg) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Penalise OVER-estimation of the preference heads on the candidates the deployed argmax would pick.

    WHY (the optimizer's curse).  Measured on D+'s navtest candidates (linear scale, a random candidate
    of the same set as control): restricted to really feasible candidates, the picked one's TTC head
    over-estimates its true TTC by +0.06 at 256 candidates while a random one is UNDER-estimated by
    0.03, and the gap grows with the set size (EP: +0.06 vs -0.17).  The pointwise BCE trains all K
    candidates alike, yet only the argmax's estimate matters, and the argmax selects exactly the
    candidates the heads over-rate.  The imitation term hides this by keeping the pick near the human,
    where the heads err least (TTC |err| 0.13 in im's top 4 vs ~0.25 elsewhere), at the price of
    capping the ranking.

    WHAT.  S = the top ``curse_topk`` candidates under the DEPLOYED rule (plan_final_rewards, i.e. the
    feasible-head gate, then the static predicted-field gate at ``feas_select_margin`` as eval runs it).
    On S, the EP / TTC / comfort heads -- the terms the argmax maximises -- get an asymmetric pinball
    loss against the label on the emitted trajectory (the same relabelled ``sim_reward`` the BCE uses):

        L = (1 - tau) * relu(p - y) + tau * relu(y - p),   tau = ``curse_tau`` < 0.5

    so over-rating a would-be winner costs (1 - tau) / tau times more than under-rating it.
    ``curse_set`` = "all" applies the same loss to all K (control: asymmetry without selection).
    The feasible head is left alone: its own target lives in feasibility_loss.
    """
    w = float(getattr(cfg, "curse_weight", 0.0))
    if w <= 0 or "plan_final_rewards" not in predictions:
        return None, {}
    sim = predictions["sim_rewards"].float()                                    # [B, H, K]
    if str(getattr(cfg, "plan_head_mode", "wote")) == "corridor":
        heads = sim[:, 1:4]                                                     # EP, TTC, C
    else:
        heads = sim[:, 2:5]
    lab = sim_reward.float()
    y = torch.stack([lab[:, EP_COL], lab[:, TTC_COL], lab[:, 4]], dim=1)          # [B, 3, K]
    valid = targets.get("sim_reward_valid")
    valid = (valid.float().view(-1) > 0) if valid is not None else torch.ones(sim.shape[0], dtype=torch.bool, device=sim.device)
    k_all = sim.shape[-1]
    mode = str(getattr(cfg, "curse_set", "topk"))
    with torch.no_grad():
        final = predictions["plan_final_rewards"].detach().float()
        if "drivable_sdf_pred" in predictions:
            p = {"trajectory_anchors": predictions["trajectory_anchors"],
                 "trajectory_offset": predictions["trajectory_offset"].detach(),
                 "drivable_sdf_pred": predictions["drivable_sdf_pred"].detach().float()}

            class _P:
                map_pc_range = cfg.map_pc_range
                feas_dyn_weight = 0.0
            adm = corridor_clearance(p, _P, float(getattr(cfg, "feas_select_margin", 0.1)), -1e9) >= 0
            gate = float(getattr(cfg, "plan_select_gate", 20.0))
            final = final - gate * (~adm & adm.any(-1, keepdim=True)).float()
        if mode == "topk":
            top = final.topk(min(int(getattr(cfg, "curse_topk", 8)), k_all), dim=-1).indices   # [B, k]
            sel = torch.zeros_like(final, dtype=torch.bool).scatter_(1, top, True)
        elif mode == "all":
            sel = torch.ones_like(final, dtype=torch.bool)
        else:
            raise ValueError(f"unknown curse_set {mode!r}")
        pick = final.argmax(-1)
    tau = float(getattr(cfg, "curse_tau", 0.2))
    d = heads - y                                                              # [B, 3, K] (+ = over-rated)
    pin = (1.0 - tau) * F.relu(d) + tau * F.relu(-d)
    m = (sel & valid[:, None]).float()[:, None].expand_as(pin)
    loss = (pin * m).sum() / m.sum().clamp_min(1.0)
    with torch.no_grad():
        dp = d.gather(2, pick[:, None, None].expand(-1, 3, 1))[:, :, 0]           # [B, 3]
        vs = valid.float().sum().clamp_min(1.0)
        logs = {"curse/loss": loss.detach(), "curse/set_n": sel.float().sum(-1).mean(),
                "curse/pick_over_ep": (dp[:, 0] * valid).sum() / vs,
                "curse/pick_over_ttc": (dp[:, 1] * valid).sum() / vs,
                "curse/pick_over_c": (dp[:, 2] * valid).sum() / vs,
                "curse/all_over_ttc": (d[:, 1].mean(-1) * valid).sum() / vs}
    return w * loss, logs


# --------------------------------------------------------------------------- driving force

def drive_terms(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                features: Dict[str, torch.Tensor], cfg) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """A pushing force and two resistances on the OFFSETS of the refined set S.

    The corridor cost only ever pushes a candidate AWAY from obstacles; imitation (WTA L1) only pulls
    one candidate toward the human.  Nothing asks a candidate to make progress, so the constraint
    bought freedom the plan never used.  Three terms, all on S (feasibility.select_weights, uniform
    on S, zero outside -- the same set the corridor cost refines, so they cannot collapse the rest):

      progress     relu(1 - ds_k / ds_human)   along the scorer's route centerline: candidates that
                   fall short of the human's progress are pushed forward, stopping at the human --
                   the same normalisation the EP label uses, so it does not reward aggression past
                   it.  A candidate the corridor verdict rejects is NOT pushed (its gate is 0), and a
                   scene whose human barely moves (< 2 m, e.g. a red light) pushes nothing.
      lateral      relu(d_k - free)^2, d_k the end centre's distance to the centerline: the
                   counter-force for lane keeping, which progress alone would trade away.
      continuity   the plan's first-step speed against the ego's measured speed and acceleration,
                   with a dead zone.  The scorer's two-frame comfort compares CONSECUTIVE plans and a
                   training sample has no previous plan, so this is the single-frame part of it: a
                   plan whose start does not continue the current motion cannot match the previous
                   frame's plan either.  It is a proxy, and named as one.

    Returns (loss, logs); (None, {}) when all three weights are 0.
    """
    wp = float(getattr(cfg, "feas_progress_weight", 0.0))
    wl = float(getattr(cfg, "feas_lateral_weight", 0.0))
    wc = float(getattr(cfg, "feas_continuity_weight", 0.0))
    if wp <= 0 and wl <= 0 and wc <= 0:
        return None, {}
    anchors = predictions["trajectory_anchors"].to(predictions["trajectory_offset"].dtype)
    traj = (anchors.unsqueeze(0) + predictions["trajectory_offset"]).float()          # [B, K, T, 3]  (grad)
    b, k, t, _ = traj.shape
    rank = predictions.get("plan_rank_rewards")
    rank = (rank if rank is not None else predictions["plan_final_rewards"]).detach().float()
    w = select_weights(rank, predictions, targets, cfg)                               # [B, K]
    loss = traj.new_zeros(())
    logs: Dict[str, torch.Tensor] = {}

    line = targets.get("route_centerline")
    if (wp > 0 or wl > 0) and line is not None:
        line = line.float().to(traj.device)
        end = traj[:, :, -1]
        cand = end[..., :2] + REAR_TO_CENTER * torch.stack([end[..., 2].cos(), end[..., 2].sin()], -1)
        with torch.no_grad():
            hum = targets["trajectory"].float()[:, -1]
            hum_c = hum[:, :2] + REAR_TO_CENTER * torch.stack([hum[:, 2].cos(), hum[:, 2].sin()], -1)
            start = torch.tensor([REAR_TO_CENTER, 0.0], device=traj.device).expand(b, 1, 2)
            s_ref = project_on_polyline(torch.cat([start, hum_c[:, None]], 1), line)
            s0, ds_h = s_ref[:, :1], (s_ref[:, 1:2] - s_ref[:, :1])
            has_route = ~torch.isnan(ds_h)
            moving = has_route & (torch.nan_to_num(ds_h) >= 2.0)                     # [B, 1]
            # the merged corridor verdict min(Phi_static, Phi_dyn) >= 0 on the fields this arm's cost
            # reads (pred arm: the model's own) -- a candidate it rejects is never pushed
            pred = str(getattr(cfg, "feas_field", "gt")) == "pred"
            fld = predictions["drivable_sdf_pred"] if pred else targets["drivable_sdf_bev"]
            dyn = predictions.get("agent_sdf_pred") if pred else targets.get("agent_sdf_bev")
            use_dyn = dyn is not None and float(getattr(cfg, "feas_dyn_weight", 0.0)) > 0
            p = {"trajectory_anchors": predictions["trajectory_anchors"],
                 "trajectory_offset": predictions["trajectory_offset"].detach(),
                 "drivable_sdf_pred": fld.detach().float()}
            if use_dyn:
                p["agent_sdf_pred"] = dyn.detach().float()

            class _G:
                map_pc_range = cfg.map_pc_range
                feas_dyn_weight = 1.0 if use_dyn else 0.0
            gate = (corridor_clearance(p, _G, 0.0, 0.0) >= 0).float()                  # [B, K]
        s_k = project_on_polyline(cand, line)                                           # [B, K]  (grad)
        if wp > 0:
            ratio = (s_k - s0) / torch.nan_to_num(ds_h, nan=1.0).clamp_min(2.0)
            short = F.relu(1.0 - torch.nan_to_num(ratio, nan=1.0))
            m = w * gate * moving.float()
            prog = (m * short).sum() / moving.float().sum().clamp_min(1.0)
            loss = loss + wp * prog
            logs["drive/progress"] = prog.detach()
            logs["drive/gate_pass_S"] = ((gate * (w > 0)).sum() / (w > 0).float().sum().clamp_min(1.0)).detach()
            logs["drive/progress_ratio_S"] = ((torch.nan_to_num(ratio) * (w > 0)).sum()
                                              / (w > 0).float().sum().clamp_min(1.0)).detach()
        if wl > 0:
            # distance of the end centre to the centerline, through the same nearest-segment search
            valid = ~torch.isnan(line[..., 0])
            ln = torch.nan_to_num(line, nan=0.0)
            a_, b_ = ln[:, :-1], ln[:, 1:]
            seg_ok = valid[:, :-1] & valid[:, 1:]
            d = b_ - a_
            tt = (((cand[:, :, None] - a_[:, None]) * d[:, None]).sum(-1)
                  / ((d ** 2).sum(-1)[:, None] + 1e-9)).clamp(0, 1)
            dist = (cand[:, :, None] - (a_[:, None] + tt[..., None] * d[:, None])).norm(dim=-1)
            dist = dist.masked_fill(~seg_ok[:, None], float("inf")).amin(-1)          # [B, K]
            free = float(getattr(cfg, "feas_lateral_free", 1.0))
            off_lane = F.relu(torch.where(torch.isfinite(dist), dist, torch.zeros_like(dist)) - free) ** 2
            m = w * has_route.float()
            lat = (m * off_lane).sum() / has_route.float().sum().clamp_min(1.0)
            loss = loss + wl * lat
            logs["drive/lateral"] = lat.detach()

    st = features.get("status_feature")
    if wc > 0 and st is not None:
        nc = int(getattr(cfg, "num_navi_cmd", 4))
        v = st[:, nc:nc + 2].float().to(traj.device)                                  # ego frame vx, vy
        a = st[:, nc + 2:nc + 4].float().to(traj.device)                              # ax, ay
        dt = 0.5
        # average speed over the first step, and what the measured motion predicts for it
        v_plan = traj[:, :, 0, :2].norm(dim=-1) / dt                                 # [B, K]
        v_meas = (v + 0.5 * dt * a).norm(dim=-1, keepdim=True)                        # [B, 1]
        tol = float(getattr(cfg, "feas_continuity_tol", 0.5))                         # m/s dead zone
        beta = float(getattr(cfg, "feas_beta", 0.2))
        mis = F.softplus(((v_plan - v_meas).abs() - tol) / beta) * beta
        cont = (w * mis).sum() / b
        loss = loss + wc * cont
        logs["drive/continuity"] = cont.detach()
        logs["drive/start_speed_err_S"] = (((v_plan - v_meas).abs() * (w > 0)).sum()
                                           / (w > 0).float().sum().clamp_min(1.0)).detach()
    return float(getattr(cfg, "feas_weight", 0.0)) * loss, logs


# --------------------------------------------------------------------------- selection


def corridor_clearance(predictions: Dict[str, torch.Tensor], cfg,
                       margin: float, dyn_margin: float) -> Optional[torch.Tensor]:
    """``-> [B, K]``: the WORST corridor clearance of ``anchor_k + offset_k``, or ``None``.

        min_{t, corner} min(Phi_static - margin, Phi_dyn - dyn_margin)

    Negative means some corner of the EMITTED footprint is inside an obstacle's margin at some step.
    This is the same ``min`` that ``feasibility_losses`` uses to JUDGE (``c_merge0``), on the same
    emitted trajectory -- but reduced by ``min`` instead of a softplus sum, because a verdict is not
    a magnitude, and read from the field with no head in between.
    """
    if "trajectory_offset" not in predictions:
        return None
    field = predictions.get("drivable_sdf_pred")
    if field is None:
        return None
    anchors = predictions["trajectory_anchors"].to(predictions["trajectory_offset"].dtype)
    traj = (anchors.unsqueeze(0) + predictions["trajectory_offset"]).float()       # [B, K, T, 3]
    pc_range = cfg.map_pc_range
    phi = corner_sdf(field.float(), traj, pc_range) - margin                       # [B, K, T, 4]
    dyn = predictions.get("agent_sdf_pred")
    if dyn is not None and float(getattr(cfg, "feas_dyn_weight", 0.0)) > 0:
        corners = footprint_corners(traj)                                          # [B, K, T, 4, 2]
        per_t = [sample_field(dyn[:, t].float(), corners[:, :, t], pc_range) for t in range(dyn.shape[1])]
        phi = torch.minimum(phi, torch.stack(per_t, dim=2) - dyn_margin)
    return phi.amin(dim=3).amin(dim=2)                                             # [B, K]


def field_gate_select(predictions: Dict[str, torch.Tensor], cfg) -> Dict[str, torch.Tensor]:
    """Inference-only re-selection: gate the K candidates on the PREDICTED FIELD ITSELF.

    ``feas_to_score`` sends the corridor into a sigmoid head and ``plan_select_mode=corridor_first``
    gates on that head, so at deployment the geometry survives only as far as one scalar per
    candidate copied it.  It does not survive intact: measured on v2_r34's own dump
    (``tools/plan_v2/geo_gate_probe.py``, 2000 navtest scenes), reading the predicted field
    geometrically at selection moves the realised DAC rate 95.35% -> 95.90% at margin 0.10, and the
    ground-truth field's ceiling is 96.45%.  The head had lost that.

    So the field is not only a training signal, it is the deployed constraint -- which is what makes
    its accuracy worth paying for, and what makes training's cost and deployment's verdict the same
    object rather than two things that merely correlate.  Nothing here trains: this is a selection
    rule over a finished checkpoint, so it applies to any v2 anchor run, ``wote`` head set included.

    The gate is additive and identical in form to ``AnchorPlanner._gate``: inadmissible candidates
    drop below every admissible one, and a scene with NO admissible candidate keeps its order rather
    than hand an argmax over 256 equally-penalised candidates.  At margin 0.10 that is 0.25% of
    scenes (0.75% at 0.40) -- rare, but it must not become an arbitrary pick.
    """
    if not {"plan_final_rewards", "plan_topk_index", "trajectory_anchors"} <= predictions.keys():
        return {}
    clear = corridor_clearance(predictions, cfg,
                               float(getattr(cfg, "feas_select_margin", 0.1)),
                               float(getattr(cfg, "feas_select_dyn_margin", 0.0)))
    if clear is None:
        return {}
    final = predictions["plan_final_rewards"].float()
    admissible = clear >= 0
    any_ok = admissible.any(dim=-1, keepdim=True)
    gate = float(getattr(cfg, "plan_select_gate", 20.0))
    gated = final - gate * (~admissible & any_ok).to(final.dtype)

    anchors = predictions["trajectory_anchors"].to(predictions["trajectory_offset"].dtype)
    all_traj = anchors.unsqueeze(0) + predictions["trajectory_offset"]             # [B, K, T, 3]
    k = predictions["plan_topk_index"].shape[1]
    top = gated.topk(k, dim=-1).indices
    top_traj = all_traj.gather(1, top[:, :, None, None].expand(-1, -1, *all_traj.shape[2:]))
    return {
        "trajectory": top_traj[:, 0],
        "plan_topk_trajectory": top_traj,
        "plan_topk_reward": gated.gather(1, top),
        "plan_topk_index": top,
        "plan_final_rewards": gated,
        "feas_select_clearance": clear.detach(),
    }
