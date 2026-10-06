"""Total PARA-SSR loss, per-task weighting, and the shared-BEV gradient valve.

Three multipliers stack, exactly as in the nuScenes config:

1. each head's internal ``loss_*_weight`` (per TERM)
2. the head-level ``loss_weight`` (per HEAD)
3. ``task_loss_weight`` here (per TASK)

``grad_balance`` is deliberately *not* in that stack.  It scales only the
gradient entering ``bev_embed`` and leaves each head's own parameter gradients
alone, which is why it can hold a task's influence on the shared feature down
without slowing the head itself.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .modules.grad_balance import GradBalancer, all_reduce_mean, balance_shared_gradients


def compute_plan_loss(
    ego_fut_preds: torch.Tensor,
    gt_offsets: torch.Tensor,
    gt_mask: torch.Tensor,
    command: torch.Tensor,
    heading_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """L1 on the commanded branch only, as in ``ParaSSRHead.loss``.

    Args:
        ego_fut_preds: ``[B, mode, T, D]`` per-step offsets
        gt_offsets: ``[B, T, D]``
        gt_mask: ``[B, T]``
        command: ``[B, num_cmd]`` one-hot
    """
    B, mode, T, D = ego_fut_preds.shape
    gt = gt_offsets.unsqueeze(1).expand(B, mode, T, D)

    # only the commanded branch is supervised, and only over valid steps
    weight = command[..., None, None] * gt_mask[:, None, :, None]
    weight = weight.expand(B, mode, T, D).clone()
    if D == 3:
        weight[..., 2] = weight[..., 2] * heading_weight

    residual = ego_fut_preds - gt
    if D == 3:
        # Heading is periodic: theta and theta +/- 2*pi are the same pose.
        heading_delta = residual[..., 2]
        wrapped_heading = torch.atan2(
            torch.sin(heading_delta), torch.cos(heading_delta)
        )
        # Do not assign the wrapped value back through a view of ``residual``:
        # sin/cos backward saves that view and an in-place write invalidates its
        # autograd version counter when diagnostics call autograd.grad first.
        residual = torch.cat([residual[..., :2], wrapped_heading[..., None]], dim=-1)
    err = residual.abs() * weight
    # mmdet's weighted L1 with avg_factor=None takes the mean over the complete
    # prediction tensor after applying weights.  Command/mask/heading weights
    # affect the numerator only; dividing by weight.sum() would rescale the
    # configured loss whenever a weight (notably heading_weight) changes.
    loss = err.mean()

    return loss, _plan_command_metrics(err, weight, mode)


def compute_plan_tail_loss(
    ego_fut_preds: torch.Tensor,
    gt_offsets: torch.Tensor,
    gt_mask: torch.Tensor,
    command: torch.Tensor,
    gt_poses: Optional[torch.Tensor] = None,
    progress_cap: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Worst lateral miss and one-sided progress shortfall of the commanded mode.

    Both are zero when that mode matches GT. Lateral backprops through the
    worst valid waypoint. Progress is polyline length from the origin.
    ``progress_cap`` stops the shortfall once the prediction reaches that
    fraction of the GT length, so the last part of the human path is not
    pulled in. Only a path shorter than ``cap * len(GT)`` is penalized.
    """
    cap = float(progress_cap)
    if not 0.0 < cap <= 1.0:
        raise ValueError(f"progress_cap must be in (0, 1], got {progress_cap}")
    batch, _, steps, _ = ego_fut_preds.shape
    cmd_idx = command.reshape(batch, -1).argmax(dim=-1)
    pred_xy = ego_fut_preds[
        torch.arange(batch, device=ego_fut_preds.device), cmd_idx
    ][..., :2].cumsum(dim=-2)
    gt_xy = gt_offsets[..., :2].cumsum(dim=-2) if gt_poses is None else gt_poses[..., :2]
    mask = gt_mask.reshape(batch, steps).to(dtype=pred_xy.dtype)

    lateral = (pred_xy[..., 1] - gt_xy[..., 1]).abs() * mask
    lat_tail = lateral.amax(dim=-1).mean()

    def _length(xy: torch.Tensor) -> torch.Tensor:
        prev = torch.cat((xy.new_zeros(batch, 1, 2), xy[:, :-1]), dim=1)
        return ((xy - prev).norm(dim=-1) * mask).sum(dim=-1)

    shortfall = (cap * _length(gt_xy) - _length(pred_xy)).clamp_min(0).mean()
    return lat_tail, shortfall


# Pacifica front bumper, rear axle to bumper. Matches
# ``nuplan...get_pacifica_parameters().front_length``, which the PDM scorer uses.
_PACIFICA_FRONT_LENGTH_M = 4.049
_TTC_STEP_S = 0.5
_TTC_PROJECTION_S = (0.5, 1.0)
_TTC_STOP_SPEED_MPS = 5e-3
_TTC_PENETRATION_CLAMP_M = 1.0


def compute_plan_ttc_proxy(
    ego_fut_preds: torch.Tensor,
    gt_mask: torch.Tensor,
    command: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    gt_fut_trajs: torch.Tensor,
    gt_fut_masks: torch.Tensor,
) -> torch.Tensor:
    """Penetration of the commanded bumper's 0.5s and 1.0s probes into GT boxes.

    The polyline is open-loop and 2 Hz. NAVSIM waypoints are converted to SSR
    with :func:`navsim_trajectory_to_ssr`. Agent boxes stay in the SSR code
    ``(x_right, y_forward, z, width, length, height, yaw, ...)``, and the yaw
    axes match the occupancy splat in ``distill/masks.py``. A step slower than
    the scorer's stopped threshold contributes 0. The hinge is the depth inside
    the box, clamped at 1 m, and is 0 for agents behind the rear axle.
    """
    from .distill.masks import navsim_trajectory_to_ssr

    batch, _, steps, _ = ego_fut_preds.shape
    agents = int(gt_boxes.size(1))
    if agents == 0 or steps == 0:
        return ego_fut_preds.new_zeros(())
    if gt_fut_trajs.size(2) != steps:
        raise ValueError(
            "gt_fut_trajs steps must match the plan horizon, "
            f"got {gt_fut_trajs.size(2)} and {steps}"
        )

    cmd_idx = command.reshape(batch, -1).argmax(dim=-1)
    offsets = ego_fut_preds[torch.arange(batch, device=ego_fut_preds.device), cmd_idx]
    poses_xy = offsets[..., :2].cumsum(dim=-2)
    heading = offsets[..., 2].cumsum(dim=-1)
    poses = torch.cat(
        (
            poses_xy,
            torch.atan2(heading.sin(), heading.cos()).unsqueeze(-1),
        ),
        dim=-1,
    )
    ssr = navsim_trajectory_to_ssr(poses)
    origin = poses.new_zeros(batch, 1, 2)
    segment = torch.cat((origin, poses_xy[:, :-1]), dim=1)
    speed = (poses_xy - segment).norm(dim=-1) / _TTC_STEP_S
    forward = torch.stack((ssr[..., 2].cos(), ssr[..., 2].sin()), dim=-1)
    bumper = ssr[..., :2] + _PACIFICA_FRONT_LENGTH_M * forward
    probes = []
    for delta_t in _TTC_PROJECTION_S:
        probes.append(bumper + (speed * delta_t).unsqueeze(-1) * forward)
    points = torch.stack(probes, dim=2)

    centers = gt_boxes[:, :, None, :2] + gt_fut_trajs.cumsum(dim=2)
    centers = centers.permute(0, 2, 1, 3)
    yaw = gt_boxes[:, :, 6]
    half_l = gt_boxes[:, :, 4] * 0.5
    half_w = gt_boxes[:, :, 3] * 0.5
    delta = points[:, :, :, None, :] - centers[:, :, None, :, :]
    cos_y = yaw.cos()[:, None, None, :]
    sin_y = yaw.sin()[:, None, None, :]
    dx = delta[..., 0]
    dy = delta[..., 1]
    along = cos_y * dx + sin_y * dy
    across = -sin_y * dx + cos_y * dy
    outside_l = along.abs() - half_l[:, None, None, :]
    outside_w = across.abs() - half_w[:, None, None, :]
    outside = torch.hypot(outside_l.clamp_min(0), outside_w.clamp_min(0))
    inside = torch.minimum(torch.maximum(outside_l, outside_w), outside.new_zeros(()))
    penetration = (-(outside + inside)).clamp(0, _TTC_PENETRATION_CLAMP_M)

    ahead = ((centers - ssr[:, :, None, :2]) * forward[:, :, None, :]).sum(dim=-1) > 0
    agent_ok = gt_valid.bool()[:, None, :] & (gt_fut_masks > 0.5).permute(0, 2, 1)
    moving = speed > _TTC_STOP_SPEED_MPS
    keep = ahead & agent_ok
    keep = keep[:, :, None, :] & moving[:, :, None, None]
    penetration = penetration.masked_fill(~keep, 0)
    step = penetration.amax(dim=(2, 3))

    mask = gt_mask.reshape(batch, steps).to(dtype=step.dtype)
    counted = mask.sum(dim=-1)
    scene = (step * mask).sum(dim=-1) / counted.clamp_min(1)
    scene = torch.where(counted > 0, scene, torch.zeros_like(scene))
    return scene.mean()


def _plan_command_metrics(
    err: torch.Tensor, weight: torch.Tensor, mode: int
) -> Dict[str, torch.Tensor]:
    # Diagnostics per command. Reported as SUMS and COUNTS, not ratios: turn
    # commands are absent from most batches, and a per-iteration ratio would
    # either read 0.0 (a lie) or NaN (which propagates through all-reduce).
    with torch.no_grad():
        num = err.sum(dim=(0, 2, 3))
        den = weight.sum(dim=(0, 2, 3))
        names = ("left", "straight", "right", "unknown")
        metrics = {}
        for i in range(min(mode, len(names))):
            metrics[f"plan_err_sum/{names[i]}"] = num[i]
            metrics[f"plan_n/{names[i]}"] = den[i]
    return metrics


class ParaSSRLoss(torch.nn.Module):
    """Stateful loss: owns the GradBalancer and the iteration counter."""

    def __init__(self, config):
        super().__init__()
        self._config = config
        self.iteration = 0
        self.balancer: Optional[GradBalancer] = None
        if config.grad_balance_target:
            target = dict(config.grad_balance_target)
            unsupported = set(target) - {"plan", "det", "map"}
            if unsupported:
                raise ValueError(
                    "unsupported grad_balance_target keys "
                    f"{sorted(unsupported)}; use 'det' for the shared "
                    "detection+motion valve and only plan/det/map targets"
                )
            self.balancer = GradBalancer(
                target=target,
                interval=config.grad_balance_interval,
                momentum=config.grad_balance_momentum,
                clamp=tuple(config.grad_balance_clamp),
                warmup_iters=config.grad_balance_warmup_iters,
            )

    def get_extra_state(self) -> Dict:
        """Persist controller state through regular Lightning checkpoints."""
        state = {"iteration": int(self.iteration)}
        if self.balancer is not None:
            state["balancer"] = {
                "scale": dict(self.balancer.scale),
                "seen": sorted(self.balancer._seen),
            }
        return state

    def set_extra_state(self, state: Dict) -> None:
        if not state:
            return
        self.iteration = int(state.get("iteration", 0))
        balancer_state = state.get("balancer")
        if self.balancer is not None and balancer_state is not None:
            restored_scale = balancer_state.get("scale", {})
            for task in self.balancer.scale:
                if task in restored_scale:
                    self.balancer.scale[task] = float(restored_scale[task])
            self.balancer._seen = set(balancer_state.get("seen", ()))

    def apply_aux_scales(self, model) -> None:
        """Synchronize the model valve before every auxiliary forward pass."""
        if self.balancer is None:
            return
        model.aux_grad_scale = {
            task: self.balancer.scale_for(task) for task in ("det", "map")
        }

    # ------------------------------------------------------------------ #
    def forward(
        self,
        model,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self._config
        tw = cfg.task_loss_weight
        logs: Dict[str, torch.Tensor] = {}
        task_losses: Dict[str, torch.Tensor] = {}

        # ---- planning -------------------------------------------------
        if getattr(cfg, "use_metric_planner", False):
            from .modules.candidate_planner import candidate_imitation_loss, candidate_metric_loss
            plan_loss, cls_loss, plan_metrics = candidate_imitation_loss(
                predictions, targets, cfg.heading_weight
            )
            if cfg.metric_loss_weight > 0:
                if "candidate_metric_targets" not in targets:
                    raise ValueError("metric planner loss requires simulator-scored candidate_metric_targets")
                metric_loss, metric_logs = candidate_metric_loss(
                    predictions["metric_logits"], targets["candidate_metric_targets"],
                    cfg.metric_loss_weights,
                )
            else:
                # Same-K imitation-only ablation: no rollout/no invented labels.
                # A zero graph edge keeps the optional critic DDP-compatible.
                metric_loss = predictions["metric_logits"].sum() * 0.0
                metric_logs = {}
            # All three objectives steer the planning branch. Group them before
            # shared-BEV diagnostics so balancing measures the actual gradient.
            plan_total = (plan_loss + cfg.candidate_cls_loss_weight * cls_loss
                          + cfg.metric_loss_weight * metric_loss)
            balance_plan = plan_total
            logs["loss_plan_cls"] = cls_loss.detach()
            logs["loss_plan_metric"] = metric_loss.detach()
            logs["loss_plan_cls_weighted"] = (cls_loss * cfg.candidate_cls_loss_weight * tw.get("plan", 1.0)).detach()
            logs["loss_plan_metric_weighted"] = (metric_loss * cfg.metric_loss_weight * tw.get("plan", 1.0)).detach()
            logs.update(metric_logs)
        else:
            plan_loss, plan_metrics = compute_plan_loss(
                predictions["ego_fut_preds"],
                targets["trajectory_offsets"],
                targets["trajectory_mask"],
                targets["command"],
                heading_weight=cfg.heading_weight,
            )
            lat_w = float(getattr(cfg, "plan_tail_lat_weight", 0.0) or 0.0)
            prog_w = float(getattr(cfg, "plan_tail_progress_weight", 0.0) or 0.0)
            if lat_w != 0.0 or prog_w != 0.0:
                lat_tail, prog_tail = compute_plan_tail_loss(
                    predictions["ego_fut_preds"],
                    targets["trajectory_offsets"],
                    targets["trajectory_mask"],
                    targets["command"],
                    gt_poses=targets.get("trajectory"),
                    progress_cap=float(getattr(cfg, "plan_tail_progress_cap", 1.0)),
                )
            else:
                lat_tail = plan_loss.detach().new_zeros(())
                prog_tail = lat_tail
            # Valve measures imitation L1. The progress shortfall and the TTC
            # proxy stay in the total loss and do not set the det/map scales.
            balance_plan = plan_loss
            ttc_w = float(getattr(cfg, "plan_ttc_proxy_weight", 0.0) or 0.0)
            if ttc_w != 0.0:
                ttc_proxy = compute_plan_ttc_proxy(
                    predictions["ego_fut_preds"],
                    targets["trajectory_mask"],
                    targets["command"],
                    targets["gt_boxes"],
                    targets["gt_valid"],
                    targets["gt_fut_trajs"],
                    targets["gt_fut_masks"],
                )
            else:
                ttc_proxy = plan_loss.detach().new_zeros(())
            plan_total = plan_loss + lat_w * lat_tail + prog_w * prog_tail + ttc_w * ttc_proxy
            logs["loss_plan_tail_lat"] = lat_tail.detach()
            logs["loss_plan_tail_progress"] = prog_tail.detach()
            logs["loss_plan_ttc_proxy"] = ttc_proxy.detach()
            if "trajectory_offset" in predictions:
                # Anchor planner: WoTE's three terms replace the L1 regression.
                # loss_plan_reg above stays the selected-trajectory L1 for logs.
                from .modules.anchor_planner import anchor_plan_losses

                if "sim_reward" not in targets or "sim_reward_valid" not in targets:
                    raise KeyError(
                        "plan_anchor needs sim_reward and sim_reward_valid targets; "
                        "set plan_score_file"
                    )
                anchor_losses = anchor_plan_losses(
                    predictions,
                    targets["trajectory"],
                    targets["sim_reward"],
                    targets["sim_reward_valid"],
                )
                plan_total = (
                    float(getattr(cfg, "plan_offset_loss_weight", 1.0)) * anchor_losses["traj_offset_loss"]
                    + float(getattr(cfg, "plan_im_reward_weight", 1.0)) * anchor_losses["im_reward_loss"]
                    + float(getattr(cfg, "plan_sim_reward_weight", 1.0)) * anchor_losses["sim_reward_loss"]
                )
                balance_plan = plan_total
                logs.update({
                    f"plan_v2/{name}": value.detach() for name, value in anchor_losses.items()
                })
        plan_w = tw.get("plan", 1.0)
        task_losses["plan"] = plan_total * plan_w
        plan_for_balance = balance_plan * plan_w
        logs["loss_plan_reg"] = plan_loss.detach()
        # Keep the historical raw metric, but expose the value that actually
        # enters total_loss so plan=2.0 is not hidden in dashboards.
        logs["loss_plan_reg_weighted"] = (plan_loss * tw.get("plan", 1.0)).detach()
        logs["loss_plan_total"] = task_losses["plan"].detach()
        logs.update(plan_metrics)

        # ---- detection + motion ---------------------------------------
        if model.det_motion_head is not None and "all_cls_scores" in predictions:
            det_losses = model.det_motion_head.loss(
                predictions,
                targets["gt_boxes"],
                targets["gt_labels"],
                targets["gt_valid"],
                targets.get("gt_fut_trajs"),
                targets.get("gt_fut_masks"),
            )
            det_sum = sum(
                v for k, v in det_losses.items() if not k.startswith("loss_traj")
            )
            motion_sum = sum(
                v for k, v in det_losses.items() if k.startswith("loss_traj")
            )
            task_losses["det"] = det_sum * tw.get("det", 1.0)
            if isinstance(motion_sum, torch.Tensor):
                task_losses["motion"] = motion_sum * tw.get("motion", 1.0)
            logs.update({k: v.detach() for k, v in det_losses.items()})

        # ---- vector map -----------------------------------------------
        if model.map_head is not None and "all_map_cls_scores" in predictions:
            map_losses = model.map_head.loss(
                predictions,
                targets["gt_map_pts"],
                targets["gt_map_labels"],
                targets["gt_map_valid"],
            )
            task_losses["map"] = sum(map_losses.values()) * tw.get("map", 1.0)
            logs.update({k: v.detach() for k, v in map_losses.items()})

        # ---- shared-BEV gradient measurement / balancing ---------------
        bev_embed = predictions["bev_embed"]
        measurement_losses = dict(task_losses)
        measurement_losses["plan"] = plan_for_balance
        if "det" in measurement_losses and "motion" in measurement_losses:
            # Detection and motion share one valve. Measure their sum so
            # reinforcement and cancellation both show up in the norm.
            measurement_losses["det"] = (
                measurement_losses["det"] + measurement_losses.pop("motion")
            )
        need_balance = (
            self.balancer is not None
            and model.training
            and bev_embed.requires_grad
            and self.balancer.should_update(self.iteration)
        )
        need_log = (
            cfg.grad_norm_log_interval
            and model.training
            and bev_embed.requires_grad
            and self.iteration % cfg.grad_norm_log_interval == 0
        )
        use_interaction = bool(getattr(model, "use_task_interaction", False))
        if use_interaction:
            # Heads were run on the unscaled BEV. Correct only the auxiliary
            # task's BEV gradient; planning through those decoders stays at 1.
            total_loss, norms = balance_shared_gradients(
                sum(task_losses.values()),
                bev_embed,
                measurement_losses,
                dict(getattr(model, "aux_grad_scale", {})),
                measure_norms=bool(need_balance or need_log),
            )
        else:
            total_loss = sum(task_losses.values())
            norms = (
                self._measure_bev_grad_norms(bev_embed, measurement_losses)
                if need_balance or need_log else {}
            )
        if need_balance or need_log:
            norms = all_reduce_mean(norms, bev_embed.device)
            total = sum(norms.values()) or 1.0
            for k, v in norms.items():
                logs[f"gnorm/{k}"] = torch.tensor(v, device=bev_embed.device)
                logs[f"gshare/{k}"] = torch.tensor(v / total, device=bev_embed.device)
            if need_balance:
                scales = self.balancer.update(norms)
                model.aux_grad_scale = {
                    k: self.balancer.scale_for(k) for k in ("det", "map")
                }
                for k, v in scales.items():
                    logs[k] = torch.tensor(v, device=bev_embed.device)

        if model.training:
            self.iteration += 1
        logs["loss"] = total_loss.detach()
        return total_loss, logs

    # ------------------------------------------------------------------ #
    @staticmethod
    def _measure_bev_grad_norms(
        bev_embed: torch.Tensor, task_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """``||dL_task / d bev_embed||`` per task.

        ``autograd.grad`` differentiates through ``_ScaleGrad``, so the measured
        norm already carries whatever valve is currently in effect -- which is
        exactly what ``GradBalancer.update`` expects and backs out.
        """
        norms: Dict[str, float] = {}
        for task, loss in task_losses.items():
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                continue
            grad = torch.autograd.grad(
                loss, bev_embed, retain_graph=True, allow_unused=True
            )[0]
            norms[task] = 0.0 if grad is None else float(grad.norm())
        return norms
