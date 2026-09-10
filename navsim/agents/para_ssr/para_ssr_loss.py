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

from .modules.grad_balance import GradBalancer, all_reduce_mean


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
    return loss, metrics


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
            plan_total = plan_loss
        task_losses["plan"] = plan_total * tw.get("plan", 1.0)
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
        if need_balance or need_log:
            measurement_losses = dict(task_losses)
            if "det" in measurement_losses and "motion" in measurement_losses:
                # Detection and motion share one ScaleGrad valve.  The valve's
                # actual BEV gradient is grad(L_det + L_motion), not either norm
                # separately and not the sum of their norms.
                measurement_losses["det"] = (
                    measurement_losses["det"] + measurement_losses.pop("motion")
                )
            norms = self._measure_bev_grad_norms(bev_embed, measurement_losses)
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
        total_loss = sum(task_losses.values())
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
