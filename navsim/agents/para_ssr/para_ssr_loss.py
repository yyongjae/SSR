"""Total PARA-SSR loss, per-task weighting, and the shared-BEV gradient valve.

Three multipliers stack, exactly as in the nuScenes config:

1. each head's internal ``loss_*_weight`` (per TERM)
2. the head-level ``loss_weight`` (per HEAD)
3. ``task_loss_weight`` here (per TASK)

``grad_balance`` is deliberately *not* in that stack.  It scales only the
gradient entering ``bev_embed`` and leaves each head's own parameter gradients
alone, which is why it can hold a task's influence on the shared feature down
without slowing the head itself.

Shared-gradient balancing requires FP32 forward/backward. Mixed-precision
training is supported only with balancing disabled and neutral manual scales.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

from .feasibility import DAC_INDEX, NC_INDEX, feasibility_losses
from .modules.anchor_planner import CORRIDOR_KEYS, LABEL_COLS, anchor_plan_losses
from .plan_map import plan_map_loss
from .readout.distill import ReadoutDistiller
from .modules.grad_balance import (
    GradBalancer,
    all_reduce_mean,
    balance_shared_gradients,
    require_full_precision_for_balancing,
)

logger = logging.getLogger(__name__)


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
        self.distiller: Optional[ReadoutDistiller] = None
        if getattr(config, "kd_mode", "none") != "none":
            self.distiller = ReadoutDistiller(config)
        # BEV selector v7 distillation (distill/anchor_distill.py): training-only, owned by the loss
        # so its parameters reach the optimizer (agent.get_optimizers) and leave the eval path alone
        if bool(getattr(config, "distill_v7", False)):
            from .distill import TrajectoryAnchorDistillation
            if not (getattr(config, "plan_anchor", False) and getattr(config, "plan_anchor_file", None)):
                raise ValueError("distill_v7 needs the anchor planner (plan_anchor, plan_anchor_file)")
            self.v7_distill = TrajectoryAnchorDistillation(
                channels=config.embed_dims, num_heads=config.distill_v7_heads, pc_range=config.pc_range,
                bev_h=config.bev_h, bev_w=config.bev_w, winner_boost=config.distill_anchor_winner_boost,
                tau=config.distill_anchor_tau, loss_scale=config.distill_anchor_loss_scale,
                attn_loss_scale=config.distill_attn_loss_scale, resmap_scale=config.distill_resmap_scale,
                spatial_sigma=config.distill_spatial_sigma, loss_type=config.distill_anchor_loss_type,
                use_proj=config.distill_anchor_proj, trajectory_anchors_file=config.plan_anchor_file)
        else:
            self.v7_distill = None
        if config.grad_balance_target:
            target = dict(config.grad_balance_target)
            unsupported = set(target) - {"plan", "det", "map", "distill"}
            if unsupported:
                raise ValueError(
                    "unsupported grad_balance_target keys "
                    f"{sorted(unsupported)}; use 'det' for the shared "
                    "detection+motion valve and only plan/det/map/distill targets"
                )
            # The distillation term reaches the model only through bev_embed
            # (its reader is frozen), so the BEV-level correction scales it
            # exactly.  Without a distillation term there is nothing to steer.
            if "distill" in target and getattr(config, "kd_mode", "none") == "none":
                logger.info("grad_balance_target drops 'distill': kd_mode is none")
                del target["distill"]
            # Head ablations (use_det_motion_head / use_map_head = false) keep
            # the default target dict -- a Hydra override merges keys, it cannot
            # remove one -- so drop the valve of a head that does not exist.
            # The balancer only ever uses target ratios against plan, so the
            # remaining tasks keep exactly the shares they had in the full
            # model (plan 0.4 : det 0.3 with the map head gone, for instance).
            for task, enabled in (
                ("det", getattr(config, "use_det_motion_head", True)),
                ("map", getattr(config, "use_map_head", True)),
            ):
                if task in target and not enabled:
                    logger.info(
                        "grad_balance_target drops %r: that head is disabled", task
                    )
                    del target[task]
            # Plan-only has no relative task share to control, even if a caller
            # retains the Hydra multi-task target dictionary.
            if set(target) == {"plan"} and not (
                getattr(config, "use_det_motion_head", True)
                or getattr(config, "use_map_head", True)
            ):
                logger.info("GradBalancer disabled: planning is the only active task")
            else:
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
        if self.distiller is not None and self.distiller.start_iter is not None:
            state["kd_start_iter"] = int(self.distiller.start_iter)
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
        if self.distiller is not None and "kd_start_iter" in state:
            self.distiller.start_iter = int(state["kd_start_iter"])
        balancer_state = state.get("balancer")
        if self.balancer is not None and balancer_state is not None:
            restored_scale = balancer_state.get("scale", {})
            for task in self.balancer.scale:
                if task in restored_scale:
                    self.balancer.scale[task] = float(restored_scale[task])
            self.balancer._seen = set(balancer_state.get("seen", ()))

    def apply_aux_scales(self, model) -> None:
        """Synchronize task-origin coefficients (never decoder input valves)."""
        if self.balancer is None:
            return
        # distill has no decoder valve; its scale is applied at BEV in forward()
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
        # Validate the original outputs before individual loss routines promote
        # predictions to FP32. Also reject AMP during controller warm-up, rather
        # than letting a long run fail only once its scales become non-neutral.
        active_scales = (
            {task: self.balancer.scale_for(task) for task in ("det", "map", "distill")}
            if self.balancer is not None
            else dict(getattr(model, "aux_grad_scale", {}))
        )
        manual_correction = any(
            prediction_key in predictions and float(active_scales.get(task, 1.0)) != 1.0
            for task, prediction_key in (
                ("det", "all_cls_scores"), ("map", "all_map_cls_scores")
            )
        )
        if model.training and (self.balancer is not None or manual_correction):
            require_full_precision_for_balancing(predictions.values())

        # ---- planning -------------------------------------------------
        plan_loss, plan_metrics = compute_plan_loss(
            predictions["ego_fut_preds"],
            targets["trajectory_offsets"],
            targets["trajectory_mask"],
            targets["command"],
            heading_weight=cfg.heading_weight,
        )
        task_losses["plan"] = plan_loss * tw.get("plan", 1.0)
        logs["loss_plan_reg"] = plan_loss.detach()
        if "trajectory_offset" in predictions:
            # v2 anchor planner: WoTE's three terms replace the L1 regression.  The L1
            # of the SELECTED trajectory stays in the logs (loss_plan_reg and the
            # per-command sums) as the v1-comparable planning error; it is not trained.
            if "sim_reward" not in targets:
                raise KeyError("plan_anchor needs the sim_reward target: set plan_score_file")
            # when the feasibility term owns the drivable-area head, the anchor label for that
            # column must go: it describes the ANCHOR, the feasibility target the emitted trajectory
            on = float(getattr(cfg, "feas_weight", 0.0)) > 0 and bool(getattr(cfg, "feas_to_score", False))
            if str(getattr(cfg, "plan_head_mode", "wote")) == "corridor":
                # heads are CORRIDOR_KEYS: the feasible head has no anchor label (feasibility.py
                # owns it), the other three read WoTE's own columns for the same metric.
                if not on:
                    raise ValueError("plan_head_mode=corridor needs feas_weight>0 and feas_to_score:"
                                     " nothing else supervises the feasible head")
                skip = None
                label_cols = [None] + [LABEL_COLS[k] for k in CORRIDOR_KEYS[1:]]
            else:
                # when the feasibility term owns the drivable-area head, the anchor label for that
                # column must go: it describes the ANCHOR, the feasibility target the emitted one
                skip = [i for i, use in ((DAC_INDEX, on),
                                         (NC_INDEX, on and float(getattr(cfg, "feas_dyn_weight", 0.0)) > 0))
                        if use] or None
                label_cols = None
            # the TTC column may be re-labelled on the emitted trajectory (feasibility.ttc_label)
            from .feasibility import ep_label, ttc_label
            sim_reward, ttc_logs = ttc_label(predictions, targets, cfg, targets["sim_reward"])
            logs.update(ttc_logs)
            # ... and the ego-progress column on the route centerline (feasibility.ep_label)
            sim_reward, ep_logs = ep_label(predictions, targets, cfg, sim_reward)
            logs.update(ep_logs)
            hold_w = float(getattr(cfg, "plan_offset_hold_weight", 0.0))
            al = anchor_plan_losses(predictions, targets["trajectory"], sim_reward,
                                    targets["sim_reward_valid"], skip_sim_index=skip,
                                    label_cols=label_cols,
                                    offset_hold_radius=float(cfg.plan_offset_hold_radius) if hold_w > 0 else None)
            task_losses["plan"] = tw.get("plan", 1.0) * (
                float(cfg.plan_offset_loss_weight) * al["traj_offset_loss"]
                + float(cfg.plan_im_reward_weight) * al["im_reward_loss"]
                + float(cfg.plan_sim_reward_weight) * al["sim_reward_loss"]
                + hold_w * al.get("offset_hold_loss", 0.0))
            logs.update({f"plan_v2/{k}": v.detach() for k, v in al.items()})
            # listwise top-1 ranking inside the corridor (feasibility.admissible_rank_loss)
            from .feasibility import admissible_rank_loss
            rk_loss, rk_logs = admissible_rank_loss(predictions, targets, sim_reward, cfg)
            if rk_loss is not None:
                task_losses["plan"] = task_losses["plan"] + rk_loss
                logs["loss_rank"] = rk_loss.detach()
                logs.update(rk_logs)
            # over-estimation on the would-be picks (feasibility.selection_curse_loss)
            from .feasibility import selection_curse_loss
            cu_loss, cu_logs = selection_curse_loss(predictions, targets, sim_reward, cfg)
            if cu_loss is not None:
                task_losses["plan"] = task_losses["plan"] + cu_loss
                logs["loss_curse"] = cu_loss.detach()
                logs.update(cu_logs)
        # Keep the historical raw metric, but expose the value that actually
        # enters total_loss so plan=2.0 is not hidden in dashboards.
        logs["loss_plan_reg_weighted"] = task_losses["plan"].detach()
        logs.update(plan_metrics)

        # ---- planning-side map consistency (plan_map.py) ---------------
        plan_map_weight = float(getattr(cfg, "plan_map_weight", 0.0))
        if plan_map_weight > 0:
            if "drivable_sdf" not in targets:
                if model.training:
                    raise KeyError("plan_map_weight > 0 needs the drivable_sdf target (DrivableAreaTargetBuilder)")
            else:
                pm_loss, pm_metrics = plan_map_loss(
                    predictions["ego_fut_preds"], targets["command"], targets["trajectory_offsets"],
                    targets["trajectory_mask"], targets["drivable_sdf"], tuple(cfg.plan_map_extent),
                    float(cfg.plan_map_margin),
                )
                task_losses["plan"] = task_losses["plan"] + plan_map_weight * pm_loss
                logs["loss_plan_map"] = pm_loss.detach()
                logs.update(pm_metrics)

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
            map_pts, map_labels, map_valid = self._map_labels(targets, logs)
            map_losses = model.map_head.loss(predictions, map_pts, map_labels, map_valid)
            task_losses["map"] = sum(map_losses.values()) * tw.get("map", 1.0)
            logs.update({k: v.detach() for k, v in map_losses.items()})

        # ---- drivable-area raster (bev_seg.py) --------------------------
        # Folded into the "map" task so the existing GradBalancer keeps one valve for map
        # supervision; it is map supervision, just in the representation DAC actually tests.
        if "drivable_seg_logits" in predictions and "drivable_seg" in targets:
            from .bev_seg import bev_seg_loss

            seg_loss, seg_logs = bev_seg_loss(
                predictions["drivable_seg_logits"], targets["drivable_seg"],
                pos_weight=float(getattr(cfg, "bev_seg_pos_weight", 1.0)),
            )
            seg_loss = float(cfg.bev_seg_weight) * seg_loss * tw.get("map", 1.0)
            task_losses["map"] = task_losses.get("map", seg_loss.new_zeros(())) + seg_loss
            logs["loss_bev_seg"] = seg_loss.detach()
            logs.update(seg_logs)

        # ---- feasibility on the refined candidates (feasibility.py) -----
        # Its own task so the GradBalancer can see it: the constraint is meant to steer planning,
        # and folding it into "map" would hide that behind the map valve.
        feas_loss, feas_logs = feasibility_losses(predictions, targets, cfg, self.iteration)
        if feas_loss is not None:
            task_losses["plan"] = task_losses.get("plan", feas_loss.new_zeros(())) + feas_loss
            logs["loss_feas"] = feas_loss.detach()
            logs.update(feas_logs)
        # ---- driving force and its resistances on the refined set (feasibility.drive_terms) ----
        if float(getattr(cfg, "feas_weight", 0.0)) > 0 and "trajectory_offset" in predictions:
            from .feasibility import drive_terms
            drive_loss, drive_logs = drive_terms(predictions, targets, features, cfg)
            if drive_loss is not None:
                task_losses["plan"] = task_losses.get("plan", drive_loss.new_zeros(())) + drive_loss
                logs["loss_drive"] = drive_loss.detach()
                logs.update(drive_logs)

        # ---- readout-space distillation (report/19) ---------------------
        if self.distiller is not None and "teacher_bev" in targets:
            ego = features.get("status_feature")
            ego = ego[:, cfg.num_navi_cmd:] if ego is not None else None
            kd = self.distiller(predictions["bev_embed"], targets, self.iteration, ego=ego)
            task_losses["distill"] = kd["loss"]
            logs["loss_distill"] = kd["loss"].detach()
            logs["kd/raw"] = kd["raw"]
            logs["kd/coef"] = kd["coef"]
            logs["kd/valid_frac"] = kd["valid_frac"]
            if "raw_uncentered" in kd:
                logs["kd/raw_uncentered"] = kd["raw_uncentered"]

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
        measurement_losses = dict(task_losses)
        if "motion" in measurement_losses:
            # Combine before differentiation to account for cancellation.
            measurement_losses["det"] = measurement_losses.get("det", 0.0) + (
                measurement_losses.pop("motion")
            )
        # The coefficients captured above belong to this graph. A controller
        # update below applies to the next microbatch.
        total_loss, norms = balance_shared_gradients(
            sum(task_losses.values()),
            bev_embed,
            measurement_losses,
            active_scales if model.training else {},
            measure_norms=bool(need_balance or need_log),
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

        # ---- BEV selector v7 distillation (distill/anchor_distill.py) ----------
        # Added after balancing, as in the teammate's run (GradBalancer off there): it reaches the
        # shared BEV unscaled.  The teammate adds weight * EVERY returned term -- both bank losses and
        # their mean loss_distill_tok, so the gradient is 1.5 x the bank sum; loss_attn_* are detached
        # logs that only shift the value.  Kept exactly.  bev_pos = the planner's own encoding.
        if self.v7_distill is not None and "v7_teacher_bevfusion" in targets:
            sf = features["status_feature"]
            bs = bev_embed.shape[0]
            tok = lambda m: m.flatten(2).transpose(1, 2).to(bev_embed.dtype)          # noqa: E731
            bev_pos = model.pts_bbox_head.positional_encoding(
                bev_embed.new_zeros((bs, cfg.bev_h, cfg.bev_w))).to(bev_embed.dtype).flatten(2).transpose(1, 2)
            v7_losses, v7_metrics = self.v7_distill(
                bev_embed, (tok(targets["v7_teacher_bevfusion"]), tok(targets["v7_teacher_resmap"])),
                predictions["trajectory_anchors"], gt_trajectory=targets["trajectory"],
                command=sf[:, :cfg.num_navi_cmd], ego_status=sf[:, cfg.num_navi_cmd:], bev_pos=bev_pos,
                sample_valid=targets.get("v7_teacher_valid"))
            for name, value in v7_losses.items():
                total_loss = total_loss + float(cfg.distill_v7_weight) * value
                logs[f"v7/{name}"] = value.detach()
            logs.update({f"v7/{k}": v for k, v in v7_metrics.items()})
            if "v7_teacher_valid" in targets:
                logs["v7/valid_frac"] = targets["v7_teacher_valid"].float().mean().detach()

        # ---- stage 2: refine the pick (refine.py) ------------------------
        # Added after balancing: it reaches neither the shared BEV nor stage 1 (its inputs are
        # detached), so the balancer has nothing to measure and must not rescale it.
        refiner = getattr(model, "refiner", None)
        if refiner is not None and "plan_query" in predictions and "drivable_sdf_pred" in predictions:
            from .refine import refine_losses, refine_select
            status = features["status_feature"][:, cfg.num_navi_cmd:cfg.num_navi_cmd + 4]
            predictions.update(refine_select(refiner, predictions, predictions["plan_query"], status, cfg,
                                             gt_trajectory=targets["trajectory"]))
            ref_loss, ref_logs = refine_losses(predictions, targets, features, cfg)
            if ref_loss is not None:
                total_loss = total_loss + ref_loss
                logs["loss_refine"] = ref_loss.detach()
                logs.update(ref_logs)

        if model.training:
            self.iteration += 1
        logs["loss"] = total_loss.detach()
        # LiDAR/camera blend per encoder layer.  Verification on real scenes
        # (report/12 5-2) showed the LiDAR-seeded BEV dominating early on, so the
        # gate and the two attention magnitudes are what to watch for a starved
        # camera branch.
        head = getattr(model, "pts_bbox_head", None)
        encoder = getattr(getattr(head, "transformer", None), "encoder", None)
        for index, layer in enumerate(getattr(encoder, "layers", [])):
            gate = getattr(layer, "lidar_gate", None)
            if gate is not None:
                logs[f"lidar_gate/layer{index}"] = torch.sigmoid(gate).mean().detach()
            for name, value in (getattr(layer, "last_attn_norms", None) or {}).items():
                logs[f"attn_norm/{name}_layer{index}"] = value

        return total_loss, logs

    # ------------------------------------------------------------------ #
    def _map_labels(self, targets, logs):
        """GT map labels, or the teacher's where the teacher has the frame."""
        gt = (targets["gt_map_pts"], targets["gt_map_labels"], targets["gt_map_valid"])
        if getattr(self._config, "map_label_source", "gt") != "teacher":
            return gt
        if "teacher_map_pts" not in targets:
            raise KeyError(
                "map_label_source='teacher' but the batch has no teacher_map_pts; "
                "is ResMapTeacherTargetBuilder registered?"
            )
        hit = targets["teacher_valid"].view(-1).bool()
        logs["kd/pseudo_frac"] = hit.float().mean().detach()
        pick = lambda t, g: torch.where(hit.view(-1, *([1] * (g.dim() - 1))), t.to(g.dtype), g)
        return (
            pick(targets["teacher_map_pts"], gt[0]),
            pick(targets["teacher_map_labels"], gt[1]),
            pick(targets["teacher_map_valid"], gt[2]),
        )

    @staticmethod
    def _measure_bev_grad_norms(
        bev_embed: torch.Tensor, task_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """``||dL_task / d bev_embed||`` per task.

        These are raw task-origin norms, including all direct and indirect
        routes. Training applies coefficients explicitly in
        ``balance_shared_gradients``; no decoder boundary scales them.
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
