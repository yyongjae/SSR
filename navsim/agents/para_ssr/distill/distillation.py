"""Stage-2 frozen-adapter BEV distillation for the navsim PARA-SSR agent.

The contract is the one in ``docs/PLANNING_DISTILLATION.md``: a branch uses one
*shared adapter instance* for the teacher and the student forward, and that
adapter is frozen.  The adapter therefore cannot learn to hide a mismatch, and
the feature-loss gradient passes through it into the student BEV encoder::

    feature loss -> frozen task adapter -> student BEV encoder

Nothing here survives to inference: the adapters and the teacher cache are used
only while training.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import PlanningBEVAdapter, bev_map_to_tokens, bev_tokens_to_map
from .kd_losses import (
    attention_imitation,
    channel_wise_kd,
    masked_mse,
    planning_look_kd,
    relation_kd,
)
from .masks import (
    combine_planning_prior,
    combine_role_mask,
    compute_corridor_mask,
    morphological_boundary,
    navsim_trajectory_to_ssr,
    rasterize_agent_mask,
    rasterize_map_class_mask,
    tokens_from_mask,
)
from .selector import BEVRegisterSelector
from .teacher_store import TeacherFeatureStore

# Stage-1 checkpoints are Lightning files whose keys carry an ``agent.`` prefix.
# Accept the plain module too so an exported adapter-only state dict works.
_ADAPTER_PREFIXES = (
    "agent.model.adapter.",
    "model.adapter.",
    "adapter.",
    "",
)


def _checkpoint_state(path: str) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(
        checkpoint, Mapping) else checkpoint
    if not isinstance(state, (dict, OrderedDict)):
        raise TypeError(f"{path}: checkpoint does not contain a state dict")
    return state


def load_prefixed_module(module: nn.Module, state: Mapping[str, torch.Tensor],
                         prefixes: Sequence[str] = _ADAPTER_PREFIXES) -> str:
    """Strictly load ``module`` from the first prefix that matches."""
    wanted = set(module.state_dict().keys())
    for prefix in prefixes:
        selected = {
            key[len(prefix):]: value for key, value in state.items()
            if key.startswith(prefix)
        }
        if selected and wanted <= set(selected):
            module.load_state_dict(
                {k: v for k, v in selected.items() if k in wanted}, strict=True)
            return prefix
    raise KeyError(
        f"checkpoint has no adapter parameters under any of {prefixes}; "
        "expected a stage-1 teacher-adapter checkpoint"
    )


class PlanningDistillation(nn.Module):
    """Frozen teacher adapters and the feature losses attached to PARA-SSR."""

    def __init__(
        self,
        feature_root: str,
        branches: Mapping[str, Mapping],
        adapter_checkpoint: Mapping[str, str],
        student_bev_size: Tuple[int, int] = (50, 100),
        cache_size: Tuple[int, int] = (50, 100),
        strict_checkpoints: bool = True,
        use_corridor_mask: bool = True,
        pc_range: Tuple[float, ...] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        corridor_sigma_base: float = 2.5,
        corridor_sigma_growth: float = 0.1,
        corridor_base_weight: float = 0.1,
        corridor_sigma_along: float = 4.0,
        corridor_sigma_cross: float = 2.5,
        corridor_sigma_along_growth: float = 0.15,
        corridor_sigma_cross_growth: float = 0.1,
        trajectory_frame: str = "navsim",
        use_role_masks: bool = True,
        agent_inflate: float = 1.5,
        agent_future_sigma: float = 1.5,
        map_sigma: float = 1.25,
        walkway_suppress: float = 0.0,
        boundary_kernel: int = 3,
        mse_weight: float = 1.0,
        cwd_weight: float = 0.0,
        cwd_tau: float = 4.0,
        relation_weight: float = 0.0,
        relation_samples: int = 64,
        attn_weight: float = 0.0,
        attn_tau: float = 0.5,
        adaptive_branch: bool = False,
        adaptive_momentum: float = 0.9,
        adaptive_clamp: Tuple[float, float] = (0.25, 4.0),
        plan_look_weight: float = 0.5,
        plan_look_tau: float = 0.5,
        selector_mode: bool = False,
        selector_registers: int = 16,
        selector_div_sigma: float = 4.0,
        selector_tok_warmup_steps: int = 53195,
        selector_channels: int = 256,
        selector_commands: int = 4,
        selector_anchor_sigma: float = 6.0,
        selector_anchor_sigma_end: float = 2.0,
        selector_plan_tau: float = 0.3,
        selector_struct_mix: float = 0.5,
    ) -> None:
        super().__init__()
        if not branches:
            raise ValueError("distillation needs at least one teacher branch")
        frame = str(trajectory_frame).lower()
        if frame not in {"navsim", "ssr"}:
            raise ValueError(
                f"trajectory_frame must be 'navsim' or 'ssr', got {trajectory_frame!r}"
            )
        self.student_bev_size = tuple(int(v) for v in student_bev_size)
        self.cache_size = tuple(int(v) for v in cache_size)
        self.use_corridor_mask = bool(use_corridor_mask)
        self.pc_range = tuple(float(v) for v in pc_range)
        self.corridor_sigma_base = float(corridor_sigma_base)
        self.corridor_sigma_growth = float(corridor_sigma_growth)
        self.corridor_base_weight = float(corridor_base_weight)
        self.corridor_sigma_along = float(corridor_sigma_along)
        self.corridor_sigma_cross = float(corridor_sigma_cross)
        self.corridor_sigma_along_growth = float(corridor_sigma_along_growth)
        self.corridor_sigma_cross_growth = float(corridor_sigma_cross_growth)
        self.trajectory_frame = frame
        self.use_role_masks = bool(use_role_masks)
        self.agent_inflate = float(agent_inflate)
        self.agent_future_sigma = float(agent_future_sigma)
        self.map_sigma = float(map_sigma)
        self.walkway_suppress = float(walkway_suppress)
        self.boundary_kernel = int(boundary_kernel)
        self.mse_weight = float(mse_weight)
        self.cwd_weight = float(cwd_weight)
        self.cwd_tau = float(cwd_tau)
        self.relation_weight = float(relation_weight)
        self.relation_samples = int(relation_samples)
        self.attn_weight = float(attn_weight)
        self.attn_tau = float(attn_tau)
        self.adaptive_branch = bool(adaptive_branch)
        self.adaptive_momentum = float(adaptive_momentum)
        self.adaptive_clamp = (float(adaptive_clamp[0]), float(adaptive_clamp[1]))
        self.plan_look_weight = float(plan_look_weight)
        self.plan_look_tau = float(plan_look_tau)
        self.selector_mode = bool(selector_mode)
        self.selector_registers = int(selector_registers)
        self.selector_div_sigma = float(selector_div_sigma)
        self.selector_tok_warmup_steps = int(selector_tok_warmup_steps)
        self.selector_channels = int(selector_channels)
        self.selector_commands = int(selector_commands)
        self.selector_anchor_sigma = float(selector_anchor_sigma)
        self.selector_anchor_sigma_end = float(selector_anchor_sigma_end)
        self.selector_plan_tau = float(selector_plan_tau)
        self.selector_struct_mix = float(selector_struct_mix)

        self.adapters = nn.ModuleDict()
        self.stores: Dict[str, TeacherFeatureStore] = {}
        self.loss_weights: Dict[str, float] = {}
        self.loaded_adapter_checkpoints: Dict[str, str] = {}
        self.teacher_tokens_student_grid: Dict[str, torch.Tensor] = {}
        self.student_tokens_student_grid: Dict[str, torch.Tensor] = {}

        if self.selector_mode:
            # Registers replace the frozen adapter. Stage-1 checkpoints are not
            # read, so the v4 launcher is the only path that requires them.
            active_branch_names = [
                name for name in ("bevfusion", "resmap") if name in branches
            ]
            if active_branch_names != ["bevfusion", "resmap"]:
                raise KeyError(
                    "BEV selector distillation needs both bevfusion and resmap "
                    f"branches, got {list(branches.keys())}"
                )
        else:
            active_branch_names = [
                name for name in branches
                if name in adapter_checkpoint and adapter_checkpoint[name]
            ]
            if strict_checkpoints and not active_branch_names:
                raise KeyError(
                    f"no stage-1 adapter checkpoint for any branch in {list(branches.keys())}. Train "
                    "them first; stage 2 is defined only against frozen adapters."
                )

        states: Dict[str, Mapping[str, torch.Tensor]] = {}
        for name in active_branch_names:
            raw = branches[name]
            cfg = dict(raw)
            if self.selector_mode:
                cfg.pop("adapter", None)
            else:
                adapter = PlanningBEVAdapter(**dict(cfg.pop("adapter", {})))
                path = adapter_checkpoint.get(name)
                if path:
                    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
                    if path not in states:
                        states[path] = _checkpoint_state(path)
                    load_prefixed_module(adapter, states[path])
                    self.loaded_adapter_checkpoints[name] = path
                elif strict_checkpoints:
                    raise KeyError(f"empty adapter checkpoint path for {name}")
                adapter.requires_grad_(False)
                adapter.eval()
                self.adapters[name] = adapter

            cache_name = cfg.pop("cache_name", name)
            store_kwargs = {k: cfg.pop(k) for k in ("cache_subdirs", "feature_key",
                                                    "flip_w") if k in cfg}
            if "cache_subdirs" not in store_kwargs:
                grid_subdirs = (f"cache_train_{self.cache_size[0]}x{self.cache_size[1]}",
                                f"cache_val_{self.cache_size[0]}x{self.cache_size[1]}")
                if os.path.isdir(os.path.join(feature_root, cache_name, grid_subdirs[0])):
                    store_kwargs["cache_subdirs"] = grid_subdirs
            self.stores[name] = TeacherFeatureStore(
                feature_root, cache_name, **store_kwargs)
            self.loss_weights[name] = float(cfg.pop("loss_weight", 1.0))
            if cfg:
                raise TypeError(f"unused distillation options for {name}: {cfg}")
            self.register_buffer(
                f"ema_loss_{name}", torch.tensor(1.0), persistent=False
            )

        self.selector: Optional[BEVRegisterSelector] = None
        if self.selector_mode:
            self.selector = BEVRegisterSelector(
                channels=self.selector_channels,
                num_registers=self.selector_registers,
                num_commands=self.selector_commands,
                bev_h=self.student_bev_size[0],
                bev_w=self.student_bev_size[1],
                pc_range=self.pc_range,
                div_sigma_m=self.selector_div_sigma,
                tok_warmup_steps=self.selector_tok_warmup_steps,
                anchor_sigma_m=self.selector_anchor_sigma,
                anchor_sigma_end_m=self.selector_anchor_sigma_end,
                plan_tau=self.selector_plan_tau,
                struct_mix=self.selector_struct_mix,
            )

    # ------------------------------------------------------------------ #
    def validate_manifests(self, config) -> None:
        for store in self.stores.values():
            store.validate_manifest(config)

    def train(self, mode: bool = True):
        super().train(mode)
        # A parent .train() must never turn the teacher feature space into a
        # moving target; dropout is configurable even though it defaults to 0.
        for adapter in self.adapters.values():
            adapter.eval()
        return self

    def _ssr_trajectory(self, trajectories: torch.Tensor) -> torch.Tensor:
        if self.trajectory_frame == "navsim":
            return navsim_trajectory_to_ssr(trajectories)
        return trajectories

    def _corridor(
        self,
        height: int,
        width: int,
        traj: Optional[torch.Tensor],
        cache: Dict,
    ) -> Optional[torch.Tensor]:
        key = ("corridor", height, width)
        if key not in cache and traj is not None:
            cache[key] = compute_corridor_mask(
                trajectories=traj,
                pc_range=self.pc_range,
                bev_h=height,
                bev_w=width,
                sigma_base=self.corridor_sigma_base,
                sigma_growth=self.corridor_sigma_growth,
                base_weight=self.corridor_base_weight,
                sigma_along=self.corridor_sigma_along,
                sigma_cross=self.corridor_sigma_cross,
                sigma_along_growth=self.corridor_sigma_along_growth,
                sigma_cross_growth=self.corridor_sigma_cross_growth,
            )
        return cache.get(key)

    def _aux_rasters(
        self,
        height: int,
        width: int,
        gt_boxes: Optional[torch.Tensor],
        gt_valid: Optional[torch.Tensor],
        gt_fut_trajs: Optional[torch.Tensor],
        gt_fut_masks: Optional[torch.Tensor],
        gt_map_pts: Optional[torch.Tensor],
        gt_map_labels: Optional[torch.Tensor],
        gt_map_valid: Optional[torch.Tensor],
        cache: Dict,
        *,
        want_agent: bool,
        want_map: bool,
    ) -> Dict[str, torch.Tensor]:
        extras: Dict[str, torch.Tensor] = {}
        if want_agent and gt_boxes is not None and gt_valid is not None:
            agent_key = ("agent", height, width)
            if agent_key not in cache:
                cache[agent_key] = rasterize_agent_mask(
                    gt_boxes, gt_valid, self.pc_range, height, width,
                    inflate=self.agent_inflate,
                    gt_fut_trajs=gt_fut_trajs, gt_fut_masks=gt_fut_masks,
                    future_sigma=self.agent_future_sigma,
                )
            extras["agent"] = cache[agent_key]
        if want_map and gt_map_pts is not None and gt_map_labels is not None \
                and gt_map_valid is not None:
            for class_id, tag in (
                (0, "road"),
                (1, "walkway"),
                (2, "centerline"),
                (3, "crosswalk"),
            ):
                class_key = (tag, height, width)
                if class_key not in cache:
                    cache[class_key] = rasterize_map_class_mask(
                        gt_map_pts, gt_map_labels, gt_map_valid, class_id,
                        self.pc_range, height, width, sigma=self.map_sigma,
                    )
                extras[tag] = cache[class_key]
            if "road" in extras:
                boundary_key = ("boundary", height, width)
                if boundary_key not in cache:
                    cache[boundary_key] = morphological_boundary(
                        extras["road"], kernel=self.boundary_kernel,
                    )
                extras["boundary"] = cache[boundary_key]
        return extras

    def _branch_mask(
        self,
        name: str,
        height: int,
        width: int,
        traj: Optional[torch.Tensor],
        gt_boxes: Optional[torch.Tensor],
        gt_valid: Optional[torch.Tensor],
        gt_fut_trajs: Optional[torch.Tensor],
        gt_fut_masks: Optional[torch.Tensor],
        gt_map_pts: Optional[torch.Tensor],
        gt_map_labels: Optional[torch.Tensor],
        gt_map_valid: Optional[torch.Tensor],
        cache: Dict,
    ) -> Optional[torch.Tensor]:
        corridor = self._corridor(height, width, traj, cache)
        if corridor is None and not self.use_role_masks:
            return None
        if not self.use_role_masks:
            return corridor

        extras = self._aux_rasters(
            height, width, gt_boxes, gt_valid, gt_fut_trajs, gt_fut_masks,
            gt_map_pts, gt_map_labels, gt_map_valid, cache,
            want_agent=name == "bevfusion",
            want_map=name == "resmap",
        )
        if corridor is None:
            template = next(iter(extras.values()), None)
            if template is None:
                return None
            corridor = torch.full_like(template, self.corridor_base_weight)
        return combine_role_mask(
            name, corridor, base_weight=self.corridor_base_weight,
            walkway_suppress=self.walkway_suppress, **extras,
        )

    def _planning_prior(
        self,
        height: int,
        width: int,
        traj: Optional[torch.Tensor],
        gt_boxes: Optional[torch.Tensor],
        gt_valid: Optional[torch.Tensor],
        gt_fut_trajs: Optional[torch.Tensor],
        gt_fut_masks: Optional[torch.Tensor],
        gt_map_pts: Optional[torch.Tensor],
        gt_map_labels: Optional[torch.Tensor],
        gt_map_valid: Optional[torch.Tensor],
        cache: Dict,
    ) -> Optional[torch.Tensor]:
        corridor = self._corridor(height, width, traj, cache)
        extras = self._aux_rasters(
            height, width, gt_boxes, gt_valid, gt_fut_trajs, gt_fut_masks,
            gt_map_pts, gt_map_labels, gt_map_valid, cache,
            want_agent=True, want_map=True,
        )
        prior_extras = {
            key: extras[key]
            for key in ("agent", "road", "centerline", "crosswalk", "boundary")
            if key in extras
        }
        if corridor is None:
            template = next(iter(prior_extras.values()), None)
            if template is None:
                return None
            corridor = torch.full_like(template, self.corridor_base_weight)
        return combine_planning_prior(
            corridor, base_weight=self.corridor_base_weight, **prior_extras,
        )

    def _adaptive_scale(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not self.adaptive_branch:
            return value
        ema_name = f"ema_loss_{name}"
        ema = getattr(self, ema_name)
        detached = value.detach()
        if not torch.isfinite(detached):
            return value
        ema.mul_(self.adaptive_momentum).add_(
            detached, alpha=1.0 - self.adaptive_momentum
        )
        emas = [getattr(self, f"ema_loss_{n}") for n in self.adapters]
        mean_ema = torch.stack(emas).mean().clamp_min(1e-6)
        scale = (mean_ema / ema.clamp_min(1e-6)).clamp(*self.adaptive_clamp)
        return value * scale

    def _feature_losses(
        self,
        name: str,
        student: torch.Tensor,
        teacher: torch.Tensor,
        mask: Optional[torch.Tensor],
        teacher_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask_tokens = None if mask is None else tokens_from_mask(mask)
        metrics: Dict[str, torch.Tensor] = {}
        terms = []
        mse = masked_mse(student, teacher, mask_tokens)
        metrics[f"distill_mse/{name}"] = mse.detach()
        if self.mse_weight != 0.0:
            terms.append(self.mse_weight * mse)

        student_map = bev_tokens_to_map(student, teacher_hw)
        teacher_map = bev_tokens_to_map(teacher, teacher_hw)
        if self.cwd_weight != 0.0:
            cwd = channel_wise_kd(
                student_map, teacher_map, mask=mask, tau=self.cwd_tau
            )
            terms.append(self.cwd_weight * cwd)
            metrics[f"distill_cwd/{name}"] = cwd.detach()
        if self.relation_weight != 0.0:
            rel = relation_kd(
                student_map, teacher_map, mask=mask,
                num_samples=self.relation_samples,
            )
            terms.append(self.relation_weight * rel)
            metrics[f"distill_rel/{name}"] = rel.detach()
        if self.attn_weight != 0.0:
            attn = attention_imitation(
                student_map, teacher_map, tau=self.attn_tau
            )
            terms.append(self.attn_weight * attn)
            metrics[f"distill_attn/{name}"] = attn.detach()

        total = sum(terms) if terms else mse
        return total * self.loss_weights[name], metrics

    def forward(
        self,
        student_bev: torch.Tensor,
        tokens: Sequence[str],
        trajectories: Optional[torch.Tensor] = None,
        gt_boxes: Optional[torch.Tensor] = None,
        gt_valid: Optional[torch.Tensor] = None,
        gt_fut_trajs: Optional[torch.Tensor] = None,
        gt_fut_masks: Optional[torch.Tensor] = None,
        gt_map_pts: Optional[torch.Tensor] = None,
        gt_map_labels: Optional[torch.Tensor] = None,
        gt_map_valid: Optional[torch.Tensor] = None,
        plan_attn: Optional[torch.Tensor] = None,
        command: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
    ):
        """``student_bev`` is ``[B, bev_h*bev_w, C]``; returns (losses, metrics)."""
        if self.selector_mode:
            return self._selector_forward(
                student_bev, tokens, plan_attn, command, ego_status,
                gt_boxes, gt_valid, gt_map_pts, gt_map_labels, gt_map_valid,
            )
        student_map = bev_tokens_to_map(student_bev, self.student_bev_size)
        traj = None
        if self.use_corridor_mask and trajectories is not None and trajectories.numel() > 0:
            if trajectories.dim() == 3 and trajectories.size(0) == student_map.size(0):
                traj = self._ssr_trajectory(
                    trajectories.to(device=student_map.device, dtype=student_map.dtype)
                )

        mask_cache: Dict = {}
        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        self.teacher_tokens_student_grid = {}
        self.student_tokens_student_grid = {}
        student_hw = self.student_bev_size

        for name, adapter in self.adapters.items():
            teacher_map = self.stores[name].load_batch(
                tokens, student_map.device, student_map.dtype)
            if teacher_map.size(0) != student_map.size(0):
                raise ValueError(
                    f"{name}: {teacher_map.size(0)} cached samples for "
                    f"{student_map.size(0)} student samples"
                )
            teacher_hw = (int(teacher_map.size(-2)), int(teacher_map.size(-1)))
            # 50x100 student, BEVFusion cache_train_50x100, and ReSMap after
            # transpose+flip are the same grid, so this is a no-op. A mismatched
            # cache is still resized onto the teacher grid before the MSE.
            if tuple(student_map.shape[-2:]) != teacher_hw:
                aligned_student = F.interpolate(
                    student_map, size=teacher_hw, mode="bilinear",
                    align_corners=False)
            else:
                aligned_student = student_map
            mask = self._branch_mask(
                name, teacher_hw[0], teacher_hw[1], traj,
                gt_boxes, gt_valid, gt_fut_trajs, gt_fut_masks,
                gt_map_pts, gt_map_labels, gt_map_valid, mask_cache,
            )
            with torch.no_grad():
                teacher = adapter(teacher_map)
            student = adapter(aligned_student)

            branch_loss, branch_metrics = self._feature_losses(
                name, student, teacher, mask, teacher_hw
            )
            losses[f"loss_distill_{name}"] = self._adaptive_scale(name, branch_loss)
            metrics.update(branch_metrics)

            with torch.no_grad():
                cos = F.cosine_similarity(student, teacher, dim=-1)
                mask_tokens = None if mask is None else tokens_from_mask(mask)
                if mask_tokens is not None:
                    metrics[f"distill_cos/{name}"] = (
                        (cos * mask_tokens.squeeze(-1)).sum() / mask_tokens.sum()
                    )
                else:
                    metrics[f"distill_cos/{name}"] = cos.mean()
                metrics[f"distill_rmse/{name}"] = losses[f"loss_distill_{name}"].detach().clamp_min(0).sqrt()
                teacher_map_out = bev_tokens_to_map(teacher.detach(), teacher_hw)
                if teacher_hw != student_hw:
                    teacher_map_out = F.interpolate(
                        teacher_map_out, size=student_hw, mode="bilinear",
                        align_corners=False,
                    )
                self.teacher_tokens_student_grid[name] = bev_map_to_tokens(
                    teacher_map_out
                )

            student_map_out = bev_tokens_to_map(student, teacher_hw)
            if teacher_hw != student_hw:
                student_map_out = F.interpolate(
                    student_map_out, size=student_hw, mode="bilinear",
                    align_corners=False,
                )
            self.student_tokens_student_grid[name] = bev_map_to_tokens(
                student_map_out
            )

        if self.plan_look_weight != 0.0:
            prior = self._planning_prior(
                student_hw[0], student_hw[1], traj,
                gt_boxes, gt_valid, gt_fut_trajs, gt_fut_masks,
                gt_map_pts, gt_map_labels, gt_map_valid, mask_cache,
            )
            if prior is not None:
                look = planning_look_kd(
                    student_map, prior, tau=self.plan_look_tau,
                )
                losses["loss_distill_plan_look"] = self.plan_look_weight * look
                metrics["distill_plan_look"] = look.detach()
        return losses, metrics

    def _selector_structure(
        self,
        student_bev: torch.Tensor,
        gt_boxes: Optional[torch.Tensor],
        gt_valid: Optional[torch.Tensor],
        gt_map_pts: Optional[torch.Tensor],
        gt_map_labels: Optional[torch.Tensor],
        gt_map_valid: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Agent occupancy and the road-boundary ring, flattened to ``[B, HW]``.

        Current boxes only. The ring is the morphological boundary of the road
        splat. Neither one is the GT trajectory corridor.
        """
        height, width = self.student_bev_size
        extras = self._aux_rasters(
            height, width, gt_boxes, gt_valid, None, None,
            gt_map_pts, gt_map_labels, gt_map_valid, {},
            want_agent=True, want_map=True,
        )

        def _flat(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None:
                return None
            flat = mask.detach().flatten(2).squeeze(1)
            if flat.shape[-1] != height * width:
                raise ValueError(
                    f"structure length {flat.shape[-1]} != {height * width}"
                )
            return flat.to(device=student_bev.device, dtype=student_bev.dtype)

        return _flat(extras.get("agent")), _flat(extras.get("boundary"))

    def _selector_forward(
        self,
        student_bev: torch.Tensor,
        tokens: Sequence[str],
        plan_attn: Optional[torch.Tensor],
        command: Optional[torch.Tensor],
        ego_status: Optional[torch.Tensor],
        gt_boxes: Optional[torch.Tensor] = None,
        gt_valid: Optional[torch.Tensor] = None,
        gt_map_pts: Optional[torch.Tensor] = None,
        gt_map_labels: Optional[torch.Tensor] = None,
        gt_map_valid: Optional[torch.Tensor] = None,
    ):
        if self.selector is None:
            raise RuntimeError("selector distillation was not constructed")
        if plan_attn is None or command is None:
            raise ValueError(
                "BEV selector distillation needs plan_bev_attn and command from "
                "the student forward"
            )
        teachers = []
        for name in ("bevfusion", "resmap"):
            teacher_map = self.stores[name].load_batch(
                tokens, student_bev.device, student_bev.dtype)
            student_hw = self.student_bev_size
            if tuple(teacher_map.shape[-2:]) != student_hw:
                teacher_map = F.interpolate(
                    teacher_map, size=student_hw, mode="bilinear", align_corners=False,
                )
            teachers.append(bev_map_to_tokens(teacher_map))
        structure = self._selector_structure(
            student_bev, gt_boxes, gt_valid, gt_map_pts, gt_map_labels, gt_map_valid,
        )
        return self.selector(
            student_bev, teachers, plan_attn, command, ego_status, structure,
        )


def build_planning_distillation(config):
    """Construct the stage-2 module from a :class:`ParaSSRConfig`, or ``None``."""
    if not getattr(config, "use_distill", False):
        return None
    branches = {
        name: dict(spec) for name, spec in config.distill_branches.items()
    }
    module = PlanningDistillation(
        feature_root=config.distill_feature_root,
        branches=branches,
        adapter_checkpoint=dict(config.distill_adapter_checkpoints),
        student_bev_size=(config.bev_h, config.bev_w),
        cache_size=tuple(config.distill_cache_size),
        use_corridor_mask=getattr(config, "use_corridor_mask", True),
        pc_range=tuple(config.pc_range),
        corridor_sigma_base=float(getattr(config, "corridor_sigma_base", 2.5)),
        corridor_sigma_growth=float(getattr(config, "corridor_sigma_growth", 0.1)),
        corridor_base_weight=float(getattr(config, "corridor_base_weight", 0.1)),
        corridor_sigma_along=float(getattr(config, "corridor_sigma_along", 4.0)),
        corridor_sigma_cross=float(getattr(config, "corridor_sigma_cross", 2.5)),
        corridor_sigma_along_growth=float(
            getattr(config, "corridor_sigma_along_growth", 0.15)
        ),
        corridor_sigma_cross_growth=float(
            getattr(config, "corridor_sigma_cross_growth", 0.1)
        ),
        trajectory_frame=str(getattr(config, "distill_trajectory_frame", "navsim")),
        use_role_masks=bool(getattr(config, "distill_use_role_masks", True)),
        agent_inflate=float(getattr(config, "distill_agent_inflate", 1.5)),
        agent_future_sigma=float(getattr(config, "distill_agent_future_sigma", 1.5)),
        map_sigma=float(getattr(config, "distill_map_sigma", 1.25)),
        walkway_suppress=float(getattr(config, "distill_walkway_suppress", 0.0)),
        boundary_kernel=int(getattr(config, "distill_boundary_kernel", 3)),
        mse_weight=float(getattr(config, "distill_mse_weight", 1.0)),
        cwd_weight=float(getattr(config, "distill_cwd_weight", 0.0)),
        cwd_tau=float(getattr(config, "distill_cwd_tau", 4.0)),
        relation_weight=float(getattr(config, "distill_relation_weight", 0.0)),
        relation_samples=int(getattr(config, "distill_relation_samples", 64)),
        attn_weight=float(getattr(config, "distill_attn_weight", 0.0)),
        attn_tau=float(getattr(config, "distill_attn_tau", 0.5)),
        adaptive_branch=bool(getattr(config, "distill_adaptive_branch", False)),
        plan_look_weight=float(getattr(config, "distill_plan_look_weight", 0.5)),
        plan_look_tau=float(getattr(config, "distill_plan_look_tau", 0.5)),
        selector_mode=bool(getattr(config, "distill_selector", False)),
        selector_registers=int(getattr(config, "distill_selector_registers", 16)),
        selector_div_sigma=float(getattr(config, "distill_selector_div_sigma", 4.0)),
        selector_tok_warmup_steps=int(
            getattr(config, "distill_selector_tok_warmup_steps", 53195)
        ),
        selector_channels=int(getattr(config, "embed_dims", 256)),
        selector_commands=int(getattr(config, "num_navi_cmd", 4)),
        selector_anchor_sigma=float(getattr(config, "distill_selector_anchor_sigma", 6.0)),
        selector_anchor_sigma_end=float(
            getattr(config, "distill_selector_anchor_sigma_end", 2.0)
        ),
        selector_plan_tau=float(getattr(config, "distill_selector_plan_tau", 0.3)),
        selector_struct_mix=float(getattr(config, "distill_selector_struct_mix", 0.5)),
    )
    module.validate_manifests(config)
    return module
