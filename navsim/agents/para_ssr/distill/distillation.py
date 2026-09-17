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

from .adapter import PlanningBEVAdapter, bev_tokens_to_map
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


def compute_corridor_mask(
    trajectories: torch.Tensor,
    pc_range: Tuple[float, ...],
    bev_h: int,
    bev_w: int,
    sigma_base: float = 2.5,
    sigma_growth: float = 0.1,
    base_weight: float = 0.1,
) -> torch.Tensor:
    """Compute Gaussian driving corridor mask along future trajectory waypoints.

    BEV regions close to the planned future driving path receive higher distillation
    gradient, while off-corridor background areas receive base_weight.
    """
    B, T, _ = trajectories.shape
    device = trajectories.device
    dtype = trajectories.dtype
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]

    ys = torch.linspace(
        y_min + (y_max - y_min) / (2 * bev_h),
        y_max - (y_max - y_min) / (2 * bev_h),
        bev_h,
        device=device,
        dtype=dtype,
    )
    xs = torch.linspace(
        x_min + (x_max - x_min) / (2 * bev_w),
        x_max - (x_max - x_min) / (2 * bev_w),
        bev_w,
        device=device,
        dtype=dtype,
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_pts = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)

    pts = trajectories[:, :, :2].unsqueeze(2).unsqueeze(2)
    d2 = ((grid_pts - pts) ** 2).sum(dim=-1)

    ts = torch.arange(T, device=device, dtype=dtype)
    sigmas = (sigma_base + sigma_growth * ts).view(1, T, 1, 1)
    w = torch.exp(-0.5 * d2 / (sigmas ** 2))

    corridor_max, _ = w.max(dim=1, keepdim=True)
    return base_weight + (1.0 - base_weight) * corridor_max


class PlanningDistillation(nn.Module):
    """Frozen teacher adapters and the feature losses attached to PARA-SSR."""

    def __init__(
        self,
        feature_root: str,
        branches: Mapping[str, Mapping],
        adapter_checkpoint: Mapping[str, str],
        student_bev_size: Tuple[int, int] = (100, 100),
        cache_size: Tuple[int, int] = (100, 100),
        strict_checkpoints: bool = True,
        use_corridor_mask: bool = True,
        pc_range: Tuple[float, ...] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        corridor_sigma_base: float = 2.5,
        corridor_sigma_growth: float = 0.1,
        corridor_base_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if not branches:
            raise ValueError("distillation needs at least one teacher branch")
        self.student_bev_size = tuple(int(v) for v in student_bev_size)
        self.cache_size = tuple(int(v) for v in cache_size)
        self.use_corridor_mask = bool(use_corridor_mask)
        self.pc_range = tuple(float(v) for v in pc_range)
        self.corridor_sigma_base = float(corridor_sigma_base)
        self.corridor_sigma_growth = float(corridor_sigma_growth)
        self.corridor_base_weight = float(corridor_base_weight)

        self.adapters = nn.ModuleDict()
        self.stores: Dict[str, TeacherFeatureStore] = {}
        self.loss_weights: Dict[str, float] = {}
        self.loaded_adapter_checkpoints: Dict[str, str] = {}

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

    def forward(
        self,
        student_bev: torch.Tensor,
        tokens: Sequence[str],
        trajectories: Optional[torch.Tensor] = None,
    ):
        """``student_bev`` is ``[B, bev_h*bev_w, C]``; returns (losses, metrics)."""
        student_map = bev_tokens_to_map(student_bev, self.student_bev_size)
        traj = None
        if self.use_corridor_mask and trajectories is not None and trajectories.numel() > 0:
            if trajectories.dim() == 3 and trajectories.size(0) == student_map.size(0):
                traj = trajectories.to(device=student_map.device, dtype=student_map.dtype)

        corridor_by_hw: Dict[Tuple[int, int], torch.Tensor] = {}

        def corridor_tokens(height: int, width: int) -> Optional[torch.Tensor]:
            if traj is None:
                return None
            key = (height, width)
            if key not in corridor_by_hw:
                mask = compute_corridor_mask(
                    trajectories=traj,
                    pc_range=self.pc_range,
                    bev_h=height,
                    bev_w=width,
                    sigma_base=self.corridor_sigma_base,
                    sigma_growth=self.corridor_sigma_growth,
                    base_weight=self.corridor_base_weight,
                )
                corridor_by_hw[key] = mask.flatten(2).transpose(1, 2)
            return corridor_by_hw[key]

        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        for name, adapter in self.adapters.items():
            teacher_map = self.stores[name].load_batch(
                tokens, student_map.device, student_map.dtype)
            if teacher_map.size(0) != student_map.size(0):
                raise ValueError(
                    f"{name}: {teacher_map.size(0)} cached samples for "
                    f"{student_map.size(0)} student samples"
                )
            teacher_hw = (int(teacher_map.size(-2)), int(teacher_map.size(-1)))
            if tuple(student_map.shape[-2:]) != teacher_hw:
                aligned_student = F.interpolate(
                    student_map, size=teacher_hw, mode="bilinear",
                    align_corners=False)
            else:
                aligned_student = student_map
            mask_tokens = corridor_tokens(*teacher_hw)
            with torch.no_grad():
                teacher = adapter(teacher_map)
            student = adapter(aligned_student)

            diff = (student - teacher).square()
            if mask_tokens is not None:
                weighted_diff = diff * mask_tokens
                mse = weighted_diff.sum() / (mask_tokens.sum() * student.size(-1))
            else:
                mse = diff.mean()
            losses[f"loss_distill_{name}"] = mse * self.loss_weights[name]

            with torch.no_grad():
                cos = F.cosine_similarity(student, teacher, dim=-1)
                if mask_tokens is not None:
                    metrics[f"distill_cos/{name}"] = (cos * mask_tokens.squeeze(-1)).sum() / mask_tokens.sum()
                else:
                    metrics[f"distill_cos/{name}"] = cos.mean()
                metrics[f"distill_rmse/{name}"] = mse.detach().sqrt()
        return losses, metrics


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
    )
    module.validate_manifests(config)
    return module
