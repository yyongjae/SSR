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
from typing import Dict, Mapping, Sequence, Tuple

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
    ) -> None:
        super().__init__()
        if not branches:
            raise ValueError("distillation needs at least one teacher branch")
        self.student_bev_size = tuple(int(v) for v in student_bev_size)
        self.cache_size = tuple(int(v) for v in cache_size)
        self.adapters = nn.ModuleDict()
        self.stores: Dict[str, TeacherFeatureStore] = {}
        self.loss_weights: Dict[str, float] = {}
        self.loaded_adapter_checkpoints: Dict[str, str] = {}

        missing = [name for name in branches if name not in adapter_checkpoint]
        if strict_checkpoints and missing:
            raise KeyError(
                f"no stage-1 adapter checkpoint for branch(es) {missing}. Train "
                "them first; stage 2 is defined only against frozen adapters."
            )

        states: Dict[str, Mapping[str, torch.Tensor]] = {}
        for name, raw in branches.items():
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

            self.stores[name] = TeacherFeatureStore(
                feature_root, cfg.pop("cache_name", name),
                **{k: cfg.pop(k) for k in ("cache_subdirs", "feature_key",
                                            "flip_w") if k in cfg})
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

    def forward(self, student_bev: torch.Tensor, tokens: Sequence[str]):
        """``student_bev`` is ``[B, bev_h*bev_w, C]``; returns (losses, metrics)."""
        student_map = bev_tokens_to_map(student_bev, self.student_bev_size)
        if tuple(student_map.shape[-2:]) != self.cache_size:
            student_map = F.interpolate(
                student_map, size=self.cache_size, mode="bilinear",
                align_corners=False)

        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        for name, adapter in self.adapters.items():
            teacher_map = self.stores[name].load_batch(
                tokens, student_map.device, student_map.dtype)
            if tuple(teacher_map.shape[-2:]) != self.cache_size:
                raise ValueError(
                    f"{name} cache has {tuple(teacher_map.shape[-2:])}, "
                    f"expected {self.cache_size}"
                )
            if teacher_map.size(0) != student_map.size(0):
                raise ValueError(
                    f"{name}: {teacher_map.size(0)} cached samples for "
                    f"{student_map.size(0)} student samples"
                )
            with torch.no_grad():
                teacher = adapter(teacher_map)
            student = adapter(student_map)
            mse = (student - teacher).square().mean()
            losses[f"loss_distill_{name}"] = mse * self.loss_weights[name]

            with torch.no_grad():
                metrics[f"distill_cos/{name}"] = F.cosine_similarity(
                    student, teacher, dim=-1).mean()
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
    )
    module.validate_manifests(config)
    return module
