"""Stage 1: train one BEV adapter from a frozen teacher's cached BEV.

The stage-1 contract from ``docs/PLANNING_DISTILLATION.md`` is::

    cached teacher BEV -> residual MLP adapter -> unchanged SSR planning
                       -> trajectory loss

No images are read and no BEV is built, so this is cheap.  What it produces is
the adapter checkpoint that stage 2 freezes: an adapter that is known to carry
*planning-relevant* information, because a planner trained through it can drive.

The planning decoder is the same :class:`ParaSSRPlannerHead` the full model
uses, entered through ``forward_from_bev``.  Its BEV transformer is never
constructed here -- there is nothing to encode.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from ..cache_key import cache_key
from ..modules.planner_head import ParaSSRPlannerHead
from ..para_ssr_agent import WarmupCosLR
from ..para_ssr_loss import compute_plan_loss
from ..para_ssr_targets import ParaSSRTargetBuilder
from .adapter import PlanningBEVAdapter
from .teacher_store import TeacherFeatureStore


class TeacherAdapterFeatureBuilder(AbstractFeatureBuilder):
    """Only the navigation command: stage 1 reads no sensors at all."""

    def __init__(self, config) -> None:
        super().__init__()
        self._config = config

    def get_unique_name(self) -> str:
        return cache_key(
            "para_ssr_teacher_adapter_feature",
            (("num_navi_cmd", self._config.num_navi_cmd),),
        )

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        import numpy as np

        command = np.asarray(
            agent_input.ego_statuses[-1].driving_command, dtype=np.float32
        )
        return {"command": torch.tensor(command)}


class TeacherAdapterPlanner(nn.Module):
    """Trainable adapter feeding the unchanged PARA-SSR planning decoder."""

    def __init__(self, config) -> None:
        super().__init__()
        cfg = config
        self.adapter = PlanningBEVAdapter(
            **dict(cfg.distill_branches[cfg.teacher_adapter_branch].get(
                "adapter", {"channels": cfg.embed_dims}))
        )
        # transformer=None: ``forward_from_bev`` never touches it, and building
        # one would add parameters that receive no gradient (a DDP error).
        self.planner = ParaSSRPlannerHead(
            transformer=None,
            bev_h=cfg.bev_h,
            bev_w=cfg.bev_w,
            embed_dims=cfg.embed_dims,
            pc_range=cfg.pc_range,
            num_scenes=cfg.num_scenes,
            num_reg_fcs=cfg.num_reg_fcs,
            fut_ts=cfg.fut_ts,
            ego_fut_mode=cfg.ego_fut_mode,
            num_navi_cmd=cfg.num_navi_cmd,
            traj_dims=cfg.traj_dims,
            latent_num_layers=cfg.latent_num_layers,
            way_num_layers=cfg.way_num_layers,
            num_heads=cfg.num_heads,
            feedforward_channels=cfg.ffn_channels,
            use_metric_planner=False,
            num_plan_candidates=cfg.num_plan_candidates,
            plan_anchor_path="",
        )
        # ``forward_from_bev`` takes the BEV from the adapter, so the learned BEV
        # queries that ``forward`` would hand to the transformer are unused here.
        # A parameter that never receives a gradient makes DDP abort, and it is
        # invisible in single-process testing. requires_grad is not part of
        # state_dict, so a stage-1 checkpoint still loads into the full model.
        self.planner.bev_embedding.weight.requires_grad_(False)

    def forward(self, teacher_bev: torch.Tensor, cmd: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``teacher_bev`` is ``[B, C, H, W]`` straight from the cache."""
        adapted = self.adapter(teacher_bev)          # [B, H*W, C]
        outs = self.planner.forward_from_bev(adapted, cmd)
        outs["trajectory"] = self.planner.select_trajectory(
            outs["ego_fut_preds"], cmd
        )
        return outs


class ParaSSRTeacherAdapterAgent(AbstractAgent):
    """Stage-1 agent: cached teacher BEV in, trajectory out."""

    def __init__(
        self,
        config,
        trajectory_sampling: TrajectorySampling,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if not config.distill_feature_root:
            raise ValueError(
                "stage 1 needs distill_feature_root pointing at the teacher cache"
            )
        branch = config.teacher_adapter_branch
        if branch not in config.distill_branches:
            raise KeyError(
                f"teacher_adapter_branch {branch!r} is not in distill_branches "
                f"{sorted(config.distill_branches)}"
            )
        # The token is privileged, so it arrives in targets; forward() therefore
        # has to see targets.
        if not config.input_target or not config.needs_scene_token:
            raise ValueError(
                "stage 1 requires input_target=True and use_distill=True so the "
                "frame token reaches forward()"
            )

        self._config = config
        self.config = config
        self._trajectory_sampling = trajectory_sampling
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._branch = branch

        spec = dict(config.distill_branches[branch])
        self._store = TeacherFeatureStore(
            config.distill_feature_root, spec.get("cache_name", branch)
        )
        self._store.validate_manifest(config)

        self.model = TeacherAdapterPlanner(config)
        self.latest_logs: Dict[str, torch.Tensor] = {}

        if resume_from_checkpoint and checkpoint_path:
            self.initialize()

    # ------------------------------------------------------------------ #
    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if not self._checkpoint_path:
            raise ValueError("checkpoint_path must be set before initialize()")
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu",
                                weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        prefix = "agent."
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()
                     if k.startswith(prefix)}
        self.load_state_dict(state, strict=True)

    def get_sensor_config(self) -> SensorConfig:
        """Stage 1 reads the cache, never the sensors."""
        return SensorConfig(
            cam_f0=False, cam_l0=False, cam_l1=False, cam_l2=False,
            cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=False,
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TeacherAdapterFeatureBuilder(self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [ParaSSRTargetBuilder(self._config, self._trajectory_sampling)]

    # ------------------------------------------------------------------ #
    def forward(
        self, features: Dict[str, torch.Tensor], targets: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        if targets is None or "scene_token" not in targets:
            raise ValueError(
                "stage 1 needs targets['scene_token']; set input_target=True "
                "and regenerate the target cache"
            )
        command = features["command"]
        teacher_bev = self._store.load_batch(
            targets["scene_token"], command.device, command.dtype
        )
        return self.model(teacher_bev, command)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss, metrics = compute_plan_loss(
            predictions["ego_fut_preds"],
            targets["trajectory_offsets"],
            targets["trajectory_mask"],
            targets["command"],
            self._config.heading_weight,
        )
        self.latest_logs = dict(metrics)
        self.latest_logs["loss_plan"] = loss.detach()
        return loss

    def get_training_callbacks(self) -> List[pl.Callback]:
        from ..para_ssr_agent import ParaSSRLoggingCallback

        return [ParaSSRLoggingCallback()]

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        cfg = self._config
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_cls = getattr(torch.optim, cfg.optimizer_type)
        optimizer = optimizer_cls(
            [{"params": params, "lr": self._lr, "lr_scale": 1.0}],
            lr=self._lr, weight_decay=cfg.weight_decay,
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer, lr=self._lr, min_lr=cfg.min_lr,
            epochs=cfg.max_epochs, warmup_epochs=cfg.warmup_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
