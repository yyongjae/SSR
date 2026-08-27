"""PARA-SSR agent for navsim.

Optimisation follows the WoTE/SeerDrive recipe (report #09 §4): AdamW, cosine
schedule with a 3-epoch linear warm-up, and the image backbone at 0.1x the base
learning rate -- which is both WoTE's ``opt_paramwise_cfg`` and the original
SSR config's ``img_backbone: lr_mult 0.1``.

Gradient clipping (``max_norm=35``) is a *trainer* setting, not an optimiser
one: pass ``trainer.params.gradient_clip_val=35.0``.  ``scripts/training/``
does this.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, _LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from .para_ssr_features import ParaSSRFeatureBuilder
from .para_ssr_loss import ParaSSRLoss
from .para_ssr_model import ParaSSRModel
from .para_ssr_targets import (
    DET_NAME_TO_INDEX,
    MAP_CLASS_NAMES,
    ParaSSRTargetBuilder,
)


class WarmupCosLR(_LRScheduler):
    """Linear warm-up then cosine decay, stepped per epoch (WoTE's schedule)."""

    def __init__(
        self,
        optimizer: Optimizer,
        min_lr: float,
        lr: float,
        warmup_epochs: int,
        epochs: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.min_lr = min_lr
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / max(self.epochs - self.warmup_epochs, 1)
                )
            )
        # per-group multipliers (the backbone's 0.1x) live in ``lr_scale``
        return [lr * group.get("lr_scale", 1.0) for group in self.optimizer.param_groups]


class ParaSSRLoggingCallback(pl.Callback):
    """Publishes the per-term loss breakdown and the shared-BEV diagnostics.

    ``AgentLightningModule`` only logs whatever ``compute_loss`` returns, and it
    sums that, so the breakdown has to reach the logger by another route.
    ``gshare/*`` in particular is the number to watch: it says which task is
    actually steering the shared BEV feature, which the loss curves do not.
    """

    def _log(self, pl_module: pl.LightningModule, prefix: str) -> None:
        logs = getattr(pl_module.agent, "latest_logs", None)
        if not logs:
            return
        for key, value in logs.items():
            if key == "loss":  # already logged by AgentLightningModule
                continue
            pl_module.log(
                f"{prefix}/{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log(pl_module, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._log(pl_module, "val")


class ParaSSRAgent(AbstractAgent):
    def __init__(
        self,
        config,
        trajectory_sampling: TrajectorySampling,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ):
        super().__init__()
        self._validate_config(config, trajectory_sampling)
        self._config = config
        # ``run_training.py`` and ``AgentLightningModule`` both reach for
        # ``agent.config`` directly, so it has to exist under that name.
        self.config = config
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path
        self._lr = lr

        self.para_ssr_model = ParaSSRModel(config)
        self._loss = ParaSSRLoss(config)
        self.latest_logs: Dict[str, torch.Tensor] = {}

        if resume_from_checkpoint and checkpoint_path:
            self.initialize()

    @staticmethod
    def _validate_config(config, trajectory_sampling: TrajectorySampling) -> None:
        """Reject overrides that silently mix frames or target conventions."""
        frame_indices = tuple(config.frame_indices)
        if not frame_indices or frame_indices[-1] != 3:
            raise ValueError(
                "frame_indices must end in NAVSIM's current history frame (3), "
                f"got {frame_indices}"
            )
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= 3
            for index in frame_indices
        ):
            raise ValueError(
                "frame_indices must contain integer NAVSIM history indices in "
                f"[0, 3], got {frame_indices}"
            )
        if tuple(sorted(set(frame_indices))) != frame_indices:
            raise ValueError(
                f"frame_indices must be unique and chronological, got {frame_indices}"
            )
        if config.ego_motion_dims < 14:
            raise ValueError(
                f"ego_motion_dims must be at least 14, got {config.ego_motion_dims}"
            )
        if config.num_navi_cmd != config.ego_fut_mode:
            raise ValueError(
                "num_navi_cmd and ego_fut_mode must match for command-branch "
                f"selection, got {config.num_navi_cmd} and {config.ego_fut_mode}"
            )
        if config.num_navi_cmd != 4:
            raise ValueError(
                "NAVSIM driving_command is fixed at 4 classes; num_navi_cmd "
                f"and ego_fut_mode must both be 4, got {config.num_navi_cmd}"
            )
        if config.traj_dims != 3:
            raise ValueError(
                "NAVSIM Trajectory targets are (x, y, heading), so traj_dims "
                f"must be 3, got {config.traj_dims}"
            )
        if config.num_feature_levels != 1 or len(config.backbone_out_indices) != 1:
            raise ValueError(
                "this port has a single-level FPN/deformable-attention path; "
                "num_feature_levels and backbone_out_indices length must both "
                f"be 1, got {config.num_feature_levels} and "
                f"{tuple(config.backbone_out_indices)}"
            )
        if config.num_det_classes != len(DET_NAME_TO_INDEX):
            raise ValueError(
                "num_det_classes must match the NAVSIM target mapping "
                f"({len(DET_NAME_TO_INDEX)}), got {config.num_det_classes}"
            )
        if config.det_code_size != 10 or len(config.det_code_weights) != 10:
            raise ValueError(
                "the detector uses the fixed 10D SECOND regression code; "
                "det_code_size and det_code_weights length must both be 10, "
                f"got {config.det_code_size} and {len(config.det_code_weights)}"
            )
        if config.map_num_classes != len(MAP_CLASS_NAMES):
            raise ValueError(
                "map_num_classes must match divider/crosswalk/boundary targets "
                f"({len(MAP_CLASS_NAMES)}), got {config.map_num_classes}"
            )
        if config.map_num_orders < 1:
            raise ValueError(
                f"map_num_orders must be positive, got {config.map_num_orders}"
            )
        allowed_cameras = {
            "cam_f0",
            "cam_l0",
            "cam_l1",
            "cam_l2",
            "cam_r0",
            "cam_r1",
            "cam_r2",
            "cam_b0",
        }
        camera_names = tuple(config.camera_names)
        if (
            not camera_names
            or len(set(camera_names)) != len(camera_names)
            or not set(camera_names) <= allowed_cameras
        ):
            raise ValueError(
                "camera_names must be a non-empty unique subset of NAVSIM "
                f"cameras, got {camera_names}"
            )
        if config.max_agents > config.num_query:
            raise ValueError(
                "max_agents cannot exceed detection queries, got "
                f"{config.max_agents} > {config.num_query}"
            )
        if not 1 <= config.map_dir_interval < config.map_num_pts_per_vec:
            raise ValueError(
                "map_dir_interval must satisfy 1 <= interval < points/vector, "
                f"got {config.map_dir_interval} and {config.map_num_pts_per_vec}"
            )
        if trajectory_sampling.num_poses != config.fut_ts:
            raise ValueError(
                "trajectory_sampling and fut_ts disagree: "
                f"{trajectory_sampling.num_poses} != {config.fut_ts}"
            )
        if not math.isclose(
            float(trajectory_sampling.interval_length), 0.5, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "NAVSIM frames/targets use a 0.5 second interval, got "
                f"{trajectory_sampling.interval_length}"
            )

    # ------------------------------------------------------------------ #
    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """Load an exact PARA-SSR agent checkpoint.

        Lightning prefixes agent parameters with ``agent.``.  Strip that
        prefix only at the start of each key, then require an exact match.
        Silently accepting missing heads here would make PDM evaluation run
        with randomly initialised parameters, which is worse than failing
        before scoring.  Loading through CPU also keeps DDP checkpoints
        portable across CUDA device layouts.
        """
        if not self._checkpoint_path:
            raise ValueError("checkpoint_path must be set before initialize()")

        checkpoint: Any = torch.load(self._checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
            raise RuntimeError(
                "checkpoint must be a Lightning checkpoint containing 'state_dict'"
            )
        raw_state = checkpoint["state_dict"]
        if not isinstance(raw_state, Mapping):
            raise RuntimeError("checkpoint 'state_dict' must be a mapping")

        prefix = "agent."
        has_lightning_prefix = any(key.startswith(prefix) for key in raw_state)
        if has_lightning_prefix:
            state_dict: Dict[str, Any] = {
                key[len(prefix) :]: value
                for key, value in raw_state.items()
                if key.startswith(prefix)
            }
        else:
            # Also accept a deliberately exported agent-only state dict.
            state_dict = dict(raw_state)
        if not state_dict:
            raise RuntimeError("checkpoint contains no PARA-SSR agent state")

        # Fresh training is required after the parity fixes.  Strict loading is
        # intentional so a pre-fix or unrelated checkpoint cannot be scored.
        self.load_state_dict(state_dict, strict=True)

    def get_sensor_config(self) -> SensorConfig:
        """Load only the cameras and frames this config actually consumes."""
        frames = list(self._config.frame_indices)
        wanted = set(self._config.camera_names)
        return SensorConfig(
            cam_f0=frames if "cam_f0" in wanted else False,
            cam_l0=frames if "cam_l0" in wanted else False,
            cam_l1=frames if "cam_l1" in wanted else False,
            cam_l2=frames if "cam_l2" in wanted else False,
            cam_r0=frames if "cam_r0" in wanted else False,
            cam_r1=frames if "cam_r1" in wanted else False,
            cam_r2=frames if "cam_r2" in wanted else False,
            cam_b0=frames if "cam_b0" in wanted else False,
            lidar_pc=False,
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ParaSSRFeatureBuilder(self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [ParaSSRTargetBuilder(self._config, self._trajectory_sampling)]

    # ------------------------------------------------------------------ #
    def forward(
        self, features: Dict[str, torch.Tensor], targets: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        # The controller is checkpointed with the loss module.  Re-applying its
        # scales here also makes zero-share ablations effective on the very first
        # forward, before the first scheduled measurement.
        self._loss.apply_aux_scales(self.para_ssr_model)
        return self.para_ssr_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns the total loss as a scalar.

        Deliberately NOT a dict: ``AgentLightningModule._step`` sums every value
        it is handed, and the per-term breakdown includes the total, so
        returning it would double-count. The breakdown is published on
        ``latest_logs`` and picked up by :class:`ParaSSRLoggingCallback`.
        """
        loss, logs = self._loss(self.para_ssr_model, features, targets, predictions)
        self.latest_logs = logs
        return loss

    def get_training_callbacks(self) -> List["pl.Callback"]:
        return [ParaSSRLoggingCallback()]

    # ------------------------------------------------------------------ #
    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        cfg = self._config
        backbone_params, other_params = [], []
        for name, param in self.para_ssr_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("image_encoder"):
                backbone_params.append(param)
            else:
                other_params.append(param)

        groups = [
            {"params": other_params, "lr": self._lr, "lr_scale": 1.0},
            {
                "params": backbone_params,
                "lr": self._lr * cfg.backbone_lr_mult,
                "lr_scale": cfg.backbone_lr_mult,
            },
        ]
        optimizer_cls = getattr(torch.optim, cfg.optimizer_type)
        optimizer = optimizer_cls(groups, lr=self._lr, weight_decay=cfg.weight_decay)

        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=cfg.min_lr,
            epochs=cfg.max_epochs,
            warmup_epochs=cfg.warmup_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
