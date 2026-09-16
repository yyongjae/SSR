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

import inspect
import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
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
from .modules.lidar_encoder import (
    LIDAR_BACKBONE_STRIDES,
    LIDAR_ENCODERS,
    pillar_downsample_ratio,
    sparse_grid,
)
from .para_ssr_model import ParaSSRModel
from .para_ssr_targets import (
    DET_NAME_TO_INDEX,
    MAP_CLASS_NAMES,
    ParaSSRTargetBuilder,
)
from .readout.distill import KD_DISTANCES, KD_MODES
from .readout.teacher_targets import ResMapTeacherTargetBuilder


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
        # torch>=2.4 dropped the LRScheduler `verbose` argument.
        if "verbose" in inspect.signature(_LRScheduler.__init__).parameters:
            super().__init__(optimizer, last_epoch, verbose)
        else:
            super().__init__(optimizer, last_epoch)

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


logger = logging.getLogger(__name__)


class ParaSSRLoggingCallback(pl.Callback):
    """Publishes the per-term loss breakdown, BEV diagnostics and wall-clock time.

    ``AgentLightningModule`` only logs whatever ``compute_loss`` returns, and it
    sums that, so the breakdown has to reach the logger by another route.
    ``gshare/*`` in particular is the number to watch: it says which task is
    actually steering the shared BEV feature, which the loss curves do not.

    Wall-clock time goes out as ``time/*`` scalars (elapsed, ETA, per-epoch and
    per-validation hours) and, at the end of the session, as
    ``train_time.json`` in the trainer's root directory.  A resumed run appends
    a new session to that file, so ``total_hours`` is the whole training time
    across resumes; a crash or interrupt still writes the session with
    ``status: "interrupted"``.
    """

    TIME_FILE = "train_time.json"

    def __init__(self) -> None:
        super().__init__()
        self._train_start: Optional[float] = None
        self._epoch_start: Optional[float] = None
        self._val_start: Optional[float] = None
        self._epoch_seconds: List[float] = []
        self._val_seconds: List[float] = []
        self._start_epoch = 0
        self._start_step = 0
        self._started_at = ""

    # ---- per-term losses ------------------------------------------------- #
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
        self._log_progress(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._log(pl_module, "val")

    # ---- wall-clock time ------------------------------------------------- #
    @staticmethod
    def _log_time(pl_module, name: str, hours: float, on_step: bool) -> None:
        pl_module.log(
            f"time/{name}", float(hours), on_step=on_step, on_epoch=not on_step,
            prog_bar=False, rank_zero_only=True,
        )

    def _elapsed(self) -> float:
        return 0.0 if self._train_start is None else time.time() - self._train_start

    def _eta_seconds(self, trainer) -> Optional[float]:
        """Session-rate extrapolation to ``estimated_stepping_batches``."""
        done = trainer.global_step - self._start_step
        total = getattr(trainer, "estimated_stepping_batches", None)
        if done <= 0 or not total or not math.isfinite(float(total)):
            return None
        return self._elapsed() / done * max(float(total) - trainer.global_step, 0.0)

    def _log_progress(self, trainer, pl_module) -> None:
        if self._train_start is None:
            return
        self._log_time(pl_module, "elapsed_hours", self._elapsed() / 3600.0, on_step=True)
        eta = self._eta_seconds(trainer)
        if eta is not None:
            self._log_time(pl_module, "eta_hours", eta / 3600.0, on_step=True)

    def on_train_start(self, trainer, pl_module) -> None:
        self._train_start = time.time()
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._start_epoch = int(trainer.current_epoch)
        self._start_step = int(trainer.global_step)
        self._epoch_seconds = []
        self._val_seconds = []
        logger.info(
            "training wall clock started %s at epoch %d, global step %d",
            self._started_at, self._start_epoch, self._start_step,
        )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._epoch_start is None:
            return
        seconds = time.time() - self._epoch_start
        self._epoch_seconds.append(seconds)
        self._log_time(pl_module, "epoch_hours", seconds / 3600.0, on_step=False)
        self._log_time(pl_module, "elapsed_hours_at_epoch_end", self._elapsed() / 3600.0, on_step=False)
        eta = self._eta_seconds(trainer)
        logger.info(
            "epoch %d took %.1f min; elapsed %.2f h%s",
            trainer.current_epoch, seconds / 60.0, self._elapsed() / 3600.0,
            "" if eta is None else f", eta {eta / 3600.0:.2f} h",
        )

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._val_start = None if getattr(trainer, "sanity_checking", False) else time.time()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self._val_start is None:
            return
        seconds = time.time() - self._val_start
        self._val_seconds.append(seconds)
        self._val_start = None
        self._log_time(pl_module, "val_hours", seconds / 3600.0, on_step=False)

    def on_train_end(self, trainer, pl_module) -> None:
        self._finish(trainer, "completed")

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self._finish(trainer, f"interrupted: {type(exception).__name__}")

    def _finish(self, trainer, status: str) -> None:
        if self._train_start is None:
            return
        seconds = self._elapsed()
        session = {
            "status": status,
            "started": self._started_at,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(seconds, 1),
            "hours": round(seconds / 3600.0, 4),
            "start_epoch": self._start_epoch,
            "end_epoch": int(trainer.current_epoch),
            "start_step": self._start_step,
            "end_step": int(trainer.global_step),
            "epochs_completed": len(self._epoch_seconds),
            "epoch_seconds": [round(s, 1) for s in self._epoch_seconds],
            "validation_seconds": [round(s, 1) for s in self._val_seconds],
            "world_size": int(getattr(trainer, "world_size", 1)),
        }
        logger.info(
            "training %s after %.2f h (%d epochs this session, mean epoch %.1f min, %d validation runs)",
            status, seconds / 3600.0, len(self._epoch_seconds),
            (sum(self._epoch_seconds) / len(self._epoch_seconds) / 60.0) if self._epoch_seconds else 0.0,
            len(self._val_seconds),
        )
        self._train_start = None
        if not getattr(trainer, "is_global_zero", True):
            return
        # The TensorBoard logger's save_dir is the experiment's output_dir;
        # default_root_dir is the same directory under Hydra's chdir, and the
        # fallback otherwise.
        root = getattr(getattr(trainer, "logger", None), "save_dir", None) or getattr(trainer, "default_root_dir", None)
        if not root:
            return
        path = Path(root) / self.TIME_FILE
        record: Dict[str, Any] = {"sessions": []}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, dict) and isinstance(loaded.get("sessions"), list):
                    record = loaded
            except (OSError, ValueError):
                logger.warning("could not read %s; starting a new training-time record", path)
        record["sessions"].append(session)
        record["total_hours"] = round(sum(float(s.get("hours", 0.0)) for s in record["sessions"]), 4)
        record["total_epochs_completed"] = int(sum(int(s.get("epochs_completed", 0)) for s in record["sessions"]))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2))
            logger.info("training time recorded in %s (total %.2f h over %d session(s))",
                        path, record["total_hours"], len(record["sessions"]))
        except OSError as exc:
            logger.warning("could not write %s: %s", path, exc)


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

        # Strict checkpoint restoration supplies every model tensor, so
        # inference must not download pretrained backbone weights.
        model_config = config
        if checkpoint_path:
            model_config = replace(config, backbone_pretrained=False)
        self.para_ssr_model = ParaSSRModel(model_config)
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
                "map_num_classes must match MAP_CLASS_NAMES "
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
        if tuple(config.map_pc_range) != tuple(config.pc_range):
            raise ValueError(
                "shared BEV/detection/map must use one physical ROI; got "
                f"pc_range={tuple(config.pc_range)} and "
                f"map_pc_range={tuple(config.map_pc_range)}"
            )
        if config.pc_range[1] < 0.0:
            raise ValueError(
                "the front-camera-only ROI cannot include rear supervision; "
                f"got y_forward_min={config.pc_range[1]}"
            )
        if not 0.0 < float(config.det_fov_half_angle_deg) <= 90.0:
            raise ValueError(
                "det_fov_half_angle_deg must be in (0, 90], got "
                f"{config.det_fov_half_angle_deg}"
            )
        if getattr(config, "use_lidar", False):
            z_range = tuple(config.lidar_z_range)
            if (
                len(z_range) != 2
                or not all(math.isfinite(float(v)) for v in z_range)
                or not float(z_range[0]) < float(z_range[1])
            ):
                raise ValueError(
                    f"lidar_z_range must be (low, high) metres with low < high, got {z_range}"
                )
            max_points = config.lidar_max_points
            if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
                raise ValueError(
                    f"lidar_max_points must be a positive integer, got {max_points!r}"
                )
            encoder = str(config.lidar_encoder)
            if encoder not in LIDAR_ENCODERS:
                raise ValueError(
                    f"lidar_encoder must be one of {LIDAR_ENCODERS}, got {encoder!r}"
                )
            if encoder == "sparse":
                # Raises with the offending axis when the voxel grid is not the
                # fixed-stride backbone's 8x of the BEV grid.  Pure arithmetic:
                # spconv itself is only required when the model is built.
                sparse_grid(
                    config.pc_range, config.bev_h, config.bev_w,
                    config.lidar_z_range, config.lidar_voxel_size,
                )
            else:
                # Raises with the offending axis when the pillar grid cannot be
                # reduced onto the BEV grid by one power-of-two stride.
                pillar_downsample_ratio(
                    config.pc_range, config.bev_h, config.bev_w, config.lidar_pillar_size
                )
                stages = len(LIDAR_BACKBONE_STRIDES)
                if (
                    len(config.lidar_backbone_channels) != stages
                    or len(config.lidar_backbone_layers) != stages
                ):
                    raise ValueError(
                        "lidar_backbone_channels and lidar_backbone_layers need one entry "
                        f"per backbone stage ({stages}), got "
                        f"{tuple(config.lidar_backbone_channels)} and "
                        f"{tuple(config.lidar_backbone_layers)}"
                    )
            if int(config.lidar_attn_points) < 1:
                raise ValueError(
                    f"lidar_attn_points must be positive, got {config.lidar_attn_points}"
                )
        if not 1 <= config.map_dir_interval < config.map_num_pts_per_vec:
            raise ValueError(
                "map_dir_interval must satisfy 1 <= interval < points/vector, "
                f"got {config.map_dir_interval} and {config.map_num_pts_per_vec}"
            )
        kd_mode = getattr(config, "kd_mode", "none")
        label_source = getattr(config, "map_label_source", "gt")
        if kd_mode not in KD_MODES:
            raise ValueError(f"kd_mode must be one of {KD_MODES}, got {kd_mode!r}")
        if label_source not in ("gt", "teacher"):
            raise ValueError(f"map_label_source must be gt|teacher, got {label_source!r}")
        if (kd_mode != "none" or label_source == "teacher") and not config.kd_teacher_cache:
            raise ValueError("kd_mode / map_label_source=teacher need kd_teacher_cache")
        if label_source == "teacher" and not config.use_map_head:
            raise ValueError("map_label_source=teacher needs use_map_head=true")
        if kd_mode != "none":
            if config.kd_distance not in KD_DISTANCES:
                raise ValueError(f"kd_distance must be one of {KD_DISTANCES}, got {config.kd_distance!r}")
            if kd_mode == "readout" and not config.kd_readout_ckpt:
                raise ValueError("kd_mode=readout needs kd_readout_ckpt")
            if min(int(config.kd_warmup_iters), int(config.kd_ramp_iters)) < 0:
                raise ValueError("kd_warmup_iters and kd_ramp_iters must be non-negative")
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
        # The one exception is the distillation adapter (readout/distill.py):
        # it is training-only, so a KD checkpoint must evaluate without it and
        # a KD fine-tune must start from a checkpoint that never had it.
        kd_prefix = "_loss.distiller."
        own = self.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith(kd_prefix) or k in own
        }
        result = self.load_state_dict(state_dict, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith(kd_prefix)]
        if missing or result.unexpected_keys:
            # same wording as torch's strict load, which callers match on
            raise RuntimeError(
                "Error(s) in loading PARA-SSR checkpoint: "
                f"Missing key(s): {missing[:10]}; "
                f"Unexpected key(s): {list(result.unexpected_keys)[:10]}"
            )

    def get_sensor_config(self) -> SensorConfig:
        """Load only the cameras, LiDAR frames and history this config consumes.

        LiDAR is loaded at every queue frame because the history BEV is built
        from the same LiDAR-seeded encoder as the current one.
        """
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
            lidar_pc=frames if self._config.use_lidar else False,
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ParaSSRFeatureBuilder(self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        builders: List[AbstractTargetBuilder] = [
            ParaSSRTargetBuilder(self._config, self._trajectory_sampling)
        ]
        cfg = self._config
        if getattr(cfg, "kd_mode", "none") != "none" or getattr(cfg, "map_label_source", "gt") == "teacher":
            builders.append(ResMapTeacherTargetBuilder(cfg))
        return builders

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

        # training-only modules owned by the loss (the distillation adapter)
        other_params += [p for p in self._loss.parameters() if p.requires_grad]

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
