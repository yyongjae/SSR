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
import os
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
from .plan_map import DrivableAreaTargetBuilder
from .plan_score_targets import AnchorScoreTargetBuilder
from .modules.corridor_head import (CorridorPlanTargetBuilder, CorridorTargetBuilder, corridor_loss, corridor_plan_loss,
                                    corridor_verdict_loss)


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
        # ParaSSRConfig.log_sync_dist = False: no per-micro-batch all-reduce (default True = unchanged)
        sync_dist = bool(getattr(getattr(pl_module.agent, "config", None), "log_sync_dist", True))
        for key, value in logs.items():
            if key == "loss":  # already logged by AgentLightningModule
                continue
            pl_module.log(
                f"{prefix}/{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=sync_dist,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log(pl_module, "train")
        self._log_progress(trainer, pl_module, batch_idx=batch_idx)

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

    def _eta_seconds(self, trainer, batch_idx: Optional[int] = None) -> Optional[float]:
        """Session-rate extrapolation to total remaining training batches (or ``estimated_stepping_batches``)."""
        if batch_idx is not None:
            num_b = getattr(trainer, "num_training_batches", 0)
            max_ep = getattr(trainer, "max_epochs", 0)
            if num_b and math.isfinite(float(num_b)) and max_ep and max_ep > 0:
                done_b = (int(trainer.current_epoch) - self._start_epoch) * float(num_b) + float(batch_idx + 1)
                rem_b = max((int(max_ep) - 1 - int(trainer.current_epoch)) * float(num_b) + (float(num_b) - float(batch_idx + 1)), 0.0)
                if done_b > 0:
                    return self._elapsed() / done_b * rem_b
        done = trainer.global_step - self._start_step
        total = getattr(trainer, "estimated_stepping_batches", None)
        if done <= 0 or not total or not math.isfinite(float(total)):
            return None
        return self._elapsed() / done * max(float(total) - trainer.global_step, 0.0)

    def _log_progress(self, trainer, pl_module, batch_idx: Optional[int] = None) -> None:
        if self._train_start is None:
            return
        self._log_time(pl_module, "elapsed_hours", self._elapsed() / 3600.0, on_step=True)
        eta = self._eta_seconds(trainer, batch_idx=batch_idx)
        if eta is not None:
            eta_h = eta / 3600.0
            self._log_time(pl_module, "eta_hours", eta_h, on_step=True)
            pl_module.log(
                "eta_h", float(eta_h), on_step=True, on_epoch=False,
                prog_bar=True, logger=False, rank_zero_only=True,
            )

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
    # CK Phase 2 helper (plain object, set in __init__ when config.ck_e2e is on).  Class-level default so agents built
    # without __init__ (tests use ParaSSRAgent.__new__) still read "CK off".
    _ck_e2e = None
    # CK2 e2e helper (ck/online2.py; config.ck_e2e2), same class-level "off" default.  The optional BEV-KD arm
    # (ck/bev_kd_arm.py) is a registered submodule `ck_bev_kd`: deliberately NO class attribute of that name (it would
    # shadow nn.Module.__getattr__ and hide the registered module); read it with getattr(self, "ck_bev_kd", None).
    _ck_e2e2 = None

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
        # Stage E (refiner/e2e.py): built only when switched on, so 'off' (E0)
        # constructs, loads and computes exactly what it did before.
        self._stage_e = None
        if getattr(config, "refiner_mode", "off") != "off":
            from .refiner.e2e import StageE, build_student

            self.ref_student = build_student(
                int(config.ref_seed), float(config.ref_bev_grad_scale), int(config.embed_dims)
            )
            self._stage_e = StageE(config)
        # CK Phase 2 (ck/online.py): built only when ck_e2e is enabled, so the default
        # (empty dict) constructs, loads and computes exactly what v2 did.  The teachers,
        # LabelStore and recorder are created lazily at the first training step and are
        # never registered modules (not in the state_dict / DDP / optimiser).
        self._ck_e2e = None
        ckcfg = self._ck_e2e_config(config)
        if ckcfg is not None and ckcfg.enabled:
            from .ck.online import CKE2E, build_student_ck

            self.ck_student = build_student_ck(ckcfg, apply_prior=not checkpoint_path)
            self._ck_e2e = CKE2E(ckcfg, max_epochs=int(config.max_epochs))
        # CK2 e2e (ck/online2.py): built only when ck_e2e2.enabled (default {} = off: v2 bit-identical).  Every
        # constructor below forks the RNG (CKNet trunk / score head, BEVKDAdapter), so the v2 init and the data order
        # are the same with and without CK2 / the BEV-KD arm.  Teachers / LabelStore2 / recorder are lazy plain objects.
        self._ck_e2e2 = None
        ck2 = self._ck_e2e2_config(config)
        if ck2 is not None and ck2.enabled:
            from .ck.online2 import CKE2E2, build_student_ck2

            self.ck_student = build_student_ck2(ck2, apply_prior=not checkpoint_path)
            self._ck_e2e2 = CKE2E2(ck2, max_epochs=int(config.max_epochs))
            self.ck_bev_kd = None
            if ck2.bev_kd["enabled"]:
                from .ck.bev_kd_arm import build_bev_kd_arm

                self.ck_bev_kd = build_bev_kd_arm(ck2)       # nn.Module: state_dict / DDP / own optimiser group
        # BEV planning distillation merged from aux_distill (distill/, docs/bev_selector_v15.md): built only when
        # config.use_distill (default off: nothing imported, built or loaded).  A registered submodule `_distill`
        # (state_dict prefix '_distill.'); deliberately NO class attribute of that name (it would hide the registered
        # module), so read it with getattr(self, "_distill", None).
        if getattr(config, "use_distill", False):
            source = str(getattr(config, "distill_teacher_source", "store"))
            if source not in ("store", "targets"):
                raise ValueError(f"distill_teacher_source must be 'store' or 'targets', got {source!r}")
            self._distill = self._build_distill(config)

        if resume_from_checkpoint and checkpoint_path:
            self.initialize()

    @staticmethod
    def _ck_e2e_config(config):
        """CKE2EConfig of config.ck_e2e, or None when it is empty (the module is then never imported)."""
        raw = getattr(config, "ck_e2e", None)
        if not raw:
            return None
        from .ck.online import CKE2EConfig

        return CKE2EConfig.from_any(raw)

    @staticmethod
    def _ck_e2e2_config(config):
        """CKE2E2Config of config.ck_e2e2, or None when it is empty (ck/online2.py is then never imported)."""
        raw = getattr(config, "ck_e2e2", None)
        if not raw:
            return None
        from .ck.online2 import CKE2E2Config

        return CKE2E2Config.from_any(raw)

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
        # ---- Stage E student refiner (exp-refine) ----
        refiner_mode = getattr(config, "refiner_mode", "off")
        if refiner_mode not in ("off", "E1", "E2"):
            raise ValueError(f"refiner_mode must be off, E1 or E2, got {refiner_mode!r}")
        if refiner_mode != "off":
            if (config.bev_h, config.bev_w) != (50, 100) or tuple(config.pc_range) != (
                -32.0, 0.0, -2.0, 32.0, 32.0, 2.0
            ):
                raise ValueError("the student refiner reads the 50 x 100 S grid (0.64 m, front ROI)")
            if config.ref_eval_traj not in ("final", "tau0"):
                raise ValueError(f"ref_eval_traj must be final or tau0, got {config.ref_eval_traj!r}")
            if len(tuple(config.kd_ramp)) != 2:
                raise ValueError(f"kd_ramp must be (start_epoch, end_epoch), got {config.kd_ramp}")
            if getattr(config, "kd_balance", "fixed") not in ("fixed", "ema"):
                raise ValueError(f"kd_balance must be fixed or ema, got {config.kd_balance!r}")
            if getattr(config, "kd_draft_source", "tau0") not in ("tau0", "human_mix"):
                raise ValueError(f"kd_draft_source must be tau0 or human_mix, got {config.kd_draft_source!r}")
            if getattr(config, "kd_balance", "fixed") == "ema" and not (
                0.0 < float(config.kd_ema_m) < 1.0 and float(config.kd_ema_floor) > 0.0 and float(config.kd_ratio) >= 0.0
            ):
                raise ValueError("kd_balance ema needs 0 < kd_ema_m < 1, kd_ema_floor > 0, kd_ratio >= 0")
            hou = getattr(config, "ref_human_only_until", None)
            if hou is not None and not float(hou) >= 0.0:
                raise ValueError(f"ref_human_only_until must be null or >= 0, got {hou!r}")
            for k in ("kd_ratio_ramp_epochs", "kd_weight_max"):
                v = getattr(config, k, None)
                if v is not None and not (float(v) > 0.0 and getattr(config, "kd_balance", "fixed") == "ema"):
                    raise ValueError(f"{k} must be null or > 0 with kd_balance ema, got {v!r}")
        # ---- CK Phase 2 e2e (ck/online.py; off unless ck_e2e.enabled) ----
        ckcfg = ParaSSRAgent._ck_e2e_config(config)
        if ckcfg is not None:
            ckcfg.validate(config)
        # ---- CK2 e2e (ck/online2.py; off unless ck_e2e2.enabled; rejects ck_e2e + ck_e2e2 both on) ----
        ck2cfg = ParaSSRAgent._ck_e2e2_config(config)
        if ck2cfg is not None:
            ck2cfg.validate(config)
        # ---- v2: readout KD / anchor planner / plan_map (km/para-ssr-v2) ----
        # Independent of Stage E: the two 'kd_*' families are separate systems
        # (v2: kd_mode/kd_teacher_cache/kd_weight/kd_*_iters = ReSMap BEV readout
        # distillation inside ParaSSRLoss; Stage E: kd_teacher_runs/kd_lambda/kd_ramp/
        # kd_space/kd_balance = correction KD of the student refiner).
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
            if kd_mode in ("readout", "attn_feature", "sens_feature") and not config.kd_readout_ckpt:
                raise ValueError("kd_mode=readout needs kd_readout_ckpt")
            if min(int(config.kd_warmup_iters), int(config.kd_ramp_iters)) < 0:
                raise ValueError("kd_warmup_iters and kd_ramp_iters must be non-negative")
        if getattr(config, "plan_anchor", False):
            if not config.plan_anchor_file:
                raise ValueError("plan_anchor=true needs plan_anchor_file")
            if config.traj_dims != 3:
                raise ValueError("plan_anchor=true needs traj_dims=3 (the anchors are x, y, heading)")
        plan_map_weight = float(getattr(config, "plan_map_weight", 0.0))
        if plan_map_weight < 0:
            raise ValueError(f"plan_map_weight must be >= 0, got {plan_map_weight}")
        if plan_map_weight > 0:
            x0, x1, y0, y1 = config.plan_map_extent
            if not (x1 > x0 and y1 > y0 and float(config.plan_map_res) > 0 and float(config.plan_map_clip) > 0):
                raise ValueError(
                    f"plan_map_extent/res/clip invalid: {config.plan_map_extent}, "
                    f"{config.plan_map_res}, {config.plan_map_clip}")
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

    # ------------------------------------------------------------------ #
    # aux_distill BEV planning distillation (fusion_v1 / corridor-cf-distill port)
    # ------------------------------------------------------------------ #
    _DISTILL_PREFIX = "_distill."

    @staticmethod
    def _build_distill(config):
        """PlanningDistillation of the trajectory-anchor path (v6..v16) on this kyungmin-v2 / corridor-cf core.

        Differences to aux_distill, none of which changes the distillation math:
        - the probe is drawn under torch.random.fork_rng(seed=distill_probe_seed), so switching the distillation on
          leaves the v2 / corridor-cf init and the global RNG stream (data order) of the distillation-off run unchanged;
        - for v6/v7/v8, everything but student_proj is frozen HERE, before Lightning's DDP wrap; for v9..v16, both
          student_proj and the PlanningGroundedProbe (+ anchor/cmd/status query encoders) remain trainable;
        - the spatial bias is built from config.plan_anchor_file (the planner's own anchors) and is not saved in
          checkpoints (it is a function of that file).
        """
        if not any(bool(getattr(config, k, False))
                   for k in ("distill_selector_v6", "distill_selector_v7", "distill_selector_v8",
                             "distill_selector_v9", "distill_selector_v10", "distill_selector_v11",
                             "distill_selector_v12", "distill_selector_v13", "distill_selector_v14",
                             "distill_selector_v15", "distill_selector_v16")):
            raise NotImplementedError(
                "use_distill on this core supports only the trajectory-anchor path "
                "(distill_selector_v6/v7/v8/v9/v10/v11/v12/v13/v14/v15/v16); the stage-1 adapter / corridor / "
                "register-selector paths need aux_distill's own v2 port")
        anchor_file = getattr(config, "plan_anchor_file", None)
        if not getattr(config, "plan_anchor", False) or not anchor_file or not os.path.isfile(str(anchor_file)):
            raise ValueError("use_distill (trajectory-anchor path) needs the anchor planner: plan_anchor true and an "
                             f"existing plan_anchor_file (got plan_anchor={getattr(config, 'plan_anchor', None)}, "
                             f"plan_anchor_file={anchor_file})")
        from .distill.distillation import build_planning_distillation

        # CPU generator only: torch.manual_seed would also re-seed the (queued) CUDA generators
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(getattr(config, "distill_probe_seed", 0)))
            module = build_planning_distillation(config)
        is_pgp = bool(
            getattr(config, "distill_selector_v9", False)
            or getattr(config, "distill_selector_v10", False)
            or getattr(config, "distill_selector_v11", False)
            or getattr(config, "distill_selector_v12", False)
            or getattr(config, "distill_selector_v13", False)
            or getattr(config, "distill_selector_v14", False)
            or getattr(config, "distill_selector_v15", False)
            or getattr(config, "distill_selector_v16", False)
        )
        trainable_prefix = "anchor_distill." if is_pgp else "anchor_distill.student_proj."
        for name, param in module.named_parameters():
            param.requires_grad_(name.startswith(trainable_prefix))
        return module

    def distill_parameters(self) -> List[torch.nn.Parameter]:
        distill = getattr(self, "_distill", None)
        if distill is None:
            return []
        return [p for p in distill.parameters() if p.requires_grad]

    def _distill_loss(self, distill, features, targets, predictions, logs):
        """aux_distill's compute_loss block (v6..v16 anchor path), added after the v2 balancer and the
        corridor / scorer terms.  bev_pos is recomputed from the planner's LearnedPositionalEncoding exactly as
        plan_from_bev does (same parameters, not detached: the distillation gradient reaches the shared positional
        embedding as in aux_distill, where the planner returned the same tensor)."""
        cfg = self._config
        if "scene_token" not in targets:
            raise ValueError("planning distillation needs targets['scene_token'] (ParaSSRTargetBuilder emits it when "
                             "use_distill is on; regenerate a target cache built with use_distill off)")
        status = features.get("ego_status")
        status_feature = features.get("status_feature")
        command = features.get("command")
        n_cmd = int(cfg.num_navi_cmd)
        if status is None and status_feature is not None and status_feature.shape[-1] > n_cmd:
            status = status_feature[:, n_cmd:]
            if command is None:
                command = status_feature[:, :n_cmd]
        bev = predictions["bev_embed"]
        head = self.para_ssr_model.pts_bbox_head
        mask = bev.new_zeros((bev.shape[0], head.bev_h, head.bev_w))
        bev_pos = head.positional_encoding(mask).to(bev.dtype).flatten(2).transpose(1, 2)
        inputs = dict(student_bev=bev, tokens=list(targets["scene_token"]), trajectories=targets.get("trajectory"),
                      command=command, ego_status=status, bev_pos=bev_pos, sim_reward=targets.get("sim_reward"),
                      sim_reward_valid=targets.get("sim_reward_valid"),
                      gt_map_pts=targets.get("gt_map_pts"), gt_map_labels=targets.get("gt_map_labels"),
                      gt_map_valid=targets.get("gt_map_valid"),
                      plan_topk_trajectory=predictions.get("plan_topk_trajectory"))
        out = {}
        anchors = predictions.get("trajectory_anchors")
        ad = getattr(distill, "anchor_distill", None)
        if ad is not None and anchors is not None and ad.sync_spatial_bias(anchors):
            logger.warning("distillation spatial bias rebuilt from the planner's anchors (they differ from the ones "
                           "it was built from, e.g. after a checkpoint restore)")
        frac, dead = 1.0, False
        if str(getattr(cfg, "distill_teacher_source", "store")) == "targets":
            # S-grid f16 targets from DistillTeacherTargetBuilder or CK2E2ETargetBuilder:
            # kd_bev_0 = DET / BEVFusion, kd_bev_1 = MAP / ReSMap
            from .distill.adapter import bev_map_to_tokens

            ok = targets["kd_ok_0"].bool() & targets["kd_ok_1"].bool()
            inputs["teacher_tokens"] = (bev_map_to_tokens(targets["kd_bev_0"]).float(),
                                        bev_map_to_tokens(targets["kd_bev_1"]).float())
            out["distill/teacher_ok_frac"] = ok.float().mean().detach()
            if not bool(ok.any()):
                # no teacher at all on this rank: run the term on the student's own (detached) BEV with weight 0, so
                # every rank logs the same keys (sync_dist logging) and every trainable distillation parameter gets
                # a (zero) gradient (plain DDP)
                dead = True
                inputs["teacher_tokens"] = (bev.detach(), bev.detach())
            elif not bool(ok.all()):
                # left-out tokens contribute 0 and every kept token keeps the per-sample weight 1 / B it has in a
                # full micro-batch (the term is a batch mean), so ranks with different teacher coverage weigh the
                # same under DDP's gradient average
                idx = ok.nonzero(as_tuple=True)[0]
                frac = float(idx.numel()) / float(ok.numel())
                inputs = {k: (None if v is None else
                              [v[i] for i in idx.tolist()] if k == "tokens" else
                              tuple(t[idx] for t in v) if k == "teacher_tokens" else v[idx])
                          for k, v in inputs.items()}
        losses, metrics = distill(
            inputs.pop("student_bev"),
            inputs.pop("tokens"),
            trajectory_anchors=anchors,
            **inputs,
        )
        weight = float(cfg.distill_loss_weight) * (0.0 if dead else frac)
        keep = 0.0 if dead else 1.0
        total = None
        for name, value in losses.items():
            total = weight * value if total is None else total + weight * value
            out[name] = value.detach() * keep           # per kept token (0 when no teacher)
        out.update({k: v.detach() * keep for k, v in metrics.items()})
        if total is None:
            total = bev.new_zeros(())
        else:
            if (
                ad is not None
                and getattr(ad, "ratio_ctrl", None) is not None
                and "gnorm/plan" in logs
                and self.training
                and bev.requires_grad
                and torch.is_grad_enabled()
            ):
                from .modules.grad_balance import all_reduce_mean
                has_probe_ctrl = getattr(ad, "probe_ratio_ctrl", None) is not None
                g_local = {}
                for tk in ("det", "map"):
                    u_loss = ad.last_unit_losses.get(tk)
                    u_in = ad.last_unit_inputs.get(tk)
                    if not dead and isinstance(u_loss, torch.Tensor) and u_loss.requires_grad and u_in is not None:
                        gg = torch.autograd.grad(weight * u_loss, u_in, retain_graph=True, allow_unused=True)[0]
                        g_local[tk] = 0.0 if gg is None else float(gg.detach().float().norm())
                    else:
                        g_local[tk] = 0.0
                    if has_probe_ctrl:
                        pu_loss = ad.last_probe_unit_losses.get(tk)
                        pu_in = ad.last_probe_unit_inputs.get(tk)
                        if not dead and isinstance(pu_loss, torch.Tensor) and pu_loss.requires_grad and pu_in is not None:
                            pgg = torch.autograd.grad(weight * pu_loss, pu_in, retain_graph=True, allow_unused=True)[0]
                            g_local[f"probe_{tk}"] = 0.0 if pgg is None else float(pgg.detach().float().norm())
                        else:
                            g_local[f"probe_{tk}"] = 0.0
                g_red = all_reduce_mean(g_local, bev.device)
                g_plan = float(logs["gnorm/plan"])
                ad.ratio_ctrl.update({"det": g_red["det"], "map": g_red["map"]}, g_plan)
                out["distill/g_unit_det"] = bev.new_tensor(float(g_red["det"]))
                out["distill/g_unit_map"] = bev.new_tensor(float(g_red["map"]))
                if has_probe_ctrl:
                    ad.probe_ratio_ctrl.update({"det": g_red["probe_det"], "map": g_red["probe_map"]}, g_plan)
                    out["distill/g_unit_probe_det"] = bev.new_tensor(float(g_red["probe_det"]))
                    out["distill/g_unit_probe_map"] = bev.new_tensor(float(g_red["probe_map"]))
            self._log_distill_bev_grad(bev, total, out)
        return total, out

    def _log_distill_bev_grad(self, bev: torch.Tensor, distill_loss: torch.Tensor, logs: Dict) -> None:
        """||d L_distill / d bev_embed|| on the gnorm/plan cadence (aux_distill telemetry; same micro-batches on every
        rank).  Logged as gnorm/distill_v8: gnorm/distill is the readout-KD slot of ParaSSRLoss."""
        interval = int(getattr(self._config, "grad_norm_log_interval", 0) or 0)
        if interval <= 0 or not self.training or not distill_loss.requires_grad or not bev.requires_grad:
            return
        if (int(getattr(self._loss, "iteration", 1)) - 1) % interval != 0:
            return
        grad = torch.autograd.grad(distill_loss, bev, retain_graph=True, allow_unused=True)[0]
        norm = 0.0 if grad is None else float(grad.detach().norm())
        from .modules.grad_balance import all_reduce_mean

        reduced = all_reduce_mean({"distill_v8": norm}, bev.device)
        logs["gnorm/distill_v8"] = bev.new_tensor(float(reduced["distill_v8"]))

    def filter_datasets_to_teacher_cache(self, train_data, val_data) -> None:
        """Keep only training tokens present in every teacher store (run_training hook); no-op when distillation is
        off.  The distillation is a training-only term here, so validation keeps all its tokens."""
        distill = getattr(self, "_distill", None)
        if distill is None or str(getattr(self._config, "distill_teacher_source", "store")) == "targets":
            return       # the 'targets' source masks a missing teacher per token (kd_ok) instead
        from .distill.teacher_store import restrict_dataset_to_stores

        restrict_dataset_to_stores(train_data, list(distill.stores.values()), name="train")

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
        # Same for the aux_distill BEV distillation module (training-only): a distillation checkpoint evaluates with
        # use_distill off, and a distillation run may start from a checkpoint that never had it.
        train_only = (kd_prefix, self._DISTILL_PREFIX)
        own = self.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith(train_only) or k in own
        }
        # v1 -> v2 init: the anchor planner replaces the single-trajectory regressor.
        v1_regressor = "para_ssr_model.pts_bbox_head.ego_fut_decoder."
        if not any(k.startswith(v1_regressor) for k in own):
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(v1_regressor)}
        result = self.load_state_dict(state_dict, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith(train_only)]
        # The v2 anchor planner may be initialised from a v1 checkpoint: its new
        # modules start fresh, but only when the checkpoint has NONE of them (a
        # partial match is a different vocabulary/architecture, not an upgrade).
        anchor_prefix = "para_ssr_model.pts_bbox_head.anchor_planner."
        if not any(k.startswith(anchor_prefix) for k in state_dict):
            fresh = [k for k in missing if k.startswith(anchor_prefix)]
            if fresh:
                logger.info("v1 checkpoint: %d anchor-planner tensors start fresh", len(fresh))
            missing = [k for k in missing if not k.startswith(anchor_prefix)]
        # km 10/25: the cross-fit planners are training-only and start fresh from a checkpoint that has none of them
        # (initialising them from the main planner would hand them its fit of every training scene)
        cf_prefix = "para_ssr_model.pts_bbox_head.cf_planners."
        if not any(k.startswith(cf_prefix) for k in state_dict):
            fresh = [k for k in missing if k.startswith(cf_prefix)]
            if fresh:
                logger.info("cross-fit planners: %d tensors start fresh", len(fresh))
            missing = [k for k in missing if not k.startswith(cf_prefix)]
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
        if float(getattr(cfg, "plan_map_weight", 0.0)) > 0:
            builders.append(DrivableAreaTargetBuilder(cfg))
        # plan_score_file is training-only: evaluation leaves it null and gets no builder
        if getattr(cfg, "plan_anchor", False) and getattr(cfg, "plan_score_file", None):
            builders.append(AnchorScoreTargetBuilder(cfg))
        if str(getattr(cfg, "corridor_format", "dt01")) == "c64" and float(getattr(cfg, "corridor_weight", 0.0)) > 0:
            from .modules.corridor_head import Corridor64TargetBuilder
            builders.append(Corridor64TargetBuilder(cfg, plan=float(getattr(cfg, "corridor_plan_weight", 0.0)) > 0))
        elif float(getattr(cfg, "corridor_plan_weight", 0.0)) > 0:
            builders.append(CorridorPlanTargetBuilder(cfg))
        elif float(getattr(cfg, "corridor_weight", 0.0)) > 0:
            builders.append(CorridorTargetBuilder(cfg))
        if float(getattr(cfg, "cand_scorer_weight", 0.0)) > 0 and getattr(cfg, "cand_official", False):
            from .modules.exact_score import ExactFullTargetBuilder
            builders.append(ExactFullTargetBuilder(cfg))
        elif float(getattr(cfg, "cand_scorer_weight", 0.0)) > 0 and getattr(cfg, "exact_dir", None):
            from .modules.exact_verdict import ExactTargetBuilder
            builders.append(ExactTargetBuilder(cfg))
        if self._ck_e2e is not None:
            from .ck.e2e_data import CKE2ETargetBuilder

            builders.append(CKE2ETargetBuilder(cfg))
        if self._ck_e2e2 is not None:
            from .ck.e2e_data2 import CK2E2ETargetBuilder

            builders.append(CK2E2ETargetBuilder(cfg))
        elif getattr(cfg, "use_distill", False) and str(getattr(cfg, "distill_teacher_source", "store")) == "targets":
            from .distill.teacher_store import DistillTeacherTargetBuilder

            builders.append(DistillTeacherTargetBuilder(cfg))
        return builders

    # ------------------------------------------------------------------ #
    def forward(
        self, features: Dict[str, torch.Tensor], targets: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        # The controller is checkpointed with the loss module.  Re-applying its
        # scales here also makes zero-share ablations effective on the very first
        # forward, before the first scheduled measurement.
        self._loss.apply_aux_scales(self.para_ssr_model)
        cf_fold = targets.get("cf_fold") if (targets is not None and self.training
                                              and int(getattr(self._config, "cand_cf_folds", 0))) else None
        predictions = (self.para_ssr_model(features, cf_fold=cf_fold) if cf_fold is not None
                       else self.para_ssr_model(features))
        if self._stage_e is not None and not self.training:
            # inference: trajectory = tau_final (or tau0 with ref_eval_traj=tau0)
            predictions = self._stage_e.infer(self.ref_student, features, predictions)
        if self._ck_e2e is not None and not self.training and self._ck_e2e.cfg.infer_outputs:
            # ck_* outputs only; predictions['trajectory'] stays the v2 one
            predictions = dict(predictions)
            predictions.update(self._ck_e2e.infer(self.ck_student, features, predictions))
        if self._ck_e2e2 is not None and not self.training and self._ck_e2e2.cfg.infer_outputs:
            # ck2_* outputs over the 96-candidate pool only; predictions['trajectory'] stays the v2 one
            predictions = dict(predictions)
            predictions.update(self._ck_e2e2.infer(self.ck_student, features, predictions))
        return predictions

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
        if self.para_ssr_model.corridor_head is not None and "corridor_valid" in targets:
            # km 10/15: corridor loss added AFTER the grad balancer (v2's plan/det/map shares unchanged)
            c_loss, c_logs = corridor_loss(predictions, targets, float(self._config.corridor_occ_weight),
                                           float(getattr(self._config, "corridor_route_weight", 0.0)),
                                           occ_pos_weight=float(getattr(self._config, "corridor_occ_pos_weight", 4.0)),
                                           occ_time_weight=float(getattr(self._config, "corridor_occ_time_weight", 0.0)),
                                           sdf_boundary_weight=float(getattr(self._config, "corridor_sdf_boundary_weight", 4.0)))
            logs["loss_v2"] = logs["loss"]
            loss = loss + float(self._config.corridor_weight) * c_loss
            logs.update(c_logs)
            if float(self._config.corridor_plan_weight) > 0 and "corridor_occ_dist" in targets:
                # km 10/15 (2): corridor constraint on every tracked candidate (GT corridor, training only)
                cfg = self._config
                p_loss, p_logs = corridor_plan_loss(
                    predictions, targets, features["status_feature"][:, cfg.num_navi_cmd:cfg.num_navi_cmd + 4],
                    float(cfg.corridor_plan_static_margin), float(cfg.corridor_plan_occ_margin),
                    float(cfg.corridor_plan_cap), float(getattr(cfg, "corridor_plan_ttc_weight", 0.0)),
                    prog_weight=float(getattr(cfg, "corridor_plan_prog_weight", 0.0)),
                    prog_cap=float(getattr(cfg, "corridor_plan_prog_cap", 1.2)),
                    comfort_weight=float(getattr(cfg, "corridor_plan_comfort_weight", 0.0)),
                    ec_smooth_weight=float(getattr(cfg, "corridor_plan_ec_smooth_weight", 0.0)))
                loss = loss + float(cfg.corridor_plan_weight) * p_loss
                if "cf_offset" in predictions and "cf_fold" in targets:
                    # km 10/25 (cross-fitting): each sample trains ONLY its own fold's planner: the imitation L1 of
                    # the anchor nearest the GT (anchor_plan_losses' WTA term) + the same corridor plan loss
                    cfo = predictions["cf_offset"]
                    rows = torch.arange(cfo.shape[0], device=cfo.device)
                    own = cfo[rows, targets["cf_fold"].long().clamp(0, 1).to(cfo.device)]          # [B, K, T, 3]
                    anc = predictions["trajectory_anchors"].to(own.dtype)
                    gt = targets["trajectory"].reshape(own.shape[0], -1).to(own.dtype)
                    win = (anc.reshape(1, anc.shape[0], -1) - gt[:, None]).norm(dim=-1).argmin(1)
                    cf_l1 = torch.nn.functional.l1_loss(own.reshape(own.shape[0], own.shape[1], -1)[rows, win],
                                      gt - anc.reshape(anc.shape[0], -1)[win])
                    cf_p, cf_logs = corridor_plan_loss(
                        {**predictions, "trajectory_offset": own}, targets,
                        features["status_feature"][:, cfg.num_navi_cmd:cfg.num_navi_cmd + 4],
                        float(cfg.corridor_plan_static_margin), float(cfg.corridor_plan_occ_margin),
                        float(cfg.corridor_plan_cap), float(getattr(cfg, "corridor_plan_ttc_weight", 0.0)),
                        prog_weight=float(getattr(cfg, "corridor_plan_prog_weight", 0.0)),
                        prog_cap=float(getattr(cfg, "corridor_plan_prog_cap", 1.2)),
                        comfort_weight=float(getattr(cfg, "corridor_plan_comfort_weight", 0.0)),
                        ec_smooth_weight=float(getattr(cfg, "corridor_plan_ec_smooth_weight", 0.0)), track_slot=1)
                    loss = loss + float(cfg.cand_cf_weight) * (cf_l1 + float(cfg.corridor_plan_weight) * cf_p)
                    logs["cf/l1"], logs["cf/plan"] = cf_l1.detach(), cf_p.detach()
                    logs.update({"cf/" + kk.split("/")[-1]: v for kk, v in cf_logs.items()})
                logs.update(p_logs)
                if float(getattr(cfg, "corridor_verdict_weight", 0.0)) > 0:
                    # km 10/16 (v3.1): the predicted corridor's verdicts of model-free footprints vs the GT verdicts
                    v_loss, v_logs = corridor_verdict_loss(
                        predictions, targets, features["status_feature"][:, cfg.num_navi_cmd:cfg.num_navi_cmd + 4],
                        int(getattr(cfg, "corridor_verdict_anchors", 64)),
                        ttc=float(getattr(cfg, "corridor_plan_ttc_weight", 0.0)) > 0)
                    loss = loss + float(cfg.corridor_verdict_weight) * v_loss
                    logs.update(v_logs)
            logs["loss"] = loss.detach()
        if self.para_ssr_model.cand_scorer is not None and "ex1_valid" in targets and "cand_states" in predictions:
            # km 10/19 (C v3): every head vs the OFFICIAL per-candidate PDMS / EPDMS terms of the same candidates,
            # tracked from the token's exact initial state (exact_score.official_scores)
            from .modules import exact_score as ES
            from .modules.candidate_scorer import cand_scorer_loss
            from .modules.torch_tracker import track
            cand = predictions["cand_traj"]                  # km 10/25: may hold anchor + u * offset rows (cand_scorer_mix_p)
            b, k = cand.shape[:2]
            with torch.no_grad():
                init = ES.ego_init(targets["ex_ego"].double())
                st9 = track(cand.reshape(b * k, 8, 3).double(), init.repeat_interleave(k, 0), full=True,
                            angular=True).reshape(b, k, 41, 9).float()
                ES.track_human(targets, init)
                ex = ES.official_scores(st9, targets, targets)
            s_loss, s_logs = cand_scorer_loss(predictions, ex,
                                              prog_weight=float(getattr(self._config, "cand_scorer_prog_weight", 1.0)))
            logs.setdefault("loss_v2", logs["loss"])
            loss = loss + float(self._config.cand_scorer_weight) * s_loss
            logs.update(s_logs)
            with torch.no_grad():
                logs["cand/ex1_trunc"] = targets["ex1_trunc"].float().mean()
                logs["cand/ex2_trunc"] = targets["ex2_trunc"].float().mean()
                logs["cand/ex2_valid"] = targets["ex2_valid"].float().mean()
                logs["cand/gt_best_pdms1"] = torch.nan_to_num(ex["pdms1"], nan=0.0).amax(1).mean()
                logs["cand/gt_best_epdms2"] = torch.nan_to_num(ex["epdms2"], nan=0.0).amax(1).mean()
                if "cf_offset" in predictions:               # km 10/25: official NC / DAC failure rate of the cf candidates
                    for nm in ("nc1", "dac1"):
                        f, v = ex[nm] < 1, torch.isfinite(ex[nm])
                        logs[f"cand/cf_fail_{nm}"] = (f & v).float().sum() / v.float().sum().clamp(min=1)
                if "cand_mix_u" in predictions:              # km 10/25: official NC / DAC failure rate, mixed vs fitted
                    mx = predictions["cand_mix_u"] < 1
                    for nm in ("nc1", "dac1"):
                        f, v = ex[nm] < 1, torch.isfinite(ex[nm])
                        logs[f"cand/mix_fail_{nm}"] = (f & v & mx).float().sum() / (v & mx).float().sum().clamp(min=1)
                        logs[f"cand/fit_fail_{nm}"] = (f & v & ~mx).float().sum() / (v & ~mx).float().sum().clamp(min=1)
            logs["loss"] = loss.detach()
        elif self.para_ssr_model.cand_scorer is not None and "ex_valid" in targets and "cand_states" in predictions:
            # km 10/19 (C): candidate scorer vs the exact GT verdicts of the same tracked candidates (after the balancer)
            from .modules.candidate_scorer import cand_scorer_loss
            from .modules.exact_verdict import exact_verdicts
            ex = exact_verdicts(predictions["cand_states"], targets)
            s_loss, s_logs = cand_scorer_loss(predictions, ex, targets["ex_valid"],
                                              prog_weight=float(getattr(self._config, "cand_scorer_prog_weight", 1.0)))
            logs.setdefault("loss_v2", logs["loss"])
            loss = loss + float(self._config.cand_scorer_weight) * s_loss
            logs.update(s_logs)
            logs["cand/ex_trunc"] = targets["ex_trunc"].float().mean()
            logs["loss"] = loss.detach()
        if self._stage_e is not None and self.training:
            # Added AFTER the grad balancer: its plan/det/map shares are unchanged.
            ref_loss, ref_logs = self._stage_e.loss(
                self.ref_student, features, targets, predictions,
                iteration=self._loss.iteration - 1,
                log_bev_grad="gnorm/plan" in logs,
                e0_loss=loss,
            )
            logs["loss_e0"] = logs["loss"]
            loss = loss + ref_loss
            logs.update(ref_logs)
            logs["loss"] = loss.detach()
        if self._ck_e2e is not None and self.training:
            # CK loss added AFTER the grad balancer (its plan/det/map shares are unchanged)
            ck_loss, ck_logs = self._ck_e2e.loss(self.ck_student, features, targets, predictions, v2_loss=loss)
            logs["loss_v2"] = logs["loss"]
            loss = loss + ck_loss
            logs.update(ck_logs)
            logs["loss"] = loss.detach()
        if self._ck_e2e2 is not None and self.training:
            # CK2 loss (+ the BEV-KD arm term) added AFTER the grad balancer (v2's plan/det/map shares unchanged);
            # v2_logs = the v2 loss logs (the BEV-KD measurement fires on the 'gnorm/plan' micro-batches)
            ck_loss, ck_logs = self._ck_e2e2.loss(self.ck_student, features, targets, predictions, v2_loss=loss,
                                                  v2_logs=logs, bev_kd=getattr(self, "ck_bev_kd", None))
            logs["loss_v2"] = logs["loss"]
            loss = loss + ck_loss
            logs.update(ck_logs)
            logs["loss"] = loss.detach()
        distill = getattr(self, "_distill", None)
        if distill is not None and self.training:
            # aux_distill BEV distillation, added after the balancer and the corridor/scorer/refiner terms
            d_loss, d_logs = self._distill_loss(distill, features, targets, predictions, logs)
            logs["loss_pre_distill"] = logs["loss"]
            loss = loss + d_loss
            logs.update(d_logs)
            logs["loss"] = loss.detach()
        self.latest_logs = logs
        return loss

    def get_training_callbacks(self) -> List["pl.Callback"]:
        callbacks: List[pl.Callback] = [ParaSSRLoggingCallback()]
        if self._stage_e is not None:
            from .refiner.e2e import make_callback

            callbacks.append(make_callback(self))
        if self._ck_e2e is not None:
            from .ck.online import make_ck_callback

            callbacks.append(make_ck_callback(self))
        if self._ck_e2e2 is not None:
            from .ck.online2 import make_ck2_callback

            callbacks.append(make_ck2_callback(self))
        return callbacks

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
        if self._stage_e is not None:
            from .refiner.e2e import refiner_parameters

            groups.append(
                {
                    "params": refiner_parameters(self),
                    "lr": self._lr * cfg.ref_lr_mult,
                    "lr_scale": cfg.ref_lr_mult,
                    "weight_decay": cfg.ref_weight_decay,
                }
            )
        if self._ck_e2e is not None:
            from .ck.online import ck_parameters

            ckc = self._ck_e2e.cfg
            groups.append(
                {
                    "params": ck_parameters(self),
                    "lr": self._lr * ckc.lr_mult,
                    "lr_scale": ckc.lr_mult,
                    "weight_decay": ckc.weight_decay,
                }
            )
        if self._ck_e2e2 is not None:
            from .ck.online2 import ck2_parameters

            ck2c = self._ck_e2e2.cfg
            groups.append(
                {
                    "params": ck2_parameters(self),
                    "lr": self._lr * ck2c.lr_mult,
                    "lr_scale": ck2c.lr_mult,
                    "weight_decay": ck2c.weight_decay,
                }
            )
            if getattr(self, "ck_bev_kd", None) is not None:
                from .ck.bev_kd_arm import bev_kd_parameters

                bk = ck2c.bev_kd
                groups.append(
                    {
                        "params": bev_kd_parameters(self),
                        "lr": self._lr * bk["lr_mult"],
                        "lr_scale": bk["lr_mult"],
                        "weight_decay": bk["weight_decay"],
                    }
                )
        distill = getattr(self, "_distill", None)
        if distill is not None:
            proj_params = [
                p for n, p in distill.named_parameters()
                if p.requires_grad and n.startswith("anchor_distill.student_proj.")
            ]
            probe_params = [
                p for n, p in distill.named_parameters()
                if p.requires_grad and not n.startswith("anchor_distill.student_proj.")
            ]
            if proj_params:
                if bool(
                    getattr(cfg, "distill_selector_v11", False)
                    or getattr(cfg, "distill_selector_v12", False)
                    or getattr(cfg, "distill_selector_v13", False)
                    or getattr(cfg, "distill_selector_v14", False)
                    or getattr(cfg, "distill_selector_v15", False)
                    or getattr(cfg, "distill_selector_v16", False)
                ):
                    proj_mult = float(getattr(cfg, "distill_proj_lr_mult", 2.0))
                    groups.append({
                        "params": proj_params,
                        "lr": self._lr * proj_mult,
                        "lr_scale": proj_mult,
                        "weight_decay": 0.0,
                    })
                else:
                    # aux_distill trains student_proj with the base group (lr, lr_scale 1, weight_decay): same AdamW math in
                    # a group of its own, so the distillation-off group layout is unchanged
                    groups.append({"params": proj_params, "lr": self._lr, "lr_scale": 1.0})
            if probe_params:
                probe_mult = float(getattr(cfg, "distill_probe_lr_mult", 3.0))
                groups.append({
                    "params": probe_params,
                    "lr": self._lr * probe_mult,
                    "lr_scale": probe_mult,
                })
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
