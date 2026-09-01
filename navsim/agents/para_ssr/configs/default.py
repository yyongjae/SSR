"""PARA-SSR (navsim) configuration.

Values fall into three groups:

* **architecture** -- carried over from ``projects/configs/SSR/PARA_SSR_e2e.py``
  unchanged wherever navsim allows it.
* **navsim adaptations** -- horizon, command count, class sets, image geometry.
* **optimisation** -- the WoTE/SeerDrive-derived recipe (see report #09 §4).
"""
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class ParaSSRConfig:
    # ------------------------------------------------------------------ #
    # navsim I/O
    # ------------------------------------------------------------------ #
    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=4, interval_length=0.5
    )

    # navsim gives 4 history frames (0..3); 3 is the current one.
    # ``frame_indices[-1]`` MUST be the current frame.
    # A longer queue costs one full image load + one no-grad BEV pass per extra
    # frame, so this is the first knob to turn if dataloading is the bottleneck.
    frame_indices: Tuple[int, ...] = (2, 3)

    camera_names: Tuple[str, ...] = (
        "cam_f0",
        "cam_l0",
        "cam_l1",
        "cam_l2",
        "cam_r0",
        "cam_r1",
        "cam_r2",
        "cam_b0",
    )

    # navsim images are 1920x1080. 0.4 -> 768x432, then drop 16 rows of sky.
    image_scale: float = 0.4
    crop_top: int = 16

    ego_motion_dims: int = 18

    # Read by the devkit, not by us. Motion GT comes from ``Scene``'s future
    # frames (annotations, always loaded), not from future *sensors*, so the
    # expensive future-image path stays off.
    use_fut_frames: bool = False
    # ``AgentLightningModule`` passes targets into forward() when True.
    input_target: bool = False

    # ------------------------------------------------------------------ #
    # shared BEV encoder
    # ------------------------------------------------------------------ #
    # VAD/SSR ego frame: x lateral (+right), y longitudinal (+forward).
    pc_range: Tuple[float, ...] = (-15.0, -30.0, -2.0, 15.0, 30.0, 2.0)
    bev_h: int = 100
    bev_w: int = 100
    embed_dims: int = 256
    num_heads: int = 8
    ffn_channels: int = 512
    num_feature_levels: int = 1

    # The tag matters: bare timm ``resnet50`` resolves to its A1 recipe, while
    # original SSR uses torchvision://resnet50 (0676ba61 / tv_in1k).
    image_architecture: str = "resnet50.tv_in1k"
    # Training starts from ImageNet weights.  Checkpoint evaluation overrides
    # this to False: the strict checkpoint load immediately replaces every
    # parameter, so asking timm to download/cache a second copy is unnecessary
    # and can make an otherwise self-contained evaluation fail offline.
    backbone_pretrained: bool = True
    backbone_out_indices: Tuple[int, ...] = (4,)   # timm feature index for C5
    frozen_stages: int = 1
    # Original mmdet config: norm_cfg=dict(type="BN", requires_grad=False).
    norm_requires_grad: bool = False
    norm_eval: bool = True

    encoder_num_layers: int = 3
    encoder_num_points_in_pillar: int = 4
    encoder_num_points_sca: int = 8
    encoder_attn_dropout: float = 0.1
    encoder_ffn_dropout: float = 0.1

    use_shift: bool = True
    use_ego_motion: bool = True
    ego_motion_norm: bool = True
    use_cams_embeds: bool = True
    use_grid_mask: bool = True

    # ------------------------------------------------------------------ #
    # planning head
    # ------------------------------------------------------------------ #
    num_scenes: int = 16
    num_reg_fcs: int = 2
    latent_num_layers: int = 3
    way_num_layers: int = 1
    # navsim: 8 poses x 0.5s = 4s, and a 4-way driving_command one-hot
    fut_ts: int = 8
    num_navi_cmd: int = 4
    ego_fut_mode: int = 4
    # navsim scores (x, y, heading); nuScenes SSR regressed (x, y) only
    traj_dims: int = 3
    heading_weight: float = 0.5

    # ------------------------------------------------------------------ #
    # auxiliary head 1: detection + motion
    # ------------------------------------------------------------------ #
    use_det_motion_head: bool = True
    num_query: int = 300
    # nuPlan TrackedObjectType, not nuScenes' 10 classes
    num_det_classes: int = 7
    # Measured over 20 navtrain scenes: in-range agents median 53, max 80,
    # and 7/20 scenes exceed 60. 100 leaves headroom without truncating.
    max_agents: int = 100
    det_num_decoder_layers: int = 3
    det_code_size: int = 10
    det_code_weights: Tuple[float, ...] = (
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2,
    )
    fut_mode: int = 6
    det_use_pe: bool = True
    loss_cls_weight: float = 2.0
    loss_bbox_weight: float = 0.25
    loss_traj_weight: float = 0.2
    loss_traj_cls_weight: float = 0.2

    # ------------------------------------------------------------------ #
    # auxiliary head 2: vectorised map
    # ------------------------------------------------------------------ #
    use_map_head: bool = True
    map_num_vec: int = 100          # predicted polyline queries
    map_max_vec: int = 100          # GT slots
    map_num_pts_per_vec: int = 20
    map_num_orders: int = 20        # equivalent GT orderings, see report #09
    map_num_classes: int = 3        # divider, ped_crossing, boundary
    map_num_decoder_layers: int = 3
    map_dir_interval: int = 1
    map_min_length: float = 1.0     # metres; shorter clipped fragments are dropped
    loss_map_cls_weight: float = 2.0
    loss_map_pts_weight: float = 1.0
    loss_map_dir_weight: float = 0.005

    # ------------------------------------------------------------------ #
    # PARA-Drive wiring: per-task loss weights and the shared-BEV gradient valve
    # ------------------------------------------------------------------ #
    task_loss_weight: Dict[str, float] = field(
        # The port averages over 4 modes x (x, y, heading), whereas SSR averaged
        # over 3 modes x (x, y). 2.0 restores the commanded x/y coefficient.
        default_factory=lambda: dict(plan=2.0, det=1.0, motion=1.0, map=1.0)
    )
    # Closed-loop control of each task's gradient share at the shared BEV.
    # Same target as the nuScenes 60-epoch arm (PARA_SSR_e2e_60ep.py:114), so
    # the two runs stay comparable.  Set to None for the unscaled reference.
    # Measured unscaled on real navsim batches: plan 0.25% / det 73.5% /
    # map 26.2% -- the planner barely steers the shared feature without this.
    grad_balance_target: Dict[str, float] = field(
        default_factory=lambda: dict(plan=0.4, det=0.3, map=0.3)
    )
    # NOTE: these count MICRO-batches (one per compute_loss call), not
    # optimiser steps.  With accumulate_grad_batches=16 one optimiser step is
    # 16 of them, so the nuScenes config's 500 does NOT transfer: it would fire
    # after 31 optimiser steps, deep inside the 3-epoch LR warm-up, and the
    # first measurement is adopted without smoothing.  10600 is one epoch at
    # 2 GPUs x batch 4 over navtrain, by which point LR is at 2/3 of peak.
    grad_balance_interval: int = 200
    grad_balance_momentum: float = 0.9
    grad_balance_clamp: Tuple[float, float] = (1e-5, 1.0)
    grad_balance_warmup_iters: int = 10600
    grad_norm_log_interval: int = 200

    # aux heads are train-only by default (PARA-Drive's runtime advantage)
    test_aux_heads: bool = False

    # ------------------------------------------------------------------ #
    # optimisation -- WoTE/SeerDrive recipe, see report #09 §4
    # ------------------------------------------------------------------ #
    optimizer_type: str = "AdamW"
    weight_decay: float = 1e-4
    max_epochs: int = 30
    min_lr: float = 1e-6
    warmup_epochs: int = 3
    # backbone learns at 0.1x, as in WoTE and in the original SSR config
    backbone_lr_mult: float = 0.1
    grad_clip_norm: float = 35.0

    @property
    def bev_grid_length(self) -> Tuple[float, float]:
        """(metres per cell longitudinal, metres per cell lateral)."""
        return (
            (self.pc_range[4] - self.pc_range[1]) / self.bev_h,
            (self.pc_range[3] - self.pc_range[0]) / self.bev_w,
        )

    @property
    def num_cams(self) -> int:
        return len(self.camera_names)

    @property
    def queue_length(self) -> int:
        return len(self.frame_indices)
