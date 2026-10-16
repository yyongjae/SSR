"""PARA-SSR (navsim) configuration.

Values fall into three groups:

* **architecture** -- carried over from ``projects/configs/SSR/PARA_SSR_e2e.py``
  unchanged wherever navsim allows it.
* **navsim adaptations** -- horizon, command count, class sets, image geometry.
* **optimisation** -- the WoTE/SeerDrive-derived recipe (see report #09 §4).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

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

    # Same front-camera set as WoTE. Keep each view separate for BEVFormer's
    # calibrated projection; camera tensors, lidar2img and embeddings all use
    # this order. num_cams is derived from this tuple below.
    camera_names: Tuple[str, ...] = (
        "cam_f0",
        "cam_l0",
        "cam_r0",
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
    # The camera-only front-view model must not receive supervision for an
    # unobserved rear half-plane.  One front-facing ROI is therefore shared by
    # the BEV encoder, detector/motion head and vector-map head:
    # x_right in [-32, 32] m and y_forward in [0, 32] m.
    #
    # The grid is 50 rows over the 32 m of y_forward by 100 columns over the
    # 64 m of x_right: square 0.64 m cells, 5,000 queries.  This is the
    # BEVFusion teacher's grid (bevfusion/configs/navsim/default.yaml: voxel
    # 0.08 m, sparse_shape [400, 800, 41], out_size_factor 8 -> 50 x 100), so
    # the teacher's cached ``bev_feature`` maps onto ``bev_embed`` cell for
    # cell with a lateral flip and no resampling (teacher_cache/*_50x100).  The
    # earlier 100 x 100 variant had 0.32 x 0.64 m cells and needed the
    # longitudinal axis resampled against that teacher.
    pc_range: Tuple[float, ...] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
    bev_h: int = 50     # y_forward: 32 m / 50 = 0.64 m per row
    bev_w: int = 100    # x_right:  64 m / 100 = 0.64 m per column

    # Deliberately identical to pc_range.  A different map range would make the
    # map decoder's normalized reference points address the wrong physical BEV
    # cells, even if both tensors happened to have compatible shapes.
    map_pc_range: Tuple[float, ...] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
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

    # Historical flag name: now controls full SE(2) history feature alignment
    # (translation AND rotation) before temporal attention, once per frame.
    use_shift: bool = True
    # Ego velocity/acceleration condition the planning query directly. Keep
    # geometric history alignment via bev_shift + yaw, without learned BEV status
    # conditioning. The legacy feature/config fields remain cache-compatible.
    use_ego_motion: bool = False
    ego_motion_norm: bool = True
    use_cams_embeds: bool = True
    use_grid_mask: bool = True

    # ------------------------------------------------------------------ #
    # LiDAR branch (SafeDrive wiring, see modules/lidar_encoder.py)
    # ------------------------------------------------------------------ #
    # The merged NAVSIM point cloud at every frame in ``frame_indices`` is
    # clipped to the front ROI, converted to SSR axes and encoded into a
    # ``[embed_dims, bev_h, bev_w]`` BEV.  That BEV replaces the learned BEV
    # query embedding and is the value of a deformable ``lidar_cross_attn``
    # in every encoder layer, gated against the camera cross-attention.
    # History frames need it too, or ``prev_bev`` would be a different kind of
    # feature from the current BEV it is aligned with.
    use_lidar: bool = False
    # Padded rows per frame.  Measured on trainval: ~50k points fall inside the
    # 32 x 64 m front ROI of a ~91k-point frame (max 53k over the sample);
    # longer clouds are thinned deterministically, never dropped at random.
    lidar_max_points: int = 65536
    # Metres, ego frame (rear axle ~0.3-0.5 m above ground).  99% of in-ROI
    # points lie below 5.1 m and none below -3 m; same bounds as the teacher.
    # Independent of pc_range's z, which only places the camera pillars.
    lidar_z_range: Tuple[float, float] = (-3.0, 5.0)
    # "sparse": SafeDrive's SpMiddleResNetFHD on spconv (pip install
    # spconv-cu126 on this rig).  "pillar": spconv-free PointPillars-style
    # fallback.  Both return [embed_dims, bev_h, bev_w].
    lidar_encoder: str = "sparse"
    # sparse: (x_right, y_forward, z) metres.  The backbone's stride is 8, so
    # x/y voxels must be cell / 8 = 0.08 m -> a 400 x 800 x 40 grid (41 deep
    # with SafeDrive's +1), which the four stride-2 stages take to 50 x 100 x 2
    # -> 128 * 2 = 256 channels = embed_dims, no projection.  This is also the
    # BEVFusion teacher's voxel size.
    lidar_voxel_size: Tuple[float, float, float] = (0.08, 0.08, 0.2)
    # pillar: (longitudinal, lateral) metres.  Must divide the BEV cell by one
    # power of two on both axes; 0.64 / 0.16 = 4 -> a 200 x 400 pillar canvas
    # that the 2D backbone reduces back to 50 x 100.
    lidar_pillar_size: Tuple[float, float] = (0.16, 0.16)
    lidar_pillar_channels: int = 64
    lidar_backbone_channels: Tuple[int, ...] = (64, 128, 256)
    lidar_backbone_layers: Tuple[int, ...] = (3, 5, 5)
    lidar_neck_channels: int = 128
    # Sampling points of the per-layer LiDAR deformable cross-attention.
    lidar_attn_points: int = 8

    # ------------------------------------------------------------------ #
    # planning head
    # ------------------------------------------------------------------ #
    # Three Pre-LN blocks; optionally read private decoder memories after BEV.
    # Scene-token settings below are retained as inert config compatibility.
    use_stl: bool = False
    use_task_interaction: bool = True
    plan_num_layers: int = 3
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
    # Half-angle about the forward axis.  The three front NAVSIM cameras cover
    # approximately -80..+80 degrees; out-of-FOV boxes must not become false
    # negative supervision for this camera-only detector.  This filter applies
    # to detection/motion GT and its auxiliary evaluator, not to HD-map GT.
    det_fov_half_angle_deg: float = 80.0
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
    map_num_classes: int = 4        # road, walkway, centerline, crosswalk (MAP_CLASS_NAMES)
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
    # Retain the previous target; re-measure shares for the new planner.
    # Planning includes direct BEV and both latent-memory paths. Set to None
    # for the unscaled reference.
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

    # ------------------------------------------------------------------ #
    # planning-readout distillation from the cached ReSMap teacher
    # (report/19_planning_readout.md).  All off by default.
    # ------------------------------------------------------------------ #
    # none | readout | random | feature   (see readout/distill.py)
    kd_mode: str = "none"
    # Root of the sharded ReSMap cache (index.json, bev/, vectors/, ...).
    kd_teacher_cache: Optional[str] = None
    # Stage-1 readout trained on the teacher BEV; its h_enc is frozen.
    kd_readout_ckpt: Optional[str] = None
    # cosine | mse, on z (readout/random) or on the BEV (feature).
    kd_distance: str = "cosine"
    # Fixed lambda.  With "distill" in grad_balance_target the balancer then
    # rescales the term's BEV gradient on top of this.
    kd_weight: float = 1.0
    # Micro-batch counts, the grad_balance_warmup_iters convention: lambda is 0
    # for kd_warmup_iters, then ramps linearly to kd_weight over kd_ramp_iters.
    # z_S is meaningless while the student BEV is still random.
    kd_warmup_iters: int = 0
    kd_ramp_iters: int = 0
    # Identity-initialised 1x1 conv on the student BEV before h_enc (design s05).
    kd_adapter: bool = False
    # none | global | command.  Remove the scene-independent part of z before the
    # distance (report/19 s3): 95% of the energy of z is the per-command mean, and
    # matching it carries no scene information.  Only for kd_mode readout|random.
    kd_center: str = "none"
    # EMA momentum of those running means over micro-batches.
    kd_center_momentum: float = 0.99
    # kd_mode=random: seed of the fixed random 256-d projection.
    kd_random_seed: int = 0
    # kd_mode=sens_feature: random trajectory-space probes per step (K backward
    # passes through the frozen reader) estimating the per-cell trajectory change.
    kd_sens_probes: int = 4

    # Planning-side map consistency (plan_map.py): hinge on the commanded
    # trajectory's footprint corners against the SDF of the drivable area the
    # DAC metric uses.  0 = off: no target builder, no term.
    # ---- v2 planner: WoTE-style anchors + PDM-score rewards (modules/anchor_planner.py) ----
    # Off = the v1 single-query L1 regression.  The two files are WoTE's released
    # extra data (K-means anchors, and the PDM scores of every anchor per token).
    plan_anchor: bool = False
    plan_anchor_file: Optional[str] = None
    plan_score_file: Optional[str] = None
    plan_reward_weights: Tuple[float, float, float, float] = (0.1, 0.5, 0.5, 1.0)   # WoTE reward_weights
    plan_topk: int = 6
    # Kinematic-bicycle output layer (modules/kinematics.py, from TOAD): the offset
    # head predicts control corrections and the poses are their rollout.
    plan_kinematic: bool = False
    # Heading from the path (modules/kinematics.bezier_xyyaw, from DiffusionDriveV2):
    # the offset head predicts x, y only.  v2 regressing heading separately left it
    # ~3 deg off the path direction (median, epoch 1), which the PDM tracker follows.
    plan_heading_from_xy: bool = False
    # evaluation-only diagnostic: select among anchor + offset re-encoded and rescored (WoTE test time)
    plan_rescore_refined: bool = False
    plan_offset_loss_weight: float = 1.0       # WoTE traj_offset_loss_weight
    plan_im_reward_weight: float = 1.0         # WoTE im_reward_loss (unweighted)
    plan_sim_reward_weight: float = 1.0        # WoTE sim_reward_loss (unweighted; x5 inside)

    # Evaluation-only post-processing: heading := direction of travel of the planned
    # path (para_ssr_model.heading_from_path).  On the v1 control it lifts PDMS
    # 84.74 -> 85.61 (DAC +0.70, EP +0.75): the PDM simulator tracks position AND
    # heading, and predicted headings that disagree with the path pull it off line.
    heading_from_path: bool = False
    # Evaluation-only: TOAD's kinematic projection (modules/kinematics.py) of the
    # planned trajectory, i.e. its test-time step without the CEM search.
    kinematic_projection: bool = False

    plan_map_weight: float = 0.0
    plan_map_margin: float = 0.0
    # (x0, x1, y0, y1) in the current ego frame of NAVSIM trajectories:
    # x forward, y left, rear axle.  4 s at 18 m/s stays inside x1.
    plan_map_extent: Tuple[float, float, float, float] = (-8.0, 72.0, -32.0, 32.0)
    plan_map_res: float = 0.25
    plan_map_clip: float = 10.0
    # gt | teacher.  "teacher" supervises the map head with the ReSMap vector
    # head (arm 3: teacher knowledge through the label path).  Map *evaluation*
    # always uses GT.
    map_label_source: str = "gt"
    map_pseudo_score_thr: float = 0.3

    # Drivable-area RASTER on the shared BEV grid (bev_seg.py).  The vector map head emits ROAD as
    # open polylines, which say where an edge is but never which side is legal; DAC is a
    # point-in-polygon test on the ego footprint.  0 = off: no head, no target, no parameters.
    bev_seg_weight: float = 0.0
    bev_seg_hidden: int = 128
    bev_seg_pos_weight: float = 1.0
    # vector: the polyline map head (default).  seg: the raster head REPLACES it and its token view
    # feeds the planner's map memories.  With seg, bev_seg_weight must be > 0.
    map_head_type: str = "vector"
    bev_seg_token_pool: int = 5

    # One differentiable feasibility cost on the 256 refined candidates (feasibility.py); 0 = off.
    # feas_field: "gt" isolates the mechanism from the field's quality, "pred" lets the same
    # gradient reach the perception path.  margin follows the measured penetration (median 0.32 m):
    # a generous margin buys DAC by making the planner timid, and EP is the larger gap.
    feas_weight: float = 0.0
    feas_field: str = "gt"                 # gt | pred
    feas_field_head: bool = True           # false + feas_field=gt: no predicted-field head at all
    feas_to_offset: bool = True
    feas_to_score: bool = False
    feas_weighting: str = "softmax"        # softmax | topk | uniform | select
    # select: uniform over (label top-k) U (model top-k) U (WTA winner), zero elsewhere.  softmax at
    # temp 1 measured to spread over 198/256 candidates and collapse them (feasibility.select_weights)
    feas_select_label_k: int = 16
    feas_select_model_k: int = 16
    # hold the candidates OUTSIDE the refined set near their anchor: hinge^2 beyond the radius.  The
    # offset head is shared, so select alone let them drift (B2: |offset| 5.6 -> 7.0 m, epoch 5->10)
    feas_offset_anchor_weight: float = 0.0
    feas_offset_anchor_radius: float = 2.0
    # TTC head target (feasibility.ttc_label): anchor | geometric | band | hybrid.  Measured on 1.5M
    # navtest candidates: the anchor's label is the more accurate below 2 m of offset and above
    # 12 m, the geometric one (agent field + velocity projection) in between -- hence "band".
    feas_ttc_target: str = "anchor"
    feas_ttc_margin: float = 0.0
    feas_ttc_band_lo: float = 2.0
    feas_ttc_band_hi: float = 12.0
    # EP head target (feasibility.ep_label): anchor | geometric -- progress on the route centerline of
    # the emitted trajectory, normalised by the human's, times the GT corridor verdict
    feas_ep_target: str = "anchor"
    feas_route_file: str = ""              # packed route centerlines, no .npy suffix
    feas_route_max_offset: float = 5.0     # a route farther than this from the ego is a failed extraction -> no route
    # driving force on the refined set (feasibility.drive_terms): progress up to the human's along
    # the route, lane deviation beyond `free` m, first-step speed vs the measured motion
    feas_progress_weight: float = 0.0
    feas_lateral_weight: float = 0.0
    feas_lateral_free: float = 1.0
    feas_continuity_weight: float = 0.0
    feas_continuity_tol: float = 0.5
    # stage 2 (refine.py): refine only the selected candidate; stage 1 keeps a diverse pool
    feas_field_priors: bool = False        # field head also reads seg + det/motion (feasibility.field_priors)
    feas_prior_topk: int = 40
    # listwise top-1 ranking inside the corridor (feasibility.admissible_rank_loss); 0 = off
    rank_weight: float = 0.0
    rank_set: str = "gate"                 # gate (deployment set) | corridor (GT set) | all (control: all K)
    rank_tau: float = 0.05                 # label temperature
    rank_temp: float = 1.0                 # model-score temperature
    # fine-tune only the score heads (reward / sim heads, offset_embed); everything else frozen, in eval mode
    # selection-curse loss (feasibility.selection_curse_loss): pinball on the EP/TTC/C heads of the
    # deployed rule's top-k candidates, over-estimation weighted (1 - tau) vs tau
    curse_weight: float = 0.0
    curse_set: str = "topk"                # topk (would-be picks) | all (control: asymmetry only)
    curse_topk: int = 8
    curse_tau: float = 0.2
    finetune_score_only: bool = False
    # fine-tune only stage 2 (the refiner); everything else frozen, in eval mode
    finetune_refiner_only: bool = False
    refine_enable: bool = False
    refine_weight: float = 1.0
    refine_hidden: int = 256
    refine_margin: float = 0.4
    refine_dyn_margin: float = 0.2
    refine_corridor_weight: float = 1.0
    refine_corridor_hinge: bool = False    # relu(-phi)^2: no push once inside the margin (refine.py)
    refine_progress_weight: float = 1.0
    refine_lateral_weight: float = 0.1
    refine_lateral_free: float = 1.0
    refine_smooth_weight: float = 1.0
    refine_trust_weight: float = 0.05
    refine_continuity_weight: float = 0.5
    refine_continuity_tol: float = 0.5
    refine_imitation_weight: float = 1.0
    refine_ttc_weight: float = 0.0         # do-no-harm on the TTC clearance of the pick (refine.py)
    refine_ttc_margin: float = 1.0
    refine_fallback: bool = True
    refine_apply: bool = True              # evaluation: false emits stage 1's pick (isolates stage 2)
    # BEV selector v7 distillation (teammate, origin/aux_distill docs/bev_selector_v7.md; distill/):
    # anchor queries cross-attend BEVFusion + ReSMap teacher BEV and the student BEV with a Gaussian
    # bias along the 256 anchors; attention KL + feature hybrid loss.  Training only.
    distill_v7: bool = False
    distill_v7_weight: float = 1.0         # teammate's distill_loss_weight
    distill_v7_root: str = "/home/external-user/datasets/teacher_cache"
    distill_v7_resmap_flip: bool = False   # teammate flips ReSMap L/R; measured wrong (distill/teacher_store.py)
    distill_v7_heads: int = 4
    distill_spatial_sigma: float = 4.0
    distill_attn_loss_scale: float = 5.0
    distill_resmap_scale: float = 3.0
    distill_anchor_winner_boost: float = 2.0
    distill_anchor_tau: float = 1.0
    distill_anchor_loss_scale: float = 10.0
    distill_anchor_loss_type: str = "hybrid"
    distill_anchor_proj: bool = True
    feas_temp: float = 1.0
    feas_topk: int = 32
    feas_margin: float = 0.4
    feas_beta: float = 0.2
    feas_clip: float = 8.0
    feas_offset_weight: float = 1.0
    feas_score_weight: float = 1.0
    feas_score_tau: float = 0.5   # measured: separates DAC=0 (c0 median 8.84) from DAC=1 (0.00) best
    # Dynamic field: one plane per ego step, agents where they will BE.  0 = off (static only).
    # The margin is small because NC is an overlap test, not a clearance test.
    feas_dyn_weight: float = 0.0
    feas_dyn_margin: float = 0.2
    # INFERENCE gate on the field itself (feasibility.field_gate_select).  feas_to_score puts the
    # corridor into a sigmoid head and the gate reads the head, so at deployment the geometry
    # survives only as far as that scalar copied it -- measured, not intact: reading the predicted
    # field at selection moves realised DAC 95.35 -> 95.90% at margin 0.10, ground-truth ceiling
    # 96.45% (tools/plan_v2/geo_gate_probe.py).  This makes the field the deployed constraint, so
    # training's cost and the deployed verdict are the same object.  Selection rule only: it needs
    # no retraining and applies to any finished v2 anchor checkpoint.
    feas_select_gate: bool = False
    feas_select_margin: float = 0.1        # probe optimum for the PREDICTED field (GT's is 0.4)
    feas_select_dyn_margin: float = 0.0
    feas_field_weight: float = 1.0
    feas_hidden: int = 128
    # TARGET grid only.  Bilinear sampling of an exact SDF errs 0.142 m at 0.64 m cells and 0.035 m
    # at 0.16 m, against a 0.32 m penetration, so the label is rasterised 4x finer.  The PREDICTED
    # field stays on the BEV grid: a head finer than every other head would buy accuracy with
    # resolution instead of with the mechanism.  The resulting gap is a result, not a defect.
    feas_target_upsample: int = 4
    # weighted: WoTE's rule.  corridor_first: DAC gates instead of voting -- only meaningful once
    # feasibility owns the DAC head, since only then does it describe the EMITTED trajectory.
    # "wote": 5 score heads (NC, DAC, EP, TTC, comfort), WoTE's label set.
    # "corridor": DAC+NC collapse into ONE "feasible" head fed by the merged corridor field,
    # which always gates; EP/TTC/comfort keep their WoTE labels and order the admissible set.
    plan_head_mode: str = "wote"
    # score heads see the (detached) offset of the candidate they score -- anchor_planner.score
    plan_score_sees_offset: bool = False
    # hold the non-winner offsets near their anchor (anchor_planner.anchor_plan_losses): hinge^2 on the
    # per-step xy offset beyond the radius.  Their score heads learn the ANCHOR's label; the car drives
    # anchor + offset.  0 = off.
    plan_offset_hold_weight: float = 0.0
    plan_offset_hold_radius: float = 1.0
    plan_select_mode: str = "weighted"
    plan_select_dac_thr: float = 0.5
    plan_select_gate: float = 20.0
    # Grounding weight = the cost's own sensitivity, sigmoid((margin + band - sdf_gt)/band), floored
    # so deep-interior cells cannot drift down and invent obstacles.
    feas_ground_band: float = 1.0
    feas_ground_floor: float = 0.05
    # Let the field's grounding loss settle before the planner is steered by it.
    # Does the feasibility cost's gradient reach the predicted field?  Measured to be degenerate
    # (see feasibility.py); kept only so the collapse can be reproduced as an ablation.
    feas_cost_grad_to_field: bool = False
    feas_warmup_iters: int = 0
    feas_ramp_iters: int = 0

    # Eval auxiliary predictions. With interaction on, decoders always run;
    # with it off, evaluation can omit their computation as well.
    test_aux_heads: bool = False

    # ------------------------------------------------------------------ #
    # optimisation -- WoTE/SeerDrive recipe, see report #09 §4
    # ------------------------------------------------------------------ #
    # Also used by the shared Hydra training entrypoint; wrappers already use 32.
    training_precision: int = 32
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
