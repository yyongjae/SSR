"""PARA-SSR (navsim) configuration.

Values fall into three groups:

* **architecture** -- carried over from ``projects/configs/SSR/PARA_SSR_e2e.py``
  unchanged wherever navsim allows it.
* **navsim adaptations** -- horizon, command count, class sets, image geometry.
* **optimisation** -- the WoTE/SeerDrive-derived recipe (see report #09 §4).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

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
    pc_range: Tuple[float, ...] = (-32.0, 0.0, -1.0, 32.0, 32.0, 3.0)
    bev_h: int = 50     # y_forward: 32 m / 50 = 0.64 m per row
    bev_w: int = 100    # x_right:  64 m / 100 = 0.64 m per column

    # Deliberately identical to pc_range.  A different map range would make the
    # map decoder's normalized reference points address the wrong physical BEV
    # cells, even if both tensors happened to have compatible shapes.
    map_pc_range: Tuple[float, ...] = (-32.0, 0.0, -1.0, 32.0, 32.0, 3.0)
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
    # km 10/27: 'seg' = raster map head (modules/map_seg_head.py): BEV segmentation of the same four classes at the 2x
    # BEV grid (GT para_ssr_targets._compute_map_seg_targets), planner map memory = map_seg_tokens^2 pooled BEV blocks
    map_repr: str = "vector"
    map_seg_tokens: int = 20
    map_seg_bce_weight: float = 1.0
    map_seg_dice_weight: float = 1.0

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
    # km 10/15: corridor head (modules/corridor_head.py) on bev_embed -> GT corridor of the metric cache
    # (tools/corridor/build_corridor_dt01.py): static drivable SDF + agent occupancy swept over 0.5 s windows
    # (8 planes, from the 0.1 s planes), 100 x 200 at 0.32 m.  Used at inference to evaluate every candidate.
    corridor_weight: float = 0.0
    corridor_dir: str = "/home/external-user/ssd/kyungmin_ck/corridor_dt01"
    corridor_tokens_root: str = "/home/external-user/ssd/kyungmin_ck/km_runs/ck2e2e_bevkd/eval/root/packed"
    corridor_occ_weight: float = 1.0
    # km 10/15 (2): corridor CONSTRAINT on the plan.  Every candidate (anchor + offset, all K) is tracked by the
    # differentiable port of the official LQR + bicycle simulation (modules/torch_tracker.py); its footprint is
    # penalised (hinge, metres) where it leaves the GT drivable SDF (margin static_margin) or enters the GT agent
    # occupancy at the same 0.1 s step (margin occ_margin, front half of the footprint).  0 = off.
    corridor_plan_weight: float = 0.0
    corridor_plan_static_margin: float = 0.1
    corridor_plan_occ_margin: float = 0.3
    corridor_plan_cap: float = 3.0
    # km 10/15 (3): TTC term of the constraint (official _calculate_ttc on the GT agent field): at every 0.1 s step the
    # front half of the footprint, shifted along the heading by speed * {0.3, 0.6, 0.9} s, vs the agents at t + dt
    # (planes past 4.0 s = the 4.0 s plane), same margin / cap as the occupancy term.  Relative weight; 0 = off.
    corridor_plan_ttc_weight: float = 0.0
    # km 10/15 (4): supervise the EP head only on anchors whose label NC * DAC > 0 (EP given feasibility; see
    # anchor_planner.anchor_plan_losses).  Normalisation = mean over the supervised entries.
    plan_ep_feasible_only: bool = False
    # km 10/15 (4): agent-occupancy planes of the corridor head: 8 = union over 0.5 s windows, 40 = every 0.1 s
    # (the official NC / TTC time steps)
    corridor_occ_planes: int = 8
    # km 10/16: route progress / distance channels (the official EP path, metric_cache.centerline) predicted at the
    # BEV resolution; GT: tools/corridor/build_route32.py (32 m) or build_corridor64.py (64 m).  0 = off.
    corridor_route_weight: float = 0.0
    corridor_route_dir: str = "/home/external-user/ssd/kyungmin_ck/route32"
    # "dt01" = 32 m raster GT (corridor_dir = corridor_dt01) | "c64" = 64 m GT (corridor_dir = corridor64,
    # tools/corridor/build_corridor64.py; needs pc_range forward 64 m, bev_h 100)
    corridor_plan_prog_weight: float = 0.0     # km 10/16 v3.1: progress reward inside the GT-safe corridor
    corridor_plan_prog_cap: float = 1.2        # reward saturates at cap x the GT trajectory's route progress
    corridor_plan_comfort_weight: float = 0.0  # official comfort thresholds as a hinge on every tracked candidate
    # km 10/22: hinge on every tracked candidate's own EC features (rms jerk > 0.25, rms yaw accel > 0.05 = half the
    # two-frame EC thresholds), corridor_head.corridor_plan_loss.  0 = off.
    corridor_plan_ec_smooth_weight: float = 0.0
    # km 10/23: fraction of the training samples whose scorer inputs are perturbed (hazards partly hidden;
    # candidate_scorer.PERT_DEFAULTS, magnitudes fixed a priori).  0 = off; never active at eval.
    cand_scorer_pert_p: float = 0.0
    # km 10/25: fraction of each training sample's scorer candidates replaced by anchor + u * offset, u ~ U(0, 1)
    # (the fitted offsets almost never fail on train scenes: NC fail 0.4% vs 6.0% on navtest, c64 ep25).  Labels stay
    # the official terms of exactly these candidates vs GT.  0 = off; never active at eval.
    cand_scorer_mix_p: float = 0.0
    # km 10/25 (cross-fitting): 2 = two CrossFitPlanner copies (planner_head.py), each trained only on the scenes of
    # its log-disjoint fold (crc32(log_name) % 2, exact_score.ExactFullTargetBuilder 'cf_fold'); the candidate scorer
    # trains on the OTHER fold's planner's candidates (scenes that planner never fitted), labelled by their own official
    # terms.  Needs input_target=true (the fold reaches the forward).  0 = off; eval always scores the main candidates.
    cand_cf_folds: int = 0
    cand_cf_weight: float = 1.0          # the cross-fit planners' own loss (imitation L1 + corridor_plan_weight x plan)
    corridor_verdict_weight: float = 0.0       # km 10/16 v3.1: predicted-corridor verdicts vs GT verdicts
    corridor_verdict_anchors: int = 64
    corridor_agent_attn: bool = False      # km 10/16 v3.1: corridor pixels attend to det/motion agent tokens
    corridor_agent_k: int = 32
    corridor_agent_detach: bool = True     # the corridor loss does not update the det/motion head
    corridor_format: str = "dt01"
    # km 10/18 (v3.2b): corridor GT loss shape.  occ: BCE pos_weight, plane weight 1 + time_weight * k / (P - 1)
    # (later planes = the TTC / late-NC planes the selector misses); sdf: boundary weight 1 + bw * exp(-|s|).
    # sdf_sigma: an aleatoric SDF scale head sigma (Gaussian NLL on the detached SDF error), the selector's DAC verdict
    # becomes p_DAC = Phi(min_corners sdf / sigma) (the verdict loss trains it at the same quantity).
    corridor_occ_pos_weight: float = 4.0
    corridor_occ_time_weight: float = 0.0
    corridor_sdf_boundary_weight: float = 4.0
    corridor_sdf_sigma: bool = False
    # km 10/17: trajectory the agent submits (modules/corridor_select.py): v2 = the anchor planner's argmax (no corridor)
    # | pred_multiply_t<tau> | pred_replace_t<tau> = every candidate tracked and judged on the predicted corridor
    # | km 10/18: pred_multiply_sigma (p_DAC from the sigma head) | suffix _ttc adds the predicted TTC verdict
    corridor_select_rule: str = "v2"
    # km 10/19 (C): candidate scorer (modules/candidate_scorer.py) on the model's own tracked candidates, trained on
    # their exact GT verdicts (modules/exact_verdict.py, GT tables of tools/corridor/build_exact.py in exact_dir).
    # corridor_select_rule = cand_expected submits the candidate of the highest expected PDMS.  0 = off.
    cand_scorer_weight: float = 0.0
    cand_scorer_fields: bool = True        # + the predicted corridor read-out along each candidate (needs corridor head)
    cand_scorer_prog_weight: float = 1.0
    exact_dir: str = "/home/external-user/ssd/kyungmin_ck/exact"
    # km 10/19 (C v3): official PDMS (v1) + EPDMS (v2, EC excluded) of every candidate (modules/exact_score.py): v1
    # tables in exact_dir (+ build_exact.py --only pdm,ego), v2 tables of the v2 metric cache in exact2_dir.
    # corridor_select_rule = cand_official: argmax expected EPDMS among the candidates within cand_select_eps of the
    # best expected PDMS (eps tuned on navtrain_train only)
    cand_official: bool = False
    exact2_dir: str = "/home/external-user/ssd/kyungmin_ck/exact2"
    cand_select_eps: float = 0.02
    # smoothness weight in the cand_official choice (km 10/20 C v3.1).  Read through getattr before; a field now so a
    # hydra override can set it (report 124 V8: the training copy pick, val and navtest share eps / lam)
    cand_select_lam: float = 0.0

    # ------------------------------------------------------------------ #
    # refiner v6c (SSR/report/124_refiner_v6c_design_on_corridor_cf.md; navsim/agents/para_ssr/refiner_v6c/).
    # Off by default: no module is built and the forward / loss are bit-identical to the baseline.
    # ------------------------------------------------------------------ #
    v6c: bool = False
    v6c_stage: str = "D"              # P: the baseline is frozen (no_grad, BN / dropout / GridMask in eval, cf copy
                                      # candidates kept); D: joint training from scratch
    v6c_mode: str = "B"               # B: R-J picks tau_base or tau_v6 (submitted); A: always tau_v6
    v6c_hid: int = 192
    v6c_r: float = 0.1                # stage D: share of R-E's loss reaching the BEV / corridor fields (GradScale); P: 0
    v6c_a_max: float = 1.0            # deceleration amplitude (m/s^2), V7
    v6c_d_max: float = 0.5            # lateral residual bound (m), V7
    v6c_decoder_mode: str = "A"       # A: deceleration only; B: the acceleration arm (V7)
    v6c_lon_st_slope: float = 0.1     # straight-through slope of the mode-A clamp (10-28: 0.1 as refiner T / E / CK;
                                      # the 10-27 run used 0 = off and every longitudinal control died)
    v6c_lon_pos_penalty: float = 1.0  # mode A: weight of relu(z_lon) (z > 0 is clamped away); 0 = off (10-28 review)
    v6c_reset_rc_adam: bool = True    # drop R-C / R-J AdamW state on the first micro-batch past the warm-up (10-28)
    v6c_rc_out: str = "per_control"   # R-C read-out: per_control (10-28) | shared (the 10-27 checkpoints)
    v6c_lon_target: str = "official"  # longitudinal controls: official decel target (10-28 review) | off (time-mean hinge)
    v6c_lon_levels: Tuple[float, ...] = (-0.25, -0.5, -1.0, -2.0)   # z of the decel variants scored, mild -> strong
    v6c_lon_target_weight: float = 1.0
    v6c_lon_fix_weight: float = 4.0   # weight of a fixable failing sample vs a passing one (target 0)
    v6c_rear_mask: Optional[bool] = None  # student reads / L_corr hinges: points behind the ego (x < 0) are out of range
                                      # (10-28 review; CH._sample's border padding repeated row 0); None = True except
                                      # v6c_rc_out 'shared' (the 10-27 checkpoints, old rule); T-E keeps the old rule
    v6c_h_arm: str = "E"              # R-C input: E (default) | EH (E + H detached) | H (H only)
    v6c_switch_delta: float = 0.0     # R-J switch margin on the expected PDMS (V9)
    v6c_switch_kappa: float = -1.0    # EC-proxy guard on the switch (< 0: off = kappa infinite)
    v6c_pos_weight: float = 4.0       # BCE pos_weight of the per-step logit channels
    v6c_w_E: float = 1.0
    v6c_w_E_re: float = 1.0           # L_E on tau_v6's states (D14)
    v6c_w_corr: float = 1.0
    v6c_w_J: float = 1.0
    v6c_corr_teacher: float = 1.0
    v6c_corr_gt: float = 1.0
    v6c_corr_road: float = 1.0
    v6c_corr_comfort: float = 1.0
    v6c_corr_ec: float = 0.1
    v6c_corr_prog: float = 0.2
    v6c_gt_margin: float = 0.1        # GT guard margin on the occupancy distance (m)
    v6c_teacher_margin: float = 0.3   # teacher term margin on the calibrated occupancy probability
    v6c_teacher_discount: float = 1.0 # teacher term step weight gamma^t (1 = time mean; arm)
    v6c_fix_source: str = "teacher"   # source of the fixing term on a failing tau_base: teacher (D) | gt (GT fixing
                                      # term); independent of v6c_kd (report 124 control B = gt + v6c_kd false)
    v6c_kd: bool = True               # K-E / K-M teacher KD into R-E (independent of v6c_fix_source, feedback 10-27)
    v6c_kd_lambda: float = 0.5        # lambda_c start value (every learned channel; per-channel overrides below)
    v6c_kd_lambda_off: float = 0.5    # K-M (off-road <- T-M)
    v6c_teacher_ckpt: str = ""        # stage-T teacher checkpoint (T-D / T-M / T-E fold copies + Platt); "" = none
    v6c_seed: int = 6                 # v6c init under fork_rng (the baseline init / RNG stream stay unchanged)
    v6c_warmup_steps: int = 0         # stage D: micro-batches of R-E-only warm-up on the path mixture (0 = off;
                                      # report 124 §4-3: epochs 0-4 = 5 * 85109 / (4 * NG) micro-batches per rank)
    # gt | teacher.  "teacher" supervises the map head with the ReSMap vector
    # head (arm 3: teacher knowledge through the label path).  Map *evaluation*
    # always uses GT.
    map_label_source: str = "gt"
    map_pseudo_score_thr: float = 0.3

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

    # ------------------------------------------------------------------ #
    # Stage E: student refiner (+ correction KD).  'off' = E0, nothing is
    # built, loaded or computed (bit-identical, tools/refiner/stageE_parity.py).
    # See navsim/agents/para_ssr/refiner/e2e.py and
    # report/refiner_T/stageE_impl_plan.md.
    # ------------------------------------------------------------------ #
    refiner_mode: str = "off"            # off | E1 (surrogate) | E2 (surrogate + KD)
    ref_bev_grad_scale: float = 0.1      # gradient scale student BEV <- refiner
    ref_w: float = 1.0                   # weight of the surrogate loss
    ref_term_weights: Dict[str, float] = field(
        default_factory=lambda: dict(col=1.0, ttc=1.0, dac=1.0, prog=2.0, cmf=0.1, mod=0.1)
    )
    ref_m_col: float = 0.15
    ref_m_dac: float = 0.05
    ref_m_ttc: float = 0.15
    ref_lon_st_slope: float = 0.1        # mode-A straight-through slope (training only)
    ref_perturb_frac: float = 0.5        # share of samples fed a perturbed sg(tau0)
    ref_seed: int = 0                    # refiner init + perturbation rng
    ref_lr_mult: float = 3.0             # refiner lr = lr * 3 (stage-T 3e-4 peak)
    ref_weight_decay: float = 0.01
    ref_clip: float = 1.0                # refiner-only grad-norm clip (before the global 35)
    ref_data_root: str = "/home/external-user/ssd/yongjae_refiner"
    kd_teacher_runs: Tuple[str, ...] = ()
    kd_lambda: float = 0.0               # lambda_KD constant (from the pilot rule)
    kd_space: str = ""                   # KD control space raw | tanh | decoded; E2 training refuses "" (user choice)
    kd_ramp: Tuple[float, float] = (5.0, 10.0)   # 0 before epoch a, linear to 1 at epoch b (REVISION 1: 5-9)
    ref_eval_traj: str = "final"         # eval output: final (tau_final) | tau0
    # KD balance / draft options (defaults = the behaviour before these options existed; refiner/e2e.py docstring)
    kd_balance: str = "fixed"            # fixed: kd_lambda x ramp | ema: w = kd_ratio * EMA(L_sur_w) / EMA(L_KD)
    kd_ratio: float = 1.0                # ema: target weighted-KD / weighted-surrogate magnitude ratio
    kd_ema_m: float = 0.99               # ema momentum (bias-corrected)
    kd_ema_floor: float = 1e-4           # ema: floor on EMA(L_KD) in the denominator
    kd_start_epoch: Optional[float] = None   # ema: KD on from this epoch (no ramp); None -> kd_ramp[0]
    kd_draft_source: str = "tau0"        # tau0: perturbed sg(tau0) | human_mix: perturbed GT human traj (else sg(tau0))
    ref_human_only_until: Optional[float] = None  # epoch < this: every draft = perturbed GT human (invalid -> unperturbed
                                         #   GT human); from then on kd_draft_source applies.  None = off
    kd_ratio_ramp_epochs: Optional[float] = None  # ema: ratio 0 -> kd_ratio linearly over [kd_start, kd_start + this]; None = off
    kd_weight_max: Optional[float] = None         # ema: cap on the KD weight (after the ratio); None = no cap
    grad_share_every: int = 50           # E1/E2: log ||dL/d bev_embed|| of E0 / surrogate / KD every n micro-batches (0 = off)

    # ------------------------------------------------------------------ #
    # CK Phase 2 (report 45; navsim/agents/para_ssr/ck/online.py): v2 + CK
    # end-to-end.  Empty dict / enabled false = off: nothing is built,
    # loaded or computed (v2 bit-identical).  Parsed by CKE2EConfig.from_any.
    # ------------------------------------------------------------------ #
    ck_e2e: Dict[str, Any] = field(default_factory=dict)
    # CK2 e2e (navsim/agents/para_ssr/ck/online2.py; SSR/report/47 s9): v2 + CK2 student
    # from scratch.  Empty dict / enabled false = off (nothing built / loaded / computed).
    # Parsed by CKE2E2Config.from_any; exclusive with ck_e2e.
    ck_e2e2: Dict[str, Any] = field(default_factory=dict)
    # self.log(..., sync_dist=...) of the per-micro-batch loss logs
    # (ParaSSRLoggingCallback, AgentLightningModule).  True = previous
    # behaviour; False removes the per-micro-batch all-reduce (DDP -14 %).
    log_sync_dist: bool = True

    # ------------------------------------------------------------------ #
    # BEV planning distillation, merged from aux_distill (ba7640d;
    # docs/bev_selector_v15.md, navsim/agents/para_ssr/distill/).  Off by
    # default: nothing below is read, built or loaded when use_distill is
    # false (refiner / v2 bit-identical).  On this kyungmin-v2 core only the
    # trajectory-anchor path (distill_selector_v6 .. v16) is supported; the
    # stage-1 adapter, corridor and register-selector paths (v1-v5) depend on
    # aux_distill's own v2 port (forward_from_bev, use_metric_planner) and are
    # not wired here.
    # ------------------------------------------------------------------ #
    use_distill: bool = False
    # Root holding BEVFusion ``cache_{train,val}_{50x100,100x100}`` and the ReSMap shards.
    distill_feature_root: str = ""
    # ``{branch: {cache_name, adapter: {...}, loss_weight[, flip_w, cache_subdirs, feature_key]}}``.
    # The anchor path (v6-v16) needs both branches and ignores ``adapter``.
    distill_branches: Dict[str, Dict] = field(
        default_factory=lambda: {
            "bevfusion": {
                "cache_name": "bevfusion",
                "adapter": {"channels": 256, "hidden_channels": 256, "dropout": 0.0},
                "loss_weight": 1.0,
            },
            "resmap": {
                "cache_name": "resmap",
                "adapter": {"channels": 256, "hidden_channels": 256, "dropout": 0.0},
                "loss_weight": 1.0,
                # fusion_v1: transpose only.  ReSMap's lateral index 0 is y_left = +32 m, i.e.
                # already student column 0; aux_distill's default flip_w=True mirrored the map
                # teacher left/right (GT check, 187 navtrain tokens: road IoU 0.965 transpose-only
                # vs 0.360 with the flip, better on 187/187; = refiner/resmap_cache.py, kd_bev_1).
                "flip_w": False,
            },
        }
    )
    # ``{branch: /path/to/stage1_adapter.ckpt}`` (stage-1 paths only; unused by v6-v16)
    distill_adapter_checkpoints: Dict[str, str] = field(default_factory=dict)
    # BEVFusion directory ``cache_train_50x100`` (or ``cache_train_100x100`` for 64m).
    # ReSMap is sharded and already 50x100 (or 100x100 for stage1_front64) after the
    # lateral/forward transpose. Same shape as the student, so stage 2 does not resample either teacher.
    distill_cache_size: Tuple[int, int] = (50, 100)
    distill_loss_weight: float = 1.0
    # Seed of the frozen random probe of the anchor path (anchor/cmd/status MLPs,
    # q/k/v/out projections): built under torch.random.fork_rng, so turning the
    # distillation on changes neither the v2 / refiner init nor the global RNG
    # stream (data order).  aux_distill drew the probe from the global RNG.
    distill_probe_seed: int = 0
    # Where the anchor path gets its two teacher BEVs (bevfusion, resmap):
    #   "store"   aux_distill's TeacherFeatureStore, read in the training process from
    #             distill_feature_root by targets['scene_token'];
    #   "targets" the dataloader targets kd_bev_0 (DET = BEVFusion) / kd_bev_1 (MAP = ReSMap,
    #             transpose only), loaded by the dataloader workers (DistillTeacherTargetBuilder
    #             for corridor-cf or CK2E2ETargetBuilder for ck_e2e2); tokens with kd_ok False
    #             are left out of the term.
    distill_teacher_source: str = "store"
    # Legacy v1-v5 knobs (inert on the anchor path; kept because the
    # aux_distill agent yamls set them).
    use_corridor_mask: bool = True
    # NAVSIM GT poses are (x_forward, y_left); BEV cells are SSR (x_right, y_forward).
    distill_trajectory_frame: str = "navsim"
    corridor_sigma_base: float = 2.5
    corridor_sigma_growth: float = 0.1
    corridor_base_weight: float = 0.1
    corridor_sigma_along: float = 4.0
    corridor_sigma_cross: float = 2.5
    corridor_sigma_along_growth: float = 0.15
    corridor_sigma_cross_growth: float = 0.1
    distill_use_role_masks: bool = True
    distill_agent_inflate: float = 1.5
    distill_agent_future_sigma: float = 1.5
    distill_map_sigma: float = 1.25
    distill_walkway_suppress: float = 0.0
    distill_boundary_kernel: int = 3
    distill_mse_weight: float = 1.0
    distill_cwd_weight: float = 0.0
    distill_cwd_tau: float = 4.0
    distill_relation_weight: float = 0.0
    distill_relation_samples: int = 64
    distill_attn_weight: float = 0.0
    distill_attn_tau: float = 0.5
    distill_adaptive_branch: bool = False
    distill_plan_look_weight: float = 0.0
    distill_plan_look_tau: float = 0.5
    distill_head_kd_weight: float = 0.0
    distill_head_kd_tau: float = 2.0
    distill_head_kd_det: bool = False
    distill_head_kd_det_geom: bool = False
    distill_selector: bool = False
    distill_selector_registers: int = 16
    distill_selector_div_sigma: float = 4.0
    distill_selector_tok_warmup_steps: int = 53195
    distill_selector_anchor_sigma: float = 6.0
    distill_selector_anchor_sigma_end: float = 2.0
    distill_selector_plan_tau: float = 0.3
    distill_selector_struct_mix: float = 0.5
    distill_selector_proj: bool = False
    distill_selector_tok_scale: float = 1.0
    distill_selector_loss_type: str = "l2"
    distill_selector_struct_mask_boost: float = 0.0
    # v6: Trajectory-Anchor-Guided Distillation
    distill_selector_v6: bool = False
    distill_anchor_winner_boost: float = 1.0
    distill_anchor_tau: float = 2.0
    distill_anchor_loss_scale: float = 10.0
    distill_anchor_loss_type: str = "hybrid"
    distill_anchor_proj: bool = True
    distill_anchor_mode: str = "cross_attn"
    # v7: + spatial bias & attention alignment
    distill_selector_v7: bool = False
    distill_spatial_sigma: float = 4.0
    distill_attn_loss_scale: float = 5.0
    distill_resmap_scale: float = 3.0
    # v8: hybrid anchor weighting & multi-scale spatial distance bias
    distill_selector_v8: bool = False
    distill_spatial_sigmas: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
    distill_lambda_gt: float = 0.6
    distill_lambda_cov: float = 0.2
    distill_lambda_risk: float = 0.2
    distill_tau_gt: float = 5.0
    distill_tau_risk: float = 8.0
    # v9/v10/v11/v12/v13/v14/v15/v16: Domain-Specific Planning-Grounded Probe (PGP) + structural fixes
    distill_selector_v9: bool = False
    distill_selector_v10: bool = False
    distill_selector_v11: bool = False
    distill_selector_v12: bool = False
    distill_selector_v13: bool = False
    distill_selector_v14: bool = False
    distill_selector_v15: bool = False
    distill_selector_v16: bool = False
    distill_proj_lr_mult: float = 3.0
    distill_probe_warmup_steps: int = 5300
    distill_probe_ramp_steps: int = 2650
    distill_probe_loss_weight: float = 0.5
    distill_probe_lr_mult: float = 3.0
    distill_probe_use_ema: bool = True
    distill_probe_ema_decay: float = 0.99
    # aux_distill stage-1 teacher adapter (not supported on this core)
    teacher_adapter_branch: str = "bevfusion"

    @property
    def needs_scene_token(self) -> bool:
        """Targets must carry the frame token for the cache-addressed distillation."""
        return bool(self.use_distill)

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
