"""PARA-SSR: SSR's navigation-guided sparse planner + PARA-Drive parallel heads.

Derived from ``SSR_e2e.py``.  Differences:

* The FFP / latent world model is gone (``latent_world_model``, ``loss_bev``,
  the ``index + 3`` future frame in the temporal queue).
* Three auxiliary modules now co-train the shared BEV feature *in parallel*:
  detection+motion, online mapping and occupancy prediction.  None of them feeds
  another module or the planner -- information passes only through the BEV
  feature (PARA-Drive, CVPR'24, Fig. 5).
* All three can be switched off at inference (``test_aux_heads=False``) and set
  to ``None`` here to reproduce the paper's module-necessity ablation (Table 5).
"""
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = len(class_names)

map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 100
queue_length = 3
total_epochs = 12

# --- occupancy settings -------------------------------------------------
# current frame + 6 future steps @0.5 s = the exact planning horizon (3 s), and
# exactly the 6 future steps stored per agent in gt_attr_labels, so occupancy and
# motion are supervised from one and the same source.
occ_n_future = 6
# Vehicles only -- UniAD's ``only_vehicle=True`` set, which is also what FIERY,
# ST-P3 and PARA-Drive supervise. A pedestrian channel was trained for 12 epochs
# and did not reach a useful operating point: 0.16% of BEV cells are pedestrian
# (vs 15.5% vehicle), each pedestrian is a ~7-cell blob, and the best IoU over
# all thresholds was 0.104 (at 0.05) against 0.574 for vehicles. At that
# threshold recall was only 20%, so no threshold traded off usefully.
#
# The head did learn *something* -- mean p is 124x higher on GT-positive cells
# than on negatives, a better ratio than vehicles get. What it cannot do at this
# input resolution (640x384, single-level C5) and cell size (0.3 x 0.6 m) is
# localise a 7-pixel blob well enough to be useful, and the top-k BCE settles
# non-discriminating pixels near n_pos/k = 16/2500 = 0.006, which makes the
# usable operating range very narrow. Dropping the channel is an empirical
# call, not a proof of impossibility.
# See report/01_baseline_vs_para_ssr.md 6.
occ_class_labels = (
    (0, 1, 2, 3, 4, 6, 7),  # vehicle
)
occ_num_classes = len(occ_class_labels)

# Kept for re-enabling; not referenced while occ_head=None.
_occ_head = dict(
        type='ParaOccHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        in_channels=_dim_,
        mid_channels=128,
        feat_channels=64,
        num_trunk_blocks=2,
        n_future=occ_n_future,
        num_classes=occ_num_classes,
        temporal_decoder=False,
        loss_weight=1.0,
        test_thresh=0.5,
        loss_mask=dict(
            type='OccBinarySegmentationLoss',
            use_top_k=True,
            top_k_ratio=0.25,
            future_discount=0.95,
            loss_weight=5.0),
        loss_dice=dict(
            type='OccDiceLoss',
            use_sigmoid=True,
            activate=True,
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0))

model = dict(
    type='ParaSSR',
    use_grid_mask=True,
    video_test_mode=True,
    # PARA-Drive's runtime advantage: aux heads are for interpretability /
    # safety checks only, so they stay off at inference. Flip to True to dump
    # detection / map / occupancy predictions.
    test_aux_heads=False,
    # --------------------------------------------------------------- #
    # Three multipliers stack on every loss term. Know which is which.  #
    # --------------------------------------------------------------- #
    #   1. `loss_weight` inside each loss cfg      -- VAD's layer, per TERM
    #   2. `loss_weight` on each aux head          -- per HEAD
    #   3. `task_loss_weight` below                -- per TASK
    #
    # effective = (1) x (2) x (3), and (2) and (3) are mathematically the same
    # knob: both multiply every term of one head. Two names for one lever is a
    # trap -- setting each to 0.5 gives 0.25, not 0.5. Both are 1.0 here, so
    # today every term sits at exactly VAD's published weight; prefer (3) if
    # one of them ever has to move.
    #
    # `grad_balance` is NOT in this stack. It scales the gradient entering
    # bev_embed and leaves each head's own parameter gradients alone, which is
    # why it can hold a task's influence on the shared feature down without
    # slowing the head itself.
    #   4. `motion` is its own task here, separate from `det`. They come out of
    #      one head, so they share a grad_balance VALVE, but they are different
    #      supervisions and get different loss weights -- setting motion=0.0
    #      leaves detection training untouched.
    #   `occ` is absent because the occupancy head is off (see below); the model
    #   still accepts it if the head is put back.
    task_loss_weight=dict(plan=1.0, det=1.0, motion=1.0, map=1.0),
    # --------------------------------------------------------------- #
    # Controlling the shared-BEV gradient: two mutually exclusive ways.  #
    # --------------------------------------------------------------- #
    # `aux_grad_scale` -- one constant on all three aux heads -- sets a KNOB and
    # lets the resulting share fall where it may. Measured on a real batch, at
    # 0.01 the planner still only received 4.1% of the BEV gradient.
    #
    # `grad_balance` sets the OUTCOME instead. Every `interval` iterations it
    # measures ||dL_k/d bev_embed|| per task, backs out the valve currently in
    # effect, and re-solves so the shares land on `target`. Planning is the
    # numeraire and is never scaled:
    #
    #   grad_balance=dict(
    #       target=dict(plan=0.4, det=0.2, map=0.2, occ=0.2),
    #       interval=200, momentum=0.9, clamp=(1e-4, 1.0), warmup_iters=500),
    #
    # Verified closed-loop on real batches (tools/verify_grad_balance.py). The
    # untouched shares are plan 0.07% / det 57% / map 27% / occ 15%, and the
    # solved valves come out det 5.5e-4, map 1.2e-3, occ 2.1e-3 -- a factor of
    # four apart from each other, which is why one shared constant cannot
    # balance them whatever value it takes.
    #
    # Detection and motion share a valve: one forward through det_motion_head
    # means one BEV input, so there can only be one scale. Motion is measured
    # separately as gshare/motion, which OVERLAPS det and is therefore excluded
    # from the shares that sum to 1.
    #
    # Neither option addresses the real limit: both balance gradient MAGNITUDE.
    # Two tasks pulling in opposite directions cancel in the actual update and
    # still read as large here.
    #
    # This file sets NEITHER. It is the reference 8-GPU definition and the v1
    # result it is compared against ran unscaled. Derived configs opt in --
    # see PARA_SSR_e2e_12ep.py.

    # Log ||d L_task / d bev_embed|| every N iterations as gnorm/* and gshare/*.
    # The four tasks share one representation and their gradients differ by
    # orders of magnitude, so the loss curves alone do not reveal which task is
    # steering the BEV feature -- gshare/plan is the number to watch. Costs a
    # few extra partial backwards per epoch (<1% of runtime).
    grad_norm_log_interval=200,
    # Log aux-head QUALITY every N iterations as occ_iou*/... and occ_sep/...
    # A falling loss does not prove a head learned anything: the v1 dense map
    # head cut its BCE 18% over 12 epochs while converging to a spatially
    # constant output (positive/negative probability ratio 1.06x). IoU and the
    # separation ratio cannot be satisfied that way, so these are the numbers
    # that catch a dead head in epoch 1 instead of after a 15-hour run.
    # Costs one extra sigmoid + a few reductions on tensors already in memory.
    aux_metric_log_interval=200,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),

    # ------------------------------------------------------------------ #
    # planning head (always active) -- unchanged SSR sparse-token planner  #
    # ------------------------------------------------------------------ #
    pts_bbox_head=dict(
        type='ParaSSRHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        embed_dims=_dim_,
        in_channels=_dim_,
        pc_range=point_cloud_range,
        num_scenes=16,
        num_reg_fcs=2,
        fut_ts=6,
        ego_fut_mode=3,
        # NOTE: every dropout below is EXPLICIT on purpose. mmcv's
        # MultiheadAttention/BaseTransformerLayer keep their dropout defaults in
        # shared (class-level) dicts, and building any layer with the deprecated
        # `dropout=`/`ffn_dropout=` kwargs mutates those defaults for every layer
        # built afterwards in the same process. SSR_e2e.py leaves them implicit
        # and effectively trains these decoders with dropout 0.0; we pin 0.0
        # explicitly so the value cannot depend on module build order.
        latent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=3,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(type='MultiheadAttention', embed_dims=_dim_,
                         num_heads=8,
                         dropout_layer=dict(type='Dropout', drop_prob=0.0)),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        way_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(type='MultiheadAttention', embed_dims=_dim_,
                         num_heads=8,
                         dropout_layer=dict(type='Dropout', drop_prob=0.0)),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.0,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        transformer=dict(
            type='SSRPerceptionTransformer',
            map_num_vec=map_num_vec,
            map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
            rotate_prev_bev=False,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention', embed_dims=_dim_,
                             num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0)),

    # ------------------------------------------------------------------ #
    # AUX 1: tracking + motion prediction (parallel, train-only)          #
    # ------------------------------------------------------------------ #
    det_motion_head=dict(
        type='ParaDetMotionHead',
        num_query=300,
        num_classes=num_classes,
        embed_dims=_dim_,
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        code_size=10,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        num_reg_fcs=2,
        fut_ts=6,
        fut_mode=6,
        valid_fut_ts=6,
        use_pe=True,
        sync_cls_avg_factor=True,
        loss_weight=1.0,
        # 3 layers, matching VAD-Tiny -- which is what SSR is built on, and what
        # the map head here already uses. v1 ran this at 6 (VAD-Base) while
        # everything else stayed Tiny-scale, and the parameter count showed what
        # that cost: the 6-layer decoder alone was 4.26M of the head's 6.67M,
        # making detection 70% of all auxiliary parameters and, measured, 82% of
        # the gradient arriving at the shared BEV. At 3 layers det drops to
        # ~3.8M and sits much closer to map (2.91M), so grad_balance has a far
        # smaller correction to apply.
        #
        # v1's 6-layer definition is preserved in
        # the archived v1 run's resolved config.
        transformer_decoder=dict(
            type='DetectionTransformerDecoder',
            num_layers=3,
            return_intermediate=True,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(type='MultiheadAttention', embed_dims=_dim_,
                         num_heads=8, dropout=0.1),
                    dict(type='CustomMSDeformableAttention', embed_dims=_dim_,
                         num_levels=1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'))),
        # agent<->agent interaction only; NO map cross-attention -- that is
        # PARA-Drive Fig. 4 edge (1), which the paper removes. Layer layout is
        # VAD's motion_decoder verbatim (query=key=value=motion queries, so the
        # 'cross_attn' op is a self-attention; VAD wires key_padding_mask
        # through this op).
        motion_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(type='MultiheadAttention', embed_dims=_dim_,
                         num_heads=8, dropout=0.1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=num_classes),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=0.2)),

    # ------------------------------------------------------------------ #
    # AUX 2: online mapping (parallel, train-only)                        #
    # VAD-style VECTORISED map: 100 polyline instances x 20 points,       #
    # Hungarian-matched, regressed as point sets.                         #
    #                                                                     #
    # Previously this was the dense ParaMapSegHead (PARA-Drive's own      #
    # representation). It failed to train while still consuming 17.7% of  #
    # the shared BEV gradient, i.e. it injected near-noise into the       #
    # representation the planner depends on.                              #
    #                                                                     #
    # The failure was NOT class imbalance -- measured GT positive rates   #
    # are divider 8.2%, ped_crossing 1.8%, boundary 10.0%, and top-k 25%  #
    # brings the effective ratio to about 1:2. It was a loss of spatial   #
    # discrimination: mean p on GT-positive vs GT-negative was 1.06x for  #
    # boundary (a near-constant 0.29 everywhere) and 1.6x for divider,    #
    # against 7.8x for vehicle occupancy. A 0.41M 3-layer conv with a     #
    # ~7x7-cell receptive field cannot localise 30 m lane structures from #
    # single-level C5 features of a 0.4x-downscaled image.                #
    #                                                                     #
    # The vector formulation helps precisely there: the query embeddings  #
    # carry a shape prior ("lanes are long smooth curves"), so weak       #
    # evidence still yields plausible geometry. It is also the setup      #
    # SSR's own lineage (VAD-Tiny) was validated with.                    #
    # See report/01_baseline_vs_para_ssr.md 6 and report/02.              #
    #                                                                     #
    # PARA-Drive is about the parallel-vs-sequential wiring, not the map  #
    # representation (paper 2 explicitly lists both dense and vectorised  #
    # as valid mapping-module choices), so this stays faithful to it.     #
    # Nothing from this head reaches motion or the planner -- edges (1)   #
    # and (5) of PARA-Drive Fig. 4, both removed.                         #
    # ------------------------------------------------------------------ #
    map_head=dict(
        type='ParaMapHead',
        map_num_vec=map_num_vec,
        map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
        map_num_classes=map_num_classes,
        embed_dims=_dim_,
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        map_dir_interval=1,
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        num_reg_fcs=2,
        sync_cls_avg_factor=True,
        loss_weight=1.0,
        # num_layers=3 is VAD-Tiny, which is what SSR is built on
        # (VAD-Base uses 6). NOTE: `dropout=0.1` / `ffn_dropout=0.1` below are
        # the deprecated mmcv kwargs that mutate class-level default dicts.
        # They are safe *here* because det_motion_head is built immediately
        # before this head with the same values, and the planning head (built
        # earlier, in super().__init__) pins every dropout explicitly.
        transformer_decoder=dict(
            type='MapDetectionTransformerDecoder',
            num_layers=3,
            return_intermediate=True,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(type='MultiheadAttention', embed_dims=_dim_,
                         num_heads=8, dropout=0.1),
                    dict(type='CustomMSDeformableAttention',
                         embed_dims=_dim_, num_levels=1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'))),
        bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=map_num_classes),
        # VAD-Tiny weights verbatim. bbox/iou are 0.0: the (cx, cy, w, h) box is
        # only a matching aid, the supervision that matters is on the points.
        loss_map_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=2.0),
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005)),
    # --- alternative: dense semantic BEV segmentation --------------------
    # Kept for the representation ablation and as a per-pixel KD interface.
    # If re-enabled, the defect to fix is receptive field and evidence, not
    # class weighting: multi-level FPN features instead of C5 only, a much
    # larger decoder (PARA-Drive uses Panoptic SegFormer against this 0.41M
    # trunk), and only then thicker rasterisation. Re-weighting the loss
    # alone will not create spatial discrimination that the features do not
    # carry -- see report/01_baseline_vs_para_ssr.md 6.3.
    # map_head=dict(
    #     type='ParaMapSegHead',
    #     bev_h=bev_h_, bev_w=bev_w_, in_channels=_dim_,
    #     mid_channels=128, feat_channels=64,
    #     num_classes=map_num_classes, num_trunk_blocks=2,
    #     pc_range=point_cloud_range, line_thickness=2,
    #     loss_weight=1.0, test_thresh=0.5,
    #     loss_mask=dict(type='OccBinarySegmentationLoss', use_top_k=True,
    #                    top_k_ratio=0.25, future_discount=1.0, loss_weight=5.0),
    #     loss_dice=dict(type='OccDiceLoss', use_sigmoid=True, activate=True,
    #                    naive_dice=True, eps=1.0, loss_weight=1.0)),

    # ------------------------------------------------------------------ #
    # AUX 3: occupancy prediction (parallel, train-only, query-free)      #
    # Single-shot decoder: one conv trunk at full BEV resolution predicts #
    # all (n_future+1) x num_classes channels at once -- dense logits are  #
    # the KD interface. temporal_decoder=True switches to the UniAD-style #
    # recurrent decoder (heavier; capacity ablation / stronger teacher).  #
    # ------------------------------------------------------------------ #
    # --------------------------------------------------------------- #
    # Occupancy is OFF, at the base, so nothing can inherit it back on. #
    # --------------------------------------------------------------- #
    # It never reached a usable operating point in v1 or ver2: occ_sep/ratio sat
    # at 1.008 against a floor of 1.0 (a spatially constant output), and the
    # corrected metric says the true separation is worse than that number, not
    # better -- report #07 2. Carrying a head that has not been shown to learn
    # only adds parameters to the clip norm and takes a slice of the BEV
    # gradient, for no teacher.
    #
    # To re-enable: set `occ_head=_occ_head` below, put occ back into
    # task_loss_weight and into any grad_balance target, and restore
    # GenerateSSROccLabels plus the gt_occ_seg / gt_occ_valid collect keys in
    # both pipelines (all four sites are marked "occupancy:" further down).
    occ_head=None,

    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range),
        map_assigner=dict(
            type='MapHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
            pc_range=point_cloud_range))))

dataset_type = 'VADCustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=True),
    # Rasterise occupancy BEFORE the range filter so agents that are outside the
    # BEV now but drive into it later are still drawn.
    # occupancy: restore this entry to re-enable the occ head.
    # dict(type='GenerateSSROccLabels',
    # pc_range=point_cloud_range,
    # bev_size=(bev_h_, bev_w_),
    # n_future=occ_n_future,
    # class_labels=occ_class_labels),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
         with_ego=True),
    dict(type='CustomCollect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
               'gt_attr_labels'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5,
         file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=True),
    # Same placement as the train pipeline: before the range filter, so agents
    # that start outside the BEV box but drive into it are still rasterised.
    # Carried through to the result dict so evaluate_occ() can score the
    # occupancy head against the very GT the loss was trained on -- rebuilding
    # it inside evaluate() would duplicate the class grouping and horizon.
    # occupancy: restore this entry to re-enable the occ head.
    # dict(type='GenerateSSROccLabels',
    # pc_range=point_cloud_range,
    # bev_size=(bev_h_, bev_w_),
    # n_future=occ_n_future,
    # class_labels=occ_class_labels),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
                 with_label=False, with_ego=True),
            dict(type='CustomCollect3D',
                 keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img',
                       'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                       'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                       'gt_attr_labels'])])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'vad_nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        # PARA-SSR has no world model, so the temporal queue does not need the
        # index+3 future frame. This also makes queue[-1] (which union2one uses
        # as the metadata container) the *current* frame, aligning every GT.
        use_future_frame=False,
        map_classes=map_classes,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        box_type_3d='LiDAR',
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    val=dict(type=dataset_type,
             data_root=data_root,
             pc_range=point_cloud_range,
             ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,
             bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             map_classes=map_classes,
             map_ann_file=data_root + 'nuscenes_map_anns_val_ssr.json',
             map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
             map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
             use_pkl_result=True,
             custom_eval_version='vad_nusc_detection_cvpr_2019'),
    test=dict(type=dataset_type,
              data_root=data_root,
              pc_range=point_cloud_range,
              ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline,
              bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality, samples_per_gpu=1,
              map_classes=map_classes,
              map_ann_file=data_root + 'nuscenes_map_anns_val_ssr.json',
              map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
              map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
              use_pkl_result=True,
              custom_eval_version='vad_nusc_detection_cvpr_2019'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=5e-5,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

# max_norm stays at SSR/VAD's 35 so this run remains comparable to the
# plan-only baseline; the hook only adds telemetry, it does not change the
# optimisation. It is needed because clipping is global: measured on the v1
# 8-GPU run the total norm reaches 38.5 and clips on 100% of iterations from
# epoch 10, while the plan-only baseline sits at 0.93 and never clips. So the
# planner's update is being scaled by ~0.91 whenever aux heads are attached,
# and `aux_grad_scale` does not help -- it scales the BEV path, not the aux
# heads' own parameter gradients, which are what dominate the norm.
# Watch clip/rate and pnorm/* before deciding whether to raise max_norm.
optimizer_config = dict(
    type='ClipMonitorOptimizerHook',
    grad_clip=dict(max_norm=35, norm_type=2),
    group_norm_interval=200)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='SSRWandbLoggerHook',
            # GPU count and per-GPU batch are launcher concerns. ``run.sh``
            # overrides them while keeping the global batch fixed at 8.
            init_kwargs=dict(
                project='para-ssr',
                name='para_ssr_e2e',
                group='para',
                tags=['para-ssr', 'no-ffp', 'aux', 'global8'],
                config=dict(model='ParaSSR', ffp=False, global_batch=8,
                            aux='det+map_vec+occ_vehicle',
                            task_loss_weight='1/1/1/1')),
            # step with the runner's iteration so train curves and the
            # per-epoch validation metrics land on the same x axis
            by_epoch=False,
            interval=100),
    ])

# Validation every epoch. dataset.evaluate() returns plan_L2_* / plan_obj_col_*
# (plus *_avg); with test_aux_heads=False only planning is evaluated, which is
# the deployment configuration.
evaluation = dict(interval=1, pipeline=test_pipeline)

checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

find_unused_parameters = True

# Same training-time hooks as SSR_e2e.py so the comparison is apples-to-apples
# (the EMA weights are what SSR evaluates with).
custom_hooks = [
    # Watches the loss / gradient / evaluation stream and appends anything odd
    # to anomalies.jsonl in the work dir, plus an anomaly/* counter for wandb.
    # It never raises: a monitor that can kill a seven-day run over a threshold
    # it got wrong is worse than no monitor.
    #
    # priority='LOW' (70) places it between EvalHook (NORMAL=50) and LoggerHook
    # (VERY_LOW=90), so after_train_epoch sees the evaluation results sitting in
    # log_buffer.output before the logger clears them.
    dict(type='TrainingAnomalyHook', priority='LOW'),
    dict(type='CustomSetEpochInfoHook'),
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]
