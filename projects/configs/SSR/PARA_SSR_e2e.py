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
occ_class_labels = (
    (0, 1, 2, 3, 4, 6, 7),  # vehicle  (UniAD's only_vehicle set)
    (8, ),                  # pedestrian
)
occ_num_classes = len(occ_class_labels)

model = dict(
    type='ParaSSR',
    use_grid_mask=True,
    video_test_mode=True,
    # PARA-Drive's runtime advantage: aux heads are for interpretability /
    # safety checks only, so they stay off at inference. Flip to True to dump
    # detection / map / occupancy predictions.
    test_aux_heads=False,
    task_loss_weight=dict(plan=1.0, det=1.0, map=1.0, occ=1.0),
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
        # num_layers=6 follows VAD-Base. NOTE: SSR itself is built on VAD-Tiny,
        # which uses 3-layer map/motion decoders (VAD paper Sec. 4.1); dropping
        # to 3 here is the SSR-lineage-consistent (and cheaper) variant worth
        # ablating -- aux-head capacity, not correctness.
        transformer_decoder=dict(
            type='DetectionTransformerDecoder',
            num_layers=6,
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
    # Dense semantic BEV segmentation (PARA-Drive's own representation).  #
    # Chosen over the VAD vector head because its per-pixel logits are a  #
    # clean KD target (no Hungarian matching between teacher/student) and #
    # it is ~14x smaller (~3 GB less activation memory). To ablate the    #
    # representation axis, swap in the ParaMapHead block kept below.      #
    # ------------------------------------------------------------------ #
    map_head=dict(
        type='ParaMapSegHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        in_channels=_dim_,
        mid_channels=128,
        feat_channels=64,
        num_classes=map_num_classes,
        num_trunk_blocks=2,
        pc_range=point_cloud_range,
        line_thickness=2,
        loss_weight=1.0,
        test_thresh=0.5,
        loss_mask=dict(
            type='OccBinarySegmentationLoss',
            use_top_k=True,
            top_k_ratio=0.25,
            future_discount=1.0,
            loss_weight=5.0),
        loss_dice=dict(
            type='OccDiceLoss',
            use_sigmoid=True,
            activate=True,
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0)),
    # --- alternative: VAD-style vectorised map (chamfer-mAP evaluable) ---
    # map_head=dict(
    #     type='ParaMapHead',
    #     map_num_vec=map_num_vec,
    #     map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
    #     map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
    #     map_num_classes=map_num_classes,
    #     embed_dims=_dim_, bev_h=bev_h_, bev_w=bev_w_,
    #     pc_range=point_cloud_range,
    #     map_code_size=2, map_code_weights=[1.0, 1.0, 1.0, 1.0],
    #     map_dir_interval=1, map_transform_method='minmax',
    #     map_gt_shift_pts_pattern='v2', num_reg_fcs=2,
    #     sync_cls_avg_factor=True, loss_weight=1.0,
    #     # 6 layers = VAD-Base; VAD-Tiny (SSR's base) uses 3.
    #     transformer_decoder=dict(
    #         type='MapDetectionTransformerDecoder',
    #         num_layers=6, return_intermediate=True,
    #         transformerlayers=dict(
    #             type='DetrTransformerDecoderLayer',
    #             attn_cfgs=[
    #                 dict(type='MultiheadAttention', embed_dims=_dim_,
    #                      num_heads=8, dropout=0.1),
    #                 dict(type='CustomMSDeformableAttention',
    #                      embed_dims=_dim_, num_levels=1),
    #             ],
    #             feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
    #             operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
    #                              'ffn', 'norm'))),
    #     bbox_coder=dict(
    #         type='MapNMSFreeCoder',
    #         post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
    #         pc_range=point_cloud_range, max_num=50,
    #         voxel_size=voxel_size, num_classes=map_num_classes),
    #     loss_map_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
    #                       alpha=0.25, loss_weight=2.0),
    #     loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
    #     loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
    #     loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
    #     loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005)),

    # ------------------------------------------------------------------ #
    # AUX 3: occupancy prediction (parallel, train-only, query-free)      #
    # Single-shot decoder: one conv trunk at full BEV resolution predicts #
    # all (n_future+1) x num_classes channels at once -- dense logits are  #
    # the KD interface. temporal_decoder=True switches to the UniAD-style #
    # recurrent decoder (heavier; capacity ablation / stronger teacher).  #
    # ------------------------------------------------------------------ #
    occ_head=dict(
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
            loss_weight=1.0)),

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
    dict(type='GenerateSSROccLabels',
         pc_range=point_cloud_range,
         bev_size=(bev_h_, bev_w_),
         n_future=occ_n_future,
         class_labels=occ_class_labels),
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
               'gt_attr_labels', 'gt_occ_seg', 'gt_occ_valid'])
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
             map_ann_file=data_root + 'nuscenes_map_anns_val.json',
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
              map_ann_file=data_root + 'nuscenes_map_anns_val.json',
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

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
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
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

find_unused_parameters = True

# Same training-time hooks as SSR_e2e.py so the comparison is apples-to-apples
# (the EMA weights are what SSR evaluates with).
custom_hooks = [
    dict(type='CustomSetEpochInfoHook'),
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]
