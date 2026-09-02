"""Stage 1: planning-specialise small adapters on frozen teacher BEV caches.

BEVDepth and HDMapNet are not instantiated here.  Their frozen, task-specific
features are generated once by ``tools/distill/cache_teacher_bev.py`` and keyed
by nuScenes sample token.  Therefore every parameter in this model belongs to
one of exactly four trainable pieces:

    BEVDepth adapter + SSR planning head
    HDMapNet adapter  + SSR planning head

The two branches are independent during training.  Evaluation averages their
trajectory offsets by default and also retains each branch prediction.
"""
_base_ = ['./PARA_SSR_e2e.py']

feature_root = \
    '/data2/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache'
total_epochs = 6

_latent_decoder = dict(
    type='CustomTransformerDecoder',
    num_layers=3,
    return_intermediate=False,
    transformerlayers=dict(
        type='BaseTransformerLayer',
        attn_cfgs=[dict(
            type='MultiheadAttention', embed_dims=256, num_heads=8)],
        feedforward_channels=512,
        operation_order=('self_attn', 'norm', 'ffn', 'norm')))

_way_decoder = dict(
    type='CustomTransformerDecoder',
    num_layers=1,
    return_intermediate=False,
    transformerlayers=dict(
        type='BaseTransformerLayer',
        attn_cfgs=[dict(
            type='MultiheadAttention', embed_dims=256, num_heads=8)],
        feedforward_channels=512,
        operation_order=('cross_attn', 'norm', 'ffn', 'norm')))

_planner = dict(
    input_size=(25, 25),
    bev_h=100,
    bev_w=100,
    embed_dims=256,
    num_scenes=16,
    num_reg_fcs=2,
    fut_ts=6,
    ego_fut_mode=3,
    num_navi_cmd=3,
    positional_encoding=dict(
        type='LearnedPositionalEncoding', num_feats=128,
        row_num_embed=100, col_num_embed=100),
    latent_decoder=_latent_decoder,
    way_decoder=_way_decoder,
    loss_plan_reg=dict(type='L1Loss', loss_weight=1.0))

_adapter = dict(channels=256, hidden_channels=256, dropout=0.0)

model = dict(
    _delete_=True,
    type='CachedTeacherAdapterPlanner',
    feature_root=feature_root,
    test_teacher='ensemble',
    teachers=dict(
        bevdepth=dict(
            cache_name='bevdepth',
            adapter=_adapter,
            planner=_planner,
            loss_weight=1.0),
        hdmapnet=dict(
            cache_name='hdmapnet',
            adapter=_adapter,
            planner=_planner,
            loss_weight=1.0)))

# Only the current token is needed; teacher temporal context was already
# folded into BEVDepth's cached two-key-frame feature.
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(queue_length=1, use_future_frame=False))

optimizer = dict(
    _delete_=True, type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing', warmup='linear', warmup_iters=500,
    warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)
find_unused_parameters = False

# EMA is intentionally absent: stage 2 loads the exact raw adapter state that
# was evaluated and checkpointed, rather than a second hidden copy.  The
# inherited epoch hook is also ParaSSR-specific (it calls model.set_epoch), so
# this standalone cached-feature model must not install it.
custom_hooks = []

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
