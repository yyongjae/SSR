"""Stage 1 parent: BEVFusion + MapTRv2 adapters on frozen npz BEV caches.

The teachers themselves are not instantiated.  Offline caches arrive at
``/data3/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache/{bevfusion,maptrv2}``
as 100x100 fp16 xy-major tokens.  Loading converts them to SSR CHW maps at
native 100x100; the planner resize is therefore identity.

Override ``model.feature_root`` with cfg-options if the rsync destination
moves.
"""
_base_ = ['./DISTILL_teacher_adapters.py']

feature_root = \
    '/data3/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache'

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
    input_size=(100, 100),
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
    feature_root=feature_root,
    test_teacher='ensemble',
    teachers=dict(
        _delete_=True,
        bevfusion=dict(
            cache_name='bevfusion',
            adapter=_adapter,
            planner=_planner,
            loss_weight=1.0),
        maptrv2=dict(
            cache_name='maptrv2',
            adapter=_adapter,
            planner=_planner,
            loss_weight=1.0)))
