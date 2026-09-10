"""Stage 1 RPED: joint privileged readout planner on BEVFusion + MapTRv2.

The two teacher caches are one KV memory.  Command-conditioned rank-16
queries read that memory; a waypoint decoder plans from the 16 evidence
tokens.  There is no per-cell adapter and no dense BEV imitation.

Gate: this privileged planner's val L2 / 1s box collision must beat
planning-only SSR before Stage 2 is worth running.
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

model = dict(
    _delete_=True,
    type='PrivilegedReadoutPlanner',
    feature_root=feature_root,
    teachers=dict(
        bevfusion=dict(cache_name='bevfusion'),
        maptrv2=dict(cache_name='maptrv2')),
    memory=dict(channels=256),
    readout=dict(
        embed_dims=256,
        num_queries=16,
        num_heads=8,
        num_layers=1,
        num_commands=3,
        ffn_channels=512,
        dropout=0.0),
    planner=dict(
        embed_dims=256,
        num_scenes=16,
        num_reg_fcs=2,
        fut_ts=6,
        ego_fut_mode=3,
        latent_decoder=_latent_decoder,
        way_decoder=_way_decoder,
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0)))
