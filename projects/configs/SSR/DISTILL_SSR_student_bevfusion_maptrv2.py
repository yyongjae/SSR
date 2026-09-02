"""Stage 2: planning-only SSR student distilled from BEVFusion + MapTRv2.

Same ``ssr_noffp_2gpu_b4`` regime as ``DISTILL_SSR_student.py``.  Frozen
stage-1 adapters convert both the cached teacher BEV and the student BEV into
the same planning feature space; only the student image/BEV/planning modules
train.
"""
_base_ = ['./DISTILL_SSR_student.py']

feature_root = \
    '/data3/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache'
adapter_checkpoints = dict(
    _delete_=True,
    bevfusion=(
        '/data2/byounggun/rideflux/pretrained_checkpoints/'
        'planning_distill_checkpoints/teacher_bevfusion/epoch_6.pth'),
    maptrv2=(
        '/data2/byounggun/rideflux/pretrained_checkpoints/'
        'planning_distill_checkpoints/teacher_maptrv2/epoch_6.pth'))

model = dict(
    distill=dict(
        feature_root=feature_root,
        adapter_checkpoint=adapter_checkpoints,
        student_bev_size=(100, 100),
        cache_size=(100, 100),
        branches=dict(
            _delete_=True,
            bevfusion=dict(
                cache_name='bevfusion',
                adapter=dict(
                    channels=256, hidden_channels=256, dropout=0.0),
                adapter_prefix='branches.bevfusion.adapter.',
                loss_weight=1.0),
            maptrv2=dict(
                cache_name='maptrv2',
                adapter=dict(
                    channels=256, hidden_channels=256, dropout=0.0),
                adapter_prefix='branches.maptrv2.adapter.',
                loss_weight=1.0))))
