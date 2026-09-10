"""Stage 2 RPED: planning-only SSR, rank-N evidence distillation.

The student is vanilla SSR-noFFP at inference.  Training adds a frozen
privileged readout: student TokenLearner tokens V match R(Q, T) from the
joint teacher memory.  There is no dense BEV MSE.

Default query_source='privileged' keeps student queries as questions and
distills answers.  Same-question ablation sets query_source='student'.
"""
_base_ = ['./DISTILL_SSR_student.py']

feature_root = \
    '/data3/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache'
rped_checkpoint = (
    '/data2/byounggun/rideflux/pretrained_checkpoints/'
    'planning_distill_checkpoints/rped_teacher_bevfusion_maptrv2/epoch_6.pth')

model = dict(
    distill=dict(
        _delete_=True,
        type='PrivilegedEvidenceDistillation',
        feature_root=feature_root,
        readout_checkpoint=rped_checkpoint,
        student_bev_size=(100, 100),
        loss_weight=1.0,
        command_triplet_weight=0.0,
        shuffle_memory=False,
        query_source='privileged',
        teachers=dict(
            bevfusion=dict(cache_name='bevfusion'),
            maptrv2=dict(cache_name='maptrv2')),
        readout=dict(
            embed_dims=256,
            num_queries=16,
            num_heads=8,
            num_layers=1,
            num_commands=3,
            ffn_channels=512,
            dropout=0.0)))
