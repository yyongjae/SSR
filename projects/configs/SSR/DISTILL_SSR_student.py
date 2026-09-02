"""Stage 2: planning-only SSR student with frozen-adapter BEV distillation.

This is the ``ssr_noffp_2gpu_b4`` regime: the student has no detection,
mapping, motion, occupancy or FFP head.  Its ordinary SSR planning loss trains
the image backbone, BEV encoder and planning head.  Two extra feature losses
pass through frozen copies of the stage-1 BEVDepth/HDMapNet planning adapters,
so they also update the student BEV encoder without adding an inference-time
module.
"""
_base_ = ['./PARA_SSR_e2e_12ep.py']

feature_root = \
    '/data2/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache'
adapter_checkpoints = dict(
    bevdepth=(
        '/data2/byounggun/rideflux/pretrained_checkpoints/'
        'planning_distill_checkpoints/teacher_bevdepth/epoch_6.pth'),
    hdmapnet=(
        '/data2/byounggun/rideflux/pretrained_checkpoints/'
        'planning_distill_checkpoints/teacher_hdmapnet/epoch_6.pth'))

model = dict(
    # Planning only.  ``None`` replaces the inherited PARA-Drive aux heads.
    det_motion_head=None,
    map_head=None,
    occ_head=None,
    test_aux_heads=False,
    task_loss_weight=dict(plan=1.0, det=0.0, motion=0.0, map=0.0, occ=0.0),
    aux_grad_scale=1.0,
    grad_balance=None,
    aux_metric_log_interval=0,
    # Includes plan and distill in gnorm/gshare so the actual pressure on the
    # shared student BEV remains observable.
    grad_norm_log_interval=200,
    distill=dict(
        feature_root=feature_root,
        adapter_checkpoint=adapter_checkpoints,
        student_bev_size=(100, 100),
        cache_size=(25, 25),
        branches=dict(
            bevdepth=dict(
                cache_name='bevdepth',
                adapter=dict(
                    channels=256, hidden_channels=256, dropout=0.0),
                adapter_prefix='branches.bevdepth.adapter.',
                loss_weight=1.0),
            hdmapnet=dict(
                cache_name='hdmapnet',
                adapter=dict(
                    channels=256, hidden_channels=256, dropout=0.0),
                adapter_prefix='branches.hdmapnet.adapter.',
                loss_weight=1.0))))

# Match the requested SSR-noFFP baseline schedule rather than PARA-SSR's 2e-4
# auxiliary-head schedule.
optimizer = dict(lr=5e-5)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
