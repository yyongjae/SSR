"""Sweep W1 -- nominal equal weights (PARA-Drive as written).

task_loss_weight = 1/1/1/1. Note this is *nominally* equal, not effectively:
measured on real batches, the gradient arriving at the shared BEV feature splits
det 89% / map 7.9% / occ 3.1% / plan 0.02%, i.e. detection alone decides the
representation. This is the reference point for the weight sweep, and the most
likely explanation for the degradation SSR's own authors report when they bolt
PARA-Drive-style aux tasks onto SSR (SSR paper, Appendix E, Table 6).
"""
_base_ = ['./PARA_SSR_e2e.py']

model = dict(task_loss_weight=dict(plan=1.0, det=1.0, map=1.0))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_w_equal',
                group='loss-weight-sweep',
                tags=['para-ssr', 'sweep'],
                config=dict(task_loss_weight='1/1/1/1', aux_grad_scale=1.0)),
            by_epoch=False,
            interval=100),
    ])
