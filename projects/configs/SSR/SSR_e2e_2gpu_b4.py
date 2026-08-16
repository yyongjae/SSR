"""SSR paper/global-batch setting on two GPUs.

The released recipe uses 8 GPUs x 1 sample/GPU.  This configuration keeps the
same global batch (8), optimizer step count, LR schedule and EMA cadence while
factoring it as 2 GPUs x 4 samples/GPU.  Validation/test intentionally remain
at one sample/GPU because streaming inference maintains one scene state.

Do not combine this config with ``--autoscale-lr``: the global batch is already
unchanged, so the original 5e-5 learning rate is the correct value.
"""

_base_ = ['./SSR_e2e.py']

data = dict(samples_per_gpu=4)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='ssr_baseline_2gpu_b4',
                group='baseline',
                tags=['ssr', 'ffp', '2gpu', 'batch4', 'global8'],
                config=dict(
                    model='SSR', ffp=True, gpus=2, batch_per_gpu=4,
                    global_batch=8)),
            by_epoch=False,
            interval=100),
    ])
