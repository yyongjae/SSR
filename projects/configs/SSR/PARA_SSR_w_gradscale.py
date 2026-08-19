"""Sweep W3 -- scale the aux gradient into the BEV instead of the aux losses.

Down-weighting an aux loss does two things at once: it reduces how much the task
reshapes the shared BEV feature, and it slows the aux head itself. Only the
first is wanted here -- once the aux heads are used as distillation teachers, a
weak teacher is a useless one. `aux_grad_scale` decouples them: the aux losses
keep weight 1.0, so the heads converge normally, but the gradient they push back
into `bev_embed` is scaled.

Arms of the scale sweep, all against the unscaled 8-GPU PARA_SSR_e2e.py:

    PARA_SSR_w_equal.py         1.0    no scaling (the control)
    this file                   0.02
    PARA_SSR_e2e_2gpu_b4.py     0.01   the working 2-GPU configuration

None of these is a published value; 0.01 came from one warm-up measurement of
gshare and needs its own ablation. Compare against W2 to separate "aux
supervision hurts the BEV" from "aux supervision is simply too strong".
"""
_base_ = ['./PARA_SSR_e2e.py']

model = dict(
    task_loss_weight=dict(plan=1.0, det=1.0, map=1.0, occ=1.0),
    aux_grad_scale=0.02)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_w_gradscale',
                group='loss-weight-sweep',
                tags=['para-ssr', 'sweep'],
                config=dict(task_loss_weight='1/1/1/1', aux_grad_scale=0.02)),
            by_epoch=False,
            interval=100),
    ])
