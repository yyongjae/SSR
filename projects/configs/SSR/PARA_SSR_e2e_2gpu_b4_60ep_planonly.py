"""The plan=1.0 control arm: aux heads present, but blind to the BEV.

This is the comparison group for the 60-epoch run. It is the same experiment
with one variable moved -- the auxiliary heads contribute nothing to the shared
BEV feature -- which is exactly SSR's original situation, where planning owned
100% of the BEV gradient because no other head existed.

Why this rather than an SSR-orig run:

    SSR-orig                    this file
    ------------------------    ---------------------------------------
    no aux heads at all         aux heads present and training
    41.4M -> 31.4M parameters   identical parameter count
    optimiser sees 31.4M        identical optimiser state
    clip norm ~= planner's      identical clip norm (pnorm/aux dominates
                                it either way -- measured 24.8 vs 3.3)
    different code path         identical code path

Only the aux->BEV gradient differs. An SSR-orig comparison would move four
things at once, and the clip norm one is not small: global gradient clipping is
driven almost entirely by the aux heads' own parameter gradients, which are
present here and absent there.

And it answers a question of its own, for the distillation work. `_ScaleGrad`
zeroes only the path back into `bev_embed`; each head's own parameters still
receive full gradients from its own loss. So the aux heads here train as pure
READERS of a planning-only BEV. Whatever map mAP / detection mAP / occupancy IoU
they reach is the quality of teacher obtainable WITHOUT letting the teacher
reshape the representation -- the floor against which co-training has to justify
itself.

Everything else (60-epoch cosine, lr 2e-4, batch 8, eval every 10) is inherited
unchanged, so the two runs are directly comparable.
"""
_base_ = ['./PARA_SSR_e2e_2gpu_b4_60ep.py']

# A target of exactly 0 is handled as OFF, not as "very small": GradBalancer
# pins the scale at 0.0 up front and never revisits it, so it does not drift up
# to the clamp floor.
# occ_head=None is inherited from the 60ep config; only det and map remain.
model = dict(
    # _delete_=True: mmcv deep-merges, so without it the parent's shares
    # survive and this stops being a control.
    grad_balance=dict(
        target=dict(_delete_=True, plan=1.0, det=0.0, map=0.0)))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_ssr_2gpu_b4_60ep_planonly',
                group='para',
                tags=['para-ssr', 'no-ffp', 'aux', '2gpu', 'batch4',
                      'global8', '60ep', 'control', 'plan1.0'],
                config=dict(
                    model='PARA-SSR', ffp=False, gpus=2, batch_per_gpu=4,
                    global_batch=8, epochs=60, eval_interval=10,
                    grad_target='plan1.0/det0/map0')),
            by_epoch=False,
            interval=100),
    ])
