"""The plan=1.0 control arm: aux heads present, but blind to the BEV.

This is the comparison group for the 60-epoch run. It is the same experiment
with one variable moved: the auxiliary heads contribute nothing to the shared
BEV feature, matching SSR's original planning-only ownership of that feature.

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

This also provides a useful floor for distillation. ``_ScaleGrad`` zeroes only
the path back into ``bev_embed``; each head's own parameters still receive full
gradients from its loss. The aux heads therefore train as pure readers of a
planning-only BEV.

Everything else (60-epoch cosine, lr 2e-4, global batch 8 and evaluation every
6 epochs) is inherited unchanged, so the two runs are directly comparable.
"""
_base_ = ['./PARA_SSR_e2e_60ep.py']

# A target of exactly 0 is handled as OFF, not as "very small": GradBalancer
# pins the scale at 0.0 up front and never revisits it.
# occ_head=None is inherited from the 60ep config; only det and map remain.
model = dict(
    # _delete_=True: mmcv deep-merges, so without it the parent's shares
    # survive and this stops being a control.
    grad_balance=dict(
        target=dict(_delete_=True, plan=1.0, det=0.0, map=0.0)))

evaluation = dict(interval=6)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='SSRWandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_ssr_60ep_planonly',
                group='para',
                tags=['para-ssr', 'no-ffp', 'aux', 'global8', '60ep',
                      'control', 'plan1.0'],
                config=dict(
                    model='PARA-SSR', ffp=False, global_batch=8, epochs=60,
                    eval_interval=6, grad_target='plan1.0/det0/map0')),
            by_epoch=False,
            interval=100),
    ])
