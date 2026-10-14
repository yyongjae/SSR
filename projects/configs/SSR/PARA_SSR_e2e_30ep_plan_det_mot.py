"""Train planning, detection, and motion for 30 epochs; disable map loss.

The map head remains instantiated so this arm keeps the same model structure,
parameter count, optimiser state, and inference code path as the full
PARA-SSR experiment. It is disabled in both places that matter:

* ``task_loss_weight.map=0`` removes every map loss from optimisation, so the
  map-head parameters are not trained (and AdamW does not decay them).
* ``grad_balance.target.map=0`` follows the plan-only control and pins the
  map-to-BEV gradient valve to zero from the first iteration.

Detection and motion are separate loss groups but share one BEV valve because
they are produced by the same head. Planning keeps its original 0.4 target;
the disabled map share is assigned to the det+motion valve.
"""
_base_ = ['./PARA_SSR_e2e_12ep.py']

total_epochs = 30
runner = dict(max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

model = dict(
    test_aux_heads=True,
    task_loss_weight=dict(
        _delete_=True, plan=1.0, det=1.0, motion=1.0, map=0.0),
    # _delete_=True prevents the inherited map/occ targets from surviving
    # mmcv's recursive config merge.
    grad_balance=dict(
        target=dict(_delete_=True, plan=0.4, det=0.6, map=0.0)))

# Five validation points, including the final epoch.
evaluation = dict(interval=6)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='SSRWandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_ssr_30ep_plan_det_mot',
                group='para',
                tags=['para-ssr', 'no-ffp', 'aux', 'global8', '30ep',
                      'plan-det-motion', 'map-off'],
                config=dict(
                    model='PARA-SSR', ffp=False, global_batch=8, epochs=30,
                    eval_interval=6,
                    task_loss_weight='plan1/det1/motion1/map0',
                    grad_target='plan0.4/det0.6/map0')),
            by_epoch=False,
            interval=100),
    ])
