"""Stage 2 of 2: switch planning on, starting from the stage-1 weights.

    ./run.sh stage1 0,1
    ./run.sh stage2 0,1

``load_from`` (not ``resume_from``): the weights carry over, the optimiser
state, epoch counter and LR schedule all restart. That is what VAD does
(``load_from = 'ckpts/VAD_base_stage_1.pth'``) and what HiP-AD does
(``load_from = './work_dirs/hipad_nusc_stage1/latest.pth'``), and it is what
makes stage 2 a fresh 12-epoch cosine from 2e-4 rather than the tail of a
48-epoch one.

WHERE THIS DEPARTS FROM VAD, deliberately.

VAD's stage 2 down-weights the perception losses 2.5x:

    loss             stage1   stage2
    loss_cls            2.0      0.8     2.5x down
    loss_bbox          0.25      0.1     2.5x down
    loss_map_cls        2.0      0.8     2.5x down
    loss_map_pts        1.0      0.4     2.5x down
    loss_traj           0.2      0.2     unchanged
    loss_map_dir      0.005    0.005     unchanged

It has to: with planning suddenly added to a representation that spent 48
epochs being shaped by perception, something has to stop perception from
continuing to dominate, and a hand-tuned constant on the losses is the only
lever VAD has.

We have grad_balance, which does that job directly and adaptively -- it
measures each task's gradient at bev_embed and solves for the scales that put
the shares on target, whatever the loss magnitudes happen to be. Stacking VAD's
2.5x on top would not change the BEV shares at all (the balancer would simply
solve for larger scales to compensate); it would only slow the perception HEADS
down. For a project whose point is to end up with good distillation teachers,
slowing the teachers is the wrong side of that trade.

So the perception loss weights stay at 1.0 here and grad_balance owns the
shared representation. If the VAD-faithful arm is wanted, it is a small config:
put loss_cls=0.8, loss_bbox=0.1, loss_map_cls=0.8, loss_map_pts=0.4 in the
det/map heads (and the matching assigner costs -- VAD changes those too, and a
matcher optimising a different objective than the loss is a real bug, not a
cosmetic mismatch).

WHAT TO COMPARE THIS AGAINST -- and how not to. 48 + 12 = 60 epochs is the same
wall-clock budget as the single-stage PARA_SSR_e2e_60ep.py, but it is
NOT "the same run with one variable changed". See the stage-1 docstring: the LR
schedule, the optimiser state, the EMA, the RNG stream and above all the number
of iterations the planner and motion heads receive (42,192 against 210,960) all
differ. Read the pair as two recipes, not as an ablation of staging.
"""
_base_ = ['./PARA_SSR_e2e_60ep.py']

total_epochs = 12
runner = dict(max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

# Weights only. Optimiser state, epoch and LR schedule restart -- see above.
# Point this at wherever stage 1 was written; the launcher passes --work-dir.
load_from = 'work_dirs/para_ssr_stage1/latest.pth'

model = dict(
    task_loss_weight=dict(plan=1.0, det=1.0, motion=1.0, map=1.0),
    # Balance from iteration 200, not 600.
    #
    # warmup_iters exists because the gradient ratios during LR warm-up are not
    # the ratios the run settles into -- measured at iteration 10 they were two
    # orders of magnitude away from the epoch-1 value. That reasoning holds for
    # a run starting from scratch. It does not hold here: stage 2 starts from a
    # 48-epoch checkpoint whose BEV encoder and perception heads are already
    # converged, so the first measurement is representative immediately.
    #
    # It also matters more here. 600 unbalanced iterations out of 210,960 is
    # noise in the monolithic run; out of stage 2's 42,192 it is 1.4%, and they
    # are the iterations where the planner -- untrained until now -- is furthest
    # from the perception heads and most easily drowned.
    #
    # The monolithic config keeps warmup_iters=500 deliberately: there the
    # warm-up gradients really are unrepresentative.
    grad_balance=dict(warmup_iters=0))

# 12 / 6 = 6,12. 10 would evaluate epoch 10 only and never the final model.
evaluation = dict(interval=6)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='SSRWandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_ssr_stage2_all',
                group='para-staged',
                tags=['para-ssr', 'no-ffp', 'aux', 'global8', 'stage2',
                      'all-tasks'],
                config=dict(
                    model='PARA-SSR', ffp=False, global_batch=8, epochs=12,
                    eval_interval=6,
                    stage=2, tasks='plan+det+motion+map')),
            by_epoch=False,
            interval=100),
    ])
