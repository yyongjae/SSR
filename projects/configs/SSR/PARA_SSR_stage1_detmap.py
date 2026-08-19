"""Stage 1 of 2: shape the BEV with everything except planning.

Follows VAD's staging (VAD_base_stage_1.py / _stage_2.py), which is the closest
reference for this model -- the auxiliary heads here ARE VAD's, at VAD's loss
weights, on VAD's schedule. HiP-AD stages too, but the other way round: its
stage 1 already trains planning and defers only motion
(``task_select = ["det", "map", "plan", "ego"]``), which is not what is wanted
here.

    stage 1   48 epochs   plan 0.0   det 1.0   motion 1.0   map 1.0
    stage 2   12 epochs   plan 1.0   det 1.0   motion 1.0   map 1.0  + grad_balance
    -----------------------------------------------------------------
    total     60 epochs, the same wall-clock budget as the single-stage
              PARA_SSR_e2e_2gpu_b4_60ep.py.

THAT IS A BUDGET MATCH, NOT A CONTROLLED EXPERIMENT. An earlier version of this
docstring called the pair "a controlled comparison of staging itself"; that was
wrong. Staged and monolithic are two RECIPES, and at least five things differ:

  * LR schedule. Two cosines, not one. Stage 1 anneals to 2.0e-07 by epoch 48
    and stage 2 re-ignites at 2.0e-04, where the monolithic run is passing
    through 1.9e-05 and still descending.
  * Optimiser state. load_from restores weights only, so Adam's first and
    second moments start from zero at epoch 49.
  * EMA. MEGVIIEMAHook re-initialises in stage 2 unless its own `resume` is
    pointed at the stage-1 EMA checkpoint, which it is not.
  * RNG. Stage 2 is a new process re-seeded from 0, so the data order and the
    augmentation stream are not the continuation of stage 1's.
  * How long the planner and motion actually train. Monolithic gives both
    210,960 iterations; staged gives 42,192 -- one fifth.

The last one is not a nuisance variable, it is most of the difference. Any gap
in the final planning number is at least as attributable to "the planner saw
five times fewer updates" as to "perception was pre-trained".

What the pair IS good for: asking whether perception-first reaches a better
map/detection mAP for the same total compute, and what that costs planning.
Attributing a difference to staging alone would need the confounds above
controlled one at a time.

VAD expresses "planning off" by zeroing the four planner loss weights. We have
``task_loss_weight``, which is the same thing one level up, so plan=0.0 here.

grad_balance is OFF in this stage, and that is not an oversight. The balancer
holds each task's share of the shared-BEV gradient against planning as the
numeraire; with no planning gradient there is nothing to hold shares against.
(GradBalancer would detect this itself -- its zero-numeraire guard leaves the
scales at 1.0 -- but an explicit None says so at the config level.) Letting
detection and mapping shape the BEV freely is the entire point of stage 1.

WHAT THIS COSTS THE PLANNER, measured rather than assumed: with plan=0.0 the
planner-only modules (navi_se, tokenlearner, latent_decoder, way_decoder,
ego_fut_decoder -- 2.42M of the 38.1M parameters) receive a zero gradient, but
AdamW's decoupled weight decay still applies. Over 48 epochs x 3516 iterations
at a cosine mean LR of ~1e-4 and weight_decay 0.01 the decay factor is
exp(-1e-4 * 0.01 * 168768) = 0.845, so those weights end stage 1 at roughly
85% of their initial norm. VAD has exactly the same behaviour and stage 2
retrains the planner from there regardless; freezing them instead
(requires_grad=False) would avoid it, at the cost of diverging from the
reference. Left as VAD has it.

The planning metrics reported at the epoch-10 validations are therefore
meaningless in this stage -- the planner has not been trained. Read map mAP and
detection mAP; ignore plan_L2.

USEFUL SIDE EFFECT: this stage answers "how long do the auxiliary tasks take to
converge" in isolation, with no planning gradient competing for the
representation. That is the cleanest possible version of that measurement.
"""
_base_ = ['./PARA_SSR_e2e_2gpu_b4_60ep.py']

total_epochs = 48
runner = dict(max_epochs=total_epochs)
# Both must be restated: the parent resolved them against its own
# total_epochs=60 at parse time, and re-declaring the variable does not
# retroactively change a dict that already used it.
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

model = dict(
    # plan=0.0 is "planning off". Every other weight is unchanged, so the
    # perception losses sit at exactly VAD stage 1's values.
    # Perception AND motion; only planning is off. plan=0.0 means OFF, not
    # "times zero": _scale drops a zero-weighted task from the loss dict
    # entirely, so the planner gets no gradient and AdamW cannot decay it.
    task_loss_weight=dict(plan=0.0, det=1.0, motion=1.0, map=1.0),
    # None, not a smaller target: see the docstring. mmcv replaces rather than
    # merges when the incoming value is not a dict, so this clears the parent's
    # grad_balance outright.
    grad_balance=None,
    aux_grad_scale=1.0)

# 48 / 8 = 8,16,24,32,40,48 -- six points and the last one IS epoch 48.
# 10 would stop at 40 and never score the model stage 2 starts from.
evaluation = dict(interval=6)

# The planner is untrained here, so its validation numbers are noise. Watch
# only what this stage trains, or the monitor cries wolf every six epochs.
custom_hooks = [
    dict(type='TrainingAnomalyHook', priority='LOW',
         eval_watch=['NuscMap_chamfer/mAP', 'pts_bbox_NuScenes/mAP',
                     'pts_bbox_NuScenes/NDS',
                     'motion_EPA/car', 'motion_EPA/pedestrian']),
    dict(type='CustomSetEpochInfoHook'),
    dict(type='MEGVIIEMAHook', init_updates=10560, priority='NORMAL'),
]

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_ssr_stage1_detmap',
                group='para-staged',
                tags=['para-ssr', 'no-ffp', 'aux', '2gpu', 'batch4',
                      'global8', 'stage1', 'no-planning'],
                config=dict(
                    model='PARA-SSR', ffp=False, gpus=2, batch_per_gpu=4,
                    global_batch=8, epochs=48, eval_interval=6,
                    stage=1, tasks='det+motion+map')),
            by_epoch=False,
            interval=100),
    ])
