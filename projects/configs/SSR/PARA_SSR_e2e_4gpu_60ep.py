"""The 60-epoch run on four GPUs. Same experiment, half the wall time.

4 x 2 = the same global batch of 8, so the optimiser step count, the LR
schedule, the EMA cadence and every iteration-based interval are unchanged and
the results are directly comparable to the 2-GPU config. Only throughput moves:
roughly 7 days to roughly 3.5.

Do NOT reach for 4 x 4. That is a global batch of 16, and then:

* the LR is no longer VAD's validated 2e-4 -- linear scaling asks for 4e-4,
  which nothing here or in VAD has run;
* iterations per epoch halve (3516 -> 1758), so warmup_iters=500 goes from 14%
  of an epoch to 28%, and grad_balance/grad_norm/aux_metric intervals (all in
  iterations) fire twice as often per epoch;
* the run stops being comparable to the paper, to VAD, and to the plan=1.0
  control unless that is rebuilt at batch 16 too.

Do NOT pass --autoscale-lr either, at any GPU count. tools/train.py applies
``lr * len(gpu_ids) / 8``, which ignores samples_per_gpu entirely: on four GPUs
it would hand back 1e-4, halving the learning rate at the moment the batch
doubled.

workers_per_gpu drops 16 -> 8 so the total stays at 32. The box has 64 cores and
three other people on it; 4 x 16 would take all of them. 32 workers already have
about 4x the headroom the training loop consumes.
"""
_base_ = ['./PARA_SSR_e2e_2gpu_b4_60ep.py']

data = dict(samples_per_gpu=2, workers_per_gpu=8)

evaluation = dict(interval=6)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_ssr_4gpu_60ep',
                group='para',
                tags=['para-ssr', 'no-ffp', 'aux', '4gpu', 'batch2',
                      'global8', '60ep', 'aux-convergence'],
                config=dict(
                    model='PARA-SSR', ffp=False, gpus=4, batch_per_gpu=2,
                    global_batch=8, epochs=60, eval_interval=6)),
            by_epoch=False,
            interval=100),
    ])
