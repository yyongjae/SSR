"""PARA-SSR on two GPUs at the paper's global batch of 8.

Same relationship to ``PARA_SSR_e2e.py`` that ``SSR_noffp_e2e_2gpu_b4.py`` has
to ``SSR_noffp_e2e.py``: 8 GPUs x 1 sample re-factored into 2 GPUs x 4 samples.
Global batch, optimizer step count, LR schedule and EMA cadence are unchanged,
so results stay comparable to the 8-GPU PARA-SSR run.

Requires the multibatch fixes (ported from SSR-orig commit 8fe89d5 plus the
matching change in ``para_ssr_head.py`` and per-sample occupancy frame masks).
Before those, ``samples_per_gpu > 1`` silently applied sample 0's navigation
command, camera visibility and image shape to the whole local batch.

Do not combine with ``--autoscale-lr``: the global batch is already 8, so 5e-5
is the correct learning rate.
"""
_base_ = ['./PARA_SSR_e2e.py']

# 2 GPUs x 4 samples = global batch 8.
#
# workers_per_gpu is raised from 4 to 16 so the total worker count (2 x 16 = 32)
# matches what the 8-GPU runs had (8 x 4). This is a dataloader-parallelism knob
# only; it does not change the experiment. Without it the run is data-bound:
# measured 4.83 s/iter of which 3.12 s was data_time (65% idle GPU) against
# 1.22 s / 0.12 s on 8 GPUs. Each sample decodes 3 frames x 6 cameras of JPEG,
# so halving the GPU count does not halve the CPU demand. The box has 64 cores.
data = dict(samples_per_gpu=4, workers_per_gpu=16)

# --------------------------------------------------------------------- #
# Give every task a fixed slice of the shared-BEV gradient.              #
# --------------------------------------------------------------------- #
# The earlier approach was `aux_grad_scale`: one constant on all three aux
# heads. It sets a knob and lets the resulting share fall where it may, and
# what it actually produced was not what its name suggested -- measured on a
# real batch at 0.01, the planner received 4.1% of the BEV gradient.
#
# `grad_balance` specifies the share and solves for the knob. Every `interval`
# iterations it measures ||dL_k/d bev_embed||, backs out the valve currently in
# effect, and re-solves. Planning is the numeraire and is never scaled.
#
# Untouched, the split is roughly
#     plan 0.07%   det 57%   map 27%   occ 15%
# and the valves that produce the target below come out around
#     det 5.5e-4   map 1.2e-3   occ 2.1e-3
# -- a factor of four apart from each other, which is why no single shared
# constant can balance them (tools/verify_grad_balance.py measures both).
#
# Three things to keep in mind while reading the resulting run:
#
#  * plan=0.4 is a 6x-500x amplification of planning's natural share. This is
#    NOT the SSR baseline regime. If planning improves, "we raised planning's
#    gradient" is as good an explanation as "the aux heads helped".
#  * this balances gradient MAGNITUDE. Two tasks pulling in opposite directions
#    cancel in the real update and both still read as large.
#  * det and motion share one valve -- one forward through det_motion_head
#    means one BEV input. gshare/motion is reported separately for information;
#    it OVERLAPS det and is excluded from the shares that sum to 1.
#
# Watch gscale/* in the log. If gscale/det pins to the lower clamp the target
# is unreachable and the clamp, not the controller, is setting the share.
model = dict(
    grad_balance=dict(
        target=dict(plan=0.4, det=0.2, map=0.2, occ=0.2),
        interval=200,        # ratios drift on an epoch timescale, not an iter one
        momentum=0.9,        # ~10 updates = 2000 iters ~ 0.6 epoch time constant
        clamp=(1e-5, 1.0),   # 55x of headroom below the measured det valve
        warmup_iters=500))   # matches the LR warm-up; gradients before it lie


# --------------------------------------------------------------------- #
# VAD's learning rate.                                                   #
# --------------------------------------------------------------------- #
# Everything else in the optimiser was already identical to VAD_base_e2e.py /
# VAD_tiny_e2e.py -- AdamW, weight_decay 0.01, img_backbone lr_mult 0.1,
# grad_clip max_norm 35 norm_type 2, CosineAnnealing, warmup linear 500 iters
# at ratio 1/3, min_lr_ratio 1e-3. The one difference was the base LR: SSR uses
# 5e-5, VAD uses 2e-4.
#
# The auxiliary heads here ARE VAD's, and VAD converges them in 60 epochs at
# 2e-4. Running them at a quarter of that rate would make "the aux tasks had
# not converged by epoch 60" ambiguous between the architecture and the LR,
# which is exactly the question the long run is meant to answer.
#
# CONSEQUENCE: planning is no longer on SSR's schedule. The reproduced
# L2 MAX avg 0.7526 was measured at 5e-5, so it is not a valid reference for
# this run. A plan-only baseline at 2e-4 on the same 60-epoch cosine is needed
# before any statement of the form "aux supervision costs/helps planning".
#
# mmcv deep-merges this into the parent dict, so type / paramwise_cfg /
# weight_decay all carry over; only lr changes. img_backbone therefore lands at
# 2e-5, which is also what VAD does.
optimizer = dict(lr=2e-4)


log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_ssr_2gpu_b4',
                group='para',
                tags=['para-ssr', 'no-ffp', 'aux', '2gpu', 'batch4',
                      'global8'],
                config=dict(
                    model='PARA-SSR', ffp=False, gpus=2, batch_per_gpu=4,
                    global_batch=8)),
            by_epoch=False,
            interval=100),
    ])
