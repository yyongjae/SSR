"""PARA-SSR long run: train until the auxiliary tasks converge.

``PARA_SSR_e2e_2gpu_b4.py`` stays the 12-epoch definition so results from it
remain comparable to the 8-GPU runs; this file is the separate long-run
experiment. Everything except the schedule and the validation cadence is
inherited unchanged.

The base resolved runner / checkpoint_config / evaluation against its own
total_epochs=12, so re-declaring the variable is not enough -- each dict has to
be overridden explicitly. mmcv deep-merges them, so keys not named here (runner
type, evaluation pipeline) still come from the base.

This is NOT "the 12-epoch run, continued", for two separate reasons.

CosineAnnealing is by_epoch, so at epoch 12 this run sits at 4.5e-5 of its base
LR where the 12-epoch run had already annealed to ~0. And the base LR itself is
now VAD's 2e-4 rather than SSR's 5e-5 (see PARA_SSR_e2e_2gpu_b4.py), because
the auxiliary heads are VAD's and VAD converges them in 60 epochs at that rate.

So the reproduced SSR-orig result -- L2 MAX avg 0.7526, measured at 5e-5 over
12 epochs -- is not a reference for anything here. Any claim about what the
auxiliary heads cost or give the planner needs a plan-only baseline on THIS
schedule: 60-epoch cosine, lr 2e-4.

Run with train_para_2gpu_b4_60ep.sh, which deliberately omits --no-validate.
Build the map GT cache first (~30 min, CPU only):

    python tools/build_map_gt_cache.py projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py

Then smoke-test the validation path before committing five days to it:

    tools/verify_dist_eval.sh 8 projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py 0,1
"""
_base_ = ['./PARA_SSR_e2e_2gpu_b4.py']

total_epochs = 60
runner = dict(max_epochs=total_epochs)

# CosineAnnealing is by_epoch, so the schedule stretches over 60 epochs on its
# own: the LR now decays far more slowly, which is the point of the longer run.
# Warmup stays at 500 iterations.

# Disk. Measured: 469 MB raw + 157 MB EMA = 626 MB per epoch, so all 60 of both
# is 37.6 GB. work_dirs is a symlink to /data1/yong/SSR/work_dirs and /data1 has
# 1.8 TB free -- do NOT size this against `df` on the repo path, which reports
# the root partition (53 GB) and has nothing to do with where checkpoints land.
#
# Keep every epoch: 37.6 GB is nothing here, and it means any epoch can be
# re-evaluated after the fact once the aux convergence point is known.
# (max_keep_ckpts prunes the RAW checkpoints only; MEGVIIEMAHook saves from its
# own after_train_epoch and ignores checkpoint_config entirely.)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

# Validate every 10 epochs -- 6 points over the run. The knee is located from
# the training-time metrics instead (map_pts_err_m, map_spread_m, map_cls_acc,
# map_conf_pos/neg, logged every 200 iterations = ~18 points per epoch); these
# validation points anchor that continuous curve in real mAP rather than being
# the primary signal. Each costs ~15-25 min.
#
# The aux heads have to be ON during evaluation -- otherwise only planning
# metrics come back and map mAP / detection mAP / occupancy IoU stay invisible.
# Costs ~15-25 min per evaluation (6019 samples + the nuScenes detection
# evaluator).
#
# Two things this in-training evaluation is NOT:
#
#  * it does not score the EMA weights. EvalHook sees the raw model (eval_model
#    is never passed to custom_train_model); EMA is dumped separately as
#    epoch_N_ema.pth. Run tools/test.py on those for final numbers.
#  * it does not reproduce a single-GPU evaluation exactly. The test sampler
#    hands rank 1 a contiguous block starting mid-scene, so that rank's first
#    sample loses its prev_bev. That is 1 sample in 6019 here, but VAD's
#    docs/train_eval.md asks for 1 GPU at evaluation time for this reason.
#    Treat these as a trend; produce final numbers with 1 GPU on the EMA
#    checkpoint.
#
# If a resume is ever needed, note that --resume-from restores the raw weights
# only: MEGVIIEMAHook re-initialises unless its own `resume` is pointed at the
# matching epoch_N_ema.pth.
# --------------------------------------------------------------------- #
# Occupancy is off for now: det (+motion) and map only.                  #
# --------------------------------------------------------------------- #
# The occupancy head never reached a usable operating point in either v1 or
# ver2 -- occ_sep/ratio sat at 1.008 against a floor of 1.0, i.e. a spatially
# constant output, and the corrected metric (report #07 2) says the true
# separation is worse than that number, not better. Carrying a head that has
# not been shown to learn adds a third of the aux parameters to the clip norm
# and a third of the aux slice of the BEV gradient, for no teacher.
#
# Re-enabling is one line: delete `occ_head=None` and put occ back into the
# grad_balance target below.
#
# NOTE: the pipeline still RASTERISES occupancy GT (GenerateSSROccLabels stays
# in train/test_pipeline and gt_occ_seg/gt_occ_valid stay in the collect keys).
# Stripping it would mean copying both pipeline lists into this file, since
# mmcv gives a derived config no handle on a base list, and a stale copy of a
# pipeline is a worse bug than the cost. Measured, that cost is 94.8 ms per
# sample, 4.19% of the pipeline -- and the dataloader has roughly 4x headroom
# at workers_per_gpu=16 (32 workers produce ~14 samples/s against the ~3.6/s
# training consumes), so it does not show up in wall time.
model = dict(
    occ_head=None,
    test_aux_heads=True,
    # plan keeps the 0.4 it had, so the planning arm of the experiment is
    # unchanged; occ's 0.2 is split evenly between the two remaining heads.
    # Motion has no separate entry because it shares det's valve -- one forward
    # through det_motion_head means one BEV input. gshare/motion is still
    # logged separately for information.
    #
    # _delete_=True is NOT optional. mmcv deep-merges dicts, so without it
    # the base's occ=0.2 survives into this target and the balancer keeps
    # solving for a head that does not exist.
    grad_balance=dict(
        target=dict(_delete_=True, plan=0.4, det=0.3, map=0.3)))
evaluation = dict(interval=10)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='para-ssr',
                name='para_ssr_2gpu_b4_60ep',
                group='para',
                tags=['para-ssr', 'no-ffp', 'aux', '2gpu', 'batch4',
                      'global8', '60ep', 'aux-convergence'],
                config=dict(
                    model='PARA-SSR', ffp=False, gpus=2, batch_per_gpu=4,
                    global_batch=8, epochs=60, eval_interval=10)),
            by_epoch=False,
            interval=100),
    ])
