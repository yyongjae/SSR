#!/usr/bin/env bash
# PARA-SSR, 60 epochs, 2 GPUs x 4 samples/GPU (= the paper's global batch of 8).
#
#   CUDA_VISIBLE_DEVICES=0,2 ./train_para_2gpu_b4_60ep.sh
#
# What this run is: det (+motion) and map as parallel auxiliary heads on the
# shared BEV, at VAD's learning rate over VAD's 60-epoch cosine, with each
# task's share of the BEV gradient held at plan 0.4 / det 0.3 / map 0.3.
# Occupancy is off. See the config docstring for why each of those is what it
# is, and report/07 for what changed and what it was measured against.
#
# Deliberately no --no-validate: the point is watching the aux tasks converge,
# and the config validates every 5 epochs with test_aux_heads=True so map mAP
# and detection mAP come back alongside the planning metrics.
#
# Never add --autoscale-lr. tools/train.py scales by len(gpu_ids)/8 and ignores
# samples_per_gpu, so it would silently change the LR without changing the batch.
#
# Before the first run:
#   1. map GT cache (~30 min, CPU only, already built for the val split):
#        python tools/build_map_gt_cache.py \
#          projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py
#   2. smoke-test the epoch-5 validation path (~10 min) so a crash there
#      surfaces now rather than five days in:
#        tools/verify_dist_eval.sh 8 \
#          projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py 0,2
#
# On resume (--resume-from work_dirs/.../epoch_N.pth), two pieces of state do
# NOT come back with the weights:
#   * the EMA shadow model -- MEGVIIEMAHook re-initialises unless its own
#     `resume` is pointed at the matching epoch_N_ema.pth;
#   * the GradBalancer scales -- they reset to 1.0 and are re-solved at the
#     first measurement after warm-up, so roughly 600 iterations (0.17 epoch)
#     run unbalanced. Pass
#       --cfg-options model.grad_balance.warmup_iters=0
#     on a resume to cut that to 200.
set -euo pipefail
cd "$(dirname "$0")"

export PATH="/home/yongjae/miniconda3/envs/ssr/bin:$PATH"
export NUMBA_CPU_NAME=generic NUMBA_CPU_FEATURES=""
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PORT=${PORT:-28537}
export WANDB_DIR=${WANDB_DIR:-$PWD/work_dirs}
WORK_DIR=${WORK_DIR:-work_dirs/para_ssr_2gpu_b4_60ep}

NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
if [ "$NGPU" -ne 2 ]; then
  echo "this config is 2 x 4 = global batch 8; got $NGPU GPUs." >&2
  echo "Changing the GPU count alone changes the global batch and invalidates" >&2
  echo "the LR. For four GPUs use train_para_4gpu_60ep.sh (4 x 2 = 8)." >&2
  exit 1
fi

exec ./tools/dist_train.sh projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py 2 \
    --work-dir "$WORK_DIR" --seed 0 "$@"
