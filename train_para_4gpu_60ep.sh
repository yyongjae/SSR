#!/usr/bin/env bash
# PARA-SSR, 60 epochs, 4 GPUs x 2 samples = the same global batch of 8.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./train_para_4gpu_60ep.sh
#
# Same experiment as train_para_2gpu_b4_60ep.sh, ~3.5 days instead of ~7.
# Deliberately no --no-validate: the point of the long run is watching the aux
# tasks converge, and the config validates every 5 epochs with
# test_aux_heads=True so map mAP / detection mAP come back alongside planning.
#
# Never add --autoscale-lr. tools/train.py scales by len(gpu_ids)/8 and ignores
# samples_per_gpu, so on 4 GPUs it would halve the learning rate.
#
# Before the first run, build the map GT cache once (~30 min, CPU only):
#   python tools/build_map_gt_cache.py projects/configs/SSR/PARA_SSR_e2e_4gpu_60ep.py
# and smoke-test the validation path:
#   tools/verify_dist_eval.sh 8 projects/configs/SSR/PARA_SSR_e2e_4gpu_60ep.py 0,1
set -euo pipefail
cd "$(dirname "$0")"

export PATH="/home/yongjae/miniconda3/envs/ssr/bin:$PATH"
export NUMBA_CPU_NAME=generic NUMBA_CPU_FEATURES=""
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PORT=${PORT:-28541}
export WANDB_DIR=${WANDB_DIR:-$PWD/work_dirs}
WORK_DIR=${WORK_DIR:-work_dirs/para_ssr_4gpu_60ep}

NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
if [ "$NGPU" -ne 4 ]; then
  echo "this config is 4 x 2 = global batch 8; got $NGPU GPUs." >&2
  echo "Changing the GPU count without changing samples_per_gpu changes the" >&2
  echo "global batch and invalidates the LR. Use the 2-GPU config instead." >&2
  exit 1
fi

exec ./tools/dist_train.sh projects/configs/SSR/PARA_SSR_e2e_4gpu_60ep.py 4 \
    --work-dir "$WORK_DIR" --seed 0 "$@"
