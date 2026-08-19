#!/usr/bin/env bash
# PARA-SSR: 2 GPUs x 4 samples/GPU = the paper's global batch of 8.
# LR, warmup, epochs and EMA cadence intentionally remain unchanged.
set -euo pipefail
cd "$(dirname "$0")"

export PATH="/home/yongjae/miniconda3/envs/ssr/bin:$PATH"
export NUMBA_CPU_NAME=generic NUMBA_CPU_FEATURES=""
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PORT=${PORT:-28535}
export WANDB_DIR=${WANDB_DIR:-$PWD/work_dirs}
WORK_DIR=${WORK_DIR:-work_dirs/para_ssr_2gpu_b4}

# Do not add --autoscale-lr: global batch is already the original value (8).
exec ./tools/dist_train.sh projects/configs/SSR/PARA_SSR_e2e_2gpu_b4.py 2 \
    --work-dir "$WORK_DIR" --seed 0 --no-validate "$@"
