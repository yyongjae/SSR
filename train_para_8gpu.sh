#!/usr/bin/env bash
# PARA-SSR ver A (no FFP, parallel aux heads) -- 8 GPUs, effective batch 8.
# Same schedule as the baseline so the two runs are directly comparable.
# Checkpoints every epoch (raw + EMA); validation is OFF.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/yongjae/miniconda3/envs/ssr/bin:$PATH"
export NUMBA_CPU_NAME=generic NUMBA_CPU_FEATURES=""
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# GPUS must match the number of entries in CUDA_VISIBLE_DEVICES
export PORT=${PORT:-28509}
GPUS=${GPUS:-8}
export WANDB_DIR=${WANDB_DIR:-$PWD/work_dirs}
CONFIG=${CONFIG:-projects/configs/SSR/PARA_SSR_e2e.py}
WORK_DIR=${WORK_DIR:-work_dirs/para_ssr_8gpu}

exec ./tools/dist_train.sh "$CONFIG" "$GPUS" \
    --work-dir "$WORK_DIR" --seed 0 --no-validate \
    --cfg-options \
        log_config.hooks.2.init_kwargs.name=para_ssr_8gpu \
        log_config.hooks.2.init_kwargs.config.gpus="$GPUS" \
    "$@"
