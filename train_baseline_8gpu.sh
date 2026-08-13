#!/usr/bin/env bash
# Original SSR (FFP world model) -- 8 GPUs, effective batch 8 == the paper setting.
# Checkpoints every epoch (raw + EMA); validation is OFF, evaluate separately with
# tools/test.py afterwards.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/yongjae/miniconda3/envs/ssr/bin:$PATH"
export NUMBA_CPU_NAME=generic NUMBA_CPU_FEATURES=""
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# GPUS must match the number of entries in CUDA_VISIBLE_DEVICES
export PORT=${PORT:-28509}
GPUS=${GPUS:-8}
export WANDB_DIR=${WANDB_DIR:-$PWD/work_dirs}
WORK_DIR=${WORK_DIR:-work_dirs/ssr_baseline_8gpu}

exec ./tools/dist_train.sh projects/configs/SSR/SSR_e2e.py "$GPUS" \
    --work-dir "$WORK_DIR" --seed 0 --no-validate \
    --cfg-options \
        log_config.hooks.2.init_kwargs.name=ssr_baseline_8gpu \
        log_config.hooks.2.init_kwargs.config.gpus="$GPUS" \
    "$@"
