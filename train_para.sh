#!/usr/bin/env bash
# PARA-SSR ver A (no FFP, parallel aux heads) -- 2 GPUs
set -e
cd "$(dirname "$0")"
# dist_train.sh calls bare `python`; make sure it is the ssr env one
export PATH="/home/yongjae/miniconda3/envs/ssr/bin:$PATH"
export NUMBA_CPU_NAME=generic NUMBA_CPU_FEATURES=""
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export PORT=${PORT:-28510}
WORK_DIR=${WORK_DIR:-work_dirs/para_ssr_v1}
exec ./tools/dist_train.sh projects/configs/SSR/PARA_SSR_e2e.py 2 \
    --work-dir "$WORK_DIR" --seed 0 "$@"
