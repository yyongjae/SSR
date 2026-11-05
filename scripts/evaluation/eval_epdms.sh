#!/usr/bin/env bash
# NAVSIM v2.2 EPDMS evaluation (navtest & navhard_two_stage) for v2_v15_v7 (64 m BEV) checkpoints.
#
#   GPUS=0,1,2,3 bash scripts/evaluation/eval_epdms.sh --run-dir <RUN> --split navtest [--ckpt last]
#   GPUS=0,1,2,3 bash scripts/evaluation/eval_epdms.sh --run-dir <RUN> --split navhard [--ckpt last]
#   GPUS=0,1,2,3 bash scripts/evaluation/eval_epdms.sh --run-dir <RUN> --split all     [--ckpt epoch:19]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${REPO}/tools/release/load_paths.sh"
GPUS="${GPUS:-0,1,2,3}"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export CK_BEV64=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case ",${GPUS}," in
  *,4,*|*,5,*|*,6,*|*,7,*)
    export CK_ALLOW_GPU4567=1
    ;;
esac

exec "${SSR_PY}" -B "${REPO}/tools/ck/e2e2/eval_epdms.py" --gpus "${GPUS}" "$@"
