#!/usr/bin/env bash
# Evaluation entry point called by `bash scripts/experiment.sh eval RUN=<name> GPUS=... [EPOCH=<n>] [EPDMS=0|1]`.
# Runs EPDMS evaluation (Option B+ default: beta=0.75, w_safe=1.50, w_cc=2.0) on navtest and navhard_two_stage.
set -euo pipefail

RUN_DIR="${1:?usage: eval.sh <RUN_DIR> <GPUS> [EPOCH] [EPDMS]}"
GPUS="${2:-0,1,2,3}"
EPOCH="${3:-}"
EPDMS="${4:-1}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CKPT_SPEC="last"
if [ -n "${EPOCH}" ]; then
  CKPT_SPEC="ep${EPOCH}"
fi

SPLIT="all"
METRIC="epdms"
if [ "${EPDMS}" = "0" ]; then
  SPLIT="navtest"
  METRIC="pdms"
elif [ "${EPDMS}" = "all" ] || [ "${EPDMS}" = "both" ]; then
  SPLIT="all"
  METRIC="all"
fi

exec env GPUS="${GPUS}" bash "${REPO}/scripts/evaluation/eval_epdms.sh" \
  --run-dir "${RUN_DIR}" \
  --ckpt "${CKPT_SPEC}" \
  --split "${SPLIT}" \
  --metric "${METRIC}"
