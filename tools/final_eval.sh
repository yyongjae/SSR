#!/usr/bin/env bash
# The numbers that go in a table. ONE GPU, sequential, on the EMA weights.
#
#   tools/final_eval.sh CONFIG CKPT [GPU]
#   tools/final_eval.sh projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py \
#                       work_dirs/para_ssr_2gpu_b4_60ep/epoch_60_ema.pth 0
#
# WHY THIS EXISTS: the EMA weights. Not the GPU count.
#
# EvalHook scores the RAW model -- eval_model is never handed to
# custom_train_model -- so every number in the training log is a raw-weight
# number. MEGVIIEMAHook writes epoch_N_ema.pth on its own, and nothing ever
# scores those. The EMA performance of a finished run is simply unknown until
# something like this is run. That gap has nothing to do with how many GPUs
# the evaluation uses.
#
# THE SINGLE GPU IS CHEAP INSURANCE, NOT A CORRECTION. The in-training 2-GPU
# numbers are fine to read. The test sampler gives each rank a contiguous
# block, so rank 1 starts mid-scene with prev_bev=None, and because prev_bev is
# a chain that perturbation carries to the end of that scene: measured on the
# val split, index 3010 through 3031, so 22 of 6019 samples (0.37%) -- one hard
# difference and twenty-one decaying ones. That does not move a mean over 6019
# samples anywhere that matters.
#
# It is pinned to one GPU anyway so that two runs being compared were scored
# identically BY CONSTRUCTION rather than by an argument that has to be made
# again every time (at four ranks it would be ~66 samples, still small, still
# an argument). VAD's docs/train_eval.md asks for one GPU for the same reason,
# and since the EMA weights need a separate pass regardless, fixing the
# protocol here costs nothing.
#
# DETECTION THRESHOLD. The dataset defaults to 0.0, which is the nuScenes
# protocol -- the official config sets no score floor, only max_boxes_per_sample
# 500, and the head emits 300 queries. Any floor above 0 truncates the PR curve
# and lowers mAP. VAD publishes a 0.2-thresholded number, so for a like-for-like
# comparison against VAD add
#     --cfg-options data.test.det_score_thresh="[0.0, 0.2]"
# which keeps 0.0 on the unsuffixed keys and adds .../mAP@0.2.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:?usage: final_eval.sh CONFIG CKPT [GPU]}
CKPT=${2:?usage: final_eval.sh CONFIG CKPT [GPU]}
GPU=${3:-0}

if [ ! -e "$CKPT" ]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 1
fi
case "$CKPT" in
  *_ema.pth) ;;
  *) echo "NOTE: $CKPT is not an *_ema.pth. The in-training validation already" >&2
     echo "      scored the raw weights; the EMA copy is the one that has not" >&2
     echo "      been measured. Continuing anyway." >&2 ;;
esac

# run.sh exports a working PATH/NUMBA before calling this; standalone use falls
# back to whatever env is active, or SSR_PYTHON.
if ! python -c 'import torch, mmcv' >/dev/null 2>&1; then
  [ -n "${SSR_PYTHON:-}" ] || { echo "activate the env, or use ./run.sh eval" >&2; exit 1; }
  export PATH="$(dirname "$SSR_PYTHON"):$PATH"
fi
export NUMBA_CPU_NAME="${NUMBA_CPU_NAME:-generic}"
export CUDA_VISIBLE_DEVICES="$GPU"

OUT=${OUT:-$(dirname "$CKPT")/final_eval_$(basename "${CKPT%.pth}")}
mkdir -p "$OUT"
echo "config : $CONFIG"
echo "ckpt   : $CKPT"
echo "gpu    : $GPU  (single, sequential)"
echo "out    : $OUT"

exec python tools/test.py "$CONFIG" "$CKPT" \
    --eval bbox --jsonfile_prefix "$OUT" "${@:4}" 2>&1 | tee "$OUT/eval.log"
