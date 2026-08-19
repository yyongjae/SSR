#!/usr/bin/env bash
# PARA-SSR, two-stage: 48 epochs det+map, then 12 epochs with planning on.
#
#   CUDA_VISIBLE_DEVICES=0,2 ./train_para_staged.sh          # both stages
#   CUDA_VISIBLE_DEVICES=0,2 ./train_para_staged.sh stage1   # just stage 1
#   CUDA_VISIBLE_DEVICES=0,2 ./train_para_staged.sh stage2   # just stage 2
#
# 48 + 12 = 60 epochs, the same wall-clock budget as the single-stage
# train_para_2gpu_b4_60ep.sh -- but NOT the same experiment with one variable
# moved. Two cosines instead of one, Adam moments reset at the boundary, EMA
# re-initialised, RNG re-seeded, and the planner and motion heads trained for
# 42,192 iterations against 210,960. See the stage-1 config docstring.
#
# Stage 2 uses load_from, not resume_from: weights carry over, optimiser state
# and LR schedule restart. That is what VAD and HiP-AD both do, and it is why
# stage 2 is a fresh 12-epoch cosine rather than the tail of a 48-epoch one.
#
# Never add --autoscale-lr (tools/train.py scales by len(gpu_ids)/8 and ignores
# samples_per_gpu).
set -euo pipefail
cd "$(dirname "$0")"

export PATH="/home/yongjae/miniconda3/envs/ssr/bin:$PATH"
export NUMBA_CPU_NAME=generic NUMBA_CPU_FEATURES=""
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PORT=${PORT:-28545}
export WANDB_DIR=${WANDB_DIR:-$PWD/work_dirs}

WHICH=${1:-both}
S1_DIR=${S1_DIR:-work_dirs/para_ssr_stage1_detmap}
S2_DIR=${S2_DIR:-work_dirs/para_ssr_stage2_all}

NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
if [ "$NGPU" -ne 2 ]; then
  echo "these configs are 2 x 4 = global batch 8; got $NGPU GPUs." >&2
  exit 1
fi

run_stage1() {
  echo "=== stage 1: 48 epochs, det + map (plan weight 0.0) -> $S1_DIR ==="
  ./tools/dist_train.sh projects/configs/SSR/PARA_SSR_stage1_detmap.py 2 \
      --work-dir "$S1_DIR" --seed 0
}

run_stage2() {
  local ckpt="$S1_DIR/latest.pth"
  if [ ! -e "$ckpt" ]; then
    echo "stage 2 needs $ckpt, which does not exist." >&2
    echo "Run stage 1 first, or point S1_DIR at an existing stage-1 run." >&2
    exit 1
  fi
  echo "=== stage 2: 12 epochs, all tasks, from $ckpt -> $S2_DIR ==="
  # The config's load_from names the default S1_DIR; override it here so a
  # non-default S1_DIR is actually honoured instead of silently ignored.
  PORT=$((PORT + 1)) ./tools/dist_train.sh \
      projects/configs/SSR/PARA_SSR_stage2_all.py 2 \
      --work-dir "$S2_DIR" --seed 0 \
      --cfg-options load_from="$ckpt"
}

case "$WHICH" in
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  both)   run_stage1; run_stage2 ;;
  *) echo "usage: $0 [stage1|stage2|both]" >&2; exit 1 ;;
esac
