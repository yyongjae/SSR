#!/usr/bin/env bash
# Cache frozen BEVDepth planning features for nuScenes train and val.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SSR_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

BEVDEPTH_PYTHON=${BEVDEPTH_PYTHON:-/home/byounggun/anaconda3/envs/bevdepth/bin/python}
BEVDEPTH_REPO=${BEVDEPTH_REPO:-/data2/byounggun/rideflux/BEVDepth}
BEVDEPTH_GPU=${BEVDEPTH_GPU:-0}
DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/nuscenes}
DISTILL_CKPT_ROOT=${DISTILL_CKPT_ROOT:-/data2/byounggun/rideflux/pretrained_checkpoints}
DISTILL_CACHE_ROOT=${DISTILL_CACHE_ROOT:-$DISTILL_CKPT_ROOT/distill_bev_cache}
DISTILL_BATCH_SIZE=${DISTILL_BATCH_SIZE:-1}
DISTILL_WORKERS=${DISTILL_WORKERS:-4}

CHECKPOINT=$DISTILL_CKPT_ROOT/bevdepth_nuscenes_r50_256x704_cbgs.pth

require_file() {
  if [ ! -f "$1" ]; then
    echo "missing file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    echo "missing directory: $1" >&2
    exit 1
  fi
}

require_file "$BEVDEPTH_PYTHON"
require_dir "$BEVDEPTH_REPO"
require_dir "$DISTILL_DATA_ROOT"
require_file "$CHECKPOINT"

for split in train val; do
  info_path=$DISTILL_CKPT_ROOT/bevdepth_infos_${split}.pkl
  require_file "$info_path"

  echo "[BEVDepth] split=$split physical_gpu=$BEVDEPTH_GPU cache=$DISTILL_CACHE_ROOT"
  CUDA_VISIBLE_DEVICES="$BEVDEPTH_GPU" "$BEVDEPTH_PYTHON" \
    "$SSR_ROOT/tools/distill/cache_teacher_bev.py" \
    --teacher bevdepth \
    --teacher-repo "$BEVDEPTH_REPO" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DISTILL_DATA_ROOT" \
    --split "$split" \
    --info-path "$info_path" \
    --cache-root "$DISTILL_CACHE_ROOT" \
    --batch-size "$DISTILL_BATCH_SIZE" \
    --workers "$DISTILL_WORKERS" \
    --device cuda:0 \
    "$@"
done
