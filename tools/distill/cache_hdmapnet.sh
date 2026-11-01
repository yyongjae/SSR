#!/usr/bin/env bash
# Cache frozen HDMapNet/P-MapNet planning features for nuScenes train and val.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SSR_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

PMAPNET_PYTHON=${PMAPNET_PYTHON:-/home/byounggun/anaconda3/envs/pmapnet/bin/python}
PMAPNET_REPO=${PMAPNET_REPO:-/data2/byounggun/rideflux/P-MapNet}
HDMAPNET_GPU=${HDMAPNET_GPU:-1}
DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/nuscenes}
DISTILL_CKPT_ROOT=${DISTILL_CKPT_ROOT:-/data2/byounggun/rideflux/pretrained_checkpoints}
DISTILL_CACHE_ROOT=${DISTILL_CACHE_ROOT:-$DISTILL_CKPT_ROOT/distill_bev_cache}
DISTILL_BATCH_SIZE=${DISTILL_BATCH_SIZE:-1}
DISTILL_WORKERS=${DISTILL_WORKERS:-4}

CHECKPOINT=$DISTILL_CKPT_ROOT/hdmapnet_nuscenes_60x30_lidar_camera.pth

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

require_file "$PMAPNET_PYTHON"
require_dir "$PMAPNET_REPO"
require_dir "$DISTILL_DATA_ROOT"
require_file "$CHECKPOINT"

for split in train val; do
  echo "[HDMapNet] split=$split physical_gpu=$HDMAPNET_GPU cache=$DISTILL_CACHE_ROOT"
  CUDA_VISIBLE_DEVICES="$HDMAPNET_GPU" "$PMAPNET_PYTHON" \
    "$SSR_ROOT/tools/distill/cache_teacher_bev.py" \
    --teacher hdmapnet \
    --teacher-repo "$PMAPNET_REPO" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DISTILL_DATA_ROOT" \
    --split "$split" \
    --cache-root "$DISTILL_CACHE_ROOT" \
    --batch-size "$DISTILL_BATCH_SIZE" \
    --workers "$DISTILL_WORKERS" \
    --device cuda:0 \
    "$@"
done
