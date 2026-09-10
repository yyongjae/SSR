#!/usr/bin/env bash
# One train batch + one val batch on a tiny model. Verifies the whole path:
# scene loading -> features -> targets -> forward -> loss -> backward -> step.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_agent experiment_name=smoke_front3 \
  scene_filter=navtrain split=trainval scene_filter.max_scenes=8 \
  dataloader.params.batch_size=1 dataloader.params.num_workers=2 \
  trainer.params.fast_dev_run=true trainer.params.precision=32 \
  +trainer.params.devices=1 trainer.params.strategy=auto \
  trainer.params.gradient_clip_val=35.0 \
  agent.config.bev_h=25 agent.config.bev_w=25 \
  agent.config.image_scale=0.125 agent.config.crop_top=5 \
  agent.config.encoder_num_layers=1 \
  "$@"
