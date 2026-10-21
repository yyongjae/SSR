#!/usr/bin/env bash
# One train batch + one val batch of the v3 factorized planner on a tiny model and
# a handful of real navtrain scenes: scene loading -> targets (rasters, route
# centerline, forced vocabulary indices) -> forward -> losses -> backward -> step.
# VOCAB defaults to the real vocabulary; set VOCAB to a smoke vocabulary directory
# (tools/fact/make_smoke_vocab.py) before the real one exists.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${SSR_NAVSIM_PYTHON:-${HOME}/conda-envs/ssr-navsim/bin/python}"
VOCAB="${VOCAB:-${REPO}/data/fact_vocab}"
PATHS="${PATHS:-${VOCAB}/paths_1024.npy}"
SPEEDS="${SPEEDS:-${VOCAB}/speeds_256.npy}"

"${PY}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_agent experiment_name="${EXPERIMENT:-smoke_fact}" \
  scene_filter=navtrain split=trainval scene_filter.max_scenes="${MAX_SCENES:-8}" \
  dataloader.params.batch_size="${BATCH_SIZE:-2}" dataloader.params.num_workers=2 \
  trainer.params.fast_dev_run="${FAST_DEV_RUN:-true}" trainer.params.precision=32 \
  +trainer.params.devices=1 trainer.params.strategy=auto \
  trainer.params.gradient_clip_val=35.0 wandb.enable=false \
  agent.config.bev_h="${BEV_H:-10}" agent.config.bev_w="${BEV_W:-20}" \
  agent.config.image_scale="${IMAGE_SCALE:-0.125}" agent.config.crop_top=5 \
  agent.config.encoder_num_layers=1 \
  agent.config.fact_enable=true agent.config.input_target=true \
  agent.config.fact_path_file="${PATHS}" agent.config.fact_speed_file="${SPEEDS}" \
  agent.config.plan_anchor_file="${REPO}/data/planning_vb/trajectory_anchors_256.npy" \
  agent.config.plan_score_file="${REPO}/data/planning_vb/pdm_score_256" \
  "$@"
