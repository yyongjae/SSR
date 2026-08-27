#!/usr/bin/env bash
# PARA-SSR on navsim -- PDM score on navtest.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CKPT="${1:?usage: eval_para_ssr.sh <absolute_checkpoint_path> [hydra overrides...]}"
shift || true

python "${REPO}/navsim/planning/script/run_pdm_score_gpu.py" \
  agent=para_ssr_agent \
  agent.checkpoint_path="${CKPT}" \
  experiment_name=eval/para_ssr \
  scene_filter=navtest \
  split=test \
  metric_cache_path="${REPO}/data/exp/metric_cache" \
  "$@"
