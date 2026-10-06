#!/usr/bin/env bash
# Precompute PDM metric cache for navtest
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT_OVERRIDE:-${REPO}/work_dirs}"

CACHE_PATH="${METRIC_CACHE_PATH:-${REPO}/data/exp/metric_cache}"
WORKERS="${WORKERS:-16}"
PYTHON="${PYTHON:-/root/miniconda3/envs/ssr/bin/python}"

"${PYTHON}" "${REPO}/navsim/planning/script/run_metric_caching.py" \
  worker=single_machine_thread_pool \
  worker.max_workers="${WORKERS}" \
  scene_filter=navtest \
  split=test \
  cache.cache_path="${CACHE_PATH}" \
  "$@"
