#!/usr/bin/env bash
# PDM metric cache for navtest -- required before eval_para_ssr.sh.
#
# Ray cannot start in this container (its GCS never binds, so the default
# worker=ray_distributed_no_torch hangs retrying).  `worker_map` is worker
# agnostic, so the Ray-free thread pool does the same job.
#
# WORKERS defaults to 6 to leave CPU for concurrent training; the caching is
# CPU-bound and reads annotations + maps only (no sensor blobs).
#
#   bash scripts/evaluation/cache_metric_navtest.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"

CACHE_PATH="${METRIC_CACHE_PATH:-${REPO}/data/exp/metric_cache}"
WORKERS="${WORKERS:-6}"

# navtest logs default to data/dataset/navsim_logs/test; NAVSIM_DOWNLOAD points at
# an unpacked NAVSIM download directory holding test_navsim_logs/test instead.
DATA_ARGS=()
if [[ -n "${NAVSIM_DOWNLOAD:-}" ]]; then
  if [[ ! -d "${NAVSIM_DOWNLOAD}/test_navsim_logs/test" ]]; then
    echo "NAVSIM_DOWNLOAD=${NAVSIM_DOWNLOAD} has no test_navsim_logs/test" >&2
    exit 2
  fi
  DATA_ARGS+=("navsim_log_path=${NAVSIM_DOWNLOAD}/test_navsim_logs/test")
fi

python "${REPO}/navsim/planning/script/run_metric_caching.py" \
  worker=single_machine_thread_pool \
  worker.max_workers="${WORKERS}" \
  scene_filter=navtest \
  split=test \
  cache.cache_path="${CACHE_PATH}" \
  "${DATA_ARGS[@]}" \
  "$@"
