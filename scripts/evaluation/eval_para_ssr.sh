#!/usr/bin/env bash
# PARA-SSR on navsim -- PDM score on navtest (front 3 cameras by default).
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

# Each ablation arm needs its own output directory: Hydra rejects a duplicate
# override of the same key, so the experiment name cannot be changed from the
# caller's argument list -- it has to come from here.
EVAL_EXPERIMENT="${EVAL_EXPERIMENT:-eval/para_ssr_front3}"

# navtest data defaults to the repository layout data/dataset/{navsim_logs,sensor_blobs}/test.
# NAVSIM_DOWNLOAD instead points at an unpacked NAVSIM download directory that
# holds test_navsim_logs/test and test_sensor_blobs/test.
DATA_ARGS=()
if [[ -n "${NAVSIM_DOWNLOAD:-}" ]]; then
  for SUBDIR in test_navsim_logs/test test_sensor_blobs/test; do
    if [[ ! -d "${NAVSIM_DOWNLOAD}/${SUBDIR}" ]]; then
      echo "NAVSIM_DOWNLOAD=${NAVSIM_DOWNLOAD} has no ${SUBDIR}" >&2
      exit 2
    fi
  done
  DATA_ARGS+=("navsim_log_path=${NAVSIM_DOWNLOAD}/test_navsim_logs/test")
  DATA_ARGS+=("sensor_blobs_path=${NAVSIM_DOWNLOAD}/test_sensor_blobs/test")
fi

python "${REPO}/navsim/planning/script/run_pdm_score_gpu.py" \
  agent=para_ssr_agent \
  agent.checkpoint_path="${CKPT}" \
  experiment_name="${EVAL_EXPERIMENT}" \
  scene_filter=navtest \
  split=test \
  metric_cache_path="${METRIC_CACHE_PATH:-${REPO}/data/exp/metric_cache}" \
  "${DATA_ARGS[@]}" \
  "$@"
