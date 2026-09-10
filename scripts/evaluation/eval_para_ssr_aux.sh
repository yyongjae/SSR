#!/usr/bin/env bash
# Held-out NAVSIM detection/map auxiliary mAP for PARA-SSR.
set -euo pipefail

AUX_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${AUX_REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${AUX_REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${AUX_REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${AUX_REPO}"
export NAVSIM_EXP_ROOT="${AUX_REPO}/work_dirs"
export HYDRA_FULL_ERROR=1

AUX_PYTHON="${SSR_NAVSIM_PYTHON:-python}"
AUX_CHECKPOINT="${AUX_CHECKPOINT:-${AUX_REPO}/work_dirs/para_ssr/para_ssr_ep30_final.ckpt}"
AUX_TRAINING_CONFIG="${AUX_TRAINING_CONFIG:-${AUX_REPO}/work_dirs/para_ssr/code/hydra/config.yaml}"
AUX_EXPERIMENT="${AUX_EXPERIMENT:-eval/para_ssr_ep30_aux}"
AUX_BATCH_SIZE="${AUX_BATCH_SIZE:-4}"
AUX_GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $0 [checkpoint.ckpt] [Hydra overrides...]"
  echo "env: GPU_IDS=0,1 AUX_EXPERIMENT=eval/name AUX_BATCH_SIZE=4"
  echo "     SSR_NAVSIM_PYTHON=/path/to/python (default: active environment's python)"
  exit 0
fi
if [[ $# -gt 0 && "$1" != *=* ]]; then
  AUX_CHECKPOINT="$1"
  shift
fi

if ! command -v "${AUX_PYTHON}" >/dev/null 2>&1; then
  echo "Python interpreter is unavailable: ${AUX_PYTHON}" >&2
  echo "activate the project environment or set SSR_NAVSIM_PYTHON" >&2
  exit 2
fi
if [[ ! -f "${AUX_CHECKPOINT}" ]]; then
  echo "checkpoint does not exist: ${AUX_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${AUX_TRAINING_CONFIG}" ]]; then
  echo "archived training config does not exist: ${AUX_TRAINING_CONFIG}" >&2
  exit 2
fi

IFS=',' read -r -a AUX_GPU_ARRAY <<< "${AUX_GPU_IDS}"
if [[ ${#AUX_GPU_ARRAY[@]} -lt 1 || -z "${AUX_GPU_ARRAY[0]}" ]]; then
  echo "GPU_IDS must contain at least one CUDA device" >&2
  exit 2
fi
declare -A AUX_SEEN_GPUS=()
for AUX_GPU in "${AUX_GPU_ARRAY[@]}"; do
  if [[ -z "${AUX_GPU}" || ! "${AUX_GPU}" =~ ^[A-Za-z0-9._:-]+$ ]]; then
    echo "GPU_IDS contains an invalid CUDA device: '${AUX_GPU}'" >&2
    exit 2
  fi
  if [[ -n "${AUX_SEEN_GPUS[${AUX_GPU}]:-}" ]]; then
    echo "GPU_IDS contains a duplicate CUDA device: ${AUX_GPU}" >&2
    exit 2
  fi
  AUX_SEEN_GPUS["${AUX_GPU}"]=1
done

AUX_RUNNER="${AUX_REPO}/navsim/planning/script/run_aux_evaluation.py"
AUX_OUTPUT_DIR="${NAVSIM_EXP_ROOT}/${AUX_EXPERIMENT}"
AUX_COMMON_ARGS=(
  "checkpoint_path=${AUX_CHECKPOINT}"
  "training_config_path=${AUX_TRAINING_CONFIG}"
  "experiment_name=${AUX_EXPERIMENT}"
  "scene_filter=navtest"
  "scene_filter_name=navtest"
  "split=test"
  "dataloader.batch_size=${AUX_BATCH_SIZE}"
)
AUX_USER_OVERRIDES=("$@")
AUX_NUM_SHARDS="${#AUX_GPU_ARRAY[@]}"
AUX_PIDS=()

aux_stop_children() {
  for AUX_PID in "${AUX_PIDS[@]}"; do
    kill "${AUX_PID}" 2>/dev/null || true
  done
  for AUX_PID in "${AUX_PIDS[@]}"; do
    wait "${AUX_PID}" 2>/dev/null || true
  done
  exit 130
}
trap aux_stop_children INT TERM

for ((AUX_SHARD_INDEX=0; AUX_SHARD_INDEX<AUX_NUM_SHARDS; AUX_SHARD_INDEX++)); do
  AUX_GPU="${AUX_GPU_ARRAY[${AUX_SHARD_INDEX}]}"
  CUDA_VISIBLE_DEVICES="${AUX_GPU}" "${AUX_PYTHON}" "${AUX_RUNNER}" \
    "${AUX_COMMON_ARGS[@]}" \
    "${AUX_USER_OVERRIDES[@]}" \
    "device=cuda:0" \
    "num_shards=${AUX_NUM_SHARDS}" \
    "shard_index=${AUX_SHARD_INDEX}" \
    "extract_only=true" \
    "aggregate_only=false" \
    "hydra.run.dir=${AUX_OUTPUT_DIR}/launcher/shard_${AUX_SHARD_INDEX}" \
    "hydra.output_subdir=code/hydra" &
  AUX_PIDS+=("$!")
done

AUX_EXTRACT_STATUS=0
for AUX_PID in "${AUX_PIDS[@]}"; do
  if ! wait "${AUX_PID}"; then
    AUX_EXTRACT_STATUS=1
  fi
done
if [[ ${AUX_EXTRACT_STATUS} -ne 0 ]]; then
  echo "one or more auxiliary extraction shards failed; aggregation skipped" >&2
  exit 1
fi
trap - INT TERM

# One CPU process performs strict global completeness/provenance validation and
# is the only process allowed to publish the final JSON/CSV completion marker.
CUDA_VISIBLE_DEVICES="" "${AUX_PYTHON}" "${AUX_RUNNER}" \
  "${AUX_COMMON_ARGS[@]}" \
  "${AUX_USER_OVERRIDES[@]}" \
  "device=cpu" \
  "num_shards=1" \
  "shard_index=0" \
  "extract_only=false" \
  "aggregate_only=true" \
  "hydra.run.dir=${AUX_OUTPUT_DIR}/launcher/aggregate" \
  "hydra.output_subdir=code/hydra"
