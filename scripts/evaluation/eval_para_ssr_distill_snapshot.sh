#!/usr/bin/env bash
# Snapshot Stage-2 last.ckpt (training overwrites it) and run NAVSIM PDM on one GPU.
#
# Default: GPU 1, frozen epoch-18 snapshot, full navtest.
#
#   bash scripts/evaluation/eval_para_ssr_distill_snapshot.sh
#   SMOKE=1 bash scripts/evaluation/eval_para_ssr_distill_snapshot.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/evaluation/eval_para_ssr_distill_snapshot.sh -- extra hydra overrides
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ -x "/home/external-user/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/external-user/miniconda3/envs/ssr/bin/python}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

SNAPSHOT_DIR="${SNAPSHOT_DIR:-${REPO}/work_dirs/eval_snapshots/paradrive_distill_stage2_epoch18}"
CKPT_SRC="${CKPT_SRC:-${REPO}/work_dirs/paradrive_distill_stage2_dual_distill/lightning_logs/version_0/checkpoints/last.ckpt}"
SNAPSHOT_CKPT="${SNAPSHOT_CKPT:-${SNAPSHOT_DIR}/last.ckpt}"
STUDENT_CKPT="${STUDENT_CKPT:-${SNAPSHOT_DIR}/last_student.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval/paradrive_distill_stage2_epoch18}"
NAVTEST_LOGS="${NAVTEST_LOGS:-/home/external-user/navsim/download/test_navsim_logs/test}"
NAVTEST_BLOBS="${NAVTEST_BLOBS:-/home/external-user/navsim/download/test_sensor_blobs/test}"
METRIC_CACHE="${METRIC_CACHE:-${REPO}/data/exp/metric_cache}"
SMOKE="${SMOKE:-0}"

SHARDS="${SHARDS:-1}"
SHARD="${SHARD:-0}"
NAVTEST_SCENE_FILTER="${REPO}/navsim/planning/script/config/common/scene_filter/navtest.yaml"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $0 [hydra overrides...]"
  echo "env: CUDA_VISIBLE_DEVICES=1 SMOKE=1 EXPERIMENT_NAME=eval/name"
  echo "     CKPT_SRC=/path/to/last.ckpt SNAPSHOT_DIR=/path/to/snapshot"
  echo "     SHARDS=2 SHARD=0   # split navtest logs across GPUs"
  exit 0
fi

if ! [[ "${SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARDS must be a positive integer, got: ${SHARDS}" >&2
  exit 2
fi
if ! [[ "${SHARD}" =~ ^[0-9]+$ ]] || (( SHARD < 0 || SHARD >= SHARDS )); then
  echo "SHARD must be in [0, SHARDS), got SHARD=${SHARD} SHARDS=${SHARDS}" >&2
  exit 2
fi

mkdir -p "${SNAPSHOT_DIR}"
if [[ ! -f "${SNAPSHOT_CKPT}" ]]; then
  if [[ ! -f "${CKPT_SRC}" ]]; then
    echo "missing training checkpoint: ${CKPT_SRC}" >&2
    exit 2
  fi
  echo "copying ${CKPT_SRC} -> ${SNAPSHOT_CKPT}"
  cp -a "${CKPT_SRC}" "${SNAPSHOT_CKPT}"
else
  echo "reusing frozen snapshot: ${SNAPSHOT_CKPT}"
fi

if [[ ! -f "${STUDENT_CKPT}" ]]; then
  echo "exporting student-only checkpoint (drops agent._distill.*)"
  "${PYTHON}" "${REPO}/scripts/evaluation/export_student_ckpt.py" \
    "${SNAPSHOT_CKPT}" "${STUDENT_CKPT}"
else
  echo "reusing student checkpoint: ${STUDENT_CKPT}"
fi

if [[ ! -d "${NAVTEST_LOGS}" ]]; then
  echo "navtest logs missing: ${NAVTEST_LOGS}" >&2
  exit 2
fi
if [[ ! -d "${NAVTEST_BLOBS}" ]]; then
  echo "navtest sensor blobs missing: ${NAVTEST_BLOBS}" >&2
  exit 2
fi
if [[ ! -e "${METRIC_CACHE}" ]]; then
  echo "metric cache missing: ${METRIC_CACHE}" >&2
  exit 2
fi

LOGS_OVERRIDE=""
if [[ "${SHARDS}" -gt 1 ]]; then
  LOGS_OVERRIDE="$(
    "${PYTHON}" - "${NAVTEST_SCENE_FILTER}" "${SHARD}" "${SHARDS}" <<'PY'
from pathlib import Path
import sys

yaml_path = Path(sys.argv[1])
shard = int(sys.argv[2])
shards = int(sys.argv[3])
logs = []
in_logs = False
for line in yaml_path.read_text().splitlines():
    if line.startswith("log_names:"):
        in_logs = True
        continue
    if in_logs:
        if line.startswith("tokens:"):
            break
        stripped = line.strip()
        if stripped.startswith("- "):
            logs.append(stripped[2:].strip().strip("'\""))
if not logs:
    raise SystemExit(f"no log_names parsed from {yaml_path}")
start = shard * len(logs) // shards
end = (shard + 1) * len(logs) // shards
part = logs[start:end]
if not part:
    raise SystemExit(f"empty shard {shard}/{shards}")
print("[" + ",".join("'" + name + "'" for name in part) + "]")
print(f"shard {shard}/{shards}: logs {start}:{end} ({len(part)}/{len(logs)})", file=sys.stderr)
PY
  )"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_shard${SHARD}of${SHARDS}"
fi

EVAL_ARGS=(
  "agent=para_ssr_agent"
  "agent.checkpoint_path=${STUDENT_CKPT}"
  "agent.config.use_stl=false"
  "agent.config.plan_num_layers=3"
  "experiment_name=${EXPERIMENT_NAME}"
  "scene_filter=navtest"
  "split=test"
  "metric_cache_path=${METRIC_CACHE}"
  "navsim_log_path=${NAVTEST_LOGS}"
  "sensor_blobs_path=${NAVTEST_BLOBS}"
)
if [[ -n "${LOGS_OVERRIDE}" ]]; then
  EVAL_ARGS+=("scene_filter.log_names=${LOGS_OVERRIDE}")
fi

if [[ "${SMOKE}" == "1" || "${SMOKE}" == "true" ]]; then
  EVAL_ARGS+=("scene_filter.max_scenes=8")
  echo "SMOKE=1: scoring at most 8 navtest scenes"
fi

echo "GPU=${CUDA_VISIBLE_DEVICES} python=${PYTHON}"
echo "student ckpt=${STUDENT_CKPT}"
echo "experiment=${EXPERIMENT_NAME}"

"${PYTHON}" "${REPO}/navsim/planning/script/run_pdm_score_gpu.py" \
  "${EVAL_ARGS[@]}" \
  "$@"
