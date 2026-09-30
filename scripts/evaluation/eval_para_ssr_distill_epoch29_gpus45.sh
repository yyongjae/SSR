#!/usr/bin/env bash
# Split Stage-2 epoch-29 NAVSIM PDM across GPU 4 and 5, then print the merged score.
#
#   bash scripts/evaluation/eval_para_ssr_distill_epoch29_gpus45.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_ONE="${REPO}/scripts/evaluation/eval_para_ssr_distill_snapshot.sh"
EXPORT_PY="${REPO}/scripts/evaluation/export_student_ckpt.py"

if [[ -x "/home/external-user/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/external-user/miniconda3/envs/ssr/bin/python}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

GPU0="${GPU0:-4}"
GPU1="${GPU1:-5}"
CKPT_SRC="${CKPT_SRC:-${REPO}/work_dirs/paradrive_distill_stage2_dual_distill/lightning_logs/version_0/checkpoints/epoch=29-step=19950.ckpt}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${REPO}/work_dirs/eval_snapshots/paradrive_distill_stage2_epoch29}"
SNAPSHOT_CKPT="${SNAPSHOT_CKPT:-${SNAPSHOT_DIR}/epoch29.ckpt}"
STUDENT_CKPT="${STUDENT_CKPT:-${SNAPSHOT_DIR}/epoch29_student.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval/paradrive_distill_stage2_epoch29}"
MERGE_DIR="${REPO}/work_dirs/${EXPERIMENT_NAME}"
SMOKE="${SMOKE:-0}"
# This repo's data/dataset/navsim_logs symlink is trainval only.
# Official navtest logs and blobs live in the download trees.
NAVTEST_LOGS="${NAVTEST_LOGS:-/home/external-user/navsim/download/test_navsim_logs/test}"
NAVTEST_BLOBS="${NAVTEST_BLOBS:-/home/external-user/navsim/download/test_sensor_blobs/test}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $0 [hydra overrides...]"
  echo "env: GPU0=4 GPU1=5 SMOKE=1 CKPT_SRC=/path/to/epoch=29-*.ckpt"
  echo "     IMAGE_ARCHITECTURE=resnet34.tv_in1k"
  echo "     or: agent.config.image_architecture=resnet34.tv_in1k"
  echo "     PDM and det/map mAP both use that backbone. Default is resnet50.tv_in1k."
  exit 0
fi

if [[ ! -f "${CKPT_SRC}" ]]; then
  echo "missing checkpoint: ${CKPT_SRC}" >&2
  exit 2
fi
if [[ ! -x "${EVAL_ONE}" && ! -f "${EVAL_ONE}" ]]; then
  echo "missing eval launcher: ${EVAL_ONE}" >&2
  exit 2
fi

mkdir -p "${SNAPSHOT_DIR}" "${MERGE_DIR}" \
  "${REPO}/work_dirs/${EXPERIMENT_NAME}_shard0of2" \
  "${REPO}/work_dirs/${EXPERIMENT_NAME}_shard1of2"

if [[ ! -f "${SNAPSHOT_CKPT}" ]]; then
  echo "copying ${CKPT_SRC} -> ${SNAPSHOT_CKPT}"
  cp -a "${CKPT_SRC}" "${SNAPSHOT_CKPT}"
fi
if [[ ! -f "${STUDENT_CKPT}" ]]; then
  echo "exporting student-only checkpoint (drops agent._distill.*)"
  "${PYTHON}" "${EXPORT_PY}" "${SNAPSHOT_CKPT}" "${STUDENT_CKPT}"
fi

# One backbone for PDM and det/map. ResNet-50 stays the default.
# A CLI override wins over IMAGE_ARCHITECTURE.
IMAGE_ARCHITECTURE="${IMAGE_ARCHITECTURE:-resnet50.tv_in1k}"
EVAL_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    agent.config.image_architecture=*)
      IMAGE_ARCHITECTURE="${arg#agent.config.image_architecture=}"
      ;;
    *)
      EVAL_ARGS+=("${arg}")
      ;;
  esac
done
if [[ ! "${IMAGE_ARCHITECTURE}" =~ ^[A-Za-z0-9._+-]+$ ]]; then
  echo "invalid image architecture: ${IMAGE_ARCHITECTURE}" >&2
  exit 2
fi
EVAL_ARGS+=("agent.config.image_architecture=${IMAGE_ARCHITECTURE}")
echo "image_architecture=${IMAGE_ARCHITECTURE}"

run_shard() {
  local gpu="$1"
  local shard="$2"
  local log="$3"
  shift 3
  echo "starting shard ${shard}/2 on GPU ${gpu}, log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHON="${PYTHON}" \
    CKPT_SRC="${SNAPSHOT_CKPT}" \
    SNAPSHOT_DIR="${SNAPSHOT_DIR}" \
    SNAPSHOT_CKPT="${SNAPSHOT_CKPT}" \
    STUDENT_CKPT="${STUDENT_CKPT}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    SHARDS=2 \
    SHARD="${shard}" \
    SMOKE="${SMOKE}" \
    bash "${EVAL_ONE}" "$@" >"${log}" 2>&1
}

LOG0="${REPO}/work_dirs/${EXPERIMENT_NAME}_shard0of2/run.log"
LOG1="${REPO}/work_dirs/${EXPERIMENT_NAME}_shard1of2/run.log"
pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

run_shard "${GPU0}" 0 "${LOG0}" "${EVAL_ARGS[@]}" &
pids+=("$!")
run_shard "${GPU1}" 1 "${LOG1}" "${EVAL_ARGS[@]}" &
pids+=("$!")

set +e
wait "${pids[0]}"
st0=$?
wait "${pids[1]}"
st1=$?
set -e
trap - INT TERM

if [[ "${st0}" -ne 0 || "${st1}" -ne 0 ]]; then
  echo "shard failed: GPU${GPU0}/shard0 exit=${st0}  GPU${GPU1}/shard1 exit=${st1}" >&2
  echo "----- shard0 log: ${LOG0} -----" >&2
  tail -n 40 "${LOG0}" >&2 || true
  echo "----- shard1 log: ${LOG1} -----" >&2
  tail -n 40 "${LOG1}" >&2 || true
  exit 1
fi

"${PYTHON}" - "${REPO}/work_dirs/${EXPERIMENT_NAME}_shard0of2" \
  "${REPO}/work_dirs/${EXPERIMENT_NAME}_shard1of2" \
  "${MERGE_DIR}/merged.csv" <<'PY'
from pathlib import Path
import sys
import pandas as pd

shard_dirs = [Path(sys.argv[1]), Path(sys.argv[2])]
out = Path(sys.argv[3])
frames = []
used = []
for root in shard_dirs:
    csvs = sorted(
        (p for p in root.glob("*.csv") if p.name != "merged.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not csvs:
        raise SystemExit(f"no PDM csv in {root}")
    path = csvs[-1]
    used.append(path)
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    if "token" not in df.columns:
        raise SystemExit(f"{path} has no token column")
    frames.append(df[df["token"] != "average"].copy())

merged = pd.concat(frames, ignore_index=True)
if merged["token"].duplicated().any():
    raise SystemExit("overlapping tokens between shards")
n = len(merged)
n_valid = int(merged["valid"].sum()) if "valid" in merged.columns else n
avg = merged.drop(columns=["token"], errors="ignore").mean(numeric_only=True, skipna=True)
print("")
print("======================================================================")
print(" NAVSIM PDM score  epoch=29  GPU 4+5 merged")
print(f" scenes={n}  valid={n_valid}  failed={n - n_valid}")
print("----------------------------------------------------------------------")
for key, value in avg.items():
    print(f" {key}: {float(value):.4f}")
print("----------------------------------------------------------------------")
print(" shard csvs:")
for path in used:
    print(f"  {path}")
print(f" merged: {out}")
print("======================================================================")
avg_row = avg.to_dict()
avg_row["token"] = "average"
avg_row["valid"] = bool(n_valid == n)
out.parent.mkdir(parents=True, exist_ok=True)
pd.concat([merged, pd.DataFrame([avg_row])], ignore_index=True).to_csv(out, index=False)
PY

# Detection and map mAP on the same navtest split and the same two GPUs.
AUX_EXPERIMENT="${EXPERIMENT_NAME}_aux"
AUX_CONFIG="${SNAPSHOT_DIR}/aux_training_config.yaml"
"${PYTHON}" - "${REPO}" "${AUX_CONFIG}" "${IMAGE_ARCHITECTURE}" <<'PY'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

repo = Path(sys.argv[1])
out = Path(sys.argv[2])
image_architecture = sys.argv[3]
config_dir = repo / "navsim/planning/script/config/training"
with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
    cfg = compose(
        config_name="default_training",
        overrides=[
            "agent=para_ssr_agent",
            "agent.config.backbone_pretrained=false",
            "agent.config.test_aux_heads=true",
            f"agent.config.image_architecture={image_architecture}",
        ],
    )
if str(cfg.agent.config.image_architecture) != image_architecture:
    raise SystemExit(
        f"aux config backbone is {cfg.agent.config.image_architecture}, "
        f"expected {image_architecture}"
    )
out.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, out)
print(f"wrote aux training config: {out} image_architecture={image_architecture}")
PY

if [[ ! -d "${NAVTEST_LOGS}" ]]; then
  echo "navtest logs missing: ${NAVTEST_LOGS}" >&2
  exit 2
fi
if [[ ! -d "${NAVTEST_BLOBS}" ]]; then
  echo "navtest sensor blobs missing: ${NAVTEST_BLOBS}" >&2
  exit 2
fi
echo "starting det/map mAP on GPU ${GPU0},${GPU1}"
echo "aux logs=${NAVTEST_LOGS}"
SSR_NAVSIM_PYTHON="${PYTHON}" \
  GPU_IDS="${GPU0},${GPU1}" \
  AUX_CHECKPOINT="${STUDENT_CKPT}" \
  AUX_TRAINING_CONFIG="${AUX_CONFIG}" \
  AUX_EXPERIMENT="${AUX_EXPERIMENT}" \
  AUX_BATCH_SIZE="${AUX_BATCH_SIZE:-4}" \
  bash "${REPO}/scripts/evaluation/eval_para_ssr_aux.sh" \
    "navsim_log_path=${NAVTEST_LOGS}" \
    "sensor_blobs_path=${NAVTEST_BLOBS}"

"${PYTHON}" - "${REPO}/work_dirs/${AUX_EXPERIMENT}/aux_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text())["metrics"]
print("")
print("======================================================================")
print(" NAVSIM aux mAP")
print(f" det mAP: {float(metrics['detection']['mAP']):.4f}")
print(f" map mAP: {float(metrics['map']['mAP']):.4f}")
print(f" json: {path}")
print("======================================================================")
PY
