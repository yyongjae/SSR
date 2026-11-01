#!/usr/bin/env bash
# ==============================================================================
# NAVSIM Evaluation Pipeline on GPUs 4, 5, 6, 7:
# 1. Multi-GPU sharded PDM Score evaluation (Planning metric)
# 2. Multi-GPU auxiliary Detection & HD-Map mAP evaluation (Perception heads)
# 3. Comprehensive Results Aggregation & Summary Report
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_ONE="${REPO}/scripts/evaluation/eval_para_ssr_distill_snapshot.sh"
EXPORT_PY="${REPO}/scripts/evaluation/export_student_ckpt.py"

# Python interpreter detection
if [[ -x "/venv/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/venv/ssr/bin/python}"
elif [[ -x "/root/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/root/miniconda3/envs/ssr/bin/python}"
elif [[ -x "/home/external-user/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/external-user/miniconda3/envs/ssr/bin/python}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

EXP_ROOT="${NAVSIM_EXP_ROOT_OVERRIDE:-${REPO}/work_dirs}"

# Target GPUs (Default: 4, 5, 6, 7)
GPU_LIST="${GPU_LIST:-${GPUS:-4,5,6,7}}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
NUM_GPUS="${#GPU_ARRAY[@]}"

# Stage-2 dual-distill checkpoint from training
DEFAULT_CKPT="${EXP_ROOT}/paradrive_distill_stage2_dual_distill/lightning_logs/version_0/checkpoints/last.ckpt"
CKPT_SRC="${CKPT_SRC:-${DEFAULT_CKPT}}"

SNAPSHOT_DIR="${SNAPSHOT_DIR:-${EXP_ROOT}/eval_snapshots/paradrive_distill_stage2_dual_distill_epoch29}"
SNAPSHOT_CKPT="${SNAPSHOT_CKPT:-${SNAPSHOT_DIR}/epoch29.ckpt}"
STUDENT_CKPT="${STUDENT_CKPT:-${SNAPSHOT_DIR}/epoch29_student.ckpt}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval/paradrive_distill_stage2_dual_distill_epoch29}"
MERGE_DIR="${EXP_ROOT}/${EXPERIMENT_NAME}"
AUX_EXPERIMENT="${AUX_EXPERIMENT:-${EXPERIMENT_NAME}_aux}"
AUX_CONFIG="${SNAPSHOT_DIR}/aux_training_config.yaml"

NAVTEST_LOGS="${NAVTEST_LOGS:-${REPO}/data/dataset/navsim_logs/test}"
NAVTEST_BLOBS="${NAVTEST_BLOBS:-${REPO}/data/dataset/sensor_blobs/test}"
METRIC_CACHE="${METRIC_CACHE:-${REPO}/data/exp/metric_cache}"

SMOKE="${SMOKE:-0}"
SKIP_PDM="${SKIP_PDM:-0}"
SKIP_AUX="${SKIP_AUX:-0}"
AUX_BATCH_SIZE="${AUX_BATCH_SIZE:-4}"
IMAGE_ARCHITECTURE="${IMAGE_ARCHITECTURE:-resnet50.tv_in1k}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash $0 [hydra overrides...]"
  echo "Env options:"
  echo "  GPU_LIST=4,5,6,7                Target GPUs (comma separated)"
  echo "  CKPT_SRC=/path/to/last.ckpt     Source Stage-2 checkpoint"
  echo "  EXPERIMENT_NAME=eval/my_eval    Experiment folder name under work_dirs"
  echo "  SKIP_PDM=1                      Skip PDM scoring (run only aux)"
  echo "  SKIP_AUX=1                      Skip aux evaluation (run only PDM)"
  echo "  SMOKE=1                         Run smoke test (small scenes for PDM)"
  exit 0
fi

if [[ ! -f "${CKPT_SRC}" ]]; then
  echo "Error: missing checkpoint: ${CKPT_SRC}" >&2
  exit 2
fi

echo "======================================================================"
echo " [Evaluation Pipeline] Stage-2 Dual Distill (Epoch 29 / last.ckpt)"
echo " Checkpoint Source : ${CKPT_SRC}"
echo " Allocated GPUs    : ${GPU_LIST} (${NUM_GPUS} GPUs)"
echo " Snapshot Dir      : ${SNAPSHOT_DIR}"
echo " Experiment Output : ${MERGE_DIR}"
echo " Python            : ${PYTHON}"
echo "======================================================================"

mkdir -p "${SNAPSHOT_DIR}" "${MERGE_DIR}"

# 1. Snapshot & Student Checkpoint Export
if [[ ! -f "${SNAPSHOT_CKPT}" ]]; then
  echo ">> Copying ${CKPT_SRC} -> ${SNAPSHOT_CKPT}..."
  cp -a "${CKPT_SRC}" "${SNAPSHOT_CKPT}"
else
  echo ">> Reusing snapshot: ${SNAPSHOT_CKPT}"
fi

if [[ ! -f "${STUDENT_CKPT}" ]]; then
  echo ">> Exporting student checkpoint (dropping teacher distill keys)..."
  "${PYTHON}" "${EXPORT_PY}" "${SNAPSHOT_CKPT}" "${STUDENT_CKPT}"
else
  echo ">> Reusing student checkpoint: ${STUDENT_CKPT}"
fi

# Parse extra Hydra args
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
EVAL_ARGS+=("agent.config.image_architecture=${IMAGE_ARCHITECTURE}")

# ------------------------------------------------------------------------------
# STEP 1: PDM Evaluation across GPUs
# ------------------------------------------------------------------------------
MERGED_CSV="${MERGE_DIR}/merged.csv"
if [[ "${SKIP_PDM}" == "1" ]]; then
  echo ">> SKIP_PDM=1: Skipping PDM evaluation."
else
  echo ""
  echo "======================================================================"
  echo " [Step 1/2] Running PDM Evaluation on ${NUM_GPUS} GPUs (${GPU_LIST})"
  echo "======================================================================"

  pids=()
  cleanup() {
    echo ">> Terminating child processes..."
    for pid in "${pids[@]:-}"; do
      kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
  }
  trap cleanup INT TERM

  run_shard() {
    local gpu="$1"
    local shard="$2"
    local log="$3"
    shift 3
    echo "   -> Starting shard ${shard}/${NUM_GPUS} on GPU ${gpu} (log: ${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHON="${PYTHON}" \
      CKPT_SRC="${SNAPSHOT_CKPT}" \
      SNAPSHOT_DIR="${SNAPSHOT_DIR}" \
      SNAPSHOT_CKPT="${SNAPSHOT_CKPT}" \
      STUDENT_CKPT="${STUDENT_CKPT}" \
      EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
      SHARDS="${NUM_GPUS}" \
      SHARD="${shard}" \
      SMOKE="${SMOKE}" \
      bash "${EVAL_ONE}" "$@" >"${log}" 2>&1
  }

  SHARD_DIRS=()
  for ((i=0; i<NUM_GPUS; i++)); do
    gpu="${GPU_ARRAY[i]}"
    shard_dir="${EXP_ROOT}/${EXPERIMENT_NAME}_shard${i}of${NUM_GPUS}"
    mkdir -p "${shard_dir}"
    SHARD_DIRS+=("${shard_dir}")
    log_file="${shard_dir}/run.log"
    run_shard "${gpu}" "${i}" "${log_file}" "${EVAL_ARGS[@]}" &
    pids+=("$!")
  done

  echo ">> Waiting for all ${NUM_GPUS} PDM shards to finish..."
  shard_fail=0
  for ((i=0; i<NUM_GPUS; i++)); do
    if ! wait "${pids[i]}"; then
      echo ">> Error: Shard ${i} (GPU ${GPU_ARRAY[i]}) failed!" >&2
      shard_fail=1
    fi
  done
  trap - INT TERM

  if [[ "${shard_fail}" -ne 0 ]]; then
    echo ">> PDM evaluation failed on one or more shards. Check run logs." >&2
    for ((i=0; i<NUM_GPUS; i++)); do
      log_file="${SHARD_DIRS[i]}/run.log"
      echo "----- Tail of ${log_file} -----" >&2
      tail -n 30 "${log_file}" >&2 || true
    done
    exit 1
  fi

  echo ">> All PDM shards completed successfully! Merging results..."
  "${PYTHON}" - "${SHARD_DIRS[@]}" "${MERGED_CSV}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

shard_dirs = [Path(p) for p in sys.argv[1:-1]]
out = Path(sys.argv[-1])
frames = []
used = []
for root in shard_dirs:
    csvs = sorted(
        (p for p in root.glob("*.csv") if p.name != "merged.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not csvs:
        raise SystemExit(f"no PDM csv found in {root}")
    path = csvs[-1]
    used.append(path)
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    if "token" not in df.columns:
        raise SystemExit(f"{path} has no token column")
    frames.append(df[df["token"] != "average"].copy())

merged = pd.concat(frames, ignore_index=True)
if merged["token"].duplicated().any():
    print("Warning: deduplicating overlapping tokens across shards")
    merged = merged.drop_duplicates(subset=["token"])

n = len(merged)
n_valid = int(merged["valid"].sum()) if "valid" in merged.columns else n
avg = merged.drop(columns=["token"], errors="ignore").mean(numeric_only=True, skipna=True)
avg_row = avg.to_dict()
avg_row["token"] = "average"
avg_row["valid"] = bool(n_valid == n)
out.parent.mkdir(parents=True, exist_ok=True)
pd.concat([merged, pd.DataFrame([avg_row])], ignore_index=True).to_csv(out, index=False)
print(f"Successfully merged {len(frames)} shards ({n} scenes) -> {out}")
PY
fi

# ------------------------------------------------------------------------------
# STEP 2: Auxiliary Tasks Evaluation (Detection & Map mAP)
# ------------------------------------------------------------------------------
AUX_JSON="${EXP_ROOT}/${AUX_EXPERIMENT}/aux_metrics.json"
if [[ "${SKIP_AUX}" == "1" ]]; then
  echo ">> SKIP_AUX=1: Skipping Aux evaluation."
else
  echo ""
  echo "======================================================================"
  echo " [Step 2/2] Running Detection & Map Aux Evaluation on GPUs ${GPU_LIST}"
  echo "======================================================================"

  # Generate aux config
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
            "agent.config.plan_score_file=null",
            f"agent.config.image_architecture={image_architecture}",
        ],
    )
out.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, out)
print(f"Wrote aux training config: {out} (image_architecture={image_architecture})")
PY

  SSR_NAVSIM_PYTHON="${PYTHON}" \
    GPU_IDS="${GPU_LIST}" \
    AUX_CHECKPOINT="${STUDENT_CKPT}" \
    AUX_TRAINING_CONFIG="${AUX_CONFIG}" \
    AUX_EXPERIMENT="${AUX_EXPERIMENT}" \
    AUX_BATCH_SIZE="${AUX_BATCH_SIZE}" \
    bash "${REPO}/scripts/evaluation/eval_para_ssr_aux.sh" \
      "navsim_log_path=${NAVTEST_LOGS}" \
      "sensor_blobs_path=${NAVTEST_BLOBS}"
fi

# ------------------------------------------------------------------------------
# STEP 3: Comprehensive Results Summary
# ------------------------------------------------------------------------------
SUMMARY_TXT="${MERGE_DIR}/evaluation_summary.txt"

"${PYTHON}" - "${MERGED_CSV}" "${AUX_JSON}" "${SUMMARY_TXT}" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

pdm_merged_csv = Path(sys.argv[1])
aux_json = Path(sys.argv[2])
summary_txt = Path(sys.argv[3])

lines = []
lines.append("=" * 72)
lines.append("  [NAVSIM EVALUATION COMPREHENSIVE SUMMARY - STAGE 2 DUAL DISTILL]")
lines.append("=" * 72)

if pdm_merged_csv.exists():
    df = pd.read_csv(pdm_merged_csv)
    avg_rows = df[df["token"] == "average"]
    if not avg_rows.empty:
        avg = avg_rows.iloc[0].to_dict()
    else:
        avg = df.drop(columns=["token"], errors="ignore").mean(numeric_only=True).to_dict()
    
    n_total = len(df[df["token"] != "average"])
    n_valid = int(df[df["token"] != "average"]["valid"].sum()) if "valid" in df.columns else n_total
    
    lines.append(" [1] Planning Performance (PDM Score)")
    lines.append("-" * 72)
    lines.append(f"  Total Scenarios      : {n_total} (Valid: {n_valid}, Failed: {n_total - n_valid})")
    if "score" in avg:
        lines.append(f"  PDM Score (PDMS)     : {float(avg['score']):.4f}  <-- Overall PDM Score")
    if "no_at_fault_collisions" in avg:
        lines.append(f"  No Collisions (NC)   : {float(avg['no_at_fault_collisions']):.4f}")
    if "drivable_area_compliance" in avg:
        lines.append(f"  Drivable Area (DAC)  : {float(avg['drivable_area_compliance']):.4f}")
    if "driving_direction_compliance" in avg:
        lines.append(f"  Driving Dir (DDC)    : {float(avg['driving_direction_compliance']):.4f}")
    if "ego_progress" in avg:
        lines.append(f"  Ego Progress (EP)    : {float(avg['ego_progress']):.4f}")
    if "time_to_collision_within_bound" in avg:
        lines.append(f"  Time-to-Coll (TTC)   : {float(avg['time_to_collision_within_bound']):.4f}")
    if "comfort" in avg:
        lines.append(f"  Comfort              : {float(avg['comfort']):.4f}")
    lines.append(f"  CSV Results          : {pdm_merged_csv}")
    lines.append("-" * 72)
else:
    lines.append(f" [1] Planning Performance: Merged CSV not found ({pdm_merged_csv})")

if aux_json.exists():
    data = json.loads(aux_json.read_text())
    metrics = data.get("metrics", {})
    det = metrics.get("detection", {})
    map_m = metrics.get("map", {})
    
    lines.append(" [2] Auxiliary Perception Performance (Detection & Map Heads)")
    lines.append("-" * 72)
    if "mAP" in det:
        lines.append(f"  Detection mAP        : {float(det['mAP']):.4f}")
        if "classes" in det:
            for cname, cinfo in sorted(det["classes"].items(), key=lambda x: -x[1].get("AP", 0)):
                lines.append(f"    - {cname:<17}: {float(cinfo.get('AP', 0)):.4f}")
    lines.append("")
    if "mAP" in map_m:
        lines.append(f"  Map mAP              : {float(map_m['mAP']):.4f}")
        if "classes" in map_m:
            for cname, cinfo in sorted(map_m["classes"].items(), key=lambda x: -x[1].get("AP", 0)):
                lines.append(f"    - {cname:<17}: {float(cinfo.get('AP', 0)):.4f}")
    lines.append(f"  JSON Results         : {aux_json}")
    lines.append("=" * 72)
else:
    lines.append(f" [2] Auxiliary Performance: JSON not found ({aux_json})")

output_str = "\n".join(lines)
print("\n" + output_str + "\n")
summary_txt.parent.mkdir(parents=True, exist_ok=True)
summary_txt.write_text(output_str + "\n")
print(f"Summary saved to: {summary_txt}")
PY
