#!/usr/bin/env bash
# ==============================================================================
# End-to-End Planning-Centric Distillation Pipeline (Stages 1A -> 1B -> 2)
#
# Stage 1A: Train BEVFusion 3DOD BEV Adapter (frozen teacher cache)
# Stage 1B: Train ReSMap Online HD-Map BEV Adapter (frozen teacher cache)
# Stage 2 : Train Student Agent with Dual-Teacher Distillation & Corridor Mask
#
# Default GPUs: 4,5 (RTX 5090 32GB x 2)
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"

# GPU Allocation (Default: GPU 4,5)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# Network & Threading: Single-host loopback avoids NCCL socket connection errors
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Teacher Cache Path
export DISTILL_FEATURE_ROOT="${DISTILL_FEATURE_ROOT:-/home/external-user/datasets/teacher_cache}"

# Experiment Naming & Run Control
EXP_PREFIX="${EXP_PREFIX:-paradrive_distill}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
# Keep a weight file every 5 epochs (epoch=4,9,14,... in Lightning's
# 0-based names) plus last.ckpt. save_top_k=-1 does not prune those files.
SAVE_TOP_K="${SAVE_TOP_K:--1}"
EVERY_N_EPOCHS="${EVERY_N_EPOCHS:-5}"

# Optional checkpoints override (if skipping or reusing existing stage 1 runs)
BEVFUSION_CKPT="${BEVFUSION_ADAPTER_CKPT:-}"
RESMAP_CKPT="${RESMAP_ADAPTER_CKPT:-}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
ONLY_STAGE="${ONLY_STAGE:-}"  # options: stage1a, stage1b, stage2
# Re-run a finished stage instead of auto-skipping its checkpoint.
# 1 / true / all  -> every selected stage
# stage1a,stage1b,stage2 -> only those
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
EXP_1A="${EXP_PREFIX}_stage1_bevfusion"
EXP_1B="${EXP_PREFIX}_stage1_resmap"
EXP_2="${EXP_PREFIX}_stage2_dual_distill"

# Telemetry. TensorBoard under lightning_logs/ is always on.
# W&B uploads to the e2ekd team by default. WANDB=0 turns that off.
# The key is read from ${REPO}/.env for this process only. --team is accepted
# and stripped so Hydra never sees it; the entity is already e2ekd.
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-v1_distill}"
WANDB_GROUP="${WANDB_GROUP:-full-pipeline}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-e2ekd}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-r50_dtd_trial}"
USER_ARGS=()
for _arg in "$@"; do
  if [[ "${_arg}" == "--team" ]]; then
    WANDB_ENTITY="e2ekd"
  else
    USER_ARGS+=("${_arg}")
  fi
done
if ((${#USER_ARGS[@]})); then
  set -- "${USER_ARGS[@]}"
else
  set --
fi
unset USER_ARGS _arg

# Python Environment (Auto-detect ssr conda environment)
if [[ -x "/home/external-user/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/external-user/miniconda3/envs/ssr/bin/python}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

# Private netrc for this process only. wandb writes API keys to $NETRC
# (default ~/.netrc). On this shared account that file is the login everyone
# else inherits, so a WANDB=1 run must not touch it.
_WANDB_NETRC_FILE=""
cleanup_personal_wandb() {
  if [[ -n "${_WANDB_NETRC_FILE}" && -f "${_WANDB_NETRC_FILE}" ]]; then
    rm -f "${_WANDB_NETRC_FILE}"
  fi
  _WANDB_NETRC_FILE=""
}

load_personal_wandb() {
  local env_file="${REPO}/.env"
  local line key=""
  if [[ ! -f "${env_file}" ]]; then
    echo "Error: ${env_file} not found; this script needs WANDB_API_KEY there." >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    if [[ "${line}" == WANDB_API_KEY=* ]]; then
      key="${line#WANDB_API_KEY=}"
      key="${key#\"}"; key="${key%\"}"
      key="${key#\'}"; key="${key%\'}"
    fi
  done < "${env_file}"
  if [[ -z "${key}" ]]; then
    echo "Error: WANDB_API_KEY is empty in ${env_file}" >&2
    exit 1
  fi
  # Process environment only. Do not call `wandb login`: that writes ~/.netrc.
  export WANDB_API_KEY="${key}"
  # Team entity e2ekd. The key stays in this process and in the temp netrc.
  export WANDB_ENTITY
  unset WANDB_DISABLED
  _WANDB_NETRC_FILE="$(mktemp "${TMPDIR:-/tmp}/wandb-netrc.XXXXXX")"
  chmod 600 "${_WANDB_NETRC_FILE}"
  printf 'machine api.wandb.ai\n  login user\n  password %s\n' "${key}" > "${_WANDB_NETRC_FILE}"
  export NETRC="${_WANDB_NETRC_FILE}"
  trap cleanup_personal_wandb EXIT
  unset key
}

if [[ "${WANDB}" != "0" ]]; then
  load_personal_wandb
fi

# One number per stage that actually starts. Display name is
# r50_dtd_trial_<n>. The next n is one past the highest name already in
# e2ekd/v1_distill and the local counter, so a second launch does not reuse it.
alloc_wandb_trial() {
  if [[ -n "${WANDB_TRIAL:-}" ]]; then
    echo "${WANDB_TRIAL}"
    return
  fi
  WANDB_SILENT=true "${PYTHON}" -c "$(cat <<'PY'
import re
import sys
from pathlib import Path

entity, project, prefix, root = sys.argv[1:5]
counter = Path(root) / f".{prefix}_counter"
counter.parent.mkdir(parents=True, exist_ok=True)
lock_path = counter.with_suffix(".lock")

import fcntl
with open(lock_path, "a") as lockf:
    fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
    local = 0
    if counter.exists():
        text = counter.read_text().strip()
        if text.isdigit():
            local = int(text)
    remote = 0
    try:
        import wandb
        api = wandb.Api(timeout=30)
        pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        for run in api.runs(f"{entity}/{project}"):
            match = pat.fullmatch(run.name or "")
            if match:
                remote = max(remote, int(match.group(1)))
    except Exception as exc:
        # A new project is created on the first upload, so a missing project
        # is the normal first-run case. Other failures still fall back locally.
        if "could not find project" not in str(exc).lower():
            print(
                f"W&B name scan failed ({type(exc).__name__}: {exc}); using the local counter.",
                file=sys.stderr,
            )
    nxt = max(local, remote) + 1
    counter.write_text(f"{nxt}\n")
    print(nxt)
PY
)" "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${WANDB_RUN_PREFIX}" "${NAVSIM_EXP_ROOT}"
}

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

echo "======================================================================"
echo " [All-in-One] Planning Distillation Pipeline Launched"
echo " Time         : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Working Dir  : ${REPO}"
echo " Python       : ${PYTHON}"
echo " Visible GPUs : ${CUDA_VISIBLE_DEVICES} (Count: ${NUM_GPUS})"
echo " Teacher Cache: ${DISTILL_FEATURE_ROOT}"
echo " Exp Prefix   : ${EXP_PREFIX}"
echo " Only stage   : ${ONLY_STAGE:-all}"
echo " Force retrain: ${FORCE_RETRAIN}"
if [[ "${WANDB}" != "0" ]]; then
  echo " W&B          : enable=true  entity=${WANDB_ENTITY}  project=${WANDB_PROJECT}  group=${WANDB_GROUP}  name=${WANDB_RUN_PREFIX}_<n>  (one-shot .env key)"
else
  echo " W&B          : disabled  (TensorBoard: work_dirs/<exp>/lightning_logs)"
fi
echo "======================================================================"

# Helper to find latest checkpoint in an experiment directory
find_checkpoint() {
  local exp_dir="$1"
  local found=""
  if [[ ! -d "${exp_dir}" ]]; then
    echo ""
    return 0
  fi
  if [[ -f "${exp_dir}/checkpoints/last.ckpt" ]]; then
    found="${exp_dir}/checkpoints/last.ckpt"
  else
    local last_f
    last_f=$(find "${exp_dir}" -name "last.ckpt" -type f 2>/dev/null | head -n 1 || true)
    if [[ -n "${last_f}" ]]; then
      found="${last_f}"
    else
      found=$(find "${exp_dir}" -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}' || true)
    fi
  fi
  echo "${found}"
}

# Skip only when the last epoch was actually written. A crash after epoch 0
# leaves last.ckpt, which must not count as a finished stage.
find_completed_checkpoint() {
  local exp_dir="$1"
  local max_epochs="$2"
  local last_epoch=$((max_epochs - 1))
  if [[ ! -d "${exp_dir}" ]]; then
    echo ""
    return 0
  fi
  local completed
  completed=$(find "${exp_dir}" -name "epoch=${last_epoch}-*.ckpt" -type f 2>/dev/null | head -n 1 || true)
  if [[ -z "${completed}" ]]; then
    echo ""
    return 0
  fi
  local last
  last=$(find "${exp_dir}" -name "last.ckpt" -type f 2>/dev/null | head -n 1 || true)
  echo "${last:-${completed}}"
}

stage_forced() {
  local stage="$1"
  local flag
  flag="$(echo "${FORCE_RETRAIN}" | tr '[:upper:]' '[:lower:]')"
  case ",${flag}," in
    *,1,*|*,true,*|*,all,*|*,"${stage}",*) return 0 ;;
  esac
  return 1
}

stage_selected() {
  local stage="$1"
  [[ -z "${ONLY_STAGE}" || "${ONLY_STAGE}" == "${stage}" ]]
}

wandb_args() {
  local stage="$1"
  if [[ "${WANDB}" != "0" ]]; then
    local name
    name="${WANDB_RUN_PREFIX}_$(alloc_wandb_trial)"
    if [[ ! "${name}" =~ ^${WANDB_RUN_PREFIX}_[0-9]+$ ]]; then
      echo "Error: W&B run name must look like ${WANDB_RUN_PREFIX}_<number>, got '${name}'." >&2
      exit 1
    fi
    echo " W&B run      : ${WANDB_ENTITY}/${WANDB_PROJECT}  ${name}  (${stage})"
    WANDB_ARGS=(
      "wandb.enable=true"
      "wandb.mode=${WANDB_MODE_ARG}"
      "wandb.entity=${WANDB_ENTITY}"
      "wandb.project=${WANDB_PROJECT}"
      "wandb.group=${WANDB_GROUP}"
      "wandb.name=${name}"
      "wandb.tags=[${stage},r50,dtd]"
    )
  else
    WANDB_ARGS=("wandb.enable=false")
  fi
}

if [[ -z "${BEVFUSION_CKPT}" || ! -f "${BEVFUSION_CKPT}" ]]; then
  BEVFUSION_CKPT="$(find_completed_checkpoint "${NAVSIM_EXP_ROOT}/${EXP_1A}" "${STAGE1_EPOCHS}")"
fi
if [[ -z "${RESMAP_CKPT}" || ! -f "${RESMAP_CKPT}" ]]; then
  RESMAP_CKPT="$(find_completed_checkpoint "${NAVSIM_EXP_ROOT}/${EXP_1B}" "${STAGE1_EPOCHS}")"
fi
STAGE2_CKPT="$(find_completed_checkpoint "${NAVSIM_EXP_ROOT}/${EXP_2}" "${STAGE2_EPOCHS}")"

echo ""
echo " Checkpoint scan:"
echo "   Stage 1A ${EXP_1A}: ${BEVFUSION_CKPT:-<missing>}"
echo "   Stage 1B ${EXP_1B}: ${RESMAP_CKPT:-<missing>}"
echo "   Stage 2  ${EXP_2}: ${STAGE2_CKPT:-<missing>}"
echo " Re-run a finished stage with FORCE_RETRAIN=1 or FORCE_RETRAIN=stage1b"
echo ""

# ------------------------------------------------------------------------------
# STAGE 1A: BEVFusion Adapter Training
# ------------------------------------------------------------------------------
if stage_selected stage1a; then
  if [[ "${SKIP_STAGE1}" == "1" ]]; then
    echo ">>> [Stage 1A] SKIP_STAGE1=1 is set, skipping BEVFusion adapter training."
  elif [[ -n "${BEVFUSION_CKPT}" && -f "${BEVFUSION_CKPT}" ]] && ! stage_forced stage1a; then
    echo ">>> [Stage 1A] Already done, skipping. Checkpoint: ${BEVFUSION_CKPT}"
  else
    echo ""
    echo "======================================================================"
    echo " >>> [Stage 1A/3] Training BEVFusion Adapter (${STAGE1_EPOCHS} epochs)"
    echo " Experiment: ${EXP_1A}"
    echo "======================================================================"

    wandb_args stage1a
    "${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
      agent=para_ssr_teacher_adapter_agent \
      agent.config.teacher_adapter_branch="bevfusion" \
      agent.config.distill_feature_root="${DISTILL_FEATURE_ROOT}" \
      agent.config.max_epochs="${STAGE1_EPOCHS}" \
      agent.lr="2e-4" \
      experiment_name="${EXP_1A}" \
      scene_filter=navtrain \
      split=trainval \
      dataloader.params.batch_size=16 \
      dataloader.params.num_workers=8 \
      trainer.params.max_epochs="${STAGE1_EPOCHS}" \
      trainer.params.accumulate_grad_batches=4 \
      trainer.params.check_val_every_n_epoch=2 \
      trainer.params.precision=32 \
      +trainer.params.devices="${NUM_GPUS}" \
      checkpoint.every_n_epochs="${EVERY_N_EPOCHS}" \
      checkpoint.save_top_k="${SAVE_TOP_K}" \
      checkpoint.save_last=true \
      trainer.params.gradient_clip_val=35.0 \
      trainer.params.gradient_clip_algorithm=norm \
      "${WANDB_ARGS[@]}" \
      "$@"

    BEVFUSION_CKPT="$(find_checkpoint "${NAVSIM_EXP_ROOT}/${EXP_1A}")"
    echo ">>> [Stage 1A] Completed! Checkpoint found: ${BEVFUSION_CKPT}"
  fi
fi

if [[ "${ONLY_STAGE}" == "stage1a" ]]; then
  echo ">>> Finished requested ONLY_STAGE=stage1a."
  exit 0
fi

# ------------------------------------------------------------------------------
# STAGE 1B: ReSMap Adapter Training
# ------------------------------------------------------------------------------
if stage_selected stage1b; then
  if [[ "${SKIP_STAGE1}" == "1" ]]; then
    echo ">>> [Stage 1B] SKIP_STAGE1=1 is set, skipping ReSMap adapter training."
  elif [[ -n "${RESMAP_CKPT}" && -f "${RESMAP_CKPT}" ]] && ! stage_forced stage1b; then
    echo ">>> [Stage 1B] Already done, skipping. Checkpoint: ${RESMAP_CKPT}"
  else
    echo ""
    echo "======================================================================"
    echo " >>> [Stage 1B/3] Training ReSMap Adapter (${STAGE1_EPOCHS} epochs)"
    echo " Experiment: ${EXP_1B}"
    echo "======================================================================"

    wandb_args stage1b
    "${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
      agent=para_ssr_teacher_adapter_agent \
      agent.config.teacher_adapter_branch="resmap" \
      agent.config.distill_feature_root="${DISTILL_FEATURE_ROOT}" \
      agent.config.max_epochs="${STAGE1_EPOCHS}" \
      agent.lr="2e-4" \
      experiment_name="${EXP_1B}" \
      scene_filter=navtrain \
      split=trainval \
      dataloader.params.batch_size=16 \
      dataloader.params.num_workers=8 \
      trainer.params.max_epochs="${STAGE1_EPOCHS}" \
      trainer.params.accumulate_grad_batches=4 \
      trainer.params.check_val_every_n_epoch=2 \
      trainer.params.precision=32 \
      +trainer.params.devices="${NUM_GPUS}" \
      checkpoint.every_n_epochs="${EVERY_N_EPOCHS}" \
      checkpoint.save_top_k="${SAVE_TOP_K}" \
      checkpoint.save_last=true \
      trainer.params.gradient_clip_val=35.0 \
      trainer.params.gradient_clip_algorithm=norm \
      "${WANDB_ARGS[@]}" \
      "$@"

    RESMAP_CKPT="$(find_checkpoint "${NAVSIM_EXP_ROOT}/${EXP_1B}")"
    echo ">>> [Stage 1B] Completed! Checkpoint found: ${RESMAP_CKPT}"
  fi
fi

if [[ "${ONLY_STAGE}" == "stage1b" ]]; then
  echo ">>> Finished requested ONLY_STAGE=stage1b."
  exit 0
fi

# ------------------------------------------------------------------------------
# STAGE 2: Dual-Teacher Planning Distillation Student Training
# ------------------------------------------------------------------------------
if [[ -n "${STAGE2_CKPT}" && -f "${STAGE2_CKPT}" ]] && ! stage_forced stage2; then
  echo ">>> [Stage 2] Already done, skipping. Checkpoint: ${STAGE2_CKPT}"
  printf '%s\n' \
    '' \
    '======================================================================' \
    ' [All-in-One] Pipeline Finished Successfully!' \
    " Final Student Model Checkpoints: ${STAGE2_CKPT}" \
    " Time: $(date '+%Y-%m-%d %H:%M:%S')" \
    '======================================================================'
  exit 0
fi

echo ""
echo "======================================================================"
echo " >>> [Stage 2/3] Full Student Training with Dual Distillation"
echo " BEVFusion Adapter Checkpoint: ${BEVFUSION_CKPT}"
echo " ReSMap Adapter Checkpoint   : ${RESMAP_CKPT}"
echo "======================================================================"

if [[ -z "${BEVFUSION_CKPT}" || ! -f "${BEVFUSION_CKPT}" ]]; then
  echo "Error: Stage 2 requires valid BEVFUSION_ADAPTER_CKPT, but got: '${BEVFUSION_CKPT}'" >&2
  exit 1
fi

if [[ -z "${RESMAP_CKPT}" || ! -f "${RESMAP_CKPT}" ]]; then
  echo "Error: Stage 2 requires valid RESMAP_ADAPTER_CKPT, but got: '${RESMAP_CKPT}'" >&2
  exit 1
fi

export BEVFUSION_ADAPTER_CKPT="${BEVFUSION_CKPT}"
export RESMAP_ADAPTER_CKPT="${RESMAP_CKPT}"

wandb_args stage2
"${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_distill_agent \
  agent.lr="1e-4" \
  agent.config.max_epochs="${STAGE2_EPOCHS}" \
  agent.config.distill_feature_root="${DISTILL_FEATURE_ROOT}" \
  agent.config.distill_adapter_checkpoints.bevfusion="${BEVFUSION_CKPT}" \
  agent.config.distill_adapter_checkpoints.resmap="${RESMAP_CKPT}" \
  agent.config.use_corridor_mask=true \
  agent.config.use_stl=false \
  agent.config.plan_num_layers=3 \
  experiment_name="${EXP_2}" \
  scene_filter=navtrain \
  split=trainval \
  dataloader.params.batch_size=4 \
  dataloader.params.num_workers=8 \
  trainer.params.max_epochs="${STAGE2_EPOCHS}" \
  trainer.params.accumulate_grad_batches=16 \
  trainer.params.check_val_every_n_epoch=5 \
  trainer.params.precision=32 \
  +trainer.params.devices="${NUM_GPUS}" \
  checkpoint.every_n_epochs="${EVERY_N_EPOCHS}" \
  checkpoint.save_top_k="${SAVE_TOP_K}" \
  checkpoint.save_last=true \
  trainer.params.gradient_clip_val=35.0 \
  trainer.params.gradient_clip_algorithm=norm \
  "${WANDB_ARGS[@]}" \
  "$@"

STAGE2_FINAL="$(find_checkpoint "${NAVSIM_EXP_ROOT}/${EXP_2}")"
printf '%s\n' \
  '' \
  '======================================================================' \
  ' [All-in-One] Pipeline Finished Successfully!' \
  " Final Student Model Checkpoints: ${STAGE2_FINAL:-${NAVSIM_EXP_ROOT}/${EXP_2}}" \
  " Time: $(date '+%Y-%m-%d %H:%M:%S')" \
  '======================================================================'
