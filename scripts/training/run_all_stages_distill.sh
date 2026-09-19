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
SAVE_TOP_K="${SAVE_TOP_K:-1}"  # Prune checkpoints: keep only latest 1 (+ last.ckpt) instead of all epochs

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

# Telemetry / W&B — this script only. Loads the personal key from ${REPO}/.env
# and pins entity to that account so runs do not land on the shared team.
WANDB_PROJECT="${WANDB_PROJECT:-para-ssr-distill}"
WANDB_GROUP="${WANDB_GROUP:-full-pipeline}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_PERSONAL_ENTITY:-comflife}"

# Python Environment (Auto-detect ssr conda environment)
if [[ -x "/home/external-user/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/external-user/miniconda3/envs/ssr/bin/python}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

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
  export WANDB_API_KEY="${key}"
  # Force the personal entity even if the parent shell exported the team.
  export WANDB_ENTITY
  unset WANDB_DISABLED
}

load_personal_wandb

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
echo " W&B          : enable=true  entity=${WANDB_ENTITY}  project=${WANDB_PROJECT}  group=${WANDB_GROUP}  (personal .env key)"
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
  local name="$1"
  WANDB_ARGS=(
    "wandb.enable=true"
    "wandb.mode=${WANDB_MODE_ARG}"
    "wandb.entity=${WANDB_ENTITY}"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.group=${WANDB_GROUP}"
    "wandb.name=${name}"
  )
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

    wandb_args "${EXP_1A}"
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
      checkpoint.save_top_k="${SAVE_TOP_K}" \
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

    wandb_args "${EXP_1B}"
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
      checkpoint.save_top_k="${SAVE_TOP_K}" \
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

wandb_args "${EXP_2}"
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
  checkpoint.save_top_k="${SAVE_TOP_K}" \
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
