#!/usr/bin/env bash
# ==============================================================================
# Stage 1 Planning Distillation: ReSMap Adapter Training
# Trains a per-cell residual BEV adapter on frozen ReSMap online HD-map features.
# No sensor images loaded; fast, lightweight training using cached teacher BEVs.
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

# Network & Threading
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
SAVE_TOP_K="${SAVE_TOP_K:-1}"

# Python Environment (Auto-detect ssr conda environment)
if [[ -x "/home/external-user/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/external-user/miniconda3/envs/ssr/bin/python}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

# Teacher Cache Settings
export DISTILL_FEATURE_ROOT="${DISTILL_FEATURE_ROOT:-/home/external-user/datasets/teacher_cache}"
export TEACHER_ADAPTER_BRANCH="resmap"

EXPERIMENT="${EXPERIMENT:-stage1_adapter_resmap}"
BATCH_SIZE="${BATCH_SIZE:-16}"       # Teacher features are lightweight (no camera backbone)
ACCUMULATE="${ACCUMULATE:-4}"        # 16 x 2 GPUs x 4 = 128 global batch
MAX_EPOCHS="${MAX_EPOCHS:-20}"
WORKERS="${WORKERS:-8}"
LR="${LR:-2e-4}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-para-ssr-distill}"
WANDB_GROUP="${WANDB_GROUP:-stage1-adapters}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"
GLOBAL_BATCH=$((BATCH_SIZE * NUM_GPUS * ACCUMULATE))

echo "======================================================================"
echo "Stage 1: Training ReSMap Adapter for Planning Distillation"
echo "Teacher Branch : ${TEACHER_ADAPTER_BRANCH}"
echo "Feature Root   : ${DISTILL_FEATURE_ROOT}"
echo "GPUs           : ${NUM_GPUS} (${CUDA_VISIBLE_DEVICES})"
echo "Batch size/GPU : ${BATCH_SIZE}, Accumulate: ${ACCUMULATE} -> Global: ${GLOBAL_BATCH}"
echo "======================================================================"

WANDB_ARGS=()
if [[ "${WANDB}" != "0" ]]; then
  WANDB_ARGS+=(
    "wandb.enable=true"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.group=${WANDB_GROUP}"
    "wandb.mode=${WANDB_MODE_ARG}"
    "wandb.name=${EXPERIMENT}"
    "wandb.tags=[stage1,resmap,adapter,global${GLOBAL_BATCH},${MAX_EPOCHS}ep]"
  )
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS+=("resume_checkpoint=${RESUME_CHECKPOINT}")
fi

"${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_teacher_adapter_agent \
  agent.lr="${LR}" \
  agent.config.max_epochs="${MAX_EPOCHS}" \
  agent.config.teacher_adapter_branch="resmap" \
  agent.config.distill_feature_root="${DISTILL_FEATURE_ROOT}" \
  experiment_name="${EXPERIMENT}" \
  scene_filter=navtrain \
  split=trainval \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${WORKERS}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.accumulate_grad_batches="${ACCUMULATE}" \
  trainer.params.check_val_every_n_epoch=2 \
  trainer.params.precision=32 \
  +trainer.params.devices="${NUM_GPUS}" \
  checkpoint.save_top_k="${SAVE_TOP_K}" \
  trainer.params.gradient_clip_val=35.0 \
  trainer.params.gradient_clip_algorithm=norm \
  "${WANDB_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "$@"
