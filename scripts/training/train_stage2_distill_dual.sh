#!/usr/bin/env bash
# ==============================================================================
# Stage 2 Planning-Centric Distillation: Dual Teachers (BEVFusion + ReSMap)
# Student: Dense BEV Direct Cross-Attention (no TokenLearner), Kinematics Conditioning
# Teachers: Frozen BEVFusion (3DOD) + Frozen ReSMap (Online HD Map)
# Distillation: Trajectory-Conditioned Gaussian Driving Corridor Masking
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

# Teacher Cache and Checkpoints
export DISTILL_FEATURE_ROOT="${DISTILL_FEATURE_ROOT:-/home/external-user/datasets/teacher_cache}"
BEVFUSION_CKPT="${BEVFUSION_ADAPTER_CKPT:-${REPO}/work_dirs/stage1_adapter_bevfusion/checkpoints/best.ckpt}"
RESMAP_CKPT="${RESMAP_ADAPTER_CKPT:-${REPO}/work_dirs/stage1_adapter_resmap/checkpoints/best.ckpt}"

export BEVFUSION_ADAPTER_CKPT="${BEVFUSION_CKPT}"
export RESMAP_ADAPTER_CKPT="${RESMAP_CKPT}"

EXPERIMENT="${EXPERIMENT:-stage2_dual_distill_corridor}"
BATCH_SIZE="${BATCH_SIZE:-4}"        # per GPU (BEVFormer encoder + dense planner)
ACCUMULATE="${ACCUMULATE:-16}"       # 4 x 2 GPUs x 16 = 128 global batch
MAX_EPOCHS="${MAX_EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
LR="${LR:-1e-4}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-para-ssr-distill}"
WANDB_GROUP="${WANDB_GROUP:-stage2-dual-distill}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"
GLOBAL_BATCH=$((BATCH_SIZE * NUM_GPUS * ACCUMULATE))

echo "======================================================================"
echo "Stage 2: Training Student with Dual-Teacher BEV Corridor Distillation"
echo "Feature Root        : ${DISTILL_FEATURE_ROOT}"
echo "BEVFusion Adapter   : ${BEVFUSION_ADAPTER_CKPT}"
echo "ReSMap Adapter      : ${RESMAP_ADAPTER_CKPT}"
echo "Corridor Masking    : Enabled (Trajectory Gaussian Weighting)"
echo "GPUs                : ${NUM_GPUS} (${CUDA_VISIBLE_DEVICES})"
echo "Batch size/GPU      : ${BATCH_SIZE}, Accumulate: ${ACCUMULATE} -> Global: ${GLOBAL_BATCH}"
echo "======================================================================"

WANDB_ARGS=()
if [[ "${WANDB}" != "0" ]]; then
  WANDB_ARGS+=(
    "wandb.enable=true"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.group=${WANDB_GROUP}"
    "wandb.mode=${WANDB_MODE_ARG}"
    "wandb.name=${EXPERIMENT}"
    "wandb.tags=[stage2,dual-distill,bevfusion,resmap,corridor-mask,dense-cross-attn,global${GLOBAL_BATCH},${MAX_EPOCHS}ep]"
  )
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS+=("resume_checkpoint=${RESUME_CHECKPOINT}")
fi

"${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_distill_agent \
  agent.lr="${LR}" \
  agent.config.max_epochs="${MAX_EPOCHS}" \
  agent.config.distill_feature_root="${DISTILL_FEATURE_ROOT}" \
  agent.config.distill_adapter_checkpoints.bevfusion="${BEVFUSION_ADAPTER_CKPT}" \
  agent.config.distill_adapter_checkpoints.resmap="${RESMAP_ADAPTER_CKPT}" \
  agent.config.use_corridor_mask=true \
  agent.config.use_stl=false \
  agent.config.plan_num_layers=3 \
  experiment_name="${EXPERIMENT}" \
  scene_filter=navtrain \
  split=trainval \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${WORKERS}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.accumulate_grad_batches="${ACCUMULATE}" \
  trainer.params.check_val_every_n_epoch=5 \
  trainer.params.precision=32 \
  +trainer.params.devices="${NUM_GPUS}" \
  checkpoint.save_top_k="${SAVE_TOP_K}" \
  trainer.params.gradient_clip_val=35.0 \
  trainer.params.gradient_clip_algorithm=norm \
  "${WANDB_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "$@"
