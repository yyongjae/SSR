#!/usr/bin/env bash
# ==============================================================================
# PARA-SSR Pure Student Training: PLAN ONLY (No aux heads, no task interaction)
#
# Exact reproduction of `km/para-ssr-v2` head ablation baseline (Report 13 & 22):
# - det+motion head: OFF
# - map head       : OFF
# - task interaction: OFF
# - grad_balance   : null (single planning task)
# - plan_anchor    : false (single-trajectory dense BEV planner)
#
# Batch size / Accumulation recipe:
#   2 GPUs: batch=4 x 2 GPUs x accumulate=16 = 128 global batch
#   4 GPUs: batch=4 x 4 GPUs x accumulate=8  = 128 global batch
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Ensure Python interpreter from ssr environment
if [[ -x "/venv/ssr/bin/python" ]]; then
  PYTHON="/venv/ssr/bin/python"
  export PATH="/venv/ssr/bin:${PATH}"
elif [[ -x "/root/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="/root/miniconda3/envs/ssr/bin/python"
  export PATH="/root/miniconda3/envs/ssr/bin:${PATH}"
elif [[ -x "/home/external-user/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="/home/external-user/miniconda3/envs/ssr/bin/python"
  export PATH="/home/external-user/miniconda3/envs/ssr/bin:${PATH}"
else
  PYTHON="python"
fi

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT_OVERRIDE:-${REPO}/work_dirs}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# OpenMP single-threaded workers
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# GPU allocation (Default: GPUs 4,5,6,7)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

EXPERIMENT="${EXPERIMENT:-para_ssr_plan_only}"
BATCH_SIZE="${BATCH_SIZE:-4}"
# Keep global batch = 128
if [[ "${NUM_GPUS}" -eq 4 ]]; then
  ACCUMULATE="${ACCUMULATE:-8}"
elif [[ "${NUM_GPUS}" -eq 2 ]]; then
  ACCUMULATE="${ACCUMULATE:-16}"
else
  ACCUMULATE="${ACCUMULATE:-$((128 / (BATCH_SIZE * NUM_GPUS)))}"
fi

MAX_EPOCHS="${MAX_EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
LR="${LR:-1e-4}"

# W&B Configuration
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-para-ssr}"
WANDB_GROUP="${WANDB_GROUP:-para-navsim}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"

GLOBAL_BATCH=$((BATCH_SIZE * NUM_GPUS * ACCUMULATE))
echo "======================================================================"
echo " [Pure Student Plan-Only] Training Launch"
echo " Experiment Name : ${EXPERIMENT}"
echo " Visible GPUs    : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo " Batch / GPU     : ${BATCH_SIZE}"
echo " Accumulate Grad : ${ACCUMULATE} (Global Batch: ${GLOBAL_BATCH})"
echo " Max Epochs      : ${MAX_EPOCHS}"
echo " Learning Rate   : ${LR}"
echo "======================================================================"

WANDB_ARGS=()
if [[ "${WANDB}" != "0" ]]; then
  WANDB_ARGS+=(
    "wandb.enable=true"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.group=${WANDB_GROUP}"
    "wandb.mode=${WANDB_MODE_ARG}"
    "wandb.name=${EXPERIMENT}"
    "wandb.tags=[para-ssr,navsim,plan-only,global${GLOBAL_BATCH},${MAX_EPOCHS}ep]"
  )
else
  WANDB_ARGS+=("wandb.enable=false")
fi

exec "${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_agent \
  agent.lr="${LR}" \
  agent.config.max_epochs="${MAX_EPOCHS}" \
  agent.config.use_task_interaction=false \
  agent.config.use_det_motion_head=false \
  agent.config.use_map_head=false \
  ~agent.config.grad_balance_target \
  +agent.config.grad_balance_target=null \
  agent.config.plan_anchor=false \
  agent.config.plan_anchor_file=null \
  agent.config.plan_score_file=null \
  experiment_name="${EXPERIMENT}" \
  scene_filter=navtrain \
  split=trainval \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${WORKERS}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.accumulate_grad_batches="${ACCUMULATE}" \
  trainer.params.check_val_every_n_epoch="${VAL_EVERY}" \
  trainer.params.precision=32 \
  +trainer.params.devices="${NUM_GPUS}" \
  checkpoint.every_n_epochs=5 \
  checkpoint.save_top_k=-1 \
  checkpoint.save_last=true \
  trainer.params.gradient_clip_val=35.0 \
  trainer.params.gradient_clip_algorithm=norm \
  "${WANDB_ARGS[@]}" \
  "$@"
