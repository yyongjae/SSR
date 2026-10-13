#!/usr/bin/env bash
# ==============================================================================
# BEV Selector v8 Distillation Launch Script
#
# Methodology (docs/bev_selector_v8.md):
# - Student: km/para-ssr-v2 Multi-Task (Aux Det+Motion + Map + Task Interaction + 256 Anchors)
# - Backbone: ResNet-34 (timm/resnet34.tv_in1k)
# - Planner: 256 K-Means Trajectory Anchors + WTA Offset + Simulation Rewards
# - Balancer: Closed-loop GradBalancer across Student tasks (plan: 0.4, det: 0.3, map: 0.3)
# - Bug Fixes:
#     * Student Proj Optimizer Registration (student_proj trainable, shared QKV probes frozen)
#     * Distill Loss Scale Fix (0.5 * bank_loss, loss_distill_tok moved to telemetry, 3x bug eliminated)
# - Distillation v8:
#     * Hybrid Anchor Weighting: w_k = lambda_gt * w_gt(tau=5.0) + lambda_cov * (1/K) + lambda_risk * r_k
#     * Multi-Scale Relative Spatial Distance Bias: per-head sigma in (2.0, 4.0, 8.0, 16.0)m
#     * Teacher Attention Map Distillation (KL Divergence on Teacher's Cross-Attention weights)
#     * Attended Feature Distillation (Hybrid L2 + Cosine on Teacher & Student features)
#     * ReSMap Domain Loss Scaling (3.0x multiplier to resolve DAC gradient starvation)
#     * Telemetry: distill/effective_anchors tracked in W&B (target: 20~60 effective anchors)
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

# Python environment setup
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Teacher feature cache
if [[ -d "/workspace/teacher_cache" ]]; then
  DISTILL_FEATURE_ROOT="${DISTILL_FEATURE_ROOT:-/workspace/teacher_cache}"
else
  DISTILL_FEATURE_ROOT="${DISTILL_FEATURE_ROOT:-/home/external-user/datasets/teacher_cache}"
fi
export DISTILL_FEATURE_ROOT

EXPERIMENT="${EXPERIMENT:-paradrive_distill_bev_selector_v8_fix}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

# Maintain global batch size = 128
if [[ "${NUM_GPUS}" -eq 2 ]]; then
  ACCUMULATE="${ACCUMULATE:-16}"
elif [[ "${NUM_GPUS}" -eq 4 ]]; then
  ACCUMULATE="${ACCUMULATE:-8}"
else
  ACCUMULATE="${ACCUMULATE:-$((128 / (BATCH_SIZE * NUM_GPUS)))}"
fi

GLOBAL_BATCH=$((BATCH_SIZE * NUM_GPUS * ACCUMULATE))

# W&B Configuration
WANDB="${WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-e2ekd}"
WANDB_PROJECT="${WANDB_PROJECT:-v1_distill}"
WANDB_GROUP="${WANDB_GROUP:-bev-selector-v8}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-r34_sel_v8}"

_WANDB_NETRC_FILE=""
cleanup_wandb() {
  if [[ -n "${_WANDB_NETRC_FILE}" && -f "${_WANDB_NETRC_FILE}" ]]; then
    rm -f "${_WANDB_NETRC_FILE}"
  fi
}
trap cleanup_wandb EXIT

WANDB_ARGS=("wandb.enable=false")
if [[ "${WANDB}" != "0" ]]; then
  key=""
  if [[ -f "${REPO}/.env" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      if [[ "${line}" == WANDB_API_KEY=* ]]; then
        key="${line#WANDB_API_KEY=}"
        key="${key%\"}"
        key="${key#\"}"
        key="${key%\'}"
        key="${key#\'}"
      fi
    done < "${REPO}/.env"
  fi
  if [[ -n "${key}" ]]; then
    export WANDB_API_KEY="${key}"
    export WANDB_ENTITY
    _WANDB_NETRC_FILE="$(mktemp "${TMPDIR:-/tmp}/wandb-netrc.XXXXXX")"
    chmod 600 "${_WANDB_NETRC_FILE}"
    printf 'machine api.wandb.ai\n  login user\n  password %s\n' "${WANDB_API_KEY}" > "${_WANDB_NETRC_FILE}"
    export NETRC="${_WANDB_NETRC_FILE}"

    counter="${NAVSIM_EXP_ROOT}/.${WANDB_RUN_PREFIX}_counter"
    mkdir -p "${NAVSIM_EXP_ROOT}"
    trial="$(
      flock "${counter}.lock" bash -c '
        c=0
        if [[ -f "'"${counter}"'" ]]; then c=$(cat "'"${counter}"'"); fi
        n=$((c + 1))
        printf "%s\n" "$n" > "'"${counter}"'"
        printf "%s\n" "$n"
      '
    )"
    run_name="${WANDB_RUN_PREFIX}_${trial}"
    echo " W&B run      : ${WANDB_ENTITY}/${WANDB_PROJECT} (${run_name})"
    WANDB_ARGS=(
      "wandb.enable=true"
      "wandb.mode=${WANDB_MODE:-online}"
      "wandb.entity=${WANDB_ENTITY}"
      "wandb.project=${WANDB_PROJECT}"
      "wandb.group=${WANDB_GROUP}"
      "wandb.name=${run_name}"
      "wandb.tags=[bev-selector,r34,v8,aux-tasks,interaction,hybrid-anchors,multi-scale-bias]"
    )
  else
    echo "Notice: No WANDB_API_KEY found. Running with W&B disabled."
    WANDB_ARGS=("wandb.enable=false")
  fi
fi

echo "======================================================================"
echo " [BEV Selector v8] Hybrid Anchor Weighting & Multi-Scale Spatial Distance Bias"
echo " Experiment Name : ${EXPERIMENT}"
echo " Visible GPUs    : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo " Feature Cache   : ${DISTILL_FEATURE_ROOT}"
echo " Global Batch    : ${GLOBAL_BATCH} (Batch ${BATCH_SIZE} x GPUs ${NUM_GPUS} x Acc ${ACCUMULATE})"
echo " Max Epochs      : ${MAX_EPOCHS}"
echo " Student Arch    : ResNet-34 (Aux Det/Motion + Map + Task Interaction + 256 Anchors)"
echo " Distill Target  : BEVFusion (NC) + ReSMap 3.0x (DAC) with Multi-Scale Spatial Bias & Attn KD"
echo "======================================================================"

exec "${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_selector_v8_agent \
  agent.lr="${LR}" \
  agent.config.max_epochs="${MAX_EPOCHS}" \
  agent.config.distill_feature_root="${DISTILL_FEATURE_ROOT}" \
  agent.config.plan_heading_from_xy=true \
  experiment_name="${EXPERIMENT}" \
  scene_filter=navtrain \
  split=trainval \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers=8 \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.accumulate_grad_batches="${ACCUMULATE}" \
  trainer.params.check_val_every_n_epoch=5 \
  trainer.params.precision=32 \
  trainer.params.strategy="ddp_find_unused_parameters_true" \
  +trainer.params.devices="${NUM_GPUS}" \
  checkpoint.every_n_epochs=5 \
  checkpoint.save_top_k=-1 \
  checkpoint.save_last=true \
  trainer.params.gradient_clip_val=35.0 \
  trainer.params.gradient_clip_algorithm=norm \
  "${WANDB_ARGS[@]}" \
  "$@"
