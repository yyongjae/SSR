#!/usr/bin/env bash
# ==============================================================================
# Pure Student Training: PARA-SSR v2 (ResNet-34 + Aux Heads + 256 Anchor Planner)
#
# Exact reproduction of `km/para-ssr-v2` (`para_ssr_interaction_v2` / commit 0f20868):
# - Backbone          : ResNet-34 (timm/resnet34.tv_in1k)
# - Det+Motion Head   : ON (300 queries)
# - Map Head          : ON (100 vectors)
# - Task Interaction  : ON (Cross-Attention across Det, Motion, Map memories)
# - 256 Anchor Planner: ON (WoTE-style 256 K-Means anchors + WTA offset + PDM simulation rewards)
# - Closed-loop Balancer: ON (plan: 0.4, det: 0.3, map: 0.3)
# - Distillation      : OFF (Pure Student baseline, target PDMS: ~0.88)
#
# Default Hardware:
#   GPUs 4, 5 (2 GPUs x batch 4 x accumulate 16 = global batch 128)
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
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# GPU allocation (Default: GPUs 4, 5)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

EXPERIMENT="${EXPERIMENT:-para_ssr_v2_r34_student}"
BATCH_SIZE="${BATCH_SIZE:-4}"
if [[ "${NUM_GPUS}" -eq 2 ]]; then
  ACCUMULATE="${ACCUMULATE:-16}"
elif [[ "${NUM_GPUS}" -eq 4 ]]; then
  ACCUMULATE="${ACCUMULATE:-8}"
else
  ACCUMULATE="${ACCUMULATE:-$((128 / (BATCH_SIZE * NUM_GPUS)))}"
fi

MAX_EPOCHS="${MAX_EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
LR="${LR:-1e-4}"

GLOBAL_BATCH=$((BATCH_SIZE * NUM_GPUS * ACCUMULATE))

# W&B Configuration
WANDB="${WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-e2ekd}"
WANDB_PROJECT="${WANDB_PROJECT:-para-ssr-v2}"
WANDB_GROUP="${WANDB_GROUP:-r34_student}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-r34_v2_student}"

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
      "wandb.tags=[para-ssr-v2,r34,student,aux-tasks,interaction,anchors-256]"
    )
  else
    echo "Notice: No WANDB_API_KEY found. Running with W&B disabled."
    WANDB_ARGS=("wandb.enable=false")
  fi
fi

echo "======================================================================"
echo " [Pure Student Training] PARA-SSR v2 ResNet-34 + Aux + 256 Anchors"
echo " Experiment Name : ${EXPERIMENT}"
echo " Visible GPUs    : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo " Global Batch    : ${GLOBAL_BATCH} (Batch ${BATCH_SIZE} x GPUs ${NUM_GPUS} x Acc ${ACCUMULATE})"
echo " Max Epochs      : ${MAX_EPOCHS}"
echo " Learning Rate   : ${LR}"
echo " Backbone        : ResNet-34 (resnet34.tv_in1k)"
echo " Planner         : 256 WoTE Trajectory Anchors + PDM Simulation Rewards"
echo " Aux Heads       : 3D Detection + Motion + Map + Task Interaction"
echo " Distillation    : OFF"
echo "======================================================================"

exec "${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_agent \
  agent.lr="${LR}" \
  agent.config.image_architecture=resnet34.tv_in1k \
  agent.config.max_epochs="${MAX_EPOCHS}" \
  agent.config.use_task_interaction=true \
  agent.config.use_det_motion_head=true \
  agent.config.use_map_head=true \
  agent.config.plan_anchor=true \
  agent.config.plan_anchor_file=/workspace/byounggun/SSR/data/planning_vb/trajectory_anchors_256.npy \
  agent.config.plan_score_file=/workspace/byounggun/SSR/data/planning_vb/pdm_score_256 \
  agent.config.plan_heading_from_xy=true \
  'agent.config.plan_reward_weights=[0.1,0.5,0.5,1.0]' \
  'agent.config.grad_balance_target={plan:0.4,det:0.3,map:0.3}' \
  agent.config.grad_balance_warmup_iters=10600 \
  agent.config.grad_balance_interval=200 \
  experiment_name="${EXPERIMENT}" \
  scene_filter=navtrain \
  split=trainval \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${WORKERS}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.accumulate_grad_batches="${ACCUMULATE}" \
  trainer.params.check_val_every_n_epoch="${VAL_EVERY}" \
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
