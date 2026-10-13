#!/usr/bin/env bash
# Advanced BEV selector distillation (v5).
# Incorporates:
#   1. Linear projection adapter (student_proj) for channel basis alignment.
#   2. Loss scale un-dilution (tok_scale=25.0) to match det/map gradient norms naturally.
#   3. Hybrid Normalized L2 + Cosine distance loss.
#   4. Spatially boosted masks to resolve bank conflicts.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

if [[ -f "${REPO}/env.vast.sh" ]]; then
  source "${REPO}/env.vast.sh"
fi

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT_OVERRIDE:-${REPO}/work_dirs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export DISTILL_FEATURE_ROOT="${DISTILL_FEATURE_ROOT:-/workspace/teacher_cache}"

if [[ -x "/root/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/root/miniconda3/envs/ssr/bin/python}"
elif [[ -x "/home/external-user/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/external-user/miniconda3/envs/ssr/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

EXPERIMENT="${EXPERIMENT:-paradrive_distill_bev_selector_v5}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
WANDB="${WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-e2ekd}"
WANDB_PROJECT="${WANDB_PROJECT:-v1_distill}"
WANDB_GROUP="${WANDB_GROUP:-bev-selector}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-r34_sel_v5}"

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
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" == WANDB_API_KEY=* ]]; then
      key="${line#WANDB_API_KEY=}"
      key="${key%\"}"
      key="${key#\"}"
      key="${key%\'}"
      key="${key#\'}"
    fi
  done < "${REPO}/.env"
  if [[ -z "${key}" ]]; then
    echo "Error: ${REPO}/.env has no WANDB_API_KEY. WANDB=0 skips the upload." >&2
    exit 1
  fi
  export WANDB_API_KEY="${key}"
  export WANDB_ENTITY
  unset key
  _WANDB_NETRC_FILE="$(mktemp "${TMPDIR:-/tmp}/wandb-netrc.XXXXXX")"
  chmod 600 "${_WANDB_NETRC_FILE}"
  printf 'machine api.wandb.ai\n  login user\n  password %s\n' "${WANDB_API_KEY}" > "${_WANDB_NETRC_FILE}"
  export NETRC="${_WANDB_NETRC_FILE}"
  if [[ -n "${WANDB_TRIAL:-}" ]]; then
    trial="${WANDB_TRIAL}"
  else
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
  fi
  run_name="${WANDB_RUN_PREFIX}_${trial}"
  echo " W&B run      : ${WANDB_ENTITY}/${WANDB_PROJECT}  ${run_name}"
  WANDB_ARGS=(
    "wandb.enable=true"
    "wandb.mode=${WANDB_MODE:-online}"
    "wandb.entity=${WANDB_ENTITY}"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.group=${WANDB_GROUP}"
    "wandb.name=${run_name}"
    "wandb.tags=[bev-selector,r34,v5,adv_distill]"
  )
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

echo "=========================================================================="
echo " Advanced BEV Selector Distillation v5"
echo " Experiment   : ${EXPERIMENT}"
echo " GPUs         : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} devices)"
echo " Feature root : ${DISTILL_FEATURE_ROOT}"
echo " Output root  : ${NAVSIM_EXP_ROOT}"
echo " Agent config : para_ssr_selector_v5_agent"
echo "=========================================================================="

"${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_selector_v5_agent \
  agent.lr="1e-4" \
  agent.config.max_epochs="${MAX_EPOCHS}" \
  agent.config.distill_feature_root="${DISTILL_FEATURE_ROOT}" \
  agent.config.image_architecture=resnet34.tv_in1k \
  agent.config.distill_selector=true \
  agent.config.distill_selector_proj=true \
  agent.config.distill_selector_tok_scale=25.0 \
  agent.config.distill_selector_loss_type=hybrid \
  agent.config.distill_selector_struct_mask_boost=2.0 \
  agent.config.distill_selector_plan_tau=0.25 \
  agent.config.distill_selector_struct_mix=0.6 \
  agent.config.use_corridor_mask=false \
  agent.config.use_stl=false \
  agent.config.plan_num_layers=3 \
  agent.config.use_task_interaction=true \
  agent.config.use_ego_motion=false \
  agent.config.grad_balance_target=null \
  experiment_name="${EXPERIMENT}" \
  scene_filter=navtrain \
  split=trainval \
  dataloader.params.batch_size=4 \
  dataloader.params.num_workers=8 \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.accumulate_grad_batches=16 \
  trainer.params.check_val_every_n_epoch=5 \
  trainer.params.precision=32 \
  ++trainer.params.devices="${NUM_GPUS}" \
  checkpoint.every_n_epochs=5 \
  checkpoint.save_top_k=-1 \
  checkpoint.save_last=true \
  trainer.params.gradient_clip_val=35.0 \
  trainer.params.gradient_clip_algorithm=norm \
  "${WANDB_ARGS[@]}" \
  "$@"
