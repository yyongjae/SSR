#!/usr/bin/env bash
# BEV selector distillation. One stage, no adapter checkpoints.
# The v4 recipe is unchanged:
#   FORCE_RETRAIN=stage1a,stage2 bash ./scripts/training/run_all_stages_distill.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export DISTILL_FEATURE_ROOT="${DISTILL_FEATURE_ROOT:-/home/external-user/datasets/teacher_cache}"

if [[ -x "/home/external-user/miniconda3/envs/ssr/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/external-user/miniconda3/envs/ssr/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

EXPERIMENT="${EXPERIMENT:-paradrive_distill_bev_selector}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
WANDB="${WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-e2ekd}"
WANDB_PROJECT="${WANDB_PROJECT:-v1_distill}"
WANDB_GROUP="${WANDB_GROUP:-bev-selector}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-r34_sel_trial}"

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
    "wandb.tags=[bev-selector,r34,v2]"
  )
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

echo " Selector distill. v4 stays: FORCE_RETRAIN=stage1a,stage2 bash ./scripts/training/run_all_stages_distill.sh"
echo " Experiment   : ${EXPERIMENT}"
echo " GPUs         : ${CUDA_VISIBLE_DEVICES}"
echo " GradBalancer : off"

"${PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_selector_agent \
  agent.lr="1e-4" \
  agent.config.max_epochs="${MAX_EPOCHS}" \
  agent.config.distill_feature_root="${DISTILL_FEATURE_ROOT}" \
  agent.config.image_architecture=resnet34.tv_in1k \
  agent.config.distill_selector=true \
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
  +trainer.params.devices="${NUM_GPUS}" \
  checkpoint.every_n_epochs=5 \
  checkpoint.save_top_k=-1 \
  checkpoint.save_last=true \
  trainer.params.gradient_clip_val=35.0 \
  trainer.params.gradient_clip_algorithm=norm \
  "${WANDB_ARGS[@]}" \
  "$@"
