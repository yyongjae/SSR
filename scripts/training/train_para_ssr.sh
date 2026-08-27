#!/usr/bin/env bash
# PARA-SSR on navsim -- 2 GPU training.
#
# Batch size: use the final-model smoke-tested B=4/GPU. Earlier peak-memory
# numbers came from a pre-audit motion head and are intentionally not repeated
# here. BEVFormer's 10,000 spatial queries plus the restored 1,800 QxM motion
# tokens make WoTE/SeerDrive's 16/GPU inapplicable to this architecture.
# Gradient accumulation recovers the requested global batch:
#
#   4 (per GPU) x 2 (GPUs) x 16 (accumulate) = 128 = WoTE's global batch
#
# Accumulation costs no extra forward/backward work beyond the microbatches.
# The earlier throughput estimate predates restoration of the original QxM
# motion decoder. Re-benchmark the final model before scheduling a 30-epoch run.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

EXPERIMENT="${EXPERIMENT:-para_ssr}"
BATCH_SIZE="${BATCH_SIZE:-4}"        # per GPU
ACCUMULATE="${ACCUMULATE:-16}"       # -> global 128 on 2 GPUs
MAX_EPOCHS="${MAX_EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
LR="${LR:-1e-4}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# Weights & Biases. Runs alongside TensorBoard and is fail-open: an SDK or
# service failure disables telemetry instead of killing a DDP rank.
# Turn off with WANDB=0. Offline runs: WANDB_MODE=offline (sync later).
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-para-ssr}"
WANDB_GROUP="${WANDB_GROUP:-para-navsim}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"
GLOBAL_BATCH=$((BATCH_SIZE * NUM_GPUS * ACCUMULATE))
echo "PARA-SSR launch: GPUs=${NUM_GPUS}, batch/GPU=${BATCH_SIZE}, accumulate=${ACCUMULATE}, global_batch=${GLOBAL_BATCH}"
if [[ "${GLOBAL_BATCH}" -ne 128 ]]; then
  echo "warning: effective global batch is ${GLOBAL_BATCH}, not the configured recipe value 128" >&2
fi

WANDB_ARGS=()
if [[ "${WANDB}" != "0" ]]; then
  WANDB_ARGS+=(
    "wandb.enable=true"
    "wandb.project=${WANDB_PROJECT}"
    "wandb.group=${WANDB_GROUP}"
    "wandb.mode=${WANDB_MODE_ARG}"
    "wandb.name=${EXPERIMENT}"
    "wandb.tags=[para-ssr,navsim,no-ffp,aux,global${GLOBAL_BATCH},${MAX_EPOCHS}ep]"
  )
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS+=("resume_checkpoint=${RESUME_CHECKPOINT}")
fi

python "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_agent \
  agent.lr="${LR}" \
  agent.config.max_epochs="${MAX_EPOCHS}" \
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
  trainer.params.gradient_clip_val=35.0 \
  trainer.params.gradient_clip_algorithm=norm \
  "${WANDB_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "$@"
