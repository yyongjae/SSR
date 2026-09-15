#!/usr/bin/env bash
# PARA-SSR on navsim -- front 3 cameras, 2 GPU training.
#
# Batch size: use the final-model smoke-tested B=4/GPU. Earlier peak-memory
# numbers came from a pre-audit motion head and are intentionally not repeated
# here. BEVFormer's 5,000 spatial queries plus the restored 1,800 QxM motion
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

# The local training environment can run directly without `conda activate`.
# Keep an explicit interpreter override for other installations; otherwise
# fall back to the active environment when the local ssr environment is absent.
if [[ -n "${SSR_NAVSIM_PYTHON:-}" ]]; then
  TRAIN_PYTHON="${SSR_NAVSIM_PYTHON}"
elif [[ -x "${HOME}/miniconda3/envs/ssr/bin/python" ]]; then
  TRAIN_PYTHON="${HOME}/miniconda3/envs/ssr/bin/python"
else
  TRAIN_PYTHON="$(command -v python)"
fi

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset"
export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${REPO}/work_dirs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# These wrappers train on one host. Loopback avoids the NCCL bootstrap
# connection failure observed with this server's default network interface.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# Every dataloader worker must stay single-threaded: the worker IS the unit of
# parallelism, so each library's own thread pool only oversubscribes the host.
# Left unset, OpenCV/OpenMP spawn one thread per core inside every worker --
# measured at 2,685 threads on 32 cores, 491k context switches/s, and a step
# 18x slower than its compute cost.  run_training.py pins cv2/torch per worker;
# these cover the OpenMP runtimes behind numpy and shapely.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

EXPERIMENT="${EXPERIMENT:-para_ssr_front3}"
BATCH_SIZE="${BATCH_SIZE:-4}"        # per GPU
ACCUMULATE="${ACCUMULATE:-16}"       # -> global 128 on 2 GPUs
MAX_EPOCHS="${MAX_EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
LR="${LR:-1e-4}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# Weights & Biases. Runs alongside TensorBoard and is fail-open: an SDK or
# service failure disables telemetry instead of killing a DDP rank.
# TensorBoard is always on. Opt in with WANDB=1; for offline W&B also set
# WANDB_MODE=offline (sync later).
WANDB="${WANDB:-0}"
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

exec "${TRAIN_PYTHON}" "${REPO}/navsim/planning/script/run_training.py" \
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
