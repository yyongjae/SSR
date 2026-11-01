#!/usr/bin/env bash
# ==============================================================================
# NAVSIM Evaluation for Student With Aux Tasks (`para_ssr_with_aux`, tmux 21)
# Model Architecture:
#   - Backbone        : ResNet-50 (timm/resnet50.tv_in1k)
#   - Task Interaction: true (Det + Motion + Map decoder hidden states cross-attention)
#   - Aux Heads       : Det + Motion Head, Map Head (active)
#   - Plan Anchor     : false (Dense BEV cross-attention single-trajectory planner)
#
# Evaluation Options:
#   - GPUs: 4, 5 (2 shards parallel PDM evaluation over 12,146 test scenarios)
#   - SKIP_AUX=1 (Default: Planning PDM Score only)
#   - SKIP_AUX=0 (Run Planning PDM Score + Perception 3D BBox/Map mAP evaluation)
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

# Target GPUs
export GPU_LIST="${GPU_LIST:-4,5}"

# Checkpoint paths
DEFAULT_CKPT="${NAVSIM_EXP_ROOT}/para_ssr_with_aux/lightning_logs/version_0/checkpoints/epoch=29-step=19950.ckpt"
if [[ ! -f "${DEFAULT_CKPT}" ]]; then
  DEFAULT_CKPT="${NAVSIM_EXP_ROOT}/para_ssr_with_aux/lightning_logs/version_0/checkpoints/last.ckpt"
fi
export CKPT_SRC="${CKPT_SRC:-${DEFAULT_CKPT}}"
export SNAPSHOT_DIR="${SNAPSHOT_DIR:-${NAVSIM_EXP_ROOT}/eval_snapshots/para_ssr_with_aux_epoch29}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval/para_ssr_with_aux_epoch29}"
export SKIP_AUX="${SKIP_AUX:-1}"
export IMAGE_ARCHITECTURE="resnet50.tv_in1k"

echo "======================================================================"
echo " [Student With Aux Evaluation] NAVSIM Evaluation Launch"
echo " Target GPUs       : ${GPU_LIST}"
echo " Checkpoint        : ${CKPT_SRC}"
echo " Experiment Name   : ${EXPERIMENT_NAME}"
echo " Skip Aux (mAP)    : ${SKIP_AUX} (1: PDM only, 0: PDM + Aux mAP)"
echo " Image Architecture: ${IMAGE_ARCHITECTURE}"
echo "======================================================================"

exec bash "${REPO}/scripts/evaluation/eval_para_ssr_distill_epoch29_gpus4567.sh" \
  agent.config.image_architecture=resnet50.tv_in1k \
  agent.config.use_task_interaction=true \
  agent.config.use_det_motion_head=true \
  agent.config.use_map_head=true \
  agent.config.plan_anchor=false \
  "$@"
