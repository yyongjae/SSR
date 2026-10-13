#!/usr/bin/env bash
# ==============================================================================
# Evaluate Student (para_ssr_v2_r34_student Epoch 29) on GPUs 6 and 7 ONLY
# Usage:
#   bash scripts/evaluation/eval_student_r34_gpus67.sh
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

CKPT="/workspace/byounggun/SSR/work_dirs/para_ssr_v2_r34_student/lightning_logs/version_0/checkpoints/epoch=29-step=19950.ckpt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="/workspace/byounggun/SSR/work_dirs/para_ssr_v2_r34_student/lightning_logs/version_0/checkpoints/last.ckpt"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Error: Checkpoint not found at ${CKPT}" >&2
  exit 1
fi

echo "======================================================================"
echo " Evaluating Student (para_ssr_v2_r34_student) on GPUs 6, 7"
echo " Checkpoint : ${CKPT}"
echo " Target GPUs: 6, 7 (2 shards)"
echo "======================================================================"

unset CUDA_VISIBLE_DEVICES || true

export GPU_LIST="6,7"
export CKPT_SRC="${CKPT}"
export SNAPSHOT_DIR="/workspace/byounggun/SSR/work_dirs/eval_snapshots/para_ssr_v2_r34_student_epoch29"
export EXPERIMENT_NAME="eval/para_ssr_v2_r34_student_epoch29"
export SKIP_AUX="1"
export IMAGE_ARCHITECTURE="resnet34.tv_in1k"

exec bash "${REPO}/scripts/evaluation/eval_para_ssr_distill_epoch29_gpus4567.sh" \
  agent.config.image_architecture=resnet34.tv_in1k \
  agent.config.use_task_interaction=true \
  agent.config.use_det_motion_head=true \
  agent.config.use_map_head=true \
  agent.config.plan_anchor=true \
  agent.config.plan_anchor_file=/workspace/byounggun/SSR/data/planning_vb/trajectory_anchors_256.npy \
  agent.config.plan_score_file=/workspace/byounggun/SSR/data/planning_vb/pdm_score_256 \
  "$@"
