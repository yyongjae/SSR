#!/usr/bin/env bash
# ==============================================================================
# Evaluate BEV Selector v8 on GPUs 6 and 7 ONLY (NAVSIM PDM score)
# Usage:
#   bash scripts/evaluation/eval_bev_selector_v8_gpus67.sh
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

# Checkpoint path: latest saved checkpoint from paradrive_distill_bev_selector_v8
CKPT="/workspace/byounggun/SSR/work_dirs/paradrive_distill_bev_selector_v8/lightning_logs/version_1/checkpoints/last.ckpt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="/workspace/byounggun/SSR/work_dirs/paradrive_distill_bev_selector_v8/lightning_logs/version_1/checkpoints/epoch=19-step=13300.ckpt"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Error: Checkpoint not found at ${CKPT}" >&2
  exit 1
fi

echo "======================================================================"
echo " Evaluating BEV Selector v8 (Last Checkpoint)"
echo " Checkpoint : ${CKPT}"
echo " Target GPUs: 6, 7 (2 shards)"
echo "======================================================================"

# Unset CUDA_VISIBLE_DEVICES in parent so eval_para_ssr_distill_epoch29_gpus4567.sh
# can bind shard 0 -> GPU 6 and shard 1 -> GPU 7 directly.
unset CUDA_VISIBLE_DEVICES || true

export GPU_LIST="6,7"
export CKPT_SRC="${CKPT}"
export SNAPSHOT_DIR="/workspace/byounggun/SSR/work_dirs/eval_snapshots/paradrive_distill_bev_selector_v8_ep20"
export EXPERIMENT_NAME="eval/paradrive_distill_bev_selector_v8_ep20"
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
