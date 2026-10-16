#!/usr/bin/env bash
# v2+seg (navtest PDMS 88.40 / EPDMS 88.11, run para_ssr_v2_r34_seg): v2 r34 with the vector map head replaced by a
# drivable-area segmentation head (navsim/agents/para_ssr/bev_seg.py).  Identical to v2 r34 except
#   agent.config.map_head_type=seg      the polyline head is not built; the seg head's token view feeds the planner's
#                                       map memories
#   agent.config.bev_seg_weight=3.0     ||dL_map/d bev_embed|| is 3.1x smaller for the seg loss at init than for the
#                                       vector loss and the GradBalancer cannot scale a task up, so 3.0 matches it
# seg labels = the GT drivable polygons (the PDM drivable-map layers), built on the fly: no teacher, no extra cache.
#   needs: data/dataset (NAVSIM, see train_para_ssr.sh), PLAN_SCORE = pdm_score_256 (+ .tokens.json; 253 MB, not in git)
#   usage: CUDA_VISIBLE_DEVICES=0,1 PLAN_SCORE=/path/pdm_score_256 bash scripts/training/train_v2_r34_seg.sh [overrides]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLAN_SCORE="${PLAN_SCORE:-$REPO/data/planning_vb/pdm_score_256}"
PLAN_ANCHORS="${PLAN_ANCHORS:-$REPO/resources/planning_vb/trajectory_anchors_256.npy}"
export WANDB="${WANDB:-1}" WANDB_PROJECT="${WANDB_PROJECT:-para-ssr-v2}" WANDB_GROUP="${WANDB_GROUP:-v2-seg}"
export EXPERIMENT="${EXPERIMENT:-para_ssr_v2_r34_seg}" WORKERS="${WORKERS:-3}"
exec bash "$REPO/scripts/training/train_para_ssr.sh" \
  agent.config.use_task_interaction=true \
  agent.config.use_det_motion_head=true \
  agent.config.use_map_head=true \
  '~agent.config.grad_balance_target' \
  '+agent.config.grad_balance_target={plan:0.4,det:0.3,map:0.3}' \
  agent.config.plan_anchor=true \
  agent.config.plan_anchor_file="$PLAN_ANCHORS" \
  agent.config.plan_score_file="$PLAN_SCORE" \
  agent.config.image_architecture=resnet34.tv_in1k \
  agent.config.plan_heading_from_xy=true \
  agent.config.map_head_type=seg \
  agent.config.bev_seg_weight=3.0 \
  "$@"
