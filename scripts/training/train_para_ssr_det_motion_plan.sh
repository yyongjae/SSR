#!/usr/bin/env bash
# PARA-SSR ablation: DET + MOTION + PLAN (no vector-map head).
#
# Same recipe as train_para_ssr.sh; task interaction is off, the map head is removed and the
# shared-BEV gradient target is plan : det = 1 : 1.  The map target builder,
# the map loss and the map valve of the GradBalancer are all skipped by the
# use_map_head flag, and no map parameters exist (DDP-safe).
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/training/train_para_ssr_det_motion_plan.sh
#
# Evaluate with the same flag (the checkpoint has no map tensors, and the
# agent loads strictly):
#   bash scripts/evaluation/eval_para_ssr.sh /abs/ckpt \
#     agent.config.use_task_interaction=false agent.config.use_map_head=false
set -euo pipefail
ABLATION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXPERIMENT="${EXPERIMENT:-para_ssr_det_motion_plan}"
exec bash "${ABLATION_REPO}/scripts/training/train_para_ssr.sh" \
  agent.config.use_task_interaction=false \
  agent.config.use_det_motion_head=true \
  agent.config.use_map_head=false \
  'agent.config.grad_balance_target={plan:0.5,det:0.5}' \
  "$@"
