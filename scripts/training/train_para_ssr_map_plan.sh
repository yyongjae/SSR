#!/usr/bin/env bash
# PARA-SSR ablation: MAP + PLAN (no detection / motion head).
#
# Same recipe as train_para_ssr.sh; only the detection+motion head is removed
# and the shared-BEV gradient target is plan : map = 1 : 1.  The agent-target
# builder, the det/motion losses and the det valve of the GradBalancer are all
# skipped by the use_det_motion_head flag, and no det parameters exist.
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/training/train_para_ssr_map_plan.sh
#
# Evaluate with the same flag:
#   bash scripts/evaluation/eval_para_ssr.sh /abs/ckpt agent.config.use_det_motion_head=false
set -euo pipefail
ABLATION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXPERIMENT="${EXPERIMENT:-para_ssr_map_plan}"
exec bash "${ABLATION_REPO}/scripts/training/train_para_ssr.sh" \
  agent.config.use_det_motion_head=false \
  'agent.config.grad_balance_target={plan:0.5,map:0.5}' \
  "$@"
