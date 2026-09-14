#!/usr/bin/env bash
# PARA-SSR ablation: PLAN only (no detection/motion head, no vector-map head).
#
# The shared BEV is then steered by the planning loss alone -- SSR's original
# situation, and the control arm the other three are measured against.  Both
# head flags are off, so no auxiliary parameters are built, no detection/map
# targets are generated (the nuPlan map query disappears from the dataloader,
# which makes this the fastest arm), and no auxiliary loss is computed.
#
# `grad_balance_target=null` disables the GradBalancer: with a single task
# there is nothing to balance.  `gshare/plan` is still logged and is trivially
# 1.0, which is a useful sanity check that no other task slipped in.
#
#   CUDA_VISIBLE_DEVICES=4,5 bash scripts/training/train_para_ssr_plan_only.sh
#
# Evaluate with the same flags (the checkpoint has no auxiliary tensors and the
# agent loads strictly):
#   bash scripts/evaluation/eval_para_ssr.sh /abs/ckpt \
#     agent.config.use_det_motion_head=false agent.config.use_map_head=false
set -euo pipefail
ABLATION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXPERIMENT="${EXPERIMENT:-para_ssr_plan_only}"
exec bash "${ABLATION_REPO}/scripts/training/train_para_ssr.sh" \
  agent.config.use_det_motion_head=false \
  agent.config.use_map_head=false \
  agent.config.grad_balance_target=null \
  "$@"
