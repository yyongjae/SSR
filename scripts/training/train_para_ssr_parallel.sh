#!/usr/bin/env bash
# Parallel multitask baseline: shared BEV, independent planner/det-motion/map heads.
# All three tasks are supervised, but the planner reads only dense BEV.
# Shares the encoder, optimizer, loss weights and FP32 recipe with interaction-on.
#
#   bash scripts/training/train_para_ssr_parallel.sh
# Evaluate with the same architecture flag for strict checkpoint loading:
#   bash scripts/evaluation/eval_para_ssr.sh /abs/ckpt \
#     agent.config.use_task_interaction=false
set -euo pipefail
ABLATION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXPERIMENT="${EXPERIMENT:-para_ssr_parallel_final}"
exec bash "${ABLATION_REPO}/scripts/training/train_para_ssr.sh" \
  agent.config.use_task_interaction=false \
  agent.config.use_det_motion_head=true \
  agent.config.use_map_head=true \
  "$@"
