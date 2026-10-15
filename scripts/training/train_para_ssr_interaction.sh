#!/usr/bin/env bash
# Each planner layer: BEV -> parallel det/motion + map -> residual sum -> FFN.
# All three tasks are supervised; planning also backpropagates through the heads.
# Shares the encoder, optimizer, loss weights and FP32 recipe with the other arms.
#
#   bash scripts/training/train_para_ssr_interaction.sh
# Evaluation uses the same flags (these are also the model defaults).
set -euo pipefail
ABLATION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXPERIMENT="${EXPERIMENT:-para_ssr_interaction_final}"
exec bash "${ABLATION_REPO}/scripts/training/train_para_ssr.sh" \
  agent.config.use_task_interaction=true \
  agent.config.use_det_motion_head=true \
  agent.config.use_map_head=true \
  "$@"
