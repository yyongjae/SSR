#!/usr/bin/env bash
# PARA-SSR v3: factorized path x speed planner (navsim/agents/para_ssr/fact/).
#
# Wraps scripts/training/train_para_ssr.sh with the v3 recipe:
#   * fact_enable + the vocabulary built by tools/fact/build_vocab.py
#   * input_target=true (the target builder hands the forced vocabulary indices to forward)
#   * the 256 WoTE anchors + their exact PDM labels as extra training candidates
#   * Bezier heading, 20 epochs (report/42: v2 converged by epoch 19)
# Environment knobs (same as train_para_ssr.sh): CUDA_VISIBLE_DEVICES, BATCH_SIZE,
# ACCUMULATE, MAX_EPOCHS, WORKERS, EXPERIMENT, SSR_NAVSIM_PYTHON, WANDB, CACHE_PATH.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VOCAB="${VOCAB:-${REPO}/data/fact_vocab}"
PATHS="${PATHS:-${VOCAB}/paths_1024.npy}"
SPEEDS="${SPEEDS:-${VOCAB}/speeds_256.npy}"
ANCHORS="${ANCHORS:-${REPO}/data/planning_vb/trajectory_anchors_256.npy}"
SCORES="${SCORES:-${REPO}/data/planning_vb/pdm_score_256}"
for f in "$PATHS" "$SPEEDS" "$ANCHORS" "${SCORES}.npy"; do
  [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
export EXPERIMENT="${EXPERIMENT:-fact_v3}"
export MAX_EPOCHS="${MAX_EPOCHS:-20}"
export SSR_NAVSIM_PYTHON="${SSR_NAVSIM_PYTHON:-${HOME}/conda-envs/ssr-navsim/bin/python}"
CACHE_ARGS=()
if [[ -n "${CACHE_PATH:-}" ]]; then
  CACHE_ARGS=("cache_path=${CACHE_PATH}")
fi
exec bash "${REPO}/scripts/training/train_para_ssr.sh" \
  agent.config.fact_enable=true \
  agent.config.fact_path_file="${PATHS}" \
  agent.config.fact_speed_file="${SPEEDS}" \
  agent.config.input_target=true \
  agent.config.plan_anchor_file="${ANCHORS}" \
  agent.config.plan_score_file="${SCORES}" \
  agent.config.grad_balance_warmup_iters="${GB_WARMUP:-5300}" \
  agent.config.grad_balance_interval=100 \
  "${CACHE_ARGS[@]}" \
  "$@"
