#!/usr/bin/env bash
# Metric-supervised candidate planning, with the normal 2-GPU training recipe.
set -euo pipefail
METRIC_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PARA_SSR_PLAN_ANCHORS:?Set PARA_SSR_PLAN_ANCHORS to the train-only anchors .npz file}"
: "${PARA_SSR_METRIC_CACHE:?Set PARA_SSR_METRIC_CACHE to the navtrain metric world cache}"
if [[ ! -f "${PARA_SSR_PLAN_ANCHORS}" ]]; then
  echo "Anchor archive does not exist: ${PARA_SSR_PLAN_ANCHORS}" >&2
  exit 1
fi
if [[ ! -d "${PARA_SSR_METRIC_CACHE}/metadata" ]]; then
  echo "World cache metadata directory does not exist: ${PARA_SSR_METRIC_CACHE}/metadata" >&2
  exit 1
fi
export PARA_SSR_PLAN_ANCHORS PARA_SSR_METRIC_CACHE
export EXPERIMENT="${EXPERIMENT:-para_ssr_front3_metric_k16}"
exec bash "${METRIC_REPO}/scripts/training/train_para_ssr.sh" \
  agent=para_ssr_metric_agent "$@"
