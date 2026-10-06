#!/usr/bin/env bash
# Stage 1 of report/19: is there planning content in the teacher BEV at all?
#
#   own       h on the teacher BEV                       -> S_own
#   ego       same h, BEV unread                         -> S_ego (floor)
#   shuffled  h on the teacher BEV, permuted labels      -> must be <= S_ego
#
#   TEACHER_CACHE=... TEACHER_CACHE_TEST=... TARGETS_TRAIN=... TARGETS_TEST=... \
#   PRESETS="h1" SEEDS="0 1 2" CUDA_VISIBLE_DEVICES=0 bash tools/readout/run_stage1.sh
#
# Gate: S_own - S_ego clearly above seed noise, else stop (report/19 s4).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
: "${TEACHER_CACHE:?}" "${TEACHER_CACHE_TEST:?}" "${TARGETS_TRAIN:?}" "${TARGETS_TEST:?}"
RUNS="${RUNS:-runs/readout}"
PRESETS="${PRESETS:-h1}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-10}"
TRAIN_ARGS=(--targets "${TARGETS_TRAIN}" --epochs "${EPOCHS}" ${READOUT_TRAIN_ARGS:-})
EVAL_ARGS=(--targets "${TARGETS_TEST}" ${READOUT_EVAL_ARGS:-})

run() {  # name, bev-cache for training, extra train args...
  local name=$1 cache=$2; shift 2
  local out="${RUNS}/${name}"
  [[ -f "${out}/readout.pt" ]] || "${PY}" "${HERE}/train_readout.py" --bev-cache "${cache}" --out "${out}" "${TRAIN_ARGS[@]}" "$@"
  [[ -f "${out}/pdms_teacher.json" ]] || "${PY}" "${HERE}/eval_readout_pdms.py" --readout "${out}/readout.pt" \
      --bev-cache "${TEACHER_CACHE_TEST}" --out "${out}/pdms_teacher.csv" "${EVAL_ARGS[@]}"
}

for p in ${PRESETS}; do
  for s in ${SEEDS}; do
    run "teacher_${p}_s${s}" "${TEACHER_CACHE}" --preset "${p}" --seed "${s}"
    run "ego_${p}_s${s}"     "${TEACHER_CACHE}" --preset "${p}" --seed "${s}" --no-bev
  done
  run "shuffled_${p}_s0" "${TEACHER_CACHE}" --preset "${p}" --seed 0 --shuffle-labels
done
"${PY}" "${HERE}/collect_results.py" "${RUNS}"
