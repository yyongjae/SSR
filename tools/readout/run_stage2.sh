#!/usr/bin/env bash
# Stage 2 of report/19: can the student BEV be read like the teacher's?
# Needs Stage 1 outputs and student caches (cache_student_bev.py).
#
#   student           h trained on the student BEV                 -> S_student
#   transfer          Stage-1 teacher h, frozen, on the student    -> S_transfer
#   transfer_adapter  same h frozen + trained 1x1 adapter          -> S_transfer+A
#
#   STUDENT_CACHE=... STUDENT_CACHE_TEST=... TARGETS_TRAIN=... TARGETS_TEST=... \
#   PRESETS="h1" SEEDS="0" bash tools/readout/run_stage2.sh
#
# Reading (report/19 s2):  S_student ~ S_own and S_transfer low  -> alignment gap
#                          S_student << S_own                     -> content gap, distil
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
: "${STUDENT_CACHE:?}" "${STUDENT_CACHE_TEST:?}" "${TARGETS_TRAIN:?}" "${TARGETS_TEST:?}"
RUNS="${RUNS:-runs/readout}"
PRESETS="${PRESETS:-h1}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-10}"
TRAIN_ARGS=(--targets "${TARGETS_TRAIN}" --epochs "${EPOCHS}" ${READOUT_TRAIN_ARGS:-})
EVAL_ARGS=(--targets "${TARGETS_TEST}" --bev-cache "${STUDENT_CACHE_TEST}" ${READOUT_EVAL_ARGS:-})

for p in ${PRESETS}; do
  for s in ${SEEDS}; do
    t="${RUNS}/teacher_${p}_s${s}"
    [[ -f "${t}/readout.pt" ]] || { echo "missing Stage-1 run ${t}" >&2; exit 2; }

    out="${RUNS}/student_${p}_s${s}"
    [[ -f "${out}/readout.pt" ]] || "${PY}" "${HERE}/train_readout.py" --bev-cache "${STUDENT_CACHE}" \
        --out "${out}" --preset "${p}" --seed "${s}" "${TRAIN_ARGS[@]}"
    [[ -f "${out}/pdms_student.json" ]] || "${PY}" "${HERE}/eval_readout_pdms.py" \
        --readout "${out}/readout.pt" --out "${out}/pdms_student.csv" "${EVAL_ARGS[@]}"

    [[ -f "${t}/pdms_transfer.json" ]] || "${PY}" "${HERE}/eval_readout_pdms.py" \
        --readout "${t}/readout.pt" --out "${t}/pdms_transfer.csv" "${EVAL_ARGS[@]}"

    out="${RUNS}/transfer_adapter_${p}_s${s}"
    [[ -f "${out}/readout.pt" ]] || "${PY}" "${HERE}/train_readout.py" --bev-cache "${STUDENT_CACHE}" \
        --out "${out}" --init-readout "${t}/readout.pt" --adapter-only --seed "${s}" "${TRAIN_ARGS[@]}"
    [[ -f "${out}/pdms_transfer_adapter.json" ]] || "${PY}" "${HERE}/eval_readout_pdms.py" \
        --readout "${out}/readout.pt" --out "${out}/pdms_transfer_adapter.csv" "${EVAL_ARGS[@]}"
  done
done
"${PY}" "${HERE}/collect_results.py" "${RUNS}"
