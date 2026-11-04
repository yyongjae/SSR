#!/usr/bin/env bash
# GT stores for the lateral surrogate: objects/navtrain (raw logs), sdf/navtrain and e2e_side (metric cache).  85,109 tokens,
# 12.5 GiB, ~1 h on the reference server, CPU.  Needs 20_metric_cache_navtrain.sh finished.
# A store left out does not stop a training by itself: ref_gt_ok goes False per token.  run-time guard: online2 _check_inputs.
#   K             sdf parts, each 4 workers (default 16)
#   SIDE_WORKERS  default 24
# --mc-root is not passed to build_future_objects: its validation mode needs a research-only module.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
K=${K:-16}
SIDE_WORKERS=${SIDE_WORKERS:-24}
TABLE=$SSR_DATA_ROOT/splits/e2e_train_trainlogs.parquet
need "$TABLE" "$SSR_DATA_ROOT/metric_cache/_state/done_tokens.txt"
refuse_unless_resume "$SSR_DATA_ROOT/objects/navtrain/summary.json" "$SSR_DATA_ROOT/sdf/navtrain" "$SSR_DATA_ROOT/e2e_side/status.json"
run_mkdir "$SETUP_LOGS"
run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/make_parts.py" --tables "$TABLE" --k "$K" --out-prefix "$SETUP_LOGS/sdf_part"

run_bg "$SETUP_LOGS/build_objects.log" "${CPU[@]}" "$SSR_PY" "$REPO/tools/refiner/build_future_objects.py" --tokens "$TABLE" \
  --logs "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" --out "$SSR_DATA_ROOT/objects/navtrain" --workers 4
for i in $(seq 1 "$K"); do
  run_bg "$SETUP_LOGS/build_sdf_part$i.log" "${CPU[@]}" "$SSR_PY" "$REPO/tools/refiner/build_sdf.py" --subset navtrain \
    --tokens "$SETUP_LOGS/sdf_part$i.parquet" --workers 4
done
run_bg "$SETUP_LOGS/build_e2e_side.log" "${CPU[@]}" "$SSR_PY" "$REPO/tools/refiner/build_e2e_side.py" build --workers "$SIDE_WORKERS"
wait_bg
verify gt refiner_noteacher
