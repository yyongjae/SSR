#!/usr/bin/env bash
# GT stores for the lateral surrogate: objects/navtrain (raw logs), sdf/navtrain and e2e_side (metric cache).  85,109 tokens,
# 12.5 GiB, ~1 h on the reference server, CPU.  Needs 20_metric_cache_navtrain.sh finished.
# A store left out does not stop a training by itself: ref_gt_ok goes False per token.  run-time guard: online2 _check_inputs.
#   K             sdf parts, each 4 workers (default 16)
#   SIDE_WORKERS  default 24
#   SPLIT         default navtrain_train.  trainval_simscale: every row of splits/trainval_simscale.parquet into the same
#                 three stores (keyed by the full token; SimScale objects come from the scene pickle, up to 4 s).  The
#                 builders skip what exists, so the navtrain stores are kept and not refused; the run status files
#                 get a _trainval_simscale suffix.
# --mc-root is not passed to build_future_objects: its validation mode needs a research-only module.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
K=${K:-16}
SIDE_WORKERS=${SIDE_WORKERS:-24}
SPLIT=${SPLIT:-navtrain_train}
case "$SPLIT" in
  navtrain_train)    TABLE=$SSR_DATA_ROOT/splits/e2e_train_trainlogs.parquet; TAG=; DATA= ;;
  trainval_simscale) TABLE=$SSR_DATA_ROOT/splits/trainval_simscale.parquet; TAG=_trainval_simscale; DATA=trainval_simscale ;;
  *) die "unknown split $SPLIT" ;;
esac
need "$TABLE" "$SSR_DATA_ROOT/metric_cache/_state/done_tokens.txt"
if [ -z "$DATA" ]; then
  refuse_unless_resume "$SSR_DATA_ROOT/objects/navtrain/summary.json" "$SSR_DATA_ROOT/sdf/navtrain" "$SSR_DATA_ROOT/e2e_side/status.json"
fi
run_mkdir "$SETUP_LOGS"
run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/make_parts.py" --tables "$TABLE" --k "$K" --out-prefix "$SETUP_LOGS/sdf${TAG}_part"

obj=(--tokens "$TABLE" --logs "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" --out "$SSR_DATA_ROOT/objects/navtrain" --workers 4)
side=(build --workers "$SIDE_WORKERS")
sdf=()
if [ -n "$DATA" ]; then
  obj+=(--report "$SSR_DATA_ROOT/objects/navtrain/summary$TAG.json")
  side+=(--tokens "$TABLE" --status "$SSR_DATA_ROOT/e2e_side/status$TAG.json")
  # build_sdf skips a metric cache younger than 60 s (one a running build may still write); this step comes after the
  # metric cache step has finished, and right after it in prepare_trainval_simscale.sh
  sdf=(--min-age 0)
fi
run_bg "$SETUP_LOGS/build_objects$TAG.log" "${CPU[@]}" "$SSR_PY" "$REPO/tools/refiner/build_future_objects.py" "${obj[@]}"
for i in $(seq 1 "$K"); do
  run_bg "$SETUP_LOGS/build_sdf${TAG}_part$i.log" "${CPU[@]}" "$SSR_PY" "$REPO/tools/refiner/build_sdf.py" --subset navtrain \
    --tokens "$SETUP_LOGS/sdf${TAG}_part$i.parquet" --workers 4 "${sdf[@]}"
done
run_bg "$SETUP_LOGS/build_e2e_side$TAG.log" "${CPU[@]}" "$SSR_PY" "$REPO/tools/refiner/build_e2e_side.py" "${side[@]}"
wait_bg
verify gt refiner_noteacher $DATA
