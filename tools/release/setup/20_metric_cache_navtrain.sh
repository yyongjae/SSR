#!/usr/bin/env bash
# Navtrain metric cache (fork lineage, v1.0-style): training labels, warm-up labels and GT stores read it.  103,288 tokens,
# 34 GiB.  Reference server: train 41 min (16 parts x 4 workers), val ~9 min (12 x 3).  CPU.
#   K       parts, each 4 workers (default 16: 64 CPU workers)
#   SPLITS  default "navtrain_train navtrain_val"; navtrain_val is only for retraining the refiner teachers
# Resume: RESUME=1 (written caches are skipped, truncated ones repaired).  Never run `build_metric_cache.py check` on the
# parts: it overwrites the manifest.  The pickles and the metadata CSV hold absolute paths, so keep this path stable.
# Reported numbers do not use this cache: they use the pristine v1.1 one (70_pristine_navsim.sh).
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
K=${K:-16}
SPLITS=${SPLITS:-navtrain_train navtrain_val}
MC=$SSR_DATA_ROOT/metric_cache
LOGS=$MC/logs
refuse_unless_resume "$MC/_state/done_tokens.txt"
need "$REPO/resources/metric_cache_saved_config.yaml"

TABLES=()
for s in $SPLITS; do
  case "$s" in
    navtrain_train) TABLES+=("$SSR_DATA_ROOT/splits/e2e_train_trainlogs.parquet") ;;
    navtrain_val)   TABLES+=("$CK/data-map/navtrain_val_tokens.parquet") ;;
    *) die "unknown split $s" ;;
  esac
done
need "${TABLES[@]}"
run_mkdir "$LOGS"
run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/make_parts.py" --tables "${TABLES[@]}" --k "$K" --out-prefix "$LOGS/mc_part" --skip-mc "$MC"
for i in $(seq 1 "$K"); do
  run_bg "$LOGS/build_mc_part$i.log" "${CPU[@]}" "$SSR_PY" "$REPO/tools/refiner/build_metric_cache.py" build \
    --splits "mc_part$i" --splits-dir "$LOGS" --workers 4 --no-seed --no-final-check
done
wait_bg
[ -e "$MC/_state/failed_tokens.tsv" ] && die "$MC/_state/failed_tokens.tsv exists: run again with RESUME=1 and read it"
verify metric_cache refiner_noteacher
