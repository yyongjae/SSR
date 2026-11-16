#!/usr/bin/env bash
# Navtrain metric cache (fork lineage, v1.0-style): training labels, warm-up labels and GT stores read it.  103,288 tokens,
# 34 GiB.  Reference server: train 41 min (16 parts x 4 workers), val ~9 min (12 x 3).  CPU.
#   K       parts, each 4 workers (default 16: 64 CPU workers)
#   SPLITS  default "navtrain_train navtrain_val"; navtrain_val is only for retraining the refiner teachers.
#           trainval_simscale: every row of splits/trainval_simscale.parquet (16_trainval_simscale_table.sh) into the same
#           cache; its SimScale scenes get the 4 + 8 frame window (resources/metric_cache_saved_config_simscale.yaml).
#           Tokens already cached are skipped, so an existing navtrain cache is kept and not refused.
# Resume: RESUME=1 (written caches are skipped, truncated ones repaired).  Never run `build_metric_cache.py check` on the
# parts: it overwrites the manifest.  The pickles and the metadata CSV hold absolute paths, so keep this path stable.
# Reported numbers do not use this cache: they use the pristine v1.1 one (70_pristine_navsim.sh).
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
K=${K:-16}
SPLITS=${SPLITS:-navtrain_train navtrain_val}
MC=$SSR_DATA_ROOT/metric_cache
LOGS=$MC/logs
need "$REPO/resources/metric_cache_saved_config.yaml"

TABLES=()
DATA=
for s in $SPLITS; do
  case "$s" in
    navtrain_train) TABLES+=("$SSR_DATA_ROOT/splits/e2e_train_trainlogs.parquet") ;;
    navtrain_val)   TABLES+=("$CK/data-map/navtrain_val_tokens.parquet") ;;
    trainval_simscale)
      TABLES+=("$SSR_DATA_ROOT/splits/trainval_simscale.parquet"); DATA=trainval_simscale
      need "$REPO/resources/metric_cache_saved_config_simscale.yaml" ;;
    *) die "unknown split $s" ;;
  esac
done
if [ -n "$DATA" ]; then
  [ "$(echo "$SPLITS" | wc -w)" = 1 ] || die "SPLITS=trainval_simscale already holds the navtrain rows: give it alone"
  PART=mc_tvsim_part
else
  refuse_unless_resume "$MC/_state/done_tokens.txt"
  PART=mc_part
fi
need "${TABLES[@]}"
run_mkdir "$LOGS"
run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/make_parts.py" --tables "${TABLES[@]}" --k "$K" --out-prefix "$LOGS/$PART" --skip-mc "$MC"
for i in $(seq 1 "$K"); do
  run_bg "$LOGS/build_${PART}$i.log" "${CPU[@]}" "$SSR_PY" "$REPO/tools/refiner/build_metric_cache.py" build \
    --splits "$PART$i" --splits-dir "$LOGS" --workers 4 --no-seed --no-final-check
done
wait_bg
[ -e "$MC/_state/failed_tokens.tsv" ] && die "$MC/_state/failed_tokens.tsv exists: run again with RESUME=1 and read it"
verify metric_cache refiner_noteacher $DATA
