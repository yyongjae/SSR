#!/usr/bin/env bash
# Token table of the trainval_simscale split (DATA=trainval_simscale): navtrain_train, navtrain_val, then the SimScale
# rounds, as splits/trainval_simscale.parquet (+ .meta.json, splits/simscale.parquet).  Needs 15_token_lists.sh and
# 12_download_simscale.sh.  ~1 min per 50k SimScale scenes, CPU.  The full table is compared with the navtrain pins.
#   ROUNDS        default "0 1 2 3 4"
#   TRAIN_TOKENS  navtrain_train table (default splits/e2e_train_trainlogs.parquet); VAL_TOKENS likewise
#   MINI=1        the development selection: the first MINI_TRAIN (1000) / MINI_VAL (300) real rows and every scene of
#                 ROUNDS with images (MINI_SIM caps it, 0 = all); no pins
#   WORKERS       default 16
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROUNDS=${ROUNDS:-0 1 2 3 4}
TRAIN_TOKENS=${TRAIN_TOKENS:-$SSR_DATA_ROOT/splits/e2e_train_trainlogs.parquet}
VAL_TOKENS=${VAL_TOKENS:-$CK/data-map/navtrain_val_tokens.parquet}
WORKERS=${WORKERS:-16}
TABLE=$SSR_DATA_ROOT/splits/trainval_simscale.parquet
refuse "$TABLE"            # written last by the builder: its other two files without it are a stopped run, replaced
need "$TRAIN_TOKENS" "$VAL_TOKENS"
for r in $ROUNDS; do need "$SSR_SIMSCALE_ROOT/synthetic_reaction_pdm_v1.0-$r/meta_datas"; done
args=(--train "$TRAIN_TOKENS" --val "$VAL_TOKENS" --sim-root "$SSR_SIMSCALE_ROOT" --rounds $ROUNDS --workers "$WORKERS")
if [ "${MINI:-0}" = 1 ]; then
  caps=("navtrain_train=${MINI_TRAIN:-1000}" "navtrain_val=${MINI_VAL:-300}")
  [ "${MINI_SIM:-0}" != 0 ] && caps+=("simscale=$MINI_SIM")
  args+=(--max-per-source "${caps[@]}")
fi
run_mkdir "$SSR_DATA_ROOT/splits"
run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/build_trainval_simscale_table.py" "${args[@]}"
