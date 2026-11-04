#!/usr/bin/env bash
# Token tables, from the official scene filter and log split: navtrain_train (85,109), navtrain_val (18,179), navtest
# (12,146).  ~1 min, CPU.  Each is compared with its pin and with the copy tracked in resources/km_corridor/packed.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need "$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
refuse "$SSR_DATA_ROOT/splits/e2e_train_trainlogs.parquet" "$CK/data-map/navtrain_val_tokens.parquet" "$SSR_DATA_ROOT/splits/navtest.parquet"
run_mkdir "$SSR_DATA_ROOT/splits" "$CK/data-map"
run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/derive_tokens.py" --split navtrain_train --out "$SSR_DATA_ROOT/splits/e2e_train_trainlogs.parquet"
run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/derive_tokens.py" --split navtrain_val --out "$CK/data-map/navtrain_val_tokens.parquet"
# navtest table: research tools only, skipped until the test logs are there
if [ -d "$SSR_TEST_DATA_ROOT/navsim_logs/test/test" ] || [ "$SETUP_DRY" = 1 ]; then
  run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/derive_tokens.py" --split navtest --out "$SSR_DATA_ROOT/splits/navtest.parquet"
fi
