#!/usr/bin/env bash
# ck/packed/<split> from the logs: tokens, gt_traj, status, ok.  No planner.  ~1 min, CPU.  Compared with the pins before the
# directory appears.  On a server with an older packed directory: EXPECT_FROM=<dir holding navtrain_train, navtrain_val>
# (compares row by row) and OUT_ROOT=<elsewhere>.
#   SPLITS  default "navtrain_train navtrain_val"; navtrain_val is only for retraining the refiner teachers.
#           trainval_simscale: the rows of splits/trainval_simscale.parquet (no pins; its real rows are compared with the
#           packed navtrain parts that exist), SIM_WORKERS processes (default 8); with RESUME=1 a finished directory of
#           the same token table is kept.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SPLITS=${SPLITS:-navtrain_train navtrain_val}
OUT_ROOT=${OUT_ROOT:-$CK/packed}
EXPECT_FROM=${EXPECT_FROM:-}
need "$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
for s in $SPLITS; do
  if [ "$s" = trainval_simscale ]; then
    need "$SSR_DATA_ROOT/splits/trainval_simscale.parquet"
    args=(--split "$s" --out "$OUT_ROOT/$s" --tokens "$SSR_DATA_ROOT/splits/trainval_simscale.parquet" --workers "${SIM_WORKERS:-8}")
    if [ "$RESUME" = 1 ]; then args+=(--skip-done); else refuse "$OUT_ROOT/$s"; fi
  else
    refuse "$OUT_ROOT/$s"
    args=(--split "$s" --out "$OUT_ROOT/$s" --workers 2)
  fi
  [ -n "$EXPECT_FROM" ] && args+=(--expect-from "$EXPECT_FROM/$s")
  run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/repack_dataset.py" "${args[@]}"
done
