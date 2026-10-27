#!/usr/bin/env bash
# km 10/16: GT corridor caches from the metric cache.  usage: build_gt.sh 32 | 64 | route32 | exact | cf [workers]
#   32      -> $KM_ROOT/corridor_dt01  (static SDF + 40 x 0.1 s occupancy planes, 0-32 m)    ~14 GB, ~2 h at 9 workers
#   64      -> $KM_ROOT/corridor64     (0-64 m SDF, route progress/distance, agent polygons) ~14 GB
#   route32 -> $KM_ROOT/route32        (first 50 rows of corridor64 route channels; v3 32 m needs it, build 64 first)
#   exact   -> $KM_ROOT/exact          (exact candidate scorer polygons + PDM-Closed states + full centerline)
#   cf      -> builds both 64 ($KM_ROOT/corridor64) and exact ($KM_ROOT/exact) for cf / cf_v15
set -uo pipefail
if [ -f "$(dirname "$0")/env.sh" ]; then source "$(dirname "$0")/env.sh"; else source "$(dirname "$0")/env.example.sh"; fi; cd $REPO
W=${2:-64}
case $1 in
  32) for s in navtrain_val navtrain_train; do nice $PY -W ignore tools/corridor/build_corridor_dt01.py --split $s --out $KM_ROOT/corridor_dt01 --workers $W --root $(dirname $TOKENS_ROOT) || exit 1; done ;;
  64) for s in navtrain_val navtrain_train; do nice $PY -W ignore tools/corridor/build_corridor64.py --split $s --out $KM_ROOT/corridor64 --workers $W --root $(dirname $TOKENS_ROOT) || exit 1; done ;;
  route32) $PY tools/corridor/crop_route32.py $KM_ROOT/corridor64 $KM_ROOT/route32 ;;
  exact) for s in navtrain_val navtrain_train; do nice $PY -W ignore tools/corridor/build_exact.py --split $s --out $KM_ROOT/exact --workers $W --root $(dirname $TOKENS_ROOT) --cl-full || exit 1; done ;;
  cf)
    for s in navtrain_val navtrain_train; do nice $PY -W ignore tools/corridor/build_corridor64.py --split $s --out $KM_ROOT/corridor64 --workers $W --root $(dirname $TOKENS_ROOT) || exit 1; done
    for s in navtrain_val navtrain_train; do nice $PY -W ignore tools/corridor/build_exact.py --split $s --out $KM_ROOT/exact --workers $W --root $(dirname $TOKENS_ROOT) --cl-full || exit 1; done
    ;;
  *) echo "usage: $0 32|64|route32|exact|cf [workers]"; exit 2 ;;
esac
