#!/bin/bash
# refiner-v3 GT-object provenance (review fix: causality finding 5, protocol D5).  CPU only, idempotent.
#
#   bash tools/ck/v3/setup_gt_objects.sh
#
# D3/gt_objects/<split> is the ONLY place build_obj_pack.py reads GT tracks from (no silent fallback):
#   navtrain_val    a COPY (cp -a) of SSR/report/53_eval_and_plugin_evidence/A_full16/gt_objects/navtrain_val
#                   (built 2026-10-11 02:57 KST; the report folder is editable, so the pack must not read it)
#   navtrain_train  a symlink to /home/external-user/ssd/yongjae_refiner/objects/navtrain (data dir, built 2026-10-03)
#   navtest         BUILT here from /home/external-user/navsim/download/test_navsim_logs/test with
#                   tools/refiner/build_future_objects.py (= the SSR copy byte for byte; the builder itself puts the
#                   SSR checkout first on sys.path, so it is run with python -B and PYTHONDONTWRITEBYTECODE=1: nothing
#                   is written into SSR).  Targets only: navtest stays descriptive (review causality finding 2).
# Each dir gets manifest.json (per-file sha256, tools/ck/v3/sha_manifest.py) under D3/gt_objects/_manifests/.
set -euo pipefail
V3=/workspace/yongjae/SSR-v3
D3=/home/external-user/ssd/yongjae_refiner/ck/v3
PY=/venv/ssr/bin/python
G=$D3/gt_objects
SRC_VAL=/workspace/yongjae/SSR/report/53_eval_and_plugin_evidence/A_full16/gt_objects/navtrain_val
SRC_TRAIN=/home/external-user/ssd/yongjae_refiner/objects/navtrain
TEST_LOGS=/home/external-user/navsim/download/test_navsim_logs/test
ENVC="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
mkdir -p "$G/_inputs" "$G/_manifests" "$D3/logs"
cd "$V3"
# navtrain_val: copy (never a link into the editable report folder)
if [ ! -e "$G/navtrain_val/index.parquet" ]; then
  rm -rf "$G/navtrain_val.tmp"
  cp -a "$SRC_VAL" "$G/navtrain_val.tmp"
  mv "$G/navtrain_val.tmp" "$G/navtrain_val"
fi
# navtrain_train: link (guarded: a second `ln -s` onto a dir link would nest a link INSIDE the target)
[ -e "$G/navtrain_train" ] || ln -s "$SRC_TRAIN" "$G/navtrain_train"
# navtest: build
if [ ! -e "$G/navtest/index.parquet" ]; then
  env $ENVC PYTHONPATH="$V3" $PY -B -c "
import pandas as pd
p = pd.read_parquet('/home/external-user/ssd/yongjae_refiner/ck/packed/navtest/tokens.parquet')
p[['token', 'log']].to_parquet('$G/_inputs/navtest_tokens.parquet', index=False)
print(len(p), 'navtest tokens')"
  env $ENVC nice -n 10 $PY -B tools/refiner/build_future_objects.py --tokens "$G/_inputs/navtest_tokens.parquet" \
    --logs "$TEST_LOGS" --out "$G/navtest" --workers 4 > "$D3/logs/gt_objects_navtest.log" 2>&1
  # metric-cache validation on 400 tokens (as for val).  NOTE: with --mc-root the builder rewrites index.parquet
  # (validation columns for the sampled rows), so it runs BEFORE the manifest below.
  env $ENVC nice -n 10 $PY -B tools/refiner/build_future_objects.py --tokens "$G/_inputs/navtest_tokens.parquet" \
    --logs "$TEST_LOGS" --out "$G/navtest" --workers 4 --mc-root /workspace/yongjae/SSR/data/exp/metric_cache \
    --sample 400 --seed 0 --report "$G/_manifests/navtest_mc_validation.json" \
    > "$D3/logs/gt_objects_navtest_validation.log" 2>&1
fi
for s in navtrain_val navtrain_train navtest; do
  [ -f "$G/_manifests/$s.json" ] || env $ENVC PYTHONPATH="$V3" nice -n 10 $PY -B tools/ck/v3/sha_manifest.py \
    --dir "$G/$s" --out "$G/_manifests/$s.json" --workers 16
done
ls -la "$G"
