#!/usr/bin/env bash
# Persistent collector lane: GPU 5 runs shards 0/2/4, GPU 7 runs 1/3/5.
set -euo pipefail

GPU="$1"
START="$2"
NSHARD=6
OUT=/data1/yong/SSR/work_dirs/analysis/teacher_stitch
PY=/home/yongjae/miniconda3/envs/ssr/bin/python
ROOT=/home/yongjae/e2e/SSR

mkdir -p "$OUT/logs"
cd "$ROOT"

for ((SHARD=START; SHARD<NSHARD; SHARD+=2)); do
  env CUDA_VISIBLE_DEVICES="$GPU" NUMBA_CPU_NAME=generic \
    "$PY" tools/teacher_stitch_worker.py \
    --shard "$SHARD" --nshard "$NSHARD" --out "$OUT" \
    --cache-stride 5 --workers 8 --teacher-workers 8 \
    > "$OUT/logs/shard_${SHARD}.log" 2>&1
done
