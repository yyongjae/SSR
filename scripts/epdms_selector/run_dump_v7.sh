#!/bin/bash
set -euo pipefail

BASE=/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main
CKPT_DIR=/workspace/yongjae/ssd/yongjae_refiner/v7/e2e/v2_v15_v7_main/train/lightning_logs/version_0/checkpoints
CKPT22=$CKPT_DIR/epoch=22-step=15295.ckpt
CKPT29=$CKPT_DIR/epoch=29-step=19950.ckpt
PY=/venv/ssr/bin/python
SCRIPT=$BASE/dump_96pool_v7.py
EP30_ROOT=/workspace/yongjae/ssd/yongjae_refiner/v7/eval/v2_v15_v7_main_ep30/eval/root

echo "=== [1/4] Dumping Ep29 (30ep) navhard 96-pool on GPUs 4,5,6,7 ==="
OUT_NH29=$BASE/pool96_ep29_navhard
mkdir -p $OUT_NH29
for i in 0 1 2 3; do
  g=$((i + 4))
  CUDA_VISIBLE_DEVICES=$g CK_BEV64=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY -B $SCRIPT dump --split navhard --out $OUT_NH29 \
    --seed-eval-root $EP30_ROOT --seed-run-name v2_v15_v7_main_ep30 \
    --ckpt $CKPT29 --part $i/4 --device cuda:0 --bs 8 --workers 2 --threads 2 > $OUT_NH29/part$i.log 2>&1 &
done
wait
$PY -B $SCRIPT merge --split navhard --out $OUT_NH29

echo "=== [2/4] Dumping Ep22 (23ep) navtest 96-pool on GPUs 4,5,6,7 ==="
OUT_NT22=$BASE/pool96_ep22_navtest
mkdir -p $OUT_NT22
for i in 0 1 2 3; do
  g=$((i + 4))
  CUDA_VISIBLE_DEVICES=$g CK_BEV64=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY -B $SCRIPT dump --split navtest --out $OUT_NT22 \
    --ckpt $CKPT22 --part $i/4 --device cuda:0 --bs 8 --workers 2 --threads 2 > $OUT_NT22/part$i.log 2>&1 &
done
wait
$PY -B $SCRIPT merge --split navtest --out $OUT_NT22

echo "=== [3/4] Dumping Ep22 (23ep) navhard 96-pool on GPUs 4,5,6,7 ==="
OUT_NH22=$BASE/pool96_ep22_navhard
mkdir -p $OUT_NH22
for i in 0 1 2 3; do
  g=$((i + 4))
  CUDA_VISIBLE_DEVICES=$g CK_BEV64=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY -B $SCRIPT dump --split navhard --out $OUT_NH22 \
    --seed-dir $OUT_NT22 \
    --ckpt $CKPT22 --part $i/4 --device cuda:0 --bs 8 --workers 2 --threads 2 > $OUT_NH22/part$i.log 2>&1 &
done
wait
$PY -B $SCRIPT merge --split navhard --out $OUT_NH22

echo "=== [4/4] Labeling Ep22 (23ep) navtest 96-pool (lat_traj and cand96) for v1 PDMS ==="
(cd /workspace/yongjae/SSR-v7 && CK_BEV64=1 PYTHONPATH=/workspace/yongjae/SSR-v7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $PY -B -c "
from pathlib import Path
from tools.ck.data import label_cands as LC
er = Path('$EP30_ROOT')
out_nt22 = Path('$OUT_NT22')
LC.run('navtest', er / 'packed/navtest', out_nt22 / 'labels_lat96', traj_file=out_nt22 / 'lat_traj.npy', workers=48)
LC.run('navtest', er / 'packed/navtest', out_nt22 / 'labels_pool96', traj_file=out_nt22 / 'cand96.npy', workers=48)
" > $OUT_NT22/label_pdms.log 2>&1)

echo "=== ALL V7 DUMPS AND LABELS DONE ==="
