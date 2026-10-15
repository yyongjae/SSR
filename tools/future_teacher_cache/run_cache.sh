#!/usr/bin/env bash
# Run the unchanged 50x100 BEVFusion teacher on future frames. Usage: run_cache.sh <gpu> <part>
# Same npz format as cache_{train,val}_50x100 (bev_feature included). Resumable (--skip-existing).
set -u
GPU=$1; PART=$2
W=/home/external-user/yongjae/SSR/tools/future_teacher_cache
OUT=/home/external-user/datasets/teacher_cache/bevfusion
export PATH=/home/external-user/miniconda3/envs/bevfusion/bin:$PATH NCCL_SOCKET_IFNAME=lo OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$GPU
cd /home/external-user/yongjae/bevfusion
for S in test train; do
  if [ $S = test ]; then D=$OUT/cache_val_50x100_future; SPLIT=val; else D=$OUT/cache_train_50x100_future; SPLIT=train; fi
  echo "[$(date '+%F %T')] start $S part$PART on GPU $GPU -> $D"
  torchpack dist-run -np 1 python $W/cache_teacher_future.py \
    configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml runs/navsim-fusion-50x100/epoch_20.pth \
    --cache-dir $D --split $SPLIT --ann-file $W/infos/future_${S}_part${PART}.pkl --mem-frac 0.3 --skip-existing
  echo "[$(date '+%F %T')] end $S part$PART exit $?"
done
