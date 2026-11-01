#!/bin/bash
# du of existing data dirs (read-only), parallel, each with timeout
O=$1
for d in /home/external-user/navsim/download/maps /home/external-user/navsim/download/trainval_navsim_logs /home/external-user/navsim/download/trainval_sensor_blobs /home/external-user/navsim/download/test_navsim_logs /home/external-user/navsim/download/test_sensor_blobs /home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100 /home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100 /home/external-user/datasets/teacher_cache/resmap /home/external-user/ssd/yongjae_refiner /home/external-user/yongjae/SSR/data/exp/metric_cache; do
  ( r=$(timeout 900 nice -n 19 ionice -c3 du -s -B1 "$d" 2>/dev/null | cut -f1); echo "$d ${r:-TIMEOUT}" >> $O ) &
done
wait
echo DONE >> $O
