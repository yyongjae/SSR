#!/bin/bash
# M6 V2 validation: builder vs official metric caches (navtest random 500, navtest NC failures, navtrain E random 500).
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PY=/home/external-user/miniconda3/envs/ssr/bin/python
B=/home/external-user/yongjae/SSR/tools/refiner/build_future_objects.py
V=/home/external-user/ssd/yongjae_refiner/objects/_validation
TEST_LOGS=/home/external-user/navsim/download/test_navsim_logs/test
TRAIN_LOGS=/home/external-user/yongjae/SSR/data/dataset/navsim_logs/trainval
nice -n 10 $PY $B --tokens $V/navtest_all_token_log.parquet --logs $TEST_LOGS --out $V/navtest --workers 2 --overwrite --sample 500 --seed 0 \
  --mc-root /home/external-user/yongjae/SSR/data/exp/metric_cache --report $V/navtest_rand500.json
nice -n 10 $PY $B --tokens $V/navtest_ncfail_token_log.parquet --logs $TEST_LOGS --out $V/navtest --workers 2 --overwrite \
  --mc-root /home/external-user/yongjae/SSR/data/exp/metric_cache --report $V/navtest_ncfail.json
nice -n 10 $PY $B --tokens /home/external-user/yongjae/SSR/report/cause_and_correction_tests/E_train_split_feasibility/tokens/sample.parquet \
  --logs $TRAIN_LOGS --out $V/navtrain_E --workers 2 --overwrite --sample 500 --seed 0 \
  --mc-root /home/external-user/yongjae/SSR/report/cause_and_correction_tests/E_train_split_feasibility/metric_cache --report $V/navtrain_E_rand500.json
echo DONE
