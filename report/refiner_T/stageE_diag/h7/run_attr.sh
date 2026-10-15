#!/usr/bin/env bash
# H7: PDM failure attribution (cause objects, DAC exit) for E2 tau0 (A) and E2 tau_final (S) on the 1,648 tokens
# failing NC/DAC/TTC in any of A, S, R_T4(A), R_M4(A). CPU only, 4 workers.
set -e
cd /home/external-user/yongjae/SSR
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUPLAN_MAPS_ROOT=/home/external-user/yongjae/SSR/data/dataset/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/home/external-user/yongjae/SSR/data/dataset
PY=/home/external-user/miniconda3/envs/ssr/bin/python
H=/home/external-user/ssd/yongjae_refiner/stageE_diag/h7
S=report/refiner_T/stageE_diag/h7/rescore_attr_h7.py
export H7_TOKENS=$H/attr_tokens.json
for arm in A:e2_tau0 S:e2_final; do
  n=${arm%%:*}; p=${arm##*:}
  mkdir -p $H/attr_$n
  H7_OUT=$H/attr_$n H7_TRAJ_PKL=/home/external-user/ssd/yongjae_refiner/stageE_diag/${p}_navtest_trajectories.pkl \
    nice -n 10 $PY $S --workers 4 --chunk 50 2>&1 | grep -v -i warning > $H/logs/attr_$n.log
done
echo ALLDONE
