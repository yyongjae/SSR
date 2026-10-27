# km 10/16: copy to scripts/km_corridor/env.sh and edit for the host (or let train.sh source env.example.sh directly).
export REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -d /home/external-user/ssd/kyungmin_ck ]; then
  export KM_ROOT=${KM_ROOT:-/home/external-user/ssd/kyungmin_ck}          # GT corridor caches + run outputs (needs ~35 GB)
else
  export KM_ROOT=${KM_ROOT:-/workspace/kyungmin/km_ck}
fi
if [ -x /venv/ssr/bin/python ]; then
  export PY=${PY:-/venv/ssr/bin/python}
else
  export PY=${PY:-/home/external-user/miniconda3/envs/ssr/bin/python}
fi
# planning labels of the 256 anchors (not in git, 253 MB): pdm_score_256.npy + pdm_score_256.tokens.json
if [ -f /home/external-user/kyungmin/SSR-v2/data/planning_vb/pdm_score_256.npy ]; then
  export PLAN_SCORE=${PLAN_SCORE:-/home/external-user/kyungmin/SSR-v2/data/planning_vb/pdm_score_256}
else
  export PLAN_SCORE=${PLAN_SCORE:-$KM_ROOT/planning_vb/pdm_score_256}
fi
export PLAN_ANCHORS=${PLAN_ANCHORS:-$REPO/resources/km_corridor/trajectory_anchors_256.npy}
export TOKENS_ROOT=${TOKENS_ROOT:-$REPO/resources/km_corridor/packed}          # <split>/tokens.parquet (row order of every cache)
# NAVSIM data (overwrite container default OPENSCENE_DATA_ROOT=/workspace/navsim_workspace/dataset which has nested trainval/trainval)
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
if [ -d /home/external-user/kyungmin/SSR/data/dataset/maps ]; then
  export NUPLAN_MAPS_ROOT=/home/external-user/kyungmin/SSR/data/dataset/maps
  export OPENSCENE_DATA_ROOT=/home/external-user/kyungmin/SSR/data/dataset
else
  export NUPLAN_MAPS_ROOT=/home/external-user/yongjae/SSR/data/dataset/maps
  export OPENSCENE_DATA_ROOT=/home/external-user/yongjae/SSR/data/dataset
fi
export NAVSIM_EXP_ROOT=$KM_ROOT/_exp
export NAVSIM_DEVKIT_ROOT=$REPO PYTHONPATH=$REPO
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# metric caches (GT corridor source = what the official scorer reads) are set in tools/ck/data/common.py (MC_ROOT)
export KM_CORR_ROOT=$(dirname $TOKENS_ROOT)
# 64m (100x100) frozen teacher BEV caches for BEV Selector v15 distillation (bevfusion/cache_{train,val}_100x100 + resmap/stage1_front64)
export TEACHER_CACHE_ROOT=${TEACHER_CACHE_ROOT:-/workspace/teacher_cache}
