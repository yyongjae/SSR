#!/bin/bash
# CK data: navtrain VAL-log builds (CPU only).  Port of ck/data-map/val_build/commands.sh.
#   REQUIRED  A2  metric cache for the 18,179 navtrain-val tokens (12 log-balanced parts, $MC_WORKERS workers each)
#             A6  lead flags -> CK_DATA/lead/navtrain_val.parquet (tools/ck/data/lead_split.py --split navtrain_val)
#   OPTIONAL  A1/A3/A4 (objects / SDF / side for val surrogate logging) are NOT needed for Phase 1 (not run here).
# Usage:  bash tools/ck/data/build_val.sh {a2|a2_status|a2_finish|a6|all}
#   a2        launches the 12 parts in the background (nohup, pid files) and returns
#   a2_status prints done/total per part
#   a2_finish waits for the parts, checks counts, restores the stage-T build_summary.json, writes impl-data/val_mc.json
#   a6        (after a2_finish) builds the lead flags in the foreground
#   all       a2 + a2_finish + a6 (foreground; use nohup for this one)
# Rules: OMP_NUM_THREADS=1, workers summed <= 48, no pkill/pgrep -f (pid files), navsim pinned to SSR-ck (run_pinned.py).
set -u
CKR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"   # repo root of this worktree
D=/home/external-user/ssd/yongjae_refiner
CKD=$D/ck
VB=$CKD/data-map/val_build
RUNS=$CKD/runs
PY=/venv/ssr/bin/python
MC_WORKERS=${MC_WORKERS:-3}          # per part; 12 parts -> 36 workers (leaves room for the dump smoke workers)
LEAD_WORKERS=${LEAD_WORKERS:-24}
export PYTHONPATH=$CKR OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
cd $CKR
mkdir -p $RUNS $D/metric_cache/logs $CKD/impl-data

a2() {
  if [ ! -f $D/metric_cache/_state/build_summary_before_val_backup.json ]; then
    cp -p $D/metric_cache/_state/build_summary.json $D/metric_cache/_state/build_summary_before_val_backup.json
  fi
  date +%s > $RUNS/val_mc_start_epoch.txt
  for i in $(seq 1 12); do
    if [ -f $RUNS/val_mc_part$i.pid ] && kill -0 $(cat $RUNS/val_mc_part$i.pid) 2>/dev/null; then
      echo "part $i already running (pid $(cat $RUNS/val_mc_part$i.pid))"; continue
    fi
    nohup nice -n 10 $PY tools/ck/data/run_pinned.py build_metric_cache build --splits val_mc_part$i --splits-dir $VB \
      --workers $MC_WORKERS --no-final-check > $RUNS/val_mc_part$i.log 2>&1 &
    echo $! > $RUNS/val_mc_part$i.pid
  done
  echo "launched 12 parts x $MC_WORKERS workers; logs $RUNS/val_mc_part*.log"
}

a2_status() {
  for i in $(seq 1 12); do
    st="done"; kill -0 $(cat $RUNS/val_mc_part$i.pid 2>/dev/null) 2>/dev/null && st="running"
    echo "part $i $st | $(grep '^\[' $RUNS/val_mc_part$i.log 2>/dev/null | tail -1 | cut -c1-160)"
  done
}

a2_finish() {
  for i in $(seq 1 12); do
    while kill -0 $(cat $RUNS/val_mc_part$i.pid) 2>/dev/null; do sleep 20; done
  done
  cp -p $D/metric_cache/_state/build_summary_before_val_backup.json $D/metric_cache/_state/build_summary.json
  $PY - <<'EOF'
import json, time
from pathlib import Path
import pandas as pd
D = Path('/home/external-user/ssd/yongjae_refiner')
v = pd.read_parquet(D / 'ck/data-map/navtrain_val_tokens.parquet')
have = [(D / 'metric_cache' / l / 'unknown' / t / 'metric_cache.pkl').is_file() for t, l in zip(v.token, v.log)]
fail = D / 'metric_cache/_state/failed_tokens.tsv'
fails = [r.split('\t')[0] for r in fail.read_text().splitlines()] if fail.exists() else []
vt = set(v.token)
res = dict(n=len(v), n_have=int(sum(have)), n_missing=int(len(v) - sum(have)),
           failed_lines_val=sum(t in vt for t in fails), finished=time.strftime('%F %T'))
logs = sorted((D / 'ck/runs').glob('val_mc_part*.log'))
res['part_tails'] = {p.name: p.read_text().strip().splitlines()[-1][:300] for p in logs}
(D / 'ck/impl-data/val_mc.json').write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != 'part_tails'}))
EOF
}

a6() {
  $PY tools/ck/data/lead_split.py --split navtrain_val --workers $LEAD_WORKERS
}

case "${1:-}" in
  a2) a2 ;;
  a2_status) a2_status ;;
  a2_finish) a2_finish ;;
  a6) a6 ;;
  all) a2; a2_finish; a6 ;;
  *) echo "usage: $0 {a2|a2_status|a2_finish|a6|all}"; exit 2 ;;
esac
