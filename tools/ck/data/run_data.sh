#!/bin/bash
# CK Phase 1 data stages (CPU + v2 dump GPU).  nohup + pid files in CK_DATA/runs/ (no pkill/pgrep -f).
# Usage: bash tools/ck/data/run_data.sh STAGE [args]
#   dump  "0 1 2"      one dump process per listed GPU (GPU 0-3 only); --claim over 24 log-balanced shards per split,
#                      splits in order navtrain_val, navtrain_train, navtest.  Re-running / adding a GPU later is safe
#                      (claims are atomic, dead claims are taken over, finished tokens are skipped).
#   label_watch S W    label v2 top-16 candidates of split S straight from the dump while it is written (W workers)
#   pack S             pack_v2.py --split S (after the dump of S is complete)
#   label S W          label_cands.py --split S --name cand (assemble; scores what is left)
#   lead               lead flags for navtrain_train / navtest (copies) and navtrain_val (extract_lead, needs val mc)
#   check [S...]       integrity checks -> CK_DATA/impl-data/check_<split>.json
#   status             dump / label progress
# Worker budget (<= 48 per agent): dump 3 x DUMP_WORKERS(12) + label LABEL_WORKERS(12) = 48.
set -u
CKR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"   # repo root of this worktree
CKD=/home/external-user/ssd/yongjae_refiner/ck
RUNS=$CKD/runs
PY=/venv/ssr/bin/python
DUMP_WORKERS=${DUMP_WORKERS:-12}
NSHARD=${NSHARD:-24}
SPLITS_ORDER=${SPLITS_ORDER:-navtrain_val,navtrain_train,navtest}
export PYTHONPATH=$CKR OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $CKR
mkdir -p $RUNS

running() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

case "${1:-}" in
  dump)
    for g in ${2:-0 1 2}; do
      case $g in 0|1|2|3) ;; *) echo "GPU $g not allowed (0-3 only)"; exit 2 ;; esac
      if running $RUNS/dump_g$g.pid; then echo "dump on GPU $g already running"; continue; fi
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
      if [ "$used" -gt 2000 ]; then echo "GPU $g busy (${used} MiB used), skipped"; continue; fi
      CUDA_VISIBLE_DEVICES=$g nohup $PY tools/ck/data/dump_v2.py --split $SPLITS_ORDER --claim --nshard $NSHARD \
        --shard $(( g * NSHARD / 4 )) --workers $DUMP_WORKERS --batch-size 8 > $RUNS/dump_g$g.log 2>&1 &
      echo $! > $RUNS/dump_g$g.pid
      echo "dump GPU $g pid $(cat $RUNS/dump_g$g.pid) -> $RUNS/dump_g$g.log"
    done ;;
  label_watch)
    S=$2; W=${3:-12}
    if running $RUNS/label_${S}_cand.pid; then echo "label $S running"; exit 0; fi
    CUDA_VISIBLE_DEVICES= nohup nice -n 5 $PY tools/ck/data/label_cands.py --split $S --name cand --from-dump --watch \
      --poll 60 --workers $W > $RUNS/label_${S}_cand.log 2>&1 &
    echo $! > $RUNS/label_${S}_cand.pid; echo "label_watch $S pid $(cat $RUNS/label_${S}_cand.pid)" ;;
  pack)
    CUDA_VISIBLE_DEVICES= $PY tools/ck/data/pack_v2.py --split $2 ;;
  label)
    S=$2; W=${3:-48}
    CUDA_VISIBLE_DEVICES= nice -n 5 $PY tools/ck/data/label_cands.py --split $S --name cand --from-dump --workers $W ;;
  lead)
    CUDA_VISIBLE_DEVICES= $PY tools/ck/data/lead_split.py --split navtrain_train
    CUDA_VISIBLE_DEVICES= $PY tools/ck/data/lead_split.py --split navtest
    CUDA_VISIBLE_DEVICES= $PY tools/ck/data/lead_split.py --split navtrain_val --workers ${LEAD_WORKERS:-24} ;;
  check)
    shift; CUDA_VISIBLE_DEVICES= $PY tools/ck/data/check_data.py ${@:-navtrain_val navtrain_train navtest} ;;
  status)
    for g in 0 1 2 3; do
      [ -f $RUNS/dump_g$g.log ] || continue
      st=done; running $RUNS/dump_g$g.pid && st=running
      echo "dump g$g $st | $(grep -a 'tok/s' $RUNS/dump_g$g.log | tail -1)"
    done
    for s in navtrain_val navtrain_train navtest; do
      d=$CKD/v2dump/$s/claims; [ -d $d ] && echo "$s shards done $(ls $d | grep -c '\.done$')/$NSHARD claimed $(ls $d | grep -c '\.json$')"
      [ -f $RUNS/label_${s}_cand.log ] && echo "  label $s | $(grep -a 'chunks\|waiting\|assembled' $RUNS/label_${s}_cand.log | tail -1)"
    done ;;
  *) echo "usage: $0 {dump \"GPUS\"|label_watch S W|pack S|label S W|lead|check [S..]|status}"; exit 2 ;;
esac
