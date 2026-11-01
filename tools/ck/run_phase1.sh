#!/bin/bash
# CK Phase 1 orchestration (report 44 §8; contract pipeline.run_phase1).  One stage per call, or a chain.
#
#   bash tools/ck/run_phase1.sh <stage> [--bg]        # --bg: detach the whole stage (nohup, runs/stage_<stage>.{log,pid})
#   bash tools/ck/run_phase1.sh from:<stage> [--bg]   # run <stage> and every later stage in order
#   bash tools/ck/run_phase1.sh status                # stage markers, live pids, dump progress / ETA
#
# stages (in order):
#   dump        v2 r34 dump of navtrain_val, navtest, navtrain_train, one shard per GPU in $GPUS_DUMP (default 0 1 2)
#   val_build   navtrain-val metric cache (A2) + val lead flags (A6) via tools/ck/data/build_val.sh, lead_split.py
#   label       pack_v2 + label_cands --name cand for every split (CPU, $LABEL_WORKERS)
#   ---- CK_APPROVED=1 required from here on (orchestrator explains to the user first) ----
#   teachers    train_ck T (GPU $GPU_T) and M (GPU $GPU_M) in parallel; epochs $EPOCHS_T
#   kd          kd_targets shards on $GPUS_KD + combine; DET infer on navtrain_val / navtest and MAP on navtest
#   label_kd    official labels of tau'_KD (labels/navtrain_train/kd_corr)
#   student     train_ck S --kd on --corr-aug on (GPU $GPU_S); epochs $EPOCHS_S   (+KD main line only; no C' control)
#   infer       student infer on navtrain_val (GPU $GPU_S) and navtest (GPU $GPU_M)
#   label_corr  official labels of the CK corrections (student val/navtest, DET val/navtest, MAP navtest)
#   eval        eval_ck: student / DET on navtrain_val (beta chosen) then navtest (--fix-from); MAP navtest (grid)
# Each stage writes runs/stage_<stage>.done on success and is skipped when it exists (FORCE=1 to redo).  Every job is
# nohup'ed with runs/<job>.{log,pid}; the stage waits for its jobs and fails if any exits non-zero.  Python steps are
# resumable, so re-running a failed stage continues.  GPUs: only 0-3 (6, 7 belong to another user; 4, 5 not used).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"   # repo root of this worktree
PY=/venv/ssr/bin/python
CKD=${CK_DATA_ROOT:-/home/external-user/ssd/yongjae_refiner/ck}
RUNS=${RUNS:-$CKD/runs}
TAG=${TAG:-phase1}
RUN_T=${RUN_T:-ckT_p1}; RUN_M=${RUN_M:-ckM_p1}; RUN_S=${RUN_S:-ckS_p1}
GPUS_DUMP=${GPUS_DUMP:-"0 1 2"}; GPU_T=${GPU_T:-0}; GPU_M=${GPU_M:-1}; GPUS_KD=${GPUS_KD:-"0 1"}; GPU_X=${GPU_X:-2}
GPU_S=${GPU_S:-0}
EPOCHS_T=${EPOCHS_T:-6}; EPOCHS_S=${EPOCHS_S:-6}
DUMP_WORKERS=${DUMP_WORKERS:-8}; LABEL_WORKERS=${LABEL_WORKERS:-48}; TRAIN_WORKERS=${TRAIN_WORKERS:-8}
DUMP_SPLITS=${DUMP_SPLITS:-"navtrain_val navtest navtrain_train"}
SPLITS_ALL="navtrain_train navtrain_val navtest"
STAGES="dump val_build label teachers kd label_kd student infer label_corr eval"
APPROVAL_STAGES="teachers kd label_kd student infer label_corr eval"
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$RUNS"
cd $REPO

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "ERROR: $*"; exit 1; }

gpu_check() {   # gpu_check <g>: allowed index and (unless ALLOW_BUSY=1) < 2 GB in use
  local g=$1
  case " 0 1 2 3 " in *" $g "*) ;; *) die "GPU $g not allowed (0-3 only)";; esac
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  [ -n "$used" ] || die "nvidia-smi failed for GPU $g"
  if [ "${ALLOW_BUSY:-0}" != 1 ] && [ "$used" -gt 2000 ]; then die "GPU $g busy (${used} MiB); ALLOW_BUSY=1 to override"; fi
}

PIDS=()
launch() {      # launch <job> <gpu|-> <cmd...>: nohup, log + pid in $RUNS; refuses a live duplicate
  local job=$1 gpu=$2; shift 2
  if [ -f "$RUNS/$job.pid" ] && kill -0 "$(cat "$RUNS/$job.pid")" 2>/dev/null; then
    die "$job already running (pid $(cat "$RUNS/$job.pid"))"
  fi
  local cvd=""
  if [ "$gpu" != "-" ]; then gpu_check "$gpu"; cvd=$gpu; fi
  CUDA_VISIBLE_DEVICES=$cvd nohup "$@" >> "$RUNS/$job.log" 2>&1 &
  local pid=$!
  echo $pid > "$RUNS/$job.pid"
  PIDS+=("$job:$pid")
  log "launched $job pid $pid gpu ${gpu} -> $RUNS/$job.log"
}

wait_all() {    # wait for every launched job; fail if any exited non-zero
  local bad=0 j p rc
  for jp in "${PIDS[@]}"; do
    j=${jp%%:*}; p=${jp##*:}
    wait "$p"; rc=$?
    echo "$rc" > "$RUNS/$j.exit"
    if [ "$rc" != 0 ]; then log "job $j (pid $p) exited $rc"; bad=1; else log "job $j done"; fi
  done
  PIDS=()
  [ $bad = 0 ] || die "a job failed (see $RUNS/*.log)"
}

need_approval() {   # a smoke root (CK_DATA_ROOT=.../smoke) needs no approval: tiny data, used to test this script
  case "$CKD" in */smoke|*/smoke/*) return 0;; esac
  case " $APPROVAL_STAGES " in *" $1 "*)
    [ "${CK_APPROVED:-0}" = 1 ] || die "stage $1 needs CK_APPROVED=1 (orchestrator sets it after explaining to the user)";;
  esac
}

# ----------------------------------------------------------------------------------------------- stages
st_dump() {
  local gs=($GPUS_DUMP) n i
  n=${#gs[@]}
  df -h /workspace | tail -1
  for i in "${!gs[@]}"; do
    launch "dump_g${gs[$i]}_s$i" "${gs[$i]}" bash -c "set -e; for s in $DUMP_SPLITS; do \
      $PY tools/ck/data/dump_v2.py --split \$s --shard $i --nshard $n --workers $DUMP_WORKERS --batch-size 8; done"
  done
  date +%s > "$RUNS/dump_start_epoch.txt"
  echo "nshard=$n gpus=$GPUS_DUMP splits=$DUMP_SPLITS workers=$DUMP_WORKERS" > "$RUNS/dump_plan.txt"
  wait_all
  $PY tools/ck/dump_status.py --check || die "dump incomplete"
}

wait_foreign() {  # wait while a data-side job (run_data.sh / build_val.sh / finish_*.sh pid files) touching $1 is alive
  local pat=$1 f p again=1
  while [ $again = 1 ]; do
    again=0
    for f in "$RUNS"/label_*"$pat"*.pid "$RUNS"/label_valtest_cand.pid "$RUNS"/finish_*.pid "$RUNS"/val_mc_part*.pid; do
      [ -f "$f" ] || continue
      case "$(basename "$f")" in label_cand_*|label_kd_*|label_corr_*) continue;; esac   # our own jobs
      p=$(cat "$f")
      if kill -0 "$p" 2>/dev/null; then log "waiting for data job $(basename "$f" .pid) (pid $p)"; again=1; sleep 60; break; fi
    done
  done
}

st_val_build() {   # the data owner usually ran this already (build_val.sh a2 / a6): skip what exists
  wait_foreign val_mc
  if [ ! -f "$CKD/impl-data/val_mc.json" ]; then bash tools/ck/data/build_val.sh a2_finish || die "val metric cache"; fi
  if [ ! -f "$CKD/lead/navtrain_val.parquet" ]; then
    launch lead_val - bash tools/ck/data/build_val.sh a6
    wait_all
  fi
  local s
  for s in navtrain_train navtest; do
    if [ ! -f "$CKD/lead/$s.parquet" ]; then
      launch lead_split_$s - $PY tools/ck/data/lead_split.py --split $s
      wait_all
    fi
  done
}

st_label() {       # pack + official cand labels; skips splits the data owner already finished
  local s n
  for s in navtrain_val navtest navtrain_train; do
    wait_foreign "$s"
    n=$($PY -c "import json,sys; m=json.load(open('$CKD/packed/$s/meta.json')); print(m.get('n', 0))" 2>/dev/null || echo 0)
    if [ "${n:-0}" -lt 1 ] || [ "${FORCE_PACK:-0}" = 1 ]; then
      launch "pack_$s" - $PY tools/ck/data/pack_v2.py --split $s
      wait_all
    fi
    if [ ! -f "$CKD/labels/$s/cand/labels.npy" ]; then
      launch "label_cand_$s" - $PY tools/ck/data/label_cands.py --split $s --name cand --workers $LABEL_WORKERS
      wait_all
    fi
    [ -f "$CKD/labels/$s/cand/labels.npy" ] || die "labels/$s/cand missing"
  done
}

st_teachers() {
  launch train_$RUN_T $GPU_T $PY tools/ck/train_ck.py --arm T --run $RUN_T --epochs $EPOCHS_T \
    --workers $TRAIN_WORKERS --resume ${TRAIN_T_ARGS:-}
  launch train_$RUN_M $GPU_M $PY tools/ck/train_ck.py --arm M --run $RUN_M --epochs $EPOCHS_T --val-split none \
    --workers $TRAIN_WORKERS --resume ${TRAIN_M_ARGS:-}
  wait_all
}

st_kd() {
  local gs=($GPUS_KD) n i
  n=${#gs[@]}
  for i in "${!gs[@]}"; do
    launch "kd_${TAG}_s$i" "${gs[$i]}" $PY tools/ck/kd_targets.py --det-run $CKD/train/$RUN_T \
      --map-run $CKD/train/$RUN_M --tag $TAG --shard $i --nshard $n --workers 6
  done
  launch "infer_${RUN_T}_x" $GPU_X bash -c "set -e; \
    $PY tools/ck/infer_ck.py --run $CKD/train/$RUN_T --split navtrain_val --workers 6; \
    $PY tools/ck/infer_ck.py --run $CKD/train/$RUN_T --split navtest --workers 6; \
    $PY tools/ck/infer_ck.py --run $CKD/train/$RUN_M --split navtest --workers 6"
  wait_all
  launch "kd_${TAG}_combine" - $PY tools/ck/kd_targets.py --det-run $CKD/train/$RUN_T --map-run $CKD/train/$RUN_M \
    --tag $TAG --combine-only --device cpu
  wait_all
}

fresh_flag() {   # fresh_flag <traj npy> <label dir>: '--fresh' when the trajectories are newer than the label plan
  local plan="$2/shards/_plan.json"
  if [ -f "$plan" ] && [ "$1" -nt "$plan" ]; then echo "--fresh"; fi
}

st_label_kd() {
  local traj=$CKD/kd_targets/$TAG/kd_corr_traj.npy
  launch label_kd_corr - $PY tools/ck/data/label_cands.py --split navtrain_train \
    --traj $traj --name kd_corr --workers $LABEL_WORKERS $(fresh_flag $traj $CKD/labels/navtrain_train/kd_corr)
  wait_all
}

st_student() {
  launch train_$RUN_S $GPU_S $PY tools/ck/train_ck.py --arm S --run $RUN_S --epochs $EPOCHS_S --kd on \
    --kd-dir $CKD/kd_targets/$TAG --corr-aug on --workers $TRAIN_WORKERS --resume ${TRAIN_S_ARGS:-}
  wait_all
}

st_infer() {
  launch "infer_${RUN_S}_val" $GPU_S $PY tools/ck/infer_ck.py --run $CKD/train/$RUN_S --split navtrain_val --workers 8
  launch "infer_${RUN_S}_test" $GPU_M $PY tools/ck/infer_ck.py --run $CKD/train/$RUN_S --split navtest --workers 8
  wait_all
}

label_corr_one() {  # label_corr_one <run> <split>
  local traj=$CKD/infer/$1/$2/corr_traj.npy
  launch "label_corr_$1_$2" - $PY tools/ck/data/label_cands.py --split $2 \
    --traj $traj --name corr_$1 --workers $LABEL_WORKERS $(fresh_flag $traj $CKD/labels/$2/corr_$1)
  wait_all
}

st_label_corr() {
  label_corr_one $RUN_S navtrain_val
  label_corr_one $RUN_S navtest
  label_corr_one $RUN_T navtrain_val
  label_corr_one $RUN_T navtest
  label_corr_one $RUN_M navtest
}

st_eval() {
  local r
  for r in $RUN_S $RUN_T; do
    launch "eval_${r}_val" - $PY tools/ck/eval_ck.py --run $r --split navtrain_val
    wait_all
    launch "eval_${r}_test" - $PY tools/ck/eval_ck.py --run $r --split navtest \
      --fix-from $CKD/eval/$r/navtrain_val/metrics.json
    wait_all
  done
  launch "eval_${RUN_M}_test" - $PY tools/ck/eval_ck.py --run $RUN_M --split navtest
  wait_all
}

run_stage() {
  local s=$1
  case " $STAGES " in *" $s "*) ;; *) die "unknown stage $s (stages: $STAGES)";; esac
  if [ -f "$RUNS/stage_$s.done" ] && [ "${FORCE:-0}" != 1 ]; then log "stage $s already done ($RUNS/stage_$s.done)"; return 0; fi
  need_approval "$s"
  log "=== stage $s ==="
  "st_$s"
  date '+%F %T' > "$RUNS/stage_$s.done"
  log "=== stage $s done ==="
}

status() {
  local s
  for s in $STAGES; do
    if [ -f "$RUNS/stage_$s.done" ]; then echo "stage $s: done $(cat "$RUNS/stage_$s.done")"; else echo "stage $s: -"; fi
  done
  for f in "$RUNS"/*.pid; do
    [ -f "$f" ] || continue
    if kill -0 "$(cat "$f")" 2>/dev/null; then echo "running $(basename "$f" .pid) pid $(cat "$f")"; fi
  done
  $PY tools/ck/dump_status.py || true
}

# ----------------------------------------------------------------------------------------------- main
# Everything runs inside main(), called on the last line together with exit: bash has then parsed the whole file,
# so editing this script while a long stage is running cannot break that stage.
main() {
  local arg=${1:-} bg=${2:-} name start go s
  [ -n "$arg" ] || { sed -n '2,30p' "$0"; return 2; }
  if [ "$bg" = "--bg" ]; then
    name=${arg//:/_}
    nohup bash "$0" "$arg" > "$RUNS/stage_$name.log" 2>&1 &
    echo $! > "$RUNS/stage_$name.pid"
    log "detached $arg pid $! -> $RUNS/stage_$name.log"
    return 0
  fi
  case "$arg" in
    status) status ;;
    from:*)
      start=${arg#from:}; go=0
      for s in $STAGES; do
        [ "$s" = "$start" ] && go=1
        [ $go = 1 ] && run_stage "$s"
      done
      [ $go = 1 ] || die "unknown stage $start" ;;
    *) run_stage "$arg" ;;
  esac
}
main "$@"; exit $?
