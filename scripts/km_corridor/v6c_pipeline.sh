#!/usr/bin/env bash
# refiner v6c (report 124 / 128): stage T teachers in parallel on 4 GPUs, then the student.
#   usage: v6c_pipeline.sh <name> [student run name]            start (run it detached, see below)
#          v6c_pipeline.sh <name> status                        progress of every stage
#   detached: setsid nohup bash scripts/km_corridor/v6c_pipeline.sh <name> <student> > <log> 2>&1 &
#
#   stages (a stage with a done marker is skipped, so a re-launch resumes; the field / T-E copies also resume from
#   their last finished epoch):
#     1 fields   T-D copy 0 @G0, T-D copy 1 @G1 (DET, BEVFusion) | T-M copy 0 @G2, T-M copy 1 @G3 (MAP, ReSMap)
#     2 platt    merge the four copies -> teachers_fields.pt; occupancy Platt @G0 -> teachers_platt_occ.pt
#     3 te       T-E copy 0 DDP @G0,G1 | T-E copy 1 DDP @G2,G3 (global batch = BATCH_T, the single-GPU recipe)
#     4 finish   T-E overlap / approach Platt + T-M off-road Platt @G0 -> teachers_te.pt (fields, platt_occ, te)
#     5 student  train.sh cf_seg_v15_v6c <student> G0,G1,G2,G3 with V6C_TEACHER=<out>/teachers_te.pt (stage D)
#   GPUS (default 4,5,6,7 = G0..G3).  Copy f of every module learns from fold-f tokens only (cf_fold = crc32(log) % 2).
#   env: EPOCHS_T (10), BATCH_T (8), LIMIT_T (0 = every token), PLATT_BATCHES (200), LR_T (2e-4),
#        WORKERS_T (16: loader workers per field copy; the corridor targets are CPU-bound, 0.2-0.4 s per token),
#        WORKERS_TE (8 per T-E rank)
#        STUDENT=0 stops after stage 4.  The student stage passes the environment on to train.sh (EPOCHS,
#        LIMIT_TRAIN, LIMIT_VAL, V6C_WARMUP, V6C_FIX, V6C_KD, WANDB, ...); WANDB_GROUP defaults to refiner-v6c.
#   outputs: $KM_RUN_ROOT/v6c_teacher/<name>/ (parts, teachers_*.pt, logs/, done_*, status.log);
#            the student run in $KM_RUN_ROOT/corr30/<student>
set -uo pipefail
export KM_ROOT=${KM_ROOT:-/workspace/navsim_workspace/corridor_cache}
if [ -f "$(dirname "$0")/env.sh" ]; then source "$(dirname "$0")/env.sh"; else source "$(dirname "$0")/env.example.sh"; fi
cd $REPO
NAME=${1:?usage: v6c_pipeline.sh <name> [student run name | status]}
OUT=${KM_RUN_ROOT:-$REPO/work_dirs/km_runs}/v6c_teacher/$NAME
mkdir -p $OUT/logs
kst() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST'; }
say() { echo "[$(kst)] $*" | tee -a $OUT/status.log; }

if [ "${2:-}" = status ]; then
  cat $OUT/status.log 2>/dev/null | tail -20
  for st in fields platt te finish student; do printf '%-8s %s\n' $st "$( [ -f $OUT/done_$st ] && cat $OUT/done_$st || echo pending)"; done
  for f in $OUT/log_td_f0.jsonl $OUT/log_td_f1.jsonl $OUT/log_tm_f0.jsonl $OUT/log_tm_f1.jsonl \
           $OUT/log_te_fold_f0.jsonl $OUT/log_te_fold_f1.jsonl; do
    [ -f $f ] && echo "$(basename $f): $(grep -o '"epoch_done": [0-9]*' $f | tail -1)  $(tail -1 $f | cut -c1-160)"
  done
  exit 0
fi
STUDENT_RUN=${2:-${NAME}_student}
IFS=',' read -ra G <<< "${GPUS:-4,5,6,7}"
[ ${#G[@]} -eq 4 ] || { echo "GPUS must name 4 GPUs"; exit 2; }
T0=(tools/refiner_v6c/train_teachers.py --out $OUT --epochs ${EPOCHS_T:-10} --batch ${BATCH_T:-8}
    --limit ${LIMIT_T:-0} --platt-batches ${PLATT_BATCHES:-200} --lr ${LR_T:-2e-4})
T=("${T0[@]}" --workers ${WORKERS_T:-16})
TE=("${T0[@]}" --workers ${WORKERS_TE:-8})
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$REPO NCCL_SOCKET_IFNAME=lo
echo $$ > $OUT/pipeline.pid
say "pipeline $NAME start (pid $$, GPUs ${G[*]}, student $STUDENT_RUN, git $(git rev-parse --short HEAD))"

run_parallel() {   # run_parallel <stage> "<label>|<cmd>" ...   -> 0 when every job exits 0
  local stage=$1; shift
  local pids=() labels=() rc=0
  for job in "$@"; do
    local label=${job%%|*} cmd=${job#*|}
    bash -c "$cmd" > $OUT/logs/$label.log 2>&1 &
    pids+=($!); labels+=($label)
    say "  $stage: $label started (pid $!)"
  done
  for i in "${!pids[@]}"; do
    wait ${pids[$i]}; local r=$?
    say "  $stage: ${labels[$i]} exit $r"
    [ $r -eq 0 ] || rc=1
  done
  return $rc
}
done_mark() { echo "done $(kst)" > $OUT/done_$1; say "stage $1 done"; }
fail() { say "stage $1 FAILED (logs in $OUT/logs); re-run the same command to resume"; exit 1; }

# 1. T-D (DET) on G0, G1 and T-M (MAP) on G2, G3: one fold copy per GPU
if [ ! -f $OUT/done_fields ]; then
  say "stage fields: T-D copies @${G[0]},${G[1]}  T-M copies @${G[2]},${G[3]}"
  run_parallel fields \
    "td0|CUDA_VISIBLE_DEVICES=${G[0]} $PY ${T[*]} --phase td --fold 0" \
    "td1|CUDA_VISIBLE_DEVICES=${G[1]} $PY ${T[*]} --phase td --fold 1" \
    "tm0|CUDA_VISIBLE_DEVICES=${G[2]} $PY ${T[*]} --phase tm --fold 0" \
    "tm1|CUDA_VISIBLE_DEVICES=${G[3]} $PY ${T[*]} --phase tm --fold 1" || fail fields
  done_mark fields
fi
# 2. merge + occupancy Platt
if [ ! -f $OUT/done_platt ]; then
  CUDA_VISIBLE_DEVICES=${G[0]} $PY ${T[*]} --phase merge_fields > $OUT/logs/merge_fields.log 2>&1 || fail platt
  CUDA_VISIBLE_DEVICES=${G[0]} $PY ${T[*]} --phase platt_occ --resume $OUT/teachers_fields.pt > $OUT/logs/platt_occ.log 2>&1 || fail platt
  done_mark platt
fi
# 3. T-E: copy 0 DDP on G0,G1 and copy 1 DDP on G2,G3
if [ ! -f $OUT/done_te ]; then
  say "stage te: T-E copy 0 @${G[0]},${G[1]}  copy 1 @${G[2]},${G[3]} (DDP)"
  run_parallel te \
    "te0|CUDA_VISIBLE_DEVICES=${G[0]},${G[1]} $PY -m torch.distributed.run --standalone --nproc_per_node 2 ${TE[*]} --phase te_fold --fold 0 --resume $OUT/teachers_platt_occ.pt" \
    "te1|CUDA_VISIBLE_DEVICES=${G[2]},${G[3]} $PY -m torch.distributed.run --standalone --nproc_per_node 2 ${TE[*]} --phase te_fold --fold 1 --resume $OUT/teachers_platt_occ.pt" || fail te
  done_mark te
fi
# 4. T-E / T-M off-road Platt -> the complete teacher checkpoint
if [ ! -f $OUT/done_finish ]; then
  CUDA_VISIBLE_DEVICES=${G[0]} $PY ${TE[*]} --phase finish --resume $OUT/teachers_platt_occ.pt > $OUT/logs/finish.log 2>&1 || fail finish
  $PY -c "
from types import SimpleNamespace
from navsim.agents.para_ssr.refiner_v6c.teacher import load_teachers
load_teachers('$OUT/teachers_te.pt', SimpleNamespace(embed_dims=256, bev_h=100, bev_w=100))
print('complete teacher checkpoint: $OUT/teachers_te.pt')" >> $OUT/logs/finish.log 2>&1 || fail finish
  done_mark finish
fi
# 5. the student (stage D, cf_seg_v15 + refiner v6c from scratch) with the finished teachers
if [ "${STUDENT:-1}" = 1 ] && [ ! -f $OUT/done_student ]; then
  say "stage student: $STUDENT_RUN @${GPUS:-4,5,6,7} (V6C_TEACHER=$OUT/teachers_te.pt)"
  V6C_TEACHER=$OUT/teachers_te.pt WANDB_GROUP=${WANDB_GROUP:-refiner-v6c} \
    bash scripts/km_corridor/train.sh cf_seg_v15_v6c $STUDENT_RUN ${G[0]},${G[1]},${G[2]},${G[3]} \
    > $OUT/logs/student_$STUDENT_RUN.log 2>&1 || fail student
  done_mark student
fi
say "pipeline $NAME finished"
