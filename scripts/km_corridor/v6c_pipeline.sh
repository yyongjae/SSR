#!/usr/bin/env bash
# refiner v6c (report 124 / 128): stage T teachers in parallel on 4 GPUs, then the student.
#   usage: v6c_pipeline.sh <name> [student run name]            start (run it detached, see below)
#          v6c_pipeline.sh <name> status                        progress of every stage
#   detached: setsid nohup bash scripts/km_corridor/v6c_pipeline.sh <name> <student> > <log> 2>&1 &
#
#   stages (a stage with a done marker is skipped, so a re-launch resumes; the field / T-E copies also resume from
#   their last finished epoch):
#     0 check    label-table coverage of navtrain_train / val (train_teachers --phase check, CPU) -> coverage.json;
#                a coverage below COVERAGE_MIN (1.0) stops the pipeline
#     1 fields   T-D copy 0 @G0, T-D copy 1 @G1 (DET, BEVFusion) | T-M copy 0 @G2, T-M copy 1 @G3 (MAP, ReSMap)
#     2 platt    merge the four copies -> teachers_fields.pt; occupancy Platt @G0 -> teachers_platt_occ.pt
#     3 te       T-E copy 0 DDP @G0,G1 | T-E copy 1 DDP @G2,G3 (global batch = BATCH_T, the single-GPU recipe)
#     4 finish   T-E overlap / approach Platt + T-M off-road Platt @G0 -> teachers_te.pt (fields, platt_occ, te)
#     5 student  train.sh cf_seg_v15_v6c <student> G0,G1,G2,G3 with V6C_TEACHER=<out>/teachers_te.pt (stage D)
#   GPUS (default 0,1,2,3 = G0..G3).  Copy f of every module learns from fold-f tokens only (cf_fold = crc32(log) % 2).
#        GPU preflight (r129): every G must hold <= GPU_MAX_USED_MIB (1000) MiB (nvidia-smi) at launch, else the launch
#        aborts (exit 2); GPU_CHECK=0 skips it ('status' never checks).
#   env: EPOCHS_T (10), BATCH_T (8), LIMIT_T (0 = every token), PLATT_BATCHES (200), LR_T (2e-4),
#        WORKERS_T (16: loader workers per field copy; the corridor targets are CPU-bound, 0.2-0.4 s per token),
#        WORKERS_TE (8 per T-E rank), SEED_T (0: train_teachers --seed of every stage-T phase),
#        COVERAGE_MIN (1.0: stage check), EXACT2_DIR (default $KM_ROOT/exact2: the v2 tables, exported to stage T
#        and to the student alike)
#        STUDENT=0 stops after stage 4.  The student stage passes the environment on to train.sh (EPOCHS,
#        LIMIT_TRAIN, LIMIT_VAL, V6C_WARMUP, V6C_FIX, V6C_KD, WANDB, ...); WANDB_GROUP defaults to refiner-v6c.
#   stale-reuse guard (r129): <out>/code_rev = 'git rev[-dirty.<hash>] read_rule=<teacher.READ_RULE>' of the first
#        launch (12-char rev; <hash> = md5 of the diff to HEAD + the untracked *.py / *.sh of navsim, tools, scripts).
#        A launch at a different code_rev, into a dir with stage outputs but no code_rev (pre-r129: done markers as in
#        v6cT_main, or only teachers / parts / logs as in v6cT_platt2 / platt3), or on a name whose pipeline.pid still
#        runs aborts (exit 2) and leaves the dir untouched: use a new name.  ALLOW_REUSE=1 turns the guard off (a
#        warning goes to status.log).
#   tables_provenance.txt (r129): per corridor64 / exact / exact2 (<split>/current when present) x navtrain_train / val:
#        realpath, meta.txt md5 and an md5 of the *.npy names / sizes / mtimes (an in-place rebuild), and the code_rev.
#        Written while no stage is done (a relaunch after a failed stage check lists the rebuilt tables); once a stage
#        is done every launch lists the tables again and aborts (exit 2) on a difference, and so does the student
#        stage (exit 1); the student run dir gets a copy + 'teacher <out>/teachers_te.pt'.
#   student run dir: must be new (one with a last.ckpt aborts) unless STUDENT_RESUME=1, which resumes from its newest
#        train/lightning_logs/*/checkpoints/last.ckpt (++resume_checkpoint; needs the r129 distill-state checkpoint
#        callback, without it the KD warm-up replays) when its tables_provenance.txt names this teacher and tables.
#   outputs: $KM_RUN_ROOT/v6c_teacher/<name>/ (code_rev, coverage.json, tables_provenance.txt, parts, teachers_*.pt,
#            logs/, done_*, status.log); the student run in $KM_RUN_ROOT/corr30/<student>
set -uo pipefail
export KM_ROOT=${KM_ROOT:-/workspace/navsim_workspace/corridor_cache}
if [ -f "$(dirname "$0")/env.sh" ]; then source "$(dirname "$0")/env.sh"; else source "$(dirname "$0")/env.example.sh"; fi
export EXACT2_DIR=${EXACT2_DIR:-$KM_ROOT/exact2}     # the train.sh default: stage T and the student read one exact2
cd $REPO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$REPO NCCL_SOCKET_IFNAME=lo
NAME=${1:?usage: v6c_pipeline.sh <name> [student run name | status]}
RUNS=${KM_RUN_ROOT:-$REPO/work_dirs/km_runs}
OUT=$RUNS/v6c_teacher/$NAME
kst() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST'; }
say() { echo "[$(kst)] $*" | tee -a $OUT/status.log; }
# before the launch is accepted only read-only tests touch OUT and messages go to stdout: a refused launch (a pre-r129
# dir, another code_rev, a running pipeline) leaves the dir as it was
note() { echo "[$(kst)] $*"; }

if [ "${2:-}" = status ]; then
  cat $OUT/status.log 2>/dev/null | tail -20
  [ -f $OUT/code_rev ] && echo "code_rev $(cat $OUT/code_rev)"
  for st in check fields platt te finish student; do printf '%-8s %s\n' $st "$( [ -f $OUT/done_$st ] && cat $OUT/done_$st || echo pending)"; done
  for f in $OUT/log_td_f0.jsonl $OUT/log_td_f1.jsonl $OUT/log_tm_f0.jsonl $OUT/log_tm_f1.jsonl \
           $OUT/log_te_fold_f0.jsonl $OUT/log_te_fold_f1.jsonl; do
    [ -f $f ] && echo "$(basename $f): $(grep -o '"epoch_done": [0-9]*' $f | tail -1)  $(tail -1 $f | cut -c1-160)"
  done
  exit 0
fi
STUDENT_RUN=${2:-${NAME}_student}
IFS=',' read -ra G <<< "${GPUS:-0,1,2,3}"
[ ${#G[@]} -eq 4 ] || { echo "GPUS must name 4 GPUs"; exit 2; }
# GPU preflight: every G must be free (GPUs 4-7 of this host may hold another person's job)
if [ "${GPU_CHECK:-1}" != 0 ]; then
  for g in "${G[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
    [[ $used =~ ^[0-9]+$ ]] || { note "abort: cannot read the memory use of GPU $g (nvidia-smi); GPU_CHECK=0 skips the check"; exit 2; }
    [ $used -le ${GPU_MAX_USED_MIB:-1000} ] || {
      note "abort: GPU $g already uses $used MiB (> GPU_MAX_USED_MIB ${GPU_MAX_USED_MIB:-1000}): pick free GPUS (GPU_CHECK=0 skips the check)"; exit 2; }
  done
fi
# stale-reuse guard: OUT keeps the code_rev of its first launch, and done stages are reused only at that code_rev.
# 'git diff' (not diff-index) compares the content of stat-dirty files, so a touched but unchanged file is not -dirty;
# a fixed-length rev (--short grows with the object count) and a hash of the dirty content, so two different
# uncommitted trees, or a new still-untracked module, are two code_revs
RRE=$(mktemp) || exit 2
RR=$(CUDA_VISIBLE_DEVICES= $PY -c 'from navsim.agents.para_ssr.refiner_v6c.teacher import READ_RULE; print(READ_RULE)' \
     2> $RRE | tail -1) || RR=
[ -n "$RR" ] || { note "abort: READ_RULE missing (navsim.agents.para_ssr.refiner_v6c.teacher): $(tail -3 $RRE)"; rm -f $RRE; exit 2; }
rm -f $RRE
dirty() {   # '' for a clean tree, else '-dirty.<md5 [:8] of the diff to HEAD + the untracked code>'
  local u; u=$(git ls-files --others --exclude-standard -- 'navsim/*.py' 'tools/*.py' 'scripts/*.py' 'navsim/*.sh' \
                 'tools/*.sh' 'scripts/*.sh' 2>/dev/null)
  git diff --quiet HEAD -- 2>/dev/null && [ -z "$u" ] && return
  echo "-dirty.$( { git diff --no-color --no-ext-diff --binary HEAD -- 2>/dev/null
                   for f in $u; do echo "untracked $f"; cat "$f"; done; } | md5sum | cut -c1-8)"
}
REV="$(git rev-parse --short=12 HEAD 2>/dev/null || echo nogit)$(dirty) read_rule=$RR"
# pre-r129 = stage outputs without a code_rev: pipeline done markers (v6cT_main), or only train_teachers' teachers /
# parts / checkpoints / logs (v6cT_platt2, platt3: a stage T there would overwrite their teachers_*.pt)
has_outputs() { { compgen -G "$OUT/done_*" || compgen -G "$OUT/teachers*.pt" || compgen -G "$OUT/part_*" ||
                  compgen -G "$OUT/*.ckpt" || compgen -G "$OUT/log_*.jsonl" || compgen -G "$OUT/coverage.json"; } > /dev/null; }
LEGACY=0; [ ! -f $OUT/code_rev ] && has_outputs && LEGACY=1
# a pipeline still running on this OUT (one in a CPU stage passes the GPU preflight): pipeline.pid names a live bash
# whose argv is v6c_pipeline.sh <name>
OLD=$(cat $OUT/pipeline.pid 2>/dev/null)
running() {
  local a=() i
  [[ $OLD =~ ^[0-9]+$ ]] && [ "$OLD" != $$ ] && kill -0 $OLD 2>/dev/null || return 1
  mapfile -d '' a 2>/dev/null < /proc/$OLD/cmdline || return 1
  for i in "${!a[@]}"; do [[ ${a[$i]} == *v6c_pipeline.sh ]] && [ "${a[$((i + 1))]:-}" = "$NAME" ] && return 0; done
  return 1
}
if [ "${ALLOW_REUSE:-0}" = 1 ]; then
  :   # warned below, once the launch may write status.log
elif running; then
  note "abort: pipeline pid $OLD still runs on $OUT: wait for it or stop it (ALLOW_REUSE=1 skips the check)"; exit 2
elif [ -f $OUT/code_rev ] && [ "$(cat $OUT/code_rev)" != "$REV" ]; then
  note "abort: OUT was produced by $(cat $OUT/code_rev), code is $REV: use a new name or ALLOW_REUSE=1"; exit 2
elif [ $LEGACY = 1 ]; then
  note "abort: OUT holds outputs of pre-r129 code (done markers / teachers / parts / logs, no code_rev), code is $REV:" \
       "use a new name or ALLOW_REUSE=1"; exit 2
fi
mkdir -p $OUT/logs
if [ "${ALLOW_REUSE:-0}" = 1 ]; then
  say "WARNING: ALLOW_REUSE=1, code_rev guard off (OUT: $(cat $OUT/code_rev 2>/dev/null || echo 'no code_rev'), code: $REV)"
  running && say "WARNING: ALLOW_REUSE=1, pipeline pid $OLD still runs on $OUT"
fi
# a pre-r129 dir (ALLOW_REUSE) is not stamped with a code_rev it was not produced by
[ -f $OUT/code_rev ] || [ $LEGACY = 1 ] || echo "$REV" > $OUT/code_rev
T0=(tools/refiner_v6c/train_teachers.py --out $OUT --epochs ${EPOCHS_T:-10} --batch ${BATCH_T:-8}
    --limit ${LIMIT_T:-0} --platt-batches ${PLATT_BATCHES:-200} --lr ${LR_T:-2e-4} --seed ${SEED_T:-0})
T=("${T0[@]}" --workers ${WORKERS_T:-16})
TE=("${T0[@]}" --workers ${WORKERS_TE:-8})
echo $$ > $OUT/pipeline.pid
say "pipeline $NAME start (pid $$, GPUs ${G[*]}, student $STUDENT_RUN, code_rev $REV, EXACT2_DIR $EXACT2_DIR -> $(realpath -m $EXACT2_DIR))"

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
prov() {   # one line per label table / split: <table> <split> <realpath> <md5 of meta.txt | no-meta>
           #   <md5 of the name / size / mtime of its *.npy | no-npy>
  local tb s d
  for tb in corridor64=$KM_ROOT/corridor64 exact=$KM_ROOT/exact exact2=$EXACT2_DIR; do
    for s in navtrain_train navtrain_val; do
      d=${tb#*=}/$s
      # a v2 split still being filled (merge_exact_parts.py) is read through <split>/current (ExactFullTargetBuilder)
      [ ${tb%%=*} = exact2 ] && [ -e $d/current ] && d=$d/current
      echo "${tb%%=*} $s $(realpath -m $d) $( [ -f $d/meta.txt ] && md5sum < $d/meta.txt | cut -d' ' -f1 || echo no-meta)" \
           "$( (cd $d 2> /dev/null && compgen -G '*.npy' > /dev/null &&
                { stat -Lc '%n %s %Y' -- *.npy 2>&1 || true; } | md5sum | cut -d' ' -f1) || echo no-npy)"
    done
  done
}
# the tables of this launch (KM_ROOT / EXACT2_DIR, the files now) vs the stored list, its git line left out
prov_same() { diff <(grep -v '^git ' $OUT/tables_provenance.txt) <(prov) > $OUT/logs/provenance_diff.log 2>&1; }

# the label tables stage T reads, listed while no stage has read them (so a relaunch after a failed stage check lists
# the rebuilt tables); once a stage is done every launch lists them again before any stage, since a resumed stage T
# would mix two table sets (the meta.txt / *.npy stat columns catch an in-place rebuild); the student stage checks
# once more (tables rebuilt while stage T ran).  A pre-r129 dir (ALLOW_REUSE) without a record gets the current list
if ! compgen -G "$OUT/done_*" > /dev/null || [ ! -f $OUT/tables_provenance.txt ]; then
  { prov; echo "git $REV"; } > $OUT/tables_provenance.txt
elif ! prov_same; then
  say "abort: the label tables differ from $OUT/tables_provenance.txt (read by the done stages), see" \
      "$OUT/logs/provenance_diff.log: a resume would mix two table sets; restore the tables or use a new name"; exit 2
fi
# 0. label-table coverage (CPU only): every navtrain_train / val token must have its rows in every table
if [ ! -f $OUT/done_check ]; then
  CUDA_VISIBLE_DEVICES= $PY ${T[*]} --phase check --min-coverage ${COVERAGE_MIN:-1.0} > $OUT/logs/check.log 2>&1 || fail check
  done_mark check
fi
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
  # the student must read the tables stage T read (here: tables rebuilt while stage T ran in this launch; a change
  # between launches stops at the check above)
  if ! prov_same; then
    say "student: the label tables differ from $OUT/tables_provenance.txt (stage T), see $OUT/logs/provenance_diff.log"
    fail student
  fi
  SDIR=$RUNS/corr30/$STUDENT_RUN
  SD=$SDIR/train/lightning_logs
  # what the student reads: the stage-T record + this teacher, kept in the student run dir (a resume must match it)
  SPROV="$(cat $OUT/tables_provenance.txt)
teacher $OUT/teachers_te.pt"
  RESUME=()
  if [ "${STUDENT_RESUME:-0}" = 1 ]; then
    CK=$(ls -t $SD/*/checkpoints/last.ckpt 2>/dev/null | head -1)
    [ -n "$CK" ] || { say "student: no last.ckpt to resume in $SD"; fail student; }
    # the resumed run started with this teacher and these tables (not one of another stage T)
    diff $SDIR/tables_provenance.txt <(echo "$SPROV") > $OUT/logs/student_provenance_diff.log 2>&1 || {
      say "student: $SDIR/tables_provenance.txt is missing or names another teacher / other tables than this stage T," \
          "see $OUT/logs/student_provenance_diff.log: resume it from its own teacher dir or use a new student name"
      fail student; }
    RESUME=("++resume_checkpoint='$CK'")    # quoted for hydra
    say "student: $STUDENT_RUN resumes from $CK"
  elif ls $SD/*/checkpoints/last.ckpt > /dev/null 2>&1; then
    # a fresh launch would start a second Lightning version from epoch 0 in the same run dir
    say "student: student run dir already has checkpoints ($SD): STUDENT_RESUME=1 to resume or use a new student name"
    fail student
  fi
  mkdir -p $SDIR && echo "$SPROV" > $SDIR/tables_provenance.txt || fail student
  say "stage student: $STUDENT_RUN @${G[*]} (V6C_TEACHER=$OUT/teachers_te.pt)"
  # the log is appended: a resumed student keeps the earlier attempts' output
  V6C_TEACHER=$OUT/teachers_te.pt WANDB_GROUP=${WANDB_GROUP:-refiner-v6c} \
    bash scripts/km_corridor/train.sh cf_seg_v15_v6c $STUDENT_RUN ${G[0]},${G[1]},${G[2]},${G[3]} "${RESUME[@]}" \
    >> $OUT/logs/student_$STUDENT_RUN.log 2>&1 || fail student
  done_mark student
fi
say "pipeline $NAME finished"
