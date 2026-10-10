#!/usr/bin/env bash
# CK Phase 2 smoke S1-S5 (report 45 §6; contract launch.smoke_e2e.sh).  Each stage writes <SR>/result_<S>.json;
# 'report' writes <SR>/smoke_report.json + kd_loss_table.{json,md} (the table the user approves the main run from).
#
#   TAG=<tag> bash tools/ck/e2e/smoke_e2e.sh S1 S2 S3 S4 S5 report      (or: all)
#   TAG=<tag> bash tools/ck/e2e/smoke_e2e.sh status
#   SR = /home/external-user/ssd/yongjae_refiner/ck/phase2/e2e_smoke/<TAG>  (TAG default: date)
#
#   S1  pytest tests/test_ck_*.py (Phase 1 + test_ck_e2e_*) + tests/test_para_ssr_*.py (v2 agent / anchor planner), CPU
#   S2  1 GPU (free one of SMOKE_GPUS, default 1,2,3), CK on, 200 micro-batches = epoch 0 replay 100 mb + epoch 1
#       on-policy (+ record) 100 mb, no labeler -> non-finite 0, peak mem <= 20 GB, ck/* logs, s/mb per phase
#   S3  1 GPU, 240 fixed tokens (scene_filter.tokens + log_names), compressed schedule record 1 / on-policy 2 / ramp 0.5,
#       4 epochs + labeler W=8 -> killed during epoch 3 (after the epoch-2 ckpt) -> --resume finishes epoch 3
#       -> record -> label -> use round trip: lab/ep001-002 DONE, every labelled row's trajectories == the recorded
#       cat(cand, kd_corr) (bit-exact), ep 2 G mostly from ep 1 (>= 90%), ep 3 G_src_phase1 < 1%, kd_ok_frac >= 99%,
#       EMA n / mb continuous across the resume
#   S4  DDP on GPUs 0-3 (or 1-3 when GPU 0 is busy), 3 epochs x 200 mb = replay / replay + record / on-policy + record,
#       labeler W=24 + watchdog (WATCHDOG_EVERY 15 s); the labeler is killed once during epoch 1 and must be restarted
#       by the watchdog and still finish lab/ep001-002 -> no hang, on-policy s/mb <= 0.55, labeler tok/s >= 1.5 x
#       training consumption, ck/gshare_ck logged.  Labeler capacity = a drain run (--once, W=24) over a copy of the S4
#       records after training (during training the labeler is supply-bound); status snapshots in s4/lab_snapshots.jsonl
#   S5  eval_e2e.py on the S3 checkpoint: navtrain_val --limit 200 (beta / variant grid) then navtest --limit 200 (fixed)
#   report: smoke_report.json, kd_loss_table.{json,md}, smoke_summary.md (per-phase s/mb, 30-epoch estimate)
#   Timeouts (s): S2_TIMEOUT 3600, S3_TIMEOUT 5400, S4_TIMEOUT 3600, LABELER_TIMEOUT 1800.
# Every job is nohup'ed through train_e2e.sh (pid files, setsid groups); this script only waits on them.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"   # repo root of this worktree
PY=/venv/ssr/bin/python
LU="$REPO/tools/ck/e2e/launch_util.py"
TE="$REPO/tools/ck/e2e/train_e2e.sh"
TAG=${TAG:-$(date -u +%m%d_%H%M)}
SMOKE_BASE=/home/external-user/ssd/yongjae_refiner/ck/phase2/e2e_smoke
SR=$SMOKE_BASE/$TAG
SMOKE_GPUS=${SMOKE_GPUS:-1,2,3}
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SR"
cd "$REPO" || exit 1

log() { echo "[$(date -u '+%F %T') UTC] [smoke $TAG] $*" | tee -a "$SR/smoke.log"; }
te() {  # te <run> <cmd> [args]: train_e2e.sh against the smoke root
  local run=$1; shift
  RUN=$run E2E_ROOT=$SR E2E_SMOKE=1 E2E_WANDB=${E2E_WANDB:-disable} bash "$TE" "$@"
}
check() {  # check <stage> [args] -> result_<stage>.json
  local st=$1; shift
  "$PY" "$LU" smoke-check --stage "$st" --out "$SR/result_$st.json" "$@" | tee -a "$SR/smoke.log"
}
one_gpu() { "$PY" "$LU" pick-gpus --n 1 --among "$SMOKE_GPUS"; }

S1() {
  log "S1 pytest"
  local out="$SR/S1_pytest.log" rc
  CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q tests/test_ck_*.py tests/test_para_ssr_*.py > "$out" 2>&1; rc=$?
  check S1 --rc "$rc" --summary "$(grep -E '(passed|failed|error)' "$out" | tail -n 1)"
}

S2() {
  local g; g=$(one_gpu) || { log "S2: no free GPU among $SMOKE_GPUS"; return 1; }
  log "S2 CK on, 200 mb (replay 100 + on-policy/record 100) on GPU $g"
  printf '%s\n' trainer.params.limit_train_batches=100 > "$SR/s2_extra.txt"
  printf '%s\n' record_from_epoch=1 onpolicy_from_epoch=1 record_until_epoch=1 kd_ramp_epochs=0.01 \
    kd_ema.start_mb=50 > "$SR/s2_ck.txt"
  MAX_EPOCHS=2 E2E_NO_LABELER=1 E2E_EXTRA=$SR/s2_extra.txt E2E_CK_SET=$SR/s2_ck.txt te s2 start --gpus "$g" || return 1
  E2E_WAIT_TIMEOUT=${S2_TIMEOUT:-3600} te s2 wait; log "S2 train exit $?"
  check S2 --run-dir "$SR/s2"
}

s3te() {  # S3 schedule: 4 trainer epochs, labeler W=8 polling every 5 s
  MAX_EPOCHS=4 LABELER_WORKERS=8 LABELER_POLL=5 E2E_EXTRA=$SR/s3_extra.txt E2E_CK_SET=$SR/s3_ck.txt te s3 "$@"
}

S3() {
  local g; g=$(one_gpu) || { log "S3: no free GPU among $SMOKE_GPUS"; return 1; }
  log "S3 compressed schedule, 240 tokens, GPU $g"
  "$PY" "$LU" smoke-tokens --n 240 --out "$SR/s3_tokens.json" --extra-out "$SR/s3_extra.txt" | tee -a "$SR/smoke.log"
  printf '%s\n' replay_until_epoch=2 record_from_epoch=1 onpolicy_from_epoch=2 kd_ramp_epochs=0.5 \
    label_refresh_every_mb=10 > "$SR/s3_ck.txt"
  s3te start --gpus "$g" || return 1
  # kill during epoch 3: wait for the epoch-2 checkpoint, then 20 s more
  local t0; t0=$(date +%s)
  while :; do
    ls "$SR"/s3/train/lightning_logs/version_*/checkpoints/epoch=2-step=*.ckpt > /dev/null 2>&1 && break
    if [ -f "$SR/s3/launch/train.exit" ]; then
      log "S3: training ended before the epoch-2 checkpoint"; break
    fi
    [ $(( $(date +%s) - t0 )) -gt "${S3_TIMEOUT:-5400}" ] && { log "S3 timeout before epoch 2"; break; }
    sleep 5
  done
  sleep 20
  log "S3 kill during epoch 3"
  te s3 stop --train-only
  local exit_a; exit_a=$(cat "$SR/s3/launch/train.exit" 2>/dev/null || echo "")
  E2E_WAIT_TIMEOUT=600 te s3 wait --labeler; log "S3 labeler after kill exit $?"
  log "S3 resume"
  s3te start --resume --gpus "$g" || return 1
  E2E_WAIT_TIMEOUT=${S3_TIMEOUT:-5400} te s3 wait; log "S3 train exit $?"
  E2E_WAIT_TIMEOUT=${LABELER_TIMEOUT:-1800} te s3 wait --labeler; log "S3 labeler exit $?"
  check S3 --run-dir "$SR/s3" --exit-a "$exit_a" --mb-per-epoch 60
}

S4() {
  local G
  if "$PY" "$LU" gpu-check --gpus 0,1,2,3 > /dev/null 2>&1; then G=0,1,2,3
  elif "$PY" "$LU" gpu-check --gpus 1,2,3 > /dev/null 2>&1; then G=1,2,3
  else log "S4: neither GPUs 0-3 nor 1-3 are free"; return 1; fi
  local NG; NG=$(awk -F, '{print NF}' <<< "$G")
  log "S4 DDP on GPUs $G: 3 epochs x 200 mb (replay / replay+record / on-policy+record), labeler kill -> watchdog"
  printf '%s\n' trainer.params.limit_train_batches=200 > "$SR/s4_extra.txt"
  printf '%s\n' record_from_epoch=1 onpolicy_from_epoch=2 record_until_epoch=2 kd_ramp_epochs=0.01 \
    kd_ema.start_mb=50 label_refresh_every_mb=50 > "$SR/s4_ck.txt"
  MAX_EPOCHS=3 LABELER_WORKERS=24 LABELER_POLL=10 LABELER_MAX_EPOCH=2 WATCHDOG_EVERY=15 E2E_EXTRA=$SR/s4_extra.txt \
    E2E_CK_SET=$SR/s4_ck.txt te s4 start --gpus "$G" || return 1
  local t0 st=$SR/s4/ck_e2e/lab/status.json ep=$SR/s4/ck_e2e/epochs.jsonl wt=$SR/s4/watchdog_test.json t1=0 killed=0
  t0=$(date +%s)
  while [ ! -f "$SR/s4/launch/train.exit" ]; do        # labeler snapshots while training (backlog / tok/s)
    [ -f "$st" ] && { tr -d '\n' < "$st"; echo; } >> "$SR/s4/lab_snapshots.jsonl"
    # watchdog test: in epoch 1 (recording), once the labeler is scoring (tokens_ok > 0; at the latest 45 s into the
    # epoch), TERM the labeler's whole process group once (as a crash / OOM / stop would)
    if [ "$killed" = 0 ] && [ -f "$ep" ] && grep -q '"event": "epoch_start", "epoch": 1,' "$ep"; then
      [ "$t1" = 0 ] && t1=$(date +%s)
      if grep -Eq '"tokens_ok": [1-9]' "$st" 2>/dev/null || [ $(( $(date +%s) - t1 )) -ge 45 ]; then
        local lp lg; lp=$(cat "$SR/s4/labeler/labeler.pid" 2>/dev/null); lg=$(cat "$SR/s4/labeler/labeler.pgid" 2>/dev/null)
        if [ -n "$lg" ] && kill -0 "$lp" 2>/dev/null; then
          kill -TERM -- "-$lg" 2>/dev/null; killed=1
          printf '{"killed_pid": %s, "killed_pgid": %s, "kill_time": %s, "tokens_ok_before": %s}\n' "$lp" "$lg" \
            "$(date +%s)" "$(grep -Eo '"tokens_ok": [0-9]+' "$st" 2>/dev/null | grep -Eo '[0-9]+' || echo null)" > "$wt"
          log "S4 watchdog test: labeler pid $lp (pgid $lg) killed during epoch 1"
        fi
      fi
    fi
    [ $(( $(date +%s) - t0 )) -gt "${S4_TIMEOUT:-3600}" ] && break
    sleep 5
  done
  E2E_WAIT_TIMEOUT=60 te s4 wait; local rc=$?
  log "S4 train exit $rc"
  if [ "$rc" = 124 ]; then log "S4 hang/timeout: stopping"; te s4 stop; fi
  E2E_WAIT_TIMEOUT=${LABELER_TIMEOUT:-1800} te s4 wait --labeler; log "S4 labeler exit $?"
  # labeler capacity on a full backlog: re-score the S4 records in a scratch io_dir (--once, W=24, GPUs idle)
  local D=$SR/s4_drain/ck_e2e
  rm -rf "$SR/s4_drain"; mkdir -p "$D"
  cp -r "$SR/s4/ck_e2e/rec" "$D/" && touch "$D/TRAIN_DONE"
  "$PY" tools/ck/e2e/labeler.py --io-dir "$D" --world-size "$NG" --workers 24 --poll 2 --status-every 5 --max-epoch 2 \
    --once > "$SR/s4_drain/labeler.log" 2>&1; log "S4 drain labeler exit $?"
  check S4 --run-dir "$SR/s4" --ngpu "$NG" --drain-dir "$D"
}

S5() {
  local g; g=$(one_gpu) || { log "S5: no free GPU among $SMOKE_GPUS"; return 1; }
  log "S5 eval_e2e on the S3 checkpoint, GPU $g"
  local EV=("$PY" tools/ck/e2e/eval_e2e.py --run-dir "$SR/s3" --limit 200 --gpus "$g" --workers-label 16
            --bootstrap 200)
  "${EV[@]}" --split navtrain_val --stage all >> "$SR/S5_eval.log" 2>&1 || log "S5 navtrain_val failed (rc $?)"
  "${EV[@]}" --split navtest --stage dump,label,infer_check,eval,summary >> "$SR/S5_eval.log" 2>&1 \
    || log "S5 navtest failed (rc $?)"
  check S5 --run-dir "$SR/s3"
}

report() { log "report"; "$PY" "$LU" smoke-report --root "$SR" | tee -a "$SR/smoke.log"; }

status() {
  for r in s2 s3 s4; do [ -d "$SR/$r" ] && RUN=$r E2E_ROOT=$SR bash "$TE" status; done
  ls "$SR"/result_*.json 2>/dev/null
}

[ $# -gt 0 ] || { sed -n '2,20p' "$0"; exit 2; }
ARGS=("$@"); [ "$1" = all ] && ARGS=(S1 S2 S3 S4 S5 report)
log "stages ${ARGS[*]} -> $SR"
FAIL=0
for s in "${ARGS[@]}"; do
  case "$s" in
    S1|S2|S3|S4|S5|report|status) "$s" || { log "$s returned $?"; FAIL=1; };;
    *) log "unknown stage $s"; exit 2;;
  esac
done
exit $FAIL
