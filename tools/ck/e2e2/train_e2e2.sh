#!/usr/bin/env bash
# CK2 e2e: v2 + CK2 student from scratch, 30 epochs, r34 recipe on 2 GPUs (SPEC ck2e2e s7-3; logic of
# tools/ck/e2e/train_e2e.sh with the CK2 constants).  Two runs (user): GPUs 0,1 = CK2 main, GPUs 2,3 = CK2 + BEV KD.
#
#   RUN=ck2e2e_main  CK2_E2E_APPROVED=1 bash tools/ck/e2e2/train_e2e2.sh start --gpus 0,1
#   RUN=ck2e2e_bevkd CK2_E2E_APPROVED=1 ARM=bevkd bash tools/ck/e2e2/train_e2e2.sh start --gpus 2,3
#   RUN=... bash tools/ck/e2e2/train_e2e2.sh start --resume --gpus <same>      # newest last.ckpt
#   RUN=... bash tools/ck/e2e2/train_e2e2.sh status [--json]
#   RUN=... bash tools/ck/e2e2/train_e2e2.sh stop [--train-only]
#   RUN=... bash tools/ck/e2e2/train_e2e2.sh wait [--labeler]
#   RUN=... bash tools/ck/e2e2/train_e2e2.sh labeler --gpus <same>             # (re)start labeler + watchdog
#   RUN=... E2E_DRY_RUN=1 CK2_E2E_APPROVED=1 bash tools/ck/e2e2/train_e2e2.sh start --gpus 0,1   # print overrides only
#   (internal) train_e2e2.sh _watchdog --gpus ...
#
# Layout <E2E_ROOT>/<RUN>/ (E2E_ROOT default /home/external-user/ssd/yongjae_refiner/ck/ck2/e2e):
#   train/     navsim output_dir (lightning_logs/version_*/checkpoints/{epoch=E-step=S.ckpt,last.ckpt}, code/hydra)
#   ck_e2e2/   ck_e2e2.io_dir (rec/, lab/, steps_rank0.jsonl, epochs.jsonl, TRAIN_DONE)
#   labeler/   labeler.log / .pid / .pgid / .exit / .cmd, watchdog.pid / .log / .restarts / .state
#   launch/    train.log / .pid / .pgid / .exit, overrides.txt, ck_e2e2_effective.json, teachers.json,
#              nvidia_smi_start.txt, launch.log
# start guards: --gpus = exactly 2 GPUs of 0-3 (user recipe devices 2), nvidia-smi <= 1 GiB used on each,
#   CK2_E2E_APPROVED=1 (main runs) or E2E_SMOKE=1 (only below .../ck/ck2/e2e_smoke/), both CK2 teachers finished
#   (done.json, trainer train_ck2, arm T / M, lon head zero, config.json ep_target == ck_e2e2.ep_target -- default
#   teachers ck2T10dep / ck2M10dep with --ep-target decoupled, user 2026-10-08; launch/teachers.json records the ckpt
#   sha16 and ep_target) -- smoke runs skip only the done.json requirement, no '=' in paths, no live train.pid, a
#   fresh start refuses an existing run dir.  KD calibration (ck_e2e2.kd_calib.enabled, user 2026-10-08 ~22:35 KST,
#   both modes): every kd_calib.det / .map file must exist and match its teacher (run, which, ckpt sha16, ep_target,
#   KD keys fitted); a missing file is REFUSED with the command to run after the teacher has finished:
#     CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=$REPO /venv/ssr/bin/python tools/ck/e2e2/fit_kd_calib.py --run <teacher run>
#       --which last      (-> .../ck/ck2/kd_calib/<run>__last.json; recorded with its sha16 in launch/teachers.json)
# -> training (setsid nohup; DDP ranks share its process group) -> labeler2 (32 workers, --world-size 2, --max-epoch
#   MAX_EPOCHS-2, --exit-when-done --train-pid-file) -> watchdog (restart a dead / hung labeler while training lives).
# ARM=main (default) | bevkd (adds tools/ck/e2e2/bevkd_arm.set); E2E_CK_SET=<file of ck_e2e2 key=value lines> is applied
#   after the arm file.  Never pkill / pgrep -f: processes are found through the pid files only.
# Env knobs: E2E_ROOT, TRAIN_WORKERS (6), MAX_EPOCHS (30), LABELER_WORKERS (32), LABELER_POLL (30),
#   LABELER_MAX_EPOCH (MAX_EPOCHS-2), E2E_NO_LABELER=1, E2E_WANDB=online|offline|disable, E2E_EXTRA=<Hydra override file>,
#   E2E_SMOKE=1, E2E_WAIT_TIMEOUT, E2E_DRY_RUN=1, WATCHDOG_EVERY (120), LABELER_MAX_RESTARTS (20), LABELER_STALL_S (1200),
#   E2E_NO_WATCHDOG=1, E2E_NO_DEEP_TEACHER_CHECK=1 (tests: skip loading the teacher ckpts).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"   # repo root of this worktree
PY=/venv/ssr/bin/python
LU2="$REPO/tools/ck/e2e2/launch_util2.py"
LU1="$REPO/tools/ck/e2e/launch_util.py"
SMOKE_PREFIX=/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke/
CMD=${1:-}; shift || true
RESUME=0; GPUS=""; TRAIN_ONLY=0; JSON=0; WAIT_LAB=0
while [ $# -gt 0 ]; do
  case "$1" in
    --resume) RESUME=1;;
    --gpus) GPUS=${2:?--gpus needs a list}; shift;;
    --train-only) TRAIN_ONLY=1;;
    --json) JSON=1;;
    --labeler) WAIT_LAB=1;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
  shift
done
RUN=${RUN:?RUN=<run name> is required}
ARM=${ARM:-main}
E2E_ROOT=${E2E_ROOT:-/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e}
R="$E2E_ROOT/$RUN"
L="$R/launch"; LB="$R/labeler"; IO="$R/ck_e2e2"
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# DataLoader workers hand tensors to the trainer through AF_UNIX sockets under $TMPDIR (pymp-*/listener-*, +32 chars;
# limit 107): with a long TMPDIR the bind fails inside a worker thread and the first batch hangs without an exit
# [실측 2026-10-08: a TMPDIR of 87 chars hung a fork DataLoader test].  Training / labeler then use /tmp.
if [ -n "${TMPDIR:-}" ] && [ "${#TMPDIR}" -gt 75 ]; then
  echo "[train_e2e2] TMPDIR '$TMPDIR' has ${#TMPDIR} chars (> 75): using TMPDIR=/tmp (AF_UNIX socket path limit)" >&2
  export TMPDIR=/tmp
fi

log() { echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*"; [ -d "$L" ] && echo "[$(TZ=Asia/Seoul date '+%F %T') KST] $*" >> "$L/launch.log"; return 0; }
die() { echo "[$(TZ=Asia/Seoul date '+%F %T') KST] REFUSED: $*" >&2; exit 3; }
WRAP='D=$1; N=$2; shift 2; echo $$ > "$D/$N.pgid"; "$@" >> "$D/$N.log" 2>&1 & c=$!; echo $c > "$D/$N.pid"
trap "kill -TERM $c 2>/dev/null" TERM
while :; do wait $c; rc=$?; kill -0 $c 2>/dev/null || break; done; echo $rc > "$D/$N.exit"'
alive() {  # alive <pid_file> <needle>: prints pid if that process lives and its cmdline contains needle
  [ -f "$1" ] || return 1
  local p; p=$(cat "$1" 2>/dev/null); [ -n "$p" ] || return 1
  [ -r "/proc/$p/cmdline" ] || return 1
  tr '\0' ' ' < "/proc/$p/cmdline" | grep -q -- "$2" || return 1
  echo "$p"
}
grp_kill() {  # grp_kill <pgid_file> <pid_file> <needle> <name>: TERM the group, KILL after 60 s
  local pg pid; pid=$(alive "$2" "$3") || { log "$4 not running"; return 0; }
  pg=$(cat "$1" 2>/dev/null || true)
  log "stopping $4 (pid $pid, pgid ${pg:-?})"
  if [ -n "$pg" ]; then kill -TERM -- "-$pg" 2>/dev/null || kill -TERM "$pid"; else kill -TERM "$pid"; fi
  for _ in $(seq 60); do
    if ! alive "$2" "$3" >/dev/null; then
      local ex=${2%.pid}.exit; for _ in $(seq 20); do [ -f "$ex" ] && break; sleep 0.5; done
      log "$4 stopped (exit $(cat "$ex" 2>/dev/null || echo ?))"; return 0
    fi
    sleep 1
  done
  log "$4 still alive after 60 s: KILL"
  if [ -n "$pg" ]; then kill -KILL -- "-$pg" 2>/dev/null || kill -KILL "$pid"; else kill -KILL "$pid"; fi
}
need_gpus() { [ -n "$GPUS" ] || die "--gpus <2 GPUs of 0-3> is required (e.g. --gpus 0,1)"; }

case "$R" in *=*) die "run dir $R contains '='";; esac
case "$ARM" in main|bevkd) ;; *) die "ARM must be main | bevkd (got $ARM)";; esac

start_labeler() {  # [retry]: add --retry-errors (watchdog restarts)
  if [ "${E2E_NO_LABELER:-0}" = 1 ]; then log "labeler disabled (E2E_NO_LABELER=1)"; return 0; fi
  if p=$(alive "$LB/labeler.pid" labeler2); then log "labeler alive (pid $p): kept"; return 0; fi
  need_gpus
  local ng me; ng=$(awk -F, '{print NF}' <<< "$GPUS"); me=${LABELER_MAX_EPOCH:-$(( ${MAX_EPOCHS:-30} - 2 ))}
  [ "$me" -lt 0 ] && me=0
  mkdir -p "$LB"; rm -f "$LB/labeler.exit" "$LB/labeler.pid"
  local LCMD=("$PY" "$REPO/tools/ck/e2e2/labeler2.py" --io-dir "$IO" --world-size "$ng" --workers "${LABELER_WORKERS:-32}"
              --poll "${LABELER_POLL:-30}" --max-epoch "$me" --exit-when-done --train-pid-file "$L/train.pid")
  [ "${1:-}" = retry ] && LCMD+=(--retry-errors)
  echo "${LCMD[*]}" > "$LB/labeler.cmd"
  setsid nohup bash -c "$WRAP" _ "$LB" labeler env PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "${LCMD[@]}" \
    < /dev/null > /dev/null 2>&1 &
  for _ in $(seq 30); do [ -s "$LB/labeler.pid" ] && break; sleep 0.2; done
  log "labeler2 started pid $(cat "$LB/labeler.pid" 2>/dev/null) W=${LABELER_WORKERS:-32} world $ng max-epoch $me -> $LB/labeler.log"
}

start_watchdog() {
  if [ "${E2E_NO_LABELER:-0}" = 1 ] || [ "${E2E_NO_WATCHDOG:-0}" = 1 ]; then log "labeler watchdog off"; return 0; fi
  if p=$(alive "$LB/watchdog.pid" _watchdog); then log "watchdog alive (pid $p): kept"; return 0; fi
  mkdir -p "$LB"; rm -f "$LB/watchdog.pid"
  RUN=$RUN ARM=$ARM E2E_ROOT=$E2E_ROOT setsid nohup bash "$REPO/tools/ck/e2e2/train_e2e2.sh" _watchdog --gpus "$GPUS" \
    < /dev/null >> "$LB/watchdog.log" 2>&1 &
  for _ in $(seq 30); do [ -s "$LB/watchdog.pid" ] && break; sleep 0.2; done
  log "labeler watchdog started pid $(cat "$LB/watchdog.pid" 2>/dev/null) every ${WATCHDOG_EVERY:-120}s -> $LB/watchdog.log"
}

do_watchdog() {
  echo $$ > "$LB/watchdog.pid"
  local n=0 every=${WATCHDOG_EVERY:-120} maxr=${LABELER_MAX_RESTARTS:-20} ex
  log "watchdog pid $$ every ${every}s max restarts $maxr"
  sleep "$every"
  while alive "$L/train.pid" run_training > /dev/null; do
    if alive "$LB/labeler.pid" labeler2 > /dev/null; then
      local why
      why=$("$PY" "$LU2" labeler-health --run-dir "$R" --state "$LB/watchdog.state" --stall-s "${LABELER_STALL_S:-1200}")
      if [ $? = 5 ]; then
        log "watchdog: labeler hung ($why): stopping it"
        grp_kill "$LB/labeler.pgid" "$LB/labeler.pid" labeler2 labeler
      fi
    fi
    if ! alive "$LB/labeler.pid" labeler2 > /dev/null; then
      ex=$(cat "$LB/labeler.exit" 2>/dev/null || echo "?")
      if [ "$ex" = 0 ] && [ -f "$IO/TRAIN_DONE" ]; then log "watchdog: labeler finished (exit 0, TRAIN_DONE)"; break; fi
      if [ "$n" -ge "$maxr" ]; then
        log "watchdog: labeler not running (exit $ex); restart limit $maxr reached: giving up"; break
      fi
      n=$((n + 1))
      log "watchdog: labeler not running (exit $ex) while training is alive: restart $n/$maxr"
      echo "$(TZ=Asia/Seoul date '+%F %T') KST restart $n exit $ex" >> "$LB/watchdog.restarts"
      start_labeler retry
    fi
    sleep "$every"
  done
  log "watchdog: exit (training not running or labeler done; restarts $n)"
}

stop_watchdog() {
  local p; p=$(alive "$LB/watchdog.pid" _watchdog) || return 0
  kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"
  for _ in $(seq 20); do alive "$LB/watchdog.pid" _watchdog > /dev/null || break; sleep 0.5; done
  log "watchdog stopped (pid $p)"
}

do_start() {
  need_gpus
  "$PY" "$LU2" gpu-check --gpus "$GPUS" --list-only > /dev/null || die "GPU list $GPUS (exactly 2 of 0-3)"
  local TMODE=strict
  if [ "${E2E_SMOKE:-0}" = 1 ]; then
    case "$R/" in "$SMOKE_PREFIX"*) ;; *) die "E2E_SMOKE=1 only below $SMOKE_PREFIX (got $R)";; esac
    TMODE=smoke
  elif [ "${CK2_E2E_APPROVED:-0}" != 1 ]; then
    die "main run needs CK2_E2E_APPROVED=1 (user approval after the GPU smoke)"
  fi
  if p=$(alive "$L/train.pid" run_training); then die "training already running (pid $p)"; fi
  if [ "$RESUME" = 0 ] && { [ -d "$R/train/lightning_logs" ] || [ -d "$IO/rec" ]; }; then
    die "$R already has a run; use --resume (or a new RUN)"
  fi
  local CKPT=""
  if [ "$RESUME" = 1 ]; then CKPT=$("$PY" "$LU1" find-last --run-dir "$R") || die "no last.ckpt to resume in $R"; fi
  if [ "${E2E_DRY_RUN:-0}" != 1 ]; then
    "$PY" "$LU2" gpu-check --gpus "$GPUS" > /dev/null || die "GPU(s) $GPUS busy (nvidia-smi > 1 GiB)"
  fi
  local OUTD="$L"
  [ "${E2E_DRY_RUN:-0}" = 1 ] && OUTD="$R/launch_dryrun"
  mkdir -p "$OUTD"
  local WB=${E2E_WANDB:-}
  [ -n "$WB" ] || WB=$("$PY" "$LU1" wandb-mode)
  local OA=(overrides --run-dir "$R" --gpus "$GPUS" --workers "${TRAIN_WORKERS:-6}" --max-epochs "${MAX_EPOCHS:-30}"
            --wandb "$WB" --teacher-check "$TMODE" --out "$OUTD/overrides.txt")
  [ "$ARM" = bevkd ] && OA+=(--ck-set-file "$REPO/tools/ck/e2e2/bevkd_arm.set")
  [ -n "${E2E_CK_SET:-}" ] && OA+=(--ck-set-file "$E2E_CK_SET")
  [ -n "${E2E_EXTRA:-}" ] && OA+=(--extra-file "$E2E_EXTRA")
  [ -n "$CKPT" ] && OA+=(--resume-ckpt "$CKPT")
  [ "${E2E_NO_DEEP_TEACHER_CHECK:-0}" = 1 ] && OA+=(--no-deep)
  "$PY" "$LU2" "${OA[@]}" || die "override generation / teacher / KD-calibration check failed (REFUSED line above)"
  mapfile -t OV < "$OUTD/overrides.txt"
  if [ "${E2E_DRY_RUN:-0}" = 1 ]; then printf '%s\n' "${OV[@]}"; echo "dry run: nothing launched (arm $ARM, $OUTD)"; return 0; fi
  mkdir -p "$R/train" "$IO"
  nvidia-smi > "$L/nvidia_smi_start.txt" 2>&1
  local NG; NG=$(awk -F, '{print NF}' <<< "$GPUS")
  log "start RUN=$RUN arm=$ARM gpus=$GPUS resume=$RESUME ckpt=${CKPT:-none} wandb=$WB overrides=${#OV[@]} (ngpu $NG)"
  rm -f "$L/train.exit" "$L/train.pid"
  setsid nohup env PYTHONPATH=$REPO NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
      NUPLAN_MAPS_ROOT=/home/external-user/yongjae/SSR/data/dataset/maps \
      OPENSCENE_DATA_ROOT=/home/external-user/yongjae/SSR/data/dataset NAVSIM_DEVKIT_ROOT=$REPO \
      NAVSIM_EXP_ROOT="$R" NCCL_SOCKET_IFNAME=lo CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      bash -c "$WRAP" _ "$L" train "$PY" "$REPO/navsim/planning/script/run_training.py" "${OV[@]}" < /dev/null > /dev/null 2>&1 &
  for _ in $(seq 50); do [ -s "$L/train.pid" ] && break; sleep 0.2; done
  [ -s "$L/train.pid" ] || die "training did not start (see $L/train.log)"
  log "training started pid $(cat "$L/train.pid") -> $L/train.log"
  start_labeler
  start_watchdog
}

do_wait() {
  local pidf=$L/train.pid needle=run_training exitf=$L/train.exit t0 to
  [ "$WAIT_LAB" = 1 ] && { pidf=$LB/labeler.pid; needle=labeler2; exitf=$LB/labeler.exit; }
  t0=$(date +%s); to=${E2E_WAIT_TIMEOUT:-0}
  while alive "$pidf" "$needle" > /dev/null; do
    if [ "$to" -gt 0 ] && [ $(( $(date +%s) - t0 )) -gt "$to" ]; then log "wait timeout ${to}s ($needle alive)"; return 124; fi
    sleep 5
  done
  for _ in $(seq 20); do [ -f "$exitf" ] && break; sleep 0.5; done
  local rc; rc=$(cat "$exitf" 2>/dev/null || echo 255)
  log "$needle exited $rc"
  return "$rc"
}

case "$CMD" in
  start) do_start;;
  status)
    if [ "$JSON" = 1 ]; then "$PY" "$LU2" status --run-dir "$R" --json; else "$PY" "$LU2" status --run-dir "$R"; fi;;
  stop)
    stop_watchdog
    grp_kill "$L/train.pgid" "$L/train.pid" run_training training
    [ "$TRAIN_ONLY" = 1 ] || grp_kill "$LB/labeler.pgid" "$LB/labeler.pid" labeler2 labeler;;
  wait) do_wait; exit $?;;
  _watchdog) do_watchdog;;
  labeler) mkdir -p "$L"; start_labeler; start_watchdog;;
  *) sed -n '2,12p' "$0"; exit 2;;
esac
