#!/usr/bin/env bash
# CK Phase 2 main run: v2 + CK from scratch, e2e 30 epochs on GPU 0-3 (report 45; contract launch.train_e2e.sh).
#
#   RUN=v2ck_d_g1 CK_E2E_APPROVED=1 bash tools/ck/e2e/train_e2e.sh start [--resume] [--gpus 0,1,2,3]
#   RUN=v2ck_d_g1 bash tools/ck/e2e/train_e2e.sh status [--json]
#   RUN=v2ck_d_g1 bash tools/ck/e2e/train_e2e.sh stop [--train-only]
#   RUN=v2ck_d_g1 bash tools/ck/e2e/train_e2e.sh wait [--labeler]        # block until train (or labeler) exits
#   RUN=v2ck_d_g1 bash tools/ck/e2e/train_e2e.sh labeler [--gpus 0,1,2,3]  # (re)start labeler + watchdog if not alive
#   (internal) train_e2e.sh _watchdog --gpus ...   labeler watchdog, started by 'start'
#   RUN=v2ck_d_g1 E2E_DRY_RUN=1 CK_E2E_APPROVED=1 bash tools/ck/e2e/train_e2e.sh start   # print overrides only
#
# Layout <E2E_ROOT>/<RUN>/ (E2E_ROOT default /home/external-user/ssd/yongjae_refiner/ck/phase2/e2e):
#   train/    navsim output_dir (lightning_logs/version_*/checkpoints/{epoch=E-step=S.ckpt,last.ckpt}, code/hydra)
#   ck_e2e/   ck_e2e.io_dir (rec/, lab/, steps_rank0.jsonl, epochs.jsonl, TRAIN_DONE)
#   labeler/  labeler.log, labeler.pid, labeler.pgid, labeler.exit, watchdog.pid, watchdog.log
#   launch/   train.log, train.pid (python), train.pgid (process group), train.exit, overrides.txt,
#             ck_e2e_effective.json, nvidia_smi_start.txt, launch.log
# start: guards (GPU list within 0-3, nvidia-smi <= 1 GiB used on each, CK_E2E_APPROVED=1, no '=' in paths, no live
#   train.pid, fresh start refuses an existing run dir) -> training (setsid nohup; DDP ranks share its process group)
#   -> labeler (setsid nohup, --exit-when-done --train-pid-file).  The training process is started first so the
#   labeler never sees a missing train.pid.  --resume: '++resume_checkpoint=<newest last.ckpt>', labeler kept if alive.
#   -> labeler watchdog (setsid nohup): every WATCHDOG_EVERY s (120) while training lives, a labeler that is not
#   running (crash, kill) is restarted (start_labeler --retry-errors; the labeler resumes cleanly), and an alive labeler
#   that is hung (launch_util labeler-health: status.json stale or no scored token for LABELER_STALL_S (1200 s) with a
#   backlog) is stopped (TERM group, KILL after 60 s) and restarted; at most LABELER_MAX_RESTARTS (20) restarts.
# stop: watchdog first, then TERM to the process groups from the pid files, KILL after 60 s.  Never pkill / pgrep -f.
# Env knobs (smoke_e2e.sh uses them; the main run needs none):
#   E2E_ROOT, TRAIN_WORKERS (6), MAX_EPOCHS (30), LABELER_WORKERS (24), LABELER_POLL (30), LABELER_MAX_EPOCH (max-2),
#   E2E_NO_LABELER=1, E2E_WANDB=online|offline|disable (default: online if the W&B API answers, else offline),
#   E2E_CK_SET=<file of ck_e2e key=value lines>, E2E_EXTRA=<file of extra Hydra overrides>,
#   E2E_SMOKE=1 (approval gate waived, only below .../phase2/e2e_smoke/), E2E_WAIT_TIMEOUT (s, wait), E2E_DRY_RUN=1,
#   WATCHDOG_EVERY (120 s), LABELER_MAX_RESTARTS (20), LABELER_STALL_S (1200 s), E2E_NO_WATCHDOG=1.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"   # repo root of this worktree
PY=/venv/ssr/bin/python
LU="$REPO/tools/ck/e2e/launch_util.py"
SMOKE_PREFIX=/home/external-user/ssd/yongjae_refiner/ck/phase2/e2e_smoke/
CMD=${1:-}; shift || true
RESUME=0; GPUS=0,1,2,3; TRAIN_ONLY=0; JSON=0; WAIT_LAB=0
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
RUN=${RUN:-v2ck_d_g1}
E2E_ROOT=${E2E_ROOT:-/home/external-user/ssd/yongjae_refiner/ck/phase2/e2e}
R="$E2E_ROOT/$RUN"
L="$R/launch"; LB="$R/labeler"; IO="$R/ck_e2e"
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

log() { echo "[$(date -u '+%F %T') UTC] $*"; [ -d "$L" ] && echo "[$(date -u '+%F %T') UTC] $*" >> "$L/launch.log"; return 0; }
die() { echo "[$(date -u '+%F %T') UTC] REFUSED: $*" >&2; exit 3; }
# job wrapper (process-group leader): $1 dir, $2 name, rest = command.  Writes <name>.pgid / .pid / .exit; a TERM
# to the group is forwarded and the wrapper still records the job's exit code.
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

case "$R" in *=*) die "run dir $R contains '='";; esac

start_labeler() {  # [retry]: add --retry-errors (watchdog restarts)
  if [ "${E2E_NO_LABELER:-0}" = 1 ]; then log "labeler disabled (E2E_NO_LABELER=1)"; return 0; fi
  if p=$(alive "$LB/labeler.pid" labeler); then log "labeler alive (pid $p): kept"; return 0; fi
  local ng me; ng=$(awk -F, '{print NF}' <<< "$GPUS"); me=${LABELER_MAX_EPOCH:-$(( ${MAX_EPOCHS:-30} - 2 ))}
  [ "$me" -lt 0 ] && me=0
  mkdir -p "$LB"; rm -f "$LB/labeler.exit" "$LB/labeler.pid"
  local LCMD=("$PY" "$REPO/tools/ck/e2e/labeler.py" --io-dir "$IO" --world-size "$ng" --workers "${LABELER_WORKERS:-24}"
              --poll "${LABELER_POLL:-30}" --max-epoch "$me" --exit-when-done --train-pid-file "$L/train.pid")
  [ "${1:-}" = retry ] && LCMD+=(--retry-errors)
  echo "${LCMD[*]}" > "$LB/labeler.cmd"
  setsid nohup bash -c "$WRAP" _ "$LB" labeler env PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "${LCMD[@]}" \
    < /dev/null > /dev/null 2>&1 &
  for _ in $(seq 30); do [ -s "$LB/labeler.pid" ] && break; sleep 0.2; done
  log "labeler started pid $(cat "$LB/labeler.pid" 2>/dev/null) W=${LABELER_WORKERS:-24} max-epoch $me -> $LB/labeler.log"
}

start_watchdog() {
  if [ "${E2E_NO_LABELER:-0}" = 1 ] || [ "${E2E_NO_WATCHDOG:-0}" = 1 ]; then log "labeler watchdog off"; return 0; fi
  if p=$(alive "$LB/watchdog.pid" _watchdog); then log "watchdog alive (pid $p): kept"; return 0; fi
  mkdir -p "$LB"; rm -f "$LB/watchdog.pid"
  setsid nohup bash "$REPO/tools/ck/e2e/train_e2e.sh" _watchdog --gpus "$GPUS" < /dev/null >> "$LB/watchdog.log" 2>&1 &
  for _ in $(seq 30); do [ -s "$LB/watchdog.pid" ] && break; sleep 0.2; done
  log "labeler watchdog started pid $(cat "$LB/watchdog.pid" 2>/dev/null) every ${WATCHDOG_EVERY:-120}s -> $LB/watchdog.log"
}

do_watchdog() {  # runs as its own session leader (pgid = pid); restarts a dead labeler while training lives
  echo $$ > "$LB/watchdog.pid"
  local n=0 every=${WATCHDOG_EVERY:-120} maxr=${LABELER_MAX_RESTARTS:-20} ex
  echo "[$(date -u '+%F %T') UTC] watchdog pid $$ every ${every}s max restarts $maxr"
  sleep "$every"
  while alive "$L/train.pid" run_training > /dev/null; do
    if alive "$LB/labeler.pid" labeler > /dev/null; then
      local why
      why=$("$PY" "$LU" labeler-health --run-dir "$R" --state "$LB/watchdog.state" --stall-s "${LABELER_STALL_S:-1200}")
      if [ $? = 5 ]; then
        log "watchdog: labeler hung ($why): stopping it"
        grp_kill "$LB/labeler.pgid" "$LB/labeler.pid" labeler labeler
      fi
    fi
    if ! alive "$LB/labeler.pid" labeler > /dev/null; then
      ex=$(cat "$LB/labeler.exit" 2>/dev/null || echo "?")
      if [ "$ex" = 0 ] && [ -f "$IO/TRAIN_DONE" ]; then log "watchdog: labeler finished (exit 0, TRAIN_DONE)"; break; fi
      if [ "$n" -ge "$maxr" ]; then
        log "watchdog: labeler not running (exit $ex); restart limit $maxr reached: giving up"; break
      fi
      n=$((n + 1))
      log "watchdog: labeler not running (exit $ex) while training is alive: restart $n/$maxr"
      echo "$(date -u '+%F %T') restart $n exit $ex" >> "$LB/watchdog.restarts"
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
  # ---- guards
  "$PY" "$LU" gpu-check --gpus "$GPUS" --list-only > /dev/null || die "GPU list $GPUS (0-3 only)"
  if [ "${E2E_SMOKE:-0}" = 1 ]; then
    case "$R/" in "$SMOKE_PREFIX"*) ;; *) die "E2E_SMOKE=1 only below $SMOKE_PREFIX (got $R)";; esac
  elif [ "${CK_E2E_APPROVED:-0}" != 1 ]; then
    die "main run needs CK_E2E_APPROVED=1 (user approves after reading the smoke KD loss table)"
  fi
  if p=$(alive "$L/train.pid" run_training); then die "training already running (pid $p)"; fi
  if [ "$RESUME" = 0 ] && { [ -d "$R/train/lightning_logs" ] || [ -d "$IO/rec" ]; }; then
    die "$R already has a run; use --resume (or a new RUN)"
  fi
  local CKPT=""
  if [ "$RESUME" = 1 ]; then CKPT=$("$PY" "$LU" find-last --run-dir "$R") || die "no last.ckpt to resume in $R"; fi
  if [ "${E2E_DRY_RUN:-0}" != 1 ]; then
    "$PY" "$LU" gpu-check --gpus "$GPUS" > /dev/null || die "GPU(s) $GPUS busy (nvidia-smi > 1 GiB)"
  fi
  mkdir -p "$L" "$R/train" "$IO"
  local WB=${E2E_WANDB:-}
  [ -n "$WB" ] || WB=$("$PY" "$LU" wandb-mode)
  local OA=(overrides --run-dir "$R" --gpus "$GPUS" --workers "${TRAIN_WORKERS:-6}" --max-epochs "${MAX_EPOCHS:-30}"
            --wandb "$WB" --out "$L/overrides.txt")
  [ -n "${E2E_CK_SET:-}" ] && OA+=(--ck-set-file "$E2E_CK_SET")
  [ -n "${E2E_EXTRA:-}" ] && OA+=(--extra-file "$E2E_EXTRA")
  [ -n "$CKPT" ] && OA+=(--resume-ckpt "$CKPT")
  "$PY" "$LU" "${OA[@]}" || die "override generation failed"
  mapfile -t OV < "$L/overrides.txt"
  if [ "${E2E_DRY_RUN:-0}" = 1 ]; then printf '%s\n' "${OV[@]}"; log "dry run: nothing launched"; return 0; fi
  nvidia-smi > "$L/nvidia_smi_start.txt" 2>&1
  local NG; NG=$(awk -F, '{print NF}' <<< "$GPUS")
  log "start RUN=$RUN gpus=$GPUS resume=$RESUME ckpt=${CKPT:-none} wandb=$WB overrides=${#OV[@]} (ngpu $NG)"
  rm -f "$L/train.exit" "$L/train.pid"
  # ---- training (process-group leader = wrapper bash; DDP ranks are its descendants)
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
  [ "$WAIT_LAB" = 1 ] && { pidf=$LB/labeler.pid; needle=labeler; exitf=$LB/labeler.exit; }
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
    if [ "$JSON" = 1 ]; then "$PY" "$LU" status --run-dir "$R" --json; else "$PY" "$LU" status --run-dir "$R"; fi;;
  stop)
    stop_watchdog
    grp_kill "$L/train.pgid" "$L/train.pid" run_training training
    [ "$TRAIN_ONLY" = 1 ] || grp_kill "$LB/labeler.pgid" "$LB/labeler.pid" labeler labeler;;
  wait) do_wait; exit $?;;
  _watchdog) do_watchdog;;
  labeler) mkdir -p "$L"; start_labeler; start_watchdog;;
  *) sed -n '2,8p' "$0"; exit 2;;
esac
