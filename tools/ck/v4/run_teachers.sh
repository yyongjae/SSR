#!/bin/bash
# refiner v4 teachers: DET (arm T, BEVFusion) then MAP (arm M, ReSMap), each 2-GPU DDP, the ck2T10 / ck2M10 recipe
# (10 epochs, 16 tokens per step, lr 6e-4, sur near_mid, eval every 2, official EP, seed 0) + the v4 network:
# READOUT=hazard (default, as before): hazard readout + P_ref query + corridor version 2 + censored station BCE on the
# time-index sidecars -> v4T10_haz / v4M10_haz;
# READOUT=range (refiner v4_3, docs/refiner_v4/V4_3_SPEC.md): range readout (V4Cfg.range_arm: LQR executed states,
# 20 bins of 0.25 s, EP evidence) + P_ref query + corridor version 2 + the GT field aux of train_ck2 (DET blocked field
# B / U, MAP SDF margin; --lambda-ev) -> v4T10_rng / v4M10_rng (logs / pid files teachers_rng.*);
# READOUT=range R2=1 (Phase 0b representation R2, spec 'Phase 0b'): the same + --v4-rg-read any4 (B / U read as the OR
# of the 2 x 2 cells around each executed point) + --v4-rg-stopped-aware 1 (B target = the stopped-aware blocked field)
# -> v4T10_rng2 / v4M10_rng2 (logs / pid files teachers_rng2.*).
# Specialist keys as ck2T10 / ck2M10 (DET nc,ttc,ep,comfort; MAP dac,ep); train_ck2 restricts the station / field
# terms to them.
#
#   V4_USER_APPROVED=1 [V4_GPUS=2,3] [READOUT=hazard|range] [R2=0|1] bash tools/ck/v4/run_teachers.sh [both|det|map]
#   V4_DRY_RUN=1 [READOUT=...] [R2=...] bash tools/ck/v4/run_teachers.sh [both|det|map]   # print the train_ck2
#                                                                     # arguments only (no approval / GPU check, no file)
#
# Refuses unless V4_USER_APPROVED=1 and the GPUs are idle (< 1 GB).  Writes, under $OUT (CK/v4/train):
#   <tag>.code.tgz + <tag>.launch.json (git head / dirty diff sha, command, KST), <tag>.{log,pid,pgid},
#   <run>.log (torchrun output) and the runs <run>/ (--resume: rerunning continues an interrupted run); <tag> =
#   teachers (hazard) | teachers_rng (range) | teachers_rng2 (range, R2=1).
# The chain runs MAP after DET even if DET fails (independent runs).  Stop: kill -TERM -- -$(cat $OUT/<tag>.pgid)
#
# WAIT mode (V4_WAIT_IDLE_MIN=<minutes>, still needs V4_USER_APPROVED=1; GPUs 0,1 or 2,3 only): instead of refusing on
# busy GPUs, a background waiter polls the chosen GPUs every V4_WAIT_POLL_S (60) s and launches (this script again, the
# normal path above with its own < 1 GB check) only after EVERY chosen GPU has been continuously idle -- < 1024 MiB
# used AND 0 % utilisation, a failed nvidia-smi query counts as busy -- for V4_WAIT_IDLE_MIN minutes; a launch refused
# by the memory check (exit 3) resets the idle clock and the waiter keeps waiting; it gives up after V4_WAIT_MAX_H (72)
# hours.  Log (one line per poll + the launch output): $OUT/<tag>.wait.log; pid / pgid: $OUT/<tag>.wait.{pid,pgid}
# (stop: kill -TERM -- -$(cat $OUT/<tag>.wait.pgid)); one waiter per tag (a foreground waiter, V4_WAIT_FOREGROUND=1
# or a dry run, writes <tag>.wait.pid too).  Neither WAIT nor a normal (non-dry) launch starts while a chain of the
# same tag is alive ($OUT/<tag>.pid, kill -0): exit 6, so a waiter whose launch meets a running chain stops.  The code
# snapshot (<tag>.code.tgz, launch.json) is taken at the launch, i.e. it records the code that actually runs; the wait
# log records the git state at the wait start.  With V4_DRY_RUN=1 the waiter runs in the foreground and its launch is the dry run (prints the
# arguments; the wait log is still written).  V4_NVIDIA_SMI overrides the nvidia-smi binary (tests).
set -euo pipefail
WHICH=${1:-both}
case "$WHICH" in both|det|map) ;; *) echo "usage: run_teachers.sh [both|det|map]"; exit 2 ;; esac
READOUT=${READOUT:-hazard}
R2=${R2:-0}
RNG_ARGS="--v4-readout range --v4-exec-model lqr --v4-ev-tbin-s 0.25 --v4-ep-evidence 1"
case "$READOUT" in
  hazard) SUF=haz; TAG=teachers; RO_ARGS="--v4-readout hazard"
          [ "$R2" = 0 ] || { echo "refusing: R2=$R2 needs READOUT=range"; exit 2; } ;;
  range) case "$R2" in
           0) SUF=rng; TAG=teachers_rng; RO_ARGS="$RNG_ARGS" ;;
           1) SUF=rng2; TAG=teachers_rng2; RO_ARGS="$RNG_ARGS --v4-rg-read any4 --v4-rg-stopped-aware 1" ;;
           *) echo "refusing: R2='$R2' (0 or 1)"; exit 2 ;;
         esac ;;
  *) echo "refusing: READOUT='$READOUT' (hazard or range)"; exit 2 ;;
esac
DRY=${V4_DRY_RUN:-0}
WAIT=${V4_WAIT_IDLE_MIN:-}
[ "$DRY" = 1 ] || [ "${V4_USER_APPROVED:-0}" = 1 ] || {
  echo "refusing: GPU training needs the user's approval (V4_USER_APPROVED=1)"; exit 2; }
V4=$(cd "$(dirname "$0")/../../.." && pwd)
SELF="$V4/tools/ck/v4/run_teachers.sh"
CK=/home/external-user/ssd/yongjae_refiner/ck
OUT=${V4_OUT:-$CK/v4/train}
TI=$CK/v4/time_idx
CVD=${V4_GPUS:-2,3}
SMI=${V4_NVIDIA_SMI:-nvidia-smi}
case "$CVD" in 0,1|2,3) ;; *) echo "refusing: V4_GPUS='$CVD' (0,1 or 2,3)"; exit 2 ;; esac
if [ -n "$WAIT" ]; then
  /venv/ssr/bin/python -c "import sys; v = float(sys.argv[1]); sys.exit(0 if v > 0 else 1)" "$WAIT" 2>/dev/null || {
    echo "refusing: V4_WAIT_IDLE_MIN='$WAIT' (minutes > 0)"; exit 2; }
fi
chain_alive() {  # a launched chain of this tag still running?
  [ -f "$OUT/$TAG.pid" ] && kill -0 "$(cat "$OUT/$TAG.pid")" 2>/dev/null
}
if { [ -n "$WAIT" ] || [ "$DRY" != 1 ]; } && chain_alive; then
  echo "refusing: a $TAG chain is running (pid $(cat "$OUT/$TAG.pid"); log $OUT/$TAG.log)"; exit 6
fi
if [ "$DRY" != 1 ] && [ -z "$WAIT" ]; then
  for g in ${CVD//,/ }; do
    used=$("$SMI" --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    [ "$used" -lt 1024 ] || { echo "refusing: GPU $g uses ${used} MiB (needs < 1024)"; exit 3; }
  done
fi
for s in navtrain_train navtrain_val; do
  [ -d "$TI/$s" ] || { echo "refusing: time-index sidecars $TI/$s missing (tools/ck/v4/build_time_idx.py)"; exit 4; }
done
cd "$V4"
COMMON="--epochs 10 --tokens-per-batch 16 --lr 6e-4 --sur-groups near_mid --eval-every 2 --best-metric none \
$RO_ARGS --v4-pref-head 1 --corridor-version 2 --time-idx-root $TI --out-root $OUT --resume"
RUN_T=v4T10_$SUF
RUN_M=v4M10_$SUF
DET="--arm T --run-name $RUN_T --score-keys nc,ttc,ep,comfort $COMMON"
MAP="--arm M --run-name $RUN_M --score-keys dac,ep $COMMON"
if [ -n "$WAIT" ]; then
  # ---------------------------------------------------------------------------------------------- WAIT mode
  mkdir -p "$OUT"
  WLOG="$OUT/$TAG.wait.log"
  if [ -f "$OUT/$TAG.wait.pid" ] && kill -0 "$(cat "$OUT/$TAG.wait.pid")" 2>/dev/null; then
    echo "refusing: a waiter for $TAG is running (pid $(cat "$OUT/$TAG.wait.pid"); log $WLOG)"; exit 2
  fi
  if [ "$DRY" = 1 ]; then
    GIT_STATE="dry run: not recorded"
  else
    GIT_STATE="head $(git rev-parse HEAD 2>/dev/null || echo n/a), dirty diff sha256 $( (git diff HEAD 2>/dev/null || true) | sha256sum | cut -c1-16)"
  fi
  {
    echo "[$(TZ=Asia/Seoul date '+%F %T KST')] WAIT request: $WHICH ($READOUT, R2=$R2) on GPUs $CVD, idle >= $WAIT min"
    echo "  git at the wait start: $GIT_STATE"
    echo "  DET $DET"
    echo "  MAP $MAP"
  } >> "$WLOG"
  WAITER=$(cat <<'EOF'
import os, subprocess, sys, time
sh_path, which, gpus, idle_min, poll_s, max_h, log_path, smi = sys.argv[1:9]
gpus = gpus.split(",")
idle_s, poll, max_s = float(idle_min) * 60.0, float(poll_s), float(max_h) * 3600.0
kst = lambda: time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime(time.time() + 9 * 3600))


def log(msg):
    with open(log_path, "a") as f:
        f.write(f"[{kst()}] {msg}\n")


def probe():
    """[(gpu, mem MiB, util %, idle)]: idle = < 1024 MiB and 0 % util; a failed query is busy."""
    out = []
    for g in gpus:
        try:
            r = subprocess.run([smi, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
                                "-i", g], capture_output=True, text=True, timeout=60)
            mem, util = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
            out.append((g, mem, util, r.returncode == 0 and mem < 1024 and util == 0))
        except Exception:
            out.append((g, None, None, False))
    return out


t0, since = time.time(), None
log(f"waiter pid {os.getpid()}: GPUs {','.join(gpus)} must be idle (< 1024 MiB, 0 % util) for {idle_min} min; "
    f"poll {poll:g} s, give up after {max_h} h")
while True:
    now = time.time()
    st = probe()
    idle = all(x[3] for x in st)
    since = (now if since is None else since) if idle else None
    dur = 0.0 if since is None else now - since
    log(" | ".join(f"gpu{g} " + ("query failed" if m is None else f"{m:.0f} MiB {u:.0f} %") for g, m, u, _ in st)
        + f" -> {'idle' if idle else 'busy'} {dur / 60:.1f} / {float(idle_min):g} min")
    if idle and dur >= idle_s:
        log(f"idle for {dur / 60:.1f} min: launching {which}")
        env = dict(os.environ)
        env.pop("V4_WAIT_IDLE_MIN", None)
        r = subprocess.run(["bash", sh_path, which], env=env, capture_output=True, text=True)
        for line in (r.stdout + r.stderr).strip().splitlines()[-40:]:
            log(f"  | {line}")
        if r.returncode == 3:
            log("launch refused by the GPU memory check (exit 3): waiting again")
            since = None
        else:
            log(f"launch exit {r.returncode}; waiter done")
            sys.exit(r.returncode)
    if now - t0 >= max_s:
        log(f"gave up after {(now - t0) / 3600:.2f} h without a launch")
        sys.exit(5)
    time.sleep(poll)
EOF
)
  WARGS=("$SELF" "$WHICH" "$CVD" "$WAIT" "${V4_WAIT_POLL_S:-60}" "${V4_WAIT_MAX_H:-72}" "$WLOG" "$SMI")
  if [ "$DRY" = 1 ] || [ "${V4_WAIT_FOREGROUND:-0}" = 1 ]; then
    echo "[run_teachers] waiting in the foreground ($([ "$DRY" = 1 ] && echo 'dry run' || echo 'real launch')): $WLOG"
    echo "$$" > "$OUT/$TAG.wait.pid"                   # exec keeps the pid: the one-waiter guard sees this waiter
    exec /venv/ssr/bin/python -B -c "$WAITER" "${WARGS[@]}"
  fi
  setsid nohup /venv/ssr/bin/python -B -c "$WAITER" "${WARGS[@]}" > "$OUT/$TAG.wait.out" 2>&1 < /dev/null &
  WPID=$!
  echo "$WPID" > "$OUT/$TAG.wait.pid"
  sleep 1
  ps -o pgid= -p "$WPID" | tr -d ' ' > "$OUT/$TAG.wait.pgid" || true
  echo "[run_teachers] $(TZ=Asia/Seoul date '+%F %T KST') waiter pid $WPID: launches $WHICH ($READOUT) on GPUs $CVD" \
       "after $WAIT min of continuous idle; log $WLOG; stop: kill -TERM -- -\$(cat $OUT/$TAG.wait.pgid)"
  exit 0
fi
if [ "$DRY" = 1 ]; then
  echo "[run_teachers] dry run ($READOUT, $WHICH): nothing launched"
  echo "DET $DET"
  echo "MAP $MAP"
  exit 0
fi
mkdir -p "$OUT"
TGZ="$OUT/$TAG.code.tgz"
tar czf "$TGZ" --exclude=__pycache__ navsim tools docs/refiner_v4
/venv/ssr/bin/python -B - "$OUT/$TAG.launch.json" "$TGZ" "$WHICH" "$CVD" "$DET" "$MAP" <<'EOF'
import hashlib, json, os, subprocess, sys, time
out, tgz, which, gpus, det, mapa = sys.argv[1:7]
sh = lambda *c: subprocess.run(list(c), capture_output=True, text=True).stdout
rec = json.load(open(out)) if os.path.isfile(out) else {"launches": []}
rec["launches"].append(dict(
    kst=time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime(time.time() + 9 * 3600)), which=which, gpus=gpus,
    det_args=det, map_args=mapa, git_head=sh("git", "rev-parse", "HEAD").strip(),
    git_dirty_diff_sha256=hashlib.sha256(sh("git", "diff", "HEAD").encode()).hexdigest(),
    git_status=sh("git", "status", "--short").splitlines(),
    code_tgz=tgz, code_tgz_sha256=hashlib.sha256(open(tgz, "rb").read()).hexdigest()))
json.dump(rec, open(out, "w"), indent=1)
EOF
CHAIN=$(cat <<EOF
cd "$V4"
run() {  # name, args
  echo "[\$(TZ=Asia/Seoul date '+%F %T KST')] start \$1 on GPUs $CVD" >> "$OUT/\$1.log"
  env CUDA_VISIBLE_DEVICES=$CVD PYTHONPATH=$V4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    nice -n 10 /venv/ssr/bin/torchrun --standalone --nproc_per_node=2 tools/ck/train_ck2.py \$2 >> "$OUT/\$1.log" 2>&1
  rc=\$?
  echo "[\$(TZ=Asia/Seoul date '+%F %T KST')] end \$1 exit \$rc" >> "$OUT/\$1.log"
  echo "[\$(TZ=Asia/Seoul date '+%F %T KST')] \$1 exit \$rc" >> "$OUT/$TAG.log"
}
EOF
)
case "$WHICH" in
  both) CHAIN="$CHAIN
run $RUN_T \"$DET\"
run $RUN_M \"$MAP\"" ;;
  det) CHAIN="$CHAIN
run $RUN_T \"$DET\"" ;;
  map) CHAIN="$CHAIN
run $RUN_M \"$MAP\"" ;;
esac
echo "[run_teachers] $(TZ=Asia/Seoul date '+%F %T KST') $WHICH ($READOUT) on GPUs $CVD -> $OUT" | tee -a "$OUT/$TAG.log"
setsid nohup bash -c "$CHAIN" > "$OUT/$TAG.chain.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/$TAG.pid"
sleep 1
ps -o pgid= -p "$PID" | tr -d ' ' > "$OUT/$TAG.pgid"
echo "[run_teachers] pid $PID pgid $(cat "$OUT/$TAG.pgid") logs $OUT/{$RUN_T,$RUN_M}.log"
