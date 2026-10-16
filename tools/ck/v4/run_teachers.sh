#!/bin/bash
# refiner v4 teachers: DET (arm T, BEVFusion) then MAP (arm M, ReSMap), each 2-GPU DDP, the ck2T10 / ck2M10 recipe
# (10 epochs, 16 tokens per step, lr 6e-4, sur near_mid, eval every 2, official EP, seed 0) + the v4 network:
# READOUT=hazard (default, as before): hazard readout + P_ref query + corridor version 2 + censored station BCE on the
# time-index sidecars -> v4T10_haz / v4M10_haz;
# READOUT=range (refiner v4_3, docs/refiner_v4/V4_3_SPEC.md): range readout (V4Cfg.range_arm: LQR executed states,
# 20 bins of 0.25 s, EP evidence) + P_ref query + corridor version 2 + the GT field aux of train_ck2 (DET blocked field
# B / U, MAP SDF margin; --lambda-ev) -> v4T10_rng / v4M10_rng (logs / pid files teachers_rng.*).
# Specialist keys as ck2T10 / ck2M10 (DET nc,ttc,ep,comfort; MAP dac,ep); train_ck2 restricts the station / field
# terms to them.
#
#   V4_USER_APPROVED=1 [V4_GPUS=2,3] [READOUT=hazard|range] bash tools/ck/v4/run_teachers.sh [both|det|map]
#   V4_DRY_RUN=1 [READOUT=...] bash tools/ck/v4/run_teachers.sh [both|det|map]   # print the train_ck2 arguments only
#                                                                                # (no approval / GPU check, no file)
#
# Refuses unless V4_USER_APPROVED=1 and the GPUs are idle (< 1 GB).  Writes, under $OUT (CK/v4/train):
#   <tag>.code.tgz + <tag>.launch.json (git head / dirty diff sha, command, KST), <tag>.{log,pid,pgid},
#   <run>.log (torchrun output) and the runs <run>/ (--resume: rerunning continues an interrupted run); <tag> =
#   teachers (hazard) | teachers_rng (range).
# The chain runs MAP after DET even if DET fails (independent runs).  Stop: kill -TERM -- -$(cat $OUT/<tag>.pgid)
set -euo pipefail
WHICH=${1:-both}
case "$WHICH" in both|det|map) ;; *) echo "usage: run_teachers.sh [both|det|map]"; exit 2 ;; esac
READOUT=${READOUT:-hazard}
case "$READOUT" in
  hazard) SUF=haz; TAG=teachers; RO_ARGS="--v4-readout hazard" ;;
  range) SUF=rng; TAG=teachers_rng
         RO_ARGS="--v4-readout range --v4-exec-model lqr --v4-ev-tbin-s 0.25 --v4-ep-evidence 1" ;;
  *) echo "refusing: READOUT='$READOUT' (hazard or range)"; exit 2 ;;
esac
DRY=${V4_DRY_RUN:-0}
[ "$DRY" = 1 ] || [ "${V4_USER_APPROVED:-0}" = 1 ] || {
  echo "refusing: GPU training needs the user's approval (V4_USER_APPROVED=1)"; exit 2; }
V4=$(cd "$(dirname "$0")/../../.." && pwd)
CK=/home/external-user/ssd/yongjae_refiner/ck
OUT=${V4_OUT:-$CK/v4/train}
TI=$CK/v4/time_idx
CVD=${V4_GPUS:-2,3}
case "$CVD" in 0,1|2,3) ;; *) echo "refusing: V4_GPUS='$CVD' (0,1 or 2,3)"; exit 2 ;; esac
if [ "$DRY" != 1 ]; then
  for g in ${CVD//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
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
