#!/bin/bash
# refiner v5 teachers (SSR/report/97_refiner_v5_design.md; Stage-1 readers + corrector A on frozen v4_3 trunks), 2-GPU DDP
# each, warm-started from the v4_3 range teachers (v4T10_rng2 / v4M10_rng2, R2 representation):
#   DET  v5T_rel4 arm T, relation channel (BEVFusion rows + causal CA) + unknown evidence + corrector A, L_refine all terms
#   MAP  v5M_unk3 arm M, unknown evidence + corrector A (L_refine road / progress / comfort) + corner margin aux
#   V5_USER_APPROVED=1 [V5_EPOCHS=3] [V5_TPB=32] [V5_LR=4e-4] [V5_WARMUP=200] [V5_DET_GPUS=0,1] [V5_MAP_GPUS=2,3] bash tools/ck/v5/run_teachers_v5.sh [both|det|map]
#   V5_DRY_RUN=1 ... (print the commands only); V5_DET_RUN / V5_MAP_RUN rename the runs (default v5T_rel3 / v5M_unk3:
#   v5T_rel / v5M_unk predate the soft-PDMS L_refine, *_rel2 / *_unk2 the report-100 fixes, v5T_rel3 the report-101 N1 fix)
# Refuses unless approved and every chosen GPU uses < 1 GB.  Outputs CK/v5/train/<run>/ (train_ck2 layout), <run>.log,
# <run>.{pid,pgid}, launch_<run>.json (git head, dirty-diff sha, command, KST).  Stop: kill -TERM -- -$(cat <run>.pgid)
set -euo pipefail
WHICH=${1:-both}
case "$WHICH" in both|det|map) ;; *) echo "usage: run_teachers_v5.sh [both|det|map]"; exit 2 ;; esac
DRY=${V5_DRY_RUN:-0}
[ "$DRY" = 1 ] || [ "${V5_USER_APPROVED:-0}" = 1 ] || { echo "refusing: GPU training needs V5_USER_APPROVED=1"; exit 2; }
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
CK=/home/external-user/ssd/yongjae_refiner/ck
OUT=${V5_OUT:-$CK/v5/train}
EP=${V5_EPOCHS:-3}
TPB=${V5_TPB:-32}       # 16 / rank: the step is CPU-launch-bound (relation + LQR rollouts), a larger batch amortises it
LR=${V5_LR:-4e-4}
WU=${V5_WARMUP:-200}
DG=${V5_DET_GPUS:-0,1}
MG=${V5_MAP_GPUS:-2,3}
DR=${V5_DET_RUN:-v5T_rel4}
MR=${V5_MAP_RUN:-v5M_unk3}
for g in $DG $MG; do case "$g" in 0,1|2,3) ;; *) echo "refusing: GPUs '$g' (0,1 or 2,3)"; exit 2 ;; esac; done
COMMON="--epochs $EP --tokens-per-batch $TPB --lr $LR --warmup-steps $WU --sur-groups near_mid --eval-every 2 \
--best-metric none --train-eval-rows 2048 --v4-readout range --v4-exec-model lqr --v4-ev-tbin-s 0.25 --v4-ep-evidence 1 \
--v4-rg-read any4 --v4-rg-stopped-aware 1 --v4-pref-head 1 --corridor-version 2 --time-idx-root $CK/v4/time_idx \
--v5-unknown 1 --v5-corrector 1 --v5-freeze-trunk 1 --out-root $OUT --resume"
DET="--arm T --run-name $DR --score-keys nc,ttc,ep,comfort --v5-relation teacher --v5-init $CK/v4/train/v4T10_rng2 $COMMON"
MAP="--arm M --run-name $MR --score-keys dac,ep --ref-w obj=0,ttc=0 --lambda-corner 0.5 --v5-init $CK/v4/train/v4M10_rng2 $COMMON"
launch() {  # launch <run> <gpus> <args>
  local run=$1 gpus=$2 args=$3
  local cmd="CUDA_VISIBLE_DEVICES=$gpus PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /venv/ssr/bin/torchrun --standalone --nproc_per_node=2 $REPO/tools/ck/train_ck2.py $args"
  if [ "$DRY" = 1 ]; then echo "$cmd"; return 0; fi
  for g in ${gpus//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    [ "$used" -lt 1024 ] || { echo "refusing: GPU $g uses ${used} MiB"; exit 3; }
  done
  if [ -f "$OUT/$run.pid" ] && kill -0 "$(cat "$OUT/$run.pid")" 2>/dev/null; then
    echo "refusing: $run is running (pid $(cat "$OUT/$run.pid"))"; exit 6
  fi
  mkdir -p "$OUT"
  ( cd "$REPO" && printf '{"run": "%s", "gpus": "%s", "git_head": "%s", "dirty_diff_sha16": "%s", "kst": "%s", "cmd": "%s"}\n' \
      "$run" "$gpus" "$(git rev-parse HEAD)" "$( (git diff HEAD || true) | sha256sum | cut -c1-16)" \
      "$(TZ=Asia/Seoul date '+%F %T KST')" "$(echo "$cmd" | sed 's/"/\\"/g')" ) > "$OUT/launch_$run.json"
  cd "$REPO"
  setsid nohup bash -c "$cmd" > "$OUT/$run.log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$OUT/$run.pid"
  ps -o pgid= -p "$pid" | tr -d ' ' > "$OUT/$run.pgid"
  echo "launched $run on GPUs $gpus: pid $pid, log $OUT/$run.log"
}
case "$WHICH" in
  both) launch "$DR" "$DG" "$DET"; launch "$MR" "$MG" "$MAP" ;;
  det) launch "$DR" "$DG" "$DET" ;;
  map) launch "$MR" "$MG" "$MAP" ;;
esac
