#!/bin/bash
# refiner-v3 trainer smoke tests (docs/refiner_v3/SPEC.md 5.4, 9 Integrate / Fix).  Few-step runs only, never a
# training run.
#
#   bash tools/ck/v3/smoke_v3.sh cpu            # stage pre + arms off / a (frozen) / a joint / b / d on CPU, 3 steps
#   bash tools/ck/v3/smoke_v3.sh gpu0 [arms]    # GPU 0, world 1, 8 tokens / step, 60 steps (default arms: pre a off)
#   bash tools/ck/v3/smoke_v3.sh gpu01 [arms]   # the real Step-1 setup: stage pre on GPU 0 (60 steps), then arms on
#                                               # GPUs 0,1, world 2, 16 tokens / step, 60 steps (default: pre a b off)
#
# Inputs: the Step-1 data itself (review fix D1/D2: K_OBJ = 96 packs D3/objpack_k96/bevfusion/{navtrain_train,
# navtrain_val}, D3/ca_scales.json fitted on the augmented train stream); norm.npz from ck2T10 (--norm-from: a data
# statistic).  Arm a uses the forecaster of THIS mode's smoke stage pre (D3/smoke/<mode>/smoke_<mode>_pre/forecaster.pt,
# frozen, as in Step 1); 'aj' = arm a with --t-fc-train joint (the optional fine-tuning path).
# Outputs D3/smoke/<mode>/<run>/; a summary line per arm (gpu_mem_gb max, median s/step and cand_per_s over the
# step-log windows after the first, skipped steps, val metrics finite) is printed by tools/ck/v3/smoke_summary.py.
# GPU modes refuse unless the GPUs used are idle (< 1 GB); each run is capped by `timeout`.
set -uo pipefail
MODE=${1:?usage: smoke_v3.sh cpu|gpu0|gpu01 [arms...]}
shift
V3=/workspace/yongjae/SSR-v3
D3=/home/external-user/ssd/yongjae_refiner/ck/v3
SM=$D3/smoke
PY=/venv/ssr/bin/python
NORM=/home/external-user/ssd/yongjae_refiner/ck/ck2/train/ck2T10
PACKS=$D3/objpack_k96
SCALES=$D3/ca_scales.json
OUT=$SM/$MODE
PRE=$OUT/smoke_${MODE}_pre
cd "$V3"
for s in navtrain_train navtrain_val; do
  [ -f "$PACKS/bevfusion/$s/build.json" ] || { echo "pack $s missing under $PACKS"; exit 4; }
done
[ -f "$SCALES" ] || { echo "$SCALES missing"; exit 4; }
RECIPE="--arm T --lr 6e-4 --sur-groups near_mid --best-metric none --score-keys nc,ttc,ep,comfort --n-var 80
  --var-name var_separate_sampler16_accstraight --seed 0 --sampler-seed 0 --norm-from $NORM --warmup-steps 500
  --wd 0.01 --clip 1.0 --ep-target official --eval-every 1 --ckpt-every 0 --restart"
TF="--t-pack-root $PACKS --t-ca-scales $SCALES"
tflags() {   # arm -> temporal flags (same values as tools/ck/v3/arms/*.json)
  case "$1" in
    off) echo "--temporal 0" ;;
    a) echo "--temporal 1 --t-forecaster learned --t-modes 6 $TF --t-fc-init $PRE/forecaster.pt --t-fc-train frozen" ;;
    aj) echo "--temporal 1 --t-forecaster learned --t-modes 6 $TF --t-fc-init $PRE/forecaster.pt --t-fc-train joint" ;;
    b) echo "--temporal 1 --t-forecaster ca --t-modes 1 $TF" ;;
    d) echo "--temporal 1 --t-forecaster oracle_gt --t-modes 1 $TF" ;;
    *) echo "bad arm $1" >&2; exit 2 ;;
  esac
}
rname() { case "$1" in d) echo "smoke_${MODE}_d_PRIV" ;; *) echo "smoke_${MODE}_$1" ;; esac; }
ENVC="PYTHONPATH=$V3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
gpu_idle() {
  for g in "$@"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    [ "$used" -lt 1024 ] || { echo "refusing: GPU $g uses ${used} MiB"; exit 3; }
  done
}
pre() {      # $1 device spec, $2 CUDA_VISIBLE_DEVICES, $3 extra args
  rm -rf "$PRE"
  # shellcheck disable=SC2086
  env $ENVC CUDA_VISIBLE_DEVICES="$2" nice -n 10 timeout "${PRE_TIMEOUT:-600}" $PY -B tools/ck/v3/pretrain_forecaster.py \
    --out "$PRE" --device "$1" --epochs 1 --pack-root "$PACKS" --ca-scales "$SCALES" --log-every 10 $3 \
    > "$OUT/smoke_${MODE}_pre.log" 2>&1
}
mkdir -p "$OUT"
rc=0
case "$MODE" in
  cpu)
    ARMS=${*:-"pre off a aj b d"}
    if [[ " $ARMS " == *" pre "* ]]; then
      pre cpu "" "--max-steps 3 --batch 8 --limit-tokens 64 --val-limit 16 --train-eval-rows 16 --workers 2" || rc=1
    fi
    pids=()
    for arm in $ARMS; do
      [ "$arm" = pre ] && continue
      # shellcheck disable=SC2046,SC2086
      env $ENVC CUDA_VISIBLE_DEVICES= nice -n 10 timeout 1800 $PY -B tools/ck/train_ck2.py $RECIPE $(tflags "$arm") \
        --device cpu --workers 2 --tokens-per-batch 2 --epochs 1 --max-steps 3 --limit-tokens 16 --val-limit 8 \
        --log-every 1 --out-root "$OUT" --run-name "$(rname "$arm")" > "$OUT/$(rname "$arm").log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    ;;
  gpu0)
    gpu_idle 0
    ARMS=${*:-"pre a off"}
    for arm in $ARMS; do
      if [ "$arm" = pre ]; then
        pre cuda 0 "--max-steps 60 --batch 64 --val-limit 512 --train-eval-rows 512 --workers 8" || rc=1
        continue
      fi
      # shellcheck disable=SC2046,SC2086
      env $ENVC CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 280 $PY -B tools/ck/train_ck2.py $RECIPE $(tflags "$arm") \
        --workers 8 --tokens-per-batch 8 --epochs 1 --max-steps 60 --limit-tokens 512 --val-limit 64 \
        --log-every 10 --out-root "$OUT" --run-name "$(rname "$arm")" > "$OUT/$(rname "$arm").log" 2>&1 || rc=1
    done
    ;;
  gpu01)
    gpu_idle 0 1
    ARMS=${*:-"pre a b off"}
    for arm in $ARMS; do
      if [ "$arm" = pre ]; then
        pre cuda 0 "--max-steps 60 --batch 64 --val-limit 512 --train-eval-rows 512 --workers 8" || rc=1
        continue
      fi
      # shellcheck disable=SC2046,SC2086
      env $ENVC CUDA_VISIBLE_DEVICES=0,1 nice -n 10 timeout 420 /venv/ssr/bin/torchrun --standalone --nproc_per_node=2 \
        tools/ck/train_ck2.py $RECIPE $(tflags "$arm") --workers 8 --tokens-per-batch 16 --epochs 1 --max-steps 60 \
        --limit-tokens 1024 --val-limit 64 --log-every 10 --out-root "$OUT" --run-name "$(rname "$arm")" \
        > "$OUT/$(rname "$arm").log" 2>&1 || rc=1
    done
    ;;
  *) echo "mode must be cpu, gpu0 or gpu01"; exit 2 ;;
esac
env $ENVC $PY -B tools/ck/v3/smoke_summary.py "$OUT"
exit $rc
