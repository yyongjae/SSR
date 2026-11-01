#!/usr/bin/env bash
# Stage E GPU / launch commands (report/refiner_T/stageE_impl_plan.md; PRESTATED_DECISION_RULE "STAGE E PLAN" + REVISION 1).
# Nothing here runs by itself on import; every step is an explicit sub-command:
#
#   bash tools/refiner/stageE_gpu_commands.sh gate                      # liveness gate of both run-4 teachers (CPU)
#   bash tools/refiner/stageE_gpu_commands.sh snapshot                  # read-only teacher snapshot + sha256 (after DONE)
#   bash tools/refiner/stageE_gpu_commands.sh pilot  E1 <gpu> <W>                         # 300 micro-batches (1 epoch
#   bash tools/refiner/stageE_gpu_commands.sh pilot  E2 <gpu> <W> <kd_space>              #  of 300), batch 4 x acc 32
#   bash tools/refiner/stageE_gpu_commands.sh lambda                    # pilot summary + lambda_c (E2 pilot, last 100)
#   bash tools/refiner/stageE_gpu_commands.sh train1 E1 <gpu> <N> <W>                     # plan: one GPU per arm,
#   bash tools/refiner/stageE_gpu_commands.sh train1 E2 <gpu> <N> <W> <lambda_c> <kd_space>  #   N epochs
#   bash tools/refiner/stageE_gpu_commands.sh train4 E1 <gpus>                            # REVISION 1: 4 GPUs,
#   bash tools/refiner/stageE_gpu_commands.sh train4 E2 <gpus> <lambda_c> <kd_space>      #   30 epochs, W = 3
#   KD_BALANCE=ema bash tools/refiner/stageE_gpu_commands.sh train4 E2 <gpus> - <kd_space>   # 1:1 EMA balance
#   bash tools/refiner/stageE_gpu_commands.sh pilot150 <gpu> <kd_space>   # 150 micro-batches, 1 GPU: EMA + grad shares
#   bash tools/refiner/stageE_gpu_commands.sh eval   E1|E2|off <ckpt> final|tau0 <gpu> <name>
# Post-pilot options (env; unset = the behaviour before they existed, identical command line):
#   KD_BALANCE=fixed|ema  KD_RATIO=1.0  KD_START_EPOCH=<epoch>|null (ema; null -> kd_ramp[0])
#   KD_DRAFT_SOURCE=tau0|human_mix  GRAD_SHARE_EVERY=<n micro-batches, 0 = off; default 50>
#   KD_RATIO_RAMP=<epochs> (ema: ratio 0 -> KD_RATIO over [KD_START_EPOCH, +this])  KD_WEIGHT_MAX=<cap on the ema weight>
#   REF_HUMAN_ONLY_UNTIL=<epoch> (all drafts = perturbed GT human before it; needs the human npz below)
#   REF_BEV_GRAD_SCALE=<x> (student BEV <- refiner gradient scale; default 0.1)
#   (human_mix reads the logged 8 s human path from <ref_data_root>/human/e2e_train_trainlogs.npz:
#    python tools/refiner/extract_human.py extract --splits e2e_train_trainlogs --workers 2)
#   bash tools/refiner/stageE_gpu_commands.sh compare                   # after the evals (edit the csv list below)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY=/home/external-user/miniconda3/envs/ssr/bin/python
export PATH="/home/external-user/miniconda3/envs/ssr/bin:$PATH" SSR_NAVSIM_PYTHON="$PY"
DATA=/home/external-user/ssd/yongjae_refiner
TEACH="$DATA/stageE/teachers"                 # write-once snapshot (stageE_prep.py snapshot)
KD_RUNS="[${TEACH}/stageT4_T_fold0_seed0,${TEACH}/stageT4_M_fold0_seed0]"
CMD="${1:?sub-command}"; shift

# W (warmup_epochs): WarmupCosLR is stepped once per EPOCH (lr in epoch e < W = lr * (e + 1) / W), so W = 1 is NO
# warm-up (epoch 0 at the peak); W = 2 -> epoch 0 at 0.5 lr; W = 3 -> E0's absolute warm-up (1/3, 2/3, 1).
# kd_space (E2): raw | tanh | decoded (refiner/e2e.py); no default -- the user's choice, fixed in the decision rule.
e2_args() {  # $1 = lambda_c, $2 = kd_space -> KD overrides; refuses a missing / non-positive lambda or kd_space
  # (KD_BALANCE=ema: lambda_c must be '-' -- the weight is the EMA ratio, kd_lambda is not used)
  local bal="${KD_BALANCE:-fixed}"
  local lam="${1:?E2 needs lambda_c (from the pilot rule), or '-' with KD_BALANCE=ema}" sp="${2:?E2 needs kd_space raw|tanh|decoded}"
  case "$sp" in raw|tanh|decoded) ;; *) echo "kd_space must be raw|tanh|decoded, got '$sp'" >&2; exit 2 ;; esac
  case "$bal" in
  fixed)
    [ -z "${KD_RATIO_RAMP:-}${KD_WEIGHT_MAX:-}" ] || { echo "KD_RATIO_RAMP / KD_WEIGHT_MAX need KD_BALANCE=ema" >&2; exit 2; }
    awk -v l="$lam" 'BEGIN { exit !(l + 0 > 0) }' || { echo "E2 needs lambda_c > 0, got '$lam'" >&2; exit 2; }
    echo "agent.config.kd_teacher_runs=$KD_RUNS agent.config.kd_lambda=$lam agent.config.kd_space=$sp" ;;
  ema)
    [ "$lam" = "-" ] || { echo "KD_BALANCE=ema does not use lambda_c; pass '-' (got '$lam')" >&2; exit 2; }
    awk -v r="${KD_RATIO:-1.0}" 'BEGIN { exit !(r + 0 > 0) }' || { echo "KD_RATIO must be > 0" >&2; exit 2; }
    local ex=""
    if [ -n "${KD_RATIO_RAMP:-}" ]; then
      awk -v r="$KD_RATIO_RAMP" 'BEGIN { exit !(r ~ /^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ && r + 0 > 0) }' || { echo "KD_RATIO_RAMP must be > 0" >&2; exit 2; }
      ex="$ex agent.config.kd_ratio_ramp_epochs=$KD_RATIO_RAMP"
    fi
    if [ -n "${KD_WEIGHT_MAX:-}" ]; then
      awk -v r="$KD_WEIGHT_MAX" 'BEGIN { exit !(r ~ /^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ && r + 0 > 0) }' || { echo "KD_WEIGHT_MAX must be > 0" >&2; exit 2; }
      ex="$ex agent.config.kd_weight_max=$KD_WEIGHT_MAX"
    fi
    echo "agent.config.kd_teacher_runs=$KD_RUNS agent.config.kd_space=$sp agent.config.kd_balance=ema agent.config.kd_ratio=${KD_RATIO:-1.0} agent.config.kd_start_epoch=${KD_START_EPOCH:-null}$ex" ;;
  *) echo "KD_BALANCE must be fixed|ema, got '$bal'" >&2; exit 2 ;;
  esac
}
opt_args() {  # options shared by E1 / E2 (only when set: unset = the old command line)
  local o=""
  if [ -n "${KD_DRAFT_SOURCE:-}" ]; then
    case "$KD_DRAFT_SOURCE" in tau0|human_mix) ;; *) echo "KD_DRAFT_SOURCE must be tau0|human_mix" >&2; exit 2 ;; esac
    if [ "$KD_DRAFT_SOURCE" = "human_mix" ] && [ ! -f /home/external-user/ssd/yongjae_refiner/human/e2e_train_trainlogs.npz ]; then
      echo "human_mix needs /home/external-user/ssd/yongjae_refiner/human/e2e_train_trainlogs.npz (extract_human.py)" >&2; exit 2
    fi
    o="$o agent.config.kd_draft_source=$KD_DRAFT_SOURCE"
  fi
  if [ -n "${REF_HUMAN_ONLY_UNTIL:-}" ]; then
    awk -v r="$REF_HUMAN_ONLY_UNTIL" 'BEGIN { exit !(r ~ /^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ && r + 0 >= 0) }' || { echo "REF_HUMAN_ONLY_UNTIL must be >= 0" >&2; exit 2; }
    [ -f /home/external-user/ssd/yongjae_refiner/human/e2e_train_trainlogs.npz ] || {
      echo "REF_HUMAN_ONLY_UNTIL needs /home/external-user/ssd/yongjae_refiner/human/e2e_train_trainlogs.npz" >&2; exit 2; }
    o="$o agent.config.ref_human_only_until=$REF_HUMAN_ONLY_UNTIL"
  fi
  if [ -n "${REF_BEV_GRAD_SCALE:-}" ]; then
    awk -v r="$REF_BEV_GRAD_SCALE" 'BEGIN { exit !(r ~ /^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ && r + 0 >= 0) }' || { echo "REF_BEV_GRAD_SCALE must be >= 0" >&2; exit 2; }
    o="$o agent.config.ref_bev_grad_scale=$REF_BEV_GRAD_SCALE"
  fi
  [ -n "${GRAD_SHARE_EVERY:-}" ] && o="$o agent.config.grad_share_every=$GRAD_SHARE_EVERY"
  echo "$o"
}
opt_tag() {  # $1 = arm; experiment-name suffix for non-default options ('' for the defaults -> old names)
  local t=""
  [ "${1:-E2}" = "E2" ] && [ "${KD_BALANCE:-fixed}" = "ema" ] && t="${t}_ema_r${KD_RATIO:-1.0}_s${KD_START_EPOCH:-ramp}"
  [ "${1:-E2}" = "E2" ] && [ "${KD_BALANCE:-fixed}" = "ema" ] && [ -n "${KD_RATIO_RAMP:-}" ] && t="${t}_rr${KD_RATIO_RAMP}"
  [ "${1:-E2}" = "E2" ] && [ "${KD_BALANCE:-fixed}" = "ema" ] && [ -n "${KD_WEIGHT_MAX:-}" ] && t="${t}_wmax${KD_WEIGHT_MAX}"
  [ "${KD_DRAFT_SOURCE:-tau0}" = "human_mix" ] && t="${t}_hmix"
  [ -n "${REF_HUMAN_ONLY_UNTIL:-}" ] && t="${t}_hwu${REF_HUMAN_ONLY_UNTIL}"
  [ -n "${REF_BEV_GRAD_SCALE:-}" ] && t="${t}_bg${REF_BEV_GRAD_SCALE}"
  echo "$t"
}

# E0 reference: 2 GPUs x batch 4 x acc 16 = 128; grad-balancer counters count MICRO-batches per rank
# (warm-up 10600 = 1 epoch, interval 200 = 12.5 optimiser steps).  Other layouts keep the same cadence in optimiser
# steps by scaling them with the per-rank micro-batches per optimiser step (acc):  counter x acc / 16.
counters() {  # $1 = accumulate
  local acc=$1
  echo "agent.config.grad_balance_warmup_iters=$((10600 * acc / 16)) agent.config.grad_balance_interval=$((200 * acc / 16)) agent.config.grad_norm_log_interval=$((200 * acc / 16))"
}

case "$CMD" in
gate)
  for r in stageT4_T_fold0_seed0 stageT4_M_fold0_seed0; do
    "$PY" tools/refiner/liveness.py --run "$DATA/runs/$r" --eval eval_train_fold0 --min-liveness 0.01 \
        --out "$DATA/stageE/liveness_$r.json"
  done ;;
snapshot)
  "$PY" tools/refiner/stageE_prep.py snapshot ;;
pilot)    # ONE epoch of 300 micro-batches at the run's epoch-0 lr (same W); E2 computes + logs KD with lambda = 0
  ARM="$1"; GPU="$2"; W="${3:?pilot needs W (warmup_epochs of the planned run)}"
  EXTRA=()
  if [ "$ARM" = "E2" ]; then
    SP="${4:?E2 pilot needs kd_space raw|tanh|decoded}"
    EXTRA+=("agent.config.kd_teacher_runs=$KD_RUNS" "agent.config.kd_lambda=0.0" "agent.config.kd_space=$SP")
  fi
  if [ -e "work_dirs/stageE_pilot_$ARM" ]; then   # stageE_steps.jsonl is appended: a rerun would mix records
    echo "work_dirs/stageE_pilot_$ARM exists; move it away before re-running the pilot" >&2; exit 2
  fi
  CUDA_VISIBLE_DEVICES="$GPU" BATCH_SIZE=4 ACCUMULATE=32 MAX_EPOCHS=1 WORKERS=6 EXPERIMENT="stageE_pilot_$ARM" \
  bash scripts/training/train_para_ssr_interaction.sh agent.config.refiner_mode="$ARM" agent.config.warmup_epochs="$W" \
    $(counters 32) trainer.params.strategy=auto trainer.params.limit_val_batches=0 \
    trainer.params.limit_train_batches=300 wandb.enable=false ${EXTRA[@]+"${EXTRA[@]}"} ;;
lambda)
  "$PY" tools/refiner/stageE_prep.py pilot --steps work_dirs/stageE_pilot_E2/stageE_steps.jsonl \
      --other work_dirs/stageE_pilot_E1/stageE_steps.jsonl ;;
train1)   # plan (stageE_impl_plan.md §9): one GPU per arm, batch 4 x acc 32 = 128, N epochs, KD ramp epochs 1-3
  ARM="$1"; GPU="$2"; N="${3:?N}"; W="${4:?W (see the note above: W=1 means no warm-up)}"
  EXTRA=()
  if [ "$ARM" = "E2" ]; then E2X="$(e2_args "${5:-}" "${6:-}")"; read -r -a EXTRA <<< "$E2X"; fi
  CUDA_VISIBLE_DEVICES="$GPU" BATCH_SIZE=4 ACCUMULATE=32 MAX_EPOCHS="$N" WORKERS=6 EXPERIMENT="stageE_${ARM}_N${N}" \
  bash scripts/training/train_para_ssr_interaction.sh agent.config.refiner_mode="$ARM" agent.config.warmup_epochs="$W" \
    "agent.config.kd_ramp=[1.0,3.0]" $(counters 32) trainer.params.strategy=auto trainer.params.limit_val_batches=0 \
    ${EXTRA[@]+"${EXTRA[@]}"} ;;
train4)   # REVISION 1: E0's recipe unchanged (30 epochs, warm-up 3, 128 = 4 GPUs x 4 x acc 8), KD ramp epochs 5-9
  ARM="$1"; GPUS="$2"
  EXTRA=()
  if [ "$ARM" = "E2" ]; then E2X="$(e2_args "${3:-}" "${4:-}")"; read -r -a EXTRA <<< "$E2X"; fi
  OX="$(opt_args)"; [ -n "$OX" ] && { read -r -a OXA <<< "$OX"; EXTRA+=("${OXA[@]}"); }
  TAG="$(opt_tag "$ARM")"
  CUDA_VISIBLE_DEVICES="$GPUS" BATCH_SIZE=4 ACCUMULATE=8 MAX_EPOCHS=30 WORKERS=6 EXPERIMENT="stageE_${ARM}_30ep${TAG}" \
  bash scripts/training/train_para_ssr_interaction.sh agent.config.refiner_mode="$ARM" agent.config.warmup_epochs=3 \
    "agent.config.kd_ramp=[5.0,10.0]" $(counters 8) trainer.params.limit_val_batches=0 \
    ${EXTRA[@]+"${EXTRA[@]}"} ;;
pilot150) # 1 GPU, 150 micro-batches of the train4 per-rank layout (batch 4 x acc 8, W = 3, 1 epoch), E2 with the
          # post-pilot options (defaults here: KD_BALANCE=ema, KD_START_EPOCH=0 so KD is on from step 0,
          # GRAD_SHARE_EVERY=10) -> EMA weight / weighted KD vs surrogate / BEV grad shares in stageE_steps.jsonl
  GPU="${1:?pilot150 needs <gpu>}"; SP="${2:?pilot150 needs kd_space raw|tanh|decoded}"
  export KD_BALANCE="${KD_BALANCE:-ema}" KD_START_EPOCH="${KD_START_EPOCH:-0}" GRAD_SHARE_EVERY="${GRAD_SHARE_EVERY:-10}"
  if [ "$KD_BALANCE" = "ema" ]; then E2X="$(e2_args - "$SP")"; else E2X="$(e2_args "${3:-}" "$SP")"; fi
  read -r -a EXTRA <<< "$E2X"
  OX="$(opt_args)"; [ -n "$OX" ] && { read -r -a OXA <<< "$OX"; EXTRA+=("${OXA[@]}"); }
  EXP="stageE_pilot150_E2$(opt_tag E2)"
  if [ -e "work_dirs/$EXP" ]; then echo "work_dirs/$EXP exists; move it away before re-running" >&2; exit 2; fi
  CUDA_VISIBLE_DEVICES="$GPU" BATCH_SIZE=4 ACCUMULATE=8 MAX_EPOCHS=1 WORKERS=6 EXPERIMENT="$EXP" \
  bash scripts/training/train_para_ssr_interaction.sh agent.config.refiner_mode=E2 agent.config.warmup_epochs=3 \
    $(counters 8) trainer.params.strategy=auto trainer.params.limit_val_batches=0 \
    trainer.params.limit_train_batches=150 wandb.enable=false "${EXTRA[@]}"
  "$PY" tools/refiner/stageE_prep.py pilot --steps "work_dirs/$EXP/stageE_steps.jsonl" --last 100 --skip 0 ;;
eval)     # navtest PDMS; trajectory = tau_final (final) or tau0; teachers are never built at eval
  ARM="$1"; CKPT="$2"; TRAJ="$3"; GPU="$4"; NAME="$5"
  RA=(); [ "$ARM" != "off" ] && RA+=("agent.config.refiner_mode=$ARM" "agent.config.ref_eval_traj=$TRAJ")
  CUDA_VISIBLE_DEVICES="$GPU" NAVSIM_DOWNLOAD=/home/external-user/navsim/download EVAL_EXPERIMENT="eval/$NAME" \
  bash scripts/evaluation/eval_para_ssr.sh "$CKPT" agent.config.use_task_interaction=true \
    agent.config.use_det_motion_head=true agent.config.use_map_head=true ${RA[@]+"${RA[@]}"} ;;
compare)
  c() { ls -t "work_dirs/eval/$1"/*.csv | head -1; }
  ARGS=(--arm "E0=work_dirs/eval/para_ssr_interaction_final/2026.09.17.00.09.41.csv")
  for n in "$@"; do ARGS+=(--arm "$n=$(c "stageE_$n")"); done
  "$PY" tools/refiner/stageE_compare.py "${ARGS[@]}" --contrast E2-E0 --contrast E2-E2_tau0 \
      $([ -d work_dirs/eval/stageE_E1 ] && echo "--contrast E2-E1 --contrast E1-E1_tau0") \
      --out report/refiner_T/stageE_compare.json ;;
*) echo "unknown sub-command $CMD" >&2; exit 2 ;;
esac
