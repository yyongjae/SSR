#!/usr/bin/env bash
# Wait until the given GPUs carry no compute process for STABLE consecutive checks
# (INTERVAL seconds apart), then launch scripts/training/train_fact.sh on them.
#
#   GPUS=2,3 ACCUMULATE=16 EXPERIMENT=fact_v3_s0 SEED=0 bash tools/fact/wait_and_train.sh
#
# Optional: WAIT_FILE=<path> also waits until that file exists (e.g. a data-transfer
# marker).  Progress goes to work_dirs/logs/wait_<EXPERIMENT>.log; the training log to
# work_dirs/logs/<EXPERIMENT>.log.  Stop the wait with: pkill -f "[w]ait_and_train.sh"
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPUS="${GPUS:-2,3}"
INTERVAL="${INTERVAL:-60}"
STABLE="${STABLE:-5}"
EXPERIMENT="${EXPERIMENT:-fact_v3}"
SEED="${SEED:-0}"
WAIT_FILE="${WAIT_FILE:-}"
mkdir -p "${REPO}/work_dirs/logs"
LOG="${REPO}/work_dirs/logs/wait_${EXPERIMENT}.log"

free_count=0
while true; do
  busy=""
  for g in ${GPUS//,/ }; do
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g")
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid" || true)
    [[ "$n" -gt 0 ]] && busy="${busy} gpu${g}(${n})"
  done
  if [[ -n "$WAIT_FILE" && ! -e "$WAIT_FILE" ]]; then
    busy="${busy} waiting-for:${WAIT_FILE}"
  fi
  if [[ -z "$busy" ]]; then
    free_count=$((free_count + 1))
  else
    free_count=0
  fi
  echo "[$(date '+%F %T')] busy:${busy:- none}  free_streak=${free_count}/${STABLE}" >> "$LOG"
  if [[ "$free_count" -ge "$STABLE" ]]; then
    break
  fi
  sleep "$INTERVAL"
done

echo "[$(date '+%F %T')] launching ${EXPERIMENT} on GPUs ${GPUS}" >> "$LOG"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" EXPERIMENT="$EXPERIMENT" \
  bash scripts/training/train_fact.sh +seed="$SEED" "$@" > "${REPO}/work_dirs/logs/${EXPERIMENT}.log" 2>&1
echo "[$(date '+%F %T')] training exited rc=$?" >> "$LOG"
