#!/usr/bin/env bash
# Stage-T refiner: GPU training + dev evaluation for one (arm, fold, seed).  NOT run by the implementation workflow.
#
#   usage: tools/refiner/stageT_gpu_commands.sh <gpu> <arm: T|none|M|TM> <fold: -1|0..4> <seed> [extra train args...]
#   e.g.   tools/refiner/stageT_gpu_commands.sh 3 T -1 0          # final dev model, arm T, seed 0 on cuda:3
#          tools/refiner/stageT_gpu_commands.sh 3 none 2 1        # cross-fitting fold 2 (OOF eval on train fold 2)
#
# Prerequisites: packed splits <data>/packed/{train,dev} with all parts (tools/refiner/pack_when_ready.sh, or
# python -m navsim.agents.para_ssr.refiner.data --split train --workers 4), loss weights chosen by cross-fitting
# (--w ...).  train_refiner.py uses the real surrogate (surrogate.surrogate_terms) by default and fails if it is missing.
# --gpu is the CUDA index as seen by the process (do not also restrict CUDA_VISIBLE_DEVICES, or pass the remapped index).
# Scoring inside eval_refiner uses 4 CPU workers: run the evaluations of different runs one after another.
# Logs: <data>/runs/logs/<run>.{train,eval}.log ; the run directory is <data>/runs/<tag>_<arm>_fold<k>_seed<s>/.
# Run 4 (AMENDMENT 6): arms M / TM and --token-subset / --dev-token-subset (train_refiner.py) need a NEW TAG (not
# stageT / stageT2 / stageT3).  The OOF train eval passes the run's train subset (--token-subset, re-checked against
# config.json by eval_refiner.py); DEV_EVAL=1 also evaluates dev with the --dev-token-subset file (default off: in run 4
# dev is evaluated separately, after the liveness gate on the OOF fold).
set -euo pipefail
GPU=${1:?gpu}; ARM=${2:?arm}; FOLD=${3:?fold}; SEED=${4:?seed}; shift 4
REPO=/home/external-user/yongjae/SSR
PY=/home/external-user/miniconda3/envs/ssr/bin/python
DATA=/home/external-user/ssd/yongjae_refiner
TAG=${TAG:-stageT}
RUN=${TAG}_${ARM}_fold${FOLD}_seed${SEED}
# run 2 (dead-zone options) must not reuse the run-1 tag: stageT_* directories are run-1 evidence
for x in "$@"; do
  case "$x" in --lon-st-slope*|--w-zdead*)
    if [ "$TAG" = "stageT" ]; then echo "refused: $x with TAG=stageT (run-1 dirs); set TAG=<new tag>" >&2; exit 2; fi;;
  esac
  # run 3 (TTC term, AMENDMENT 4) must not reuse the run-1 / run-2 tags either
  case "$x" in --m-ttc*|*ttc=*)
    case "$TAG" in stageT|stageT2)
      echo "refused: $x with TAG=$TAG (run-1 / run-2 dirs); set TAG=<new tag, e.g. stageT3>" >&2; exit 2;;
    esac;;
  esac
  # run 4 (AMENDMENT 6): token subsets must not reuse the run-1 / 2 / 3 tags
  case "$x" in --token-subset*|--dev-token-subset*|--resmap-root*)
    case "$TAG" in stageT|stageT2|stageT3)
      echo "refused: $x with TAG=$TAG (run-1 / 2 / 3 dirs); set TAG=<new tag, e.g. stageT4>" >&2; exit 2;;
    esac;;
  esac
done
case "$ARM" in
  T|none) ;;
  M|TM)
    case "$TAG" in stageT|stageT2|stageT3)
      echo "refused: arm $ARM with TAG=$TAG (run-1 / 2 / 3 dirs); set TAG=<new tag, e.g. stageT4>" >&2; exit 2;;
    esac;;
  *) echo "refused: unknown arm $ARM (T|none|M|TM)" >&2; exit 2;;
esac
# subset files given to training are passed on to the evaluations
TRAIN_SUBSET=""; DEV_SUBSET=""; PREV=""
for x in "$@"; do
  case "$PREV" in --token-subset) TRAIN_SUBSET="$x";; --dev-token-subset) DEV_SUBSET="$x";; esac
  case "$x" in --token-subset=*) TRAIN_SUBSET="${x#*=}";; --dev-token-subset=*) DEV_SUBSET="${x#*=}";; esac
  PREV="$x"
done
SUB_TRAIN=(); SUB_DEV=()
if [ -n "$TRAIN_SUBSET" ]; then SUB_TRAIN=(--token-subset "$TRAIN_SUBSET"); fi
if [ -n "$DEV_SUBSET" ]; then SUB_DEV=(--token-subset "$DEV_SUBSET"); fi
mkdir -p "$DATA/runs/logs"
cd "$REPO"
echo "=== train $(date)" >> "$DATA/runs/logs/$RUN.train.log"
OMP_NUM_THREADS=2 nice -n 10 "$PY" tools/refiner/train_refiner.py --arm "$ARM" --fold "$FOLD" --seed "$SEED" \
    --gpu "$GPU" --workers 2 --tag "$TAG" "$@" >> "$DATA/runs/logs/$RUN.train.log" 2>&1
if [ "$FOLD" -lt 0 ]; then EVAL=(--split dev ${SUB_DEV[@]+"${SUB_DEV[@]}"}); else
  EVAL=(--split train --fold "$FOLD" ${SUB_TRAIN[@]+"${SUB_TRAIN[@]}"}); fi
# evaluations of concurrent runs are serialised (flock): each one scores with 4 CPU workers
echo "=== eval $(date)" >> "$DATA/runs/logs/$RUN.eval.log"
OMP_NUM_THREADS=1 nice -n 10 flock "$DATA/runs/.eval.lock" "$PY" tools/refiner/eval_refiner.py all \
    --run "$DATA/runs/$RUN" "${EVAL[@]}" --gpu "$GPU" --workers 4 --theta 0.5 --sweep --budget-ep 0.5 \
    >> "$DATA/runs/logs/$RUN.eval.log" 2>&1
# reduced single-run plan (DEV_EVAL=1): a fold-k run also evaluates dev (theta is chosen later on its OOF fold k)
if [ "$FOLD" -ge 0 ] && [ "${DEV_EVAL:-0}" = "1" ]; then
  echo "=== eval dev $(date)" >> "$DATA/runs/logs/$RUN.eval.log"
  OMP_NUM_THREADS=1 nice -n 10 flock "$DATA/runs/.eval.lock" "$PY" tools/refiner/eval_refiner.py all \
      --run "$DATA/runs/$RUN" --split dev ${SUB_DEV[@]+"${SUB_DEV[@]}"} --gpu "$GPU" --workers 4 --theta 0.5 \
      --sweep --budget-ep 0.5 >> "$DATA/runs/logs/$RUN.eval.log" 2>&1
fi
