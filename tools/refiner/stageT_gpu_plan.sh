#!/usr/bin/env bash
# Stage-T GPU plan: queue every (arm, fold, seed) run of one phase over the assigned GPUs.  NOT run by the workflow.
#
#   usage: tools/refiner/stageT_gpu_plan.sh <phase: crossfit|final> "<gpu indices>" [extra train_refiner args...]
#     crossfit : arms {T, none} x seeds {0,1,2} x folds {0..4} = 30 runs; each trains on the other 4 folds of the TRAIN
#                split and evaluates out-of-fold on its own fold (eval_train_fold<k>/, theta sweep)
#     final    : arms {T, none} x seeds {0,1,2}, fold -1 = 6 runs; trains on all folds, evaluates on DEV (eval_dev/)
#   e.g.  nohup tools/refiner/stageT_gpu_plan.sh crossfit "2 3" > <data>/runs/logs/plan_crossfit.log 2>&1 &
#         TAG=w_col2 tools/refiner/stageT_gpu_plan.sh crossfit "2 3" --w col=2,dac=1,prog=2,cmf=0.1,mod=0.1
#
# Order: runs are interleaved (seed, fold, arm) so that both arms progress together; GPU i takes runs i, i+n, i+2n, ...
# and runs them one after another.  Resumable: a finished run (DONE) is skipped by train_refiner.py, an interrupted one
# resumes from ckpt_last.pt; its evaluation is redone (scoring shards resume).  Evaluations of concurrent runs are
# serialised by stageT_gpu_commands.sh (flock), each scoring with 4 CPU workers.
# Guard: the packed split the phase needs must be complete (<data>/packed/<split>/follow_status.json, written by
# tools/refiner/pack_follow.py); FORCE=1 skips the guard.
# After the phases:
#   python tools/refiner/stageT_decision.py select --wtags stageT --out report/refiner_T/crossfit_selection.json
#   python tools/refiner/stageT_decision.py compare --selection report/refiner_T/crossfit_selection.json \
#       --out report/refiner_T/decision_dev.json
set -euo pipefail
PHASE=${1:?phase crossfit|final}; read -r -a GPUS <<< "${2:?gpu list, e.g. \"2 3\"}"; shift 2
REPO=/home/external-user/yongjae/SSR
PY=/home/external-user/miniconda3/envs/ssr/bin/python
DATA=/home/external-user/ssd/yongjae_refiner
TAG=${TAG:-stageT}
export TAG
cd "$REPO"
mkdir -p "$DATA/runs/logs"

need_split() {   # $1 = split whose pack must be complete
    if [ "${FORCE:-0}" = "1" ]; then return 0; fi
    "$PY" - "$DATA/packed/$1/follow_status.json" <<'EOF'
import json, sys
p = sys.argv[1]
try:
    st = json.load(open(p))
except FileNotFoundError:
    sys.exit(f"{p} missing: run tools/refiner/pack_follow.py first (or FORCE=1)")
if not st.get("complete"):
    sys.exit(f"{p}: pack not complete (ready {st.get('ready')}/{st.get('n_usable')}, parts {st.get('parts')}); FORCE=1 to override")
EOF
}

JOBS=()
case "$PHASE" in
    crossfit)
        need_split train
        for SEED in 0 1 2; do for FOLD in 0 1 2 3 4; do for ARM in T none; do JOBS+=("$ARM $FOLD $SEED"); done; done; done ;;
    final)
        need_split train; need_split dev
        for SEED in 0 1 2; do for ARM in T none; do JOBS+=("$ARM -1 $SEED"); done; done ;;
    *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac

NG=${#GPUS[@]}
echo "[plan] $(date) phase=$PHASE tag=$TAG gpus=${GPUS[*]} runs=${#JOBS[@]} extra=$*"
for i in "${!GPUS[@]}"; do
    (
        for j in "${!JOBS[@]}"; do
            if (( j % NG == i )); then
                read -r ARM FOLD SEED <<< "${JOBS[$j]}"
                echo "[plan gpu ${GPUS[$i]}] $(date) start $TAG $ARM fold $FOLD seed $SEED"
                if tools/refiner/stageT_gpu_commands.sh "${GPUS[$i]}" "$ARM" "$FOLD" "$SEED" "$@"; then
                    echo "[plan gpu ${GPUS[$i]}] $(date) done  $TAG $ARM fold $FOLD seed $SEED"
                else
                    echo "[plan gpu ${GPUS[$i]}] $(date) FAILED $TAG $ARM fold $FOLD seed $SEED (see $DATA/runs/logs/)"
                fi
            fi
        done
    ) &
done
wait
echo "[plan] $(date) phase=$PHASE finished"
