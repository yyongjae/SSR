#!/usr/bin/env bash
# km 10/16: corridor selection, official scoring.  CAND=1 (km 10/19, C runs): navtest = forward choice (cand_official) + v2 (anchor-label comparison only).  usage: eval.sh <val|navtest> <run name> <ckpt> <gpu> <tag> [planes 8|40] [route 0|1]
#   val     : navtrain_val ALL 18,179 tokens -> dump -> corridor verdicts -> every pre-declared rule, official PDMS (select_0.json)
#   navtest : 12,146 tokens -> dump -> verdicts -> RULE (default pred_multiply_t0.25, fixed on val) + the same model's v2
#             selection -> official PDMS and EPDMS (score_npz.sh of the SSR-v2 checkout: V2_REPO)
set -uo pipefail
if [ -f "$(dirname "$0")/env.sh" ]; then source "$(dirname "$0")/env.sh"; else source "$(dirname "$0")/env.example.sh"; fi; cd $REPO
SPL=$1; RUN=$2; CKPT=$3; GPU=$4; TAG=$5; P=${6:-8}; RT=${7:-0}; FR=${8:-100}
R=$KM_ROOT/km_runs/corr30/$RUN; OUT=$R/eval_$TAG
export CUDA_VISIBLE_DEVICES=$GPU
ROUTE=(); [ "$RT" = 1 ] && ROUTE=(--route)
[ "${SIGMA:-0}" = 1 ] && ROUTE+=(--sigma)       # km 10/18: v32b runs (sigma head) -> csig.npy
if [ "$SPL" = val ]; then
  $PY tools/corridor/dump_corr_model.py --run-dir $R --ckpt $CKPT --split navtrain_val --out $OUT --workers 4 --planes $P --fwd-rows $FR "${ROUTE[@]}" || exit 1
  $PY tools/corridor/corr_verdict.py --dump $OUT --split navtrain_val || exit 1
  nice $PY tools/corridor/corr_select.py --dump $OUT --split navtrain_val --n 0 --workers ${WORKERS:-8}
elif [ "${CAND:-0}" = 1 ]; then   # km 10/19 (C): the forward's cand_official choice + the same model's v2 choice
  $PY tools/corridor/dump_corr_model.py --run-dir $R --ckpt $CKPT --split navtest --out $OUT --workers 4 --planes $P --fwd-rows $FR \
    --traj32 --cand --no-fields --cand-eps ${CAND_EPS:-0.05} --cand-lam ${CAND_LAM:-0.05} || exit 1   # km 10/20: C v3.1 (train-tuned)
  for r in model v2; do
    $PY tools/corridor/select_npz.py --dump $OUT --split navtest --rule $r --out $OUT/sel_$r.npz || exit 1
    (cd ${V2_REPO:-/home/external-user/kyungmin/SSR-v2} && WORKERS=${WORKERS:-8} bash data/logs/score_npz.sh ${RUN}_${TAG}_${r}_navtest $OUT/sel_$r.npz)
  done
else
  $PY tools/corridor/dump_corr_model.py --run-dir $R --ckpt $CKPT --split navtest --out $OUT --workers 4 --planes $P --fwd-rows $FR "${ROUTE[@]}" --traj32 || exit 1
  $PY tools/corridor/corr_verdict.py --dump $OUT --split navtest || exit 1
  for r in ${RULE:-pred_multiply_t0.25} v2 model; do     # model = the forward's own corridor_select_rule (km 10/17)
    $PY tools/corridor/select_npz.py --dump $OUT --split navtest --rule $r --out $OUT/sel_$r.npz || exit 1
    (cd ${V2_REPO:-/home/external-user/kyungmin/SSR-v2} && WORKERS=${WORKERS:-8} bash data/logs/score_npz.sh ${RUN}_${TAG}_${r}_navtest $OUT/sel_$r.npz)
  done
fi
