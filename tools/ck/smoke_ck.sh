#!/bin/bash
# CK Phase 1 end-to-end smoke (contract pipeline.smoke_ck): <= 200 tokens, GPU 3 only, every stage of run_phase1.sh
# in miniature under CK_DATA/smoke (CK_DATA_ROOT) so nothing touches the real outputs.
#   bash tools/ck/smoke_ck.sh [from_step]          steps: dump pack label lead teachers kd label_kd student infer
#                                                  label_corr eval report   (default: dump)
# Env: SMOKE_GPU (3), STEPS (30), N_TRAIN 100 / N_VAL 50 / N_TEST 50, LABEL_WORKERS 16, FRESH=1 wipes smoke train/
#      kd/infer/eval/labels(kd_corr, corr_*) first.
# Prints cand/s per arm (train_log.jsonl) -> used to set --epochs of the real runs (teacher ~2 h, student ~4 h).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"   # repo root of this worktree
PY=/venv/ssr/bin/python
REAL=/home/external-user/ssd/yongjae_refiner/ck
export CK_DATA_ROOT=${CK_DATA_ROOT:-$REAL/smoke}
S=$CK_DATA_ROOT
G=${SMOKE_GPU:-3}
STEPS=${STEPS:-30}
N_TRAIN=${N_TRAIN:-100}; N_VAL=${N_VAL:-50}; N_TEST=${N_TEST:-50}
LW=${LABEL_WORKERS:-16}
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $REPO
mkdir -p $S/runs
T0=$(date +%s)
log() { echo "[smoke $(date '+%T') +$(( $(date +%s) - T0 ))s] $*"; }
die() { log "FAILED: $*"; exit 1; }
case "$G" in 0|1|2|3) ;; *) die "SMOKE_GPU must be 0-3";; esac
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G | tr -d ' ')
[ "${free:-0}" -gt 16000 ] || die "GPU $G has only ${free} MiB free"
gpu() { CUDA_VISIBLE_DEVICES=$G "$@"; }
cpu() { CUDA_VISIBLE_DEVICES= "$@"; }
run() { local name=$1; shift; log "$name: $*"; local t=$(date +%s)
  "$@" > $S/runs/smoke_$name.log 2>&1 || { tail -30 $S/runs/smoke_$name.log; die "$name (log $S/runs/smoke_$name.log)"; }
  log "$name ok ($(( $(date +%s) - t ))s)"; }

STEPS_ALL="dump pack label lead teachers kd label_kd student infer label_corr eval report"
start=${1:-dump}; go=0
want() { [ $go = 1 ] && return 0; [ "$1" = "$start" ] && { go=1; return 0; }; return 1; }

if [ "${FRESH:-0}" = 1 ]; then
  rm -rf $S/train $S/kd_targets $S/infer $S/eval $S/labels/*/kd_corr $S/labels/*/corr_*
fi

if want dump; then
  for sp in "navtest $N_TEST" "navtrain_val $N_VAL" "navtrain_train $N_TRAIN"; do set -- $sp
    run dump_$1 gpu $PY tools/ck/data/dump_v2.py --split $1 --limit $2 --batch-size 8 --workers 8
  done
fi
if want pack; then
  for sp in navtrain_train navtrain_val navtest; do
    run pack_$sp cpu $PY tools/ck/data/pack_v2.py --split $sp --tokens dumped
  done
fi
if want label; then
  for sp in navtrain_train navtrain_val navtest; do
    if [ -f $S/labels/$sp/cand/labels.npy ]; then log "label_$sp exists (skip)"; continue; fi
    run label_$sp cpu $PY tools/ck/data/label_cands.py --split $sp --name cand --workers $LW --chunk 8
  done
fi
if want lead; then   # the smoke reads the real lead flags (token-keyed; extra rows are harmless)
  mkdir -p $S/lead
  for sp in navtrain_train navtrain_val navtest; do
    [ -f $REAL/lead/$sp.parquet ] && ln -sfn $REAL/lead/$sp.parquet $S/lead/$sp.parquet && log "lead $sp linked"
  done
fi
TRAIN_COMMON="--max-steps $STEPS --epochs 4 --workers 4 --log-every 5 --ckpt-every 10 --train-eval-rows 50 --val-limit 50 --resume"
if want teachers; then
  run train_T gpu $PY tools/ck/train_ck.py --arm T --run smT $TRAIN_COMMON
  run train_M gpu $PY tools/ck/train_ck.py --arm M --run smM --val-split none $TRAIN_COMMON
fi
if want kd; then
  run kd gpu $PY tools/ck/kd_targets.py --det-run $S/train/smT --map-run $S/train/smM --tag smoke --workers 4
fi
if want label_kd; then
  run label_kd cpu $PY tools/ck/data/label_cands.py --split navtrain_train --name kd_corr \
    --traj $S/kd_targets/smoke/kd_corr_traj.npy --workers $LW --chunk 8
fi
if want student; then
  run train_S gpu $PY tools/ck/train_ck.py --arm S --run smS --kd on --kd-dir $S/kd_targets/smoke --corr-aug on $TRAIN_COMMON
fi
if want infer; then
  for sp in navtrain_val navtest; do
    run infer_S_$sp gpu $PY tools/ck/infer_ck.py --run $S/train/smS --split $sp --workers 4
  done
  run infer_T_val gpu $PY tools/ck/infer_ck.py --run $S/train/smT --split navtrain_val --workers 4
  run infer_M_test gpu $PY tools/ck/infer_ck.py --run $S/train/smM --split navtest --workers 4
fi
if want label_corr; then
  for sp in navtrain_val navtest; do
    run label_corr_$sp cpu $PY tools/ck/data/label_cands.py --split $sp --name corr_smS \
      --traj $S/infer/smS/$sp/corr_traj.npy --workers $LW --chunk 8
  done
fi
if want eval; then
  run eval_S_val cpu $PY tools/ck/eval_ck.py --run smS --split navtrain_val --bootstrap 200
  run eval_S_test cpu $PY tools/ck/eval_ck.py --run smS --split navtest --bootstrap 200 \
    --fix-from $S/eval/smS/navtrain_val/metrics.json
  run eval_T_val cpu $PY tools/ck/eval_ck.py --run smT --split navtrain_val --labels-corr none --bootstrap 200
fi
if want report; then
  $PY - <<EOF
import json, pathlib
S = pathlib.Path("$S")
rep = {}
for r in ("smT", "smM", "smS"):
    p = S / "train" / r / "train_log.jsonl"
    if not p.is_file():
        continue
    recs = [json.loads(l) for l in p.read_text().splitlines() if '"step"' in l]
    steps = [x for x in recs if x.get("kind") == "step"]
    late = steps[len(steps) // 2:] or steps
    cps = [x["cand_per_s"] for x in late]
    rep[r] = dict(steps=len(steps), cand_per_s_median=sorted(cps)[len(cps) // 2] if cps else None,
                  gpu_mem_gb=max([x.get("gpu_mem_gb", 0) for x in steps] or [0]),
                  loss_last=steps[-1]["loss"] if steps else None,
                  nonfinite=sum(x.get("skipped", 0) for x in steps))
    # epoch time estimate on the full train split: 85,109 tokens x 16 candidates (x2 for the student corr_aug)
    if cps:
        mult = 2 if r == "smS" else 1
        rep[r]["est_min_per_epoch_full"] = round(85109 * 16 * mult / rep[r]["cand_per_s_median"] / 60, 1)
for r, sp in (("smS", "navtrain_val"), ("smS", "navtest")):
    m = S / "eval" / r / sp / "metrics.json"
    if m.is_file():
        d = json.loads(m.read_text())
        rep[f"eval_{r}_{sp}"] = {"n_eval": d["n_eval"], "v2_mismatch": d.get("v2_mismatch"),
                                 "v2_check": d.get("v2_check"), "best_beta": d.get("best_beta")}
(S / "smoke_report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
EOF
fi
log "SMOKE DONE in $(( $(date +%s) - T0 ))s"
