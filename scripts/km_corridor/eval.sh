#!/usr/bin/env bash
# km 10/16: corridor selection, official scoring.  CAND=1 (km 10/19, C runs): navtest = forward choice (cand_official) + v2 (anchor-label comparison only).  usage: eval.sh <val|navtest> <run name> <ckpt> <gpu> <tag> [planes] [route 0|1] [fwd rows]
#   val     : navtrain_val ALL 18,179 tokens -> dump (f32 trajectories) -> corridor verdicts -> every pre-declared rule, official PDMS (select_0.json)
#             (a c64 field dump of all 18,179 val tokens is ~36 GB: 49 x 200 x 200 u8 per token)
#   navtest : 12,146 tokens -> dump -> verdicts -> RULE (default pred_multiply_t0.25, fixed on val) + the same model's v2
#             selection -> official PDMS and EPDMS (tools/corridor/score_npz.py: its inputs are checked before the dump,
#             a failed scoring exits 1)
#   refiner v6c run (agent.config.v6c of the archived run config; val or navtest, CAND or not): one dump --v6 --traj32
#             --cand --no-fields; the submission is rule v6 (R-J's switch ? tau_v6 : tau_base), never 'model' (= tau_base).
#             val: val_score.py (v1 PDMS of base / v6 / out) + val_epdms.py (v2.2 EPDMS; C0_DUMP=<dump> adds the C0 pick)
#             navtest: sel_{v6,v6base,v6A,v2}.npz + val_score.py --split navtest + val_epdms.py --split navtest (v2 added)
#   planes / fwd rows default to the archived run config (corridor_occ_planes; 200 rows for corridor_format c64, else 100);
#   args 6 / 8 override them.  EVAL_DRY=1 prints every dump / select / score command instead of running it.
set -uo pipefail
if [ -f "$(dirname "$0")/env.sh" ]; then source "$(dirname "$0")/env.sh"; else source "$(dirname "$0")/env.example.sh"; fi; cd $REPO
SPL=$1; RUN=$2; CKPT=$3; GPU=$4; TAG=$5; RT=${7:-0}
R=${KM_RUN_ROOT:-$REPO/work_dirs/km_runs}/corr30/$RUN; OUT=$R/eval_$TAG
# the archived run config: v6c, corridor_occ_planes, fwd rows (c64: 200), cand_select_eps / lam (no file: the old defaults)
CFG=$($PY - "$R/train/code/hydra/config.yaml" <<'PYCFG'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print("0 8 100 0.05 0.05")
    sys.exit(0)
from omegaconf import OmegaConf
c = OmegaConf.load(p).agent.config
g = lambda k, d: d if c.get(k) is None else c.get(k)
print(int(bool(g("v6c", False))), int(g("corridor_occ_planes", 8)), 200 if c.get("corridor_format") == "c64" else 100,
      float(g("cand_select_eps", 0.05)), float(g("cand_select_lam", 0.05)))
PYCFG
)
read -r V6C PL0 FR0 EPS0 LAM0 <<< "$CFG"
[ -n "${LAM0:-}" ] || { echo "eval.sh: cannot read $R/train/code/hydra/config.yaml" >&2; exit 1; }
P=${6:-$PL0}; FR=${8:-$FR0}
export CUDA_VISIBLE_DEVICES=$GPU
ROUTE=(); [ "$RT" = 1 ] && ROUTE=(--route)
[ "${SIGMA:-0}" = 1 ] && ROUTE+=(--sigma)       # km 10/18: v32b runs (sigma head) -> csig.npy
run() { if [ "${EVAL_DRY:-0}" = 1 ]; then printf '%q ' "$@"; echo; else "$@"; fi; }
if [ "$V6C" = 1 ]; then          # refiner v6c (A49): score the submitted trajectory (rule v6), not tau_base (rule model)
  S=navtest; [ "$SPL" = val ] && S=navtrain_val
  if [ "$S" = navtest ]; then run $PY tools/corridor/score_npz.py --check --split navtest || exit 1; fi
  run $PY tools/corridor/dump_corr_model.py --run-dir $R --ckpt $CKPT --split $S --out $OUT --workers 4 --planes $P --fwd-rows $FR \
    --traj32 --cand --no-fields --v6 --cand-eps ${CAND_EPS:-$EPS0} --cand-lam ${CAND_LAM:-$LAM0} || exit 1
  if [ "$S" = navtrain_val ]; then
    run nice $PY tools/refiner_v6c/val_score.py --dump $OUT --split navtrain_val --n 0 || exit 1
    run nice $PY tools/refiner_v6c/val_epdms.py all --dump $OUT --out $OUT/epdms_val ${C0_DUMP:+--c0-dump $C0_DUMP} || exit 1
  else
    for r in v6 v6base v6A v2 model; do     # model (= traj[msel]) only for the tau_base check below
      run $PY tools/corridor/select_npz.py --dump $OUT --split navtest --rule $r --out $OUT/sel_$r.npz || exit 1
    done
    run $PY -c '
import sys, numpy as np
z = [np.load(f"{sys.argv[1]}/sel_{r}.npz", allow_pickle=True) for r in ("v6base", "model")]
m = dict(zip(z[1]["tokens"].astype(str), z[1]["trajectory"]))
d = max([float(np.abs(x - m[t]).max()) for t, x in zip(z[0]["tokens"].astype(str), z[0]["trajectory"]) if t in m] or [0.0])
print(f"max |sel_v6base - sel_model| = {d:.3g}" + ("  WARNING: tau_base != traj[msel]" if d > 1e-5 else ""))' $OUT || exit 1
    run nice $PY tools/refiner_v6c/val_score.py --dump $OUT --split navtest --n 0 || exit 1
    run nice $PY tools/refiner_v6c/val_epdms.py all --split navtest --dump $OUT --out $OUT/epdms_navtest \
      --npz v2=$OUT/sel_v2.npz || exit 1
  fi
  exit 0
fi
if [ "$SPL" = val ]; then
  run $PY tools/corridor/dump_corr_model.py --run-dir $R --ckpt $CKPT --split navtrain_val --out $OUT --workers 4 --planes $P --fwd-rows $FR "${ROUTE[@]}" --traj32 || exit 1
  run $PY tools/corridor/corr_verdict.py --dump $OUT --split navtrain_val || exit 1
  run nice $PY tools/corridor/corr_select.py --dump $OUT --split navtrain_val --n 0 --workers ${WORKERS:-8}
elif [ "${CAND:-0}" = 1 ]; then   # km 10/19 (C): the forward's cand_official choice + the same model's v2 choice
  run $PY tools/corridor/score_npz.py --check --split navtest || exit 1
  run $PY tools/corridor/dump_corr_model.py --run-dir $R --ckpt $CKPT --split navtest --out $OUT --workers 4 --planes $P --fwd-rows $FR \
    --traj32 --cand --no-fields --cand-eps ${CAND_EPS:-0.05} --cand-lam ${CAND_LAM:-0.05} || exit 1   # km 10/20: C v3.1 (train-tuned)
  for r in model v2; do
    run $PY tools/corridor/select_npz.py --dump $OUT --split navtest --rule $r --out $OUT/sel_$r.npz || exit 1
    run nice $PY tools/corridor/score_npz.py --npz $OUT/sel_$r.npz --split navtest --out $OUT --name ${RUN}_${TAG}_${r} \
      ${WORKERS:+--workers $WORKERS} || exit 1
  done
else
  run $PY tools/corridor/score_npz.py --check --split navtest || exit 1
  run $PY tools/corridor/dump_corr_model.py --run-dir $R --ckpt $CKPT --split navtest --out $OUT --workers 4 --planes $P --fwd-rows $FR "${ROUTE[@]}" --traj32 || exit 1
  run $PY tools/corridor/corr_verdict.py --dump $OUT --split navtest || exit 1
  for r in ${RULE:-pred_multiply_t0.25} v2 model; do     # model = the forward's own corridor_select_rule (km 10/17)
    run $PY tools/corridor/select_npz.py --dump $OUT --split navtest --rule $r --out $OUT/sel_$r.npz || exit 1
    run nice $PY tools/corridor/score_npz.py --npz $OUT/sel_$r.npz --split navtest --out $OUT --name ${RUN}_${TAG}_${r} \
      ${WORKERS:+--workers $WORKERS} || exit 1
  done
fi
