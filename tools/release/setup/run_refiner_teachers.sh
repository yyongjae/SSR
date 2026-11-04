#!/usr/bin/env bash
# The two refiner teachers: the exact recipe of ck2T10_64m_v7 and ck2M10_64m_v7, except --r34-evals 0 (no planner candidates in the
# monitoring evals, which do not feed training; weights are expected to stay the same, not compared).
# 2.77 h on 4 + 4 GPUs at the same time.  Needs: packed, raw256 and variant labels of navtrain_train and navtrain_val, GT stores,
# the BEV cache of each teacher.  A new run has a new checkpoint sha (training is not bitwise): put it in refiner_ckpt_sha16 of
# tools/release/registry.yaml, which also changes the hash in run names.
#   DET_TEACHER, MAP_TEACHER   registry names (default bevfusion, resmap): the output run dir is the entry's refiner_run
#   GPUS_DET, GPUS_MAP         default 0,1,2,3 and 4,5,6,7; 2 or 4 GPUs each (the global batch stays 16 tokens)
#   WHICH                      both, det, map (default both)
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DET_TEACHER=${DET_TEACHER:-bevfusion}
MAP_TEACHER=${MAP_TEACHER:-resmap}
GPUS_DET=${GPUS_DET:-0,1,2,3}
GPUS_MAP=${GPUS_MAP:-4,5,6,7}
WHICH=${WHICH:-both}
case "$WHICH" in both|det|map) ;; *) die "WHICH must be both, det or map" ;; esac

# export lines of the selected teachers + DET_RUN, MAP_RUN; refuses a teacher that is not ready
_exports=$("$SSR_PY" "$_SETUP_HERE/teacher_env.py" "$DET_TEACHER" "$MAP_TEACHER") || die "teacher registry"
eval "$_exports"

export CK_BEV64=1 CK_ALLOW_GPU4567=1 CK_TEACHER_MAX_GPUS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RECIPE=(--epochs 10 --tokens-per-batch 16 --lr 6e-4 --sur-groups near_mid --val-split navtrain_val --eval-every 2 --best-metric none \
        --r34-evals 0 --resume)
for s in navtrain_train navtrain_val; do
  need "$CK/packed/$s/tokens.parquet" "$CK/labels/$s/raw256/labels.npy" "$CK/ck2/labels/$s/var_separate_sampler16_accstraight/labels.npy"
done

# start ARM RUN_DIR GPUS SCORE_KEYS
start() {
  local arm=$1 rd=$2 gpus=$3 keys=$4 n
  refuse "$rd/done.json"
  refuse_unless_resume "$rd/config.json"
  n=$(echo "$gpus" | tr ',' '\n' | wc -l)
  case "$n" in 2|4) ;; *) die "GPUS for arm $arm: $gpus has $n GPUs, the recipe allows 2 or 4" ;; esac
  run_mkdir "$(dirname "$rd")"
  run_bg "$rd.log" env CUDA_VISIBLE_DEVICES="$gpus" nice -n 10 "$SSR_TORCHRUN" --standalone --nproc_per_node="$n" "$REPO/tools/ck/train_ck2.py" \
    --arm "$arm" --run-name "$(basename "$rd")" --out-root "$(dirname "$rd")" "${RECIPE[@]}" --score-keys "$keys"
}

case "$WHICH" in both|det) start T "$DET_RUN" "$GPUS_DET" nc,ttc,ep,comfort ;; esac
case "$WHICH" in both|map) start M "$MAP_RUN" "$GPUS_MAP" dac,ep ;; esac
wait_bg
say "finished.  Put the new sha16 of ckpt_last.pt in tools/release/registry.yaml (refiner_ckpt_sha16); verify_assets --only teachers compares them."
