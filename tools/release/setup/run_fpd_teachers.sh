#!/usr/bin/env bash
# The two FPD-structured teacher refiners (ck_e2e2.arch fpd: FPD-Drive's residual DiT + candidate scorer,
# navsim/agents/para_ssr/ck/fpd_net.py), one per cached teacher BEV, the counterpart of run_refiner_teachers.sh (CK2).
# Per teacher, in order (tools/ck/fpd/train_fpd_teacher.py; every stage resumes, a finished stage is skipped):
#   1 dit     the DiT on the GT target, 10 epochs, 16 tokens per batch, lr 2e-4, wd 0.01, clip 1.0     (GPUs)
#   2 bank    2 DiT draws of each of the 16 near + mid anchors of every navtrain_train / navtrain_val token (GPUs)
#   3 labels  official PDM labels of the 32 bank trajectories per token, tools/ck/fpd/label_fpd_bank.py   (CPU)
#   4 scorer  scorer (64-candidate sets: 32 raw anchors + 32 bank) + DiT, 10 epochs; writes the teacher run dir
# DET on GPUS_DET and MAP on GPUS_MAP at the same time; the CPU labels of one teacher can overlap the other's GPU stages.
# Needs: packed and raw256 labels of navtrain_train and navtrain_val, the navtrain metric cache, the BEV cache of each
# teacher.  Training is not bitwise: a new run has a new ckpt_last.pt sha16 (printed at the end).
#   DET_TEACHER, MAP_TEACHER   registry names (default bevfusion, resmap): the BEV caches teacher_env.py exports
#   FPD_DET_RUN, FPD_MAP_RUN   output run dirs (default $SSR_TEACHER_CACHE/refiner_fpd/det and .../map: the launch
#                              defaults of ck_e2e2.fpd.teacher_det_run / teacher_map_run)
#   GPUS_DET, GPUS_MAP         default 0,1,2,3 and 4,5,6,7; 2 or 4 GPUs each (the global batch stays 16 tokens)
#   WHICH                      both, det, map (default both)
#   LABEL_WORKERS              CPU processes of the bank labeler per teacher (default 48, at most 48)
#   RESUME=1                   continue a partial run; a finished teacher (done.json) is always refused
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DET_TEACHER=${DET_TEACHER:-bevfusion}
MAP_TEACHER=${MAP_TEACHER:-resmap}
GPUS_DET=${GPUS_DET:-0,1,2,3}
GPUS_MAP=${GPUS_MAP:-4,5,6,7}
WHICH=${WHICH:-both}
LABEL_WORKERS=${LABEL_WORKERS:-48}
FPD_DET_RUN=${FPD_DET_RUN:-$SSR_TEACHER_CACHE/refiner_fpd/det}
FPD_MAP_RUN=${FPD_MAP_RUN:-$SSR_TEACHER_CACHE/refiner_fpd/map}
# the default FPD run dirs are what every REFINER_ARCH=fpd run reads (with the DEFAULT teachers' kd_bev_0 / 1): a
# non-default teacher must write elsewhere (FPD_*_RUN), or its BEV would silently train the default dir
case "$WHICH" in both|det) [ "$DET_TEACHER" = bevfusion ] || [ "$FPD_DET_RUN" != "$SSR_TEACHER_CACHE/refiner_fpd/det" ] || die "DET_TEACHER=$DET_TEACHER: set FPD_DET_RUN (the default dir is the bevfusion FPD teacher)" ;; esac
case "$WHICH" in both|map) [ "$MAP_TEACHER" = resmap ] || [ "$FPD_MAP_RUN" != "$SSR_TEACHER_CACHE/refiner_fpd/map" ] || die "MAP_TEACHER=$MAP_TEACHER: set FPD_MAP_RUN (the default dir is the resmap FPD teacher)" ;; esac
case "$WHICH" in both|det|map) ;; *) die "WHICH must be both, det or map" ;; esac
case "$LABEL_WORKERS" in ''|*[!0-9]*) die "LABEL_WORKERS must be a number" ;; esac
[ "$LABEL_WORKERS" -ge 1 ] && [ "$LABEL_WORKERS" -le 48 ] || die "LABEL_WORKERS must be in 1..48"

# export lines of the selected teachers' BEV caches (+ DET_RUN, MAP_RUN of the CK2 teachers, unused here); refuses a
# teacher that is not ready
_exports=$("$SSR_PY" "$_SETUP_HERE/teacher_env.py" "$DET_TEACHER" "$MAP_TEACHER") || die "teacher registry"
eval "$_exports"

export CK_BEV64=1 CK_ALLOW_GPU4567=1 CK_TEACHER_MAX_GPUS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RECIPE=(--epochs 10 --tokens-per-batch 16 --lr 2e-4 --wd 0.01 --warmup-steps 500 --clip 1.0 --split navtrain_train \
        --val-split navtrain_val --eval-every 2 --resume)
for s in navtrain_train navtrain_val; do
  need "$CK/packed/$s/tokens.parquet" "$CK/labels/$s/raw256/labels.npy"
done
need "$SSR_DATA_ROOT/metric_cache"

# pipeline ARM RUN_DIR GPUS SCORE_KEYS NPROC: the four steps of one teacher, in order
pipeline() {
  local arm=$1 rd=$2 gpus=$3 keys=$4 n=$5 stage
  local trainer=(env CUDA_VISIBLE_DEVICES="$gpus" nice -n 10 "$SSR_TORCHRUN" --standalone --nproc_per_node="$n"
                 "$REPO/tools/ck/fpd/train_fpd_teacher.py" --arm "$arm" --run-dir "$rd" "${RECIPE[@]}" --score-keys "$keys")
  run "${trainer[@]}" --stage dit || return
  run "${trainer[@]}" --stage bank || return
  run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/fpd/label_fpd_bank.py" --run-dir "$rd" --workers "$LABEL_WORKERS" \
    --splits navtrain_train,navtrain_val || return
  run "${trainer[@]}" --stage scorer || return
}

# start ARM RUN_DIR GPUS SCORE_KEYS
start() {
  local arm=$1 rd=$2 gpus=$3 keys=$4 n
  refuse "$rd/done.json"
  refuse_unless_resume "$rd/dit/config.json"
  n=$(echo "$gpus" | tr ',' '\n' | wc -l)
  case "$n" in 2|4) ;; *) die "GPUS for arm $arm: $gpus has $n GPUs, the recipe allows 2 or 4" ;; esac
  run_mkdir "$rd"
  if [ "$SETUP_DRY" = 1 ]; then
    echo "# arm $arm -> $rd (log $rd.log)"
    pipeline "$arm" "$rd" "$gpus" "$keys" "$n"
  else
    run_bg "$rd.log" pipeline "$arm" "$rd" "$gpus" "$keys" "$n"
  fi
}

SELECTED=()
case "$WHICH" in both|det) start T "$FPD_DET_RUN" "$GPUS_DET" nc,ttc,ep,comfort; SELECTED+=("$FPD_DET_RUN") ;; esac
case "$WHICH" in both|map) start M "$FPD_MAP_RUN" "$GPUS_MAP" dac,ep; SELECTED+=("$FPD_MAP_RUN") ;; esac
wait_bg
for rd in "${SELECTED[@]}"; do
  if [ "$SETUP_DRY" = 1 ]; then
    say "FPD teacher run dir: $rd"
  else
    say "FPD teacher run dir: $rd (ckpt_last.pt sha16 $(sha256sum "$rd/ckpt_last.pt" | cut -c1-16))"
  fi
done
say "finished.  ck_e2e2.fpd.teacher_det_run / teacher_map_run: $FPD_DET_RUN / $FPD_MAP_RUN"
