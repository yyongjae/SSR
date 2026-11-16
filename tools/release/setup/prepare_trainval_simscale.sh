#!/usr/bin/env bash
# Every asset of the trainval_simscale split (DATA=trainval_simscale), step by step, then the check of all of them:
#   12_download_simscale.sh        only with DOWNLOAD=1: SimScale meta + history images, ~2.23 TB for rounds 0-4
#   16_trainval_simscale_table.sh  the token table (kept when it exists)
#   20_metric_cache_navtrain.sh    SPLITS=trainval_simscale: every row, SimScale scenes with the 4 + 8 frame window
#   30_gt_stores.sh                SPLIT=trainval_simscale: objects, sdf, e2e_side
#   40_packed_from_dataset.sh      SPLITS=trainval_simscale
#   50_warmup_labels.sh            SPLITS=trainval_simscale: raw256, pdm_score_256_officialEP_trainval_simscale, variants
#   65_cam_teacher_bev.sh          BEVFormer + Mask2Map caches of every row (GPU, TEACHER_GPUS); skipped when the file is
#                                  absent or its done.json exists
#   verify_assets.py --profile full --data trainval_simscale   every row of every asset (VERIFY_QUICK=1: a sample)
# Needs 15_token_lists.sh before (TRAIN_TOKENS / VAL_TOKENS name other tables).  The shared stores (metric cache, GT
# stores, teacher caches) of an existing navtrain setup are reused: their tokens are skipped, not rebuilt.  Resumable:
# every step runs with RESUME=1 and skips what it finished, so run the same command again after a stop.  The wall time
# of each step goes to $SETUP_LOGS/prepare_trainval_simscale.times.tsv.  CPU except 65_*.
#   SETUP_DRY=1   print the plan (the commands of every step), run nothing
#   MINI=1        the development selection of 16_* (MINI_TRAIN, MINI_VAL, MINI_SIM; ROUNDS the rounds on disk)
#   SKIP="30 65"  leave steps out by number; K, LABEL_WORKERS, SIDE_WORKERS, SIM_WORKERS, ROUNDS go to the steps
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TABLE=$SSR_DATA_ROOT/splits/trainval_simscale.parquet
TIMES=$SETUP_LOGS/prepare_trainval_simscale.times.tsv
TEACHER_DONE=$SSR_TEACHER_CACHE/_work/cam_teachers/done.json
run_mkdir "$SETUP_LOGS"

# step NUMBER SCRIPT [VAR=VALUE...]: one setup step with RESUME=1, timed (SKIP leaves it out)
step() {
  local num=$1 name=$2 t0=$SECONDS
  shift 2
  case " ${SKIP:-} " in *" $num "*) say "skip $name (SKIP)"; return 0 ;; esac
  say "step $name $*"
  env RESUME=1 "$@" bash "$_SETUP_HERE/$name"
  [ "$SETUP_DRY" = 1 ] || printf '%s\t%s\t%s\n' "$name" "$((SECONDS - t0))" "$(date +%FT%T)" >> "$TIMES"
}

if [ "${DOWNLOAD:-0}" = 1 ]; then step 12 12_download_simscale.sh; fi
if [ -e "$TABLE" ]; then
  say "$TABLE exists: kept (move it away, with every asset built from it, for another table)"
else
  step 16 16_trainval_simscale_table.sh
fi
step 20 20_metric_cache_navtrain.sh SPLITS=trainval_simscale
step 30 30_gt_stores.sh SPLIT=trainval_simscale
step 40 40_packed_from_dataset.sh SPLITS=trainval_simscale
step 50 50_warmup_labels.sh SPLITS=trainval_simscale
if [ ! -f "$_SETUP_HERE/65_cam_teacher_bev.sh" ]; then
  say "no 65_cam_teacher_bev.sh here: the camera teacher caches are not built by this script"
elif [ -e "$TEACHER_DONE" ]; then
  say "camera teacher caches finished ($TEACHER_DONE)"
else
  step 65 65_cam_teacher_bev.sh GPUS="${TEACHER_GPUS:-0}" TABLE="$TABLE"
fi
q=(); [ "${VERIFY_QUICK:-0}" = 1 ] && q=(--quick)
t0=$SECONDS
rc=0
run "${CPU[@]}" "$SSR_PY" "$REPO/tools/release/verify_assets.py" --profile full --data trainval_simscale "${q[@]}" \
  --json "$SETUP_LOGS/verify_trainval_simscale.json" || rc=$?
[ "$SETUP_DRY" = 1 ] || printf '%s\t%s\t%s\n' verify_assets "$((SECONDS - t0))" "$(date +%FT%T)" >> "$TIMES"
exit "$rc"
