#!/usr/bin/env bash
# The trainval + SimScale recipe in one command: scripts/experiment.sh with five settings fixed,
#   DATA=trainval_simscale BACKBONE=dino_vits14_1148 DET_TEACHER=bevformer MAP_TEACHER=mask2map TF32=on
# = navtrain train + val + SimScale rounds 0-4 (every loss and KD on every token), DINOv2 ViT-S + LoRA at 1148 x 672,
# the camera-only teachers BEVFormer (det) and Mask2Map (map), TF32 (H100).  Everything else is the main recipe.
#
#   bash scripts/train_simscale.sh [train|check|smoke|status|stop|resume|name] [NGPU=8 | GPUS=0,...,7] [KEY=VALUE ...]
#
# No command = train.  train runs check first (launcher dry run + CPU agent build) and stops when it fails, then
# experiment.sh train with its guards (clean tree, MIN_FREE_GB, no second training, idle GPUs, verify_assets).
# NGPU=8 (the default, GPUs 0-7) is the reference count: one 8 x H100 node.  NGPU=4 / 2 also run (name part g4 / g2,
# per-rank counters scaled, the same global batch 128).  GPUS=<list> picks the indices instead.
# KEY=VALUE, passed on to experiment.sh: RUN_ROOT (default $SSR_RUN_ROOT of tools/release/paths.env), EPOCHS, SEED,
#   TAG, WANDB, SAVE_EVERY, DRY, MIN_FREE_GB (default 300), MEM_LIMIT_GB (smoke, default 30: dino_vits14_1148), RUN
#   (status / stop / resume by run name), CHECK_ROOT, ALLOW_DIRTY, ALLOW_COMMIT_CHANGE, FORCE_CONCURRENT, SMOKE_KEEP.
# The five fixed settings are refused with another value; any other setting is another recipe, for
#   bash scripts/experiment.sh <command> DATA=trainval_simscale ...   (docs/EXPERIMENTS.md).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXP="$REPO/scripts/experiment.sh"
# tests/test_simscale_launcher.py: a stub of experiment.sh, to run this script's sequence without data or GPUs
if [ -n "${SSR_RELEASE_TEST_STUBS:-}" ] && [ -f "$SSR_RELEASE_TEST_STUBS/experiment.sh" ]; then
  EXP=$SSR_RELEASE_TEST_STUBS/experiment.sh
fi
FIXED=(DATA=trainval_simscale BACKBONE=dino_vits14_1148 DET_TEACHER=bevformer MAP_TEACHER=mask2map TF32=on)
KEYS=(RUN_ROOT EPOCHS SEED TAG WANDB SAVE_EVERY DRY MIN_FREE_GB MEM_LIMIT_GB RUN CHECK_ROOT ALLOW_DIRTY
      ALLOW_COMMIT_CHANGE FORCE_CONCURRENT SMOKE_KEEP)
SMOKE_MEM_LIMIT_GB=30       # tools/release/registry.yaml dino_vits14_1148: peak 23.1 GiB (28.8 GiB in nvidia-smi)

say() { echo "[simscale] $*"; }
die() { echo "[simscale] REFUSED: $*" >&2; exit 3; }
_in() { local x=$1; shift; local y; for y in "$@"; do [ "$x" = "$y" ] && return 0; done; return 1; }

CMD=train
case "${1:-}" in
  train|check|smoke|status|stop|resume|name) CMD=$1; shift;;
  ""|*=*) ;;
  *) sed -n '2,17p' "$0"; exit 2;;
esac
NGPU=""; GPUS=""; RUN=""; RUN_ROOT=""; DRY=0; GOT_ROOT=0; GOT_MEM=0; ARGS=()
for a in "$@"; do
  case "$a" in
    *=*) k=${a%%=*}; v=${a#*=};;
    *) die "'$a': arguments are KEY=VALUE";;
  esac
  case "$k" in NGPU) NGPU=$v; continue;; GPUS) GPUS=$v; continue;; esac
  fixed=""; for f in "${FIXED[@]}"; do [ "${f%%=*}" = "$k" ] && fixed=$f; done
  if [ -n "$fixed" ]; then
    [ "$a" = "$fixed" ] || die "$k is fixed to ${fixed#*=} by this script (the trainval + SimScale recipe); $a is another recipe: bash scripts/experiment.sh"
    continue
  fi
  _in "$k" "${KEYS[@]}" || die "$k: not a key of this script (NGPU GPUS ${KEYS[*]}); another recipe: bash scripts/experiment.sh DATA=trainval_simscale ..."
  case "$k" in RUN) RUN=$v;; RUN_ROOT) RUN_ROOT=$v; GOT_ROOT=1;; DRY) DRY=$v;; MEM_LIMIT_GB) GOT_MEM=1;; esac
  ARGS+=("$a")
done
if [ -n "$NGPU" ]; then
  [ -z "$GPUS" ] || die "NGPU=$NGPU and GPUS=$GPUS: give one of them"
  case "$NGPU" in 2|4|8) ;; *) die "NGPU=$NGPU: 2, 4 or 8 (8 = the reference count, one 8 x H100 node)";; esac
  GPUS=$(seq -s, 0 $((NGPU - 1)))
fi
GOT_GPUS=0; [ -z "$GPUS" ] || GOT_GPUS=1
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
case "$CMD" in status|stop|resume) ;; *) [ -z "$RUN" ] || die "RUN=$RUN: only for status, stop and resume";; esac
source "$REPO/tools/release/load_paths.sh"   # SSR_RUN_ROOT: this script's runs go there unless RUN_ROOT is given
[ "$GOT_ROOT" = 1 ] || ARGS+=("RUN_ROOT=")   # experiment.sh: an empty RUN_ROOT is $SSR_RUN_ROOT, not its block's folder
ROOT=${RUN_ROOT:-$SSR_RUN_ROOT}
RECIPE=("${FIXED[@]}" "GPUS=$GPUS")

exp() { bash "$EXP" "$@"; }
print_where() {  # print_where <run folder>: where its records and logs are
  say "  folder  $1"
  say "  logs    $1/launch/train.log (training), $1/labeler/labeler.log (labeler), $1/launch/launch.log (launcher)"
  say "  steps   $1/refiner/steps_rank0.jsonl, epochs $1/refiner/epochs.jsonl, TensorBoard $1/train/lightning_logs"
}

case "$CMD" in
  name|status|stop) exp "$CMD" "${RECIPE[@]}" "${ARGS[@]}"; exit $?;;
  resume)
    # with RUN= the settings come from the run's settings.json: only how it is launched may be given
    if [ -n "$RUN" ]; then
      g=(); [ "$GOT_GPUS" = 0 ] || g=("GPUS=$GPUS")
      exp resume "${ARGS[@]}" "${g[@]}"; exit $?
    fi
    exp resume "${RECIPE[@]}" "${ARGS[@]}"; exit $?;;
esac

NAME=$(exp name "${RECIPE[@]}" "${ARGS[@]}") || exit 3
NG=$(awk -F, '{print NF}' <<< "$GPUS")
if [ "$CMD" = check ]; then
  say "check of $NAME on $NG GPUs ($GPUS): launcher dry run + CPU agent build, nothing starts"
  exp check "${RECIPE[@]}" "${ARGS[@]}"; exit $?
fi
if [ "$CMD" = smoke ]; then
  m=(); [ "$GOT_MEM" = 1 ] || m=("MEM_LIMIT_GB=${MEM_LIMIT_GB:-$SMOKE_MEM_LIMIT_GB}")
  say "smoke of $NAME on $NG GPUs ($GPUS): 2 epochs x 60 micro-batches"
  print_where "$ROOT/_smoke/smoke_$NAME"
  exp smoke "${RECIPE[@]}" "${ARGS[@]}" "${m[@]}"; exit $?
fi

# train: check first, then the start
R=$ROOT/$NAME
say "run $NAME on $NG GPUs ($GPUS)"
if [ "$DRY" != 1 ] && [ -e "$R" ]; then die "$R exists: bash scripts/train_simscale.sh resume (with the same keys)"; fi
say "check (launcher dry run + CPU agent build) before the start"
exp check "${RECIPE[@]}" "${ARGS[@]}" || die "check failed: nothing started ($NAME)"
exp train "${RECIPE[@]}" "${ARGS[@]}"; rc=$?
[ "$rc" = 0 ] || exit "$rc"
if [ "$DRY" = 1 ]; then say "(DRY) nothing started: $NAME"; exit 0; fi
say "training $NAME"
print_where "$R"
say "  status  bash scripts/train_simscale.sh status (the same keys), or bash scripts/experiment.sh status RUN=$NAME RUN_ROOT=$ROOT"
