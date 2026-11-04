#!/usr/bin/env bash
# The one entry point for experiments on this code: name, check, smoke, train, resume, status, stop, wait, eval, table,
# assets, teachers, purge.  docs/EXPERIMENTS.md has every setting and every arm as a one-line command.
#
#   bash scripts/experiment.sh <command> [KEY=VALUE ...]
#
# A KEY=VALUE on the command line wins over the block below.  Prefer the command line: editing this block makes the
# working tree dirty, and train refuses a dirty tree.
#
# ---------------------------------------------------------------------------------------------------- settings
REFINER=on                  # on | off: the refiner (critic + lateral correction on the v2 head's candidates)
REFINER_KD=on               # on | off | score (lateral KD off) | lat (score KD off).  off: no refiner teacher is loaded
BEV_KD=on                   # on | off: BEV KD v15 from both teachers
DET_KD=on                   # on | off (both KD paths) | off_bev | off_ref: KD from the DET teacher
MAP_KD=on                   # on | off (both KD paths) | off_bev | off_ref: KD from the MAP teacher
BACKBONE=r34                # r18 r34 r50 r101 r152 r50d convnext_tiny regnety_032 efficientnet_b0 (tools/release/registry.yaml)
DET_TEACHER=bevfusion       # registry name
MAP_TEACHER=resmap          # registry name
INTERACTION=on              # on | off: task interaction
DET_HEAD=on                 # on | off (off needs INTERACTION=off)
MAP_HEAD=seg                # seg | vector | off (off needs INTERACTION=off)
FRAMES=2,3                  # 2,3 | 3 | 0,1,2,3: history frames
UNDISTORT=on                # on | off: lens undistortion
HEADING=xy                  # xy (heading from the path) | free (own head)
PILLAR_Z=-1,3               # a,b: camera pillar height range in metres
BEV_RANGE=64                # 64 | 32 (32 only with REFINER=off BEV_KD=off)
BN=moving                   # moving | frozen: backbone BatchNorm statistics in training
SEED=0                      # global + ck_e2e2.seed + distill_probe_seed; warmup.sampler_seed stays 0
EPOCHS=30                   # trainer + agent.max_epochs + labeler max
GPUS=0,1,2,3,4,5,6,7        # 2, 4 or 8 distinct indices, each with < 1 GiB used; the reference recipe is 8
SAVE_EVERY=1                # checkpoint every N epochs (must divide EPOCHS); not in the name, not in the hash
RUN_ROOT=/home/external-user/experiments  # empty: $SSR_RUN_ROOT
TAG=                        # [A-Za-z0-9.-]+, a letter or digit first; goes into the run name
WANDB=offline               # online | offline | disable; not in the hash
EXTRA=                      # raw Hydra lines, ';'-separated: adds _x to the name and enters the hash
DRY=0                       # 1: train and smoke print what they would do and launch nothing
# ------------------------------------------------------------------------------------------------ end of settings
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$REPO/tools/release/load_paths.sh"   # tools/release/paths.env, then SSR_PY, SSR_DATA_ROOT, SSR_RUN_ROOT, ...
PY=$SSR_PY
E2E="$REPO/tools/ck/e2e2/train_e2e2.sh"
E2E_START=$E2E
VERIFY_ASSETS="$REPO/tools/release/verify_assets.py"
EVAL_SH="$REPO/tools/release/eval.sh"
SMOKE_CHECK="$REPO/tools/release/smoke_check.py"
COMPOSE="$REPO/tools/release/compose.py"
LU2="$REPO/tools/ck/e2e2/launch_util2.py"
# tests/test_release_compose.py: stubs of the launcher, asset check and eval, to run the guards and records without a GPU
if [ -n "${SSR_RELEASE_TEST_STUBS:-}" ]; then
  E2E_START=$SSR_RELEASE_TEST_STUBS/train_e2e2.sh; VERIFY_ASSETS=$SSR_RELEASE_TEST_STUBS/verify_assets.py
  EVAL_SH=$SSR_RELEASE_TEST_STUBS/eval.sh; SMOKE_CHECK=$SSR_RELEASE_TEST_STUBS/smoke_check.py
fi
SETTINGS=(REFINER REFINER_KD BEV_KD DET_KD MAP_KD BACKBONE DET_TEACHER MAP_TEACHER INTERACTION DET_HEAD MAP_HEAD FRAMES
          UNDISTORT HEADING PILLAR_Z BEV_RANGE BN SEED EPOCHS GPUS SAVE_EVERY RUN_ROOT TAG WANDB EXTRA DRY)
# options of single commands; these may also come from the environment
RUN=""; EPOCH=""; EPDMS=""; CONFIRM=${CONFIRM:-0}; REGISTRY=${REGISTRY:-}; CHECK_ROOT=${CHECK_ROOT:-}
ALLOW_DIRTY=${ALLOW_DIRTY:-0}; ALLOW_MIXED=${ALLOW_MIXED:-0}; ALLOW_COMMIT_CHANGE=${ALLOW_COMMIT_CHANGE:-0}
FORCE_CONCURRENT=${FORCE_CONCURRENT:-0}; MIN_FREE_GB=${MIN_FREE_GB:-70}; MEM_LIMIT_GB=${MEM_LIMIT_GB:-22.5}
SMOKE_KEEP=${SMOKE_KEEP:-1}; QUICK=${QUICK:-0}; BASE=${BASE:-}; GPUS_DET=${GPUS_DET:-}; GPUS_MAP=${GPUS_MAP:-}; WHICH=${WHICH:-}
OPTIONS=(RUN EPOCH EPDMS CONFIRM REGISTRY CHECK_ROOT ALLOW_DIRTY ALLOW_MIXED ALLOW_COMMIT_CHANGE FORCE_CONCURRENT MIN_FREE_GB
         MEM_LIMIT_GB SMOKE_KEEP QUICK BASE GPUS_DET GPUS_MAP WHICH)

say() { echo "[experiment] $*"; }
die() { echo "[experiment] REFUSED: $*" >&2; exit 3; }
_in() { local x=$1; shift; local y; for y in "$@"; do [ "$x" = "$y" ] && return 0; done; return 1; }

CMD=${1:-}; shift || true
GIVEN=()
for a in "$@"; do
  case "$a" in
    *=*) k=${a%%=*}; v=${a#*=};;
    *) die "'$a': arguments are KEY=VALUE (docs/EXPERIMENTS.md)";;
  esac
  _in "$k" "${SETTINGS[@]}" "${OPTIONS[@]}" || die "unknown key $k (docs/EXPERIMENTS.md)"
  printf -v "$k" '%s' "$v"
  GIVEN+=("$k")
done
[ -n "$REGISTRY" ] && export SSR_RELEASE_REGISTRY=$REGISTRY
RUN_ROOT=${RUN_ROOT:-$SSR_RUN_ROOT}

py() { PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" "$PY" "$@"; }
setting_args() { local k; for k in "${SETTINGS[@]}"; do printf '%s=%s\n' "$k" "${!k}"; done; }
cp_name() { local args; mapfile -t args < <(setting_args); py "$COMPOSE" name "${args[@]}"; }
cp_emit() {  # cp_emit <out dir> [compose emit options]: write the composition there and load its environment
  local out=$1; shift
  local args; mapfile -t args < <(setting_args)
  py "$COMPOSE" emit --out-dir "$out" "$@" "${args[@]}" > /dev/null || exit 3
  # shellcheck disable=SC1091
  . "$out/env.sh"
}
only_keys() {  # only_keys <what> <allowed keys...>: any other key on the command line is refused
  local what=$1; shift; local k
  [ "${#GIVEN[@]}" -gt 0 ] || return 0
  for k in "${GIVEN[@]}"; do _in "$k" "$@" || die "$what takes only $*; got $k"; done
}
need_run_name() { [[ "$RUN" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || die "RUN='$RUN': a run name, not a path"; }
resolve_run() {  # RUN from the command line, else from the settings; R = its folder (smoke runs are under _smoke)
  if [ -z "$RUN" ]; then RUN=$(cp_name) || exit 3; fi
  need_run_name
  case "$RUN" in smoke_*) RROOT=$RUN_ROOT/_smoke;; *) RROOT=$RUN_ROOT;; esac
  R=$RROOT/$RUN
}
free_gb() {  # free_gb <path>: GiB free on the nearest existing parent
  local d=$1
  while [ ! -e "$d" ] && [ "$d" != / ]; do d=$(dirname "$d"); done
  df -BG --output=avail "$d" | tail -1 | tr -dc 0-9
}
git_dirty() { [ -n "$(git -C "$REPO" status --porcelain)" ]; }
running_trainings() {  # pid files under RUN_ROOT whose process lives and is a training; never pgrep
  local f p
  for f in "$RUN_ROOT"/*/launch/train.pid "$RUN_ROOT"/_smoke/*/launch/train.pid; do
    [ -f "$f" ] || continue
    p=$(cat "$f" 2>/dev/null); [ -n "$p" ] && [ -r "/proc/$p/cmdline" ] || continue
    if tr '\0' ' ' < "/proc/$p/cmdline" | grep -q run_training; then echo "$f (pid $p)"; fi
  done
}
print_pressure() {
  if [ -r /proc/pressure/memory ]; then say "memory pressure: $(head -1 /proc/pressure/memory)"; fi
  say "free on $RUN_ROOT: $(free_gb "$RUN_ROOT") GiB"
}
refuse_or_dry() {  # with DRY=1 a guard reports and goes on
  if [ "$DRY" = 1 ]; then say "(DRY) would refuse: $*"; else die "$*"; fi
}
guard_dirty() {
  if git_dirty; then
    if [ "$ALLOW_DIRTY" = 1 ]; then say "dirty tree accepted (ALLOW_DIRTY=1): recorded as git_dirty, diff saved into the run"
    else refuse_or_dry "uncommitted changes in $REPO: commit them, or ALLOW_DIRTY=1 to record the run as git_dirty"; fi
  fi
}
guard_disk() {
  local f; f=$(free_gb "$RUN_ROOT")
  [ "$f" -ge "$1" ] || refuse_or_dry "$f GiB free on $RUN_ROOT < $1 GiB (MIN_FREE_GB); a 30-epoch run peaks at ~60 GiB"
}
guard_second_training() {
  local r; r=$(running_trainings)
  if [ -n "$r" ] && [ "$FORCE_CONCURRENT" != 1 ]; then
    refuse_or_dry "a training is already running, FORCE_CONCURRENT=1 to run beside it: $r"
  fi
}
guard_gpus() {
  [ -z "${SSR_RELEASE_TEST_STUBS:-}" ] || return 0
  SSR_GPU_POLICY=any py "$LU2" gpu-check --gpus "$GPUS" > /dev/null || refuse_or_dry "GPU(s) $GPUS busy (> 1 GiB used)"
}
save_dirty_if_any() { if git_dirty; then py "$COMPOSE" dirty-record --out-dir "$1" || die "cannot save the diff into $1"; fi; }

# launcher dry run + CPU agent build under <root>/_check; no GPU, no training.  Sets CHECK_DIR.
preflight() {  # preflight [--smoke]
  local base=${CHECK_ROOT:-$RUN_ROOT}/_check name; name=$(cp_name) || exit 3
  local keep_wandb=$WANDB; WANDB=disable
  [ "${1:-}" = --smoke ] && name=smoke_$name
  CHECK_DIR=$base/$name
  local out=$CHECK_DIR/gen
  cp_emit "$out" --run-root "$base" "$@"
  WANDB=$keep_wandb
  if ! E2E_DRY_RUN=1 bash "$E2E" start --gpus "$GPUS" > "$out/launcher.log" 2>&1; then
    tail -5 "$out/launcher.log" >&2; die "the launcher refused this run (full log: $out/launcher.log)"
  fi
  say "launcher dry run ok: $(tail -1 "$out/launcher.log")"
  py "$COMPOSE" verify --composition "$out/composition.json" --overrides "$CHECK_DIR/launch_dryrun/overrides.txt" \
    --teachers-json "$CHECK_DIR/launch_dryrun/teachers.json" || die "check failed"
  CUDA_VISIBLE_DEVICES= py "$REPO/tools/release/check_build.py" --overrides "$CHECK_DIR/launch_dryrun/overrides.txt" \
    --out "$out/build.json" || die "the agent refused this run"
}

do_name() { cp_name || exit 3; }

do_check() {
  print_pressure
  preflight
  say "check ok: $CP_RUN"
  say "written only under $CHECK_DIR"
}

undo_failed_start() {  # the launcher refused before training: leave no run folder behind
  if [ ! -e "$R/launch/train.pid" ] && [ ! -d "$R/train/lightning_logs" ]; then
    rm -rf "$R/launch/gen"
    rm -f "$R/launch/settings.json" "$R/launch/verify_assets.log" "$R/launch/overrides.txt" "$R/launch/refiner_effective.json" \
      "$R/launch/teachers.json" "$R/launch/git_status.txt" "$R/launch/git_dirty.diff" "$R/launch/git_untracked.tar.gz"
    rmdir "$R/train" "$R/refiner" "$R/launch" "$R" 2> /dev/null
  fi
  return 0
}

do_train() {
  local name; name=$(cp_name) || exit 3
  R=$RUN_ROOT/$name
  [ ! -e "$R" ] || refuse_or_dry "$R exists: use 'resume RUN=$name'"
  guard_dirty
  guard_disk "$MIN_FREE_GB"
  guard_second_training
  guard_gpus
  print_pressure
  preflight
  if [ "$DRY" = 1 ]; then
    cp_emit "$CHECK_DIR/gen_real"
    say "(DRY) would run: RUN=$CP_RUN E2E_ROOT=$E2E_ROOT bash $E2E start --gpus $GPUS"
    say "(DRY) env, set file, extra file: $CHECK_DIR/gen_real/{env.sh,ck.set,arm.extra}"
    sed 's/^/  set   /' "$CHECK_DIR/gen_real/ck.set"; sed 's/^/  extra /' "$CHECK_DIR/gen_real/arm.extra"
    return 0
  fi
  local prof; prof=$CP_PROFILE
  local alog=$CHECK_DIR/gen/verify_assets.log arc=0
  [ -f "$VERIFY_ASSETS" ] || die "$VERIFY_ASSETS is missing"
  py "$VERIFY_ASSETS" --profile "$prof" --quick --det-teacher "$DET_TEACHER" --map-teacher "$MAP_TEACHER" > "$alog" 2>&1 || arc=$?
  [ "$arc" = 0 ] || { tail -20 "$alog" >&2; die "assets of profile $prof are not ready (full log: $alog)"; }
  local gen=$R/launch/gen
  cp_emit "$gen"
  cp "$alog" "$R/launch/verify_assets.log"
  py "$COMPOSE" record --composition "$gen/composition.json" --out "$R/launch/settings.json" --asset-profile "$prof" \
    --asset-exit "$arc" --asset-log launch/verify_assets.log || die "cannot write settings.json"
  save_dirty_if_any "$R/launch"
  if ! bash "$E2E_START" start --gpus "$GPUS"; then undo_failed_start; die "the launcher refused the start"; fi
  if ! py "$COMPOSE" verify --composition "$gen/composition.json" --overrides "$R/launch/overrides.txt" > /dev/null; then
    bash "$E2E_START" stop; die "launch/overrides.txt differs from settings.json: training stopped"
  fi
  say "started $CP_RUN"
  say "status: bash scripts/experiment.sh status RUN=$CP_RUN    next: wait, eval"
}

do_resume() {
  # with RUN= the settings come from the run's settings.json; only how it is launched may be given
  [ -z "$RUN" ] || only_keys "resume RUN=..." RUN GPUS WANDB SAVE_EVERY RUN_ROOT ALLOW_DIRTY ALLOW_COMMIT_CHANGE FORCE_CONCURRENT \
    MIN_FREE_GB REGISTRY
  resolve_run
  local sj=$R/launch/settings.json
  [ -f "$sj" ] || die "$sj not found: not a run of this script"
  local given_gpus="" given_wandb="" given_save="" line k
  _in GPUS "${GIVEN[@]:-}" && given_gpus=$GPUS
  _in WANDB "${GIVEN[@]:-}" && given_wandb=$WANDB
  _in SAVE_EVERY "${GIVEN[@]:-}" && given_save=$SAVE_EVERY
  local keep_root=$RUN_ROOT
  while IFS= read -r line; do k=${line%%=*}; printf -v "$k" '%s' "${line#*=}"; done < <(py "$COMPOSE" settings-args --settings-json "$sj")
  RUN_ROOT=$keep_root
  [ -z "$given_gpus" ] || GPUS=$given_gpus
  [ -z "$given_wandb" ] || WANDB=$given_wandb
  [ -z "$given_save" ] || SAVE_EVERY=$given_save
  guard_dirty
  guard_disk "$MIN_FREE_GB"
  guard_second_training
  guard_gpus
  local tmp=$RUN_ROOT/_check/$RUN/gen_resume
  cp_emit "$tmp"
  local rc_args=(--composition "$tmp/composition.json" --settings-json "$sj")
  [ "$ALLOW_COMMIT_CHANGE" = 1 ] && rc_args+=(--allow-commit-change)
  py "$COMPOSE" resume-check "${rc_args[@]}" || exit 3
  local stamp; stamp=$(TZ=Asia/Seoul date +%m%d_%H%M%S)
  [ -f "$R/launch/overrides.txt" ] && cp "$R/launch/overrides.txt" "$R/launch/overrides.$stamp.txt"
  cp "$sj" "$R/launch/settings.$stamp.json"
  cp_emit "$R/launch/gen"
  py "$COMPOSE" record --composition "$R/launch/gen/composition.json" --out "$R/launch/resume_$stamp.json" || die "cannot record the resume"
  save_dirty_if_any "$R/launch/dirty_$stamp"
  bash "$E2E_START" start --resume --gpus "$GPUS" || die "the launcher refused the resume"
  say "resumed $RUN on GPUs $GPUS"
}

passthrough() {  # status | stop | wait
  resolve_run
  RUN=$RUN E2E_ROOT=$RROOT bash "$E2E" "$1"
}

gpu_sampler() {  # gpu_sampler <csv> <gpu list>: memory.used of the run's GPUs every 5 s, for the refiner-off smoke check
  while :; do
    nvidia-smi --id="$2" --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$1" 2> /dev/null
    sleep 5
  done
}

do_smoke() {
  local name; name=smoke_$(cp_name) || exit 3
  R=$RUN_ROOT/_smoke/$name
  if [ "$DRY" = 1 ]; then
    preflight --smoke
    sed 's/^/  set   /' "$CHECK_DIR/gen/ck.set"; sed 's/^/  extra /' "$CHECK_DIR/gen/arm.extra"
    say "(DRY) smoke overrides checked, nothing launched: $CHECK_DIR"
    return 0
  fi
  [ ! -e "$R" ] || die "$R exists: remove it, or TAG=... for another smoke of this recipe"
  guard_disk 5
  guard_second_training
  guard_gpus
  print_pressure
  local gen=$R/launch/gen
  cp_emit "$gen" --smoke
  py "$COMPOSE" record --composition "$gen/composition.json" --out "$R/launch/settings.json" || die "cannot write settings.json"
  save_dirty_if_any "$R/launch"
  if ! bash "$E2E_START" start --gpus "$GPUS"; then undo_failed_start; die "the launcher refused the smoke"; fi
  gpu_sampler "$R/launch/gpu_mem.csv" "$GPUS" < /dev/null > /dev/null 2>&1 &
  local sampler=$!
  bash "$E2E_START" wait; local rc=$?
  kill "$sampler" 2> /dev/null; wait "$sampler" 2> /dev/null
  [ "$rc" = 0 ] || say "training exit $rc (see $R/launch/train.log)"
  py "$SMOKE_CHECK" --run-dir "$R" --mem-limit-gb "$MEM_LIMIT_GB" || die "smoke failed: $R"
  if [ "$SMOKE_KEEP" = 0 ]; then
    rm -rf "$R/refiner/rec" "$R/refiner/lab"
    local c; for c in "$R"/train/lightning_logs/version_*/checkpoints; do rm -rf "$c"; done
    say "removed this smoke's checkpoints, rec, lab; logs and json kept"
  fi
  say "smoke passed: $R"
}

do_eval() {
  only_keys eval RUN GPUS EPOCH EPDMS RUN_ROOT ALLOW_DIRTY
  [ -n "$RUN" ] || die "eval needs RUN=<name>"
  need_run_name
  R=$RUN_ROOT/$RUN
  [ -d "$R" ] || die "$R not found"
  [ -f "$EVAL_SH" ] || die "$EVAL_SH is missing"
  export ALLOW_DIRTY
  exec bash "$EVAL_SH" "$R" "$GPUS" "$EPOCH" "${EPDMS:-1}"
}

do_table() {
  only_keys table RUN_ROOT ALLOW_MIXED BASE
  local b=(); [ -z "$BASE" ] || b=(--base "$BASE")
  [ -f "$REPO/tools/release/table.py" ] || die "tools/release/table.py is missing (not part of this commit)"
  ALLOW_MIXED=$ALLOW_MIXED py "$REPO/tools/release/table.py" --run-root "$RUN_ROOT" "${b[@]}"
}

do_assets() {
  local prof
  if [ -n "$RUN" ]; then
    resolve_run
    prof=$(py - "$R/launch/settings.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["profile"])
PY
    ) || die "no settings.json for $RUN"
  else
    local tmp; tmp=$(mktemp -d)
    cp_emit "$tmp"
    prof=$CP_PROFILE
    rm -rf "$tmp"
  fi
  [ -f "$VERIFY_ASSETS" ] || die "$VERIFY_ASSETS is missing"
  local q=(); [ "$QUICK" = 1 ] && q=(--quick)
  py "$VERIFY_ASSETS" --profile "$prof" --det-teacher "$DET_TEACHER" --map-teacher "$MAP_TEACHER" "${q[@]}"
}

do_teachers() {
  only_keys teachers DET_TEACHER MAP_TEACHER GPUS_DET GPUS_MAP WHICH REGISTRY
  [ -f "$REPO/tools/release/setup/run_refiner_teachers.sh" ] || die "tools/release/setup/run_refiner_teachers.sh is missing"
  [ -z "$GPUS_DET" ] || export GPUS_DET
  [ -z "$GPUS_MAP" ] || export GPUS_MAP
  [ -z "$WHICH" ] || export WHICH
  DET_TEACHER=$DET_TEACHER MAP_TEACHER=$MAP_TEACHER SSR_GPU_POLICY=any exec bash "$REPO/tools/release/setup/run_refiner_teachers.sh"
}

do_purge() {
  only_keys purge RUN CONFIRM RUN_ROOT
  [ -n "$RUN" ] || die "purge needs RUN=<name>"
  resolve_run
  [ "$CONFIRM" = 1 ] || die "purge removes $R/refiner/{lab,rec}: add CONFIRM=1"
  [ -f "$R/refiner/TRAIN_DONE" ] || die "$R has no refiner/TRAIN_DONE: the run is not finished"
  local p; p=$(cat "$R/launch/train.pid" 2>/dev/null || true)
  if [ -n "$p" ] && [ -r "/proc/$p/cmdline" ] && tr '\0' ' ' < "/proc/$p/cmdline" | grep -q run_training; then
    die "training of $RUN is still running (pid $p)"
  fi
  du -sh "$R/refiner/lab" "$R/refiner/rec" 2> /dev/null
  rm -rf "$R/refiner/lab" "$R/refiner/rec"
  say "removed $R/refiner/{lab,rec}"
}

case "$CMD" in
  name) do_name;;
  check) do_check;;
  smoke) do_smoke;;
  train) do_train;;
  resume) do_resume;;
  status|stop|wait) passthrough "$CMD";;
  eval) do_eval;;
  table) do_table;;
  assets) do_assets;;
  teachers) do_teachers;;
  purge) do_purge;;
  *) sed -n '2,7p' "$0"; exit 2;;
esac
