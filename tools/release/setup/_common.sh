# Sourced by every script of this directory.  Sets REPO, the roots of navsim/common/ssr_paths.py (paths.env first), CK.
#   SETUP_DRY=1  prints the commands instead of running them, creates and removes nothing
#   RESUME=1     lets a step continue over its own partial output; a finished output is still refused
set -euo pipefail
_SETUP_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$_SETUP_HERE/../../.." && pwd)
. "$REPO/tools/release/load_paths.sh"
CK=$SSR_DATA_ROOT/ck
SETUP_LOGS=$SSR_DATA_ROOT/setup_logs
SETUP_DRY=${SETUP_DRY:-0}
RESUME=${RESUME:-0}
export PYTHONPATH="$REPO" NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_DEVKIT_ROOT="$REPO"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$SSR_DATA_ROOT/setup_logs/navsim_exp}"
CPU=(env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10)

die() { echo "[$(basename "$0")] $*" >&2; exit 3; }
say() { echo "[$(basename "$0")] $*"; }

# run CMD...: run it, or print it under SETUP_DRY=1
run() {
  if [ "$SETUP_DRY" = 1 ]; then echo "+ $*"; else "$@"; fi
}

# run_mkdir DIR...: mkdir -p, or print it
run_mkdir() { run mkdir -p "$@"; }

# run_bg LOG CMD...: run in the background with stdout + stderr in LOG; wait_bg waits for all of them, any failure stops
_BG_PIDS=()
run_bg() {
  local log=$1; shift
  if [ "$SETUP_DRY" = 1 ]; then echo "+ $* > $log 2>&1 &"; return; fi
  mkdir -p "$(dirname "$log")"
  "$@" > "$log" 2>&1 &
  _BG_PIDS+=("$!")
}
wait_bg() {
  local rc=0 p
  for p in "${_BG_PIDS[@]:-}"; do
    [ -n "$p" ] || continue
    wait "$p" || rc=1
  done
  _BG_PIDS=()
  [ "$rc" = 0 ] || die "a background job failed: see $SETUP_LOGS"
}

# need PATH...: stop when a prerequisite is missing (under SETUP_DRY=1: print it and go on)
need() {
  local p
  for p in "$@"; do
    [ -e "$p" ] && continue
    [ "$SETUP_DRY" = 1 ] && { echo "# needs $p"; continue; }
    die "missing $p (docs/SETUP.md lists the order of the steps)"
  done
}

# refuse PATH...: stop when an output exists.  This check also holds under SETUP_DRY=1.
refuse() {
  local p
  for p in "$@"; do
    [ -e "$p" ] && die "refusing to overwrite $p: move it away first"
  done
  return 0
}

# refuse_unless_resume PATH...: same, but RESUME=1 goes on (the builders skip what they already wrote)
refuse_unless_resume() {
  [ "$RESUME" = 1 ] && return 0
  local p
  for p in "$@"; do
    [ -e "$p" ] && die "$p exists: RESUME=1 continues over it, or move it away first"
  done
  return 0
}

# verify GROUP PROFILE: the checks of tools/release/verify_assets.py for what this step built
verify() {
  run "${CPU[@]}" "$SSR_PY" "$REPO/tools/release/verify_assets.py" --profile "$2" --only "$1"
}
