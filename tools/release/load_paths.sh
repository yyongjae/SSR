# Source with REPO set to the repo root: exports every root of navsim/common/ssr_paths.py into this shell, after
# tools/release/paths.env when that file exists.  Stops the calling script when no python can read ssr_paths.py.
if [ -f "$REPO/tools/release/paths.env" ]; then
  set -a; . "$REPO/tools/release/paths.env"; set +a
fi
_ssr_py=${SSR_PY:-python3}
# SSR_BOOTSTRAP=1: the steps that create the environment run before the interpreter of SSR_PY exists
if [ -n "${SSR_BOOTSTRAP:-}" ] && [ ! -x "$_ssr_py" ]; then _ssr_py=python3; fi
_ssr_paths=$("$_ssr_py" -I "$REPO/navsim/common/ssr_paths.py" --shell) \
  || { echo "[load_paths] cannot run navsim/common/ssr_paths.py with $_ssr_py" >&2; exit 3; }
eval "$_ssr_paths"
unset _ssr_paths _ssr_py
