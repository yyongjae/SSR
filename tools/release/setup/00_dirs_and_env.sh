#!/usr/bin/env bash
# Step 0: show every root and make the empty directories.  Touches nothing that exists.
# First copy tools/release/paths.env.example to tools/release/paths.env and edit it.  Without that file the reference
# server's paths are used, which is right only on that server.
_given=${SSR_DATA_ROOT:-}
export SSR_BOOTSTRAP=1     # SSR_PY may not exist yet
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ ! -f "$REPO/tools/release/paths.env" ] && [ -z "$_given" ] && [ ! -d "$SSR_DATA_ROOT" ]; then
  die "no tools/release/paths.env and $SSR_DATA_ROOT does not exist: copy tools/release/paths.env.example first"
fi
say "repo $REPO"
for v in SSR_DATA_ROOT SSR_TEACHER_CACHE OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT SSR_TEST_DATA_ROOT SSR_PLAN_DATA SSR_NAVTEST_MC \
         SSR_PY SSR_TORCHRUN SSR_RUN_ROOT NAVSIM_V11 NAVSIM_V11_CACHE NAVSIM_V22_POST NAVSIM_V22_PRE NAVSIM_V22_CACHE; do
  printf '  %-20s %s\n' "$v" "${!v}"
done
run_mkdir "$SSR_DATA_ROOT/splits" "$SSR_DATA_ROOT/metric_cache/logs" "$SSR_DATA_ROOT/objects/logs" "$SSR_DATA_ROOT/sdf/logs" \
  "$CK/data-map" "$CK/packed" "$CK/labels" "$CK/raw256" "$CK/ck2/labels" "$SETUP_LOGS" \
  "$SSR_TEACHER_CACHE" "$SSR_PLAN_DATA" "$SSR_RUN_ROOT" "$OPENSCENE_DATA_ROOT/navsim_logs" "$OPENSCENE_DATA_ROOT/sensor_blobs"
for d in "$SSR_DATA_ROOT" "$SSR_TEACHER_CACHE" "$OPENSCENE_DATA_ROOT"; do
  [ -d "$d" ] && df -h "$d" | tail -1
done
if [ -x "$SSR_PY" ]; then
  "$SSR_PY" -V
else
  say "no interpreter at SSR_PY=$SSR_PY yet: run install_env.sh next"
fi
