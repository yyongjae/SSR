#!/usr/bin/env bash
# SimScale (Hugging Face dataset OpenDriveLab/SimScale), only for DATA=trainval_simscale: meta_datas + sensor_blobs_hist of
# the planner-based rounds into SSR_SIMSCALE_ROOT/synthetic_reaction_pdm_v1.0-<r>/, with each hist part's scenes checked
# against the part index the dataset ships (simscale_download.py).  Rounds 0-4: ~2.23 TB, ~6 h at 100 MB/s [estimate
# from the listed sizes].  Resumable: each finished archive leaves a stamp in DOWNLOAD_DIR/.done.  sensor_blobs_fut is
# not downloaded.  Only PLAN_ONLY=1 was run in testing.
#   ROUNDS        default "0 1 2 3 4"
#   DOWNLOAD_DIR  archives in flight and the stamps (default $SSR_SIMSCALE_ROOT/_download)
#   PLAN_ONLY=1   list what is left, its size, time and disk; download nothing
#   HF_TOKEN      optional, sent when set
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROUNDS=${ROUNDS:-0 1 2 3 4}
DL=${DOWNLOAD_DIR:-$SSR_SIMSCALE_ROOT/_download}
command -v curl >/dev/null && command -v tar >/dev/null || die "needs curl and tar"
args=(--rounds $ROUNDS --root "$SSR_SIMSCALE_ROOT" --download-dir "$DL")
[ "${PLAN_ONLY:-0}" = 1 ] && args+=(--plan-only)
run_mkdir "$SSR_SIMSCALE_ROOT"
run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/simscale_download.py" "${args[@]}"
