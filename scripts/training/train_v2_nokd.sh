#!/usr/bin/env bash
# v2_nokd: v2_v15_v7_main without any teacher -- baseline v2-seg + refiner v2 with no refiner
# teacher and no KD (tools/ck/e2e2/v2_nokd.set) and no BEV KD (v15 off, tools/ck/e2e2/v2_nokd_arm.extra), front 0-64 m
# 100 x 100 BEV, lens undistortion, camera pillars z [-1, 3].  Thin wrapper around the CK2 e2e launcher
# (tools/ck/e2e2/train_e2e2.sh: training + CPU labeler + watchdog):
#   - CK_BEV64=1 for every process it starts (refiner S grid 100 x 100);
#   - E2E_EXTRA = tools/ck/e2e2/v2_nokd_arm.extra, E2E_CK_SET = tools/ck/e2e2/v2_nokd.set (callers may replace either);
#   - GPUs 4-7 (CK_ALLOW_GPU4567=1) only when listed in --gpus; the launcher keeps the global batch 128 (4 per GPU x
#     ngpu x accumulate) and rescales its per-rank counters (4 GPUs: accumulate 8, GradBalancer 5300 / 100);
#   - outputs under $E2E_ROOT/<RUN>.
#
#   RUN=v2_nokd_main CK2_E2E_APPROVED=1 bash scripts/training/train_v2_nokd.sh start --gpus 0,1,2,3
#   RUN=v2_nokd_main bash scripts/training/train_v2_nokd.sh status | stop | wait
#   RUN=v2_nokd_main E2E_DRY_RUN=1 CK2_E2E_APPROVED=1 bash scripts/training/train_v2_nokd.sh start --gpus 0,1,2,3
#   smoke: GPUS=0,1,2,3 bash scripts/training/smoke_v2_nokd.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$REPO/tools/release/load_paths.sh"   # SSR_PY, SSR_DATA_ROOT, SSR_RUN_ROOT, ... (navsim/common/ssr_paths.py)
export CK_BEV64=1 SSR_ALLOW_DIRECT=1
export E2E_ROOT=${E2E_ROOT:-$SSR_RUN_ROOT}
export E2E_CK_SET=${E2E_CK_SET:-$REPO/tools/ck/e2e2/v2_nokd.set}
export E2E_EXTRA=${E2E_EXTRA:-$REPO/tools/ck/e2e2/v2_nokd_arm.extra}
GPUS=""; prev=""
for a in "$@"; do [ "$prev" = "--gpus" ] && GPUS=$a; prev=$a; done
case ",$GPUS," in *,4,*|*,5,*|*,6,*|*,7,*) export CK_ALLOW_GPU4567=1;; esac
exec bash "$REPO/tools/ck/e2e2/train_e2e2.sh" "$@"
