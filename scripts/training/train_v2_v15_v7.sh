#!/usr/bin/env bash
# v2_v15_v7 (report 132): baseline v2-seg + BEV selector v15 (km 64 m port) + refiner v2 on the front 0-64 m,
# 100 x 100 BEV, with lens undistortion and camera pillars at z [-1, 3].  Thin wrapper around the CK2 e2e launcher
# (tools/ck/e2e2/train_e2e2.sh: training + CPU labeler + watchdog, same recipe as byounggun's fusion_v15_main):
#   - CK_BEV64=1 for every process it starts (refiner S grid / teacher caches 100 x 100; ck_e2e2 refuses a mismatch
#     between this and the BEV config);
#   - E2E_EXTRA = tools/ck/e2e2/v2_v15_v7_arm.extra (grid, seg head, v15), E2E_CK_SET = tools/ck/e2e2/v2_v15_v7.set
#     (64 m refiner teachers + KD calibration);
#   - outputs under /home/external-user/ssd/yongjae_refiner/v7/e2e/<RUN> (never under /workspace/byounggun).
#
#   RUN=v2_v15_v7_main CK2_E2E_APPROVED=1 bash scripts/training/train_v2_v15_v7.sh start --gpus 0,1,2,3
#   RUN=v2_v15_v7_main bash scripts/training/train_v2_v15_v7.sh status | stop | wait
#   RUN=v2_v15_v7_main E2E_DRY_RUN=1 CK2_E2E_APPROVED=1 bash scripts/training/train_v2_v15_v7.sh start --gpus 0,1,2,3
#   smoke: RUN=<name> E2E_SMOKE=1 E2E_ROOT=/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke ... (launcher rules)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export CK_BEV64=1
export E2E_ROOT=${E2E_ROOT:-/home/external-user/ssd/yongjae_refiner/v7/e2e}
export E2E_EXTRA=${E2E_EXTRA:-$REPO/tools/ck/e2e2/v2_v15_v7_arm.extra}
export E2E_CK_SET=${E2E_CK_SET:-$REPO/tools/ck/e2e2/v2_v15_v7.set}
exec bash "$REPO/tools/ck/e2e2/train_e2e2.sh" "$@"
