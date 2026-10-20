#!/usr/bin/env bash
# refiner v5 KD student (ARM=v5kd) GPU smoke on the MAIN layout (4 GPUs 0-3, labeler2 + watchdog), modelled on
# tools/ck/e2e2/smoke_e2e2.sh G1M: 3 epochs x G1_MB micro-batches per rank (warmup / warmup_record / onpolicy) on a
# fixed token set, then launch_util2 smoke-check2.  Not a training run (E2E_SMOKE=1 under .../ck/ck2/e2e_smoke/).
#   REPO=<worktree> TAG=<tag> CKSET=<file of ck_e2e2 key=value lines: calib / z-stats / teachers> G1_MB=100 bash smoke_v5kd_gpu.sh
set -uo pipefail
REPO=${REPO:?REPO=<worktree>}
PY=/venv/ssr/bin/python
TE=$REPO/tools/ck/e2e2/train_e2e2.sh
LU2=$REPO/tools/ck/e2e2/launch_util2.py
TAG=${TAG:-v5kd_$(TZ=Asia/Seoul date +%m%d_%H%M)}
SR=/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke/$TAG
G1_MB=${G1_MB:-100}
NG=4
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SR"
log() { echo "[$(TZ=Asia/Seoul date '+%F %T') KST] [smoke_v5kd] $*" | tee -a "$SR/smoke.log"; }
cd "$REPO"
ntok=$(( NG * 4 * G1_MB ))
if [ ! -s "$SR/g_tokens.txt" ]; then
  "$PY" - "$ntok" "$SR/g_tokens.json" "$SR/g_tokens.txt" <<'EOF' | tee -a "$SR/smoke.log"
import json, sys
from tools.ck.e2e import launch_util as LU
n, out, extra = int(sys.argv[1]), sys.argv[2], sys.argv[3]
toks, logs = LU.smoke_tokens(n)
q = lambda xs: "[" + ",".join("'" + str(x) + "'" for x in xs) + "]"
open(out, "w").write(json.dumps({"tokens": toks, "logs": logs, "n": len(toks)}))
open(extra, "w").write(f"scene_filter.tokens={q(toks)}\nscene_filter.log_names={q(logs)}\n")
print(f"smoke token set: {len(toks)} tokens from {len(logs)} logs -> {extra}")
EOF
fi
# E2E_EXTRA = the v5 student extra (v13 distillation, det-head fixes, 3-frame input) + the smoke limits / token set
{ /usr/bin/grep -v '^[[:space:]]*#' "$REPO/tools/ck/e2e2/fusion_v13_v5.extra" | /usr/bin/grep -v '^[[:space:]]*$'
  printf '%s\n' "trainer.params.limit_train_batches=$G1_MB"; cat "$SR/g_tokens.txt"; } > "$SR/g_extra.txt"
{ printf '%s\n' record_from_epoch=1 onpolicy_from_epoch=2 record_until_epoch=2 label_refresh_every_mb=20 lat_kd.start_mb=50
  [ -n "${CKSET:-}" ] && /usr/bin/grep -v '^[[:space:]]*#' "$CKSET" | /usr/bin/grep -v '^[[:space:]]*$'; } > "$SR/g_main.set"
"$PY" "$LU2" gpu-check --gpus 0,1,2,3 > /dev/null || { log "GPUs 0-3 not free"; exit 1; }
log "v5kd smoke on GPUs 0-3: 3 epochs x $G1_MB mb/rank, labeler2; repo $REPO ($(git -C "$REPO" rev-parse --short HEAD))"
RUN=v5kd_g1 ARM=v5kd E2E_ROOT=$SR E2E_SMOKE=1 E2E_WANDB=disable MAX_EPOCHS=3 LABELER_MAX_EPOCH=2 WATCHDOG_EVERY=30 \
  E2E_EXTRA=$SR/g_extra.txt E2E_CK_SET=$SR/g_main.set bash "$TE" start --gpus 0,1,2,3 || { log "start failed"; exit 1; }
t0=$(date +%s); frc=0
while [ ! -f "$SR/v5kd_g1/launch/train.exit" ]; do
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$SR/nvsmi.csv" 2>/dev/null
  [ $(( $(date +%s) - t0 )) -gt "${G1_TIMEOUT:-5400}" ] && { log "timeout: stopping"; RUN=v5kd_g1 ARM=v5kd E2E_ROOT=$SR bash "$TE" stop; frc=1; break; }
  sleep 15
done
log "train exit $(cat "$SR/v5kd_g1/launch/train.exit" 2>/dev/null)"
RUN=v5kd_g1 ARM=v5kd E2E_ROOT=$SR E2E_WAIT_TIMEOUT=1800 bash "$TE" wait --labeler; log "labeler wait rc $?"
"$PY" "$LU2" smoke-check2 --run-dir "$SR/v5kd_g1" --out "$SR/result_G1M_v5kd.json" --onpolicy-epochs 2 --gen-epochs 1,2 \
  | tee -a "$SR/smoke.log" || frc=1
log "done rc $frc"
exit $frc
