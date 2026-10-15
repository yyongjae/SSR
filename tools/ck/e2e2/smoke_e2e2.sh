#!/usr/bin/env bash
# CK2 e2e GPU smoke (PREPARED, NOT RUN: needs the user's go and free GPUs 0-3, i.e. after the CK2 teachers finished).
# Each stage writes <SR>/result_<stage>.json (launch_util2 smoke-check2).  SR = .../ck/ck2/e2e_smoke/<TAG>.
#
#   TAG=<tag> bash tools/ck/e2e2/smoke_e2e2.sh G0 G1 G2 G3     (or: all)
#   TAG=<tag> bash tools/ck/e2e2/smoke_e2e2.sh status
#
#   G0  CPU: pytest tests/test_ck2_e2e_*.py (CUDA_VISIBLE_DEVICES=) -> result_G0.json (rc + summary line)
#   G1  the real layout at once: main arm DDP on GPUs 0,1 + BEV-KD arm DDP on GPUs 2,3, each with its labeler2
#       (32 workers) + watchdog; 3 epochs x G1_MB (300) micro-batches per rank = warmup / warmup_record / onpolicy on
#       a FIXED set of 2 x 4 x G1_MB tokens (whole logs, same tokens every epoch, write_sets)
#       (ck sets record_from 1, onpolicy_from 2, record_until 2, label_refresh_every_mb 20, lat_kd.start_mb 200;
#       arm: bev_kd.start_mb 200) -> per phase s/mb, wall/mb, peak memory, variants / teacher / student ms
#       (SPEC R1 / R2), labels of epoch 1 used in epoch 2 (G_src prev), gen 1 = recorded trajectories bitwise, BEV-KD
#       per teacher (bevkd_arm.set: det + map; result_G1_bevkd.json 'bevkd'): lam_t, ratio_now_t (= lam_t g_t /
#       g_plan at the measurement, target 0.1), ok_frac_t (masked tokens), gnorm/bev_bevkd_<t> vs gnorm/bev_v2 (the
#       gradient shares to check before the real arm), checks every teacher logged and lam_t > 0 after start_mb;
#       labeler under the load of 2 trainings + 64 workers (R3), 30-epoch estimate on 2 GPUs.
#   G2  resume: main arm only, stopped (train-only) after its epoch-1 checkpoint, resumed with --resume, finishes
#       epoch 2 (callback state / EMA / labeler continuity).  GPUs 0,1.
#   G3  eval_e2e2.py on the G1 main checkpoint: navtrain_val --limit 200 (all stages; grid choice) then navtest
#       --limit 200 (fixed from val), GPU 0, label workers 32.
#   Teachers: the yaml defaults ck2T10dep / ck2M10dep (decoupled EP = ck_e2e2.ep_target; teacher_check refuses a
#   mismatch) with their KD calibration files (ck_e2e2.kd_calib, yaml default enabled: run fit_kd_calib.py for both
#   teachers first; train_e2e2.sh refuses a missing / mismatched file).  E2E_SMOKE=1 waives the approval gate below .../e2e_smoke/
#   only.  W&B disabled.  Timeouts (s): G1_TIMEOUT 5400, G2_TIMEOUT 5400, LABELER_TIMEOUT 1800.
#   Exit code (report 48 F2): nonzero iff any requested stage failed -- G0 = the pytest rc; G1 = either arm's start /
#   labeler wait / smoke-check2 / the G1 timeout (all accumulated; a refused BEV-KD start stops the already started
#   main arm); G2 = no epoch-1 checkpoint, a train / labeler wait or its check; G3 = either eval_e2e2 rc, a missing
#   or stale (older than this G3 run) metrics / infer_check / summary file, or result_G3.json pass false.  The exit
#   code alone is the gate; every stage still writes its result_*.json.  G2_STOP_DELAY (30 s): wait before the stop.
#   Tests only: SMOKE_PY / SMOKE_TE / SMOKE_ROOT replace the python, train_e2e2.sh and the e2e_smoke root.
# no 'set -e' on purpose: every stage runs and returns its own rc, the loop at the end aggregates them
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
PY=${SMOKE_PY:-/venv/ssr/bin/python}
LU2="$REPO/tools/ck/e2e2/launch_util2.py"
TE=${SMOKE_TE:-$REPO/tools/ck/e2e2/train_e2e2.sh}
TAG=${TAG:-$(TZ=Asia/Seoul date +%m%d_%H%M)}
SR=${SMOKE_ROOT:-/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke}/$TAG
G1_MB=${G1_MB:-300}
export PYTHONPATH=$REPO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# AF_UNIX socket paths under $TMPDIR (DataLoader workers, G0 pytest) are limited to 107 chars (see train_e2e2.sh)
if [ -n "${TMPDIR:-}" ] && [ "${#TMPDIR}" -gt 75 ]; then export TMPDIR=/tmp; fi
mkdir -p "$SR"
cd "$REPO" || exit 1

log() { echo "[$(TZ=Asia/Seoul date '+%F %T') KST] [smoke2 $TAG] $*" | tee -a "$SR/smoke.log"; }
te() {  # te <run> <arm> <cmd> [args]: train_e2e2.sh against the smoke root
  local run=$1 arm=$2; shift 2
  RUN=$run ARM=$arm E2E_ROOT=$SR E2E_SMOKE=1 E2E_WANDB=${E2E_WANDB:-disable} bash "$TE" "$@"
}
check() { "$PY" "$LU2" smoke-check2 --run-dir "$SR/$1" --out "$SR/result_$2.json" "${@:3}" | tee -a "$SR/smoke.log"; }

write_sets() {
  # Fixed token set of 2 GPUs x 4 tokens x G1_MB (whole navtrain_train logs with a metric cache, launch_util.smoke_tokens,
  # as the old S3): every epoch sees the SAME tokens, so the on-policy epoch 2 finds the labels the labeler wrote from
  # epoch 1 (G_src prev).  With the full navtrain filter + limit_train_batches each epoch would draw ~2400 different
  # tokens of 85k and G_src prev would be ~3 % (the smoke check needs >= 90 %).  Tokens / logs are QUOTED: Hydra parses
  # an unquoted digit-only token (e.g. 6540354015965607, 01170848407050e2) as a number (6 of these 2400 tokens).
  local ntok=$(( 2 * 4 * G1_MB ))
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
    [ -s "$SR/g_tokens.txt" ] || { log "smoke token set failed"; return 1; }
  fi
  { printf '%s\n' "trainer.params.limit_train_batches=$G1_MB"; cat "$SR/g_tokens.txt"; } > "$SR/g_extra.txt"
  printf '%s\n' record_from_epoch=1 onpolicy_from_epoch=2 record_until_epoch=2 label_refresh_every_mb=20 \
    lat_kd.start_mb=200 > "$SR/g_main.set"
  { cat "$SR/g_main.set"; echo bev_kd.start_mb=200; } > "$SR/g_bevkd.set"
}

G0() {
  log "G0 CPU pytest tests/test_ck2_e2e_*.py"
  local out="$SR/G0_pytest.log" rc
  CUDA_VISIBLE_DEVICES="" nice -n 10 "$PY" -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_*.py \
    --basetemp="$SR/pytest" > "$out" 2>&1; rc=$?
  printf '{"stage": "G0", "rc": %s, "summary": "%s", "pass": %s}\n' "$rc" "$(grep -E '(passed|failed|error)' "$out" \
    | tail -n 1)" "$([ $rc = 0 ] && echo true || echo false)" > "$SR/result_G0.json"
  log "G0 rc $rc"
  return "$rc"
}

G1() {
  write_sets || return 1
  "$PY" "$LU2" gpu-check --gpus 0,1 > /dev/null && "$PY" "$LU2" gpu-check --gpus 2,3 > /dev/null \
    || { log "G1: GPUs 0-3 not all free"; return 1; }
  log "G1 main (GPUs 0,1) + bevkd (GPUs 2,3): 3 epochs x $G1_MB mb/rank, labeler2 W=32 each"
  MAX_EPOCHS=3 LABELER_MAX_EPOCH=2 WATCHDOG_EVERY=30 E2E_EXTRA=$SR/g_extra.txt E2E_CK_SET=$SR/g_main.set \
    te g1_main main start --gpus 0,1 || return 1
  MAX_EPOCHS=3 LABELER_MAX_EPOCH=2 WATCHDOG_EVERY=30 E2E_EXTRA=$SR/g_extra.txt E2E_CK_SET=$SR/g_bevkd.set \
    te g1_bevkd bevkd start --gpus 2,3 \
    || { log "G1: bevkd start failed: stopping the already started g1_main"; te g1_main main stop; return 1; }
  local t0 frc=0 lrc; t0=$(date +%s)
  while [ ! -f "$SR/g1_main/launch/train.exit" ] || [ ! -f "$SR/g1_bevkd/launch/train.exit" ]; do
    for r in g1_main g1_bevkd; do
      [ -f "$SR/$r/ck_e2e2/lab/status.json" ] && { tr -d '\n' < "$SR/$r/ck_e2e2/lab/status.json"; echo; } \
        >> "$SR/$r/lab_snapshots.jsonl"
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$SR/g1_nvsmi.csv" 2>/dev/null
    [ $(( $(date +%s) - t0 )) -gt "${G1_TIMEOUT:-5400}" ] && { log "G1 timeout: stopping"; te g1_main main stop;
                                                                te g1_bevkd bevkd stop; frc=1; break; }
    sleep 15
  done
  for r in g1_main:main g1_bevkd:bevkd; do
    E2E_WAIT_TIMEOUT=${LABELER_TIMEOUT:-1800} te "${r%%:*}" "${r##*:}" wait --labeler; lrc=$?
    log "G1 ${r%%:*} labeler exit $lrc"; [ "$lrc" = 0 ] || frc=1
  done
  # both checks always run (each writes its result json); either failing (or a labeler wait above) fails G1
  check g1_main G1_main --onpolicy-epochs 2 --gen-epochs 1,2 || { log "G1 main check failed"; frc=1; }
  check g1_bevkd G1_bevkd --onpolicy-epochs 2 --gen-epochs 1,2 || { log "G1 bevkd check failed"; frc=1; }
  return "$frc"
}

G2() {
  write_sets || return 1
  "$PY" "$LU2" gpu-check --gpus 0,1 > /dev/null || { log "G2: GPUs 0,1 not free"; return 1; }
  log "G2 resume test (main arm, GPUs 0,1)"
  MAX_EPOCHS=3 LABELER_MAX_EPOCH=2 WATCHDOG_EVERY=30 E2E_EXTRA=$SR/g_extra.txt E2E_CK_SET=$SR/g_main.set \
    te g2 main start --gpus 0,1 || return 1
  local t0 frc=0 w; t0=$(date +%s)
  until ls "$SR"/g2/train/lightning_logs/version_*/checkpoints/epoch=1-step=*.ckpt > /dev/null 2>&1; do
    [ -f "$SR/g2/launch/train.exit" ] && { log "G2: training ended before the epoch-1 checkpoint"; frc=1; break; }
    [ $(( $(date +%s) - t0 )) -gt "${G2_TIMEOUT:-5400}" ] && { log "G2 timeout"; frc=1; break; }
    sleep 10
  done
  sleep "${G2_STOP_DELAY:-30}"
  log "G2 stop (train only) during epoch 2"
  te g2 main stop --train-only
  MAX_EPOCHS=3 LABELER_MAX_EPOCH=2 WATCHDOG_EVERY=30 E2E_EXTRA=$SR/g_extra.txt E2E_CK_SET=$SR/g_main.set \
    te g2 main start --resume --gpus 0,1 || return 1
  E2E_WAIT_TIMEOUT=${G2_TIMEOUT:-5400} te g2 main wait; w=$?; log "G2 train exit $w"; [ "$w" = 0 ] || frc=1
  E2E_WAIT_TIMEOUT=${LABELER_TIMEOUT:-1800} te g2 main wait --labeler; w=$?; log "G2 labeler exit $w"
  [ "$w" = 0 ] || frc=1
  check g2 G2 --onpolicy-epochs 2 --gen-epochs 1,2 || { log "G2 check failed"; frc=1; }
  return "$frc"
}

G3() {
  log "G3 eval_e2e2 on the G1 main checkpoint (GPU 0)"
  local EV=("$PY" tools/ck/e2e2/eval_e2e2.py --run-dir "$SR/g1_main" --limit 200 --gpus 0 --workers-label 32
            --bootstrap 200)
  local rv=0 rt=0 t0; t0=$(date +%s)
  "${EV[@]}" --split navtrain_val --stage all >> "$SR/G3_eval.log" 2>&1 \
    || { rv=$?; log "G3 navtrain_val failed (rc $rv)"; }
  "${EV[@]}" --split navtest --stage dump,label,infer_check,eval,summary >> "$SR/G3_eval.log" 2>&1 \
    || { rt=$?; log "G3 navtest failed (rc $rt)"; }
  # every stage of eval_e2e2 rewrites its output, so an output older than this G3 run is stale (an earlier attempt
  # under the same TAG): it counts as missing
  "$PY" - "$SR" "$rv" "$rt" "$t0" <<'EOF' | tee -a "$SR/smoke.log"
import json, sys
from pathlib import Path
sr = Path(sys.argv[1]); r = sr / "g1_main/eval"
t0 = float(sys.argv[4])
fresh = lambda p: p.is_file() and p.stat().st_mtime >= t0 - 1.0  # noqa: E731
out = {"stage": "G3", "rc": {"navtrain_val": int(sys.argv[2]), "navtest": int(sys.argv[3])},
       "summary": fresh(r / "summary.json"), "started_epoch_s": t0}
for s in ("navtrain_val", "navtest"):
    m = r / "root/eval/g1_main" / s / "metrics.json"
    ic = r / "root/infer/g1_main" / s / "infer_check.json"
    out[s] = {"metrics": fresh(m), "metrics_stale": m.is_file() and not fresh(m), "infer_check_stale":
              ic.is_file() and not fresh(ic),
              "infer_check": json.loads(ic.read_text()).get("pass") if fresh(ic) else None}
out["pass"] = bool(not any(out["rc"].values()) and out["summary"]
                   and all(out[s]["metrics"] and out[s]["infer_check"] for s in ("navtrain_val", "navtest")))
(sr / "result_G3.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
sys.exit(0 if out["pass"] else 4)
EOF
}

status() {
  for r in g1_main:main g1_bevkd:bevkd g2:main; do
    [ -d "$SR/${r%%:*}" ] && te "${r%%:*}" "${r##*:}" status
  done
  ls "$SR"/result_*.json 2>/dev/null
}

[ $# -gt 0 ] || { sed -n '2,32p' "$0"; exit 2; }
ARGS=("$@"); [ "$1" = all ] && ARGS=(G0 G1 G2 G3)
log "stages ${ARGS[*]} -> $SR"
FAIL=0
for s in "${ARGS[@]}"; do
  case "$s" in
    G0|G1|G2|G3|status) "$s" || { log "$s returned $?"; FAIL=1; };;
    *) log "unknown stage $s"; exit 2;;
  esac
done
exit $FAIL
