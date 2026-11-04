#!/usr/bin/env bash
# GPU smoke of branch v2_final (online KD calibration, kd_calib.mode online, user 2026-11-02 option B) with the
# v2_v15_v7 launcher: 2 epochs x 60 micro-batches per rank (epoch 0 = warm-up + recording for the labeler, epoch 1 =
# on-policy), wandb off, then the checks below.  Same smoke recipe as the 2026-11-02 8-GPU smoke of v2_v15_v7_main
# (ssd/yongjae_refiner/v7/chain_student.sh) plus kd_calib.min_n=50 (the smoke's on-policy epoch has few labelled rows).
#
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/training/smoke_v2_final.sh          (8 GPUs as the main run; 2 or 4 also run)
#   env: TAG (run name), MEM_LIMIT_GB (22.5, peak torch memory per GPU), SMOKE_ROOT (launcher-enforced smoke prefix)
#
# Checks (exit 1 on failure): training exit 0, two epochs (warmup_record, onpolicy), no non-finite loss / step value,
# peak memory < MEM_LIMIT_GB, launch/teachers.json kd_calib mode 'online', ck_e2e2/kd_calib_online.jsonl with >= 5
# warm-up refits whose every key (DET nc / ep / ttc, MAP dac / ep) is fitted with finite a > 0, >= 1 on-policy window,
# ck2/cal_* in the step log.  Printed (not checked): per-phase sec_step and ck2/teacher_ms (cost of the extra teacher
# pass on the label set in on-policy), the last (a, b) per key, whether an on-policy fit (EMA reset) happened.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$REPO/tools/release/load_paths.sh"   # SSR_PY, SSR_DATA_ROOT, SSR_RUN_ROOT, ... (navsim/common/ssr_paths.py)
export SSR_ALLOW_DIRECT=1
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
SMOKE_ROOT=${SMOKE_ROOT:-$SSR_DATA_ROOT/ck/ck2/e2e_smoke}
TAG=${TAG:-v2final_smoke_$(TZ=Asia/Seoul date +%m%d_%H%M)}
MEM_LIMIT_GB=${MEM_LIMIT_GB:-22.5}
SD=$SMOKE_ROOT/_cfg_$TAG
mkdir -p "$SD"
{ cat "$REPO/tools/ck/e2e2/v2_v15_v7_arm.extra"; printf '%s\n' "trainer.params.limit_train_batches=60" \
    "++agent.config.distill_probe_warmup_steps=20" "++agent.config.distill_probe_ramp_steps=20" \
    "agent.config.grad_balance_warmup_iters=20" "agent.config.grad_balance_interval=10" \
    "agent.config.grad_norm_log_interval=10"; } > "$SD/smoke.extra"
{ cat "$REPO/tools/ck/e2e2/v2_v15_v7.set"; printf '%s\n' record_from_epoch=0 record_until_epoch=0 onpolicy_from_epoch=1 \
    label_refresh_every_mb=20 lat_kd.start_mb=30 kd_calib.min_n=50; } > "$SD/smoke.set"
echo "[smoke_v2_final] $(TZ=Asia/Seoul date '+%F %T KST') start $TAG on GPUs $GPUS (git $(git -C "$REPO" rev-parse --short HEAD))"
RUN=$TAG E2E_SMOKE=1 E2E_ROOT=$SMOKE_ROOT E2E_WANDB=disable MAX_EPOCHS=2 LABELER_MAX_EPOCH=1 E2E_EXTRA=$SD/smoke.extra \
  E2E_CK_SET=$SD/smoke.set bash "$REPO/scripts/training/train_v2_v15_v7.sh" start --gpus "$GPUS" \
  || { echo "[smoke_v2_final] start refused"; exit 1; }
RUN=$TAG E2E_ROOT=$SMOKE_ROOT bash "$REPO/scripts/training/train_v2_v15_v7.sh" wait
R=$SMOKE_ROOT/$TAG
[ "$(cat "$R/launch/train.exit" 2>/dev/null)" = "0" ] || { echo "[smoke_v2_final] training exit $(cat "$R/launch/train.exit" 2>/dev/null) (see $R/launch/train.log)"; exit 1; }
"$SSR_PY" - "$R" "$MEM_LIMIT_GB" <<'PY' || { echo "[smoke_v2_final] FAILED (run dir $R)"; exit 1; }
import json, math, sys
R, mem_limit = sys.argv[1], float(sys.argv[2])
ep = [json.loads(l) for l in open(f"{R}/ck_e2e2/epochs.jsonl")]
ends = [e for e in ep if e["event"] == "epoch_end"]
phases = [e.get("phase") for e in ep if e["event"] == "epoch_start"]
assert len(ends) == 2 and phases == ["warmup_record", "onpolicy"], (len(ends), phases)
cum = ends[-1]["cum"]
assert cum["nonfinite_loss"] == 0 and cum["sur_nonfinite"] == 0, cum
st = [json.loads(l) for l in open(f"{R}/ck_e2e2/steps_rank0.jsonl")]
bad = sorted({k for s in st for k, v in s.items() if isinstance(v, float) and not math.isfinite(v)})
assert not bad, bad[:10]
mem = max(s.get("mem_gb", 0) for s in st)
assert mem < mem_limit, f"peak mem {mem} GB >= {mem_limit}"
tj = json.load(open(f"{R}/launch/teachers.json"))
assert (tj.get("kd_calib") or {}).get("mode") == "online", tj.get("kd_calib")
assert any("ck2/cal_det_nc_a" in s for s in st), "no ck2/cal_* in steps_rank0.jsonl"
cal = [json.loads(l) for l in open(f"{R}/ck_e2e2/kd_calib_online.jsonl")]
KEYS = ("det/nc", "det/ep", "det/ttc", "map/dac", "map/ep")
wu = [c for c in cal if c["window_phase"] == "warmup_record" and "keys" in c]
assert len(wu) >= 5, f"{len(wu)} warm-up refits"
for c in wu:
    for k in KEYS:
        r = c["keys"][k]
        assert r["fitted"] and r["a"] > 0 and math.isfinite(r["a"]) and math.isfinite(r["b"]), (c["refit"], k, r)
op = [c for c in cal if c["window_phase"] == "onpolicy"]
assert op, "no on-policy window record"
op_fit = [c for c in op if "keys" in c and any(r.get("fitted") for r in c["keys"].values())]
for ph in ("warmup_record", "onpolicy"):
    e = [i for i, x in enumerate(ep) if x["event"] == "epoch_start" and x.get("phase") == ph][0]
    ss = [s for s in st if s["epoch"] == ep[e]["epoch"] and s.get("sec_step")]
    tm = [s["ck2/teacher_ms"] for s in ss if "ck2/teacher_ms" in s]
    print(f"{ph:14s} steps {len(ss)}  sec_step mean {sum(s['sec_step'] for s in ss) / max(len(ss), 1):.3f}  "
          f"teacher_ms mean {sum(tm) / max(len(tm), 1):.1f}")
last = cal[-1] if "keys" in cal[-1] else ([c for c in cal if "keys" in c] or [{}])[-1]
print("last refit", last.get("refit"), "window", last.get("window_phase"), "reset", last.get("reset"))
for k in KEYS:
    r = (last.get("keys") or {}).get(k, {})
    print(f"  {k:8s} a {r.get('a', r.get('a_prev')):.3f} b {r.get('b', r.get('b_prev')):.3f} n_win {r.get('n_win', 0):.0f} "
          f"mean label {r.get('mean_label_win', float('nan')):.3f} raw {r.get('mean_pred_raw_win', float('nan')):.3f} "
          f"prev {r.get('mean_pred_prev_win', float('nan')):.3f}")
print(f"smoke OK: {len(st)} steps, peak mem {mem:.1f} GB, {len(wu)} warm-up refits, {len(op)} on-policy windows "
      f"({len(op_fit)} fitted{', EMA reset' if any(c.get('reset') for c in op) else ', no on-policy fit (few labelled rows)'})")
PY
echo "[smoke_v2_final] $(TZ=Asia/Seoul date '+%F %T KST') passed: $R"
