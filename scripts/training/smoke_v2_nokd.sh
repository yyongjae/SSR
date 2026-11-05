#!/usr/bin/env bash
# GPU smoke of the refiner without any teacher, no BEV KD, with its launcher
# (scripts/training/train_v2_nokd.sh): 2 epochs x 60 micro-batches per rank (epoch 0 = warm-up + recording for the
# labeler, epoch 1 = on-policy), wandb off, then the checks below.  Same smoke recipe as the v2_v15_v7_main smoke
# without the v15 probe counters.
#
#   GPUS=0,1,2,3 bash scripts/training/smoke_v2_nokd.sh          (4 GPUs as the main run; 2 or 8 also run)
#   env: TAG (run name), MEM_LIMIT_GB (22.5, peak torch memory per GPU), SMOKE_ROOT (launcher-enforced smoke prefix)
#
# Checks (exit 1 on failure): training exit 0, two epochs (warmup_record, onpolicy), no non-finite loss / step value,
# peak memory < MEM_LIMIT_GB, launch/teachers.json mode 'none' (use_teachers false), every step: score KD 0, lateral-KD
# weight 0, teacher time < 5 ms, no teacher-derived log (ck2/e_abs_T ...), no v15 / distill loss in the step log, the
# labeler2 labelled generation 0 completely without errors, last.ckpt without any '_distill.' / 'ck_bev_kd.'
# / teacher tensor and with the refiner student 'ck_student.'.  Printed: per-phase sec_step and the 30-epoch estimate.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$REPO/tools/release/load_paths.sh"   # SSR_PY, SSR_DATA_ROOT, SSR_RUN_ROOT, ... (navsim/common/ssr_paths.py)
export SSR_ALLOW_DIRECT=1
GPUS=${GPUS:-0,1,2,3}
SMOKE_ROOT=${SMOKE_ROOT:-$SSR_DATA_ROOT/ck/ck2/e2e_smoke}
TAG=${TAG:-nokd_smoke_$(TZ=Asia/Seoul date +%m%d_%H%M)}
MEM_LIMIT_GB=${MEM_LIMIT_GB:-22.5}
SD=$SMOKE_ROOT/_cfg_$TAG
mkdir -p "$SD"
{ cat "$REPO/tools/ck/e2e2/v2_nokd_arm.extra"; printf '%s\n' "trainer.params.limit_train_batches=60" \
    "agent.config.grad_balance_warmup_iters=20" "agent.config.grad_balance_interval=10" \
    "agent.config.grad_norm_log_interval=10"; } > "$SD/smoke.extra"
{ cat "$REPO/tools/ck/e2e2/v2_nokd.set"; printf '%s\n' record_from_epoch=0 record_until_epoch=0 onpolicy_from_epoch=1 \
    label_refresh_every_mb=20; } > "$SD/smoke.set"
echo "[smoke_v2_nokd] $(TZ=Asia/Seoul date '+%F %T KST') start $TAG on GPUs $GPUS (git $(git -C "$REPO" rev-parse --short HEAD))"
RUN=$TAG E2E_SMOKE=1 E2E_ROOT=$SMOKE_ROOT E2E_WANDB=disable MAX_EPOCHS=2 LABELER_MAX_EPOCH=1 E2E_EXTRA=$SD/smoke.extra \
  E2E_CK_SET=$SD/smoke.set bash "$REPO/scripts/training/train_v2_nokd.sh" start --gpus "$GPUS" \
  || { echo "[smoke_v2_nokd] start refused"; exit 1; }
RUN=$TAG E2E_ROOT=$SMOKE_ROOT bash "$REPO/scripts/training/train_v2_nokd.sh" wait
R=$SMOKE_ROOT/$TAG
[ "$(cat "$R/launch/train.exit" 2>/dev/null)" = "0" ] || { echo "[smoke_v2_nokd] training exit $(cat "$R/launch/train.exit" 2>/dev/null) (see $R/launch/train.log)"; exit 1; }
NG=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
PYTHONPATH=$REPO "$SSR_PY" - "$R" "$MEM_LIMIT_GB" "$NG" <<'PY' || { echo "[smoke_v2_nokd] FAILED (run dir $R)"; exit 1; }
import glob, json, math, sys
import torch
R, mem_limit, ng = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
ep = [json.loads(l) for l in open(f"{R}/refiner/epochs.jsonl")]
ends = [e for e in ep if e["event"] == "epoch_end"]
phases = [e.get("phase") for e in ep if e["event"] == "epoch_start"]
assert len(ends) == 2 and phases == ["warmup_record", "onpolicy"], (len(ends), phases)
cum = ends[-1]["cum"]
assert cum["nonfinite_loss"] == 0 and cum["sur_nonfinite"] == 0, cum
# generation 0 (epoch-0 recording) fully labelled by labeler2; the overlap of the two 60-mb epochs is only a few tokens
# per rank, so G_src_prev of the smoke is printed, not checked
lab = json.load(open(f"{R}/refiner/lab/status.json"))
g0 = (lab.get("epochs") or {}).get("0") or {}
assert lab.get("tokens_ok", 0) > 0 and lab.get("tokens_err", 1) == 0 and g0.get("done") and g0.get("rows_done", 0) > 0, lab
st = [json.loads(l) for l in open(f"{R}/refiner/steps_rank0.jsonl")]
bad = sorted({k for s in st for k, v in s.items() if isinstance(v, float) and not math.isfinite(v)})
assert not bad, bad[:10]
mem = max(s.get("mem_gb", 0) for s in st)
assert mem < mem_limit, f"peak mem {mem} GB >= {mem_limit}"
tj = json.load(open(f"{R}/launch/teachers.json"))
assert tj.get("mode") == "none" and tj.get("use_teachers") is False and "det" not in tj and "map" not in tj, tj
eff = json.load(open(f"{R}/launch/refiner_effective.json"))
assert eff["use_teachers"] is False and eff["_arm"] == "nokd" and eff["lambda_kd_score"] == 0.0, eff
for s in st:
    assert s.get("ck2/kd_score", 0.0) == 0.0 and s.get("ck2/w_lat", 0.0) == 0.0 and s.get("ck2/kd_ok_frac", 0.0) == 0.0, s
    assert s.get("ck2/teacher_ms", 0.0) < 5.0, s.get("ck2/teacher_ms")
keys = sorted({k for s in st for k in s})
leak = [k for k in keys if k in ("ck2/e_abs_T", "ck2/e_abs_det", "ck2/e_abs_map", "ck2/lat_disagree")
        or "distill" in k or "probe" in k or k.startswith(("loss_attn_", "bevkd/"))]
assert not leak, leak
assert any(s.get("ck2/bce", 0.0) > 0 for s in st) and any(s.get("ck2/sur", 0.0) > 0 for s in st)
ck = sorted(glob.glob(f"{R}/train/**/last.ckpt", recursive=True))
assert ck, "no last.ckpt"
sd = torch.load(ck[-1], map_location="cpu", weights_only=False)["state_dict"]
bad_sd = [k for k in sd if k.startswith(("agent._distill.", "agent.ck_bev_kd.")) or "teacher" in k]
assert not bad_sd, bad_sd[:10]
assert any(k.startswith("agent.ck_student.") for k in sd), "no refiner student in the checkpoint"
mb_ep = 85109 // (4 * ng)
tot = 0.0
for ph, n_ep in (("warmup_record", 5), ("onpolicy", 25)):
    e = [x for x in ep if x["event"] == "epoch_start" and x.get("phase") == ph][0]["epoch"]
    ss = [s for s in st if s["epoch"] == e and s.get("sec_step")][10:]
    sec = sum(s["sec_step"] + (s.get("sec_wait") or 0.0) for s in ss) / max(len(ss), 1)
    tot += n_ep * mb_ep * sec
    print(f"{ph:14s} steps {len(ss)}  wall/mb {sec:.3f} s  ({mb_ep} mb/epoch -> {mb_ep * sec / 60:.0f} min/epoch)")
print(f"smoke OK: {len(st)} steps, peak mem {mem:.1f} GB, gen 0 labelled rows {g0['rows_done']}, G_src rank 0 "
      f"{{prev {cum['G_src_prev']}, fallback {cum['G_src_fallback']}}}, 30-epoch estimate {tot / 3600:.1f} h "
      f"(no epoch overhead) [추정]")
PY
echo "[smoke_v2_nokd] $(TZ=Asia/Seoul date '+%F %T KST') passed: $R"
