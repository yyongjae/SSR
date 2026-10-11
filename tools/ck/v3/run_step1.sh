#!/bin/bash
# refiner-v3 Step-1 launcher (docs/refiner_v3/SPEC.md 5.4, 7; docs/refiner_v3/STEP1_RUN.md).
#
#   V3_USER_APPROVED=1 bash tools/ck/v3/run_step1.sh pre|a|b|d
#
# NOT run by the builders / fixer: the main session launches Step 1 only with the user's approval, in this order:
#   pre  forecaster pre-training (review fix D3), 1 process on GPU 0: tools/ck/v3/pretrain_forecaster.py
#   a    learned arm, critic training with the pre-trained forecaster FROZEN, 2-GPU DDP on GPUs 0,1
#   b    analytic CA arm, 2-GPU DDP on GPUs 0,1
#   d    (optional) GT-future oracle, PRIVILEGED, 2-GPU DDP on GPUs 0,1
# V3_GPUS=4,5 (user 2026-10-12: GPUs 4,5 only for short jobs and only when idle) runs a / b / d on GPUs 4,5 instead;
#   pre accepts V3_GPUS=0 | 4.  Any other list is refused.
# Arguments come from `tools/ck/v3/arms.py cli --arm <stage>` (the single source of the arm definitions:
# tools/ck/v3/arms/*.json = ck2T10 recipe + the temporal flags).
# Refuses unless: V3_USER_APPROVED=1; the GPUs used are idle (< 1 GB); the K_OBJ = 96 object packs of navtrain_train /
# navtrain_val are complete; D3/ca_scales.json exists, was fitted on THAT train pack (pack_sha256 = sha256 of its
# build.json), with source bevfusion and k_obj 96 (review fix D5 / protocol finding 6); arm a: stage pre finished
# (done.json + forecaster.pt).  Before the launch it writes D3/train/<run>.code.tgz (navsim/, tools/, docs/refiner_v3/,
# no __pycache__) and D3/train/<run>.launch.json (sha256 of the tarball, train-code sha, pack / ca_scales shas,
# command, KST time).  Writes D3/train/<run>.{launch.log,pid,pgid}; the run itself goes to D3/train/<run>/ (--resume:
# rerunning the same command continues an interrupted run).  Stop with: kill -TERM -- -$(cat D3/train/<run>.pgid)
set -euo pipefail
ARM=${1:?usage: V3_USER_APPROVED=1 run_step1.sh pre|a|b|d}
case "$ARM" in pre|a|b|d) ;; *) echo "stage must be pre, a, b or d (c is the existing ck2T10)"; exit 2 ;; esac
[ "${V3_USER_APPROVED:-0}" = 1 ] || { echo "refusing: Step 1 needs the user's approval (set V3_USER_APPROVED=1)"; exit 2; }
V3=/workspace/yongjae/SSR-v3
D3=/home/external-user/ssd/yongjae_refiner/ck/v3
PK=$D3/objpack_k96/bevfusion
PY=/venv/ssr/bin/python
ENVP="PYTHONPATH=$V3 PYTHONDONTWRITEBYTECODE=1"
cd "$V3"
if [ "$ARM" = pre ]; then CVD=${V3_GPUS:-0}; else CVD=${V3_GPUS:-0,1}; fi
if [ "$ARM" = pre ]; then
  case "$CVD" in 0|4) ;; *) echo "refusing: V3_GPUS='$CVD' for pre (0 or 4)"; exit 2 ;; esac
else
  case "$CVD" in 0,1|4,5) ;; *) echo "refusing: V3_GPUS='$CVD' (0,1 or 4,5)"; exit 2 ;; esac
fi
GPUS=${CVD//,/ }
for g in $GPUS; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
  [ "$used" -lt 1024 ] || { echo "refusing: GPU $g uses ${used} MiB (needs < 1024)"; exit 3; }
done
for s in navtrain_train navtrain_val; do
  st=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" "$PK/$s/build.json" 2>/dev/null || echo missing)
  [ "$st" = complete ] || { echo "refusing: object pack $s status '$st' (tools/ck/v3/build_obj_pack.py)"; exit 4; }
done
[ -f "$D3/ca_scales.json" ] || { echo "refusing: $D3/ca_scales.json missing (tools/ck/v3/fit_ca_scales.py)"; exit 4; }
env $ENVP $PY -B - "$D3/ca_scales.json" "$PK/navtrain_train/build.json" <<'EOF' || exit 4
import hashlib, json, sys
d = json.load(open(sys.argv[1]))
sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
bad = [k for k, ok in (("pack_sha256", d.get("pack_sha256") == sha), ("source", d.get("source") == "bevfusion"),
                       ("k_obj", d.get("k_obj") == 96), ("augmented fit", bool(d.get("aug")))) if not ok]
if bad:
    sys.exit(f"refusing: ca_scales.json does not match the train pack / rule: {bad}")
EOF
ARGS=$(env $ENVP $PY -B tools/ck/v3/arms.py cli --arm "$ARM" --out-root "$D3/train")
NAME=$(env $ENVP $PY -B -c "import sys; sys.path.insert(0, '$V3'); from tools.ck.v3 import arms; print(arms.load('$ARM')['name'])")
if [ "$ARM" = a ]; then
  PRE=$(env $ENVP $PY -B -c "import sys; sys.path.insert(0, '$V3'); from tools.ck.v3 import arms; print(arms.load('pre')['out'])")
  [ -f "$PRE/done.json" ] && [ -f "$PRE/forecaster.pt" ] || { echo "refusing: stage pre not finished ($PRE)"; exit 4; }
fi
mkdir -p "$D3/train"
TGZ="$D3/train/$NAME.code.tgz"
[ -f "$TGZ" ] || tar czf "$TGZ" --exclude=__pycache__ -C "$V3" navsim tools docs/refiner_v3
env $ENVP $PY -B - "$D3/train/$NAME.launch.json" "$TGZ" "$ARM" "$ARGS" "$CVD" <<'EOF'
import hashlib, json, os, sys, time
sys.path.insert(0, "/workspace/yongjae/SSR-v3")
from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg
from tools.ck.v3 import provenance as PV
out, tgz, arm, args, gpus = sys.argv[1:6]
rec = json.load(open(out)) if os.path.isfile(out) else {"launches": []}
prov = PV.run_provenance(TemporalCfg(on=True, forecaster="ca", n_modes=1), splits=["navtrain_train", "navtrain_val"])
rec["launches"].append(dict(kst=time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime(time.time() + 9 * 3600)), arm=arm,
                            args=args, gpus=gpus, code_tgz=tgz, code_tgz_sha256=PV.sha256_file(tgz), provenance=prov))
json.dump(rec, open(out, "w"), indent=1)
print("[run_step1] train-code sha", prov["code"]["train"]["sha256"][:16], "worktree", prov["code"]["worktree"].get("sha256", "")[:16])
EOF
echo "[run_step1] $(TZ=Asia/Seoul date '+%F %T KST') stage $ARM (GPUs $CVD) -> $D3/train/$NAME"
if [ "$ARM" = pre ]; then
  echo "[run_step1] pretrain_forecaster.py $ARGS"
  # shellcheck disable=SC2086
  setsid nohup env CUDA_VISIBLE_DEVICES="$CVD" PYTHONPATH="$V3" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    nice -n 10 $PY -B tools/ck/v3/pretrain_forecaster.py $ARGS > "$D3/train/$NAME.launch.log" 2>&1 < /dev/null &
else
  echo "[run_step1] train_ck2.py $ARGS"
  # shellcheck disable=SC2086
  setsid nohup env CUDA_VISIBLE_DEVICES="$CVD" PYTHONPATH="$V3" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    nice -n 10 /venv/ssr/bin/torchrun --standalone --nproc_per_node=2 tools/ck/train_ck2.py $ARGS \
    > "$D3/train/$NAME.launch.log" 2>&1 < /dev/null &
fi
PID=$!
echo "$PID" > "$D3/train/$NAME.pid"
sleep 1
ps -o pgid= -p "$PID" | tr -d ' ' > "$D3/train/$NAME.pgid"
echo "[run_step1] pid $PID pgid $(cat "$D3/train/$NAME.pgid") log $D3/train/$NAME.launch.log"
