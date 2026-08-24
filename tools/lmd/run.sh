#!/usr/bin/env bash
# LMD BEV analysis -- runs on GPU 0 and 1 (2 and 3 are busy).
#
#   bash tools/lmd/run.sh e0          # the gate: is the real model exactly affine
#   bash tools/lmd/run.sh e1 200      # planning-centric BEV maps, 200 val samples
#   bash tools/lmd/run.sh all 200
#
# Each job is single-GPU, so the four conditions are sharded two per GPU and run
# in parallel. Logs land in out/lmd/logs/, results in out/lmd/*.npz.
set -uo pipefail

REPO=/home/byounggun/SSR
OUT=$REPO/out/lmd
GPUS=(0 1)

cd "$REPO"
mkdir -p "$OUT/logs"
eval "$(conda shell.bash hook)"
conda activate ssr
export PYTHONPATH="$REPO:$REPO/tools/lmd:${PYTHONPATH:-}"
# nuScenes lives at /data/nuscenes; the configs all say data/nuscenes/
[ -L data/nuscenes ] || { mkdir -p data && ln -sfn /data/nuscenes data/nuscenes; }

STAGE=${1:-all}
N=${2:-200}

# condition -> extra args.  aux_only has no trained planner of its own (plan=0
# for 48 epochs), so it is read with stage2's planner -- legitimate only because
# stage2 forked from it.  See report #10 section 2.
declare -A EXTRA=(
  [plan_only]=""
  [both]=""
  [staged]=""
  [aux_only]="--plan-from staged"
)
CONDS=(plan_only both staged aux_only)

run_shard () {              # $1 = stage, $2 = extra python args
  local stage=$1 extra=$2 rc=0 pids=() names=()
  local i=0
  for c in "${CONDS[@]}"; do
    local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    local log="$OUT/logs/${stage}_${c}.log"
    echo "  [gpu $gpu] $stage $c  -> $log"
    CUDA_VISIBLE_DEVICES=$gpu python "tools/lmd/run_${stage}.py" \
        --ckpt "$c" --device cuda:0 ${EXTRA[$c]} ${extra//COND/$c} > "$log" 2>&1 &
    pids+=($!); names+=("$c"); i=$((i + 1))
    # two per GPU: let the first wave finish before launching the second
    if (( i % ${#GPUS[@]} == 0 )); then
      for j in "${!pids[@]}"; do
        wait "${pids[$j]}" || { echo "  FAILED: ${names[$j]}"; rc=1; }
      done
      pids=(); names=()
    fi
  done
  for j in "${!pids[@]}"; do
    wait "${pids[$j]}" || { echo "  FAILED: ${names[$j]}"; rc=1; }
  done
  return $rc
}

if [[ "$STAGE" == "e0" || "$STAGE" == "all" ]]; then
  echo "=== E0  gate: is the real BEV -> trajectory path exactly affine? ==="
  if run_shard e0 "--samples 3"; then
    touch "$OUT/.e0_passed"
    echo "=== E0 PASSED ==="
    grep -h "worst\|PASS\|FAIL" "$OUT"/logs/e0_*.log
  else
    rm -f "$OUT/.e0_passed"
    echo "=== E0 FAILED -- stop here. Nothing downstream is meaningful until the"
    echo "    residual is small. Check the per-condition logs in $OUT/logs/ ==="
    exit 1
  fi
fi

if [[ "$STAGE" == "e1" || "$STAGE" == "all" ]]; then
  [ -f "$OUT/.e0_passed" ] || {
    echo "E0 has not passed. Run: bash tools/lmd/run.sh e0"; exit 1; }
  echo
  echo "=== E1/E2  planning-centric BEV maps ($N samples per condition) ==="
  run_shard e1 "--samples $N --out $OUT/e1_COND.npz" || exit 1
fi

echo
echo "results: $OUT"
