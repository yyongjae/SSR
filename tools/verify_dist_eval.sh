#!/usr/bin/env bash
# Two-rank smoke test of the real validation path. See verify_dist_eval.py.
#
#   tools/verify_dist_eval.sh [N_SAMPLES] [CONFIG] [GPUS]
#
# Random weights, so the numbers are noise -- this checks that epoch 10 of a
# long run reaches the end of dataset.evaluate() instead of crashing there.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-8}
CONFIG=${2:-projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py}
GPUS=${3:-0,1}
NRANKS=$(awk -F, '{print NF}' <<<"$GPUS")
PORT=${PORT:-$((29000 + RANDOM % 1000))}

echo "config : $CONFIG"
echo "samples: $N over $NRANKS ranks (GPU $GPUS)"

CUDA_VISIBLE_DEVICES="$GPUS" N_SAMPLES="$N" \
python -m torch.distributed.launch \
  --nproc_per_node="$NRANKS" --master_port="$PORT" \
  tools/verify_dist_eval.py "$CONFIG" --launcher pytorch
