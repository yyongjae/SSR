#!/usr/bin/env bash
# One entry point for every PARA-SSR run.
#
#   ./run.sh <what> [gpus] [config]
#
#   ./run.sh 12ep 3,6          12-epoch single-stage
#   ./run.sh 60ep 0,1,2,3      60-epoch single-stage
#   ./run.sh 60ep 0,1 projects/configs/SSR/custom.py
#                              60-epoch preset with a custom config
#   ./run.sh staged 3,6        48ep no-planning -> 12ep all
#   ./run.sh staged 3,6 stage1.py stage2.py
#                              staged run with both configs overridden
#   ./run.sh planonly 3,6      control: aux cannot touch the BEV
#
#   ./run.sh smoke 3,6         validation path, 8 samples (~10 min) -- run this
#                              BEFORE committing days to a training run
#   ./run.sh calib 3,6         200 iterations, prints real s/iter and the ETA
#   ./run.sh test              CPU regression suite, no GPU
#   ./run.sh doctor            check this machine: env, GPUs, dataset symlinks
#   ./run.sh eval CKPT [gpu] [config]
#                              final numbers: 1 GPU, sequential, EMA weights
#
# gpus defaults to $CUDA_VISIBLE_DEVICES, or 0,1. For training, the number of
# GPUs must divide 8. The launcher sets samples_per_gpu=8/N automatically, so
# every supported layout keeps the experiment's global batch fixed at 8.
# DataLoader workers default to 8 per GPU; override with
# SSR_WORKERS_PER_GPU=<N> when benchmarking a different host.
set -euo pipefail
cd "$(dirname "$0")"

# Interpreter, in order: $SSR_PYTHON, an already-activated env that can import
# torch and mmcv, then the usual conda locations. Nothing here is specific to
# one machine -- set SSR_PYTHON if the env lives somewhere else.
has_deps() { "$1" -c 'import torch, mmcv, mmdet3d' >/dev/null 2>&1; }
pick_python() {
  [ -n "${SSR_PYTHON:-}" ] && { echo "$SSR_PYTHON"; return; }
  command -v python >/dev/null && has_deps python && { command -v python; return; }
  local p
  for p in "$HOME"/miniconda3/envs/ssr/bin/python \
           "$HOME"/anaconda3/envs/ssr/bin/python \
           "$HOME"/miniforge3/envs/ssr/bin/python \
           /opt/conda/envs/ssr/bin/python; do
    [ -x "$p" ] && has_deps "$p" && { echo "$p"; return; }
  done
  return 1
}
if ! PY=$(pick_python); then
  echo "No interpreter with torch + mmcv + mmdet3d found." >&2
  echo "Activate the env ('conda activate ssr') or set SSR_PYTHON=/path/to/python." >&2
  exit 1
fi
PY_DIR=$(dirname "$PY")
export PATH="$PY_DIR:$PATH"

# numba 0.48/LLVM 8 segfaults probing some newer CPUs (Sapphire Rapids here).
# 'generic' is a safe target everywhere; override if a machine wants otherwise.
export NUMBA_CPU_NAME="${NUMBA_CPU_NAME:-generic}"
export NUMBA_CPU_FEATURES="${NUMBA_CPU_FEATURES-}"
export WANDB_DIR="${WANDB_DIR:-$PWD/work_dirs}"

C=projects/configs/SSR
WHAT=${1:-}
GPUS=${2:-${CUDA_VISIBLE_DEVICES:-0,1}}
NG=$(awk -F, '{print NF}' <<<"$GPUS")
GLOBAL_BATCH=8
WORKERS_PER_GPU=${SSR_WORKERS_PER_GPU:-8}

usage() { sed -n '2,20p' "$0" | sed 's/^# \?//'; exit "${1:-1}"; }
[ -z "$WHAT" ] && usage 0

validate_gpu_list() {
  if [[ ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid GPU list: '$GPUS' (expected e.g. 0,1 or 0,1,2,3)" >&2
    exit 1
  fi
}

prepare_batch() {
  validate_gpu_list
  if (( NG < 1 || GLOBAL_BATCH % NG != 0 )); then
    echo "training needs a GPU count that divides $GLOBAL_BATCH; got $NG ($GPUS)." >&2
    echo "Supported counts: 1, 2, 4, or 8." >&2
    exit 1
  fi
  if [[ ! "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "SSR_WORKERS_PER_GPU must be a positive integer; got '$WORKERS_PER_GPU'." >&2
    exit 1
  fi
  BATCH_PER_GPU=$((GLOBAL_BATCH / NG))
}

# Things that are set up per machine and fail late if missing: the dataset
# symlinks, the annotation pickles, a writable work_dirs. Checked before a run
# rather than after the first epoch of image loading.
preflight() {
  local bad=0 f
  for f in data/nuscenes/vad_nuscenes_infos_temporal_train.pkl \
           data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
           data/nuscenes/nuscenes_map_anns_val_ssr.json \
           data/nuscenes/samples data/nuscenes/maps data/can_bus; do
    [ -e "$f" ] || { echo "missing: $f" >&2; bad=1; }
  done
  mkdir -p work_dirs 2>/dev/null || { echo "work_dirs not writable" >&2; bad=1; }
  [ "$bad" -eq 0 ] || {
    echo "Set the dataset up as in the README (data/nuscenes, data/can_bus may" >&2
    echo "be symlinks) before training." >&2; exit 1; }
}

# Use the preset when no override is supplied. An override may be relative to
# the repository or absolute, but it must name an existing config file.
resolve_config() {
  local preset=$1 override=${2:-} cfg
  cfg=${override:-$C/$preset.py}
  if [ ! -f "$cfg" ]; then
    echo "config not found: $cfg" >&2
    exit 1
  fi
  printf '%s\n' "$cfg"
}

# train <preset-name> <work-dir-name> <config-override> [extra cfg-options...]
train() {
  local preset=$1 wd=$2 override=${3:-} cfg
  shift 3
  cfg=$(resolve_config "$preset" "$override")
  prepare_batch
  preflight
  echo "=== $cfg -> work_dirs/$wd ==="
  echo "    GPU $GPUS: $NG x $BATCH_PER_GPU = global batch $GLOBAL_BATCH"
  CUDA_VISIBLE_DEVICES="$GPUS" PORT="${PORT:-$((28500 + RANDOM % 500))}" \
    ./tools/dist_train.sh "$cfg" "$NG" \
      --work-dir "work_dirs/$wd" --seed 0 \
      --cfg-options data.samples_per_gpu="$BATCH_PER_GPU" \
        data.workers_per_gpu="$WORKERS_PER_GPU" "$@"
}

case "$WHAT" in
  12ep)     train PARA_SSR_e2e_12ep          para_ssr_12ep     "${3:-}" ;;
  60ep)     train PARA_SSR_e2e_60ep          para_ssr_60ep     "${3:-}" ;;
  planonly) train PARA_SSR_e2e_60ep_planonly para_ssr_planonly "${3:-}" ;;
  stage1)   train PARA_SSR_stage1_detmap     para_ssr_stage1   "${3:-}" ;;

  stage2)
    CKPT=${CKPT:-work_dirs/para_ssr_stage1/latest.pth}
    if [ ! -e "$CKPT" ]; then
      echo "stage 2 starts from stage 1's weights, and $CKPT does not exist." >&2
      echo "Run './run.sh stage1 $GPUS' first, or set CKPT=<path>." >&2
      exit 1
    fi
    # The config names a default checkpoint; override it so a non-default CKPT
    # is actually honoured instead of silently ignored.
    train PARA_SSR_stage2_all para_ssr_stage2 "${3:-}" load_from="$CKPT"
    ;;

  staged)
    "$0" stage1 "$GPUS" "${3:-}"
    CKPT=work_dirs/para_ssr_stage1/latest.pth \
      "$0" stage2 "$GPUS" "${4:-}"
    ;;

  smoke)
    validate_gpu_list
    if (( 8 % NG != 0 )); then
      echo "smoke uses 8 samples, so the GPU count must divide 8; got $NG." >&2
      exit 1
    fi
    echo "=== validation path, 8 samples. Catches an epoch-6 crash now, not in five days. ==="
    CFG=$(resolve_config PARA_SSR_e2e_60ep "${3:-}")
    tools/verify_dist_eval.sh 8 "$CFG" "$GPUS"
    ;;

  calib)
    prepare_batch
    CFG=$(resolve_config PARA_SSR_e2e_60ep "${3:-}")
    OUT=$(mktemp -d)
    echo "=== 200 iterations to measure the real s/iter (~15 min) ==="
    CUDA_VISIBLE_DEVICES="$GPUS" PORT="${PORT:-$((28500 + RANDOM % 500))}" \
      timeout 2400 ./tools/dist_train.sh \
        "$CFG" "$NG" --work-dir "$OUT" --seed 0 \
        --no-validate --cfg-options data.samples_per_gpu="$BATCH_PER_GPU" \
          data.workers_per_gpu="$WORKERS_PER_GPU" log_config.interval=20 \
        2>&1 | tee "$OUT/log"
    python - "$OUT/log" <<'EOF'
import re, sys
t = [float(m) for m in re.findall(r'time: ([0-9.]+)', open(sys.argv[1]).read())]
d = [float(m) for m in re.findall(r'data_time: ([0-9.]+)', open(sys.argv[1]).read())]
if len(t) < 3:
    print('not enough iterations logged'); raise SystemExit
t, d = t[2:], d[2:]                      # drop warm-up
s = sum(t) / len(t)
print(f'\n  {len(t)} iterations   {s:.3f} s/iter   '
      f'(data {sum(d)/len(d):.3f} s = {100*sum(d)/sum(t):.0f}%)')
for ep in (48, 60):
    h = s * 3516 * ep / 3600
    print(f'  {ep} epochs -> {h:.1f} h = {h/24:.1f} days  (+ ~20 min per eval)')
EOF
    rm -rf "$OUT"
    ;;

  eval)
    CKPT=${2:?usage: ./run.sh eval CKPT [gpu]}
    CFG=$(resolve_config PARA_SSR_e2e_60ep "${4:-}")
    tools/final_eval.sh "$CFG" "$CKPT" "${3:-0}"
    ;;

  test)
    fail=0
    for t in verify_diagnostics verify_anomaly_hook verify_grad_balance \
             verify_aux_metrics verify_multibatch_and_metrics \
             verify_wandb_logger; do
      printf '%-32s ' "$t"
      if CUDA_VISIBLE_DEVICES="" python "tools/$t.py" >/dev/null 2>&1; then
        echo ok
      else
        echo FAIL; fail=1
      fi
    done
    exit $fail
    ;;

  doctor)
    echo "python  : $PY"
    "$PY" - <<'EOF'
import torch, mmcv, mmdet3d
print(f'  torch {torch.__version__}  cuda {torch.version.cuda}  '
      f'mmcv {mmcv.__version__}  mmdet3d {mmdet3d.__version__}')
print(f'  visible GPUs: {torch.cuda.device_count()}'
      + (f'  ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))
EOF
    echo "numba   : NUMBA_CPU_NAME=$NUMBA_CPU_NAME"
    echo "dataset :"
    preflight && echo "  ok"
    ;;

  *) echo "unknown: $WHAT" >&2; usage ;;
esac
