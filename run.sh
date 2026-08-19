#!/usr/bin/env bash
# One entry point for every PARA-SSR run.
#
#   ./run.sh <what> [gpus]
#
#   ./run.sh 60ep 3,6          60-epoch single-stage      (2 GPU, ~3-5 days)
#   ./run.sh staged 3,6        48ep no-planning -> 12ep all  (2 GPU)
#   ./run.sh planonly 3,6      control: aux cannot touch the BEV  (2 GPU)
#   ./run.sh 4gpu 0,1,2,3      60ep on four GPUs, same batch, half the time
#
#   ./run.sh smoke 3,6         validation path, 8 samples (~10 min) -- run this
#                              BEFORE committing days to a training run
#   ./run.sh calib 3,6         200 iterations, prints real s/iter and the ETA
#   ./run.sh test              CPU regression suite, no GPU
#   ./run.sh doctor            check this machine: env, GPUs, dataset symlinks
#   ./run.sh eval CKPT [gpu]   final numbers: 1 GPU, sequential, EMA weights
#
# gpus defaults to $CUDA_VISIBLE_DEVICES, or 0,1. The GPU count has to match
# what the config expects -- every config fixes samples_per_gpu so that
# gpus x samples = the global batch of 8, and changing one without the other
# silently changes the experiment.
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
export PATH="$(dirname "$PY"):$PATH"

# numba 0.48/LLVM 8 segfaults probing some newer CPUs (Sapphire Rapids here).
# 'generic' is a safe target everywhere; override if a machine wants otherwise.
export NUMBA_CPU_NAME="${NUMBA_CPU_NAME:-generic}"
export NUMBA_CPU_FEATURES="${NUMBA_CPU_FEATURES-}"
export WANDB_DIR="${WANDB_DIR:-$PWD/work_dirs}"

C=projects/configs/SSR
WHAT=${1:-}
GPUS=${2:-${CUDA_VISIBLE_DEVICES:-0,1}}
NG=$(awk -F, '{print NF}' <<<"$GPUS")

usage() { sed -n '2,20p' "$0" | sed 's/^# \?//'; exit "${1:-1}"; }
[ -z "$WHAT" ] && usage 0

need_gpus() {
  [ "$NG" -eq "$1" ] && return 0
  echo "'$WHAT' needs $1 GPU(s); got $NG ($GPUS)." >&2
  echo "Every config pins samples_per_gpu so gpus x samples = 8. Changing the" >&2
  echo "GPU count alone changes the global batch and invalidates the LR." >&2
  exit 1
}

# Things that are set up per machine and fail late if missing: the dataset
# symlinks, the annotation pickles, a writable work_dirs. Checked before a run
# rather than after the first epoch of image loading.
preflight() {
  local bad=0 f
  for f in data/nuscenes/vad_nuscenes_infos_temporal_train.pkl \
           data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
           data/nuscenes/nuscenes_map_anns_val.json \
           data/nuscenes/samples data/nuscenes/maps data/can_bus; do
    [ -e "$f" ] || { echo "missing: $f" >&2; bad=1; }
  done
  mkdir -p work_dirs 2>/dev/null || { echo "work_dirs not writable" >&2; bad=1; }
  [ "$bad" -eq 0 ] || {
    echo "Set the dataset up as in the README (data/nuscenes, data/can_bus may" >&2
    echo "be symlinks) before training." >&2; exit 1; }
}

# train <config-name> <work-dir-name> <n-gpus> [extra args...]
train() {
  local cfg=$1 wd=$2 n=$3; shift 3
  need_gpus "$n"
  preflight
  echo "=== $cfg -> work_dirs/$wd  (GPU $GPUS) ==="
  CUDA_VISIBLE_DEVICES="$GPUS" PORT="${PORT:-$((28500 + RANDOM % 500))}" \
    ./tools/dist_train.sh "$C/$cfg.py" "$n" \
      --work-dir "work_dirs/$wd" --seed 0 "$@"
}

case "$WHAT" in
  60ep)     train PARA_SSR_e2e_2gpu_b4_60ep          para_ssr_60ep      2 ;;
  4gpu)     train PARA_SSR_e2e_4gpu_60ep             para_ssr_4gpu_60ep 4 ;;
  planonly) train PARA_SSR_e2e_2gpu_b4_60ep_planonly para_ssr_planonly  2 ;;
  stage1)   train PARA_SSR_stage1_detmap             para_ssr_stage1    2 ;;

  stage2)
    CKPT=${CKPT:-work_dirs/para_ssr_stage1/latest.pth}
    if [ ! -e "$CKPT" ]; then
      echo "stage 2 starts from stage 1's weights, and $CKPT does not exist." >&2
      echo "Run './run.sh stage1 $GPUS' first, or set CKPT=<path>." >&2
      exit 1
    fi
    # The config names a default checkpoint; override it so a non-default CKPT
    # is actually honoured instead of silently ignored.
    train PARA_SSR_stage2_all para_ssr_stage2 2 --cfg-options load_from="$CKPT"
    ;;

  staged)
    "$0" stage1 "$GPUS"
    CKPT=work_dirs/para_ssr_stage1/latest.pth "$0" stage2 "$GPUS"
    ;;

  smoke)
    need_gpus 2
    echo "=== validation path, 8 samples. Catches an epoch-6 crash now, not in five days. ==="
    tools/verify_dist_eval.sh 8 "$C/PARA_SSR_e2e_2gpu_b4_60ep.py" "$GPUS"
    ;;

  calib)
    need_gpus 2
    OUT=$(mktemp -d)
    echo "=== 200 iterations to measure the real s/iter (~15 min) ==="
    CUDA_VISIBLE_DEVICES="$GPUS" PORT="${PORT:-$((28500 + RANDOM % 500))}" \
      timeout 2400 ./tools/dist_train.sh \
        "$C/PARA_SSR_e2e_2gpu_b4_60ep.py" 2 --work-dir "$OUT" --seed 0 \
        --no-validate --cfg-options log_config.interval=20 2>&1 | tee "$OUT/log"
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
    tools/final_eval.sh "$C/PARA_SSR_e2e_2gpu_b4_60ep.py" "$CKPT" "${3:-0}"
    ;;

  test)
    fail=0
    for t in verify_diagnostics verify_anomaly_hook verify_grad_balance \
             verify_aux_metrics verify_multibatch_and_metrics; do
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
