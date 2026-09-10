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
#   ./run.sh teacher-bevdepth 0,1
#                              train BEVDepth adapter + planner for 6 epochs
#   ./run.sh teacher-hdmapnet 2,3
#                              train HDMapNet adapter + planner for 6 epochs
#   ./run.sh teacher-bevfusion 0
#                              train BEVFusion adapter + planner for 6 epochs
#   ./run.sh teacher-maptrv2 1
#                              train MapTRv2 adapter + planner for 6 epochs
#   ./run.sh teacher-adapters 0,1
#                              legacy joint two-teacher adapter training
#   ./run.sh distill 3,6       planning-only SSR student + feature distillation
#   ./run.sh distill-bevfusion-maptr 3,6
#                              same, from frozen BEVFusion + MapTRv2 adapters
#   ./run.sh rped-teacher 0,1  stage-1 RPED readout planner (BEVFusion+MapTRv2)
#   ./run.sh rped-teacher-pair-a 0,1
#                              same, on BEVDepth+HDMapNet 25x25 caches
#   ./run.sh rped-distill 3,6  stage-2 RPED student (rank-N evidence, no dense MSE)
#   ./run.sh rped-same-question 3,6
#   ./run.sh rped-triplet 3,6
#   ./run.sh rped-shuffle 3,6  negative control: shuffled teacher memory
#   ./run.sh rped-dense 3,6    punchline ablation: dense adapter-MSE on pair B
#   ./run.sh eval-rped-teacher [CKPT] [gpu]
#                              stage-1 privileged planner L2 / collision gate
#   ./run.sh eval-rped-teacher-pair-a [CKPT] [gpu]
#
#   ./run.sh smoke 3,6         validation path, 8 samples (~10 min) -- run this
#                              BEFORE committing days to a training run
#   ./run.sh calib 3,6         200 iterations, prints real s/iter and the ETA
#   ./run.sh test              CPU regression suite, no GPU
#   ./run.sh doctor            check this machine: env, GPUs, dataset symlinks
#   ./run.sh eval CKPT [gpu] [config]
#                              final numbers: 1 GPU, sequential, EMA weights
#   ./run.sh eval-teacher-bevdepth [CKPT] [gpu]
#   ./run.sh eval-teacher-hdmapnet [CKPT] [gpu]
#   ./run.sh eval-teacher-bevfusion [CKPT] [gpu]
#   ./run.sh eval-teacher-maptrv2 [CKPT] [gpu]
#                              stage-1 adapter/planner L2 evaluation
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
DISTILL_CKPT_OUT_ROOT=${DISTILL_CKPT_OUT_ROOT:-/data2/byounggun/rideflux/pretrained_checkpoints/planning_distill_checkpoints}
DISTILL_STUDENT_WORK_DIR=${DISTILL_STUDENT_WORK_DIR:-$DISTILL_CKPT_OUT_ROOT/student}
4
usage() { sed -n '2,57p' "$0" | sed 's/^# \?//'; exit "${1:-1}"; }
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
  local preset=$1 wd=$2 override=${3:-} cfg work_dir
  shift 3
  cfg=$(resolve_config "$preset" "$override")
  prepare_batch
  preflight
  if [[ "$wd" = /* ]]; then
    work_dir=$wd
  else
    work_dir=work_dirs/$wd
  fi
  mkdir -p "$work_dir" || {
    echo "checkpoint directory is not writable: $work_dir" >&2
    exit 1
  }
  echo "=== $cfg -> $work_dir ==="
  echo "    GPU $GPUS: $NG x $BATCH_PER_GPU = global batch $GLOBAL_BATCH"
  CUDA_VISIBLE_DEVICES="$GPUS" PORT="${PORT:-$((28500 + RANDOM % 500))}" \
    ./tools/dist_train.sh "$cfg" "$NG" \
      --work-dir "$work_dir" --seed 0 \
      --cfg-options data.samples_per_gpu="$BATCH_PER_GPU" \
        data.workers_per_gpu="$WORKERS_PER_GPU" "$@"
}

case "$WHAT" in
  12ep)     train PARA_SSR_e2e_12ep          para_ssr_12ep     "${3:-}" ;;
  60ep)     train PARA_SSR_e2e_60ep          para_ssr_60ep     "${3:-}" ;;
  planonly) train PARA_SSR_e2e_60ep_planonly para_ssr_planonly "${3:-}" ;;
  teacher-bevdepth)
    train DISTILL_teacher_bevdepth \
      "$DISTILL_CKPT_OUT_ROOT/teacher_bevdepth" "${3:-}"
    ;;
  teacher-hdmapnet)
    train DISTILL_teacher_hdmapnet \
      "$DISTILL_CKPT_OUT_ROOT/teacher_hdmapnet" "${3:-}"
    ;;
  teacher-adapters)
    train DISTILL_teacher_adapters \
      "$DISTILL_CKPT_OUT_ROOT/teacher_joint" "${3:-}"
    ;;
  teacher-bevfusion)
    train DISTILL_teacher_bevfusion \
      "$DISTILL_CKPT_OUT_ROOT/teacher_bevfusion" "${3:-}"
    ;;
  teacher-maptrv2)
    train DISTILL_teacher_maptrv2 \
      "$DISTILL_CKPT_OUT_ROOT/teacher_maptrv2" "${3:-}"
    ;;

  distill)
    if [ -n "${ADAPTER_CKPT:-}" ]; then
      if [ ! -e "$ADAPTER_CKPT" ]; then
        echo "student distillation needs $ADAPTER_CKPT." >&2
        exit 1
      fi
      # Backward-compatible path for a legacy joint stage-1 checkpoint.
      train DISTILL_SSR_student "$DISTILL_STUDENT_WORK_DIR" "${3:-}" \
        model.distill.adapter_checkpoint="$ADAPTER_CKPT"
    else
      BEVDEPTH_ADAPTER_CKPT=${BEVDEPTH_ADAPTER_CKPT:-$DISTILL_CKPT_OUT_ROOT/teacher_bevdepth/epoch_6.pth}
      HDMAPNET_ADAPTER_CKPT=${HDMAPNET_ADAPTER_CKPT:-$DISTILL_CKPT_OUT_ROOT/teacher_hdmapnet/epoch_6.pth}
      missing=0
      for checkpoint in "$BEVDEPTH_ADAPTER_CKPT" "$HDMAPNET_ADAPTER_CKPT"; do
        if [ ! -e "$checkpoint" ]; then
          echo "student distillation needs $checkpoint." >&2
          missing=1
        fi
      done
      if [ "$missing" -ne 0 ]; then
        echo "Run './run.sh teacher-bevdepth 0,1' and " \
             "'./run.sh teacher-hdmapnet 2,3' first." >&2
        exit 1
      fi
      train DISTILL_SSR_student "$DISTILL_STUDENT_WORK_DIR" "${3:-}" \
        model.distill.adapter_checkpoint.bevdepth="$BEVDEPTH_ADAPTER_CKPT" \
        model.distill.adapter_checkpoint.hdmapnet="$HDMAPNET_ADAPTER_CKPT"
    fi
    ;;

  distill-bevfusion-maptr)
    BEVFUSION_ADAPTER_CKPT=${BEVFUSION_ADAPTER_CKPT:-$DISTILL_CKPT_OUT_ROOT/teacher_bevfusion/epoch_6.pth}
    MAPTRV2_ADAPTER_CKPT=${MAPTRV2_ADAPTER_CKPT:-$DISTILL_CKPT_OUT_ROOT/teacher_maptrv2/epoch_6.pth}
    FUSION_STUDENT_WORK_DIR=${DISTILL_FUSION_STUDENT_WORK_DIR:-$DISTILL_CKPT_OUT_ROOT/student_bevfusion_maptrv2}
    missing=0
    for checkpoint in "$BEVFUSION_ADAPTER_CKPT" "$MAPTRV2_ADAPTER_CKPT"; do
      if [ ! -e "$checkpoint" ]; then
        echo "student distillation needs $checkpoint." >&2
        missing=1
      fi
    done
    if [ "$missing" -ne 0 ]; then
      echo "Run './run.sh teacher-bevfusion 0' and " \
           "'./run.sh teacher-maptrv2 1' first." >&2
      exit 1
    fi
    train DISTILL_SSR_student_bevfusion_maptrv2 "$FUSION_STUDENT_WORK_DIR" "${3:-}" \
      model.distill.adapter_checkpoint.bevfusion="$BEVFUSION_ADAPTER_CKPT" \
      model.distill.adapter_checkpoint.maptrv2="$MAPTRV2_ADAPTER_CKPT"
    ;;

  rped-teacher)
    train RPED_teacher_bevfusion_maptrv2 \
      "$DISTILL_CKPT_OUT_ROOT/rped_teacher_bevfusion_maptrv2" "${3:-}"
    ;;
  rped-teacher-pair-a)
    train RPED_teacher_bevdepth_hdmapnet \
      "$DISTILL_CKPT_OUT_ROOT/rped_teacher_bevdepth_hdmapnet" "${3:-}"
    ;;

  rped-distill)
    RPED_TEACHER_CKPT=${RPED_TEACHER_CKPT:-$DISTILL_CKPT_OUT_ROOT/rped_teacher_bevfusion_maptrv2/epoch_6.pth}
    RPED_STUDENT_WORK_DIR=${RPED_STUDENT_WORK_DIR:-$DISTILL_CKPT_OUT_ROOT/rped_student}
    if [ ! -e "$RPED_TEACHER_CKPT" ]; then
      echo "RPED student needs $RPED_TEACHER_CKPT." >&2
      echo "Run './run.sh rped-teacher 0,1' first, then eval-rped-teacher." >&2
      exit 1
    fi
    train RPED_SSR_student "$RPED_STUDENT_WORK_DIR" "${3:-}" \
      model.distill.readout_checkpoint="$RPED_TEACHER_CKPT"
    ;;
  rped-same-question)
    RPED_TEACHER_CKPT=${RPED_TEACHER_CKPT:-$DISTILL_CKPT_OUT_ROOT/rped_teacher_bevfusion_maptrv2/epoch_6.pth}
    if [ ! -e "$RPED_TEACHER_CKPT" ]; then
      echo "RPED student needs $RPED_TEACHER_CKPT." >&2
      exit 1
    fi
    train RPED_SSR_student_same_question \
      "$DISTILL_CKPT_OUT_ROOT/rped_student_same_question" "${3:-}" \
      model.distill.readout_checkpoint="$RPED_TEACHER_CKPT"
    ;;
  rped-triplet)
    RPED_TEACHER_CKPT=${RPED_TEACHER_CKPT:-$DISTILL_CKPT_OUT_ROOT/rped_teacher_bevfusion_maptrv2/epoch_6.pth}
    if [ ! -e "$RPED_TEACHER_CKPT" ]; then
      echo "RPED student needs $RPED_TEACHER_CKPT." >&2
      exit 1
    fi
    train RPED_SSR_student_triplet \
      "$DISTILL_CKPT_OUT_ROOT/rped_student_triplet" "${3:-}" \
      model.distill.readout_checkpoint="$RPED_TEACHER_CKPT"
    ;;
  rped-shuffle)
    RPED_TEACHER_CKPT=${RPED_TEACHER_CKPT:-$DISTILL_CKPT_OUT_ROOT/rped_teacher_bevfusion_maptrv2/epoch_6.pth}
    if [ ! -e "$RPED_TEACHER_CKPT" ]; then
      echo "RPED student needs $RPED_TEACHER_CKPT." >&2
      exit 1
    fi
    train RPED_SSR_student_shuffle \
      "$DISTILL_CKPT_OUT_ROOT/rped_student_shuffle" "${3:-}" \
      model.distill.readout_checkpoint="$RPED_TEACHER_CKPT"
    ;;
  rped-dense)
    BEVFUSION_ADAPTER_CKPT=${BEVFUSION_ADAPTER_CKPT:-$DISTILL_CKPT_OUT_ROOT/teacher_bevfusion/epoch_6.pth}
    MAPTRV2_ADAPTER_CKPT=${MAPTRV2_ADAPTER_CKPT:-$DISTILL_CKPT_OUT_ROOT/teacher_maptrv2/epoch_6.pth}
    missing=0
    for checkpoint in "$BEVFUSION_ADAPTER_CKPT" "$MAPTRV2_ADAPTER_CKPT"; do
      if [ ! -e "$checkpoint" ]; then
        echo "dense ablation needs $checkpoint." >&2
        missing=1
      fi
    done
    if [ "$missing" -ne 0 ]; then
      echo "Run './run.sh teacher-bevfusion 0' and " \
           "'./run.sh teacher-maptrv2 1' first. Dense MSE is an ablation, " \
           "not the RPED method." >&2
      exit 1
    fi
    train RPED_SSR_student_dense_ablation \
      "$DISTILL_CKPT_OUT_ROOT/rped_student_dense" "${3:-}" \
      model.distill.adapter_checkpoint.bevfusion="$BEVFUSION_ADAPTER_CKPT" \
      model.distill.adapter_checkpoint.maptrv2="$MAPTRV2_ADAPTER_CKPT"
    ;;
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

  eval-teacher-bevdepth)
    CKPT=${2:-$DISTILL_CKPT_OUT_ROOT/teacher_bevdepth/epoch_6.pth}
    EXPECT_RAW=1 tools/final_eval.sh \
      "$C/DISTILL_teacher_bevdepth.py" "$CKPT" "${3:-0}"
    ;;

  eval-teacher-hdmapnet)
    CKPT=${2:-$DISTILL_CKPT_OUT_ROOT/teacher_hdmapnet/epoch_6.pth}
    EXPECT_RAW=1 tools/final_eval.sh \
      "$C/DISTILL_teacher_hdmapnet.py" "$CKPT" "${3:-2}"
    ;;

  eval-teacher-bevfusion)
    CKPT=${2:-$DISTILL_CKPT_OUT_ROOT/teacher_bevfusion/epoch_6.pth}
    EXPECT_RAW=1 tools/final_eval.sh \
      "$C/DISTILL_teacher_bevfusion.py" "$CKPT" "${3:-0}"
    ;;

  eval-teacher-maptrv2)
    CKPT=${2:-$DISTILL_CKPT_OUT_ROOT/teacher_maptrv2/epoch_6.pth}
    EXPECT_RAW=1 tools/final_eval.sh \
      "$C/DISTILL_teacher_maptrv2.py" "$CKPT" "${3:-1}"
    ;;

  eval-rped-teacher)
    CKPT=${2:-$DISTILL_CKPT_OUT_ROOT/rped_teacher_bevfusion_maptrv2/epoch_6.pth}
    EXPECT_RAW=1 tools/final_eval.sh \
      "$C/RPED_teacher_bevfusion_maptrv2.py" "$CKPT" "${3:-0}"
    ;;

  eval-rped-teacher-pair-a)
    CKPT=${2:-$DISTILL_CKPT_OUT_ROOT/rped_teacher_bevdepth_hdmapnet/epoch_6.pth}
    EXPECT_RAW=1 tools/final_eval.sh \
      "$C/RPED_teacher_bevdepth_hdmapnet.py" "$CKPT" "${3:-0}"
    ;;

  test)
    fail=0
    for t in verify_diagnostics verify_anomaly_hook verify_grad_balance \
             verify_aux_metrics verify_multibatch_and_metrics \
             verify_wandb_logger verify_planning_distill; do
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
