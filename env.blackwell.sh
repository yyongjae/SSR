# Blackwell (RTX 5090 x6, sm_120) environment for SSR aux_distill.
#   usage:  source env.blackwell.sh   then   ./run.sh <target> <gpus>
#
# run.sh's pick_python only probes envs named 'ssr'. That env exists on this
# box but belongs to a different project and has no mmcv/mmdet3d, so
# SSR_PYTHON must be set explicitly.
export SSR_PYTHON="$HOME/miniconda3/envs/ssr_aux/bin/python"

# Stage-1 adapters and stage-2 student checkpoints land here.
export DISTILL_CKPT_OUT_ROOT="$HOME/byounggun/checkpoints/planning_distill_checkpoints"

# Teacher BEV cache root. The configs ship /data2 and /data3 paths that do not
# exist on this machine; pass this through --cfg-options when running distill.
export DISTILL_FEATURE_ROOT="$HOME/datasets/teacher_cache"

# numba 0.57 probes the host CPU on import; 'generic' is the safe target.
export NUMBA_CPU_NAME=generic

# SSRWandbLoggerHook imports wandb unconditionally; offline needs no login.
export WANDB_MODE="${WANDB_MODE:-offline}"

# 6 GPUs but only 32 logical CPUs, so 8 workers/GPU would oversubscribe.
export SSR_WORKERS_PER_GPU="${SSR_WORKERS_PER_GPU:-4}"

mkdir -p "$DISTILL_CKPT_OUT_ROOT"
echo "SSR env ready: $SSR_PYTHON"
