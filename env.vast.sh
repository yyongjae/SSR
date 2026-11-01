# Vast 8x RTX4090 environment for scripts/training/run_bev_selector_distill.sh
#   usage:  source env.vast.sh && bash scripts/training/run_bev_selector_distill.sh
# data/dataset -> symlink tree over /workspace/navsim_workspace/dataset (see docs/bev_selector_v2.md)
export PYTHON=/root/miniconda3/envs/ssr/bin/python
export DISTILL_FEATURE_ROOT=/workspace/teacher_cache
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# W&B on by default (key read from .env); WANDB=0 to disable
export WANDB="${WANDB:-1}"
# Outputs (hydra dir, lightning_logs/version_N/checkpoints, W&B counter) land in
# ${NAVSIM_EXP_ROOT_OVERRIDE}/<experiment_name>/. The script otherwise hardcodes <repo>/work_dirs.
export NAVSIM_EXP_ROOT_OVERRIDE=/workspace/byounggun/ssr_outputs
mkdir -p "${NAVSIM_EXP_ROOT_OVERRIDE}"
