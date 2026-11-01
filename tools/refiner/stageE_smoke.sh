#!/usr/bin/env bash
# Stage E CPU smoke: a few real PARA-SSR training steps (Lightning, run_training.py) with the student refiner.
#   bash tools/refiner/stageE_smoke.sh E1|E2|off  [extra hydra overrides]
# Env: OUT (NAVSIM_EXP_ROOT, default scratch), TEACHERS (dir holding stageT4_{T,M}_fold0_seed0 snapshots, E2 only).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:?E1|E2|off}"; shift || true
PY="${SSR_NAVSIM_PYTHON:-/home/external-user/miniconda3/envs/ssr/bin/python}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" NUPLAN_MAPS_ROOT="${REPO}/data/dataset/maps"
export OPENSCENE_DATA_ROOT="${REPO}/data/dataset" NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT="${OUT:-/tmp/stageE_smoke}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TEACHERS="${TEACHERS:-/home/external-user/ssd/yongjae_refiner/stageE/teachers}"
# 2 tokens with every GT store, 2 without objects (GT-missing path), 2 more of the same logs
TOKENS='[1aa44d46e4ab5bc7,153c6b07f09d53d1,cc9b708a380b5a8a,aa43f9c9b3c455ae,2570fbfdf1835706,c59175106e2f5b26]'
EXTRA=()
if [[ "${MODE}" == "E2" ]]; then
  EXTRA+=("agent.config.kd_teacher_runs=[${TEACHERS}/stageT4_T_fold0_seed0,${TEACHERS}/stageT4_M_fold0_seed0]"
          "agent.config.kd_lambda=1.0" "agent.config.kd_ramp=[0.0,0.0]" "agent.config.kd_space=${KD_SPACE:-decoded}")
fi
exec "${PY}" "${REPO}/navsim/planning/script/run_training.py" \
  agent=para_ssr_agent agent.lr=1e-4 agent.config.max_epochs=1 agent.config.warmup_epochs=1 \
  agent.config.backbone_pretrained=false agent.config.refiner_mode="${MODE}" \
  agent.config.grad_norm_log_interval=2 \
  experiment_name="stageE_smoke_${MODE}" scene_filter=navtrain split=trainval \
  "scene_filter.tokens=${TOKENS}" \
  dataloader.params.batch_size=2 dataloader.params.num_workers=2 \
  trainer.params.max_epochs=1 trainer.params.accumulate_grad_batches=1 \
  trainer.params.accelerator=cpu +trainer.params.devices=1 trainer.params.strategy=auto \
  trainer.params.precision=32 trainer.params.gradient_clip_val=35.0 trainer.params.gradient_clip_algorithm=norm \
  trainer.params.limit_val_batches=0 trainer.params.num_sanity_val_steps=0 \
  wandb.enable=false "${EXTRA[@]}" "$@"
