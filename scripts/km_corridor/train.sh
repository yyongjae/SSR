#!/usr/bin/env bash
# km 10/16: v2 r34 recipe + corridor head, 30 epochs from scratch, mid-training val every 5 epochs (val = evaluation only).
#   usage: train.sh <variant> <run name> <gpus e.g. 0,1> [extra hydra overrides...]
#   variant  v2   : + corridor head only (static SDF + 8 x 0.5 s swept occupancy)          == corr30_v2 (navtest 89.30 / 88.57)
#            v31  : + 40 x 0.1 s planes, plan constraint (static / occ / TTC), conditional EP, route channel,
#                   progress reward inside the GT-safe corridor, official-comfort hinge, verdict supervision  == corr30_v31
#            v31_64 : v31 on the 64 m BEV / GT (NOT smoke-tested yet, see README)
#   smoke: EPOCHS=1 LIMIT_TRAIN=40 LIMIT_VAL=5 VAL_EVERY=1 WANDB=false
#   SELECT_RULE (km 10/17): trajectory the agent submits at inference, default pred_multiply_t0.25 (corridor); v2 = no corridor
set -uo pipefail
source "$(dirname "$0")/env.sh"; cd $REPO
VAR=$1; RUN=$2; GPUS=$3; shift 3
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
OUT=$KM_ROOT/km_runs/corr30/$RUN; mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=$GPUS CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_SOCKET_IFNAME=lo OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
V31=(++agent.config.corridor_plan_weight=20.0 ++agent.config.corridor_plan_ttc_weight=1.0 ++agent.config.plan_ep_feasible_only=true
     ++agent.config.corridor_occ_planes=40 ++agent.config.corridor_route_weight=1.0
     ++agent.config.corridor_plan_prog_weight=0.2 ++agent.config.corridor_plan_comfort_weight=1.0 ++agent.config.corridor_verdict_weight=1.0)
case $VAR in
  v2)  EXTRA=(++agent.config.corridor_dir=$KM_ROOT/corridor_dt01) ;;
  v31) EXTRA=("${V31[@]}" ++agent.config.corridor_dir=$KM_ROOT/corridor_dt01 ++agent.config.corridor_route_dir=$KM_ROOT/route32) ;;
  v31_64) EXTRA=("${V31[@]}" ++agent.config.corridor_format=c64 ++agent.config.corridor_dir=$KM_ROOT/corridor64
           "++agent.config.pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]" "++agent.config.map_pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]"
           ++agent.config.bev_h=100) ;;
  *) echo "variant: v2 | v31 | v31_64"; exit 2 ;;
esac
exec $PY navsim/planning/script/run_training.py \
  agent=para_ssr_agent agent.lr=1e-4 agent.config.max_epochs=30 experiment_name=$RUN \
  scene_filter=navtrain split=trainval \
  dataloader.params.batch_size=4 dataloader.params.num_workers=5 \
  trainer.params.max_epochs=${EPOCHS:-30} trainer.params.limit_train_batches=${LIMIT_TRAIN:-1.0} trainer.params.accumulate_grad_batches=$((32 / NG)) \
  trainer.params.check_val_every_n_epoch=${VAL_EVERY:-5} trainer.params.limit_val_batches=${LIMIT_VAL:-250} trainer.params.num_sanity_val_steps=0 \
  trainer.params.precision=32 +trainer.params.devices=$NG trainer.params.strategy=ddp \
  trainer.params.gradient_clip_val=35.0 trainer.params.gradient_clip_algorithm=norm \
  agent.config.use_task_interaction=true agent.config.use_det_motion_head=true agent.config.use_map_head=true \
  '~agent.config.grad_balance_target' '+agent.config.grad_balance_target={plan:0.4,det:0.3,map:0.3}' \
  agent.config.plan_anchor=true agent.config.plan_anchor_file=$PLAN_ANCHORS agent.config.plan_score_file=$PLAN_SCORE \
  agent.config.image_architecture=resnet34.tv_in1k agent.config.plan_heading_from_xy=true \
  agent.config.grad_balance_warmup_iters=10600 agent.config.grad_balance_interval=200 agent.config.grad_norm_log_interval=200 \
  ++agent.config.log_sync_dist=false ++agent.config.corridor_weight=1.0 ++agent.config.corridor_tokens_root=$TOKENS_ROOT \
  ++agent.config.corridor_select_rule=${SELECT_RULE:-pred_multiply_t0.25} \
  "${EXTRA[@]}" \
  wandb.enable=${WANDB:-true} wandb.project=para-ssr-v2 wandb.name=$RUN wandb.mode=online wandb.group=corr30 \
  output_dir=$OUT/train "$@"
