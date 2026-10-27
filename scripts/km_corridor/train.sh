#!/usr/bin/env bash
# km 10/16: v2 r34 recipe + corridor head, 30 epochs from scratch, mid-training val every 5 epochs (val = evaluation only).
#   usage: train.sh <variant> <run name> <gpus e.g. 0,1> [extra hydra overrides...]
#   variant  v2   : + corridor head only (static SDF + 8 x 0.5 s swept occupancy)          == corr30_v2 (navtest 89.30 / 88.57)
#            v31  : + 40 x 0.1 s planes, plan constraint (static / occ / TTC), conditional EP, route channel,
#                   progress reward inside the GT-safe corridor, official-comfort hinge, verdict supervision  == corr30_v31
#            v31_64 : v31 on the 64 m BEV / GT (NOT smoke-tested yet, see README)
#            v32b : v31 + 49 x 0.1 s planes (to 4.9 s, GT corridor32_p49: tools/corridor/build_corridor_p49.py), occ
#                   pos_weight 8 + plane weight 1 -> 2 over time, SDF boundary weight 8 + sigma head (p_DAC = Phi(z)),
#                   forward selection pred_multiply_sigma_ttc (km 10/18)
#            c64  : (km 10/19, C) v31_64 + 49 x 0.1 s planes + the candidate scorer on the model's own tracked
#                   candidates (exact GT verdicts, tools/corridor/build_exact.py) + fixed LQR port; no verdict loss on
#                   random anchors.  km 10/19 (C v3): every scorer head trained on the OFFICIAL per-candidate PDMS (v1
#                   cache) / EPDMS (v2 cache, EC excluded) terms (modules/exact_score.py); submits
#                   corridor_select_rule=cand_official (expected PDMS first, expected EPDMS among the near-ties; no
#                   anchor-label term)
#            c64_traj : c64 with the scorer WITHOUT the predicted-corridor read-out (ablation)
#            pert : (km 10/23) c64 + candidate EC-smoothness hinge (corridor_plan_ec_smooth_weight 0.1) + scorer input
#                   perturbation on half the training samples (cand_scorer_pert_p 0.5, candidate_scorer.PERT_DEFAULTS)
#            mix : (km 10/25) c64 + EC-smoothness hinge 0.1 + half of each sample's scorer candidates anchor + u * offset,
#                  u ~ U(0, 1) (cand_scorer_mix_p 0.5), labelled by their own official terms; no input perturbation
#            cf  : (km 10/25) c64 + EC-smoothness hinge 0.1 + cross-fitting (cand_cf_folds 2): the scorer trains on the
#                  candidates of a planner copy that never fitted the scene (log-disjoint folds); no perturbation / mix
#            cf_seg : (km 10/27) cf with the raster map head (map_repr seg: BEV segmentation of road / walkway / centerline /
#                  crosswalk, modules/map_seg_head.py) in place of the vector map head
#   LR / MAXEP (km 10/25): agent.lr and the cosine schedule length (default 1e-4 / 30)
#   smoke: EPOCHS=1 LIMIT_TRAIN=40 LIMIT_VAL=5 VAL_EVERY=1 WANDB=false
#   SELECT_RULE (km 10/17): trajectory the agent submits at inference, default pred_multiply_t0.25 (corridor); v2 = no corridor
set -uo pipefail
source "$(dirname "$0")/env.sh"; cd $REPO
VAR=$1; RUN=$2; GPUS=$3; shift 3
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
OUT=$KM_ROOT/km_runs/corr30/$RUN; mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=$GPUS CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_SOCKET_IFNAME=lo OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[ "$VAR" = v32b ] && PLANES=49 && SELECT_RULE=${SELECT_RULE:-pred_multiply_sigma_ttc}
case $VAR in c64|c64_traj|pert|mix|cf|cf_seg) PLANES=49; SELECT_RULE=${SELECT_RULE:-cand_official} ;; esac
V31=(++agent.config.corridor_plan_weight=20.0 ++agent.config.corridor_plan_ttc_weight=1.0 ++agent.config.plan_ep_feasible_only=true
     ++agent.config.corridor_occ_planes=${PLANES:-40} ++agent.config.corridor_route_weight=1.0
     ++agent.config.corridor_plan_prog_weight=0.2 ++agent.config.corridor_plan_comfort_weight=1.0 ++agent.config.corridor_verdict_weight=1.0)
case $VAR in
  v2)  EXTRA=(++agent.config.corridor_dir=$KM_ROOT/corridor_dt01) ;;
  v31) EXTRA=("${V31[@]}" ++agent.config.corridor_dir=$KM_ROOT/corridor_dt01 ++agent.config.corridor_route_dir=$KM_ROOT/route32) ;;
  v31_64) EXTRA=("${V31[@]}" ++agent.config.corridor_format=c64 ++agent.config.corridor_dir=$KM_ROOT/corridor64
           "++agent.config.pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]" "++agent.config.map_pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]"
           ++agent.config.bev_h=100) ;;
  v32b) EXTRA=("${V31[@]}" ++agent.config.corridor_dir=$KM_ROOT/corridor32_p49 ++agent.config.corridor_route_dir=$KM_ROOT/route32
         ++agent.config.corridor_occ_pos_weight=8.0 ++agent.config.corridor_occ_time_weight=1.0
         ++agent.config.corridor_sdf_boundary_weight=8.0 ++agent.config.corridor_sdf_sigma=true) ;;
  c64|c64_traj|pert|mix|cf|cf_seg) V31C=(); for x in "${V31[@]}"; do [[ $x == *corridor_verdict_weight* ]] || V31C+=("$x"); done
           EXTRA=("${V31C[@]}" ++agent.config.corridor_format=c64
           ++agent.config.corridor_dir=$KM_ROOT/corridor64
           "++agent.config.pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]" "++agent.config.map_pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]"
           ++agent.config.bev_h=100 ++agent.config.cand_scorer_weight=1.0 ++agent.config.exact_dir=$KM_ROOT/exact
           ++agent.config.cand_official=true ++agent.config.exact2_dir=$KM_ROOT/exact2
           ++agent.config.cand_scorer_fields=$([ "$VAR" = c64_traj ] && echo false || echo true))
       [ "$VAR" = pert ] && EXTRA+=(++agent.config.corridor_plan_ec_smooth_weight=0.1 ++agent.config.cand_scorer_pert_p=0.5)
       [ "$VAR" = mix ] && EXTRA+=(++agent.config.corridor_plan_ec_smooth_weight=0.1 ++agent.config.cand_scorer_mix_p=0.5)
       case $VAR in cf|cf_seg) EXTRA+=(++agent.config.corridor_plan_ec_smooth_weight=0.1 ++agent.config.cand_cf_folds=2
                                       ++agent.config.input_target=true) ;; esac
       [ "$VAR" = cf_seg ] && EXTRA+=(++agent.config.map_repr=seg) ;;
  *) echo "variant: v2 | v31 | v31_64 | v32b | c64 | c64_traj | pert | mix | cf | cf_seg"; exit 2 ;;
esac
exec $PY navsim/planning/script/run_training.py \
  agent=para_ssr_agent agent.lr=${LR:-1e-4} agent.config.max_epochs=${MAXEP:-30} experiment_name=$RUN \
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
