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
#            cf_v15 / cf_seg_v15 : cf_seg (map_repr=seg by default, override with MAP_REPR=vector) + BEV Selector v15
#                  distillation on the 64 m (100x100) BEV grid (BEVFusion + ReSMap stage1_front64)
#            cf_vec_v15 : cf (vector map head) + BEV Selector v15 distillation
#            cf_seg_v15_v6c : (yongjae 10/27, refiner v6c stage D, SSR/report/124 + 125) cf_seg_v15 + refiner v6c trained
#                  jointly from scratch; V6C_TEACHER=<stage-T teachers_te.pt> (the checkpoint written after --phase te)
#                  feeds the OOF teachers (K-E / K-M, teacher object term).  Without V6C_TEACHER training is refused
#                  unless the GT-only arm is chosen explicitly: V6C_FIX=gt V6C_KD=false (= report 124 control B).
#                  V6C_WARMUP (micro-batches of R-E-only warm-up, default 5 epochs).
#                  cand_select eps / lam 0.05 / 0.05 in training, val and navtest (report 124 V8)
#            cf_seg_v6cP : refiner v6c stage P: C0_CKPT=<a cf_seg_v15 checkpoint> frozen (BN / dropout / GridMask in
#                  eval, the cf copy candidates kept), only v6c learns; v15 distillation OFF (it would keep training its
#                  probes), the teacher BEVs still load when V6C_TEACHER is set
#            extra v6c knobs: V6C_FIX=gt (GT fixing term), V6C_KD=false (no K-E / K-M), V6C_MODE=A, V6C_HARM=EH|H,
#            V6C_DISCOUNT=0.93 (arms, report 124 §4-3).  Control B of report 124 = V6C_FIX=gt V6C_KD=false.
#   LR / MAXEP (km 10/25): agent.lr and the cosine schedule length (default 1e-4 / 30)
#   EXACT2_DIR (10/29): the v2 (EPDMS) tables of the scorer labels, default $KM_ROOT/exact2 (agent rows = the v1 ones);
#                  $KM_ROOT/exact2_v22 = the v2.2 agent rows (tools build_exact2_v22.py, rebuilt 10-28)
#   smoke: EPOCHS=1 LIMIT_TRAIN=40 LIMIT_VAL=5 VAL_EVERY=1 WANDB=false
#   SELECT_RULE (km 10/17): trajectory the agent submits at inference, default pred_multiply_t0.25 (corridor); v2 = no corridor
set -uo pipefail
if [ -f "$(dirname "$0")/env.sh" ]; then source "$(dirname "$0")/env.sh"; else source "$(dirname "$0")/env.example.sh"; fi; cd $REPO
VAR=$1; RUN=$2; GPUS=$3; shift 3
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
OUT=${KM_RUN_ROOT:-$REPO/work_dirs/km_runs}/corr30/$RUN; mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=$GPUS CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_SOCKET_IFNAME=lo OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[ "$VAR" = v32b ] && PLANES=49 && SELECT_RULE=${SELECT_RULE:-pred_multiply_sigma_ttc}
case $VAR in c64|c64_traj|pert|mix|cf|cf_seg|cf_v15|cf_seg_v15|cf_vec_v15|cf_seg_v15_v6c|cf_seg_v6cP) PLANES=49; SELECT_RULE=${SELECT_RULE:-cand_official} ;; esac
V31=(++agent.config.corridor_plan_weight=20.0 ++agent.config.corridor_plan_ttc_weight=1.0 ++agent.config.plan_ep_feasible_only=true
     ++agent.config.corridor_occ_planes=${PLANES:-40} ++agent.config.corridor_route_weight=1.0
     ++agent.config.corridor_plan_prog_weight=0.2 ++agent.config.corridor_plan_comfort_weight=1.0 ++agent.config.corridor_verdict_weight=1.0)
V15_DISTILL=(
  ++agent.config.use_distill=true
  ++agent.config.distill_feature_root=${TEACHER_CACHE_ROOT:-/workspace/teacher_cache}
  ++agent.config.distill_teacher_source=${DISTILL_SOURCE:-targets}
  "++agent.config.distill_cache_size=[100,100]"
  ++agent.config.distill_selector_v15=true
  ++agent.config.distill_anchor_mode=cross_attn
  ++agent.config.distill_anchor_proj=true
  ++agent.config.distill_anchor_loss_type=hybrid
  ++agent.config.distill_anchor_loss_scale=10.0
  ++agent.config.distill_attn_loss_scale=5.0
  ++agent.config.distill_resmap_scale=3.0
  ++agent.config.distill_lambda_gt=0.45
  ++agent.config.distill_lambda_cov=0.20
  ++agent.config.distill_lambda_risk=0.35
  ++agent.config.distill_tau_gt=4.5
  ++agent.config.distill_tau_risk=8.0
  ++agent.config.distill_loss_weight=1.0
  ++agent.config.distill_probe_seed=0
  ++agent.config.distill_proj_lr_mult=3.0
  ++agent.config.distill_probe_lr_mult=3.0
  ++agent.config.distill_probe_warmup_steps=${DISTILL_WARMUP_STEPS:-$((21200 / NG))}
  ++agent.config.distill_probe_ramp_steps=${DISTILL_RAMP_STEPS:-$((10600 / NG))}
  ++agent.config.distill_probe_loss_weight=0.5
  ++agent.config.distill_probe_use_ema=true
  ++agent.config.distill_probe_ema_decay=0.99
)
V6C=(
  ++agent.config.v6c=true ++agent.config.v6c_mode=${V6C_MODE:-B} ++agent.config.v6c_h_arm=${V6C_HARM:-E}
  ++agent.config.v6c_fix_source=${V6C_FIX:-teacher} ++agent.config.v6c_kd=${V6C_KD:-true} ++agent.config.v6c_teacher_discount=${V6C_DISCOUNT:-1.0}
  "++agent.config.v6c_teacher_ckpt=${V6C_TEACHER:-}"
  ++agent.config.cand_select_eps=0.05 ++agent.config.cand_select_lam=0.05
)
case $VAR in
  v2)  EXTRA=(++agent.config.corridor_dir=$KM_ROOT/corridor_dt01) ;;
  v31) EXTRA=("${V31[@]}" ++agent.config.corridor_dir=$KM_ROOT/corridor_dt01 ++agent.config.corridor_route_dir=$KM_ROOT/route32) ;;
  v31_64) EXTRA=("${V31[@]}" ++agent.config.corridor_format=c64 ++agent.config.corridor_dir=$KM_ROOT/corridor64
           "++agent.config.pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]" "++agent.config.map_pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]"
           ++agent.config.bev_h=100) ;;
  v32b) EXTRA=("${V31[@]}" ++agent.config.corridor_dir=$KM_ROOT/corridor32_p49 ++agent.config.corridor_route_dir=$KM_ROOT/route32
         ++agent.config.corridor_occ_pos_weight=8.0 ++agent.config.corridor_occ_time_weight=1.0
         ++agent.config.corridor_sdf_boundary_weight=8.0 ++agent.config.corridor_sdf_sigma=true) ;;
  c64|c64_traj|pert|mix|cf|cf_seg|cf_v15|cf_seg_v15|cf_vec_v15|cf_seg_v15_v6c|cf_seg_v6cP) V31C=(); for x in "${V31[@]}"; do [[ $x == *corridor_verdict_weight* ]] || V31C+=("$x"); done
           EXTRA=("${V31C[@]}" ++agent.config.corridor_format=c64
           ++agent.config.corridor_dir=$KM_ROOT/corridor64
           "++agent.config.pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]" "++agent.config.map_pc_range=[-32.0,0.0,-2.0,32.0,64.0,2.0]"
           ++agent.config.bev_h=100 ++agent.config.cand_scorer_weight=1.0 ++agent.config.exact_dir=$KM_ROOT/exact
           ++agent.config.cand_official=true ++agent.config.exact2_dir=${EXACT2_DIR:-$KM_ROOT/exact2}
           ++agent.config.cand_scorer_fields=$([ "$VAR" = c64_traj ] && echo false || echo true))
       [ "$VAR" = pert ] && EXTRA+=(++agent.config.corridor_plan_ec_smooth_weight=0.1 ++agent.config.cand_scorer_pert_p=0.5)
       [ "$VAR" = mix ] && EXTRA+=(++agent.config.corridor_plan_ec_smooth_weight=0.1 ++agent.config.cand_scorer_mix_p=0.5)
       case $VAR in cf|cf_seg|cf_v15|cf_seg_v15|cf_vec_v15|cf_seg_v15_v6c|cf_seg_v6cP) EXTRA+=(++agent.config.corridor_plan_ec_smooth_weight=0.1 ++agent.config.cand_cf_folds=2
                                       ++agent.config.input_target=true) ;; esac
       [ "$VAR" = cf_seg ] && EXTRA+=(++agent.config.map_repr=seg)
       case $VAR in cf_v15|cf_seg_v15) EXTRA+=(++agent.config.map_repr=${MAP_REPR:-seg} "${V15_DISTILL[@]}") ;; esac
       [ "$VAR" = cf_vec_v15 ] && EXTRA+=(++agent.config.map_repr=vector "${V15_DISTILL[@]}")
       if [ "$VAR" = cf_seg_v15_v6c ]; then
         EXTRA+=(++agent.config.map_repr=seg "${V15_DISTILL[@]}" "${V6C[@]}" ++agent.config.v6c_stage=D
                 ++agent.config.v6c_warmup_steps=${V6C_WARMUP:-$((5 * 85109 / (4 * NG)))})
       fi
       if [ "$VAR" = cf_seg_v6cP ]; then
         [ -n "${C0_CKPT:-}" ] || { echo "cf_seg_v6cP needs C0_CKPT=<cf_seg_v15 checkpoint>"; exit 2; }
         EXTRA+=(++agent.config.map_repr=seg "${V6C[@]}" ++agent.config.v6c_stage=P ++agent.config.use_distill=false
                 "++agent.config.distill_feature_root=${TEACHER_CACHE_ROOT:-/workspace/teacher_cache}"
                 "++agent.config.distill_cache_size=[100,100]"
                 "++agent.checkpoint_path=$C0_CKPT" ++agent.resume_from_checkpoint=true)
       fi ;;
  *) echo "variant: v2 | v31 | v31_64 | v32b | c64 | c64_traj | pert | mix | cf | cf_seg | cf_v15 | cf_seg_v15 | cf_vec_v15 | cf_seg_v15_v6c | cf_seg_v6cP"; exit 2 ;;
esac
if [ "${DISTILL:-}" = v15 ] && [[ "$VAR" != cf_v15 && "$VAR" != cf_seg_v15 && "$VAR" != cf_vec_v15 ]]; then
  case $VAR in
    v31_64|c64|c64_traj|pert|mix|cf|cf_seg) EXTRA+=("${V15_DISTILL[@]}") ;;
    *) echo "error: DISTILL=v15 requires a 64m variant (v31_64 | c64 | c64_traj | pert | mix | cf | cf_seg | cf_v15), got $VAR"; exit 2 ;;
  esac
fi
if [ "${WANDB:-true}" != "false" ] && [ "${WANDB:-1}" != "0" ] && [ -f "$REPO/.env" ]; then
  _wkey=""
  while IFS= read -r _line || [ -n "$_line" ]; do
    _line="${_line%$'\r'}"
    case "$_line" in
      WANDB_API_KEY=*)
        _wkey="${_line#WANDB_API_KEY=}"
        _wkey="${_wkey%\"}"; _wkey="${_wkey#\"}"
        _wkey="${_wkey%\'}"; _wkey="${_wkey#\'}"
        ;;
    esac
  done < "$REPO/.env"
  if [ -n "$_wkey" ]; then
    export WANDB_API_KEY="$_wkey"
    _WANDB_NETRC_FILE="$(mktemp "${TMPDIR:-/tmp}/wandb-netrc.XXXXXX")"
    chmod 600 "$_WANDB_NETRC_FILE"
    printf 'machine api.wandb.ai\n  login user\n  password %s\n' "$WANDB_API_KEY" > "$_WANDB_NETRC_FILE"
    export NETRC="$_WANDB_NETRC_FILE"
    trap 'rm -f "$_WANDB_NETRC_FILE"' EXIT
  fi
  unset _wkey _line
fi
# DRYRUN=1: print the composed hydra job config and exit (no data, no GPU)
[ "${DRYRUN:-0}" = 1 ] && set -- "$@" --cfg job
$PY navsim/planning/script/run_training.py \
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
  wandb.enable=${WANDB:-true} wandb.entity=${WANDB_ENTITY:-e2ekd} wandb.project=${WANDB_PROJECT:-v1_distill} wandb.name=$RUN wandb.mode=online wandb.group=${WANDB_GROUP:-bev-selector} \
  output_dir=$OUT/train "$@"
