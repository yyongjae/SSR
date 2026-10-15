#!/usr/bin/env bash
# Stage 3 of report/19: the same ReSMap teacher, injected through different paths.
#
#   ARM=<arm> bash scripts/training/train_para_ssr_kd.sh [hydra overrides...]
#
#   ARM          heads      what reaches the student BEV from the teacher
#   plan_only    none       nothing                              (arm 1, control)
#   map_gt       map        GT map labels                        (arm 2)
#   map_teacher  map        teacher vectors as map labels        (arm 3: label path)
#   kd_readout   none       d(h_enc(F_S), h_enc(F_T))            (arm 4: readout path)
#   kd_random    none       d over a fixed random 256-d projection (control: bottleneck only)
#   kd_feature   none       d(F_S, F_T) over the whole BEV       (control: excess included)
#   kd_attn      none       sum_u a_T(u) d(F_S(u), F_T(u)): cells weighted by the frozen
#                           readout's attention on the teacher BEV (spatial method, report/22)
#   kd_sens      none       sum_u ||J_u (A F_S(u) - F_T(u))||^2 / sum_u ||J_u F_T(u)||^2, J = d traj /
#                           d F_T of the frozen readout: only the gap that moves the plan (KD_PROBES)
#
# Arms 3 vs 4 carry identical teacher knowledge (scene-long temporal memory and
# satellite prior included); only the injection point differs.
#
# Environment:
#   TEACHER_CACHE   ReSMap cache root                     (required except plan_only/map_gt)
#   PLAN_V2=1       v2 planner: WoTE anchors + PDM-score rewards (data/planning_vb, see tools/plan_v2)
#   PLAN_MAP        weight of the planning-side map-consistency hinge (plan_map.py), any ARM;
#                   PLAN_MAP_MARGIN its margin in metres (default 0)
#   READOUT_CKPT    Stage-1 readout (kd_readout)
#   KD_WEIGHT       lambda                                 (default 1.0)
#   KD_DISTANCE     cosine | mse          (default cosine; mse for kd_feature)
#   KD_WARMUP       micro-batches with lambda = 0          (default 10600 = 1 epoch
#                   at 2 GPUs x 4; 0 when fine-tuning)
#   KD_RAMP         micro-batches of linear ramp           (default 10600)
#   KD_ADAPTER      1 = 1x1 alignment adapter before h_enc (default 0)
#   KD_CENTER       none (default) | global | command: subtract the running mean of
#                   z before the distance, so the scene-independent part (95% of
#                   the energy of z, per-command) costs nothing (report/19 s3)
#   KD_BALANCE      e.g. "plan:0.7,distill:0.3" hands lambda to the GradBalancer
#                   (FP32 only); unset = fixed lambda, balancer off
#   PSEUDO_THR      teacher score threshold for map_teacher (default 0.3)
#   MAP_PROBE       1 = add a map head with GradBalancer target map:0 to the arms
#                   without one (plan_only, kd_*): the head learns, its BEV
#                   gradient is removed, so map mAP measures what the BEV holds
#                   without shaping it.  Use it on every compared arm or none.
#   INIT_CKPT       fine-tune from this checkpoint (step 1 of report/19 s4); the
#                   matching control is ARM=plan_only with the same INIT_CKPT,
#                   MAX_EPOCHS and LR
#   HEADS           off (default) | parallel | interaction.  parallel/interaction
#                   keep det+motion and map heads on (use_task_interaction
#                   false/true), i.e. distill into the full PARA-SSR model.  Arms:
#                   control (no teacher), map_teacher, kd_readout, kd_random,
#                   kd_feature.  MAP_PROBE does not apply (the map head is real).
#   AUX_BALANCE     full-head GradBalancer target without distill
#                   (default "plan:0.4,det:0.3,map:0.3", the recipe of the base runs)
#   KD_SHARE        full heads: distill target added to AUX_BALANCE, in the same
#                   units.  plan:det:map keep their ratio, so det/map scales match
#                   the control arm and ||dL_distill/dBEV|| = KD_SHARE/plan x
#                   ||dL_plan/dBEV||.  Unset = fixed lambda for distill.
#                   KD_BALANCE, if given, replaces the whole dict verbatim.
#   WANDB_PROJECT   W&B project when WANDB=0 leaves the config default on
#                   (default para-ssr-readout)
#
# Examples
#   TEACHER_CACHE=/data/kd_teacher_resmap READOUT_CKPT=runs/teacher_h1_s0/readout.pt \
#     ARM=kd_readout bash scripts/training/train_para_ssr_kd.sh
#   INIT_CKPT=/abs/plan_only.ckpt MAX_EPOCHS=5 LR=2e-5 KD_WARMUP=0 KD_RAMP=2000 \
#     TEACHER_CACHE=... READOUT_CKPT=... ARM=kd_readout bash scripts/training/train_para_ssr_kd.sh
#
# DRY_RUN=1 prints the resolved overrides instead of launching.
#
# Evaluate every arm with eval_para_ssr.sh and the head flags of its row:
# the KD modules are training-only and are not needed (or loaded) at test time.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM="${ARM:?set ARM (plan_only|map_gt|map_teacher|kd_readout|kd_random|kd_feature|kd_attn|kd_sens|control)}"
HEADS="${HEADS:-off}"
PROBE_TAG=""
[[ "${MAP_PROBE:-0}" == 1 && "${ARM}" != map_* ]] && PROBE_TAG=_probe
HEADS_TAG=""
case "${HEADS}" in
  off) ;;
  parallel) HEADS_TAG=par_ ;;
  interaction) HEADS_TAG=int_ ;;
  *) echo "unknown HEADS=${HEADS}" >&2; exit 2 ;;
esac
export EXPERIMENT="${EXPERIMENT:-para_ssr_${HEADS_TAG}${ARM}${PROBE_TAG}$([[ "${PLAN_V2:-0}" == 1 ]] && echo _v2)${PLAN_MAP:+_pm${PLAN_MAP}}${INIT_CKPT:+_ft}}"

HEADS_OFF=(agent.config.use_task_interaction=false
           agent.config.use_det_motion_head=false
           agent.config.use_map_head=false)
MAP_ONLY=(agent.config.use_task_interaction=false
          agent.config.use_det_motion_head=false
          agent.config.use_map_head=true
          'agent.config.grad_balance_target={plan:0.5,map:0.5}')

# Hydra rejects a key given twice, so the per-arm default is resolved here.
DEFAULT_DISTANCE=cosine
[[ "${ARM}" == kd_feature ]] && DEFAULT_DISTANCE=mse
KD=(agent.config.kd_teacher_cache="${TEACHER_CACHE:-null}"
    agent.config.kd_weight="${KD_WEIGHT:-1.0}"
    agent.config.kd_distance="${KD_DISTANCE:-${DEFAULT_DISTANCE}}"
    agent.config.kd_warmup_iters="${KD_WARMUP:-10600}"
    agent.config.kd_ramp_iters="${KD_RAMP:-10600}"
    agent.config.kd_adapter="$([[ "${KD_ADAPTER:-0}" == 1 ]] && echo true || echo false)"
    agent.config.kd_center="${KD_CENTER:-none}")
# One grad_balance_target override per run (Hydra rejects duplicates), so the
# probe and the distillation budget are composed into a single dict here.
BALANCE="${KD_BALANCE:-}"
if [[ "${MAP_PROBE:-0}" == 1 ]]; then
  BALANCE="${BALANCE:-plan:1.0},map:0.0"
fi
balance_arg() {  # grad_balance_target for arms without their own map task
  if [[ -n "${BALANCE}" ]]; then
    echo "agent.config.grad_balance_target={${BALANCE}}"
  else
    echo "agent.config.grad_balance_target=null"
  fi
}

need_teacher() { [[ -n "${TEACHER_CACHE:-}" ]] || { echo "TEACHER_CACHE is required for ARM=${ARM}" >&2; exit 2; }; }

if [[ -n "${KD_BALANCE:-}" && "${ARM}" != kd_* ]]; then
  echo "KD_BALANCE only applies to kd_* arms" >&2; exit 2
fi
# The probe switches the map head on inside HEADS_OFF itself: Hydra rejects a
# key given twice.  With INIT_CKPT the checkpoint must then contain a map head.
if [[ "${MAP_PROBE:-0}" == 1 ]]; then
  HEADS_OFF=(agent.config.use_task_interaction=false
             agent.config.use_det_motion_head=false
             agent.config.use_map_head=true)
fi

if [[ "${HEADS}" != off ]]; then
  # Full PARA-SSR: det+motion and map stay on, exactly as in the base run.
  [[ "${MAP_PROBE:-0}" == 1 ]] && { echo "MAP_PROBE does not apply with HEADS=${HEADS}" >&2; exit 2; }
  [[ -n "${KD_SHARE:-}" && -n "${KD_BALANCE:-}" ]] && { echo "give KD_SHARE or KD_BALANCE, not both" >&2; exit 2; }
  [[ -n "${KD_SHARE:-}" && "${ARM}" != kd_* ]] && { echo "KD_SHARE only applies to kd_* arms" >&2; exit 2; }
  FULL=(agent.config.use_task_interaction="$([[ "${HEADS}" == interaction ]] && echo true || echo false)"
        agent.config.use_det_motion_head=true
        agent.config.use_map_head=true)
  AUX="${AUX_BALANCE:-plan:0.4,det:0.3,map:0.3}"
  if [[ -n "${KD_BALANCE:-}" ]]; then FULL_BALANCE="${KD_BALANCE}"
  elif [[ -n "${KD_SHARE:-}" ]]; then FULL_BALANCE="${AUX},distill:${KD_SHARE}"
  else FULL_BALANCE="${AUX}"; fi
  FULL+=("agent.config.grad_balance_target={${FULL_BALANCE}}")
  case "${ARM}" in
    control)     ARGS=("${FULL[@]}") ;;
    map_teacher) need_teacher
                 ARGS=("${FULL[@]}" agent.config.map_label_source=teacher
                       agent.config.kd_teacher_cache="${TEACHER_CACHE}"
                       agent.config.map_pseudo_score_thr="${PSEUDO_THR:-0.3}") ;;
    kd_readout)  need_teacher
                 ARGS=("${FULL[@]}" "${KD[@]}" agent.config.kd_mode=readout
                       agent.config.kd_readout_ckpt="${READOUT_CKPT:?READOUT_CKPT is required}") ;;
    kd_random)   need_teacher
                 ARGS=("${FULL[@]}" "${KD[@]}" agent.config.kd_mode=random) ;;
    kd_feature)  need_teacher
                 ARGS=("${FULL[@]}" "${KD[@]}" agent.config.kd_mode=feature) ;;
    kd_attn)     need_teacher
                 ARGS=("${FULL[@]}" "${KD[@]}" agent.config.kd_mode=attn_feature
                       agent.config.kd_readout_ckpt="${READOUT_CKPT:?READOUT_CKPT is required}") ;;
    kd_sens)     need_teacher
                 ARGS=("${FULL[@]}" "${KD[@]}" agent.config.kd_mode=sens_feature
                       agent.config.kd_sens_probes="${KD_PROBES:-4}"
                       agent.config.kd_readout_ckpt="${READOUT_CKPT:?READOUT_CKPT is required}") ;;
    *) echo "ARM=${ARM} is not defined for HEADS=${HEADS}" >&2; exit 2 ;;
  esac
else case "${ARM}" in
  plan_only)   ARGS=("${HEADS_OFF[@]}" "$(balance_arg)") ;;
  map_gt)      ARGS=("${MAP_ONLY[@]}") ;;   # MAP_PROBE is moot: the map task is real here
  map_teacher) need_teacher
               ARGS=("${MAP_ONLY[@]}" agent.config.map_label_source=teacher
                     agent.config.kd_teacher_cache="${TEACHER_CACHE}"
                     agent.config.map_pseudo_score_thr="${PSEUDO_THR:-0.3}") ;;
  kd_readout)  need_teacher
               ARGS=("${HEADS_OFF[@]}" "${KD[@]}" "$(balance_arg)" agent.config.kd_mode=readout
                     agent.config.kd_readout_ckpt="${READOUT_CKPT:?READOUT_CKPT is required}") ;;
  kd_random)   need_teacher
               ARGS=("${HEADS_OFF[@]}" "${KD[@]}" "$(balance_arg)" agent.config.kd_mode=random) ;;
  kd_feature)  need_teacher
               ARGS=("${HEADS_OFF[@]}" "${KD[@]}" "$(balance_arg)" agent.config.kd_mode=feature) ;;
  kd_attn)     need_teacher
               ARGS=("${HEADS_OFF[@]}" "${KD[@]}" "$(balance_arg)" agent.config.kd_mode=attn_feature
                     agent.config.kd_readout_ckpt="${READOUT_CKPT:?READOUT_CKPT is required}") ;;
  kd_sens)     need_teacher
               ARGS=("${HEADS_OFF[@]}" "${KD[@]}" "$(balance_arg)" agent.config.kd_mode=sens_feature
                     agent.config.kd_sens_probes="${KD_PROBES:-4}"
                     agent.config.kd_readout_ckpt="${READOUT_CKPT:?READOUT_CKPT is required}") ;;
  *) echo "unknown ARM=${ARM}" >&2; exit 2 ;;
esac; fi

# grad_balance_target is a dict in para_ssr_agent.yaml.  A plain `key={...}`
# override MERGES into it: new keys such as `distill` are rejected, and keys
# left out (e.g. det:0.3) survive.  Delete it and add it back so the dict is
# exactly what was asked for.
RESOLVED=()
for a in "${ARGS[@]}"; do
  if [[ "${a}" == agent.config.grad_balance_target=* ]]; then
    RESOLVED+=("~agent.config.grad_balance_target" "+${a}")
  else
    RESOLVED+=("${a}")
  fi
done
ARGS=("${RESOLVED[@]}")

if [[ "${PLAN_V2:-0}" == 1 ]]; then
  # v2 planner (modules/anchor_planner.py): WoTE's 256 anchors + PDM-score rewards, any ARM.
  PLAN_VB="${PLAN_VB:-${REPO}/data/planning_vb}"
  ARGS+=(agent.config.plan_anchor=true
         agent.config.plan_anchor_file="${PLAN_VB}/trajectory_anchors_256.npy"
         agent.config.plan_score_file="${PLAN_VB}/pdm_score_256")
fi

if [[ -n "${PLAN_MAP:-}" ]]; then
  # Planning-side map consistency (plan_map.py), orthogonal to ARM: control +
  # PLAN_MAP is "constraint only", kd_* + PLAN_MAP is "both" in the 2x2 design.
  ARGS+=(agent.config.plan_map_weight="${PLAN_MAP}" agent.config.plan_map_margin="${PLAN_MAP_MARGIN:-0.0}")
fi

if [[ "${WANDB:-0}" == 0 ]]; then
  # The training config turns W&B on by default; keep these runs in their own project.
  ARGS+=(wandb.project="${WANDB_PROJECT:-para-ssr-readout}")
fi

if [[ -n "${INIT_CKPT:-}" ]]; then
  # Weights only: the agent loads them at construction.  Optimiser state and the
  # epoch counter start fresh, unlike RESUME_CHECKPOINT.
  ARGS+=(agent.checkpoint_path="${INIT_CKPT}" agent.resume_from_checkpoint=true
         agent.config.warmup_epochs="${FT_WARMUP_EPOCHS:-0}")
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%s\n' "EXPERIMENT=${EXPERIMENT}" "${ARGS[@]}" "$@"
  exit 0
fi
exec bash "${REPO}/scripts/training/train_para_ssr.sh" "${ARGS[@]}" "$@"
