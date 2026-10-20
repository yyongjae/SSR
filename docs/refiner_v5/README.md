# refiner v5 (branch `refiner-v5`)

Base: fusion v13 (`aux_distill` 04844d1, BEV selector v13 distillation) + `refiner-v4` (v4_3 range readout, torch LQR,
corrector, computed edits), merged in 79cab50.  Design: SSR/report 97 (v5 design), reviews 98-102 and 104 (all findings
fixed), runs 103 (teachers, student completion), 105 (speed analysis, no-LQR student, final structure).  Everything
v5 is opt-in (`ck_e2e2.v4.v5.*`, `train_ck2.py --v5-*`): with the defaults the v4_3 / fusion-v13 networks, losses and
outputs are unchanged (tests below).

## What v5 is

```
cameras t-1.0 / t-0.5 / t0 -> ResNet34 + BEVFormer -> det head (boxes, velocity with stop gate, accel aux, 6 futures),
map head, planner -> top-16 plans x (orig + 3 speed + 2 lateral variants) = 96 candidates
-> 41 states per candidate (student: 'ref' = the tracker's reference, no LQR; teachers: LQR rollout)
-> critic: v4_3 range fields (B/U/Bp, D) read at the states + v5 relation (student boxes x {hold, CV, 6 det futures}
   -> overlap r, reliability a, hazard H) + unknown evidence -> 5 scores + lateral correction
-> deployment pick tau0 -> corrector A_S (ctx, scores, ego speed, per-state reads Z in teacher units, top-3 related
   objects with delta-r probes) -> 12 controls -> tau1 = decode(tau0, controls)  = the v5 output (ck2_v5_traj)
```

Training only: two frozen-trunk teachers (DET: BEVFusion BEV + BEVFusion boxes t0 + 2 past frames, relation with
hold / CV / CA, blocking fields, unknown, corrector A_T; MAP: ReSMap BEV road margin, unknown, corner aux, A_T), score KD
(NC <- DET, DAC <- MAP, Platt), Z-KD (the teachers' fields re-read at the student's states), lateral KD, GT losses
(label BCE, field GT aux, L_rel / L_rel,cv / L_a / L_H, L_refine soft PDMS, L_edit), AWR on official PDMS(tau1) -
PDMS(tau0) from epoch 5 (on-policy).

## What is in the branch

| Piece | Where | Switch |
|---|---|---|
| Teachers: relation channel on BEVFusion rows (KDObjRows), unknown evidence, per-state reads Z, corrector A_T trained by L_refine(X1) (log-space soft PDMS on the LQR rollout of tau1, in-path braking shaping), frozen v4_3 trunk | `ck/v5/{relation,zread,corrector,refine_loss}.py`, `tools/ck/train_ck2.py` | `--v5-relation teacher --v5-unknown 1 --v5-corrector 1 --v5-freeze-trunk 1` (`tools/ck/v5/run_teachers_v5.sh`) |
| Stage-1 gate (A_T vs no-edit on fixed r34 top-1 drafts, official scorer; DET held-out 12,179 val, MAP navtest) | `tools/ck/v5/eval_refine_gate.py` | -- |
| Student det-head fixes: velocity code weights in det order, V1b moving-only velocity rows, V1c stop branch, ACC aux head; det outputs at inference | `modules/det_motion_head.py`, `para_ssr_model.py` | `tools/ck/e2e2/fusion_v13_v5.extra` |
| 3-frame student input (t-1.0, t-0.5, t0) | hydra `agent.config.frame_indices=[1,2,3]` | `fusion_v13_v5.extra` |
| Student relation: own det boxes (matched rows keep the box gradient, V2), det futures as modes + learned CV prior, reliability a_i with p_stop, log-domain relation BCE (report 102 N1) | `ck/v5/{relation,student_objects}.py`, `online2._v5_rel_terms` | `v4.v5.relation=student`, `rel_modes=cv+fut`, `rel_pstop`, `rel_det_grad(_matched)`, `lambda_rel(_cv)` |
| Corrector A_S with top-3 v2 features + delta-r probes, tau1 record + AWR, L_edit-Bernoulli | `ck/model.py` (`refine_drafts`, `_v5_top_v2`, `relation_reread`), `online2._v5_edit_terms` | `v4.v5.corrector`, `top_feats=v2`, `record_tau1`, `lambda_awr`, `lambda_edit` |
| Z adapters in teacher units (frozen teacher stats, H sigma floor 0.25) + Z-KD | `ck/v5/zadapt.py`, `tools/ck/v5/fit_z_stats.py` | `v4.v5.z_adapt`, `z_stats_det/_map`; Z-KD: `v5kd_arm.set` (`z_kd=true`) |
| **No-LQR student** (2026-10-20): exec model 'ref' = the tracker's 41-state reference (heading change clamped to the bicycle turn limit, comfort factor off in L_refine); teachers keep LQR, their two rollouts shared; Z-KD targets re-read at the student's states; z-stats records carry `zq_exec` and are checked | `ck/v4/lqr.reference_rollout`, `range_readout.{candidate_states,shared_states,zq_at}`, `CKNet.score_set(zq_exec)`, `zadapt.check_z_stats(exec_model)` | `v4.exec_model=ref` (in `v5_arm.set`); `fit_z_stats --zq-exec ref` |

Arms: `ARM=v5` = `v5_arm.set` (Stage-2 arm B, GT-only control), `ARM=v5kd` = + `v5kd_arm.set` (arm D, + Z-KD).
Teacher / calibration / z-stats paths come from an `E2E_CK_SET` file (`tools/ck/e2e2/v5_teachers_rel5_unk5.set` is the
3-epoch-teacher set; its z-stats records are LQR-state ones and are refused by a 'ref' student -- refit with
`--zq-exec ref`).

## How to run (GPU runs need the user's approval; GPUs 0-3)

Teachers (DET on 0,1, MAP on 2,3; warm start from v4T10_rng2 / v4M10_rng2):

    V5_USER_APPROVED=1 V5_EPOCHS=10 V5_DET_RUN=v5T_rel6 V5_MAP_RUN=v5M_unk6 bash tools/ck/v5/run_teachers_v5.sh both

After a teacher finished (gate, KD calibration, z-stats at the student's states):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python tools/ck/v5/eval_refine_gate.py --run T=v5T_rel6 --which ep9 \
      --split navtrain_val --device cuda --approved --label-workers 32 --tag rel6      # MAP: --split navtest
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD python tools/ck/e2e2/fit_kd_calib.py --run <CK>/v5/train/v5T_rel6 \
      --which last --n-tokens 16384
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD python tools/ck/v5/fit_z_stats.py --run <CK>/v5/train/v5T_rel6 \
      --which last --tokens 4096 --device cuda --amp 1 --zq-exec ref                   # -> v5T_rel6__last__zqref.json

Student (smoke first, then the main run; a set file with teacher_det_run / teacher_map_run / kd_calib.* /
v4.v5.z_stats_*):

    CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python tools/ck/v5/smoke_student_v5.py --roundtrip 1 --zkd 1 \
      --teacher-det <CK>/v5/train/v5T_rel6 --teacher-map <CK>/v5/train/v5M_unk6          # CPU, minutes
    REPO=$PWD TAG=<tag> CKSET=<set file> bash tools/ck/v5/smoke_v5kd_gpu.sh                # 4-GPU, 3 x 100 mb
    RUN=<run> CK2_E2E_APPROVED=1 ARM=v5kd E2E_EXTRA=$PWD/tools/ck/e2e2/fusion_v13_v5.extra \
      E2E_CK_SET=<set file> bash tools/ck/e2e2/train_e2e2.sh start --gpus 0,1,2,3

Evaluation: `tools/ck/e2e2/eval_e2e2.py` dumps / labels / evaluates the v5 output (`v5_traj`, tau1 vs the pool pick it
was computed from).

## Status (2026-10-20)

- Teachers, 3 epochs (Stage-1 gate, tau1 - draft PDMS): DET v5T_rel5 **+3.42** [+2.68, +4.09] (val held-out 12,179),
  MAP v5M_unk5 **+1.75** [+1.09, +2.47] (navtest).  The break clause (passing tokens turned failing <= 1.3 %) fails for
  both (2.73 % / 3.02 %).  10-epoch teachers v5T_rel6 / v5M_unk6 are training (report 105).
- Student main run with the LQR student (`ck2e2e_v5kd_main`) was stopped at epoch 1 (it ran ~2.6x the fusion-v10 step
  time); the no-LQR student is not trained yet.
- Speed (GPU 0, batch 1, same condition): fusion v10 0.44 s, v5kd LQR student 1.44 s, 'ref' student 0.97 s per
  micro-batch (report 105).

## Open items

- Teacher break rate above the registered 1.3 %.
- Resuming an LQR-student checkpoint with the 'ref' config restores its old z-stats buffers without a check (report 104
  P2); a fresh run is not affected.
- 4-GPU step time of the 'ref' student (warmup and on-policy) not measured yet.
- 'ref' drops the tracker: the GT-field selection ceiling falls from +1.91 to +0.59 (report 89 §3-2, unclamped variant).

## Tests

    CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python -m pytest -q tests/test_v5_*.py tests/test_v4_*.py tests/test_ck2_e2e_*.py

At 08f9498: 530 passed, 1 skipped (CPU, ~22 min).  `tests/test_v5_no_lqr_student.py` covers the 'ref' executor, the
shared rollout, the Z-KD re-read (incl. a real-teacher TeacherPair2 check, skipped without the v5 teacher runs) and the
z-stats exec-model guard.
