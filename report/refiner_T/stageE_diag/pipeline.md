# Stage E diagnostics: H8 (evaluation / pipeline differences) and H9 (single seed, training length, recipe)

Question: why is E2 (86.15) below E0 + R_T4 (86.91) and E0 + R_M4 (86.81)? The gap is -0.76 against R_T4 and -0.66
against R_M4 (PDMS points).

This page tests two explanations:
- **H8:** the two numbers come from pipelines that are not comparable.
- **H9:** the gap is within single-seed / training noise, or comes from recipe differences.

All numbers are navtest, 12,146 tokens, PDMS x100. CIs are paired over tokens, with a log-cluster bootstrap over 136 logs
(10,000 draws, seed 0, percentile 95 %). The bootstrap reflects scene sampling only; it does not include training-seed
variance. Machine-readable results are in `pipeline.json`.

## Verdict

- **H8: not supported.** Every pipeline difference is either absent or at the 0.01-point level.
  - Ego inputs are identical, and the teacher checkpoints are byte-identical.
  - The stage-E code path matches the stage-T path to within 1.2e-5 m.
  - The slope has no effect on the forward pass, and the gate is handled the same way (ignored) in both arms.
  - The scorer shows 0 difference.
  - fp16 vs fp32 moves the result by -0.008 [-0.026, 0.000], and pkl vs CSV drafts by +0.007 [-0.011, +0.030].
  - After harmonising the pipelines, the gap is -0.75 ± 0.02, so H8 explains about 0.015 of 0.76 (about 2 %, at most
    about 4 %).
- **The gap is real under the same pipeline and on the same drafts.** R_T4 applied to **E2's own** tau0 gives 86.80.
  The -0.76 splits as:
  - **Draft:** -0.11 [-0.62, +0.39], i.e. E2's planner draft under the same teacher refiner. Not significant.
  - **Refiner:** -0.65 [-0.94, -0.35], i.e. the student refiner vs the R_T4 teacher on identical drafts, significant.
    That is 85 % of the gap.
  - For R_M4 the split is -0.06 [-0.54, +0.40] (draft) and -0.60 [-0.95, -0.26] (refiner).
- **H9: partly.** The draft component (-0.11) is well inside the literature seed spread of NAVSIM planners (std 0.4 to
  0.6 PDMS). The refiner component (-0.65, 68 fixed vs 153 broken tokens against R_T4 on the same drafts) compares two
  single-seed trainings, so its seed std is sqrt(2) * sigma:
  - At sigma = 0.4 to 0.6 (a whole E2E planner), -0.65 is 0.8 to 1.1 sigma_diff.
  - At sigma = 0.21 (kyungmin's frozen-feature readout head h1), it is 2.2 sigma_diff.

  Nothing in our logs measures sigma for the refiner (every arm is one seed). The late-checkpoint numbers below give a
  floor.
  The E0 vs E2 recipe differences, 2 x acc 16 vs 4 x acc 8, keep the same global batch, the same number of optimiser
  steps and a grad-balancer schedule scaled to match. They are not a plausible source of 0.65.

## H8: every pipeline difference between "E0 + R_*" and "E2"

| # | item | E0 + R_* (e0_teacher_refine) | E2 (official eval) | measured difference | effect on PDMS |
|---|---|---|---|---|---|
| 1 | ego inputs v0 / a0 / eds / cmd | packed navtest `human` part (raw `ego_dynamic_state` of frame t0, `argmax(driving_command)`) | `status_feature` -> `e2e.ego_inputs` (`ego_statuses[-1]` velocity / acceleration, command one-hot) | eds, a0: 0.0 exactly. v0: <= 9.5e-7 (1 float32 ulp: torch vs numpy hypot). cmd equal on 100 %. No token has cmd = -1. | 0 |
| 2 | frame / units | ego frame, m/s, m/s^2 | same (NAVSIM ego frame; the same numbers, see 1) | none | 0 |
| 3 | teacher weights | `runs/stageT4_{T,M}_fold0_seed0` ckpt_best | KD used `stageE/teachers/stageT4_{T,M}_fold0_seed0` | sha256 of ckpt_best.pt, config.json and norm(_map).npz are identical | 0 |
| 4 | teacher BEV | `TeacherCache/ResmapCache.for_subset('navtest')` S grid, fp16 -> float | KD (training): the same classes, `for_subset('navtrain')` (the caches the teachers were trained on); `load_bev(s_grid=True)` fp16 -> float | same loaders; only the split differs, as intended | n/a |
| 5 | code path of the teacher's correction | `refine_external_drafts` (stage-T loader, `decode_batch`) | `StageE._run` + `StageE._decode(slope 0)`, i.e. the function the KD targets were computed with | on E0 drafts, 3,037 tokens (every 4th), fp32 both: max 1.2e-5 m (T) and 3.9e-6 m (M), 0 tokens > 1 cm | 0 |
| 6 | precision | autocast fp16 (stage-T convention) | E2 training and eval in fp32 (trainer precision 32; KD teachers `.float()`) | T: max 2.6 cm, 0.10 % of tokens > 1 cm. M: max 0.75 cm. | R_T4 fp32 - fp16 = **-0.008 [-0.026, 0.000]** (1 token broken). R_M4: **0.000**. |
| 7 | straight-through slope | decode slope 0 | infer slope 0 (training used 0.1 for the backward pass only) | `decode(..., slope 0)` vs `slope 0.1` on all E2 outputs: max abs 0.0 (the forward pass does not depend on it) | 0 |
| 8 | gate / theta | theta 0: correction always applied, p_g recorded | `StageE.infer` has no gate: always applied | same rule. The student's gate head is untrained (0 * gate_logit), so it could not be used anyway. | 0 |
| 9 | drafts | E0 pkl from `tools/dump_navtest_trajectories.py` (batch size 8) | official eval (batch size 1) | 3 tokens flip a multiplier (below). EP differs by <= 8.8e-4 on 6.9k tokens. | E0_pkl - E0_csv = **+0.007 [-0.011, +0.030]** (fixed 1, broken 1, plus one TTC token 0.58 -> 1) |
| 10 | scorer | stage-T batched scorer (bitwise equal to `pdm_score`, scorer_equivalence.json) | official `run_pdm_score_gpu.py` CSV | E2 re-scored with the batched scorer: 0 tokens differ (> 1e-9) on any sub-score, in both tau0 and tau_final | 0 |
| 11 | scorer version / config | WoTE-derived v1.0 fork (report 29), same metric caches | same; E0 CSV and E2 CSV hydra configs are identical apart from experiment_name (stageE_eval) | none | 0 |
| 12 | token set | all 12,146, including the 45 frame_gap tokens | all 12,146, all valid | same | 0 |

The 3 tokens that flip between the E0 pkl and the official E0 CSV (values are CSV -> pkl):
- `14d53eb06a7d582a`: DAC 0 -> 1, PDMS 0 -> 1.
- `72be63ed04f15f97`: NC 1 -> 0, PDMS 0.58 -> 0.
- `8f60912c624e5f5f`: TTC 0 -> 1, PDMS 0.58 -> 1.

The refined arms inherit whichever version of the draft they were fed. The largest possible effect on E0 + R_* is
(1 + 0.58 + 0.42) / 12,146, about 0.016 points.

**Harmonised comparison.** Take E0 + R_T4 in fp32 (86.903) and allow ±0.016 for the pkl draft. That gives
86.89 to 86.92 against E2 at 86.151, so the gap is **-0.75 ± 0.02**. The pipeline accounts for 0.01 to 0.03 of the
0.76 points: about 2 % as a point estimate, 4 % at most.

## The like-for-like decomposition (from the H8 runs)

The frozen teachers were applied to **E2's own planner draft** (`e2_tau0_navtest_trajectories.pkl`), using the same
path as e0_teacher_refine (fp16, packed ego, stage-T loader). The fed drafts are bitwise equal to E2's tau0. Arm means:

| arm | PDMS | NC | DAC | TTC | EP | PDMS = 0 tokens |
|---|---|---|---|---|---|---|
| E0 (pkl) | 84.875 | 97.85 | 93.31 | 93.57 | 79.77 | 1038 |
| E0 + R_T4 | 86.911 | 98.40 | 94.99 | 94.34 | 81.59 | 780 |
| E0 + R_M4 | 86.811 | 97.98 | 95.35 | 93.86 | 81.49 | 786 |
| E2 tau0 | 84.656 | 97.93 | 92.92 | 93.66 | 79.67 | 1080 |
| **E2 tau0 + R_T4** | **86.797** | 98.49 | 94.69 | 94.36 | 81.56 | 810 |
| **E2 tau0 + R_M4** | **86.748** | 98.08 | 95.12 | 93.90 | 81.51 | 808 |
| E2 tau0 + R_none4 | 84.753 | 97.95 | 93.00 | 93.75 | 79.71 | 1067 |
| E2 (tau_final, student) | 86.151 | 98.17 | 94.29 | 94.04 | 80.99 | 895 |

Contrasts (PDMS diff [95 % CI], then tokens fixed 0 -> >0 / broken >0 -> 0):

| contrast | PDMS | fixed / broken |
|---|---|---|
| E2 - (E0 + R_T4) (the gap) | -0.760 [-1.254, -0.235] | 339 / 454 |
| (E2 tau0 + R_T4) - (E0 + R_T4): draft, under the same teacher | -0.114 [-0.618, +0.391] | 342 / 372 |
| E2 - (E2 tau0 + R_T4): student vs teacher, same drafts | **-0.646 [-0.943, -0.345]** | 68 / 153 |
| E2 - (E0 + R_M4) | -0.660 [-1.193, -0.130] | 336 / 445 |
| (E2 tau0 + R_M4) - (E0 + R_M4) | -0.063 [-0.540, +0.402] | 342 / 364 |
| E2 - (E2 tau0 + R_M4) | **-0.597 [-0.948, -0.255]** | 71 / 158 |
| (E2 tau0 + R_T4) - E2 tau0: teacher gain on E2 drafts | +2.142 [+1.628, +2.723] | 296 / 26 |
| (E2 tau0 + R_M4) - E2 tau0 | +2.092 [+1.509, +2.768] | 298 / 26 |
| (E2 tau0 + R_none4) - E2 tau0 | +0.097 [-0.017, +0.225] | 27 / 14 |
| E2 - E2 tau0: student gain | +1.495 [+1.057, +2.034] | 194 / 9 |

Reading:
- E2's planner draft is not harder to correct. The teachers gain slightly more on it (+2.14 / +2.09) than on E0's
  (+2.04 / +1.94).
- The student reaches 70 % (vs R_T4) or 71 % (vs R_M4) of the teacher gain on the same drafts.
- The student is also much less active than either teacher on the same drafts:

| refiner (drafts) | lon_live | mean max disp (m) | disp > 0.5 m | mean abs(e_lat[2:]) (m) |
|---|---|---|---|---|
| R_T4 (E2 tau0) | 5.0 % | 0.104 | 6.0 % | 0.028 |
| R_M4 (E2 tau0) | 3.1 % | 0.096 | 5.8 % | 0.031 |
| R_none4 (E2 tau0) | 1.4 % | 0.027 | 0.7 % | 0.008 |
| student (E2 tau0) | 2.2 % | 0.042 | 2.2 % | 0.016 |

Why the student is weaker (KD fidelity, training-draft distribution, capacity, the student BEV) is outside H8 / H9.
The pipeline check does show that the KD targets during training were exactly the correction function evaluated
post hoc (row 5), so a mis-wired KD target is ruled out. One observation for the other hypotheses, from the training
log (`stageE_steps.jsonl`, epoch-29 means):
- Teacher lon-live on the training drafts is 9.9 % (R_T) and 7.2 % (R_M).
- Student `ref/live` is 8.5 %.
- The final KD L1 is 0.033 (T) and 0.027 (M), in m/s and m.
- About 47 % of the training drafts were perturbed GT-human drafts.

So in training the student is about as active as the teachers. At navtest, on clean tau0 drafts, it is about 0.4 to
0.5 times as active as they are.

## H9: seed variance, training length, recipe

**Literature.** NAVSIM (Dauner et al., NeurIPS 2024, [arXiv 2406.15349](https://arxiv.org/abs/2406.15349)):
- Table 2: three TransFuser training seeds (configs A1-A3) have a PDMS std of ±0.56. The seeds differ by up to
  1.7 DAC points (91.3 / 92.8 / 93.0) and 1.4 TTC points.
- Leaderboard 1.1 (Table 3, 3 seeds): TransFuser 83.9 ± 0.4, LTF 83.5 ± 0.6, Ego-status MLP 66.4 ± 0.9.

Single-seed differences below about 1 PDMS between two E2E trainings are therefore not decisive.

**Our logs.** No PARA-SSR or refiner arm has been trained with more than one seed:
- report 22 §9 (head ablation): "arm당 seed 1개".
- report 29 §6: "모든 arm은 seed 1개".
- report 13: arms are single seed, and the CI does not include seed variance.
- The stage-T refiners (run 1 to run 4) are fold 0, seed 0 only, and the stage-E pre-registration fixed one seed.

The only seed measurement is from the collaborator: kyungmin/SSR report 22 §3. It trains planning-readout heads on
frozen teacher BEV (10 epochs, 3 seeds) and reports navtest PDMS ± seed spread:
- h0 76.96 ± 0.49, h1 81.49 ± 0.21, h2 82.06 ± 0.12, h1 cmd-late ± 0.07.
- S_ego ± 0.14 to 0.31, S_student ± 0.04.

This is the closest analogue to a small head trained on fixed features, i.e. a refiner. It puts the seed spread of
such a head at 0.05 to 0.5 PDMS. That spread is a lower bound for from-scratch E2E runs, where NAVSIM reports 0.4 to 0.6.

**Late-checkpoint floor (epoch 28 vs epoch 29 of the same run).** See the table below. It is filled from
`pipeline/score_h9`. The CI is the log bootstrap. Checkpoint-to-checkpoint noise at the end of the cosine schedule is a
floor for seed variance, not an estimate of it.

| quantity | epoch 29 (last.ckpt, the reported arms) | epoch 28 | ep28 - ep29 [95 % CI] |
|---|---|---|---|
| E0 (planner draft) | 84.875 | 84.706 | -0.169 [-0.323, -0.026] (27 fixed / 51 broken) |
| E0 + R_T4 (fp16, stage-T path) | 86.911 | 86.813 | n/a |
| teacher gain R_T4 on E0 draft | +2.036 | +2.106 [+1.688, +2.546] | +0.07 |
| E2 tau0 (planner draft) | 84.656 | 84.632 | -0.024 [-0.198, +0.159] |
| E2 tau_final | 86.151 | 86.075 | -0.076 [-0.234, +0.082] |
| student gain (E2 - E2 tau0) | +1.495 [+1.057, +2.034] | +1.443 [+1.016, +1.963] | -0.05 |
| E2 tau0 - E0 (planner-vs-planner) | -0.219 [-0.854, +0.400] | -0.074 [-0.658, +0.516] | +0.15 |
| **gap E2 - (E0 + R_T4)** | **-0.760 [-1.254, -0.235]** | **-0.737 [-1.212, -0.233]** | +0.02 |

Mean draft distance, from the dumps:
- Adjacent checkpoints of the same run: 0.029 m (E0) and 0.043 m (E2 tau0); final pose 0.07 m and 0.105 m.
- E2 vs E0 at epoch 29 (two different trainings): 0.30 m; final pose 0.74 m, with 23.6 % of tokens more than 1 m apart.

Findings:
- The late-checkpoint floor is 0.02 to 0.17 PDMS for a planner and 0.05 to 0.07 for either refiner gain.
- The gap does not move between checkpoints (-0.76 / -0.74).
- Two independent trainings differ about 7 to 10 times more in the drafts than adjacent checkpoints do. So the
  planner-vs-planner term (-0.22 at epoch 29, -0.07 at epoch 28) is exactly the kind of quantity a seed can move by
  0.4 to 0.6 (NAVSIM).

**Recipe differences between E0 and E2.** From the archived hydra configs; everything else is identical.

| item | E0 (interaction_final) | E2 | consequence |
|---|---|---|---|
| devices x accumulate x batch | 2 x 16 x 4 = 128 | 4 x 8 x 4 = 128 | same global batch, same 665 optimiser steps / epoch, 19,950 steps in both |
| grad-balance warmup / interval (micro-batches per rank) | 10,600 / 200 | 5,300 / 100 | the same in optimiser steps (about 1 epoch / every 12.5 steps) |
| per-GPU batch (BatchNorm statistics) | 4 | 4 | same |
| sessions | interrupted at step 108, then resumed from epoch 6 (version_1 last.ckpt -> version_2) | one session | different data order / RNG stream: seed-like noise |
| validation | every 5 epochs | none | RNG only |
| dataloader workers | 4 | 6 | augmentation RNG only |
| refiner losses into the BEV encoder | none | `ref_bev_grad_scale` 1.0, plus 12 KD controls and the surrogate, with the refiner at lr x3 | changes the shared BEV. This is a real treatment difference, not noise, and a candidate for the E2 tau0 - E0 = -0.22 [-0.85, +0.40] |

The GPU count and accumulation do not change the optimisation at the level of the global batch. The draft component
(-0.11 under the same teacher) is within any plausible seed spread. The refiner component (-0.65) is measured on
identical drafts, so it does not involve the planner's seed at all. What remains uncertain there is the seed variance
of the student refiner's training (one E2 run) and of the teachers' training (one fold-0 seed-0 run each). Neither is
measured. R_T3 vs R_T4 (+1.47 vs +2.04 on E0) shows that teacher recipes alone move the teacher gain by about 0.6,
although that is a recipe change, not a seed.

## Answer

- **H8 explains about 2 % of the gap.** Point estimate 0.015 of 0.76, at most about 0.03 (about 4 %). Not supported.
- **Of the -0.76:**
  - -0.11 [-0.62, +0.39] is draft-side. It is not significant, is consistent with seed / recipe noise, and is the only
    part H9 can plausibly absorb, about 15 %.
  - -0.65 [-0.94, -0.35] is the student refiner gaining less than the teacher on the same drafts. This part is robust to
    scene resampling, and the pipeline cannot explain it. Seed variance could explain it only in one case. The
    difference of two single-seed gains has std sqrt(2) * sigma. Explaining -0.65 at 2 sigma needs sigma >= 0.23 per
    refiner. That is plausible for a whole planner (0.4 to 0.6), but at the top of the readout-head range (0.05 to 0.5).
    It is unmeasured for the refiner.
- **H9: partly.** It covers the draft part, but not the refiner part without new seeds.
  - At the epoch-28 checkpoints of both runs the gap reproduces: -0.74 [-1.21, -0.23] vs -0.76 at epoch 29.
  - The student gain (+1.44 vs +1.50) and the teacher gain (+2.11 vs +2.04) are stable to within 0.07.
  - So the gap is not a last-checkpoint accident. Training-length / checkpoint choice explains about 0 of it.
  - Only an independent seed of the student refiner, or of E2, would test the remaining seed explanation.

## Files

- `pipeline_h8.py`:
  - `ext`: the stage-T path, with `--fp32` to force autocast off.
  - `stagee`: the stage-E code-path teachers.
  - `stack`: builds the stacked drafts for scoring.
- `pipeline_h8_analysis.py`: produces `pipeline.json`.
- Large files are under `/home/external-user/ssd/yongjae_refiner/stageE_diag/pipeline/`:
  - `E0_RT4_fp32`, `E0_RM4_fp32`
  - `E0_R{T,M}4_stageE`, `E2t0_R{T,M}4_stageE` (every 4th token)
  - `E2t0_RT4`, `E2t0_RM4`, `E2t0_Rnone4`
  - `score_e0`, `score_e2t0`, `score_h9`
  - `h9/` (E0 epoch-28 pkl, E2 epoch-28 dump)
  - `logs/`
  - `run_*.sh`, `finish_h9.sh`, `score.sh`
- No existing run, eval or report was modified.
