# Stage-E diagnosis: KD design (H5) and training draft mix (H6)

Question: why does E2 (86.15) score below E0+R_T4 (86.91) on navtest? Gap: 0.76 PDMS points, CI [+0.24, +1.25].

Everything below is descriptive. Bootstrap: paired by token, 136 navtest log clusters, 10,000 draws, seed 0, percentile 95% CI. PDMS is in points (x100). All 12,146 navtest tokens are used, with 0 scorer errors.

## What was run

- **Teachers.** The two frozen E2 KD teachers were applied to E2's own navtest tau0 drafts: run-4 R_T (BEVFusion) and run-4 R_M (ReSMap), `ckpt_best`, sha-identical to `stageE/teachers/*`. Each ran twice. The fp16 run uses the same autocast as `e0_teacher_refine`. The fp32 run matches E2 training (`trainer.params.precision=32`), so its decoded controls are exactly the KD targets for these drafts. The teachers get the same ego inputs as the student: largest difference in v0 is 1e-6, and a0, eds and cmd match exactly. fp16 and fp32 score the same: R_T4 differs by 0.000 points, R_M4 by 0.003 points.
- **Two KD-consensus corrections, decoded from the teachers' fp32 controls.** These are re-encoded to (z, w) and decoded in mode A. A round trip of each teacher's own controls reproduces its trajectory to within 9e-5 m.
  - **MID** is the element-wise mean of the two teachers' decoded controls (c_lon[2:], e_lat[2:]). It is the midpoint of the L1-KD minimiser set.
  - **WEAK** is the KD-optimal point with the least correction. For lon it uses c = max(c_T, c_M), i.e. less braking. For lat it takes the smaller |e| when the two teachers agree in sign, and 0 when they have opposite signs.
  - Why these two matter: with two teachers, L = ½(|s−T| + |s−M|) is flat on the whole element-wise interval [min, max]. Its subgradient there is 0, so inside the interval only the GT surrogate places the student. The surrogate includes progress with weight 2.
- **Training log.** `stageE_steps.jsonl` has 159,600 rank-0 micro-batches of batch size 4. Per-draft-type estimates come from "pure" micro-batches: 2,171 all-tau0 and 1,292 all-human-perturbed in epochs 25–29. An OLS fit on `frac_draft_tau0` agrees with them.

Scripts are in `kd_design/`:
- `teachers_on_e2_tau0.py`
- `build_consensus.py`
- `train_mix.py`
- `analyze_kd_design.py`

Large files are in `/home/external-user/ssd/yongjae_refiner/stageE_diag/kd_design/`. The full numbers are in `kd-design.json`.

## Arm means on navtest (x100)

| arm | PDMS | NC | DAC | TTC | EP |
|---|---|---|---|---|---|
| E2 tau0 | 84.66 | 97.93 | 92.92 | 93.66 | 79.67 |
| **E2 (student)** | **86.15** | 98.17 | 94.29 | 94.04 | 80.99 |
| E2tau0 + R_T4 | 86.80 | 98.49 | 94.69 | 94.36 | 81.56 |
| E2tau0 + R_M4 | 86.75 | 98.08 | 95.12 | 93.90 | 81.51 |
| **E2tau0 + MID** | **87.09** | 98.40 | 95.08 | 94.29 | 81.83 |
| E2tau0 + WEAK | 86.01 | 98.07 | 94.24 | 93.93 | 80.88 |
| E0 + R_T4 (reference) | 86.91 | 98.40 | 94.99 | 94.34 | 81.59 |

## Gap decomposition (PDMS points)

| component | value [95% CI] | share of 0.76 |
|---|---|---|
| gap: (E0+R_T4) − E2 | +0.76 [+0.24, +1.25] | 100% |
| draft: (E0+R_T4) − (E2tau0+R_T4) | +0.11 [−0.39, +0.62] | 15% (n.s.) |
| same draft, teacher vs student: (E2tau0+R_T4) − E2 | +0.65 [+0.35, +0.94] | 85% |
| averaging itself: (E2tau0+R_T4) − (E2tau0+MID) | **−0.30 [−0.55, −0.06]** | −39% (averaging helps) |
| MID − E2 | +0.94 [+0.65, +1.25] | — |
| WEAK − E2 | −0.14 [−0.36, +0.05] | — |
| (E2tau0+R_T4) − WEAK (upper bound for "sits at the weak end") | +0.79 [+0.55, +1.03] | 104% |

The same-draft part, (E2tau0+R_T4) − E2, is split below by whether the teachers correct and agree. Material correction means braking with min c_lon < −0.1 m/s, or lateral with max |e_lat| > 0.1 m. "Agree" means both teachers brake or neither does, and both teachers move laterally with the same sign or neither does.

| token group | n | E2tau0 fail_any | (R_T4) − E2 contrib | MID − E2 contrib | WEAK − E2 contrib | fix rate of E2tau0 fails: student / WEAK / MID / R_T |
|---|---|---|---|---|---|---|
| no teacher correction | 9041 | 7.2% | −0.03 [−0.11, +0.04] | −0.01 [−0.07, +0.06] | −0.03 [−0.09, +0.02] | 2.6 / 1.2 / 1.9 / 2.0% |
| teachers agree and correct | 785 | 47.3% | **+0.42 [+0.24, +0.60]** | +0.63 [+0.42, +0.87] | **+0.18 [+0.05, +0.33]** | 31.8 / 38.3 / 55.5 / 48.0% |
| teachers disagree | 2320 | 23.4% | **+0.26 [+0.05, +0.46]** | +0.32 [+0.17, +0.47] | −0.29 [−0.44, −0.17] | 14.0 / 5.7 / 23.4 / 24.4% |

## H5: two-teacher L1 KD makes the student under-correct where the teachers disagree

The teachers disagree often, and mostly on *whether* to correct at all, not on the direction.

- **Longitudinal.**
  - R_T brakes on 5.0% of tokens [4.0, 6.1] and R_M on 3.1% [2.5, 3.9].
  - Only R_T brakes on 3.5% of tokens, only R_M on 1.6%, and both on 1.4%.
  - Braking depth correlates between the two teachers at r = 0.38.
- **Lateral.**
  - Only R_T corrects on 7.1% of tokens, only R_M on 7.7%, both with the same sign on 5.9%, and both with opposite signs on 0.5%.
  - e_lat correlates at r = 0.49.
- **Overall.** The mean token L1 between the teachers is 0.033, larger than the student's distance to either teacher (0.024 to R_T, 0.022 to R_M). 23.7% of tokens have teacher disagreement above 0.02.

The student sits at the weak end of the interval where the KD loss is flat.
- **Elements where the teachers disagree** (|T−M| > 0.02, 16,707 elements):
  - 62% of student values fall between the two teachers, 23% fall beyond the weaker teacher toward no correction, and 4% beyond the stronger one.
  - On a scale where 0 is the weak teacher and 1 the strong one, the student's median position is λ = −0.003 (IQR −0.15 to 0.09). For lon alone the median is 0.00 with IQR [0, 0], and 96% of values have λ < 0.5.
- **Tokens where only one teacher brakes.** The student brakes on 13% of R_T-only tokens (depth 0.09 vs 1.54 m/s) and 10% of M-only tokens (0.07 vs 1.52).
- **Tokens where only one teacher corrects laterally.** The student corrects on 20% of R_T-only tokens and 17% of M-only tokens.
- **Magnitude.** On tokens where the teachers disagree, the student's correction is 0.36× the teachers' mean magnitude (n = 2,402).
- **Fixes, among the 1,560 E2tau0 fail_any tokens:**
  - Where only R_T fixes the token (n = 147), the student fixes 25% and MID 56%.
  - Where only R_M fixes it (n = 141), the student fixes 21% and MID 57%.
  - WEAK fixes only 3–4% of these.

The student also under-corrects where the teachers agree, and the flat-interval effect cannot explain that.
- On tokens where both teachers brake (n = 172), the student brakes on 52%, with depth 0.65 vs 2.19 / 2.14 m/s.
- On tokens both teachers fix (n = 176), the student fixes 63%, while WEAK fixes 97% and MID 100%.
- In the "agree and correct" group the student is below even WEAK: +0.18 [+0.05, +0.33].

**Verdict on H5: partly supported. As worded, it is not supported.**
- Averaging the two teachers is not harmful. MID beats R_T4 alone by +0.30 [+0.06, +0.55] and E0+R_T4 by +0.18, so the consensus target itself is better than either teacher.
- What is supported is the mechanism of the L1 loss. With two teachers, L1 leaves [T, M] as a zero-gradient zone, and the surrogate (progress weight 2) pulls the student to the less-correcting end. The student behaves like WEAK: E2 − WEAK = +0.14 [−0.05, +0.36].
- **Share of the 0.76 gap:**
  - Measured on the tokens where the teachers disagree: +0.26 [+0.05, +0.46], about 35% (CI about 6–61%).
  - If the student had matched MID on those tokens, E2 would gain +0.32 [+0.17, +0.47] there.
  - The loose upper bound, R_T − WEAK = 0.79, is not specific to H5: it also contains the under-correction on agreeing tokens.

## H6: the training draft mix teaches the student to correct too little on test tau0

Training draft mix:
- Epochs 0–4 use 100% GT-perturbed drafts.
- From epoch 5 on, 46.6% of drafts are GT-perturbed and 53.4% are sg(tau0). The tau0 share includes the 6.9% of perturbations that were invalid and fell back to tau0.
- The KD weight w_ema averages 0.60 in epochs 25–29. Weighted KD and weighted surrogate have equal loss value there (0.28% each of the total loss value).

| draft type (epochs 25–29 train; navtest = E2 tau0) | R_T brakes | R_M brakes | student brakes | student / teacher-mean | KD L1 to R_T | KD L1 to R_M |
|---|---|---|---|---|---|---|
| train, GT-perturbed | 15.8% [14.8, 16.8] | 13.7% [12.9, 14.5] | 15.6% [14.7, 16.5] | **1.06** | 0.055 | 0.049 |
| train, in-sample tau0 | 4.7% [4.2, 5.2] | 2.0% [1.7, 2.2] | 2.1% [1.7, 2.4] | **0.62** | 0.016 | 0.0095 |
| navtest tau0 | 5.0% [4.0, 6.1] | 3.1% [2.5, 3.9] | 2.2% [1.6, 3.0] | **0.55** | 0.024 | 0.022 |

- **The student has learned to correct differently by draft type.** On GT-perturbed drafts it brakes about as often as the teachers (1.06×). On its own tau0 it brakes about 0.6× as often, and that is already true in-sample. Tau0-type drafts rarely need correction in training: E0's in-sample collision fail rate is 0.70% vs 2.30% on navtest (report 32, L253).
- **Test tau0 drafts need more correction than training tau0 drafts,** by the teachers' own rates: R_T goes from 4.7% to 5.0% and R_M from 2.0% to 3.1%. The student stays at 2.1% → 2.2%. The student-to-teacher ratio drops only from 0.62 to 0.55, and those CIs overlap.
- **KD fit is worse on navtest.** The student–teacher L1 is 1.5× higher for R_T and 2.3× higher for R_M than on in-sample tau0.
- **Recall.** The student brakes on 28% of the tokens where R_T brakes and on 36% of those where R_M brakes. It brakes on 0.6% of the tokens where neither teacher brakes.

**Verdict on H6: partly supported.**
- Under-correction on tau0 is real, and it depends on the draft type: 0.62× in-sample vs 1.06× on perturbed drafts.
- The part of H6 about the test set is weak. The under-correction exists in-sample too, and the extra drop on navtest (0.62 → 0.55) is within noise.
- H6 is the main candidate for the "teachers agree and correct" share of the gap: +0.42 [+0.24, +0.60], about 55%. There the student falls short of even WEAK. This analysis cannot separate H6 from other causes of imitation error there, such as the student's BEV quality or capacity; that would need a training intervention.
- **Share of the 0.76 gap:** not identifiable on its own. At most the agreeing-token share of about 0.42 (≈55%), shared with other imitation causes.

## Candidate design changes (options for the user; nothing decided)

- **Mean-seeking KD target.** Use L1 or L2 toward the element-wise mean of the teachers' controls, i.e. MID, instead of the mean of per-teacher L1. Post hoc, MID on E2's drafts scores 87.09 (+0.94 over E2). This is one look at a variant defined in advance and not tuned, but it is still descriptive, not a validated arm.
- **Union / max-correction rule.** Brake whenever either teacher brakes, or use the stronger teacher per channel: R_T for lon (NC/TTC) and R_M for lat (DAC). This is not evaluated here. The R_T4 − R_M4 contrasts on E2's drafts (NC +0.41, DAC −0.43) show the two teachers are complementary by channel.
- **Target draft mix and weighting.** Up-weight KD on tau0 drafts where the teachers correct, or raise the tau0 share, so the student does not learn "tau0 needs no correction". This targets the 0.62× in-sample under-braking.
- **Check the surrogate's pull inside the flat zone.** Inside the flat zone only the surrogate acts, and its progress weight (2) favours not braking.

## Caveats

- E2tau0 + R_T4 / MID / WEAK use the teachers' privileged BEVs (BEVFusion, ReSMap). They show what the student would score if it reproduced the targets exactly, not what a student reading its own BEV can reach.
- The training-log rates are rank-0 micro-batches in train mode, on the tau0 drafts of the model as it trained. The navtest rates are from the final checkpoint.
- The teachers were trained on 14,161 navtrain tokens, so part of the training-time teacher targets are teacher in-sample.
- One seed, one checkpoint (epoch 29).
