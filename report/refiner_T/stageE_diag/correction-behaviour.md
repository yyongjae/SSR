# Stage E, E2 vs E0+R_T: H3 (student under-corrects) and H4 (student corrects in another direction)

Descriptive diagnostics on navtest, 12,146 tokens and 136 logs. Every CI is a log-cluster paired bootstrap (10,000 draws, seed 0, 95 %). Numbers come from `correction-behaviour.json`, produced by `correction_behaviour.py`. Counterfactual drafts come from `build_control_swaps.py`.

## Verdict

- **H3 (under-correction): supported.**
  - The student's corrections point the right way but are too small and too rare.
  - It explains about all of the refiner part of the gap, which is 0.65 of the 0.76 PDMS points (85 %).
  - It is mostly lateral, not braking.
- **H4 (direction disagreement): not supported.**
  - Only 1.4 % of the teacher's lateral corrections are opposed by the student.
  - Those tokens account for about 0.01 points.

## 1. Where the 0.76 points go (same-draft decomposition)

| quantity | PDMS points (95 % CI) |
|---|---|
| E0+R_T4 − E2 (total gap) | +0.76 [+0.24, +1.25] |
| draft part: E0+R_T4 − (R_T4 on E2's tau0) | +0.11 [−0.39, +0.62] |
| refiner part: (R_T4 on E2's tau0) − E2 | **+0.65 [+0.34, +0.94]** |
| same for R_M4: (R_M4 on E2's tau0) − E2 | +0.60 [+0.25, +0.95] |

- R_T4 gains +2.14 on E2's tau0 [+1.63, +2.72], against +2.04 on E0's drafts. E2's draft is not harder to refine.
- The earlier rough split (−0.21 draft, −0.54 refiner) came from comparing gains on different drafts. Measured on the same draft, the gap is almost all refiner.
- Checks:
  - Teachers re-decoded on CPU with the student's v0 match their pred tau1 to 8e-6 m.
  - The packed v0/a0/eds/cmd are identical to the student's status_feature inputs (max 1e-6).
  - The re-decoded student reproduces the official E2 PDMS on every token.

## 2. Counterfactual control swaps on E2's tau0 (official scorer, PDMS × 100)

| arm | PDMS | vs S (E2 final) | fixed / new fails vs S |
|---|---|---|---|
| E2 tau0 | 84.66 | | |
| S = E2 final | 86.15 | — | — |
| T = R_T4 controls | 86.80 | +0.65 [+0.34, +0.94] | 184 / 98 |
| M = R_M4 controls | 86.75 | +0.60 [+0.25, +0.95] | 188 / 111 |
| T lon + S lat | 86.42 | +0.26 [+0.15, +0.39] | 44 / 6 |
| S lon + T lat | 86.53 | +0.38 [+0.09, +0.68] | 141 / 96 |
| M lon + S lat | 86.18 | +0.03 [−0.02, +0.09] | 15 / 6 |
| S lon + M lat | 86.76 | +0.60 [+0.27, +0.95] | 176 / 102 |
| **S × 2** (student's own decoded controls doubled) | **87.00** | **+0.85 [+0.59, +1.13]** | 145 / 30 |
| S × 3 | 87.16 | +1.01 [+0.66, +1.40] | 205 / 69 |

- Doubling the student's own corrections, with no teacher information, reaches:
  - 87.00, compared with E0+R_T4 at 86.91 (+0.09 [−0.40, +0.60]).
  - +0.20 [−0.12, +0.55] over R_T4 on the same draft.
- S × 3 is +0.37 [+0.01, +0.75] over R_T4 on the same draft.
- So the student's direction already carries the information. What is missing is magnitude.
- Caveat: k was not tuned (only 2 and 3 tried), but it is chosen on navtest, so this is descriptive only.
- Channels:
  - Lateral carries the larger share: T lateral gives +0.38 and M lateral gives +0.60, which is all of M's advantage.
  - Braking adds +0.26, and only from R_T (NC +0.24, TTC +0.30). R_M's braking adds nothing.

## 3. Control-level comparison on the same draft (E2 tau0)

Controls are the KD controls: c_lon[2:] and e_lat[2:].

| | S | T (R_T4) | M (R_M4) |
|---|---|---|---|
| brake rate, all | 2.2 % [1.6, 3.0] | 5.0 % [4.0, 6.1] | 3.1 % [2.5, 3.9] |
| brake rate, tau0 fail_any (n 1560) | 4.9 % [2.6, 7.6] | 12.1 % [8.6, 15.7] | 5.1 % [3.1, 7.5] |
| mean shortening at 4 s, fail_any (m) | 0.029 | 0.124 | 0.065 |
| lateral deviation > 0.1 m, all | 7.9 % | 13.0 % | 13.8 % |
| mean lateral deviation, fail_any (m) | 0.137 | 0.203 | 0.220 |

Paired differences on fail_any tokens (S − T):
- brake −7.2 pp [−9.4, −4.9]
- shortening −0.095 m [−0.138, −0.056]
- lateral −0.066 m [−0.086, −0.047]

Against R_M, the brake rate does not differ (−0.3 pp [−1.9, +1.3]), but lateral is −0.083 m [−0.114, −0.057].

**Agreement (all tokens; the fail_any values are in the json):**

| | vs T | vs M |
|---|---|---|
| L1 lon (m/s) / L1 lat (m) | 0.024 / 0.024 | 0.016 / 0.028 |
| P(S brakes \| teacher brakes) | 28 % [22, 35] | 36 % [26, 46] |
| P(teacher brakes \| S brakes) | 64 % | 50 % |
| lon gain ⟨c_S, c_X⟩/\|c_X\|² on teacher-brake drafts | median 0.00, mean 0.13 | median 0.00, mean 0.21 |
| shortening ratio S/X when X shortens > 0.5 m | 0.14 [0.08, 0.20] | 0.17 [0.08, 0.26] |
| lateral, teacher active (\|e\| > 0.1 m): same sign / **opposite** / S < 5 cm | 46 % / **1.4 %** / 52 % | 42 % / **1.8 %** / 56 % |
| cos(e_S, e_X) when both active | 0.92 [0.88, 0.94] | 0.89 [0.85, 0.92] |
| lat gain on teacher-active drafts | median 0.13, mean 0.43 | median 0.07, mean 0.32 |

- When the student brakes, the teacher usually agrees (64 %).
- When the student moves laterally, it moves the teacher's way (cos 0.9).
- It simply acts less often and less strongly. This is under-correction, not disagreement.

## 4. Outcome split (tau0 failures, n = 1560)

Vs T: 175 tokens where T fixes and S does not; 63 where S fixes and T does not; 148 fixed by both; 1174 fixed by neither.

On the 175 tokens that T fixes and S misses:

| | S | T |
|---|---|---|
| brake rate | 11 % | 30 % |
| shortening (m) | 0.07 | 0.43 |
| lateral deviation (m) | 0.11 | 0.41 |

- On those 175 tokens the student opposes the teacher's lateral direction on only 1.5 % of the lateral-active ones.
- These tokens are worth +1.14 [+0.90, +1.40] points.
- The 63 tokens that S fixes and T misses are worth −0.43 back. On those, S moves more laterally (0.43 m vs 0.28 m).
- Vs M the picture is the same: 177 missed fixes (S lateral 0.17 m vs M 0.57 m) and 71 reverse cases.

**By failure type on E2 tau0 (non-exclusive).** In the table, "contrib" is sum(PDMS_T − PDMS_S)/N.

| type | n | S fix | T fix | M fix | S / T brake | S / T lat (m) | contrib T−S | contrib M−S |
|---|---|---|---|---|---|---|---|---|
| NC agent, moving (> 0.5 m/s) | 70 | 3 % | 11 % | 9 % | 16 / 33 % | 0.02 / 0.13 | +0.05 | +0.03 |
| NC agent, stationary | 164 | 6 % | 17 % | 2 % | 9 / 32 % | 0.13 / 0.22 | +0.22 | −0.07 |
| NC static object (nc 0.5) | 34 | 32 % | 24 % | 26 % | 9 / 29 % | 0.34 / 0.59 | −0.01 | −0.01 |
| TTC | 505 | 5 % | 11 % | 7 % | 5 / 13 % | 0.05 / 0.08 | +0.11 | +0.06 |
| DAC | 860 | 20 % | 26 % | 32 % | 3 / 6 % | 0.20 / 0.27 | **+0.46** | **+0.83** |

- DAC, where lateral magnitude matters, carries the largest share.
- The rest comes from R_T's braking on stationary-agent collisions and TTC.
- The student's brake rate on those types is about 1/3 of R_T's.

**Token-level attribution of the refiner gap.**

| class vs T | n | contrib to T − S (pts) |
|---|---|---|
| under (S shortens < 0.75× or lateral gain < 0.75 where T acts) | 1653 | **+0.82 [+0.55, +1.08]** |
| of which lateral only / lon only / both | 1143 / 435 / 75 | +0.61 / +0.14 / +0.07 |
| direction (lateral cos < 0, \|e_S\| > 5 cm) | 24 | +0.007 [−0.02, +0.04] |
| S matches or exceeds T | 336 | −0.08 |
| S acts, T passive | 330 | −0.16 [−0.24, −0.09] |
| both passive | 9803 | +0.06 |
| **total** | | +0.65 |

Vs M: under +0.80 (lateral only +0.79); direction +0.006; S acts while M is passive −0.26.

Class thresholds are arbitrary (0.1 m, 0.75). The ranking is robust to them because the direction class is tiny under any threshold: at most 4.5 % of both-active tokens have cos < 0.

## 5. z_lon saturation (dead zone)

Saturation is **not** what blocks the missed brakes.

- Student z is saturated overall:
  - all six z > 3 on 74 % of drafts; all six > 5 on 56 %
  - mean relu(z)² = 69 on navtest, against 54 in training (all micro-batches, epochs 25–29) and 62 on the training tau0-only micro-batches
- The teachers are more saturated still: all z > 3 on 83 % (T) and 88 % (M); relu² 100 and 152. Saturation on drafts that need no brake is normal teacher behaviour.
- On drafts where R_T brakes, only 12 % [7, 17] of student drafts are dead at z > 3. The student's median z there is about (0.15, 0.25, 0.04, 0.04, −0.18, 0.13): at the brake boundary, not in the flat tail.
  - The student makes tiny or zero speed offsets exactly where R_T's z is −1 to −2.
  - At z ≈ 0 the tanh gradient is fully alive. The shrinkage comes from the objective, not from vanishing gradients.
- Correlation with missed fixes (323 tau0 failures that T fixes):
  - P(S misses | all z > 3) = 0.55 [0.46, 0.65] vs 0.53 [0.41, 0.65] when not dead. No association.
  - Among T-brake fixes (73), only 5.5 % are dead for S.

## 6. Training-time KD distance vs navtest (train–test gap)

Train = rank-0 micro-batches of epochs 25–29 whose four drafts were all unperturbed sg(tau0) (n = 2171, navtrain; teachers in fp32 on navtrain caches). Test = navtest E2 tau0 (teachers in fp16 on navtest caches).

| | train, tau0-only | train, all drafts | navtest tau0 |
|---|---|---|---|
| L1 lon vs T | 0.018 [0.016, 0.021] | 0.039 | 0.024 [0.019, 0.030] |
| L1 lat vs T | 0.013 [0.013, 0.014] | 0.027 | 0.024 [0.022, 0.027] |
| L1 lon vs M | 0.007 [0.005, 0.008] | 0.029 | 0.016 [0.012, 0.021] |
| L1 lat vs M | 0.013 [0.012, 0.013] | 0.026 | 0.028 [0.024, 0.032] |
| S brake / T brake / M brake | 2.1 / 4.7 / 2.0 % | 8.1 / 9.7 / 7.2 % | 2.2 / 5.0 / 3.1 % |

- The **braking under-correction is already present in training**. The S/T brake ratio is 0.44 on training tau0 drafts and 0.44 on navtest, so it is not a generalisation gap.
- In training the student matches R_M's brake rate (2.0 %), not R_T's.
- The KD distances grow about 1.3× (lon vs T) to 1.8–2.4× (lat, and vs M) from train to navtest. Part of this is expected, because navtest drafts fail more often than the planner's in-sample navtrain drafts, so the teachers correct more. The lateral gap may also carry some generalisation loss. With these logs the two cannot be separated.

## 7. Mechanism candidates

These are consistent with the data. They are for you to weigh, not decided here.

1. **Two-teacher L1 target.** KD is the mean of |c_S − c_T| and |c_S − c_M|. Between the two teacher values that sum is flat, so any value between them is optimal and the surrogate or prior decides.
   - The teachers agree on only 29 % of R_T's brakes. On lateral, both are active with cos > 0.5 on 716 tokens.
   - The student brakes on 60 % [46, 72] of drafts where both teachers brake, but only 16 % where just R_T does and 15 % where just R_M does.
   - Lateral: when both teachers act in the same direction, max|e_S| is 0.28 m against R_T's 0.46 m; when only R_T acts, 0.08 m against 0.32 m.
   - Where the teachers disagree, the student settles near "no correction".
2. **Gate never trained, correction always applied.** The student cannot pay for a large correction with a gate. The −0.16 to −0.26 points from "S acts while the teacher is passive" suggests the student has learned to hedge.
3. **Tiny share of the refiner loss.** In the last epoch KD is about 0.3 % of the loss value and about 10 % of the BEV gradient norm, and the surrogate is similar. Magnitude is weakly constrained.
4. **Training draft mix.** After epoch 5, 50 % of drafts are perturbed human drafts with large corrections, and the other 50 % are in-sample tau0 drafts that rarely need correction (teacher brake rate 4.7 % / 2.0 %). Few training examples resemble navtest failures.

H3's evidence does not depend on which of these mechanisms holds.

## Files

- Counterfactual swap arms and their scores: /home/external-user/ssd/yongjae_refiner/stageE_diag/teacher_on_e2/ (swaps.npz, scores.parquet, swaps_meta.json, logs/)
- Teacher predictions on E2's tau0 are reused from the sibling same-draft run: /home/external-user/ssd/yongjae_refiner/stageE_diag/same_draft/{R_T4,R_M4}/pred.npz (pkl sha256 checked)
- Scripts:
  - /home/external-user/yongjae/SSR/report/refiner_T/stageE_diag/build_control_swaps.py
  - /home/external-user/yongjae/SSR/report/refiner_T/stageE_diag/correction_behaviour.py

No existing run, eval or report was modified.
