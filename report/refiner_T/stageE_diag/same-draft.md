# Same-draft test: why E2 (86.15) is below E0 + R_T4 (86.91) and E0 + R_M4 (86.81)

**Verdict: H1 is supported. The student refiner is weaker than the teachers on the same drafts. H2 is not supported: E2's drafts are not measurably worse or harder to refine.** On E2's own tau0 drafts, the frozen R_T4 scores +0.65 PDMS points above the student [+0.34, +0.94] and R_M4 scores +0.60 above it [+0.25, +0.95]. This same-draft refiner difference accounts for about 85 % of the 0.76-point gap to E0+R_T4 (95 % CI of the share 43 % to 233 %). The draft difference (E2_tau0 vs E0) is −0.22 points with a CI of [−0.85, +0.40], so it is not significant. The teachers also gain as much on E2's drafts as on E0's (+2.14 vs +2.04).

요약: E2가 E0+R_T/R_M보다 낮은 이유는 대부분(약 85~90 %) 학생 refiner 자체가 약하기 때문입니다. 같은 E2 draft에 teacher를 적용하면 학생보다 +0.6~0.65점 높습니다. E2 planner draft가 E0보다 나쁘거나 고치기 어려워서 생긴 차이는 유의하지 않습니다. 학생은 teacher의 보정을 약 25~50 % 크기로만 따라 하고, 두 teacher가 모두 보정하는 토큰에서만 제대로 움직입니다. 그 결과 fix는 적고(211 vs 323) new-fail도 적은(16 vs 42) 보수적인 refiner가 되었습니다.

## Setup
The frozen stage-T refiners were run with `tools/refiner/refine_external_drafts.py predict` (ckpt_best, theta 0, amp fp16, GPU 0/1) on E2's tau0 drafts (`e2_tau0_navtest_trajectories.pkl`, sha e7e1bfa1…). Scoring used `tools/refiner/score_trajectories.py`: 12,146 tokens, 136 logs, 0 errors in every arm. E0 and E0+R_X come from `e0_teacher_refine/`. E2_tau0 and E2_final come from `stageE_diag/scores.parquet` and match the official CSVs exactly. All contrasts are paired on the same tokens with a log-cluster bootstrap (10,000 draws, seed 0, 95 % percentile CI). Share CIs are the ratio of bootstrap means taken over the same log resamples. fail_any = NC, DAC, DDC or TTC < 1.
Code: `same_draft_report.py` (this folder). Data: `/home/external-user/ssd/yongjae_refiner/stageE_diag/same_draft/{R_T4,R_M4,R_none4,R_T3,E2_tau0}`, with the launch scripts `run_predict.sh` and `score_all.sh`.

## Arms (x100)
| arm | PDMS | NC | DAC | TTC | EP | fail_any |
|---|---|---|---|---|---|---|
| E0 | 84.88 | 97.85 | 93.31 | 93.57 | 79.77 | 1523 |
| E0+R_T4 | 86.91 | 98.40 | 94.99 | 94.34 | 81.59 | 1255 |
| E0+R_M4 | 86.81 | 97.98 | 95.35 | 93.86 | 81.49 | 1272 |
| E0+R_none4 | 84.78 | 97.88 | 93.19 | 93.63 | 79.64 | 1530 |
| E0+R_T3 | 86.35 | 98.21 | 94.56 | 94.15 | 81.05 | 1324 |
| **E2_tau0** | 84.66 | 97.93 | 92.92 | 93.66 | 79.67 | 1560 |
| **E2_final (student)** | 86.15 | 98.17 | 94.29 | 94.04 | 80.99 | 1365 |
| **E2_tau0+R_T4** | 86.80 | 98.49 | 94.69 | 94.36 | 81.56 | 1279 |
| **E2_tau0+R_M4** | 86.75 | 98.08 | 95.12 | 93.90 | 81.51 | 1288 |
| E2_tau0+R_none4 | 84.75 | 97.95 | 93.00 | 93.75 | 79.71 | 1542 |
| E2_tau0+R_T3 | 86.53 | 98.38 | 94.50 | 94.27 | 81.31 | 1313 |

(E0 uses the batched score of the pkl, 84.88. The official CSV gives 84.87; see e0_teacher_refine.md.)

## H1: teacher vs student on the same (E2 tau0) drafts
| contrast | PDMS | NC | DAC | TTC | EP | fixed / new_fail |
|---|---|---|---|---|---|---|
| E2+R_T4 − E2_final | **+0.65 [+0.34, +0.94]** | +0.32 | +0.40 | +0.32 | +0.57 | 184 / 98 |
| E2+R_M4 − E2_final | **+0.60 [+0.25, +0.95]** | −0.09 | +0.83 | −0.14 | +0.52 | 188 / 111 |
| E2+R_T3 − E2_final | +0.38 [+0.12, +0.62] | +0.21 | +0.21 | +0.23 | +0.32 | 145 / 93 |
| E2+R_none4 − E2_final | −1.40 [−1.92, −0.99] | −0.21 | −1.28 | −0.29 | −1.28 | 38 / 215 |
| E2+R_T4 − E2+R_M4 | +0.05 [−0.28, +0.38] | +0.41 | −0.43 | +0.46 | +0.05 | 187 / 178 |

Gains over the unrefined draft:
| refiner | on E2 tau0 | on E0 | gain@E2 − gain@E0 |
|---|---|---|---|
| student | **+1.50 [+1.06, +2.03]** (fixed 211, new 16) | n/a (the student needs E2's BEV) | vs R_T4@E0: −0.54 [−0.93, −0.13] |
| R_T4 | +2.14 [+1.63, +2.72] (323 / 42) | +2.04 [+1.62, +2.49] (307 / 39) | +0.11 [−0.24, +0.47] |
| R_M4 | +2.09 [+1.51, +2.77] (317 / 45) | +1.94 [+1.44, +2.48] (300 / 49) | +0.16 [−0.25, +0.59] |
| R_T3 | +1.87 [+1.41, +2.38] (274 / 27) | +1.47 [+1.16, +1.82] (222 / 23) | +0.40 [+0.08, +0.73] |
| R_none4 | +0.10 [−0.02, +0.22] (36 / 18) | −0.10 [−0.22, +0.02] (28 / 35) | +0.20 [+0.06, +0.36] |

On the same drafts, the student recovers 70 % of R_T4's gain (+1.50 / +2.14) and 65 % of its fixes (211 / 323). Its level sits between R_T3 and R_T4, and it is closest to run-3's R_T: E0+R_T3 − E2_final = +0.20 [−0.34, +0.70].

## H2: draft difference
- E2_tau0 − E0: **−0.22 [−0.85, +0.40]** PDMS. By metric: NC +0.08, DAC −0.40, TTC +0.09, EP −0.10. Token-level churn is large: 568 fixed and 605 new failures.
- Failing pools: E0 has 1523 failing tokens and E2_tau0 has 1560. 955 fail in both, 568 fail only in E0 and 605 only in E2_tau0 (Jaccard 0.45). By metric (E0 / E2 / both): NC 279 / 268 / 181, DAC 812 / 860 / 468, TTC 781 / 770 / 516.
- Every teacher gains at least as much on E2's drafts as on E0's (table above), so E2's drafts are not harder to refine.
- Fix rate per pool (fixed / failing):

| refiner | on | both_fail (955) | own-only fail pool | total fixed | new_fail |
|---|---|---|---|---|---|
| R_T4 | E0 | 173 | 134 / 568 | 307 (20.2 %) | 39 |
| R_M4 | E0 | 156 | 144 / 568 | 300 (19.7 %) | 49 |
| student | E2 | **99** | 112 / 605 | **211 (13.5 %)** | **16** |
| R_T4 | E2 | 177 | 146 / 605 | 323 (20.7 %) | 42 |
| R_M4 | E2 | 158 | 159 / 605 | 317 (20.3 %) | 45 |
| R_none4 | E2 | 13 | 23 / 605 | 36 | 18 |

  On the 955 tokens that fail in both E0 and E2, the student fixes 99 and R_T4 fixes 177 on the same drafts. Of E2_tau0's 1560 failures, 148 are fixed by both the student and R_T4, 175 only by R_T4 and 63 only by the student. R_T4 or R_M4 fixes 464, and the student fixes 34 that neither teacher fixes.

## Gap decomposition: (E0+X) − E2_final = [E0 − E2_tau0] + [gain_X@E0 − gain_X@E2] + [(E2+X) − E2_final]
| X | gap | draft (raw) | draft × refiner interaction | draft total (seen through X) | **refiner, same draft** |
|---|---|---|---|---|---|
| R_T4 | 0.76 [0.24, 1.25] | 0.22 [−0.40, 0.85] | −0.11 [−0.47, 0.24] | 0.11 [−0.39, 0.62] = 15 % [−133 %, 57 %] | **0.65 [0.34, 0.94] = 85 % [43 %, 233 %]** |
| R_M4 | 0.66 [0.13, 1.19] | 0.22 [−0.40, 0.85] | −0.16 [−0.59, 0.25] | 0.06 [−0.40, 0.54] = 10 % [−194 %, 60 %] | **0.60 [0.25, 0.95] = 90 % [40 %, 294 %]** |
| R_T3 | 0.20 [−0.34, 0.70] | 0.22 | −0.40 [−0.73, −0.08] | −0.18 | 0.38 [0.12, 0.62] (gap not significant) |

The earlier split in the previous step (−0.21 draft, the rest refiner) put the draft term first, without the teacher. With the teacher applied to both draft sets, the draft term falls to 0.11, because R_T4 gains more on E2's drafts. Either way, most of the gap is the refiner, and every draft term's CI includes 0. The shares have wide CIs because the gap itself is small, with a lower bound of 0.24.

## Where the student falls short (E2 drafts, split by what the teachers do)
Tokens are split by whether R_T4 and R_M4 correct: "active" means lon_live or |e_lat_final| > 0.1 m. Numbers are PDMS points vs E2_tau0, with the contribution to the mean in brackets.

| stratum | n | student | R_T4 | R_M4 | R_none4 |
|---|---|---|---|---|---|
| both teachers active | 911 | +13.3 (0.99) | +18.7 (1.41) | +21.2 (1.59) | +0.9 |
| exactly one active | 2127 | +2.3 (0.41) | +3.9 (0.67) | +2.2 (0.39) | +0.3 |
| neither active | 9108 | +0.12 (0.09) | +0.08 (0.06) | +0.15 (0.12) | −0.03 |

The R_T4 − student difference of 0.65 comes 0.41 from the both-active tokens (7.5 % of tokens), 0.26 from the one-active tokens and −0.03 from the rest.

## Correction size on the same drafts
| | lon_live | mean shortening m | lat_final > 0.1 m | mean abs(e_lat[2:]) m | mean disp m | disp > 0.5 m | z_lon mean |
|---|---|---|---|---|---|---|---|
| student | 2.2 % | 0.008 | 7.9 % | 0.016 | 0.042 | 2.2 % | 7.3 |
| R_T4 | 5.0 % | 0.045 | 13.0 % | 0.028 | 0.104 | 6.0 % | 9.0 |
| R_M4 | 3.1 % | 0.029 | 13.7 % | 0.031 | 0.096 | 5.8 % | 11.0 |
| R_T3 | 4.3 % | 0.036 | 8.4 % | 0.020 | 0.078 | 5.0 % | 10.7 |
| R_none4 | 1.4 % | 0.011 | 3.0 % | 0.008 | 0.027 | 0.7 % | 8.6 |

KD-space agreement on navtest uses the 12 decoded controls c_lon[2:] and e_lat[2:], the E2 KD space:

| | mean L1 to R_T4, R_M4 (KD loss) |
|---|---|
| student | 0.0230 |
| zero correction | 0.0253 |
| teacher midpoint | 0.0167 |

The student is only 9 % closer to the teachers than doing nothing. Its regression slope on the teacher midpoint is 0.25 for c_lon[7] (corr 0.56) and 0.50 for e_lat[7] (corr 0.68): the student is a shrunk copy of the teachers.

## Suspects (status)
1. **Student under-corrects on real planner drafts. CONFIRMED; this is the main cause.** Evidence: the H1 table, the strata table, the shrink slopes 0.25 / 0.50, and the fixed / new counts. The student fixes fewer tokens (211 vs 323) but also breaks fewer (16 vs 42). It is conservative, not noisy.
2. **KD target: L1 to two teachers that often disagree. PLAUSIBLE mechanism, not tested causally.** R_T4 vs R_M4 L1 is 0.033, larger than either teacher's distance from zero (0.027 / 0.024). Both teachers brake on only 1.4 % of tokens and exactly one brakes on 5.3 %. For the final lateral offset, both are active on 6.2 % and exactly one on 14.3 %. With L1 against two targets, any value between them is a minimizer, so when one teacher says "do nothing" KD gives no push to correct. 87 % of student controls lie inside the teachers' interval. The student acts in 60 % of both-active cases but only 15 % (braking) to 19 % (lateral) of one-active cases. A causal test needs a single-teacher KD arm or a KD target such as the midpoint or the larger correction.
3. **Train/test draft and BEV shift. PLAUSIBLE, not tested.** At the end of training (epoch 29, stageE_steps.jsonl), the student brakes on training drafts about as often as the teachers do (ref/live 8.5 % vs 9.9 % / 7.2 %). On navtest it brakes far less (2.2 % vs 5.0 % / 3.1 %). The training drafts are 47 % perturbed GT-human (hmix) and 53 % sg(tau0) on navtrain. On navtrain the planner is well fit: loss_plan_reg drops 0.058 → 0.013 and the surrogate col / ttc terms fall about 10x (t_ttc 0.058 → 0.007). Those sg(tau0) drafts are therefore nearly clean, unlike navtest drafts. The student's BEV on training tokens is also in-sample. The frozen teachers see out-of-sample BEVFusion caches on navtest and still transfer. Test: evaluate the student on held-out logs (dev) against the teachers, on the same drafts.
4. **The student's own BEV carries less information than BEVFusion / ReSMap. PLAUSIBLE, cannot be separated here.** R_none4 gains about 0, so the refiner gain comes from the BEV. The student's +1.50 shows its BEV is informative, and this experiment cannot tell capacity/information apart from training signal (suspects 2 and 3).
5. **E2's drafts are different or harder (H2). NOT SUPPORTED as a cause of the gap.** Draft effect is −0.22 [−0.85, +0.40], teacher gains on E2 drafts are equal or larger, and the draft share of the gap is 15 % with a CI including 0. The drafts do differ (Jaccard 0.45 of the failing pools, mean final-pose distance 0.74 m). A plausible source is bg1.0 (ref_bev_grad_scale 1.0, the spec default is 0.1): at epoch 29 the refiner losses carry 23 % of the bev_embed gradient norm (gshare_sur 0.13, gshare_kd 0.10), against 2 % averaged over epoch 0. This changes the draft but costs no significant PDMS.
6. **Longitudinal saturation (z_lon in the tanh dead zone). WEAK.** Student z_lon mean is 7.3 (the training ref/zdead grows from 3 to 54), but the teachers are just as saturated (9.0 to 11.0). It is not specific to the student.
7. **Ruled out:** ego inputs (dump v0 / a0 / eds / cmd vs the packed navtest split: a0, eds and cmd are bitwise equal, v0 ≤ 1e-6); decode path (both mode A, lon_st_slope 0 at inference); scorer (E2 dump == official CSVs); gate (theta 0 in every arm, the student gate is untrained but never applied); teacher weights (the KD teachers in stageE/teachers are sha256-identical to runs/stageT4_{T,M}_fold0_seed0/ckpt_best.pt, the checkpoints used here).
8. **Not tested:** a single seed per arm; E2 uses last.ckpt (epoch 29) while the teachers use ckpt_best; the KD balance (w_ema about 0.6 at the end; KD and surrogate each about 0.3 % of the total loss value).

## Per city (PDMS points, CI)
| city | n | E2+R_T4 − E2_final | E2+R_M4 − E2_final | E2_tau0 − E0 | E0+R_T4 − E2_final |
|---|---|---|---|---|---|
| sg-one-north | 1914 | +1.49 [+0.37, +2.48] | +2.32 [+1.05, +3.49] | −1.54 [−4.05, +0.38] | +2.20 [+0.74, +3.77] |
| us-ma-boston | 3731 | +0.37 [−0.27, +0.96] | +0.19 [−0.29, +0.65] | +0.06 [−1.40, +1.50] | +0.22 [−1.09, +1.40] |
| us-nv-las-vegas-strip | 4064 | +0.50 [+0.25, +0.69] | +0.21 [−0.02, +0.45] | +0.00 [−0.67, +0.67] | +0.48 [−0.07, +1.04] |
| us-pa-pittsburgh-hazelwood | 2437 | +0.65 [+0.13, +1.46] | +0.51 [−0.27, +1.16] | +0.02 [−0.88, +0.86] | +0.93 [+0.13, +1.70] |

sg-one-north carries the largest share of both the refiner deficit and the only (non-significant) draft deficit.

## Options (for the user to decide)
- Is the KD target the bottleneck? Train an E2 variant with a single teacher (R_T4 only), or with a KD target that does not reward staying at zero when the teachers disagree (the midpoint, or the larger-magnitude correction).
- Is it train/test shift? Compare the student and the teachers on a held-out split with the same drafts, and log train-time live rates separately for the sg(tau0) half and the perturbed-human half.
- A frozen teacher on E2's drafts already reaches 86.80 (R_T4) / 86.75 (R_M4), close to E0+R_T4 at 86.91. If the E2 BEV training is kept, the drafts themselves are not the limitation.
