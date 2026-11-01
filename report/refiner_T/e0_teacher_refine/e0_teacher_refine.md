# E0 navtest trajectories + frozen stage-T refiners (theta 0, official batched scorer)

Side experiment of STAGE E — E2 EVALUATION SPEC (descriptive): E0 = PARA-SSR interaction_final (epoch 29), one draft per token = E0's navtest trajectory, corrected by the frozen refiners (ckpt_best), correction always applied (gate ignored), scored with tools/refiner/score_trajectories.py (bitwise == pdm_score).

Tokens: 12146 of 12146 E0 navtest tokens (136 logs); excluded: {'not_in_pack': 0, 'not_scored_in_all_arms': 0}; the 45 frame_gap tokens (irregular FUTURE frames; stage T dropped them because the human bank / labels need the future) are included (ego state at t0 is finite for all). Scoring errors: E0 0, R_T4 0, R_M4 0, R_none4 0, R_T3 0.

## Per arm (means over tokens, x100)

| arm | PDMS | NC | DAC | DDC | TTC | EP | C | fail_any |
|---|---|---|---|---|---|---|---|---|
| E0 | 84.88 | 97.85 | 93.31 | 100.00 | 93.57 | 79.77 | 99.99 | 1523 |
| R_T4 | 86.91 | 98.40 | 94.99 | 100.00 | 94.34 | 81.59 | 99.99 | 1255 |
| R_M4 | 86.81 | 97.98 | 95.35 | 100.00 | 93.86 | 81.49 | 99.99 | 1272 |
| R_none4 | 84.78 | 97.88 | 93.19 | 100.00 | 93.63 | 79.64 | 99.99 | 1530 |
| R_T3 | 86.35 | 98.21 | 94.56 | 100.00 | 94.15 | 81.05 | 99.99 | 1324 |

## Paired contrasts (points, log-cluster bootstrap 95 % CI)

| contrast | PDMS | NC | DAC | DDC | TTC | EP | C | fixed | new_fail | helped / harmed |
|---|---|---|---|---|---|---|---|---|---|---|
| R_T4-E0 | +2.04 [+1.62, +2.49] | +0.55 [+0.32, +0.80] | +1.68 [+1.26, +2.14] | +0.00 [+0.00, +0.00] | +0.77 [+0.51, +1.03] | +1.81 [+1.43, +2.22] | +0.00 [+0.00, +0.00] | 307 | 39 | 4210 / 3241 |
| R_M4-E0 | +1.94 [+1.44, +2.48] | +0.13 [-0.00, +0.28] | +2.03 [+1.52, +2.62] | +0.00 [+0.00, +0.00] | +0.29 [+0.08, +0.49] | +1.71 [+1.28, +2.19] | +0.00 [+0.00, +0.00] | 300 | 49 | 3958 / 3477 |
| R_none4-E0 | -0.10 [-0.22, +0.02] | +0.02 [-0.03, +0.08] | -0.12 [-0.24, -0.02] | +0.00 [+0.00, +0.00] | +0.06 [-0.05, +0.15] | -0.13 [-0.25, -0.02] | +0.00 [+0.00, +0.00] | 28 | 35 | 3617 / 3521 |
| R_T3-E0 | +1.47 [+1.16, +1.82] | +0.36 [+0.22, +0.50] | +1.24 [+0.93, +1.60] | +0.00 [+0.00, +0.00] | +0.58 [+0.31, +0.86] | +1.28 [+1.00, +1.58] | +0.00 [+0.00, +0.00] | 222 | 23 | 4005 / 3352 |
| R_T4-R_none4 | +2.13 [+1.71, +2.59] | +0.52 [+0.31, +0.76] | +1.80 [+1.38, +2.28] | +0.00 [+0.00, +0.00] | +0.71 [+0.46, +0.97] | +1.95 [+1.56, +2.36] | +0.00 [+0.00, +0.00] | 321 | 46 | 4129 / 3351 |
| R_M4-R_none4 | +2.03 [+1.55, +2.58] | +0.11 [-0.03, +0.26] | +2.16 [+1.64, +2.75] | +0.00 [+0.00, +0.00] | +0.23 [+0.02, +0.45] | +1.84 [+1.41, +2.32] | +0.00 [+0.00, +0.00] | 321 | 63 | 3926 / 3543 |
| R_T4-R_M4 | +0.10 [-0.22, +0.42] | +0.42 [+0.22, +0.63] | -0.35 [-0.64, -0.08] | +0.00 [+0.00, +0.00] | +0.48 [+0.23, +0.72] | +0.10 [-0.21, +0.41] | +0.00 [+0.00, +0.00] | 190 | 173 | 3932 / 3650 |

fixed = B fails nc/dac/ddc/ttc and A passes all four; new_fail = B passes nc/dac/ddc/ttc/comfort and A fails one. Per-metric fixed / new counts (A vs B):

| contrast | nc fixed/new | dac fixed/new | ddc fixed/new | ttc fixed/new | comfort fixed/new |
|---|---|---|---|---|---|
| R_T4-E0 | 87 / 15 | 221 / 17 | 0 / 0 | 117 / 24 | 0 / 0 |
| R_M4-E0 | 39 / 15 | 270 / 23 | 0 / 0 | 64 / 29 | 0 / 0 |
| R_none4-E0 | 9 / 8 | 15 / 30 | 0 / 0 | 18 / 11 | 0 / 0 |
| R_T3-E0 | 63 / 12 | 159 / 8 | 0 / 0 | 86 / 16 | 0 / 0 |
| R_T4-R_none4 | 87 / 16 | 239 / 20 | 0 / 0 | 114 / 28 | 0 / 0 |
| R_M4-R_none4 | 42 / 19 | 289 / 27 | 0 / 0 | 68 / 40 | 0 / 0 |
| R_T4-R_M4 | 72 / 24 | 97 / 140 | 0 / 0 | 104 / 46 | 0 / 0 |

## Per city (PDMS x100; delta vs E0 with CI)

| city | n | E0 | R_T4 - E0 | R_M4 - E0 | R_none4 - E0 | R_T3 - E0 | R_T4 - R_none4 |
|---|---|---|---|---|---|---|---|
| sg-one-north | 1914 | 75.90 | +4.15 [+2.95, +5.87] | +4.93 [+3.33, +7.14] | +0.09 [-0.27, +0.48] | +2.74 [+1.78, +4.40] | +4.06 [+2.86, +5.91] |
| us-ma-boston | 3731 | 84.48 | +1.79 [+1.00, +2.60] | +1.67 [+0.95, +2.41] | -0.20 [-0.38, -0.05] | +1.36 [+0.77, +2.02] | +2.00 [+1.26, +2.77] |
| us-nv-las-vegas-strip | 4064 | 90.50 | +1.23 [+0.80, +1.64] | +0.99 [+0.60, +1.39] | +0.08 [-0.05, +0.22] | +0.95 [+0.64, +1.25] | +1.15 [+0.71, +1.57] |
| us-pa-pittsburgh-hazelwood | 2437 | 83.15 | +2.09 [+1.28, +3.33] | +1.57 [+0.66, +2.81] | -0.38 [-0.91, -0.06] | +1.52 [+0.72, +2.34] | +2.47 [+1.53, +3.99] |

## Modification statistics (what is scored; theta 0)

| arm | subset | n | unchanged | brake live | short>0.5 m | short>2 m | mean short m | p90 short m | max short m | lat>0.1 m | lat>0.5 m | mean lat m | max lat m | disp>0.5 m | p_g>=0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R_T4 | all | 12146 | 0.049 | 0.054 | 0.032 | 0.007 | 0.05 | 0.00 | 7.5 | 0.135 | 0.035 | 0.064 | 1.97 | 0.065 | 0.106 |
| R_T4 | E0_fail_any | 1523 | 0.001 | 0.119 | 0.072 | 0.021 | 0.12 | 0.17 | 7.5 | 0.389 | 0.163 | 0.219 | 1.94 | 0.223 | 0.401 |
| R_T4 | E0_pass | 10623 | 0.055 | 0.044 | 0.026 | 0.004 | 0.03 | 0.00 | 4.9 | 0.099 | 0.017 | 0.042 | 1.97 | 0.042 | 0.063 |
| R_M4 | all | 12146 | 0.049 | 0.032 | 0.019 | 0.004 | 0.03 | 0.00 | 5.4 | 0.149 | 0.043 | 0.072 | 2.00 | 0.060 | 0.101 |
| R_M4 | E0_fail_any | 1523 | 0.001 | 0.047 | 0.026 | 0.006 | 0.03 | 0.01 | 4.1 | 0.406 | 0.177 | 0.229 | 2.00 | 0.199 | 0.339 |
| R_M4 | E0_pass | 10623 | 0.055 | 0.030 | 0.018 | 0.004 | 0.03 | 0.00 | 5.4 | 0.112 | 0.024 | 0.049 | 1.91 | 0.041 | 0.067 |
| R_none4 | all | 12146 | 0.049 | 0.014 | 0.005 | 0.002 | 0.01 | 0.00 | 4.9 | 0.047 | 0.003 | 0.023 | 1.99 | 0.007 | 0.044 |
| R_none4 | E0_fail_any | 1523 | 0.003 | 0.045 | 0.018 | 0.006 | 0.03 | 0.00 | 4.9 | 0.090 | 0.007 | 0.037 | 1.99 | 0.021 | 0.118 |
| R_none4 | E0_pass | 10623 | 0.055 | 0.010 | 0.004 | 0.002 | 0.01 | 0.00 | 4.3 | 0.040 | 0.002 | 0.020 | 1.30 | 0.005 | 0.034 |
| R_T3 | all | 12146 | 0.048 | 0.047 | 0.028 | 0.005 | 0.04 | 0.00 | 5.2 | 0.086 | 0.024 | 0.042 | 1.99 | 0.049 | 0.083 |
| R_T3 | E0_fail_any | 1523 | 0.003 | 0.093 | 0.063 | 0.014 | 0.10 | 0.05 | 5.2 | 0.316 | 0.117 | 0.163 | 1.92 | 0.168 | 0.331 |
| R_T3 | E0_pass | 10623 | 0.055 | 0.040 | 0.023 | 0.003 | 0.03 | 0.00 | 4.7 | 0.054 | 0.011 | 0.025 | 1.99 | 0.032 | 0.048 |

| arm | d PDMS pts on E0 fail_any tokens | d PDMS pts on E0 passing tokens | d EP pts on E0 passing |
|---|---|---|---|
| R_T4 | +18.22 | -0.28 | -0.22 |
| R_M4 | +18.04 | -0.37 | -0.32 |
| R_none4 | +1.40 | -0.31 | -0.30 |
| R_T3 | +12.88 | -0.16 | -0.12 |

## Sensitivity: stage-T token convention (frame_gap excluded, n = 12101)

| E0 | R_T4 | R_M4 | R_none4 | R_T3 | R_T4-E0 | R_M4-E0 | R_none4-E0 | R_T3-E0 | R_T4-R_none4 | R_M4-R_none4 | R_T4-R_M4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 84.83 | 86.87 | 86.77 | 84.73 | 86.31 | +2.04 [+1.63, +2.49] | +1.94 [+1.45, +2.49] | -0.10 [-0.22, +0.02] | +1.48 [+1.16, +1.82] | +2.14 [+1.71, +2.60] | +2.04 [+1.55, +2.59] | +0.10 [-0.22, +0.42] |

## Check: E0 batched scores vs the official E0 csv

n 12146; PDMS csv 84.8683 vs batched 84.8751 (csv all rows 84.8683, 12146 rows, valid=False 0); exact mismatches {'nc': 1, 'dac': 1, 'ddc': 0, 'ep': 7142, 'ttc': 1, 'comfort': 0, 'pdms': 7144}; > 1e-6 {'nc': 1, 'dac': 1, 'ddc': 0, 'ep': 6927, 'ttc': 1, 'comfort': 0, 'pdms': 6587}; max abs {'nc': 1.0, 'dac': 1.0, 'ddc': 0.0, 'ep': 1.0, 'ttc': 1.0, 'comfort': 0.0, 'pdms': 1.0}.

## Notes

- Inference path: tools/refiner/refine_external_drafts.py (stage-T net + decoder + teacher cache selection of eval_refiner.predict; the 13-draft bank replaced by E0's single trajectory; ego state v0/a0/eds/cmd from the packed navtest split; ckpt_best; autocast fp16 on GPU 0 as in stage T). The refiner has no cross-draft interaction: tools/refiner/tests/test_refine_external_drafts.py (CPU fp32, arms none / T) shows the single-draft path equals the bank path for the identity (human) slot and a perturbed slot (atol 1e-5). Real-data check (run-3 R_T, 640 navtest tokens, bank slot 0 fed alone on GPU vs the stored eval_navtest bank prediction): max |d tau1| 3.0 mm, max |d p_g| 1.1e-3 (fp16 kernels depend on the batch composition).
- E0 check: the pkl trajectories scored with the batched scorer give PDMS 84.875 vs 84.868 in the official csv; the same metric caches are used, and the batched scorer is bitwise equal to pdm_score (scorer_equivalence.json), so the differences come from the pkl poses themselves (EP differs by <= 8.8e-4, median 5e-6, on 7,142 tokens as in scorer_equivalence.json 'model_vs_official_csv'; two tokens flip a multiplier: 14d53eb06a7d582a csv DAC 0 -> 1, 72be63ed04f15f97 csv NC 1 -> 0). All arms start from the same pkl, so the contrasts are unaffected.
- theta 0 = correction always applied; p_g is recorded only (p_g >= 0.5 on ~10 % of E0 drafts for R_T4 / R_M4).
- Teachers: BEVFusion cache_val_50x100 (navtest is out-of-sample for BEVFusion, DECISION_NAVTEST.md); ReSMap navtest cache (its training data were not re-checked here). One seed, one fold-0 refiner per arm; descriptive.
