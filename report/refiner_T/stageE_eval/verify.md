# Stage E verification: E2 eval (Task A) and E0 + teacher refiners (Task B)

Date: 2026-09-30. I recomputed everything below from the per-token files: the official CSVs and `scores.parquet` / `pred.npz` / `refined.npz`. I did not use any script summaries. I wrote my own log-cluster bootstrap (136 logs, 10,000 draws, seed 12345), so CI endpoints can differ from the reported ones by about 0.01–0.03.
Scripts and raw output are in the session scratchpad (`verify.py`, `verify2.py`, `verify.out`).

**Verdict: PASS.** Every headline number reproduces. I found three minor documentation discrepancies (listed at the end). None of them changes a conclusion.

## Task A: E2 navtest eval

### Inputs
- **E2 checkpoint:** `last.ckpt` is byte-identical (`cmp`) to `epoch=29-step=19950.ckpt`. Its mtime (19:25:01) is before the eval start (19:37). Both E2 overrides point to that `last.ckpt`.
- **E0 checkpoint:** `para_ssr_interaction_final/version_2/last.ckpt`. The pkl meta says `checkpoint_epoch: 29`.
- **Configs:** the E2 and E2_tau0 Hydra configs differ only in `experiment_name` and `ref_eval_traj` (final vs tau0). E2 and E0 differ only in the name, the checkpoint path and the added refiner/KD block. Metric cache, scene_filter=navtest, split, log paths and scorer are identical.
- **Pairing:** each of the three CSVs has 12,146 tokens, all valid, with no duplicates and no NaN. The three token sets are identical, and they equal the pkl / tokens.parquet set (136 logs).

### Means (×100), recomputed
In each CSV, the per-token mean equals the file's `average` row exactly.

| arm | PDMS | NC | DAC | DDC | TTC | EP | C |
|---|---|---|---|---|---|---|---|
| E0 | 84.87 | 97.86 | 93.31 | 100.00 | 93.56 | 79.77 | 99.99 |
| E2 | 86.15 | 98.17 | 94.29 | 100.00 | 94.04 | 80.99 | 99.99 |
| E2_tau0 | 84.66 | 97.93 | 92.92 | 100.00 | 93.66 | 79.67 | 99.99 |

All values match the report. The E0 baseline is 84.868, which matches the reported 84.87.

### Contrasts, recomputed (points, 95% CI)

| contrast | PDMS | NC | DAC | TTC | EP |
|---|---|---|---|---|---|
| E2 − E0 | +1.28 [+0.74, +1.85] | +0.31 [+0.04, +0.61] | +0.98 [+0.49, +1.49] | +0.48 [+0.01, +0.96] | +1.22 [+0.72, +1.73] |
| E2_tau0 − E0 | −0.21 [−0.85, +0.42] | +0.07 [−0.19, +0.34] | −0.39 [−0.97, +0.19] | +0.10 [−0.37, +0.56] | −0.10 [−0.69, +0.48] |
| E2 − E2_tau0 | +1.50 [+1.06, +2.03] | +0.23 [+0.11, +0.37] | +1.37 [+0.96, +1.85] | +0.38 [+0.22, +0.56] | +1.32 [+0.94, +1.79] |

- DDC and C are 0 in every contrast.
- All point estimates match. CIs differ only through bootstrap noise, and none of the "excludes 0" statements changes. The E2 − E0 TTC lower bound is thin (+0.01 here vs +0.02 reported).
- **Fixed / broken counts:** E2 vs E0 (4326/3853/3967; 735/554; 504/361) and E2_tau0 vs E0 match exactly. E2 vs E2_tau0 matches on 226/17 and 194/9. The plain up/down/tie counts match only with a 1e-9 tie tolerance (3659/3603/4884). With an exact comparison they are 3663/3611/4872.

## Task B: E0 + frozen teacher refiners (theta 0)

### Inputs
- **Drafts:** the pkl sha256 is `c520714b…` in every arm's meta, and the pkl meta names the E0 checkpoint (epoch 29). E0 `refined.npz` is bitwise equal to the pkl trajectories for all 12,146 tokens. In every refined arm, `pred.npz` has the same token order as E0 and `tau0` is bitwise equal to the E0 drafts.
- **Checkpoints:**
  - R_T4 = `runs/stageT4_T_fold0_seed0/ckpt_best.pt`
  - R_M4 = `stageT4_M…/ckpt_best.pt`
  - R_none4 = `stageT4_none…/ckpt_best.pt`
  - R_T3 = `stageT3_T…/ckpt_best.pt`

  The run-4 T and M `ckpt_best.pt` files have the same sha256 as the copies E2 used as KD teachers (`stageE/teachers/…`): `21f63028…` and `7ee602dd…`.
- **Teacher caches:**
  - Arms T (run-3 and run-4) use BEVFusion `cache_val_50x100`. Its manifest says 12,146 of 12,146 samples written, and its checkpoint sha head `cddf943f` matches the runs' `teacher_sha_head`. All 12,146 E0 tokens have a sample file (0 missing).
  - Arm M uses ReSMap `resmap/navtest`. The script's sha-head check and coverage check (SystemExit on any missing token) passed.
  - Arm none uses no BEV.
  - This is the same navtest cache selection that stage-T3 `eval_navtest` used.
- **Theta 0:**
  - Every meta has `theta: 0.0`.
  - The code writes `refined.npz = tau1` unconditionally (`_write_common`). I verified `refined.npz == pred.tau1` bitwise in all four arms.
  - Among tokens with p_g < 0.5, 94.6% of trajectories are modified (R_T4), so the gate was not applied.
  - p_g ≥ 0.5 fractions: R_T4 10.6%, R_M4 10.1%, R_none4 4.4%, R_T3 8.3%.
- **Scores:** in every arm, 12,146 rows with unique tokens, `error` empty, `rec_ok` all true, k = 0, no NaN. The token sets are identical to each other and to the CSVs.

### Means (×100), recomputed

| arm | PDMS | NC | DAC | DDC | TTC | EP | C | fail_any |
|---|---|---|---|---|---|---|---|---|
| E0 | 84.88 | 97.85 | 93.31 | 100.00 | 93.57 | 79.77 | 99.99 | 1523 |
| R_T4 | 86.91 | 98.40 | 94.99 | 100.00 | 94.34 | 81.59 | 99.99 | 1255 |
| R_M4 | 86.81 | 97.98 | 95.35 | 100.00 | 93.86 | 81.49 | 99.99 | 1272 |
| R_none4 | 84.78 | 97.88 | 93.19 | 100.00 | 93.63 | 79.64 | 99.99 | 1530 |
| R_T3 | 86.35 | 98.21 | 94.56 | 100.00 | 94.15 | 81.05 | 99.99 | 1324 |

All values match.

### Contrasts, recomputed (PDMS points, 95% CI; fixed / new fail)

| contrast | PDMS | fixed / new |
|---|---|---|
| R_T4 − E0 | +2.04 [+1.62, +2.48] | 307 / 39 |
| R_M4 − E0 | +1.94 [+1.45, +2.45] | 300 / 49 |
| R_none4 − E0 | −0.10 [−0.22, +0.02] | 28 / 35 |
| R_T3 − E0 | +1.47 [+1.16, +1.81] | 222 / 23 |
| R_T4 − R_none4 | +2.13 [+1.71, +2.59] | 321 / 46 |
| R_M4 − R_none4 | +2.03 [+1.55, +2.56] | 321 / 63 |
| R_T4 − R_M4 | +0.10 [−0.21, +0.41] | 190 / 173 |

- The NC, DAC, TTC and EP contrasts also match to within ±0.01 on the point estimates.
- DDC and C are 0 everywhere.
- One sign detail: R_M4 − E0 NC has a lower bound of +0.00 here vs −0.00 reported. That is bootstrap noise at the boundary.

### E0: scores from the pkl vs the official CSV
- PDMS is 84.875 (pkl) vs 84.868 (CSV).
- Excluding the tokens listed below, the maximum |ΔEP| is 8.77e-4 (6,925 tokens differ by more than 1e-6), and the maximum |ΔPDMS| is 3.7e-4.
- **Three** tokens flip a pass/fail score, not two:
  - `14d53eb06a7d582a`: DAC passes in the pkl, fails in the CSV.
  - `72be63ed04f15f97`: the pkl has NC = 0 and TTC = 0. The CSV has NC = 1, TTC = 0 and PDMS 0.583.
  - **`8f60912c624e5f5f`: TTC passes in the pkl, fails in the CSV** (CSV PDMS 0.583, pkl 1.0). The report does not mention this token.

  All arms of Task B start from the same pkl drafts, so the paired contrasts are unaffected.

## Discrepancies
1. **Task B, E0 vs CSV:** the report says two tokens flip a pass/fail score. There are three; `8f60912c624e5f5f` (TTC) is missing. This is cosmetic: the net effect is +0.007 PDMS and it does not touch any contrast.
2. **Task A, E2 vs E2_tau0 up/down/tie:** the reported 3659/3603/4884 counts ties with a tolerance of about 1e-9. With an exact comparison the counts are 3663/3611/4872. The reported thresholded counts (226/17, 194/9) are exact.
3. **Task B, "mean change about 0.1 m (R_T4)":** this is the mean of each trajectory's maximum displacement over the horizon (0.106 m), or equally the displacement at 4 s (0.106 m). The mean over all 8 waypoints is 0.028 m. The wording is ambiguous, but it is not wrong.

## Not re-checked here
- Per-city splits.
- Brake / shortening / lateral rates.
- The single-draft-vs-bank equivalence test.
- Re-running the official scorer.

I checked that the arms' files are consistent with each other; I did not re-derive the simulator itself.
