# Stage T navtest evaluation: adversarial verification (2026-09-29)

**Verdict: CONFIRMED. The navtest outcome is PASS.** Every endpoint and count that was checked reproduces from the raw rows without `stageT_decision.py`. The navtest inputs match the official sources and dev's build. The models are the frozen run-3 `ckpt_best.pt`, and no navtest number fed back into any choice. The teacher-shuffle numbers also reproduce.

Scripts and outputs are in `report/refiner_T/navtest/verify/`: `recompute_navtest.py/.json`, `inputs_check.py/.json`, `bank_check.py/.json`, `rescore_check.py/.json` (+ `_rows.csv`), `repred_compare.py/.json`, `shuffle_check.py/.json`, `extra_check.py/.json`. Nothing was written inside `runs/` or to train/dev artifacts. Re-predictions went to the session scratchpad.

## 1. Independent recompute of the navtest endpoints

The recompute reads only `eval_navtest/report_rows.parquet` and `tokens.parquet` for both arms, plus the raw score parquets.
- **Pairing:** (token, k) rows are identical in order across arms. All `*_orig`, `valid`, `family` and `log` columns are identical across arms.
- **Scores vs raw files:** `*_orig` equals `scores/navtest.parquet`, and `*_tau1` equals each arm's `scores_tau1.parquet` (max abs diff 0.0). There are 0 missing rows, 0 scorer errors and 0 `rec_ok = False`.
- **PDMS formula:** `NC·DAC·DDC·(5EP+5TTC+2C)/12` reproduces `pdms` in all four columns (max abs diff 0.0).
- **Valid mask:** `valid` in both arms and all metrics finite gives 148,517 of 157,313 drafts. That is 12,101 tokens from 136 logs. The minimum p_g is 0.0045 (T) and 0.0078 (none), so θ = 0 means the refined draft is used for every valid draft.
- **Bootstrap:** my own log-cluster bootstrap, 10,000 resamples of log indices with a different RNG. It uses the ratio of summed per-log differences to summed per-log counts.

| Endpoint (R_T − R_none) | claimed | recomputed (mean [95% CI]) |
|---|---|---|
| P1 PDMS, points | +1.355 [1.047, 1.677] | +1.355 [1.054, 1.677] |
| P2 NC+TTC reduction, pp | +0.641 [0.395, 0.884] | +0.641 [0.401, 0.890] |
| P2-ni DAC excess, pp | −0.922 [−1.230, −0.643] | −0.922 [−1.232, −0.646] |
| P2-ni DDC excess, pp | +0.003 [−0.001, 0.011] | +0.003 [−0.001, 0.011] |
| P3 new-failure excess, pp | −0.273 [−0.384, −0.167] | −0.273 [−0.386, −0.164] |
| Sanity: R_T vs no correction, points | +7.254 [6.812, 7.723] | +7.254 [6.809, 7.723] |
| Sanity: R_none vs no correction, points | +5.899 [5.543, 6.275] | +5.899 [5.539, 6.275] |

- **Means:** equal to about 1e-15. CI bounds differ by at most 0.007 because the RNG differs. By the pre-stated rule this is PASS: P1 lo > 0, P2 lo > 0, P3 and DAC/DDC within +0.5 pp.
- **Also reproduced exactly:**
  - PDMS: 0.8096 / 0.8822 / 0.8686 (no correction / R_T / R_none).
  - Failure rates for NC, TTC, DAC, DDC, comfort and NC+TTC, in all three conditions.
  - EP loss on drafts that pass NC, DAC and DDC both before and after: 0.0923 (n 129,263) for R_T and 0.1232 (n 128,731) for R_none.
  - NC+TTC counts: 17,002 failed originally. Fixed: 7,688 (R_T) and 6,696 (R_none). New: 456 and 416.
  - New failures of any kind: 334 and 740.
  - DAC counts: 13,031 failed originally. Fixed: 6,850 and 6,045. New: 131 and 695.
  - Per-family P1 and P2 means for all 9 families.
  - City P1 and P2 means, token counts and log counts for all 4 cities and for "other cities". CIs agree within about 0.05.
- **Robustness (not claimed):** I also included the 8,796 invalid drafts, which are 8,720 L-const and 76 small. The result is still P1 +1.27 and P2 +0.60.

## 2. Were the navtest inputs built like dev?

I spot-checked 20 random evaluated tokens with my own transforms, without the navsim scene loader or the builder code.

- **Human trajectory vs the raw test log:** max xy error 1.3e-6 m, heading error 5e-8. Frame spacing is at most 0.5005 s. The t0 pose from the raw log equals the metric-cache `ego_state` exactly, v0 error is 1e-6, and the command matches for 20 of 20 tokens.
  - One mismatch on my side, resolved: my first check took yaw from the 4×4 `ego2global` matrix and gave up to 2.2 cm error. NAVSIM and the metric cache use pyquaternion's `yaw_pitch_roll`, which is atan2(2(wz − xy), …). With that convention the error is about 1e-6.
  - The npz follows the official convention.
- **Objects vs the official metric-cache occupancy polygons**, keyframes 0–10:
  - 12,426 boxes compared, max corner error 4.1e-6 m.
  - 0 presence mismatches, once singleton tracks are placed at every time as the metric cache does. Tracks the cache has and the npz lacks are all outside R (for example, one sits at 80.3 m when R is 80 m).
- **Identity draft:** equal to the human trajectory byte for byte (20 of 20).
- **Draft bank, all files:**
  - Every one of the 12,101 navtest files and 7,930 dev files has `cfg_hash` 3b7db8d7e630 and version `draft_bank_v1`.
  - `_config.json` is identical except for the timestamp.
  - `valid` in report_rows equals the draft npz for 12,101 of 12,101 tokens.
  - Draft validity rate is 94.41% on navtest vs 94.47% on dev, and per-family validity rates are equal (L-const 85.6 vs 85.7%).
  - Family counts by slot follow the same rules. Slot 7 is ignore-brake for 26.0% of navtest tokens and 26.4% of dev. Slot 8 is creep for 3.5% and 3.1%. Slot 12 falls back to hdrift for 3 navtest tokens and 2 dev tokens.
  - Per-family failure rates reproduce the claimed dev and navtest table.
- **Labels, re-scored with the official single-trajectory `navsim.evaluate.pdm_score.pdm_score`:**
  - The plain navsim simulator and scorer came from `default_scoring_parameters.yaml`, with the metric cache read from the official navtest cache directory.
  - 80 drafts: 50 original bank drafts (27 with PDMS 0), 15 R_T-refined and 15 R_none-refined failing drafts.
  - All seven metrics match with max abs diff 0.0.
- **Train/dev artifacts:** the newest files in packed/, drafts/, scores/, human/, objects/ and splits/ for train and dev, and in sdf/navtrain, all date from 2026-09-28. `human_validation.json`, `draft_bank_dev.json` and `run3/*.json` are also unchanged since 09-28 / 09-29 02:30.

## 3. No navtest feedback, and the models are frozen

- **Checkpoint timing and identity:**
  - `ckpt_best.pt`, `norm.npz` and `config.json` were last written at 01:59 / 00:21 (T) and 01:43 / 00:20 (none) on 09-29. Navtest prediction ran at 14:19.
  - sha256 of `ckpt_best.pt`: T `ddf3c4f8…e2e8f`, none `d72d672b…9db7b`. No hash had been recorded earlier, so I checked identity by function instead.
  - The checkpoints hold epoch 31 (T) and 29 (none), the same best epochs as DECISION_RUN3.
  - Re-predicting 96 navtest tokens and 64 dev tokens with these checkpoints reproduces `eval_navtest/pred.npz` and `eval_dev/pred.npz` bit for bit (tau1, p_g, z_lon, w_lat: max diff 0.0). So the models scored on navtest are the ones scored on dev.
- **θ and stale evaluations:**
  - θ comes from `run3/selection.json` (written 02:20; θ 0 for both arms).
  - The navtest `report.json` holds only θ = 0 and no `theta_at_budget` or sweep.
  - `predict_meta.teacher_shuffle` is null.
  - There is no other `eval_navtest*` directory under `runs/`, so no earlier navtest refiner evaluation exists.
- **Code changes:**
  - `eval_refiner.py` (13:53, before the navtest prediction) only adds `--shuffle-teacher-seed`, which is off by default. Re-prediction with the current code matches the pred files exactly.
  - `stageT_decision.py` (14:20, after prediction) only adds `--eval-name`. Its default run reproduces `run3/decision_dev.json` exactly; the only extra key is `eval_name`.
  - The model, decoder, surrogate and data modules were all last changed 09-29 00:15 or earlier.
- **Run directories:** the only new entries are `eval_navtest/` and `eval_dev_shuffle/`.

## 4. Teacher-shuffle control (dev)

Recomputed from `eval_dev_shuffle/report_rows.parquet` against run-3 `eval_dev` for T and none: 97,386 valid drafts, 7,930 tokens, orig columns identical across the three runs.
- **PDMS:** no correction 0.8215, R_T 0.8956, shuffled 0.7877, R_none 0.8812.
- **Paired differences (my bootstrap):**
  - Original minus shuffled: PDMS +10.78 [10.17, 11.41] points; NC+TTC −5.70 [−6.29, −5.14] pp.
  - Shuffled minus none: PDMS −9.34 [−9.95, −8.74] points; NC+TTC +4.91 [4.33, 5.52] pp.
  - Shuffled minus no correction: −3.37 points. DAC failure rises from 7.97% to 9.91%.
  - All of these match the claim.
- **Shuffle map:** a permutation of the 7,930 evaluated tokens with 0 fixed points and 0.61% same-log pairs.

## Discrepancies

None of substance. Minor:
- **Bootstrap:** CI bounds differ from the decision script's by up to 0.007 because the RNG differs. The means are identical and no outcome changes.
- **Build timing:** the prep summary says the build ran 13:54–14:30. File times show split and human at 13:54, labels at 14:11, pack at 14:17 and prediction at 14:19. The 550 older navtest SDFs date from 09-28 06:47, as the summary says. Only the stated end time is loose.

## Caveats (not verified or limits)

- **Not independently re-checked:** SDF and centerline values for navtest, the liveness values (0.119 / 0.084, reported only), the NC/TTC/DAC columns of the city table, the dev-to-navtest city-reweighting prediction, and the shuffle's "10 no-overlap logs" subset. My checks were samples: 20 tokens for human/objects and 80 drafts for labels.
- **Teacher out-of-sample:** navtest is out of sample for the teacher only if BEVFusion was trained on navtrain alone. I did not verify this; it rests on the teacher's provenance.
- **Statistics:** one seed. The CIs cover log variability only, not training variability.
- **Earlier choices:** the PASS is for run 3, whose TTC loss was added after the run-2 dev breakdown. navtest was used exactly once, but it does not undo decisions made on dev.
- **θ = 0 edits every draft, including human ones.** R_T adds 38 new failures on the 12,101 identity (human) drafts, and PDMS on those drafts drops 0.24 points. R_none drops 0.36 points. This is the same pattern as dev.
- **Invalid drafts:** the 8,796 excluded drafts (mostly L-const) are excluded identically in both arms. Including them does not change the direction.
