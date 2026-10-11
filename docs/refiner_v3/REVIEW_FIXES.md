# refiner-v3 review fixes: finding → fix → verification

Written 2026-10-11 18:00 KST by the review fixer. Branch `refiner-v3`, worktree `/workspace/yongjae/SSR-v3`, base ck2 @ 8ae8b1d.
Nothing is committed. No training run was launched.

Tags: [실측] = measured (the log, file or test is named), [추정] = estimate, [미확인] = not checked.
`D3 = /home/external-user/ssd/yongjae_refiner/ck/v3`.

The main session's decisions D1–D8 were applied as given. SPEC.md §12 summarises the changes; the launch procedure is in `STEP1_RUN.md`.

Reviews: **R1** = causality review (5 findings), **R2** = correctness review (9), **R3** = protocol review (11).

## 0. Summary

| result | findings |
|---|---|
| fixed | R1.1, R1.4, R1.5, R2.1, R2.2, R2.3, R2.4, R2.7, R2.9, R3.1, R3.3, R3.4, R3.5, R3.6, R3.7, R3.8, R3.9 |
| fixed in part | R1.2 (navtest GT and out-of-sample forecaster eval: done; vnoise recalibration: not done), R1.3 (only reporting), R3.2 (rule and flag: done; no extra seed run), R3.10 (CA scales tied to a source: done; class contract and v2det history: not built) |
| not fixed | R2.5, R2.6, R2.8, R3.11 (reasons in §3) |

The verification runs, all [실측]:
- **v3 tests:** 133 passed, 0 failed, 0 skipped, 4 min 53 s (`D3/logs/fix_v3_tests.log`). The earlier count was 117; 16 tests are new.
- **Existing CK2 tests (15 files):** 238 passed, 1 failed. The failure is the known one that predates v3 and fails the same way in the reference tree (`test_ck_e2e_launch.py::test_ck_yaml_matches_contract_and_config`, `kd_ramp_epochs` 0.0 vs 3.0).
- **OFF bit-identity against 8ae8b1d:** passes (`test_v3_integration_off.py`). A copy of the record is kept in `D3/fix/off_equiv/compare.json`.
- **GPU work:** a smoke on GPU 0, the Step-1-setup smoke on GPUs 0,1, and the arm-(c) dump. Each job took under 3 min.

## 1. Data changes (D1, D2; R1.1, R1.4, R1.5, R2.2, R2.7, R3.1)

### 1.1 Object packs at K_OBJ = 96 (D1)

The packs live in `D3/objpack_k96/bevfusion/<split>/`. The K=64 packs in `D3/objpack/` were left untouched; nothing reads them now.

The builds and verifies below [실측] come from `build.json` / `verify.json`, with logs in `D3/logs/objpack_k96_*.log`:

| | navtrain_train | navtrain_val | navtest |
|---|---|---|---|
| rows | 85,109 | 18,179 | 12,146 |
| build wall time | 54.9 s | 14.1 s | 8.7 s |
| truncated frames at 96 | 1,332 (**0.575 %**, rule ≤ 1 %) | 0 | 1 (0.003 %) |
| agent t0 slots matched to GT (2 m) | 99.310 % | 99.337 % | 98.295 % |
| matched singletons → FC_SINGLE | 4,979 | 502 | 337 |
| supervised (object, step) | 94.27 % | 94.29 % | 93.86 % |
| verify (1,000 rows) | ok | ok | ok |
| build.json sha16 | d004c23dc99f6348 | cff3369f02b483a5 | e3efeef97c621ef4 |

Notes:
- navtest now has targets, because GT objects were built for it (§1.3). They are used only by the forecaster evaluation; navtest stays descriptive.
- The val match rate moved from 99.384 % to 99.337 %. The cause is the singleton rule below, not the larger K.

Code changes:
- `config.K_OBJ = 96` and `PACK_ROOT = D3/objpack_k96`. The folder name carries K, which is the R2.2 fix.
- `arms/*.json` point at the new root.
- `fit_ca_scales.py` refuses a pack whose K differs from the code.
- `provenance` records `k_obj`, and `run_step1.sh` checks the `k_obj` in `ca_scales.json`.
- The model was already K-agnostic: it trims to the batch maximum.

### 1.2 Singleton rule (R1.4)

The changes in `objects.match_targets` (now `OBJECTS_VERSION v3_objects_2`):
1. A GT track first annotated after t0 (`meta first_k > 0`) is never matched. Before, `gt_future.query` held such a singleton at its later pose, so it looked present at t = 0.
2. A matched one-keyframe track gets the new state `FC_SINGLE = 4`.
   - The oracle reads it as OBS, which is the official hold rule (`forecast.oracle_forecast`).
   - The forecast loss, the metrics and the CA-scale fit use `FC_OBS` only, so they never see it.

Both counts are written to `build.json`: `n_single_matched` and `n_gt_future_first_excluded`.

Tests: `test_v3_review_fixes::test_match_targets_singleton_rule` and `::test_oracle_reads_single_as_obs_and_loss_ignores_it`. `verify_obj_pack.py` also checks that FC_SINGLE is set exactly on singleton tracks.

### 1.3 GT-object provenance (R1.5)

`tools/ck/v3/setup_gt_objects.sh` (idempotent) set up `D3/gt_objects/`:

| split | what was done |
|---|---|
| navtrain_val | **copied** from the report folder (18,179 npz, 488 MB) |
| navtrain_train | **symlinked** to `/home/external-user/ssd/yongjae_refiner/objects/navtrain` |
| navtest | **built** from `test_navsim_logs/test` by `tools/refiner/build_future_objects.py`, 33 s, 0 errors |

The navtest build was validated against the navtest metric cache on 400 tokens [실측 `D3/gt_objects/_manifests/navtest_mc_validation.json`]: max corner error 7.6e-6 m, 0 presence mismatches out of 1,396,329 pairs, 0 unexplained exclusions.

Further changes:
- Each split has a per-file sha256 manifest in `D3/gt_objects/_manifests/<split>.json`. The manifest sha16 values are: val c8c29ea11f370b79, train 57f68626927c4c49, navtest 95dca9055b61120b.
- `build_obj_pack.gt_dir_for` now reads only `D3/gt_objects/<split>` and fails when it is missing. The silent fallback is gone.
- `D3/phase0/phase0_copies.json` records the Phase-0 copies after the fact: per file, the sha256 of the source, the pristine copy and the destination. The two dataset files edited by v3 are flagged.

### 1.4 CA scales fitted on the augmented stream (D2; R3.1, R2.7)

The fit:
- `fit_ca_scales.py` fits `b` on every train row × epochs 0–9 through `augment_objects(token, epoch, seed 0, cfg)`. These are exactly the realisations that arm (b) sees in training.
- A slot counts only if it survives the augmentation at t0.
- The clean fit is kept as `b_clean`.
- The output `D3/ca_scales.json` (version `v3_ca_scales_2`, sha16 e1ae75dc742cecd3, 42.5 s) records `source`, `k_obj`, `aug`, and `pack_sha256`, which equals the sha256 of the K96 train pack's `build.json`.

Results [실측]:

| | 0.5 s | 2.5 s | 5 s |
|---|---|---|---|
| vehicle `b` (aug) / `b_clean` (m) | 0.191 / 0.112 | 1.472 / 0.812 | 4.298 / 2.436 (×1.76) |
| VRU `b` / `b_clean` (m) | 0.165 / 0.080 | 1.188 / 0.448 | 3.383 / 1.235 (×2.74) |

These ratios match the R3.1 measurement (×1.67 / ×2.75).

Why this choice [추정, documented]:
- Arm (a)'s forecaster is pre-trained on the same augmentation (§2.1). So both forecasters are fitted on the augmented train distribution, both are frozen during critic training, and both see the same clean inputs at val.
- `v_noise = 0` was not chosen. The noise models out-of-sample BEVFusion velocity error (R1.2), and D2 asked to keep the same inputs for (a) and (b).

Further changes:
- `branch.load_ca_b(path, source)` refuses scales fitted for another source. This is part of R3.10.
- `run_step1.sh` refuses a `ca_scales.json` that does not match the train pack, the source, K = 96 or the augmented rule.

Tests: `test_fit_ca_scales_augmented` checks that b > b_clean on every k, the records, the no-overwrite rule and the real file's pack sha. `test_load_ca_b_source` checks the source rule.

## 2. Code changes (D3–D6)

### 2.1 Forecaster pre-training and the frozen arm (a) (D3; R3.5)

The pre-training stage, `tools/ck/v3/pretrain_forecaster.py` (stage `pre`, arm file `arms/pre_forecaster.json`):

| aspect | what it does |
|---|---|
| model | `LearnedForecaster(6 modes, seed 0)`, the same module and init as arm (a)'s branch (tested) |
| data | navtrain_train pack rows |
| augmentation | the same as the critic (0.1 / 0.05 / 0.3) on its own noise seed (seed + 7919) |
| loss | forecast loss only (WTA Laplace NLL + mode CE, FC_OBS steps) |
| optimiser | AdamW 5e-4, warm-up 500 + cosine, clip 1.0, fp32 |
| length | 8 epochs, batch 64 |
| weights kept | the last epoch's, as `forecaster.pt` (no selection) |

Evaluation after every epoch uses clean inputs, with the learned forecaster and the CA reference side by side:
- navtrain_val (BEVFusion in-sample);
- navtest (out-of-sample; R1.2);
- a 4,096-row in-sample train subset.

From these it writes `overfit_index` = (ADE_learned / ADE_CA) on val divided by the same ratio on the train subset, and flags values above 1.10.

Changes for arm (a):
- `TemporalCfg` gains:
  - `fc_init`: the path of the pre-trained checkpoint;
  - `fc_train`: `frozen` (default) or `joint`.
- Rules:
  - `fc_init` and `fc_train` apply to the learned forecaster only.
  - `fc_backprop` needs `joint`.
  - A frozen forecaster without `fc_init` is refused at training time (`check_trainable`).
  - Old learned configs that lack the key map to `joint`, which is what they did.
- `train_ck2`:
  - New flags `--t-fc-init` and `--t-fc-train`.
  - `load_pretrained_forecaster` does a strict load, then sets requires_grad False. The trainable count then equals arm (b)'s: 3,629,131 [실측 smoke].
  - With a frozen forecaster, the loss adds no forecast term; the forecast stats are logged under no_grad, and there is a single optimiser group.
  - `dummy_backward` skips frozen outputs.
- `TemporalBranch.scene` runs a frozen forecaster under `no_grad`.
- `joint` remains as the optional fine-tuning path. It was smoked on CPU (`smoke_cpu_aj`); the trainable count is then 4,101,216.
- The pre-training run itself is a Step-1 stage. The main session launches it (`STEP1_RUN.md`); here it was only smoked:
  - GPU 0, 60 steps at batch 64: 0.04 s/step, 3.13 GB [실측];
  - CPU, 2–3 steps.

Tests:
- `test_v3_train::test_frozen_pretrained_arm_three_cpu_steps` checks:
  - the weights are loaded and unchanged after training;
  - loss = loss_ck;
  - one optimiser group;
  - the trainable count equals (b)'s;
  - the guards: wrong n_modes, missing file, frozen without fc_init, fc_init on CA;
  - resume with another forecaster is refused.
- `test_v3_review_fixes::test_temporal_cfg_fc_rules`, `::test_frozen_branch_has_no_forecaster_gradient`, `::test_pretrain_stage_cpu_and_strict_load`.

### 2.2 Provenance (D5; R3.6)

`tools/ck/v3/provenance.py` records the following:

| field | content |
|---|---|
| `code.train.sha256` | sha256 over every training-code file: `ck/**`, `refiner/**`, `tools/ck/*.py`, `tools/ck/data/*.py`, the v3 trainer files and the arm jsons |
| `code.worktree` | HEAD, dirty flag, sha256 of `git diff HEAD -- navsim tools` and of the untracked files |
| pack shas | sha256 of each pack's `build.json`, which itself lists the sha256 of every array |
| `ca_scales` sha | sha256 of the CA-scales file |
| `fc_init` | sha256 of the checkpoint, plus the train-pack sha and code sha of its pre-training run |

Where it is written and checked:
- **train_ck2:** `config.json["provenance"]`, for every run, OFF included (code only).
  - A fresh temporal run is refused if a pack is missing or incomplete, or if `fc_init` is missing or was pre-trained on another train pack (`provenance.check_self`).
  - A resume is refused when the code, the data or the forecaster changed.
- **infer_val_v3:** `infer_meta.json["provenance"] = {train, infer}`.
- **decide_step1:** refuses arms (a), (b) and (d) when they differ in:
  - the training-code sha, K, the train or val pack shas, or the CA-scales sha;
  - the pre-training of (a), when it used another train pack;
  - the val pack that an arm's inference read, when it is not the one it was trained with.
  A worktree difference is a warning only.
- **run_step1.sh:** writes `D3/train/<run>.code.tgz` (navsim, tools, docs/refiner_v3) and `<run>.launch.json` (tarball sha256 + provenance).

Tests: `test_provenance_compare`, `test_v3_decide::test_provenance_check_between_arms`, `::test_check_arm_meta_rules`, plus the resume refusal in `test_frozen_pretrained_arm_three_cpu_steps`.

### 2.3 Inference reproduce check (D6; R2.1, R3.8)

`train_ck2` saves the logits of every temporal r34 eval, with packed rows, to `<run>/eval_r34_<split>_last.npz`. These are the same weights as `ckpt_last`.

`infer_val_v3.reproduce_check`:
- For temporal runs it compares per token and per logit: the 99.9 % quantile of |Δlogit| must be ≤ 2e-4, at most 0.1 % of picks may flip, and the eval step must be the same.
- For temporal-off runs it compares pdms_ck_noim within 1e-4 and reports `exact` at 1e-6.
- The result goes to `infer_meta.json['check']`. A failure no longer exits.
- `decide_step1.check_arm_meta` requires `check.ok`.

Rows without objects (`obj_ok` False) are now scored too and recorded in `obj_ok.npy` (R2.4).

Verification [실측]:
- GPU infer on the two 2-GPU smoke runs (64 rows; `D3/smoke/infer/gpu01_{a,b}.log`): max |Δlogit| 0, 0 flips, ok. The training eval was sharded over 2 ranks; the inference ran as one stream.
- The arm-(c) full dump: |Δ| = 0.
- `test_infer_logit_check_tolerance` covers the pass and fail cases, a step mismatch and a missing file.

### 2.4 Decision rule (D4; R3.2, R3.3, R3.4, R3.9; R1.2 and R1.3 reporting)

The changes in `decide_step1.py` (`v3_decide_2`):

| gate | rule |
|---|---|
| G1 | Unchanged. The output records that one seed per arm leaves seed noise out of the CI. If \|Δ − 0.20\| ≤ 0.15 it sets `second_seed_required` and adds a note. |
| G1, also reported (not gating) | the same Δ on the full-history subset and split by past-frame cache source (current vs `_future`-named); a descriptive variant that takes only NC / TTC from the arm and freezes DAC / EP / C from ck2T |
| G2, G3 | Absolute thresholds as before. Also reported: the (a), (b), (c) values and the paired (a)−(b), (a)−(c) deltas with log-cluster CIs. For AUC this is a new weighted-AUC bootstrap on the same multinomial log weights as `log_bootstrap`; the test checks it against brute force. |
| G4 | Non-inferiority: pref(a) − pref(c) ≥ −0.01 and CI lower bound ≥ −0.03. (a)−(b) is reported. |

Further changes:
- The frozen-DAC token order (`lead_remedy/infer/ck2T/tokens.parquet`) is asserted to equal the packed order (R3.9).
- The full-history and cache-source subsets now come from the K96 val pack.

Tests: `test_v3_decide::test_auc_boot_matches_roc_auc_and_brute_force`, `::test_decide_d4_reports_and_second_seed_flag`, `::test_g4_noninferiority_edges`. The updated real self-compare still gives PDMS 90.814 and 1,583 hazard pairs, with every delta 0.

### 2.5 Smaller fixes

- **R2.3:** `TemporalCfg.normalised()`. OFF configs compare equal whatever their other fields (`build_ck(init_from)`, resume guard). Test: `test_temporal_cfg_fc_rules`.
- **R2.9:** new stat `t_n_valid` (valid t0 slots; 0 for an object-free batch) next to `t_n_obj`.
- **Smoke tooling:** `smoke_v3.sh` uses the real K96 packs and `D3/ca_scales.json`. The modes are cpu, gpu0 and gpu01; each runs stage pre, then arms a (frozen on that pre), aj (joint), b, d and off. `smoke_summary.py` understands the pre stage.

## 3. Findings not fixed, and why

| finding | status | reason |
|---|---|---|
| R1.2 vnoise calibration (0.3 → about 0.35 out-of-sample) | not changed | D2 keeps the arms on the same augmentation. Changing σ is a protocol change for the user. Navtest is now measurable instead: stage pre logs learned vs CA ADE on navtest (out-of-sample) every epoch. |
| R1.2 navtest critic metrics in infer | not built | infer and decide are val-only by design (SPEC §8). The forecaster-level navtest check is in stage pre. |
| R1.3 past-frame availability is an artefact | reporting only | The real fix runs BEVFusion on the missing pos−1 / pos−2 frames. That is GPU teacher inference, out of Fix scope. The full-history and cache-source deltas are reported next to G1. |
| R2.5 forecast loss normalised per GPU | not changed | With D3, Step 1 has no forecast loss under DDP: a is frozen, and pre-training is one process. It matters only for the optional `joint` mode; use one world size there. |
| R2.6 KD / e2e cannot use a temporal teacher | not changed | Step-2 work; it fails loudly (`_temporal_required`). |
| R2.8 speed variants are not shared | not changed | A design choice (SPEC finding 7: per-candidate stations) that needs the user's confirmation. |
| R3.2 one seed | rule only | D4: no extra run without approval. G1 near the threshold sets `second_seed_required`. The per-epoch checkpoints `ckpt_ep7` / `ckpt_ep8` exist for a descriptive check (`STEP1_RUN.md` §5). |
| R3.5 6 modes vs 1 | not changed | Inherent to "learned vs analytic"; documented. |
| R3.10 class-mapping contract, v2det history | not built | Step-2 work. The CA scales and ROI are already tied to a source. |
| R3.11 arm (c) code era | not changed | A c′ rerun at v3 code costs about 2.8 h of GPU and needs approval. The fresh dump (D8) reproduces exactly, and G4 is non-inferiority against it. |

## 4. Verification log

- **v3 tests:** 133 passed (`D3/logs/fix_v3_tests.log`):

  | file | tests |
  |---|---|
  | objects | 15 |
  | temporal_model | 34 |
  | model_patch | 5 |
  | temporal_real | 3 |
  | loss_metrics | 21 |
  | arms | 6 |
  | decide | 23 |
  | train | 14 |
  | integration_off | 1 |
  | review_fixes | 11 |

- **Existing CK2 tests:** 238 passed, 1 failed (`D3/logs/fix_existing_tests.log`). The failure is `test_ck_e2e_launch.py::test_ck_yaml_matches_contract_and_config` (`kd_ramp_epochs` 0.0 vs 3.0). It predates v3: the integrator saw it, and the reference tree gives the same result.
- **After the last edit** (`provenance.check_self` in train_ck2), these were rerun:
  - `test_v3_review_fixes`, `test_v3_train`: 25 passed;
  - `test_v3_integration_off`, `test_ck2_train`, `test_v3_arms`: 31 passed;
  - `smoke_v3.sh cpu pre a b`.
- **OFF == 8ae8b1d, bitwise:** `test_off_bitwise_identical_to_8ae8b1d` passed. It checks 1,151 arrays and 736 batch hashes, the param counts 3,516,178 / 3,391,243, and that `config.json` only gains the whitelisted keys (+ `t_fc_init`, `t_fc_train`, `provenance`).
- **CPU smoke** (`D3/smoke/cpu/`): pre, off, a (frozen), aj (joint), b and d are all finite with 0 skipped steps. The trainable counts are a = b = d = 3,629,131, aj 4,101,216 and off 3,391,243.
- **GPU smokes** [실측 `D3/smoke/{gpu0,gpu01}/summary.json`], 60 steps each:

  | setup | arm | s/step | vs off | peak GB / GPU | skipped (GradScaler) |
  |---|---|---|---|---|---|
  | GPU 0, world 1, 8 tok / rank | off | 0.180 | – | 5.74 | 2 |
  | | a (frozen) | 0.234 | ×1.30 | 6.86 | 0 |
  | | b | 0.232 | ×1.29 | 6.86 | 1 |
  | GPUs 0,1, world 2, 8 tok / rank (Step-1 setup) | off | 0.168 | – | 5.75 | 2 |
  | | a (frozen) | 0.237 | ×1.41 | 6.86 | 1 |
  | | b | 0.224 | ×1.33 | 6.86 | 2 |
  | GPU 0 | pre | 0.040 (batch 64) | – | 3.13 | – |

  Every val split was finite. ck2T10 itself had 3 skipped steps by step 100.
- **Arm (c) dump** (D8): `D3/infer/ck2T10`, 156 s on GPU 0. Check: pdms_ck_noim 0.9032467603683472, equal to ck2T10's `val_metrics.jsonl` (|Δ| = 0, n_sel 18,179).
- **GPU time used:** about 9 min in total, as separate jobs of at most 156 s, on GPUs 0 and 1 only. GPUs 2–7 were never touched.
