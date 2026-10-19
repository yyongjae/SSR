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
| G1 | Unchanged. The output records that one seed per arm leaves seed noise out of the CI. ~~If \|Δ − 0.20\| ≤ 0.15 it sets `second_seed_required` and adds a note.~~ → **2026-10-11 tooling fix F1(§5.1)로 대체:** seed ablation 없음(사용자 결정), 단일 seed로 결정 확정, 경고 필드 `near_threshold`만 남김. |
| G1, also reported (not gating) | the same Δ on the full-history subset and split by past-frame cache source (current vs `_future`-named); a descriptive variant that takes only NC / TTC from the arm and freezes DAC / EP / C from ck2T |
| G2, G3 | Absolute thresholds as before. Also reported: the (a), (b), (c) values and the paired (a)−(b), (a)−(c) deltas with log-cluster CIs. For AUC this is a new weighted-AUC bootstrap on the same multinomial log weights as `log_bootstrap`; the test checks it against brute force. |
| G4 | Non-inferiority: pref(a) − pref(c) ≥ −0.01 and CI lower bound ≥ −0.03. (a)−(b) is reported. |

Further changes:
- The frozen-DAC token order (`lead_remedy/infer/ck2T/tokens.parquet`) is asserted to equal the packed order (R3.9).
- The full-history and cache-source subsets now come from the K96 val pack.

Tests: `test_v3_decide::test_auc_boot_matches_roc_auc_and_brute_force`, `::test_decide_d4_reports_and_second_seed_flag` (F1 이후 `::test_decide_d4_reports_and_band_flag`), `::test_g4_noninferiority_edges`. The updated real self-compare still gives PDMS 90.814 and 1,583 hazard pairs, with every delta 0.

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
| R3.2 one seed | rule only | D4: no extra run without approval. ~~G1 near the threshold sets `second_seed_required`.~~ **2026-10-11 사용자 결정: seed ablation은 하지 않음.** 결정은 단일 seed로 확정이고, G1이 임계값 ±0.15 안이며 G2–G4가 통과하면 `near_threshold` 경고(낮은 신뢰도)만 기록합니다(§5.1). The per-epoch checkpoints `ckpt_ep7` / `ckpt_ep8` exist for a descriptive check (`STEP1_RUN.md` §5). |
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

## 5. 2026-10-11 tooling fixes

작성 2026-10-12 00:30 KST, tooling fixer. 적대적 검증(코드 근거 + CPU 실험)으로 확인된 결함 4개(F1–F4)를 고쳤습니다. 커밋 없음, 학습 없음, GPU 사용 없음. 실행 중인 arm (b) 학습과 `SSR-ck2` 워크트리는 건드리지 않았습니다.

**현재 Step-1 산출물에 대한 영향: 넷 다 없음** [실측, 검증 보고]. decide는 아직 실행되지 않았고, arm (a)/(b) infer 폴더도 아직 없습니다. arm (c) 덤프, K96 pack 3개, `v3fc_pre_learned6`은 각 결함의 발동 조건을 거치지 않았습니다. 수정은 이후 실행(arm a/b infer, decide, 재개, 재빌드)부터 적용됩니다.

| id | 파일 | 결함 | 수정 |
|---|---|---|---|
| F1 | decide_step1.py | "second seed 필요" 조건에서도 `decision: GO`, `go: true`로 기록됨 | seed ablation 없음(사용자 결정) → 단일 seed로 결정 확정, `near_threshold` 경고 |
| F2 | infer_val_v3.py, decide_step1.py | 입력이 바뀌어도 done 행을 재사용하고 meta만 새 입력으로 덮어씀 | `cache_key.json` + 거부, decide에서 ckpt 재해시와 cache key 검사, arm (c) legacy 경로 |
| F3 | pretrain_forecaster.py | `--batch` / `n_train`을 바꿔 재개하면 sample을 건너뛰거나 중복 학습함 | step geometry를 config와 ckpt에 기록, 다르면 재개 거부 |
| F4 | build_obj_pack.py | 중단된 빌드를 입력이 바뀐 뒤 재개하면 옛 chunk와 새 chunk가 섞임 | part_key에 입력 fingerprint 포함 |

### 5.1 F1: 단일 seed로 결정 확정 (사용자 결정 2026-10-11)

**결함.** |Δ − 0.20| ≤ 0.15이면 `G1.second_seed_required`를 켜고 "before the decision is final" note를 남기면서도, 결론 필드는 `decision: "GO"`, `go: true`였습니다. 그래서 잠정 결론이 확정처럼 보였습니다. 두 seed를 합치는 규칙도 사전등록돼 있지 않았습니다.

**결정.** seed ablation은 하지 않습니다(사용자 결정 2026-10-11). 그러므로 결정은 seed 1개로 확정합니다.

**수정** (`VERSION v3_decide_3`):
- 삭제: `second_seed_required`, `GO_RULE.G1.second_seed_band`, "second seed … before the decision is final" note, stdout의 `SECOND SEED REQUIRED`.
- `GO_RULE.G1.near_threshold_band = 0.15`. `gates.G1.within_band`는 G2–G4와 무관한 원래 구간 판정입니다.
- 최상위 `near_threshold`: |Δ − 0.20| ≤ 0.15 **이고** G2–G4가 모두 통과할 때만 `"single seed; delta within 0.15 of the threshold -> low confidence"`, 그 외에는 `null`입니다. G2–G4 중 하나라도 fail이면 G1과 상관없이 NO-GO이므로 경고할 이유가 없습니다.
- 최상위 `seed_policy = "single seed (user decision 2026-10-11)"`, `final = valid_for_decision`.
- decision 텍스트는 `GO` / `NO-GO` / `INVALID (would be …)` 그대로입니다. stdout 첫 줄은 `decision: GO  [final; single seed (user decision 2026-10-11)]`이고, 경고가 있으면 다음 줄에 `WARNING near_threshold: …`가 찍힙니다. INVALID는 final이 아닙니다.

**테스트.** `test_v3_decide::test_single_seed_decision_final_and_near_threshold`가 G2–G4를 통과하도록 설계한 world에서 다음을 확인합니다:

| Δ (설계값) | G2 | 기대 결론 | near_threshold |
|---|---|---|---|
| +0.30 | pass | GO (final) | 경고 |
| +1.00 | pass | GO (final) | 없음 |
| +0.10 | pass | NO-GO (final) | 경고 |
| +0.30 | fail | NO-GO (final) | 없음 |

`--no-arm-check`(INVALID)이면 `final: false`입니다. 이 밖에 `::test_decide_d4_reports_and_band_flag`(구 `…_second_seed_flag`)와 `::test_preregistered_constants`도 갱신했습니다.

### 5.2 F2: infer cache key, decide의 ckpt / cache 검사, legacy 경로

**결함.**
- `infer_val_v3.py`는 `--out`에 이미 done인 행과 memmap이 있으면 그대로 재사용했습니다. ckpt, which, cand, variants, object pack, ca_scales, 코드가 같은지는 확인하지 않았습니다. 그러면서 `infer_meta.json`은 새 입력으로 덮어써서, 옛 점수에 새 메타가 붙었습니다.
- variants가 바뀐 경우는 logit16 check로도 잡히지 않았습니다. 80개 variant는 16개 후보와 따로 점수가 매겨지기 때문입니다.
- decide는 `ckpt_sha16`을 기록만 하고 비교하지 않았습니다.
- `STEP1_RUN.md` §5의 "`--out`을 안 주면 last 덤프를 덮어쓴다"는 틀린 설명이었습니다. 실제로는 last 점수를 그대로 둔 채 meta만 ep7로 바꿨고, check도 ok로 나왔습니다 [실측: 검증 실험].

**수정 (infer).** `<out>/cache_key.json`을 행 계산 전에 씁니다. `version v3_infer_cache_1`이고 내용은 다음과 같습니다:
- run (resolved), which, ckpt 파일명, `ckpt_sha256`(모델 로드 전에 해시);
- packed `tokens.parquet` / `ok.npy` / `cand.npy` / `status.npy`의 sha256과 variants sha256;
- temporal cfg, privileged, 정밀도(`fp16 autocast` / `fp32 (cpu)`);
- `data` = 이 추론의 provenance data(val pack build.json sha, ca_scales sha, fc_init sha);
- `code` = forward 코드 sha256: `infer_val_v3.py`에서 시작한 **정적 import closure**입니다. ast로 module 수준과 함수 안의 `import` / `from .. import`를 모두 따라가고, 상대 import를 풀고, 부모 package를 포함하며, 범위는 repo의 `navsim/`, `tools/`입니다. 현재 40개 파일이고 model, temporal branch / forecast / raster, refiner, CKDataset, obj_pack, ckutil, 이 스크립트가 들어갑니다. 파일별 sha16도 함께 기록합니다.
  - 처음에는 `ck/**` 전체를 glob했습니다. 그런데 전체 테스트 중 다른 세션이 `online2.py`를 고치자 같은 키 재실행이 거부됐습니다 [실측, 아래 §5.5]. 그래서 실제로 import되는 코드만 보도록 바꿨습니다. e2e student 코드(`online2.py`, `e2e_data2.py`)와 `train_ck2.py`는 키에 들어가지 않습니다.
  - 정적 분석이므로, 그 프로세스가 다른 모듈을 import했는지와 무관하게 키가 같습니다.

검사와 기록:
- `--out`에 done 행이 있는데 `cache_key.json`이 없거나(F2 이전 덤프) 키가 다르면, 아무것도 쓰지 않고 SystemExit합니다. 메시지에 다른 필드 이름이 나오고 새 `--out`을 쓰라고 안내합니다. 키가 같으면 이어서 계산합니다.
- `infer_meta.json`에 `cache_key`를 기록하고, `ckpt`는 절대경로로 씁니다. `sessions`는 키가 같을 때만 이어 붙입니다.
- 한계: BEV feature 파일 자체는 해시하지 않습니다. 공용 teacher dump이고 read-only입니다.

**수정 (decide `check_arm_meta`).**
- `infer_meta.ckpt`를 다시 해시해 `ckpt_sha16`과 비교합니다. 파일이 없거나 sha가 다르면 거부합니다.
- `cache_key`가 필수입니다. infer_meta의 키와 폴더의 `cache_key.json`이 같아야 하고, ckpt / which / temporal이 meta와 맞아야 하며, cand / variants sha가 현재 파일과 같아야 합니다.

**Legacy 경로** (F2 이전 덤프, 대표적으로 arm (c) `D3/infer/ck2T10`; 이 덤프는 다시 쓰지 않습니다). 키가 없는 덤프는 다음을 모두 만족할 때만 받아들입니다:
- 세션이 정확히 1개이고 그 세션의 n이 n_done, n_rows와 같습니다. 즉 모든 행을 한 세션에서 계산했으므로 다른 입력으로 계산된 행이 섞일 수 없습니다.
- 위의 ckpt / cand / variants 검사를 통과합니다.

받아들인 legacy 덤프는 step1.json에 `inputs.<arm>.cache = {mode: "legacy", legacy_rule, …}`, `legacy_dumps`, note로 표시됩니다.

**실측.**
- ck2T10 덤프는 legacy 조건을 만족합니다: sessions [18179], ckpt sha16 7d42718ef99ca6c4 = 현재 파일, cand / variants sha 일치 (`::test_real_arm_c_legacy_dump_accepted`).
- 실제 decide를 arm check와 함께 dry run했습니다(`--a/--b/--c D3/infer/ck2T10`). 거부된 것은 a와 b뿐이고 이유는 "must be temporal on"이었습니다. c는 legacy로 통과했습니다.
- `--no-arm-check` dry run(`/tmp/v3fix_decide_dry/step1_dry.json`): PDMS 90.919, lead TTC AUC .8369, lead fail 6.17 %, a-1.0 pref .6526으로 STEP1_RUN의 arm (c) 기준값과 같습니다. 결과는 `INVALID (would be NO-GO)`, `final: false`, `legacy_dumps [a, b, c]`(NOT checked)입니다.
- temporal 경로는 CPU에서 확인했습니다(`smoke_gpu01_a`, 4행, `/tmp`): cache_key가 생성되고(data: packs / ca_scales / fc_init, code 40 files), 같은 키로 다시 실행하면 sessions [4, 0], decide cache / ckpt 문제 0입니다. check는 CPU fp32 vs GPU fp16이라 not ok인데, 기존과 같고 예상된 결과입니다.

**테스트.** `test_v3_tooling_fixes::test_key_diff_and_check_cache`, `::test_cache_key_fields`, `::test_real_infer_cache_reuse_and_refusals`. 마지막 테스트는 ck2T 8행을 CPU로 계산한 뒤 다음을 확인합니다:
- 같은 키로 재실행하면 기존 행을 재사용합니다.
- 같은 `--out`에 `--which ep28`를 주면 거부되고, 출력 파일 바이트가 그대로입니다.
- `cache_key.json`을 지우면 거부됩니다.

`test_v3_decide::test_check_arm_meta_ckpt_and_cache_key`, `::test_check_arm_meta_legacy_dump`, `::test_check_arm_meta_rules`(갱신)도 같이 고쳤습니다.

### 5.3 F3: forecaster pre-training 재개 검사

**결함.** 재개 위치를 sample이 아니라 step 수로 저장하고, 새 `--batch`로 계산한 `steps_per_epoch`로 해석했습니다. 그 결과:
- batch를 키우면 sample이 누락되고, 줄이면 같은 순서·같은 augmentation으로 중복 학습됩니다. 그런데도 `done.json`은 정상 완료처럼 보입니다.
- `--limit-tokens`로 `n_train`을 바꿔도 같은 문제가 생깁니다.
- `config.json`은 첫 실행 때만 써서 옛 batch가 그대로 남습니다.

**수정.**
- `config.json`에 최상위 `batch`를 추가하고, `ckpt_last.pt`에 `batch` / `n_train` / `steps_per_epoch` / `total_steps`를 저장합니다.
- `--resume`이면 n_modes / seed / epochs / aug와 위 4개 값을 config.json 및 ckpt(기록돼 있을 때)와 비교합니다. F3 이전 config는 `args.batch`로 비교합니다. 다르면 SystemExit합니다.
- batch를 바꿔 이어 학습하는 기능은 지원하지 않습니다. LR schedule도 step 단위라서, 위치를 다시 계산하기보다 거부하는 편이 단순하고 안전합니다.

**테스트** (CPU, 실제 K96 train pack 40행, batch 8, 3 epoch; `::test_pretrain_resume_geometry`, `::test_resume_mismatch_unit`):
- epoch 0 ckpt 직후에 kill하고 같은 설정으로 재개하면, 남은 epoch마다 모든 sample을 1회씩 학습하고 최종 forecaster 가중치가 중단 없는 실행과 **bitwise 동일**합니다.
- `--batch 16`, `--limit-tokens 32`, `--max-steps`로 재개하면 모두 거부됩니다.
- 옛 결함도 재현했습니다. 수정 전 코드(커밋 aa7a7b2)는 같은 상태에서 `--batch 16`으로 재개하면 epoch 1의 sample 일부만 학습하고, `done.json`에 `epochs_done 3`을 쓰고 조용히 끝납니다.

기존 `v3fc_pre_learned6`은 재개한 적이 없어 영향이 없습니다. 그 config(F3 이전 형식)도 같은 인자로 재개하면 통과합니다.

**부작용 (중요).** `pretrain_forecaster.py`는 `provenance.TRAIN_CODE`에 들어 있습니다. 그래서 이 수정만으로 training-code sha가 **eb1bb78f96af8e99 → 3aeff8960e5d2200**으로 바뀝니다. 원본 파일로 sha를 다시 계산하면 eb1bb78f96af8e99로 돌아오는 것을 확인했습니다 [실측].

**이 수정과 무관한 동시 변경 [실측].** 다른 세션이 2026-10-12 00:18 KST부터 같은 워크트리의 `navsim/agents/para_ssr/ck/e2e_data2.py`, `online2.py`를 고치고 있습니다(Step-2 temporal DET teacher, `docs/refiner_v3/STEP2_E2E.md` 참조). 둘 다 TRAIN_CODE에 들어 있습니다. 00:19 KST 기준 sha는 3d7a862f30c5ce3b이고, pretrain만 원본으로 돌려도 4ec95ac6674bc12e입니다. 세 파일을 모두 HEAD로 돌려야 eb1bb78f96af8e99가 됩니다. 이 두 파일은 건드리지 않았습니다.

- 지금 학습 중인 arm (b)가 끝나기 전(≈ 02:15 KST)에 죽어서 `run_step1.sh b`로 재개하면, train_ck2의 D5 재개 검사가 "code.train.sha256 differs"로 거부합니다. 이 경우 아래 절차를 따릅니다(2026-10-12 리뷰 후속으로 수정, §6 finding 3; 이전 판의 `git stash push -- <files>` / `git stash pop` 절차는 **폐기**: stash 스택은 다른 워크트리·세션과 공유되므로 `pop`이 남의 항목을 적용할 수 있고, 다른 세션의 작업 중 파일을 빼앗으며, `git diff --stat`은 sha에 포함되는 untracked `.py`를 보여주지 않습니다).
  1. **sha를 직접 확인합니다.** `cd /workspace/yongjae/SSR-v3 && PYTHONPATH=/workspace/yongjae/SSR-v3 /venv/ssr/bin/python -B -c "from tools.ck.v3 import provenance as PV; print(PV.train_code()['sha256'][:16])"` 출력이 b의 `config.json` 기록값 **eb1bb78f96af8e99**여야 재개됩니다(2026-10-12 00:46 KST 현재 2386cecba71b2ce2). 어떤 파일이 다른지는 `PV.train_code()['files']`를 b의 `config.json` `provenance.code.train`과 비교해 봅니다.
  2. **F3 파일만 되돌립니다.** `cp tools/ck/v3/pretrain_forecaster.py /tmp/pretrain_forecaster.f3.py && git show HEAD:tools/ck/v3/pretrain_forecaster.py > tools/ck/v3/pretrain_forecaster.py` (HEAD 판 = `/tmp/pretrain_forecaster.orig.py`, sha256 `1aaaa0ca…`).
  3. **다른 세션의 파일(`e2e_data2.py`, `online2.py` 등)은 여기서 되돌리지 않습니다.** 사용자나 그 세션에 잠시 HEAD 판으로 돌려 달라고 요청하고, 1번 sha가 eb1bb78f96af8e99가 될 때까지 기다립니다.
  4. `V3_USER_APPROVED=1 bash tools/ck/v3/run_step1.sh b`로 재개합니다.
  5. `D3/train/v3T10_b_ca.launch.log`에 `resumed …` 줄이 나오고 `train_log.jsonl`에 새 step 기록이 생긴 뒤(두 rank 모두 provenance 검사를 통과한 뒤) `cp /tmp/pretrain_forecaster.f3.py tools/ck/v3/pretrain_forecaster.py`로 F3 수정본을 복원하고, 다른 세션에도 복원해도 된다고 알립니다.
  6. stash가 꼭 필요하면 `git stash push -u -m <고유-tag> -- <files>` 후 `git stash list --format='%H %gs'`로 자기 항목의 sha를 잡아 `git stash apply <sha>`로 복원하고, tag로 다시 찾아 drop합니다. bare `git stash pop`은 쓰지 않습니다.
- 앞으로 새로 학습하는 arm(예: d)은 a / b와 code sha가 달라 decide provenance 검사에 걸립니다.
- a, b, c로 decide하는 데는 영향이 없습니다. decide는 각 run의 config.json에 기록된 학습 시 provenance만 비교합니다(a = b = eb1bb78f96af8e99).

### 5.4 F4: object pack 빌드 재개 키

**결함.** chunk 재사용 키(part_key)에는 layout(version, K_OBJ, D_OBJ, threshold, targets)만 들어 있었습니다. 그래서 GT 내용, BEVFusion cache나 frames parquet, `MATCH_GATE_M`, `FC_T`, `N_FRAMES`, 코드가 바뀐 뒤 중단된 빌드를 재개하면 옛 chunk와 새 chunk가 섞였고, build.json에는 새 provenance만 기록됐습니다.

**수정.**
- provenance를 chunk를 만들기 **전에** 계산합니다. build.json에도 이 값이 기록됩니다.
- 입력 fingerprint = sha256(provenance에서 git_head / git_note를 뺀 것, `MATCH_GATE_M`, `FC_T`, `N_FRAMES`, GT 파일별 manifest `D3/gt_objects/_manifests/<split>.json`의 sha256)입니다.
- 이 fingerprint를 part_key의 `inputs`에 넣고 worker init에도 넘깁니다. assembly 단계에서도 모든 part의 키를 다시 검사합니다.
- build.json에 `inputs_sha256`과 `inputs`를 기록합니다.
- 기존 complete pack(K96 3개, fingerprint 없음)은 그대로 로드됩니다. ObjPack은 이 필드를 읽지 않습니다.
- 한계: GT 파일 내용은 manifest와 index sha로만 감지합니다. manifest를 다시 만들지 않고 GT를 바꾸면 잡지 못합니다.

**테스트** (`/tmp`, navtrain_val 32행, chunk 16; `::test_build_resume_rebuilds_on_changed_gate`, `::test_part_key_includes_fingerprint`, `::test_existing_complete_pack_loads_unchanged`):
- 입력이 그대로인 재개: "0 to build", 결과 바이트 동일.
- gate를 2.0에서 0.5로 바꾼 재개: "2 to build". 결과는 처음부터 gate 0.5로 만든 pack과 바이트가 같고, gate 2.0 결과와는 target이 다릅니다. 즉 수정 전이었다면 섞인 pack이 됐을 상황입니다.

### 5.5 검증과 남은 일

**실행 결과** [실측; CPU only, `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=1`, `python -B`; 임시 파일은 `/tmp`, 기존 테스트가 쓰는 `D3/test_tmp`만 예외]:
- **전체 v3 테스트(`tests/test_v3_*.py`): 146 passed, 0 failed, 0 skipped, 8 min 9 s.**
  - 파일별 개수: arms 6, decide 27(+4), integration_off 1, loss_metrics 21, model_patch 5, objects 15, review_fixes 11, temporal_model 34, temporal_real 3, tooling_fixes 9(신규), train 14.
  - 다른 세션이 00:31 KST에 만든 `test_v3_e2e_teacher.py`는 실행 시작 뒤에 생겨서 포함되지 않았습니다.
- **첫 전체 실행: 144 passed, 1 failed.** 실패한 것은 `test_real_infer_cache_reuse_and_refusals`였습니다.
  - 두 infer 호출 사이에 다른 세션이 `navsim/agents/para_ssr/ck/online2.py`를 고쳤고, 같은 키여야 할 재실행이 `code.files.…online2.py` 차이로 거부됐습니다.
  - 즉 cache key는 설계대로 작동했지만, 범위가 넓어서 무관한 코드 변경에도 거부했습니다. 그래서 forward 코드를 정적 import closure로 좁혔고(§5.2), 그 뒤 위의 146 passed를 얻었습니다.
- **decide known-answer 테스트**(전체 실행에 포함):
  - `test_real_ck2T_selfcompare`: PDMS 90.814, 1,605 lead, 1,583 hazard pairs, 모든 Δ 0.
  - `test_reference_equality_with_lib`, `test_auc_boot_matches_roc_auc_and_brute_force`, `test_g4_noninferiority_edges`.
- **decide dry run** (a/b infer 폴더가 아직 없어 arm (c) 덤프로만, 출력은 `/tmp/v3fix_decide_dry/`):
  - arm check를 켜면 a / b만 "must be temporal on"으로 거부되고, c는 legacy로 통과합니다.
  - `--no-arm-check`이면 arm (c) 기준값(90.919 / .8369 / 6.17 % / .6526)이 그대로 나오고, `INVALID (would be NO-GO)`, `final false`, `legacy_dumps [a, b, c]`(NOT checked)입니다.
- 수정 전 원본 4개는 `/tmp/*.orig.py`에 있습니다(diff용).

**main session이 할 일.** a / b infer 명령은 STEP1_RUN §4 그대로 쓰면 됩니다. 기본 출력 폴더가 아직 없으므로 처음부터 계산됩니다. 그다음 decide를 실행합니다. arm (c)는 legacy로 받아들여지고 step1.json에 표시됩니다.
- 하지 않은 것:
  - arm (a) / (b) infer와 decide 본 실행: main session이 할 일입니다.
  - BEV feature 해시(F2 한계).
  - GT 파일 내용 전체 해시(F4 한계).
  - 2-seed 결합 규칙: 사용자 결정으로 필요 없어졌습니다.


## 6. 2026-10-12 F1–F4 리뷰 후속 수정

작성 2026-10-12 01:15 KST, tooling fixer. F1–F4 수정에 대한 리뷰(차단 결함 0, medium 1, low 5)를 반영했습니다. 커밋 없음, 학습 없음, GPU 사용 없음. 실행 중인 arm (b) 학습, `tools/ck/train_ck2.py`, `navsim/**`, `tools/ck/data/**`, `SSR-ck2` 워크트리는 건드리지 않았습니다. 바꾼 파일: `tools/ck/v3/decide_step1.py`, `tools/ck/v3/infer_val_v3.py`, `tests/test_v3_decide.py`, `tests/test_v3_tooling_fixes.py`, 이 문서, `STEP1_RUN.md`, `SPEC.md` §12.1. 둘 다 TRAIN_CODE가 아니므로 training-code sha는 바뀌지 않습니다.

| # | 등급 | 내용 | 조치 |
|---|---|---|---|
| 1 | medium | decide가 arm 간 forward 코드를 비교하지 않음 | **수정**: `code_check` |
| 2 | low | 사전등록이 아닌 bootstrap(`--n-boot 200`)도 `final: true` | **수정**: INVALID, final 아님 |
| 3 | low (문서) | arm (b) 재개 절차의 `git stash push/pop`이 위험 | **수정**: §5.3 절차 교체 |
| 4 | low | 기본 `--out`이 `--which`를 무시 | **수정**: last가 아니면 `<run>_<which>` |
| 5 | low | 같은 키 재개 시 `fc_metrics.json`이 첫 세션 행만 반영 | **수정**: 합계 저장·합산 + decide note |
| 6 | low | ckpt를 모델 로드 전에만 해시 | **수정**: 로드 전후 sha 비교 |
| (추가) | – | 학습 중인 run의 `ckpt_last`를 infer할 수 있음 | **수정**: `done.json` 필수 (infer, decide) |

### 6.1 Finding 1: arm 간 forward 코드 검사 (medium)

**결함.** cache key에 `code.sha256`과 파일별 sha가 기록되지만 arm 사이에서 비교하지 않았습니다. 다른 세션이 같은 워크트리의 `navsim/agents/para_ssr/ck/`를 고치는 중이라, a와 b infer 사이에 `model.py`, `temporal/*`, `tools/ck/data/*`가 바뀌면 두 arm이 다른 코드로 채점돼도 경고가 없었습니다.

**수정** (`decide_step1.code_check`, `VERSION v3_decide_4`):
- cache key가 있는 모든 arm(a, b, d, 그리고 c도 키가 있으면)은 `cache_key.code.sha256`과 `cache_key.amp`가 기준 arm(a, b, d, c 순으로 첫 keyed arm)과 같아야 합니다. 다르면 **오류**이고 `code.files`에서 다른 파일(한쪽에만 있는 파일 포함)을 나열합니다. `run()`은 arm check 단계에서 SystemExit합니다.
- legacy arm(c `D3/infer/ck2T10`)은 기록된 `code_sha256`(MODEL_FILES 7개; F2가 바꾼 `infer_val_v3.py`는 제외)과 `amp`를 기준 arm의 코드와 비교해 **경고**만 냅니다. legacy 덤프는 전체 closure를 기록하지 않았기 때문입니다.
- 결과는 `step1.json`의 `code_check`(`ref`, arm별 sha, `legacy_compared`, errors, warnings)와 stdout `forward code:` 줄에 나옵니다. `decide()`는 `code_check`가 없거나 오류가 있으면 INVALID입니다.
- 실측: 현재 forward 코드 sha **3475c5a4ffc2a8e2**(40개 파일, HEAD와 다른 것은 `infer_val_v3.py`뿐). legacy c의 MODEL_FILES 7개는 현재 코드와 모두 같습니다(differ `[]`).

### 6.2 Finding 2: 사전등록 bootstrap이 아니면 INVALID

`run()`이 `preregistered = (n_boot == 2000 and seed == 0)`을 `decide()`에 넘깁니다. 아니면 `valid_for_decision: false`, `final: false`, `go: false`, `decision: "INVALID (would be …)"`, `preregistered_bootstrap: false`, note가 붙습니다. 합성 테스트는 `decide()`를 직접 부르므로(작은 n_boot) 영향이 없습니다. CLI는 `[warn] … diagnostic (INVALID, never final)`을 찍습니다.

### 6.3 Finding 3: arm (b) 재개 절차 (문서)

§5.3과 `STEP1_RUN.md` "Still open"의 `git stash push -- <files>` / `git stash pop` 절차를 폐기하고, sha 직접 확인(eb1bb78f96af8e99) → F3 파일만 `git show HEAD:…`로 되돌림 → 다른 세션 파일은 그 세션과 조율 → 재개 → `resumed` 줄과 새 step 기록 확인 후 복원, 으로 바꿨습니다(§5.3). arm (b)가 정상 종료하면 필요 없는 절차입니다.

### 6.4 Finding 4: `--which`별 기본 출력 폴더

`--out`이 없으면 `which == last`일 때 `D3/infer/<run>`, 아니면 `D3/infer/<run>_<which>`입니다(`infer_val_v3.default_out_name`). 예전에는 ep7 infer를 먼저 하면 기본 폴더를 차지해 last infer가 (안전하게) 거부되어 막혔습니다.

### 6.5 Finding 5: forecast metrics 합산

- `fc_metrics.json`에 `ForecastMetrics`의 가산 합계(`sums.arm`, `sums.ca_reference`)를 함께 저장합니다.
- 같은 cache key로 재개하면 이전 합계에 이번 세션 합계를 더해 다시 씁니다(`n_rows` 누적, `n_sessions_merged`). 키가 다른 폴더는 F2에서 이미 거부되므로 섞이지 않습니다. 합계가 없는 옛 파일은 이번 세션 값으로 바꾸고 로그를 남깁니다.
- decide는 `fc_metrics.n_rows != infer_meta.n_done`이면 note를 남깁니다(forecast metrics는 서술용이고 gate가 아님). 세션이 행 계산 중 죽은 경우 등을 표시합니다.

### 6.6 Finding 6: 로드 전후 ckpt sha 비교, 학습 완료 검사

- `infer`는 키에 쓸 sha256을 모델 로드 전에 계산하고, `load_ck`가 파일을 읽은 뒤 계산한 `net.init_report["ckpt_sha16"]`과 비교합니다. 다르면(로드 중 파일 교체) 아무것도 쓰지 않고 SystemExit합니다.
- (추가) `--which last`는 `<run>/done.json`이 있어야 합니다. 학습 중인 run의 `ckpt_last`는 중간 가중치이고 eval 직후라면 reproduce check까지 통과할 수 있기 때문입니다. `--allow-unfinished`는 점검용이며 `infer_meta.run_done`에 기록됩니다. decide도 meta의 `run`에 `done.json`이 없으면 그 arm을 거부합니다. ck2T, ck2T10, v3T10_a_learned에는 있고, v3T10_b_ca는 학습 종료 후 생깁니다.

### 6.7 테스트

신규·확장 (`tests/test_v3_decide.py`, `tests/test_v3_tooling_fixes.py`):
- `test_v3_decide::test_code_check_between_arms`: 같은 코드 통과, 다른 파일·한쪽에만 있는 파일·amp 차이·d·keyed c → 오류(파일명 포함), legacy c 차이 → 경고만, 기준 arm 순서.
- `::test_decide_code_check_preregistered_and_fc_coverage`: GO 세계에서 code_check 없음/오류 → `INVALID (would be GO)`, 경고만 → GO final, `preregistered=False` → INVALID·final 아님, fc 행 수 note.
- `::test_check_arm_meta_requires_finished_run`, `::test_real_arm_c_legacy_dump_accepted`(확장: 실제 c 덤프 vs 현재 closure 비교, 7개 파일 일치).
- `test_v3_tooling_fixes::test_default_out_name`, `::test_infer_refuses_unfinished_run_and_changed_ckpt`, `::test_merge_fc_sums`.
- `::test_real_temporal_fc_merge_and_decide_code_check`(실제 v3T10_a_learned, CPU): 4행 + 4행 두 세션의 forecast 합계 = 한 번에 8행(rel 1e-4), sessions [4, 4]. 이어서 ck2T10 배열을 /tmp에 복사한 전체 크기 a / b 대역 폴더(이 infer의 meta와 cache key, b는 forecaster ca, check.ok 지정)와 실제 legacy c로 **arm check를 켠** `decide_step1.run`: valid, final, `NO-GO`(a = b = c 점수), `legacy_dumps ['c']`, code check 오류 없음(a = b sha, legacy c 7개 파일 일치, CPU 대역이라 amp 경고), fc note. `n_boot 50` → `INVALID (would be NO-GO)`. b의 코드 sha와 `model.py`만 바꾸면 `forward code differs … model.py`로 거부되고 출력 파일이 생기지 않습니다.

**실행 결과** [실측; CPU only, `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=1`, `python -B`; 임시 파일은 `/tmp`, 로그 `/tmp/v3fix_rev2/pytest_all.log`]:
- **전체 `tests/test_v3_*.py`(다른 세션의 e2e 3개 파일 포함): 165 passed, 0 failed, 0 skipped, 7 min 59 s** (2026-10-12 00:55–01:03 KST).
  - 파일별: arms 6, decide 30(+3), e2e_cpu_dryrun 1, e2e_launch 4, e2e_teacher 7, integration_off 1, loss_metrics 21, model_patch 5, objects 15, review_fixes 11, temporal_model 34, temporal_real 3, tooling_fixes 13(+4), train 14.
  - decide known-answer 테스트 포함: `test_real_ck2T_selfcompare`(PDMS 90.814, 1,605 lead, 1,583 hazard pairs), `test_reference_equality_with_lib`, `test_auc_boot_matches_roc_auc_and_brute_force`, `test_g4_noninferiority_edges`; OFF bit-identity `test_off_bitwise_identical_to_8ae8b1d`.
- 실행 후 상태: forward 코드 sha 3475c5a4ffc2a8e2(40개 파일), training-code sha 2386cecba71b2ce2(이번 수정으로 바뀌지 않음; 다른 세션 파일 + F3 때문). arm (b)는 01:03 KST에 epoch 6 / step 34,200 / 53,190으로 정상 진행 중.

### 6.8 main session이 할 일 (arm (b) `done.json` 생긴 뒤)

`STEP1_RUN.md` §4 명령 그대로입니다: (0) a, b의 `done.json` 확인, (1) forward 코드 sha 출력(3475c5a4ffc2a8e2 예상; 다르면 어떤 파일이 바뀌었는지 확인), (2) GPU 0에서 a → b infer를 연달아 기본 `--out`으로, (3) decide를 사전등록 기본값(`--n-boot`/`--seed` 생략)으로. decide가 `forward code differs`로 거부하면 두 infer 사이에 forward 코드가 바뀐 것이므로, 코드가 고정된 상태에서 a와 b를 새 `--out`으로 다시 돌립니다. smoke는 `/tmp`에만 합니다.
