# refiner-v3 INTEGRATION (SPEC §4.5, §5.2–5.4, §6, §9 Integrate)

Written 2026-10-11 16:40 KST by the integrator. Branch `refiner-v3`, worktree `/workspace/yongjae/SSR-v3`, base ck2 @ 8ae8b1d.
Nothing is committed. No training was run.

Tags: [실측] = measured (log or command named), [추정] = estimate, [미확인] = not checked.
`D3 = /home/external-user/ssd/yongjae_refiner/ck/v3`. Every log named below is under `D3/integrate/logs/` or `D3/smoke/`.

## 0. Status

| item | result |
|---|---|
| model.py patch applied (`docs/refiner_v3/model_py_temporal.patch`) | done |
| datasets: `objects` kwarg (CK2Dataset, CKDataset) | done |
| train_ck2: flags, loss, optimiser groups, logging, config, resume guard, eval | done |
| inference: `infer_val_v3.py` on temporal checkpoints | done (CPU, 16 rows, arms a and b) |
| OFF == CK2 @ 8ae8b1d, bitwise | **identical** [실측] |
| existing CK2 tests (15 files) | 238 passed, 1 failed. The reference tree gives the same result: one failure that has nothing to do with v3 (§3) |
| v3 tests (9 files) | **117 passed**, 0 skipped, 0 failed [실측 `final_v3.log`] |
| CPU smoke, arms off / a / b / d, real data | all finite, 0 skipped steps |
| GPU smoke, GPU 0, arm a vs off | memory **+1.28 GB** (pass). Step time **×1.50** in the train loop and ×1.37 in compute only. **Both fail the ≤ +25 % budget** (§5) |
| Step-1 prerequisites | **open**: a decision on K_OBJ, then the train pack and `D3/ca_scales.json` (§7) |

## 1. What changed (code)

### 1.1 Edits to existing files

**`navsim/agents/para_ssr/ck/model.py`**: the B2 patch, applied with `patch -p1` (+65/−15).
- `CKTrunk.candidates(..., tfeat=None)`: `x = cat([draft_mlp(ud) + ud_add, st + st_add])`.
  - When the net has a branch and `tfeat` is None, it raises RuntimeError.
- `CKNet(..., temporal=None, ca_b=None)`: the branch is built after every CK2 module, from its own seeded RNG.
- `forward(..., objs=None)`:
  - Raises if `objs` is passed to an OFF net, or is missing for an ON net.
  - Calls `scene()` once. Every candidates call (cand, extra, rescore_corr) gets its own hook.
  - Returns `out['fc']` (learned arm only) and `out['t_stats']`.
- `build_ck(..., temporal=None)`: `init_from` a run whose temporal cfg differs raises ValueError.
- `load_ck(..., allow_privileged=False)`: loading an oracle or gt-source run raises PermissionError.
- `param_counts()`: adds the temporal keys only when the branch is on.

**`tools/ck/data/ck2_dataset.py`** (Phase-0 copy; the diff is against `D3/ref_8ae8b1d`):
- `ObjectRows` adapter for the object pack, shared by both datasets. It:
  - checks the spec keys;
  - refuses a cfg that is off;
  - requires targets for the oracle;
  - checks that the pack covers every dataset row in the same token order;
  - computes `n_obj_bad`.
- `CK2Dataset(objects=None)`:
  - `__getitem__` (never `labels_item`) appends `obj_in / obj_mask / frame_valid / obj_ok`, plus `fc_xyh / fc_state`.
  - With `aug`, it applies `augment_objects(token, epoch, seed)` on the stream `'objaug'`.
  - For the oracle it adds `priv_fc_*` copies.
  - `describe()` gains `objects` only when on.
- New constants `OBJ_INPUT_KEYS`, `MODEL_INPUT_KEYS_TEMPORAL`, and `OBJ_BAD_MAX = 0.005`.

**`tools/ck/data/ck_dataset.py`**: `CKDataset(objects=None)` for evaluation. `aug=True` raises, because CKDataset has no epoch or seed.

**`tools/ck/train_ck2.py`** (+179/−16):
- **CLI flags**:
  - `--temporal {0,1}` (dest `temporal_on`), `--t-source`, `--t-forecaster`.
  - `--t-modes`: the default is None, which resolves to 6 for learned and 1 for ca / oracle_gt. This removes B4's "--t-modes 1" trap; an explicit mismatching value still raises.
  - `--t-lambda-fc`, `--t-fc-backprop`, `--t-frame-drop`, `--t-obj-drop`, `--t-vnoise`, `--t-pack-root`, `--t-ca-scales`.
- **`resolve_args` → `a.temporal_cfg`**: off unless `--temporal 1`. It then:
  - runs `validate(arm)`;
  - requires a `_PRIV` run name for a privileged cfg;
  - requires the CA scales file to exist for ca / oracle.
- **`build_datasets`**:
  - Train: `objects = {aug True, targets True}`.
  - Evals (r34, raw, train_eval r34 / raw): no aug, and targets when the pack has them.
  - `check_obj_bad`: SystemExit when more than 0.5 % of a split's tokens are obj_ok False.
  - OFF passes no kwarg at all.
- **`forward(..., tcfg)`**: when on, passes `objs = branch.objs_from_batch(batch, tcfg)` (the whitelist).
- **`compute_loss`**:
  - When the branch is on, adds `loss_ck`, `t_n_obj` and `n_obj_bad`.
  - In arm a, adds `+ lambda_fc * forecast_loss` (via `add_forecast_loss`, which emits the `fc_*` stats). `loss` is the total.
- **`dummy_backward`**: adds `forecast_touch(out['fc'])`.
- **Optimiser and clipping**:
  - `param_groups(net)`: the critic group, and the forecaster group (`temporal.forecaster.*`).
  - With a forecaster: AdamW with 2 groups, each clipped to `--clip`. A non-finite norm in either group skips the step. `grad_norm_fc` is logged.
  - Otherwise the CK2 optimiser and clip code run unchanged.
- **config.json**:
  - New keys: `temporal`, `privileged`, `param_count_temporal_shared`, `param_count_forecaster`, plus the raw flag values.
  - The resume guard compares `temporal`; an old config without the key counts as off.
  - In privileged runs, every val row gets `privileged: 1`.
- **`run_eval`**:
  - Counts `n_obj_bad`.
  - In arm a with targets, accumulates `ForecastMetrics`; under DDP the accumulators are gathered and summed.
  - Adds `fc_nll / fc_minade / fc_minfde`, plus the `fc_*` detail keys.
- **`SUM_KEYS += fc_n_obj, n_obj_bad`**: counts are summed over ranks, not averaged. OFF stats never contain these keys.

### 1.2 Fixes to builder files

**`navsim/agents/para_ssr/ck/temporal/raster.py`** (B2). These are GPU-only fixes, from the smoke profile; the CPU path is byte-identical:
1. **`frenet` station search on CUDA**:
   - `torch.cdist(..., 'donot_use_mm_for_euclid_dist')` took about 15 ms per chunk, which was **46 % of the CUDA time of an arm-a step** [실측 `D3/smoke/bench_gpu_step.json` profile].
   - It is replaced on CUDA by the broadcast `sqrt(dx*dx + dy*dy)`. The distances are **bitwise equal** to cdist, with 0 argmin mismatches over 8.5 M points, tie-prone points included [실측 `D3/integrate/scripts/check_argmin.py`].
2. **Chunk size**: `CHUNK_CAP = {"cpu": 128, "cuda": 1024}` (used when the caller passes `chunk=None`, now the default), and `ELEM_BUDGET["cuda"]` raised from 2^24 to 2^26. The GPU raster was kernel-launch bound.
   - On CUDA, `chunk=128` vs the new default differs by at most 3.6e-7 (atomic `scatter_add` order) [실측].

GPU vs CPU agreement on 2 real 8-token batches [실측 `D3/integrate/scripts/gpu_cpu_equiv.py`]:
- Gap scalars: max |Δ| 9.5e-7.
- Raster: max |Δ| 1.3e-2, but only for rare points.
  - The cause is the pinned keep rule `|d| − e_d ≤ LAT_DROP`. It is discontinuous: at the cut, P_lat is about 0.5·exp(−(10 − H)/b), around 0.01 for CA b ≈ 2.8 m.
  - CPU/GPU rounding moves points across that cut. This is inherent to the SPEC formula and not caused by these fixes.

**`tools/ck/v3/infer_val_v3.py`, `decide_step1.py`** (B4): the output-path guard compared `out.resolve()` with the unresolved D3. D3 sits behind a symlink (`/home/external-user/ssd -> /workspace/yongjae/ssd`), so **every D3 output, including the default `D3/infer/<run>`, was refused**. The guard now compares resolved paths. B4's tests used /tmp, which is why they missed it.

### 1.3 New files
| file | purpose |
|---|---|
| `tools/ck/v3/__init__.py` | package marker (a B0 item) |
| `tools/ck/v3/off_equiv_probe.py`, `off_equiv_compare.py` | OFF-equivalence probe and its bitwise comparer (§2) |
| `tools/ck/v3/run_step1.sh` | Step-1 launcher (§7). It refuses to start unless all of these hold: `V3_USER_APPROVED=1`, GPUs 0 and 1 below 1 GB, complete packs, and `D3/ca_scales.json` present. It writes pid and pgid files, and its arguments come from `arms.py cli` |
| `tools/ck/v3/smoke_v3.sh`, `smoke_summary.py` | smoke runner: modes `cpu`, `gpu0`, `gpu01`; capped by timeout |
| `tools/ck/v3/bench_gpu_step.py` | GPU step breakdown on one real batch, with an optional profiler |
| `tests/test_v3_train.py` (13), `tests/test_v3_integration_off.py` (1) | pipeline and OFF tests (§3) |
| `v2_source` | symlink to `/workspace/yongjae/SSR-ck/v2_source` (a B0 item; it shows as untracked in git) |
| `D3/ref_8ae8b1d/` | reference tree: `git archive 8ae8b1d` plus the Phase-0 pristine `tools/ck/data` and `future_teacher.py`. `diff -r` against the SSR-ck2 data dir is empty |

## 2. OFF is bit-identical to CK2 @ 8ae8b1d [실측]

`off_equiv_probe.py` runs the same probe code in two fresh `python -B` interpreters, with CPU, 1 thread and PYTHONPATH unset. One run uses `D3/ref_8ae8b1d`, the other this worktree. `off_equiv_compare.py` then checks:
- **1,151 arrays, all bitwise identical**:
  - the initial state_dict;
  - the CPU forward (cand + extra + decode + rescore_corr), at init and with the strictly loaded `ck2T10/ckpt_last.pt`;
  - `train_ck2.train` on synthetic data for 1 and for 3 steps: 45 step stats per step, the final weights, the AdamW state, and the val metrics. This covers "one optimizer step equal".
- **736 sha256 of collated batch keys, all identical**: 64 navtrain_val tokens × 2 epochs, for CK2Dataset (train style, gt) and CKDataset (r34 style, BEV T).
- **Facts**:
  - param_count 3,516,178 and trainable 3,391,243 (ck2T10);
  - the state_dict keys;
  - 1 optimiser group;
  - config.json only **gains** 15 keys (`temporal`, `privileged`, two param counts, 11 raw flags).

Result: `D3/integrate/off_equiv/compare.json` → `"identical": true`. The same check runs as `tests/test_v3_integration_off.py`.

`test_v3_model_patch.py::test_off_identical_to_8ae8b1d` is trivial now that model.py is patched (M0 is the patched module). The real OFF reference is the test above.

## 3. Tests [실측]

The CPU environment for every run:
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$V3 nice -n 10 /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider`

- **Existing CK2 tests** (the list is in `D3/integrate/logs/existing_tests.txt`): train_ck2, ck_data, ck_core, ck_core_losses, ck_pipeline, ck_e2e_model, ck2_e2e_{teacher, kd_calib, data, config, sampling, launch, teacher_summary_ep, ep_target}, ck_e2e_launch.
  - v3 tree: **238 passed, 1 failed** (`final_existing.log`).
  - Reference tree: **238 passed, 1 failed** (`baseline_ref_existing.log`).
  - The failure is the same in both: `test_ck_e2e_launch.py::test_ck_yaml_matches_contract_and_config` (`kd_ramp_epochs` 0.0 vs contract 3.0). It is a CK-e2e yaml/contract mismatch that predates v3.
- **v3 tests: 117 passed** (`final_v3.log`, 6 min):

  | file | tests |
  |---|---|
  | objects | 15 |
  | temporal_model | 34 |
  | model_patch | 5 |
  | temporal_real | 3 |
  | loss_metrics | 21 |
  | arms | 6 (the trainer-parser test that used to be skipped now runs) |
  | decide | 19 |
  | train | 13 |
  | integration_off | 1 |

- **What `test_v3_train.py` covers**:
  - flag resolution and its guards (n_modes, `_PRIV`, CA scales, arm M, v2det);
  - the obj_ok threshold;
  - param groups and dummy_backward;
  - **model-input isolation** in arms a and b: perturbing fc targets, lead flags, anc_*/var_* metadata, labels, gt_traj or ref_* leaves the outputs bitwise unchanged, while moving obj_in does change them;
  - 3 CPU steps for each of arms a, b and d: config, stats, privileged val rows, strict load_ck (d needs allow_privileged), 2 or 1 optimiser groups, ca_b restored;
  - the resume guard: lambda_fc, obj_drop, forecaster and OFF→ON are each refused; an old config without the key resumes as OFF;
  - **2-process gloo DDP == 1 process for arm a**: losses within rel 1e-4, grad norms within 1e-3, weights within 1e-5.
    - Excluded from the weight check: `temporal.forecaster.dec_pi.bias`. That bias is shared by all mode logits, so softmax shift invariance makes its true gradient exactly 0, and Adam turns noise into ±lr steps (1.9e-5). It is harmless; B2 could drop the bias.
  - **On real navtrain_val**:
    - the batch streams of arms a and b are byte-identical, augmentation included, over 16 tokens × 2 epochs;
    - every CK2 key equals the OFF dataset;
    - augmentation never touches the targets and is keyed by the epoch;
    - OFF and aug=False never call the augmentation;
  - CKDataset guards and the oracle's privileged keys; the 2,048-row smoke pack refuses the full train rows.

## 4. CPU smoke: `smoke_v3.sh cpu` [실측 `D3/smoke/cpu/`]

Setup: real navtrain_train (smoke pack, §7) and navtrain_val, the full ck2T10 recipe flags, `--limit-tokens 16 --max-steps 3 --tokens-per-batch 2 --val-limit 8`, 1 thread. 4 runs in parallel, 61 s wall.

| arm | params (trainable) | s/step | loss step 1 / 3 | fc_loss | skipped | val |
|---|---|---|---|---|---|---|
| off | 3,516,178 (3,391,243) | 8.60 | 0.6847 / 0.5572 | – | 0 | finite |
| a learned | 4,226,151 (4,101,216) | 9.20 | 0.6847 / 4.78 (critic part 0.5557) | 4.22 | 0 | finite, with fc_nll / minADE / minFDE |
| b CA | 3,754,066 (3,629,131) | 9.22 | 0.6847 / 0.5557 | – | 0 | finite |
| d oracle_PRIV | 3,754,066 | 10.14 | 0.6847 / 0.5556 | – | 0 | finite, every row `privileged: 1` |

- **Step 1**: the critic loss is identical in all arms (branch no-op at init). The grad norm differs only by the branch's own gradients (0.4439 vs about 0.4525).
- **Inference path**: `infer_val_v3.py --device cpu --limit 16` on `smoke_cpu_a` and `smoke_cpu_b` writes every SPEC §6 file plus `fc_metrics.json` (arm and CA reference). obj_bad was 0 [실측 `D3/smoke/infer/`].

## 5. GPU smoke: `smoke_v3.sh gpu0 a off` [실측 `D3/smoke/gpu0/summary.json`]

Setup: GPU 0, world 1, 8 tokens per step (= 8 tokens/rank, as in Step 1), 60 steps, 512 train tokens, val-limit 64, workers 8. Each arm took under 60 s; GPU 0 was used for about 7 min in total over separate runs of under 2 min each, benchmarks included; GPUs 0 and 1 are idle again.

| | off | arm a, before the raster fixes | **arm a, after** | budget |
|---|---|---|---|---|
| peak gpu_mem_gb (max_memory_allocated) | 5.74 | 7.02 | **7.02 (+1.28)** | ≤ +3 → **pass** |
| median s/step (cand_per_s windows 2–6) | 0.166 | 0.361 (×2.22) | **0.248 (×1.50)** | ≤ ×1.25 → **fail** |
| skipped steps (fp16 GradScaler back-off) | 2 | 1 | 1 | ck2T10 itself had 3 by step 100 → OK |
| val metrics finite (4 splits) | yes | yes | yes | pass |
| val fc after 60 steps | – | – | nll 1.22, minADE 1.25 m, minFDE 2.91 m | – |

`bench_gpu_step.py` (one real 8-token batch, compute only: forward + loss + scaled backward; 10 iterations) [실측]:

| | off | a | b | raster + gap per step (a / b) |
|---|---|---|---|---|
| before | 0.152 s | 0.304 (×2.00) | 0.265 (×1.75) | 133 / 77 ms |
| + cdist fix | 0.148 | 0.226 (×1.52) | 0.261 | 49 / 66 ms |
| + chunk fix | 0.151 | **0.207 (×1.37)** | **0.213 (×1.41)** | 32 / 30 ms |

- **What is left of the arm-a overhead** (about 56 ms):
  - raster + gap: 32 ms (raster 20, gap 13). It is launch-bound: cand32 (256 candidates) costs about the same as var80 (640).
  - st_enc + GroupNorm forward and backward: about 18 ms.
  - forecaster forward and backward: about 10 ms.
- **The train loop shows ×1.50 rather than ×1.37** [추정]: it adds host syncs from the stat `float()` calls, the second clip group, and the data pipeline.
- **Projected Step-1 time per arm**: ck2T10 took 2.8 h, so about 3.9–4.2 h for each of arms a and b [추정]. That is over the 3.5 h / +25 % budget.
- **Options for the main session** (none built):
  1. Accept it: a and b in sequence take about 8 h instead of 7 h.
  2. Rasterise cand and extra in one call (saves about 11 ms of fixed launch overhead).
  3. Statics fast path (B2 reserve: one time sample, broadcast over the 21 bins).
  4. A bounding-box prefilter before Frenet.

## 6. Commands

```bash
V3=/workspace/yongjae/SSR-v3; D3=/home/external-user/ssd/yongjae_refiner/ck/v3; cd $V3
E="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$V3"
# tests
env $E nice -n 10 /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider tests/test_v3_*.py
env $E nice -n 10 /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider $(cat $D3/integrate/logs/existing_tests.txt)
# OFF probe by hand (the test does the same)
for t in ref v3; do R=$([ $t = ref ] && echo $D3/ref_8ae8b1d || echo $V3); env -u PYTHONPATH CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 /venv/ssr/bin/python -B tools/ck/v3/off_equiv_probe.py --repo $R --out $D3/integrate/off_equiv/$t.npz \
  --work $D3/integrate/off_equiv/${t}_work; done
/venv/ssr/bin/python -B tools/ck/v3/off_equiv_compare.py $D3/integrate/off_equiv/{ref,v3}.npz
# smoke inputs (done): 2,048-row train pack + smoke-only CA scales; val pack linked
env $E nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/build_obj_pack.py --split navtrain_train --limit 2048 --out-root $D3/smoke/objpack
env $E nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/fit_ca_scales.py --pack $D3/smoke/objpack/bevfusion/navtrain_train --out $D3/smoke/ca_scales_SMOKE_limit2048.json
# smokes
bash tools/ck/v3/smoke_v3.sh cpu; bash tools/ck/v3/smoke_v3.sh gpu0 a off      # gpu01 (spec 5.4) not run
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$V3 /venv/ssr/bin/python -B tools/ck/v3/bench_gpu_step.py \
  --pack-root $D3/smoke/objpack --ca-scales $D3/smoke/ca_scales_SMOKE_limit2048.json --out $D3/smoke/bench_gpu_step_fix2.json
```

## 7. What Step 1 still needs (main session; training only with the user's approval)

1. **K_OBJ decision** (B1, `BUILD_data.md` §6):
   - At K_OBJ = 64, 2.62 % of train frames are truncated (limit 1 %), so the train pack stayed `failed_truncation` and `D3/ca_scales.json` was not fitted.
   - B1 recommends K_OBJ = 96 (0.58 % of train frames truncated). After changing `config.K_OBJ`: rebuild the 3 packs, run the verifies, run `fit_ca_scales.py`, then rerun `tests/test_v3_*.py` and the smoke.
   - The CPU smokes here used the K = 64 smoke pack. At K = 96, the trim to the batch maximum keeps typical batches the same; about 24 % of batches get wider [추정 B1]. The GPU smoke should be repeated then.
2. **Step time** (§5): accept ×1.4–1.5, or ask for one of the listed optimisations first.
3. **Launch**, one arm at a time:
   - `V3_USER_APPROVED=1 bash tools/ck/v3/run_step1.sh a`, then `b` (`d` is optional).
   - Arguments: `arms.py cli`, i.e. the ck2T10 recipe + `--temporal 1 --t-forecaster ... --t-modes ...`.
4. **Evaluation**: the commands are in `BUILD_loss_eval.md` §5.
   - It starts with `infer_val_v3.py --run .../ck2/train/ck2T10` on GPU 0. This fresh arm-(c) r34 dump is a blocker for `decide_step1.py`.
   - The output-path guard fix (§1.2) is needed for any D3 output.
5. **Still to confirm** (from the builders): G4 operationalisation; the per-object π expectation in gap features 9–17; the CA knot representation (f1 bound relaxed to 95 %).
