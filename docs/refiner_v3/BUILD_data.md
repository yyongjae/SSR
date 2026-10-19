# refiner-v3 B1 (data builder): build record

Written 2026-10-11 15:35 KST by the B1 builder. It covers SPEC sections 3, 4.2 and 5.1 and the B1 data jobs of section 9.
Tags: [실측] = measured (the file or command is named), [추정] = estimate, [미확인] = not checked.
`D3 = /home/external-user/ssd/yongjae_refiner/ck/v3`. Every job ran on CPU (`CUDA_VISIBLE_DEVICES=""`,
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `nice -n 10`, at most 16 workers; the longest job took 70 s). No GPU, no training,
no git commit. Nothing was written into `SSR` or `SSR-ck2` [실측: `find -newer SPEC.md` is empty in both].

## 0. Status

| item | status |
|---|---|
| code (objects.py, obj_pack.py, build / fit / verify tools) | done |
| tests `tests/test_v3_objects.py` | **15 passed**, 0 failed, 0 skipped (7.3 s) [실측] |
| pack `navtrain_val` (18,179 rows) | **complete**, verified [실측] |
| pack `navtest` (12,146 rows, no targets) | **complete**, verified [실측] |
| pack `navtrain_train` (85,109 rows) | **failed_truncation**: 2.62 % of frames are truncated at K_OBJ = 64, and the SPEC 3.4 limit is 1 %. The arrays are built and verified, but they are left as `*.npy.tmp` files with `build.json` status `failed_truncation` [실측]. **This needs a spec decision (section 6).** |
| `D3/ca_scales.json` | **not written**. It is fitted on the complete navtrain_train pack only, so it waits on the same decision. The tool is ready and was tested on a small dev pack. |

## 1. Phase 0 (done here because B0 had not started)

When B1 started (15:00 KST), `tools/ck/data/`, `refiner/future_teacher.py` and `D3` were all missing. B1 needs all of
them, so B1 ran only the idempotent copy steps of SPEC section 2. The commands were exactly those of the SPEC:
- `cp -a` of `SSR-ck2/tools/ck/data/*.py *.sh` into `SSR-v3/tools/ck/data/` and into `D3/phase0/pristine/tools/ck/data/`.
- `cp -a` of `SSR/navsim/agents/para_ssr/refiner/future_teacher.py` into the v3 tree and into `D3/phase0/pristine/refiner/`.
- `diff -r -x __pycache__` of the data directory was empty, and the `future_teacher.py` diff was empty [실측].
- `!tools/ck/data/` was appended to `.gitignore`.

B1 did **not** do these B0 steps: the symlinks (`v2_source`, `D3/gt_objects/*`), `phase0_copies.json`,
`make_ref_tree.sh`, `D3/ref_8ae8b1d`, and `tools/ck/v3/__init__.py`. B1 needs none of them: the GT directories fall back
to their source paths. Two cautions for whoever runs B0:
1. B0's `diff -r` of `tools/ck/data` will now also print `Only in SSR-v3/tools/ck/data: obj_pack.py`. This is the B1
   file, and it is expected.
2. Do not rerun the bare `ln -s` lines once the links exist. With GNU `ln`, a second `ln -s` onto an existing symlink to
   a directory creates a nested link inside the target, that is inside `SSR-ck` or `SSR/report/53...`. Guard each
   `ln -s` with `[ -e ... ] ||`.

## 2. Files (all new)

| file | content |
|---|---|
| `navsim/agents/para_ssr/ck/temporal/objects.py` | `CausalityError`, `canonical_rows`, `teacher_frame_rows`, `CausalPastIndex` (`t0`, `past`, `pose`; `bypos[(log, pos - j)]` only), `BevfusionObjectSource.frames`, stubs `V2DetObjectSource` / `GtObjectSource` (both raise `NotImplementedError`), `match_targets`, `ca_history` / `ca_positions` / `ca_forecast_np`. No torch code: torch is imported only indirectly, by `future_teacher` -> `adapters`. All constants come from `ck/temporal/config.py` (B2). |
| `tools/ck/data/obj_pack.py` | `PACK_VERSION = "v3_objpack_1"`, `pack_dir`, `packed_tokens`, `ObjPack(split, source, root, targets, expect_tokens)` (checks version, status, file shapes and dtypes, and token order; lazy mmap per pid; picklable), `.item(row)`, `.has_targets`, `augment_objects`. |
| `tools/ck/v3/build_obj_pack.py` | The pack builder: fork pool, chunks of 64 tokens, atomic part files under `P/_parts`, resumable (a part is reused only if its tokens and layout key match), final arrays renamed from `*.npy.tmp`, `build.json` written last. |
| `tools/ck/v3/fit_ca_scales.py` | Writes `D3/ca_scales.json` (SPEC 3.6). It refuses any split other than navtrain_train (`--allow-split` is for tests), refuses an incomplete pack, and never overwrites. |
| `tools/ck/v3/verify_obj_pack.py` | Independent re-checks of a pack. It writes `<pack>/verify.json`, or `verify_incomplete.json` with `--incomplete`. |
| `tests/test_v3_objects.py` | The 15 tests of section 5. |

## 3. Commands

```bash
cd /workspace/yongjae/SSR-v3
ENV="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/yongjae/SSR-v3"
env $ENV /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v3_objects.py
env $ENV nice -n 10 /venv/ssr/bin/python tools/ck/v3/build_obj_pack.py --split navtrain_val   --workers 16
env $ENV nice -n 10 /venv/ssr/bin/python tools/ck/v3/build_obj_pack.py --split navtest        --workers 16 --no-targets
env $ENV nice -n 10 /venv/ssr/bin/python tools/ck/v3/build_obj_pack.py --split navtrain_train --workers 16   # -> exit 2, failed_truncation
env $ENV nice -n 10 /venv/ssr/bin/python tools/ck/v3/verify_obj_pack.py --split navtrain_val --n 1000
env $ENV nice -n 10 /venv/ssr/bin/python tools/ck/v3/verify_obj_pack.py --split navtest --n 1000
env $ENV nice -n 10 /venv/ssr/bin/python tools/ck/v3/verify_obj_pack.py --split navtrain_train --n 1000 --incomplete
# after the section 6 decision:
env $ENV nice -n 10 /venv/ssr/bin/python tools/ck/v3/fit_ca_scales.py   # --pack D3/objpack/bevfusion/navtrain_train --out D3/ca_scales.json
```
- Logs are in `D3/logs/objpack_<split>.log` and `D3/logs/verify_<split>*.log`.
- Each pack's `build.json` records the version, the thresholds, K_OBJ, the sha256 of both cache manifests, the frames
  parquet and the GT `index.parquet`, the sha256 of every code file (`objects.py` c8685660…, `build_obj_pack.py`
  0718d602…, `obj_pack.py` 4fad9c69…, `config.py` 4b20990b…), the git head 8ae8b1d (the code itself is uncommitted),
  every count, the KST start and end times, and the sha256 of every output file.
- Wall times [실측]: val 13.6 s, navtest 3.9 s, train 46.6 s.

## 4. Data produced (`D3/objpack/bevfusion/<split>/`) [실측 build.json, verify.json]

| | navtrain_val | navtest | navtrain_train (tmp) |
|---|---|---|---|
| rows (= packed order) | 18,179 | 12,146 | 85,109 |
| status | complete | complete | failed_truncation |
| obj_ok False / token errors / non-finite boxes | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| t-0.5 valid | 91.56 % (cur 15,134, fut 1,511) | 90.77 % (10,045 / 980) | 90.33 % (70,139 / 6,740) |
| t-1.0 valid | 84.50 % (12,883 / 2,479) | 83.22 % (8,521 / 1,587) | 82.05 % (58,935 / 10,898) |
| no past frame | 8.05 % | 8.79 % | 9.25 % |
| all 3 frames valid | 84.12 % | 82.78 % | 81.63 % |
| invalid frames by reason (t-0.5 / t-1.0) | not_in_parquet 1,534 / 2,799; dt_range 0 / 18 | 1,121 / 2,026; 0 / 12 | 8,230 / 15,167; 0 / 109 |
| no_npz frames | 0 | 0 | 0 |
| dt median, t-0.5 / t-1.0 (s) | 0.49998 / 0.99997 | 0.49999 / 0.99999 | 0.50000 / 1.00000 |
| boxes above threshold, mean (t0, t-0.5, t-1.0) | 8.9 / 9.5 / 9.3 | 10.8 / 11.6 / 11.4 | 19.1 / 20.5 / 20.1 |
| frames truncated at 64 | 168 (0.33 %) | 192 (0.58 %) | **6,085 (2.62 %)** |
| agent t0 slots matched to GT (2 m Hungarian) | **99.384 %** (87,147 / 87,687) | — | 99.394 % (741,399 / 745,917) |
| vehicle <-> VRU cross matches | 6 | — | 100 |
| GT target OBS fraction, k = 1 / 5 / 10 | .9938 / .9601 / .8880 | — | .9939 / .9604 / .8950 |
| (object, step) supervised | 94.92 % | — | 95.07 % |
| size | 383 MB | 157 MB | 3.6 GB (tmp arrays + parts) |

Consistency with earlier measurements [실측]:
- Past-frame availability equals `a2_availability.json` exactly for t-0.5 (cur / fut counts are identical). At t-1.0 the
  pack has fewer frames than a2 by exactly the `dt_range` frames: val 18, navtest 12, train 109. a2's
  `n_dt_irregular` counts are 18 / 12 / 109.
- A t-0.5 frame can be 1.0 s old at a log gap (val dt max 1.0005 s). It is still inside DT_RANGE, so it stays valid,
  and its age column carries the real dt (SPEC 3.3).
- GT target coverage per k on val equals f4 `gt_target_by_k` to within 0.0002 at every k. The match rate is
  99.384 %, against f4's 99.382 % [`A_full16/out/f4_target_coverage_gt_val.json`].
- CA position error (`ca_forecast_np` on the clean val pack, matched agents at OBS steps) has medians by k of
  .097 .159 .223 .288 .347 .403 .453 .500 .544 .587 m. f4 (a3 CA on uncapped boxes) has
  .097 .159 .223 .288 .348 .403 .454 .501 .545 .587. Means agree to within 0.003 m.

## 5. Checks

### Unit and real-data tests (`tests/test_v3_objects.py`, 15 passed) [실측]

**Synthetic tests**
- Box conversion known answer: heading = wrap(−yaw − π/2), L = dy, W = dx, and the position / velocity chain through
  pose_k -> pose0. With pose_k = pose0 the chain is the identity.
- Score threshold, top-64 cap, and stable tie order by proposal index; non-finite boxes are dropped and counted.
- Causal index: a planted pos − 1 frame with ts > ts0 raises `CausalityError`. A recording `bypos` proves no key with
  pos ≥ pos0 is ever read, and the pos + 1 frame is never loaded. Also covered: `dt_range`, `not_in_parquet`, j ≤ 0
  raises, the cache read order is current first and then `_future`, and the age column equals dt.
- `match_targets` known answers: OBS / ABSENT / BEYOND (log ends early), statics get no target, the 2 m gate, and a
  vehicle <-> VRU cross match is counted.
- CA synthetic: deceleration with tstop, the 1.5 m history gate, a = 0 for |v0| ≤ 0.5, statics held.
- Augmentation: the same token / epoch / seed gives the same bytes; the expected values follow the pinned draw order;
  the input is unmodified; the targets are untouched; with cfg off the item object is returned and no RNG is drawn
  (`token_rng` is monkeypatched to raise). The rates are 0.1 / 0.05 / σ 0.3 over 400 tokens.
- `ObjPack` refuses a wrong token order, a wrong version, an incomplete status, and targets on a no-target pack.

**Real-data tests**
- 32 val tokens through the builder (in `D3/test_tmp`, removed afterwards):
  - shapes, dtypes and token order are correct;
  - refuses to overwrite a complete pack;
  - a resume from the parts gives identical bytes;
  - the t0 rows equal the rows rebuilt from `data.TeacherCache.load_det` to within 1e-4;
  - `frame_dt` equals the parquet timestamps.
- Matched vehicles at t0, 60 val tokens, 136 boxes:
  - The heading error median is **0.012 rad** (limit < 0.02).
  - The signed median of L − L_gt is **−0.010 m**, and of W − W_gt +0.003 m.
  - The absolute median of |L − L_gt| is 0.086 m. This is detector size noise, not a conversion error (deviation 2).
  - Swapping the convention (dx as L) gives errors of 2.6 m, so L = dy is correct.
- Parked vehicles: t-1.0 detections moved into the t0 N frame land on the GT t0 boxes with median **0.111 m** (352
  vehicles; the critic measured 0.10 m; limit < 0.2).
- `ca_forecast_np` equals a3 `Worker.history` + `world('CA')` on 442 agents of 50 val tokens (376 with history; tokens
  with n_raw ≤ 64 and no dt_range). Max |Δ| is **4.3e-10 m** (limit < 1e-9); `acc` and `has` are identical.
- Full val pack match rate is 99.384 %, within 0.5 pp of 99.4 %.

### Pack re-checks (`verify_obj_pack.py`, 1,000 seeded rows per split; all `ok: true`) [실측]

| check | val | navtest | train (tmp) |
|---|---|---|---|
| t0 rows vs `data.TeacherCache.load_det`, max abs diff (slot counts all equal) | 5.8e-11 | 4.7e-10 | 2.3e-10 |
| past frames checked: token = `past(token, f)`, source cache correct, rows rebuilt from the raw npz | 1,759, max diff 0 | 1,753, 0 | 1,746, 0 |
| past dt vs parquet, max error (s) | 6.0e-8 | 6.0e-8 | 5.9e-8 |
| invalid frames really missing or out of range | 241 / 241 | 247 / 247 | 254 / 254 |
| targets recomputed from the clean t0 rows + GT file: bitwise equal; OBS steps equal `gt_future.query` of the track | 1,000 / 1,000, max err 0 | — | 1,000 / 1,000 (GT = `/home/external-user/ssd/yongjae_refiner/objects/navtrain`), 0 |

## 6. Decision needed: K_OBJ (SPEC 3.4 "raised in a spec change, not silently")

The train pack breaks the 1 % truncation rule. Val and navtest do not. Fraction of valid frames with more boxes above
the threshold than K [실측: train from `n_raw.npy.tmp`; val and navtest from all frames,
`D3/scratch/count_above.py`]:

| K | train | val | navtest |
|---|---|---|---|
| 64 (now) | **2.62 %** (tokens with ≥ 1 truncated frame: 3.3 %) | 0.34 % | 0.58 % |
| 80 | 1.17 % | 0.06 % | 0.14 % |
| **96** | **0.58 %** | 0.00 % | 0.003 % |
| 128 | 0.14 % | 0 | 0 |

- **What the cap drops in train (6,085 truncated frames)** [실측 `D3/scratch/trunc_classes.py`]:

  | | count |
  |---|---|
  | pedestrian | 67,577 |
  | generic_object | 48,723 |
  | vehicle | 6,509 |
  | barrier | 4,131 |
  | other classes | 1,954 |
  | **share that are agents** | **58 %** |
  | frames losing ≥ 1 agent | 5,403 |

  Median distance of the dropped boxes is 26 m. These are crowd scenes, mostly in the Las Vegas strip [추정 from the
  class mix]. Val looks the same but is 30 times rarer.
- **Recommendation [추정]**: raise `config.K_OBJ` to 96. It is the smallest value that meets the 1 % rule on every
  split. The model trims to the batch maximum, so typical batches are unaffected. A batch is padded to more than 64
  slots only when a token in it is truncated at 64. At 8 tokens per rank that is about 1 − 0.967^8 ≈ 24 % of train
  batches [추정]. B2 / Integrate should check the raster cost of those batches in the smoke budget.
- **Alternative**: keep 64 and accept 2.6 % as an explicit, recorded decision. Run
  `build_obj_pack.py --split navtrain_train --max-trunc-frac 0.03`. It reuses the existing parts and finishes in about
  1 min.
- **Code readiness**: the B1 code is K-agnostic. Every shape comes from `config.K_OBJ`, and the tests use `TC.K_OBJ`.
  A sanity build with K_OBJ patched to 96 (2,000 train rows) completed, and `ObjPack` served [3, 96, 17] items
  [실측]. Part files carry a layout key, so a resume never mixes K values.

**After a K change**:
- Delete the three pack directories `D3/objpack/bevfusion/{navtrain_train,navtrain_val,navtest}`. They are B1's own
  outputs. `ObjPack` would refuse them anyway, because of the shape check.
- Rerun the three builds (about 1.1 min), the three verifies, `fit_ca_scales.py` (navtrain_train only) and the tests.

## 7. Deviations from the SPEC and task text

1. **Output root.** The task text said `D3/data/`. The packs are in `D3/objpack/bevfusion/<split>/`, the SPEC 3.4 path
   that `config.PACK_ROOT` (B2) points to.
2. **Size test.** The test "median |L − L_gt| < 0.05 m" was changed to a signed median below 0.05 m, plus
   absolute median < 0.15 m and a check that the swapped convention is off by more than 1 m. The critic's 0.003 m was
   a signed median. The absolute median is 0.086 m for vehicles and 0.048 m for all agents, which is detector size
   noise [실측].
3. **Matching formulation.** `match_targets` uses the f3 / f4 formulation: every valid t0 slot against every GT track
   that is OBS at t = 0, with an agent / static group gate. Targets are kept only for agent slots, so statics give
   `fc_state` 0 and `fc_track` −1. Hungarian on matrices with big-M blocks is equivalent to agent-only matching except
   for exact ties, and this formulation reproduces f4's 99.38 %. Matching runs on the stored float32 rows, so it can be
   reproduced from the pack alone.
4. **Extra pack files** (audit only, not in `OBJ_INPUT_KEYS`): `frame_why.npy` (0 ok, 1 not_in_parquet, 2 no_npz,
   3 dt_range) and `frame_token.npy` (`<U16`). Also `verify.json` sits next to each pack. `frames()` also returns
   `frame_token`, `n_nonfinite` and `dt_raw`. `teacher_frame_rows(..., return_info=True)` returns
   `(rows, {n_raw, idx, n_nonfinite})`.
5. **Truncation rule** is applied to full splits only. `--limit` builds record the fraction and skip the rule.
   `--max-trunc-frac` is an explicit override that is recorded in `build.json`.
6. **Token failures.** A token whose t0 npz or parquet row is missing would be recorded as `obj_ok` False with its
   error. There were 0 such tokens. A `CausalityError` aborts the build. A missing GT file fails it
   (`failed_gt_missing`).
7. **`ca_forecast_np`** also returns `agent` [K] and accepts an optional `t`. `ca_history` and `ca_positions` are
   exposed separately for B2's torch port.
