# refiner-v3 SPEC: temporal critic (report-51 idea A), Step 1

Status: v1, revised after review (2026-10-11 KST); code-review fixes D1-D8 applied 2026-10-11 evening KST (section 12,
`docs/refiner_v3/REVIEW_FIXES.md`; where a line below and section 12 disagree, section 12 wins). Branch `refiner-v3`, worktree `/workspace/yongjae/SSR-v3`, from ck2 @ 8ae8b1d.
Tags: [실측] = measured (file or command named), [추정] = estimate or judgement, [미확인] = not checked.
"Builder" = an implementing agent of section 9. Every path in this file is absolute unless it is a repo path
(relative to `/workspace/yongjae/SSR-v3`).

## 0. Hard rules (all builders)

- Code only inside `/workspace/yongjae/SSR-v3`. Never modify `/workspace/yongjae/SSR-ck2` (a 53-h no-KD run reads its
  code) or `/workspace/yongjae/SSR`. Read them freely. Never run Python from them without `-B` /
  `PYTHONDONTWRITEBYTECODE=1` (that would write `__pycache__` into them).
- New data/outputs only under `D3 = /home/external-user/ssd/yongjae_refiner/ck/v3/`. Never modify existing data files.
- GPUs 2, 3 busy, GPUs 4-7 not ours: never. GPU 0/1 only for smoke tests < 15 min, only in the Integrate / Fix phases.
  Default `CUDA_VISIBLE_DEVICES=""`.
- CPU jobs: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10`, at most 16 worker processes, each job < 60 min.
  No `pkill -f` / `pgrep -f`.
- No training run (smoke tests of a few steps are fine). No git commit / push / stash.
- Python: `/venv/ssr/bin/python`, `PYTHONPATH=/workspace/yongjae/SSR-v3`. Tests: `-m pytest -q -p no:cacheprovider`.
- Not allowed anywhere: a gamma conservativeness exponent, a rule-based decel gate, ranking losses (rejected earlier).

## 1. What Step 1 tests

The DET critic (CK2 arm T) reads one t0 BEVFusion BEV. On navtrain_val r34 top-16, stacking analytic conflict features on
the frozen ck2T logits gives CV +0.14 (A_novelty) / +0.22 (full16), history constant-acceleration (CA) +0.375, and a
perfect forecast of the t0 objects (FUTt) +0.80. A learned forecaster can therefore add at most about +0.42 (GBM) /
+0.34 (logistic) over CA [실측 `SSR/report/53_eval_and_plugin_evidence/A_full16/notes.md` §0, §2].

Step 1 adds a temporal branch to CKNet and retrains the DET teacher with the ck2T10 recipe in two main arms that differ
ONLY in the forecaster: (a) learned, (b) analytic CA. Decision = (a) - (b). Section 8 has the GO rule.

User requirements (fixed):
1. Teacher-agnostic: the branch is a shared CKNet component. Its objects come from a configurable source:
   `bevfusion` (DET teacher, Step 1), `v2det` (student, later), `gt` (oracle ablation only, later). The MAP teacher
   has it off.
2. Causal inputs only: perception outputs at t-1.0, t-0.5, t0. Never GT at test time. Never future frames as inputs.
3. Future data only as training targets: GT future tracks 0.5-5.0 s (masked where missing). Teacher future
   detections as an alternative target are deferred (GT target is as good where both exist: GTt - FUTt = +0.01
   [실측 A_full16 notes §5]).
4. Default OFF = bit-identical to CK2 @ 8ae8b1d (section 5.6).
5. No gamma exponent, no decel gate, no ranking loss.

## 2. Phase 0: setup (builder B0, serial, before everything)

The worktree lacks code it needs [실측 2026-10-11]:
- `tools/ck/data/` is caught by `.gitignore` line `data/`, so it is absent. `tools/ck/train_ck2.py` (lines 333, 334,
  362, 400), `tools/ck/train_ck.py` (176, 196) and 8 tests import it.
- `navsim/agents/para_ssr/refiner/future_teacher.py` exists only untracked in `/workspace/yongjae/SSR`.
- `v2_source` (symlink in SSR-ck2 -> `/workspace/yongjae/SSR-ck/v2_source`) is absent.

Steps (exact):
```bash
set -e
V3=/workspace/yongjae/SSR-v3; CK2=/workspace/yongjae/SSR-ck2; D3=/home/external-user/ssd/yongjae_refiner/ck/v3
mkdir -p $D3/phase0/pristine/tools/ck/data $D3/phase0/pristine/refiner $D3/gt_objects
mkdir -p $V3/tools/ck/data
cp -a $CK2/tools/ck/data/*.py $CK2/tools/ck/data/*.sh $V3/tools/ck/data/
cp -a $CK2/tools/ck/data/*.py $CK2/tools/ck/data/*.sh $D3/phase0/pristine/tools/ck/data/
cp -a /workspace/yongjae/SSR/navsim/agents/para_ssr/refiner/future_teacher.py $V3/navsim/agents/para_ssr/refiner/
cp -a /workspace/yongjae/SSR/navsim/agents/para_ssr/refiner/future_teacher.py $D3/phase0/pristine/refiner/
ln -s /workspace/yongjae/SSR-ck/v2_source $V3/v2_source
ln -s /workspace/yongjae/SSR/report/53_eval_and_plugin_evidence/A_full16/gt_objects/navtrain_val $D3/gt_objects/navtrain_val
ln -s /home/external-user/ssd/yongjae_refiner/objects/navtrain $D3/gt_objects/navtrain_train
diff -r -x __pycache__ $CK2/tools/ck/data $V3/tools/ck/data        # must print nothing
diff /workspace/yongjae/SSR/navsim/agents/para_ssr/refiner/future_teacher.py $V3/navsim/agents/para_ssr/refiner/future_teacher.py
```
- Append to `$V3/.gitignore` (do not commit): `!tools/ck/data/`.
- Write `$D3/phase0/phase0_copies.json`: for each copied file `{src, dst, pristine, sha256, bytes, mtime_src}` plus
  `ck2_head` (`git -C $CK2 rev-parse HEAD`), `v3_head`, KST time.
- Build the 8ae8b1d reference tree for OFF-equivalence tests with `tools/ck/v3/make_ref_tree.sh` (B0 writes it):
  `REF=$D3/ref_8ae8b1d`; `git -C $V3 archive 8ae8b1d | tar -x -C $REF`; copy the pristine files above into it
  (verify sha256 against `phase0_copies.json`); `ln -s /workspace/yongjae/SSR-ck/v2_source $REF/v2_source`. The
  reference is run only with `python -B`.
- Acceptance: `diff -r` empty; `PYTHONPATH=$V3 python -B -c "import tools.ck.data.ck2_dataset, tools.ck.train_ck2"`
  works; the existing CPU tests `tests/test_ck2_train.py tests/test_ck_data.py tests/test_ck_core.py` pass in v3
  (record pass / skip counts in `phase0_copies.json`).

## 3. Data contract

### 3.1 BEVFusion box conversion (corrects the task text and the `TeacherCache.load_det` docstring)

Raw `pred_boxes_3d` [200, 9] = (x, y, z_bottom, dx = WIDTH, dy = LENGTH, dz, yaw_mmdet, vx, vy), in the frame's own
ego (rear-axle) axes; velocity is absolute in those axes [실측 critic: 60 val tokens, 514 GT matches; raw yaw as heading
median error 1.48 rad, wrap(-yaw - pi/2) 0.0099 rad; dy vs L median 0.003 m]. Every frame, t0 included, goes through
`future_teacher.teacher_box_to_local(b)` then `future_teacher.local_to_n(local, pose_k, pose0)` with
`pose = (x, y, yaw)` from `/home/external-user/ssd/yongjae_refiner/future_index/future_frames_{train,test}_10f.parquet`.
For t0, `pose_k = pose0` (same chain, no shortcut). `data.py` is not edited (errata live here and in `objects.py`).

### 3.2 Canonical object row (D_OBJ = 17, float32, metric, t0 N frame, unnormalised)

| col | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9..15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | x | y | cos h | sin h | L | W | vx | vy | score | class one-hot (7, BEVFusion labels 0..6) | age_s |

- h = N-frame heading after `local_to_n`; (vx, vy) = N-frame axes (rotated by `local_to_n`); age_s = (ts0 - ts_frame)
  * 1e-6 (0 for t0). Padding rows are all zero with mask False.
- Classes (gt_future.CLASS_NAMES order): 0 vehicle, 1 pedestrian, 2 bicycle, 3 traffic_cone, 4 barrier, 5 czone_sign,
  6 generic_object. Groups: `GROUP_OF_CLASS = (0, 1, 1, 2, 2, 2, 2)` = vehicle / VRU / static. "Agent" = classes 0-2.
- Model-side normalisation (fixed constants only, never statistics): x, y / 32; vx, vy / 15; L / 10; W / 5; age in s;
  cos, sin, score, one-hot as is.

### 3.3 Causal past frames (`CausalPastIndex`)

- Frames per token: f = 0 (t0), 1 (t-0.5), 2 (t-1.0). Past frame f = the frame at log position `pos - f` of the same
  log, looked up as `bypos[(log, pos - f)]` built from the frames parquet columns (token, log, pos, timestamp, x, y,
  yaw, lidar2ego_identity). Never `pos + j`, never the future index json.
- Box source: current-frame cache (`cache_{train,val}_50x100`) first, then the `_future`-named cache (a past frame that
  is not an index key lives only there: val t-0.5 1,511 / 16,645 and t-1.0 2,485 / 15,380 [실측 a2_availability.json]).
  Use two `future_teacher.TeacherDetCache` objects (kind current / future); `data.TeacherCache` keeps its `_future`
  refusal untouched.
- Checks: `ts_past >= ts0` raises `CausalityError` (data bug: never silently used). `dt = (ts0 - ts_past) * 1e-6`
  outside (0.25, 1.25) s marks the frame invalid with reason `dt_range` (counted; the real dt is the age otherwise).
  Missing parquet row -> invalid (`not_in_parquet`); no npz in either cache -> invalid (`no_npz`).
- Score thresholds per frame `SCORE_THR = (0.3, 0.2, 0.2)` (a3 THR / THR_PAST). Keep the top `K_OBJ = 96` (was 64; section 12 D1) per frame by
  score (descending, ties by proposal index). Never sort or truncate by anything GT-derived.
- Coverage to expect: val no past frame 8.1%, no t-1.0 15.4%; navtest t-0.5 90.8%, t-1.0 83.3% [실측 a2].

### 3.4 Object pack (built once on CPU, shared by every arm)

Directory `P = D3/objpack/<source>/<split>/`, source `bevfusion`, split in {navtrain_train, navtrain_val, navtest}.
Row r = packed row r of `/home/external-user/ssd/yongjae_refiner/ck/packed/<split>/tokens.parquet` (85,109 / 18,179 /
12,146 rows).

| file | dtype, shape | content |
|---|---|---|
| tokens.parquet | (token, row) | must equal the packed token order (checked by the reader) |
| obj.npy | f32 [N, 3, K_OBJ = 96, 17] | canonical rows, frames (t0, t-0.5, t-1.0) |
| obj_mask.npy | bool [N, 3, 96] | valid slots |
| frame_valid.npy | bool [N, 3] | frame 0 is True whenever the t0 npz was read |
| frame_dt.npy | f32 [N, 3] | 0, dt1, dt2 seconds; NaN where invalid |
| frame_src.npy | i8 [N, 3] | 0 none, 1 current cache, 2 `_future`-named cache |
| n_raw.npy | i16 [N, 3] | boxes above threshold before the 64 cap |
| obj_ok.npy | bool [N] | False only if the token failed (expected 0 rows) |
| fc_xyh.npy | f32 [N, 96, 10, 3] | GT target (x, y, heading_unwrapped) for t0 slot i at t = 0.5k, k = 1..10 (train, val, navtest since section 12) |
| fc_state.npy | i8 [N, 96, 10] | 0 no target (unmatched / not agent / padding), 1 OBS, 2 matched but ABSENT_OFFICIAL, 3 t > t_avail, 4 FC_SINGLE (section 12) |
| fc_track.npy | i32 [N, 96] | matched GT track row, -1 if none |
| build.json | | version `v3_objpack_1`, source, split, thresholds, K_OBJ, sha256 of both cache manifests, frames parquet, GT index.parquet, builder file; git head; counts (frames invalid by reason, truncated frames, match rate); KST start / end |

Sizes [추정]: train obj 1.11 GB, fc_xyh 0.65 GB. The builder fails if more than 1% of frames are truncated at 64
(then K_OBJ is raised in a spec change, not silently).

### 3.5 Forecast targets (`match_targets`)

- GT objects: train `D3/gt_objects/navtrain_train/<tok>.npz` (85,109), val `D3/gt_objects/navtrain_val/<tok>.npz`
  (18,179); navtest has none. Load with `gt_future.load_objects`.
- Match on the CLEAN t0 pack rows (before any augmentation): t0 agent slots (classes 0-2) vs GT tracks with
  `query(obj, [0])` state OBS and `meta[:, MT_AGENT]`; cost = centre distance if same agent/static group and < 2.0 m,
  else 1e6; `scipy.optimize.linear_sum_assignment`; accept cost < 1e5. Same rule as f3/f4 (99.4% of t0 agent boxes
  matched on val [실측 f4_target_coverage_gt_val.json]); vehicle <-> VRU cross matches are allowed and counted.
- Targets: `boxes, state = gt_future.query(obj, 0.5 * arange(1, 11))` (index-based time); x, y, unwrapped heading.
  `fc_state` = 1 where state == OBS, 3 where t > t_avail (`query(..., return_unknown=True)`), else 2.
- Static classes: no target (state 0); they are held at t0 in every arm. Unmatched detections: no target, still
  rasterised.
- Never select or order objects using GT (the GT file order is by distance to the GT ego path: a leak).

### 3.6 CA rule and CA Laplace scales

CA = `a3_conflict_features.Worker.history` + `world('CA')` exactly, except statics are held (a3 moved them by their
near-zero velocity):
- Objects: t0 slots (score >= 0.3); past boxes of frames 2 then 1 (prefer t-1.0; frame 1 only for objects still
  without history); past boxes all have score >= 0.2 by construction.
- Association: predicted past centre c0 - v0 * dt_f; nearest past box of the same agent/static group within 1.5 m;
  agents only.
- a = ((v0 - v_past) . u0) / dt_f with u0 = v0 / max(|v0|, 1e-6); a = clip(a, -6, 6) if |v0| > 0.5 else 0.
  No history -> a = 0 (constant velocity).
- Position: dist(t) = |v0| te + 0.5 a te^2, te = min(t, tstop), tstop = -|v0| / a if a < -1e-6 else inf;
  c(t) = c0 + u0 dist(t). Heading held (h0). Statics: c(t) = c0.
- Laplace scales: `D3/ca_scales.json` = `{version, split: navtrain_train, b: {vehicle: [10], vru: [10]}, n: {...},
  pack_sha256, builder_sha256}`; b[g][k] = max(0.05, mean over matched agent slots of group g with fc_state == 1 at
  step k of (|r_x| + |r_y|) / 2), r = CA(inputs) - GT [section 12 D2: inputs = the AUGMENTED train stream, epochs 0-9,
  seed 0, as arm (b) sees them; the clean fit is kept as b_clean]. Fitted on navtrain_train only (was: clean pack, no
  augmentation). M = 1, pi = 1.

## 4. Model

### 4.1 Config (`navsim/agents/para_ssr/ck/temporal/config.py`, owner B2; constants used by every builder)

```python
K_OBJ = 96; N_FRAMES = 3; D_OBJ = 17; N_CLS = 7          # was 64 (section 12 D1); PACK_ROOT = D3 / "objpack_k96"
COL = dict(x=0, y=1, cos=2, sin=3, L=4, W=5, vx=6, vy=7, score=8, cls0=9, age=16)
SCORE_THR = (0.3, 0.2, 0.2); DT_RANGE = (0.25, 1.25)
GROUP_OF_CLASS = (0, 1, 1, 2, 2, 2, 2); AGENT_CLASSES = (0, 1, 2)          # 0 vehicle, 1 VRU, 2 static
FC_STEPS = 10; FC_T = tuple(0.5 * k for k in range(1, 11))                 # forecast knots (s)
N_RT = 21; R_T = tuple(0.25 * k for k in range(21))                        # raster time bins 0..5 s
EGO_L, EGO_W, EGO_OFF = 5.176, 2.297, 1.461                                 # Pacifica, centre ahead of rear axle
EGO_REAR, EGO_FRONT = EGO_OFF - EGO_L / 2, EGO_OFF + EGO_L / 2             # -1.127, +4.049 (rear-axle s offsets)
TAPS = (0.0, 0.3, 0.6, 0.9); TAP_T_MAX = 4.0                                # pdm_scorer.py 470-506
TJ = tuple(0.25 * k for k in range(1, 17))                                 # gap-scalar ego times (a3)
ROI = {"bevfusion": (0.0, -32.0, 32.0, 32.0)}                               # x0, y0, x1, y1 (manifest point_cloud_range)
KNOWN_MARGIN = 0.5; LAT_DROP = 10.0
B0 = 0.1; B_MIN, B_MAX = 0.05, 20.0; B_ORACLE = 0.25                        # Laplace scales (m)
NORM = dict(xy=32.0, v=15.0, L=10.0, W=5.0)
TEMPORAL_SEED_OFFSET = 52361; FORECASTER_SEED_OFFSET = 60493
D3 = Path("/home/external-user/ssd/yongjae_refiner/ck/v3")
PACK_ROOT = D3 / "objpack"; CA_SCALES = D3 / "ca_scales.json"
SOURCES = ("bevfusion", "v2det", "gt"); FORECASTERS = ("learned", "ca", "oracle_gt")
OBJ_INPUT_KEYS = ("obj_in", "obj_mask", "frame_valid")                      # causal model inputs
PRIV_INPUT_KEYS = ("priv_fc_xyh", "priv_fc_state")                          # oracle arm only (PRIVILEGED)
TARGET_KEYS = ("fc_xyh", "fc_state")                                        # loss / metrics only, never forward()
FC_NONE, FC_OBS, FC_ABSENT, FC_BEYOND = 0, 1, 2, 3

@dataclass(frozen=True)
class TemporalCfg:
    on: bool = False
    source: str = "bevfusion"
    forecaster: str = "learned"
    n_modes: int = 6
    lambda_fc: float = 1.0
    fc_backprop: bool = False          # default: critic loss does NOT reach the forecaster (section 4.6)
    p_frame_drop: float = 0.1          # per past frame
    p_obj_drop: float = 0.05           # per object slot, every frame
    v_noise: float = 0.3               # m/s, N(0, v_noise^2) per velocity component
    pack_root: str = str(PACK_ROOT)
    ca_scales: str = str(CA_SCALES)
    version: str = "v3_temporal_1"
    @property
    def privileged(self) -> bool: return self.on and (self.forecaster == "oracle_gt" or self.source == "gt")
    def to_dict(self) -> dict          # all fields + "privileged"
    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TemporalCfg"   # None / {} / missing key -> off; unknown key -> ValueError
    def validate(self, arm: str) -> None
        # on and arm != "T" -> ValueError (MAP teacher: off; S arm later)
        # source in ("v2det", "gt") -> NotImplementedError in Step 1
        # learned: 1 <= n_modes <= 8; ca / oracle_gt: n_modes must be 1
```

### 4.2 Object sources (`navsim/agents/para_ssr/ck/temporal/objects.py`, owner B1, numpy only, no torch)

```python
class CausalityError(RuntimeError): ...
def canonical_rows(b_n: np.ndarray, scores, labels, age_s: float) -> np.ndarray   # b_n [n, 9] future_teacher BX order -> [n, 17] f64
def teacher_frame_rows(boxes_raw, scores, labels, pose_k, pose0, age_s, thr) -> np.ndarray      # threshold, top-64, convert
class CausalPastIndex:
    def __init__(self, frames_path: str): ...
    def t0(self, token: str) -> Tuple[str, int, int, Tuple[float, float, float]]               # log, pos, ts_us, pose
    def past(self, token: str, j: int) -> Tuple[Optional[str], float, str]                     # (token | None, dt_s, why)
class BevfusionObjectSource:
    name = "bevfusion"; roi = ROI["bevfusion"]; n_frames = 3
    def __init__(self, subset: str): ...                                                       # 'navtrain' | 'navtest'
    def frames(self, token: str) -> Dict[str, np.ndarray]
        # obj f32 [3, 64, 17], mask bool [3, 64], valid bool [3], dt f32 [3], src i8 [3], n_raw i16 [3], why <U16 [3]
class V2DetObjectSource:   # stub: raises NotImplementedError; docstring records the conversion for later:
                           # 300 boxes, SSR axes, x = y_ssr, y = -x_ssr, heading = -syaw - pi, (vx, vy) = (vy_ssr, -vx_ssr)
                           # (a6_v2_forecast.py 12-15, 58-59); t0 only (frame_valid[1:] False)
class GtObjectSource:      # stub: raises NotImplementedError; PRIVILEGED; needs a causal GT-history build from raw
                           # logs (frames cur-2..cur) under D3, crop to the source ROI, select by score / distance to
                           # the origin only
def match_targets(obj0: np.ndarray, mask0: np.ndarray, gt_obj: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        # -> fc_xyh f32 [64, 10, 3], fc_state i8 [64, 10], fc_track i32 [64]   (section 3.5)
def ca_forecast_np(obj: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray]
        # numpy reference of section 3.6 -> xy f64 [64, 10, 2] at FC_T, acc [64], has [64]; dt from obj[f, :, 16]
```

### 4.3 Forecast generators (`.../ck/temporal/forecast.py`, owner B2, torch)

```python
@dataclass
class Forecast:                       # per token, K = trimmed slot count, M modes
    mu: Tensor        # [T, K, M, 10, 2] positions at FC_T (statics: c0 repeated)
    b: Tensor         # [T, K, M, 10, 2] Laplace scale per axis
    pi: Tensor        # [T, K, M] (sums to 1)
    present: Tensor   # [T, K, M, 10] 1 = present (0 only for oracle ABSENT steps)
    c0: Tensor; v0: Tensor            # [T, K, 2]
    h0: Tensor; L: Tensor; W: Tensor  # [T, K]
    group: Tensor                     # [T, K] long (0 veh, 1 VRU, 2 static)
    valid: Tensor                     # [T, K] bool (t0 slot valid)
    log_b: Optional[Tensor] = None; logit_pi: Optional[Tensor] = None   # learned only (loss)
def trim_objects(obj_in, obj_mask, frame_valid) -> Tuple[Tensor, Tensor, Tensor]
    # K' = 1 + max slot index with any True mask over the batch (>= 1); slices [..., :K', :]
def ca_forecast(obj_in, obj_mask, frame_valid, ca_b: Tensor) -> Forecast          # torch port of 3.6, M = 1, no grad
def oracle_forecast(obj_in, obj_mask, frame_valid, priv_fc_xyh, priv_fc_state, ca_b) -> Forecast
    # M = 1: state 1 -> GT (x, y), b = B_ORACLE; state 2 -> present 0; state 0 / 3 and unmatched -> CA (x, y, b)
class LearnedForecaster(nn.Module):
    def __init__(self, n_modes: int = 6, d: int = 128, n_heads: int = 4, d_ffn: int = 256,
                 n_past_layers: int = 2, n_social_layers: int = 1): ...
    def forward(self, obj_in, obj_mask, frame_valid) -> Forecast                     # fp32, autocast disabled
```

LearnedForecaster (pinned):
- Token embedding: phi(row) (normalised 17-d, section 3.2) -> Linear(17, 128), GELU, Linear(128, 128), plus
  `frame_emb = Embedding(3, 128)`.
- Past attention x 2 (pre-LN, d 128, 4 heads, FFN 256 GELU, dropout 0). Query = t0 token i. Keys / values =
  [learned null key/value] + [t0 self token i] + [every past token j of frames 1, 2 with obj_mask & frame_valid].
  Pair feature r_ij = ((x_j - x_i + vx_i a_j) / 4, (y_j - y_i + vy_i a_j) / 4, (vx_j - vx_i) / 5, (vy_j - vy_i) / 5,
  a_j, 1[agent(i) == agent(j)]) with a_j = age_j (metric inputs, before normalisation); self key r = (0, 0, 0, 0, 0, 1);
  attention bias per head = MLP(6 -> 32 -> 4); value = W_v LN(e_j) + MLP(6 -> 32 -> 128)(r_ij); null key bias 0.
  Masked keys get -inf; the null and self keys are always valid, so no row is fully masked (no NaN).
- Social attention x 1 over valid t0 tokens (+ self), bias MLP(((x_j - x_i) / 32, (y_j - y_i) / 32): 2 -> 32 -> 4).
- Decoder per mode m: z = LN(e_i) + mode_emb[m] (Parameter [M, 128], randn * 0.02); h = GELU(Linear(128, 256)(z));
  out = Linear(256, 40)(h) -> [10, 4] = (dx, dy, log b_x, log b_y); logit_pi = Linear(256, 1)(h).
  mu_k = c0 + v0 * t_k + d_k (residual over constant velocity); log b clamped to [log B_MIN, log B_MAX].
- Statics (group 2) and padding: mu = c0, b = B0, outputs ignored by loss and raster.
- Init: everything under `torch.random.fork_rng(devices=[])` + `torch.manual_seed(seed + FORECASTER_SEED_OFFSET)`,
  PyTorch defaults; null key / value randn * 0.02. No zero-init of the head (modes must differ at init).
- Yaw rule (all generators, Step 1): heading held at h0 for footprints; no yaw output, no yaw loss. `fc_xyh[..., 2]`
  is stored for analysis only.

### 4.4 Path-time rasteriser (`.../ck/temporal/raster.py`, owner B2)

Parameter-free; implemented with differentiable ops; float32 outside autocast. Per candidate on its OWN corridor
stations (`corridor_geometry` of that candidate; variants are rasterised like any candidate: no parent sharing).
Forecasts are computed once per token; only the projection is per candidate. B = T * Kc, token of candidate b = b // Kc.

```python
RASTER_CHANNELS = ("known", "ego_occ", "ego_tap03", "ego_tap06", "ego_tap09",
                   "veh_occ", "veh_vclose", "veh_dapp", "veh_inc", "veh_ext",
                   "vru_occ", "vru_vclose", "vru_dapp", "vru_inc", "vru_ext",
                   "st_occ", "st_inc", "st_ext")                                   # C_R = 18
def station_frame(geom) -> Dict[str, Tensor]       # pe = geom.path.eval(geom.s) under no_grad: p = pe.xy,
                                                   # u = pe.tangent, n = pe.normal [B,48,2]; s [B,48]; ds [B]
def ego_schedule(geom, t: Tensor) -> Tuple[Tensor, Tensor]          # s_e, v_e [B, nt]
def ego_channels(geom) -> Tensor                                    # [B, 4, 48, 21]
def known_channel(geom, roi) -> Tensor                              # [B, 1, 48, 21]
def forecast_at(fc: Forecast, t: Tensor) -> Dict[str, Tensor]       # xy, v, b [T,K,M,nt,2], present [T,K,M,nt]
def frenet(xy: Tensor, sf: Dict, chunk: int = 128) -> Dict[str, Tensor]   # s, d, u_bar, n_bar, ok  per point
def rasterize(fc: Forecast, geom, T: int, Kc: int, roi) -> Tensor   # [B, 18, 48, 21]
def gap_scalars(fc: Forecast, tau_flat: Tensor, T: int, Kc: int) -> Tensor   # [B, 20]
```

Formulas (pinned):
- Stations j = 0..47: s_j = (j + 0.5) ds, bin j = [j ds, (j + 1) ds]. Time bins t_k = 0.25 k, k = 0..20.
- Forecast at time t: piecewise-linear through (0, c0), (0.5, mu_1), ..., (5.0, mu_10); b linear through (0, B0),
  (0.5, b_1), ...; velocity V = segment velocity (mu_{i+1} - mu_i) / 0.5 for t in [t_i, t_{i+1}) (last segment at
  t = 5); present = present of knot round(t / 0.5) (1 at t = 0). Statics: c0, V = 0, b = B0, pi = 1.
- Frenet (no_grad index search, chunks <= 128 candidates): j0 = argmin_j |X - p_j|; a = (X - p_j0) . u_j0;
  j1 = j0 + 1 if a >= 0 else j0 - 1, clipped to [0, 47]. If j1 == j0: s = s_j0 + a, d = (X - p_j0) . n_j0,
  u_bar = u_j0, n_bar = n_j0. Else e = p_j1 - p_j0, lam = clamp((X - p_j0) . e / |e|^2, 0, 1), q = p_j0 + lam e,
  s = s_j0 + lam (s_j1 - s_j0), u_bar / n_bar = normalised lam-blend of u / n, d = (X - q) . n_bar.
- Footprint extents along the path: with h_hat = (cos h0, sin h0), h_perp = (-sin h0, cos h0):
  e_s = 0.5 (L |h_hat . u_bar| + W |h_perp . u_bar|), e_d = 0.5 (L |h_hat . n_bar| + W |h_perp . n_bar|).
- Keep a point iff -ds / 2 <= s < 48 ds, |d| - e_d <= LAT_DROP, present > 0, slot valid.
- Lateral overlap probability: H = EGO_W / 2 + e_d; b_d = max(B_MIN, sqrt((b_x n_bar_x)^2 + (b_y n_bar_y)^2));
  P_lat = F(H - d) - F(-H - d), F = Laplace CDF with scale b_d (F(z) = 0.5 e^{z / b_d} for z < 0, else
  1 - 0.5 e^{-z / b_d}).
- Mass w = pi_m * present * P_lat. Linear splat in s: u = s / ds - 0.5, j_lo = floor(u), f = u - j_lo; `scatter_add`
  w (1 - f) attr into (j_lo, k) and w f attr into (j_lo + 1, k), indices outside [0, 47] dropped. Times are evaluated
  exactly at t_k (no time splat needed).
- Attributes (attr = 1 for occ): vclose = clip(sgn(s - s_e(t_k)) * (v_e(t_k) - V . u_bar), -15, 15) / 15;
  dapp = clip(-sgn(d) * (V . n_bar), -5, 5) / 5; inc = clamp(H - |d|, 0, 3) / 3; ext = e_s / 5. Agent groups (veh,
  VRU): occ, vclose, dapp, inc, ext. Static: occ, inc, ext. No clamp of the summed channels.
- Ego: s_e(t) piecewise-linear through (0.5 i, S_i), i = 0..8 (geom.S, S_0 = 0), beyond 4 s S_8 + v_end (t - 4);
  v_e(t) = (S_{i+1} - S_i) / 0.5 on [0.5 i, 0.5 (i + 1)), v_end for t >= 4. cov(lo, hi, j) = clamp(min(hi, (j+1) ds) -
  max(lo, j ds), 0, ds) / ds. ego_occ = cov(s_e + EGO_REAR, s_e + EGO_FRONT); ego_tap_delta = cov(s_e + v_e delta +
  EGO_REAR, s_e + v_e delta + EGO_FRONT) for t_k <= TAP_T_MAX, else 0 (delta = 0.3, 0.6, 0.9). The exact TTC geometry
  (agents at t + delta) lives in the gap scalars.
- known[j, k] = 1 iff the 4 corners p_j +- (ds / 2) u_j +- (EGO_W / 2 + KNOWN_MARGIN) n_j lie inside the source ROI
  (x0 <= x <= x1, y0 <= y <= y1), broadcast over k. ROI is a source property, never hard-coded in the raster. Reason:
  21.6% of r34 top-16 candidates have S_look > 32 m, 10.7% have S_8 > 32 m [실측 critic, 4,000 val tokens].
- Zero objects: agent / static channels 0, ego and known as usual.

Gap scalars (port of a3 `interp_traj`, `sep_rect` and the 9-feature reduction, a3_conflict_features.py 74-115,
264-300): ego rear-axle poses at TJ (linear interpolation of [origin + 8 poses], heading unwrapped), segment speed
sp_j = seg[min(ceil(TJ_j / 0.5 - 1e-9), 8) - 1]; ego centre = pose + EGO_OFF u; for delta in TAPS: centre + sp delta u,
heading kept; objects at tau = TJ + delta (64 slots, up to 4.9 s). SAT separation, "ahead" = (c_o - c_e) . u_e > 0;
s_ag / s_agF / s_st = min over agents / agents ahead / statics (1e3 if none or not present).
f0 = min_t s_ag[delta 0], f1 = min_t s_agF[0], f2 = min_t s_st[0], f3 = min s_ag[delta > 0], f4 = min s_agF[delta > 0],
f5 = min s_st[delta > 0], f6 = first TJ with s_ag[0] < 0.3 (4.5 if none), f7 = first TJ with min_delta>0 s_agF < 0.3
(4.5), f8 = number of TJ with min_delta s_agF < 0.5; f0..f5 clipped to [-5, 15]; no objects -> (15 x 6, 4.5, 4.5, 0).
GAP (20) = 9 S-world features (t0 boxes held, all classes) + 9 forecast-world features (agents follow the forecast,
statics held; per mode, then pi-weighted mean) + 2 = min over modes with pi >= 0.1 (argmax mode if none) of the
forecast-world f0 and f3. Normalisation: separations / 5, times / 4.5, count / 16. For M = 1 the last two equal f0, f3.

### 4.5 Branch and injection (`.../ck/temporal/branch.py`, owner B2; `ck/model.py` edits, owner B2)

```python
@dataclass
class TemporalCtx:
    fc: Forecast          # used by raster / gap (detached unless cfg.fc_backprop)
    fc_raw: Forecast      # with grad, for the forecast loss (learned only)
    n_obj: int            # trimmed K
class TemporalBranch(nn.Module):
    def __init__(self, cfg: TemporalCfg, d_model: int = 192, seed: int = 0): ...
    def scene(self, objs: Dict[str, Tensor]) -> TemporalCtx       # once per token set; fp32, autocast off
    def features(self, ctx: TemporalCtx, geom, flat_tau: Tensor, T: int, Kc: int) -> Tuple[Tensor, Tensor]
        # -> st_add [B, 48, d], ud_add [B, d]; raster + gap under no_grad unless cfg.fc_backprop
    def param_counts(self) -> Dict[str, int]                       # {"temporal_shared": .., "forecaster": ..}
```
- Shared modules (identical in arms a, b, d), built first under `fork_rng` + `manual_seed(seed + TEMPORAL_SEED_OFFSET)`:
  `st_enc` = Conv2d(18, 32, 3, p 1), GroupNorm(4, 32), GELU, Conv2d(32, 48, 3, p 1), GroupNorm(8, 48), GELU,
  Conv2d(48, 48, 3, p 1), GroupNorm(8, 48), GELU on R [B, 18, 48, 21] -> [B, 48, 48, 21] -> permute to
  [B, 48 stations, 48 * 21]; `st_proj` = Linear(1008, d) ZERO-init (weight and bias); `gap_proj` = Linear(20, d)
  ZERO-init. Expected shared parameter count 237,888 [추정: B2 asserts and records the exact value].
- Forecaster (arm a only) built afterwards under `seed + FORECASTER_SEED_OFFSET`. Arms b and d build NO encoder or
  forecaster (DDP runs without `find_unused_parameters`); they register a persistent buffer `ca_b` [2, 10] read from
  `cfg.ca_scales` at build time (so the checkpoint carries the scales). Log both parameter counts in config.json.
- st_enc / projections run under the caller's autocast; the raster and gap scalars are float32 with autocast disabled.
- `CKTrunk.candidates(feat, mem, tau, v0, a0, eds, cmd, tfeat: Optional[Callable] = None)`: after `ud` is computed,
  if `tfeat is None` and `getattr(self, "_temporal_required", False)`: raise RuntimeError (catches online2 /
  fit_kd_calib callers that use the trunk directly). If tfeat is given: `st_add, ud_add = tfeat(geom, flat)`;
  `x = cat([draft_mlp(ud) + ud_add, st + st_add])`. If tfeat is None the two original lines run unchanged.
- `CKNet.__init__(arm, seed=0, norm=None, score_hidden=256, lead_aux=False, temporal: Optional[TemporalCfg] = None)`:
  existing modules first and unchanged; then if `temporal is not None and temporal.on`: `temporal.validate(arm)`,
  `self.temporal = TemporalBranch(temporal, self.trunk.d_model, seed)`, `self.trunk._temporal_required = True`
  (plain attribute, not state); else `self.temporal = None`.
- `CKNet.forward(bev, cand, status, extra=None, decode=True, slope=0.0, rescore_corr=False, objs=None)`:
  temporal on and objs None -> ValueError; temporal off and objs not None -> ValueError. On: `ctx =
  self.temporal.scene(objs)` once; every `trunk.candidates` call (cand, extra, rescore_corr) gets `tfeat` bound to
  ctx and its own Kc. Adds `out["fc"] = ctx.fc_raw` (learned only) and `out["t_stats"] = {"n_obj": ctx.n_obj}`.
- `objs` dict keys: OBJ_INPUT_KEYS, plus PRIV_INPUT_KEYS only for `oracle_gt` (else ValueError if present).
- `build_ck(..., temporal: Optional[TemporalCfg] = None)`; `init_from` a run whose temporal cfg differs -> ValueError.
- `load_ck(run_dir, which="last", device="cpu", allow_privileged: bool = False)`: `TemporalCfg.from_dict(
  cfg.get("temporal"))` (old configs -> off); privileged and not allowed -> PermissionError naming the run. This is the
  single choke point for online2 KD teachers, `eval_teacher_navtest_*`, eval_e2e*, fit_kd_calib.
- `param_counts()` adds `temporal_shared`, `forecaster`.

### 4.6 Loss (`.../ck/temporal/loss.py`, owner B2)

```python
def forecast_loss(fc: Forecast, fc_xyh: Tensor, fc_state: Tensor, obj_mask0: Tensor) -> Tuple[Tensor, Dict[str, float]]
```
- Valid (object, step): slot valid & agent & obj_mask0 (after augmentation) & fc_state == 1, targets trimmed to K'.
- Winner-take-all: m* = argmin_m mean over valid steps of |mu - y|_1 (no grad). NLL = mean over valid steps and both
  axes of log(2 b) + |mu - y| / b for m*; CE = -log pi_{m*}. Per-object loss NLL + CE averaged over objects with >= 1
  valid step. Invalid targets are replaced by mu.detach() before any arithmetic (no NaN). Zero valid objects -> the
  same expression times 0 (stays in the graph: DDP in step).
- Total = existing CK2 teacher loss (unchanged) + lambda_fc (1.0) * forecast_loss (arm a only).
- Default `fc_backprop = False`: raster and gap scalars see `fc.detach()`, so the critic loss never reaches the
  forecaster and (a) - (b) measures forecast quality, not a critic-shaped forecaster. End-to-end is an untested option.
- Stats: fc_loss, fc_nll, fc_ce, fc_minade, fc_minfde (last valid step), fc_n_obj.

### 4.7 OFF definition (must hold exactly)

With `temporal` off (default, and every old config.json):
- No new module, parameter or buffer. state_dict keys identical; `param_count` 3,516,178 and trainable 3,391,243 after
  `freeze_unused` [실측 ck2T10 config.json]; strict load of `ck2T10/ckpt_last.pt`.
- Datasets emit no new keys and draw no RNG. The augmentation stream is new (`token_rng(token, epoch, seed,
  "objaug")`), never 'var', '' or 'perm'.
- New CLI keys default off; config.json gets `"temporal": TemporalCfg().to_dict()`; the resume guard compares it
  (old config without the key = off).
- Tests against the 8ae8b1d reference tree (section 2): CPU forward max |delta| = 0; identical initial state_dict;
  identical batch bytes for 64 tokens x 2 epochs (CK2Dataset train-style and CKDataset r34-style); identical 3-step
  CPU train loss trace.

## 5. Training plumbing (owner B3)

### 5.1 Pack reader and augmentation (`tools/ck/data/obj_pack.py`, owner B1)

```python
PACK_VERSION = "v3_objpack_1"
def pack_dir(source: str, split: str, root: Optional[str] = None) -> Path
class ObjPack:
    def __init__(self, split: str, source: str = "bevfusion", root: Optional[str] = None, targets: bool = True,
                 expect_tokens: Optional[np.ndarray] = None): ...    # checks version, token order; lazy mmap per pid
    @property
    def has_targets(self) -> bool
    def item(self, row: int) -> Dict[str, np.ndarray]
        # obj_in f32 [3,64,17], obj_mask bool [3,64], frame_valid bool [3], obj_ok bool, (+ fc_xyh, fc_state)
def augment_objects(it: Dict[str, np.ndarray], token: str, epoch: int, seed: int, cfg: TemporalCfg) -> Dict
```
augment_objects (training only; after matching; identical for arms a, b, d): rng = `token_rng(token, epoch, seed,
"objaug")`; draws in this order, always all of them: (1) U(0,1) x 2 for past frames 1, 2: u < p_frame_drop ->
frame_valid False, mask False; (2) U(0,1) [3, 64]: u < p_obj_drop -> mask False; (3) N(0, v_noise^2) [3, 64, 2] added
to cols 6, 7 where mask (rows zeroed elsewhere). Targets untouched. Basis: in-sample velocity error median 0.27 m/s
[실측 critic, 48 moving vehicles]; navtest recall .846 (A_novelty notes).

### 5.2 Datasets (`tools/ck/data/ck2_dataset.py`, `ck_dataset.py`: copies from Phase 0, edited by B3)

- New kwarg `objects: Optional[Dict] = None` on CK2Dataset and CKDataset: `{"cfg": TemporalCfg.to_dict(), "aug": bool,
  "targets": bool}`. None -> old code path, no new keys.
- `__getitem__` (not `labels_item`): `it.update(pack.item(r))`; if aug: `augment_objects(it, token, self.epoch,
  self.seed, cfg)`; if `cfg.forecaster == "oracle_gt"`: add `priv_fc_xyh`, `priv_fc_state` (copies of the targets).
- `MODEL_INPUT_KEYS` unchanged; new `MODEL_INPUT_KEYS_TEMPORAL = MODEL_INPUT_KEYS + OBJ_INPUT_KEYS`.
- `collate_ck` unchanged (fixed shapes; the model trims to the batch maximum).
- Train dataset: aug True, targets True. Evals (r34 val, val_raw, train_eval r34 / raw): aug False, targets True when
  the split has them.

### 5.3 Trainer (`tools/ck/train_ck2.py`, B3)

- CLI: `--temporal {0,1}` (0), `--t-source bevfusion`, `--t-forecaster {learned,ca,oracle_gt}`, `--t-modes 6`,
  `--t-lambda-fc 1.0`, `--t-fc-backprop 0`, `--t-frame-drop 0.1`, `--t-obj-drop 0.05`, `--t-vnoise 0.3`,
  `--t-pack-root`, `--t-ca-scales`. `resolve_args` builds `a.temporal_cfg` (off unless `--temporal 1`), validates
  against `--arm`.
- `forward(net, batch, use_amp, dev, with_var, tcfg)`: passes `objs = {k: batch[k] for k in OBJ_INPUT_KEYS}` (+
  PRIV_INPUT_KEYS for oracle) when on; nothing else new reaches the model.
- `compute_loss`: + `tcfg.lambda_fc * forecast_loss(out["fc"], batch["fc_xyh"], batch["fc_state"],
  batch["obj_mask"][:, 0])` when `"fc" in out`; stats prefixed `fc_`.
- `dummy_backward`: also zero-weights `out["fc"].mu / log_b / logit_pi` when present.
- Optimiser: one AdamW, two param groups (critic = every trainable parameter not under `temporal.forecaster.`;
  forecaster). Clip separately (both `--clip`); skip the step if either grad norm is non-finite; log `grad_norm`
  (critic, as before) and `grad_norm_fc`. Off / arm b: forecaster group empty, behaviour as before.
- config.json: `temporal` dict, top-level `privileged` (0/1), `param_count_temporal_shared`,
  `param_count_forecaster`. Resume guard list += `temporal`. If privileged: every val_metrics.jsonl row gets
  `"privileged": 1` and the run name must end in `_PRIV` (refuse otherwise).
- run_eval passes objs; when the forecaster is learned and targets exist, adds `fc_nll`, `fc_minade`, `fc_minfde` to
  that split's metrics. Existing metrics unchanged (they stay descriptive; the decision is offline, section 8).
- Fail loudly: obj_ok False on > 0.5% of a split's tokens -> SystemExit.

### 5.4 Scripts (B3; written, never executed by builders except the smoke in Integrate)

- `tools/ck/v3/run_step1.sh {a|b|d}`: refuses unless `nvidia-smi` shows < 1 GB used on GPUs 0 and 1; writes pid /
  pgid files under `D3/train/`; `CUDA_VISIBLE_DEVICES=0,1 nice -n 10 torchrun --standalone --nproc_per_node=2
  tools/ck/train_ck2.py --arm T --out-root D3/train --run-name <name> --epochs 10 --tokens-per-batch 16 --lr 6e-4
  --sur-groups near_mid --eval-every 2 --best-metric none --score-keys nc,ttc,ep,comfort --n-var 80 --var-name
  var_separate_sampler16_accstraight --seed 0 --sampler-seed 0 --norm-from compute --warmup-steps 500 --wd 0.01
  --clip 1.0 --workers 8 --ep-target official --temporal 1 --t-forecaster <learned|ca|oracle_gt> --resume`.
  Names: `v3T10_a_learned`, `v3T10_b_ca`, `v3T10_d_oracle_PRIV`.
- `tools/ck/v3/smoke_v3.sh`: arms off / a / b, `--max-steps 60 --limit-tokens 512 --val-limit 64 --eval-every 1`,
  first on GPU 0 (world 1, `--tokens-per-batch 8`), then on GPUs 0,1 (world 2, `--tokens-per-batch 16`); out-root
  `D3/smoke/`; prints gpu_mem_gb and cand_per_s per arm. Each invocation < 15 min.

## 6. Inference and decision tools (owner B4)

- `tools/ck/v3/infer_val_v3.py --run <dir> [--which last] [--out D3/infer/<run name>] [--allow-privileged]
  [--device cuda] [--batch 8] [--workers 6]`: port of `ck/ck2/analysis/lead_remedy/scripts/infer_val_ck2T.py` stage
  infer. All 18,179 navtrain_val rows, CKDataset(bev 'T', k 16, labels None, objects aug False), extra =
  `lead_remedy/infer/variants/traj80.npy`, fp16 autocast. Writes score_logit16 [N,16,5], prob16, score_logit80
  [N,80,5], prob80, w_lat16 [N,16,6], done [N], bev_ok [N], obj_ok [N], infer_meta.json (ckpt sha16, temporal cfg,
  privileged, code sha256, KST times). Check: pdms_ck_noim recomputed from prob16 + cand labels equals the run's last
  navtrain_val r34 row within 1e-6. Works for temporal-off runs (arm c = `ck/ck2/train/ck2T10`).
- `tools/ck/v3/decide_step1.py --a <infer dir> --b <infer dir> --c <infer dir> [--d <infer dir>] [--out
  D3/decide/step1.json]`: section 8 metrics, imports `B_ceiling/scripts/lib.py` (`ck_final`, `select`, `boot`) by
  path with `sys.dont_write_bytecode = True`.

## 7. Step-1 experiment (launched later by the main session, only with user approval)

- Arms: (a) learned, (b) CA in the same branch, (c) existing `ck2T10` (descriptive: trained at 4f29592 + code snapshot,
  not 8ae8b1d), optional (d) oracle `v3T10_d_oracle_PRIV` (forecasts = matched GT futures of the BEVFusion t0 objects =
  GTt world, M = 1; analysis upper bound only, PRIVILEGED).
- Same world size for a and b: 2-GPU DDP on GPUs 0,1, one after the other (a, then b), 8 tokens per rank as ck2T10.
  1-GPU parallel is rejected: 16 tokens on one rank is about 15 GB before the branch [실측 critic: ck2T10 7.49 GB/GPU at
  8 tokens/rank].
- ck2T10 ran 53,190 steps in 2.8 h [실측 critic]. Budget per arm <= 3.5 h (+25%) and <= +3 GB/GPU, checked by the
  smoke (section 5.4) before Step 1. Sequential a + b <= 7 h; inference 4 x ~4 min [추정].
- One seed per arm (seed variance not covered; extra seeds only with user approval).

## 8. Decision metrics and GO rule (pre-registered)

All on navtrain_val r34 top-16 (18,179 tokens, 214 logs), `ckpt_last` of each arm (epoch 10), no best-epoch selection.
- Decision PDMS: pick = `lib.select(lib.ck_final(p, w = (0, .5, .5, 1)), allowed = all, tie = arange(16))`; p_nc, p_ep,
  p_ttc, p_comfort from the arm; p_dac FROZEN for all arms = `ck/ck2/analysis/lead_remedy/infer/ck2T/prob16.npy[..., 1]`
  (report-53 protocol; the ck2T10 recipe does not train DAC: val `auc_fail_dac` 0.339, `bce_dac` 0.363 at epoch 9
  [실측 ck2T10/val_metrics.jsonl]). PDMS = official `labels/navtrain_val/cand/labels.npy[:, :16, 6]` of the pick.
  Delta = 100 x per-token difference; CI = `lib.boot(diff, logs, n_boot=2000, seed=0)` (log-cluster, paired).
  Variants reported: w_dac = 0; p_dac = packed `v2_sim[..., 1]`.
- Lead-decel slice: `lead_decel == 1` in `ck/lead/navtrain_val.parquet` (1,605 tokens; equals lead_remedy index
  `lead_slice` [실측]). TTC AUC = roc_auc(ttc < 1, 1 - p_ttc) over all 16 candidates of those tokens. NC|TTC fail =
  % of those tokens whose pick has nc < 1 or ttc < 1 (f2_stack_full16.py 152-165).
- a-1.0 hazard preference (pool96 only): `lead_remedy/infer/sample3000/labels96.npz` (layout c = 6k + v), arm probs
  via `lib.to_pool(prob16, prob80)`; pairs = parent k safe (nc >= 1 and ttc >= 1), variant a-1.0 (c = 6k + 1) unsafe,
  both valid (`valid96 | identity`, finite); pref = share with log p_nc + log p_ttc higher for the parent (A_novelty
  cv_st_baseline.py 435-467; 1,583 pairs [실측 A_novelty notes]). CI by `lib.boot` on paired pair indicators by log.
- GO iff all of:
  - G1: Delta PDMS (a - b) >= +0.20 and CI lower bound > 0.
  - G2: lead-decel TTC AUC of (a) >= .855.
  - G3: lead-decel NC|TTC fail of (a) <= 6.97%.
  - G4 (superseded by section 12 D4: non-inferiority, pref(a) - pref(c) >= -0.01 and CI lower bound >= -0.03): a-1.0
    hazard preference of (a) not worse than (c): the CI upper bound of pref(a) - pref(c) >= 0. Also reported
    vs (b). [추정: operationalisation of "not worse"; main session confirms it before launch.]
- Also reported (descriptive): (b) - (c) (expected about +0.38; about 0 means the branch is broken, not the forecaster),
  (a) - (c), (d) - (b) (oracle headroom), every metric on the full-history subset (frame_valid all True in the clean val
  pack), val forecast NLL / minADE / minFDE of (a) vs CA (ADE is not a proxy for critic value: CA beats CV in the stack
  although its 5-s ADE is worse [실측 A_full16 notes §4-4]).
- Expectation [추정]: NO-GO is likely. G1 needs about 50-60% of the perfect-forecast headroom (FUTt - CA = +0.42 GBM /
  +0.34 logistic, from a val-fitted stack). BEVFusion is in-sample on val, so val gains overstate navtest; navtest stays
  descriptive only.

## 9. Work split

Order: B0 (serial) -> B1, B2, B3, B4 in parallel against the interfaces above -> B1 data jobs -> Integrate (all tests,
GPU smoke) -> Fix. Every builder: CPU only, writes only its files plus its tests, tags claims in docstrings.

### B0 setup (section 2)
- Files: `tools/ck/data/*` (copies), `navsim/agents/para_ssr/refiner/future_teacher.py` (copy), `v2_source`
  (symlink), `.gitignore` (+1 line), `tools/ck/v3/__init__.py` (empty), `tools/ck/v3/make_ref_tree.sh`.
- Data: `D3/phase0/{pristine/, phase0_copies.json}`, `D3/gt_objects/{navtrain_val, navtrain_train}` (symlinks),
  `D3/ref_8ae8b1d/`.
- Done when: section 2 acceptance passes.

### B1 objects and data (sections 3, 4.2, 5.1)
- Files: `navsim/agents/para_ssr/ck/temporal/objects.py`; `tools/ck/data/obj_pack.py`;
  `tools/ck/v3/build_obj_pack.py` (`--split --source bevfusion --workers <=16 --out-root D3/objpack [--limit N]
  [--no-targets]`; fork pool, chunks of 64 tokens, atomic writes, build.json); `tools/ck/v3/fit_ca_scales.py`
  (`--pack D3/objpack/bevfusion/navtrain_train --out D3/ca_scales.json`); `tests/test_v3_objects.py`.
- Imports allowed: `future_teacher` (TeacherDetCache, teacher_box_to_local, local_to_n, BX_*), `gt_future`,
  `ck.temporal.config`, `anchor_sampler.token_rng`. No torch in `objects.py`.
- Tests: synthetic known-answer box conversion (heading = wrap(-yaw - pi/2), L = dy, W = dx, velocity rotation);
  causal loader refuses a planted frame at pos - 1 with ts > ts0 (CausalityError) and never reads pos + j (mock);
  dt_range invalidation; pack shapes / dtypes / token order on 32 real val tokens built into a tmp dir under
  `D3/test_tmp/`; @real: 60 val tokens matched to GT at t0, median heading error < 0.02 rad, median |L - L_gt| <
  0.05 m; t-1.0 parked vehicles moved to t0 vs GT median < 0.2 m (critic measured 0.10 m); `ca_forecast_np` equals a3
  `Worker.history` + `world('CA')` on agents of 50 real val tokens (max |delta| < 1e-9 where n_raw <= 64); match rate on
  val within 0.5 pp of 99.4%; augment determinism (same token / epoch / seed -> same bytes), cfg off -> no draw.
- Data jobs after tests pass (each < 60 min, 16 procs): packs for navtrain_val, navtrain_train, navtest (no targets);
  then `fit_ca_scales.py`. Record wall times and shas in each build.json.

### B2 model (sections 4.1, 4.3-4.7)
- Files: `navsim/agents/para_ssr/ck/temporal/{__init__.py, config.py, forecast.py, raster.py, branch.py, loss.py}`;
  edits in `navsim/agents/para_ssr/ck/model.py` (CKTrunk.candidates tfeat hook, CKNet temporal, build_ck, load_ck
  allow_privileged, param_counts); `tests/test_v3_temporal_model.py`.
- Must not edit: `refiner/*`, `ck/losses.py`, `ck/constants.py`, `ck/online*.py`.
- Tests: OFF identity vs `D3/ref_8ae8b1d` (forward bitwise, state_dict, param counts 3,516,178 / 3,391,243, strict load
  of ck2T10 ckpt_last); ON at init == OFF outputs bitwise for the same seed (zero-init projections); (a) and (b) shared
  modules have identical initial weights; raster known answers (one object at a known station and time: occ splat
  weights, P_lat -> 1 for b -> 0 on the path, ego cov values, known mask for a 40 m straight candidate: 1 below
  32 m - margin, 0 beyond); gap scalars equal a numpy port of a3 on synthetic scenes (< 1e-4) and @real CA-world features
  vs `A_full16/out/f1_feats_full16.npz` feats[..., 2] (median |delta| < 1e-4, >= 99% within 0.01); torch `ca_forecast`
  == `objects.ca_forecast_np` (< 1e-5); no NaN with fully masked frames / zero objects / one object; forecast_loss known
  answer (M = 2 toy) and zero-target batch gives finite 0 with gradients present; every trainable parameter gets a
  finite (possibly zero) gradient in arms a, b; candidate slot permutation permutes outputs; trim invariance
  (padding to 64 vs trimmed: identical outputs); privileged load raises PermissionError; temporal-on forward without
  objs raises; direct `trunk.candidates` without tfeat on a temporal net raises; temporal on with arm M raises.
- Record in the test log: shared / forecaster parameter counts, CPU ms per token for scene() and per candidate for
  features().

### B3 pipeline (sections 5.2-5.4)
- Files: edits in `tools/ck/data/ck2_dataset.py`, `tools/ck/data/ck_dataset.py`, `tools/ck/train_ck2.py`;
  `tools/ck/v3/run_step1.sh`, `tools/ck/v3/smoke_v3.sh`; `tests/test_v3_train.py`.
- Tests: OFF batch bytes identical to the reference tree for 64 tokens x 2 epochs (CK2Dataset navtrain_val n_var 32
  bev none, CKDataset navtrain_val k 16 labels cand); OFF 3-step CPU train loss trace identical (synthetic in-memory
  data as test_ck2_train); `test_model_sees_only_model_inputs` extended (perturbing fc_xyh, fc_state, lead flags,
  anc_*, var metadata leaves outputs bitwise unchanged in arms a and b); arms a / b / d run 3 CPU steps on a tiny
  synthetic pack; batch streams identical between arms a and b (same bytes, including augmentation); 2-process gloo
  DDP == 1 process for arm a; resume refuses a changed temporal cfg; privileged run name rule; obj_ok threshold.

### B4 evaluation (sections 6, 8)
- Files: `tools/ck/v3/infer_val_v3.py`, `tools/ck/v3/decide_step1.py`, `tests/test_v3_decide.py`.
- Tests: decide with a = b = ck2T dump -> every delta 0, CI (0, 0); ck2T noim PDMS with its own DAC = 90.814
  (report 53 §1, tolerance 0.001); frozen-DAC rule changes only p_dac; lead slice size 1,605; hazard-pair count 1,583
  and ck2T a-1.0 preference within 0.005 of .745 (A_novelty 'none'; exact equality [미확인]); infer_val_v3 on
  `ck2T` for 64 rows on CPU reproduces `lead_remedy/infer/ck2T/prob16.npy` rows within 2e-3 (fp16 vs CPU fp32).

### Integrate / Fix (one agent, after B1-B4)
- Run every new and touched test on CPU; then `smoke_v3.sh` on GPU 0 / GPUs 0,1 (< 15 min each). Pass: arm a <= +3
  GB/GPU and <= +25% s/step vs off at 8 tokens/rank; no skipped steps; val metrics finite.
- Report: test counts, smoke numbers, pack build stats, the exact Step-1 commands. Then stop: Step 1 is the main
  session's call with user approval.

## 10. Open items

- G4 operationalisation (vs (c), CI upper bound >= 0) [추정]: confirm before launch.
- A fresh no-temporal baseline at v3 code (instead of the descriptive (c)) would cost another ~2.8 h GPU run: only
  with user approval.
- `gt` source with causal GT history, `v2det` adapter, future-detection targets, `fc_backprop = True`: interfaces
  fixed, not built in Step 1.

## 11. Changes after review (critic findings 1-21, 2026-10-11 14:04-14:35 KST)

The spec did not exist when the critic reviewed; this v1 is written with every finding applied.

| # | finding | where |
|---|---|---|
| 1 | missing tools/ck/data, future_teacher.py, v2_source | §2 Phase 0 (cp -a, sha record, .gitignore exception, diff -r, ref tree) |
| 2 | box format wrong in task / docstring | §3.1, §3.2 (17-d canonical row), B1 known-answer tests |
| 3 | causal past loader | §3.3 (pos - j only, current then future-named cache, CausalityError, dt range, planted-frame test) |
| 4 | untrained DAC contaminates the GO metric | §8 frozen p_dac from ck2T prob16, ckpt_last, variants; §6 fresh dump for (c) |
| 5 | full recipe, world size | §5.4 full flag list; §7 2-GPU sequential; (c) descriptive; OFF tests vs 8ae8b1d |
| 6 | CA must be a clean control | §3.6 exact a3 rule + fitted fixed scales, M = 1; §4.5 shared-module seeds, (b) builds no encoder; §5.1 same aug |
| 7 | variants do not share agent channels | §4.4 per-candidate stations, no parent sharing |
| 8 | raster memory | §4.4 score >= .3 objects, trim to batch max, chunked no_grad search, scatter_add, fp32; default detach (§4.6) puts raster / gap under no_grad; §9 smoke budget |
| 9 | known mask | §4.4 known channel from the source ROI |
| 10 | GT / oracle leaks | §3.4-3.5 no GT ordering; §7 (d) = GTt world; §4.5 load_ck PermissionError, `_PRIV` names, privileged tags; gt source deferred with crop rule |
| 11 | precise OFF definition | §4.7 |
| 12 | plumbing and whitelist | §4.5, §5.2-5.3 (OBJ_INPUT_KEYS only, targets never in forward, masked forecast loss, raises) |
| 13 | missing history / empty scenes | §4.3 null + self keys, masks; §4.4 zero objects; §5.1 frame dropout; §8 full-history subset |
| 14 | target rules | §3.5 (Hungarian on clean boxes, 2 m, OBS mask, index time, statics held), §4.6 WTA, lambda_fc 1.0, yaw rule §4.3 |
| 15 | pin raster formulas | §4.1 constants, §4.4 formulas and gap list, §4.6 backprop default |
| 16 | normalisation | §3.2 fixed constants; CA scales from navtrain_train only |
| 17 | augmentation | §5.1 ('objaug' stream, fixed draw order, train only, after matching, same for all arms) |
| 18 | GO criteria code | §8 (lib.boot, lead_decel parquet, f2 fail rule, pool96 hazard definition), one seed noted |
| 19 | pre-register NO-GO | §8 expectation and (b) - (c) check |
| 20 | precompute packs | §3.4, B1 data jobs; navtest descriptive |
| 21 | room for v2det | §4.2 source interface with the v2det conversion recorded; t0-only sources work (frame_valid) |

Deliberate deviations from the critic text, with reasons:
- `collate_ck` is not changed: pack items have fixed shapes (64 slots), and the model trims to the batch maximum
  (`trim_objects`), which gives the same memory effect.
- Association / matching groups follow f3 / f4 exactly (agent vs static), so the measured 99.4% match rate and the a3
  CA equality hold; vehicle <-> VRU cross matches are counted, not forbidden.
- Raster TTC taps are ego-side channels (ego projected along the path at bin k); the exact "agents at t + delta" TTC
  geometry is carried by the gap scalars (a3 port), which is what produced the measured CA gain.
- Past frames with dt outside (0.25, 1.25) s are marked invalid instead of raising; only ts_past >= ts0 raises.

## 12. Changes after the code review (main-session decisions D1-D8, 2026-10-11 evening KST)

Full record (finding -> fix -> verification): `docs/refiner_v3/REVIEW_FIXES.md`; launch: `docs/refiner_v3/STEP1_RUN.md`.

| id | change | where |
|---|---|---|
| D1 | K_OBJ 64 -> 96; packs rebuilt under `D3/objpack_k96/bevfusion/{navtrain_train, navtrain_val, navtest}` (train truncation 0.575 % <= 1 %); GT objects only from `D3/gt_objects/<split>` (val copied, train linked, navtest built), no fallback | config.py, build_obj_pack.py |
| D1' | singleton rule in match_targets: GT tracks first annotated after t0 never matched; matched one-keyframe tracks -> state 4 FC_SINGLE (oracle: as OBS; loss / metrics / CA scales: excluded) | objects.py, forecast.py |
| D2 | CA Laplace scales fitted on the augmented train stream (every row x epochs 0-9, `augment_objects` seed 0) -> `D3/ca_scales.json` v3_ca_scales_2 (records source, k_obj, aug, b_clean) | fit_ca_scales.py |
| D3 | forecaster pre-training stage `pre` (`tools/ck/v3/pretrain_forecaster.py`, forecast loss only, train split, own dir `D3/train/v3fc_pre_learned6`); arm (a) loads it (`--t-fc-init`) and keeps it FROZEN (`TemporalCfg.fc_train = 'frozen'`, default): no forecast-loss term, no gradient, one optimiser group, same trainable count as (b). `fc_train 'joint'` = optional fine-tuning | config.py, branch.py, train_ck2.py |
| D4 | G1 unchanged (single seed; second seed if abs(Delta - 0.20) <= 0.15); G2 / G3 absolute + paired (a) - (b), (a) - (c) deltas with log-cluster CIs; G4 non-inferiority vs (c): Delta >= -0.01 and CI lo >= -0.03 (vs (b) reported) | decide_step1.py |
| D5 | provenance (training-code sha, worktree diff sha, pack / ca_scales / forecaster shas) in every run's config.json and infer_meta.json; decide refuses arms whose code / data differ; run_step1 writes a code tarball | provenance.py, train_ck2.py, infer_val_v3.py, decide_step1.py, run_step1.sh |
| D6 | infer reproduce check: per-token logits vs the trainer's saved last r34 eval (q99.9 of abs(dlogit) <= 2e-4, <= 0.1 % flipped picks), reported (decide requires it), never an exit | infer_val_v3.py, train_ck2.py |
| D7 | runtime x1.3-1.45 per step accepted (about 4 h per arm on 2 GPUs) | - |
| D8 | fresh arm-(c) dump `D3/infer/ck2T10` (pdms_ck_noim 0.9032467603683472 = val_metrics exactly) | - |
