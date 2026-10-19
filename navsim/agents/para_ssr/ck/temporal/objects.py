"""refiner-v3 causal object sources, GT forecast targets and the numpy CA reference (SPEC sections 3, 4.2; owner B1).

numpy only: no torch code here (``future_teacher`` pulls torch in through ``refiner.adapters`` at import time; this
module never uses it).  Every constant comes from ``ck.temporal.config`` (SPEC 4.1).

BEVFusion box conversion (SPEC 3.1; corrects the task text and the ``data.TeacherCache.load_det`` docstring)
  raw pred_boxes_3d [200, 9] = (x, y, z_bottom, dx = WIDTH, dy = LENGTH, dz, yaw_mmdet, vx, vy) in the frame's OWN ego
  (rear-axle) axes, velocity absolute in those axes.  Every frame, t0 included, goes through
  ``future_teacher.teacher_box_to_local`` (heading = wrap(-yaw - pi/2), L = dy, W = dx) and then
  ``future_teacher.local_to_n(local, pose_k, pose0)`` (for t0 pose_k = pose0: same chain, no shortcut).
  Poses (x, y, yaw) come from the future-index frames parquet (bit-exact copies of the log poses).

Canonical object row (D_OBJ = 17, metric, t0 N frame, unnormalised; SPEC 3.2)
  0 x  1 y  2 cos h  3 sin h  4 L  5 W  6 vx  7 vy  8 score  9..15 one-hot of the BEVFusion label 0..6  16 age_s
  age_s = (ts0 - ts_frame) * 1e-6 (0 for t0).  Padding rows are all zero with mask False.

Causal past frames (SPEC 3.3, ``CausalPastIndex``)
  frame f = 0 (t0), 1 (t-0.5 s), 2 (t-1.0 s) = the frame at log position pos - f of the same log, looked up ONLY as
  ``bypos[(log, pos - f)]``.  Never pos + j, never the future index json.  ts_past >= ts0 raises CausalityError (a
  data bug, never silently used); dt outside DT_RANGE (0.25, 1.25) s marks the frame invalid ('dt_range'); a
  missing parquet row -> 'not_in_parquet'; no npz in either cache -> 'no_npz'.
  Box source: the current-frame cache (cache_{train,val}_50x100) first, then the '_future'-named cache (a past frame
  that is not an index key lives only there).  ``data.TeacherCache`` keeps its '_future' refusal untouched.
  Per frame: score >= SCORE_THR[f] = (0.3, 0.2, 0.2), top K_OBJ = 64 by score (descending, ties by proposal index).
  Nothing is ever sorted or truncated by anything GT-derived.

Forecast targets (SPEC 3.5, ``match_targets``): Hungarian on the CLEAN t0 rows, f3_gt_worlds / f4 rule (every valid
t0 slot vs every GT track OBS at t = 0, cost = centre distance if the same agent/static group and < 2.0 m, else 1e6;
accept < 1e5), targets only for agent slots (classes 0-2): ``gt_future.query`` at t = 0.5 k (index time), (x, y,
heading_unwrapped); state 1 OBS, 3 t > t_avail, else 2; 0 = no target (unmatched / static / padding).
Singleton rule (review fix, v3_objects_2): a GT track with one keyframe is held at that pose at every t by
``gt_future.query`` (the official metric-cache rule).  (i) A track first annotated after t0 (meta first_k > 0) is never
matched (its t = 0 "position" is a future one); (ii) a matched singleton annotated at t0 gets state 4 = FC_SINGLE
instead of OBS: the oracle reads it as OBS (official rule), the forecast loss / metrics / CA scales never use it
("stays still for 5 s" is not an observed motion).

CA reference (SPEC 3.6, ``ca_forecast_np``) = a3_conflict_features ``Worker.history`` + ``world('CA')`` exactly, except
statics are held at c0 (a3 moved them by their near-zero velocity).

Tags: [실측] measured, [추정] estimate, [미확인] not checked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from navsim.agents.para_ssr.ck.temporal.config import (AGENT_CLASSES, D_OBJ, DT_RANGE, FC_ABSENT, FC_BEYOND, FC_NONE,
                                                       FC_OBS, FC_SINGLE, FC_STEPS, FC_T, GROUP_OF_CLASS, K_OBJ, N_CLS,
                                                       N_FRAMES, ROI, SCORE_THR)
from navsim.agents.para_ssr.refiner import future_teacher as FT
from navsim.agents.para_ssr.refiner import gt_future as GF

OBJECTS_VERSION = "v3_objects_2"      # 2: singleton rule in match_targets (review fix, REVIEW_FIXES.md)

# frame-status codes (pack frame_why.npy) and source codes (pack frame_src.npy; = future_teacher SRC_* codes)
WHY = ("", "not_in_parquet", "no_npz", "dt_range")
WHY_CODE = {w: i for i, w in enumerate(WHY)}
SRC_NONE, SRC_CURRENT, SRC_FUTURE = FT.SRC_NONE, FT.SRC_CURRENT, FT.SRC_FUTURE        # 0, 1, 2

# canonical-row columns (SPEC 3.2)
C_X, C_Y, C_COS, C_SIN, C_L, C_W, C_VX, C_VY, C_SCORE, C_CLS0, C_AGE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16
_GROUP = np.asarray(GROUP_OF_CLASS, np.int64)
_AGENT = np.asarray(AGENT_CLASSES, np.int64)

# association / matching constants of a3 / f3 (not model constants)
HIST_GATE_M = 1.5          # a3 Worker.history: nearest past box of the same group within 1.5 m of c0 - v0 dt
ACC_CLIP = 6.0             # m/s^2
ACC_MIN_SPEED = 0.5        # m/s: a = 0 below
MATCH_GATE_M = 2.0         # f3 / f4 GATE

assert D_OBJ == 17 and N_CLS == 7 and C_AGE == D_OBJ - 1 and C_CLS0 + N_CLS == C_AGE
assert tuple(FT.CLASS_NAMES) == tuple(GF.CLASS_NAMES[:N_CLS])
assert tuple(int(c) for c in FT.AGENT_LABELS) == tuple(AGENT_CLASSES)


class CausalityError(RuntimeError):
    """A 'past' frame whose timestamp is not strictly before t0 (data bug: never silently used)."""


# ----------------------------------------------------------------------------------------------- rows
def row_classes(rows: np.ndarray) -> np.ndarray:
    """class index from the one-hot columns of canonical rows [..., 17] -> int64 [...] (padding rows give 0: mask!)."""
    return np.argmax(np.asarray(rows)[..., C_CLS0:C_CLS0 + N_CLS], -1).astype(np.int64)


def row_agent(rows: np.ndarray) -> np.ndarray:
    """agent flag (classes 0-2) of canonical rows [..., 17] (padding rows -> True: always combine with the mask)."""
    return np.isin(row_classes(rows), _AGENT)


def canonical_rows(b_n: np.ndarray, scores, labels, age_s: float) -> np.ndarray:
    """N-frame boxes [n, 9] in future_teacher BX order (x, y, z, L, W, H, heading, vx, vy) -> canonical rows [n, 17]
    float64.  Labels outside 0..6 raise."""
    b = np.asarray(b_n, np.float64).reshape(-1, 9)
    s = np.asarray(scores, np.float64).reshape(-1)
    lab = np.asarray(labels).astype(np.int64).reshape(-1)
    n = b.shape[0]
    if s.shape[0] != n or lab.shape[0] != n:
        raise ValueError(f"canonical_rows: {n} boxes vs {s.shape[0]} scores / {lab.shape[0]} labels")
    if n and ((lab < 0) | (lab >= N_CLS)).any():
        raise ValueError(f"canonical_rows: labels outside 0..{N_CLS - 1}: {sorted(set(lab[(lab < 0) | (lab >= N_CLS)].tolist()))}")
    out = np.zeros((n, D_OBJ), np.float64)
    out[:, C_X], out[:, C_Y] = b[:, FT.BX_X], b[:, FT.BX_Y]
    out[:, C_COS], out[:, C_SIN] = np.cos(b[:, FT.BX_HEAD]), np.sin(b[:, FT.BX_HEAD])
    out[:, C_L], out[:, C_W] = b[:, FT.BX_L], b[:, FT.BX_W]
    out[:, C_VX], out[:, C_VY] = b[:, FT.BX_VX], b[:, FT.BX_VY]
    out[:, C_SCORE] = s
    out[np.arange(n), C_CLS0 + lab] = 1.0
    out[:, C_AGE] = float(age_s)
    return out


def teacher_frame_rows(boxes_raw, scores, labels, pose_k, pose0, age_s, thr, k_obj: int = K_OBJ,
                       return_info: bool = False):
    """One raw BEVFusion frame -> canonical rows [m <= k_obj, 17] float64 in the t0 N frame.

    Keep finite boxes with score >= thr, order by score descending (ties by proposal index: stable sort), keep the
    first k_obj, convert with teacher_box_to_local -> local_to_n(pose_k, pose0).  ``return_info`` -> (rows, info)
    with info = dict(n_raw = boxes above thr before the cap, idx = kept proposal indices, n_nonfinite)."""
    braw = np.asarray(boxes_raw)
    sc = np.asarray(scores)
    lab = np.asarray(labels)
    if braw.ndim != 2 or braw.shape[1] != 9 or sc.shape != (braw.shape[0],) or lab.shape != (braw.shape[0],):
        raise ValueError(f"teacher_frame_rows: boxes {braw.shape}, scores {sc.shape}, labels {lab.shape}")
    fin = np.isfinite(braw).all(1) & np.isfinite(sc)
    above = sc >= thr
    idx = np.flatnonzero(above & fin)
    n_nonfinite = int((above & ~fin).sum())
    order = idx[np.argsort(-sc[idx].astype(np.float64), kind="stable")]
    keep = order[:int(k_obj)]
    local = FT.teacher_box_to_local(braw[keep])
    b_n = FT.local_to_n(local, pose_k, pose0)
    rows = canonical_rows(b_n, sc[keep], lab[keep], age_s)
    if return_info:
        return rows, dict(n_raw=int(idx.size), idx=keep, n_nonfinite=n_nonfinite)
    return rows


# ----------------------------------------------------------------------------------------------- causal index
FRAMES_COLS = ("token", "log", "pos", "timestamp", "x", "y", "yaw", "lidar2ego_identity")


class CausalPastIndex:
    """frames parquet (build_future_index.py: every index key and every non-null future frame) -> causal past lookup.

    ``bypos`` maps (log, pos) -> token and is the ONLY structure used to find a past frame (tests replace it with a
    recording mapping to prove that no pos + j key is ever read)."""

    def __init__(self, frames_path: Union[str, Path]):
        import pandas as pd

        self.frames_path = Path(frames_path)
        df = pd.read_parquet(self.frames_path, columns=list(FRAMES_COLS))
        if df.token.duplicated().any():
            raise ValueError(f"{self.frames_path}: duplicated tokens")
        if df.duplicated(["log", "pos"]).any():
            raise ValueError(f"{self.frames_path}: duplicated (log, pos)")
        toks = df.token.astype(str).tolist()
        logs = df["log"].astype(str).tolist()
        pos = df["pos"].astype(int).tolist()
        ts = df["timestamp"].astype("int64").tolist()
        xs, ys, yaws = (df[c].astype(float).tolist() for c in ("x", "y", "yaw"))
        l2e = df["lidar2ego_identity"].astype(bool).tolist()
        self.rows: Dict[str, tuple] = {t: (lg, p, s, (x, y, h), e)
                                       for t, lg, p, s, x, y, h, e in zip(toks, logs, pos, ts, xs, ys, yaws, l2e)}
        self.bypos = {(lg, p): t for t, lg, p in zip(toks, logs, pos)}

    def has(self, token: str) -> bool:
        return token in self.rows

    def t0(self, token: str) -> Tuple[str, int, int, Tuple[float, float, float]]:
        """(log, pos, ts_us, pose) of the t0 token; KeyError if the token has no parquet row; ValueError if its
        lidar2ego is not the identity."""
        r = self.rows[token]
        if not r[4]:
            raise ValueError(f"{token}: lidar2ego is not the identity")
        return r[0], r[1], r[2], r[3]

    def pose(self, token: str) -> Tuple[float, float, float]:
        return self.t0(token)[3]

    def past(self, token: str, j: int) -> Tuple[Optional[str], float, str]:
        """frame j >= 1 log positions before the t0 token -> (token | None, dt_s, why).

        why '' = usable; 'not_in_parquet' (no row at pos - j; token None, dt NaN); 'dt_range' (dt outside DT_RANGE:
        the token and its dt are returned, the caller marks the frame invalid).  ts_past >= ts0 -> CausalityError."""
        j = int(j)
        if j < 1:
            raise ValueError(f"past(): j must be >= 1 (causal), got {j}")
        log_, pos, ts0, _ = self.t0(token)
        pt = self.bypos.get((log_, pos - j))
        if pt is None:
            return None, float("nan"), "not_in_parquet"
        ts_p = self.rows[pt][2]
        if ts_p >= ts0:
            raise CausalityError(f"{token}: frame at pos - {j} ({pt}) has timestamp {ts_p} >= t0 {ts0}")
        dt = (ts0 - ts_p) * 1e-6
        if not (DT_RANGE[0] < dt < DT_RANGE[1]):
            return pt, float(dt), "dt_range"
        return pt, float(dt), ""


# ----------------------------------------------------------------------------------------------- sources
class BevfusionObjectSource:
    """BEVFusion (DET teacher) detections at t0, t-0.5, t-1.0 s as canonical rows (SPEC 3.1-3.3).

    subset 'navtrain' (navtrain_train / navtrain_val tokens) or 'navtest'.  Two read-only TeacherDetCache roots
    (kind current / future; manifest, sha head, class names, proposals and point_cloud_range checked) + the frames
    parquet of the same subset.  Picklable (fork pools)."""

    name = "bevfusion"
    roi = ROI["bevfusion"]
    n_frames = N_FRAMES

    def __init__(self, subset: str, current_root=None, future_root=None, frames_path=None):
        if subset not in FT.INDEX_SPLIT:
            raise ValueError(f"subset must be one of {sorted(FT.INDEX_SPLIT)}, got {subset!r}")
        self.subset = subset
        self.current = FT.TeacherDetCache(current_root or FT.TEACHER_ROOTS[subset], kind="current")
        self.future = FT.TeacherDetCache(future_root or FT.FUTURE_ROOTS[subset], kind="future")
        if self.current.sha_head != self.future.sha_head:
            raise ValueError(f"sha heads differ: current {self.current.sha_head} vs future {self.future.sha_head}")
        if self.current.pcr != self.future.pcr:
            raise ValueError(f"point_cloud_range differs: {self.current.pcr} vs {self.future.pcr}")
        if tuple(self.current.roi) != tuple(self.roi):
            raise ValueError(f"cache ROI {self.current.roi} != ROI['bevfusion'] {self.roi}")
        sp = FT.INDEX_SPLIT[subset]
        self.frames_path = Path(frames_path or FT.INDEX_ROOT / f"future_frames_{sp}_{FT.K_INDEX}f.parquet")
        self.index = CausalPastIndex(self.frames_path)

    def read_raw(self, token: str) -> Tuple[Optional[Dict[str, np.ndarray]], int]:
        """raw boxes of one frame token: current-frame cache first, then the '_future'-named cache ->
        (dict(pred_boxes_3d, pred_scores_3d, pred_labels_3d) | None, src code 0/1/2)."""
        if self.current.has(token):
            return self.current.load_boxes(token), SRC_CURRENT
        if self.future.has(token):
            return self.future.load_boxes(token), SRC_FUTURE
        return None, SRC_NONE

    def frames(self, token: str, dtype=np.float32, k_obj: int = K_OBJ) -> Dict[str, np.ndarray]:
        """-> obj [3, k_obj, 17] (dtype), mask bool [3, k_obj], valid bool [3], dt f32 [3] (0, dt1, dt2; NaN where
        invalid), src i8 [3], n_raw i16 [3], why <U16 [3], frame_token <U16 [3], n_nonfinite i16 [3], dt_raw f64 [3]
        (the timestamp offset also for 'dt_range' frames; NaN without a parquet row).

        valid[0] is True whenever the t0 npz was read.  A t0 token without a parquet row raises KeyError."""
        obj = np.zeros((N_FRAMES, k_obj, D_OBJ), dtype)
        mask = np.zeros((N_FRAMES, k_obj), bool)
        valid = np.zeros(N_FRAMES, bool)
        dt = np.full(N_FRAMES, np.nan, np.float32)
        src = np.zeros(N_FRAMES, np.int8)
        n_raw = np.zeros(N_FRAMES, np.int16)
        n_bad = np.zeros(N_FRAMES, np.int16)
        why = np.full(N_FRAMES, "", "<U16")
        ftok = np.full(N_FRAMES, "", "<U16")
        dt_raw = np.full(N_FRAMES, np.nan, np.float64)
        _, _, _, pose0 = self.index.t0(token)
        for f in range(N_FRAMES):
            if f == 0:
                tok_f, age, w = token, 0.0, ""
            else:
                tok_f, age, w = self.index.past(token, f)
            if tok_f is not None:
                ftok[f] = tok_f
                dt_raw[f] = age
            if w:
                why[f] = w
                continue
            d, s = self.read_raw(tok_f)
            if d is None:
                why[f] = "no_npz"
                continue
            pose_k = pose0 if f == 0 else self.index.pose(tok_f)
            rows, info = teacher_frame_rows(d["pred_boxes_3d"], d["pred_scores_3d"], d["pred_labels_3d"], pose_k,
                                            pose0, age, SCORE_THR[f], k_obj=k_obj, return_info=True)
            m = rows.shape[0]
            obj[f, :m] = rows.astype(dtype)
            mask[f, :m] = True
            valid[f] = True
            dt[f] = age
            src[f] = s
            n_raw[f] = info["n_raw"]
            n_bad[f] = info["n_nonfinite"]
        return dict(obj=obj, mask=mask, valid=valid, dt=dt, src=src, n_raw=n_raw, why=why, frame_token=ftok,
                    n_nonfinite=n_bad, dt_raw=dt_raw)


class V2DetObjectSource:
    """Stub (Step 1 does not build it).  Conversion recorded for later (a6_v2_forecast.py 12-15, 58-59): v2 student
    detections, 300 boxes in SSR axes -> N frame: x = y_ssr, y = -x_ssr, heading = -syaw - pi,
    (vx, vy) = (vy_ssr, -vx_ssr); t0 only (frame_valid[1:] False); ROI and score threshold to be pinned then."""

    name = "v2det"

    def __init__(self, *a, **kw):
        raise NotImplementedError("v2det object source is not built in Step 1 (SPEC 4.2)")


class GtObjectSource:
    """Stub (Step 1 does not build it).  PRIVILEGED.  Needs a causal GT-history build from the raw logs (frames
    cur-2..cur) under D3, cropped to the source ROI and selected by score / distance to the origin only (never by the
    GT file order, which is sorted by distance to the GT ego path: a leak)."""

    name = "gt"
    privileged = True

    def __init__(self, *a, **kw):
        raise NotImplementedError("gt object source is not built in Step 1 (SPEC 4.2)")


# ----------------------------------------------------------------------------------------------- targets
def match_targets(obj0: np.ndarray, mask0: np.ndarray, gt_obj: Dict, return_stats: bool = False):
    """CLEAN t0 rows [K, 17] + mask [K] + gt_future objects -> fc_xyh f32 [K, 10, 3], fc_state i8 [K, 10],
    fc_track i32 [K] (SPEC 3.5).  fc_track is the matched GT row for AGENT slots only (-1 otherwise).

    ``return_stats`` adds dict(n_agent, n_matched, n_cross (vehicle <-> VRU), n_static_matched, n_gt_obs0,
    n_gt_future_first (GT tracks OBS at t = 0 only through the singleton hold, first annotated later: not matched),
    n_single_matched (matched agent slots whose track is a singleton: FC_SINGLE))."""
    from scipy.optimize import linear_sum_assignment

    obj0 = np.asarray(obj0, np.float64)
    mask0 = np.asarray(mask0, bool)
    K = obj0.shape[0]
    fc_xyh = np.zeros((K, FC_STEPS, 3), np.float32)
    fc_state = np.full((K, FC_STEPS), FC_NONE, np.int8)
    fc_track = np.full(K, -1, np.int32)
    cls = row_classes(obj0)
    ag = np.isin(cls, _AGENT) & mask0
    st = dict(n_agent=int(ag.sum()), n_matched=0, n_cross=0, n_static_matched=0, n_gt_obs0=0)
    slots = np.flatnonzero(mask0)
    A = len(gt_obj["track"])
    st.update(n_gt_future_first=0, n_single_matched=0)
    if A and slots.size:
        bz, sz = GF.query(gt_obj, np.zeros(1), dtype=np.float64)
        gmeta = np.asarray(gt_obj["meta"])
        first_after = gmeta[:, GF.MT_FIRST].astype(np.int64) > 0
        single = gmeta[:, GF.MT_SINGLE].astype(bool)
        st["n_gt_future_first"] = int(((sz[:, 0] == GF.OBS) & first_after).sum())
        pres0 = (sz[:, 0] == GF.OBS) & ~first_after          # never match against a future pose (singleton rule i)
        gi = np.flatnonzero(pres0)
        st["n_gt_obs0"] = int(gi.size)
        if gi.size:
            gag = gmeta[:, GF.MT_AGENT].astype(bool)
            b = obj0[slots]
            sl_ag = np.isin(cls[slots], _AGENT)
            dd = np.hypot(b[:, None, C_X] - bz[gi, 0, 0][None], b[:, None, C_Y] - bz[gi, 0, 1][None])
            cost = np.where((sl_ag[:, None] == gag[gi][None]) & (dd < MATCH_GATE_M), dd, 1e6)
            ri, ci = linear_sum_assignment(cost)
            ok = cost[ri, ci] < 1e5
            ri, ci = ri[ok], gi[ci[ok]]
            is_ag = sl_ag[ri]
            st["n_static_matched"] = int((~is_ag).sum())
            ri, ci = ri[is_ag], ci[is_ag]
            if ri.size:
                bx, state, unk = GF.query(gt_obj, np.asarray(FC_T, np.float64), dtype=np.float64, return_unknown=True)
                s_idx = slots[ri]
                fc_xyh[s_idx] = bx[ci, :, :3].astype(np.float32)
                stt = np.where(unk[ci], FC_BEYOND, np.where(state[ci] == GF.OBS, FC_OBS, FC_ABSENT))
                sg = single[ci]
                stt = np.where(sg[:, None] & (stt == FC_OBS), FC_SINGLE, stt)      # singleton rule ii
                st["n_single_matched"] = int(sg.sum())
                fc_state[s_idx] = stt.astype(np.int8)
                fc_track[s_idx] = ci.astype(np.int32)
                gcls = np.asarray(gt_obj["meta"])[ci, GF.MT_CLASS].astype(np.int64)
                st["n_matched"] = int(ri.size)
                st["n_cross"] = int(((_GROUP[cls[s_idx]] == 0) != (gcls == 0)).sum())
    if return_stats:
        return fc_xyh, fc_state, fc_track, st
    return fc_xyh, fc_state, fc_track


# ----------------------------------------------------------------------------------------------- CA reference
def ca_history(obj: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """a3 ``Worker.history`` on canonical rows: per t0 slot (acc_long [K], has_hist [K]).  Past frames 2 then 1
    (prefer t-1.0 s); dt of frame f = obj[f, :, 16] (the age column); agents only; same agent/static group; 1.5 m."""
    obj = np.asarray(obj, np.float64)
    mask = np.asarray(mask, bool)
    valid = np.asarray(valid, bool)
    K = obj.shape[1]
    m0 = mask[0] & bool(valid[0])
    b0 = obj[0]
    acc = np.zeros(K)
    has = np.zeros(K, bool)
    v0 = b0[:, [C_VX, C_VY]]
    sp0 = np.hypot(b0[:, C_VX], b0[:, C_VY])
    u0 = v0 / np.maximum(sp0, 1e-6)[:, None]
    grp0 = row_agent(b0) & m0
    for j in (2, 1):                                                   # prefer the 1.0-s window
        if j >= obj.shape[0] or not valid[j]:
            continue
        sel = mask[j]
        if not sel.any():
            continue
        bp = obj[j][sel]
        dtp = float(bp[0, C_AGE])
        if dtp <= 0:
            raise CausalityError(f"ca_history: past frame {j} has age {dtp} <= 0")
        pred = b0[:, :2] - v0 * dtp
        dd = np.hypot(pred[:, None, 0] - bp[None, :, 0], pred[:, None, 1] - bp[None, :, 1])
        same = row_agent(b0)[:, None] == row_agent(bp)[None]
        dd = np.where(same, dd, np.inf)
        jj = dd.argmin(1)
        ok = (dd[np.arange(K), jj] < HIST_GATE_M) & ~has & grp0
        vp = bp[jj][:, [C_VX, C_VY]]
        a = ((v0 - vp) * u0).sum(-1) / dtp
        a = np.where(sp0 > ACC_MIN_SPEED, np.clip(a, -ACC_CLIP, ACC_CLIP), 0.0)
        acc = np.where(ok, a, acc)
        has |= ok
    return acc, has


def ca_positions(c0: np.ndarray, v0: np.ndarray, acc: np.ndarray, t: Sequence[float]) -> np.ndarray:
    """a3 world('CA') positions [K, T, 2]: dist = |v0| te + 0.5 a te^2, te = min(t, tstop), tstop = -|v0| / a for
    a < -1e-6, c = c0 + u0 dist (heading held)."""
    c0 = np.asarray(c0, np.float64)
    v = np.asarray(v0, np.float64)
    tau = np.asarray(t, np.float64)
    T = tau.shape[0]
    c = np.broadcast_to(c0[None, :, :2], (T, len(c0), 2)).copy()
    sp0 = np.hypot(v[:, 0], v[:, 1])
    u = v / np.maximum(sp0, 1e-6)[:, None]
    tt = tau[:, None]
    a = np.asarray(acc, np.float64)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tstop = np.where(a < -1e-6, -sp0[None] / a, np.inf)
    te = np.minimum(tt, tstop)
    dist = sp0[None] * te + 0.5 * a * te ** 2
    c += u[None] * dist[..., None]
    return c.transpose(1, 0, 2)


def ca_forecast_np(obj: np.ndarray, mask: np.ndarray, valid: np.ndarray, t: Optional[Sequence[float]] = None
                   ) -> Dict[str, np.ndarray]:
    """numpy reference of SPEC 3.6 -> xy f64 [K, T, 2] at t (default FC_T), acc [K], has [K], agent [K] (valid agent
    t0 slots).  Statics and padding are held at c0 (padding rows are zero)."""
    obj = np.asarray(obj, np.float64)
    mask = np.asarray(mask, bool)
    tau = np.asarray(FC_T if t is None else t, np.float64)
    acc, has = ca_history(obj, mask, valid)
    b0 = obj[0]
    agent = row_agent(b0) & mask[0] & bool(np.asarray(valid)[0])
    xy = ca_positions(b0[:, :2], b0[:, [C_VX, C_VY]], acc, tau)
    hold = np.broadcast_to(b0[:, None, :2], xy.shape)
    xy = np.where(agent[:, None, None], xy, hold)
    return dict(xy=xy, acc=acc, has=has, agent=agent)


__all__ = ["OBJECTS_VERSION", "WHY", "WHY_CODE", "SRC_NONE", "SRC_CURRENT", "SRC_FUTURE", "CausalityError",
           "row_classes", "row_agent", "canonical_rows", "teacher_frame_rows", "CausalPastIndex",
           "BevfusionObjectSource", "V2DetObjectSource", "GtObjectSource", "match_targets", "ca_history",
           "ca_positions", "ca_forecast_np"]
