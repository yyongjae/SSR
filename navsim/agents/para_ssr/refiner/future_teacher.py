"""Future teacher targets (report 40 §4 step 8): BEVFusion detections on FUTURE log frames moved into the t0 N frame,
the GT / teacher future occupancy rasters on the S grid, and their readers.

Contents
  TeacherDetCache     one read-only BEVFusion 50x100 cache root (current-frame or '_future'), manifest-checked;
                      reads only the detection members (dense_heatmap, pred_boxes_3d, pred_scores_3d, pred_labels_3d);
                      ``load_bev`` refuses caches written with --drop-bev (bev_feature_stored = false).
  FutureTeacherCache  t0 token -> its K future frames (index json) -> detections (future cache first, then the
                      current-frame cache) -> N frame (poses from the frames parquet of build_future_index.py; no log
                      pickles are read at run time).  ``future_boxes(t0)``.
  teacher_raster / gt_raster   per-token occupancy rasters occ [2K, 50, 100] u8 (agent / static per k), unknown
                      [K, 50, 100] u8, valid [K], dt [K], frames [K], src [K].
  FutureRasterCache   reader of $D/future_raster/<source>/<tok[:2]>/<tok>.npz (step 10 consumer; raises on a missing
                      token, never zero-fills, so e2e.GTLoader's kd_ok masking keeps working).
  TeacherCache (data.py) is NOT touched: its '_future' refusal stays.

Frames, time
  N frame   NAVSIM ego (rear-axle) frame at t0: x forward, y left, heading CCW (gt_future docstring).
  pose(f)   gt_future.frame_pose(log frame f) = (ego2global_translation[:2], yaw of ego2global_rotation), the frame's
            OWN pose (for future frame k: the log entry of that future token).  Stored bit-exactly in the frames
            parquet (build_future_index.py).
  lidar2ego identity in every NAVSIM frame (bevfusion navsim_converter.py:323-325, navsim_dataset.py:8-11; 0 of
            798,141 log frames differ) -> teacher boxes live in the ego frame of their own frame.  The frames table
            carries a per-frame flag; a non-identity row raises when used.
  k         channel k = LOG FRAME OFFSET k (k = 1..K), the rule of make_token_lists.py and gt_future (index based);
            the actual elapsed time is carried in ``dt[k]`` = (timestamp_k - timestamp_t0) * 1e-6 s
            (|dt - 0.5k| > 0.1 s for 0.47 % of E2E entries, 0.22 % navtest: log gaps).

Teacher box conversion (evidence: bevfusion tools/data_converter/navsim_converter.py:176-192, navsim_dataset.py:349-350;
  rescore_teacher_detection.py is absent on this server, so the conversion is re-derived from the converter and checked
  empirically: centre error median 4-7 cm at every k in- and out-of-sample, heading median 0.010 rad, dx-W / dy-L
  medians 0.000 / -0.01 m)
  raw pred_boxes_3d = (x_fwd, y_left, z_bottom, dx = WIDTH, dy = LENGTH, dz = height, yaw, vx, vy)  (frame-k ego frame)
  heading_navsim = wrap(-yaw - pi/2)  (involution of the converter's yaw = -heading - pi/2)
  L = dy, W = dx, H = dz; z is the BOTTOM (mmdet3d 0.x boxes.tensor, cache_teacher_future.py:260) in BOTH cache
  generations -- the manifests' "z_gravity_centre" is wrong (errata; measured on current-frame vehicles: teacher z -
  (gt z - H/2) median 0.001-0.04 m vs teacher z - gt z -0.85..-0.94 m); z_centre = z + dz/2.
  (vx, vy) = absolute velocity in frame-k ego axes (converter copies gt_velocity_3d[:, :2]) -> rotation only.
  Output column order everywhere = NAVSIM gt_boxes order + velocity: (x, y, z, L, W, H, heading, vx, vy)
  (BX_* indices), so rows go straight into gt_future.box_corners (cx, cy, h, L, W).
  local -> N copies gt_future.build_from_frames:149-160 operation for operation (rotate by h_k, translate, wrap twice,
  then _to_n with pose0, heading - h0, velocity rotated by -h0).

Classes
  CLASS_NAMES = gt_future.CLASS_NAMES[:7] == bevfusion configs/navsim/default.yaml object_classes == every cache
  manifest (checked at open).  AGENT_LABELS = (0, 1, 2) (vehicle, pedestrian, bicycle; nuplan AGENT_TYPES, the GT
  penalty's agent flag meta[:, MT_AGENT], surrogate W_AGENT 1.0 / W_STATIC 0.5), STATIC_LABELS = (3, 4, 5, 6).

S grid (adapters.py:32-38, 78-80; written directly in S layout -- no teacher_to_s_grid flip on rasters)
  row r -> x = (r + 0.5) * 0.64 (0..32 m forward); col c -> y_left = 32 - (c + 0.5) * 0.64 (col 0 = 32 m LEFT).

Raster rule (both sources, float64): cell (r, c) of channel ch at k is occupied iff
  (a) its centre lies inside the CLOSED footprint of a box (|d.u| <= L/2 + 1e-6 and |d.v| <= W/2 + 1e-6), or
  (b) it is the cell containing a box centre (centre_cell) -- an object whose centre is in the grid never vanishes
      (45 % of GT objects are smaller than one cell in both L and W).
  A box with a finite centre but non-finite / non-positive size gets rule (b) only and is counted in n_bad_box.
  Occupancy is NOT masked by unknown; teacher boxes are NOT ROI-filtered for occupancy.
  Teacher boxes: score >= score_thr (0.3).  GT boxes: gt_future.query at t = 0.5k, state OBS ('query', the
  metric-cache / GT-penalty rule: singletons held at every t, interior gaps interpolated) or annotated at keyframe k
  ('annotated', kf present, L, W of first appearance).

Unknown rule
  teacher: cell centre (N) -> global (pose0) -> frame-k ego frame (pose_k) outside point_cloud_range[0,1,3,4]
           = [0, 32] x [-32, 32] (closed) of the LOGGED ego pose at frame k.  Occlusion is not modelled.  Teacher
           training targets were all in-ROI boxes (camera_fov.enabled = false in runs/navsim-fusion-50x100/
           configs.yaml), so there is no camera-FOV wedge to add.
  gt:      gt_future.unknown_space at t = 0.5k (t > t_avail, > 75 m from the GT ego at t, > R - 10 m from the origin).
  invalid k (either source): occ 0, unknown 1, dt NaN, frames '', src 0.

src codes: 0 none / invalid, 1 current-frame teacher cache (in-sample: a navtrain token frame the teacher was trained
on), 2 future teacher cache (out-of-sample), 3 GT objects.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import gt_future as G
from .adapters import BEV_H, BEV_W, CELL, s_grid_cell, teacher_to_s_grid
from .data import DATA_ROOT, REPO, TEACHER_LAYOUT, TEACHER_ROOTS, TEACHER_SHA_HEAD

# ----------------------------------------------------------------------------------------------- constants
TC_ROOT = Path("/home/external-user/datasets/teacher_cache/bevfusion")
FUTURE_ROOTS = {
    "navtrain": TC_ROOT / "cache_train_50x100_future",
    "navtest": TC_ROOT / "cache_val_50x100_future",
}
INDEX_ROOT = DATA_ROOT / "future_index"
RASTER_ROOT = DATA_ROOT / "future_raster"
INDEX_SPLIT = {"navtrain": "train", "navtest": "test"}
TEACHER_RUN_CONFIG = Path("/home/external-user/yongjae/bevfusion/runs/navsim-fusion-50x100/configs.yaml")
BEVFUSION_CLASS_CONFIG = Path("/home/external-user/yongjae/bevfusion/configs/navsim/default.yaml")

K_INDEX = 10            # horizon of the stored index (frames)
K_DEFAULT = 10          # raster horizon (5 s); 8 = 4 s
SCORE_THR = 0.3
RASTER_EPS = 1e-6       # [m] closed-footprint tolerance
VERSION = "future_teacher_v1"
RASTER_VERSION = "future_raster_v1"
INDEX_VERSION = "future_index_v1"

NUM_PROPOSALS = 200
CLASS_NAMES = tuple(G.CLASS_NAMES[:7])
AGENT_LABELS = tuple(c for c in G.AGENT_CLASSES if c < 7)                 # (0, 1, 2)
STATIC_LABELS = tuple(c for c in range(len(CLASS_NAMES)) if c not in AGENT_LABELS)   # (3, 4, 5, 6)
CH_AGENT, CH_STATIC = 0, 1
SRC_NONE, SRC_CURRENT, SRC_FUTURE, SRC_GT = 0, 1, 2, 3
GT_PRESENCE = ("query", "annotated")

# box columns (NAVSIM gt_boxes order + velocity)
BX_X, BX_Y, BX_Z, BX_L, BX_W, BX_H, BX_HEAD, BX_VX, BX_VY = range(9)

DET_SPEC = {
    "dense_heatmap": ((len(CLASS_NAMES), BEV_H, BEV_W), np.float16),
    "pred_boxes_3d": ((NUM_PROPOSALS, 9), np.float32),
    "pred_scores_3d": ((NUM_PROPOSALS,), np.float32),
    "pred_labels_3d": ((NUM_PROPOSALS,), np.int16),
}
BOX_KEYS = ("pred_boxes_3d", "pred_scores_3d", "pred_labels_3d")
RASTER_KEYS = ("occ", "unknown", "valid", "dt", "frames", "src", "version", "source")
FRAME_COLS = ("token", "log", "pos", "frame_idx", "timestamp", "x", "y", "yaw", "lidar2ego_identity", "is_key")

ERRATA = {
    "z": "pred_boxes_3d z is the BOTTOM (mmdet3d 0.x boxes.tensor) in both the current-frame (cf3e02a) and future "
         "(ee05061) caches; their manifests' 'z_gravity_centre' is wrong (measured: teacher z - (gt z - H/2) median "
         "0.001-0.04 m on current-frame vehicles)",
    "note": "future-cache manifest note says 't+0.5..4 s'; the real range is k = 1..10 (0.5-5 s)",
    "dtypes": "future-cache manifest dtypes list bev_feature although it is not stored (--drop-bev)",
}
FOV_NOTE = ("teacher targets were all in-ROI boxes (camera_fov.enabled=false in runs/navsim-fusion-50x100/configs.yaml); "
            "the unknown rule is ROI only, occlusion not modelled")


# ----------------------------------------------------------------------------------------------- small helpers
def sha256_file(path: Union[str, Path], chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def inside_repo(path: Union[str, Path]) -> bool:
    """True if ``path`` (resolved, may not exist yet) lies inside the SSR repository."""
    p = Path(os.path.abspath(str(path)))
    rp = Path(os.path.realpath(str(p)))
    for repo in {REPO, Path(os.path.realpath(str(REPO)))}:
        for q in (p, rp):
            try:
                q.relative_to(repo)
                return True
            except ValueError:
                pass
    return False


def read_token_list(path: Union[str, Path]) -> List[str]:
    """Token list file -> tokens (order kept, duplicates removed keeping the first).
    .parquet/.pq: column 'token'; .yaml/.yml: list or {tokens: [...]} (scene_filter format); .json: list or dict (keys);
    .txt (other): first whitespace field of each non-empty line."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        import pandas as pd
        toks = pd.read_parquet(path, columns=["token"]).token.astype(str).tolist()
    elif suf in (".yaml", ".yml"):
        import yaml
        d = yaml.safe_load(path.read_text())
        toks = d["tokens"] if isinstance(d, dict) else d
    elif suf == ".json":
        d = json.loads(path.read_text())
        toks = list(d.keys()) if isinstance(d, dict) else d
    else:
        toks = [ln.split()[0] for ln in path.read_text().splitlines() if ln.strip()]
    return list(dict.fromkeys(str(t) for t in toks))


# ----------------------------------------------------------------------------------------------- conversion
def teacher_box_to_local(b, z: str = "bottom") -> np.ndarray:
    """raw teacher boxes [..., 9] (x, y, z_bottom, dx=W, dy=L, dz=H, yaw, vx, vy) -> local NAVSIM boxes [..., 9]
    (x, y, z_bottom | z_centre, L, W, H, heading (wrapped), vx, vy), float64, same (frame-k ego) frame."""
    b = np.asarray(b, np.float64)
    if b.shape[-1] != 9:
        raise ValueError(f"teacher boxes need 9 columns, got {b.shape}")
    out = np.empty(b.shape, np.float64)
    out[..., BX_X], out[..., BX_Y] = b[..., 0], b[..., 1]
    if z == "bottom":
        out[..., BX_Z] = b[..., 2]
    elif z == "centre":
        out[..., BX_Z] = b[..., 2] + 0.5 * b[..., 5]
    else:
        raise ValueError(f"z must be 'bottom' or 'centre', got {z!r}")
    out[..., BX_L], out[..., BX_W], out[..., BX_H] = b[..., 4], b[..., 3], b[..., 5]
    out[..., BX_HEAD] = G.wrap(-b[..., 6] - np.pi / 2)
    out[..., BX_VX], out[..., BX_VY] = b[..., 7], b[..., 8]
    return out


def teacher_box_to_navsim(b) -> np.ndarray:
    """fact_features.md name: heading = -yaw - pi/2, z = z + dz/2 (gravity centre, as NAVSIM gt_boxes), (L, W) order."""
    return teacher_box_to_local(b, z="centre")


def local_to_n(local, pose_k, pose0) -> np.ndarray:
    """local NAVSIM boxes [..., 9] in the ego frame of a frame with global pose ``pose_k`` -> N frame of ``pose0``.
    Same operations as gt_future.build_from_frames:149-160 (velocity: rotation only)."""
    b = np.asarray(local, np.float64)
    xk, yk, hk = (float(v) for v in pose_k)
    h0 = float(pose0[2])
    c, s = np.cos(hk), np.sin(hk)
    x, y = b[..., BX_X], b[..., BX_Y]
    gx = x * c - y * s + xk
    gy = x * s + y * c + yk
    gh = G.wrap(G.wrap(b[..., BX_HEAD] + hk))
    gvx = b[..., BX_VX] * c - b[..., BX_VY] * s
    gvy = b[..., BX_VX] * s + b[..., BX_VY] * c
    nx, ny = G._to_n(gx, gy, (float(pose0[0]), float(pose0[1]), h0))
    c0, s0 = np.cos(h0), np.sin(h0)
    out = b.copy()
    out[..., BX_X], out[..., BX_Y] = nx, ny
    out[..., BX_HEAD] = G.wrap(gh - h0)
    out[..., BX_VX], out[..., BX_VY] = c0 * gvx + s0 * gvy, -s0 * gvx + c0 * gvy
    return out


def n_to_local_xy(x, y, pose0, pose_k):
    """N-frame points (of pose0) -> ego frame of pose_k (global pose), float64 (teacher_unknown's chain)."""
    x0, y0, h0 = (float(v) for v in pose0)
    c0, s0 = np.cos(h0), np.sin(h0)
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    gx = x0 + c0 * x - s0 * y
    gy = y0 + s0 * x + c0 * y
    return G._to_n(gx, gy, tuple(float(v) for v in pose_k))


# ----------------------------------------------------------------------------------------------- S grid
def s_grid_centres() -> Tuple[np.ndarray, np.ndarray]:
    """cell-centre coordinates (X, Y) [50, 100] float64 in N: X = (r + 0.5) * 0.64, Y = 32 - (c + 0.5) * 0.64."""
    xs = (np.arange(BEV_H) + 0.5) * CELL
    ys = 32.0 - (np.arange(BEV_W) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return X, Y


_XY = None


def _centres():
    global _XY
    if _XY is None:
        _XY = s_grid_centres()
    return _XY


def centre_cell(x, y):
    """cell containing (x, y): r = floor(x / 0.64), c = floor((32 - y) / 0.64) (= floor(s_grid_cell + 0.5)).
    -> (r, c, inside) arrays; r = c = -1 where outside the grid or non-finite."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    with np.errstate(invalid="ignore"):
        r = np.floor(x / CELL)
        c = np.floor((32.0 - y) / CELL)
        ok = np.isfinite(r) & np.isfinite(c) & (r >= 0) & (r < BEV_H) & (c >= 0) & (c < BEV_W)
    r = np.where(ok, r, -1).astype(np.int64)
    c = np.where(ok, c, -1).astype(np.int64)
    return r, c, ok


def rasterize_boxes(cx, cy, h, L, W, eps: float = RASTER_EPS) -> Tuple[np.ndarray, int]:
    """Union raster of boxes (N frame, float64) on the S grid with the module's rule -> (occ bool [50, 100], n_bad_box).
    Rule (a) is evaluated only in each box's row/col window (corner bounding box, +-1 cell, clipped)."""
    cx, cy, h, L, W = (np.asarray(a, np.float64).reshape(-1) for a in (cx, cy, h, L, W))
    occ = np.zeros((BEV_H, BEV_W), bool)
    if cx.size == 0:
        return occ, 0
    fin_c = np.isfinite(cx) & np.isfinite(cy)
    good = fin_c & np.isfinite(h) & np.isfinite(L) & np.isfinite(W) & (L > 0) & (W > 0)
    n_bad = int((~good).sum())
    r0, c0, ok = centre_cell(cx[fin_c], cy[fin_c])                  # rule (b)
    occ[r0[ok], c0[ok]] = True
    g = np.flatnonzero(good)
    if g.size == 0:
        return occ, n_bad
    bx, by, bh, bl, bw = cx[g], cy[g], h[g], L[g], W[g]
    co = G.box_corners(np.stack([bx, by, bh, bl, bw], -1))          # [n, 4, 2]
    xmin, xmax = co[..., 0].min(1) - eps, co[..., 0].max(1) + eps
    ymin, ymax = co[..., 1].min(1) - eps, co[..., 1].max(1) + eps
    rlo = np.maximum(np.ceil(xmin / CELL - 0.5) - 1, 0)
    rhi = np.minimum(np.floor(xmax / CELL - 0.5) + 1, BEV_H - 1)
    clo = np.maximum(np.ceil((32.0 - ymax) / CELL - 0.5) - 1, 0)
    chi = np.minimum(np.floor((32.0 - ymin) / CELL - 0.5) + 1, BEV_W - 1)
    m = (rlo <= rhi) & (clo <= chi)
    if not m.any():
        return occ, n_bad
    bx, by, bh, bl, bw = bx[m], by[m], bh[m], bl[m], bw[m]
    rlo, clo = rlo[m].astype(np.int64), clo[m].astype(np.int64)
    nr = rhi[m].astype(np.int64) - rlo + 1
    nc = chi[m].astype(np.int64) - clo + 1
    WR, WC = int(nr.max()), int(nc.max())
    ar, ac = np.arange(WR), np.arange(WC)
    rr = rlo[:, None] + ar[None]                                     # [n, WR]
    cc = clo[:, None] + ac[None]                                     # [n, WC]
    vr = ar[None] < nr[:, None]
    vc = ac[None] < nc[:, None]
    px = (rr + 0.5) * CELL                                           # same formula as s_grid_centres
    py = 32.0 - (cc + 0.5) * CELL
    ch, sh = np.cos(bh)[:, None, None], np.sin(bh)[:, None, None]
    dx = px[:, :, None] - bx[:, None, None]                          # [n, WR, 1]
    dy = py[:, None, :] - by[:, None, None]                          # [n, 1, WC]
    du = dx * ch + dy * sh
    dv = -dx * sh + dy * ch
    inside = (np.abs(du) <= 0.5 * bl[:, None, None] + eps) & (np.abs(dv) <= 0.5 * bw[:, None, None] + eps)
    inside &= vr[:, :, None] & vc[:, None, :]
    b, i, j = np.nonzero(inside)
    occ[rr[b, i], cc[b, j]] = True
    return occ, n_bad


def labels_to_channel(labels) -> np.ndarray:
    """teacher labels -> channel (CH_AGENT / CH_STATIC); labels outside 0..6 raise."""
    lab = np.asarray(labels).astype(np.int64)
    if lab.size and ((lab < 0) | (lab >= len(CLASS_NAMES))).any():
        bad = sorted(set(lab[(lab < 0) | (lab >= len(CLASS_NAMES))].tolist()))
        raise ValueError(f"teacher labels outside 0..{len(CLASS_NAMES) - 1}: {bad}")
    return np.where(np.isin(lab, AGENT_LABELS), CH_AGENT, CH_STATIC)


def teacher_unknown(pose0, pose_k, roi=(0.0, -32.0, 32.0, 32.0)) -> np.ndarray:
    """bool [50, 100]: cell centre outside the closed teacher ROI (x0, y0, x1, y1) of the logged ego pose at frame k."""
    X, Y = _centres()
    lx, ly = n_to_local_xy(X, Y, pose0, pose_k)
    known = (lx >= roi[0]) & (lx <= roi[2]) & (ly >= roi[1]) & (ly <= roi[3])
    return ~known


def gt_unknown(obj, K: int, reach: float = G.REACH_M, radius_margin: Optional[float] = G.RADIUS_MARGIN) -> np.ndarray:
    """bool [K, 50, 100]: gt_future.unknown_space at the cell centres, t = 0.5 k, k = 1..K."""
    X, Y = _centres()
    cells = np.stack([X.ravel(), Y.ravel()], -1)                     # [5000, 2]
    xy = np.broadcast_to(cells[:, None, :], (cells.shape[0], K, 2))
    t = G.KF_DT * np.arange(1, K + 1)
    u = G.unknown_space(obj, xy, t, reach=reach, radius_margin=radius_margin)   # [5000, K]
    return np.ascontiguousarray(u.T.reshape(K, BEV_H, BEV_W))


def _empty_raster(K: int) -> Dict[str, np.ndarray]:
    return dict(occ=np.zeros((2 * K, BEV_H, BEV_W), np.uint8), unknown=np.ones((K, BEV_H, BEV_W), np.uint8),
                valid=np.zeros(K, bool), dt=np.full(K, np.nan, np.float32), frames=np.full(K, "", "<U16"),
                src=np.zeros(K, np.int8))


def _stats_init(K):
    return dict(n_agent=np.zeros(K, np.int64), n_static=np.zeros(K, np.int64), n_bad_box=0,
                occ_agent=np.zeros(K, np.int64), occ_static=np.zeros(K, np.int64), unk_frac=np.ones(K, np.float64))


def teacher_raster(ftc: "FutureTeacherCache", t0: str, k: Optional[int] = None,
                   score_thr: Optional[float] = None) -> Dict[str, object]:
    """teacher future raster of token t0 (module rules) -> dict(occ, unknown, valid, dt, frames, src, stats)."""
    fb = ftc.future_boxes(t0, k=k, score_thr=score_thr)
    K = len(fb["valid"])
    out = _empty_raster(K)
    st = _stats_init(K)
    for i in range(K):
        if not fb["valid"][i]:
            continue
        keep = fb["keep"][i]
        b = fb["boxes"][i][keep]
        ch = labels_to_channel(fb["labels"][i][keep])
        for c, key in ((CH_AGENT, "agent"), (CH_STATIC, "static")):
            m = ch == c
            occ, nb = rasterize_boxes(b[m, BX_X], b[m, BX_Y], b[m, BX_HEAD], b[m, BX_L], b[m, BX_W])
            out["occ"][2 * i + c] = occ
            st[f"n_{key}"][i] = int(m.sum())
            st[f"occ_{key}"][i] = int(occ.sum())
            st["n_bad_box"] += nb
        u = teacher_unknown(fb["pose0"], fb["poses"][i], ftc.roi)
        out["unknown"][i] = u
        st["unk_frac"][i] = float(u.mean())
        out["valid"][i] = True
        out["dt"][i] = fb["dt"][i]
        out["frames"][i] = fb["frames"][i]
        out["src"][i] = fb["src"][i]
    out["stats"] = st
    return out


def gt_raster(obj, k: int = K_DEFAULT, presence: str = "query", reach: float = G.REACH_M,
              radius_margin: Optional[float] = G.RADIUS_MARGIN, frames: Optional[Sequence] = None,
              dt: Optional[Sequence] = None) -> Dict[str, object]:
    """GT future raster from gt_future objects (module rules); ``frames`` / ``dt`` (length K, from the index / frames
    table) are passed through for valid k.  valid[k] = (0.5 k <= t_avail + EPS_T), i.e. k <= n_kf - 1."""
    K = int(k)
    if not 1 <= K <= G.N_KF - 1:
        raise ValueError(f"K must be in 1..{G.N_KF - 1}, got {K}")
    if presence not in GT_PRESENCE:
        raise ValueError(f"presence must be one of {GT_PRESENCE}, got {presence!r}")
    out = _empty_raster(K)
    st = _stats_init(K)
    t = G.KF_DT * np.arange(1, K + 1)
    valid = t <= G.t_avail(obj) + G.EPS_T
    meta = np.asarray(obj["meta"])
    agent = meta[:, G.MT_AGENT] > 0 if len(meta) else np.zeros(0, bool)
    kf = np.asarray(obj["kf"], np.float64)
    if presence == "query":
        boxes, state = G.query(obj, t, dtype=np.float64)              # [A, K, 5]
        pres = state == G.OBS
        cx, cy, hh, LL, WW = (boxes[..., j] for j in range(5))
    else:
        first = np.asarray(obj["first"], np.float64)
        pres = kf[:, 1:K + 1, G.KF_PRESENT] == 1
        cx, cy, hh = kf[:, 1:K + 1, G.KF_X], kf[:, 1:K + 1, G.KF_Y], kf[:, 1:K + 1, G.KF_H]
        LL = np.broadcast_to(first[:, None, G.FI_L], pres.shape)
        WW = np.broadcast_to(first[:, None, G.FI_W], pres.shape)
    unk = gt_unknown(obj, K, reach=reach, radius_margin=radius_margin)
    for i in range(K):
        if not valid[i]:
            continue
        for c, key, msk in ((CH_AGENT, "agent", agent), (CH_STATIC, "static", ~agent)):
            sel = pres[:, i] & msk
            occ, nb = rasterize_boxes(cx[sel, i], cy[sel, i], hh[sel, i], LL[sel, i], WW[sel, i])
            out["occ"][2 * i + c] = occ
            st[f"n_{key}"][i] = int(sel.sum())
            st[f"occ_{key}"][i] = int(occ.sum())
            st["n_bad_box"] += nb
        out["unknown"][i] = unk[i]
        st["unk_frac"][i] = float(unk[i].mean())
        out["valid"][i] = True
        out["src"][i] = SRC_GT
        if frames is not None and frames[i]:
            out["frames"][i] = str(frames[i])
        if dt is not None and dt[i] is not None and np.isfinite(dt[i]):
            out["dt"][i] = np.float32(dt[i])
    out["stats"] = st
    return out


# ----------------------------------------------------------------------------------------------- caches
class FutureIndex:
    """index json (token -> list of >= k future frame tokens or null) + frames parquet (build_future_index.py) ->
    frame tokens, poses, timestamps.  Used by FutureTeacherCache and by the GT raster (frames / dt per k)."""

    def __init__(self, index_path, frames_path, k: int = K_DEFAULT):
        import pandas as pd

        self.k = int(k)
        self.index_path = Path(index_path)
        self.frames_path = Path(frames_path)
        self.index: Dict[str, list] = json.loads(self.index_path.read_text())
        short = next((t for t, v in self.index.items() if not isinstance(v, list) or len(v) < self.k), None)
        if short is not None:
            raise ValueError(f"{self.index_path}: entry {short} shorter than k={self.k}")
        df = pd.read_parquet(self.frames_path)
        miss = [c for c in FRAME_COLS if c not in df.columns]
        if miss:
            raise ValueError(f"{self.frames_path}: missing columns {miss}")
        self.frames: Dict[str, tuple] = dict(zip(
            df.token.astype(str).tolist(),
            zip(df["log"].astype(str).tolist(), df["pos"].astype(int).tolist(), df["timestamp"].astype("int64").tolist(),
                df["x"].astype(float).tolist(), df["y"].astype(float).tolist(), df["yaw"].astype(float).tolist(),
                df["lidar2ego_identity"].astype(bool).tolist())))

    def frames_dt(self, t0: str, k: Optional[int] = None):
        """-> (frame tokens [K] ('' if null / no row), dt [K] seconds (NaN if null / no row)) or (None, None) if t0 is
        not an index key or has no pose row."""
        K = self.k if k is None else int(k)
        v = self.index.get(t0)
        r0 = self.frames.get(t0)
        if v is None or r0 is None:
            return None, None
        fr, dt = [], []
        for tok in v[:K]:
            r = self.frames.get(tok) if tok is not None else None
            fr.append(tok if r is not None else "")
            dt.append((r[2] - r0[2]) * 1e-6 if r is not None else float("nan"))
        return fr, dt


class TeacherDetCache:
    """One read-only BEVFusion 50x100 cache root.  kind = 'current' (no '_future' in the root's basename) or 'future'
    ('_future' required) -- guards against swapped arguments.  Manifest checks as data.TeacherCache plus class names,
    num_proposals and point_cloud_range."""

    def __init__(self, root: Union[str, Path], expect_sha_head: str = TEACHER_SHA_HEAD, kind: str = "current"):
        root = Path(root)
        if kind not in ("current", "future"):
            raise ValueError(f"kind must be 'current' or 'future', got {kind!r}")
        names = (root.name, Path(os.path.realpath(str(root))).name)
        fut = any("_future" in n for n in names)
        if kind == "future" and not fut:
            raise ValueError(f"future teacher cache {root}: basename has no '_future' (swapped roots?)")
        if kind == "current" and fut:
            raise ValueError(f"current-frame teacher cache {root}: basename contains '_future' (swapped roots?)")
        mf = root / "manifest.json"
        if not mf.is_file():
            raise FileNotFoundError(f"teacher cache without manifest: {mf}")
        man = json.loads(mf.read_text())
        sha = str(man.get("checkpoint_sha256_head", ""))
        if sha != expect_sha_head:
            raise ValueError(f"teacher cache {root}: checkpoint_sha256_head {sha!r} != {expect_sha_head!r}")
        if man.get("layout") != TEACHER_LAYOUT:
            raise ValueError(f"teacher cache {root}: layout {man.get('layout')!r} != {TEACHER_LAYOUT!r}")
        if list(man.get("target_bev_shape", [])) != [BEV_H, BEV_W]:
            raise ValueError(f"teacher cache {root}: target_bev_shape {man.get('target_bev_shape')}")
        if list(man.get("class_names", [])) != list(CLASS_NAMES):
            raise ValueError(f"teacher cache {root}: class_names {man.get('class_names')} != {list(CLASS_NAMES)}")
        if int(man.get("num_proposals", -1)) != NUM_PROPOSALS:
            raise ValueError(f"teacher cache {root}: num_proposals {man.get('num_proposals')} != {NUM_PROPOSALS}")
        pcr = man.get("point_cloud_range")
        if pcr is None or len(pcr) != 6:
            raise ValueError(f"teacher cache {root}: point_cloud_range {pcr}")
        self.root = root
        self.kind = kind
        self.manifest = man
        self.manifest_path = mf
        self.sha_head = sha
        self.pcr = tuple(float(v) for v in pcr)
        self.roi = (self.pcr[0], self.pcr[1], self.pcr[3], self.pcr[4])
        self.bev_feature_stored = bool(man.get("bev_feature_stored", True))

    def path(self, token: str) -> Path:
        return self.root / "samples" / token[:2] / f"{token}.npz"

    def has(self, token: str) -> bool:
        return self.path(token).is_file()

    def _read(self, token: str, keys) -> Dict[str, np.ndarray]:
        with np.load(self.path(token), allow_pickle=False) as z:
            out = {k: z[k] for k in keys}
        for k, v in out.items():
            shp, dt = DET_SPEC[k]
            if v.shape != shp or v.dtype != dt:
                raise ValueError(f"{self.path(token)}: {k} {v.shape} {v.dtype} != {shp} {np.dtype(dt)}")
        return out

    def load_boxes(self, token: str) -> Dict[str, np.ndarray]:
        """pred_boxes_3d [200, 9] f32 (raw, frame's own ego frame), pred_scores_3d [200] f32, pred_labels_3d [200] i16."""
        return self._read(token, BOX_KEYS)

    def load_det(self, token: str, s_grid: bool = True) -> Dict[str, np.ndarray]:
        """detection members only (never bev_feature).  dense_heatmap [7, 50, 100] is in the grid of THIS frame's ego
        pose (for a future frame: frame k's grid, not t0's); S layout (col 0 = left) unless s_grid=False."""
        out = self._read(token, tuple(DET_SPEC))
        if s_grid:
            out["dense_heatmap"] = teacher_to_s_grid(out["dense_heatmap"])
        return out

    def load_bev(self, token: str, s_grid: bool = True) -> np.ndarray:
        if not self.bev_feature_stored:
            raise ValueError(f"teacher cache {self.root}: bev_feature not stored (--drop-bev, bev_feature_stored=false)")
        with np.load(self.path(token), allow_pickle=False) as z:
            x = z["bev_feature"]
        if x.shape != (256, BEV_H, BEV_W) or x.dtype != np.float16:
            raise ValueError(f"{token}: bev_feature {x.shape} {x.dtype}")
        return teacher_to_s_grid(x) if s_grid else x


class FutureTeacherCache:
    """t0 token -> teacher detections of its K future frames in the t0 N frame (see module docstring).

    index_path : json token -> list (>= k) of future frame tokens or null (build_future_index.py; only [:k] is used,
                 so K = 8 is a prefix of the 10-frame index).
    frames_path: parquet (token, log, pos, frame_idx, timestamp, x, y, yaw, lidar2ego_identity, is_key) =
                 frame_pose of every index key and every non-null future frame.
    Lookup order future -> current (report 40 line 408); ``strict`` also checks the other cache and raises on a token
    found in both (measured overlap 0).  Picklable (the per-instance LRU is dropped)."""

    def __init__(self, future_root, current_root, index_path, frames_path, k: int = K_DEFAULT,
                 score_thr: float = SCORE_THR, expect_sha_head: str = TEACHER_SHA_HEAD, lru: int = 64,
                 strict: bool = False):
        self.future = TeacherDetCache(future_root, expect_sha_head, kind="future")
        self.current = TeacherDetCache(current_root, expect_sha_head, kind="current")
        if self.future.sha_head != self.current.sha_head:
            raise ValueError(f"sha heads differ: future {self.future.sha_head} vs current {self.current.sha_head}")
        if self.future.pcr != self.current.pcr:
            raise ValueError(f"point_cloud_range differs: future {self.future.pcr} vs current {self.current.pcr}")
        if list(self.future.manifest["class_names"]) != list(self.current.manifest["class_names"]):
            raise ValueError("class_names differ between the future and the current cache")
        self.roi = self.future.roi
        self.sha_head = self.future.sha_head
        self.k = int(k)
        if not 1 <= self.k <= G.N_KF - 1:
            raise ValueError(f"k must be in 1..{G.N_KF - 1}, got {k}")
        self.score_thr = float(score_thr)
        self.strict = bool(strict)
        self.fidx = FutureIndex(index_path, frames_path, k=self.k)
        self.index_path, self.frames_path = self.fidx.index_path, self.fidx.frames_path
        self.index, self.frames = self.fidx.index, self.fidx.frames
        self.lru = int(lru)
        self._lru: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    @classmethod
    def for_subset(cls, subset: str = "navtrain", k: int = K_DEFAULT, **kw) -> "FutureTeacherCache":
        sp = INDEX_SPLIT[subset]
        kw.setdefault("index_path", INDEX_ROOT / f"future_index_{sp}_{K_INDEX}f.json")
        kw.setdefault("frames_path", INDEX_ROOT / f"future_frames_{sp}_{K_INDEX}f.parquet")
        return cls(kw.pop("future_root", FUTURE_ROOTS[subset]), kw.pop("current_root", TEACHER_ROOTS[subset]),
                   k=k, **kw)

    # pickling (fork pools): drop the LRU
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_lru"] = OrderedDict()
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    # --------------------------------------------------------------------------------- lookups
    def frame_tokens(self, t0: str, k: Optional[int] = None) -> List[Optional[str]]:
        K = self.k if k is None else int(k)
        if t0 not in self.index:
            raise KeyError(f"{t0}: not an index key ({self.index_path})")
        v = self.index[t0]
        if len(v) < K:
            raise ValueError(f"{t0}: index entry has {len(v)} < {K} frames")
        return list(v[:K])

    def frame_row(self, tok: str) -> Optional[tuple]:
        """(log, pos, timestamp_us, x, y, yaw, lidar2ego_identity) or None."""
        return self.frames.get(tok)

    def pose(self, tok: str) -> Optional[Tuple[float, float, float]]:
        r = self.frames.get(tok)
        if r is None:
            return None
        if not r[6]:
            raise ValueError(f"{tok}: lidar2ego is not the identity in its log frame")
        return (r[3], r[4], r[5])

    def source_of(self, tok: str, strict: Optional[bool] = None) -> int:
        strict = self.strict if strict is None else strict
        f = self.future.has(tok)
        if f and not strict:
            return SRC_FUTURE
        c = self.current.has(tok)
        if f and c:
            raise ValueError(f"{tok}: present in BOTH the future and the current-frame cache")
        return SRC_FUTURE if f else (SRC_CURRENT if c else SRC_NONE)

    def _boxes(self, tok: str, src: int) -> Dict[str, np.ndarray]:
        d = self._lru.get(tok)
        if d is not None:
            self._lru.move_to_end(tok)
            return d
        d = (self.future if src == SRC_FUTURE else self.current).load_boxes(tok)
        if self.lru > 0:
            self._lru[tok] = d
            while len(self._lru) > self.lru:
                self._lru.popitem(last=False)
        return d

    def load_det(self, tok: str, s_grid: bool = True) -> Dict[str, np.ndarray]:
        src = self.source_of(tok)
        if src == SRC_NONE:
            raise FileNotFoundError(f"{tok}: in neither teacher cache")
        d = (self.future if src == SRC_FUTURE else self.current).load_det(tok, s_grid=s_grid)
        d["src"] = np.int8(src)
        return d

    def future_boxes(self, t0: str, k: Optional[int] = None, score_thr: Optional[float] = None) -> Dict[str, np.ndarray]:
        """teacher boxes of the K future frames of t0 in the t0 N frame.

        boxes [K, 200, 9] f64 (x, y, z_bottom, L, W, H, heading, vx, vy; NaN rows where invalid), scores [K, 200] f32
        (0 where invalid), labels [K, 200] i16 (-1 where invalid), keep [K, 200] = valid & score >= thr,
        valid [K] (frame non-null AND npz in a cache AND pose row present), why [K] ('' | null_in_index | no_npz |
        no_pose), dt [K] f32 seconds from timestamps (NaN if invalid), frames [K] <U16 (frame token, '' if null),
        src [K] i8 (0/1/2), pose0 [3] f64, poses [K, 3] f64 (NaN if invalid), k_nominal_s [K] = 0.5 * (1..K).
        An unreadable npz raises."""
        K = self.k if k is None else int(k)
        thr = self.score_thr if score_thr is None else float(score_thr)
        fr = self.frame_tokens(t0, K)
        r0 = self.frames.get(t0)
        if r0 is None:
            raise KeyError(f"{t0}: no pose row in {self.frames_path}")
        pose0 = self.pose(t0)
        ts0 = r0[2]
        boxes = np.full((K, NUM_PROPOSALS, 9), np.nan, np.float64)
        scores = np.zeros((K, NUM_PROPOSALS), np.float32)
        labels = np.full((K, NUM_PROPOSALS), -1, np.int16)
        valid = np.zeros(K, bool)
        why = np.full(K, "", "<U16")
        dt = np.full(K, np.nan, np.float32)
        frames = np.full(K, "", "<U16")
        src = np.zeros(K, np.int8)
        poses = np.full((K, 3), np.nan, np.float64)
        for i, tok in enumerate(fr):
            if tok is None:
                why[i] = "null_in_index"
                continue
            frames[i] = tok
            s = self.source_of(tok)
            if s == SRC_NONE:
                why[i] = "no_npz"
                continue
            row = self.frames.get(tok)
            if row is None:
                why[i] = "no_pose"
                continue
            pk = self.pose(tok)
            d = self._boxes(tok, s)
            boxes[i] = local_to_n(teacher_box_to_local(d["pred_boxes_3d"]), pk, pose0)
            scores[i] = d["pred_scores_3d"]
            labels[i] = d["pred_labels_3d"]
            valid[i] = True
            src[i] = s
            poses[i] = pk
            dt[i] = (row[2] - ts0) * 1e-6
        keep = valid[:, None] & (scores >= thr)
        return dict(boxes=boxes, scores=scores, labels=labels, keep=keep, valid=valid, why=why, dt=dt, frames=frames,
                    src=src, pose0=np.asarray(pose0, np.float64), poses=poses,
                    k_nominal_s=G.KF_DT * np.arange(1, K + 1))

    def manifest_info(self) -> Dict[str, object]:
        def cache_info(c: TeacherDetCache):
            return dict(root=str(c.root), manifest_sha256=sha256_file(c.manifest_path),
                        git_commit=c.manifest.get("git_commit"), checkpoint=c.manifest.get("checkpoint"),
                        bev_feature_stored=c.bev_feature_stored)
        info = dict(version=VERSION, future=cache_info(self.future), current=cache_info(self.current),
                    sha_head=self.sha_head, class_names=list(CLASS_NAMES), point_cloud_range=list(self.future.pcr),
                    roi=list(self.roi), index=dict(path=str(self.index_path), sha256=sha256_file(self.index_path)),
                    frames=dict(path=str(self.frames_path), sha256=sha256_file(self.frames_path)),
                    k=self.k, score_thr=self.score_thr, lookup_order="future -> current", errata=ERRATA)
        if TEACHER_RUN_CONFIG.is_file():
            info["teacher_run_config"] = dict(path=str(TEACHER_RUN_CONFIG), sha256=sha256_file(TEACHER_RUN_CONFIG))
        return info


class FutureRasterCache:
    """Reader of build_future_raster.py output (<root>/<tok[:2]>/<tok>.npz + manifest.json).
    load_bev(token) -> float16 [C, 50, 100] on the S grid, C = 2K (+K with unknown); raises FileNotFoundError for a
    missing token (never zero-fills)."""

    def __init__(self, root: Union[str, Path], expect_source: Optional[str] = None, expect_k: Optional[int] = None,
                 with_unknown: bool = False):
        root = Path(root)
        mf = root / "manifest.json"
        if not mf.is_file():
            raise FileNotFoundError(f"future raster root without manifest: {mf}")
        man = json.loads(mf.read_text())
        cfg = man.get("config", man)
        if cfg.get("version") != RASTER_VERSION:
            raise ValueError(f"{root}: raster version {cfg.get('version')!r} != {RASTER_VERSION!r}")
        if list(cfg.get("grid", {}).get("shape", [])) != [BEV_H, BEV_W]:
            raise ValueError(f"{root}: grid shape {cfg.get('grid', {}).get('shape')}")
        if expect_source is not None and cfg.get("source") != expect_source:
            raise ValueError(f"{root}: source {cfg.get('source')!r} != {expect_source!r}")
        if expect_k is not None and int(cfg.get("K", -1)) != int(expect_k):
            raise ValueError(f"{root}: K {cfg.get('K')} != {expect_k}")
        self.root = root
        self.manifest = man
        self.config = cfg
        self.source = cfg.get("source")
        self.K = int(cfg["K"])
        self.with_unknown = bool(with_unknown)
        self.channels = 2 * self.K + (self.K if self.with_unknown else 0)
        self.sha_head = (cfg.get("teacher") or {}).get("sha_head", "gt")

    def path(self, token: str) -> Path:
        return self.root / token[:2] / f"{token}.npz"

    def has(self, token: str) -> bool:
        return self.path(token).is_file()

    def load(self, token: str) -> Dict[str, np.ndarray]:
        p = self.path(token)
        if not p.is_file():
            raise FileNotFoundError(f"future raster missing for {token}: {p}")
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def load_bev(self, token: str, s_grid: bool = True) -> np.ndarray:
        if not s_grid:
            raise ValueError("FutureRasterCache only serves the S grid")
        d = self.load(token)
        occ = d["occ"]
        if occ.shape != (2 * self.K, BEV_H, BEV_W):
            raise ValueError(f"{token}: occ {occ.shape} != {(2 * self.K, BEV_H, BEV_W)}")
        x = occ
        if self.with_unknown:
            x = np.concatenate([occ, d["unknown"]], 0)
        return np.ascontiguousarray(x.astype(np.float16))


__all__ = [
    "FUTURE_ROOTS", "INDEX_ROOT", "RASTER_ROOT", "INDEX_SPLIT", "K_INDEX", "K_DEFAULT", "SCORE_THR", "RASTER_EPS",
    "VERSION", "RASTER_VERSION", "INDEX_VERSION", "CLASS_NAMES", "AGENT_LABELS", "STATIC_LABELS", "CH_AGENT",
    "CH_STATIC", "SRC_NONE", "SRC_CURRENT", "SRC_FUTURE", "SRC_GT", "teacher_box_to_local", "teacher_box_to_navsim",
    "local_to_n", "s_grid_centres", "centre_cell", "rasterize_boxes", "teacher_unknown", "gt_unknown",
    "teacher_raster", "gt_raster", "TeacherDetCache", "FutureTeacherCache", "FutureRasterCache", "read_token_list",
    "s_grid_cell",
]
