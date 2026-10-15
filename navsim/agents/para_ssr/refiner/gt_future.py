"""GT future objects (M6, spec IMPL_SPEC.md §3.4): every annotated track, 360 deg, 0..5 s, in the N frame, with the
official metric-cache presence/interpolation rule, plus the UNKNOWN (outside annotation reach) test.

Frames, units, time
  N frame   : NAVSIM ego frame at t0 = the token's frame (rear-axle origin, x forward, y left, heading CCW from +x;
              metres, radians).  Everything returned by this module is in N.
  log frame : raw log ``anns`` of frame k (gt_boxes (x_fwd, y_left, z, L, W, H, yaw), gt_velocity_3d) are in the ego
              rear-axle frame of that frame; its pose is (ego2global_translation[:2], yaw of ego2global_rotation),
              exactly as navsim.common.dataclasses.Scene._build_ego_status.
  time      : INDEX based: keyframe k (k = 0..10, log frames cur..cur+10) is at t = 0.5 k s whatever the logged
              timestamp says (metric_cache_processor._interpolate_gt_observation L127-137).  Tokens whose keyframes
              are not 0.5 s apart keep this rule and carry ``dt_max`` > 0.75 as a flag.  Dense grids: t = 0.1 n.

Construction (build_from_frames) -- identical to the metric cache (metric_cache_processor.py L115-238,
navsim_scenario_utils.annotations_to_detection_tracks):
  * centre / heading: local box -> global with the pose of its own keyframe (rotate, normalise, translate)
    -> N with the t0 pose.  Heading unwrapped along the track's annotated keyframes (StateInterpolator np.unwrap).
  * velocity: filled only for nuplan AGENT_TYPES (vehicle, pedestrian, bicycle, ego) and rotated local -> global -> N
    (rotate_vector with the keyframe yaw, then by -yaw0).  Non-agent (static) objects get 0, as in the metric cache.
  * one track = one ``track_token``; L, W, class taken at FIRST appearance (the metric cache keeps the first
    detection's box size / type for every interpolated step and in ``unique_objects``); ``first`` also stores the
    first-appearance heading and velocity (used by the official collision classification).
  * duplicate track_token inside one frame (not observed so far): first occurrence kept, counted in ``n_dup``.
  * interior keyframe gaps of a track (none observed, 0/5,847) are filled by linear interpolation (present = 0),
    which reproduces interp1d over the annotated keyframes exactly.
  * track kept iff its centre is within R = max(R_MIN, S_avail + R_PATH_MARGIN) = max(80, S_avail + 25) m of the t0
    origin at ANY keyframe.  S_avail = GT ego rear-axle polyline length over the next <= 8 s of the log (<= 16 frames).
    The metric cache itself has no radius filter; tracks beyond R are dropped here (validation reports them).
  * track order: ascending distance to the GT ego path (min over the track's annotated keyframes and the GT ego
    keyframe positions 0..5 s), ties by track_token -- so padding / truncation to A_max drops the farthest tracks
    (A reaches 523 in a navtrain sample).
  * red-light pseudo objects: the metric-cache observation built from the log contains none, and the official NC
    (pdm_scorer.py L343) and TTC (L516) skip them anyway -> nothing to exclude from log anns; the validation checks
    that the metric-cache maps contain 0 red-light tokens.

Stored per token (save_objects / load_objects, npz, no pickle):
  kf     [A, 11, 6] f32 : x, y, heading_unwrapped, vx, vy, present   (present = annotated at keyframe k; values
                          outside [first_k, last_k] are 0)
  first  [A, 6] f32     : L, W, heading, vx, vy, first_k               (first appearance)
  meta   [A, 5] i16     : class, is_agent, first_k, last_k, singleton  (class = CLASS_NAMES index = nuplan
                          TrackedObjectType value; is_agent = class in AGENT_TYPES)
  track  [A] <U16       : track_token
  ego_kf [11, 3] f32    : GT ego rear-axle pose (x, y, heading_unwrapped) in N at the keyframes (rows >= n_kf are 0)
  pose0  [3] f64        : global t0 pose (x, y, yaw);  R, S_avail, dt_max [f32 scalars];  n_kf, n_dup [i32 scalars]

Query rule (query / query_torch), metric-cache rule:
  * multi-keyframe track: OBS for first_k*0.5 <= t <= last_k*0.5; x, y, unwrapped heading linearly interpolated
    between keyframes; L, W fixed at first appearance; ABSENT_OFFICIAL before / after (the official scorer sees
    nothing there; boxes returned there are the nearest observed pose so that they stay finite -- mask with state).
  * singleton track (one annotated keyframe): its only pose at EVERY t in [0, 5] s (metric cache places
    ``initial_detection_track`` in all 51 maps).
  * t > t_avail (= 5 s, or less if the log ends early): state ABSENT_OFFICIAL and ``unknown_obj`` True.

UNKNOWN space (unknown_space):  a point (x, y) at time t is UNKNOWN iff
    t > t_avail                                   (spec: t > 5 s)
 or |p - GT ego(t)| > REACH_M = 75 m              (spec: outside the annotation reach, 77-83 m measured)
 or |p| > R - radius_margin  (default 10 m)       (EXTENSION, see deviations)
Deviations from IMPL_SPEC §3.4 (documented, interface unchanged):
  1. UNKNOWN also includes points farther than R - 10 m from the t0 origin: tracks beyond R are dropped by the
     builder, so that space is not "known empty" (critic_logic.md issue 9).  The 10 m margin covers the half length
     of long objects whose centre is just outside R.  ``radius_margin=None`` gives the spec definition exactly.
  2. The reach test uses the GT ego REAR AXLE (not the lidar); the ~1.5 m difference is far below the 77-83 m spread.
  3. is_agent also counts class ``ego`` (nuplan AGENT_TYPES; never present in the logs).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

VERSION = "gt_future_v1"

KF_DT = 0.5          # keyframe spacing [s] (index based)
N_KF = 11            # keyframes 0..5 s
T_MAX = 5.0          # last keyframe [s]
N_PATH_FRAMES = 16   # log frames used for S_avail (8 s)
R_MIN = 80.0         # [m]
R_PATH_MARGIN = 25.0 # [m]
REACH_M = 75.0       # annotation reach around the GT ego [m]
RADIUS_MARGIN = 10.0 # default UNKNOWN margin inside R [m]
EPS_T = 1e-6         # time tolerance [s]

ABSENT_OFFICIAL, OBS = 0, 1

# class index == nuplan TrackedObjectType value (checked in tests)
CLASS_NAMES = ("vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object", "ego")
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
AGENT_CLASSES = (0, 1, 2, 7)   # nuplan AGENT_TYPES: vehicle, pedestrian, bicycle, ego

# column indices
KF_X, KF_Y, KF_H, KF_VX, KF_VY, KF_PRESENT = range(6)
FI_L, FI_W, FI_H, FI_VX, FI_VY, FI_K = range(6)
MT_CLASS, MT_AGENT, MT_FIRST, MT_LAST, MT_SINGLE = range(5)


def wrap(a):
    """angle -> [-pi, pi] (same formula as navsim normalize_angle)."""
    return np.arctan2(np.sin(a), np.cos(a))


def frame_pose(frame: dict):
    """global rear-axle pose (x, y, yaw) of a raw log frame, float64 (as Scene._build_ego_status)."""
    from pyquaternion import Quaternion
    tr = frame["ego2global_translation"]
    return float(tr[0]), float(tr[1]), float(Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0])


def _to_n(gx, gy, pose0):
    x0, y0, h0 = pose0
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = np.asarray(gx, np.float64) - x0, np.asarray(gy, np.float64) - y0
    return c * dx + s * dy, -s * dx + c * dy


def build_from_frames(frames: Sequence[dict], r_min: float = R_MIN, r_margin: float = R_PATH_MARGIN) -> Dict[str, np.ndarray]:
    """Build the per-token object arrays from raw log frames ``frames`` = log[cur : cur + 17] (cur = token frame).

    Keyframes are frames[0..10] (fewer if the log ends: ``n_kf``); frames[0..16] only feed S_avail.
    Returns the dict described in the module docstring (numpy arrays)."""
    frames = list(frames)
    assert len(frames) >= 1
    n_kf = min(N_KF, len(frames))
    poses = [frame_pose(f) for f in frames[: N_PATH_FRAMES + 1]]      # 0..8 s (keyframes are the first n_kf)
    pose0 = poses[0]

    # GT ego in N, S_avail
    pxy = np.array([p[:2] for p in poses], np.float64)
    ex, ey = _to_n(pxy[:, 0], pxy[:, 1], pose0)
    n_path = min(len(poses), N_PATH_FRAMES + 1)
    s_avail = float(np.hypot(np.diff(ex[:n_path]), np.diff(ey[:n_path])).sum()) if n_path > 1 else 0.0
    R = max(r_min, s_avail + r_margin)
    ego_kf = np.zeros((N_KF, 3), np.float64)
    ego_kf[:n_kf, 0], ego_kf[:n_kf, 1] = ex[:n_kf], ey[:n_kf]
    ego_kf[:n_kf, 2] = np.unwrap(wrap(np.array([p[2] for p in poses[:n_kf]]) - pose0[2]))
    ts = np.array([float(f.get("timestamp", 0)) for f in frames[:n_kf]]) * 1e-6
    dt_max = float(np.diff(ts).max()) if n_kf > 1 else 0.0

    tracks: Dict[str, dict] = {}
    n_dup = 0
    for k in range(n_kf):
        a = frames[k]["anns"]
        b = np.asarray(a["gt_boxes"], np.float64).reshape(-1, 7)
        if len(b) == 0:
            continue
        v = np.asarray(a["gt_velocity_3d"], np.float64).reshape(-1, 3)
        names = list(a["gt_names"])
        toks = list(a["track_tokens"])
        xk, yk, hk = poses[k]
        c, s = np.cos(hk), np.sin(hk)
        # local -> global exactly like gt_boxes_oriented_box / rotate_state_se2
        gx = b[:, 0] * c - b[:, 1] * s + xk
        gy = b[:, 0] * s + b[:, 1] * c + yk
        gh = wrap(wrap(b[:, 6] + hk))
        gvx = v[:, 0] * c - v[:, 1] * s
        gvy = v[:, 0] * s + v[:, 1] * c
        # global -> N
        nx, ny = _to_n(gx, gy, pose0)
        nh = wrap(gh - pose0[2])
        c0, s0 = np.cos(pose0[2]), np.sin(pose0[2])
        nvx, nvy = c0 * gvx + s0 * gvy, -s0 * gvx + c0 * gvy
        seen = set()
        for j, tt in enumerate(toks):
            if tt in seen:
                n_dup += 1
                continue
            seen.add(tt)
            name = str(names[j])
            if name not in CLASS_ID:
                raise ValueError(f"unknown object class {name!r}")
            cls = CLASS_ID[name]
            agent = cls in AGENT_CLASSES      # velocity filled per frame by that frame's type (metric cache L150-156)
            rec = tracks.setdefault(tt, {"k": [], "st": [], "cls": cls, "agent": agent, "LW": (b[j, 3], b[j, 4])})
            rec["k"].append(k)
            rec["st"].append((nx[j], ny[j], nh[j], nvx[j] if agent else 0.0, nvy[j] if agent else 0.0))

    keep = []
    exy = np.stack([ex[:n_kf], ey[:n_kf]], -1)
    for tt, rec in tracks.items():
        st = np.asarray(rec["st"], np.float64)
        if np.hypot(st[:, 0], st[:, 1]).min() <= R:
            d_path = float(np.linalg.norm(st[:, None, :2] - exy[None], axis=-1).min())
            keep.append((d_path, tt))
    keep.sort()
    A = len(keep)
    kf = np.zeros((A, N_KF, 6), np.float64)
    first = np.zeros((A, 6), np.float64)
    meta = np.zeros((A, 5), np.int16)
    track = np.array([t for _, t in keep], dtype="<U16") if A else np.zeros((0,), "<U16")
    for i, (_, tt) in enumerate(keep):
        rec = tracks[tt]
        ks = np.asarray(rec["k"], int)
        st = np.asarray(rec["st"], np.float64)
        st[:, 2] = np.unwrap(st[:, 2])
        fk, lk = int(ks[0]), int(ks[-1])
        span = np.arange(fk, lk + 1)
        for col in range(5):
            kf[i, span, col] = np.interp(span, ks, st[:, col])
        kf[i, ks, KF_PRESENT] = 1.0
        first[i] = (rec["LW"][0], rec["LW"][1], st[0, 2], st[0, 3], st[0, 4], fk)
        meta[i] = (rec["cls"], int(rec["agent"]), fk, lk, int(fk == lk))
    return dict(kf=kf.astype(np.float32), first=first.astype(np.float32), meta=meta, track=track,
                ego_kf=ego_kf.astype(np.float32), pose0=np.asarray(pose0, np.float64),
                R=np.float32(R), S_avail=np.float32(s_avail), dt_max=np.float32(dt_max),
                n_kf=np.int32(n_kf), n_dup=np.int32(n_dup), n_tracks_all=np.int32(len(tracks)),
                version=np.array(VERSION))


def save_objects(path, obj: Dict[str, np.ndarray]) -> None:
    """write ``obj`` as an uncompressed npz atomically (tmp file + rename)."""
    import os
    path = str(path)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **obj)
    os.replace(tmp, path)


def load_objects(path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def t_avail(obj) -> float:
    """last annotated time [s] (5.0 unless the log ends early)."""
    return min(T_MAX, (int(obj.get("n_kf", N_KF)) - 1) * KF_DT)


def query(obj: Dict[str, np.ndarray], t, dtype=np.float32, return_unknown: bool = False):
    """Object boxes at times ``t`` [T] (s, e.g. 0.1*arange(41) or 0.1*arange(51)) with the metric-cache rule.

    Returns boxes [A, T, 5] (cx, cy, heading_unwrapped, L, W) in N, state [A, T] int8 in {OBS, ABSENT_OFFICIAL}
    and, if ``return_unknown``, unknown_obj [A, T] bool (t beyond the annotated horizon)."""
    kf = np.asarray(obj["kf"], np.float64)
    meta = np.asarray(obj["meta"])
    first = np.asarray(obj["first"], np.float64)
    t = np.asarray(t, np.float64).reshape(-1)
    A, T = kf.shape[0], t.shape[0]
    ta = t_avail(obj)
    fk = meta[:, MT_FIRST].astype(np.float64)[:, None]
    lk = meta[:, MT_LAST].astype(np.float64)[:, None]
    single = meta[:, MT_SINGLE].astype(bool)[:, None]
    tc = np.clip(t[None, :], fk * KF_DT, lk * KF_DT)                     # [A, T] (singleton -> its keyframe)
    u = np.clip(tc / KF_DT, 0.0, N_KF - 1)
    k0 = np.minimum(np.floor(u + 1e-9).astype(np.int64), N_KF - 2)
    w = np.clip(u - k0, 0.0, 1.0)[..., None]
    ia = np.arange(A)[:, None]
    lo, hi = kf[ia, k0, :3], kf[ia, k0 + 1, :3]
    xyh = lo * (1.0 - w) + hi * w
    boxes = np.concatenate([xyh, np.broadcast_to(first[:, None, FI_L:FI_W + 1], (A, T, 2))], -1)
    in_t = (t[None, :] >= -EPS_T) & (t[None, :] <= ta + EPS_T)
    span = (t[None, :] >= fk * KF_DT - EPS_T) & (t[None, :] <= lk * KF_DT + EPS_T)
    obs = in_t & (single | span)
    state = np.where(obs, OBS, ABSENT_OFFICIAL).astype(np.int8)
    if return_unknown:
        return boxes.astype(dtype), state, np.broadcast_to(t[None, :] > ta + EPS_T, (A, T)).copy()
    return boxes.astype(dtype), state


def box_corners(boxes):
    """boxes [..., 5] (cx, cy, h, L, W) -> corners [..., 4, 2] ordered FL, RL, RR, FR (= nuplan OrientedBox /
    sf_common.box_corners order)."""
    b = np.asarray(boxes, np.float64)
    cx, cy, h, L, W = (b[..., i] for i in range(5))
    c, s = np.cos(h), np.sin(h)
    hl, hw = L / 2, W / 2
    ux, uy, nx, ny = c * hl, s * hl, -s * hw, c * hw
    return np.stack([np.stack([cx + ux + nx, cy + uy + ny], -1), np.stack([cx - ux + nx, cy - uy + ny], -1),
                     np.stack([cx - ux - nx, cy - uy - ny], -1), np.stack([cx + ux - nx, cy + uy - ny], -1)], -2)


def gt_ego_xy(obj, t) -> np.ndarray:
    """GT ego rear-axle position in N at times t [T] (linear in keyframes, clamped to the annotated horizon)."""
    n = int(obj.get("n_kf", N_KF))
    e = np.asarray(obj["ego_kf"], np.float64)[:n]
    tk = np.arange(n) * KF_DT
    t = np.asarray(t, np.float64).reshape(-1)
    return np.stack([np.interp(t, tk, e[:, 0]), np.interp(t, tk, e[:, 1])], -1)


def unknown_space(obj, xy, t, reach: float = REACH_M, radius_margin: Optional[float] = RADIUS_MARGIN) -> np.ndarray:
    """UNKNOWN test for query points ``xy`` [..., T, 2] (N frame) at times ``t`` [T] -> bool [..., T].

    True iff t > t_avail, or farther than ``reach`` from the GT ego at t, or (radius_margin not None) farther than
    R - radius_margin from the t0 origin.  radius_margin=None is the IMPL_SPEC §3.4 definition."""
    xy = np.asarray(xy, np.float64)
    t = np.asarray(t, np.float64).reshape(-1)
    ego = gt_ego_xy(obj, t)                                              # [T, 2]
    unk = np.broadcast_to(t > t_avail(obj) + EPS_T, xy.shape[:-1])
    unk = unk | (np.linalg.norm(xy - ego, axis=-1) > reach)
    if radius_margin is not None:
        unk = unk | (np.linalg.norm(xy, axis=-1) > float(obj["R"]) - radius_margin)
    return unk


# ------------------------------------------------------------------------------------------------ batched (torch)
def pad_objects(objs: List[Dict[str, np.ndarray]], a_max: Optional[int] = None) -> Dict[str, np.ndarray]:
    """stack per-token objects into padded arrays: kf [B, A, 11, 6], first [B, A, 6], meta [B, A, 5] (i64),
    valid [B, A] bool, n_kf [B], R [B].  Tracks beyond ``a_max`` (kept in the stored order) are dropped and
    counted in ``n_dropped`` [B]."""
    B = len(objs)
    n = [o["kf"].shape[0] for o in objs]
    A = max(n + [1]) if a_max is None else int(a_max)
    kf = np.zeros((B, A, N_KF, 6), np.float32)
    first = np.zeros((B, A, 6), np.float32)
    meta = np.zeros((B, A, 5), np.int64)
    valid = np.zeros((B, A), bool)
    for b, o in enumerate(objs):
        m = min(n[b], A)
        kf[b, :m], first[b, :m], meta[b, :m], valid[b, :m] = o["kf"][:m], o["first"][:m], o["meta"][:m], True
    return dict(kf=kf, first=first, meta=meta, valid=valid,
                n_kf=np.array([int(o.get("n_kf", N_KF)) for o in objs], np.int64),
                R=np.array([float(o["R"]) for o in objs], np.float32),
                n_dropped=np.array([max(0, x - A) for x in n], np.int64))


def query_torch(kf, first, meta, t, valid=None, n_kf=None):
    """torch version of ``query`` for padded batches: kf [B, A, 11, 6], first [B, A, 6], meta [B, A, 5], t [T] (s),
    valid [B, A] bool, n_kf [B].  Returns boxes [B, A, T, 5] (kf dtype, on kf.device) and state [B, A, T] int8.
    No gradient flows (GT data)."""
    import torch
    B, A = kf.shape[:2]
    dev = kf.device
    t = torch.as_tensor(t, dtype=torch.float64, device=dev).reshape(-1)
    T = t.shape[0]
    kf64 = kf.to(torch.float64)
    fk = meta[..., MT_FIRST].to(torch.float64)[..., None]
    lk = meta[..., MT_LAST].to(torch.float64)[..., None]
    single = meta[..., MT_SINGLE].bool()[..., None]
    tt = t.view(1, 1, T)
    tc = torch.minimum(torch.maximum(tt, fk * KF_DT), lk * KF_DT)
    u = (tc / KF_DT).clamp(0.0, N_KF - 1)
    k0 = torch.floor(u + 1e-9).long().clamp(max=N_KF - 2)
    w = (u - k0).clamp(0.0, 1.0)[..., None]
    src = kf64[..., :3]                                                   # [B, A, 11, 3]
    lo = torch.gather(src, 2, k0[..., None].expand(B, A, T, 3))
    hi = torch.gather(src, 2, (k0 + 1)[..., None].expand(B, A, T, 3))
    xyh = lo * (1.0 - w) + hi * w
    lw = first[..., FI_L:FI_W + 1].to(torch.float64)[:, :, None, :].expand(B, A, T, 2)
    boxes = torch.cat([xyh, lw], -1).to(kf.dtype)
    if n_kf is None:
        ta = torch.full((B, 1, 1), T_MAX, dtype=torch.float64, device=dev)
    else:
        ta = ((torch.as_tensor(n_kf, device=dev).to(torch.float64) - 1) * KF_DT).clamp(max=T_MAX).view(B, 1, 1)
    in_t = (tt >= -EPS_T) & (tt <= ta + EPS_T)
    span = (tt >= fk * KF_DT - EPS_T) & (tt <= lk * KF_DT + EPS_T)
    obs = in_t & (single | span)
    if valid is not None:
        obs = obs & valid.bool()[..., None]
    return boxes, obs.to(torch.int8)
