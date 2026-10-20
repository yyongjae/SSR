"""km 10/19 (C): the official PDMS (navsim v1) verdicts of TRACKED candidates, on the GPU, from exact geometry
(tools/corridor/build_exact.py: metric-cache agent boxes, drivable polygons, route centerline).  Used as the training
targets of the candidate scorer on the model's own candidates; no raster, no read-out.

states [B, K, 41, 7] = torch_tracker.track(..., full=True): (x, y, heading, vx, vy, ax, ay) of the rear axle at
0.1 s, index 0 = now, ego frame.  The ego box is the Pacifica footprint about the rear axle (corridor_head.FRONT /
REAR / HALF_W), as the scorer's _ego_coords.
  DAC  (_calculate_drivable_area_compliance): 0 iff some corner, at some of the 41 steps, lies in no drivable polygon
  NC   (_calculate_no_at_fault_collision): min over the tracks whose FIRST overlap (steps 1..40) is at fault of 0
       (AGENT_TYPES) or 0.5 (static); at fault = get_collision_type: ego moving (> 0.05 m/s) and (track stopped, or
       not behind (> 150 deg) and (front bumper hit, or a lateral hit while in multiple lanes / non-drivable area));
       a first overlap that is not at fault puts the track in the proposal's memory (ignored afterwards)
  TTC  (_calculate_ttc): 0 iff for some ego step 0..40 (moving, >= 5e-3 m/s) and dt in {0, 0.3, 0.6, 0.9} s, in
       the scorer's (t, dt) loop order, the box shifted by speed * dt along the heading first overlaps a track at
       t + dt (<= 4.9 s) that is ahead (< 30 deg) or not behind while the ego is in multiple lanes / non-drivable /
       an intersection (rear axle); otherwise the track goes to the memory
  ego areas (_calculate_ego_area): exact point-in-polygon of the 4 corners in the DAC / lane polygons
  progress (_calculate_progress): centerline arc length of the box centre at 4.0 s minus at 0 s, clipped at 0
  comfort (ego_is_comfortable): the official Savitzky-Golay filters are linear, so they are applied as matrices built
       by calling the official helpers on the identity; the 1e-8 rounding is left out
Objects that the scorer ignores (red light, tracks overlapping the ego at t = 0) are dropped by the builder.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np
import torch

from .corridor_head import FRONT, HALF_W, REAR, SPLITS
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder

HALF_L = (FRONT + REAR) / 2
TABLES = ("agent_rows", "drv_seg", "lane_seg", "int_seg", "route_seg", "red_seg", "cl")
CENTRE = (FRONT - REAR) / 2           # box centre ahead of the rear axle
STOPPED = 5e-3                        # PDMScorerConfig.stopped_speed_threshold (TTC)
A_MAX, S_MAX, LANE_MAX, INT_MAX, ROUTE_MAX, RED_MAX, L_MAX, T_OBS = 512, 2048, 4096, 1024, 2048, 1024, 1024, 50


# ------------------------------------------------------------------------------------------------- geometry
_DCONST = {}


def dconst(vals, device, dtype=torch.float32) -> torch.Tensor:
    """km 10/20: torch.tensor(vals, device, dtype) built once per (vals, device, dtype): a fresh host -> device copy
    per call blocks the host until the stream drains (same values, so the results are unchanged)"""
    key = (vals, device, dtype)
    if key not in _DCONST:
        _DCONST[key] = torch.tensor(vals, device=device, dtype=dtype)
    return _DCONST[key]


def _dev_ops(ops: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    """the lru-cached comfort operators moved to device once"""
    key = (id(ops), device)
    if key not in _DCONST:
        _DCONST[key] = {k: v.to(device) for k, v in ops.items()}
    return _DCONST[key]


def _ego_box(st: torch.Tensor):
    """-> centre [.., 2], unit heading [.., 2] of the ego box"""
    h = st[..., 2]
    u = torch.stack([torch.cos(h), torch.sin(h)], -1)
    return st[..., :2] + CENTRE * u, u


def _overlap(c1, u1, e1, c2, u2, e2) -> torch.Tensor:
    """oriented boxes (centre, unit axis, half extents (along, across)) -> intersects (touching counts), broadcast"""
    v1 = torch.stack([-u1[..., 1], u1[..., 0]], -1)
    v2 = torch.stack([-u2[..., 1], u2[..., 0]], -1)
    d = c2 - c1
    sep = torch.zeros(d.shape[:-1], dtype=torch.bool, device=d.device)
    for a in (u1, v1, u2, v2):
        dist = (d * a).sum(-1).abs()
        r1 = e1[..., 0] * (u1 * a).sum(-1).abs() + e1[..., 1] * (v1 * a).sum(-1).abs()
        r2 = e2[..., 0] * (u2 * a).sum(-1).abs() + e2[..., 1] * (v2 * a).sum(-1).abs()
        sep |= dist > r1 + r2
    return ~sep


def _in_polys(p: torch.Tensor, seg: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """points p [N, 2]; seg [S, 5] (x1, y1, x2, y2, polygon id) -> inside [N, npoly] (crossing number, ray to +x)"""
    if seg.shape[0] == 0:
        return torch.zeros(p.shape[0], 0, dtype=torch.bool, device=p.device)
    pid = seg[:, 4].long()
    npoly = int(pid.max()) + 1
    onehot = torch.zeros(seg.shape[0], npoly, device=p.device)
    onehot[torch.arange(seg.shape[0], device=p.device), pid] = 1.0
    x1, y1, x2, y2 = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    out = torch.empty(p.shape[0], npoly, dtype=torch.bool, device=p.device)
    for p0 in range(0, p.shape[0], chunk):
        px, py = p[p0:p0 + chunk, :1], p[p0:p0 + chunk, 1:]
        straddle = (y1 > py) != (y2 > py)
        xint = x1 + (py - y1) * (x2 - x1) / torch.where(y2 == y1, torch.ones_like(y2), y2 - y1)
        out[p0:p0 + chunk] = ((straddle & (px < xint)).float() @ onehot).remainder(2) > 0.5
    return out


def _corners(st: torch.Tensor) -> torch.Tensor:
    """[.., 41, 7] -> ego box corners [.., 41, 4, 2] (FL, RL, RR, FR as the scorer's coords)"""
    c, u = _ego_box(st)
    v = torch.stack([-u[..., 1], u[..., 0]], -1)
    sx = dconst((HALF_L, -HALF_L, -HALF_L, HALF_L), st.device, st.dtype)
    sy = dconst((HALF_W, HALF_W, -HALF_W, -HALF_W), st.device, st.dtype)
    return c[..., None, :] + u[..., None, :] * sx[:, None] + v[..., None, :] * sy[:, None]


@torch.no_grad()
def ego_areas(st: torch.Tensor, drv: torch.Tensor, lane: torch.Tensor, inter: torch.Tensor, route: torch.Tensor = None):
    """st [K, 41, 7] of one sample; polygon edge tables -> (non_drivable, multiple_lanes, rear axle in intersection),
    each [K, 41], as PDMScorer._calculate_ego_area / is_in_layer; with route (on-route lane polygons): + (oncoming =
    box centre in no on-route lane polygon, box centre in an intersection) -- the driving-direction / lane-keeping masks"""
    k = st.shape[0]
    pc = _corners(st).reshape(-1, 2)
    ind = _in_polys(pc, drv).reshape(k, 41, 4, -1)
    nondrv = ind.any(-1).sum(-1) < 4
    inl = _in_polys(pc, lane).reshape(k, 41, 4, -1)
    multi = (inl.any(2).sum(-1) > 1) & (inl.sum(2) != 4).all(-1)
    inter_ = _in_polys(st[..., :2].reshape(-1, 2), inter).any(-1).reshape(k, 41)
    if route is None:
        return nondrv, multi, inter_
    cen = _ego_box(st)[0].reshape(-1, 2)
    onc = ~_in_polys(cen, route).any(-1).reshape(k, 41)
    cint = _in_polys(cen, inter).any(-1).reshape(k, 41)
    return nondrv, multi, inter_, onc, cint


def _first_fault(hit: torch.Tensor, fault: torch.Tensor, order: torch.Tensor, tid: torch.Tensor, ntrack: int):
    """the scorer's per-proposal track memory: a track counts iff its FIRST hit (in the scorer's loop order) is a fault.
    hit, fault [K, N]; order [N] loop position of each (step, object) entry; tid [N] track id -> counts [K, ntrack]"""
    k = hit.shape[0]
    big = torch.iinfo(torch.long).max
    key = torch.where(hit, order[None].expand(k, -1), torch.full_like(hit, big, dtype=torch.long))
    first = torch.full((k, ntrack), big, dtype=torch.long, device=hit.device)
    first.scatter_reduce_(1, tid[None].expand(k, -1), key, reduce="amin")
    at_first = hit & (key == first.gather(1, tid[None].expand(k, -1)))
    counts = torch.zeros(k, ntrack, dtype=torch.uint8, device=hit.device)          # (bool scatter: no CUDA kernel)
    counts.scatter_reduce_(1, tid[None].expand(k, -1), (at_first & fault).to(torch.uint8), reduce="amax")
    return counts.bool()


def _rel(c_ego: torch.Tensor, u_ego: torch.Tensor, c_ag: torch.Tensor) -> torch.Tensor:
    """cos of get_agent_relative_angle (rear axle pose -> agent centroid); NaN at zero distance as the scorer"""
    d = c_ag - c_ego
    return (d * u_ego).sum(-1) / d.norm(dim=-1)


@torch.no_grad()
def exact_nc_ttc(st: torch.Tensor, t: Dict[str, torch.Tensor], chunk: int = 512, ttc_last: int = 40, areas=None):
    """st [B, K, 41, 7]; targets of ExactTargetBuilder -> (nc [B, K] in {0, 0.5, 1}, ttc [B, K]) with the scorer's
    at-fault classification (get_collision_type), TTC direction rule and per-proposal track memory.  ttc_last = the
    last ego step of the TTC loop: 40 in v1, num_poses - max(future idx) = 31 in v2 (n_proposal_steps_to_evaluate)"""
    b, k = st.shape[:2]
    dev = st.device
    nc_out, ttc_out = torch.ones(b, k, device=dev), torch.ones(b, k, device=dev)
    ee = dconst((HALF_L, HALF_W), dev, st.dtype)
    eb = dconst((0.0, HALF_W), dev, st.dtype)                       # front bumper FL-FR
    cos30, cos150 = float(np.cos(np.deg2rad(30))), float(np.cos(np.deg2rad(150)))
    for i in range(b):
        m = t["ex_amask"][i]                                                           # [50, A]
        na = int(m.sum(0).nonzero().max()) + 1 if m.any() else 0
        if na == 0:
            continue
        # km 10/20: the slots past the last occupied one (pack_token fills every step from slot 0) are padding that
        # never hits (masked) -- dropping them leaves every value unchanged and the cost scales with the real count
        m = m[:, :na]
        ag = t["ex_agents"][i][:, :na]                                                 # [50, A, 8]
        nobs = int(t["ex_nobs"][i])
        ac, ah, ae = ag[..., :2], ag[..., 2], ag[..., 3:5]
        au = torch.stack([torch.cos(ah), torch.sin(ah)], -1)
        kind, tid, stopped = ag[..., 5], ag[..., 6].long(), ag[..., 7] > 0.5
        ntrack = int(tid[m].max()) + 1
        tid = torch.where(m, tid, torch.zeros_like(tid))
        val = torch.where(kind > 0.5, 0.0, 0.5)
        tmax = torch.arange(T_OBS, device=dev) <= nobs
        if areas is not None:                                       # (nondrv, multi, inter_) of ego_areas, already computed
            nondrv, multi, inter_ = areas[i]
        else:
            nondrv, multi, inter_ = ego_areas(st[i], t["ex_seg"][i][t["ex_smask"][i]], t["ex_lane"][i][t["ex_lmask"][i]],
                                              t["ex_int"][i][t["ex_imask"][i]])
        for k0 in range(0, k, chunk):
            s = st[i, k0:k0 + chunk]                                                    # [c, 41, 7]
            kc = s.shape[0]
            c, u = _ego_box(s)
            ra = s[..., :2]
            spd = torch.hypot(s[..., 3], s[..., 4])                                     # [c, 41]
            area = multi[k0:k0 + kc] | nondrv[k0:k0 + kc]
            # ---- NC: step t (1..40; t = 0 overlaps only the ignored start-collided tracks) vs objects at t
            ti = torch.arange(1, 41, device=dev)
            A_ = (lambda x: x[ti][None])
            hit = _overlap(c[:, 1:, None], u[:, 1:, None], ee, A_(ac), A_(au), A_(ae)) & A_(m) & tmax[ti][None, :, None]
            rel = _rel(ra[:, 1:, None], u[:, 1:, None], A_(ac))
            behind = rel < cos150
            front = _overlap(c[:, 1:, None] + u[:, 1:, None] * HALF_L, u[:, 1:, None], eb, A_(ac), A_(au), A_(ae))
            ego_stop = (spd[:, 1:] <= 5e-2)[..., None]
            fault = ~ego_stop & (A_(stopped) | (~behind & (front | area[:, 1:, None])))
            order = (ti[:, None] * 0 + torch.arange(40, device=dev)[:, None]).expand(-1, m.shape[1]).reshape(-1)
            cnt = _first_fault(hit.reshape(kc, -1), fault.reshape(kc, -1), order, A_(tid).reshape(-1), ntrack)
            tv = torch.ones(ntrack, device=dev)
            tv.scatter_reduce_(0, tid[m], val[m], reduce="amin")
            nc_out[i, k0:k0 + kc] = torch.where(cnt, tv[None], torch.ones_like(tv)[None]).amin(1)
            # ---- TTC: step t (0..40) shifted by speed * dt vs objects at t + dt, loop order (t, dt)
            hits, faults, orders, tids = [], [], [], []
            for j, dd in enumerate((0, 3, 6, 9)):
                ct = torch.arange(41, device=dev) + dd
                okt = (ct >= 1) & (ct < T_OBS) & (ct <= nobs)
                cc = ct.clamp(max=T_OBS - 1)
                B_ = (lambda x: x[cc][None])
                cs = c + u * (spd * dd * 0.1)[..., None]
                h = _overlap(cs[:, :, None], u[:, :, None], ee, B_(ac), B_(au), B_(ae)) & B_(m)
                h &= (okt[None] & (spd >= STOPPED) & (torch.arange(41, device=dev) <= ttc_last)[None])[..., None]
                r = _rel(ra[:, :, None], u[:, :, None], B_(ac))
                f = (r > cos30) | ((area | inter_[k0:k0 + kc])[..., None] & ~(r < cos150))
                hits.append(h)
                faults.append(f)
                orders.append((torch.arange(41, device=dev) * 4 + j)[:, None].expand(-1, m.shape[1]))
                tids.append(B_(tid)[0])
            hit = torch.stack(hits, 2).reshape(kc, -1)
            fault = torch.stack(faults, 2).reshape(kc, -1)
            order = torch.stack(orders, 1).reshape(-1)
            cnt = _first_fault(hit, fault, order, torch.stack(tids, 1).reshape(-1), ntrack)
            ttc_out[i, k0:k0 + kc] = (~cnt.any(1)).float()
    return nc_out, ttc_out


@torch.no_grad()
def exact_dac(st: torch.Tensor, seg: torch.Tensor, smask: torch.Tensor) -> torch.Tensor:
    """st [B, K, 41, 7]; seg [B, S, 5]; smask [B, S] -> dac [B, K] (1 = every corner of every step in a DAC polygon)"""
    b, k = st.shape[:2]
    out = torch.ones(b, k, device=st.device)
    for i in range(b):
        ind = _in_polys(_corners(st[i]).reshape(-1, 2), seg[i][smask[i]]).reshape(k, 41, 4, -1)
        out[i] = (ind.any(-1).sum(-1) == 4).all(-1).float()
    return out


@torch.no_grad()
def exact_progress(st: torch.Tensor, cl: torch.Tensor, clmask: torch.Tensor) -> torch.Tensor:
    """st [B, K, 41, 7]; cl [B, L, 3] (x, y, s); clmask [B, L] -> raw progress (m) [B, K] of the box centre"""
    b, k = st.shape[:2]
    c, _ = _ego_box(st)
    out = torch.zeros(b, k, device=st.device)
    for i in range(b):
        m = clmask[i]
        if m.sum() < 2:
            continue
        p = cl[i, m]
        a, d, s0 = p[:-1, :2], p[1:, :2] - p[:-1, :2], p[:-1, 2]
        ln2 = (d * d).sum(-1).clamp(min=1e-12)

        def proj(q):                                                                    # q [n, 2] -> s [n]
            t = (((q[:, None] - a[None]) * d[None]).sum(-1) / ln2).clamp(0, 1)
            dist = ((a[None] + t[..., None] * d[None] - q[:, None]) ** 2).sum(-1)
            j = dist.argmin(1)
            return s0[j] + t[torch.arange(q.shape[0], device=q.device), j] * ln2[j].sqrt()

        out[i] = (proj(c[i, :, -1]) - proj(c[i, :, 0])).clamp(min=0)
    return out


@lru_cache(maxsize=None)
def _comfort_ops(n: int = 41, dt: float = 0.1):
    """the official comfort filters as [n, n] matrices (out = x @ M), from the official helpers on the identity"""
    from scipy.signal import savgol_filter
    from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_comfort_metrics as cm
    eye = np.eye(n)
    tp = np.arange(n) * dt
    acc_n = savgol_filter(eye, polyorder=2, window_length=n, axis=-1)                   # lon / lat accel (window n)
    acc_8 = savgol_filter(eye, polyorder=2, window_length=min(8, n), axis=-1)           # inside _extract_ego_jerk
    jerk = acc_8 @ cm._approximate_derivatives(eye, tp, window_length=n, poly_order=2, deriv_order=1)
    yaw_rate = cm._approximate_derivatives(eye, tp, poly_order=2, deriv_order=1)        # window 5 (the default)
    yaw_acc = cm._approximate_derivatives(eye, tp, poly_order=3, deriv_order=2)
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in
            (("acc", acc_n), ("jerk", jerk), ("yaw_rate", yaw_rate), ("yaw_acc", yaw_acc))}


@torch.no_grad()
def exact_comfort(st: torch.Tensor) -> torch.Tensor:
    """st [..., 41, 7] -> comfortable [...] (float), the official ego_is_comfortable on these states"""
    ops = _dev_ops(_comfort_ops(st.shape[-2]), st.device)
    s = st.double()
    ax, ay = s[..., 5], s[..., 6]
    h = s[..., 2]
    hu = h.clone()
    hu[..., 1:] = h[..., 1:] - 2 * np.pi * torch.cumsum(torch.round(torch.diff(h, dim=-1) / (2 * np.pi)), -1)
    lon, lat = ax @ ops["acc"], ay @ ops["acc"]
    jm, jx = torch.hypot(ax, ay) @ ops["jerk"], ax @ ops["jerk"]
    yr, ya = hu @ ops["yaw_rate"], hu @ ops["yaw_acc"]
    ok = ((lon >= -4.05) & (lon <= 2.40)).all(-1) & (lat.abs() <= 4.89).all(-1) & (jm.abs() <= 8.37).all(-1) \
        & (jx.abs() <= 4.13).all(-1) & (ya.abs() <= 1.93).all(-1) & (yr.abs() <= 0.95).all(-1)
    return ok.float()


@torch.no_grad()
def exact_verdicts(st: torch.Tensor, t: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """all five on the targets of ExactTargetBuilder -> {nc, dac, ttc, progress, comfort} [B, K]"""
    nc, ttc = exact_nc_ttc(st, t)
    return {"nc": nc, "dac": exact_dac(st, t["ex_seg"], t["ex_smask"]), "ttc": ttc,
            "progress": exact_progress(st, t["ex_cl"], t["ex_clmask"]), "comfort": exact_comfort(st)}


# ------------------------------------------------------------------------------------------------- targets
def _pad_seg(seg: np.ndarray, n_max: int):
    sg = np.zeros((n_max, 5), np.float32)
    sm = np.zeros(n_max, bool)
    n = min(len(seg), n_max)
    sg[:n], sm[:n] = seg[:n], True
    return torch.from_numpy(sg), torch.from_numpy(sm), max(len(seg) - n_max, 0)


def pack_token(ag: np.ndarray, seg: np.ndarray, lane: np.ndarray, inter: np.ndarray, route: np.ndarray, red: np.ndarray,
               cl: np.ndarray) -> Dict[str, torch.Tensor]:
    """build_exact.py rows of one token -> fixed-size tensors (objects stored once with t = 0 go to every t)"""
    t = ag[:, 0].astype(np.int64)
    nobs = int(t.max()) if len(t) else T_OBS - 1
    agents = np.zeros((T_OBS, A_MAX, 8), np.float32)
    amask = np.zeros((T_OBS, A_MAX), bool)
    const = ag[t == 0, 1:]
    trunc = 0
    for ti in range(1, T_OBS):
        rows = np.concatenate([ag[t == ti, 1:], const], 0)
        if len(rows) > A_MAX:                                    # keep the nearest (navtest max per step: 381)
            trunc += len(rows) - A_MAX
            rows = rows[np.argsort(np.hypot(rows[:, 0], rows[:, 1]))[:A_MAX]]
        agents[ti, :len(rows)], amask[ti, :len(rows)] = rows, True
    sg, sm, t1 = _pad_seg(seg, S_MAX)
    lg, lm, t2 = _pad_seg(lane, LANE_MAX)
    ig, im, t3 = _pad_seg(inter, INT_MAX)
    rg, rm, t4 = _pad_seg(route, ROUTE_MAX)
    red6 = np.zeros((RED_MAX, 6), np.float32)
    redm = np.zeros(RED_MAX, bool)
    nr = min(len(red), RED_MAX)
    red6[:nr], redm[:nr] = red[:nr], True
    c = np.zeros((L_MAX, 3), np.float32)
    cm = np.zeros(L_MAX, bool)
    if len(cl) > L_MAX:                                          # keep the vertices nearest the ego
        j = int(np.argmin(np.hypot(cl[:, 0], cl[:, 1])))
        j0 = int(np.clip(j - L_MAX // 4, 0, len(cl) - L_MAX))
        cl = cl[j0:j0 + L_MAX]
    c[:len(cl)], cm[:len(cl)] = cl, True
    return {"ex_agents": torch.from_numpy(agents), "ex_amask": torch.from_numpy(amask), "ex_nobs": torch.tensor(nobs),
            "ex_seg": sg, "ex_smask": sm, "ex_lane": lg, "ex_lmask": lm, "ex_int": ig, "ex_imask": im,
            "ex_route": rg, "ex_rmask": rm, "ex_red": torch.from_numpy(red6), "ex_redmask": torch.from_numpy(redm),
            "ex_cl": torch.from_numpy(c), "ex_clmask": torch.from_numpy(cm),
            "ex_trunc": torch.tensor(float(trunc + t1 + t2 + t3 + t4 + max(len(red) - RED_MAX, 0)))}


def empty_token() -> Dict[str, torch.Tensor]:
    z5 = np.zeros((0, 5), np.float32)
    out = pack_token(np.zeros((0, 9), np.float32), z5, z5, z5, z5, np.zeros((0, 6), np.float32), np.zeros((0, 3), np.float32))
    out["ex_valid"] = torch.tensor(0.0)
    return out


class ExactTargetBuilder(AbstractTargetBuilder):
    """tools/corridor/build_exact.py tables (config.exact_dir), looked up by the current-frame token"""

    def __init__(self, config):
        self._dir = str(config.exact_dir)
        self._troot = str(config.corridor_tokens_root)
        self._arr = None

    def get_unique_name(self) -> str:
        return "para_ssr_exact_verdict"

    def _open(self):
        if self._arr is None:
            import pandas as pd
            self._index, self._arr = {}, {}
            for sp in SPLITS:
                toks = pd.read_parquet(f"{self._troot}/{sp}/tokens.parquet").token.astype(str).to_numpy()
                ok = np.load(f"{self._dir}/{sp}/ok.npy")
                self._index.update({t: (sp, i) for i, t in enumerate(toks) if ok[i]})
                self._arr[sp] = {k: (np.load(f"{self._dir}/{sp}/{k}.npy", mmap_mode="r"),
                                     np.load(f"{self._dir}/{sp}/{k}_off.npy")) for k in TABLES}

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        self._open()
        token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
        hit = self._index.get(token)
        if hit is None:
            return empty_token()
        A, r = self._arr[hit[0]], hit[1]
        rows = {k: np.asarray(v[o[r]:o[r + 1]], np.float32) for k, (v, o) in A.items()}
        out = pack_token(*(rows[k] for k in TABLES))
        out["ex_valid"] = torch.tensor(1.0)
        return out
