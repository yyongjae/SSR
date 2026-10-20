"""km 10/19 (C v3): the OFFICIAL per-candidate PDMS (navsim v1) and EPDMS (NAVSIM v2.2, two-frame EC excluded) of the
model's tracked candidates, on the GPU, from exact geometry (tools/corridor/build_exact.py: v1 tables from the v1 metric
cache, v2 tables from the v2 metric cache -- the two caches differ in observation and PDM-Closed).

Every term follows the scorer code line by line (references are to the scorers in navsim/ and navsim_v2/):
  v1 pdm_score:  proposals [PDM-Closed, candidate]; multiplicative NC * DAC * DDC; weighted (5 EP + 5 TTC + 2 C) / 12
    DDC v1   _calculate_driving_direction_compliance sums oncoming_progress[max(0, i - 10): i + 1] over the PROPOSAL axis
             (len(oncoming_progress) = number of proposals), so the candidate (i = 1) gets max_t of (PDM-Closed + candidate)
             per-step oncoming progress and PDM-Closed (i = 0) its own; no intersection exemption; < 2 m 1, < 6 m 0.5
    EP v1    raw progress (>= 0) * own multiplicative / max over the two; threshold 0.1 m (else 1, 0 if multiplicative 0)
    C        ego_is_comfortable (v1: window = n for every filter, rear-axle accelerations)
  v2 pdm_score:  proposals [PDM-Closed, candidate]; multiplicative NC * DAC * DDC * TLC; weighted (5 EP + 5 TTC + 2 LK +
                 2 HC [+ 2 EC, not here]) / 14; then the human_penalty_filter: every metric the human trajectory (scored
                 alone) gets 0 for is set to 1 for the candidate, multiplicative product recomputed; EP is not touched
    TTC v2   ego steps 0 .. 31 only (n_proposal_steps_to_evaluate = 40 - 9)
    DDC v2   per proposal, oncoming steps outside intersections (box centre), summed over the 1 s window [t - 10, t]
    TLC      ego polygon intersects a red-light polygon of the observation at t (0 .. 40)
    LK       box-centre distance to the whole centerline > 0.5 m for 20 consecutive steps; steps with the centre in an
             intersection are skipped (the counter neither grows nor resets)
    HC       ego_is_comfortable (v2: accel window 8, jerk derivative window 15, yaw windows 5, x-acceleration shifted to
             the box centre a_x + L (w^2 + dw/dt)) of [past human states (15) ; the 41 simulated states], dt 0.1
    EP v2    raw progress / max(raw * multiplicative) over the two, clipped to [0, 1]; threshold 5 m (else 1)
Not reproduced: the 1e-8 rounding inside the comfort helpers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np
import torch

from . import exact_verdict as EV
from .torch_tracker import track

LK_LIMIT, LK_STEPS = 0.5, 20
DDC_OK, DDC_BAD, DDC_H = 2.0, 6.0, 10
# km 10/20: the official v1 PDMS run (navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml) sets
# progress_distance_threshold 5.0 m; the 0.1 m used before was the PDMScorerConfig dataclass default (the 10/19
# fidelity check built PDMScorer(fs) without the yaml and so shared the error): EP off on ~4 % of navtest tokens
EP_THR_V1 = 5.0


def _thr_ddc(p: torch.Tensor) -> torch.Tensor:
    return torch.where(p < DDC_OK, 1.0, torch.where(p < DDC_BAD, 0.5, 0.0))


def _step_progress(st: torch.Tensor) -> torch.Tensor:
    """[.., 41, >=3] -> per-step box-centre displacement [.., 41] (index 0 = 0)"""
    c = EV._ego_box(st)[0]
    d = torch.zeros(c.shape[:-1], dtype=c.dtype, device=c.device)
    d[..., 1:] = (c[..., 1:, :] - c[..., :-1, :]).norm(dim=-1)
    return d


def _seg_dist(p: torch.Tensor, cl: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """points [N, 2], polyline vertices [L, 2] -> distance to the polyline [N] (shapely Point.distance(LineString))"""
    a, d = cl[:-1], cl[1:] - cl[:-1]
    ln2 = (d * d).sum(-1).clamp(min=1e-12)
    out = torch.empty(p.shape[0], device=p.device, dtype=p.dtype)
    for i in range(0, p.shape[0], chunk):
        q = p[i:i + chunk]
        t = (((q[:, None] - a[None]) * d[None]).sum(-1) / ln2).clamp(0, 1)
        out[i:i + chunk] = ((a[None] + t[..., None] * d[None] - q[:, None]) ** 2).sum(-1).min(1).values.sqrt()
    return out


def lane_keeping(st: torch.Tensor, cl: torch.Tensor, cint: torch.Tensor) -> torch.Tensor:
    """st [K, 41, .]; centerline vertices [L, 2]; cint [K, 41] box centre in an intersection -> LK [K]"""
    k = st.shape[0]
    if cl.shape[0] < 2:
        return torch.ones(k, device=st.device)
    dev = _seg_dist(EV._ego_box(st)[0].reshape(-1, 2), cl).reshape(k, 41)
    cnt = torch.zeros(k, device=st.device)
    fail = torch.zeros(k, dtype=torch.bool, device=st.device)
    for t in range(41):
        ex = dev[:, t] > LK_LIMIT
        skip = cint[:, t]
        cnt = torch.where(skip, cnt, torch.where(ex, cnt + 1, torch.zeros_like(cnt)))
        fail |= ~skip & (cnt >= LK_STEPS)
    return (~fail).float()


def _segs_cross(p1, p2, q1, q2) -> torch.Tensor:
    """segments p1-p2 [.., 2] and q1-q2 [.., 2] (broadcast) intersect (touching counts)"""
    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    return ((o1 * o2) <= 0) & ((o3 * o4) <= 0)


def traffic_light(st: torch.Tensor, red: torch.Tensor, red_t: torch.Tensor) -> torch.Tensor:
    """st [K, 41, .]; red [S, 5] polygon edges (pid); red_t [P, 41] red steps -> TLC [K] (0 iff the ego polygon at some
    step 0..40 intersects a polygon red at that step): corner in polygon | polygon vertex in box | edges cross"""
    k = st.shape[0]
    if red.shape[0] == 0:
        return torch.ones(k, device=st.device)
    cor = EV._corners(st)                                                        # [K, 41, 4, 2]
    pid = red[:, 4].long()
    npoly = red_t.shape[0]
    inside = EV._in_polys(cor.reshape(-1, 2), red).reshape(k, 41, 4, -1).any(2)  # [K, 41, P]
    c, u = EV._ego_box(st)
    v = torch.stack([-u[..., 1], u[..., 0]], -1)
    rel = red[None, None, :, :2] - c[:, :, None]                                  # polygon vertices in the box frame
    vin = ((rel * u[:, :, None]).sum(-1).abs() <= EV.HALF_L) & ((rel * v[:, :, None]).sum(-1).abs() <= EV.HALF_W)
    e1, e2 = cor, cor.roll(-1, dims=2)                                           # 4 box edges
    cross = _segs_cross(e1[:, :, :, None], e2[:, :, :, None], red[None, None, None, :, :2], red[None, None, None, :, 2:4])
    hit_seg = vin | cross.any(2)                                                 # [K, 41, S]
    hit = torch.zeros(k, 41, npoly, dtype=torch.uint8, device=st.device)
    hit.scatter_reduce_(2, pid[None, None].expand(k, 41, -1), hit_seg.to(torch.uint8), reduce="amax")
    hit = hit.bool() | inside
    hit &= (red_t.T > 0.5)[None]
    return (~hit.any(-1).any(-1)).float()


@lru_cache(maxsize=None)
def _comfort_ops_v2(n: int, dt: float = 0.1):
    """the v2 comfort filters as [n, n] matrices (out = x @ M) from scipy / the shared _approximate_derivatives"""
    from scipy.signal import savgol_filter
    from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_comfort_metrics as cm
    eye = np.eye(n)
    tp = np.arange(n) * dt
    acc = savgol_filter(eye, polyorder=2, window_length=min(8, n), axis=-1)
    jerk = acc @ cm._approximate_derivatives(eye, tp, window_length=min(15, n), poly_order=2, deriv_order=1)
    yaw_rate = cm._approximate_derivatives(eye, tp, poly_order=2, deriv_order=1)          # window 5
    yaw_acc = cm._approximate_derivatives(eye, tp, poly_order=3, deriv_order=2)           # window 5
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in
            (("acc", acc), ("jerk", jerk), ("yaw_rate", yaw_rate), ("yaw_acc", yaw_acc))}


def comfort_v2(s9: torch.Tensor) -> torch.Tensor:
    """[..., n, 9] (x, y, heading, vx, vy, ax, ay, w, dw) -> ego_is_comfortable of navsim_v2 [...]"""
    ops = EV._dev_ops(_comfort_ops_v2(s9.shape[-2]), s9.device)
    s = s9.double()
    ax, ay, w, dw, h = s[..., 5], s[..., 6], s[..., 7], s[..., 8], s[..., 2]
    ax_c = ax + EV.CENTRE * w ** 2 + EV.CENTRE * dw                             # get_acceleration_shifted, d = (L, 0)
    hu = h.clone()
    hu[..., 1:] = h[..., 1:] - 2 * np.pi * torch.cumsum(torch.round(torch.diff(h, dim=-1) / (2 * np.pi)), -1)
    lon, lat = ax_c @ ops["acc"], ay @ ops["acc"]
    jm, jx = torch.hypot(ax, ay) @ ops["jerk"], ax_c @ ops["jerk"]
    yr, ya = hu @ ops["yaw_rate"], hu @ ops["yaw_acc"]
    ok = ((lon >= -4.05) & (lon <= 2.40)).all(-1) & (lat.abs() <= 4.89).all(-1) & (jm.abs() <= 8.37).all(-1) \
        & (jx.abs() <= 4.13).all(-1) & (ya.abs() <= 1.93).all(-1) & (yr.abs() <= 0.95).all(-1)
    return ok.float()


def _per_sample(t: Dict[str, torch.Tensor], i: int, key: str, mask: str) -> torch.Tensor:
    return t[key][i][t[mask][i]]


@torch.no_grad()
def official_scores(st9: torch.Tensor, t1: Dict[str, torch.Tensor], t2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """st9 [B, K, 41, 9] tracked candidates (track(..., full=True, angular=True), init = the token's ego state);
    t1 / t2 the packed v1 / v2 tables (pack_full) -> per-candidate official terms [B, K]:
      v1: nc1 dac1 ddc1 ttc1 ep1 c1 pdms1 (+ prog1 raw progress, eff1 PDM-Closed raw * multiplicative)
      v2: nc2 dac2 ddc2 tlc2 ttc2 lk2 hc2 ep2 epdms2 (EC excluded), after the human_penalty_filter
    rows whose table is missing (ex1_valid / ex2_valid 0) are NaN"""
    b, k = st9.shape[:2]
    dev = st9.device
    out = {n: torch.full((b, k), float("nan"), device=dev) for n in
           ("nc1", "dac1", "ddc1", "ttc1", "ep1", "c1", "pdms1", "prog1", "eff1",
            "nc2", "dac2", "ddc2", "tlc2", "ttc2", "lk2", "hc2", "ep2", "epdms2", "prog2", "eff2")}
    c1 = EV.exact_comfort(st9[..., :7])
    for i in range(b):
        for ver, t in ((1, t1), (2, t2)):
            if t is None or float(t[f"ex{ver}_valid"][i]) < 0.5:
                continue
            g = {kk[4:]: v[i:i + 1] for kk, v in t.items() if kk.startswith(f"ex{ver}_")}   # one sample, ex_* names
            pdm = g["pdm"][0]                                                              # [41, 9]
            rows = [st9[i], pdm[None]]
            if ver == 2 and float(g["hvalid"][0]) > 0.5:
                rows.append(g["hum_st"][0][None])
            S = torch.cat(rows, 0)                                                         # [K + 1 (+1), 41, 9]
            n = S.shape[0]
            gg = {"ex_" + kk: v for kk, v in g.items()}
            seg, lane = g["seg"][0][g["smask"][0]], g["lane"][0][g["lmask"][0]]
            inter, route = g["int"][0][g["imask"][0]], g["route"][0][g["rmask"][0]]
            nondrv, multi, inter_, onc, cint = EV.ego_areas(S[..., :7], seg, lane, inter, route)
            nc, ttc = EV.exact_nc_ttc(S[None, ..., :7], gg, ttc_last=40 if ver == 1 else 31, areas=[(nondrv, multi, inter_)])
            nc, ttc = nc[0], ttc[0]
            dac = (~nondrv.any(-1)).float()
            cl = g["cl"][0][g["clmask"][0]]
            prog = EV.exact_progress(S[None, ..., :7], cl[None], torch.ones(1, cl.shape[0], dtype=torch.bool, device=dev))[0]
            step = _step_progress(S)
            if ver == 1:
                o = step * onc.float()                                                     # no intersection exemption
                ddc = _thr_ddc((o[:-1] + o[-1:]).max(-1).values)                          # candidate rows: + PDM-Closed
                ddc = torch.cat([ddc, _thr_ddc(o[-1].max())[None]])                        # PDM-Closed: its own
                mult = nc * dac * ddc
                eff = prog[-1] * mult[-1]
                masked = prog[:-1] * mult[:-1]
                norm = torch.maximum(masked, eff)
                ep = torch.where(norm > EP_THR_V1, masked / norm.clamp(min=1e-12), (mult[:-1] != 0).float())
                pdms = mult[:-1] * (5 * ep + 5 * ttc[:-1] + 2 * c1[i]) / 12
                for nm, v in (("nc1", nc[:-1]), ("dac1", dac[:-1]), ("ddc1", ddc[:-1]), ("ttc1", ttc[:-1]), ("ep1", ep),
                              ("c1", c1[i]), ("pdms1", pdms), ("prog1", prog[:-1]), ("eff1", eff.expand(k))):
                    out[nm][i] = v
            else:
                o = step * (onc & ~cint).float()
                win = torch.stack([o[:, max(0, tt - DDC_H):tt + 1].sum(-1) for tt in range(41)], -1)
                ddc = _thr_ddc(win.max(-1).values)
                tlc = traffic_light(S, g["red"][0][g["redmask"][0]], g["red_t"][0][g["redtmask"][0]])
                lk = lane_keeping(S, cl[:, :2], cint)
                past = g["past"][0][g["pmask"][0]]                                          # [15, 9]
                hc = comfort_v2(torch.cat([past[None].expand(n, -1, -1), S], 1)) if past.shape[0] else comfort_v2(S)
                mult = nc * dac * ddc * tlc
                pi = k                                                                     # PDM-Closed row
                norm = torch.maximum(prog[:k] * mult[:k], prog[pi] * mult[pi])
                ep = torch.where(norm > 5.0, (prog[:k] / norm.clamp(min=1e-12)).clamp(0, 1), torch.ones_like(norm))
                terms = {"nc2": nc[:k], "dac2": dac[:k], "ddc2": ddc[:k], "tlc2": tlc[:k], "ttc2": ttc[:k],
                         "lk2": lk[:k], "hc2": hc[:k]}
                if n == k + 2:                                                             # human_penalty_filter
                    hu = {"nc2": nc[-1], "dac2": dac[-1], "ddc2": ddc[-1], "tlc2": tlc[-1], "ttc2": ttc[-1],
                          "lk2": lk[-1], "hc2": hc[-1]}
                    terms = {nm: torch.where(hu[nm] == 0, torch.ones_like(v), v) for nm, v in terms.items()}
                m2 = terms["nc2"] * terms["dac2"] * terms["ddc2"] * terms["tlc2"]
                epdms = m2 * (5 * ep + 5 * terms["ttc2"] + 2 * terms["lk2"] + 2 * terms["hc2"]) / 14
                for nm, v in terms.items():
                    out[nm][i] = v
                out["ep2"][i], out["epdms2"][i], out["prog2"][i] = ep, epdms, prog[:k]
                out["eff2"][i] = (prog[pi] * mult[pi]).expand(k)
    return out


# ------------------------------------------------------------------------------------------------- packing
P_MAX, PAST_MAX, REDT_MAX, CL_FULL_MAX = 41, 32, 64, 8192


def pack_full(tab: Dict[str, np.ndarray], prefix: str, ego: np.ndarray = None) -> Dict[str, torch.Tensor]:
    """build_exact.py rows of one token (all tables) -> {prefix + name: tensor}: the 7 geometry tables of
    exact_verdict.pack_token (centerline up to CL_FULL_MAX vertices), red_t, pdm [41, 9], past [PAST_MAX, 9], hum [8, 3]"""
    cl = tab["cl"]
    base = EV.pack_token(tab["agent_rows"], tab["drv_seg"], tab["lane_seg"], tab["int_seg"], tab["route_seg"],
                         np.zeros((0, 6), np.float32), np.zeros((0, 3), np.float32))
    out = {prefix + kk[3:]: v for kk, v in base.items()}
    red = tab["red_seg"]
    r = np.zeros((EV.RED_MAX, 5), np.float32)
    rm = np.zeros(EV.RED_MAX, bool)
    nr = min(len(red), EV.RED_MAX)
    r[:nr], rm[:nr] = red[:nr], True
    rt = np.zeros((REDT_MAX, 41), np.float32)
    rtm = np.zeros(REDT_MAX, bool)
    npl = min(len(tab["red_t"]), REDT_MAX)
    rt[:npl], rtm[:npl] = tab["red_t"][:npl], True
    c = np.zeros((CL_FULL_MAX, 3), np.float32)
    cm = np.zeros(CL_FULL_MAX, bool)
    ncl = min(len(cl), CL_FULL_MAX)
    if len(cl) > CL_FULL_MAX:                                     # (navtest max 3821) keep the vertices nearest the ego
        j = int(np.argmin(np.hypot(cl[:, 0], cl[:, 1])))
        j0 = int(np.clip(j - CL_FULL_MAX // 4, 0, len(cl) - CL_FULL_MAX))
        cl = cl[j0:j0 + CL_FULL_MAX]
    c[:ncl], cm[:ncl] = cl[:ncl], True
    pdm = np.zeros((P_MAX, 9), np.float32)
    if len(tab["pdm"]) == P_MAX:
        pdm[:] = tab["pdm"]
    past = np.zeros((PAST_MAX, 9), np.float32)
    pm = np.zeros(PAST_MAX, bool)
    npa = min(len(tab["past"]), PAST_MAX)
    past[:npa], pm[:npa] = tab["past"][:npa], True
    hum = np.zeros((8, 3), np.float32)
    hv = len(tab["hum"]) == 8
    if hv:
        hum[:] = tab["hum"]
    trunc = float(base["ex_trunc"]) + max(len(red) - EV.RED_MAX, 0) + max(len(tab["red_t"]) - REDT_MAX, 0) \
        + max(len(tab["cl"]) - CL_FULL_MAX, 0) + max(len(tab["past"]) - PAST_MAX, 0)
    out.update({prefix + "red": torch.from_numpy(r), prefix + "redmask": torch.from_numpy(rm),
                prefix + "red_t": torch.from_numpy(rt), prefix + "redtmask": torch.from_numpy(rtm),
                prefix + "cl": torch.from_numpy(c), prefix + "clmask": torch.from_numpy(cm),
                prefix + "pdm": torch.from_numpy(pdm), prefix + "past": torch.from_numpy(past),
                prefix + "pmask": torch.from_numpy(pm), prefix + "hum": torch.from_numpy(hum),
                prefix + "hvalid": torch.tensor(float(hv)), prefix + "trunc": torch.tensor(trunc)})
    if ego is not None:
        out[prefix + "ego"] = torch.from_numpy(np.array(ego, np.float32).reshape(9))
    return out


def track_human(t2: Dict[str, torch.Tensor], init: torch.Tensor) -> None:
    """t2['ex2_hum'] [B, 8, 3] -> t2['ex2_hum_st'] [B, 41, 9]: the human trajectory through the same simulation"""
    t2["ex2_hum_st"] = track(t2["ex2_hum"].double(), init.double(), full=True, angular=True).float()


def ego_init(ego9: torch.Tensor) -> torch.Tensor:
    """build_exact ego row [.., 9] (vx, vy, ax, ay, steer, steer rate, w, dw, 0) -> track() init [.., 8]"""
    return ego9[..., :8]


# ------------------------------------------------------------------------------------------------- target builder
V1_TABLES = ("agent_rows", "drv_seg", "lane_seg", "int_seg", "route_seg", "cl", "pdm", "ego")
V2_TABLES = ("agent_rows", "drv_seg", "lane_seg", "int_seg", "route_seg", "red_seg", "red_t", "cl", "pdm", "hum", "past", "ego")


def _empty(names):
    w = dict(zip(("agent_rows", "drv_seg", "lane_seg", "int_seg", "route_seg", "red_seg", "red_t", "cl", "pdm", "hum",
                  "past", "ego"), (9, 5, 5, 5, 5, 5, 41, 3, 9, 3, 9, 9)))
    return {k: np.zeros((0, w[k]), np.float32) for k in w}


from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder  # noqa: E402


class ExactFullTargetBuilder(AbstractTargetBuilder):
    """km 10/19 (C v3): the v1 tables (config.exact_dir/<split>: geometry of the v1 metric cache + pdm / ego of
    build_exact.py --only pdm,ego) and the v2 tables (config.exact2_dir/<split>, build_exact.py --v2) of the current
    frame -> ex1_* / ex2_* (pack_full), ex1_valid / ex2_valid, ex_ego [9].  The v1 progress / centerline is the whole
    centerline of the v2 table when there is one (the two caches share map, route and centerline: navtrain_val 40/40)."""

    def __init__(self, config):
        from .corridor_head import SPLITS
        self._splits = SPLITS
        self._d1, self._d2 = str(config.exact_dir), str(config.exact2_dir)
        self._troot = str(config.corridor_tokens_root)
        self._arr = None

    def get_unique_name(self) -> str:
        return "para_ssr_exact_official"

    def _open(self):
        if self._arr is not None:
            return
        import os
        import pandas as pd
        self._index, self._arr = {}, {}
        self._clf = {}
        for sp in self._splits:
            toks = pd.read_parquet(f"{self._troot}/{sp}/tokens.parquet").token.astype(str).to_numpy()
            ok1 = np.load(f"{self._d1}/{sp}/ok.npy") & np.load(f"{self._d1}/{sp}/ok_ego_pdm.npy")
            # a split still being filled (merge_exact_parts.py) is read through <split>/current -> a complete version,
            # resolved once per open (= once per epoch for the non-persistent loader workers)
            d2 = f"{self._d2}/{sp}/current" if os.path.exists(f"{self._d2}/{sp}/current") else f"{self._d2}/{sp}"
            d2 = os.path.realpath(d2)
            ok2 = np.load(f"{d2}/ok.npy") if os.path.exists(f"{d2}/ok.npy") else np.zeros(len(toks), bool)
            self._index.update({t: (sp, i, bool(ok1[i]), bool(ok2[i])) for i, t in enumerate(toks) if ok1[i] or ok2[i]})
            self._arr[sp] = ({k: (np.load(f"{self._d1}/{sp}/{k}.npy", mmap_mode="r"), np.load(f"{self._d1}/{sp}/{k}_off.npy"))
                              for k in V1_TABLES},
                             {k: (np.load(f"{d2}/{k}.npy", mmap_mode="r"), np.load(f"{d2}/{k}_off.npy"))
                              for k in V2_TABLES} if ok2.any() else None)
            # the whole v1 centerline (build_exact.py --only cl --cl-full -> <exact_dir>_clfull; ok_cl.npy is written last)
            dc = f"{self._d1}_clfull/{sp}"
            if os.path.exists(f"{dc}/ok_cl.npy"):
                self._clf[sp] = (np.load(f"{dc}/ok_cl.npy"), np.load(f"{dc}/cl.npy", mmap_mode="r"), np.load(f"{dc}/cl_off.npy"))

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        self._open()
        token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
        hit = self._index.get(token)
        t1, t2 = _empty(V1_TABLES), _empty(V2_TABLES)
        v1 = v2 = False
        if hit is not None:
            sp, r, v1, v2 = hit
            A1, A2 = self._arr[sp]
            if v2:
                t2.update({k: np.asarray(v[o[r]:o[r + 1]], np.float32) for k, (v, o) in A2.items()})
            if v1:
                t1.update({k: np.asarray(v[o[r]:o[r + 1]], np.float32) for k, (v, o) in A1.items()})
                if v2:
                    t1["cl"] = t2["cl"]
                elif sp in self._clf and self._clf[sp][0][r]:
                    okc, v, o = self._clf[sp]
                    t1["cl"] = np.asarray(v[o[r]:o[r + 1]], np.float32)
        ego = t1["ego"] if len(t1["ego"]) else (t2["ego"] if len(t2["ego"]) else np.zeros((1, 9), np.float32))
        out = {**pack_full(t1, "ex1_"), **pack_full(t2, "ex2_")}
        out["ex1_valid"], out["ex2_valid"] = torch.tensor(float(v1)), torch.tensor(float(v2))
        out["ex_ego"] = torch.from_numpy(ego.reshape(9).astype(np.float32))
        return out
