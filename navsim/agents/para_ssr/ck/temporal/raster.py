"""refiner-v3 path-time rasteriser and gap scalars (docs/refiner_v3/SPEC.md section 4.4).  torch, parameter-free.

Per candidate b (B = T * Kc, token of candidate b = b // Kc), on the candidate's OWN corridor stations
(refiner.corridor.corridor_geometry; speed variants are rasterised like any other candidate, nothing is shared with
the parent except the per-token forecast).  float32 with autocast disabled; differentiable in the forecast (mu, b, pi,
present) except for the index searches (no_grad).

rasterize(fc, geom, T, Kc, roi) -> R [B, 18, 48, 21]   (RASTER_CHANNELS; station j = 0..47, time bin t_k = 0.25 k s)
  known            1 iff the 4 corners p_j +- (ds/2) u_j +- (EGO_W/2 + KNOWN_MARGIN) n_j lie inside the source ROI
  ego_occ          cov(s_e(t_k) + EGO_REAR, s_e(t_k) + EGO_FRONT, j): overlap of the ego footprint's arc-length
                   interval with station bin j, / ds
  ego_tap{03,06,09} same for the ego projected by v_e(t_k) * delta (PDM TTC layout), 0 for t_k > TAP_T_MAX
  {veh,vru}_{occ,vclose,dapp,inc,ext}, st_{occ,inc,ext}
                   object mass w = pi_m * present * P_lat splatted linearly in s at the exact time t_k, attributes
                   occ = 1, vclose = clip(sgn(s - s_e) (v_e - V.u_bar), +-15) / 15, dapp = clip(-sgn(d) V.n_bar, +-5) / 5,
                   inc = clamp(H - |d|, 0, 3) / 3, ext = e_s / 5 (H = EGO_W/2 + e_d; P_lat = Laplace overlap of
                   [-H, H] with scale b_d = max(B_MIN, |(b_x n_x, b_y n_y)|)).
gap_scalars(fc, tau_flat, T, Kc) -> G [B, 20]
  port of a3_conflict_features interp_traj / sep_rect / 9-feature reduction:
    0..8   S world (t0 boxes held, all classes)
    9..17  forecast world (agents follow the forecast, statics held): per object, the pi-weighted mean over modes of
           the clipped separation (see "Deviation" below)
    18, 19 min over the plausible modes (pi >= 0.1, argmax if none) of the forecast-world f0, f3
  normalisation: separations / 5, first-hit times / 4.5, conflict count / 16.

Deviation from SPEC 4.4 (documented in docs/refiner_v3/BUILD_model.md)
  "per mode, then pi-weighted mean" is applied PER OBJECT (pi is per object; there is no scene-level mode, mode m of
  two objects is not a joint world): each object's separation series (clipped to [-5, 15], 15 where it is absent or,
  for the "ahead" features, not ahead) is averaged over its modes with its pi, then the a3 min / first-hit / count
  reductions run on the averaged series.  For M = 1 (arms b, d) this is exactly the a3 world.
  forecast_at rounds the presence knot with floor(t / 0.5 + 0.5) (round half up; torch.round is half-to-even).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from ...refiner.corridor import DS_MIN, N_ST
from ...refiner.geometry import N_POSE, T_POSE, unwrap_anchored
from .config import (B0, B_MIN, EGO_FRONT, EGO_L, EGO_OFF, EGO_REAR, EGO_W, FC_DT, FC_STEPS, KNOWN_MARGIN, LAT_DROP,
                     N_RT, R_T, TAP_T_MAX, TAPS, TJ)
from .forecast import Forecast

Tensor = torch.Tensor
RASTER_CHANNELS = ("known", "ego_occ", "ego_tap03", "ego_tap06", "ego_tap09",
                   "veh_occ", "veh_vclose", "veh_dapp", "veh_inc", "veh_ext",
                   "vru_occ", "vru_vclose", "vru_dapp", "vru_inc", "vru_ext",
                   "st_occ", "st_inc", "st_ext")
C_R = len(RASTER_CHANNELS)
GAP_NAMES = tuple(f"{w}_{n}" for w in ("S", "F") for n in ("g0_ag", "g0_agF", "g0_st", "gT_ag", "gT_agF", "gT_st",
                                                         "tf0", "tfT", "nconf")) + ("Fmin_g0_ag", "Fmin_gT_ag")
N_GAP = len(GAP_NAMES)
# object channel of (group, attribute a in occ, vclose, dapp, inc, ext); -1 = not rasterised
_OBJ_CH = ((5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, -1, -1, 16, 17))
SEP_CLIP = (-5.0, 15.0)
HIT0, HIT_T, CONF = 0.3, 0.3, 0.5          # a3 thresholds
T_NONE = 4.5
PI_PLAUSIBLE = 0.1
ELEM_BUDGET = {"cpu": 1 << 23, "cuda": 1 << 26}   # elements per chunk temporary ([b, P, 48] search, [b, P, 16],
                                                  # [b, 64, K, M+1]); module-level so the smoke can tune it


def _time(vals: Sequence[float], like: Tensor) -> Tensor:
    return torch.as_tensor(vals, dtype=torch.float32, device=like.device)


# candidates per chunk when the caller passes chunk=None (the default).  CUDA (Integrate 2026-10-11): the raster is
# kernel-launch bound (~13k launches per arm-a step at cap 128 / budget 2^24 [실측 bench_gpu_step]), so the GPU runs
# few large chunks; chunking is exact (per-candidate ops), CPU values unchanged (128, 2^23)
CHUNK_CAP = {"cpu": 128, "cuda": 1024}


def _chunks(n: int, per: int, cap: Optional[int], device=None):
    dt = getattr(device, "type", "cpu")
    budget = ELEM_BUDGET.get(dt, ELEM_BUDGET["cpu"])
    if cap is None:
        cap = CHUNK_CAP.get(dt, CHUNK_CAP["cpu"])
    step = max(1, min(cap, budget // max(per, 1)))
    for i in range(0, n, step):
        yield i, min(n, i + step)


# ----------------------------------------------------------------------------------------------- stations / ego
def station_frame(geom) -> Dict[str, Tensor]:
    """Path frame at the 48 stations: p (= Gamma(s_j)), u (tangent), n (left normal) [B,48,2], s [B,48], ds [B]."""
    with torch.no_grad():
        pe = geom.path.eval(geom.s)
        ds = torch.clamp(geom.S_look / N_ST, min=DS_MIN)
    return {"p": pe.xy.float(), "u": pe.tangent.float(), "n": pe.normal.float(), "s": geom.s.float(),
            "ds": ds.float()}


def ego_schedule(geom, t: Tensor) -> Tuple[Tensor, Tensor]:
    """Draft arc length s_e and speed v_e at times t [nt] -> [B, nt] each (piecewise linear through (0.5 i, S_i);
    beyond 4 s: S_8 + v_end (t - 4), v_e = v_end)."""
    S = geom.S.float()
    v_end = geom.v_end.float()
    B = S.shape[0]
    t = t.float()
    k = torch.clamp(torch.floor(t / T_POSE), 0, N_POSE - 1).long()
    Sk = S[:, k]
    Sk1 = S[:, k + 1]
    after = (t >= N_POSE * T_POSE)[None]
    s_in = Sk + (Sk1 - Sk) * (t - k.float() * T_POSE)[None] / T_POSE
    s_out = S[:, -1:] + v_end[:, None] * (t - N_POSE * T_POSE)[None]
    s_e = torch.where(after, s_out, s_in)
    v_e = torch.where(after, v_end[:, None].expand(B, t.numel()), (Sk1 - Sk) / T_POSE)
    return s_e, v_e


def _cov(lo: Tensor, hi: Tensor, ds: Tensor) -> Tensor:
    """Overlap of [lo, hi] [B, nt] with station bins [j ds, (j+1) ds] / ds -> [B, 48, nt]."""
    j = torch.arange(N_ST, device=lo.device, dtype=lo.dtype)
    b_lo = (j[None] * ds[:, None])[:, :, None]
    b_hi = ((j[None] + 1) * ds[:, None])[:, :, None]
    ov = torch.minimum(hi[:, None, :], b_hi) - torch.maximum(lo[:, None, :], b_lo)
    return torch.minimum(torch.clamp(ov, min=0.0), ds[:, None, None]) / ds[:, None, None]


def ego_channels(geom) -> Tensor:
    """[B, 4, 48, 21]: ego_occ, ego_tap03, ego_tap06, ego_tap09."""
    ds = torch.clamp(geom.S_look.float() / N_ST, min=DS_MIN)
    t = _time(R_T, ds)
    s_e, v_e = ego_schedule(geom, t)
    out = [_cov(s_e + EGO_REAR, s_e + EGO_FRONT, ds)]
    gate = (t <= TAP_T_MAX + 1e-9).float()[None, None, :]
    for delta in TAPS[1:]:
        sh = s_e + v_e * delta
        out.append(_cov(sh + EGO_REAR, sh + EGO_FRONT, ds) * gate)
    return torch.stack(out, 1)


def known_channel(geom, roi, sf: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """[B, 1, 48, 21]: 1 iff the station's widened ego-band rectangle lies inside the source ROI (x0, y0, x1, y1)."""
    sf = station_frame(geom) if sf is None else sf
    x0, y0, x1, y1 = (float(v) for v in roi)
    hs = (0.5 * sf["ds"])[:, None, None]
    hw = EGO_W / 2 + KNOWN_MARGIN
    ok = torch.ones(sf["p"].shape[:2], dtype=torch.bool, device=sf["p"].device)
    for a in (-1.0, 1.0):
        for c in (-1.0, 1.0):
            q = sf["p"] + a * hs * sf["u"] + c * hw * sf["n"]
            ok = ok & (q[..., 0] >= x0) & (q[..., 0] <= x1) & (q[..., 1] >= y0) & (q[..., 1] <= y1)
    return ok.float()[:, None, :, None].expand(-1, 1, N_ST, N_RT)


# ----------------------------------------------------------------------------------------------- forecast sampling
def forecast_at(fc: Forecast, t: Tensor) -> Dict[str, Tensor]:
    """Piecewise-linear forecast through (0, c0), (0.5, mu_1) .. (5.0, mu_10) at times t [nt] (clamped to [0, 5]):
    xy, v (segment velocity), b (linear through (0, B0), (0.5, b_1), ..) [T,K,M,nt,2], present [T,K,M,nt]."""
    T, K, M = fc.mu.shape[:3]
    dt = fc.mu.dtype
    t = t.to(dt)
    c0 = fc.c0[:, :, None, None, :].expand(T, K, M, 1, 2).to(dt)
    P = torch.cat([c0, fc.mu], 3)                                                     # [T,K,M,11,2]
    Bk = torch.cat([torch.full_like(fc.b[:, :, :, :1], B0), fc.b], 3)
    Pr = torch.cat([torch.ones_like(fc.present[..., :1]), fc.present], 3)             # [T,K,M,11]
    x = torch.clamp(t / FC_DT, 0.0, float(FC_STEPS))
    i = torch.clamp(torch.floor(x), 0, FC_STEPS - 1).long()
    f = (x - i.to(dt))[:, None]
    P0, P1 = P[:, :, :, i], P[:, :, :, i + 1]
    B0k, B1k = Bk[:, :, :, i], Bk[:, :, :, i + 1]
    r = torch.clamp(torch.floor(t / FC_DT + 0.5), 0, FC_STEPS).long()
    return {"xy": P0 + f * (P1 - P0), "v": (P1 - P0) / FC_DT, "b": B0k + f * (B1k - B0k), "present": Pr[..., r]}


# ----------------------------------------------------------------------------------------------- Frenet projection
def _gather_st(a: Tensor, j: Tensor) -> Tensor:
    """a [b, 48] or [b, 48, 2], j [b, P] -> [b, P] or [b, P, 2]."""
    if a.dim() == 2:
        return torch.gather(a, 1, j)
    return torch.gather(a, 1, j[..., None].expand(*j.shape, a.shape[-1]))


def frenet(xy: Tensor, sf: Dict[str, Tensor], chunk: Optional[int] = None) -> Dict[str, Tensor]:
    """Points xy [b, P, 2] of each candidate -> s, d [b, P], u_bar, n_bar [b, P, 2], ok [b, P] (finite).
    Nearest-station search under no_grad (chunks of <= `chunk` candidates); the projection is differentiable in xy."""
    p, u, n, s = sf["p"], sf["u"], sf["n"], sf["s"]
    nb, P = xy.shape[:2]
    with torch.no_grad():
        j0 = torch.empty(nb, P, dtype=torch.long, device=xy.device)
        for i0, i1 in _chunks(nb, P * N_ST, chunk, xy.device):
            if xy.device.type == "cuda":
                # GPU (Integrate 2026-10-11): the non-matmul cdist kernel took ~15 ms per chunk, 46 % of an arm-a
                # train step [실측 bench_gpu_step profile]; the broadcast form evaluates the same float32
                # sqrt(dx*dx + dy*dy) and gives the identical argmin [실측: 0 mismatches, INTEGRATION.md]
                dd = xy[i0:i1, :, None, :] - p[i0:i1, None, :, :]
                j0[i0:i1] = torch.sqrt(dd[..., 0] * dd[..., 0] + dd[..., 1] * dd[..., 1]).argmin(-1)
            else:
                # exact pairwise distances (no |a|^2 - 2ab + |b|^2 matmul shortcut); one fused [b, P, 48] temporary
                j0[i0:i1] = torch.cdist(xy[i0:i1].contiguous(), p[i0:i1].contiguous(),
                                        compute_mode="donot_use_mm_for_euclid_dist").argmin(-1)
    pj0, uj0, nj0, sj0 = _gather_st(p, j0), _gather_st(u, j0), _gather_st(n, j0), _gather_st(s, j0)
    rel = xy - pj0
    a = (rel * uj0).sum(-1)
    with torch.no_grad():
        j1 = torch.where(a >= 0, j0 + 1, j0 - 1).clamp(0, N_ST - 1)
        same = j1 == j0
    pj1, uj1, nj1, sj1 = _gather_st(p, j1), _gather_st(u, j1), _gather_st(n, j1), _gather_st(s, j1)
    e = pj1 - pj0
    ee = (e * e).sum(-1)
    lam = torch.clamp((rel * e).sum(-1) / torch.where(same, torch.ones_like(ee), ee), 0.0, 1.0)
    lam = torch.where(same, torch.zeros_like(lam), lam)
    q = pj0 + lam[..., None] * e
    ub = (1 - lam)[..., None] * uj0 + lam[..., None] * uj1
    nbv = (1 - lam)[..., None] * nj0 + lam[..., None] * nj1
    ub = ub / torch.clamp(torch.linalg.vector_norm(ub, dim=-1, keepdim=True), min=1e-6)
    nbv = nbv / torch.clamp(torch.linalg.vector_norm(nbv, dim=-1, keepdim=True), min=1e-6)
    s_pt = torch.where(same, sj0 + a, sj0 + lam * (sj1 - sj0))
    d_pt = torch.where(same, (rel * nj0).sum(-1), ((xy - q) * nbv).sum(-1))
    ub = torch.where(same[..., None], uj0, ub)
    nbv = torch.where(same[..., None], nj0, nbv)
    ok = torch.isfinite(s_pt) & torch.isfinite(d_pt)
    return {"s": s_pt, "d": d_pt, "u_bar": ub, "n_bar": nbv, "ok": ok}


def _laplace_cdf(z: Tensor, b: Tensor) -> Tensor:
    e = 0.5 * torch.exp(-z.abs() / b)
    return torch.where(z < 0, e, 1.0 - e)


# ----------------------------------------------------------------------------------------------- raster
def rasterize(fc: Forecast, geom, T: int, Kc: int, roi, chunk: Optional[int] = None) -> Tensor:
    """Path-time raster R [T*Kc, 18, 48, 21] (module docstring)."""
    with torch.autocast(device_type=fc.mu.device.type, enabled=False):
        return _rasterize(fc, geom, T, Kc, roi, chunk)


def _rasterize(fc: Forecast, geom, T: int, Kc: int, roi, chunk: Optional[int]) -> Tensor:
    B = T * Kc
    if geom.s.shape[0] != B or fc.mu.shape[0] != T:
        raise ValueError(f"rasterize: geom has {geom.s.shape[0]} candidates, forecast {fc.mu.shape[0]} tokens, "
                         f"T * Kc = {B}")
    sf = station_frame(geom)
    dev = sf["p"].device
    ego = ego_channels(geom)
    known = known_channel(geom, roi, sf)
    t = _time(R_T, sf["p"])
    _, K, M = fc.mu.shape[:3]
    fa = {k: v.float() for k, v in forecast_at(fc, t).items()}
    P = K * M * N_RT
    # per-point attributes in token space [T, P, ...], point = (slot, mode, time bin)
    h0 = fc.h0.float()
    hh = torch.stack([torch.cos(h0), torch.sin(h0)], -1)                               # [T,K,2]
    hp = torch.stack([-hh[..., 1], hh[..., 0]], -1)
    rs = lambda x: x.reshape(T, P, *x.shape[4:])                                       # noqa: E731
    ex = lambda x: x[:, :, None, None].expand(T, K, M, N_RT, *x.shape[2:]).reshape(T, P, *x.shape[2:])  # noqa: E731
    xy_t, v_t, b_t, pres_t = rs(fa["xy"]), rs(fa["v"]), rs(fa["b"]), rs(fa["present"])
    w0 = rs(fc.pi.float()[:, :, :, None] * fa["present"] * fc.valid[:, :, None, None].float())   # [T, P]
    hh_t, hp_t, L_t, W_t, g_t = ex(hh), ex(hp), ex(fc.L.float()), ex(fc.W.float()), ex(fc.group)
    k_t = torch.arange(N_RT, device=dev)[None, None, None, :].expand(T, K, M, N_RT).reshape(T, P)
    # compaction (exact): drop points with zero mass before the per-candidate expansion (padding slots, statics'
    # zero-pi modes 1..M-1, absent oracle steps); the kept points keep their order, dropped ones only added zeros
    with torch.no_grad():
        live = w0 > 0
        P = max(1, int(live.sum(1).max()))
        keep_idx = torch.argsort((~live).to(torch.int8), dim=1, stable=True)[:, :P]           # [T, P']
    take = lambda x: torch.gather(x, 1, keep_idx.view(T, P, *([1] * (x.dim() - 2))).expand(T, P, *x.shape[2:]))  # noqa: E731
    xy_t, v_t, b_t, pres_t, w0 = take(xy_t), take(v_t), take(b_t), take(pres_t), take(w0)
    hh_t, hp_t, L_t, W_t, g_t, k_t = take(hh_t), take(hp_t), take(L_t), take(W_t), take(g_t), take(k_t)
    ch_tab = torch.as_tensor(_OBJ_CH, dtype=torch.long, device=dev)                   # [3, 5]
    s_e_all, v_e_all = ego_schedule(geom, t)                                          # [B, nt]
    cell = N_ST * N_RT
    out = []
    for i0, i1 in _chunks(B, P * 16, chunk, dev):
        tok = torch.arange(i0, i1, device=dev) // Kc
        nb = i1 - i0
        sfc = {k: v[i0:i1] for k, v in sf.items()}
        fr = frenet(xy_t[tok], sfc, chunk)
        ub, nbv, s, d = fr["u_bar"], fr["n_bar"], fr["s"], fr["d"]
        hhc, hpc, Lc, Wc = hh_t[tok], hp_t[tok], L_t[tok], W_t[tok]
        e_s = 0.5 * (Lc * (hhc * ub).sum(-1).abs() + Wc * (hpc * ub).sum(-1).abs())
        e_d = 0.5 * (Lc * (hhc * nbv).sum(-1).abs() + Wc * (hpc * nbv).sum(-1).abs())
        ds = sfc["ds"][:, None]
        H = EGO_W / 2 + e_d
        bc = b_t[tok]
        b_d = torch.clamp(torch.sqrt((bc[..., 0] * nbv[..., 0]) ** 2 + (bc[..., 1] * nbv[..., 1]) ** 2), min=B_MIN)
        p_lat = _laplace_cdf(H - d, b_d) - _laplace_cdf(-H - d, b_d)
        w0c = w0[tok]
        keep = ((s >= -0.5 * ds) & (s < N_ST * ds) & ((d.abs() - e_d) <= LAT_DROP) & (pres_t[tok] > 0)
                & (w0c > 0) & fr["ok"])
        w = torch.where(keep, w0c * p_lat, torch.zeros_like(p_lat))
        kk = k_t[tok]
        s_e = torch.gather(s_e_all[i0:i1], 1, kk)
        v_e = torch.gather(v_e_all[i0:i1], 1, kk)
        V = v_t[tok]
        attr = torch.stack([torch.ones_like(s),
                            torch.clamp(torch.sign(s - s_e) * (v_e - (V * ub).sum(-1)), -15.0, 15.0) / 15.0,
                            torch.clamp(-torch.sign(d) * (V * nbv).sum(-1), -5.0, 5.0) / 5.0,
                            torch.clamp(H - d.abs(), 0.0, 3.0) / 3.0,
                            e_s / 5.0], -1)                                            # [nb, P, 5]
        ch = ch_tab[g_t[tok]]                                                          # [nb, P, 5]
        with torch.no_grad():
            j_lo = torch.floor(s / ds - 0.5)
        f = (s / ds - 0.5) - j_lo
        j_lo = j_lo.long()
        vals, idxs = [], []
        for jj, wt in ((j_lo, 1.0 - f), (j_lo + 1, f)):
            m5 = ((jj >= 0) & (jj < N_ST))[..., None] & (ch >= 0)
            vals.append(torch.where(m5, (w * wt)[..., None] * attr, torch.zeros_like(attr)))
            idxs.append(ch.clamp(min=0) * cell + jj.clamp(0, N_ST - 1)[..., None] * N_RT + kk[..., None])
        val = torch.cat(vals, 1).reshape(nb, -1)
        idx = torch.cat(idxs, 1).reshape(nb, -1)
        out.append(torch.zeros(nb, C_R * cell, dtype=torch.float32, device=dev).scatter_add(1, idx, val))
    obj = torch.cat(out, 0).reshape(B, C_R, N_ST, N_RT)
    return torch.cat([known, ego, obj[:, 5:]], 1)


# ----------------------------------------------------------------------------------------------- gap scalars (a3)
def ego_tj(tau_flat: Tensor) -> Dict[str, Tensor]:
    """a3 interp_traj: rear-axle poses at TJ = 0.25 .. 4 s [B,16,3] (linear interpolation of [origin; poses], heading
    unwrapped) and segment speeds sp [B,16] (segment min(ceil(TJ / 0.5 - 1e-9), 8))."""
    tau = tau_flat.float()
    B = tau.shape[0]
    P9 = torch.cat([tau.new_zeros(B, 1, 3), tau], 1)
    P9 = torch.cat([P9[..., :2], unwrap_anchored(P9[..., 2])[..., None]], -1)
    tj = torch.as_tensor(TJ, dtype=torch.float64)
    i = torch.clamp(torch.floor(tj / T_POSE), 0, N_POSE - 1).long()
    f = ((tj - i.double() * T_POSE) / T_POSE).float().to(tau.device)
    i = i.to(tau.device)
    pose = P9[:, i] + f[None, :, None] * (P9[:, i + 1] - P9[:, i])
    seg = torch.hypot(P9[:, 1:, 0] - P9[:, :-1, 0], P9[:, 1:, 1] - P9[:, :-1, 1]) / T_POSE   # [B, 8]
    idx = torch.clamp(torch.ceil(tj / T_POSE - 1e-9), max=N_POSE).long().to(tau.device)
    return {"pose": pose, "sp": seg[:, idx - 1]}


def sat_sep(ce: Tensor, he: Tensor, co: Tensor, ho: Tensor, Lo: Tensor, Wo: Tensor) -> Tuple[Tensor, Tensor]:
    """a3 sep_rect: SAT separation between ego boxes (centre ce [..., 2], heading he) and object boxes (co, ho, Lo,
    Wo); all arguments broadcast.  -> (separation, forward = (co - ce) . u_e)."""
    cue, sue = torch.cos(he), torch.sin(he)
    cuo, suo = torch.cos(ho), torch.sin(ho)
    dx, dy = co[..., 0] - ce[..., 0], co[..., 1] - ce[..., 1]
    le, we = EGO_L / 2, EGO_W / 2
    lo, wo = Lo / 2, Wo / 2
    # axes: ue = (cue, sue), ve = (-sue, cue), uo = (cuo, suo), vo = (-suo, cuo)
    uo_ue = cuo * cue + suo * sue
    vo_ue = -suo * cue + cuo * sue
    uo_ve = -cuo * sue + suo * cue
    vo_ve = suo * sue + cuo * cue
    d_ue = dx * cue + dy * sue
    d_ve = -dx * sue + dy * cue
    d_uo = dx * cuo + dy * suo
    d_vo = -dx * suo + dy * cuo
    g1 = d_ue.abs() - le - (lo * uo_ue.abs() + wo * vo_ue.abs())
    g2 = d_ve.abs() - we - (lo * uo_ve.abs() + wo * vo_ve.abs())
    g3 = d_uo.abs() - (le * uo_ue.abs() + we * uo_ve.abs()) - lo
    g4 = d_vo.abs() - (le * vo_ue.abs() + we * vo_ve.abs()) - wo
    return torch.maximum(torch.maximum(g1, g2), torch.maximum(g3, g4)), d_ue


def _reduce9(s_ag: Tensor, s_agF: Tensor, s_st: Tensor) -> Tensor:
    """a3 9-feature reduction of clipped separations [b, 16, 4] -> [b, 9] (raw units)."""
    tj = torch.as_tensor(TJ, dtype=s_ag.dtype, device=s_ag.device)
    f = [s_ag[:, :, 0].amin(1), s_agF[:, :, 0].amin(1), s_st[:, :, 0].amin(1),
         s_ag[:, :, 1:].amin((1, 2)), s_agF[:, :, 1:].amin((1, 2)), s_st[:, :, 1:].amin((1, 2))]
    hit0 = s_ag[:, :, 0] < HIT0
    hitT = s_agF[:, :, 1:].amin(2) < HIT_T
    first = lambda h: torch.where(h.any(1), tj[h.float().argmax(1)], torch.full_like(f[0], T_NONE))  # noqa: E731
    f += [first(hit0), first(hitT), (s_agF.amin(2) < CONF).sum(1).to(s_ag.dtype)]
    return torch.stack(f, -1)


def _norm9(f: Tensor) -> Tensor:
    s = torch.tensor([5.0] * 6 + [T_NONE, T_NONE, 16.0], dtype=f.dtype, device=f.device)
    return f / s


def gap_scalars(fc: Forecast, tau_flat: Tensor, T: int, Kc: int, chunk: Optional[int] = None,
                raw: bool = False) -> Tensor:
    """[B, 20] gap scalars (module docstring); raw=True returns unnormalised values (tests)."""
    with torch.autocast(device_type=tau_flat.device.type, enabled=False):
        return _gap_scalars(fc, tau_flat, T, Kc, chunk, raw)


def _gap_scalars(fc: Forecast, tau_flat: Tensor, T: int, Kc: int, chunk: Optional[int], raw: bool) -> Tensor:
    B = T * Kc
    if tau_flat.shape[0] != B or fc.mu.shape[0] != T:
        raise ValueError(f"gap_scalars: {tau_flat.shape[0]} candidates, {fc.mu.shape[0]} tokens, T * Kc = {B}")
    dev = tau_flat.device
    _, K, M = fc.mu.shape[:3]
    eg = ego_tj(tau_flat)
    pose, sp = eg["pose"], eg["sp"]
    hd = pose[..., 2]
    u = torch.stack([torch.cos(hd), torch.sin(hd)], -1)
    deltas = torch.as_tensor(TAPS, dtype=torch.float32, device=dev)
    cen = pose[..., :2] + EGO_OFF * u                                                  # [B,16,2]
    ce = cen[:, :, None, :] + (sp[:, :, None, None] * deltas[None, None, :, None]) * u[:, :, None, :]
    ce = ce.reshape(B, 64, 2)
    he = hd[:, :, None].expand(B, 16, 4).reshape(B, 64)
    tau = (torch.as_tensor(TJ, dtype=torch.float64)[:, None] + torch.as_tensor(TAPS, dtype=torch.float64)[None]).reshape(-1)
    fa = forecast_at(fc, tau.to(device=dev, dtype=fc.mu.dtype))                        # xy [T,K,M,64,2]
    xyF = fa["xy"].float()
    presF = fa["present"].float() > 0                                                  # [T,K,M,64]
    c0 = fc.c0.float()
    h0, L, W = fc.h0.float(), fc.L.float(), fc.W.float()
    valid = fc.valid
    agent = valid & (fc.group < 2)
    static = valid & (fc.group == 2)
    pi = fc.pi.float()
    big = SEP_CLIP[1]
    plaus = pi >= PI_PLAUSIBLE
    plaus = torch.where(plaus.any(-1, keepdim=True), plaus,
                        torch.nn.functional.one_hot(pi.argmax(-1), M).bool())          # [T,K,M]
    outs = []
    for i0, i1 in _chunks(B, 64 * K * (M + 1), chunk, dev):
        tok = torch.arange(i0, i1, device=dev) // Kc
        cec = ce[i0:i1][:, :, None, :]                                                 # [b,64,1,2]
        hec = he[i0:i1][:, :, None]
        # S world: t0 boxes held
        sS, fwS = sat_sep(cec, hec, c0[tok][:, None], h0[tok][:, None], L[tok][:, None], W[tok][:, None])  # [b,64,K]
        sS = torch.clamp(sS, *SEP_CLIP)
        ag = agent[tok][:, None]
        st = static[tok][:, None]
        s_ag = torch.where(ag, sS, torch.full_like(sS, big)).amin(-1).reshape(-1, 16, 4)
        s_agF = torch.where(ag & (fwS > 0), sS, torch.full_like(sS, big)).amin(-1).reshape(-1, 16, 4)
        s_st = torch.where(st, sS, torch.full_like(sS, big)).amin(-1).reshape(-1, 16, 4)
        fS = _reduce9(s_ag, s_agF, s_st)
        # forecast world: agents follow each mode, statics held (= S-world statics)
        co = xyF[tok].permute(0, 3, 1, 2, 4)                                           # [b,64,K,M,2]
        sF, fwF = sat_sep(cec[:, :, :, None, :], hec[:, :, :, None], co, h0[tok][:, None, :, None],
                          L[tok][:, None, :, None], W[tok][:, None, :, None])          # [b,64,K,M]
        sF = torch.clamp(sF, *SEP_CLIP)
        pr = presF[tok].permute(0, 3, 1, 2)                                            # [b,64,K,M]
        sF_p = torch.where(pr, sF, torch.full_like(sF, big))
        sFF_p = torch.where(pr & (fwF > 0), sF, torch.full_like(sF, big))
        pic = pi[tok][:, None]                                                         # [b,1,K,M]
        e_ag = (pic * sF_p).sum(-1)                                                    # [b,64,K]
        e_agF = (pic * sFF_p).sum(-1)
        w_ag = torch.where(plaus[tok][:, None], sF_p, torch.full_like(sF_p, big)).amin(-1)
        F_ag = torch.where(ag, e_ag, torch.full_like(e_ag, big)).amin(-1).reshape(-1, 16, 4)
        F_agF = torch.where(ag, e_agF, torch.full_like(e_agF, big)).amin(-1).reshape(-1, 16, 4)
        fF = _reduce9(F_ag, F_agF, s_st)
        W_ag = torch.where(ag, w_ag, torch.full_like(w_ag, big)).amin(-1).reshape(-1, 16, 4)
        fW = torch.stack([W_ag[:, :, 0].amin(1), W_ag[:, :, 1:].amin((1, 2))], -1)
        if raw:
            outs.append(torch.cat([fS, fF, fW], -1))
        else:
            outs.append(torch.cat([_norm9(fS), _norm9(fF), fW / 5.0], -1))
    return torch.cat(outs, 0)
