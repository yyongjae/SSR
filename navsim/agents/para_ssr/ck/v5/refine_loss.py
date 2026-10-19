"""refiner v5 GT losses (report 97 §2-7 / §2-9; add.md §4): the corrector's executed-state surrogate L_refine(X1), the
per-state GT block targets of the relation hazard H, the reliability targets of a_i, and the corner-weighted margin aux.
GT is a TARGET only (never a network input): the batch's ref_* (GTLoader: refiner.gt_future objects 360 deg / 5 s with
the metric-cache presence rule, the official DAC SDF) and the object pack's matched GT futures (fc_xyh / fc_state).

L_refine(X1), X1 = LQR(tau1) (ck/v4/lqr.simulate, 41 states, differentiable), per draft (mean over drafts):
  road  mean over states x corners of softplus((m_road - SDF(corner)) / s) s       (SDF at the official footprint corners;
        corners outside the SDF grid skipped)
  obj   mean over states of max over GT objects of softplus((m_obj - sep) / s) s     (sep = SAT separation of the official
        footprint at state n and the GT box at t_n; counted when the object is observed (metric-cache rule), the ego
        moves (|v| > 0.05 m/s) and the object is not behind (centre >= -1.3 m along the heading from the CURRENT ego
        box at state n, also for the TTC projections) unless static / stopped: the at-fault proxy of the range field B)
  ttc   the same with the footprint projected |v| delta (0.3 / 0.6 / 0.9 s) along the heading against the GT boxes at
        t_n + delta, margin 0, max over delta
  prog  relu(min(1, S0 / P) - min(1, S1 / P)), S = executed 4 s path length (lqr.executed_progress), P = max(pdm_progress,
        5 m): progress LOST is penalised, progress gained is not credited (add.md: no decelerate-to-win)
  cmf   official comfort thresholds (pdm_comfort_metrics) on X1's finite differences: relu(a - 2.40) + relu(-4.05 - a) +
        relu(|v yaw_rate| - 4.89) + relu(|jerk_lon| - 4.13) + relu(|yaw_rate| - 0.95) + relu(|yaw_acc| - 1.93), means
  loss = sum_k w_k term_k (weights REF_W; caller adds w_u mean(mu^2))
Stats (no grad) compare X1 with X0 = LQR(tau0) -- the callers pass tau0 = the corrector's baseline (mu = 0: today's
laterally corrected draft, CKNet.refine_drafts 'tau_base'), so *0 vs *1 is the corrector's own change; L_prog uses the
same baseline (progress lost relative to it).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ...refiner import gt_future as GF
from ...refiner.sdf import ego_corners, sample_sdf
from ..temporal.config import EGO_OFF, FC_OBS, FC_SINGLE
from ..temporal.raster import sat_sep
from ..v4 import lqr as LQR

Tensor = torch.Tensor
REF_W = {"road": 1.0, "obj": 1.0, "ttc": 0.5, "prog": 1.0, "cmf": 0.1, "u": 1e-3}
M_ROAD, M_OBJ, SCALE = 0.1, 0.2, 0.1          # [m]
BEHIND_M = -1.3
STOP_NC, STOP_TTC = 0.05, 5e-3
TTC_D = (0.3, 0.6, 0.9)
A_MAX = 64                                    # GT tracks kept per token (stored nearest-to-the-GT-path first)
CMF = {"lon_max": 2.40, "lon_min": -4.05, "lat": 4.89, "jerk_lon": 4.13, "yaw_rate": 0.95, "yaw_acc": 1.93}
CMF_LAT_JERK_NOTE = "max_abs_mag_jerk 8.37 not used: lon jerk 4.13 is the binding one on a smooth LQR rollout"


def parse_weights(s: Optional[str]) -> Dict[str, float]:
    """'road=1,obj=0,...' -> REF_W updated (unknown key -> ValueError)."""
    w = dict(REF_W)
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        k = k.strip()
        if k not in w:
            raise ValueError(f"refine weight {k!r} not in {sorted(w)}")
        w[k] = float(v)
    return w


def gt_of(batch: Dict, idx: Tensor) -> Dict[str, Tensor]:
    """GT of the token rows idx from a train batch (ref_*), objects trimmed to A_MAX."""
    g = lambda k: batch[f"ref_{k}"][idx]                                               # noqa: E731
    n_obj = g("obj_n").long()
    A = int(max(1, min(A_MAX, int(n_obj.max()) if idx.numel() else 1)))
    return {"kf": g("obj_kf")[:, :A].float(), "first": g("obj_first")[:, :A].float(),
            "meta": g("obj_meta")[:, :A].long(),
            "valid": torch.arange(A, device=idx.device)[None] < n_obj[:, None],
            "n_kf": g("obj_n_kf").long(), "sdf": g("sdf"), "p_pdm": g("p_pdm").float()}


def _gt_boxes(gt: Dict[str, Tensor], t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """boxes [n, A, len(t), 5] (cx, cy, h, L, W; GT box centre), obs [n, A, len(t)] bool, exempt [n, A] bool (static /
    stopped: no not-behind filter)."""
    boxes, st = GF.query_torch(gt["kf"], gt["first"], gt["meta"], t, valid=gt["valid"], n_kf=gt["n_kf"])
    agent = gt["meta"][..., GF.MT_AGENT].bool()
    v_first = torch.linalg.vector_norm(gt["first"][..., GF.FI_VX:GF.FI_VY + 1], dim=-1)
    return boxes.float(), st.bool(), (~agent) | (v_first < 0.5)


def object_sep(states: Tensor, t: Tensor, gt: Dict[str, Tensor], deltas: Sequence[float]) -> Tuple[Tensor, Tensor]:
    """states [n, R, Q, >=4] (N frame, rear axle; Q states at times t [Q] s), deltas (s) -> sep [n, R, Q, D, A] (SAT
    separation of the official footprint projected |v| delta along the heading vs the GT boxes at t + delta;
    differentiable in states), count [n, R, Q, D, A] bool (observed, not-behind or exempt, ego moving: NC |v| > 0.05
    for delta 0, TTC |v| >= 5e-3 otherwise)."""
    n, R, Q = states.shape[:3]
    dev = states.device
    dl = torch.as_tensor(list(deltas), dtype=torch.float64, device=dev)
    D = int(dl.numel())
    tt = (t.to(dev).double()[:, None] + dl[None]).reshape(-1)                          # [Q * D]
    boxes, obs, exempt = _gt_boxes(gt, tt)                                              # [n, A, Q*D, 5]
    A = boxes.shape[1]
    boxes = boxes.reshape(n, A, Q, D, 5).permute(0, 2, 3, 1, 4)[:, None]               # [n, 1, Q, D, A, 5]
    obs = obs.reshape(n, A, Q, D).permute(0, 2, 3, 1)[:, None]                         # [n, 1, Q, D, A]
    hd, spd = states[..., 2], states[..., 3].abs()
    eh = torch.stack([torch.cos(hd), torch.sin(hd)], -1)
    ce = states[..., :2] + EGO_OFF * eh                                                 # [n, R, Q, 2]
    shift = spd[..., None] * dl.to(states.dtype)                                        # [n, R, Q, D]
    ce_d = ce[..., None, :] + shift[..., None] * eh[..., None, :]                       # [n, R, Q, D, 2]
    sep, _ = sat_sep(ce_d[..., None, :], hd[..., None, None], boxes[..., :2], boxes[..., 2], boxes[..., 3],
                     boxes[..., 4])                                                      # [n, R, Q, D, A]
    # not-behind from the CURRENT ego box at state n (official TTC judges ahead / behind from the state's rear axle;
    # the projected box is only used for the overlap) -- review 98 R3
    fwd = ((boxes[..., :2] - ce[:, :, :, None, None, :]) * eh[:, :, :, None, None, :]).sum(-1)
    mv = torch.where(dl[None, None, None] == 0, (spd > STOP_NC)[..., None], (spd >= STOP_TTC)[..., None])
    count = obs & mv[..., None] & ((fwd.detach() >= BEHIND_M) | exempt[:, None, None, None, :])
    return sep, count


def _comfort(X: Tensor) -> Tensor:
    """X [..., 41, 6] -> mean comfort penalty [...] (module docstring)."""
    a, v, h = X[..., 4], X[..., 3], X[..., 2]
    yr = LQR.principal_value(h[..., 1:] - h[..., :-1]) / LQR.DT
    ya = (yr[..., 1:] - yr[..., :-1]) / LQR.DT
    lat = v[..., :-1] * yr
    jk = (a[..., 1:] - a[..., :-1]) / LQR.DT
    return (F.relu(a - CMF["lon_max"]).mean(-1) + F.relu(CMF["lon_min"] - a).mean(-1)
            + F.relu(lat.abs() - CMF["lat"]).mean(-1) + F.relu(jk.abs() - CMF["jerk_lon"]).mean(-1)
            + F.relu(yr.abs() - CMF["yaw_rate"]).mean(-1) + F.relu(ya.abs() - CMF["yaw_acc"]).mean(-1))


def _terms(X: Tensor, gt: Dict[str, Tensor], t: Tensor, w: Dict[str, float], S0: Optional[Tensor]) -> Dict[str, Tensor]:
    n, R, Q = X.shape[:3]
    out: Dict[str, Tensor] = {}
    sp = lambda x, m: F.softplus((m - x) / SCALE) * SCALE                               # noqa: E731
    cor = ego_corners(X[..., :3])                                                       # [n, R, Q, 4, 2]
    val, ok = sample_sdf(gt["sdf"], cor.reshape(n, -1, 2))
    val, ok = val.reshape(n, R, Q * 4), ok.reshape(n, R, Q * 4)
    okf = ok.float()
    out["road"] = (sp(val, M_ROAD) * okf).sum(-1) / okf.sum(-1).clamp(min=1.0)
    out["offroad"] = ((val < 0) & ok).any(-1).float()
    if w.get("obj", 0.0) or w.get("ttc", 0.0) or S0 is None:
        sep, cnt = object_sep(X, t, gt, (0.0,) + TTC_D)
        pen = torch.where(cnt[..., 0, :], sp(sep[..., 0, :], M_OBJ), torch.zeros_like(sep[..., 0, :]))
        out["obj"] = pen.amax(-1).mean(-1)
        pt = torch.where(cnt[..., 1:, :], sp(sep[..., 1:, :], 0.0), torch.zeros_like(sep[..., 1:, :]))
        out["ttc"] = pt.amax(-1).amax(-1).mean(-1)
        out["coll"] = ((sep[..., 0, :] < 0) & cnt[..., 0, :]).flatten(2).any(-1).float()
        out["ttc_hit"] = ((sep[..., 1:, :] < 0) & cnt[..., 1:, :]).flatten(2).any(-1).float()
    S = LQR.executed_progress(X)[0]                                                     # [n, R]
    out["S"] = S
    if S0 is not None:
        P = gt["p_pdm"].float().clamp(min=5.0)[:, None]
        out["prog"] = F.relu(torch.clamp(S0 / P, max=1.0) - torch.clamp(S / P, max=1.0))
    out["cmf"] = _comfort(X)
    return out


def refine_loss(tau1: Tensor, tau0: Tensor, v0: Tensor, a0: Tensor, gt: Dict[str, Tensor], w: Dict[str, float]
                ) -> Tuple[Tensor, Dict[str, float]]:
    """tau1 (differentiable) / tau0 [n, R, 8, 3], v0 / a0 [n], gt = gt_of(batch, idx) -> (loss scalar f32, stats)."""
    n, R = tau1.shape[:2]
    dev = tau1.device
    with torch.autocast(device_type=dev.type, enabled=False):
        ego = torch.stack([v0.float(), a0.float()], -1)[:, None, :]                      # [n, 1, 2]
        t = torch.arange(LQR.NT, dtype=torch.float64, device=dev) * LQR.DT
        with torch.no_grad():
            X0 = LQR.simulate(tau0.float(), ego, dtype=torch.float32)
            T0 = _terms(X0, gt, t, dict(w, obj=1.0, ttc=1.0), None)
        X1 = LQR.simulate(tau1.float(), ego, dtype=torch.float32)
        T1 = _terms(X1, gt, t, w, T0["S"])
        loss = tau1.new_zeros((), dtype=torch.float32)
        for k in ("road", "obj", "ttc", "prog", "cmf"):
            if w.get(k, 0.0) and k in T1:
                loss = loss + float(w[k]) * torch.nan_to_num(T1[k], nan=0.0, posinf=1e3).mean()
        st = {f"ref_{k}": float(T1[k].detach().mean()) for k in ("road", "obj", "ttc", "prog", "cmf") if k in T1}
        for k in ("offroad", "coll", "ttc_hit"):
            if k in T1:
                st[f"ref_{k}1"] = float(T1[k].detach().mean())
            st[f"ref_{k}0"] = float(T0[k].mean())
        st["ref_dS"] = float((T1["S"] - T0["S"]).detach().mean())
        st["ref_n"] = float(n * R)
    return loss, st


@torch.no_grad()
def gt_state_block(states: Tensor, t: Tensor, gt: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    """per-state GT block targets of the relation hazard: states [n, K, Q, 6] at t [Q] -> y [n, K, Q, 2] (NC: overlap
    at delta 0; TTC: overlap at any delta of (0, 0.3, 0.6, 0.9) s; counted objects only) float, known [n, K, Q] bool
    (all times <= the token's annotated horizon)."""
    sep, cnt = object_sep(states.float(), t, gt, (0.0,) + TTC_D)
    hit = (sep < 0) & cnt                                                               # [n, K, Q, D, A]
    y = torch.stack([hit[..., 0, :].any(-1), hit.any(-1).any(-1)], -1).float()
    t_av = ((gt["n_kf"].double() - 1) * GF.KF_DT).clamp(max=GF.T_MAX)                  # [n]
    known = (t.to(states.device).double()[None] + max(TTC_D) <= t_av[:, None] + 1e-6)  # [n, Q]
    return y, known[:, None, :].expand(states.shape[0], states.shape[1], -1)


@torch.no_grad()
def gt_t0_objects(gt: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    """GT objects at t0 (metric-cache rule): centres [n, A, 2], annotated velocity [n, A, 2] (keyframe 0; 0 for
    non-agents), observed [n, A] bool -> reliability targets compare t0 with t0 (review 98 R4)."""
    boxes, obs = GF.query_torch(gt["kf"], gt["first"], gt["meta"], [0.0], valid=gt["valid"], n_kf=gt["n_kf"])
    return boxes[:, :, 0, :2].float(), gt["kf"][:, :, 0, GF.KF_VX:GF.KF_VY + 1].float(), obs[:, :, 0].bool()


def corner_margin_aux(Dc: Tensor, ok: Tensor, states: Tensor, sdf: Tensor, w_edge: float = 4.0) -> Tuple[Tensor, float]:
    """MAP: predicted margin at the executed corners Dc [n, K, Q*4] (readout read, differentiable; ok [n, K, Q*4] =
    readable), states [n, K, Q, 6], sdf [n, 320, 256] -> (Huber(delta 1 m) of Dc vs the GT SDF clipped +-5 m, weight
    w_edge where |SDF| <= 1 m (exits are shallow and late, report 95 §3-1), else 1; masked by both validities, mean,
    weight-normalised), n_used."""
    n, K = Dc.shape[:2]
    cor = ego_corners(states[..., :3].float())                                          # [n, K, Q, 4, 2]
    val, okg = sample_sdf(sdf, cor.reshape(n, -1, 2))
    val, okg = val.reshape(n, K, -1).clamp(-5.0, 5.0), okg.reshape(n, K, -1)
    m = (ok & okg).float()
    wt = torch.where(val.abs() <= 1.0, torch.full_like(val, float(w_edge)), torch.ones_like(val)) * m
    h = F.huber_loss(Dc.float(), val.float(), delta=1.0, reduction="none")
    return (h * wt).sum() / wt.sum().clamp(min=1.0), float(m.sum())
