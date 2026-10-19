"""refiner v5 GT losses (report 97 §2-7 / §2-9; add.md §4): the corrector's executed-state surrogate L_refine(X1), the
per-state GT block targets of the relation hazard H, the reliability targets of a_i, and the corner-weighted margin aux.
GT is a TARGET only (never a network input): the batch's ref_* (GTLoader: refiner.gt_future objects 360 deg / 5 s with
the metric-cache presence rule, the official DAC SDF) and the object pack's matched GT futures (fc_xyh / fc_state).

L_refine(X1), X1 = LQR(tau1) (ck/v4/lqr.simulate, 41 states, differentiable): a SOFT PDMS of the executed rollout,
  S_hat = p_nc p_dac (5 p_ttc + 2 p_c + 5 ep) / 12 (the official PDMS structure: NC and DAC multiply, TTC / comfort /
  EP are the weighted average), with soft passes of the WORST state (not means over the states -- review 2026-10-20:
  state-averaged hazards against a per-draft progress fraction ranked braking that fixes a collision as worse, PT-6
  agreement 0.55-0.65):
    p_nc  = sigmoid(min sep / s)   min over the moving states and the counted GT objects at t_n (SAT separation of the
          official footprint; counted: observed (metric-cache rule), not behind the CURRENT ego box (centre >= -1.3 m
          along its heading) unless static / stopped)
    p_ttc = sigmoid(min sep / s)   the same with the footprint projected |v| delta (0.3 / 0.6 / 0.9 s) along the heading
          against the GT boxes at t_n + delta
    p_dac = sigmoid(min SDF / s)   min over the states x 4 corners of the official DAC SDF (corners off the SDF grid skipped)
    p_c   = sigmoid(-v_max / 0.5)  v_max = the largest official-comfort violation of X1 (pdm_comfort_metrics with its
          Savitzky-Golay filters as fixed linear operators: lon accel w41 p2 in [-4.05, 2.40], lon jerk w41 d1 of a w8-
          smoothed a < 4.13, yaw rate w5 d1 < 0.95, yaw accel w5 p3 d2 < 1.93; the official lateral check reads
          ACCELERATION_Y = 0 and never fails)
    ep    = min(1, S_exec / P)     executed 4 s path length over the token's pdm_progress_eff (>= 5 m)
  loss = mean over drafts of log(S_hat(X0) + 1e-3) - log(S_hat(X1) + 1e-3) (relative to the baseline tau0 the caller
  passes = the corrector's mu = 0 draft; log keeps the gradient of a deep overlap alive), s = 0.1 m.
  A weight 0 in w ('obj', 'ttc', 'road', 'prog', 'cmf') turns that factor off (p = 1; the MAP teacher has no object
  input: obj = ttc = 0); 'u' is the caller's mu^2 weight.
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
REF_W = {"road": 1.0, "obj": 1.0, "ttc": 1.0, "prog": 1.0, "cmf": 1.0, "u": 1e-3}   # factor switches + mu^2
SCALE = 0.1                                   # [m] soft pass sigmoid(margin / SCALE)
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


_CMF_OPS: Dict[str, Tensor] = {}


def _cmf_ops(dev) -> Dict[str, Tensor]:
    """official comfort filters (pdm_comfort_metrics) as fixed 41 x 41 float64 operators (savgol mode 'interp' is linear in
    the signal; reproduces the official per-metric verdicts on LQR rollouts, review 2026-10-20 f7)."""
    key = str(dev)
    if key not in _CMF_OPS:
        import numpy as np
        from scipy.signal import savgol_filter
        E = np.eye(LQR.NT)
        ops = {"a": savgol_filter(E, 41, 2, axis=0),
               "j": savgol_filter(E, 41, 2, deriv=1, delta=LQR.DT, axis=0) @ savgol_filter(E, 8, 2, axis=0),
               "r": savgol_filter(E, 5, 2, deriv=1, delta=LQR.DT, axis=0),
               "y": savgol_filter(E, 5, 3, deriv=2, delta=LQR.DT, axis=0)}
        _CMF_OPS[key] = {k: torch.from_numpy(v).to(dev) for k, v in ops.items()}
    return _CMF_OPS[key]


def _comfort(X: Tensor) -> Tensor:
    """X [..., 41, 6] -> the largest official-comfort violation [...] (> 0 = fails; module docstring)."""
    M = _cmf_ops(X.device)
    Xd = X.double()
    a = Xd[..., 4]
    h = Xd[..., 2]
    hu = torch.cat([h[..., :1], h[..., :1] + torch.cumsum(LQR.principal_value(h[..., 1:] - h[..., :-1]), -1)], -1)
    la, lj, yr, ya = a @ M["a"].T, a @ M["j"].T, hu @ M["r"].T, hu @ M["y"].T
    v = torch.stack([(la - CMF["lon_max"]).amax(-1), (CMF["lon_min"] - la).amax(-1),
                     (lj.abs() - CMF["jerk_lon"]).amax(-1), (yr.abs() - CMF["yaw_rate"]).amax(-1),
                     (ya.abs() - CMF["yaw_acc"]).amax(-1)], -1)
    return v.amax(-1).float()


def _terms(X: Tensor, gt: Dict[str, Tensor], t: Tensor, w: Dict[str, float]) -> Dict[str, Tensor]:
    """soft PDMS factors of rollouts X [n, R, 41, 6] -> dict of [n, R] (module docstring) + hard stats."""
    n, R, Q = X.shape[:3]
    out: Dict[str, Tensor] = {}
    sg = lambda x: torch.sigmoid(x / SCALE)                                             # noqa: E731
    one = X.new_ones(n, R)
    cor = ego_corners(X[..., :3])                                                       # [n, R, Q, 4, 2]
    val, ok = sample_sdf(gt["sdf"], cor.reshape(n, -1, 2))
    val, ok = val.reshape(n, R, Q * 4), ok.reshape(n, R, Q * 4)
    dmin = torch.where(ok, val, torch.full_like(val, 10.0)).amin(-1)
    out["offroad"] = (dmin < 0).float()
    out["p_dac"] = sg(dmin) if w.get("road", 0.0) else one
    if w.get("obj", 0.0) or w.get("ttc", 0.0):
        sep, cnt = object_sep(X, t, gt, (0.0,) + TTC_D)
        big = torch.full_like(sep, 10.0)
        s_nc = torch.where(cnt[..., 0, :], sep[..., 0, :], big[..., 0, :]).flatten(2).amin(-1)
        s_tt = torch.where(cnt[..., 1:, :], sep[..., 1:, :], big[..., 1:, :]).flatten(2).amin(-1)
        out["coll"] = (s_nc < 0).float()
        out["ttc_hit"] = (s_tt < 0).float()
        out["p_nc"] = sg(s_nc) if w.get("obj", 0.0) else one
        out["p_ttc"] = sg(s_tt) if w.get("ttc", 0.0) else one
    else:
        out["p_nc"], out["p_ttc"] = one, one
    S = LQR.executed_progress(X)[0]                                                     # [n, R]
    out["S"] = S
    P = gt["p_pdm"].float().clamp(min=5.0)[:, None]
    out["ep"] = torch.clamp(S / P, max=1.0) if w.get("prog", 0.0) else one
    cv = _comfort(X)
    out["cmf_fail"] = (cv > 0).float()
    out["p_c"] = torch.sigmoid(-cv / 0.5) if w.get("cmf", 0.0) else one
    out["S_hat"] = out["p_nc"] * out["p_dac"] * (5 * out["p_ttc"] + 2 * out["p_c"] + 5 * out["ep"]) / 12.0
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
            T0 = _terms(X0, gt, t, w)
        X1 = LQR.simulate(tau1.float(), ego, dtype=torch.float32)
        T1 = _terms(X1, gt, t, w)
        L = torch.log(T0["S_hat"] + 1e-3) - torch.log(T1["S_hat"] + 1e-3)
        loss = torch.nan_to_num(L, nan=0.0, posinf=10.0, neginf=-10.0).mean()
        st = {f"ref_{k}{i}": float(T[k].detach().mean()) for i, T in (("0", T0), ("1", T1))
              for k in ("S_hat", "p_nc", "p_dac", "p_ttc", "ep", "p_c", "offroad", "cmf_fail")}
        H0, H1 = T0, T1
        if "coll" not in T0:                    # object factors off (MAP): the hazard stats still reported
            with torch.no_grad():
                hw = dict(w, obj=1.0, ttc=1.0)
                H0, H1 = _terms(X0, gt, t, hw), _terms(X1.detach(), gt, t, hw)
        for i, H in (("0", H0), ("1", H1)):
            st[f"ref_coll{i}"] = float(H["coll"].mean())
            st[f"ref_ttc_hit{i}"] = float(H["ttc_hit"].mean())
        st["ref_dS"] = float((T1["S"] - T0["S"]).detach().mean())
        st["ref_n"] = float(n * R)
    return loss, st


@torch.no_grad()
def gt_state_block(states: Tensor, t: Tensor, gt: Dict[str, Tensor], fov_deg: Optional[float] = None
                   ) -> Tuple[Tensor, Tensor]:
    """per-state GT block targets of the relation hazard: states [n, K, Q, 6] at t [Q] -> y [n, K, Q, 2] (NC: overlap
    at delta 0; TTC: overlap at any delta of (0, 0.3, 0.6, 0.9) s; counted objects only) float, known [n, K, Q] bool
    (all times <= the token's annotated horizon).  Only GT objects the t0 detections could explain count: observed at
    t0 with the centre inside the detector window (x in [0, 32], |y| <= 32; + |bearing| <= fov_deg for the student's
    camera head) -- the relation reads t0 detections only, an object absent at t0 or outside the window is an
    irreducible positive that would inflate eps / a_i (review 2026-10-20 f6)."""
    sep, cnt = object_sep(states.float(), t, gt, (0.0,) + TTC_D)
    c0, _, obs0 = gt_t0_objects(gt)
    win = obs0 & (c0[..., 0] >= 0) & (c0[..., 0] <= 32.0) & (c0[..., 1].abs() <= 32.0)
    if fov_deg is not None:
        win = win & (torch.rad2deg(torch.atan2(c0[..., 1], c0[..., 0])).abs() <= float(fov_deg) + 1e-4)
    hit = (sep < 0) & cnt & win[:, None, None, None, :]                                   # [n, K, Q, D, A]
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
