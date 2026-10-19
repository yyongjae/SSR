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
  computed in LOG space (report 100 F1: log(S + 1e-3) scaled every gradient by S / (S + 1e-3), ~1e-13 on a deep
  overlap): log S = log p_nc + log p_dac + log(5 p_ttc + 2 p_c + 5 ep) - log 12 with log p = logsigmoid (linear tail:
  a constant 1 / s per metre however deep), and log p_nc additionally carries the IN-PATH SHAPING term
  sum over objects a of c_a logsigmoid(g_a / s): c_a = sigmoid(-min_n sep_na / s) (soft "collides with a"),
  g_a = min over the states where a overlaps the ego footprint laterally of the SIGNED longitudinal gap (a's centre
  ahead along the ego heading minus both half-extents; no abs, no not-behind cut) -- the worst-state SAT term alone is
  flat in the longitudinal control on a head-on overlap (its least-overlapping axis is lateral), and a penetration sum
  over the states rewards driving THROUGH faster; g_a only grows when the ego progresses less, so braking is always
  the local direction until a is cleared, after which c_a -> 0 (non-colliding rollouts: ~0, worst-state ranking).
  loss = mean over drafts of (log S(X0) - log S(X1)) / max(1, -log S(X0)) (relative to the baseline tau0 the caller
  passes = the corrector's mu = 0 draft; the per-draft normaliser keeps a deep-overlap draft from dominating the batch
  and does not change the sign), s = 0.1 m.
  GT objects (report 100 F2): gt_of keeps EVERY track of the token (the old cap of 64 nearest-to-the-GT-path tracks
  turned correct detections into reliability negatives); object_sep's callers restrict the tracks to the ones whose
  4.9 s footprint can reach the rollouts (relevant_objects: spatial, time-agnostic, + the TTC projection and a 4 m
  corrector slack; nothing within reach is dropped), gt_state_block runs in object chunks.
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
REACH_SLACK_M = 4.0                           # relevant_objects: corrector deviation slack beyond the drafts' sweep
CHUNK_A = 64                                  # gt_state_block object chunk (no grad; exact)
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
    """GT of the token rows idx from a train batch (ref_*), EVERY track (A = the rows' largest count; report 100 F2)."""
    g = lambda k: batch[f"ref_{k}"][idx]                                               # noqa: E731
    n_obj = g("obj_n").long()
    A = int(max(1, min(int(batch["ref_obj_kf"].shape[1]), int(n_obj.max()) if idx.numel() else 1)))
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


@torch.no_grad()
def relevant_objects(gt: Dict[str, Tensor], states: Tensor, slack: float = REACH_SLACK_M) -> Tuple[Dict[str, Tensor], Dict]:
    """GT restricted to the tracks that can reach the rollouts states [n, R, Q, >=4]: a track is kept when some point of
    its 0..4.9 s centre path (0.25 s) lies within ego half-diagonal + its own half-diagonal + 0.9 s x the token's max
    speed (TTC projection) + slack of some executed rear-axle / box-centre position of any draft (time-agnostic: a
    slower tau1 sweeps a subset of the path, the slack covers the corrector's lateral / extra-progress change).  Kept
    tracks first (stable order), A' = the rows' largest kept count -> (gt', stats).  Nothing within reach is dropped."""
    from ..temporal.config import EGO_L, EGO_W
    n, A = gt["valid"].shape
    dev = states.device
    tq = torch.arange(0.0, GF.T_MAX + 1e-6, 0.25, dtype=torch.float64, device=dev)
    boxes, obs = GF.query_torch(gt["kf"], gt["first"], gt["meta"], tq, valid=gt["valid"], n_kf=gt["n_kf"])
    oc = boxes[..., :2].float()                                                         # [n, A, T, 2]
    ohd = 0.5 * torch.sqrt(boxes[..., 3].float() ** 2 + boxes[..., 4].float() ** 2)     # [n, A, T]
    ovm = ((oc[:, :, 1:] - oc[:, :, :-1]).norm(dim=-1) / 0.25).amax(-1, keepdim=True)  # [n, A, 1] max track speed
    st = states.float()[:, :, ::2]                                                      # every 2nd state (gap term below)
    eh = torch.stack([torch.cos(st[..., 2]), torch.sin(st[..., 2])], -1)
    pts = (st[..., :2] + EGO_OFF * eh).reshape(n, -1, 2)                                # box centres [n, R*Q', 2]
    vmax = states[..., 3].float().abs().reshape(n, -1).amax(-1)                         # [n]
    dt_gap = 2 * float(LQR.DT) * (1 if states.shape[2] > 21 else 2)                    # spacing of the kept states [s]
    # ego half-diagonal + TTC projection (0.9 s) + half the kept-state spacing + slack; object half-diagonal + half the
    # 0.25 s sampling of its path
    reach = 0.5 * math.hypot(EGO_L, EGO_W) + (0.9 + 0.5 * dt_gap) * vmax + float(slack)   # [n]
    d = torch.cdist(oc.reshape(n, -1, 2), pts).amin(-1).reshape(n, A, -1)               # [n, A, T]
    near = ((d <= ohd + 0.125 * ovm + reach[:, None, None]) & obs.bool()).any(-1) & gt["valid"]   # [n, A]
    order = torch.argsort((~near).to(torch.int8), dim=1, stable=True)
    k = int(max(1, int(near.sum(1).max()) if n else 1))
    idx = order[:, :k]
    take = lambda x: torch.gather(x, 1, idx.reshape(n, k, *([1] * (x.dim() - 2))).expand(n, k, *x.shape[2:]))  # noqa
    out = dict(gt)
    for key in ("kf", "first", "meta"):
        out[key] = take(gt[key])
    out["valid"] = take(near)
    return out, {"n_tracks": float(gt["valid"].sum(1).float().mean()), "n_reach": float(near.sum(1).float().mean()),
                 "n_reach_max": float(k)}


def lon_lat_gaps(ce: Tensor, he: Tensor, co: Tensor, ho: Tensor, Lo: Tensor, Wo: Tensor) -> Tuple[Tensor, Tensor]:
    """signed longitudinal gap (object centre ahead along the ego heading minus the ego half-length and the object's
    extent on that axis) and the lateral SAT gap (|lateral offset| minus both half-extents on the ego lateral axis);
    arguments as temporal.raster.sat_sep (broadcast)."""
    from ..temporal.config import EGO_L, EGO_W
    cue, sue = torch.cos(he), torch.sin(he)
    cuo, suo = torch.cos(ho), torch.sin(ho)
    dx, dy = co[..., 0] - ce[..., 0], co[..., 1] - ce[..., 1]
    lo, wo = Lo / 2, Wo / 2
    d_ue = dx * cue + dy * sue
    d_ve = -dx * sue + dy * cue
    ext_ue = lo * (cuo * cue + suo * sue).abs() + wo * (-suo * cue + cuo * sue).abs()
    ext_ve = lo * (-cuo * sue + suo * cue).abs() + wo * (suo * sue + cuo * cue).abs()
    return d_ue - EGO_L / 2 - ext_ue, d_ve.abs() - EGO_W / 2 - ext_ve


def object_sep(states: Tensor, t: Tensor, gt: Dict[str, Tensor], deltas: Sequence[float], extras: bool = False):
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
    if not extras:
        return sep, count
    b0 = boxes[:, :, :, 0]                                                              # delta 0: [n, 1, Q, A, 5]
    lon, lat = lon_lat_gaps(ce[..., None, :], hd[..., None], b0[..., :2], b0[..., 2], b0[..., 3], b0[..., 4])
    inpath = (lat.detach() < 0) & obs[:, :, :, 0] & mv[..., 0][..., None]               # [n, R, Q, A]
    return sep, count, lon, inpath


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
    """soft PDMS factors of rollouts X [n, R, 41, 6] -> dict of [n, R] (module docstring): log factors lp_nc / lp_dac /
    lp_ttc, logS, S_hat = exp(logS), p_* = exp(lp_*), + hard stats."""
    n, R, Q = X.shape[:3]
    out: Dict[str, Tensor] = {}
    lsg = lambda x: F.logsigmoid(x / SCALE)                                             # noqa: E731
    zero, one = X.new_zeros(n, R), X.new_ones(n, R)
    cor = ego_corners(X[..., :3])                                                       # [n, R, Q, 4, 2]
    val, ok = sample_sdf(gt["sdf"], cor.reshape(n, -1, 2))
    val, ok = val.reshape(n, R, Q * 4), ok.reshape(n, R, Q * 4)
    dmin = torch.where(ok, val, torch.full_like(val, 10.0)).amin(-1)
    out["offroad"] = (dmin < 0).float()
    out["lp_dac"] = lsg(dmin) if w.get("road", 0.0) else zero
    if w.get("obj", 0.0) or w.get("ttc", 0.0):
        sep, cnt, lon, inpath = object_sep(X, t, gt, (0.0,) + TTC_D, extras=True)
        big = torch.full_like(sep, 10.0)
        s_nc = torch.where(cnt[..., 0, :], sep[..., 0, :], big[..., 0, :]).flatten(2).amin(-1)
        s_tt = torch.where(cnt[..., 1:, :], sep[..., 1:, :], big[..., 1:, :]).flatten(2).amin(-1)
        out["coll"] = (s_nc < 0).float()
        out["ttc_hit"] = (s_tt < 0).float()
        ms_a = torch.where(cnt[..., 0, :], sep[..., 0, :], big[..., 0, :]).amin(2)       # [n, R, A] per object
        c_a = torch.sigmoid(-ms_a / SCALE)
        g_a = torch.where(inpath, lon, torch.full_like(lon, 10.0)).amin(2)              # [n, R, A]
        shp = (c_a * F.logsigmoid(g_a / SCALE)).sum(-1)
        out["shape_nc"] = shp
        out["lp_nc"] = lsg(s_nc) + shp if w.get("obj", 0.0) else zero
        out["lp_ttc"] = lsg(s_tt) if w.get("ttc", 0.0) else zero
    else:
        out["lp_nc"], out["lp_ttc"] = zero, zero
    S = LQR.executed_progress(X)[0]                                                     # [n, R]
    out["S"] = S
    P = gt["p_pdm"].float().clamp(min=5.0)[:, None]
    out["ep"] = torch.clamp(S / P, max=1.0) if w.get("prog", 0.0) else one
    cv = _comfort(X)
    out["cmf_fail"] = (cv > 0).float()
    out["p_c"] = torch.sigmoid(-cv / 0.5) if w.get("cmf", 0.0) else one
    for k in ("nc", "dac", "ttc"):
        out[f"p_{k}"] = torch.exp(out[f"lp_{k}"])
    out["logS"] = (out["lp_nc"] + out["lp_dac"]
                   + torch.log(5 * out["p_ttc"] + 2 * out["p_c"] + 5 * out["ep"] + 1e-6) - math.log(12.0))
    out["S_hat"] = torch.exp(out["logS"])
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
            sweep = torch.cat([X0, LQR.simulate(tau1.detach().float(), ego, dtype=torch.float32)], 1)
            gt, rs = relevant_objects(gt, sweep) if (w.get("obj", 0.0) or w.get("ttc", 0.0)) else (gt, {})
            T0 = _terms(X0, gt, t, w)
        X1 = LQR.simulate(tau1.float(), ego, dtype=torch.float32)
        T1 = _terms(X1, gt, t, w)
        L = (T0["logS"] - T1["logS"]) / (-T0["logS"]).clamp(min=1.0)
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
        st["ref_logS0"], st["ref_logS1"] = float(T0["logS"].mean()), float(T1["logS"].detach().mean())
        st.update({f"ref_gt_{k}": v for k, v in rs.items()})
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
    gt, _ = relevant_objects(gt, states.float(), slack=1.0)       # tracks out of reach never overlap these states
    c0, _, obs0 = gt_t0_objects(gt)
    win = obs0 & (c0[..., 0] >= 0) & (c0[..., 0] <= 32.0) & (c0[..., 1].abs() <= 32.0)
    if fov_deg is not None:
        win = win & (torch.rad2deg(torch.atan2(c0[..., 1], c0[..., 0])).abs() <= float(fov_deg) + 1e-4)
    n, K, Q = states.shape[:3]
    y_nc = torch.zeros(n, K, Q, dtype=torch.bool, device=states.device)
    y_tt = torch.zeros_like(y_nc)
    A = gt["valid"].shape[1]
    for a0 in range(0, A, CHUNK_A):                                  # object chunks: OR over the tracks is exact
        sl = slice(a0, min(A, a0 + CHUNK_A))
        g = dict(gt, kf=gt["kf"][:, sl], first=gt["first"][:, sl], meta=gt["meta"][:, sl], valid=gt["valid"][:, sl])
        sep, cnt = object_sep(states.float(), t, g, (0.0,) + TTC_D)
        hit = (sep < 0) & cnt & win[:, sl][:, None, None, None, :]                        # [n, K, Q, D, a]
        y_nc |= hit[..., 0, :].any(-1)
        y_tt |= hit.any(-1).any(-1)
    y = torch.stack([y_nc, y_tt], -1).float()
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
