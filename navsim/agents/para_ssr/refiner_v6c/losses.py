"""Losses of refiner v6c (report 124 §4-2, detail D2; 2026-10-27 revision).

L_E   R-E per-step labels (GT, full weight) + lambda_c * soft KD (object 6 <- T-E, off-road <- T-M), input gates only
L_corr R-C, read on S_v6_g (the ONLY consumer of the grad states; report 124 §4-2a):
      teacher term   mean_t [h_T(v6, t) - h_T(base, t)]+ + 1[tau_base official NC / TTC fail] mean_t h_T(v6, t)
      GT guard       mean_t [h_GT(v6, t) - h_GT(base, t)]+        (122's L_preserve; 0 at Delta = 0)
                     h = front-half footprint (cf GT hinge's 4 x 3 points) occupancy + TTC-shifted occupancy hinge;
                     teacher: probability field, margin 0.3; GT: occupancy distance field, margin eps (small)
      road           mean_t max_corner [0.1 - SDF_GT]+ (cap 3, in range)              (cf static form)
      comfort        max_t hinges of the official ego_is_comfortable features (lon / lat accel, jerk magnitude, lon
                     jerk, yaw rate / accel; yaw masked below 1 m/s) on exact_verdict's filters (report 129 A19)
      EC             [rms jerk(v6) - max(rms jerk(base), 0.25)]+ / 0.5 + [rms yaw acc(v6, masked) - max(.., 0.05)]+ / 0.1
      progress       relu((s_end(base) - s_end(v6)) / 10)   (F.relu: 0 gradient at the Delta = 0 kink)
L_J   R-J per row: CE(NC 3-class) + BCE(DAC fail) + BCE(TTC fail) + SmoothL1(prog / 10), + SmoothL1(eff / 10)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..modules import corridor_head as CH
from ..modules import exact_verdict as EV
from ..modules.candidate_scorer import _ec_ops
from .nets import CONT_NORM, cont_norm

LOGIT_LABELS = {"ovl": 0, "ttc": 1, "offroad": 6}


def _bce_pw(logit, y, pw):
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(pw, device=logit.device),
                                              reduction="none")


def risk_loss(E: torch.Tensor, lab: Dict[str, torch.Tensor], kd: Optional[Dict[str, torch.Tensor]] = None,
              lam: Optional[Dict[str, float]] = None, pos_weight: float = 4.0) -> Tuple[torch.Tensor, Dict]:
    """E [B, R, T, 12] raw R-E output; lab = labels.stride2(exact_step_masks(...)) on the same rows' exact states
    ([B, R, T]; pos [B, R, T, 2]); kd (optional): {'p' [B, R, T, 7] teacher targets in E's raw units for the
    continuous channels and PROBABILITIES for the 3 logit channels, 'g' [B, R, T, 7] gates (0 / 1)};
    lam: per-channel KD weights (names of nets.E_NAMES).  -> (loss, logs)
    Masks (report 129 N-stride2-odd-valid): ovl / ttc use lab['valid_bin'] (validity pooled over the same two steps as
    the labels; 'valid' when absent), offroad valid_off, the continuous channels 'valid' (their even step).
    The continuous prediction is normalised without a clip, the target with it (report 129 A39): an output beyond
    +-4 units is pulled back instead of getting no gradient; |o| <= 4 is unchanged bitwise.
    Eskill_{nm} (report 129 A66) = 1 - BCE / the BCE of the best constant logit on the same masked steps (same
    pos_weight); Eprior_{nm} is that constant's BCE.  A batch without positives has a prior ~ 0 and a large negative
    Eskill, so the pooled skill over many batches is 1 - mean(E_{nm}) / mean(Eprior_{nm})."""
    v = lab["valid"]
    nv = v.sum().clamp(min=1.0)
    vb = lab.get("valid_bin", v)
    logs, loss = {}, E.new_zeros(())
    for nm, ch in LOGIT_LABELS.items():
        vm = lab["valid_off"] if nm == "offroad" else vb                 # DAC does not depend on the object horizon
        nvm = vm.sum().clamp(min=1.0)
        l = (_bce_pw(E[..., ch], lab[nm], pos_weight) * vm).sum() / nvm
        loss = loss + l
        logs[f"v6c/E_{nm}"] = l.detach()
        with torch.no_grad():
            pr = (lab[nm] * vm).sum() / nvm
            logs[f"v6c/pos_{nm}"] = pr
            q = (pos_weight * pr / (pos_weight * pr + 1 - pr)).clamp(1e-6, 1 - 1e-6)     # argmin of the constant BCE
            prior = -(pos_weight * pr * torch.log(q) + (1 - pr) * torch.log(1 - q))
            logs[f"v6c/Eprior_{nm}"] = prior
            logs[f"v6c/Eskill_{nm}"] = 1 - l.detach() / prior.clamp(min=1e-6)
    # continuous 4 (gap, vapp, px, py): fixed values where there is no object, so every valid step is supervised
    y = torch.stack([lab["gap"], lab["vapp"], lab["pos"][..., 0], lab["pos"][..., 1]], -1)
    o = E[..., 2:6] / E.new_tensor(CONT_NORM)                                                      # not clipped (A39)
    lc = F.smooth_l1_loss(o, cont_norm(y), beta=0.5, reduction="none")                            # [B, R, T, 4]
    for j, nm in enumerate(("gap", "vapp", "px", "py")):
        l = (lc[..., j] * v).sum() / nv
        loss = loss + l
        logs[f"v6c/E_{nm}"] = l.detach()
    with torch.no_grad():
        logs["v6c/obj_frac"] = (lab["obj"] * v).sum() / nv
    if kd is not None:
        lam = lam or {}
        p, g = kd["p"], kd["g"]
        for ch, nm in enumerate(("ovl", "ttc", "gap", "vapp", "px", "py", "offroad")):
            w = float(lam.get(nm, 0.0))
            gc = g[..., ch]
            logs[f"v6c/KDcov_{nm}"] = (gc.float().mean()).detach()                # every call (report 129 A70)
            if w <= 0 or not bool(gc.any()):
                continue
            ng = gc.sum().clamp(min=1.0)
            if ch in (0, 1, 6):
                l = (F.binary_cross_entropy_with_logits(E[..., ch], p[..., ch].clamp(0, 1), reduction="none") * gc).sum() / ng
            else:
                j = ch - 2
                l = (F.smooth_l1_loss((E[..., 2:6] / E.new_tensor(CONT_NORM))[..., j], cont_norm(p[..., 2:6])[..., j],
                                      beta=0.5, reduction="none") * gc).sum() / ng
            loss = loss + w * l
            logs[f"v6c/KD_{nm}"] = l.detach()
    return loss, logs


# ------------------------------------------------------------------------------------------------- path hinges
def _front_points(S: torch.Tensor):
    """S [B, 41, 7] -> front-half footprint points at steps 1..40 [B, 40, 12, 2], the heading cos/sin and speed"""
    st = S[:, 1:]
    x, y, h = st[..., 0], st[..., 1], st[..., 2]
    c, s = torch.cos(h)[..., None], torch.sin(h)[..., None]
    fx, fy = torch.meshgrid(torch.linspace(0.0, CH.FRONT, 4, device=S.device),
                            torch.linspace(-CH.HALF_W, CH.HALF_W, 3, device=S.device), indexing="ij")
    fx, fy = fx.reshape(-1), fy.reshape(-1)
    q = torch.stack([x[..., None] + c * fx - s * fy, y[..., None] + s * fx + c * fy], -1)
    # the state's own speed (state j + 1 for step index j), detached, as the official TTC, field_readout and the T-E
    # shift; cf's plan loss keeps its lagged displacement speed (base, left as km; report 129 A20)
    spd = torch.hypot(st[..., 3], st[..., 4]).detach()
    return q, c, s, spd


def front_hinge(S: torch.Tensor, field: torch.Tensor, kind: str, margin: float, cap: float = 3.0,
                rear_mask: bool = False) -> torch.Tensor:
    """per-step hinge h(S, t) [B, 40] of the front-half footprint at steps 1..40 vs field planes (plane j = step j+1,
    0.1 s; TTC: shifted by the state's own speed * dt, dt 0.3 / 0.6 / 0.9 s, vs plane of step + dt, moving ego only:
    own speed >= 5e-3 m/s, report 129 A20) = overlap hinge + TTC hinge (max over the 3 shifts), as cf's plan loss adds
    its occupancy and TTC terms.  kind 'prob': field = occupancy PROBABILITY [B, P, Hf, 200], h = [p - margin]+;
    kind 'dist': field = signed distance to agents (m) [B, P, Hf, 200], h = [margin - d]+.  Out of range -> 0.
    rear_mask: points behind the ego (x < 0) out of range; redundant with CH._sample's own x >= 0 bound since 2e674b4,
    kept for call compatibility (report 129 N-rear-mask-noop-docs)."""
    b = S.shape[0]
    q, c, s, spd = _front_points(S)
    npl = field.shape[1]

    def hinge(vals):
        return ((vals - margin) if kind == "prob" else (margin - vals)).clamp(0, cap)

    v, inr = CH._sample(field[:, :40].float(), q.reshape(b, 40, 12, 2))
    if rear_mask:
        inr = inr & (q.reshape(b, 40, 12, 2)[..., 0] >= 0)
    h_occ = (hinge(v) * inr).amax(-1)                                                     # [B, 40]
    moving = (spd >= 5e-3).float()
    steps = torch.arange(40, device=S.device)
    h_ttc = torch.zeros_like(h_occ)
    for d in (3, 6, 9):                                                                    # dt = 0 is the occupancy part
        sh = (spd * d * 0.1)[..., None]
        tq = torch.stack([q[..., 0] + c * sh, q[..., 1] + s * sh], -1)
        fld = field[:, (steps + d).clamp(max=npl - 1)].float()
        tv, ti = CH._sample(fld, tq.reshape(b, 40, 12, 2))
        if rear_mask:
            ti = ti & (tq.reshape(b, 40, 12, 2)[..., 0] >= 0)
        h_ttc = torch.maximum(h_ttc, (hinge(tv) * ti).amax(-1) * moving)
    return h_occ + h_ttc


def road_hinge(S: torch.Tensor, sdf: torch.Tensor, margin: float = 0.1, cap: float = 3.0,
               rear_mask: bool = False) -> torch.Tensor:
    """cf static form: [B, 41, 7] states vs GT SDF [B, Hf, 200] -> [B] = mean_t max_corner [margin - SDF]+ (in range;
    rear_mask: a corner behind the ego, x < 0, is out of range; redundant with CH._sample's own x >= 0 bound since
    2e674b4, kept for call compatibility)"""
    st = S[:, 1:]
    x, y, h = st[..., 0], st[..., 1], st[..., 2]
    c, s = torch.cos(h)[..., None], torch.sin(h)[..., None]
    cx = torch.tensor([CH.FRONT, CH.FRONT, -CH.REAR, -CH.REAR], device=S.device)
    cy = torch.tensor([CH.HALF_W, -CH.HALF_W, -CH.HALF_W, CH.HALF_W], device=S.device)
    px, py = x[..., None] + c * cx - s * cy, y[..., None] + s * cx + c * cy
    b = S.shape[0]
    v, inr = CH._sample(sdf[:, None].float(), torch.stack([px, py], -1).reshape(b, 1, -1, 2))
    if rear_mask:
        inr = inr & (px.reshape(b, 1, -1) >= 0)
    return ((margin - v).clamp(0, cap) * inr).reshape(b, 40, 4).amax(-1).mean(-1)


def comfort_hinge(S: torch.Tensor, yaw_mask_speed: float = 1.0) -> torch.Tensor:
    """[B, 41, 7] -> [B] max over the 41 states of hinges on the official ego_is_comfortable features, built from the
    official filters as exact_verdict.exact_comfort (report 129 A19): lon / lat accel (Savitzky-Golay window 41),
    jerk magnitude and lon jerk (accel window 8, then the derivative), yaw rate / yaw accel (window 5).
    h = [lon - 2.40]+ + [-4.05 - lon]+ + [|lat| - 4.89]+ (m/s^2, the scale of the old term) + [|jm| - 8.37]+ / 8.37
    + [|jx| - 4.13]+ / 4.13 + m ([|ya| - 1.93]+ / 1.93 + [|yr| - 0.95]+ / 0.95), m = own speed >= yaw_mask_speed.
    DESIGN (not official): the low-speed yaw mask as in ec_rms (the decoder's heading is noise at a stop); 0 = the
    official rule (h == 0 <=> exact_comfort == 1).  jm uses sqrt(ax^2 + ay^2 + 1e-12) (hypot's gradient is NaN at 0).
    The base corridor_plan_loss comfort hinge is left as km has it."""
    ops = EV._dev_ops(EV._comfort_ops(S.shape[-2]), S.device)
    s = S.double()
    ax, ay = s[..., 5], s[..., 6]
    h = s[..., 2]
    unwrap = 2 * torch.pi * torch.cumsum(torch.round(torch.diff(h, dim=-1) / (2 * torch.pi)), -1).detach()
    hu = torch.cat([h[..., :1], h[..., 1:] - unwrap], -1)
    lon, lat = ax @ ops["acc"], ay @ ops["acc"]
    jm, jx = (ax.pow(2) + ay.pow(2) + 1e-12).sqrt() @ ops["jerk"], ax @ ops["jerk"]
    yr, ya = hu @ ops["yaw_rate"], hu @ ops["yaw_acc"]
    m = (torch.hypot(s[..., 3], s[..., 4]) >= yaw_mask_speed).double()
    hh = F.relu(lon - 2.40) + F.relu(-4.05 - lon) + F.relu(lat.abs() - 4.89) + F.relu(jm.abs() - 8.37) / 8.37 \
        + F.relu(jx.abs() - 4.13) / 4.13 + m * (F.relu(ya.abs() - 1.93) / 1.93 + F.relu(yr.abs() - 0.95) / 0.95)
    return hh.amax(-1).float()


def ec_rms(S: torch.Tensor):
    """official EC features (rms jerk, rms yaw acceleration) of [B, 41, 7]; yaw acceleration masked below 1 m/s
    (report 124 §2-4 low-speed yaw mask; Bezier-free decoder still turns the heading at a stop)"""
    key = (S.shape[-2], S.device)
    mj, my = (m.to(S.device) for m in _ec_ops(S.shape[-2]))
    s = S.double()
    h = s[..., 2]
    unwrap = 2 * torch.pi * torch.cumsum(torch.round(torch.diff(h, dim=-1) / (2 * torch.pi)), -1).detach()
    hu = torch.cat([h[..., :1], h[..., 1:] - unwrap], -1)
    j = (s[..., 5].pow(2) + s[..., 6].pow(2) + 1e-12).sqrt() @ mj
    ya = (hu @ my) * (torch.hypot(s[..., 3], s[..., 4]) >= 1.0).double()
    return (j.pow(2).mean(-1) + 1e-12).sqrt().float(), (ya.pow(2).mean(-1) + 1e-12).sqrt().float()


def route_end(S: torch.Tensor, route_s: torch.Tensor) -> torch.Tensor:
    """progress along the GT route at the last state [B]; route_s [B, Hr, Wr] (0.64 m) upsampled to 0.32 m as cf"""
    rs = route_s.float()[:, None]
    if rs.shape[-1] != 200:
        rs = F.interpolate(rs, scale_factor=2, mode="bilinear", align_corners=False)
    v, _ = CH._sample(rs, S[:, -1:, :2][:, None])
    return v[:, 0, 0]


def corr_loss(S_g: torch.Tensor, S_base: torch.Tensor, targets: Dict[str, torch.Tensor],
              teacher_occ: Optional[torch.Tensor], base_fail: torch.Tensor, w: Dict[str, float],
              gt_margin: float = 0.1, teacher_margin: float = 0.3, valid: Optional[torch.Tensor] = None,
              teacher_discount: float = 1.0, fix_source: str = "teacher", rear_mask: bool = False,
              S_fix: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
    """S_g [B, 41, 7] tau_v6 states WITH grad (status init); S_base [B, 41, 7] tau_base states (no grad, same init);
    targets: corridor_sdf [B, 200, 200], corridor_occ_dist [B, 49, 200, 200], corridor_route_s [B, 100, 100],
    corridor_valid [B]; teacher_occ [B, 49, 200, 200] OOF probabilities or None; base_fail [B] (0 / 1, tau_base's
    official NC < 1 or TTC < 1, from the labels); w: weights {teacher, gt, road, comfort, ec, prog};
    teacher_discount gamma: the teacher term's steps weighted gamma^t (normalised), 1 = the plain time mean (default).
    At Delta = 0 the time mean asks a decel-only corrector to pass crossing objects faster (a dead zone, report 124
    §4-4); gamma < 1 makes a later contact cheaper (arm).
    fix_source 'gt': the fixing part 1[tau_base fails] * mean_t h(S_v6) reads the GT occupancy distance instead of the
    teacher field (and the teacher term is off).  K-E / K-M are switched separately (cfg.v6c_kd, feedback 10-27), so
    the source of the fixing signal and the KD can be compared one at a time; report 124 control B = gt + no KD.
    S_fix (10-28 review): states of tau_base edited by the LATERAL controls only (with grad); when given, the fixing
    part 1[tau_base fails] * h(.) reads them, so the time-mean hinge no longer pushes the longitudinal controls (it asked
    for 'faster' through a stopped object; the longitudinal target comes from the official scorer, hook.py).
    The GT guard [h_GT(v6) - h_GT(base)]+ only penalises an edit that RAISES the GT risk (a surrogate on 12 front
    points, not the official collision rule); at Delta = 0 it is 0 with zero gradient, so it never fixes a risky
    tau_base by itself: fixing comes only from the term gated by the official tau_base failure.
    The GT guard and the fix_source='gt' arm read the BASE cell-centre corridor_occ_dist on purpose (a refiner-only
    distance field would cost 49 distance transforms per token in the loader); the T-D teacher target uses
    train_teachers.raster_agents_min1 (every object covers at least one cell, report 129 A04)."""
    S_base = S_base.detach()
    v = targets["corridor_valid"].float() if valid is None else valid.float()
    nv = v.sum().clamp(min=1.0)
    logs, terms = {}, {}
    if float(w.get("gt", 0.0)) > 0:
        od = targets["corridor_occ_dist"]
        hv = front_hinge(S_g, od, "dist", gt_margin, rear_mask=rear_mask)
        with torch.no_grad():
            hb = front_hinge(S_base, od, "dist", gt_margin, rear_mask=rear_mask)
        terms["gt"] = F.relu(hv - hb).mean(-1)
        if fix_source == "gt":
            hf = hv if S_fix is None else front_hinge(S_fix, od, "dist", gt_margin, rear_mask=rear_mask)
            terms["gt"] = terms["gt"] + base_fail.float() * hf.mean(-1)
        logs["v6c/frac_gt_worse"] = (((hv - hb) > 0).any(-1).float() * v).sum() / nv
    if float(w.get("teacher", 0.0)) > 0 and teacher_occ is not None and fix_source == "teacher":
        hv = front_hinge(S_g, teacher_occ, "prob", teacher_margin, rear_mask=rear_mask)
        with torch.no_grad():
            hb = front_hinge(S_base, teacher_occ, "prob", teacher_margin, rear_mask=rear_mask)
        tw = teacher_discount ** torch.arange(hv.shape[-1], device=hv.device, dtype=hv.dtype)
        tw = tw / tw.sum()
        hf = hv if S_fix is None else front_hinge(S_fix, teacher_occ, "prob", teacher_margin, rear_mask=rear_mask)
        terms["teacher"] = (F.relu(hv - hb) * tw).sum(-1) + base_fail.float() * (hf * tw).sum(-1)
        logs["v6c/frac_teacher_worse"] = (((hv - hb) > 0).any(-1).float() * v).sum() / nv
    if float(w.get("road", 0.0)) > 0:
        terms["road"] = road_hinge(S_g, targets["corridor_sdf"], rear_mask=rear_mask)
    if float(w.get("comfort", 0.0)) > 0:
        terms["comfort"] = comfort_hinge(S_g)
    if float(w.get("ec", 0.0)) > 0:
        rj, ry = ec_rms(S_g)
        with torch.no_grad():
            rjb, ryb = ec_rms(S_base)
        terms["ec"] = F.relu(rj - rjb.clamp(min=0.25)) / 0.5 + F.relu(ry - ryb.clamp(min=0.05)) / 0.1
    if float(w.get("prog", 0.0)) > 0 and "corridor_route_s" in targets:
        se = route_end(S_g, targets["corridor_route_s"])
        with torch.no_grad():
            sb = route_end(S_base, targets["corridor_route_s"])
        terms["prog"] = F.relu((sb - se) / 10.0)
    loss = S_g.new_zeros(())
    for nm, t in terms.items():
        l = (t * v).sum() / nv
        loss = loss + float(w[nm]) * l
        logs[f"v6c/corr_{nm}"] = l.detach()
        logs[f"v6c/corr_nz_{nm}"] = (((t > 0).float() * v).sum() / nv).detach()
    return loss, logs


# ------------------------------------------------------------------------------------------------- judge
def judge_loss(j: Dict[str, torch.Tensor], lab: Dict[str, torch.Tensor], eff: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """j = RowJudge outputs ([B, R, ..], eff [B]); lab: official terms per row {nc1, dac1, ttc1, prog1} [B, R]
    (NaN where the token has no table); eff [B] = ex['eff1'][:, 0] -> (loss, logs)"""
    logs, loss = {}, j["dac"].new_zeros(())
    m = torch.isfinite(lab["nc1"])
    if bool(m.any()):
        y = lab["nc1"]
        cls = torch.where(y >= 1, 0, torch.where(y > 0, 1, 2)).long()
        l_nc = F.cross_entropy(j["nc"][m], cls[m])
        l_dac = F.binary_cross_entropy_with_logits(j["dac"][m], (lab["dac1"][m] < 1).float())
        l_ttc = F.binary_cross_entropy_with_logits(j["ttc"][m], (lab["ttc1"][m] < 1).float())
        l_p = F.smooth_l1_loss(j["prog"][m] / 10, lab["prog1"][m] / 10, beta=0.5)
        loss = l_nc + l_dac + l_ttc + l_p
        logs.update({"v6c/J_nc": l_nc.detach(), "v6c/J_dac": l_dac.detach(), "v6c/J_ttc": l_ttc.detach(),
                     "v6c/J_prog": l_p.detach()})
    me = torch.isfinite(eff)
    if bool(me.any()):
        l_e = F.smooth_l1_loss(j["eff"][me] / 10, eff[me] / 10, beta=0.5)
        loss = loss + l_e
        logs["v6c/J_eff"] = l_e.detach()
    return loss, logs
