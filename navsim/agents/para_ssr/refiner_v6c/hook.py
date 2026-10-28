"""refiner v6c inside ParaSSRModel.forward (report 124 §2, §4-2a) and its loss block in ParaSSRAgent.compute_loss.

Forward (train and eval share the code path):
  pick      train: select_official(expected_official(scorer on the OTHER fold's copy candidates)) under no_grad, with the
            same eps / lam as eval (cand_select_eps / cand_select_lam, V8); eval: plan_select_index (cand_official)
  reads     path_reads on tau_base's states S_base (status init, cand_states) from the student BEV / fields; the BEV and
            field inputs pass GradScale(r) (stage D: r = v6c_r, stage P: 0 = detached)
  R-E       E_base [B, 1, T, 12] (+ H)
  R-C       controls from DETACHED E / M / geometry (H only in the H arms) -> decoder -> tau_v6 (Delta = 0: tau_base)
  LQR       S_v6_g = track(tau_v6) WITH grad (L_corr only); S_v6_d = S_v6_g.detach() for the re-read, R-J and M
  R-E again E_v6 on S_v6_d (L_E_re trains R-E on tau_v6's states)
  R-J       rows (tau_base, tau_v6), every input detached -> expected PDMS -> switch (mode B) | always tau_v6 (mode A)
Loss (compute_loss, needs the exact tables): official terms of the 2 rows (float64 exact-init track, official_scores),
per-step labels (labels.exact_step_masks), teacher outputs (teacher.py, optional), then L_E + L_E_re + L_corr + L_J.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from . import nets as N
from ..modules import exact_score as ES

AGENT_K = 32


class GradScale(torch.autograd.Function):
    """identity forward; backward x s (s = 0 -> no gradient, same as detach but keeps the graph shape)"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, s: float) -> torch.Tensor:
        ctx.s = float(s)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return g * ctx.s, None


V6C_FIX_SLOT = 6                 # torch_tracker graph slot of the lateral-only fixing path (0: tau_v6, 1: cf planners)


def _trk(poses: torch.Tensor, init: torch.Tensor, full: bool = True, angular: bool = False, slot: int = 0) -> torch.Tensor:
    """torch_tracker.track, but eager under inference_mode: a CUDA graph first captured inside Lightning's val
    (inference tensors as static inputs) cannot be replayed later in a training step (report 125, risk list).
    slot: a second GRAD call of the same shape in one forward needs its own graph (the backward reads the static
    buffers; the 10-28 lateral-only path on slot 0 overwrote tau_v6's and the loss went NaN)"""
    from ..modules import torch_tracker as TT
    if torch.is_inference_mode_enabled():
        return TT._track(poses, init, full, angular)
    return TT.track(poses, init, full=full, angular=angular, slot=slot)


def _scaled(x: torch.Tensor, r: float) -> torch.Tensor:
    if r <= 0 or not torch.is_grad_enabled() or not x.requires_grad:
        return x.detach()
    return GradScale.apply(x, r)


class RefinerV6c(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        cin, hid = int(cfg.embed_dims), int(getattr(cfg, "v6c_hid", 192))
        self.bev_h, self.bev_w = int(cfg.bev_h), int(cfg.bev_w)
        self.fwd_m = float(cfg.pc_range[4] - cfg.pc_range[1])
        agent_dim = cin + 10 + 7 + 9 * 4 * 16 + 1
        self.re = N.RiskNet(cin, hid, agent_dim=agent_dim)
        arm = str(getattr(cfg, "v6c_h_arm", "E"))
        if arm not in ("E", "EH", "H"):
            raise ValueError(f"v6c_h_arm must be E | EH | H, got {arm!r}")
        self.rc = N.Corrector(hid, h_dim=hid, use_e=arm != "H", use_h=arm != "E", n_cmd=int(cfg.num_navi_cmd),
                              out_mode=str(getattr(cfg, "v6c_rc_out", "per_control")))
        self.rj = N.RowJudge(hid, n_cmd=int(cfg.num_navi_cmd))

    # ------------------------------------------------------------------ #
    def pick(self, predictions: Dict[str, torch.Tensor], training: bool) -> torch.Tensor:
        if not training and "plan_select_index" in predictions:
            return predictions["plan_select_index"].long()
        from ..modules.candidate_scorer import ec_roughness, expected_official, select_official
        with torch.no_grad():
            st = predictions["cand_states"]
            ex = expected_official(predictions, st)
            lam = float(getattr(self.cfg, "cand_select_lam", 0.0))
            rough = ec_roughness(st) if lam > 0 else None
            return select_official(ex, float(getattr(self.cfg, "cand_select_eps", 0.02)), lam, rough)

    def forward(self, predictions: Dict[str, torch.Tensor], features: Dict[str, torch.Tensor],
                det_out: Optional[Dict[str, torch.Tensor]], training: bool) -> Dict[str, torch.Tensor]:
        from ..modules.exact_verdict import exact_comfort
        cfg = self.cfg
        nc = int(cfg.num_navi_cmd)
        status = features["status_feature"][:, nc:nc + 4].float()
        cmd = features["command"].float().reshape(status.shape[0], -1)
        cand, st = predictions["cand_traj"], predictions["cand_states"]
        b = cand.shape[0]
        rows = torch.arange(b, device=cand.device)
        idx = self.pick(predictions, training)
        tau_base = cand[rows, idx].float().detach()                                      # [B, 8, 3]
        S_base = st[rows, idx].float().detach()                                          # [B, 41, 7]
        anchors = predictions["trajectory_anchors"].float()
        anchor_h = anchors[idx][..., 2]                                                  # [B, 8]
        stage_p = str(getattr(cfg, "v6c_stage", "D")) == "P"
        r = 0.0 if stage_p else float(getattr(cfg, "v6c_r", 0.1))
        bev = _scaled(predictions["bev_embed"], r)
        sdf = _scaled(predictions["corridor_sdf"].float(), r)
        occ = torch.sigmoid(_scaled(predictions["corridor_occ"].float(), r))             # PROBABILITIES (§3-1)
        agents = N.agent_features(det_out, AGENT_K) if det_out is not None else None
        # ---- R-E on tau_base
        rm = rear_mask_of(cfg)
        reads_b = N.path_reads(S_base[:, None], bev, sdf, occ, self.bev_h, self.bev_w, self.fwd_m, rear_mask=rm)
        re_b = self.re(reads_b, agents)
        E_b, H_b, M_b = re_b["E"], re_b["H"], reads_b["M"]
        # ---- R-C (inputs detached) -> decoder -> tau_v6
        H_in = H_b[:, 0].detach() if self.rc.use_h else None
        ctrl = self.rc(E_b[:, 0].detach(), M_b[:, 0].detach(), S_base, status, cmd, H=H_in)
        dec = N.decode_v6(tau_base, ctrl, status[:, 0], float(cfg.v6c_a_max), float(cfg.v6c_d_max),
                          str(cfg.v6c_decoder_mode), float(cfg.v6c_lon_st_slope) if training else 0.0,
                          anchor_h=anchor_h)
        tau_v6 = dec["traj"].float()
        # ---- one LQR of tau_v6: grad for L_corr, detached for everything else (report 124 §4-2a)
        init = torch.cat([status, status.new_zeros(b, 3)], -1)
        S_v6_g = _trk(tau_v6, init, full=True).float()                                   # [B, 41, 7]
        S_v6_d = S_v6_g.detach()
        reads_v = N.path_reads(S_v6_d[:, None], bev, sdf, occ, self.bev_h, self.bev_w, self.fwd_m, rear_mask=rm)
        re_v = self.re(reads_v, agents)
        E_v = re_v["E"]
        # ---- R-J on the two rows, every input detached
        E2 = torch.cat([E_b, E_v], 1).detach()
        M2 = torch.cat([M_b, reads_v["M"]], 1).detach()
        S2 = torch.stack([S_base, S_v6_d], 1)
        t21 = N.T21.to(S2.device)
        d = (S_v6_d[:, t21, :2] - S_base[:, t21, :2])                                    # edit in the t0 frame
        hb = S_base[:, t21, 2]
        dl = torch.stack([d[..., 0] * torch.cos(hb) + d[..., 1] * torch.sin(hb),
                          -d[..., 0] * torch.sin(hb) + d[..., 1] * torch.cos(hb)], -1) / 10.0   # along / across base
        dsum = torch.stack([torch.zeros_like(dl), dl], 1).detach()                       # [B, 2, T, 2]
        j = self.rj(E2, M2, S2, dsum, status, cmd, S_base)
        with torch.no_grad():
            comf = exact_comfort(S2)                                                     # [B, 2]
            ep = N.expected_pdms({k: v.detach() for k, v in j.items()}, comf)
            delta = float(getattr(cfg, "v6c_switch_delta", 0.0))
            switch = ep["pdms"][:, 1] - ep["pdms"][:, 0] > delta
            kappa = float(getattr(cfg, "v6c_switch_kappa", -1.0))
            if kappa >= 0:
                from ..modules.candidate_scorer import ec_roughness
                switch &= (ec_roughness(S_v6_d) - ec_roughness(S_base)) <= kappa
            if str(cfg.v6c_mode) == "A":
                switch = torch.ones_like(switch)
            out_traj = torch.where(switch[:, None, None], tau_v6.detach(), tau_base)
        out = {"v6c_idx": idx, "v6c_tau_base": tau_base, "v6c_tau_v6": tau_v6, "v6c_S_base": S_base,
               "v6c_S_v6_g": S_v6_g, "v6c_E_base": E_b, "v6c_E_v6": E_v, "v6c_M_base": M_b, "v6c_ctrl": ctrl,
               "v6c_j_nc": j["nc"], "v6c_j_dac": j["dac"], "v6c_j_ttc": j["ttc"], "v6c_j_prog": j["prog"],
               "v6c_j_eff": j["eff"], "v6c_pdms_hat": ep["pdms"], "v6c_m_hat": ep["m"], "v6c_ep_hat": ep["ep"],
               "v6c_switch": switch, "v6c_out": out_traj, "v6c_anchor_h": anchor_h, "v6c_c_rows": comf}
        if dec.get("lon_dead") is not None:
            out["v6c_lon_dead"] = dec["lon_dead"]
        return out


def rear_mask_of(cfg) -> bool:
    """v6c_rear_mask: True / False when set; None (default) = True, except for the 10-27 checkpoints (v6c_rc_out
    'shared'), which were trained with the old rule (review 10-28: evaluating them must not shift R-E's inputs)"""
    rm = getattr(cfg, "v6c_rear_mask", None)
    if rm is None:
        return str(getattr(cfg, "v6c_rc_out", "per_control")) != "shared"
    return bool(rm)


def zero_touch(mod: nn.Module) -> torch.Tensor:
    """0 * sum(p): every parameter gets a (zero) gradient on micro-batches without a v6c term (DDP,
    find_unused_parameters False)"""
    return sum(p.sum() for p in mod.parameters() if p.requires_grad) * 0.0


# ------------------------------------------------------------------------------------------------- loss block
def _area_flags(S: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """GT area flags along the paths (T-E input, V3): (multiple lanes, non-drivable, rear axle in an intersection)
    at the 21 times from the exact v1 polygon tables -> [B, R, T, 3] (0 where the token has no table)"""
    from ..modules import exact_verdict as EV
    b, r = S.shape[:2]
    out = S.new_zeros(b, r, len(N.T21), 3)
    t21 = N.T21.to(S.device)
    for i in range(b):
        if float(targets["ex1_valid"][i]) < 0.5:
            continue
        g = {k[4:]: v[i] for k, v in targets.items() if k.startswith("ex1_")}
        nd, mu, it = EV.ego_areas(S[i].float(), g["seg"][g["smask"]], g["lane"][g["lmask"]], g["int"][g["imask"]])
        out[i] = torch.stack([mu[:, t21], nd[:, t21], it[:, t21]], -1).float()
    return out


LON_LEVELS = (-0.25, -0.5, -1.0, -2.0)        # z of the constant decel variants, mild -> strong (a_max tanh z m/s^2)


@torch.no_grad()
def official_decel_target(tau_b: torch.Tensor, v0: torch.Tensor, anchor_h: torch.Tensor, init: torch.Tensor,
                          t_obs: Dict[str, torch.Tensor], ex: Dict[str, torch.Tensor], base_fail: torch.Tensor, cfg,
                          levels=LON_LEVELS) -> Dict[str, torch.Tensor]:
    """10-28 review: the longitudinal target from the official scorer instead of the time-mean occupancy hinge (which
    asked a decel-only corrector to drive through a stopped object faster).  tau_base and L constant-decel variants
    (z_lon = level on all 6 controls, lateral 0, the corrector's own decoder) are scored on the exact init:
      tau_base fails (NC or TTC) and some variant passes NC and TTC without a worse DAC -> target = the mildest one
      tau_base fails and none passes -> no target (decel cannot fix it)
      tau_base passes -> target 0 (no edit)
    -> {z [B] target, fixable [B], unfixable [B], use [B]}"""
    b = tau_b.shape[0]
    nl = len(levels)
    valid = torch.isfinite(ex["nc1"][:, 0])
    ok = torch.zeros(b, nl, dtype=torch.bool, device=tau_b.device)
    rows = base_fail.nonzero(as_tuple=True)[0]                                           # only failing tau_base (~7 %)
    if rows.numel():
        m = rows.numel()
        ctrl = tau_b.new_zeros(m * nl, 12)
        ctrl[:, :6] = tau_b.new_tensor(levels).repeat(m)[:, None]
        dec = N.decode_v6(tau_b[rows].repeat_interleave(nl, 0), ctrl, v0[rows].repeat_interleave(nl, 0),
                          float(cfg.v6c_a_max), float(cfg.v6c_d_max), str(cfg.v6c_decoder_mode), 0.0,
                          anchor_h=anchor_h[rows].repeat_interleave(nl, 0))
        st9 = _trk(dec["traj"].double(), init[rows].repeat_interleave(nl, 0), full=True,
                   angular=True).reshape(m, nl, 41, 9).float()
        sub = {k: v[rows] for k, v in t_obs.items() if k.startswith("ex1_") and torch.is_tensor(v) and v.shape[:1] == (b,)}
        exv = ES.official_scores(st9, sub, None)                                          # [m, L], v1 terms only
        ok[rows] = (exv["nc1"] >= 1) & (exv["ttc1"] >= 1) & (exv["dac1"] >= ex["dac1"][rows, :1])
    fixable = base_fail & ok.any(1)
    unfixable = base_fail & ~ok.any(1)
    first = torch.argmax(ok.float(), 1)                                                  # mildest passing level
    z = torch.where(fixable, tau_b.new_tensor(levels)[first], torch.zeros_like(first, dtype=tau_b.dtype))
    return {"z": z, "fixable": fixable, "unfixable": unfixable, "use": valid & ~unfixable}


def v6c_loss(v6c: RefinerV6c, teachers, cfg, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
             predictions: Dict[str, torch.Tensor], training: bool):
    """-> (loss, logs).  Official terms of the 2 rows (tau_base, tau_v6) on the exact init (float64 LQR, as the
    scorer's labels), per-step labels, optional OOF teachers, then L_E (+ L_E_re), L_corr, L_J."""
    from ..modules import exact_score as ES
    from . import labels as LB
    from . import losses as LS
    logs: Dict[str, torch.Tensor] = {}
    tau_b, tau_v = predictions["v6c_tau_base"], predictions["v6c_tau_v6"]
    b = tau_b.shape[0]
    dev = tau_b.device
    has = targets["ex1_valid"].float() > 0.5
    if not bool(has.any()):
        z = zero_touch(v6c)
        logs["v6c/rows_valid"] = z.detach()
        return z, logs
    with torch.no_grad():
        init = ES.ego_init(targets["ex_ego"].double())
        rows2 = torch.stack([tau_b, tau_v.detach()], 1).double()                       # [B, 2, 8, 3]
        st9 = _trk(rows2.reshape(b * 2, 8, 3), init.repeat_interleave(2, 0), full=True,
                   angular=True).reshape(b, 2, 41, 9).float()
        if "ex2_hum_st" not in targets:
            ES.track_human(targets, init)
        # the official row terms (R-J targets, the tau_base failure flag that gates the fixing term) on the SAME
        # observation horizon as the R-E step labels (feedback 10-27 #1); the baseline's tables stay untouched
        t_obs = LB.with_observed_horizon(targets)
        logs["v6c/nobs_raised"] = (t_obs["ex1_nobs"].reshape(-1) != targets["ex1_nobs"].reshape(-1)).float().mean()
        ex = ES.official_scores(st9, t_obs, t_obs)                                     # [B, 2] per term
        lab = LB.stride2(LB.exact_step_masks(st9[..., :7], targets))                   # [B, 2, T]
        base_fail = ((ex["nc1"][:, 0] < 1) | (ex["ttc1"][:, 0] < 1)) & torch.isfinite(ex["nc1"][:, 0])
        nc = int(cfg.num_navi_cmd)
        status = features["status_feature"][:, nc:nc + 4].float()
        lon_mode = str(getattr(cfg, "v6c_lon_target", "official"))
        lon_t = None
        warm_now = training and int(getattr(cfg, "v6c_warmup_steps", 0)) > 0 and \
            int(predictions.get("_v6c_iteration", 0)) <= int(getattr(cfg, "v6c_warmup_steps", 0))
        if lon_mode == "official" and training and not warm_now:
            lv = tuple(float(x) for x in getattr(cfg, "v6c_lon_levels", LON_LEVELS))
            lon_t = official_decel_target(tau_b, status[:, 0], predictions["v6c_anchor_h"], init, t_obs, ex,
                                          base_fail, cfg, lv)
        # teachers (online, OOF copy of the other fold)
        kd = teacher_occ = None
        if teachers is not None and "kd_bev_0" in targets and "cf_fold" in targets:
            copy = teachers.oof_copy(targets["cf_fold"].to(dev))
            flds = teachers.fields(targets["kd_bev_0"], targets["kd_bev_1"], copy)
            # per-teacher gates (report 125): T-D needs BEVFusion, T-M needs ReSMap, T-E reads both
            ok0 = targets.get("kd_ok_0", torch.ones(b, dtype=torch.bool, device=dev)).bool()
            ok1 = targets.get("kd_ok_1", torch.ones(b, dtype=torch.bool, device=dev)).bool()
            teacher_occ = flds["occ_prob"] * ok0.float()[:, None, None, None]             # missing teacher -> no term
            S_rows = torch.stack([predictions["v6c_S_base"], predictions["v6c_S_v6_g"].detach()], 1)
            area = _area_flags(S_rows, targets)
            po = teachers.path_outputs(S_rows, flds, area, copy)
            g_obj = po["g_obj"] * (ok0 & ok1).float()[:, None, None]
            g_ttc = po["g_ttc"] * (ok0 & ok1).float()[:, None, None]                    # ttc, gap, vapp, px, py
            g_off = po["g_off"] * ok1.float()[:, None, None]
            g = torch.stack([g_obj, g_ttc, g_ttc, g_ttc, g_ttc, g_ttc, g_off], -1)
            if not getattr(teachers, "loaded_parts", {}).get("te", True):
                g[..., :6] = 0.0                                                         # no T-E in this checkpoint
            kd = {"p": po["p"], "g": g}
            logs["v6c/teacher_ok_frac_det"] = ok0.float().mean()
            logs["v6c/teacher_ok_frac_map"] = ok1.float().mean()
    lam = {nm: float(getattr(cfg, "v6c_kd_lambda", 0.5)) for nm in ("ovl", "ttc", "gap", "vapp", "px", "py")}
    lam["offroad"] = float(getattr(cfg, "v6c_kd_lambda_off", 0.5))
    if not bool(getattr(cfg, "v6c_kd", True)):
        kd = None                                                                       # K-E / K-M off (independent of
                                                                                        # v6c_fix_source, feedback 10-27)
    rowmask = has.float()[:, None, None]
    lab = {k: (v * rowmask if v.dim() == 3 else v) for k, v in lab.items()}
    pw = float(getattr(cfg, "v6c_pos_weight", 4.0))

    def sl(d, j):
        return None if d is None else {k: v[:, j:j + 1] for k, v in d.items()}

    # ParaSSRLoss.iteration is already incremented for this micro-batch (1 on the first one), so <= gives exactly
    # v6c_warmup_steps warm-up micro-batches
    warm = training and int(getattr(cfg, "v6c_warmup_steps", 0)) > 0 and \
        int(predictions.get("_v6c_iteration", 0)) <= int(getattr(cfg, "v6c_warmup_steps", 0))
    logs["v6c/warmup"] = torch.tensor(float(warm), device=dev)
    if warm:
        # report 124 §4-3 (D epochs 0-4): the scorer is untrained, so the pick is meaningless; R-E alone learns on
        # the stage-T path mixture (anchors near / mid / random, rule speed variants, GT variants) of every token
        lw, lgw = warmup_risk_loss(v6c, cfg, features, targets, predictions, kd_teachers=teachers)
        logs.update({k + "_warm": v for k, v in lgw.items()})
        loss = lw + zero_touch(v6c)
        logs["v6c/loss"] = loss.detach()
        return loss, logs
    l_e, lg = LS.risk_loss(predictions["v6c_E_base"], sl(lab, 0), sl(kd, 0), lam, pw)
    logs.update({k + "_base": v for k, v in lg.items()})
    loss = float(getattr(cfg, "v6c_w_E", 1.0)) * l_e
    w_re = float(getattr(cfg, "v6c_w_E_re", 1.0))
    if w_re > 0:
        l_r, lg = LS.risk_loss(predictions["v6c_E_v6"], sl(lab, 1), sl(kd, 1), lam, pw)
        logs.update({k + "_v6": v for k, v in lg.items()})
        loss = loss + w_re * l_r
    # L_corr on S_v6_g only (status init, the same states R-E read)
    w = {"teacher": float(cfg.v6c_corr_teacher), "gt": float(cfg.v6c_corr_gt), "road": float(cfg.v6c_corr_road),
         "comfort": float(cfg.v6c_corr_comfort), "ec": float(cfg.v6c_corr_ec), "prog": float(cfg.v6c_corr_prog)}
    if "corridor_occ_dist" in targets and training and predictions["v6c_S_v6_g"].requires_grad:
        S_fix = None
        if lon_t is not None:
            # the fixing term (1[tau_base fails] h(S)) moves only the lateral controls; the longitudinal ones follow
            # the official decel target below.  The path keeps the corrector's own (detached) longitudinal controls,
            # so the lateral push fades once the decel clears the object (review 10-28: at lon = 0 it never did)
            ctrl = predictions["v6c_ctrl"]
            dec_l = N.decode_v6(tau_b, torch.cat([ctrl[:, :6].detach(), ctrl[:, 6:]], -1), status[:, 0],
                                float(cfg.v6c_a_max), float(cfg.v6c_d_max), str(cfg.v6c_decoder_mode), 0.0,
                                anchor_h=predictions["v6c_anchor_h"])
            S_fix = _trk(dec_l["traj"].float(), torch.cat([status, status.new_zeros(b, 3)], -1), full=True,
                         slot=V6C_FIX_SLOT).float()
        l_c, lg = LS.corr_loss(predictions["v6c_S_v6_g"], predictions["v6c_S_base"], targets, teacher_occ,
                               base_fail.float(), w, float(cfg.v6c_gt_margin), float(cfg.v6c_teacher_margin),
                               valid=targets["corridor_valid"].float(),
                               teacher_discount=float(getattr(cfg, "v6c_teacher_discount", 1.0)),
                               fix_source=str(getattr(cfg, "v6c_fix_source", "teacher")),
                               rear_mask=rear_mask_of(cfg), S_fix=S_fix)
        logs.update(lg)
        loss = loss + float(cfg.v6c_w_corr) * l_c
    if lon_t is not None:
        zl = predictions["v6c_ctrl"][:, :6]
        wt = torch.where(lon_t["fixable"], torch.full_like(lon_t["z"], float(getattr(cfg, "v6c_lon_fix_weight", 4.0))),
                         torch.ones_like(lon_t["z"])) * lon_t["use"].float()
        lt = torch.nn.functional.smooth_l1_loss(zl, lon_t["z"][:, None].expand_as(zl).to(zl.dtype), beta=0.1,
                                                reduction="none").mean(-1)
        l_t = (lt * wt).sum() / wt.sum().clamp(min=1.0)
        loss = loss + float(getattr(cfg, "v6c_lon_target_weight", 1.0)) * l_t
        with torch.no_grad():
            logs["v6c/lon_tgt"] = l_t.detach()
            logs["v6c/lon_fixable"] = lon_t["fixable"].float().mean()
            logs["v6c/lon_unfixable"] = lon_t["unfixable"].float().mean()
            zm = zl.mean(-1)
            if bool(lon_t["fixable"].any()):
                logs["v6c/z_lon_fix"] = zm[lon_t["fixable"]].mean()
                logs["v6c/z_lon_fix_tgt"] = lon_t["z"][lon_t["fixable"]].mean()
            pas = lon_t["use"] & ~lon_t["fixable"]
            if bool(pas.any()):
                logs["v6c/z_lon_pass"] = zm[pas].mean()
    # L_J: rows' official terms (NaN where the token has no v1 table)
    jl = {"nc": predictions["v6c_j_nc"], "dac": predictions["v6c_j_dac"], "ttc": predictions["v6c_j_ttc"],
          "prog": predictions["v6c_j_prog"], "eff": predictions["v6c_j_eff"]}
    pen = float(getattr(cfg, "v6c_lon_pos_penalty", 0.0))
    if pen > 0 and training and str(cfg.v6c_decoder_mode) == "A":
        # 10-28 review: in mode A a longitudinal control above 0 is clamped away (no effect, no gradient without a
        # straight-through slope); pull it back to the live boundary so the clamp is never an absorbing state
        l_pen = torch.relu(predictions["v6c_ctrl"][:, :6]).sum(-1).mean()
        loss = loss + pen * l_pen
        logs["v6c/lon_pos_pen"] = l_pen.detach()
    l_j, lg = LS.judge_loss(jl, {k: ex[k] for k in ("nc1", "dac1", "ttc1", "prog1")}, ex["eff1"][:, 0])
    logs.update(lg)
    loss = loss + float(getattr(cfg, "v6c_w_J", 1.0)) * l_j
    loss = loss + zero_touch(v6c)
    with torch.no_grad():
        v = torch.isfinite(ex["pdms1"]).all(1)
        if bool(v.any()):
            d = ex["pdms1"][v, 1] - ex["pdms1"][v, 0]
            sw = predictions["v6c_switch"][v].float()
            logs["v6c/pdms1_base"] = ex["pdms1"][v, 0].mean()
            logs["v6c/pdms1_v6_minus_base"] = d.mean()
            logs["v6c/pdms1_out_minus_base"] = (d * sw).mean()
            logs["v6c/switch_rate"] = sw.mean()
            logs["v6c/frac_v6_better"] = (d > 1e-6).float().mean()
            logs["v6c/frac_v6_worse"] = (d < -1e-6).float().mean()
            logs["v6c/base_fail"] = base_fail[v].float().mean()
            pred_better = predictions["v6c_pdms_hat"][v, 1] > predictions["v6c_pdms_hat"][v, 0]
            flip = d.abs() > 1e-6
            if bool(flip.any()):
                logs["v6c/J_pair_acc"] = ((pred_better == (d > 0)) & flip).float().sum() / flip.float().sum()
            # R-J's comfort C is computed on the status-init states; the label c1 on the exact init (report 124 §2-5a)
            logs["v6c/c1_mismatch"] = (ex["c1"][v] != predictions["v6c_c_rows"][v]).float().mean()
        dz = predictions["v6c_ctrl"][:, :6]
        logs["v6c/z_lon_mean"] = dz.mean()
        logs["v6c/z_lon_zero"] = (dz == 0).float().mean()
        logs["v6c/z_lon_std"] = dz.mean(-1).std() if dz.shape[0] > 1 else dz.new_zeros(())
        if "v6c_lon_dead" in predictions:
            logs["v6c/lon_dead"] = predictions["v6c_lon_dead"].float().mean()
        logs["v6c/lat_abs"] = predictions["v6c_ctrl"][:, 6:].abs().mean()
        logs["v6c/rows_valid"] = has.float().mean()
    logs["v6c/loss"] = loss.detach()
    return loss, logs


def warmup_risk_loss(v6c: RefinerV6c, cfg, features, targets, predictions, kd_teachers=None, n_paths: int = 24):
    """L_E (GT, + KD if teachers) on a per-token path mixture (tools/refiner_v6c/train_teachers.path_mixture), read
    from the same student BEV / fields / det tokens as the on-policy rows"""
    from ..modules import exact_score as ES
    from . import labels as LB
    from . import losses as LS
    from .mixture import path_mixture
    dev = predictions["bev_embed"].device
    nc = int(cfg.num_navi_cmd)
    status = features["status_feature"][:, nc:nc + 4].float()
    b = status.shape[0]
    has = targets["ex1_valid"].float() > 0.5
    gen = torch.Generator().manual_seed(int(predictions.get("_v6c_iteration", 0)))
    with torch.no_grad():
        paths = path_mixture(predictions["trajectory_anchors"].float().cpu(), targets["ex2_hum"].float().cpu(),
                             targets["ex2_hvalid"].cpu(), gen)
        sel = torch.randperm(paths.shape[1], generator=gen)[:n_paths]
        paths = paths[:, sel].to(dev)
        r = paths.shape[1]
        st9 = _trk(paths.reshape(b * r, 8, 3).double(), ES.ego_init(targets["ex_ego"].double()).repeat_interleave(r, 0),
                   full=True, angular=True).reshape(b, r, 41, 9).float()
        lab = LB.stride2(LB.exact_step_masks(st9[..., :7], targets))
        lab = {k: (v * has.float()[:, None, None] if v.dim() == 3 else v) for k, v in lab.items()}
        S = _trk(paths.reshape(b * r, 8, 3).float(), torch.cat([status, status.new_zeros(b, 3)], -1)
                 .repeat_interleave(r, 0), full=True).reshape(b, r, 41, 7).float()
    stage_p = str(getattr(cfg, "v6c_stage", "D")) == "P"
    rr = 0.0 if stage_p else float(getattr(cfg, "v6c_r", 0.1))
    bev = _scaled(predictions["bev_embed"], rr)
    sdf = _scaled(predictions["corridor_sdf"].float(), rr)
    occ = torch.sigmoid(_scaled(predictions["corridor_occ"].float(), rr))
    det = {k: predictions[k] for k in ("all_cls_scores", "all_bbox_preds", "traj_preds", "traj_cls_preds",
                                        "motion_hidden") if k in predictions}
    agents = N.agent_features(det, AGENT_K) if len(det) == 5 else None
    reads = N.path_reads(S, bev, sdf, occ, v6c.bev_h, v6c.bev_w, v6c.fwd_m,
                         rear_mask=rear_mask_of(cfg))
    E = v6c.re(reads, agents)["E"]
    kd = None
    if kd_teachers is not None and "kd_bev_0" in targets and "cf_fold" in targets and bool(getattr(cfg, "v6c_kd", True)):
        with torch.no_grad():
            copy = kd_teachers.oof_copy(targets["cf_fold"].to(dev))
            flds = kd_teachers.fields(targets["kd_bev_0"], targets["kd_bev_1"], copy)
            po = kd_teachers.path_outputs(S, flds, _area_flags(S, targets), copy)
            ok0 = targets["kd_ok_0"].bool() if "kd_ok_0" in targets else torch.ones(b, dtype=torch.bool, device=dev)
            ok1 = targets["kd_ok_1"].bool() if "kd_ok_1" in targets else torch.ones(b, dtype=torch.bool, device=dev)
            g_obj = po["g_obj"] * (ok0 & ok1).float()[:, None, None]
            g_ttc = po["g_ttc"] * (ok0 & ok1).float()[:, None, None]
            g = torch.stack([g_obj] + [g_ttc] * 5 + [po["g_off"] * ok1.float()[:, None, None]], -1)
            if not getattr(kd_teachers, "loaded_parts", {}).get("te", True):
                g[..., :6] = 0.0                                                         # no T-E: same rule as v6c_loss
            kd = {"p": po["p"], "g": g}
    lam = {nm: float(getattr(cfg, "v6c_kd_lambda", 0.5)) for nm in ("ovl", "ttc", "gap", "vapp", "px", "py")}
    lam["offroad"] = float(getattr(cfg, "v6c_kd_lambda_off", 0.5))
    return LS.risk_loss(E, lab, kd, lam, float(getattr(cfg, "v6c_pos_weight", 4.0)))
