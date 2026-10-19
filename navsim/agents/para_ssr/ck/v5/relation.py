"""refiner v5 object relation channel (V5Cfg.relation 'teacher'; report 97 §2-3, add.md (2)).

Objects: the t0 rows of the refiner-v3 object pack (OBJ_INPUT_KEYS: obj_in [T, 3, K, 17] t0 / t-0.5 s / t-1.0 s,
  obj_mask [T, 3, K], frame_valid [T, 3]; BEVFusion detections, rows sorted by score), the first n_obj t0 slots.  Per
  object: centre c0, heading h0, L, W, velocity v0, speed / direction, the 0-parameter constant acceleration of the
  causal past frames (temporal.forecast.ca_history: nearest past box of the same group within 1.5 m, a = dv / dt,
  +-6 m/s^2; has = a past match exists), class group.
Modes at the times t_n (+ delta for TTC): hold (c0), constant velocity (c0 + v0 t), constant acceleration with the stop
  clamp (temporal.forecast.ca_positions; only where has).  Weights (no hard rule): w_hold = pi_s (static classes 1,
  else sigmoid((0.1 - |v0|) / 0.05): a soft hold, so a slow mover is not frozen), w_ca = (1 - pi_s) has sigmoid(c_g)
  (learned per group, vehicle / VRU; 0 without rel_ca), w_cv = the rest.
Executed states: the candidate's LQR states [B, Q, 6] (every z_stride-th of the 41), ego box = the official footprint
  (centre 1.461 m ahead of the rear axle, 5.176 x 2.297 m) at the state (NC, delta 0) and projected |v| delta along the
  heading (TTC, delta 0.3 / 0.6 / 0.9 s; objects at t + delta).
Overlap: p = sum_m w_m sigmoid(-sep_m / eps) (SAT separation temporal.raster.sat_sep, eps = exp(log_eps) learned, init
  V5Cfg.rel_eps), times the soft not-behind factor pi_s + (1 - pi_s) sigmoid((fwd + 1.3) / 0.3) (the at-fault proxy of
  the range field B: stopped / static objects count from any side; measured from the CURRENT ego box at state n also
  for the TTC projections, as the official TTC), times the official stopped-ego skips (NC |v| >
  0.05 m/s, TTC |v| >= 5e-3).  NC reads delta 0, TTC the max over the four deltas.
Reliability a_i = sigmoid(MLP(score, class one-hot, range / 32, bearing cos / sin, speed / 10, has, |a| / 6, L / 5,
  W / 3)) (learned; target in tools/ck/train_ck2.py: matched agent with an agreeing velocity -- never a gate).
  Relation 'student': the student rows' feats (ck/v5/student_objects.py), + p_stop_i of the det head's V1c
  stop_branch with V5Cfg.rel_pstop.
Outputs per candidate b:
  r_i^k = max_n p^k_ni (per object), lp_k = sum_i log(1 - a_i r_i^k) (<= 0; k = NC, TTC);
  H^k_n = 1 - prod_i (1 - a_i p^k_ni) (per-state attributed hazard, the 'not bbox-anchored' quantity of Z);
  ev [B, 4] = [lp_nc / 5, lp_ttc / 5, max_i a_i r_i^nc, sum_i a_i / 10] (residual inputs);
  top [B, 21] = the 3 objects with the largest a_i max(r^nc, r^ttc): [a r^nc, a r^ttc, x / 32, y / 32, vx / 10,
  vy / 10, t* / 4] (t* = the state time of the largest overlap; explanation + corrector input).
Gradients: eps, the CA weights and the reliability MLP (object rows and states are inputs; the separations run under
no_grad).  Float32 (autocast off).
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from ..temporal.config import COL, EGO_L, EGO_OFF, EGO_W, GROUP_OF_CLASS, N_CLS
from ..temporal.forecast import ca_history, effective_mask
from ..temporal.raster import sat_sep
from .config import N_TOP, N_TOP_FEAT, V5Cfg

Tensor = torch.Tensor
DELTAS = (0.0, 0.3, 0.6, 0.9)          # official TTC future_time_idcs x 0.1 s (delta 0 = the NC read)
BEHIND_M = -1.3                        # not-behind proxy of the range field B (range_targets.BEHIND_M)
STOP_NC, STOP_TTC = 0.05, 5e-3         # [m/s] official stopped-ego skips
P_MAX = 1.0 - 1e-4
N_FEAT = 1 + N_CLS + 1 + 2 + 1 + 1 + 1 + 2
assert abs(EGO_L - 5.176) < 1e-6 and abs(EGO_W - 2.297) < 1e-6 and abs(EGO_OFF - 1.461) < 1e-6


def object_rows(obj_in: Tensor, obj_mask: Tensor, frame_valid: Tensor, n: int) -> Dict[str, Tensor]:
    """t0 attributes of the first n slots (padded to n, valid False) -> dict of [T, n] / [T, n, 2] float32 tensors +
    feats [T, n, N_FEAT] (reliability MLP input)."""
    obj_in, obj_mask, frame_valid = obj_in.float(), obj_mask.bool(), frame_valid.bool()
    T, _, K, _ = obj_in.shape
    hist = ca_history(obj_in, obj_mask, frame_valid)
    v0m = effective_mask(obj_mask, frame_valid)[:, 0]                                   # [T, K]
    any_slot = v0m.any(0)
    k_valid = int(torch.nonzero(any_slot).max()) + 1 if bool(any_slot.any()) else 1
    # every valid t0 row up to n, trimmed to the batch's last valid slot (rows are a score-sorted prefix; review
    # 2026-10-20 f1: the old top-32 cut dropped ~16 % of matched agents, ~30 % of pedestrians)
    k = max(1, min(int(n), K, k_valid))
    n = k
    x0 = obj_in[:, 0, :k]
    valid = effective_mask(obj_mask, frame_valid)[:, 0, :k]
    cls1h = x0[..., COL["cls0"]:COL["cls0"] + N_CLS]
    cls = cls1h.argmax(-1)
    group = torch.as_tensor(GROUP_OF_CLASS, dtype=torch.long, device=obj_in.device)[cls]
    c0, v0 = x0[..., 0:2], x0[..., COL["vx"]:COL["vy"] + 1]
    sp0, u0 = hist["sp0"][:, :k], hist["u0"][:, :k]
    acc, has = hist["acc"][:, :k], hist["has"][:, :k] & (group < 2)
    static = group == 2
    pi_s = torch.where(static, torch.ones_like(sp0), torch.sigmoid((0.1 - sp0) / 0.05))
    rng = torch.linalg.vector_norm(c0, dim=-1)
    bear = torch.atan2(c0[..., 1], c0[..., 0])
    L, W = x0[..., COL["L"]], x0[..., COL["W"]]
    feats = torch.cat([x0[..., COL["score"]:COL["score"] + 1], cls1h, (rng / 32.0)[..., None],
                       torch.cos(bear)[..., None], torch.sin(bear)[..., None], (sp0 / 10.0)[..., None],
                       has.float()[..., None], (acc.abs() / 6.0)[..., None], (L / 5.0)[..., None],
                       (W / 3.0)[..., None]], -1)
    out = dict(c0=c0, v0=v0, h0=torch.atan2(x0[..., COL["sin"]], x0[..., COL["cos"]]), L=L, W=W, sp0=sp0, u0=u0,
               acc=acc, has=has, group=group, pi_s=pi_s, valid=valid, feats=feats)
    if k < n:                                                    # pad to n slots (valid False, finite zeros)
        for key, v in out.items():
            pad = torch.zeros(T, n - k, *v.shape[2:], dtype=v.dtype, device=v.device)
            if key == "L" or key == "W":
                pad = pad + 1.0
            out[key] = torch.cat([v, pad], 1)
    for key in ("c0", "v0", "h0", "L", "W", "sp0", "u0", "acc", "pi_s", "feats"):
        out[key] = torch.where(out["valid"].reshape(*out["valid"].shape, *([1] * (out[key].dim() - 2))),
                               out[key], torch.zeros_like(out[key]) + (1.0 if key in ("L", "W") else 0.0))
    return out


class RelationReader(nn.Module):
    def __init__(self, cfg: V5Cfg):
        super().__init__()
        self.n_obj = int(cfg.n_obj)
        self.use_ca = bool(cfg.rel_ca)
        self.log_eps = nn.Parameter(torch.tensor(math.log(float(cfg.rel_eps))))
        # CA mode weight per group (vehicle, VRU); only built with the CA mode (an unused parameter breaks DDP)
        self.ca_logit = nn.Parameter(torch.zeros(2)) if self.use_ca else None
        h = int(cfg.rel_hidden)
        self.source = str(cfg.relation)                                    # 'teacher' (pack rows) | 'student' (det head)
        if self.source == "student":
            from .student_objects import N_FEAT_S
            n_feat = N_FEAT_S + (1 if cfg.rel_pstop else 0)              # + p_stop_i (V5Cfg.rel_pstop)
        else:
            n_feat = N_FEAT
        self.rel_mlp = nn.Sequential(nn.Linear(n_feat, h), nn.GELU(), nn.Linear(h, 1))
        with torch.no_grad():
            self.rel_mlp[-1].bias.fill_(math.log(0.9 / 0.1))               # teacher boxes: ~90 % match GT [measured]

    def reliability(self, O: Dict[str, Tensor]) -> Tensor:
        return torch.sigmoid(self.rel_mlp(O["feats"]).float()[..., 0]) * O["valid"].float()

    def forward(self, objs: Dict[str, Tensor], states: Tensor, t: Tensor, tok: Tensor) -> Dict[str, Tensor]:
        """objs: the teacher's OBJ_INPUT_KEYS of T tokens ('teacher') or the student's object rows (student_objects.
        student_object_rows, already per token: 'student'); states [B, Q, 6] (rows of token tok [B]), t [Q] (s) -> module
        docstring outputs + a_tok / valid_tok [T, n_obj] (token level, for the reliability loss), objs_tok (rows)."""
        dev = states.device
        with torch.autocast(device_type=dev.type, enabled=False):
            if "c0" in objs:                                     # already rows (student det head)
                if self.source != "student":
                    raise ValueError("RelationReader('teacher') got student object rows")
                O = {k: v.to(dev) for k, v in objs.items()}
            else:
                if self.source != "teacher":
                    raise ValueError("RelationReader('student') needs student_object_rows, got pack rows")
                O = object_rows(objs["obj_in"].to(dev), objs["obj_mask"].to(dev), objs["frame_valid"].to(dev),
                                self.n_obj)
            a_tok = self.reliability(O)
            tok = tok.to(dev).long()
            g = lambda x: x.index_select(0, tok)                                         # noqa: E731
            st = states.float()
            B, Q = st.shape[:2]
            tq = t.to(dev).float()
            dl = torch.as_tensor(DELTAS, dtype=torch.float32, device=dev)
            tt = tq[:, None] + dl[None]                                                  # [Q, D]
            c0, v0, h0 = g(O["c0"]), g(O["v0"]), g(O["h0"])                              # [B, N, ..]
            L, W, sp0, u0, acc = g(O["L"]), g(O["W"]), g(O["sp0"]), g(O["u0"]), g(O["acc"])
            pi_s, valid, has, grp = g(O["pi_s"]), g(O["valid"]), g(O["has"]), g(O["group"])
            # geometry: no graph for the teacher's pack rows (data); the student's rows keep their det-head graph on
            # c0 / v0 (design V2, report 99 N3) -- with detached rows nothing in here requires grad either way
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.source == "student"):
                hd, spd = st[..., 2].detach(), st[..., 3].detach().abs()
                eh = torch.stack([torch.cos(hd), torch.sin(hd)], -1)                     # [B, Q, 2]
                ce = st[..., :2].detach() + EGO_OFF * eh
                ce_d = ce[:, :, None, :] + (spd[..., None] * dl)[..., None] * eh[:, :, None, :]   # [B, Q, D, 2]
                ce_x = ce_d[:, :, :, None, :]                                            # [B, Q, D, 1, 2]
                he_x = hd[:, :, None, None]
                T5 = tt[None, :, :, None]                                                # [1, Q, D, 1]
                ho, Lo, Wo = h0[:, None, None, :], L[:, None, None, :], W[:, None, None, :]
                pos_hold = c0[:, None, None, :, :]
                pos_cv = pos_hold + v0[:, None, None, :, :] * T5[..., None]
                s_hold, fwd = sat_sep(ce_x, he_x, pos_hold, ho, Lo, Wo)                   # [B, Q, D, N]
                s_cv, fwd_cv = sat_sep(ce_x, he_x, pos_cv, ho, Lo, Wo)
                if self.use_ca:
                    a_ = acc[:, None, None, :]
                    s_ = sp0[:, None, None, :]
                    neg = a_ < -1e-6
                    tstop = torch.where(neg, -s_ / torch.where(neg, a_, -torch.ones_like(a_)),
                                        torch.full_like(a_, float("inf")))
                    te = torch.minimum(T5, tstop)                                      # [B, Q, D, N]
                    dist = s_ * te + 0.5 * a_ * te ** 2
                    pos_ca = pos_hold + u0[:, None, None, :, :] * dist[..., None]
                    s_ca, _ = sat_sep(ce_x, he_x, pos_ca, ho, Lo, Wo)
                # not-behind from the CURRENT ego box centre at state n (official TTC: is_agent_ahead of the state's
                # rear axle; the projected box is only used for the overlap) -- review 98 R3; delta 0 = unchanged
                ce_now, eh_now = ce[:, :, None, None, :], eh[:, :, None, None, :]
                fwd = ((pos_hold - ce_now) * eh_now).sum(-1).expand(-1, -1, len(DELTAS), -1)
                fwd_cv = ((pos_cv - ce_now) * eh_now).sum(-1)
                moving_fwd = torch.where(pi_s[:, None, None, :] > 0.5, fwd, fwd_cv)
                nb = pi_s[:, None, None, :] + (1 - pi_s[:, None, None, :]) * torch.sigmoid(
                    (moving_fwd - BEHIND_M) / 0.3)
                gate = nb * valid[:, None, None, :].float()
                mv_nc = (spd > STOP_NC).float()[:, :, None]                               # [B, Q, 1]
                mv_tt = (spd >= STOP_TTC).float()[:, :, None]
            eps = torch.exp(self.log_eps.float()).clamp(min=0.05)
            w_hold = pi_s
            if self.use_ca:
                w_ca = (1 - pi_s) * has.float() * torch.sigmoid(self.ca_logit.float())[grp.clamp(max=1)]
            else:
                w_ca = torch.zeros_like(pi_s)
            w_cv = (1 - w_hold - w_ca).clamp(min=0.0)
            ww = lambda w: w[:, None, None, :]                                           # noqa: E731
            p = ww(w_hold) * torch.sigmoid(-s_hold / eps) + ww(w_cv) * torch.sigmoid(-s_cv / eps)
            if self.use_ca:
                p = p + ww(w_ca) * torch.sigmoid(-s_ca / eps)
            p = p * gate                                                                 # [B, Q, D, N]
            p_nc = p[:, :, 0] * mv_nc                                                    # [B, Q, N]
            p_tt = p.amax(2) * mv_tt
            a = g(a_tok)                                                                 # [B, N]
            r_nc, r_tt = p_nc.amax(1), p_tt.amax(1)                                      # [B, N]
            ar_nc, ar_tt = (a * r_nc).clamp(max=P_MAX), (a * r_tt).clamp(max=P_MAX)
            lp = torch.stack([torch.log1p(-ar_nc).sum(-1), torch.log1p(-ar_tt).sum(-1)], -1)   # [B, 2]
            H = torch.stack([1 - torch.exp(torch.log1p(-(a[:, None] * p_nc).clamp(max=P_MAX)).sum(-1)),
                             1 - torch.exp(torch.log1p(-(a[:, None] * p_tt).clamp(max=P_MAX)).sum(-1))], -1)
            ev = torch.stack([lp[:, 0] / 5.0, lp[:, 1] / 5.0, ar_nc.amax(-1), (a * valid.float()).sum(-1) / 10.0], -1)
            # top related objects (explanation + corrector input)
            score = torch.maximum(ar_nc, ar_tt).detach()
            k = min(N_TOP, score.shape[1])
            idx = torch.topk(score, k, dim=1).indices                                    # [B, k]
            with torch.no_grad():
                q_obj = (p_nc + p_tt).argmax(1)                                          # [B, N] state of max overlap
                t_obj = tq[q_obj]
            gi = lambda x: torch.gather(x, 1, idx)                                       # noqa: E731
            top = torch.stack([gi(ar_nc), gi(ar_tt), gi(c0[..., 0]) / 32.0, gi(c0[..., 1]) / 32.0,
                               gi(v0[..., 0]) / 10.0, gi(v0[..., 1]) / 10.0, gi(t_obj) / 4.0], -1)   # [B, k, 7]
            if k < N_TOP:
                top = torch.cat([top, torch.zeros(B, N_TOP - k, N_TOP_FEAT, device=dev)], 1)
        return {"lp": lp, "ev": ev, "H": H, "top": top.reshape(B, N_TOP * N_TOP_FEAT), "a_tok": a_tok,
                "valid_tok": O["valid"], "objs_tok": O, "r_nc": r_nc, "r_ttc": r_tt}
