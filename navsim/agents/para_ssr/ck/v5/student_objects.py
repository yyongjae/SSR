"""refiner v5 student object rows (V5Cfg.relation 'student'; report 97 §2-3): the e2e student's OWN det head as the object
source of the relation channel (no teacher input at inference).

Det head output (modules/det_motion_head.py, final decoder layer): box code [B, Q, 10] = (cx, cy, log w, log l, cz,
  log h, sin yaw, cos yaw, vx, vy) in the det frame x_right / y_forward (pc_range x -32..32, y 0..32; the aux records'
  frame, navsim/evaluate/aux_metrics.decode = losses.denormalize_bbox of it), class logits [B, Q, 7] (nuPlan
  TrackedObjectType 0..6 = refiner.gt_future.CLASS_NAMES[:7]), det_hidden [B, Q, 256].
N frame (refiner): x_N = y_rec, y_N = -x_rec, heading h_N = wrap(-yaw - pi), L = l, W = w, vx_N = vy_rec, vy_N = -vx_rec
  (the conversion verified against GT in the bevkd failure analysis, det/scripts/common.rec_box_to_n).
Selection (declared input-selection hyper-parameters, report 97 §9-16; no evidence gate): per token the queries with max
  class probability >= score_thr inside the student window (x_N in [0, 32], |y_N| <= 32, |bearing| <= 80 deg), top n_obj
  by score.  Static classes (cone, barrier, czone sign, generic object) are held (pi_s = 1); agents get the soft hold
  sigmoid((0.1 - |v|) / 0.05).  No past frames (two camera frames give no finite difference): has = False (no CA mode).
  Velocity = the box code's entries 8, 9 = the det head's DEPLOYED velocity: with the V1c stop_branch
  (agent.config.det_stop_branch) that is already (1 - sigmoid(stop)) v_hat, so the soft hold and the relation read the
  gated speed and, with a graph-carrying code (V2), the relation losses reach both v_hat and the stop branch.
  acc = 0 and has = False always: the ACC head's a_lon is an aux target, the critic's CA mode 7 is off (design §3.3).
feats [T, n, 256 + 7 + 7 (+ 1)] = [det_hidden, class probabilities, score, range / 32, bearing cos / sin, speed / 10,
  L / 5, W / 3 (, p_stop = sigmoid(stop logit) with stop_logit given: V5Cfg.rel_pstop)] -> the reliability MLP.
Gradient (design V2, report 97 §2-2 "det outputs into the relation module without detach"; report 99 N3): the slot
  selection, class, heading, size and feats are detached; the centre c0 = (x, y) and velocity v0 = (vx, vy) (and what
  derives from them: sp0, u0, pi_s) keep the box code's graph when `code` requires grad, so the relation losses (L_H,
  the corrector's L_refine / AWR through Z, Z-KD) reach the det head's position / velocity regression.  A detached
  `code` (online2.student_rel_rows with rel_det_grad off, inference) gives the old rows exactly.
  Matched boxes only (design V2 'the matched boxes'; online2 v4.v5.rel_det_grad_matched): rows_keep_graph keeps that
  graph on the rows matched one-to-one to a GT track at t0 (refine_loss.match_t0, 2 m) and detaches the others (ghost /
  duplicate boxes) for every relation consumer; training only (inference rows carry no graph).
Futures (V5Cfg.rel_modes 'cv+fut', design §3.2 modes m1..6): with traj [T, Q, M, 8, 2] / traj_cls [T, Q, M] (the det
  head's traj_preds / traj_cls_preds: PER-STEP offsets in the det frame (x_right, y_forward) of the t0 ego frame, 0.5 s
  steps to 4 s, the first relative to the t0 box centre -- para_ssr_targets._track_future, det_motion_head
  ._last_valid_fde) the rows gain fut [T, n, M, 8, 2] = the cumulative displacement from c0 in the N frame
  (cumsum of (d_y, -d_x)) and pi_fut [T, n, M] = softmax(traj_cls); both detached (the futures are trained by the det
  head's L_traj / L_traj_cls only); padding / invalid slots: fut 0, pi_fut uniform.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ..temporal.config import GROUP_OF_CLASS, N_CLS

Tensor = torch.Tensor
WIN_X, WIN_Y, FOV_DEG = 32.0, 32.0, 80.0
N_FEAT_S = 256 + N_CLS + 7      # hidden, class probabilities, score, range, cos, sin, speed, L, W


def student_object_rows(code: Tensor, cls_logit: Tensor, hidden: Tensor, n: int, score_thr: float = 0.3,
                        stop_logit: Optional[Tensor] = None, traj: Optional[Tensor] = None,
                        traj_cls: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """code [T, Q, 10], cls_logit [T, Q, 7], hidden [T, Q, 256] (final det layer; stop_logit [T, Q] = its V1c stop
    logit, only for V5Cfg.rel_pstop; traj [T, Q, M, 8, 2] / traj_cls [T, Q, M] = its futures, only for rel_modes
    'cv+fut') -> the relation.object_rows dict (c0, v0, h0, L, W, sp0, u0, acc, has, group, pi_s, valid, feats) of n
    slots per token, + q_idx [T, n] (query of each slot, -1 = padding) (+ fut / pi_fut with traj, module docstring).
    c0 / v0 / sp0 / u0 / pi_s carry code's graph (module docstring 'Gradient'); the rest (feats incl. p_stop, the
    futures) is detached."""
    if stop_logit is not None and tuple(stop_logit.shape) != tuple(code.shape[:2]):
        raise ValueError(f"student_object_rows: stop_logit {tuple(stop_logit.shape)} != code {tuple(code.shape[:2])}")
    if (traj is None) != (traj_cls is None):
        raise ValueError("student_object_rows: traj and traj_cls go together")
    if traj is not None and (traj.dim() != 5 or tuple(traj.shape[:2]) != tuple(code.shape[:2]) or traj.shape[3] != 8
                             or traj.shape[4] != 2 or tuple(traj_cls.shape) != tuple(traj.shape[:3])):
        raise ValueError(f"student_object_rows: traj {tuple(traj.shape)} / traj_cls {tuple(traj_cls.shape)} vs code "
                         f"{tuple(code.shape)} (expected [T, Q, M, 8, 2] / [T, Q, M])")
    code_g = code.float()                                   # position / velocity entries: graph kept (V2)
    code = code_g.detach()
    with torch.no_grad():
        prob = torch.sigmoid(cls_logit.float())
        T, Q, _ = code.shape
        dev = code.device
        score, cls = prob.max(-1)
        xr, yf = code[..., 0], code[..., 1]
        x, y = yf, -xr
        yaw = torch.atan2(code[..., 6], code[..., 7])
        h = torch.remainder((-yaw - math.pi) + math.pi, 2 * math.pi) - math.pi      # wrap(-yaw - pi)
        L, W = torch.exp(code[..., 3]).clamp(0.2, 30.0), torch.exp(code[..., 2]).clamp(0.2, 10.0)
        bearing = torch.rad2deg(torch.atan2(-y, x)).abs()
        inwin = (x >= 0) & (x <= WIN_X) & (y.abs() <= WIN_Y) & (bearing <= FOV_DEG + 1e-4)
        ok = (score >= float(score_thr)) & inwin & torch.isfinite(code).all(-1)
        k = max(1, min(int(n), Q, int(ok.sum(1).max()) if T else 1))   # trimmed to the batch's largest valid count
        n = k
        s_sel = torch.where(ok, score, torch.full_like(score, -1.0))
        top = torch.topk(s_sel, k, dim=1)
        qi, valid = top.indices, top.values >= 0
    g = lambda a: torch.gather(a, 1, qi.reshape(T, k, *([1] * (a.dim() - 2))).expand(T, k, *a.shape[2:]))  # noqa
    cg = torch.nan_to_num(code_g, nan=0.0, posinf=0.0, neginf=0.0)      # non-finite rows are not selected
    c0 = torch.stack([g(cg[..., 1]), -g(cg[..., 0])], -1)               # (x_N, y_N) = (y_rec, -x_rec)
    v0 = torch.stack([g(cg[..., 9]), -g(cg[..., 8])], -1)               # (vx_N, vy_N) = (vy_rec, -vx_rec)
    sp0 = torch.linalg.vector_norm(v0, dim=-1)
    u0 = v0 / sp0.clamp(min=1e-6)[..., None]
    with torch.no_grad():
        clsk = g(cls)
        group = torch.as_tensor(GROUP_OF_CLASS, dtype=torch.long, device=dev)[clsk]
    pi_s = torch.where(group == 2, torch.ones_like(sp0), torch.sigmoid((0.1 - sp0) / 0.05))
    with torch.no_grad():
        Lk, Wk = g(L), g(W)
        c0d, sp0d = c0.detach(), sp0.detach()
        rng = torch.linalg.vector_norm(c0d, dim=-1)
        bear = torch.atan2(c0d[..., 1], c0d[..., 0])
        cols = [g(hidden.float()), g(prob), g(score)[..., None], (rng / 32.0)[..., None],
                torch.cos(bear)[..., None], torch.sin(bear)[..., None], (sp0d / 10.0)[..., None],
                (Lk / 5.0)[..., None], (Wk / 3.0)[..., None]]
        if stop_logit is not None:                          # V5Cfg.rel_pstop: p_stop_i of the det head's stop_branch
            cols.append(g(torch.sigmoid(stop_logit.detach().float()))[..., None])   # non-finite rows: not selected
        feats = torch.cat(cols, -1)
    out = dict(c0=c0, v0=v0, h0=g(h), L=Lk, W=Wk, sp0=sp0, u0=u0, acc=torch.zeros_like(sp0),
               has=torch.zeros_like(valid), group=group, pi_s=pi_s, valid=valid, feats=feats,
               q_idx=torch.where(valid, qi, torch.full_like(qi, -1)))
    M = 0
    if traj is not None:                                    # rel_modes 'cv+fut': the det head's futures (detached)
        with torch.no_grad():
            d = torch.nan_to_num(traj.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            fut = torch.cumsum(torch.stack([d[..., 1], -d[..., 0]], -1), 3)            # [T, Q, M, 8, 2] N frame
            M = int(traj.shape[2])
            pf = torch.softmax(traj_cls.detach().float(), -1)
            pf = torch.where(torch.isfinite(pf).all(-1, keepdim=True), pf, torch.full_like(pf, 1.0 / M))
            out["fut"], out["pi_fut"] = g(fut), g(pf)
    if k < n:
        for key, v in out.items():
            pad = torch.zeros(T, n - k, *v.shape[2:], dtype=v.dtype, device=dev)
            if key in ("L", "W"):
                pad = pad + 1.0
            if key == "q_idx":
                pad = pad - 1
            if key == "pi_fut":
                pad = pad + 1.0 / M
            out[key] = torch.cat([v, pad], 1)
    vm = out["valid"]
    for key in ("c0", "v0", "h0", "sp0", "u0", "pi_s", "feats"):
        out[key] = torch.where(vm.reshape(*vm.shape, *([1] * (out[key].dim() - 2))), out[key],
                               torch.zeros_like(out[key]))
    out["L"] = torch.where(vm, out["L"], torch.ones_like(out["L"]))
    out["W"] = torch.where(vm, out["W"], torch.ones_like(out["W"]))
    if traj is not None:
        out["fut"] = torch.where(vm[..., None, None, None], out["fut"], torch.zeros_like(out["fut"]))
        out["pi_fut"] = torch.where(vm[..., None], out["pi_fut"], torch.full_like(out["pi_fut"], 1.0 / M))
    return out


GRAPH_KEYS = ("c0", "v0", "sp0", "u0", "pi_s")          # the row entries that carry the box code's graph


def rows_keep_graph(O: Dict[str, Tensor], keep: Tensor) -> Dict[str, Tensor]:
    """design V2 'matched boxes only' (online2 v4.v5.rel_det_grad_matched): the rows O with the box code's graph kept on
    the slots keep [T, n] (matched one-to-one to a GT track at t0) and detached on every other slot (ghost / duplicate
    boxes: no relation loss moves their regression); values unchanged."""
    out = dict(O)
    kp = keep.to(O["valid"].device).bool()
    for key in GRAPH_KEYS:
        v = O[key]
        out[key] = torch.where(kp.reshape(*kp.shape, *([1] * (v.dim() - 2))), v, v.detach())
    return out


@torch.no_grad()
def student_reliability_targets(O: Dict[str, Tensor], gt_boxes0: Tensor, gt_vel0: Tensor, gt_obs0: Tensor,
                                gate_m: float = 2.0, v_tol: float = 1.0):
    """a_i targets of the student rows against the GT objects at t0 (GT = target only): O (student_object_rows of n
    tokens), gt_boxes0 [n, A, 2] centres, gt_vel0 [n, A, 2], gt_obs0 [n, A] -> y, m [n, N]: y = a GT object within
    gate_m of the row centre whose velocity agrees within v_tol m/s (nearest one); m = valid rows.
    v_tol = 1.0 m/s is a DECLARED DEVIATION from design §3.3's 0.5 m/s (main-session decision D5, 2026-10-20): the
    teachers' a_i^T (tools/ck/train_ck2.py) is trained with 1.0, and its H feeds the student's Z-KD (arm D), so the
    student reliability keeps the same tolerance."""
    d = torch.cdist(O["c0"].float(), gt_boxes0.float())                                   # [n, N, A]
    d = torch.where(gt_obs0[:, None, :], d, torch.full_like(d, float("inf")))
    dmin, j = d.min(-1)
    vg = torch.gather(gt_vel0.float(), 1, j[..., None].expand(*j.shape, 2))
    err = torch.linalg.vector_norm(O["v0"].float() - vg, dim=-1)
    y = ((dmin <= gate_m) & (err < v_tol)).float()
    return y, O["valid"]
