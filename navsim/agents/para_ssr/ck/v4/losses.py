"""refiner v4 losses (BCE-family only; no ranking / margin terms).

station_bce  censored station BCE of the hazard readout (report 70 P1 M7, 72 I1), per key k in (nc, ttc, dac) (or the
  `keys` subset: a specialist teacher only gets the station terms of its own score keys):
  inputs  a [B, K, 48] station hazard logits, t_in / t_out [B, K, 48] the candidate's time interval in the parent's
          station bins (entry, min(leave, horizon)), m [B, K, 48] stations read, tidx [B, K] first-failure time index
          of the official scorer (n = 0..40 -> tau = 0.1 n s on the 41-state LQR grid; -1 no event; -2 unknown / not
          available), y [B, K] the key's official label (fail iff y < 1), ok [B, K] candidate ok.
  targets failing candidate (y < 1, tidx >= 0): station j* = the read station whose interval contains tau (else the
          nearest interval) -> positive (a stopped / slow candidate's failure lands on the bin it dwells in);
          stations left before tau - before_s -> negative; j* +- 1 and later stations ignored (censored: after a
          failure the trajectory is not evaluated by the metric any more);
          passing candidate (y >= 1, tidx == -1): every read station negative;
          tidx == -2, y / tidx inconsistent, or no read station: no station term.
  (haz dicts without t_in / t_out, e.g. older callers: t_in = t_out = t_arr, the bin-centre arrival time.)
  value   mean BCE over the weighted (candidate, station) entries of each key, then the mean over the keys present.
  TTC note: the scorer's TTC index is the ego time of the flagged step (the contact may be up to 0.9 s later), so the
  TTC positive marks where the ego is when the infraction is flagged.
pref_loss    readout.pref_loss (Huber on log1p P_ref)."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..constants import CK_KEYS
from .readout import pref_loss  # noqa: F401  (re-export)

TIDX_COLS = ("nc", "ttc", "dac")          # = ck/time_idx.TIDX_COLS (column order of the time-index sidecars)
TIDX_DT = 0.1                             # s per index (official scorer, 41-state LQR grid)
HAZ_KEYS = (("nc", "a_nc"), ("ttc", "a_ttc"), ("dac", "a_dac"))


def _jstar(t_in: torch.Tensor, t_out: torch.Tensor, m: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """[.., 48] intervals, m read, tau [..] -> index of the read interval [t_in, t_out) containing tau, else the nearest
    interval (distance to it; ties -> the lower index; zero-length intervals t_in == t_out = the old nearest arrival
    time) [..]."""
    tt = tau[..., None]
    dist = (t_in - tt).clamp(min=0) + (tt - t_out).clamp(min=0)
    dist = torch.where((t_in <= tt) & (tt < t_out), torch.full_like(dist, -1.0), dist)
    return torch.where(m, dist, torch.full_like(dist, 1e6)).argmin(-1)


def _intervals(haz: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if "t_in" in haz and "t_out" in haz:
        return haz["t_in"].float(), haz["t_out"].float()
    t = haz["t_arr"].float()
    return t, t


def station_targets(t_in: torch.Tensor, t_out: torch.Tensor, m: torch.Tensor, tidx: torch.Tensor, y: torch.Tensor,
                    ok: torch.Tensor, before_s: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (target [B, K, 48] in {0, 1}, weight [B, K, 48] in {0, 1}) for one key (module docstring)."""
    m = m.bool() & ok.bool()[..., None]
    ti = tidx.long()
    fail = (y < 1.0) & (ti >= 0)
    pas = (y >= 1.0) & (ti == -1)
    tau = ti.clamp(min=0).to(t_in.dtype) * TIDX_DT
    jstar = _jstar(t_in, t_out, m, tau)                                           # [B, K]
    has = torch.gather(m, -1, jstar[..., None])[..., 0]
    j = torch.arange(t_in.shape[-1], device=t_in.device).view(1, 1, -1)
    pos = (j == jstar[..., None]) & (fail & has)[..., None]
    near = ((j - jstar[..., None]).abs() <= 1) & ~pos
    neg_f = m & (t_out < (tau - before_s)[..., None]) & ~near & (fail & has)[..., None]
    neg_p = m & pas[..., None]
    target = pos.to(t_in.dtype)
    weight = (pos | neg_f | neg_p).to(t_in.dtype)
    return target, weight


def station_bce(haz: Dict[str, torch.Tensor], tidx: torch.Tensor, y5: torch.Tensor, ok: torch.Tensor,
                before_s: float = 0.25, keys=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """haz = score_set / forward 'haz' dict ([B, K, 48] a_nc, a_ttc, a_dac, haz_mask, t_in, t_out (or t_arr)); tidx
    [B, K, 3] int (TIDX_COLS order); y5 [B, K, 5] CK targets (CK_KEYS order); ok [B, K]; keys: the hazard keys to
    supervise (None = nc, ttc, dac) -> (loss, stats).  0 (graph-connected) when no key has a station term."""
    t_in, t_out = _intervals(haz)
    m = haz["haz_mask"].bool()
    terms, st = [], {}
    for c, (key, ak) in enumerate(HAZ_KEYS):
        if keys is not None and key not in keys:
            continue
        a = haz[ak].float()
        tg, w = station_targets(t_in, t_out, m, tidx[..., c], y5[..., CK_KEYS.index(key)].float(), ok, before_s)
        n = w.sum()
        st[f"st_n_{key}"] = float(n.detach())
        st[f"st_npos_{key}"] = float((tg * w).sum().detach())
        if float(n) > 0:
            l = (F.binary_cross_entropy_with_logits(a, tg, reduction="none") * w).sum() / n
            terms.append(l)
            st[f"st_bce_{key}"] = float(l.detach())
    if not terms:
        return sum(haz[ak].float().sum() for _, ak in HAZ_KEYS) * 0.0, st
    loss = torch.stack(terms).mean()
    st["st_bce"] = float(loss.detach())
    return loss, st


def localisation_hit(haz: Dict[str, torch.Tensor], tidx: torch.Tensor, y5: torch.Tensor,
                     ok: torch.Tensor) -> Dict[str, float]:
    """Diagnostic (report 70 P1 (i)): for failing candidates, fraction whose argmax hazard station is the labelled
    failure station (|j - j*| <= 1) per key; the chance level is 3 / (number of read stations)."""
    t_in, t_out = _intervals(haz)
    m = haz["haz_mask"].bool() & ok.bool()[..., None]
    out = {}
    for c, (key, ak) in enumerate(HAZ_KEYS):
        ti = tidx[..., c].long()
        fail = (y5[..., CK_KEYS.index(key)] < 1.0) & (ti >= 0) & m.any(-1)
        if not bool(fail.any()):
            continue
        tau = ti.clamp(min=0).float() * TIDX_DT
        jstar = _jstar(t_in, t_out, m, tau)
        a = torch.where(m, haz[ak].float(), torch.full_like(t_in, -1e9))
        jhat = a.argmax(-1)
        hit = ((jhat - jstar).abs() <= 1) & fail
        out[f"loc_hit_{key}"] = float(hit.sum()) / float(fail.sum())
        out[f"loc_chance_{key}"] = float((3.0 / m.sum(-1).clamp(min=1).float())[fail].mean())
    return out


# ----------------------------------------------------------------------------------------------- evidence aux
EV_REF_KEYS = ("ref_obj_kf", "ref_obj_first", "ref_obj_meta", "ref_obj_n", "ref_obj_n_kf", "ref_sdf", "ref_gt_ok")


def _points_in_boxes(p: torch.Tensor, boxes: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """p [N, Q, 2], boxes [N, A, M, 5] (cx, cy, h, L, W), obs [N, A, M] bool -> occupied [N, Q, M] (any OBS box
    contains the point; box edges inclusive)."""
    c = boxes[..., :2][:, :, :, None, :]                                            # [N, A, M, 1, 2]
    h = boxes[..., 2][..., None]
    rel = p[:, None, None, :, :] - c                                               # [N, A, M, Q, 2]
    ch, sh = torch.cos(h), torch.sin(h)
    x = rel[..., 0] * ch + rel[..., 1] * sh
    y = -rel[..., 0] * sh + rel[..., 1] * ch
    inside = (x.abs() <= 0.5 * boxes[..., 3][..., None]) & (y.abs() <= 0.5 * boxes[..., 4][..., None])
    inside = inside & obs[..., None]
    return inside.any(1).permute(0, 2, 1)                                          # [N, Q, M]


def evidence_aux(haz: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor], max_parents: int = 4,
                 margin_clip: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Auxiliary targets of the evidence readout on the parent grid points (GT-ok tokens only, the candidates that
    are their own parents, at most max_parents per token):
      occupancy: GT future objects (gt_future.query_torch, metric-cache rule; OBS only) cover the grid point at
                 t_m = 0.5 (m + 1) s (m < M) -> BCE with the occ logits (times after the token's annotated span are
                 masked);
      margin:    the official DAC SDF (ref_sdf, sdf.sample_sdf) at the grid point, clipped to +-margin_clip m ->
                 Huber (delta 1 m) with the predicted margin (points outside the SDF grid masked).
    haz: score_set 'haz' of the evidence readout ([T, K, ...]); ref: the batch's ref_* GT tensors ([T, ...]).
    -> (occ_loss, margin_loss, stats); zeros (graph-connected) when nothing is selected."""
    from ...refiner import gt_future as GF
    from ...refiner.sdf import sample_sdf
    O, D, pts = haz["occ_logit"], haz["margin"], haz["pts"]
    T, K = O.shape[:2]
    M = O.shape[-1]
    zero = O.sum() * 0.0 + D.sum() * 0.0
    if any(k not in ref for k in EV_REF_KEYS):
        return zero, zero, {"ev_n": 0.0}
    gt_ok = ref["ref_gt_ok"].reshape(-1).bool().to(O.device)
    sel = haz["is_self"].bool() & gt_ok[:, None]
    rank = torch.cumsum(sel.long(), 1)
    sel = sel & (rank <= int(max_parents))
    tok, col = torch.nonzero(sel, as_tuple=True)
    if tok.numel() == 0:
        return zero, zero, {"ev_n": 0.0}
    dev = O.device
    n_obj = int(max(int(ref["ref_obj_n"][tok].max()), 1))
    kf = ref["ref_obj_kf"][tok][:, :n_obj].to(dev)
    first = ref["ref_obj_first"][tok][:, :n_obj].to(dev)
    meta = ref["ref_obj_meta"][tok][:, :n_obj].to(dev).long()
    valid = torch.arange(n_obj, device=dev)[None] < ref["ref_obj_n"][tok].to(dev).long()[:, None]
    tm = (torch.arange(M, device=dev, dtype=torch.float64) + 1) * 0.5
    with torch.no_grad():
        boxes, st = GF.query_torch(kf, first, meta, tm, valid=valid, n_kf=ref["ref_obj_n_kf"][tok].to(dev))
        obs = st.bool() & (st.to(torch.int64) == 1)
        p = pts[tok, col].reshape(len(tok), -1, 2).to(torch.float64)                # [N, 816, 2]
        occ = _points_in_boxes(p, boxes.to(torch.float64), obs).float()            # [N, 816, M]
        t_av = ((ref["ref_obj_n_kf"][tok].to(dev).double() - 1) * 0.5).clamp(max=5.0)
        known = (tm[None] <= t_av[:, None] + 1e-6)[:, None, :].expand_as(occ)
        sv, sval = sample_sdf(ref["ref_sdf"][tok].to(dev), p.float())
    ol = O[tok, col].reshape(len(tok), -1, M)
    l_occ = (F.binary_cross_entropy_with_logits(ol, occ, reduction="none") * known.float()).sum() / \
        known.float().sum().clamp(min=1.0)
    dm = D[tok, col].reshape(len(tok), -1)
    tgt = sv.float().clamp(-margin_clip, margin_clip)
    l_m = (F.huber_loss(dm, tgt, delta=1.0, reduction="none") * sval.float()).sum() / sval.float().sum().clamp(min=1.0)
    st_ = {"ev_n": float(len(tok)), "ev_occ_rate": float((occ * known).sum() / known.float().sum().clamp(min=1.0)),
           "ev_occ_bce": float(l_occ.detach()), "ev_margin_huber": float(l_m.detach()),
           "ev_sdf_valid": float(sval.float().mean())}
    return l_occ, l_m, st_
