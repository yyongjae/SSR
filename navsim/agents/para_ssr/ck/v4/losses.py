"""refiner v4 losses (BCE-family only; no ranking / margin terms).

station_bce  censored station BCE of the hazard readout (report 70 P1 M7, 72 I1), per key k in (nc, ttc, dac):
  inputs  a [B, K, 48] station hazard logits, t [B, K, 48] the candidate's arrival time at the parent's stations,
          m [B, K, 48] stations read (t <= horizon), tidx [B, K] first-failure time index of the official scorer
          (n = 0..40 -> tau = 0.1 n s on the 41-state LQR grid; -1 no event; -2 unknown / not available), y [B, K] the
          key's official label (fail iff y < 1), ok [B, K] candidate ok.
  targets failing candidate (y < 1, tidx >= 0): station j* = the read station whose arrival time is closest to tau
          -> positive; stations reached before tau - before_s -> negative; j* +- 1 and later stations ignored
          (censored: after a failure the trajectory is not evaluated by the metric any more);
          passing candidate (y >= 1, tidx == -1): every read station negative;
          tidx == -2, y / tidx inconsistent, or no read station: no station term.
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


def station_targets(t: torch.Tensor, m: torch.Tensor, tidx: torch.Tensor, y: torch.Tensor, ok: torch.Tensor,
                    before_s: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (target [B, K, 48] in {0, 1}, weight [B, K, 48] in {0, 1}) for one key (module docstring)."""
    m = m.bool() & ok.bool()[..., None]
    ti = tidx.long()
    fail = (y < 1.0) & (ti >= 0)
    pas = (y >= 1.0) & (ti == -1)
    tau = ti.clamp(min=0).to(t.dtype) * TIDX_DT
    big = torch.full_like(t, 1e6)
    dist = torch.where(m, (t - tau[..., None]).abs(), big)
    jstar = dist.argmin(-1)                                                       # [B, K]
    has = torch.gather(m, -1, jstar[..., None])[..., 0]
    j = torch.arange(t.shape[-1], device=t.device).view(1, 1, -1)
    pos = (j == jstar[..., None]) & (fail & has)[..., None]
    near = ((j - jstar[..., None]).abs() <= 1) & ~pos
    neg_f = m & (t < (tau - before_s)[..., None]) & ~near & (fail & has)[..., None]
    neg_p = m & pas[..., None]
    target = pos.to(t.dtype)
    weight = (pos | neg_f | neg_p).to(t.dtype)
    return target, weight


def station_bce(haz: Dict[str, torch.Tensor], tidx: torch.Tensor, y5: torch.Tensor, ok: torch.Tensor,
                before_s: float = 0.25) -> Tuple[torch.Tensor, Dict[str, float]]:
    """haz = score_set / forward 'haz' dict ([B, K, 48] a_nc, a_ttc, a_dac, haz_mask, t_arr); tidx [B, K, 3] int
    (TIDX_COLS order); y5 [B, K, 5] CK targets (CK_KEYS order); ok [B, K] -> (loss, stats).  0 (graph-connected) when
    no key has a station term."""
    t = haz["t_arr"].float()
    m = haz["haz_mask"].bool()
    terms, st = [], {}
    for c, (key, ak) in enumerate(HAZ_KEYS):
        a = haz[ak].float()
        tg, w = station_targets(t, m, tidx[..., c], y5[..., CK_KEYS.index(key)].float(), ok, before_s)
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
    t = haz["t_arr"].float()
    m = haz["haz_mask"].bool() & ok.bool()[..., None]
    out = {}
    for c, (key, ak) in enumerate(HAZ_KEYS):
        ti = tidx[..., c].long()
        fail = (y5[..., CK_KEYS.index(key)] < 1.0) & (ti >= 0) & m.any(-1)
        if not bool(fail.any()):
            continue
        tau = ti.clamp(min=0).float() * TIDX_DT
        jstar = torch.where(m, (t - tau[..., None]).abs(), torch.full_like(t, 1e6)).argmin(-1)
        a = torch.where(m, haz[ak].float(), torch.full_like(t, -1e9))
        jhat = a.argmax(-1)
        hit = ((jhat - jstar).abs() <= 1) & fail
        out[f"loc_hit_{key}"] = float(hit.sum()) / float(fail.sum())
        out[f"loc_chance_{key}"] = float((3.0 / m.sum(-1).clamp(min=1).float())[fail].mean())
    return out
