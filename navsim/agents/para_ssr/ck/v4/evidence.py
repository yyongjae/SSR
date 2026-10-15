"""refiner v4 evidence-decomposed readout (V4Cfg.readout 'evidence', report 77 Q4-2; config.py docstring).

Scene quantities (learned, candidate-free; predicted on the PARENT's station grid by small per-cell MLPs on the corridor
conv features Hc [B, 128, 48, 17] and the station tokens x' [B, 48, d]):
  O [B, 48, 17, M] occupancy logits: another object covers the grid point (station j, lateral offset i) at time bin m
                   (t_m = 0.5 (m + 1) s, M = ev_tbins <= 8);
  D [B, 48, 17]    drivable margin [m]: signed distance of the grid point to the official DAC boundary (+ inside).
Candidate quantities (geometry, no parameters; ck/v4/sibling.on_parent knots on the parent's grid):
  knots k = 1..8 (t = 0.5 k s): arc length s_k, lateral offset d_k, speed v_k = (s_k - s_{k-1}) / 0.5.
  Ego box (geometry.HALF_LEN 2.588, HALF_WID 1.1485, centre RA2C 1.461 m ahead of the rear axle) approximated in the
  parent's path frame (heading relative to the path ignored): 9 footprint points (3 along x 3 across), 4 corners.
Evidence (the official rules, differentiable).  Points of one footprint (and the TTC deltas of one knot) see the SAME
objects, so they are combined with a soft maximum of their logits (smax: tau logsumexp(z / tau), tau 0.5 -- "the
footprint overlaps something"), and the knots with a noisy-OR (log P_pass = sum_k log(1 - p_k)):
  NC   log P = sum_k logsigmoid(-smax_{9 points} O(point, t_k))              (bilinear in (j, i), bin of t_k)
  TTC  log P = sum_k logsigmoid(-smax_{delta in 0.3 / 0.6 / 0.9 s, 9 points} O(point + v_k delta, t_k + delta))
       (linear in time between bins; beyond the last bin: the last bin)
  DAC  log P = sum_k logsigmoid(-smax_{4 corners} (-D(corner) / eps)), eps = exp(log_eps) learned (init ev_eps)
  Points outside the grid (station range or |d| > 4.8 m) are not read (their term is 0): "unknown" is not "free"
  in the scorer, but the parent grid cannot say more; the residual can learn the out-of-grid share.
Score: logit_k = logit(P_k) + residual_k([log P_nc, log P_ttc, log P_dac, min corner margin, g_c]) (zero-initialised
  MLP, ev_resid), EP / comfort heads as the hazard readout, lateral head on [g_c, margin-weighted station features].
Outputs also O, D (aux losses: evidence_aux in ck/v4/losses.py) and per-candidate reasons (argmax point / time)."""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...refiner.corridor import LAT_MAX, N_LAT, N_ST
from ...refiner.decoder import N_FREE
from ...refiner.geometry import HALF_LEN, HALF_WID, RA2C
from ..constants import CK_KEYS
from .config import V4Cfg
from .readout import EP_MIN_DIST, HC_CH, I_C, I_DAC, I_EP, I_NC, I_TTC, LAT_STEP, _mlp, lateral_read

TTC_DELTAS = (0.3, 0.6, 0.9)
T_BIN = 0.5
LOGP_MAX = -1e-6
_ALONG = (-HALF_LEN, 0.0, HALF_LEN)
_ACROSS = (-HALF_WID, 0.0, HALF_WID)
FOOT = tuple((a, c) for a in _ALONG for c in _ACROSS)                     # 9 (along, across) offsets of the centre
CORNERS = tuple((a, c) for a in (-HALF_LEN, HALF_LEN) for c in (-HALF_WID, HALF_WID))


def grid_read(G: torch.Tensor, s_st: torch.Tensor, s: torch.Tensor, d: torch.Tensor):
    """Bilinear read of a per-candidate station grid G [B, 48, 17, C] at path points (s [B, Q] arc length on the
    parent's path, d [B, Q] lateral offset) -> (values [B, Q, C], valid [B, Q]).  Stations s_st [B, 48] are uniform
    (s_j = (j + 0.5) ds); lateral offsets d_i = -4.8 + 0.6 i."""
    B, NS, NL, C = G.shape
    ds = (s_st[:, 1] - s_st[:, 0]).clamp(min=1e-3)[:, None]
    us = s / ds - 0.5
    ud = (d + LAT_MAX) / LAT_STEP
    valid = (us >= 0) & (us <= NS - 1) & (ud >= 0) & (ud <= NL - 1) & torch.isfinite(us) & torch.isfinite(ud)
    us = torch.where(valid, us, torch.zeros_like(us))
    ud = torch.where(valid, ud, torch.zeros_like(ud))
    j0 = torch.floor(us.detach()).long().clamp(0, NS - 2)
    i0 = torch.floor(ud.detach()).long().clamp(0, NL - 2)
    a = (us - j0.to(us.dtype)).clamp(0, 1)[..., None]
    b = (ud - i0.to(ud.dtype)).clamp(0, 1)[..., None]
    flat = G.reshape(B, NS * NL, C)
    idx = j0 * NL + i0
    g = lambda k: torch.gather(flat, 1, k[..., None].expand(*k.shape, C))
    v = (g(idx) * (1 - a) * (1 - b) + g(idx + NL) * a * (1 - b) + g(idx + 1) * (1 - a) * b
         + g(idx + NL + 1) * a * b)
    return v * valid[..., None].to(v.dtype), valid


SMAX_TAU = 0.5
NO_READ = -20.0                     # logit of a knot without any readable point (p ~ 2e-9)


def smax(z: torch.Tensor, valid: torch.Tensor, tau: float = SMAX_TAU) -> torch.Tensor:
    """Soft maximum over the last axis of logits z [..., P] restricted to valid [..., P]: tau logsumexp(z / tau) (an
    upper bound of the max, <= max + tau ln P); no valid entry -> NO_READ."""
    zz = torch.where(valid, z, torch.full_like(z, -1e4))
    out = tau * torch.logsumexp(zz / tau, -1)
    return torch.where(valid.any(-1), out, torch.full_like(out, NO_READ))


def _logit_of_logp(lp: torch.Tensor) -> torch.Tensor:
    lp = lp.clamp(max=LOGP_MAX)
    return lp - torch.log(-torch.expm1(lp))


class EvidenceReadout(nn.Module):
    """V4Cfg.readout == 'evidence' heads (module docstring).  Built under the caller's seeded RNG."""

    def __init__(self, cfg: V4Cfg, d_model: int):
        super().__init__()
        self.cfg = cfg
        d = d_model
        M = int(cfg.ev_tbins)
        self.occ_head = _mlp(HC_CH + d, 128, M)
        self.marg_head = _mlp(HC_CH + d, 128, 1)
        self.log_eps = nn.Parameter(torch.tensor(math.log(float(cfg.ev_eps))))
        self.resid = _mlp(4 + d, 128, 3, zero_last=True) if cfg.ev_resid else None
        self.ep_head = _mlp(3 * d + 3, int(cfg.ep_hidden), 1)
        self.comfort_head = _mlp(d, 128, 1)
        self.lat_head = _mlp(2 * d + HC_CH, 256, N_FREE, zero_last=True)

    @torch.no_grad()
    def set_prior(self, prior: Sequence[float]) -> None:
        """Occupancy logits biased so that 8 knots give the NC pass prior (per-knot p = 1 - p_nc^(1/8), minus the soft-max
        lift tau ln 9), margins to +2 m; EP / comfort heads logit(prior); the NC / TTC / DAC pass rates then follow from
        the evidence (no per-key bias: the residual starts at 0)."""
        p = np.clip(np.asarray(prior, np.float64), 1e-4, 1 - 1e-4)
        q = 1.0 - p[I_NC] ** (1.0 / 8.0)
        self.occ_head[-1].bias.fill_(math.log(q / (1 - q)) - SMAX_TAU * math.log(len(FOOT)))
        self.marg_head[-1].bias.fill_(2.0)
        for head, i in ((self.ep_head, I_EP), (self.comfort_head, I_C)):
            head[-1].bias.fill_(math.log(p[i] / (1 - p[i])))

    def scene(self, xs: torch.Tensor, Hc: torch.Tensor):
        """-> O [B, 48, 17, M] occupancy logits (f32), D [B, 48, 17] margin [m] (f32)."""
        B = xs.shape[0]
        z = torch.cat([Hc.permute(0, 2, 3, 1), xs[:, :, None, :].expand(B, N_ST, N_LAT, xs.shape[-1]).to(Hc.dtype)],
                      -1)
        return self.occ_head(z).float(), self.marg_head(z)[..., 0].float()

    def forward(self, xs: torch.Tensor, xd: torch.Tensor, Hc: torch.Tensor, g: torch.Tensor, s_st: torch.Tensor,
                s_k: torch.Tensor, d_k: torch.Tensor, d_st: torch.Tensor, t_st: torch.Tensor, S8: torch.Tensor,
                pref: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """xs [B, 48, d], xd [B, d], Hc [B, 128, 48, 17] (parent pass), g [B, d] (candidate draft embedding),
        s_st [B, 48] parent station arc lengths, s_k / d_k [B, 9] candidate knots on the parent's grid, d_st / t_st
        [B, 48] candidate lateral offset / arrival time at the stations, S8 [B], pref [B] log1p P_ref | None."""
        B = xs.shape[0]
        dt = xs.dtype
        O, D = self.scene(xs, Hc)
        M = O.shape[-1]
        s1, d1 = s_k[:, 1:].float(), d_k[:, 1:].float()                          # knots k = 1..8 [B, 8]
        v1 = ((s_k[:, 1:] - s_k[:, :-1]) / T_BIN).float().clamp(min=0)
        tk = torch.arange(1, 9, device=xs.device, dtype=torch.float32) * T_BIN
        kk = torch.arange(8, device=xs.device).clamp(max=M - 1)                   # bin of t_k
        sst = s_st.float()

        def foot(s_c, d_c, offs):
            """[B, 8] centres -> points s, d [B, 8 * P] (knot-major)."""
            a = torch.tensor([o[0] for o in offs], device=xs.device)
            c = torch.tensor([o[1] for o in offs], device=xs.device)
            return ((s_c + RA2C)[..., None] + a).reshape(B, -1), (d_c[..., None] + c).reshape(B, -1)

        # NC: footprint at t_k, occupancy of bin k
        ps, pd = foot(s1, d1, FOOT)
        vals, val = grid_read(O, sst, ps, pd)                                     # [B, 72, M]
        nf = len(FOOT)
        bin_nc = kk.repeat_interleave(nf)[None, :, None].expand(B, -1, 1)
        o_nc = torch.gather(vals, 2, bin_nc)[..., 0].reshape(B, 8, nf)
        z_nc = smax(o_nc, val.reshape(B, 8, nf))                                   # [B, 8] per knot
        lp_nc = F.logsigmoid(-z_nc).sum(-1)
        # TTC: projected footprint at t_k + delta (linear in time between bins), soft max over deltas x points
        zt, vt = [], []
        for dl in TTC_DELTAS:
            ps2, pd2 = foot(s1 + v1 * dl, d1, FOOT)
            v2, ok2 = grid_read(O, sst, ps2, pd2)
            tb = ((tk + dl) / T_BIN - 1.0).clamp(0, M - 1)                        # fractional bin
            b0 = torch.floor(tb).long().clamp(max=max(M - 2, 0))
            w = (tb - b0.float()).clamp(0, 1)
            b0e = b0.repeat_interleave(len(FOOT))[None, :, None].expand(B, -1, 1)
            b1e = (b0e + 1).clamp(max=M - 1)
            we = w.repeat_interleave(len(FOOT))[None, :]
            o2 = torch.gather(v2, 2, b0e)[..., 0] * (1 - we) + torch.gather(v2, 2, b1e)[..., 0] * we
            zt.append(o2.reshape(B, 8, nf))
            vt.append(ok2.reshape(B, 8, nf))
        z_ttc = smax(torch.cat(zt, -1), torch.cat(vt, -1))
        lp_ttc = F.logsigmoid(-z_ttc).sum(-1)
        # DAC: the 4 corners at every knot (soft max of the corners' out-of-drivable logits -D / eps)
        cs, cd = foot(s1, d1, CORNERS)
        dv, okc = grid_read(D[..., None], sst, cs, cd)
        eps = torch.exp(self.log_eps).clamp(min=0.01)
        z_dac = smax((-dv[..., 0] / eps).reshape(B, 8, len(CORNERS)), okc.reshape(B, 8, len(CORNERS)))
        lp_dac = F.logsigmoid(-z_dac).sum(-1)
        min_marg = torch.where(okc, dv[..., 0], torch.full_like(dv[..., 0], 10.0)).min(-1).values.clamp(max=10.0)
        logit = torch.zeros(B, len(CK_KEYS), dtype=torch.float32, device=xs.device)
        logit[:, I_NC] = _logit_of_logp(lp_nc)
        logit[:, I_TTC] = _logit_of_logp(lp_ttc)
        logit[:, I_DAC] = _logit_of_logp(lp_dac)
        if self.resid is not None:
            r_in = torch.cat([torch.stack([lp_nc / 5.0, lp_ttc / 5.0, lp_dac / 5.0, min_marg / 5.0], -1).to(dt), g], -1)
            r = self.resid(r_in).float()
            logit[:, I_NC] = logit[:, I_NC] + r[:, 0]
            logit[:, I_TTC] = logit[:, I_TTC] + r[:, 1]
            logit[:, I_DAC] = logit[:, I_DAC] + r[:, 2]
        # EP / comfort (as the hazard readout)
        m = (t_st <= float(self.cfg.horizon)) & torch.isfinite(t_st)
        mf = m.to(dt)[..., None]
        xs_mean = (xs * mf).sum(1) / mf.sum(1).clamp(min=1.0)
        S = S8.float().clamp(min=0.0)
        if pref is None:
            lp_ref, ratio = torch.zeros_like(S), torch.ones_like(S)
        else:
            lp_ref = pref.detach().float()
            P = torch.expm1(lp_ref).clamp(min=0.0)
            ratio = S / torch.maximum(torch.maximum(P, S), torch.full_like(S, EP_MIN_DIST))
        ep_in = torch.cat([g, xd, xs_mean, torch.stack([torch.log1p(S) / 4.0, lp_ref / 4.0, ratio], -1).to(dt)], -1)
        logit[:, I_EP] = self.ep_head(ep_in)[:, 0].float()
        logit[:, I_C] = self.comfort_head(g)[:, 0].float()
        # lateral head: stations weighted by a low predicted margin along the candidate's path
        dpath = lateral_read(D[:, None], d_st)[..., 0]                             # [B, 48]
        w = torch.where(m, -dpath.detach() / eps.detach(), torch.full_like(dpath, -1e4))
        w = torch.softmax(w, -1)
        w = torch.where(m.any(-1, keepdim=True), w, torch.zeros_like(w)).to(dt)
        f = lateral_read(Hc, d_st).to(dt)
        w_lat = self.lat_head(torch.cat([g, (w[..., None] * xs).sum(1), (w[..., None] * f).sum(1)], -1)).float()
        return {"score_logit": logit, "w_lat": w_lat, "occ_logit": O, "margin": D, "mask": m,
                "lp_nc": lp_nc, "lp_ttc": lp_ttc, "lp_dac": lp_dac, "min_margin": min_marg}
