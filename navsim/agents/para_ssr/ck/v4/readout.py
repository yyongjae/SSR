"""refiner v4 heads (config.py docstring): HazardReadout (I1 station x arrival-time hazard + noisy-OR, structured EP
head, scene-free comfort head, hazard-localised lateral head) and PRefHead (supervised P_ref scene query).

Shapes (B = candidates, flattened over tokens; every per-candidate tensor is already aligned with its parent's pass):
  xs   [B, 48, d]        the parent's station tokens after the 4 layers (+ norm_out)
  xd   [B, d]            the parent's draft token after the 4 layers
  Hc   [B, 128, 48, 17]  the parent's corridor conv features (lateral axis kept)
  g    [B, d]            draft_mlp(u_d of the CANDIDATE) (kinematics only, no scene)
  t, v, d [B, 48]        the candidate's arrival time [s], speed [m/s], lateral offset [m] at the parent's stations
  S8   [B]               the candidate's draft length [m];  pref [B] log1p(P_ref) per candidate's token or None
Returns score_logit [B, 5] (CK_KEYS nc, dac, ep, ttc, comfort; f32), a_nc / a_ttc / a_dac [B, 48] (station hazard
logits, f32), mask [B, 48] (stations read), w_lat [B, 6] (raw decoder controls).
Numerics: log P_pass = sum_j m_j logsigmoid(-a_j) <= 0, clamped <= -1e-6; logit(P) = log P - log(-expm1(log P))."""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...refiner.corridor import LAT_MAX, N_LAT, N_ST
from ...refiner.decoder import N_FREE
from ..constants import CK_KEYS
from .config import V4Cfg

HC_CH = 128
I_NC, I_DAC, I_EP, I_TTC, I_C = (CK_KEYS.index(k) for k in ("nc", "dac", "ep", "ttc", "comfort"))
LAT_STEP = 2 * LAT_MAX / (N_LAT - 1)            # 0.6 m
EP_MIN_DIST = 5.0                                # progress_distance_threshold (ep_target.EP_DIST_THRESHOLD)
LOGP_MAX = -1e-6


def time_embed(t: torch.Tensor, n_freq: int) -> torch.Tensor:
    """t [...] seconds -> [..., 1 + 2 n_freq]: t / 4, sin / cos(2 pi m t / 8), m = 1..n_freq (t clamped to [0, 8])."""
    t = t.clamp(0.0, 8.0)
    out = [t[..., None] / 4.0]
    for m in range(1, n_freq + 1):
        w = 2 * math.pi * m / 8.0
        out += [torch.sin(w * t)[..., None], torch.cos(w * t)[..., None]]
    return torch.cat(out, -1)


def lateral_read(Hc: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Hc [B, C, 48, 17], d [B, 48] (m, left +) -> [B, 48, C]: linear interpolation between the two lateral offsets
    around d (clamped to +-4.8 m)."""
    B, C = Hc.shape[:2]
    u = ((d.clamp(-LAT_MAX, LAT_MAX) + LAT_MAX) / LAT_STEP).to(Hc.dtype)
    i0 = torch.floor(u).long().clamp(0, N_LAT - 2)
    w = (u - i0.to(u.dtype)).clamp(0.0, 1.0)
    H = Hc.permute(0, 2, 3, 1)                                                     # [B, 48, 17, C]
    f0 = torch.gather(H, 2, i0[..., None, None].expand(B, N_ST, 1, C))[:, :, 0]
    f1 = torch.gather(H, 2, (i0 + 1)[..., None, None].expand(B, N_ST, 1, C))[:, :, 0]
    return f0 + w[..., None] * (f1 - f0)


def _mlp(i: int, h: int, o: int, zero_last: bool = False) -> nn.Sequential:
    m = nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))
    if zero_last:
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
    return m


def noisy_or_logit(a: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """a [B, 48] station hazard logits, m [B, 48] bool -> logit(P_pass) [B] with P_pass = prod_j (1 - sigmoid(a_j))^m_j."""
    lp = (F.logsigmoid(-a.float()) * m.float()).sum(-1).clamp(max=LOGP_MAX)
    return lp - torch.log(-torch.expm1(lp))


class HazardReadout(nn.Module):
    """V4Cfg.readout == 'hazard' heads (module docstring).  Built under the caller's seeded RNG."""

    def __init__(self, cfg: V4Cfg, d_model: int):
        super().__init__()
        self.cfg = cfg
        d = d_model
        nt = 1 + 2 * int(cfg.n_freq)
        hh = int(cfg.haz_hidden)
        self.h_nc = _mlp(d + HC_CH + nt + 1, hh, 1)
        self.h_ttc = _mlp(d + HC_CH + nt + 1, hh, 1)
        self.h_dac = _mlp(d + HC_CH + 1, hh, 1)
        self.ep_head = _mlp(3 * d + 3, int(cfg.ep_hidden), 1)
        self.comfort_head = _mlp(d, 128, 1)
        self.lat_head = _mlp(2 * d + HC_CH if cfg.lat_haz else 2 * d, 256, N_FREE, zero_last=True)

    @torch.no_grad()
    def set_prior(self, prior: Sequence[float]) -> None:
        """Output biases from per-key pass rates (CK_KEYS order): station hazards so that a candidate reading
        prior_n_eff stations has P_pass = prior; EP / comfort heads logit(prior)."""
        p = np.clip(np.asarray(prior, np.float64), 1e-4, 1 - 1e-4)
        n = float(self.cfg.prior_n_eff)
        for head, i in ((self.h_nc, I_NC), (self.h_ttc, I_TTC), (self.h_dac, I_DAC)):
            q = 1.0 - p[i] ** (1.0 / n)
            head[-1].bias.fill_(math.log(q / (1 - q)))
        for head, i in ((self.ep_head, I_EP), (self.comfort_head, I_C)):
            head[-1].bias.fill_(math.log(p[i] / (1 - p[i])))

    def forward(self, xs: torch.Tensor, xd: torch.Tensor, Hc: torch.Tensor, g: torch.Tensor, t: torch.Tensor,
                v: torch.Tensor, d: torch.Tensor, S8: torch.Tensor, pref: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        B = xs.shape[0]
        dt = xs.dtype
        m = (t <= float(self.cfg.horizon)) & torch.isfinite(t)
        f = lateral_read(Hc, d).to(dt)                                             # [B, 48, 128]
        te = time_embed(t.float(), int(self.cfg.n_freq)).to(dt)
        vv = (v.float() / 15.0).clamp(-1.0, 3.0)[..., None].to(dt)
        dd = (d.float() / LAT_MAX).clamp(-2.0, 2.0)[..., None].to(dt)
        z_t = torch.cat([xs, f, te, vv], -1)
        a_nc = self.h_nc(z_t)[..., 0].float()
        a_ttc = self.h_ttc(z_t)[..., 0].float()
        a_dac = self.h_dac(torch.cat([xs, f, dd], -1))[..., 0].float()
        mf = m.to(dt)[..., None]
        xs_mean = (xs * mf).sum(1) / mf.sum(1).clamp(min=1.0)
        S = S8.float().clamp(min=0.0)
        if pref is None:
            lp = torch.zeros_like(S)
            ratio = torch.ones_like(S)
        else:
            lp = pref.detach().float()
            P = torch.expm1(lp).clamp(min=0.0)
            ratio = S / torch.maximum(torch.maximum(P, S), torch.full_like(S, EP_MIN_DIST))
        ep_in = torch.cat([g, xd, xs_mean, torch.stack([torch.log1p(S) / 4.0, lp / 4.0, ratio], -1).to(dt)], -1)
        z_ep = self.ep_head(ep_in)[:, 0].float()
        z_c = self.comfort_head(g)[:, 0].float()
        logit = torch.zeros(B, len(CK_KEYS), dtype=torch.float32, device=xs.device)
        logit[:, I_NC] = noisy_or_logit(a_nc, m)
        logit[:, I_TTC] = noisy_or_logit(a_ttc, m)
        logit[:, I_DAC] = noisy_or_logit(a_dac, m)
        logit[:, I_EP] = z_ep
        logit[:, I_C] = z_c
        if self.cfg.lat_haz:
            w = torch.where(m, a_dac.detach(), torch.full_like(a_dac, -1e4))
            w = torch.softmax(w, -1).to(dt)
            w = torch.where(m.any(-1, keepdim=True), w, torch.zeros_like(w))
            lat_in = torch.cat([g, (w[..., None] * xs).sum(1), (w[..., None] * f).sum(1)], -1)
        else:
            lat_in = torch.cat([g, xs_mean], -1)
        w_lat = self.lat_head(lat_in).float()
        return {"score_logit": logit, "a_nc": a_nc, "a_ttc": a_ttc, "a_dac": a_dac, "mask": m, "w_lat": w_lat}


class PRefHead(nn.Module):
    """One learned query over the scene tokens mem [T, 200, d] + ego (v0 / 15, a0 / 4, command one-hot) -> log1p(P_ref)
    [T] (P_ref = labels 'pdm_progress_eff', metres)."""

    def __init__(self, d_model: int, n_heads: int = 6):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = _mlp(d_model + 6, 128, 1)
        with torch.no_grad():
            self.out[-1].bias.fill_(math.log1p(20.0))     # ~ median P_ref (val: 23.8 m)

    def forward(self, mem: torch.Tensor, v0: torch.Tensor, a0: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        T = mem.shape[0]
        kv = self.ln(mem)
        q = self.q.expand(T, 1, -1).to(kv.dtype)
        z = self.attn(q, kv, kv, need_weights=False)[0][:, 0]
        oh = F.one_hot(cmd.reshape(T).long().clamp(0, 3), 4).to(z.dtype)
        e = torch.cat([(v0.reshape(T).float() / 15.0).to(z.dtype)[:, None], (a0.reshape(T).float() / 4.0).to(z.dtype)
                       [:, None], oh], -1)
        return self.out(torch.cat([z, e], -1))[:, 0].float()


def pref_target(pref_m: torch.Tensor) -> torch.Tensor:
    """P_ref [m] -> regression target log1p(max(P_ref, 0)) (NaN stays NaN = masked by callers)."""
    return torch.log1p(pref_m.float().clamp(min=0.0))


def pref_loss(pred: torch.Tensor, pref_m: torch.Tensor, ok: Optional[torch.Tensor] = None,
              delta: float = 0.2) -> torch.Tensor:
    """Huber(log1p P_ref) over finite targets (& ok); 0 (graph-connected) when there is none."""
    y = pref_target(pref_m)
    m = torch.isfinite(y)
    if ok is not None:
        m = m & ok.bool()
    if not bool(m.any()):
        return pred.sum() * 0.0
    return F.huber_loss(pred.float()[m], y[m], delta=delta)
