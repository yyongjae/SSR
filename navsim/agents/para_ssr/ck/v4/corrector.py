"""refiner v4 learned corrector A (reports 61 / 64 / 75): edits the critic's top-K shortlist with 12 decoder controls
(z_lon 6 + w_lat 6, the PPT refiner's control format; decoder mode B: deceleration, limited acceleration and lateral
offsets), the edits are re-scored by the same critic on their source's parent corridor, and the final pick is over
the 96 rule candidates + the K edits.

Corrector (Gaussian policy):  ctx = [g_c, x_draft(parent), masked station mean(parent), critic logits (5), v0 / 15]
  of each shortlisted candidate (detached: the corrector learns its own head, it never moves the critic's features)
  -> MLP -> mu [12], log_sigma [12] (mu zero-initialised = the identity edit, i.e. the rule pool at the start;
  log_sigma init log 0.3, clamped to [log 0.05, log 1.5]).  Inference: u = mu.  Training: u ~ N(mu, sigma) (recorded
  with log pi(u)).
Decode: refiner.decoder.decode(src, z = u[:6], w = u[6:], v0, mode 'B') (u = 0 returns the source; tests).
Credit (report 61 A; official labels only, never the critic score's gradient):
  A_sel(i) = f(S) - f(S \\ i),  f(S) = sum_j softmax(s_j / tau) R_j over the pool + edits (s = the critic's selection
            score, R = official PDMS): positive when the edit is picked AND good, negative when it is a trap;
  A_loc(i) = R_i - R_src(i) (the edit vs its own source);  A = A_sel + lambda_loc A_loc.
AWR loss: L = - sum_i w_i log pi(u_i | ctx_i) / sum_i 1,  w_i = min(exp(A_i / eta), w_max) normalised over the batch
  (no importance ratio: the labels come one generation late; Peng et al. 2019).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ...refiner.decoder import N_FREE, decode
from ..constants import CK_KEYS
from .readout import _mlp

N_U = 2 * N_FREE                                    # 12 controls
LOG_SIG_MIN, LOG_SIG_MAX = math.log(0.05), math.log(1.5)


class Corrector(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, sigma0: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * d_model + len(CK_KEYS) + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, N_U)
        self.log_sig = nn.Linear(hidden, N_U)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_sig.weight)
        nn.init.constant_(self.log_sig.bias, math.log(sigma0))

    def forward(self, ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(ctx)
        return {"mu": self.mu(h).float(), "log_sig": self.log_sig(h).float().clamp(LOG_SIG_MIN, LOG_SIG_MAX)}

    @staticmethod
    def log_prob(mu: torch.Tensor, log_sig: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """diagonal Gaussian log density [...] of u [..., 12]."""
        z = (u.float() - mu) / torch.exp(log_sig)
        return (-0.5 * z * z - log_sig - 0.5 * math.log(2 * math.pi)).sum(-1)

    def act(self, ctx: torch.Tensor, sample: bool = False, generator: Optional[torch.Generator] = None
            ) -> Dict[str, torch.Tensor]:
        """-> mu, log_sig, u [.., 12] (mu if not sample), logp [..] (of u)."""
        o = self(ctx)
        u = o["mu"]
        if sample:
            eps = torch.randn(u.shape, generator=generator, device=u.device, dtype=u.dtype)
            u = (u + torch.exp(o["log_sig"]) * eps).detach()
        o["u"] = u
        o["logp"] = self.log_prob(o["mu"], o["log_sig"], u)
        return o


def critic_select_score(score_logit: torch.Tensor) -> torch.Tensor:
    """The critic-only selection score of the shortlist (select.ck_final with the 'noim' weights, beta 1):
    0.5 log nc + 0.5 log dac + log(5 ttc + 2 comfort + 5 ep) [..] from score_logit [.., 5] (CK_KEYS order)."""
    p = torch.sigmoid(score_logit.float())
    nc, dac, ep, ttc, c = (p[..., CK_KEYS.index(k)] for k in ("nc", "dac", "ep", "ttc", "comfort"))
    e = 1e-6
    return 0.5 * torch.log(nc + e) + 0.5 * torch.log(dac + e) + torch.log(5 * ttc + 2 * c + 5 * ep + e)


def decode_edits(src: torch.Tensor, u: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """src [T, K, 8, 3] (the shortlisted candidates), u [T, K, 12], v0 [T] -> edited trajectories [T, K, 8, 3]
    (decoder mode B, float32, no gradient: the corrector is trained by AWR on official labels)."""
    T, K = src.shape[:2]
    with torch.no_grad(), torch.autocast(device_type=src.device.type, enabled=False):
        v = torch.nan_to_num(v0.float().reshape(T), nan=0.0).repeat_interleave(K)
        uu = u.reshape(T * K, N_U).float()
        dec = decode(src.reshape(T * K, 8, 3).float(), uu[:, :N_FREE], uu[:, N_FREE:], v0=v, mode="B")
    return dec["traj"].reshape(T, K, 8, 3)


def credit(s_all: torch.Tensor, R_all: torch.Tensor, ok_all: torch.Tensor, edit_cols: torch.Tensor,
           src_cols: Optional[torch.Tensor] = None, tau: float = 0.05, lambda_loc: float = 0.5,
           R_src: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """s_all / R_all / ok_all [B, N] (selection scores, official PDMS, label ok of the set + edits), edit_cols [B, K]
    (column of each edit in the same N); the source's PDMS either from src_cols [B, K] (columns in N) or given as
    R_src [B, K] (the source need not be in the set, e.g. the on-policy label set) -> A_sel, A_loc, A [B, K] (0 where
    not ok)."""
    s = torch.where(ok_all, s_all.float(), torch.full_like(s_all.float(), -1e4)) / tau
    R = torch.where(ok_all, R_all.float(), torch.zeros_like(R_all.float()))
    p = torch.softmax(s, -1)
    f_all = (p * R).sum(-1, keepdim=True)                                          # [B, 1]
    pe = torch.gather(p, 1, edit_cols)
    Re = torch.gather(R, 1, edit_cols)
    # f(S \\ i): renormalise without i
    f_wo = (f_all - pe * Re) / (1.0 - pe).clamp(min=1e-6)
    a_sel = f_all - f_wo
    if R_src is not None:
        a_loc = Re - R_src.float()
        ok = torch.gather(ok_all, 1, edit_cols)
    else:
        a_loc = Re - torch.gather(R, 1, src_cols)
        ok = torch.gather(ok_all, 1, edit_cols) & torch.gather(ok_all, 1, src_cols)
    A = torch.where(ok, a_sel + lambda_loc * a_loc, torch.zeros_like(a_sel))
    return {"A_sel": torch.where(ok, a_sel, torch.zeros_like(a_sel)), "A_loc": torch.where(ok, a_loc, torch.zeros_like(a_loc)),
            "A": A, "ok": ok}


def awr_loss(logp: torch.Tensor, A: torch.Tensor, ok: torch.Tensor, eta: float = 0.05,
             w_max: float = 20.0) -> torch.Tensor:
    """- mean_i w_i log pi(u_i), w_i = min(exp(A_i / eta), w_max) normalised to mean 1 over the ok entries."""
    if not bool(ok.any()):
        return logp.sum() * 0.0
    w = torch.exp((A.float() / eta).clamp(max=math.log(w_max))).detach()
    w = torch.where(ok, w, torch.zeros_like(w))
    w = w / w[ok].mean().clamp(min=1e-8)
    return -(w * logp)[ok].mean()
