"""refiner v5 corrector A (V5Cfg.corrector; report 97 §2-7, add.md (4)): tau1 = decode(tau0, mu_lon, w_lat_head + mu_lat)
is the final trajectory (the lateral head is folded in; no gate, no re-selection; 'keep' = mu ~ 0).

Inputs per draft (T tokens x R drafts):
  ctx    [T, R, 3 d]  critic context [g_c, x_draft, masked station mean] (detached, as the v4 corrector A)
  logit  [T, R, 5]    critic score logits (detached)
  v0     [T]          ego speed (/ 15)
  zq     [T, R, Q, C] per-state reads Z (ck/v5/zread.py + relation H; NOT detached: the correction loss trains the
                      reader, add.md (3)), + the state time t / 4 -> per-state MLP -> mean and max over the states
  top    [T, R, 21]   the top-3 related objects (relation channel; absent without it)
Output mu [T, R, 12] = (z_lon 6, w_lat residual 6), zero-initialised (identity edit at the start); log_sig [T, R, 12]
(init log 0.3; for a later AWR stage, unused by L_refine).
decode_grad: refiner.decoder.decode mode 'B' (deceleration + limited acceleration + lateral) WITH gradient (the v4
decode_edits is no_grad: it was AWR-only).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ...refiner.decoder import N_FREE, decode
from ..constants import CK_KEYS

N_U = 2 * N_FREE


class CorrectorV5(nn.Module):
    def __init__(self, d_model: int, n_z: int, n_top: int, hidden: int = 256, z_hidden: int = 32,
                 sigma0: float = 0.3):
        super().__init__()
        self.n_z, self.n_top = int(n_z), int(n_top)
        self.z_enc = nn.Sequential(nn.Linear(self.n_z + 1, z_hidden), nn.GELU(), nn.Linear(z_hidden, z_hidden),
                                   nn.GELU())
        n_in = 3 * d_model + len(CK_KEYS) + 1 + 2 * z_hidden + self.n_top
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, N_U)
        self.log_sig = nn.Linear(hidden, N_U)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_sig.weight)
        nn.init.constant_(self.log_sig.bias, math.log(sigma0))

    def forward(self, ctx: torch.Tensor, logit: torch.Tensor, v0: torch.Tensor, zq: torch.Tensor, tq: torch.Tensor,
                top: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        T, R = ctx.shape[:2]
        Q = zq.shape[2]
        tt = (tq.float().to(zq.device) / 4.0).reshape(1, 1, Q, 1).expand(T, R, Q, 1)
        h = self.z_enc(torch.cat([zq.float(), tt], -1))                                    # [T, R, Q, z_hidden]
        parts = [ctx.float(), logit.float(), (v0.float().reshape(T, 1, 1) / 15.0).expand(T, R, 1),
                 h.mean(2), h.amax(2)]
        if self.n_top:
            parts.append(top.float())
        x = self.net(torch.cat(parts, -1))
        return {"mu": self.mu(x).float(), "log_sig": self.log_sig(x).float().clamp(math.log(0.05), math.log(1.5))}


def decode_grad(src: torch.Tensor, z: torch.Tensor, w: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """src [T, R, 8, 3] drafts (inputs), z / w [T, R, 6] raw controls, v0 [T] -> tau1 [T, R, 8, 3] (decoder mode B,
    float32, differentiable in z / w).  The lateral spline keeps mode A's normalisation length S_L = S8 (lat_len):
    the lateral head w_lat_head was trained through mode A (ck/correct.py), and mode B's default S_L = S8 + L_ext
    turns the same control into another curve (review 98 R1: up to ~1 m at mu = 0) -- with lat_len = S8, mu = 0
    reproduces today's laterally corrected draft exactly (tests/test_v5_review_fixes.py)."""
    from ...refiner.decoder import DraftPath
    T, R = src.shape[:2]
    with torch.autocast(device_type=src.device.type, enabled=False):
        v = torch.nan_to_num(v0.float().reshape(T), nan=0.0).repeat_interleave(R)
        sf = src.reshape(T * R, 8, 3).float()
        S8 = DraftPath(sf).knots()[:, -1]
        dec = decode(sf, z.reshape(T * R, N_FREE).float(), w.reshape(T * R, N_FREE).float(), v0=v, mode="B",
                     lat_len=S8)
    return dec["traj"].reshape(T, R, 8, 3)
