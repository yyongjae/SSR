"""km: listwise selector over the CK2 96-candidate pool (cross-candidate attention, trained on official PDMS).

CK2 scores every pool column independently (ck/model.py: one candidate's tokens never attend to another's) and selects
with the hand formula of ck/select2.py (score_c = (1 - beta) v2_final + beta ck_final).  This head reads the frozen CK2
outputs of ALL columns of one scene at once and returns a correction of that score:

    r_c = s0_c + delta_c,    s0 = select2.pool_scores(beta)        (CK2's deployed score, beta 0.5)
    delta = head(Encoder_cross([x_1 .. x_96], allowed))             (self-attention ACROSS the scene's columns)

x_c = [h_c, h_lat_c (CK2 score-head inputs of the pool and of the laterally corrected trajectory), CK2 logits of both,
s0_c, v2_final / log v2_im of the parent, variant type, parent rank, lateral offsets e_lat].  The last layer is zero
initialised, so an untrained head selects exactly CK2's column (select2.select96, same tie order).  cross=False is the
same head with the encoder replaced by a per-column MLP (no information between columns).

Loss (listwise, official labels of the SUBMITTED trajectory = lateral label, lat_mode 'on'):
    L = - sum_c softmax(r / tau)_c * PDMS_c      over the allowed columns (expected PDMS of the softmax selection)
Selection: argmax r over the allowed columns, ties -> select2.TIE_ORDER.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from . import select2 as S2

N_TYPE, N_RANK, D_H = S2.NV, S2.K16, 384
D_SCALAR = 5 + 5 + 3 + 6           # logits pool / lat, s0 / v2_final / log im, e_lat
D_IN = 2 * D_H + D_SCALAR + N_TYPE + N_RANK


def pool_features(h, h_lat, logit, logit_lat, s0, v2_final16, v2_im16, e_lat) -> torch.Tensor:
    """-> x [B, 96, D_IN] f32.  h / h_lat [B, 96, 384], logit / logit_lat [B, 96, 5], s0 [B, 96], v2_* [B, 16],
    e_lat [B, 96, 6]."""
    B, K = s0.shape
    dev = s0.device
    par = lambda x: x.float().repeat_interleave(S2.NV, 1)
    c = torch.arange(K, device=dev)
    typ = nn.functional.one_hot(c % S2.NV, N_TYPE).float()[None].expand(B, -1, -1)
    rank = nn.functional.one_hot(c // S2.NV, N_RANK).float()[None].expand(B, -1, -1)
    sc = torch.cat([logit.float(), logit_lat.float(), s0.float()[..., None], par(v2_final16)[..., None],
                    torch.log(par(v2_im16).clamp_min(1e-6))[..., None], e_lat.float()], -1)
    return torch.cat([h.float(), h_lat.float(), sc, typ, rank], -1)


class ListwiseHead(nn.Module):
    def __init__(self, d_in: int = D_IN, d: int = 256, n_layers: int = 2, n_heads: int = 8, d_ffn: int = 512,
                 dropout: float = 0.1, cross: bool = True):
        super().__init__()
        self.cross = bool(cross)
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        if self.cross:
            layer = nn.TransformerEncoderLayer(d, n_heads, d_ffn, dropout, activation="gelu", batch_first=True,
                                               norm_first=True)
            self.mix = nn.TransformerEncoder(layer, n_layers)
        else:
            self.mix = nn.Sequential(*[m for _ in range(n_layers) for m in
                                       (nn.LayerNorm(d), nn.Linear(d, d_ffn), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(d_ffn, d))])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """x [B, 96, D_IN], allowed [B, 96] bool -> delta [B, 96] (0 on disallowed columns)."""
        z = self.inp(x)
        if self.cross:
            z = self.mix(z, src_key_padding_mask=~allowed)
        else:
            z = z + self.mix(z)
        return self.out(self.norm(z))[..., 0].masked_fill(~allowed, 0.0)


def listwise_loss(r: torch.Tensor, pdms: torch.Tensor, allowed: torch.Tensor, tau: float) -> Dict[str, torch.Tensor]:
    """expected PDMS of softmax(r / tau) over the allowed columns (pdms [B, 96] in [0, 1], NaN only on disallowed)."""
    logit = (r / tau).masked_fill(~allowed, float("-inf"))
    p = torch.softmax(logit, -1)
    y = torch.nan_to_num(pdms, nan=0.0)
    exp = (p * y).sum(-1)
    return {"loss": -exp.mean(), "exp_pdms": exp.detach().mean()}


def select(r: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """argmax r over the allowed columns, ties in select2.TIE_ORDER (identity columns first) -> [N] int64."""
    s = np.where(np.asarray(allowed, bool) & np.isfinite(r), np.asarray(r, np.float64), -np.inf)
    j = np.argmax(s[:, S2.TIE_ORDER], axis=1)
    return S2.TIE_ORDER[j].astype(np.int64)
