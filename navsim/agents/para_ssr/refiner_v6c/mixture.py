"""Stage-T path mixture (report 124 §3-2 / 122 §11), shared by the teacher training and the stage-D R-E warm-up."""
from __future__ import annotations

import torch


def path_mixture(anchors: torch.Tensor, hum: torch.Tensor, hvalid: torch.Tensor, gen: torch.Generator,
                 n_anchor: int = 32, n_rule: int = 32, n_gt: int = 16) -> torch.Tensor:
    """report 124 §3-2 / 122 §11: per token 32 anchors (8 nearest the human path, 8 mid, 16 random), 32 rule speed
    variants of the 8 nearest (time scaling 0.6 / 0.8 / 1.2 / and a 1 m/s^2-like late brake), 16 GT variants (lateral
    +-0.5 / +-1.0 m, speed 0.8 / 1.2).  anchors [K, 8, 3]; hum [B, 8, 3] -> [B, 80, 8, 3] (ego frame)"""
    b = hum.shape[0]
    k = anchors.shape[0]
    d = torch.cdist(hum.reshape(b, -1), anchors.reshape(k, -1).to(hum))                    # [B, K]
    order = d.argsort(-1)
    near = order[:, :8]
    mid = order[:, 8 + torch.randperm(56, generator=gen)[:8]]
    rnd = torch.randint(0, k, (b, n_anchor - 16), generator=gen)
    sel = torch.cat([near, mid, rnd.to(near)], 1)
    A = anchors.to(hum)[sel]                                                               # [B, 32, 8, 3]

    def retime(p, s):                                                                      # speed x s along the path
        t = torch.arange(1, 9, dtype=p.dtype) * 0.5
        full = torch.cat([torch.zeros_like(p[..., :1, :]), p], -2)                          # [.., 9, 3]
        q = (t * s).clamp(max=4.0) / 0.5                                                    # fractional pose index
        i0 = q.floor().long().clamp(max=7)
        w = (q - i0.to(q))[..., None]
        return full[..., i0, :] * (1 - w) + full[..., i0 + 1, :] * w

    base = anchors.to(hum)[near]                                                            # [B, 8, 8, 3]
    rule = torch.cat([retime(base, s) for s in (0.6, 0.8, 1.2)] +
                     [torch.cat([base[..., :4, :], retime(base, 0.85)[..., 4:, :]], -2)], 1)   # [B, 32, 8, 3]
    lat = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=hum.dtype)
    gv = [hum[:, None] + torch.stack([torch.zeros_like(lat), lat, torch.zeros_like(lat)], -1)[None, :, None, :]
          * (torch.arange(1, 9, dtype=hum.dtype) / 8)[None, None, :, None]]
    gv.append(torch.stack([retime(hum, s) for s in (0.8, 1.2, 0.6, 1.0)], 1))
    gv.append(torch.stack([retime(hum, s) for s in (0.9, 1.1, 0.7, 0.5)], 1))
    gv.append(torch.stack([retime(hum, s) for s in (0.95, 1.05, 0.75, 0.4)], 1))
    G = torch.cat(gv, 1)[:, :n_gt]
    G = torch.where(hvalid.bool()[:, None, None, None], G, A[:, :n_gt])                    # no human path -> anchors
    return torch.cat([A, rule, G], 1)
