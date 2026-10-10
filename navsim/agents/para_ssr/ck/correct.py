"""CK correction branch: decode the K candidates' controls with the v1 M4 decoder (mode A, H1 heading).

correct(tau [T,K,8,3], z [T,K,6], w [T,K,6], v0 [T], slope) flattens to B = T*K (b = t*K + k), calls
refiner.decoder.decode(mode='A', lon_st_slope=slope) in float32 outside autocast, and reshapes back.
z = w = 0 returns tau bit-exact (decoder anchoring).  KD control space = decoded c_lon[..., 2:] (m/s, <= 0 in mode A)
and e_lat[..., 2:] (m, |e| <= 2) (c_0 = c_1 = e_0 = e_1 = 0 always).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from ..refiner.decoder import D_MAX, N_FREE, decode, lon_q_from_c, lon_z_from_q
from .constants import DECODE_MODE


def correct(tau: torch.Tensor, z: torch.Tensor, w: torch.Tensor, v0: torch.Tensor, slope: float = 0.0,
            mode: str = DECODE_MODE) -> Dict:
    """-> dict traj [T,K,8,3], c_lon [T,K,8], e_lat [T,K,8], raw (the flat decode dict, B = T*K, for the surrogate).
    tau is an input (the caller detaches it); gradients flow to z / w.  slope = mode-A straight-through backward
    slope (training 0.1, eval / KD targets 0.0; the forward does not depend on it)."""
    T, K = tau.shape[:2]
    B = T * K
    with torch.autocast(device_type=tau.device.type, enabled=False):
        v = torch.nan_to_num(v0.float().reshape(T), nan=0.0).repeat_interleave(K)
        dec = decode(tau.reshape(B, *tau.shape[2:]).float(), z.reshape(B, N_FREE).float(),
                     w.reshape(B, N_FREE).float(), v0=v, mode=mode, lon_st_slope=float(slope))
    return {"traj": dec["traj"].reshape(T, K, *dec["traj"].shape[1:]),
            "c_lon": dec["c_lon"].reshape(T, K, -1), "e_lat": dec["e_lat"].reshape(T, K, -1), "raw": dec}


def kd_space(corr: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """(c_lon[..., 2:], e_lat[..., 2:]) = the 6 + 6 decoded KD controls of a correct() output."""
    return corr["c_lon"][..., 2:], corr["e_lat"][..., 2:]


def controls_from_kd(c_lon: torch.Tensor, e_lat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of the KD space (c_lon [.., 6 or 8], e_lat [.., 6 or 8]) -> raw (z [.., 6], w [.., 6]).  Exact where
    the decoder's mode-A clamp and curvature projection were inactive (all pre-clamp c <= 0, alpha = 1); used for
    round-trip checks and to turn decoded teacher targets back into controls."""
    c = c_lon if c_lon.shape[-1] == N_FREE + 2 else torch.cat([torch.zeros_like(c_lon[..., :2]), c_lon], -1)
    e = e_lat[..., 2:] if e_lat.shape[-1] == N_FREE + 2 else e_lat
    q = lon_q_from_c(c.double())[..., 1:]
    z = lon_z_from_q(q)
    w = torch.atanh(torch.clamp(e.double() / D_MAX, -1 + 1e-12, 1 - 1e-12))
    return z.to(c_lon.dtype), w.to(e_lat.dtype)
