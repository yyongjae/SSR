"""CK KD target combination (report 44 §5): one target per quantity, taken from the teacher whose privileged BEV
carries that information (v1 E2 showed that averaging per-teacher L1 losses hurts).

  scores     (KD_SCORE_SOURCE)  nc, ttc <- DET;  dac <- MAP;  ep, comfort <- mean of the two teacher probabilities
  controls   (KD_CTRL_SOURCE)   longitudinal c_lon <- DET;  lateral e_lat <- MAP
  corrected KD candidate        tau'_KD = correct(cand, z_DET, w_MAP, v0, slope=0)['traj']
No MAP teacher (map None) -> DET for everything; a candidate whose MAP output is not ok -> DET values there and
kd_ok False (kd_ok = det ok AND (map ok or map None)).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .constants import CK_KEYS, KD_CTRL_SOURCE, KD_SCORE_SOURCE, N_CTRL
from .correct import correct


def _sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x.float())
    return (0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, np.float32)))).astype(np.float32)


def _ctrl6(x):
    return x[..., 2:] if x.shape[-1] == N_CTRL + 2 else x


def combine_teacher(det: Dict, map: Optional[Dict] = None) -> Dict:  # noqa: A002 (contract name)
    """det / map: dict(score_logit [N, K, 5], ok [N, K], optional c_lon [N, K, 6 | 8], e_lat [N, K, 6 | 8]); all
    numpy or all torch.  -> kd_score_prob [N, K, 5] f32, kd_ok [N, K] bool, and (if c_lon / e_lat given)
    kd_c_lon [N, K, 6] (DET), kd_e_lat [N, K, 6] (MAP, else DET).  Same array type as the inputs."""
    is_t = isinstance(det["score_logit"], torch.Tensor)
    where = torch.where if is_t else np.where
    as_b = (lambda x: torch.as_tensor(x).bool()) if is_t else (lambda x: np.asarray(x, bool))
    p_det = _sigmoid(det["score_logit"])
    ok_det = as_b(det["ok"])
    prob = p_det.clone() if is_t else p_det.copy()
    out: Dict = {}
    if map is None:
        kd_ok = ok_det
    else:
        p_map = _sigmoid(map["score_logit"])
        ok_map = as_b(map["ok"])
        if is_t:
            ok_map = ok_map.to(ok_det.device)
            p_map = p_map.to(p_det.device)
        kd_ok = ok_det & ok_map
        for i, k in enumerate(CK_KEYS):
            src = KD_SCORE_SOURCE[k]
            if src == "map":
                prob[..., i] = where(ok_map, p_map[..., i], p_det[..., i])
            elif src == "mean":
                prob[..., i] = where(ok_map, 0.5 * (p_det[..., i] + p_map[..., i]), p_det[..., i])
            elif src != "det":
                raise ValueError(f"KD_SCORE_SOURCE[{k}] = {src!r}")
    out["kd_score_prob"] = prob
    out["kd_ok"] = kd_ok
    if "c_lon" in det:
        if KD_CTRL_SOURCE["lon"] != "det":
            raise ValueError("KD_CTRL_SOURCE['lon'] must be 'det'")
        out["kd_c_lon"] = _ctrl6(det["c_lon"])
    if "e_lat" in det:
        e_det = _ctrl6(det["e_lat"])
        if map is not None and "e_lat" in map and KD_CTRL_SOURCE["lat"] == "map":
            e_map = _ctrl6(map["e_lat"])
            if is_t:
                e_map = e_map.to(e_det.device)
            out["kd_e_lat"] = where(ok_map[..., None], e_map, e_det)
        else:
            out["kd_e_lat"] = e_det
    return out


def combine_controls(z_det, w_map, w_det=None) -> Tuple:
    """The control rule: corrected KD candidate uses z from DET and w from MAP (w_det when there is no MAP teacher).
    -> (z, w); then tau'_KD = correct(cand, z, w, v0, slope=0)['traj'] (kd_corrected)."""
    return z_det, (w_det if w_map is None else w_map)


@torch.no_grad()
def kd_corrected(cand: torch.Tensor, z_det: torch.Tensor, w_map: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """tau'_KD [T, K, 8, 3] = decode(cand, z_DET, w_MAP) (mode A, slope 0)."""
    z, w = combine_controls(z_det, w_map)
    return correct(cand, z, w, v0, slope=0.0)["traj"]
