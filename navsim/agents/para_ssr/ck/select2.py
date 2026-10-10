"""CK2 e2e selection over the 96-candidate pool (SPEC s1-5 / s5-2; user: old ck_final blended with v2 by beta, no gamma,
calibration offset OFF by default, all 96 candidates, lateral applied AFTER selection).

Pool layout (= CKE2E2.infer, e2e_data2): column c = k * 6 + v, k = v2 top-16 rank (plan_final_rewards desc), v = VNAMES
(id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5); identity columns c = k * 6 are the v2 top-16, column 0 = v2's trajectory.

  pool_scores(v2_final16, v2_im16, prob96, beta)  score_c = (1 - beta) v2_final[c // 6] + beta ck_final(p_c, im[c // 6])
                                                  (variants inherit the parent's v2_final / im, NU26; ck_final / blend
                                                  = ck/select.py, SEL_W / SEL_EPS); optional per-key logit `offset`
                                                  (calibration; default None = OFF); w = 4 weights or a SEL_W_SETS
                                                  name ('default' = SEL_W, 'noim', 'plugin' = (0, 1, 1, 1)).
  select96(..., valid96, beta, types)             argmax over the allowed columns (identity always allowed; variant
                                                  columns: valid96 & type in `types`; non-finite score never chosen);
                                                  ties -> identity columns first (k ascending), then variants in c order.
  final_traj(cand96, lat_traj96, idx, lat_mode)   'on': the selected column's laterally corrected trajectory (b-style,
                                                  user); 'on_except_latvar': uncorrected when the selected column is a
                                                  lateral variant (l-0.5 / l+0.5), else corrected (NU27); 'off': the
                                                  selected pool trajectory.
numpy in / numpy out for the scores and indices; final_traj keeps the type of cand96 (torch or numpy).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import SEL_W, SEL_W_SETS  # noqa: F401  (SEL_W_SETS re-exported for the eval tools)
from .select import blend, ck_final, resolve_w, sel_w_name  # noqa: F401

K16, NV, G_K = 16, 6, 96
VNAMES = ("id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5")
VARIANT_SETS: Dict[str, Tuple[int, ...]] = {
    "none": (0,), "decel": (0, 1, 2), "speed": (0, 1, 2, 3), "lat": (0, 4, 5),
    "all_noaccel": (0, 1, 2, 4, 5), "all": (0, 1, 2, 3, 4, 5)}                       # NU28 (type ids = VNAMES index)
LAT_MODES = ("on", "on_except_latvar", "off")                                       # NU27
LAT_TYPES = (4, 5)
# identity columns first (k ascending), then the variant columns in c order (tie-break order of select96)
TIE_ORDER = np.concatenate([np.arange(0, G_K, NV), np.array([c for c in range(G_K) if c % NV], np.int64)])


def _np(x, dtype=None) -> np.ndarray:
    a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return a if dtype is None else a.astype(dtype)


def parent_of(x16: np.ndarray) -> np.ndarray:
    """[N, 16, ...] -> [N, 96, ...] (column c takes row c // 6)."""
    return np.repeat(np.asarray(x16), NV, axis=1)


def apply_offset(prob: np.ndarray, offset: Optional[Sequence[float]]) -> np.ndarray:
    """sigmoid(logit(p) + offset) per CK key (calibration; None = unchanged)."""
    if offset is None:
        return prob
    off = np.asarray(offset, np.float64).reshape((1,) * (prob.ndim - 1) + (-1,))
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-(np.log(p) - np.log1p(-p) + off)))


def pool_scores(v2_final16, v2_im16, prob96, beta: float, w=SEL_W, offset=None) -> np.ndarray:
    """v2_final16 / v2_im16 [N, 16], prob96 [N, 96, 5] (CK2 sigmoid, CK_KEYS order) -> blended score [N, 96] f64."""
    v2f = parent_of(_np(v2_final16, np.float64))
    im = parent_of(_np(v2_im16, np.float64))
    p = apply_offset(_np(prob96, np.float64), offset)
    if p.shape[1] != G_K or v2f.shape[1] != G_K:
        raise ValueError(f"pool_scores: expected 16 parents and 96 columns, got {v2f.shape} / {p.shape}")
    return blend(v2f, ck_final(p, im, w), beta)


def allowed_mask(valid96, types: Sequence[int] = VARIANT_SETS["all"]) -> np.ndarray:
    """[N, 96] bool: identity columns always; variant columns valid & type in `types`."""
    v = _np(valid96).astype(bool)
    t = np.arange(G_K) % NV
    in_set = np.isin(t, np.asarray(tuple(types), np.int64)) | (t == 0)
    return (v | (t == 0)[None]) & in_set[None]


def select96(v2_final16, v2_im16, prob96, valid96, beta: float, types: Sequence[int] = VARIANT_SETS["all"],
             offset=None, w=SEL_W) -> np.ndarray:
    """-> selected column [N] int64 (0..95).  Non-finite / disallowed columns get -inf; all -inf -> column 0."""
    s = pool_scores(v2_final16, v2_im16, prob96, beta, w, offset)
    ok = allowed_mask(valid96, types) & np.isfinite(s)
    s = np.where(ok, s, -np.inf)
    j = np.argmax(s[:, TIE_ORDER], axis=1)                 # first maximum in tie order
    return TIE_ORDER[j].astype(np.int64)


def use_lat(idx, lat_mode: str) -> np.ndarray:
    """[N] bool: whether the submitted trajectory is the laterally corrected one."""
    if lat_mode not in LAT_MODES:
        raise ValueError(f"lat_mode must be one of {LAT_MODES}, got {lat_mode!r}")
    i = _np(idx).astype(np.int64)
    if lat_mode == "on":
        return np.ones(i.shape, bool)
    if lat_mode == "off":
        return np.zeros(i.shape, bool)
    return ~np.isin(i % NV, LAT_TYPES)


def final_traj(cand96, lat_traj96, idx, lat_mode: str):
    """-> (traj [N, 8, 3] (type of cand96), src [N] str 'pool' | 'lat')."""
    lat = use_lat(idx, lat_mode)
    src = np.where(lat, "lat", "pool")
    if isinstance(cand96, torch.Tensor):
        ii = torch.as_tensor(_np(idx).astype(np.int64), device=cand96.device)
        ar = torch.arange(cand96.shape[0], device=cand96.device)
        a, b = cand96[ar, ii], torch.as_tensor(lat_traj96, device=cand96.device)[ar, ii]
        m = torch.as_tensor(lat, device=cand96.device)[:, None, None]
        return torch.where(m, b.to(a.dtype), a), src
    c, lt = np.asarray(cand96), np.asarray(lat_traj96)
    ii = _np(idx).astype(np.int64)
    ar = np.arange(len(ii))
    return np.where(lat[:, None, None], lt[ar, ii], c[ar, ii]), src


def chosen_labels(lab_pool: np.ndarray, lab_lat: np.ndarray, idx, lat_mode: str) -> np.ndarray:
    """official labels of the submitted trajectory: [N, 96, C] pool / lateral-corrected labels -> [N, C]."""
    ii = _np(idx).astype(np.int64)
    ar = np.arange(len(ii))
    lat = use_lat(ii, lat_mode)
    return np.where(lat[:, None], lab_lat[ar, ii], lab_pool[ar, ii])
