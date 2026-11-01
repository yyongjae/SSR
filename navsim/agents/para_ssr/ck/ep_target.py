"""CK EP training target (user decision 2026-10-08 ~20:10 KST): official EP or a decoupled EP.

The official per-trajectory EP (navsim PDM scorer, label column 'ep') is multiplied by the multiplicative metrics:
    M = NC * DAC * DDC,  EP_off = clip(r M / max(p, r M), 0, 1) if max(p, r M) > 5 m else (1 if M > 0 else 0)
(r = raw_progress, p = pdm_progress_eff; reproduced to 1.3e-7 on the label files), so EP = 0 whenever NC, DAC or DDC
fails and the CK EP head partly re-learns NC / DAC.  The decoupled target drops M:
    EP_dec = clip(r / max(p, r), 0, 1) if max(p, r) > 5.0 else 1.0        (EP_dec == EP_off wherever M == 1)
    non-finite r or p -> NaN (the candidate becomes not-ok exactly like any other non-finite label).
5.0 = progress_distance_threshold of navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml.

Every place that turns the 9 label columns (constants.LABEL_COLS: nc, dac, ep, ttc, comfort, ddc, pdms, raw_progress,
pdm_progress_eff) into the 5 CK targets (CK_KEYS: nc, dac, ep, ttc, comfort) goes through ck_targets(labels9,
ep_target):  'official' -> exactly labels9[..., CK_LABEL_IDX] (same values, same dtype, a copy);  'decoupled' -> the
same array with the EP column replaced by decoupled_ep(labels9).  numpy or torch in, the same type out (torch: same
device).  Selection / evaluation PDMS stay on the official 'pdms' column; the AnchorSampler pass mask uses NC / DAC /
TTC / C only (never EP).  No navsim / tools imports (data workers import this module).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .constants import CK_KEYS, CK_LABEL_IDX, LBL

EP_TARGETS = ("official", "decoupled")
EP_DEFAULT = "official"                     # teacher trainer default (old behaviour); the e2e2 student sets its own
EP_DIST_THRESHOLD = 5.0                     # progress_distance_threshold (m), default_scoring_parameters.yaml
EP_KEY_IDX = CK_KEYS.index("ep")            # 2: EP column of a [..., 5] CK target array
R_COL, P_COL = LBL["raw_progress"], LBL["pdm_progress_eff"]
N_LABEL_COLS = len(LBL)                     # 9
EP_RULE = ("decoupled EP = clip(raw_progress / max(pdm_progress_eff, raw_progress), 0, 1) if max(.) > 5.0 m else 1.0 "
           "(official EP without the NC * DAC * DDC factor; non-finite -> NaN)")


def check_ep_target(ep_target: Any) -> str:
    s = "official" if ep_target is None else str(ep_target)
    if s not in EP_TARGETS:
        raise ValueError(f"ep_target must be one of {EP_TARGETS}, got {ep_target!r}")
    return s


def _is_torch(x) -> bool:
    try:
        import torch
    except ImportError:   # pragma: no cover
        return False
    return isinstance(x, torch.Tensor)


def _check_cols(shape) -> None:
    if len(shape) < 1 or shape[-1] != N_LABEL_COLS:
        raise ValueError(f"labels9 must have {N_LABEL_COLS} columns (LABEL_COLS) on the last axis, got shape "
                         f"{tuple(shape)}")


def decoupled_ep(labels9):
    """labels9 [..., 9] (LABEL_COLS) -> decoupled EP [...] (float64 arithmetic; dtype of labels9 when floating, else
    float32; numpy or torch like the input)."""
    _check_cols(labels9.shape)
    thr = EP_DIST_THRESHOLD
    if _is_torch(labels9):
        import torch
        dt = labels9.dtype if labels9.is_floating_point() else torch.float32
        r = labels9[..., R_COL].double()
        p = labels9[..., P_COL].double()
        fin = torch.isfinite(r) & torch.isfinite(p)
        m = torch.maximum(p, r)
        big = fin & (m > thr)
        one = torch.ones_like(r)
        e = torch.where(big, (r / torch.where(big, m, one)).clamp(0.0, 1.0), one)
        e = torch.where(fin, e, torch.full_like(r, float("nan")))
        return e.to(dt)
    a = np.asarray(labels9)
    dt = a.dtype if np.issubdtype(a.dtype, np.floating) else np.dtype(np.float32)
    r = a[..., R_COL].astype(np.float64)
    p = a[..., P_COL].astype(np.float64)
    fin = np.isfinite(r) & np.isfinite(p)
    with np.errstate(invalid="ignore"):
        m = np.maximum(p, r)
        big = fin & (m > thr)
        e = np.where(big, np.clip(r / np.where(big, m, 1.0), 0.0, 1.0), 1.0)
    e = np.where(fin, e, np.nan)
    return e.astype(dt)


def ck_targets(labels9, ep_target: str = EP_DEFAULT):
    """labels9 [..., 9] (LABEL_COLS) -> CK targets [..., 5] (CK_KEYS order nc, dac, ep, ttc, comfort).
    'official': labels9[..., CK_LABEL_IDX] (bit-identical, same dtype); 'decoupled': EP column = decoupled_ep."""
    ep_target = check_ep_target(ep_target)
    _check_cols(labels9.shape)
    idx = list(CK_LABEL_IDX)
    if _is_torch(labels9):
        y = labels9[..., idx]
        if ep_target == "official":
            return y
        if not y.is_floating_point():
            y = y.float()
        y = y.clone()
        y[..., EP_KEY_IDX] = decoupled_ep(labels9).to(y.dtype)
        return y
    a = np.asarray(labels9)
    y = a[..., idx]
    if ep_target == "official":
        return y
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32)
    y[..., EP_KEY_IDX] = decoupled_ep(a).astype(y.dtype)
    return y


def run_ep_target(cfg: Any) -> str:
    """ep_target recorded in a CK2 teacher run's config.json dict (runs written before the option = 'official')."""
    return check_ep_target((cfg or {}).get("ep_target", "official"))


__all__ = ["EP_TARGETS", "EP_DEFAULT", "EP_DIST_THRESHOLD", "EP_KEY_IDX", "EP_RULE", "check_ep_target",
           "decoupled_ep", "ck_targets", "run_ep_target"]
