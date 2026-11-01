"""CK selection among the v2 top-K candidates (report 44 §6).

  ck_final(prob, im)  = w_im log(im + eps) + w_nc log(nc + eps) + w_dac log(dac + eps) + w_rest log(5 ttc + 2 c + 5 ep
                        + eps)  (= AnchorPlanner.weighted_reward with the CK probabilities; eps 1e-6; im = v2 softmax)
  resolve_w(w)        weights (w_im, w_nc, w_dac, w_rest) from a tuple or a constants.SEL_W_SETS name ('default' =
                      SEL_W, 'noim', 'plugin' = (0, 1, 1, 1)); every function here accepts either (default SEL_W).
  blend(v2_final, ck_fin, beta) = (1 - beta) v2_final + beta ck_fin
  select_all          v2 (index 0 = submitted), a (argmax blend over the K originals), b (= a, its corrected trajectory
                      is submitted), c (argmax over originals then corrected, 2K pool; the corrected candidates use their
                      own CK probabilities and the original's im / v2_final).  np.argmax takes the first maximum, so a
                      tie keeps the original (e.g. an identity correction).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .constants import SEL_EPS, SEL_W, SEL_W_SETS


def resolve_w(w=None) -> tuple:
    """None -> SEL_W; a SEL_W_SETS name ('default', 'noim', 'plugin') -> its weights; else 4 floats (w_im, w_nc,
    w_dac, w_rest)."""
    if w is None:
        return tuple(SEL_W)
    if isinstance(w, str):
        if w not in SEL_W_SETS:
            raise ValueError(f"selection weight set {w!r} not in {sorted(SEL_W_SETS)}")
        return tuple(SEL_W_SETS[w])
    t = tuple(float(x) for x in w)
    if len(t) != 4:
        raise ValueError(f"selection weights must be (w_im, w_nc, w_dac, w_rest), got {w!r}")
    return t


def sel_w_name(w=None) -> str:
    """name of a weight set (the SEL_W_SETS key whose weights equal w; else the weights as text)."""
    t = resolve_w(w)
    for k, v in SEL_W_SETS.items():
        if tuple(float(x) for x in v) == t:
            return k
    return ",".join(f"{x:g}" for x in t)


def ck_final(prob, im, w=SEL_W):
    """prob [.., K, 5] (CK_KEYS order: nc, dac, ep, ttc, comfort), im [.., K] -> [.., K] (numpy or torch).
    w: 4 weights or a SEL_W_SETS name (resolve_w)."""
    w = resolve_w(w) if isinstance(w, str) or w is None else w
    log = torch.log if isinstance(prob, torch.Tensor) else np.log
    nc, dac, ep, ttc, cmf = (prob[..., i] for i in range(5))
    return (w[0] * log(im + SEL_EPS) + w[1] * log(nc + SEL_EPS) + w[2] * log(dac + SEL_EPS)
            + w[3] * log(5 * ttc + 2 * cmf + 5 * ep + SEL_EPS))


def blend(v2_final, ck_fin, beta: float):
    return (1.0 - float(beta)) * v2_final + float(beta) * ck_fin


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _argmax(s: np.ndarray) -> np.ndarray:
    s = np.where(np.isfinite(s), s, -np.inf)
    return np.argmax(s, axis=-1).astype(np.int64)


def select_all(v2_final, v2_im, prob, prob_corr=None, beta: float = 1.0, w=SEL_W) -> Dict[str, np.ndarray]:
    """v2_final / v2_im [N, K], prob / prob_corr [N, K, 5] (CK sigmoid) -> dict of int64 [N]: 'v2' (0), 'a', 'b'
    (index into the K originals) and, with prob_corr, 'c' (index into [originals; corrected], 0 .. 2K-1)."""
    v2f = _np(v2_final).astype(np.float64)
    im = _np(v2_im).astype(np.float64)
    s = blend(v2f, ck_final(_np(prob).astype(np.float64), im, w), beta)
    a = _argmax(s)
    out = {"v2": np.zeros(len(a), np.int64), "a": a, "b": a.copy()}
    if prob_corr is not None:
        sc = blend(v2f, ck_final(_np(prob_corr).astype(np.float64), im, w), beta)
        out["c"] = _argmax(np.concatenate([s, sc], -1))
    return out
