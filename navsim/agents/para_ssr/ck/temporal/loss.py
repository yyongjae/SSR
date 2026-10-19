"""refiner-v3 forecast loss (docs/refiner_v3/SPEC.md section 4.6; owner: builder LOSS+EVAL).

NOTE 2026-10-11 KST: this file was accidentally overwritten by the MODEL builder at 06:15 UTC; it was rebuilt by the
MODEL builder against the LOSS+EVAL builder's contract (tests/test_v3_loss_metrics.py and the imports of
metrics.py: AGENT_GROUPS, LOG2, forecast_valid, safe_targets, wta_index, _check_shapes).  Re-checked by LOSS+EVAL
(2026-10-11 15:3x KST): equal to the original on 30 random configurations (masks, NaN targets, M = 1..6), max |dloss|
4.8e-7, gradients identical [실측]; tests/test_v3_loss_metrics.py 21 passed.  metrics.py no longer imports this file.

forecast_loss(fc, fc_xyh, fc_state, obj_mask0) -> (loss, stats)
  fc         learned Forecast WITH grad (TemporalCtx.fc_raw): mu [T,K,M,10,2], log_b [T,K,M,10,2], logit_pi [T,K,M],
             group [T,K], valid [T,K] (duck-typed: only these fields are read).  A forecast without log_b / logit_pi
             (CA / oracle) is refused (ValueError): those arms have no forecast loss.
  fc_xyh     [T, >=K, 10, 3] GT target (x, y, heading_unwrapped) of t0 slot i at FC_T (heading unused: yaw rule)
  fc_state   [T, >=K, 10] int (FC_NONE / FC_OBS / FC_ABSENT / FC_BEYOND)
  obj_mask0  [T, >=K] bool t0 mask AFTER augmentation (an object dropped from the input is not supervised)
  Targets are trimmed to K.  Valid (object, step) = slot valid & agent (group 0 / 1) & obj_mask0 & state == FC_OBS.
  Winner-take-all: m* = argmin_m mean over valid steps of |mu - y|_1 (no grad; ties -> first mode).  NLL = mean over
  valid steps and both axes of log(2 b) + |mu - y| / b for m*; CE = -log pi_{m*}.  Per-object loss NLL + CE, averaged
  over objects with >= 1 valid step.  Invalid targets are replaced by mu.detach() before any arithmetic (no NaN
  reaches the graph); no valid object -> the same expression times 0 (finite 0 that stays in the graph: DDP in step).
  float32 with autocast disabled.  stats (floats): fc_loss, fc_nll, fc_ce, fc_minade, fc_minfde (L2, min over ALL
  modes; FDE at the object's last valid step), fc_n_obj.  No ranking term, no gamma exponent.
add_forecast_loss(total, stats, out, batch, lambda_fc)   trainer hook: total + lambda_fc * forecast_loss(out['fc'],
  batch['fc_xyh'], batch['fc_state'], batch['obj_mask'][:, 0]) when out has 'fc'; else total unchanged.
forecast_touch(fc)   0 * (sum of every learned output) -> keeps the forecaster in the graph (dummy_backward).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from .config import FC_OBS, FC_STEPS

Tensor = torch.Tensor
LOG2 = math.log(2.0)
AGENT_GROUPS = (0, 1)
STAT_KEYS = ("fc_loss", "fc_nll", "fc_ce", "fc_minade", "fc_minfde", "fc_n_obj")


def _check_shapes(fc, fc_xyh: Tensor, fc_state: Tensor, obj_mask0: Tensor) -> Tuple[int, int, int]:
    if fc.mu.dim() != 5 or fc.mu.shape[3] != FC_STEPS or fc.mu.shape[4] != 2:
        raise ValueError(f"fc.mu must be [T, K, M, {FC_STEPS}, 2], got {tuple(fc.mu.shape)}")
    T, K, M = (int(x) for x in fc.mu.shape[:3])
    if fc_xyh.dim() != 4 or fc_xyh.shape[0] != T or fc_xyh.shape[1] < K or fc_xyh.shape[2] != FC_STEPS \
            or fc_xyh.shape[3] < 2:
        raise ValueError(f"fc_xyh {tuple(fc_xyh.shape)} must be [T={T}, >={K}, {FC_STEPS}, 3]")
    if fc_state.dim() != 3 or fc_state.shape[0] != T or fc_state.shape[1] < K or fc_state.shape[2] != FC_STEPS:
        raise ValueError(f"fc_state {tuple(fc_state.shape)} must be [T={T}, >={K}, {FC_STEPS}]")
    if obj_mask0.dim() != 2 or obj_mask0.shape[0] != T or obj_mask0.shape[1] < K:
        raise ValueError(f"obj_mask0 {tuple(obj_mask0.shape)} must be [T={T}, >={K}]")
    if tuple(fc.group.shape) != (T, K) or tuple(fc.valid.shape) != (T, K):
        raise ValueError(f"fc.group / fc.valid must be [T, K] = {(T, K)}")
    return T, K, M


def forecast_valid(fc, fc_state: Tensor, obj_mask0: Tensor) -> Tensor:
    """[T, K, 10] bool: slot valid & agent & obj_mask0 & fc_state == FC_OBS (targets trimmed to K)."""
    K = fc.mu.shape[1]
    dev = fc.mu.device
    grp = fc.group.to(dev).long()
    agent = torch.zeros_like(grp, dtype=torch.bool)
    for g in AGENT_GROUPS:
        agent = agent | (grp == g)
    slot = fc.valid.to(dev).bool() & agent & obj_mask0[:, :K].to(dev).bool()
    return slot[..., None] & (fc_state[:, :K].to(dev).long() == FC_OBS)


def safe_targets(mu: Tensor, fc_xyh: Tensor, valid: Tensor) -> Tensor:
    """Targets broadcast to mu [T, K, M, 10, 2]; invalid entries (NaN allowed) replaced by mu.detach()."""
    K = mu.shape[1]
    y = fc_xyh[:, :K, :, :2].to(device=mu.device, dtype=mu.dtype)
    return torch.where(valid[:, :, None, :, None], y[:, :, None].expand_as(mu), mu.detach())


@torch.no_grad()
def wta_index(absd: Tensor, vmask: Tensor, den: Tensor) -> Tensor:
    """m* [T, K] = argmin_m mean over valid steps of |mu - y|_1 (first mode on ties).
    absd [T,K,M,10,2], vmask broadcastable to absd (1 = valid step), den [T,K] = max(n_valid, 1)."""
    l1 = (absd * vmask).sum((-1, -2)) / den[..., None].to(absd.dtype)
    return l1.argmin(-1)


def forecast_loss(fc, fc_xyh: Tensor, fc_state: Tensor, obj_mask0: Tensor) -> Tuple[Tensor, Dict[str, float]]:
    if getattr(fc, "log_b", None) is None or getattr(fc, "logit_pi", None) is None:
        raise ValueError("forecast_loss needs a learned forecast (log_b and logit_pi); CA / oracle have no loss")
    T, K, M = _check_shapes(fc, fc_xyh, fc_state, obj_mask0)
    with torch.autocast(device_type=fc.mu.device.type, enabled=False):
        mu = fc.mu.float()
        log_b = fc.log_b.float()
        logit_pi = fc.logit_pi.float()
        valid = forecast_valid(fc, fc_state, obj_mask0)                               # [T,K,10]
        y = safe_targets(mu, fc_xyh, valid)
        vmf = valid[:, :, None, :].to(mu.dtype)                                       # [T,K,1,10]
        nv = valid.sum(-1)
        obj_ok = (nv > 0).to(mu.dtype)
        den = nv.clamp(min=1).to(mu.dtype)
        absd = (mu - y).abs()
        m_star = wta_index(absd, vmf[..., None], den)                                 # [T,K]
        nll_step = (LOG2 + log_b + absd * torch.exp(-log_b)).mean(-1)                # [T,K,M,10]
        nll_m = (nll_step * vmf).sum(-1) / den[..., None]                             # [T,K,M]
        nll = torch.gather(nll_m, -1, m_star[..., None])[..., 0]
        ce = -torch.gather(torch.log_softmax(logit_pi, -1), -1, m_star[..., None])[..., 0]
        n_obj = obj_ok.sum()
        loss = ((nll + ce) * obj_ok).sum() / n_obj.clamp(min=1.0)
        with torch.no_grad():
            l2 = torch.linalg.vector_norm((mu - y).detach(), dim=-1) * vmf          # [T,K,M,10]
            ade = (l2.sum(-1) / den[..., None]).amin(-1)                              # [T,K]
            steps = torch.arange(FC_STEPS, device=mu.device)
            last = torch.where(valid, steps, torch.full_like(steps, -1)).amax(-1).clamp(min=0)
            fde = torch.gather(l2, -1, last[:, :, None, None].expand(T, K, M, 1))[..., 0].amin(-1)
            n = max(float(n_obj), 1.0)
            stats = {"fc_loss": float(loss.detach()), "fc_nll": float((nll.detach() * obj_ok).sum()) / n,
                     "fc_ce": float((ce.detach() * obj_ok).sum()) / n, "fc_minade": float((ade * obj_ok).sum()) / n,
                     "fc_minfde": float((fde * obj_ok).sum()) / n, "fc_n_obj": float(n_obj)}
    return loss, stats


def add_forecast_loss(total: Tensor, stats: Dict, out: Dict, batch: Dict, lambda_fc: float) -> Tensor:
    """Trainer hook (SPEC 5.3): total + lambda_fc * forecast loss when the model returned a learned forecast."""
    fc = out.get("fc") if isinstance(out, dict) else None
    if fc is None:
        return total
    for k in ("fc_xyh", "fc_state", "obj_mask"):
        if k not in batch:
            raise KeyError(f"batch lacks {k!r} needed by the forecast loss")
    l_fc, s = forecast_loss(fc, batch["fc_xyh"], batch["fc_state"], batch["obj_mask"][:, 0])
    stats.update(s)
    stats["fc_lambda"] = float(lambda_fc)
    return total + float(lambda_fc) * l_fc


def forecast_touch(fc) -> Tensor:
    """0 * (sum of mu, log_b, logit_pi): a zero-valued term that keeps every forecaster output in the graph."""
    if fc is None:
        return torch.zeros(())
    z = fc.mu.float().sum() * 0.0
    for name in ("log_b", "logit_pi"):
        v = getattr(fc, name, None)
        if v is not None:
            z = z + v.float().sum() * 0.0
    return z


__all__ = ["forecast_loss", "add_forecast_loss", "forecast_touch", "forecast_valid", "safe_targets", "wta_index",
           "AGENT_GROUPS", "LOG2", "FC_OBS", "FC_STEPS", "STAT_KEYS"]
