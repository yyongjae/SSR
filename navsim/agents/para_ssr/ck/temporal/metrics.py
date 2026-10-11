"""Forecast metrics of the refiner-v3 temporal branch (SPEC sections 4.6, 5.3, 8; builder LOSS+EVAL, 2026-10-11 KST).

Works for every generator (learned, CA, oracle): it reads mu / b / pi (and log_b when present), never logit_pi.
Self-contained (no import of loss.py, owned by the model builder); same validity rule as loss.forecast_loss:
slot valid & agent (group 0 / 1 = Forecast.agent) & obj_mask0 & fc_state == FC_OBS, and the same WTA rule.

  forecast_errors(fc, fc_xyh, fc_state, obj_mask0)  per-object tensors (no grad):
      valid [T, K, 10], has [T, K], group [T, K], l2 [T, K, M, 10] (masked 0), ade [T, K, M], fde [T, K, M]
      (L2 at the object's LAST valid step), minade / minfde [T, K] (min over ALL modes), de_step_min [T, K, 10]
      (min over modes of the L2 at each step, 0 where invalid), wta_nll [T, K] (the training NLL term at the WTA mode
      m* of the loss: mean over valid steps and axes of log 2b + |mu - y| / b), mix_nll [T, K] (proper mixture NLL
      -log sum_m pi_m prod_{k, axis} Laplace(y | mu, b), divided by 2 * n_valid_steps; equals wta_nll for M = 1)
  ForecastMetrics   object-weighted accumulator over batches / DDP ranks (sums, merge, to_tensor / from_tensor for
      all_reduce) -> result(prefix='fc_'): n_obj, n_steps, minade, minfde, nll (= WTA NLL, the training term),
      mixnll, miss2m (share of objects with minFDE > 2 m), veh_* / vru_* minade / minfde / n_obj, minde_1s / 3s / 5s
      (min-over-modes displacement at t = 1, 3, 5 s over objects valid at that step).

ADE is descriptive only: it is not a proxy for critic value (CA beats CV in the stack although its 5-s ADE is worse
[실측 A_full16 notes section 4-4]).  The Step-1 decision is offline (decide_step1.py, SPEC section 8).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch

from .config import AGENT_CLASSES, B_MIN, FC_OBS, FC_STEPS, GROUP_OF_CLASS

AGENT_GROUPS = tuple(sorted({GROUP_OF_CLASS[c] for c in AGENT_CLASSES}))   # (0 vehicle, 1 VRU); 2 static: no target
LOG2 = math.log(2.0)
B_FLOOR = B_MIN                     # scale floor for generators that give b without log_b
MISS_M = 2.0
HORIZON_STEPS = {"1s": 1, "3s": 5, "5s": 9}       # step index k - 1 of t = 0.5 k


def _check_shapes(fc, fc_xyh: torch.Tensor, fc_state: torch.Tensor, obj_mask0: torch.Tensor):
    if fc.mu.dim() != 5 or fc.mu.shape[-1] != 2 or fc.mu.shape[-2] != FC_STEPS:
        raise ValueError(f"fc.mu must be [T, K, M, {FC_STEPS}, 2], got {tuple(fc.mu.shape)}")
    T, K, M = fc.mu.shape[:3]
    if fc_xyh.dim() != 4 or fc_xyh.shape[0] != T or fc_xyh.shape[1] < K or fc_xyh.shape[2] != FC_STEPS \
            or fc_xyh.shape[3] < 2:
        raise ValueError(f"fc_xyh must be [T={T}, >= K={K}, {FC_STEPS}, >= 2], got {tuple(fc_xyh.shape)}")
    if fc_state.dim() != 3 or fc_state.shape[0] != T or fc_state.shape[1] < K or fc_state.shape[2] != FC_STEPS:
        raise ValueError(f"fc_state must be [T={T}, >= K={K}, {FC_STEPS}], got {tuple(fc_state.shape)}")
    if obj_mask0.dim() != 2 or obj_mask0.shape[0] != T or obj_mask0.shape[1] < K:
        raise ValueError(f"obj_mask0 must be [T={T}, >= K={K}], got {tuple(obj_mask0.shape)}")
    if tuple(fc.valid.shape) != (T, K) or tuple(fc.group.shape) != (T, K):
        raise ValueError(f"fc.valid / fc.group must be [T={T}, K={K}]")
    return T, K, M


def forecast_valid(fc, fc_state: torch.Tensor, obj_mask0: torch.Tensor) -> torch.Tensor:
    """bool [T, K, 10] = slot valid & agent (group 0 / 1, = Forecast.agent) & obj_mask0 & fc_state == FC_OBS
    (the forecast_loss rule of SPEC 4.6), targets trimmed to K."""
    K = fc.mu.shape[1]
    dev = fc.mu.device
    grp = fc.group.to(dev).long()
    agent = torch.zeros_like(grp, dtype=torch.bool)
    for g in AGENT_GROUPS:
        agent |= grp == g
    ok = fc.valid.to(dev).bool() & agent & obj_mask0[:, :K].to(dev).bool()
    return ok[..., None] & (fc_state[:, :K].to(dev).long() == FC_OBS)


def safe_targets(mu: torch.Tensor, fc_xyh: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """y [T, K, M, 10, 2] = GT (x, y) where valid, mu elsewhere (no NaN from missing targets)."""
    K = mu.shape[1]
    y = fc_xyh[:, :K, :, :2].to(device=mu.device, dtype=mu.dtype)
    return torch.where(valid[:, :, None, :, None], y[:, :, None].expand_as(mu), mu.detach())


def wta_index(absd: torch.Tensor, vmf: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """m* [T, K] = argmin_m mean over valid steps of |mu - y|_1 (first minimum on ties; = the loss's WTA rule)."""
    with torch.no_grad():
        return ((absd.detach() * vmf).sum((-1, -2)) / den[..., None]).argmin(-1)


@torch.no_grad()
def forecast_errors(fc, fc_xyh: torch.Tensor, fc_state: torch.Tensor,
                    obj_mask0: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-object error tensors (module docstring).  All float32, on fc.mu's device."""
    T, K, M = _check_shapes(fc, fc_xyh, fc_state, obj_mask0)
    with torch.autocast(device_type=fc.mu.device.type, enabled=False):
        mu = fc.mu.detach().float()
        if getattr(fc, "log_b", None) is not None:
            log_b = fc.log_b.detach().float()
        else:
            log_b = torch.log(fc.b.detach().float().clamp(min=B_FLOOR))
        pi = fc.pi.detach().float().to(mu.device)
        if tuple(pi.shape) != (T, K, M):
            raise ValueError(f"fc.pi {tuple(pi.shape)} must be [T, K, M] = {(T, K, M)}")
        valid = forecast_valid(fc, fc_state, obj_mask0)
        y = safe_targets(mu, fc_xyh, valid)
        vm = valid[:, :, None, :]                                             # [T, K, 1, 10]
        vmf = vm.float()
        nstep = valid.sum(-1)
        has = nstep > 0
        den = nstep.clamp(min=1).float()
        diff = mu - y
        l2 = diff.pow(2).sum(-1).sqrt() * vmf                                 # [T, K, M, 10]
        ade = l2.sum(-1) / den[..., None]                                     # [T, K, M]
        steps = torch.arange(FC_STEPS, device=mu.device)
        last = torch.where(valid, steps, torch.full_like(steps, -1)).amax(-1).clamp(min=0)     # [T, K]
        fde = l2.gather(-1, last[:, :, None, None].expand(T, K, M, 1))[..., 0]                 # [T, K, M]
        absd = diff.abs()
        m_star = wta_index(absd, vmf[..., None], den)[..., None]
        nll_e = (LOG2 + log_b + absd * torch.exp(-log_b)) * vmf[..., None]
        nll_sum = nll_e.sum((-1, -2))                                         # [T, K, M] summed over steps, axes
        wta = nll_sum.gather(-1, m_star)[..., 0] / (2.0 * den)
        log_pi = torch.log(pi.clamp(min=1e-12))
        mix = -torch.logsumexp(log_pi - nll_sum, -1) / (2.0 * den)
        de_step = l2.amin(2) * valid.float()                                  # [T, K, 10]
        z = torch.zeros_like(den)
        return {"valid": valid, "has": has, "group": fc.group.to(mu.device).long(), "l2": l2, "ade": ade,
                "fde": fde, "minade": torch.where(has, ade.amin(-1), z), "minfde": torch.where(has, fde.amin(-1), z),
                "de_step_min": de_step, "wta_nll": torch.where(has, wta, z), "mix_nll": torch.where(has, mix, z),
                "n_step": nstep}


class ForecastMetrics:
    """Object-weighted forecast metrics accumulated over batches (and DDP ranks via to_tensor / all_reduce)."""

    GROUPS = {"veh": 0, "vru": 1}
    KEYS = (("n_obj", "n_steps", "s_minade", "s_minfde", "s_wta_nll", "s_mix_nll", "n_miss")
            + tuple(f"{g}_{k}" for g in ("veh", "vru") for k in ("n_obj", "s_minade", "s_minfde"))
            + tuple(f"h{h}_{k}" for h in HORIZON_STEPS for k in ("n", "s_de")))

    def __init__(self, miss_m: float = MISS_M):
        self.miss_m = float(miss_m)
        self.sums: Dict[str, float] = {k: 0.0 for k in self.KEYS}

    def update(self, fc, fc_xyh: torch.Tensor, fc_state: torch.Tensor,
               obj_mask0: torch.Tensor) -> Dict[str, torch.Tensor]:
        e = forecast_errors(fc, fc_xyh, fc_state, obj_mask0)
        h = e["has"]
        s = self.sums
        s["n_obj"] += float(h.sum())
        s["n_steps"] += float(e["valid"].sum())
        s["s_minade"] += float(e["minade"][h].sum())
        s["s_minfde"] += float(e["minfde"][h].sum())
        s["s_wta_nll"] += float(e["wta_nll"][h].sum())
        s["s_mix_nll"] += float(e["mix_nll"][h].sum())
        s["n_miss"] += float((e["minfde"][h] > self.miss_m).sum())
        for g, gi in self.GROUPS.items():
            hg = h & (e["group"] == gi)
            s[f"{g}_n_obj"] += float(hg.sum())
            s[f"{g}_s_minade"] += float(e["minade"][hg].sum())
            s[f"{g}_s_minfde"] += float(e["minfde"][hg].sum())
        for hn, k in HORIZON_STEPS.items():
            vk = e["valid"][..., k]
            s[f"h{hn}_n"] += float(vk.sum())
            s[f"h{hn}_s_de"] += float(e["de_step_min"][..., k][vk].sum())
        return e

    def merge(self, other: "ForecastMetrics") -> "ForecastMetrics":
        for k in self.KEYS:
            self.sums[k] += other.sums[k]
        return self

    def to_tensor(self, device="cpu") -> torch.Tensor:
        return torch.tensor([self.sums[k] for k in self.KEYS], dtype=torch.float64, device=device)

    @classmethod
    def from_tensor(cls, t: torch.Tensor, miss_m: float = MISS_M) -> "ForecastMetrics":
        m = cls(miss_m)
        v = t.detach().double().cpu().tolist()
        if len(v) != len(cls.KEYS):
            raise ValueError(f"expected {len(cls.KEYS)} sums, got {len(v)}")
        m.sums = dict(zip(cls.KEYS, (float(x) for x in v)))
        return m

    def result(self, prefix: str = "fc_") -> Dict[str, float]:
        s = self.sums
        nan = float("nan")

        def r(a, b):
            return s[a] / s[b] if s[b] > 0 else nan

        out = {"n_obj": s["n_obj"], "n_steps": s["n_steps"], "minade": r("s_minade", "n_obj"),
               "minfde": r("s_minfde", "n_obj"), "nll": r("s_wta_nll", "n_obj"), "mixnll": r("s_mix_nll", "n_obj"),
               "miss2m" if self.miss_m == 2.0 else f"miss{self.miss_m:g}m": r("n_miss", "n_obj")}
        for g in self.GROUPS:
            out[f"{g}_n_obj"] = s[f"{g}_n_obj"]
            out[f"{g}_minade"] = r(f"{g}_s_minade", f"{g}_n_obj")
            out[f"{g}_minfde"] = r(f"{g}_s_minfde", f"{g}_n_obj")
        for hn in HORIZON_STEPS:
            out[f"minde_{hn}"] = r(f"h{hn}_s_de", f"h{hn}_n")
        return {f"{prefix}{k}": float(v) for k, v in out.items()}


def summarize(metrics: Iterable[ForecastMetrics], prefix: str = "fc_") -> Dict[str, float]:
    """Merge several accumulators (e.g. per-worker) -> result()."""
    acc: Optional[ForecastMetrics] = None
    for m in metrics:
        acc = ForecastMetrics(m.miss_m).merge(m) if acc is None else acc.merge(m)
    return (acc or ForecastMetrics()).result(prefix)


__all__ = ["forecast_errors", "ForecastMetrics", "summarize", "AGENT_GROUPS", "HORIZON_STEPS"]
