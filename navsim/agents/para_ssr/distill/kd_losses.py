"""Feature and response distillation losses used by Stage-2 PlanningDistillation."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


def masked_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask_tokens: Optional[torch.Tensor],
) -> torch.Tensor:
    """``student``/``teacher`` are ``[B, N, C]``. Mask is ``[B, N, 1]`` or None."""
    diff = (student - teacher).square()
    if mask_tokens is None:
        return diff.mean()
    weighted = diff * mask_tokens
    return weighted.sum() / (mask_tokens.sum() * student.size(-1)).clamp_min(1e-6)


def channel_wise_kd(
    student_map: torch.Tensor,
    teacher_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tau: float = 4.0,
) -> torch.Tensor:
    """CWD (Shu et al., ICCV 2021): KL on per-channel spatial softmax.

    ``student_map``/``teacher_map`` are ``[B, C, H, W]``.  A spatial mask, if
    given, is added in log-space so low-weight cells do not dominate the
    softmax (the planning-aware analogue of CWD's saliency focus).
    """
    bsz, channels, height, width = student_map.shape
    tau = max(float(tau), 1e-3)
    student = student_map.reshape(bsz, channels, height * width) / tau
    teacher = teacher_map.reshape(bsz, channels, height * width) / tau
    if mask is not None:
        log_w = mask.reshape(bsz, 1, height * width).clamp_min(1e-6).log()
        student = student + log_w
        teacher = teacher + log_w
    log_s = F.log_softmax(student, dim=-1)
    p_t = F.softmax(teacher.detach(), dim=-1)
    return (p_t * (p_t.clamp_min(1e-8).log() - log_s)).sum(dim=-1).mean()


def relation_kd(
    student_map: torch.Tensor,
    teacher_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_samples: int = 64,
) -> torch.Tensor:
    """Pairwise cosine-relation MSE on the highest-weight cells (MapDistill/FGD)."""
    bsz, channels, height, width = student_map.shape
    num = height * width
    k = min(int(num_samples), num)
    student = student_map.flatten(2).transpose(1, 2)
    teacher = teacher_map.flatten(2).transpose(1, 2)
    if mask is None:
        weights = torch.ones(bsz, num, device=student.device, dtype=student.dtype)
    else:
        weights = mask.reshape(bsz, num)
    index = torch.topk(weights, k, dim=1).indices
    gather = index.unsqueeze(-1).expand(-1, -1, channels)
    s_sel = F.normalize(student.gather(1, gather), dim=-1)
    t_sel = F.normalize(teacher.gather(1, gather).detach(), dim=-1)
    s_rel = torch.bmm(s_sel, s_sel.transpose(1, 2))
    t_rel = torch.bmm(t_sel, t_sel.transpose(1, 2))
    return (s_rel - t_rel).square().mean()


def attention_imitation(
    student_map: torch.Tensor,
    teacher_map: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """DistillBEV spatial-attention L1: match where each BEV is 'looking'."""
    tau = max(float(tau), 1e-3)
    def _attn(feat: torch.Tensor) -> torch.Tensor:
        pooled = feat.abs().mean(dim=1).flatten(1)
        return F.softmax(pooled / tau, dim=-1)

    return (_attn(student_map) - _attn(teacher_map.detach())).abs().mean()


def planning_look_kd(
    student_map: torch.Tensor,
    prior: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """KL between student BEV spatial energy and an aux-derived planning prior.

    ``student_map`` is ``[B, C, H, W]`` — the BEV the planner reads, before the
    frozen adapter.  ``prior`` is ``[B, 1, H, W]`` in ``[ε, 1]`` from corridor
    ∪ agents ∪ road/boundary.  DistillBEV-style "where to look", with the
    target taken from det/map GT instead of teacher energy.
    """
    if student_map.dim() != 4:
        raise ValueError(f"expected [B,C,H,W] student map, got {tuple(student_map.shape)}")
    if prior.shape[0] != student_map.shape[0] or prior.shape[-2:] != student_map.shape[-2:]:
        raise ValueError(
            f"prior shape {tuple(prior.shape)} does not match student "
            f"{tuple(student_map.shape)}"
        )
    tau = max(float(tau), 1e-3)
    energy = student_map.abs().mean(dim=1).flatten(1)
    log_s = F.log_softmax(energy / tau, dim=-1)
    target = prior.reshape(energy.shape).clamp_min(1e-6)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (target * (target.clamp_min(1e-8).log() - log_s)).sum(dim=-1).mean()


def _kl_logits(student: torch.Tensor, teacher: torch.Tensor, tau: float) -> torch.Tensor:
    tau = max(float(tau), 1e-3)
    log_s = F.log_softmax(student / tau, dim=-1)
    p_t = F.softmax(teacher.detach() / tau, dim=-1)
    return (p_t * (p_t.clamp_min(1e-8).log() - log_s)).sum(dim=-1).mean() * (tau * tau)


def head_response_kd(
    student_pred: Dict[str, torch.Tensor],
    teacher_pred: Dict[str, torch.Tensor],
    *,
    cls_key: str,
    geom_key: Optional[str] = None,
    tau: float = 2.0,
    extra_keys: Sequence[str] = (),
) -> torch.Tensor:
    """Same-query response distillation (MapDistill head / DualDistill CRD).

    Classification/geometry keys that are stacked over decoder layers should
    already be sliced to the last layer by the caller.  Teacher tensors are
    detached inside this function.

    Geometry and extra keys are optional.  Do not pass unmatched DETR
    ``traj_preds`` / metric boxes here: mean L1 on those is unbounded and
    dominated Stage-2 loss in the first improved run.
    """
    if cls_key not in student_pred or cls_key not in teacher_pred:
        raise KeyError(f"head KD missing classification key {cls_key!r}")
    loss = _kl_logits(student_pred[cls_key], teacher_pred[cls_key], tau)
    if geom_key:
        if geom_key not in student_pred or geom_key not in teacher_pred:
            raise KeyError(f"head KD missing geometry key {geom_key!r}")
        loss = loss + (
            student_pred[geom_key] - teacher_pred[geom_key].detach()
        ).abs().mean()
    for key in extra_keys:
        if key in student_pred and key in teacher_pred:
            loss = loss + (student_pred[key] - teacher_pred[key].detach()).abs().mean()
    return loss
