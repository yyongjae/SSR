"""Train-only BEV registers for selective dual-teacher distillation.

The deployed student is unchanged. These registers are not a planner and are
not read at inference. They choose where a frozen teacher supervises
``bev_embed``, on top of the student's own plan / det / motion / map losses.

The attention that picks cells is trained by planner-attention coverage and a
spatial diversity penalty. The feature match uses that attention detached, so
the registers cannot lower the match loss by sliding onto easy cells.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import bev_cell_centers


class BEVRegisterSelector(nn.Module):
    """Two banks of persistent registers over one student BEV grid.

    Bank 0 reads the BEVFusion cache. Bank 1 reads the ReSMap cache. Both
    banks see the same front-only ``50 x 100`` grid the student already uses.
    """

    def __init__(
        self,
        channels: int = 256,
        num_registers: int = 16,
        num_commands: int = 4,
        bev_h: int = 50,
        bev_w: int = 100,
        pc_range: Sequence[float] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        div_sigma_m: float = 4.0,
        tok_warmup_steps: int = 53195,
    ) -> None:
        super().__init__()
        if num_registers < 2:
            raise ValueError("diversity needs at least 2 registers per bank")
        self.channels = int(channels)
        self.num_registers = int(num_registers)
        self.num_commands = int(num_commands)
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.pc_range = tuple(float(v) for v in pc_range)
        self.div_sigma_m = float(div_sigma_m)
        self.tok_warmup_steps = max(int(tok_warmup_steps), 1)

        registers = torch.randn(2, self.num_registers, self.channels) * 1e-6
        self.registers = nn.Parameter(registers)
        self.cmd_to_channel = nn.Linear(self.num_commands, self.channels)
        self.status_to_channel = nn.Linear(4, self.channels)
        self.query_proj = nn.Linear(self.channels, self.channels)
        self.key_proj = nn.Linear(self.channels, self.channels)
        self.register_buffer("steps", torch.zeros((), dtype=torch.long), persistent=True)
        grid_y, grid_x = bev_cell_centers(
            self.pc_range, self.bev_h, self.bev_w, device=torch.device("cpu"), dtype=torch.float32,
        )
        coords = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        self.register_buffer("cell_xy", coords, persistent=False)

    def ramp(self) -> torch.Tensor:
        """0 at step 0, 1 once ``tok_warmup_steps`` forwards have run."""
        return (self.steps.float() / float(self.tok_warmup_steps)).clamp(0.0, 1.0)

    def command_prior(self, command: torch.Tensor) -> torch.Tensor:
        """Wide forward wedge in SSR metres. Not the GT trajectory.

        Command order matches the planner: left, straight, right, unknown.
        ``x`` is right, ``y`` is forward. Left looks toward negative ``x``.
        """
        if command.dim() != 2 or command.size(-1) != self.num_commands:
            raise ValueError(
                f"command must be [B, {self.num_commands}], got {tuple(command.shape)}"
            )
        index = command.argmax(dim=-1)
        x_mean = command.new_tensor([-10.0, 0.0, 10.0, 0.0])[index]
        xy = self.cell_xy.to(device=command.device, dtype=command.dtype)
        x = xy[:, 0].unsqueeze(0)
        y = xy[:, 1].unsqueeze(0)
        peak = torch.exp(
            -0.5 * ((x - x_mean.unsqueeze(1)) / 12.0) ** 2
            - 0.5 * ((y - 16.0) / 10.0) ** 2
        )
        peak = peak / peak.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        prior = 0.05 + 0.95 * peak
        return prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def attend(
        self,
        bev: torch.Tensor,
        command: torch.Tensor,
        ego_status: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return attention ``[B, 2, R, HW]``. ``bev`` is detached inside."""
        batch, num_tokens, channels = bev.shape
        if num_tokens != self.bev_h * self.bev_w or channels != self.channels:
            raise ValueError(
                f"expected bev [B, {self.bev_h * self.bev_w}, {self.channels}], "
                f"got {tuple(bev.shape)}"
            )
        if ego_status is None:
            status = bev.new_zeros((batch, 4))
        else:
            status = ego_status.to(device=bev.device, dtype=bev.dtype)
            if status.shape != (batch, 4):
                raise ValueError(f"ego_status must be [B, 4], got {tuple(status.shape)}")
        cmd = command.to(device=bev.device, dtype=bev.dtype)
        cond = self.cmd_to_channel(cmd) + self.status_to_channel(status)
        slots = self.registers.unsqueeze(0) + cond[:, None, None, :]
        query = self.query_proj(slots)
        key = self.key_proj(bev.detach())
        scale = self.channels ** -0.5
        logits = torch.einsum("bark,bnk->barn", query, key) * scale
        return logits.softmax(dim=-1)

    def forward(
        self,
        student_bev: torch.Tensor,
        teacher_by_bank: Sequence[torch.Tensor],
        plan_attn: torch.Tensor,
        command: torch.Tensor,
        ego_status: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Match pooled tokens and train the registers on coverage.

        ``teacher_by_bank`` is ``(bevfusion, resmap)``, each ``[B, HW, C]``
        already on the student grid and without gradient.
        """
        if len(teacher_by_bank) != 2:
            raise ValueError("selector expects BEVFusion then ReSMap")
        attention = self.attend(student_bev, command, ego_status)
        ramp = self.ramp()
        prior = self.command_prior(command)
        target_p = _normalize_rows(plan_attn.detach().to(dtype=student_bev.dtype))
        cover_target = (1.0 - ramp) * prior + ramp * target_p
        cover_target = _normalize_rows(cover_target)

        names = ("bevfusion", "resmap")
        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {"distill_selector_ramp": ramp.detach()}
        cover_terms = []
        token_terms = []
        attentions = []
        for bank, name in enumerate(names):
            weights = attention[:, bank]
            attentions.append(weights)
            cover_terms.append(coverage_loss(weights, cover_target))
            teacher = teacher_by_bank[bank]
            if teacher.shape != student_bev.shape:
                raise ValueError(
                    f"{name} tokens {tuple(teacher.shape)} != student {tuple(student_bev.shape)}"
                )
            token_terms.append(token_match_loss(student_bev, teacher.detach(), weights, target_p))
            with torch.no_grad():
                gathered_s = torch.matmul(weights, student_bev)
                gathered_t = torch.matmul(weights, teacher)
                metrics[f"distill_sel_cos/{name}"] = F.cosine_similarity(
                    _layernorm(gathered_s), _layernorm(gathered_t), dim=-1
                ).mean()

        stacked = torch.cat(attentions, dim=1)
        losses["loss_distill_cover"] = torch.stack(cover_terms).mean()
        losses["loss_distill_div"] = diversity_loss(stacked, self.cell_xy, self.div_sigma_m)
        losses["loss_distill_tok"] = ramp * torch.stack(token_terms).mean()
        with torch.no_grad():
            metrics.update(_selection_metrics(stacked, self.cell_xy))
        self.steps += 1
        return losses, metrics


def _layernorm(tokens: torch.Tensor) -> torch.Tensor:
    """Channel LayerNorm with no affine, so a per-cell MLP is not in the match."""
    return F.layer_norm(tokens, (tokens.size(-1),), weight=None, bias=None)


def _normalize_rows(mass: torch.Tensor) -> torch.Tensor:
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def coverage_loss(attention: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``attention`` is ``[B, R, HW]``. Its sum should follow ``target`` ``[B, HW]``."""
    mass = _normalize_rows(attention.sum(dim=1))
    goal = _normalize_rows(target)
    return (1.0 - F.cosine_similarity(mass, goal, dim=-1)).mean()


def token_match_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    attention: torch.Tensor,
    plan_attn: torch.Tensor,
) -> torch.Tensor:
    """Pooled LN match. Attention is detached, so this loss does not move it.

    ``student`` and ``teacher`` are ``[B, HW, C]``. Registers the planner does
    not read contribute nothing, via ``w_k = <A_k, P>``.
    """
    weights = attention.detach()
    student_tok = _layernorm(torch.matmul(weights, student))
    teacher_tok = _layernorm(torch.matmul(weights, teacher))
    per_register = (student_tok - teacher_tok).pow(2).mean(dim=-1)
    importance = (weights * plan_attn.detach().unsqueeze(1)).sum(dim=-1)
    importance = importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (importance * per_register).sum(dim=-1).mean()


def diversity_loss(
    attention: torch.Tensor,
    cell_xy: torch.Tensor,
    sigma_m: float,
) -> torch.Tensor:
    """Repel spatial means and attention-map clones. ``attention`` is ``[B, K, HW]``."""
    xy = cell_xy.to(device=attention.device, dtype=attention.dtype)
    means = torch.matmul(attention, xy)
    delta = means.unsqueeze(2) - means.unsqueeze(1)
    dist2 = delta.pow(2).sum(dim=-1)
    attract = torch.exp(-dist2 / (2.0 * float(sigma_m) ** 2))
    eye = torch.eye(attract.size(-1), device=attract.device, dtype=torch.bool)
    spatial = attract.masked_fill(eye.unsqueeze(0), 0.0).sum(dim=(1, 2))
    spatial = spatial / float(attract.size(-1) * (attract.size(-1) - 1))
    flat = F.normalize(attention, dim=-1)
    cosine = torch.matmul(flat, flat.transpose(1, 2)).masked_fill(eye.unsqueeze(0), 0.0)
    angular = cosine.pow(2).sum(dim=(1, 2)) / float(attention.size(1) * (attention.size(1) - 1))
    return (spatial + angular).mean()


def _selection_metrics(attention: torch.Tensor, cell_xy: torch.Tensor) -> Dict[str, torch.Tensor]:
    mass = attention.sum(dim=1)
    xy = cell_xy.to(device=attention.device, dtype=attention.dtype)
    means = torch.matmul(attention, xy)
    delta = means.unsqueeze(2) - means.unsqueeze(1)
    dist = delta.pow(2).sum(dim=-1).clamp_min(0.0).sqrt()
    eye = torch.eye(dist.size(-1), device=dist.device, dtype=torch.bool)
    nearest = dist.masked_fill(eye.unsqueeze(0), 1e6).amin(dim=-1)
    prob = attention.clamp_min(1e-8)
    entropy = -(prob * prob.log()).sum(dim=-1)
    return {
        "distill_sel_mass_frac": (mass > 0.01).float().mean(),
        "distill_sel_entropy": entropy.mean(),
        "distill_sel_nearest_m": nearest.mean(),
    }
