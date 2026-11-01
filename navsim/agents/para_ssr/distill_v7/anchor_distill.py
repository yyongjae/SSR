"""Trajectory-anchor-guided teacher distillation, BEV selector v7 (teammate's method).

Ported from origin/aux_distill ``navsim/agents/para_ssr/distill/anchor_distill.py`` (v7 branch,
commit 8f6d88d, docs/bev_selector_v7.md).  Only the v7 path is kept; the math is unchanged:

    q_k        = LN(anchor_mlp(anchor_k) + cmd_mlp(command) + status_mlp(ego_status))   256 anchors
    M_k(x, y)  = -min_t |(x, y) - anchor_k,t|^2 / (2 sigma^2),  sigma = 4 m              fixed [K, HW]
    per bank (0 = BEVFusion, 1 = ReSMap), q/k/v/out projections shared by teacher and student:
        A_t = softmax(q K_t^T / sqrt(d) + M),  K_t = k_proj(t_bev + bev_pos),  V_t = v_proj(t_bev)
        A_s = softmax(q K_s^T / sqrt(d) + M),  K_s = k_proj(s_bev + bev_pos),  s_bev = student_proj(bev)
        feat = out_proj(A V)
        L_bank = scale_bank * (10 * hybrid(feat_s, feat_t.detach()) + 5 * KL(A_s || A_t.detach()))
        scale_bank = 3 for ReSMap, 1 for BEVFusion
    per-anchor weights w_k = normalise(softmax(-|anchor_k - gt| / tau) + winner_boost * 1[k = winner])
    loss_distill_tok = mean over banks

Training-only: nothing here is on the inference path.  The one addition is ``sample_valid``: a
frame the teacher cache does not have contributes nothing (the teammate drops such frames from
the dataset instead); with every frame valid the loss is exactly the teammate's.
"""
from typing import Dict, Optional, Sequence, Tuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BANKS = ("bevfusion", "resmap")


class TrajectoryAnchorDistillation(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        num_heads: int = 4,
        pc_range: Sequence[float] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        bev_h: int = 50,
        bev_w: int = 100,
        winner_boost: float = 2.0,
        tau: float = 1.0,
        loss_scale: float = 10.0,
        attn_loss_scale: float = 5.0,
        resmap_scale: float = 3.0,
        spatial_sigma: float = 4.0,
        loss_type: str = "hybrid",
        use_proj: bool = True,
        trajectory_anchors_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.pc_range = tuple(float(v) for v in pc_range)
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.winner_boost = float(winner_boost)
        self.tau = float(tau)
        self.loss_scale = float(loss_scale)
        self.attn_loss_scale = float(attn_loss_scale)
        self.resmap_scale = float(resmap_scale)
        self.spatial_sigma = float(spatial_sigma)
        self.loss_type = str(loss_type).lower()
        self.use_proj = bool(use_proj)

        self.anchor_mlp = nn.Sequential(nn.Linear(8 * 3, 128), nn.ReLU(inplace=True), nn.Linear(128, self.channels))
        self.cmd_mlp = nn.Linear(4, self.channels)
        self.status_mlp = nn.Linear(4, self.channels)
        self.query_norm = nn.LayerNorm(self.channels)

        self.q_proj = nn.Linear(self.channels, self.channels)
        self.k_proj = nn.ModuleList([nn.Linear(self.channels, self.channels) for _ in range(2)])
        self.v_proj = nn.ModuleList([nn.Linear(self.channels, self.channels) for _ in range(2)])
        self.out_proj = nn.ModuleList([nn.Linear(self.channels, self.channels) for _ in range(2)])

        if self.use_proj:
            self.student_proj = nn.ModuleList([nn.Linear(self.channels, self.channels) for _ in range(2)])
            for proj in self.student_proj:
                nn.init.eye_(proj.weight)
                nn.init.zeros_(proj.bias)
        else:
            self.student_proj = None

        if trajectory_anchors_file is not None and os.path.exists(trajectory_anchors_file):
            anchors = torch.tensor(np.load(trajectory_anchors_file), dtype=torch.float32)
            self.register_buffer("spatial_bias", self._build_spatial_bias(anchors))
        else:
            self.spatial_bias = None

    def _build_spatial_bias(self, anchors: torch.Tensor) -> torch.Tensor:
        """Gaussian distance bias [K, HW]: row-major over (bev_h = y_forward, bev_w = x_right)."""
        r = torch.arange(self.bev_h, dtype=torch.float32, device=anchors.device)
        c = torch.arange(self.bev_w, dtype=torch.float32, device=anchors.device)
        y_min, y_max = self.pc_range[1], self.pc_range[4]
        x_min, x_max = self.pc_range[0], self.pc_range[3]
        grid_y = y_min + (r + 0.5) * ((y_max - y_min) / self.bev_h)
        grid_x = x_min + (c + 0.5) * ((x_max - x_min) / self.bev_w)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid_pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)             # [HW, 2]
        wp = torch.stack([-anchors[..., 1], anchors[..., 0]], dim=-1)             # SSR: x = -y_left, y = x_fwd
        min_dist = torch.norm(wp.unsqueeze(2) - grid_pts[None, None], dim=-1).min(dim=1).values
        return -0.5 * (min_dist / max(self.spatial_sigma, 1e-4)) ** 2

    def _build_anchor_queries(self, anchors, batch_size, command=None, ego_status=None, device=None, dtype=None):
        k = anchors.size(0)
        flat = anchors[..., :3].reshape(1, k, -1).to(device=device, dtype=dtype)
        q = self.anchor_mlp(flat).expand(batch_size, -1, -1)
        if command is not None:
            cmd = command.to(device=device, dtype=dtype)
            if cmd.dim() == 1:
                cmd = F.one_hot(cmd, num_classes=4).to(dtype=dtype)
            q = q + self.cmd_mlp(cmd).unsqueeze(1)
        if ego_status is not None:
            q = q + self.status_mlp(ego_status[..., :4].to(device=device, dtype=dtype)).unsqueeze(1)
        return self.query_norm(q)

    def _compute_anchor_weights(self, anchors, gt_trajectory):
        bs, k = gt_trajectory.size(0), anchors.size(0)
        flat_anchors = anchors[..., :2].reshape(1, k, -1).to(device=gt_trajectory.device, dtype=gt_trajectory.dtype)
        dist = torch.norm(flat_anchors - gt_trajectory[..., :2].reshape(bs, 1, -1), dim=-1)   # [B, K]
        winner = dist.argmin(dim=-1)
        w = F.softmax(-dist / self.tau, dim=-1)
        if self.winner_boost > 0.0:
            w = w + torch.zeros_like(w).scatter_add(
                1, winner.unsqueeze(1), torch.full((bs, 1), self.winner_boost, device=w.device, dtype=w.dtype))
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6), winner

    @staticmethod
    def _batch_mean(per_sample: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
        if valid is None:
            return per_sample.mean()
        return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)

    def _feature_pair_loss(self, student_feat, teacher_feat, weights, valid=None):
        c = student_feat.size(-1)
        ln_s, ln_t = F.layer_norm(student_feat, (c,)), F.layer_norm(teacher_feat, (c,))
        l2 = ((ln_s - ln_t) ** 2).mean(dim=-1)
        cos_sim = F.cosine_similarity(ln_s, ln_t, dim=-1)
        per_anchor = 0.5 * l2 + 0.5 * (1.0 - cos_sim) if self.loss_type == "hybrid" else l2
        loss = self._batch_mean((weights * per_anchor).sum(dim=-1), valid)
        return loss, self._batch_mean((weights * cos_sim).sum(dim=-1), valid)

    def forward(
        self,
        student_bev: torch.Tensor,
        teacher_by_bank: Sequence[torch.Tensor],
        trajectory_anchors: torch.Tensor,
        gt_trajectory: Optional[torch.Tensor] = None,
        command: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
        bev_pos: Optional[torch.Tensor] = None,
        sample_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """student_bev [B, HW, C]; teacher_by_bank (BEVFusion, ReSMap) [B, HW, C]; anchors [K, T, 3];
        bev_pos [B, HW, C] (the planner's); sample_valid [B] (1 = teacher cache has the frame)."""
        if len(teacher_by_bank) != 2:
            raise ValueError(f"TrajectoryAnchorDistillation expects 2 teachers, got {len(teacher_by_bank)}")
        if bev_pos is None:
            raise ValueError("bev_pos must be provided by planner_head for TrajectoryAnchorDistillation")
        bs, hw = student_bev.size(0), student_bev.size(1)
        device, dtype = student_bev.device, student_bev.dtype
        k = trajectory_anchors.size(0)
        valid = None if sample_valid is None else sample_valid.to(device=device, dtype=dtype).view(bs)

        if gt_trajectory is not None and gt_trajectory.numel() > 0:
            weights, winner = self._compute_anchor_weights(trajectory_anchors, gt_trajectory)
        else:
            weights = torch.full((bs, k), 1.0 / k, device=device, dtype=dtype)
            winner = torch.zeros((bs,), device=device, dtype=torch.long)

        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        terms = []
        q = self._build_anchor_queries(trajectory_anchors, bs, command=command, ego_status=ego_status,
                                       device=device, dtype=dtype)
        q_proj = self.q_proj(q).view(bs, k, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, K, D]
        if self.spatial_bias is None:
            self.register_buffer("spatial_bias", self._build_spatial_bias(trajectory_anchors.float()))
        bias = self.spatial_bias.to(device=device, dtype=dtype)[None, None]                    # [1, 1, K, HW]
        scale = 1.0 / (self.head_dim ** 0.5)

        def heads(x):
            return x.view(bs, hw, self.num_heads, self.head_dim).transpose(1, 2)

        for bank, name in enumerate(BANKS):
            t_bev = teacher_by_bank[bank].detach()
            s_bev = self.student_proj[bank](student_bev) if self.use_proj else student_bev
            k_t, v_t = heads(self.k_proj[bank](t_bev + bev_pos)), heads(self.v_proj[bank](t_bev))
            k_s, v_s = heads(self.k_proj[bank](s_bev + bev_pos)), heads(self.v_proj[bank](s_bev))
            attn_t = F.softmax(torch.matmul(q_proj, k_t.transpose(-2, -1)) * scale + bias, dim=-1)
            attn_s = F.softmax(torch.matmul(q_proj, k_s.transpose(-2, -1)) * scale + bias, dim=-1)
            feat_t = self.out_proj[bank](torch.matmul(attn_t, v_t).transpose(1, 2).contiguous().view(bs, k, self.channels))
            feat_s = self.out_proj[bank](torch.matmul(attn_s, v_s).transpose(1, 2).contiguous().view(bs, k, self.channels))

            feat_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t.detach(), weights, valid)
            kl = F.kl_div(attn_s.clamp_min(1e-8).log(), attn_t.detach(), reduction="none").sum(dim=-1).mean(dim=1)
            attn_loss = self._batch_mean((weights * kl).sum(dim=-1), valid)

            domain_scale = self.resmap_scale if name == "resmap" else 1.0
            bank_loss = domain_scale * (self.loss_scale * feat_loss + self.attn_loss_scale * attn_loss)
            losses[f"loss_distill_{name}"] = bank_loss
            losses[f"loss_attn_{name}"] = (domain_scale * self.attn_loss_scale * attn_loss).detach()
            terms.append(bank_loss)
            metrics[f"distill_cos/{name}"] = bank_cos.detach()
            with torch.no_grad():
                ix = torch.arange(bs, device=device)
                wc = F.cosine_similarity(F.layer_norm(feat_s[ix, winner], (self.channels,)),
                                         F.layer_norm(feat_t[ix, winner], (self.channels,)), dim=-1)
                metrics[f"distill_winner_cos/{name}"] = self._batch_mean(wc, valid).detach()

        losses["loss_distill_tok"] = torch.stack(terms).mean()
        return losses, metrics
