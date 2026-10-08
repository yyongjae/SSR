"""Trajectory-Anchor-Guided Interaction Distillation (BEV Selector v6).

Method 2: 256 Anchor Query Cross-Attention (Interaction-Aware Selection)
Directly guides the student's 256 trajectory anchors by dynamically querying
the multi-modal teachers for interaction-relevant context across the BEV:
- Teacher 1 (BEVFusion): 3D Dynamic Obstacles & Collision context (NC)
- Teacher 2 (ReSMap): HD Map, Road Boundaries & Drivable Area context (DAC)

Rather than merely sampling points on the centerline, the 256 trajectory anchor
queries dynamically cross-attend to the entire BEV scene to discover all objects,
cut-in agents, and road boundaries that interact with each specific trajectory plan.

Winner-aware Softmax weighting focuses distillation capacity on viable paths,
eliminating the need for arbitrary learnable registers, coverage/diversity losses,
and closed-loop gradient balancers.
"""

from typing import Dict, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryAnchorDistillation(nn.Module):
    """v6 Trajectory-Anchor-Guided Interaction Distillation module."""

    def __init__(
        self,
        channels: int = 256,
        num_heads: int = 4,
        pc_range: Sequence[float] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        bev_h: int = 50,
        bev_w: int = 100,
        winner_boost: float = 1.0,
        tau: float = 2.0,
        loss_scale: float = 10.0,
        loss_type: str = "hybrid",
        use_proj: bool = True,
        mode: str = "cross_attn",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.pc_range = tuple(float(v) for v in pc_range)
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.winner_boost = float(winner_boost)
        self.tau = float(tau)
        self.loss_scale = float(loss_scale)
        self.loss_type = str(loss_type).lower()
        self.use_proj = bool(use_proj)
        self.mode = str(mode).lower()  # "cross_attn" (Method 2) or "corridor" (Method 1)

        # 256 Anchor Query Generator: embeds [fut_ts=8 * traj_dims=3] into C
        self.anchor_mlp = nn.Sequential(
            nn.Linear(8 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.channels),
        )
        self.cmd_mlp = nn.Linear(4, self.channels)
        self.status_mlp = nn.Linear(4, self.channels)

        # Cross-Attention interaction decoders:
        # Bank 0: Obstacle interaction (BEVFusion)
        # Bank 1: Road Boundary / Map interaction (ReSMap)
        if self.mode == "cross_attn":
            self.attn_det = nn.MultiheadAttention(
                self.channels, self.num_heads, batch_first=True
            )
            self.attn_map = nn.MultiheadAttention(
                self.channels, self.num_heads, batch_first=True
            )

        # Bank-specific Linear Projection Adapter for student feature alignment
        if self.use_proj:
            self.student_proj = nn.ModuleList([
                nn.Linear(self.channels, self.channels) for _ in range(2)
            ])
            for proj in self.student_proj:
                nn.init.eye_(proj.weight)
                nn.init.zeros_(proj.bias)
        else:
            self.student_proj = None

    def _build_anchor_queries(
        self,
        anchors: torch.Tensor,
        batch_size: int,
        command: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Encode 256 trajectory anchors into conditioning queries [B, 256, C]."""
        k = anchors.size(0)
        flat = anchors[..., :3].reshape(1, k, -1).to(device=device, dtype=dtype)
        q = self.anchor_mlp(flat).expand(batch_size, -1, -1)

        if command is not None:
            cmd = command.to(device=device, dtype=dtype)
            if cmd.dim() == 1:
                cmd = F.one_hot(cmd, num_classes=4).to(dtype=dtype)
            q = q + self.cmd_mlp(cmd).unsqueeze(1)

        if ego_status is not None:
            stat = ego_status[..., :4].to(device=device, dtype=dtype)
            q = q + self.status_mlp(stat).unsqueeze(1)

        return q

    def _compute_anchor_weights(
        self,
        anchors: torch.Tensor,
        gt_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Softmax + Winner importance weights for 256 anchors."""
        bs = gt_trajectory.size(0)
        k = anchors.size(0)
        flat_anchors = anchors[..., :2].reshape(1, k, -1).to(device=gt_trajectory.device, dtype=gt_trajectory.dtype)
        flat_gt = gt_trajectory[..., :2].reshape(bs, 1, -1)

        # L2 distance across all waypoints
        dist = torch.norm(flat_anchors - flat_gt, dim=-1)  # [B, K]
        winner = dist.argmin(dim=-1)  # [B]

        # Softmax over distance
        w = F.softmax(-dist / self.tau, dim=-1)

        # Winner anchor boost
        if self.winner_boost > 0.0:
            boost = torch.zeros_like(w).scatter_add(
                1, winner.unsqueeze(1),
                torch.full((bs, 1), self.winner_boost, device=w.device, dtype=w.dtype)
            )
            w = w + boost

        # Normalize weights so sum is 1.0 per sample
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return w, winner

    def _feature_pair_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted hybrid (L2 + Cosine) loss between student and teacher."""
        c = student_feat.size(-1)
        ln_s = F.layer_norm(student_feat, (c,))
        ln_t = F.layer_norm(teacher_feat, (c,))

        l2 = ((ln_s - ln_t) ** 2).mean(dim=-1)  # [B, K]
        cos_sim = F.cosine_similarity(ln_s, ln_t, dim=-1)  # [B, K]

        if self.loss_type == "hybrid":
            per_anchor_loss = 0.5 * l2 + 0.5 * (1.0 - cos_sim)
        else:
            per_anchor_loss = l2

        loss = (weights * per_anchor_loss).sum(dim=-1).mean()
        weighted_cos = (weights * cos_sim).sum(dim=-1).mean()
        return loss, weighted_cos

    def _navsim_to_bev_grid(
        self,
        anchors: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert NAVSIM trajectory coordinates [K, T, >=2] to normalized grid [-1, 1]."""
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]

        x_fwd = anchors[..., 0].to(device=device, dtype=dtype)
        y_left = anchors[..., 1].to(device=device, dtype=dtype)

        x_ssr = -y_left
        y_ssr = x_fwd

        u = ((x_ssr - x_min) / (x_max - x_min) * 2.0 - 1.0).clamp(-1.0, 1.0)
        v = ((y_ssr - y_min) / (y_max - y_min) * 2.0 - 1.0).clamp(-1.0, 1.0)
        return torch.stack([u, v], dim=-1)

    def forward(
        self,
        student_bev: torch.Tensor,
        teacher_by_bank: Sequence[torch.Tensor],
        trajectory_anchors: torch.Tensor,
        gt_trajectory: Optional[torch.Tensor] = None,
        command: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute trajectory-anchor interaction distillation loss.

        Args:
            student_bev: [B, HW, C]
            teacher_by_bank: (BEVFusion [B, HW, C], ReSMap [B, HW, C])
            trajectory_anchors: [K, T, 3]
            gt_trajectory: [B, T, 3] (optional)
            command: [B, 4] (optional)
            ego_status: [B, 4] (optional)
        """
        if len(teacher_by_bank) != 2:
            raise ValueError(f"TrajectoryAnchorDistillation expects 2 teachers, got {len(teacher_by_bank)}")

        bs = student_bev.size(0)
        device = student_bev.device
        dtype = student_bev.dtype

        # Compute importance weights
        if gt_trajectory is not None and gt_trajectory.numel() > 0:
            weights, winner = self._compute_anchor_weights(trajectory_anchors, gt_trajectory)
        else:
            k = trajectory_anchors.size(0)
            weights = torch.full((bs, k), 1.0 / k, device=device, dtype=dtype)
            winner = torch.zeros((bs,), device=device, dtype=torch.long)

        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        names = ("bevfusion", "resmap")
        total_distill_terms = []

        if self.mode == "cross_attn":
            # -------------------------------------------------------------
            # Method 2: Dynamic Cross-Attention Interaction Selection
            # -------------------------------------------------------------
            # Generate 256 anchor queries conditioned on driving intent
            q = self._build_anchor_queries(
                trajectory_anchors, bs, command=command, ego_status=ego_status,
                device=device, dtype=dtype,
            )

            attns = (self.attn_det, self.attn_map)
            for bank, name in enumerate(names):
                t_bev = teacher_by_bank[bank].detach()
                proj = self.student_proj[bank] if self.use_proj else None
                s_bev = proj(student_bev) if proj is not None else student_bev
                attn_layer = attns[bank]

                # Teacher extracts interaction features for all 256 trajectories
                feat_t, _ = attn_layer(q, t_bev, t_bev)
                # Student extracts interaction features for all 256 trajectories
                feat_s, _ = attn_layer(q, s_bev, s_bev)

                bank_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t, weights)
                scaled_loss = self.loss_scale * bank_loss

                losses[f"loss_distill_{name}"] = scaled_loss
                total_distill_terms.append(scaled_loss)
                metrics[f"distill_cos/{name}"] = bank_cos.detach()

                with torch.no_grad():
                    winner_cos = F.cosine_similarity(
                        F.layer_norm(feat_s[torch.arange(bs), winner], (self.channels,)),
                        F.layer_norm(feat_t[torch.arange(bs), winner], (self.channels,)),
                        dim=-1,
                    ).mean()
                    metrics[f"distill_winner_cos/{name}"] = winner_cos.detach()

        else:
            # -------------------------------------------------------------
            # Method 1: Spatial Trajectory Corridor Sampling
            # -------------------------------------------------------------
            grid_base = self._navsim_to_bev_grid(trajectory_anchors, device, dtype)
            grid = grid_base.unsqueeze(0).expand(bs, -1, -1, -1)

            for bank, name in enumerate(names):
                t_bev = teacher_by_bank[bank].detach()
                proj = self.student_proj[bank] if self.use_proj else None

                s_map = (proj(student_bev) if proj is not None else student_bev).view(
                    bs, self.bev_h, self.bev_w, self.channels
                ).permute(0, 3, 1, 2)
                t_map = t_bev.view(bs, self.bev_h, self.bev_w, self.channels).permute(0, 3, 1, 2)

                feat_s = F.grid_sample(s_map, grid, mode="bilinear", align_corners=True).permute(0, 2, 3, 1).mean(dim=2)
                feat_t = F.grid_sample(t_map, grid, mode="bilinear", align_corners=True).permute(0, 2, 3, 1).mean(dim=2)

                bank_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t, weights)
                scaled_loss = self.loss_scale * bank_loss

                losses[f"loss_distill_{name}"] = scaled_loss
                total_distill_terms.append(scaled_loss)
                metrics[f"distill_cos/{name}"] = bank_cos.detach()

        losses["loss_distill_tok"] = torch.stack(total_distill_terms).mean()
        return losses, metrics
