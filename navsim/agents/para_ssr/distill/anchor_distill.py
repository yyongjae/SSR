"""Trajectory-Anchor-Guided Interaction Distillation (BEV Selector v6 & v7).

v7 Upgrade:
- 2D Positional Encoding on BEV Keys (Spatial Coordinate Grounding)
- Soft Relative Spatial Gaussian Distance Bias per Trajectory Anchor
- Teacher-Guided Attention Map Distillation (KL Divergence on Attention Weights)
- Attended Feature Distillation (Hybrid L2 + Cosine)
- ReSMap Domain Loss Scaling (3.0x multiplier to eliminate gradient starvation)
- Sharpened Winner & Softmax Importance Weighting (tau=1.0, winner_boost=2.0)
"""

from typing import Dict, Optional, Sequence, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer_blocks import LearnedPositionalEncoding


class TrajectoryAnchorDistillation(nn.Module):
    """Trajectory-Anchor-Guided Interaction Distillation module (v6, v7, v8, v9, v10, v11)."""

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
        spatial_sigmas: Optional[Sequence[float]] = None,
        lambda_gt: float = 0.6,
        lambda_cov: float = 0.2,
        lambda_risk: float = 0.2,
        tau_gt: float = 5.0,
        tau_risk: float = 8.0,
        loss_type: str = "hybrid",
        use_proj: bool = True,
        mode: str = "cross_attn",
        version: str = "v8",
        trajectory_anchors_file: Optional[str] = "/workspace/byounggun/SSR/data/planning_vb/trajectory_anchors_256.npy",
        probe_warmup_steps: int = 1000,
        probe_ramp_steps: int = 2000,
        probe_loss_weight: float = 1.0,
        probe_use_ema: bool = True,
        probe_ema_decay: float = 0.99,
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
        if spatial_sigmas is not None:
            self.spatial_sigmas = tuple(float(s) for s in spatial_sigmas)
        else:
            self.spatial_sigmas = (2.0, 4.0, 8.0, 16.0)
        self.lambda_gt = float(lambda_gt)
        self.lambda_cov = float(lambda_cov)
        self.lambda_risk = float(lambda_risk)
        self.tau_gt = float(tau_gt)
        self.tau_risk = float(tau_risk)
        self.loss_type = str(loss_type).lower()
        self.use_proj = bool(use_proj)
        self.mode = str(mode).lower()
        self.version = str(version).lower()
        self.probe_warmup_steps = int(probe_warmup_steps)
        self.probe_ramp_steps = int(probe_ramp_steps)
        self.probe_loss_weight = float(probe_loss_weight)
        self.probe_use_ema = bool(probe_use_ema)
        self.probe_ema_decay = float(probe_ema_decay)

        # 256 Anchor Query Generator: embeds [fut_ts=8 * traj_dims=3] into C
        self.anchor_mlp = nn.Sequential(
            nn.Linear(8 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.channels),
        )
        self.cmd_mlp = nn.Linear(4, self.channels)
        self.status_mlp = nn.Linear(4, self.channels)
        self.query_norm = nn.LayerNorm(self.channels)

        # Cross-Attention interaction decoders:
        # Bank 0: Obstacle interaction (BEVFusion)
        # Bank 1: Road Boundary / Map interaction (ReSMap)
        self.probes: Optional[nn.ModuleList] = None
        if self.mode == "cross_attn":
            if self.version in ("v9", "v10", "v11"):
                from .planning_probe import PlanningGroundedProbe

                self.probes = nn.ModuleList([
                    PlanningGroundedProbe(
                        domain="bevfusion",
                        channels=self.channels,
                        num_heads=self.num_heads,
                        use_ema=self.probe_use_ema,
                        ema_decay=self.probe_ema_decay,
                        version=self.version,
                    ),
                    PlanningGroundedProbe(
                        domain="resmap",
                        channels=self.channels,
                        num_heads=self.num_heads,
                        use_ema=self.probe_use_ema,
                        ema_decay=self.probe_ema_decay,
                        version=self.version,
                    ),
                ])
                self.register_buffer("micro_step", torch.zeros((), dtype=torch.long), persistent=False)
            elif self.version in ("v7", "v8"):
                # v7/v8: Explicit Q, K, V projections for asymmetric teacher-guided alignment
                self.q_proj = nn.Linear(self.channels, self.channels)
                self.k_proj = nn.ModuleList([
                    nn.Linear(self.channels, self.channels) for _ in range(2)
                ])
                self.v_proj = nn.ModuleList([
                    nn.Linear(self.channels, self.channels) for _ in range(2)
                ])
                self.out_proj = nn.ModuleList([
                    nn.Linear(self.channels, self.channels) for _ in range(2)
                ])
            else:
                # v6 fallback: standard MultiheadAttention
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
                if self.version == "v11":
                    # v11 Fix 2: Zero-initialize 1x1 projection adapter (matching ck/bev_kd.py init="zero")
                    # so step-0 gradient into BEV is identically zero and no arbitrary identity channel mapping is forced.
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                else:
                    nn.init.eye_(proj.weight)
                    nn.init.zeros_(proj.bias)
        else:
            self.student_proj = None

        # v10/v11 Fix 5: Per-channel frozen Teacher Z-Score statistics (norm.npz for BEVFusion, norm_map.npz for ReSMap),
        # matching ck/bev_kd.py TeacherZ so heavy-tailed post-ReLU BEVFusion channels are standardized per channel.
        self.ratio_ctrl = None
        if self.version in ("v10", "v11"):
            import numpy as np
            from ..ck.bev_kd_arm import TeacherRatioControllers
            default_norms = (
                "/home/external-user/ssd/yongjae_refiner/ck/ck2/train/ck2T10/norm.npz",
                "/home/external-user/ssd/yongjae_refiner/ck/ck2/train/ck2M10/norm_map.npz",
            )
            for bank_idx, norm_path in enumerate(default_norms):
                if os.path.isfile(norm_path):
                    with np.load(norm_path, allow_pickle=False) as zf:
                        zm = torch.as_tensor(zf["mean"].astype(np.float32)).view(1, 1, self.channels)
                        zs = torch.as_tensor(zf["std"].astype(np.float32)).clamp_min(1e-6).view(1, 1, self.channels)
                    self.register_buffer(f"z_mean_{bank_idx}", zm, persistent=False)
                    self.register_buffer(f"z_std_{bank_idx}", zs, persistent=False)
                else:
                    self.register_buffer(f"z_mean_{bank_idx}", None, persistent=False)
                    self.register_buffer(f"z_std_{bank_idx}", None, persistent=False)
            self.ratio_ctrl = TeacherRatioControllers(
                ("det", "map"),
                ratio=0.10,
                cap=0.25,
                m=0.9,
                start_mb=self.probe_warmup_steps,
            )
        self.last_unit_inputs: Dict[str, torch.Tensor] = {}
        self.last_unit_losses: Dict[str, torch.Tensor] = {}

        # Precompute spatial distance bias for 256 anchors if anchors file is available
        if trajectory_anchors_file is not None and os.path.exists(trajectory_anchors_file):
            import numpy as np
            anchors_np = np.load(trajectory_anchors_file)
            anchors_t = torch.tensor(anchors_np, dtype=torch.float32)
            bias = self._build_spatial_bias(anchors_t)
            # fusion_v1: not persistent -- a pure function of the anchors (20.5 MB per checkpoint otherwise);
            # bias_anchors records which anchors built it (sync_spatial_bias)
            self.register_buffer("spatial_bias", bias, persistent=False)
            self.register_buffer("bias_anchors", anchors_t.clone(), persistent=False)
        elif trajectory_anchors_file is not None and str(version).lower() in ("v7", "v8", "v9", "v10", "v11"):
            # fusion_v1: a missing file used to leave a plain attribute and fail at the
            # first forward (register_buffer KeyError); fail at construction instead.
            raise FileNotFoundError(f"trajectory anchors file not found: {trajectory_anchors_file}")
        else:
            self.register_buffer("spatial_bias", None)
        if not hasattr(self, "bias_anchors"):
            self.register_buffer("bias_anchors", None)

    @torch.no_grad()
    def sync_spatial_bias(self, anchors: torch.Tensor) -> bool:
        """fusion_v1: keep the spatial bias a function of the planner's own anchors.  A checkpoint restore replaces the
        planner's anchor buffer but not this non-persistent bias, so a changed anchor set would silently disagree.
        Rebuilt in place (same shape, so DDP's buffer list stays valid).  Returns True when it was rebuilt."""
        if self.spatial_bias is None or self.bias_anchors is None:
            return False
        a = anchors.detach().to(device=self.bias_anchors.device, dtype=self.bias_anchors.dtype)
        if a.shape != self.bias_anchors.shape:
            raise ValueError(f"planner anchors {tuple(a.shape)} != spatial-bias anchors {tuple(self.bias_anchors.shape)}")
        if torch.equal(a, self.bias_anchors):
            return False
        self.spatial_bias.copy_(self._build_spatial_bias(a).to(self.spatial_bias))
        self.bias_anchors.copy_(a)
        return True

    def _build_spatial_bias(self, anchors: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian spatial distance bias [K, HW] (v7) or [H, K, HW] (v8/v9/v10/v11) for 256 anchors."""
        device = anchors.device
        r = torch.arange(self.bev_h, dtype=torch.float32, device=device)
        c = torch.arange(self.bev_w, dtype=torch.float32, device=device)
        y_min, y_max = self.pc_range[1], self.pc_range[4]
        x_min, x_max = self.pc_range[0], self.pc_range[3]
        grid_y = y_min + (r + 0.5) * ((y_max - y_min) / self.bev_h)
        grid_x = x_min + (c + 0.5) * ((x_max - x_min) / self.bev_w)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid_pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # [HW, 2]

        # Convert anchors to SSR coordinates: x_ssr = -y_left, y_ssr = x_fwd
        x_ssr = -anchors[..., 1]
        y_ssr = anchors[..., 0]
        wp_pts = torch.stack([x_ssr, y_ssr], dim=-1)  # [K, T, 2]

        dists = torch.norm(wp_pts.unsqueeze(2) - grid_pts.unsqueeze(0).unsqueeze(0), dim=-1)  # [K, T, HW]
        min_dist = dists.min(dim=1).values  # [K, HW]

        if self.version in ("v8", "v9", "v10", "v11"):
            sigmas = list(self.spatial_sigmas)
            if len(sigmas) != self.num_heads:
                sigmas = [sigmas[i % len(sigmas)] for i in range(self.num_heads)]
            biases = []
            for s in sigmas:
                biases.append(-0.5 * (min_dist / max(float(s), 1e-4)) ** 2)
            spatial_bias = torch.stack(biases, dim=0)  # [H, K, HW]
        else:
            spatial_bias = -0.5 * (min_dist / max(self.spatial_sigma, 1e-4)) ** 2  # [K, HW]
        return spatial_bias

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
        elif self.version in ("v9", "v10", "v11"):
            q = q + 0.0 * (self.cmd_mlp.weight.sum() + self.cmd_mlp.bias.sum())

        if ego_status is not None:
            stat = ego_status[..., :4].to(device=device, dtype=dtype)
            q = q + self.status_mlp(stat).unsqueeze(1)
        elif self.version in ("v9", "v10", "v11"):
            q = q + 0.0 * (self.status_mlp.weight.sum() + self.status_mlp.bias.sum())

        return self.query_norm(q)

    def _compute_anchor_weights(
        self,
        anchors: torch.Tensor,
        gt_trajectory: torch.Tensor,
        sim_reward: Optional[torch.Tensor] = None,
        domain: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Softmax + Uniform Coverage + Risk importance weights for 256 anchors."""
        bs = gt_trajectory.size(0)
        k = anchors.size(0)
        flat_anchors = anchors[..., :2].reshape(1, k, -1).to(device=gt_trajectory.device, dtype=gt_trajectory.dtype)
        flat_gt = gt_trajectory[..., :2].reshape(bs, 1, -1)

        # L2 distance across all waypoints
        dist = torch.norm(flat_anchors - flat_gt, dim=-1)  # [B, K]
        winner = dist.argmin(dim=-1)  # [B]

        if self.version in ("v8", "v9", "v10", "v11"):
            # 1. Relaxed GT softmax: tau_gt (v8: 5.0, v9/v10: 2.0, v11: 4.0)
            w_gt = F.softmax(-dist / max(self.tau_gt, 1e-4), dim=-1)  # [B, K]

            # 2. Uniform coverage across all 256 anchors
            w_cov = torch.full((bs, k), 1.0 / k, device=dist.device, dtype=dist.dtype)

            # 3. Hard negative / Risk exploration term r_k
            risk_proximity = F.softmax(-dist / max(self.tau_risk, 1e-4), dim=-1)
            if sim_reward is not None and sim_reward.dim() >= 3 and sim_reward.size(-1) == k:
                # sim_reward: [B, 5, K], 0=NC, 1=DAC, 2=EP, 3=TTC, 4=Comfort
                nc = sim_reward[:, 0, :].to(device=dist.device, dtype=dist.dtype).clamp(0.0, 1.0)
                dac = sim_reward[:, 1, :].to(device=dist.device, dtype=dist.dtype).clamp(0.0, 1.0)
                if self.version == "v11" and domain in ("bevfusion", "resmap"):
                    ttc = sim_reward[:, 3, :].to(device=dist.device, dtype=dist.dtype).clamp(0.0, 1.0)
                    if domain == "bevfusion":
                        # v11 Fix 3 (DET): Explore anchors that collide or have TTC hazard with obstacles,
                        # prioritizing obstacles inside the drivable road (0.5 + 0.5 * DAC).
                        risk_factor = (1.0 - (nc * ttc).clamp(0.0, 1.0)) * (0.5 + 0.5 * dac)
                    else:
                        # v11 Fix 3 (MAP): Explore anchors that cross road/lane drivable area boundaries (1 - DAC).
                        risk_factor = 1.0 - dac
                    r_k = risk_proximity * risk_factor
                    r_sum = r_k.sum(dim=-1, keepdim=True)
                    r_k = torch.where(r_sum > 1e-6, r_k / r_sum.clamp_min(1e-6), w_cov)
                else:
                    safety = (nc * dac).clamp(0.0, 1.0)
                    risk_factor = 1.0 - safety  # collision or DAC violation
                    r_k = risk_proximity * risk_factor
                    r_k = r_k / r_k.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            else:
                # Borderline distance-banded risk
                r_k = risk_proximity * (1.0 - w_gt)
                r_k = r_k / r_k.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            # Hybrid Weighting
            w = self.lambda_gt * w_gt + self.lambda_cov * w_cov + self.lambda_risk * r_k
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            # v7 legacy: sharpened softmax + winner boost
            w = F.softmax(-dist / self.tau, dim=-1)
            if self.winner_boost > 0.0:
                boost = torch.zeros_like(w).scatter_add(
                    1, winner.unsqueeze(1),
                    torch.full((bs, 1), self.winner_boost, device=w.device, dtype=w.dtype)
                )
                w = w + boost
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        entropy = -(w * torch.log(w.clamp_min(1e-12))).sum(dim=-1)
        n_eff = torch.exp(entropy).mean()
        return w, winner, n_eff

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
        bev_pos: Optional[torch.Tensor] = None,
        sim_reward: Optional[torch.Tensor] = None,
        sim_reward_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute trajectory-anchor interaction distillation loss.

        Args:
            student_bev: [B, HW, C]
            teacher_by_bank: (BEVFusion [B, HW, C], ReSMap [B, HW, C])
            trajectory_anchors: [K, T, 3]
            gt_trajectory: [B, T, 3] (optional)
            command: [B, 4] (optional)
            ego_status: [B, 4] (optional)
            bev_pos: [B, HW, C] (optional 2D positional encoding)
            sim_reward: [B, 5, K] (optional PDM simulation scores)
            sim_reward_valid: [B] (optional validity mask for sim_reward)
        """
        if len(teacher_by_bank) != 2:
            raise ValueError(f"TrajectoryAnchorDistillation expects 2 teachers, got {len(teacher_by_bank)}")

        bs = student_bev.size(0)
        device = student_bev.device
        dtype = student_bev.dtype
        k = trajectory_anchors.size(0)
        hw = student_bev.size(1)

        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}

        # Compute importance weights
        if gt_trajectory is not None and gt_trajectory.numel() > 0:
            weights, winner, n_eff = self._compute_anchor_weights(
                trajectory_anchors, gt_trajectory, sim_reward=sim_reward,
            )
        else:
            weights = torch.full((bs, k), 1.0 / k, device=device, dtype=dtype)
            winner = torch.zeros((bs,), device=device, dtype=torch.long)
            n_eff = torch.tensor(float(k), device=device, dtype=dtype)

        metrics["distill/effective_anchors"] = n_eff.detach()

        names = ("bevfusion", "resmap")
        total_distill_terms = []

        if self.mode == "cross_attn":
            # Generate 256 anchor queries conditioned on driving intent
            q = self._build_anchor_queries(
                trajectory_anchors, bs, command=command, ego_status=ego_status,
                device=device, dtype=dtype,
            )

            if bev_pos is None:
                raise ValueError("bev_pos must be provided by planner_head for TrajectoryAnchorDistillation")

            if self.version in ("v9", "v10", "v11"):
                # -------------------------------------------------------------
                # v9/v10/v11: Domain-Specific Planning-Grounded Probe (PGP) +
                #             PGP-Weighted Dense 1x1 Cell KD (v10/v11)
                # -------------------------------------------------------------
                if not hasattr(self, "spatial_bias") or self.spatial_bias is None:
                    spatial_bias = self._build_spatial_bias(trajectory_anchors)
                    self.register_buffer("spatial_bias", spatial_bias, persistent=False)

                if self.spatial_bias.dim() == 2:
                    bias = self.spatial_bias.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(1)
                else:
                    bias = self.spatial_bias.to(device=device, dtype=dtype).unsqueeze(0)

                step = int(self.micro_step.item())
                if self.training:
                    self.micro_step.add_(1)
                if step < self.probe_warmup_steps:
                    ramp = 0.0
                elif self.probe_ramp_steps > 0:
                    ramp = min(1.0, float(step - self.probe_warmup_steps + 1) / float(self.probe_ramp_steps))
                else:
                    ramp = 1.0
                metrics["distill/ramp"] = student_bev.new_tensor(ramp)

                import math
                max_ent = math.log(max(hw, 2))
                self.last_unit_inputs = {}
                self.last_unit_losses = {}
                tkey = {"bevfusion": "det", "resmap": "map"}

                for bank, name in enumerate(names):
                    probe = self.probes[bank]
                    t_bev = teacher_by_bank[bank].detach()
                    if self.version in ("v10", "v11"):
                        zm = getattr(self, f"z_mean_{bank}", None)
                        zs = getattr(self, f"z_std_{bank}", None)
                        if zm is not None and zs is not None:
                            t_bev = ((t_bev.float() - zm.to(device=device)) / zs.to(device=device)).to(dtype=dtype)
                    proj = self.student_proj[bank] if self.use_proj else None
                    if self.version in ("v10", "v11"):
                        # Scale-pinned per-cell LayerNorm (no affine) before 1x1 student_proj (matching ck/bev_kd.py)
                        # Put GradScale before student_proj so student_proj always gets 100% unscaled gradients
                        # while bev_embed gradient is gated by ramp * RatioController weight!
                        from ..ck.bev_kd_arm import GradScale
                        tk = tkey[name]
                        if self.ratio_ctrl is not None and self.ratio_ctrl[tk].n > 0:
                            lam_b = float(self.ratio_ctrl[tk].weight(step))
                            bev_scale = float(ramp) * lam_b
                        else:
                            lam_b = float(ramp)
                            bev_scale = float(ramp)
                        metrics[f"distill/lam_{tk}"] = student_bev.new_tensor(bev_scale)
                        x_unit = GradScale.apply(student_bev, bev_scale)
                        s_normed = F.layer_norm(x_unit, (self.channels,))
                        s_bev = proj(s_normed) if proj is not None else s_normed
                        s_pos = bev_pos.detach()
                    else:
                        x_unit = student_bev
                        s_bev = proj(student_bev) if proj is not None else student_bev
                        s_pos = bev_pos

                    # 1. Teacher Probe Pass (learns P_b via domain-grounded planning loss)
                    attn_t, feat_t, preds_t = probe.forward_teacher(q, t_bev, bev_pos, bias)
                    probe_loss, probe_metrics = probe.compute_probe_loss(
                        preds_t,
                        trajectory_anchors,
                        gt_trajectory=gt_trajectory,
                        sim_reward=sim_reward,
                        sim_reward_valid=sim_reward_valid,
                    )
                    losses[f"loss_probe_{name}"] = self.probe_loss_weight * probe_loss
                    metrics.update(probe_metrics)

                    domain_scale = self.resmap_scale if name == "resmap" else 1.0
                    if self.version == "v11":
                        # v11 Fix 1 + Fix 3: Domain-Specific Risk Exploration Anchor Weighting (w_det vs w_map)
                        # + Pure PGP-Weighted Pointwise 1x1 Cell KD (pooled feat_loss & attn_loss detached).
                        with torch.no_grad():
                            if gt_trajectory is not None and gt_trajectory.numel() > 0:
                                weights_b, _, n_eff_b = self._compute_anchor_weights(
                                    trajectory_anchors, gt_trajectory, sim_reward=sim_reward, domain=name,
                                )
                            else:
                                weights_b = weights
                                n_eff_b = n_eff
                            metrics[f"distill/effective_anchors_{name}"] = n_eff_b.detach()
                            if name == "bevfusion":
                                metrics["distill/effective_anchors"] = n_eff_b.detach()

                            ent_t = -(attn_t.detach() * attn_t.detach().clamp_min(1e-12).log()).sum(dim=-1)
                            metrics[f"probe/head_entropy/{name}"] = ent_t.mean().detach()
                            head_conf = (1.0 - ent_t / max_ent).clamp(0.1, 1.0)
                            head_w = head_conf / head_conf.mean(dim=1, keepdim=True).clamp_min(1e-6)
                            attn_hw = (head_w.unsqueeze(-1) * attn_t.detach()).mean(dim=1)  # [B, K, HW]
                            m_spatial = torch.bmm(weights_b.unsqueeze(1), attn_hw).squeeze(1)  # [B, HW]
                            m_norm = m_spatial * (float(hw) / m_spatial.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            w_pgp = 0.5 + 0.5 * m_norm
                            metrics[f"distill/w_pgp_max/{name}"] = w_pgp.max(dim=-1).values.mean().detach()
                            metrics[f"distill/w_pgp_std/{name}"] = w_pgp.std(dim=-1).mean().detach()

                            attn_s, feat_s, preds_s = probe.forward_student(q.detach(), s_bev.detach(), s_pos, bias)
                            feat_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t.detach(), weights_b)
                            kl_per_head = F.kl_div(
                                attn_s.clamp_min(1e-8).log(), attn_t.detach(), reduction="none"
                            ).sum(dim=-1)
                            kl = (head_w * kl_per_head).mean(dim=1)
                            attn_loss = (weights_b * kl).sum(dim=-1).mean()

                        cell_mse = (s_bev - t_bev).pow(2).mean(dim=-1)
                        cell_loss = (w_pgp * cell_mse).mean()
                        metrics[f"distill_cell_mse/{name}"] = cell_loss.detach()
                        metrics[f"distill_uniform_mse/{name}"] = cell_mse.mean().detach()
                        raw_bank_loss = self.loss_scale * cell_loss
                        bank_loss = raw_bank_loss
                        self.last_unit_inputs[tkey[name]] = x_unit
                        self.last_unit_losses[tkey[name]] = 0.5 * raw_bank_loss
                    else:
                        # 2. Student Stop-Grad Probe Pass (gradients flow ONLY to student_bev / student_proj)
                        attn_s, feat_s, preds_s = probe.forward_student(q, s_bev, s_pos, bias)

                        # 3. Subspace Attended Feature KD
                        feat_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t.detach(), weights)

                        # 4. Confidence-Weighted (Inverse-Entropy) Attention KD (v10 Fix 3)
                        kl_per_head = F.kl_div(
                            attn_s.clamp_min(1e-8).log(), attn_t.detach(), reduction="none"
                        ).sum(dim=-1)  # [B, H, K]
                        with torch.no_grad():
                            ent_t = -(attn_t.detach() * attn_t.detach().clamp_min(1e-12).log()).sum(dim=-1)
                            metrics[f"probe/head_entropy/{name}"] = ent_t.mean().detach()
                        if self.version == "v10":
                            with torch.no_grad():
                                head_conf = (1.0 - ent_t / max_ent).clamp(0.1, 1.0)
                                head_w = head_conf / head_conf.mean(dim=1, keepdim=True).clamp_min(1e-6)
                                # PGP Spatial Importance Map w_pgp(u) in [B, HW], spatial mean == 1.0:
                                # 0.5 uniform baseline (full-grid 1x1 conv KD) + 0.5 PGP anchor-focused importance
                                attn_hw = (head_w.unsqueeze(-1) * attn_t.detach()).mean(dim=1)  # [B, K, HW]
                                m_spatial = torch.bmm(weights.unsqueeze(1), attn_hw).squeeze(1)  # [B, HW]
                                m_norm = m_spatial * (float(hw) / m_spatial.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                                w_pgp = 0.5 + 0.5 * m_norm
                            kl = (head_w * kl_per_head).mean(dim=1)
                            cell_mse = (s_bev - t_bev).pow(2).mean(dim=-1)
                            cell_loss = (w_pgp * cell_mse).mean()
                            metrics[f"distill_cell_mse/{name}"] = cell_loss.detach()
                        else:
                            kl = kl_per_head.mean(dim=1)
                            cell_loss = feat_loss.new_zeros(())
                        attn_loss = (weights * kl).sum(dim=-1).mean()

                        if self.version == "v10":
                            # In v10, x_unit = GradScale(student_bev, ramp) already gates the gradient into student_bev,
                            # so student_proj trains at 100% weight even during warm-up (ramp == 0.0)!
                            raw_bank_loss = domain_scale * (
                                self.loss_scale * cell_loss
                                + 0.5 * self.loss_scale * feat_loss
                                + self.attn_loss_scale * attn_loss
                            )
                            bank_loss = raw_bank_loss
                            self.last_unit_inputs[tkey[name]] = x_unit
                            self.last_unit_losses[tkey[name]] = 0.5 * raw_bank_loss
                        else:
                            raw_bank_loss = domain_scale * (self.loss_scale * feat_loss + self.attn_loss_scale * attn_loss)
                            bank_loss = ramp * raw_bank_loss + 0.0 * raw_bank_loss

                    losses[f"loss_distill_{name}"] = 0.5 * bank_loss
                    metrics[f"loss_attn_{name}"] = (domain_scale * self.attn_loss_scale * attn_loss).detach()
                    total_distill_terms.append(raw_bank_loss)
                    metrics[f"distill_cos/{name}"] = bank_cos.detach()

                    with torch.no_grad():
                        winner_cos = F.cosine_similarity(
                            F.layer_norm(feat_s[torch.arange(bs), winner], (self.channels,)),
                            F.layer_norm(feat_t[torch.arange(bs), winner], (self.channels,)),
                            dim=-1,
                        ).mean()
                        metrics[f"distill_winner_cos/{name}"] = winner_cos.detach()
                        probs_t = preds_t["sim_logits"].detach().float().sigmoid()
                        probs_s = preds_s["sim_probs"].detach().float()
                        metrics[f"probe/student_gap/{name}"] = (probs_s - probs_t).abs().mean().detach()

            elif self.version in ("v7", "v8"):
                # -------------------------------------------------------------
                # v7/v8: Asymmetric Teacher-Guided Alignment with Multi-Scale Spatial Bias
                # -------------------------------------------------------------
                q_proj = self.q_proj(q).view(bs, k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, D]

                # Ensure spatial distance bias is available
                if not hasattr(self, "spatial_bias") or self.spatial_bias is None:
                    spatial_bias = self._build_spatial_bias(trajectory_anchors)
                    self.register_buffer("spatial_bias", spatial_bias, persistent=False)

                if self.spatial_bias.dim() == 2:
                    bias = self.spatial_bias.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(1)  # [1, 1, K, HW]
                else:
                    bias = self.spatial_bias.to(device=device, dtype=dtype).unsqueeze(0)              # [1, H, K, HW]

                scale = 1.0 / (self.head_dim ** 0.5)

                for bank, name in enumerate(names):
                    t_bev = teacher_by_bank[bank].detach()
                    proj = self.student_proj[bank] if self.use_proj else None
                    s_bev = proj(student_bev) if proj is not None else student_bev

                    # Keys with 2D Spatial Positional Encoding
                    k_t = self.k_proj[bank](t_bev + bev_pos).view(bs, hw, self.num_heads, self.head_dim).transpose(1, 2)
                    v_t = self.v_proj[bank](t_bev).view(bs, hw, self.num_heads, self.head_dim).transpose(1, 2)

                    k_s = self.k_proj[bank](s_bev + bev_pos).view(bs, hw, self.num_heads, self.head_dim).transpose(1, 2)
                    v_s = self.v_proj[bank](s_bev).view(bs, hw, self.num_heads, self.head_dim).transpose(1, 2)

                    # Attention Logits with Relative Spatial Bias
                    logits_t = torch.matmul(q_proj, k_t.transpose(-2, -1)) * scale + bias
                    logits_s = torch.matmul(q_proj, k_s.transpose(-2, -1)) * scale + bias

                    attn_t = F.softmax(logits_t, dim=-1)  # [B, H, K, HW]
                    attn_s = F.softmax(logits_s, dim=-1)

                    # Teacher-Attended Features
                    out_t = torch.matmul(attn_t, v_t).transpose(1, 2).contiguous().view(bs, k, self.channels)
                    feat_t = self.out_proj[bank](out_t)

                    out_s = torch.matmul(attn_s, v_s).transpose(1, 2).contiguous().view(bs, k, self.channels)
                    feat_s = self.out_proj[bank](out_s)

                    # 1. Attended Feature Loss (Hybrid L2 + Cosine)
                    feat_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t.detach(), weights)

                    # 2. Attention Alignment Loss (KL Divergence: Student matches Teacher Attention)
                    kl = F.kl_div(attn_s.clamp_min(1e-8).log(), attn_t.detach(), reduction="none").sum(dim=-1).mean(dim=1)
                    attn_loss = (weights * kl).sum(dim=-1).mean()

                    # Domain Scaling: ReSMap (3.0x) vs BEVFusion (1.0x)
                    domain_scale = self.resmap_scale if name == "resmap" else 1.0
                    bank_loss = domain_scale * (self.loss_scale * feat_loss + self.attn_loss_scale * attn_loss)

                    # Fix: Each bank contributes 0.5 * bank_loss so sum equals (det + map) / 2
                    losses[f"loss_distill_{name}"] = 0.5 * bank_loss
                    metrics[f"loss_attn_{name}"] = (domain_scale * self.attn_loss_scale * attn_loss).detach()
                    total_distill_terms.append(bank_loss)
                    metrics[f"distill_cos/{name}"] = bank_cos.detach()

                    with torch.no_grad():
                        winner_cos = F.cosine_similarity(
                            F.layer_norm(feat_s[torch.arange(bs), winner], (self.channels,)),
                            F.layer_norm(feat_t[torch.arange(bs), winner], (self.channels,)),
                            dim=-1,
                        ).mean()
                        metrics[f"distill_winner_cos/{name}"] = winner_cos.detach()

            else:
                # v6 fallback: MultiheadAttention without explicit guidance
                attns = (self.attn_det, self.attn_map)
                for bank, name in enumerate(names):
                    t_bev = teacher_by_bank[bank].detach()
                    proj = self.student_proj[bank] if self.use_proj else None
                    s_bev = proj(student_bev) if proj is not None else student_bev
                    attn_layer = attns[bank]

                    feat_t, _ = attn_layer(q, t_bev + bev_pos, t_bev)
                    feat_s, _ = attn_layer(q, s_bev + bev_pos, s_bev)

                    bank_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t, weights)
                    scaled_loss = self.loss_scale * bank_loss

                    losses[f"loss_distill_{name}"] = 0.5 * scaled_loss
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
            # Method 1: Spatial Trajectory Corridor Sampling
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
                domain_scale = self.resmap_scale if name == "resmap" else 1.0
                scaled_loss = domain_scale * self.loss_scale * bank_loss

                losses[f"loss_distill_{name}"] = 0.5 * scaled_loss
                total_distill_terms.append(scaled_loss)
                metrics[f"distill_cos/{name}"] = bank_cos.detach()

        # Fix: loss_distill_tok is purely for telemetry/W&B logging, NOT returned in losses to avoid 3x double counting
        metrics["loss_distill_tok"] = torch.stack(total_distill_terms).mean().detach()
        return losses, metrics
