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
    """Trajectory-Anchor-Guided Interaction Distillation module (v6, v7, v8, v9, v10, v11, v12)."""

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
            if self.version in ("v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"):
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
                if self.version == "v12":
                    # v12: Damped identity init (0.1 * I) so step-0 student BEV has small norm (0.1 vs teacher 1.0)
                    # while preserving spatial variation across BEV cells from step 0 (avoiding zero-variance LayerNorm).
                    nn.init.eye_(proj.weight)
                    proj.weight.data.mul_(0.1)
                    nn.init.zeros_(proj.bias)
                elif self.version in ("v13", "v14", "v15", "v16"):
                    # v13/v14/v15/v16 Fix: Near-zero orthogonal init (gain=0.02, energy=0.0004) avoids v12's arbitrary identity
                    # channel prior while keeping full-rank spatial variation at step 0 so LayerNorm/cosine_similarity
                    # in feat_loss and diff_loss never divide by eps on step 0.
                    nn.init.orthogonal_(proj.weight, gain=0.02)
                    nn.init.zeros_(proj.bias)
                elif self.version == "v11":
                    # v11 Fix: Zero-initialize 1x1 projection adapter (matching ck/bev_kd.py init="zero")
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                else:
                    nn.init.eye_(proj.weight)
                    nn.init.zeros_(proj.bias)
        else:
            self.student_proj = None

        # v10/v11/v12/v13/v14/v15 Fix 5: Per-channel frozen Teacher Z-Score statistics (norm.npz for BEVFusion, norm_map.npz for ReSMap),
        # matching ck/bev_kd.py TeacherZ so heavy-tailed post-ReLU BEVFusion channels are standardized per channel.
        self.ratio_ctrl = None
        self.probe_ratio_ctrl = None
        if self.version in ("v10", "v11", "v12", "v13", "v14", "v15", "v16"):
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
            if self.version == "v15":
                # v15 Pillar 1: Rebalanced Dual-Stream Decoupled Gradient Controller
                # - Grid Stream (subspace-weighted cell_1x1 + edge_3x3): ratio=0.085, cap=0.22
                # - Probe Stream (feat + horizon_sibling_diff + score_kd + attn_kl): ratio=0.035, cap=0.08
                self.ratio_ctrl = TeacherRatioControllers(
                    ("det", "map"),
                    ratio=0.085,
                    cap=0.22,
                    m=0.9,
                    start_mb=self.probe_warmup_steps,
                )
                self.probe_ratio_ctrl = TeacherRatioControllers(
                    ("det", "map"),
                    ratio=0.035,
                    cap=0.08,
                    m=0.9,
                    start_mb=self.probe_warmup_steps,
                )
            elif self.version in ("v14", "v16"):
                # v14/v16 Fix 1: Dual-Stream Decoupled Gradient Controller
                # - Grid Stream (cell_1x1 + multi-dilation edge): ratio=0.090, cap=0.22 (restoring ck2e2e_bevkd spatial alignment budget)
                # - Probe Stream (feat + sibling_diff + attn_kl): ratio=0.025, cap=0.06 (+ late-epoch cosine taper in v16)
                self.ratio_ctrl = TeacherRatioControllers(
                    ("det", "map"),
                    ratio=0.09,
                    cap=0.22,
                    m=0.9,
                    start_mb=self.probe_warmup_steps,
                )
                self.probe_ratio_ctrl = TeacherRatioControllers(
                    ("det", "map"),
                    ratio=0.025,
                    cap=0.06,
                    m=0.9,
                    start_mb=self.probe_warmup_steps,
                )
            else:
                self.ratio_ctrl = TeacherRatioControllers(
                    ("det", "map"),
                    ratio=0.10,
                    cap=0.25,
                    m=0.9,
                    start_mb=self.probe_warmup_steps,
                )
        self.last_unit_inputs: Dict[str, torch.Tensor] = {}
        self.last_unit_losses: Dict[str, torch.Tensor] = {}
        self.last_probe_unit_inputs: Dict[str, torch.Tensor] = {}
        self.last_probe_unit_losses: Dict[str, torch.Tensor] = {}

        # Precompute spatial distance bias (and v12/v13/v14/v15 sibling pairs + horizon row gain) for 256 anchors
        if trajectory_anchors_file is not None and os.path.exists(trajectory_anchors_file):
            import numpy as np
            anchors_np = np.load(trajectory_anchors_file)
            anchors_t = torch.tensor(anchors_np, dtype=torch.float32)
            bias = self._build_spatial_bias(anchors_t)
            # fusion_v1: not persistent -- a pure function of the anchors (20.5 MB per checkpoint otherwise);
            # bias_anchors records which anchors built it (sync_spatial_bias)
            self.register_buffer("spatial_bias", bias, persistent=False)
            self.register_buffer("bias_anchors", anchors_t.clone(), persistent=False)
            if self.version in ("v12", "v13", "v14", "v15", "v16"):
                self.register_buffer("sibling_pairs", self._build_sibling_pairs(anchors_t), persistent=False)
                self.register_buffer("row_gain", self._build_row_gain(anchors_t.device), persistent=False)
        elif trajectory_anchors_file is not None and str(version).lower() in ("v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"):
            # fusion_v1: a missing file used to leave a plain attribute and fail at the
            # first forward (register_buffer KeyError); fail at construction instead.
            raise FileNotFoundError(f"trajectory anchors file not found: {trajectory_anchors_file}")
        else:
            self.register_buffer("spatial_bias", None)
        if not hasattr(self, "bias_anchors"):
            self.register_buffer("bias_anchors", None)
        if not hasattr(self, "sibling_pairs"):
            self.register_buffer("sibling_pairs", None, persistent=False)
        if not hasattr(self, "row_gain"):
            self.register_buffer("row_gain", None, persistent=False)

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
        if self.version in ("v12", "v13", "v14", "v15", "v16"):
            new_pairs = self._build_sibling_pairs(a).to(device=self.bias_anchors.device)
            if self.sibling_pairs is not None and self.sibling_pairs.shape == new_pairs.shape:
                self.sibling_pairs.copy_(new_pairs)
            else:
                self.register_buffer("sibling_pairs", new_pairs, persistent=False)
        self.bias_anchors.copy_(a)
        return True

    def _build_row_gain(self, device: torch.device) -> torch.Tensor:
        """Horizon density compensation gain in [1, HW].
        - v12: 0.65 + 0.90 * (y / 32.0)
        - v13/v14/v15/v16: Non-penalizing far-field boost 1.0 + 0.50 * clamp((y - 10.0) / 22.0, 0.0, 1.0) in [1.0, 1.5]
                           so near-field (0..10m) is never penalized below 1.0 while mid/far-field gets up to 1.5x boost.
        """
        r = torch.arange(self.bev_h, dtype=torch.float32, device=device)
        y_min, y_max = self.pc_range[1], self.pc_range[4]
        grid_y = y_min + (r + 0.5) * ((y_max - y_min) / self.bev_h)
        if self.version in ("v13", "v14", "v15", "v16"):
            gain_r = 1.0 + 0.50 * ((grid_y - 10.0) / max(y_max - 10.0, 1e-4)).clamp(0.0, 1.0)
        else:
            gain_r = 0.65 + 0.90 * (grid_y / max(y_max, 1e-4)).clamp(0.0, 1.0)
        return gain_r.unsqueeze(1).expand(self.bev_h, self.bev_w).reshape(1, -1)

    def _build_sibling_pairs(self, anchors: torch.Tensor) -> torch.Tensor:
        """v12/v13 Fix 2: Build velocity-sibling pairs (i_slow, j_fast) along the same geometric path.
        For each anchor i with terminal progress S_8 >= 1.0m, pairs it with the closest cross-track
        faster sibling j in [2.0m, 6.0m] (+0.5~1.0 m/s^2) and [5.0m, 12.0m] (+1.0~2.0 m/s^2).
        """
        pts0 = anchors[..., :2].detach().float()  # [K, T, 2]
        k = pts0.size(0)
        origin = torch.zeros((k, 1, 2), dtype=pts0.dtype, device=pts0.device)
        poly = torch.cat([origin, pts0], dim=1)  # [K, T+1, 2]
        diffs = poly[:, 1:, :] - poly[:, :-1, :]  # [K, T, 2]
        arc_len = torch.norm(diffs, dim=-1).sum(dim=-1)  # [K]

        # Densely sample each anchor polyline (5 sub-steps per segment -> 40 points)
        # so fast anchors with wide waypoint spacing have exact cross-track distance to slow siblings.
        alphas = torch.linspace(0.2, 1.0, 5, dtype=pts0.dtype, device=pts0.device).view(1, 1, 5, 1)
        dense_pts = (poly[:, :-1, :].unsqueeze(2) + alphas * diffs.unsqueeze(2)).reshape(k, -1, 2)  # [K, 40, 2]

        # Asymmetric prefix cross-track distance: every point of slow anchor i must lie on fast anchor j's curve
        pw_dist = torch.cdist(pts0.unsqueeze(1), dense_pts.unsqueeze(0))  # [K, K, T, 40]
        ct_slow_to_fast = pw_dist.min(dim=-1).values.mean(dim=-1)  # [K, K]
        ds = arc_len.unsqueeze(0) - arc_len.unsqueeze(1)  # ds[i, j] = S_8(j) - S_8(i)

        pairs = []
        seen = set()
        for i in range(k):
            if float(arc_len[i].item()) < 1.0:
                continue
            for ds_min, ds_max in ((2.0, 6.0), (5.0, 12.0)):
                mask = (ct_slow_to_fast[i] < 1.0) & (ds[i] >= ds_min) & (ds[i] <= ds_max)
                cand = mask.nonzero(as_tuple=True)[0]
                if cand.numel() > 0:
                    best_j = int(cand[ct_slow_to_fast[i, cand].argmin()].item())
                    key = (i, best_j)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
        if not pairs:
            return torch.zeros((0, 2), dtype=torch.long, device=anchors.device)
        return torch.tensor(pairs, dtype=torch.long, device=anchors.device)

    def _build_spatial_bias(self, anchors: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian spatial distance bias [K, HW] (v7) or [H, K, HW] (v8/v9/v10/v11/v12/v13) for 256 anchors."""
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

        if self.version in ("v12", "v13", "v14", "v15", "v16") and wp_pts.size(1) >= 2:
            # v12/v13/v14/v15/v16 Fix 1: 1.0s TTC braking look-ahead extension (T=8 -> 10 points)
            delta_end = wp_pts[:, -1, :] - wp_pts[:, -2, :]  # [K, 2]
            step_norm = torch.norm(delta_end, dim=-1, keepdim=True)  # [K, 1]
            v_end = step_norm / 0.5  # [K, 1]
            u_end = delta_end / step_norm.clamp_min(1e-4)  # [K, 2]
            moving = (torch.norm(wp_pts[:, -1, :], dim=-1, keepdim=True) >= 2.0).to(wp_pts.dtype)
            l_look = (v_end * 1.0).clamp(8.0, 12.0) * moving  # [K, 1]
            wp_9 = wp_pts[:, -1, :] + 0.5 * l_look * u_end
            wp_10 = wp_pts[:, -1, :] + 1.0 * l_look * u_end
            wp_pts = torch.cat([wp_pts, wp_9.unsqueeze(1), wp_10.unsqueeze(1)], dim=1)  # [K, T+2, 2]

        dists = torch.norm(wp_pts.unsqueeze(2) - grid_pts.unsqueeze(0).unsqueeze(0), dim=-1)  # [K, T, HW]
        min_dist = dists.min(dim=1).values  # [K, HW]

        if self.version == "v16" and dists.size(1) >= 10 and self.num_heads == 4:
            # v16 Pillar 2: Lateral-Aware Horizon-Partitioned 4-Head Spatial Bias (Sweet-Spot Entropy ~5.9, ~380 cells)
            # - Head 0 (Near-Core, steps 0..4, t=0.5..2.0s, sigma=1.8m): Ego width + ±0.5m lateral offset clearance
            # - Head 1 (Mid-Maneuver & Lateral, steps 2..7, t=1.5..3.5s, sigma=2.8m): Full ±2.5m lateral variant envelope (v2_lat / CK2)
            # - Head 2 (Far-Braking & Cut-In TTC, steps 5..10, t=3.0..5.0s, sigma=4.2m): Ego & adjacent-lane cut-in TTC
            # - Head 3 (Full-Corridor Context, steps 0..10, t=0.5..5.0s, sigma=7.5m): CK2 ±4.8m corridor & road boundary context
            horizon_specs = (
                (0, 4, 1.8),
                (2, 7, 2.8),
                (5, dists.size(1), 4.2),
                (0, dists.size(1), 7.5),
            )
            biases = []
            for s_idx, e_idx, sig in horizon_specs:
                d_h = dists[:, s_idx:e_idx, :].min(dim=1).values
                biases.append(-0.5 * (d_h / sig) ** 2)
            spatial_bias = torch.stack(biases, dim=0)  # [4, K, HW]
        elif self.version == "v15" and dists.size(1) >= 10 and self.num_heads == 4:
            # v15 Pillar 2: Horizon-Partitioned 4-Head Spatial Bias & Corridor-Matched Sharp Sigmas
            # - Head 0 (Near-Core, steps 0..4, t=0.5..2.0s, sigma=1.5m): Immediate curb & collision clearance
            # - Head 1 (Mid-Maneuver, steps 2..7, t=1.5..3.5s, sigma=2.2m): Lane-change & intersection turn apex
            # - Head 2 (Far-Braking & TTC, steps 5..10, t=3.0..5.0s, sigma=2.8m): Pure far-field TTC & lead vehicle
            # - Head 3 (Full-Corridor Context, steps 0..10, t=0.5..5.0s, sigma=4.5m): CK2 ±4.8m corridor context
            horizon_specs = (
                (0, 4, 1.5),
                (2, 7, 2.2),
                (5, dists.size(1), 2.8),
                (0, dists.size(1), 4.5),
            )
            biases = []
            for s_idx, e_idx, sig in horizon_specs:
                d_h = dists[:, s_idx:e_idx, :].min(dim=1).values
                biases.append(-0.5 * (d_h / sig) ** 2)
            spatial_bias = torch.stack(biases, dim=0)  # [4, K, HW]
        elif self.version in ("v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"):
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

    @torch.no_grad()
    def _compute_on_policy_corridor_mask(
        self,
        plan_topk_trajectory: Optional[torch.Tensor],
        bs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """v15 Pillar 4: Compute normalized BEV corridor mask [B, HW] (mean == 1.0) around the student's
        own top-K refined parent trajectories (anchor + offset) with 1.0s look-ahead extension.
        """
        if (
            plan_topk_trajectory is None
            or plan_topk_trajectory.dim() != 4
            or plan_topk_trajectory.size(0) != bs
            or plan_topk_trajectory.size(2) < 2
        ):
            return None
        r = torch.arange(self.bev_h, dtype=torch.float32, device=device)
        c = torch.arange(self.bev_w, dtype=torch.float32, device=device)
        y_min, y_max = self.pc_range[1], self.pc_range[4]
        x_min, x_max = self.pc_range[0], self.pc_range[3]
        grid_y = y_min + (r + 0.5) * ((y_max - y_min) / self.bev_h)
        grid_x = x_min + (c + 0.5) * ((x_max - x_min) / self.bev_w)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid_pts = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2)  # [1, HW, 2]

        traj = plan_topk_trajectory.detach().to(device=device, dtype=torch.float32)  # [B, K_top, T, >=2]
        x_ssr = -traj[..., 1]
        y_ssr = traj[..., 0]
        wp_pts = torch.stack([x_ssr, y_ssr], dim=-1)  # [B, K_top, T, 2]

        delta_end = wp_pts[:, :, -1, :] - wp_pts[:, :, -2, :]
        step_norm = torch.norm(delta_end, dim=-1, keepdim=True)
        v_end = step_norm / 0.5
        u_end = delta_end / step_norm.clamp_min(1e-4)
        moving = (torch.norm(wp_pts[:, :, -1, :], dim=-1, keepdim=True) >= 2.0).to(wp_pts.dtype)
        l_look = (v_end * 1.0).clamp(8.0, 12.0) * moving
        wp_9 = wp_pts[:, :, -1, :] + 0.5 * l_look * u_end
        wp_10 = wp_pts[:, :, -1, :] + 1.0 * l_look * u_end
        wp_all = torch.cat([wp_pts, wp_9.unsqueeze(2), wp_10.unsqueeze(2)], dim=2)  # [B, K_top, T+2, 2]

        flat_pts = wp_all.reshape(bs, -1, 2)  # [B, K_top*(T+2), 2]
        dists = torch.cdist(flat_pts, grid_pts)  # [B, N_pts, HW]
        min_d = dists.min(dim=1).values  # [B, HW]
        m_op = torch.exp(-0.5 * (min_d / 2.4).pow(2))
        hw = float(self.bev_h * self.bev_w)
        m_op_norm = m_op * (hw / m_op.sum(dim=-1, keepdim=True).clamp_min(1e-12))
        return m_op_norm.to(dtype=dtype)

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
            cmd = command.to(device=device)
            if cmd.dim() == 1:
                cmd = F.one_hot(cmd.long(), num_classes=4).to(dtype=dtype)
            else:
                cmd = cmd.to(dtype=dtype)
            q = q + self.cmd_mlp(cmd).unsqueeze(1)
        elif self.version in ("v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"):
            q = q + 0.0 * (self.cmd_mlp.weight.sum() + self.cmd_mlp.bias.sum())

        if ego_status is not None:
            stat = ego_status[..., :4].to(device=device, dtype=dtype)
            q = q + self.status_mlp(stat).unsqueeze(1)
        elif self.version in ("v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"):
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

        if self.version in ("v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"):
            if self.version in ("v13", "v14", "v15", "v16") and domain in ("bevfusion", "resmap"):
                if domain == "bevfusion":
                    # v13/v14/v15/v16 Fix 1 (DET): Wide ±4.8m corridor & adjacent-lane exploration (N_eff ~ 125)
                    lam_gt, lam_cov, lam_risk = 0.45, 0.20, 0.35
                    t_gt, t_risk = 4.5, 8.0
                else:
                    # v13/v14/v15/v16 Fix 1 (MAP): Wide road-boundary exploration (N_eff ~ 130)
                    lam_gt, lam_cov, lam_risk = 0.35, 0.20, 0.45
                    t_gt, t_risk = 4.5, 8.0
            elif self.version == "v12" and domain in ("bevfusion", "resmap"):
                if domain == "bevfusion":
                    # v12 Fix 3 (DET): Tight ego-corridor focus (N_eff ~ 54) for obstacle & TTC precision
                    lam_gt, lam_cov, lam_risk = 0.58, 0.15, 0.27
                    t_gt, t_risk = 2.0, 2.8
                else:
                    # v12 Fix 3 (MAP): Wide road-boundary exploration (N_eff ~ 115) for DAC compliance
                    lam_gt, lam_cov, lam_risk = 0.30, 0.20, 0.50
                    t_gt, t_risk = 4.0, 8.0
            else:
                lam_gt, lam_cov, lam_risk = self.lambda_gt, self.lambda_cov, self.lambda_risk
                t_gt, t_risk = self.tau_gt, self.tau_risk

            # 1. Relaxed GT softmax
            w_gt = F.softmax(-dist / max(t_gt, 1e-4), dim=-1)  # [B, K]

            # 2. Uniform coverage across all 256 anchors
            w_cov = torch.full((bs, k), 1.0 / k, device=dist.device, dtype=dist.dtype)

            # 3. Hard negative / Risk exploration term r_k
            risk_proximity = F.softmax(-dist / max(t_risk, 1e-4), dim=-1)
            if sim_reward is not None and sim_reward.dim() >= 3 and sim_reward.size(-1) == k:
                # sim_reward: [B, 5, K], 0=NC, 1=DAC, 2=EP, 3=TTC, 4=Comfort
                nc = sim_reward[:, 0, :].to(device=dist.device, dtype=dist.dtype).clamp(0.0, 1.0)
                dac = sim_reward[:, 1, :].to(device=dist.device, dtype=dist.dtype).clamp(0.0, 1.0)
                if self.version in ("v11", "v12", "v13", "v14", "v15", "v16") and domain in ("bevfusion", "resmap"):
                    ttc = sim_reward[:, 3, :].to(device=dist.device, dtype=dist.dtype).clamp(0.0, 1.0)
                    if domain == "bevfusion":
                        # v11/v12/v13/v14/v15/v16 Fix 3 (DET): Explore anchors that collide or have TTC hazard with obstacles,
                        # prioritizing obstacles inside the drivable road (0.5 + 0.5 * DAC).
                        s_det = (nc * ttc).clamp(0.0, 1.0)
                        risk_factor = (1.0 - s_det) * (0.5 + 0.5 * dac)
                        if self.version in ("v12", "v13", "v14", "v15", "v16") and self.sibling_pairs is not None and self.sibling_pairs.numel() > 0:
                            pairs = self.sibling_pairs.to(device=dist.device)
                            i_slow, j_fast = pairs[:, 0], pairs[:, 1]
                            cliff = (s_det[:, i_slow] - s_det[:, j_fast]).relu() * dac[:, j_fast]
                            cliff_boost = torch.zeros_like(s_det)
                            idx_slow = i_slow.unsqueeze(0).expand(bs, -1)
                            idx_fast = j_fast.unsqueeze(0).expand(bs, -1)
                            cliff_boost.scatter_reduce_(1, idx_slow, cliff, reduce="amax", include_self=True)
                            cliff_boost.scatter_reduce_(1, idx_fast, cliff, reduce="amax", include_self=True)
                            risk_factor = risk_factor + 0.5 * cliff_boost
                    else:
                        # v11/v12/v13/v14/v15/v16 Fix 3 (MAP): Explore anchors that cross road/lane drivable area boundaries (1 - DAC),
                        # plus in v14/v15/v16: Lane Keeping (LK, slot 3) & Driving Direction Compliance (DDC, slot 4).
                        if self.version in ("v14", "v15", "v16"):
                            lk = sim_reward[:, 3, :].to(device=dist.device, dtype=dist.dtype).clamp(0.0, 1.0)
                            ddc = sim_reward[:, 4, :].to(device=dist.device, dtype=dist.dtype).clamp(0.0, 1.0)
                            s_map = (dac * ddc * (0.4 + 0.6 * lk)).clamp(0.0, 1.0)
                        else:
                            s_map = dac
                        risk_factor = 1.0 - s_map
                        if self.version in ("v12", "v13", "v14", "v15", "v16") and self.sibling_pairs is not None and self.sibling_pairs.numel() > 0:
                            pairs = self.sibling_pairs.to(device=dist.device)
                            i_slow, j_fast = pairs[:, 0], pairs[:, 1]
                            if self.version in ("v14", "v15", "v16"):
                                cliff = (s_map[:, i_slow] - s_map[:, j_fast]).abs()
                            else:
                                cliff = (s_map[:, i_slow] - s_map[:, j_fast]).relu()
                            cliff_boost = torch.zeros_like(s_map)
                            idx_slow = i_slow.unsqueeze(0).expand(bs, -1)
                            idx_fast = j_fast.unsqueeze(0).expand(bs, -1)
                            cliff_boost.scatter_reduce_(1, idx_slow, cliff, reduce="amax", include_self=True)
                            cliff_boost.scatter_reduce_(1, idx_fast, cliff, reduce="amax", include_self=True)
                            risk_factor = risk_factor + 0.5 * cliff_boost
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
            w = lam_gt * w_gt + lam_cov * w_cov + lam_risk * r_k
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

    def _sibling_diff_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        weights: torch.Tensor,
        sim_reward: Optional[torch.Tensor] = None,
        domain: str = "bevfusion",
        u_far_s: Optional[torch.Tensor] = None,
        u_far_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """v12 Fix 2 + v15 Pillar 3: Velocity-Sibling Differential Feature KD (L_sibling_diff).
        Aligns the normalized anchor feature difference Delta f(i_slow, j_fast) = LN(f(j_fast)) - LN(f(i_slow))
        between student and teacher across velocity siblings, weighted by speed-cliff magnitude.
        In v15, blends 65% full post-FFN diff with 35% Mid+Far horizon head diff (u_far, t=1.5..5.0s).
        """
        if self.sibling_pairs is None or self.sibling_pairs.numel() == 0:
            zero = student_feat.new_zeros(())
            return zero, zero
        pairs = self.sibling_pairs.to(device=student_feat.device)
        i_slow, j_fast = pairs[:, 0], pairs[:, 1]
        c = student_feat.size(-1)
        k = student_feat.size(1)

        ln_s = F.layer_norm(student_feat, (c,))
        ln_t = F.layer_norm(teacher_feat, (c,))

        diff_s = ln_s[:, j_fast, :] - ln_s[:, i_slow, :]  # [B, P, C]
        diff_t = ln_t[:, j_fast, :] - ln_t[:, i_slow, :]  # [B, P, C]

        l2 = ((diff_s - diff_t) ** 2).mean(dim=-1)  # [B, P]
        cos_sim = F.cosine_similarity(diff_s, diff_t, dim=-1, eps=1e-2)  # [B, P]
        per_pair_loss = 0.5 * l2 + 0.5 * (1.0 - cos_sim)

        if self.version in ("v15", "v16") and u_far_s is not None and u_far_t is not None:
            c_far = u_far_s.size(-1)
            ln_far_s = F.layer_norm(u_far_s, (c_far,))
            ln_far_t = F.layer_norm(u_far_t.detach(), (c_far,))
            df_s = ln_far_s[:, j_fast, :] - ln_far_s[:, i_slow, :]
            df_t = ln_far_t[:, j_fast, :] - ln_far_t[:, i_slow, :]
            l2_far = ((df_s - df_t) ** 2).mean(dim=-1)
            cos_far = F.cosine_similarity(df_s, df_t, dim=-1, eps=1e-2)
            loss_far = 0.5 * l2_far + 0.5 * (1.0 - cos_far)
            w_far = 0.30 if self.version == "v16" else 0.35
            per_pair_loss = (1.0 - w_far) * per_pair_loss + w_far * loss_far
            cos_sim = (1.0 - w_far) * cos_sim + w_far * cos_far

        with torch.no_grad():
            pair_anchor_w = 0.5 * (weights[:, i_slow] + weights[:, j_fast])  # [B, P]
            if sim_reward is not None and sim_reward.dim() >= 3 and sim_reward.size(-1) == k:
                nc = sim_reward[:, 0, :].to(device=student_feat.device, dtype=student_feat.dtype).clamp(0.0, 1.0)
                dac = sim_reward[:, 1, :].to(device=student_feat.device, dtype=student_feat.dtype).clamp(0.0, 1.0)
                ttc = sim_reward[:, 3, :].to(device=student_feat.device, dtype=student_feat.dtype).clamp(0.0, 1.0)
                if domain == "bevfusion":
                    s_b = (nc * ttc).clamp(0.0, 1.0)
                    cliff_b = (s_b[:, i_slow] - s_b[:, j_fast]).relu() * dac[:, j_fast]
                elif self.version in ("v14", "v15", "v16"):
                    lk = sim_reward[:, 3, :].to(device=student_feat.device, dtype=student_feat.dtype).clamp(0.0, 1.0)
                    ddc = sim_reward[:, 4, :].to(device=student_feat.device, dtype=student_feat.dtype).clamp(0.0, 1.0)
                    s_b = (dac * ddc * (0.4 + 0.6 * lk)).clamp(0.0, 1.0)
                    cliff_b = (s_b[:, i_slow] - s_b[:, j_fast]).abs()
                else:
                    s_b = dac
                    cliff_b = (s_b[:, i_slow] - s_b[:, j_fast]).relu()
                raw_w = pair_anchor_w * (0.2 + cliff_b)
            else:
                raw_w = pair_anchor_w
            w_sum = raw_w.sum(dim=-1, keepdim=True)
            w_pair = torch.where(
                w_sum > 1e-8,
                raw_w / w_sum.clamp_min(1e-8),
                torch.full_like(raw_w, 1.0 / max(pairs.size(0), 1)),
            )

        loss = (w_pair * per_pair_loss).sum(dim=-1).mean()
        weighted_cos = (w_pair * cos_sim).sum(dim=-1).mean()
        return loss, weighted_cos

    def _multiscale_ck2_grid_loss(
        self,
        s_bev: torch.Tensor,
        t_bev: torch.Tensor,
        w_pgp: torch.Tensor,
        w_ch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute CK2-matched multi-scale grid KD (M3b 5x5 global context pool + M3a local spatial edge).

        Args:
            s_bev: [B, HW, C] projected student BEV tokens
            t_bev: [B, HW, C] z-normalized teacher BEV tokens
            w_pgp: [B, HW] soft-capped wide corridor spatial weights (mean == 1.0)
            w_ch: [1, 1, C] optional subspace channel importance weights (mean == 1.0)

        Returns:
            (global_5x5_loss, edge_3x3_loss, cos_5x5)
        """
        bs = s_bev.size(0)
        # 1. M3b 5x5 Global Context Pool KD ([B, C, 50, 100] -> [B, 200, C], matching corridor.py:global_tokens)
        if self.version in ("v14", "v15", "v16"):
            with torch.no_grad():
                s_map = s_bev.detach().transpose(1, 2).reshape(bs, self.channels, self.bev_h, self.bev_w)
                t_map = t_bev.detach().transpose(1, 2).reshape(bs, self.channels, self.bev_h, self.bev_w)
                s_5x5 = F.avg_pool2d(s_map, kernel_size=5, stride=5).flatten(2).transpose(1, 2)
                t_5x5 = F.avg_pool2d(t_map, kernel_size=5, stride=5).flatten(2).transpose(1, 2)
                global_5x5_loss = (s_5x5 - t_5x5).pow(2).mean()
                cos_5x5 = F.cosine_similarity(s_5x5, t_5x5, dim=-1, eps=1e-6).mean()
        else:
            s_map = s_bev.transpose(1, 2).reshape(bs, self.channels, self.bev_h, self.bev_w)
            t_map = t_bev.transpose(1, 2).reshape(bs, self.channels, self.bev_h, self.bev_w)
            s_5x5 = F.avg_pool2d(s_map, kernel_size=5, stride=5).flatten(2).transpose(1, 2)
            t_5x5 = F.avg_pool2d(t_map, kernel_size=5, stride=5).flatten(2).transpose(1, 2)
            global_5x5_loss = (s_5x5 - t_5x5).pow(2).mean()
            with torch.no_grad():
                cos_5x5 = F.cosine_similarity(s_5x5, t_5x5, dim=-1, eps=1e-6).mean()

        # 2. M3a Corridor Spatial-Derivative (3x3 Edge) KD along forward rows and lateral columns
        s_grid = s_bev.view(bs, self.bev_h, self.bev_w, self.channels)
        t_grid = t_bev.view(bs, self.bev_h, self.bev_w, self.channels)
        w_grid = w_pgp.view(bs, self.bev_h, self.bev_w)
        ch_w = w_ch.view(1, 1, 1, self.channels) if w_ch is not None else 1.0

        ds_r = s_grid[:, 1:, :, :] - s_grid[:, :-1, :, :]
        dt_r = t_grid[:, 1:, :, :] - t_grid[:, :-1, :, :]
        w_r = 0.5 * (w_grid[:, 1:, :] + w_grid[:, :-1, :])
        loss_edge_r = (w_r * ((ds_r - dt_r).pow(2) * ch_w).mean(dim=-1)).mean()

        ds_c = s_grid[:, :, 1:, :] - s_grid[:, :, :-1, :]
        dt_c = t_grid[:, :, 1:, :] - t_grid[:, :, :-1, :]
        w_c = 0.5 * (w_grid[:, :, 1:] + w_grid[:, :, :-1])
        loss_edge_c = (w_c * ((ds_c - dt_c).pow(2) * ch_w).mean(dim=-1)).mean()

        edge_d1_loss = 0.5 * (loss_edge_r + loss_edge_c)
        if self.version == "v16":
            # v16 Pillar 5: Multi-Dilation Corridor Edge KD (70% d=1 0.64m local boundary + 30% d=2 1.28m lane/vehicle contrast)
            ds_r2 = s_grid[:, 2:, :, :] - s_grid[:, :-2, :, :]
            dt_r2 = t_grid[:, 2:, :, :] - t_grid[:, :-2, :, :]
            w_r2 = 0.5 * (w_grid[:, 2:, :] + w_grid[:, :-2, :])
            loss_edge_r2 = (w_r2 * ((ds_r2 - dt_r2).pow(2) * ch_w).mean(dim=-1)).mean()

            ds_c2 = s_grid[:, :, 2:, :] - s_grid[:, :, :-2, :]
            dt_c2 = t_grid[:, :, 2:, :] - t_grid[:, :, :-2, :]
            w_c2 = 0.5 * (w_grid[:, :, 2:] + w_grid[:, :, :-2])
            loss_edge_c2 = (w_c2 * ((ds_c2 - dt_c2).pow(2) * ch_w).mean(dim=-1)).mean()

            edge_d2_loss = 0.5 * (loss_edge_r2 + loss_edge_c2)
            edge_3x3_loss = 0.70 * edge_d1_loss + 0.30 * edge_d2_loss
        else:
            edge_3x3_loss = edge_d1_loss
        return global_5x5_loss, edge_3x3_loss, cos_5x5

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

    @torch.no_grad()
    def _compute_map_epdms_targets(
        self,
        trajectory_anchors: torch.Tensor,
        gt_map_pts: Optional[torch.Tensor],
        gt_map_labels: Optional[torch.Tensor],
        gt_map_valid: Optional[torch.Tensor],
        bs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """v14 Fix 5: Compute per-anchor EPDMS Lane Keeping (LK) & Driving Direction Compliance (DDC)
        targets [B, K] and a normalized BEV Centerline + Road-Boundary geometry splat [B, HW] directly
        from gt_map_pts[:, :, 0, :, :] (order 0 = canonical forward direction).
        """
        if (
            gt_map_pts is None
            or gt_map_labels is None
            or gt_map_valid is None
            or gt_map_pts.numel() == 0
            or gt_map_pts.size(0) != bs
        ):
            return None, None, None

        pts_norm = gt_map_pts[:, :, 0, :, :].to(device=device, dtype=torch.float32)  # [B, M, P, 2]
        labels = gt_map_labels.to(device=device)
        valid = gt_map_valid.to(device=device).bool()
        _, m_max, n_pts, _ = pts_norm.shape
        n_seg = n_pts - 1
        k, t_steps, _ = trajectory_anchors.shape

        x_min, y_min = float(self.pc_range[0]), float(self.pc_range[1])
        x_max, y_max = float(self.pc_range[3]), float(self.pc_range[4])
        x_ssr = x_min + (x_max - x_min) * pts_norm[..., 0]
        y_ssr = y_min + (y_max - y_min) * pts_norm[..., 1]
        # Convert to NAVSIM ego frame: x_fwd = y_ssr, y_left = -x_ssr
        cl_xy = torch.stack([y_ssr, -x_ssr], dim=-1)  # [B, M, P, 2]

        # 1. Exact point-to-segment projection onto centerline segments (label == 2)
        p0 = cl_xy[:, :, :-1, :].reshape(bs, 1, m_max * n_seg, 2)
        p1 = cl_xy[:, :, 1:, :].reshape(bs, 1, m_max * n_seg, 2)
        seg_v = p1 - p0
        seg_len2 = (seg_v * seg_v).sum(dim=-1).clamp_min(1e-6)  # [B, 1, MS]
        seg_tan = seg_v / seg_len2.sqrt().unsqueeze(-1)         # [B, 1, MS, 2]

        is_cl = (labels == 2) & valid                           # [B, M]
        has_cl = is_cl.any(dim=1)                               # [B]
        cl_mask = is_cl.unsqueeze(-1).expand(bs, m_max, n_seg).reshape(bs, 1, m_max * n_seg)  # [B, 1, MS]

        anc = trajectory_anchors.to(device=device, dtype=torch.float32)
        a_xy = anc[..., :2].reshape(1, k * t_steps, 1, 2)       # [1, KT, 1, 2]
        a_hd = anc[..., 2]
        a_dir = torch.stack([torch.cos(a_hd), torch.sin(a_hd)], dim=-1).reshape(1, k * t_steps, 1, 2)

        in_patch = ((anc[..., 0] >= y_min) & (anc[..., 0] <= y_max - 0.5) & (anc[..., 1].abs() <= x_max - 0.5)).float()
        step_w = in_patch / in_patch.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [K, T]

        rel = a_xy - p0                                         # [B, KT, MS, 2]
        alpha = ((rel * seg_v).sum(dim=-1) / seg_len2).clamp(0.0, 1.0)
        proj = p0 + alpha.unsqueeze(-1) * seg_v
        dist = (a_xy - proj).norm(dim=-1)                       # [B, KT, MS]
        cos_dir = (a_dir * seg_tan).sum(dim=-1)                 # [B, KT, MS]
        dist_valid = torch.where(cl_mask, dist, torch.full_like(dist, 1e4))

        # Per-anchor LK (Lane Keeping): distance to nearest direction-compatible centerline segment (sigma = 0.5m)
        compat_mask = cl_mask & (cos_dir > 0.25)
        has_compat = compat_mask.any(dim=-1, keepdim=True)
        dist_compat = torch.where(compat_mask, dist, torch.full_like(dist, 1e4))
        min_d_lk = torch.where(has_compat, dist_compat, dist_valid).min(dim=-1).values.view(bs, k, t_steps)
        step_lk = torch.exp(-0.5 * (min_d_lk / 0.50).pow(2))
        s_lk = (step_lk * step_w.unsqueeze(0)).sum(dim=-1)      # [B, K]

        # Per-anchor DDC (Driving Direction Compliance): best tangent cosine within 1.2m (or nearest segment)
        near_mask = cl_mask & (dist < 1.2)
        has_near = near_mask.any(dim=-1, keepdim=True)
        cos_near = torch.where(near_mask, cos_dir, torch.full_like(cos_dir, -1e4)).max(dim=-1).values
        nearest_idx = dist_valid.argmin(dim=-1, keepdim=True)
        cos_nearest = torch.gather(cos_dir, -1, nearest_idx).squeeze(-1)
        best_cos = torch.where(has_near.squeeze(-1), cos_near, cos_nearest).view(bs, k, t_steps)
        step_ddc = torch.sigmoid((best_cos + 0.10) / 0.15)
        s_ddc = (step_ddc * step_w.unsqueeze(0)).sum(dim=-1)    # [B, K]

        s_lk = torch.where(has_cl.unsqueeze(-1), s_lk, torch.ones_like(s_lk)).to(dtype=dtype)
        s_ddc = torch.where(has_cl.unsqueeze(-1), s_ddc, torch.ones_like(s_ddc)).to(dtype=dtype)

        return s_lk, s_ddc, None

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
        gt_map_pts: Optional[torch.Tensor] = None,
        gt_map_labels: Optional[torch.Tensor] = None,
        gt_map_valid: Optional[torch.Tensor] = None,
        plan_topk_trajectory: Optional[torch.Tensor] = None,
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

            if self.version in ("v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"):
                # -------------------------------------------------------------
                # v9/v10/v11/v12/v13/v14/v15/v16: Domain-Specific Planning-Grounded Probe (PGP) +
                #                                 Multi-Scale / PGP-Weighted Cell KD
                # -------------------------------------------------------------
                if not hasattr(self, "spatial_bias") or self.spatial_bias is None:
                    spatial_bias = self._build_spatial_bias(trajectory_anchors)
                    self.register_buffer("spatial_bias", spatial_bias, persistent=False)
                if self.version in ("v12", "v13", "v14", "v15", "v16") and (not hasattr(self, "sibling_pairs") or self.sibling_pairs is None):
                    self.register_buffer("sibling_pairs", self._build_sibling_pairs(trajectory_anchors), persistent=False)
                if self.version in ("v12", "v13", "v14", "v15", "v16") and (not hasattr(self, "row_gain") or self.row_gain is None):
                    self.register_buffer("row_gain", self._build_row_gain(device), persistent=False)

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
                self.last_probe_unit_inputs = {}
                self.last_probe_unit_losses = {}
                tkey = {"bevfusion": "det", "resmap": "map"}

                s_lk_map, s_ddc_map, m_geom_map = (None, None, None)
                if self.version in ("v14", "v15", "v16"):
                    s_lk_map, s_ddc_map, m_geom_map = self._compute_map_epdms_targets(
                        trajectory_anchors, gt_map_pts, gt_map_labels, gt_map_valid, bs, device, dtype
                    )
                    if s_lk_map is not None:
                        metrics["distill/map_s_lk_mean"] = s_lk_map.mean().detach()
                    if s_ddc_map is not None:
                        metrics["distill/map_s_ddc_mean"] = s_ddc_map.mean().detach()

                m_on_policy = None
                if self.version == "v15":
                    m_on_policy = self._compute_on_policy_corridor_mask(plan_topk_trajectory, bs, device, dtype)

                for bank, name in enumerate(names):
                    probe = self.probes[bank]
                    t_bev = teacher_by_bank[bank].detach()
                    sim_reward_b = sim_reward
                    if (
                        self.version in ("v14", "v15", "v16")
                        and name == "resmap"
                        and sim_reward is not None
                        and sim_reward.dim() >= 3
                        and sim_reward.size(-1) == k
                        and s_lk_map is not None
                        and s_ddc_map is not None
                    ):
                        # v14/v15/v16 Fix 5: Populate unused slots 3 and 4 of resmap sim_reward with EPDMS LK & DDC targets
                        sim_reward_b = sim_reward.clone()
                        sim_reward_b[:, 3, :] = s_lk_map.to(device=sim_reward.device, dtype=sim_reward.dtype)
                        sim_reward_b[:, 4, :] = s_ddc_map.to(device=sim_reward.device, dtype=sim_reward.dtype)
                    if self.version in ("v10", "v11", "v12", "v13", "v14", "v15", "v16"):
                        zm = getattr(self, f"z_mean_{bank}", None)
                        zs = getattr(self, f"z_std_{bank}", None)
                        if zm is not None and zs is not None:
                            t_bev = ((t_bev.float() - zm.to(device=device)) / zs.to(device=device)).to(dtype=dtype)
                    proj = self.student_proj[bank] if self.use_proj else None
                    if self.version in ("v10", "v11", "v12", "v13", "v14", "v15", "v16"):
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
                        sim_reward=sim_reward_b,
                        sim_reward_valid=sim_reward_valid,
                        sibling_pairs=self.sibling_pairs if self.version in ("v12", "v13", "v14", "v15", "v16") else None,
                    )
                    losses[f"loss_probe_{name}"] = self.probe_loss_weight * probe_loss
                    metrics.update(probe_metrics)

                    domain_scale = self.resmap_scale if name == "resmap" else 1.0
                    if self.version == "v16":
                        # v16 Pillars 1–5 (docs/bev_selector_v16.md):
                        # - Pillar 1: 100% Stop-Grad Adapter Isolation (proj.weight.detach(), proj.bias.detach())
                        #             + Subspace-Importance Weighted Grid KD (w_ch in [0.85, 1.38] on 1x1 Cell & Multi-Dilation Edge)
                        # - Pillar 2: Lateral-Aware Horizon-Partitioned 4-Head Spatial Bias (1.8, 2.8, 4.2, 7.5)m + gamma_h in (0.15, 2.20)
                        #             + 70% Full Post-FFN / 30% Mid+Far Heads (u_far) Velocity-Sibling Diff KD
                        # - Pillar 3: Golden 0.090 / 0.025 Dual-Stream Budget + Late-Epoch Probe Cosine Taper (1.0 -> 0.65 over Ep 20..29)
                        # - Pillar 4: Pure Teacher-Grounded Ribbon (DET 0.75 / MAP 0.85 floor, no on-policy confirmation bias)
                        #             & Pure 40/40/40 Geometric/Relational Probe Loss (no uncalibrated L_score_kd)
                        # - Pillar 5: Multi-Dilation Corridor Edge KD (70% d=1 0.64m + 30% d=2 1.28m lane/vehicle contrast, 18% of Grid KD)
                        from ..ck.bev_kd_arm import GradScale
                        tk = tkey[name]
                        taper_start, taper_end = 106400, 159600
                        if step <= taper_start:
                            probe_taper = 1.0
                        else:
                            prog = min(1.0, max(0.0, float(step - taper_start) / float(taper_end - taper_start)))
                            probe_taper = 0.65 + 0.35 * 0.5 * (1.0 + math.cos(math.pi * prog))
                        metrics["distill/probe_taper"] = student_bev.new_tensor(probe_taper)
                        if self.probe_ratio_ctrl is not None and self.probe_ratio_ctrl[tk].n > 0:
                            lam_p = float(self.probe_ratio_ctrl[tk].weight(step))
                            probe_bev_scale = float(ramp) * probe_taper * lam_p
                        else:
                            lam_p = float(ramp) * 0.25 * probe_taper
                            probe_bev_scale = lam_p
                        metrics[f"distill/lam_probe_{tk}"] = student_bev.new_tensor(probe_bev_scale)
                        x_probe_unit = GradScale.apply(student_bev, probe_bev_scale)
                        s_probe_normed = F.layer_norm(x_probe_unit, (self.channels,))
                        if proj is not None:
                            s_bev_probe = F.linear(s_probe_normed, proj.weight.detach(), proj.bias.detach())
                        else:
                            s_bev_probe = s_probe_normed

                        with torch.no_grad():
                            if gt_trajectory is not None and gt_trajectory.numel() > 0:
                                weights_b, _, n_eff_b = self._compute_anchor_weights(
                                    trajectory_anchors, gt_trajectory, sim_reward=sim_reward_b, domain=name,
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
                            if name == "bevfusion" and self.row_gain is not None and self.row_gain.numel() == hw:
                                m_spatial = m_spatial * self.row_gain.to(device=device, dtype=m_spatial.dtype)
                            m_norm = m_spatial * (float(hw) / m_spatial.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            m_cap = 4.0 * torch.tanh(m_norm / 4.0)
                            m_cap_norm = m_cap * (float(hw) / m_cap.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            if name == "bevfusion":
                                w_pgp = 0.75 + 0.25 * m_cap_norm
                            else:
                                m_map = 0.5 * m_cap_norm + 0.5 * m_geom_map if m_geom_map is not None else m_cap_norm
                                w_pgp = 0.85 + 0.15 * m_map
                            w_ch = probe.channel_importance(device, dtype)
                            metrics[f"distill/w_pgp_max/{name}"] = w_pgp.max(dim=-1).values.mean().detach()
                            metrics[f"distill/w_pgp_std/{name}"] = w_pgp.std(dim=-1).mean().detach()
                            metrics[f"distill/w_pgp_far/{name}"] = w_pgp[:, hw // 2:].mean().detach()

                        # Stream 2: Student Stop-Grad Probe Pass (100% isolated from student_proj)
                        attn_s, feat_s, preds_s = probe.forward_student(q.detach(), s_bev_probe, s_pos, bias)
                        feat_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t.detach(), weights_b)
                        diff_loss, diff_cos = self._sibling_diff_loss(
                            feat_s,
                            feat_t.detach(),
                            weights_b,
                            sim_reward=sim_reward_b,
                            domain=name,
                            u_far_s=preds_s.get("u_far"),
                            u_far_t=preds_t.get("u_far"),
                        )
                        kl_per_head = F.kl_div(
                            attn_s.clamp_min(1e-8).log(), attn_t.detach(), reduction="none"
                        ).sum(dim=-1)
                        kl = (head_w * kl_per_head).mean(dim=1)
                        attn_loss = (weights_b * kl).sum(dim=-1).mean()

                        # Stream 1: Subspace-Weighted High-Res Corridor Grid KD (82% 1x1 Cell + 18% Multi-Dilation Edge)
                        raw_cell_sq = (s_bev - t_bev).pow(2)
                        cell_mse = (raw_cell_sq * w_ch).mean(dim=-1)
                        uniform_mse = raw_cell_sq.mean(dim=-1)
                        cell_loss = (w_pgp * cell_mse).mean()
                        global_5x5_loss, edge_3x3_loss, cos_5x5 = self._multiscale_ck2_grid_loss(
                            s_bev, t_bev, w_pgp, w_ch=w_ch
                        )
                        metrics[f"distill_cell_mse/{name}"] = cell_loss.detach()
                        metrics[f"distill_uniform_mse/{name}"] = uniform_mse.mean().detach()
                        metrics[f"distill_global_5x5/{name}"] = global_5x5_loss.detach()
                        metrics[f"distill_global_5x5_cos/{name}"] = cos_5x5.detach()
                        metrics[f"distill_edge_3x3/{name}"] = edge_3x3_loss.detach()
                        metrics[f"distill_sibling_diff/{name}"] = diff_loss.detach()
                        metrics[f"distill_sibling_cos/{name}"] = diff_cos.detach()

                        grid_bank_loss = self.loss_scale * (
                            0.82 * cell_loss
                            + 0.18 * edge_3x3_loss
                        )
                        probe_bank_loss = (
                            self.loss_scale * (
                                0.40 * feat_loss
                                + 0.40 * diff_loss
                            )
                            + 0.40 * self.attn_loss_scale * attn_loss
                        )
                        raw_bank_loss = grid_bank_loss + probe_bank_loss
                        bank_loss = raw_bank_loss
                        self.last_unit_inputs[tk] = x_unit
                        self.last_unit_losses[tk] = 0.5 * grid_bank_loss
                        self.last_probe_unit_inputs[tk] = x_probe_unit
                        self.last_probe_unit_losses[tk] = 0.5 * probe_bank_loss
                    elif self.version == "v15":
                        # v15 Pillar 1 + 2 + 3 + 4 + 5 (docs/bev_selector_v15.md):
                        # - Pillar 1: Subspace-Guided Adapter (w_ch channel importance on Grid KD + 0.08x GradScale on proj.weight, 0.0 on proj.bias)
                        #             + Rebalanced Dual-Stream (Grid ratio=0.085, cap=0.22 + Probe ratio=0.035, cap=0.08)
                        # - Pillar 2: Horizon-Partitioned 4-Head Spatial Bias (Near 1.5m / Mid 2.2m / Far-TTC 2.8m / Full 4.5m) + gamma_h in (0.2, 3.5)
                        # - Pillar 3: Horizon-Decoupled Velocity-Sibling Diff KD (65% full post-FFN + 35% Mid+Far heads u_far)
                        # - Pillar 4: On-Policy Student-Refined Corridor Ribbon (80% Teacher Probe + 20% student plan_topk_trajectory)
                        # - Pillar 5: Functional Safety Decision-Boundary KD (15% stop-grad sim_head Binary KL)
                        from ..ck.bev_kd_arm import GradScale
                        tk = tkey[name]
                        if self.probe_ratio_ctrl is not None and self.probe_ratio_ctrl[tk].n > 0:
                            lam_p = float(self.probe_ratio_ctrl[tk].weight(step))
                            probe_bev_scale = float(ramp) * lam_p
                        else:
                            lam_p = float(ramp) * 0.35
                            probe_bev_scale = float(ramp) * 0.35
                        metrics[f"distill/lam_probe_{tk}"] = student_bev.new_tensor(probe_bev_scale)
                        x_probe_unit = GradScale.apply(student_bev, probe_bev_scale)
                        s_probe_normed = F.layer_norm(x_probe_unit, (self.channels,))
                        if proj is not None:
                            w_proj_probe = GradScale.apply(proj.weight, 0.08)
                            s_bev_probe = F.linear(s_probe_normed, w_proj_probe, proj.bias.detach())
                        else:
                            s_bev_probe = s_probe_normed

                        with torch.no_grad():
                            if gt_trajectory is not None and gt_trajectory.numel() > 0:
                                weights_b, _, n_eff_b = self._compute_anchor_weights(
                                    trajectory_anchors, gt_trajectory, sim_reward=sim_reward_b, domain=name,
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
                            m_cap = 4.0 * torch.tanh(m_norm / 4.0)
                            m_cap_norm = m_cap * (float(hw) / m_cap.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            m_comb = 0.80 * m_cap_norm + 0.20 * m_on_policy if m_on_policy is not None else m_cap_norm
                            if name == "bevfusion":
                                if self.row_gain is not None and self.row_gain.numel() == hw:
                                    m_comb = m_comb * self.row_gain.to(device=device, dtype=m_comb.dtype)
                                m_comb = m_comb * (float(hw) / m_comb.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                                w_pgp = 0.75 + 0.25 * m_comb
                            else:
                                m_comb = m_comb * (float(hw) / m_comb.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                                w_pgp = 0.82 + 0.18 * m_comb
                            w_ch = probe.channel_importance(device, dtype)
                            metrics[f"distill/w_pgp_max/{name}"] = w_pgp.max(dim=-1).values.mean().detach()
                            metrics[f"distill/w_pgp_std/{name}"] = w_pgp.std(dim=-1).mean().detach()
                            metrics[f"distill/w_pgp_far/{name}"] = w_pgp[:, hw // 2:].mean().detach()

                        # Stream 2: Student Probe Pass (0.08x damped rotation on proj.weight, 0.0 on proj.bias)
                        attn_s, feat_s, preds_s = probe.forward_student(q.detach(), s_bev_probe, s_pos, bias)
                        feat_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t.detach(), weights_b)
                        diff_loss, diff_cos = self._sibling_diff_loss(
                            feat_s,
                            feat_t.detach(),
                            weights_b,
                            sim_reward=sim_reward_b,
                            domain=name,
                            u_far_s=preds_s.get("u_far"),
                            u_far_t=preds_t.get("u_far"),
                        )
                        score_kd_loss = probe.compute_score_kd_loss(
                            preds_s["sim_logits_s"],
                            preds_t["sim_probs_ema"],
                            weights_b,
                        )
                        kl_per_head = F.kl_div(
                            attn_s.clamp_min(1e-8).log(), attn_t.detach(), reduction="none"
                        ).sum(dim=-1)
                        kl = (head_w * kl_per_head).mean(dim=1)
                        attn_loss = (weights_b * kl).sum(dim=-1).mean()

                        # Stream 1: Subspace-Importance Weighted High-Res Corridor Grid KD (85% 1x1 Cell + 15% 3x3 Edge)
                        raw_cell_sq = (s_bev - t_bev).pow(2)
                        cell_mse = (raw_cell_sq * w_ch).mean(dim=-1)
                        uniform_mse = raw_cell_sq.mean(dim=-1)
                        cell_loss = (w_pgp * cell_mse).mean()
                        global_5x5_loss, edge_3x3_loss, cos_5x5 = self._multiscale_ck2_grid_loss(
                            s_bev, t_bev, w_pgp, w_ch=w_ch
                        )
                        metrics[f"distill_cell_mse/{name}"] = cell_loss.detach()
                        metrics[f"distill_uniform_mse/{name}"] = uniform_mse.mean().detach()
                        metrics[f"distill_global_5x5/{name}"] = global_5x5_loss.detach()
                        metrics[f"distill_global_5x5_cos/{name}"] = cos_5x5.detach()
                        metrics[f"distill_edge_3x3/{name}"] = edge_3x3_loss.detach()
                        metrics[f"distill_sibling_diff/{name}"] = diff_loss.detach()
                        metrics[f"distill_sibling_cos/{name}"] = diff_cos.detach()
                        metrics[f"distill_score_kd/{name}"] = score_kd_loss.detach()

                        grid_bank_loss = self.loss_scale * (
                            0.85 * cell_loss
                            + 0.15 * edge_3x3_loss
                        )
                        probe_bank_loss = (
                            self.loss_scale * (
                                0.35 * feat_loss
                                + 0.35 * diff_loss
                                + 0.15 * score_kd_loss
                            )
                            + 0.35 * self.attn_loss_scale * attn_loss
                        )
                        raw_bank_loss = grid_bank_loss + probe_bank_loss
                        bank_loss = raw_bank_loss
                        self.last_unit_inputs[tk] = x_unit
                        self.last_unit_losses[tk] = 0.5 * grid_bank_loss
                        self.last_probe_unit_inputs[tk] = x_probe_unit
                        self.last_probe_unit_losses[tk] = 0.5 * probe_bank_loss
                    elif self.version == "v14":
                        # v14 Fix 1 + Fix 2 + Fix 3 + Fix 4 + Fix 5 (docs/bev_selector_v14.md):
                        # - Fix 1: Dual-Stream Decoupled Gradient Controller (Grid ratio=0.09, cap=0.22 + Probe ratio=0.025, cap=0.06)
                        # - Fix 2: Stop-Grad Adapter Isolation (s_bev_probe detaches proj.weight/bias so student_proj is trained 100% by Grid KD at lr_mult=3.0)
                        # - Fix 3: Prune dead 5x5 AvgPool loss (g_BEV=0.00012) -> High-Res 85% 1x1 Cell + 15% 3x3 Edge Grid KD
                        # - Fix 4: Domain-Specific Spatial Ribbon Shaping (DET keeps far-field row_gain in [1.0, 1.5] & w_pgp in [0.75, 2.30];
                        #          MAP uses unskewed row_gain=1.0 to preserve 0..18m intersection curbs & raised floor w_pgp = 0.85 + 0.15*m_map)
                        # - Fix 5: EPDMS-Grounded Map Supervision (LK + DDC in P_map sim_head/cliff_loss/w_risk/sibling_diff + Centerline/Boundary m_geom splat)
                        from ..ck.bev_kd_arm import GradScale
                        tk = tkey[name]
                        if self.probe_ratio_ctrl is not None and self.probe_ratio_ctrl[tk].n > 0:
                            lam_p = float(self.probe_ratio_ctrl[tk].weight(step))
                            probe_bev_scale = float(ramp) * lam_p
                        else:
                            lam_p = float(ramp) * 0.25
                            probe_bev_scale = float(ramp) * 0.25
                        metrics[f"distill/lam_probe_{tk}"] = student_bev.new_tensor(probe_bev_scale)
                        x_probe_unit = GradScale.apply(student_bev, probe_bev_scale)
                        s_probe_normed = F.layer_norm(x_probe_unit, (self.channels,))
                        if proj is not None:
                            s_bev_probe = F.linear(s_probe_normed, proj.weight.detach(), proj.bias.detach())
                        else:
                            s_bev_probe = s_probe_normed

                        with torch.no_grad():
                            if gt_trajectory is not None and gt_trajectory.numel() > 0:
                                weights_b, _, n_eff_b = self._compute_anchor_weights(
                                    trajectory_anchors, gt_trajectory, sim_reward=sim_reward_b, domain=name,
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
                            if name == "bevfusion" and self.row_gain is not None and self.row_gain.numel() == hw:
                                m_spatial = m_spatial * self.row_gain.to(device=device, dtype=m_spatial.dtype)
                            m_norm = m_spatial * (float(hw) / m_spatial.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            m_cap = 4.0 * torch.tanh(m_norm / 4.0)
                            m_cap_norm = m_cap * (float(hw) / m_cap.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            if name == "bevfusion":
                                w_pgp = 0.75 + 0.25 * m_cap_norm
                            else:
                                m_map = 0.5 * m_cap_norm + 0.5 * m_geom_map if m_geom_map is not None else m_cap_norm
                                w_pgp = 0.85 + 0.15 * m_map
                            metrics[f"distill/w_pgp_max/{name}"] = w_pgp.max(dim=-1).values.mean().detach()
                            metrics[f"distill/w_pgp_std/{name}"] = w_pgp.std(dim=-1).mean().detach()
                            metrics[f"distill/w_pgp_far/{name}"] = w_pgp[:, hw // 2:].mean().detach()

                        # Stream 2: Student Stop-Grad Probe Pass (s_bev_probe detaches proj.weight/bias so probe never corrupts student_proj)
                        attn_s, feat_s, preds_s = probe.forward_student(q.detach(), s_bev_probe, s_pos, bias)
                        feat_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t.detach(), weights_b)
                        diff_loss, diff_cos = self._sibling_diff_loss(
                            feat_s, feat_t.detach(), weights_b, sim_reward=sim_reward_b, domain=name
                        )
                        kl_per_head = F.kl_div(
                            attn_s.clamp_min(1e-8).log(), attn_t.detach(), reduction="none"
                        ).sum(dim=-1)
                        kl = (head_w * kl_per_head).mean(dim=1)
                        attn_loss = (weights_b * kl).sum(dim=-1).mean()

                        # Stream 1: High-Resolution Corridor Grid KD (85% 1x1 Cell + 15% 3x3 Edge; 5x5 logged no_grad)
                        cell_mse = (s_bev - t_bev).pow(2).mean(dim=-1)
                        cell_loss = (w_pgp * cell_mse).mean()
                        global_5x5_loss, edge_3x3_loss, cos_5x5 = self._multiscale_ck2_grid_loss(
                            s_bev, t_bev, w_pgp
                        )
                        metrics[f"distill_cell_mse/{name}"] = cell_loss.detach()
                        metrics[f"distill_uniform_mse/{name}"] = cell_mse.mean().detach()
                        metrics[f"distill_global_5x5/{name}"] = global_5x5_loss.detach()
                        metrics[f"distill_global_5x5_cos/{name}"] = cos_5x5.detach()
                        metrics[f"distill_edge_3x3/{name}"] = edge_3x3_loss.detach()
                        metrics[f"distill_sibling_diff/{name}"] = diff_loss.detach()
                        metrics[f"distill_sibling_cos/{name}"] = diff_cos.detach()

                        grid_bank_loss = self.loss_scale * (
                            0.85 * cell_loss
                            + 0.15 * edge_3x3_loss
                        )
                        probe_bank_loss = (
                            self.loss_scale * (
                                0.40 * feat_loss
                                + 0.40 * diff_loss
                            )
                            + 0.40 * self.attn_loss_scale * attn_loss
                        )
                        raw_bank_loss = grid_bank_loss + probe_bank_loss
                        bank_loss = raw_bank_loss
                        self.last_unit_inputs[tk] = x_unit
                        self.last_unit_losses[tk] = 0.5 * grid_bank_loss
                        self.last_probe_unit_inputs[tk] = x_probe_unit
                        self.last_probe_unit_losses[tk] = 0.5 * probe_bank_loss
                    elif self.version == "v13":
                        # v13 Fix 1 + Fix 2 + Fix 3 + Fix 4:
                        # - Fix 1: Wide Corridor Coverage (tau_gt=4.5, tau_risk=8.0, N_eff~125-130) + Soft-Capped Ribbon w_pgp (0.75 + 0.25*m_cap_norm)
                        # - Fix 2: Non-Penalizing Far-Field Boost (row_gain in [1.0, 1.5]) + 10-pt Look-Ahead Spatial Bias
                        # - Fix 3: CK2-Matched Multi-Scale Grid KD (M3b 5x5 Global Pool + M3a 3x3 Spatial-Derivative Edge KD)
                        # - Fix 4: Balanced Dual-Consumer Allocation (55% 1x1 Cell + 15% 5x5 Pool + 10% 3x3 Edge + 10% Feat + 10% Diff + 0.20x Attn KL)
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
                            if self.row_gain is not None and self.row_gain.numel() == hw:
                                m_spatial = m_spatial * self.row_gain.to(device=device, dtype=m_spatial.dtype)
                            m_norm = m_spatial * (float(hw) / m_spatial.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            m_cap = 4.0 * torch.tanh(m_norm / 4.0)
                            m_cap_norm = m_cap * (float(hw) / m_cap.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            w_pgp = 0.75 + 0.25 * m_cap_norm
                            metrics[f"distill/w_pgp_max/{name}"] = w_pgp.max(dim=-1).values.mean().detach()
                            metrics[f"distill/w_pgp_std/{name}"] = w_pgp.std(dim=-1).mean().detach()
                            metrics[f"distill/w_pgp_far/{name}"] = w_pgp[:, hw // 2:].mean().detach()

                        # Student Stop-Grad Probe Pass (gradients flow through s_bev into student_proj and x_unit)
                        attn_s, feat_s, preds_s = probe.forward_student(q.detach(), s_bev, s_pos, bias)
                        feat_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t.detach(), weights_b)
                        diff_loss, diff_cos = self._sibling_diff_loss(
                            feat_s, feat_t.detach(), weights_b, sim_reward=sim_reward, domain=name
                        )
                        kl_per_head = F.kl_div(
                            attn_s.clamp_min(1e-8).log(), attn_t.detach(), reduction="none"
                        ).sum(dim=-1)
                        kl = (head_w * kl_per_head).mean(dim=1)
                        attn_loss = (weights_b * kl).sum(dim=-1).mean()

                        cell_mse = (s_bev - t_bev).pow(2).mean(dim=-1)
                        cell_loss = (w_pgp * cell_mse).mean()
                        global_5x5_loss, edge_3x3_loss, cos_5x5 = self._multiscale_ck2_grid_loss(
                            s_bev, t_bev, w_pgp
                        )
                        metrics[f"distill_cell_mse/{name}"] = cell_loss.detach()
                        metrics[f"distill_uniform_mse/{name}"] = cell_mse.mean().detach()
                        metrics[f"distill_global_5x5/{name}"] = global_5x5_loss.detach()
                        metrics[f"distill_global_5x5_cos/{name}"] = cos_5x5.detach()
                        metrics[f"distill_edge_3x3/{name}"] = edge_3x3_loss.detach()
                        metrics[f"distill_sibling_diff/{name}"] = diff_loss.detach()
                        metrics[f"distill_sibling_cos/{name}"] = diff_cos.detach()

                        raw_bank_loss = (
                            self.loss_scale * (
                                0.55 * cell_loss
                                + 0.15 * global_5x5_loss
                                + 0.10 * edge_3x3_loss
                                + 0.10 * feat_loss
                                + 0.10 * diff_loss
                            )
                            + 0.20 * self.attn_loss_scale * attn_loss
                        )
                        bank_loss = raw_bank_loss
                        self.last_unit_inputs[tkey[name]] = x_unit
                        self.last_unit_losses[tkey[name]] = 0.5 * raw_bank_loss
                    elif self.version == "v12":
                        # v12 Fix 1 + Fix 2 + Fix 3 + Fix 4:
                        # - Fix 1: Look-ahead spatial bias (10 pts) + Horizon density gain g(y) on w_pgp
                        # - Fix 2: Velocity-Sibling Differential Feature KD (10% of loss_scale)
                        # - Fix 3: Domain-Asymmetric Anchor Focus (DET tight tau=2.0/2.8 vs MAP wide tau=4.0/8.0) + Speed-Cliff Boost
                        # - Fix 4: Dual-Consumer 80/20 Hybrid KD (80% 1x1 Cell MSE + 10% Feat + 10% Sibling Diff + 0.25x Attn KL)
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
                            if self.row_gain is not None and self.row_gain.numel() == hw:
                                m_spatial = m_spatial * self.row_gain.to(device=device, dtype=m_spatial.dtype)
                            m_norm = m_spatial * (float(hw) / m_spatial.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            w_pgp = 0.5 + 0.5 * m_norm
                            metrics[f"distill/w_pgp_max/{name}"] = w_pgp.max(dim=-1).values.mean().detach()
                            metrics[f"distill/w_pgp_std/{name}"] = w_pgp.std(dim=-1).mean().detach()
                            metrics[f"distill/w_pgp_far/{name}"] = w_pgp[:, hw // 2:].mean().detach()

                        # Student Stop-Grad Probe Pass (gradients flow through s_bev into student_proj and x_unit)
                        attn_s, feat_s, preds_s = probe.forward_student(q.detach(), s_bev, s_pos, bias)
                        feat_loss, bank_cos = self._feature_pair_loss(feat_s, feat_t.detach(), weights_b)
                        diff_loss, diff_cos = self._sibling_diff_loss(
                            feat_s, feat_t.detach(), weights_b, sim_reward=sim_reward, domain=name
                        )
                        kl_per_head = F.kl_div(
                            attn_s.clamp_min(1e-8).log(), attn_t.detach(), reduction="none"
                        ).sum(dim=-1)
                        kl = (head_w * kl_per_head).mean(dim=1)
                        attn_loss = (weights_b * kl).sum(dim=-1).mean()

                        cell_mse = (s_bev - t_bev).pow(2).mean(dim=-1)
                        cell_loss = (w_pgp * cell_mse).mean()
                        metrics[f"distill_cell_mse/{name}"] = cell_loss.detach()
                        metrics[f"distill_uniform_mse/{name}"] = cell_mse.mean().detach()
                        metrics[f"distill_sibling_diff/{name}"] = diff_loss.detach()
                        metrics[f"distill_sibling_cos/{name}"] = diff_cos.detach()

                        raw_bank_loss = (
                            self.loss_scale * (0.80 * cell_loss + 0.10 * feat_loss + 0.10 * diff_loss)
                            + 0.25 * self.attn_loss_scale * attn_loss
                        )
                        bank_loss = raw_bank_loss
                        self.last_unit_inputs[tkey[name]] = x_unit
                        self.last_unit_losses[tkey[name]] = 0.5 * raw_bank_loss
                    elif self.version == "v11":
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
