"""Domain-Specific Planning-Grounded Probe (PGP) for BEV Selector Distillation (v9 & v10).

v10 Upgrades over v9:
1. Pre-Attention BEV LayerNorm (`bev_norm`) + Detached Positional Encoding (`bev_pos.detach()` on Teacher pass)
   to eliminate cross-domain norm mismatch and prevent probe training gradients from polluting Student PE.
2. BEV-Pure Post-Attention FFN & No-Shortcut Reward Heads (no additive query skip-connection into reward heads,
   forcing 100% of NC/TTC/DAC discrimination gradients to flow through BEV K/V projections) + Domain-Pure
   Loss Masking (Comfort excluded as purely kinematic; P_det focuses on NC/TTC/EP, P_map on DAC/EP/Imitation).
3. Bounded Multi-Scale Spatial Distance Bias (`[2, 4, 8, 16]m`) with Learnable Per-Head Gate `gamma_h`
   and Confidence-Weighted (Inverse-Entropy) Attention KD to prevent diffuse global-head KL noise.
4. Step-Aware Warm-up / Ramp Schedule + Bias-Corrected Fast EMA Probe (`decay = min(0.99, (1+t)/(10+t))`)
   with 100% plain-DDP (`strategy=ddp`) gradient coverage on every micro-batch.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Order matches navsim/agents/para_ssr/modules/anchor_planner.py SIM_KEYS:
# 0: no_at_fault_collisions (NC)
# 1: drivable_area_compliance (DAC)
# 2: ego_progress (EP)
# 3: time_to_collision_within_bound (TTC)
# 4: comfort (C)
PROBE_SIM_Masks: Dict[str, Tuple[float, float, float, float, float]] = {
    # P_det (BEVFusion): dynamic obstacle collision (NC, TTC) + progress (EP); DAC=0, Comfort=0
    "bevfusion": (1.0, 0.0, 0.5, 1.0, 0.0),
    # P_map (ReSMap): road/lane compliance (DAC) + progress (EP); NC=0, TTC=0, Comfort=0
    "resmap": (0.0, 1.0, 0.5, 0.0, 0.0),
}


class PlanningGroundedProbe(nn.Module):
    """Single-Layer (L=1) Domain-Specific Cross-Attention Planning Probe."""

    def __init__(
        self,
        domain: str,
        channels: int = 256,
        num_heads: int = 4,
        fut_ts: int = 8,
        traj_dims: int = 3,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        version: str = "v10",
    ) -> None:
        super().__init__()
        if domain not in ("bevfusion", "resmap"):
            raise ValueError(f"domain must be 'bevfusion' or 'resmap', got {domain!r}")
        self.domain = str(domain)
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.fut_ts = int(fut_ts)
        self.traj_dims = int(traj_dims)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.version = str(version).lower()

        # v10 Fix 1: Pre-attention LayerNorm on BEV tokens (matches PlanTaskMemoryLayer.bev_memory_norm)
        self.bev_norm = nn.LayerNorm(self.channels)

        # Single-layer Multi-Head Cross-Attention projections
        self.q_proj = nn.Linear(self.channels, self.channels)
        self.k_proj = nn.Linear(self.channels, self.channels)
        self.v_proj = nn.Linear(self.channels, self.channels)
        self.out_proj = nn.Linear(self.channels, self.channels)

        # v10 Fix 3: Learnable per-head spatial distance bias gate gamma_h in (0.1, 2.0), initialized at 1.0
        # 0.1 + 1.9 * sigmoid(theta_0) = 1.0  =>  sigmoid(theta_0) = 0.9 / 1.9  =>  theta_0 = log(0.9 / 1.0)
        init_theta = math.log(0.9 / 1.0)
        self.bias_theta = nn.Parameter(torch.full((self.num_heads,), init_theta, dtype=torch.float32))

        # v10 Fix 2: Post-attention FFN operating purely on BEV-attended features (no query skip shortcut)
        self.ffn_norm = nn.LayerNorm(self.channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, self.channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels * 2, self.channels),
        )
        self.head_norm = nn.LayerNorm(self.channels)

        # Lightweight planning prediction heads (trained only by Teacher BEV probe loss)
        hidden = 128
        self.sim_head = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 5),
        )
        self.im_head = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(self.channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.fut_ts * self.traj_dims),
        )

        if self.version == "v14" and self.domain == "resmap":
            # v14 Fix 5 (EPDMS-Grounded Map Probe): Slots 3 and 4 of P_map.sim_head supervise
            # per-anchor Lane Keeping (LK, weight 1.0) and Driving Direction Compliance (DDC, weight 0.75)
            # alongside DAC (slot 1, weight 1.0) and EP (slot 2, weight 0.5).
            sim_mask = (0.0, 1.0, 0.5, 1.0, 0.75)
        else:
            sim_mask = PROBE_SIM_Masks[self.domain]
        self.register_buffer("sim_mask", torch.tensor(sim_mask, dtype=torch.float32), persistent=False)
        self.im_weight = 0.2 if self.domain == "bevfusion" else 0.5
        self.offset_weight = 0.5

        # Non-persistent EMA shadow buffers for stable KD targets (v10 Fix 4)
        self.register_buffer("ema_step", torch.zeros((), dtype=torch.long), persistent=False)
        self._ema_names = (
            "bev_norm.weight", "bev_norm.bias",
            "q_proj.weight", "q_proj.bias",
            "k_proj.weight", "k_proj.bias",
            "v_proj.weight", "v_proj.bias",
            "out_proj.weight", "out_proj.bias",
            "bias_theta",
            "ffn_norm.weight", "ffn_norm.bias",
            "ffn.0.weight", "ffn.0.bias",
            "ffn.2.weight", "ffn.2.bias",
            "head_norm.weight", "head_norm.bias",
        )
        param_dict = dict(self.named_parameters())
        for name in self._ema_names:
            buf_name = "ema__" + name.replace(".", "__")
            self.register_buffer(buf_name, param_dict[name].detach().clone(), persistent=False)

    def _gamma(self, theta: torch.Tensor) -> torch.Tensor:
        """Map unconstrained theta [H] to bounded spatial bias scale gamma in (0.1, 2.0)."""
        return 0.1 + 1.9 * torch.sigmoid(theta)

    @torch.no_grad()
    def update_ema(self) -> None:
        """Bias-corrected fast EMA update: decay = min(ema_decay, (1 + step) / (10 + step))."""
        if not self.use_ema:
            return
        step = int(self.ema_step.item())
        decay = min(self.ema_decay, float(1 + step) / float(10 + step))
        param_dict = dict(self.named_parameters())
        for name in self._ema_names:
            buf = getattr(self, "ema__" + name.replace(".", "__"))
            p = param_dict[name].detach()
            buf.mul_(decay).add_(p, alpha=1.0 - decay)
        self.ema_step.add_(1)

    def _encode_with_params(
        self,
        q: torch.Tensor,
        bev: torch.Tensor,
        bev_pos: torch.Tensor,
        spatial_bias: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run L=1 cross-attention + FFN using explicit parameter dictionary `params`."""
        bs, k, c = q.shape
        hw = bev.size(1)
        scale = 1.0 / (self.head_dim ** 0.5)

        bev_normed = F.layer_norm(
            bev, (c,), params["bev_norm.weight"], params["bev_norm.bias"]
        )
        q_h = F.linear(q, params["q_proj.weight"], params["q_proj.bias"]).view(
            bs, k, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k_h = F.linear(bev_normed + bev_pos, params["k_proj.weight"], params["k_proj.bias"]).view(
            bs, hw, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v_h = F.linear(bev_normed, params["v_proj.weight"], params["v_proj.bias"]).view(
            bs, hw, self.num_heads, self.head_dim
        ).transpose(1, 2)

        gamma = self._gamma(params["bias_theta"]).to(dtype=q.dtype, device=q.device).view(1, self.num_heads, 1, 1)
        logits = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale + gamma * spatial_bias
        attn = F.softmax(logits, dim=-1)  # [B, H, K, HW]

        out = torch.matmul(attn, v_h).transpose(1, 2).contiguous().view(bs, k, c)
        feat = F.linear(out, params["out_proj.weight"], params["out_proj.bias"])

        # v10 Fix 2: Pure BEV post-attention representation (no additive q skip-connection)
        ffn_in = F.layer_norm(feat, (c,), params["ffn_norm.weight"], params["ffn_norm.bias"])
        ffn_h = F.relu(F.linear(ffn_in, params["ffn.0.weight"], params["ffn.0.bias"]), inplace=False)
        ffn_out = F.linear(ffn_h, params["ffn.2.weight"], params["ffn.2.bias"])
        h_bev = F.layer_norm(feat + ffn_out, (c,), params["head_norm.weight"], params["head_norm.bias"])
        return attn, h_bev, gamma.squeeze()

    def forward_teacher(
        self,
        q: torch.Tensor,
        teacher_bev: torch.Tensor,
        bev_pos: torch.Tensor,
        spatial_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Teacher forward: updates probe parameters via L_probe; returns KD targets (attn_t, h_t)."""
        # Teacher BEV and positional encoding are detached so L_probe only updates the probe
        t_bev = teacher_bev.detach()
        t_pos = bev_pos.detach()

        live_params = {name: p for name, p in self.named_parameters() if name in self._ema_names}
        attn_live, h_live, gamma_live = self._encode_with_params(q, t_bev, t_pos, spatial_bias, live_params)

        # Prediction heads for domain-grounded planning loss
        bs, k, _ = h_live.shape
        sim_logits = self.sim_head(h_live).permute(0, 2, 1)  # [B, 5, K]
        im_logits = self.im_head(h_live).squeeze(-1)         # [B, K]
        raw_offset = self.offset_head(h_live).view(bs, k, self.fut_ts, self.traj_dims)
        if self.traj_dims == 3:
            offset = torch.cat([raw_offset[..., :2], raw_offset[..., 2:].tanh() * math.pi], dim=-1)
        else:
            offset = raw_offset

        preds = {
            "sim_logits": sim_logits,
            "im_logits": im_logits,
            "offset": offset,
            "gamma": gamma_live.detach(),
        }

        # Update EMA shadow weights and compute stable KD targets if enabled
        if self.training and self.use_ema:
            self.update_ema()
            with torch.no_grad():
                ema_params = {
                    name: getattr(self, "ema__" + name.replace(".", "__"))
                    for name in self._ema_names
                }
                attn_t, h_t, _ = self._encode_with_params(
                    q.detach(), t_bev, t_pos, spatial_bias, ema_params
                )
        else:
            attn_t = attn_live.detach()
            h_t = h_live.detach()

        return attn_t, h_t, preds

    def forward_student(
        self,
        q: torch.Tensor,
        student_bev: torch.Tensor,
        bev_pos: torch.Tensor,
        spatial_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Student forward: stop-grad on probe parameters; gradients flow ONLY to student_bev / student_proj."""
        if self.use_ema:
            detached_params = {
                name: getattr(self, "ema__" + name.replace(".", "__")).detach()
                for name in self._ema_names
            }
        else:
            detached_params = {
                name: p.detach() for name, p in self.named_parameters() if name in self._ema_names
            }

        attn_s, h_s, _ = self._encode_with_params(
            q.detach(), student_bev, bev_pos, spatial_bias, detached_params
        )

        with torch.no_grad():
            sim_s = self.sim_head(h_s.detach()).permute(0, 2, 1).sigmoid()
        return attn_s, h_s, {"sim_probs": sim_s}

    def compute_probe_loss(
        self,
        preds: Dict[str, torch.Tensor],
        trajectory_anchors: torch.Tensor,
        gt_trajectory: Optional[torch.Tensor] = None,
        sim_reward: Optional[torch.Tensor] = None,
        sim_reward_valid: Optional[torch.Tensor] = None,
        sibling_pairs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute domain-specific planning loss L_probe,b on Teacher BEV predictions."""
        sim_logits = preds["sim_logits"]  # [B, 5, K]
        im_logits = preds["im_logits"]    # [B, K]
        offset = preds["offset"]          # [B, K, T, 3]
        bs, _, k = sim_logits.shape
        device = sim_logits.device
        dtype = sim_logits.dtype

        # Touch all output heads with 0.0 so plain DDP never sees an unused parameter
        zero_touch = 0.0 * (sim_logits.sum() + im_logits.sum() + offset.sum())
        metrics: Dict[str, torch.Tensor] = {}
        tag = "det" if self.domain == "bevfusion" else "map"

        # 1. Domain-masked PDM simulation reward BCE loss (+ v12 Velocity-Sibling Cliff Ranking Loss)
        cliff_loss = zero_touch
        if sim_reward is not None and sim_reward.numel() > 0:
            target = sim_reward.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            bce = F.binary_cross_entropy_with_logits(sim_logits.float(), target, reduction="none")  # [B, 5, K]
            mask = self.sim_mask.to(device=device, dtype=torch.float32).view(1, 5, 1)
            if sim_reward_valid is not None and sim_reward_valid.numel() == bs:
                valid = sim_reward_valid.to(device=device, dtype=torch.float32).view(bs, 1, 1)
            else:
                valid = torch.ones((bs, 1, 1), device=device, dtype=torch.float32)
            denom = (valid.sum() * mask.sum() * k).clamp_min(1.0)
            sim_loss = ((bce * mask * valid).sum() / denom).to(dtype=dtype) * 5.0

            if self.version in ("v12", "v13", "v14") and sibling_pairs is not None and sibling_pairs.numel() > 0:
                sib = sibling_pairs.to(device=device, dtype=torch.long)
                i_slow, j_fast = sib[:, 0], sib[:, 1]
                if self.domain == "bevfusion":
                    safe_t = (target[:, 0] * target[:, 3]).clamp(0.0, 1.0)
                    safe_z = 0.5 * (sim_logits[:, 0].float() + sim_logits[:, 3].float())
                    cliff = (safe_t[:, i_slow] - safe_t[:, j_fast]).clamp(0.0, 1.0)
                    active = (cliff > 0.25).float() * valid.view(bs, 1)
                    c_w = active * cliff
                    d_z = safe_z[:, i_slow] - safe_z[:, j_fast]
                elif self.version == "v14":
                    # v14 Fix 5 (EPDMS Map Cliff): Rank sibling anchors by composite Map Compliance
                    # safe_t = DAC * DDC * (0.4 + 0.6 * LK) in both directions (|diff| > 0.20),
                    # breaking the in-road DAC=1.0 tie so P_map distinguishes lane-centered vs drifting/oncoming siblings.
                    safe_t = (target[:, 1] * target[:, 4] * (0.4 + 0.6 * target[:, 3])).clamp(0.0, 1.0)
                    safe_z = (sim_logits[:, 1].float() + sim_logits[:, 3].float() + sim_logits[:, 4].float()) / 3.0
                    diff_t = safe_t[:, i_slow] - safe_t[:, j_fast]
                    cliff = diff_t.abs().clamp(0.0, 1.0)
                    active = (cliff > 0.20).float() * valid.view(bs, 1)
                    c_w = active * cliff
                    sign_t = torch.where(diff_t >= 0.0, 1.0, -1.0)
                    d_z = sign_t * (safe_z[:, i_slow] - safe_z[:, j_fast])
                else:
                    safe_t = target[:, 1].clamp(0.0, 1.0)
                    safe_z = sim_logits[:, 1].float()
                    cliff = (safe_t[:, i_slow] - safe_t[:, j_fast]).clamp(0.0, 1.0)
                    active = (cliff > 0.25).float() * valid.view(bs, 1)
                    c_w = active * cliff
                    d_z = safe_z[:, i_slow] - safe_z[:, j_fast]
                cliff_loss = ((c_w * F.softplus(1.0 - d_z)).sum() / c_w.sum().clamp_min(1.0)).to(dtype=dtype)
                metrics[f"probe/{tag}/cliff_loss"] = cliff_loss.detach()

            with torch.no_grad():
                probs = sim_logits.detach().float().sigmoid()
                if self.domain == "bevfusion":
                    metrics["probe/det/nc_err"] = (probs[:, 0] - target[:, 0]).abs().mean()
                    metrics["probe/det/ttc_err"] = (probs[:, 3] - target[:, 3]).abs().mean()
                else:
                    metrics["probe/map/dac_err"] = (probs[:, 1] - target[:, 1]).abs().mean()
                    if self.version == "v14":
                        metrics["probe/map/lk_err"] = (probs[:, 3] - target[:, 3]).abs().mean()
                        metrics["probe/map/ddc_err"] = (probs[:, 4] - target[:, 4]).abs().mean()
        else:
            sim_loss = zero_touch

        # 2. Imitation soft-CE and WTA offset L1 loss
        if gt_trajectory is not None and gt_trajectory.numel() > 0:
            gt = gt_trajectory[..., :self.traj_dims].reshape(bs, 1, -1).to(device=device, dtype=dtype)
            flat_anchors = trajectory_anchors[..., :self.traj_dims].reshape(1, k, -1).to(device=device, dtype=dtype)
            dist = torch.norm(flat_anchors[..., :2] - gt[..., :2], dim=-1)  # [B, K]
            winner = dist.argmin(dim=-1)
            rows = torch.arange(bs, device=device)
            gt_offset = gt[:, 0] - flat_anchors[0, winner]
            offset_loss = F.l1_loss(offset.reshape(bs, k, -1)[rows, winner], gt_offset, reduction="mean")

            reward_target = F.softmax(-dist.float(), dim=-1)
            log_im = F.log_softmax(im_logits.float(), dim=-1)
            im_loss = (-(reward_target * log_im).sum(dim=-1).mean()).to(dtype=dtype)
        else:
            offset_loss = zero_touch
            im_loss = zero_touch

        total_probe_loss = (
            sim_loss
            + 0.5 * cliff_loss
            + self.im_weight * im_loss
            + self.offset_weight * offset_loss
            + zero_touch
        )
        metrics[f"probe/{tag}/loss"] = total_probe_loss.detach()
        metrics[f"probe/{tag}/gamma_mean"] = preds["gamma"].mean().detach()
        return total_probe_loss, metrics
