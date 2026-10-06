"""Anchor-based planning with PDM-score rewards (PARA-SSR v2), ported from WoTE.

Source: WoTE (github.com/liyingyanUCAS/WoTE), ``navsim/agents/WoTE`` in this
repo -- ``WoTE_model.py`` / ``WoTE_loss.py``.  Everything below follows that code
and its constants; the one part left out is WoTE's latent world model (future
BEV roll-out feeding the reward heads).  Here the reward heads read the anchor
queries after PARA-SSR's own planner layers (BEV + det/motion + map memories),
the role WoTE's ``offset_tf_decoder`` has.

    anchors   256 K-means trajectories [256, 8, 3] (x fwd, y left, heading), fixed
    query_k   = MLP([ego/command feature ; encoder(MLP(anchor_k))])       (WoTE: mlp_planning_vb,
                cluster_encoder, encode_ego_feat_mlp)
    -> PARA-SSR planner layers
    offset_k  = TrajectoryOffsetHead(query_k), heading = tanh * pi          (WoTE offset_head)
    im_k      = softmax_k(reward_head(query_k))                             (WoTE reward_head)
    sim_k,m   = sigmoid(sim_reward_heads[m](query_k)), m = NC, DAC, EP, TTC, C

Losses (``anchor_plan_losses``): WTA L1 on the offset of the anchor nearest the
human trajectory, soft cross-entropy of ``im`` against softmax(-L2 to the human
trajectory), and BCE of ``sim`` against the PDM scores of the 256 anchors
precomputed per token (WoTE's ``formatted_pdm_score_256.npy``), times 5.

Selection (``weighted_reward``): WoTE's
    w0 log im + w1 log NC + w2 log DAC + w3 log(5 TTC + 2 C + 5 EP),  w = [0.1, 0.5, 0.5, 1.0]
argmax over anchors; the output is anchor + offset.  ``plan_topk`` ranked
candidates are returned as well.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kinematics import bezier_xyyaw, bicycle_rollout, clamp_controls, poses_to_controls

SIM_KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
            "time_to_collision_within_bound", "comfort")


class AnchorPlanner(nn.Module):
    def __init__(
        self,
        anchor_file: str,
        embed_dims: int = 256,
        fut_ts: int = 8,
        traj_dims: int = 3,
        reward_weights: Sequence[float] = (0.1, 0.5, 0.5, 1.0),
        topk: int = 6,
        kinematic: bool = False,
        dt: float = 0.5,
        heading_from_xy: bool = False,
    ):
        super().__init__()
        anchors = torch.tensor(np.load(anchor_file), dtype=torch.float32)
        if anchors.ndim != 3 or anchors.shape[1:] != (fut_ts, traj_dims):
            raise ValueError(f"anchors must be [K, {fut_ts}, {traj_dims}], got {tuple(anchors.shape)}")
        # A buffer, not WoTE's frozen Parameter: it must not reach the optimiser
        # and a checkpoint should carry the vocabulary it was trained with.
        self.register_buffer("trajectory_anchors", anchors)
        self.num_anchors = anchors.shape[0]
        self.fut_ts, self.traj_dims = fut_ts, traj_dims
        self.reward_weights = tuple(float(w) for w in reward_weights)
        if len(self.reward_weights) != 4:
            raise ValueError("reward_weights must be (w_im, w_nc, w_dac, w_rest)")
        self.topk = int(topk)
        # kinematic=True: the offset head predicts control corrections (da, domega) per
        # step on top of the anchor's own controls, and the poses are their bicycle
        # rollout (modules/kinematics.py, from TOAD) -- heading consistent with the path.
        self.kinematic = bool(kinematic)
        self.dt = float(dt)
        if self.kinematic and traj_dims != 3:
            raise ValueError("kinematic anchor planner needs (x, y, heading) anchors")
        # heading_from_xy=True (DiffusionDriveV2): the offset head predicts x, y only and
        # the heading is the tangent of the Bezier curve through the predicted points.
        self.heading_from_xy = bool(heading_from_xy)
        if self.heading_from_xy and (self.kinematic or traj_dims != 3):
            raise ValueError("heading_from_xy needs (x, y, heading) anchors and kinematic=False")

        hidden = 128                                          # WoTE SCORE_HEAD_HIDDEN_DIM
        self.mlp_planning_vb = nn.Sequential(
            nn.Linear(fut_ts * traj_dims, hidden), nn.ReLU(), nn.Linear(hidden, embed_dims))
        layer = nn.TransformerEncoderLayer(d_model=embed_dims, nhead=8, dim_feedforward=512,
                                           dropout=0.1, batch_first=True)
        self.cluster_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.encode_ego_feat_mlp = nn.Sequential(
            nn.Linear(2 * embed_dims, embed_dims), nn.ReLU(), nn.Linear(embed_dims, embed_dims))
        self.offset_head = nn.Sequential(                      # WoTE TrajectoryOffsetHead, d_ffn 1024
            nn.Linear(embed_dims, 1024), nn.ReLU(),
            nn.Linear(1024, fut_ts * (2 if (self.kinematic or self.heading_from_xy) else traj_dims)))

        def score_head():
            return nn.Sequential(nn.Linear(embed_dims, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        self.reward_head = score_head()
        self.sim_reward_heads = nn.ModuleList(score_head() for _ in SIM_KEYS)

    def queries(self, ego_feat: torch.Tensor, trajectories: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ego/command feature [B, 1, C] -> one query per anchor [B, K, C].

        ``trajectories`` [B, K, T, 3] encodes those instead of the fixed anchors (WoTE's
        test-time re-encoding of the refined trajectories, ``WoTE_model.py`` is_eval branch).
        """
        bs = ego_feat.shape[0]
        if trajectories is None:
            flat = self.trajectory_anchors.reshape(1, self.num_anchors, -1).to(ego_feat.dtype).expand(bs, -1, -1)
        else:
            flat = trajectories.reshape(bs, self.num_anchors, -1).to(ego_feat.dtype)
        anchor_feat = self.cluster_encoder(self.mlp_planning_vb(flat))
        return self.encode_ego_feat_mlp(torch.cat([ego_feat.expand(-1, self.num_anchors, -1), anchor_feat], dim=-1))

    def weighted_reward(self, im: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
        """im [B, K] (softmax), sim [B, 5, K] (sigmoid, SIM_KEYS order) -> [B, K]."""
        eps = 1e-6
        w = self.reward_weights
        nc, dac, ep, ttc, comfort = sim.unbind(dim=1)
        return (w[0] * torch.log(im + eps) + w[1] * torch.log(nc + eps) + w[2] * torch.log(dac + eps)
                + w[3] * torch.log(5 * ttc + 2 * comfort + 5 * ep + eps))

    def score(self, h: torch.Tensor):
        """Queries after the planner layers [B, K, C] -> (im [B, K], sim [B, 5, K], final [B, K])."""
        im = torch.softmax(self.reward_head(h).squeeze(-1).float(), dim=-1)
        sim = torch.cat([head(h) for head in self.sim_reward_heads], dim=-1).permute(0, 2, 1).float().sigmoid()
        return im, sim, self.weighted_reward(im, sim)

    def forward(self, h: torch.Tensor, init_speed: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Anchor queries after the planner layers [B, K, C] -> predictions.

        ``init_speed`` [B] (current longitudinal speed) is needed when ``kinematic``.
        """
        bs = h.shape[0]
        if self.kinematic:
            if init_speed is None:
                raise ValueError("kinematic anchor planner needs the current speed")
            anchors = self.trajectory_anchors.to(h.dtype).unsqueeze(0).expand(bs, -1, -1, -1)    # [B, K, T, 3]
            v0 = init_speed.to(h.dtype).view(bs, 1).expand(bs, self.num_anchors)
            base = poses_to_controls(anchors, v0, self.dt)                                        # [B, K, T, 2]
            delta = self.offset_head(h).view(bs, self.num_anchors, self.fut_ts, 2)
            poses, _ = bicycle_rollout(clamp_controls(base + delta), v0, self.dt)
            # expressed as an offset from the anchor, so WoTE's losses apply unchanged
            offset = poses - anchors
        elif self.heading_from_xy:
            anchors = self.trajectory_anchors.to(h.dtype).unsqueeze(0).expand(bs, -1, -1, -1)    # [B, K, T, 3]
            xy = anchors[..., :2] + self.offset_head(h).view(bs, self.num_anchors, self.fut_ts, 2)
            poses = bezier_xyyaw(xy, anchors[..., 2])
            offset = poses - anchors          # WoTE's losses apply unchanged; the heading term trains xy
        else:
            offset = self.offset_head(h).view(bs, self.num_anchors, self.fut_ts, self.traj_dims)
            if self.traj_dims == 3:
                offset = torch.cat([offset[..., :2], offset[..., 2:].tanh() * np.pi], dim=-1)
        im, sim, final = self.score(h)
        all_traj = self.trajectory_anchors.unsqueeze(0).to(offset.dtype) + offset          # [B, K, T, 3]
        k = min(self.topk, self.num_anchors)
        top = final.topk(k, dim=-1).indices                                               # [B, k]
        gather = top[:, :, None, None].expand(-1, -1, self.fut_ts, self.traj_dims)
        top_traj = all_traj.gather(1, gather)
        return {
            "trajectory": top_traj[:, 0],
            "plan_topk_trajectory": top_traj,
            "plan_topk_reward": final.gather(1, top),
            "plan_topk_index": top,
            "plan_final_rewards": final,
            "trajectory_offset": offset,
            "im_rewards": im,
            "sim_rewards": sim,
            "trajectory_anchors": self.trajectory_anchors,
        }


def anchor_plan_losses(
    predictions: Dict[str, torch.Tensor],
    gt_trajectory: torch.Tensor,
    sim_reward: torch.Tensor,
    sim_valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """WoTE's three planning terms.  gt_trajectory [B, T, 3]; sim_reward [B, 5, K]; sim_valid [B]."""
    anchors = predictions["trajectory_anchors"]
    offset = predictions["trajectory_offset"]
    bs, k = offset.shape[:2]
    gt = gt_trajectory.reshape(bs, 1, -1).to(offset.dtype)
    flat = anchors.reshape(1, k, -1).to(offset.dtype)
    dist = torch.norm(flat - gt, dim=2)                                   # [B, K]
    winner = dist.argmin(dim=1)
    rows = torch.arange(bs, device=offset.device)
    gt_offset = gt[:, 0] - flat[0, winner]
    traj_offset_loss = F.l1_loss(offset.reshape(bs, k, -1)[rows, winner], gt_offset, reduction="mean")

    reward_target = torch.softmax(-dist.float(), dim=-1)
    im = predictions["im_rewards"].clamp(1e-6, 1 - 1e-6)
    im_reward_loss = -(reward_target * im.log()).sum() / bs

    eps = 1e-6
    sim = predictions["sim_rewards"]
    target = sim_reward.to(sim.dtype)
    bce = -(target * (sim + eps).log() + (1 - target) * (1 - sim + eps).log())   # [B, 5, K]
    valid = sim_valid.to(sim.dtype).view(bs, 1, 1)
    sim_reward_loss = (bce * valid).sum() / (valid.sum() * bce.shape[1] * bce.shape[2]).clamp(min=1.0) * 5
    return {"traj_offset_loss": traj_offset_loss, "im_reward_loss": im_reward_loss,
            "sim_reward_loss": sim_reward_loss}
