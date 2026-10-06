"""A scorer that can only see perception ENTITIES -- no image, no BEV.

Same job as the v2 anchor planner's score heads (predict the five PDM sub-scores of every
candidate trajectory), but the only inputs are the GT agents (box, velocity, future) and the GT
map polylines, as tokens.  With no BEV to fall back on, the scores must come from the entities,
so the model is (a) the reference for the intervention diagnostic -- how much a score SHOULD move
when an entity is removed -- and (b) the teacher candidate for distilling that dependence into a
sensor model.

Entities enter through ``key_padding_mask``, so removing one at inference is exact.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .anchor_planner import SIM_KEYS


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class EntityCritic(nn.Module):
    def __init__(self, anchor_file: str, embed_dims: int = 128, num_layers: int = 3, num_heads: int = 8,
                 fut_ts: int = 8, map_pts: int = 20, map_classes: int = 4, agent_fut: int = 8):
        super().__init__()
        anchors = torch.tensor(np.load(anchor_file), dtype=torch.float32)
        self.register_buffer("trajectory_anchors", anchors)
        self.num_anchors, self.fut_ts, self.traj_dims = anchors.shape[0], anchors.shape[1], anchors.shape[2]
        self.traj_mlp = mlp(self.fut_ts * self.traj_dims, 128, embed_dims)
        layer = nn.TransformerEncoderLayer(embed_dims, num_heads, 512, dropout=0.1, batch_first=True)
        self.cluster_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.ego_mlp = mlp(8, 128, embed_dims)                               # vx, vy, ax, ay + command (4)
        self.agent_mlp = mlp(9 + agent_fut * 3, 128, embed_dims)             # box + future offsets + mask
        self.map_mlp = mlp(map_pts * 2 + map_classes, 128, embed_dims)
        self.type_embed = nn.Embedding(2, embed_dims)                        # agent / map
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(embed_dims, num_heads, 512, dropout=0.1, batch_first=True, norm_first=True)
            for _ in range(num_layers))
        self.norm = nn.LayerNorm(embed_dims)
        self.score_heads = nn.ModuleList(mlp(embed_dims, 128, 1) for _ in SIM_KEYS)
        self.map_classes = map_classes

    def tokens(self, b: Dict[str, torch.Tensor], drop_agent: Optional[torch.Tensor] = None):
        """-> entity tokens [B, N, C] and their padding mask [B, N] (True = ignored)."""
        agents = torch.cat([b["agents"], b["agent_fut"].flatten(2), b["agent_fut_mask"]], dim=-1)
        at = self.agent_mlp(agents) + self.type_embed.weight[0]
        one_hot = torch.nn.functional.one_hot(b["map_labels"].clamp(min=0), self.map_classes).float()
        mt = self.map_mlp(torch.cat([b["map_pts"].flatten(2), one_hot], dim=-1)) + self.type_embed.weight[1]
        valid = torch.cat([b["agent_valid"], b["map_valid"]], dim=1) > 0.5
        if drop_agent is not None:                                            # [B] index or -1
            hit = torch.nn.functional.one_hot(drop_agent.clamp(min=0), at.shape[1]).bool()
            valid[:, : at.shape[1]] &= ~(hit & (drop_agent >= 0).unsqueeze(-1))
        return torch.cat([at, mt], dim=1), ~valid

    def forward(self, b: Dict[str, torch.Tensor], trajectories: Optional[torch.Tensor] = None,
                drop_agent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """-> predicted sub-scores [B, 5, K] (sigmoid, SIM_KEYS order)."""
        bs = b["agents"].shape[0]
        traj = self.trajectory_anchors.unsqueeze(0).expand(bs, -1, -1, -1) if trajectories is None else trajectories
        q = self.cluster_encoder(self.traj_mlp(traj.flatten(2)))
        q = q + self.ego_mlp(torch.cat([b["ego_status"], b["command"]], dim=-1)).unsqueeze(1)
        mem, pad = self.tokens(b, drop_agent)
        for layer in self.layers:
            q = layer(q, mem, memory_key_padding_mask=pad)
        q = self.norm(q)
        return torch.cat([h(q) for h in self.score_heads], dim=-1).permute(0, 2, 1).sigmoid()


def critic_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """WoTE's sim_reward_loss: BCE over the five metrics and all anchors, x5."""
    eps = 1e-6
    bce = -(target * (pred + eps).log() + (1 - target) * (1 - pred + eps).log())
    v = valid.view(-1, 1, 1).to(pred.dtype)
    return (bce * v).sum() / (v.sum() * bce.shape[1] * bce.shape[2]).clamp(min=1.0) * 5
