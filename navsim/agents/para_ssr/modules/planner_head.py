"""SSR's navigation-guided sparse planner, ported to navsim.

Path (unchanged from the nuScenes ``ParaSSRHead``):

    BEV encoder -> navigation SE gating -> TokenLearner (16 scene tokens)
    -> latent self-attention decoder -> waypoint cross-attention decoder -> MLP

Two adaptations, both forced by navsim's output contract:

* **4 command branches, not 3.**  navsim's ``driving_command`` is a 4-way
  one-hot (left / straight / right / unknown); nuScenes' converter produced 3.
* **3 output dims per waypoint, not 2.**  navsim scores a ``Trajectory`` of
  ``(x, y, heading)`` poses, so the per-step offset regression carries heading.
  ``loss_plan_reg`` weights heading separately (``heading_weight``) because it
  is in radians while x/y are in metres.

The horizon follows navsim's ``TrajectorySampling(time_horizon=4,
interval_length=0.5)`` -> ``fut_ts = 8``, against nuScenes' 6.

``use_stl=False`` replaces the scene-token path with PARA-Drive's planner
(Weng et al., CVPR 2024, Sec. 4): a learnable plan query concatenated with the
command embedding cross-attends to the full BEV, then an MLP regresses the
trajectory.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .tokenlearner import TokenLearnerV11
from .transformer_blocks import (
    LearnedPositionalEncoding,
    build_self_attn_decoder,
)


class SELayer(nn.Module):
    """Channel gating of the BEV feature by the navigation-command embedding."""

    def __init__(self, channels: int, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x: torch.Tensor, x_se: torch.Tensor) -> torch.Tensor:
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)


class ParaSSRPlannerHead(nn.Module):
    """BEV encoder wrapper + navigation-guided sparse-token planner.

    Like the nuScenes head, this owns the BEV encoder so ``only_bev=True`` can
    short-circuit it for history frames, and it exposes ``bev_embed`` so the
    detector can fan it out to the parallel auxiliary heads.
    """

    def __init__(
        self,
        transformer: nn.Module,
        bev_h: int = 100,
        bev_w: int = 100,
        embed_dims: int = 256,
        pc_range: Sequence[float] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        num_scenes: int = 16,
        num_reg_fcs: int = 2,
        fut_ts: int = 8,
        ego_fut_mode: int = 4,
        num_navi_cmd: int = 4,
        traj_dims: int = 3,
        latent_num_layers: int = 3,
        way_num_layers: int = 1,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        use_lidar: bool = False,
        use_stl: bool = True,
        plan_num_layers: int = 3,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.pc_range = list(pc_range)
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_scenes = num_scenes
        self.num_reg_fcs = num_reg_fcs
        self.fut_ts = fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_navi_cmd = num_navi_cmd
        self.traj_dims = traj_dims

        self.transformer = transformer
        self.positional_encoding = LearnedPositionalEncoding(
            embed_dims // 2, bev_h, bev_w
        )

        # With LiDAR the encoder starts from the LiDAR BEV (SafeDrive), so the
        # learned query table is not built at all: an unused parameter would
        # also trip DDP's unused-parameter check.
        self.bev_embedding = None if use_lidar else nn.Embedding(bev_h * bev_w, embed_dims)
        self.navi_embedding = nn.Embedding(num_navi_cmd, embed_dims)
        self.use_stl = use_stl

        def reg_fcs(out_dims: int) -> nn.Sequential:
            layers = []
            for _ in range(num_reg_fcs):
                layers.append(nn.Linear(embed_dims, embed_dims))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(embed_dims, out_dims))
            return nn.Sequential(*layers)

        # Branches construct modules in the same order as before use_stl
        # existed, so a seeded STL run initialises identically.
        if use_stl:
            self.navi_se = SELayer(embed_dims)

            self.tokenlearner = TokenLearnerV11(num_scenes, embed_dims * 2)
            self.latent_decoder = build_self_attn_decoder(
                latent_num_layers,
                embed_dims,
                num_heads,
                feedforward_channels,
                ("self_attn", "norm", "ffn", "norm"),
                attn_dropout=0.0,
                ffn_dropout=0.0,
            )

            self.way_point = nn.Embedding(ego_fut_mode * fut_ts, embed_dims * 2)
            self.way_decoder = build_self_attn_decoder(
                way_num_layers,
                embed_dims,
                num_heads,
                feedforward_channels,
                ("cross_attn", "norm", "ffn", "norm"),
                attn_dropout=0.0,
                ffn_dropout=0.0,
            )
            self.ego_fut_decoder = reg_fcs(traj_dims)
            decoders = (self.latent_decoder, self.way_decoder)
        else:
            # PARA-Drive / UniAD planner: one learnable plan query, fused with
            # the command embedding, cross-attends to the full BEV and an MLP
            # regresses the whole horizon.  The command conditions the query
            # instead of selecting an output branch.
            self.plan_query = nn.Embedding(1, embed_dims)
            self.plan_query_pos = nn.Embedding(1, embed_dims)
            self.plan_fuser = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
            self.plan_decoder = build_self_attn_decoder(
                plan_num_layers,
                embed_dims,
                num_heads,
                feedforward_channels,
                ("cross_attn", "norm", "ffn", "norm"),
                attn_dropout=0.0,
                ffn_dropout=0.0,
            )
            self.ego_fut_decoder = reg_fcs(fut_ts * traj_dims)
            decoders = (self.plan_decoder,)

        # BaseModule called this in the original implementation.  Keep the
        # planner decoders on the same Xavier initialization rather than
        # PyTorch Linear's default Kaiming-uniform initialization.
        for decoder in decoders:
            for parameter in decoder.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        mlvl_feats: Sequence[torch.Tensor],
        lidar2img: torch.Tensor,
        image_hw: torch.Tensor,
        ego_motion: torch.Tensor,
        bev_shift: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
        only_bev: bool = False,
        cmd: Optional[torch.Tensor] = None,
        lidar_bev: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Args:
            mlvl_feats: list of ``[bs, num_cam, C, H, W]``
            cmd: one-hot navigation command ``[bs, num_navi_cmd]``
            lidar_bev: ``[bs, C, bev_h, bev_w]``; required iff built with
                ``use_lidar``
        Returns:
            ``only_bev``: the BEV feature. Otherwise a dict with ``bev_embed``
            ``[bs, bev_h*bev_w, C]``, ``scene_query``, ``token_attn`` and
            ``ego_fut_preds`` ``[bs, ego_fut_mode, fut_ts, traj_dims]``.
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        if self.bev_embedding is None:
            if lidar_bev is None:
                raise ValueError("planner head built with use_lidar needs lidar_bev")
            bev_queries = None
            lidar_bev = lidar_bev.to(dtype)
        else:
            if lidar_bev is not None:
                raise ValueError("camera-only planner head received lidar_bev")
            bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=device, dtype=dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            bev_pos=bev_pos,
            lidar2img=lidar2img,
            image_hw=image_hw,
            ego_motion=ego_motion,
            bev_shift=bev_shift,
            prev_bev=prev_bev,
            lidar_bev=lidar_bev,
        )
        if only_bev:
            return bev_embed

        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)

        if cmd is None:
            raise ValueError("cmd is required when only_bev=False")
        if cmd.size(0) != bs or cmd.numel() != bs * self.num_navi_cmd:
            raise ValueError(
                f"expected one {self.num_navi_cmd}-way command per sample, "
                f"but got cmd shape {tuple(cmd.shape)} for batch size {bs}"
            )
        cmd = cmd.reshape(bs, self.num_navi_cmd)
        cmd_idx = cmd.argmax(dim=-1)

        if not self.use_stl:
            navi = self.navi_embedding(cmd_idx)  # [B, C]
            plan_query = self.plan_query.weight.to(dtype).expand(bs, -1)
            plan_query = self.plan_fuser(torch.cat((plan_query, navi), -1))
            plan_query_pos = self.plan_query_pos.weight.to(dtype).expand(bs, -1)
            plan_query = self.plan_decoder(
                query=plan_query.unsqueeze(0),  # [1, B, C]
                key=bev_embed.permute(1, 0, 2),  # [HW, B, C]
                value=bev_embed.permute(1, 0, 2),
                query_pos=plan_query_pos.unsqueeze(0),
                key_pos=pos_embd.permute(1, 0, 2),
            )
            plan = self.ego_fut_decoder(plan_query[0]).view(
                bs, 1, self.fut_ts, self.traj_dims
            )
            # The plan is already command-conditioned; filling every command
            # slot keeps the [B, mode, T, D] contract, so the commanded-branch
            # loss (and its mean over all slots) and select_trajectory are
            # unchanged from the STL planner.
            return {
                "bev_embed": bev_embed,
                "scene_query": plan_query,
                "token_attn": None,
                "ego_fut_preds": plan.expand(bs, self.ego_fut_mode, self.fut_ts, self.traj_dims),
            }

        navi_embed = self.navi_embedding(cmd_idx).unsqueeze(1)  # [B, 1, C]
        bev_navi_embed = self.navi_se(bev_embed, navi_embed)

        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)
        learned_latent_query, selected = self.tokenlearner(bev_query)

        learned_latent_query = learned_latent_query.permute(1, 0, 2)
        latent_query, latent_pos = torch.split(
            learned_latent_query, self.embed_dims, dim=2
        )

        latent_query = self.latent_decoder(
            query=latent_query,
            key=latent_query,
            value=latent_query,
            query_pos=latent_pos,
            key_pos=latent_pos,
        )

        way_point = self.way_point.weight.to(dtype)
        wp_pos, way_point = torch.split(way_point, self.embed_dims, dim=1)
        wp_pos = wp_pos.unsqueeze(0).expand(bs, -1, -1).permute(1, 0, 2)
        way_point = way_point.unsqueeze(0).expand(bs, -1, -1).permute(1, 0, 2)

        way_point = self.way_decoder(
            query=way_point,
            key=latent_query,
            value=latent_query,
            query_pos=wp_pos,
            key_pos=latent_pos,
        )

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        outputs_ego_trajs = outputs_ego_trajs.permute(1, 0, 2).view(
            bs, self.ego_fut_mode, self.fut_ts, self.traj_dims
        )

        return {
            "bev_embed": bev_embed,
            "scene_query": latent_query,
            "token_attn": selected,
            "ego_fut_preds": outputs_ego_trajs,
        }

    # navsim driving_command index order, from
    # navsim/planning/scenario_builder: [left, straight, right, unknown]
    CMD_NAMES = ("left", "straight", "right", "unknown")

    def select_trajectory(
        self, ego_fut_preds: torch.Tensor, cmd: torch.Tensor
    ) -> torch.Tensor:
        """Pick the commanded branch and cumsum offsets into absolute poses."""
        bs = ego_fut_preds.shape[0]
        cmd_idx = cmd.reshape(bs, self.num_navi_cmd).argmax(dim=-1)
        chosen = ego_fut_preds[torch.arange(bs, device=ego_fut_preds.device), cmd_idx]
        return chosen.cumsum(dim=-2)
