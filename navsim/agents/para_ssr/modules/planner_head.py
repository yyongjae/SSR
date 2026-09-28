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
from .candidate_planner import load_plan_anchors, poses_to_offsets, commanded_candidates


class PlanTaskMemoryLayer(nn.Module):
    """One planner layer: BEV, then parallel det and map attention, then FFN.

    Both task attentions read the residual left by the BEV attention. Neither
    consumes the other branch's update. The two updates are added together.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        use_task_interaction: bool = True,
    ):
        super().__init__()
        self.use_task_interaction = use_task_interaction
        self.bev_query_norm = nn.LayerNorm(embed_dims)
        self.bev_memory_norm = nn.LayerNorm(embed_dims)
        self.bev_cross_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        if use_task_interaction:
            self.det_query_norm = nn.LayerNorm(embed_dims)
            self.map_query_norm = nn.LayerNorm(embed_dims)
            self.det_memory_norm = nn.LayerNorm(embed_dims)
            self.map_memory_norm = nn.LayerNorm(embed_dims)
            self.plan_det_cross_attn = nn.MultiheadAttention(
                embed_dims, num_heads, batch_first=True
            )
            self.plan_map_cross_attn = nn.MultiheadAttention(
                embed_dims, num_heads, batch_first=True
            )
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_channels, embed_dims),
        )

    def forward(
        self,
        h: torch.Tensor,
        plan_query_pos: torch.Tensor,
        bev: torch.Tensor,
        bev_pos: torch.Tensor,
        det_memory: Optional[torch.Tensor] = None,
        det_position: Optional[torch.Tensor] = None,
        det_confidence: Optional[torch.Tensor] = None,
        map_memory: Optional[torch.Tensor] = None,
        map_position: Optional[torch.Tensor] = None,
        map_confidence: Optional[torch.Tensor] = None,
        return_bev_attn: bool = False,
    ):
        value = self.bev_memory_norm(bev)
        # The default path keeps need_weights=False so the fused kernel, and
        # therefore the v4 training graph, stays as it was. Attention is read
        # only for the train-only BEV selector.
        if return_bev_attn:
            attended, bev_attn = self.bev_cross_attn(
                self.bev_query_norm(h) + plan_query_pos,
                value + bev_pos,
                value,
                need_weights=True,
                average_attn_weights=True,
            )
            h = h + attended
            bev_attn = bev_attn.squeeze(1)
        else:
            h = h + self.bev_cross_attn(
                self.bev_query_norm(h) + plan_query_pos,
                value + bev_pos,
                value,
                need_weights=False,
            )[0]
            bev_attn = None
        if self.use_task_interaction:
            value = self.det_memory_norm(det_memory)
            det_update = self.plan_det_cross_attn(
                self.det_query_norm(h) + plan_query_pos,
                value + det_position + det_confidence,
                value,
                need_weights=False,
            )[0]
            value = self.map_memory_norm(map_memory)
            map_update = self.plan_map_cross_attn(
                self.map_query_norm(h) + plan_query_pos,
                value + map_position + map_confidence,
                value,
                need_weights=False,
            )[0]
            h = h + det_update + map_update
        h = h + self.ffn(self.ffn_norm(h))
        if return_bev_attn:
            return h, bev_attn
        return h


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
        use_metric_planner: bool = False,
        num_plan_candidates: int = 16,
        plan_anchor_path: str = "",
        use_stl: bool = True,
        plan_num_layers: int = 3,
        use_task_interaction: bool = False,
    ):
        super().__init__()
        if use_task_interaction and (use_metric_planner or use_stl):
            raise ValueError("task interaction requires the dense planner (use_stl=False)")
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
        self.use_metric_planner = use_metric_planner
        self.num_plan_candidates = num_plan_candidates if use_metric_planner else 1
        self.use_stl = True if use_metric_planner else use_stl
        self.use_task_interaction = bool(use_task_interaction) and not self.use_stl

        self.transformer = transformer
        self.positional_encoding = LearnedPositionalEncoding(
            embed_dims // 2, bev_h, bev_w
        )

        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.navi_embedding = nn.Embedding(num_navi_cmd, embed_dims)

        def reg_fcs(out_dims: int) -> nn.Sequential:
            layers = []
            for _ in range(num_reg_fcs):
                layers.append(nn.Linear(embed_dims, embed_dims))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(embed_dims, out_dims))
            return nn.Sequential(*layers)

        if self.use_stl:
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
            self.way_point = nn.Embedding(
                ego_fut_mode * self.num_plan_candidates * fut_ts, embed_dims * 2
            )
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
            self.plan_query = nn.Embedding(1, embed_dims)
            self.plan_query_pos = nn.Embedding(1, embed_dims)
            self.plan_fuser = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
            # Physical ego dynamics [vx, vy, ax, ay] conditioning
            self.ego_status_encoder = nn.Sequential(
                nn.Linear(4, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims),
            )
            if self.use_task_interaction:
                self.det_projection = nn.Linear(embed_dims, embed_dims)
                self.motion_projection = nn.Linear(embed_dims, embed_dims)
                self.map_projection = nn.Linear(embed_dims, embed_dims)
                self.det_memory_norm = nn.LayerNorm(embed_dims)
                self.map_memory_norm = nn.LayerNorm(embed_dims)

                def metadata_encoder(input_dims: int) -> nn.Sequential:
                    return nn.Sequential(
                        nn.Linear(input_dims, embed_dims),
                        nn.ReLU(inplace=True),
                        nn.Linear(embed_dims, embed_dims),
                    )

                self.det_position_encoder = metadata_encoder(2)
                self.map_position_encoder = metadata_encoder(2)
                self.det_confidence_encoder = metadata_encoder(1)
                self.map_confidence_encoder = metadata_encoder(1)
                self.planner_layers = nn.ModuleList(
                    PlanTaskMemoryLayer(
                        embed_dims, num_heads, feedforward_channels, True
                    )
                    for _ in range(plan_num_layers)
                )
                self.final_norm = nn.LayerNorm(embed_dims)
                decoders = tuple(self.planner_layers)
            else:
                self.plan_decoder = build_self_attn_decoder(
                    plan_num_layers,
                    embed_dims,
                    num_heads,
                    feedforward_channels,
                    ("cross_attn", "norm", "ffn", "norm"),
                    attn_dropout=0.0,
                    ffn_dropout=0.0,
                )
                decoders = (self.plan_decoder,)
            self.ego_fut_decoder = reg_fcs(fut_ts * traj_dims)

        # Xavier initialization matching BaseModule convention
        for decoder in decoders:
            for parameter in decoder.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

        if self.use_metric_planner:
            anchors = (load_plan_anchors(plan_anchor_path, num_plan_candidates, fut_ts)
                       if plan_anchor_path else torch.zeros(num_plan_candidates, fut_ts, 3))
            self.register_buffer("plan_anchors", anchors)
            self.register_buffer("anchors_ready", torch.tensor(bool(plan_anchor_path)))
            self.candidate_cls = nn.Linear(embed_dims, 1)
            # Start at physically valid, distinct train-derived trajectories.
            nn.init.zeros_(self.ego_fut_decoder[-1].weight)
            nn.init.zeros_(self.ego_fut_decoder[-1].bias)

    def forward(
        self,
        mlvl_feats: Sequence[torch.Tensor],
        lidar2img: torch.Tensor,
        image_hw: torch.Tensor,
        ego_motion: torch.Tensor,
        bev_shift: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
        bev_yaw: Optional[torch.Tensor] = None,
        only_bev: bool = False,
        cmd: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Args:
            mlvl_feats: list of ``[bs, num_cam, C, H, W]``
            cmd: one-hot navigation command ``[bs, num_navi_cmd]``
            ego_status: ``[bs, 4]`` [vx, vy, ax, ay] kinematics
        Returns:
            ``only_bev``: the BEV feature. Otherwise a dict with ``bev_embed``
            ``[bs, bev_h*bev_w, C]``, ``scene_query``, ``token_attn`` and
            ``ego_fut_preds`` ``[bs, ego_fut_mode, fut_ts, traj_dims]``.
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device, dtype=dtype
        )
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
            bev_yaw=bev_yaw,
        )
        if only_bev:
            return bev_embed
        return self.forward_from_bev(bev_embed, cmd, bev_pos=bev_pos, ego_status=ego_status)

    def prepare_task_memories(
        self,
        det_out: Dict[str, torch.Tensor],
        map_out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pool decoder latents once. Box and score metadata are detached.

        Detection XY is metric and is normalised by the BEV range. Map points
        are already in ``[0, 1]``. Confidence is the max sigmoid class score.
        """
        det_hidden = det_out["det_hidden"]
        motion_hidden = det_out["motion_hidden"]
        map_point_hidden = map_out["map_point_hidden"]
        if det_hidden.ndim != 3 or motion_hidden.ndim != 4 or map_point_hidden.ndim != 4:
            raise ValueError("expected det [B,Q,C], motion [B,Q,M,C], map [B,V,P,C] latents")
        det_memory = self.det_memory_norm(
            self.det_projection(det_hidden) + self.motion_projection(motion_hidden.mean(dim=2))
        )
        map_memory = self.map_memory_norm(self.map_projection(map_point_hidden.mean(dim=2)))
        det_xy = det_out["all_bbox_preds"][-1, ..., :2].detach()
        xy_origin = det_xy.new_tensor(self.pc_range[:2])
        xy_extent = det_xy.new_tensor((self.real_w, self.real_h))
        det_xy = (det_xy - xy_origin) / xy_extent
        map_xy = map_out["all_map_pts_preds"][-1].detach()
        det_score = det_out["all_cls_scores"][-1].detach().sigmoid().amax(dim=-1, keepdim=True)
        map_score = map_out["all_map_cls_scores"][-1].detach().sigmoid().amax(dim=-1, keepdim=True)
        return {
            "det_memory": det_memory,
            "det_position": self.det_position_encoder(det_xy),
            "det_confidence": self.det_confidence_encoder(det_score),
            "map_memory": map_memory,
            "map_position": self.map_position_encoder(map_xy).mean(dim=2),
            "map_confidence": self.map_confidence_encoder(map_score),
        }

    def forward_from_bev(
        self,
        bev_embed: torch.Tensor,
        cmd: Optional[torch.Tensor],
        bev_pos: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
        det_out: Optional[Dict[str, torch.Tensor]] = None,
        map_out: Optional[Dict[str, torch.Tensor]] = None,
        return_bev_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the planning decoder on an existing BEV feature.

        Split out of :meth:`forward` so stage-1 planning distillation can drive
        the *unchanged* decoder from a cached teacher BEV instead of one built
        from images. ``forward`` passes the ``bev_pos`` it already computed;
        when called directly it is recomputed, which is deterministic for a
        given batch size and grid.
        """
        bs = bev_embed.size(0)
        dtype = bev_embed.dtype
        expected = self.bev_h * self.bev_w
        if bev_embed.size(1) != expected:
            raise ValueError(
                f"BEV has {bev_embed.size(1)} tokens, planner grid is "
                f"{self.bev_h}x{self.bev_w}={expected}"
            )
        if bev_pos is None:
            bev_mask = torch.zeros(
                (bs, self.bev_h, self.bev_w), device=bev_embed.device, dtype=dtype
            )
            bev_pos = self.positional_encoding(bev_mask).to(dtype)

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
            navi = self.navi_embedding(cmd_idx).unsqueeze(1)  # [B, 1, C]
            query = self.plan_query.weight.to(dtype).unsqueeze(0).expand(bs, -1, -1)
            h = self.plan_fuser(torch.cat((query, navi), dim=-1))
            if ego_status is None:
                status = bev_embed.new_zeros((bs, 4))
            else:
                status = ego_status.to(dtype=dtype, device=bev_embed.device)
                if status.shape != (bs, 4):
                    raise ValueError(
                        f"expected ego_status [B, 4] (vx, vy, ax, ay), got {tuple(status.shape)}"
                    )
            h = h + self.ego_status_encoder(status).unsqueeze(1)
            plan_pos = self.plan_query_pos.weight.to(dtype).unsqueeze(0).expand(bs, -1, -1)
            plan_attn = None
            if self.use_task_interaction:
                if det_out is None or map_out is None:
                    raise ValueError("task interaction planning requires det/motion and map outputs")
                memories = self.prepare_task_memories(det_out, map_out)
                last = len(self.planner_layers) - 1
                for index, layer in enumerate(self.planner_layers):
                    if return_bev_attn and index == last:
                        h, plan_attn = layer(
                            h, plan_pos, bev_embed, pos_embd,
                            return_bev_attn=True, **memories,
                        )
                    else:
                        h = layer(h, plan_pos, bev_embed, pos_embd, **memories)
                h = self.final_norm(h)
                scene_query = h.transpose(0, 1)
            else:
                plan_query = self.plan_decoder(
                    query=h.transpose(0, 1),  # [1, B, C]
                    key=bev_embed.permute(1, 0, 2),  # [HW, B, C]
                    value=bev_embed.permute(1, 0, 2),
                    query_pos=plan_pos.transpose(0, 1),
                    key_pos=pos_embd.permute(1, 0, 2),
                )
                h = plan_query.transpose(0, 1)
                scene_query = plan_query
            plan = self.ego_fut_decoder(h[:, 0]).view(
                bs, 1, self.fut_ts, self.traj_dims
            )
            planned = {
                "bev_embed": bev_embed,
                "scene_query": scene_query,
                "token_attn": None,
                "ego_fut_preds": plan.expand(bs, self.ego_fut_mode, self.fut_ts, self.traj_dims),
            }
            if plan_attn is not None:
                planned["plan_bev_attn"] = plan_attn
            return planned

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
        if self.use_metric_planner:
            if not self.anchors_ready.item():
                raise RuntimeError(
                    "metric planner needs a train-only plan_anchor_path or an initialized checkpoint"
                )
            residuals = outputs_ego_trajs.permute(1, 0, 2).reshape(
                bs, self.ego_fut_mode, self.num_plan_candidates, self.fut_ts, self.traj_dims
            )
            candidates = residuals + poses_to_offsets(self.plan_anchors)[None, None]
            candidate_features = way_point.permute(1, 0, 2).reshape(
                bs, self.ego_fut_mode, self.num_plan_candidates, self.fut_ts, self.embed_dims
            ).mean(dim=-2)
            logits = self.candidate_cls(candidate_features).squeeze(-1)
            return {
                "bev_embed": bev_embed,
                "bev_pos": pos_embd,
                "scene_query": latent_query,
                "token_attn": selected,
                "candidate_offsets": candidates,
                "candidate_logits": commanded_candidates(logits, cmd),
                "plan_anchors": self.plan_anchors,
            }
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
