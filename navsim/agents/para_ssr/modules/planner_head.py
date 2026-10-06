"""Command/ego-conditioned planner over BEV and private decoder memories.

Three independent Pre-LN layers update the single planning query with
BEV -> parallel detection/motion and map attention -> residual sum -> FFN.
Both task attentions read the same post-BEV hidden. Object and map memories are built once
from the final decoder latents; only the planning query is updated. Prediction
coordinates and foreground confidence condition attention keys through detached
metadata, while content values preserve gradients into both private decoders.
With task interaction disabled, the same layers use only BEV -> FFN and the
planner neither constructs interaction parameters nor consumes task memories.
All modes inject current ego velocity/acceleration into the planning query and
normalize the final residual stream before trajectory regression.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .anchor_planner import AnchorPlanner
from .transformer_blocks import LearnedPositionalEncoding


class PlanTaskMemoryLayer(nn.Module):
    """One residual planner layer, with separate query/memory Pre-LNs."""

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
        if use_task_interaction:
            self.det_query_norm = nn.LayerNorm(embed_dims)
            self.map_query_norm = nn.LayerNorm(embed_dims)
        self.bev_memory_norm = nn.LayerNorm(embed_dims)
        if use_task_interaction:
            self.det_memory_norm = nn.LayerNorm(embed_dims)
            self.map_memory_norm = nn.LayerNorm(embed_dims)
        self.bev_cross_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        if use_task_interaction:
            self.plan_det_cross_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
            self.plan_map_cross_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
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
    ) -> torch.Tensor:
        value = self.bev_memory_norm(bev)
        h = h + self.bev_cross_attn(
            self.bev_query_norm(h) + plan_query_pos,
            value + bev_pos,
            value,
            need_weights=False,
        )[0]
        if self.use_task_interaction:
            # Fork from the same post-BEV residual; neither task attention
            # consumes the other branch's update within this layer.
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
        return h + self.ffn(self.ffn_norm(h))


class ParaSSRPlannerHead(nn.Module):
    """BEV encoder wrapper and three-layer planner using all task queries.

    ``forward(..., only_bev=True)`` encodes current/history frames. The model
    runs the two private heads once on the current BEV and then calls
    ``plan_from_bev`` with their predictions and final hidden states when
    ``use_task_interaction=True``. Command and current ego status condition
    planning in all modes; disabling interaction removes the task memories.
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
        use_stl: bool = False,
        plan_num_layers: int = 3,
        det_embed_dims: Optional[int] = None,
        motion_embed_dims: Optional[int] = None,
        map_embed_dims: Optional[int] = None,
        use_task_interaction: bool = True,
        plan_anchor_file: Optional[str] = None,
        plan_reward_weights: Sequence[float] = (0.1, 0.5, 0.5, 1.0),
        plan_topk: int = 6,
        plan_kinematic: bool = False,
        plan_heading_from_xy: bool = False,
        plan_rescore_refined: bool = False,
    ):
        super().__init__()
        if use_stl:
            raise ValueError("PARA-SSR task-memory planner requires use_stl=False")
        if plan_num_layers != 3:
            raise ValueError("PARA-SSR task-memory planner requires plan_num_layers=3")
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.pc_range = list(pc_range)
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_scenes = num_scenes  # retained constructor/config compatibility
        self.num_reg_fcs = num_reg_fcs
        self.fut_ts = fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_navi_cmd = num_navi_cmd
        self.traj_dims = traj_dims
        self.use_stl = False
        self.use_task_interaction = use_task_interaction

        self.transformer = transformer
        self.positional_encoding = LearnedPositionalEncoding(embed_dims // 2, bev_h, bev_w)
        self.bev_embedding = None if use_lidar else nn.Embedding(bev_h * bev_w, embed_dims)
        self.navi_embedding = nn.Embedding(num_navi_cmd, embed_dims)
        self.plan_query = nn.Embedding(1, embed_dims)
        self.plan_query_pos = nn.Embedding(1, embed_dims)
        self.plan_fuser = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        # Native NAVSIM [vx, vy, ax, ay] retains its physical magnitude. This
        # private planning input is separate from BEV temporal alignment.
        self.ego_status_encoder = nn.Sequential(
            nn.Linear(4, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        # The head dimensions come from configuration; no query/mode/point
        # count is encoded in the projections or attention parameter shapes.
        if use_task_interaction:
            self.det_projection = nn.Linear(det_embed_dims or embed_dims, embed_dims)
            self.motion_projection = nn.Linear(motion_embed_dims or embed_dims, embed_dims)
            self.map_projection = nn.Linear(map_embed_dims or embed_dims, embed_dims)
            self.det_memory_norm = nn.LayerNorm(embed_dims)
            self.map_memory_norm = nn.LayerNorm(embed_dims)

        def metadata_encoder(input_dims: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims),
            )

        if use_task_interaction:
            self.det_position_encoder = metadata_encoder(2)
            self.map_position_encoder = metadata_encoder(2)
            self.det_confidence_encoder = metadata_encoder(1)
            self.map_confidence_encoder = metadata_encoder(1)
        self.planner_layers = nn.ModuleList(
            PlanTaskMemoryLayer(embed_dims, num_heads, feedforward_channels, use_task_interaction)
            for _ in range(plan_num_layers)
        )
        # SafeDrive/WoTE regress from normalized decoder output (Post-LN).
        # Our Pre-LN stack needs an explicit output norm for that same role.
        self.final_norm = nn.LayerNorm(embed_dims)
        reg_layers = []
        for _ in range(num_reg_fcs):
            reg_layers.extend([nn.Linear(embed_dims, embed_dims), nn.ReLU()])
        reg_layers.append(nn.Linear(embed_dims, fut_ts * traj_dims))
        # v2: WoTE-style anchor vocabulary + PDM-score rewards (anchor_planner.py)
        # replaces the single-trajectory regressor, which is then not built: an
        # unused module would fail DDP's unused-parameter check.
        self.anchor_planner = (
            AnchorPlanner(plan_anchor_file, embed_dims, fut_ts, traj_dims, plan_reward_weights, plan_topk,
                          kinematic=plan_kinematic, heading_from_xy=plan_heading_from_xy)
            if plan_anchor_file else None
        )
        self.ego_fut_decoder = nn.Sequential(*reg_layers) if self.anchor_planner is None else None
        # Evaluation-only diagnostic (never used in training): score the refined
        # trajectories (anchor + offset) instead of the fixed anchors, as WoTE does at
        # test time -- a second pass of the planner layers on re-encoded trajectories.
        self.plan_rescore_refined = bool(plan_rescore_refined)

        for layer in self.planner_layers:
            for parameter in layer.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        mlvl_feats: Sequence[torch.Tensor],
        lidar2img: torch.Tensor,
        image_hw: torch.Tensor,
        ego_motion: Optional[torch.Tensor],
        bev_shift: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
        only_bev: bool = False,
        cmd: Optional[torch.Tensor] = None,
        lidar_bev: Optional[torch.Tensor] = None,
        det_out: Optional[Dict[str, torch.Tensor]] = None,
        map_out: Optional[Dict[str, torch.Tensor]] = None,
        ego_status: Optional[torch.Tensor] = None,
        bev_yaw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """Encode BEV; interaction-enabled planning also needs both heads."""
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
            bev_yaw=bev_yaw,
            prev_bev=prev_bev,
            lidar_bev=lidar_bev,
        )
        if only_bev:
            return bev_embed
        return self.plan_from_bev(
            bev_embed, cmd, det_out, map_out, bev_pos=bev_pos, ego_status=ego_status,
        )

    def prepare_task_memories(
        self,
        det_out: Dict[str, torch.Tensor],
        map_out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pool latent content once and encode detached prediction metadata.

        Detection box XY is metric, so normalize it once using the BEV range.
        Map points are already in [0, 1] and enter their coordinate MLP as-is.
        Both heads use sigmoid focal classification without a background logit;
        the maximum sigmoid score is their foreground confidence.
        """
        if not self.use_task_interaction:
            raise ValueError("task memories are unavailable when use_task_interaction=False")
        det_hidden = det_out["det_hidden"]
        motion_hidden = det_out["motion_hidden"]
        map_point_hidden = map_out["map_point_hidden"]
        if det_hidden.ndim != 3 or motion_hidden.ndim != 4 or map_point_hidden.ndim != 4:
            raise ValueError("expected batch-first det [B,Q,C], motion [B,Q,M,C], map [B,V,P,C] latents")
        if det_hidden.shape[:2] != motion_hidden.shape[:2]:
            raise ValueError("detection and motion hidden states must share batch and object dimensions")
        if det_hidden.shape[0] != map_point_hidden.shape[0]:
            raise ValueError("detection and map hidden states must share the batch dimension")
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

    def plan_from_bev(
        self,
        bev_embed: torch.Tensor,
        cmd: Optional[torch.Tensor],
        det_out: Optional[Dict[str, torch.Tensor]] = None,
        map_out: Optional[Dict[str, torch.Tensor]] = None,
        *,
        bev_pos: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Plan from BEV and current [vx, vy, ax, ay] ego status.

        Task head outputs are consumed only when interaction is enabled. Ego
        status is required in every planning mode and enters the initial query
        once, before the first attention layer.
        """
        if self.use_task_interaction and (det_out is None or map_out is None):
            raise ValueError("planning requires det/motion and map outputs with return_hidden=True")
        bs, _, _ = bev_embed.shape
        if cmd is None:
            raise ValueError("cmd is required for planning")
        if cmd.size(0) != bs or cmd.numel() != bs * self.num_navi_cmd:
            raise ValueError(
                f"expected one {self.num_navi_cmd}-way command per sample, "
                f"but got cmd shape {tuple(cmd.shape)} for batch size {bs}"
            )
        if ego_status is None:
            raise ValueError("ego_status [B,4] containing vx, vy, ax, ay is required for planning")
        if ego_status.shape != (bs, 4):
            raise ValueError(
                f"expected ego_status [B,4] containing vx, vy, ax, ay, "
                f"but got shape {tuple(ego_status.shape)} for batch size {bs}"
            )
        if bev_pos is None:
            mask = bev_embed.new_zeros((bs, self.bev_h, self.bev_w))
            bev_pos = self.positional_encoding(mask).to(bev_embed.dtype)
        bev_pos = bev_pos.flatten(2).transpose(1, 2)
        cmd_idx = cmd.reshape(bs, self.num_navi_cmd).argmax(dim=-1)
        navi = self.navi_embedding(cmd_idx).unsqueeze(1)
        query = self.plan_query.weight.to(bev_embed.dtype).unsqueeze(0).expand(bs, -1, -1)
        h = self.plan_fuser(torch.cat((query, navi), dim=-1))
        ego_status = ego_status.to(device=h.device, dtype=h.dtype)
        h = h + self.ego_status_encoder(ego_status).unsqueeze(1)
        plan_pos = self.plan_query_pos.weight.to(bev_embed.dtype).unsqueeze(0).expand(bs, -1, -1)
        memories = self.prepare_task_memories(det_out, map_out) if self.use_task_interaction else {}
        h_ego = h
        if self.anchor_planner is not None:
            h = self.anchor_planner.queries(h)                      # [B, K, C]
            plan_pos = plan_pos.expand(-1, h.shape[1], -1)
        for layer in self.planner_layers:
            h = layer(h, plan_pos, bev_embed, bev_pos, **memories)
        h = self.final_norm(h)
        if self.anchor_planner is not None:
            # NAVSIM ego status is (vx, vy, ax, ay) in the ego frame: vx is the speed along heading
            out = self.anchor_planner(h, init_speed=ego_status[:, 0])
            if self.plan_rescore_refined and not self.training:
                out.update(self._rescore_refined(out, h_ego, plan_pos, bev_embed, bev_pos, memories))
            poses = out["trajectory"]                               # absolute poses [B, T, 3]
            steps = torch.diff(torch.cat([poses.new_zeros(bs, 1, self.traj_dims), poses], dim=1), dim=1)
            out.update(
                bev_embed=bev_embed, scene_query=h.transpose(0, 1), token_attn=None,
                # per-step offsets of the selected trajectory, so every consumer of the
                # v1 API (select_trajectory, validation metrics, plan_map) keeps working
                ego_fut_preds=steps.unsqueeze(1).expand(bs, self.ego_fut_mode, self.fut_ts, self.traj_dims),
            )
            return out
        plan = self.ego_fut_decoder(h[:, 0]).view(bs, 1, self.fut_ts, self.traj_dims)
        # Keep the existing command-branch loss and evaluation API. The single
        # trajectory has already been conditioned on the command embedding.
        return {
            "bev_embed": bev_embed,
            "scene_query": h.transpose(0, 1),
            "token_attn": None,
            "ego_fut_preds": plan.expand(bs, self.ego_fut_mode, self.fut_ts, self.traj_dims),
        }

    def _rescore_refined(self, out, h_ego, plan_pos, bev_embed, bev_pos, memories):
        """WoTE-style test-time selection: re-encode anchor + offset, rescore, pick among the refined."""
        ap = self.anchor_planner
        refined = (ap.trajectory_anchors.unsqueeze(0).to(out["trajectory_offset"].dtype)
                   + out["trajectory_offset"])                                  # [B, K, T, 3]
        h2 = ap.queries(h_ego, trajectories=refined)
        for layer in self.planner_layers:
            h2 = layer(h2, plan_pos, bev_embed, bev_pos, **memories)
        _, sim2, final2 = ap.score(self.final_norm(h2))
        k = min(ap.topk, ap.num_anchors)
        top = final2.topk(k, dim=-1).indices
        gather = top[:, :, None, None].expand(-1, -1, ap.fut_ts, ap.traj_dims)
        top_traj = refined.gather(1, gather)
        return {"trajectory": top_traj[:, 0], "plan_rescore_final_rewards": final2,
                "plan_rescore_sim_rewards": sim2, "plan_rescore_topk_index": top,
                "plan_rescore_topk_trajectory": top_traj}

    CMD_NAMES = ("left", "straight", "right", "unknown")

    def select_trajectory(self, ego_fut_preds: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        """Pick the commanded branch and cumsum offsets into absolute poses."""
        bs = ego_fut_preds.shape[0]
        cmd_idx = cmd.reshape(bs, self.num_navi_cmd).argmax(dim=-1)
        chosen = ego_fut_preds[torch.arange(bs, device=ego_fut_preds.device), cmd_idx]
        return chosen.cumsum(dim=-2)
