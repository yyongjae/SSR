"""Parallel vectorised map head (``ParaMapHead``), mmdet-free.

VAD's formulation, unchanged in shape: ``map_num_vec`` polyline instances of
``map_num_pts_per_vec`` points each, Hungarian-matched against ground-truth
polylines and regressed as point sets.

The query set is the outer product of an instance embedding and a point
embedding, so instance *i* point *j* is query ``i * num_pts + j``.  Classification
is per instance (read off the instance-mean of its point features); regression
is per point.

Nothing produced here reaches the planner or the motion decoder -- PARA-Drive
Fig. 4 edges (1) and (5), both removed.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .losses import (
    distributed_avg_factor,
    hungarian_assign_map,
    pts_dir_cos_loss,
    pts_l1_loss,
    sigmoid_focal_loss,
)
from .ms_deform_attn import CustomMSDeformableAttention
from .transformer_blocks import (
    BaseTransformerLayer,
    MultiheadAttention,
    inverse_sigmoid,
)


def _mlp(in_dim: int, hidden: int, out_dim: int, num_fcs: int = 2) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(num_fcs):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU(inplace=True))
        in_dim = hidden
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _cls_mlp(
    in_dim: int, hidden: int, out_dim: int, num_fcs: int = 2
) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(num_fcs):
        layers.extend(
            [
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
            ]
        )
        in_dim = hidden
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class MapDetectionTransformerDecoder(nn.Module):
    """Deformable decoder over the BEV, refining 2D point references."""

    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        num_levels: int = 1,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            attentions = [
                MultiheadAttention(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    proj_drop=0.0,
                    dropout_layer=attn_dropout,
                ),
                CustomMSDeformableAttention(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    dropout=attn_dropout,
                    batch_first=False,
                ),
            ]
            layers.append(
                BaseTransformerLayer(
                    attentions=attentions,
                    embed_dims=embed_dims,
                    feedforward_channels=feedforward_channels,
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    ffn_dropout=ffn_dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        reg_branches: nn.ModuleList,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = query
        intermediate, intermediate_refs = [], []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            # ``output`` is sequence-first; the branches and the reference
            # points are batch-first.
            tmp = reg_branches[lid](output.permute(1, 0, 2))
            new_reference_points = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
            reference_points = new_reference_points.sigmoid().detach()

            intermediate.append(output)
            intermediate_refs.append(reference_points)
        return torch.stack(intermediate), torch.stack(intermediate_refs)


class ParaMapHead(nn.Module):
    def __init__(
        self,
        map_num_vec: int = 100,
        map_num_pts_per_vec: int = 20,
        map_num_classes: int = 3,
        embed_dims: int = 256,
        bev_h: int = 100,
        bev_w: int = 100,
        pc_range: Sequence[float] = (-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
        num_reg_fcs: int = 2,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        map_dir_interval: int = 1,
        sync_cls_avg_factor: bool = True,
        loss_weight: float = 1.0,
        loss_map_cls_weight: float = 2.0,
        loss_map_pts_weight: float = 1.0,
        loss_map_dir_weight: float = 0.005,
        assigner_cls_weight: float = 2.0,
        assigner_pts_weight: float = 1.0,
    ):
        super().__init__()
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_classes = map_num_classes
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = list(pc_range)
        self.map_dir_interval = map_dir_interval
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.loss_weight = loss_weight
        self.loss_map_cls_weight = loss_map_cls_weight
        self.loss_map_pts_weight = loss_map_pts_weight
        self.loss_map_dir_weight = loss_map_dir_weight
        self.assigner_cls_weight = assigner_cls_weight
        self.assigner_pts_weight = assigner_pts_weight

        self.instance_embedding = nn.Embedding(map_num_vec, embed_dims * 2)
        self.pts_embedding = nn.Embedding(map_num_pts_per_vec, embed_dims * 2)
        self.reference_points = nn.Linear(embed_dims, 2)

        self.decoder = MapDetectionTransformerDecoder(
            num_decoder_layers, embed_dims, num_heads, feedforward_channels
        )

        cls_branch = _cls_mlp(embed_dims, embed_dims, map_num_classes, num_reg_fcs)
        reg_branch = _mlp(embed_dims, embed_dims, 2, num_reg_fcs)
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(cls_branch) for _ in range(num_decoder_layers)]
        )
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(num_decoder_layers)]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for parameter in self.decoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        for module in self.decoder.modules():
            if isinstance(module, CustomMSDeformableAttention):
                module.init_weights()
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.0)
        bias_init = float(-torch.log(torch.tensor((1 - 0.01) / 0.01)))
        for branch in self.cls_branches:
            nn.init.constant_(branch[-1].bias, bias_init)

    def forward(self, bev_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs = bev_embed.size(0)
        device = bev_embed.device
        num_q = self.map_num_vec * self.map_num_pts_per_vec

        pts_embeds = self.pts_embedding.weight.unsqueeze(0)            # [1, P, 2C]
        instance_embeds = self.instance_embedding.weight.unsqueeze(1)  # [V, 1, 2C]
        query_embed = (pts_embeds + instance_embeds).flatten(0, 1)     # [V*P, 2C]
        query_embed = query_embed.to(bev_embed.dtype)

        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()

        value = bev_embed
        spatial_shapes = torch.tensor(
            [[self.bev_h, self.bev_w]], dtype=torch.long, device=device
        )
        level_start_index = torch.tensor([0], dtype=torch.long, device=device)

        inter_states, inter_refs = self.decoder(
            query=query.permute(1, 0, 2),
            key=value.permute(1, 0, 2),
            value=value.permute(1, 0, 2),
            query_pos=query_pos.permute(1, 0, 2),
            reference_points=reference_points,
            reg_branches=self.reg_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        all_cls, all_pts = [], []
        for lvl in range(inter_states.shape[0]):
            hidden = inter_states[lvl].permute(1, 0, 2)  # [bs, V*P, C]
            reference = reference_points if lvl == 0 else inter_refs[lvl - 1]
            reference = inverse_sigmoid(reference)

            hidden_inst = hidden.view(
                bs, self.map_num_vec, self.map_num_pts_per_vec, self.embed_dims
            )
            cls = self.cls_branches[lvl](hidden_inst.mean(dim=2))  # [bs, V, num_cls]

            tmp = self.reg_branches[lvl](hidden)
            pts = (tmp[..., :2] + reference[..., :2]).sigmoid()
            pts = pts.view(bs, self.map_num_vec, self.map_num_pts_per_vec, 2)

            all_cls.append(cls)
            all_pts.append(pts)

        return {
            "all_map_cls_scores": torch.stack(all_cls),
            "all_map_pts_preds": torch.stack(all_pts),
        }

    # ------------------------------------------------------------------ #
    def loss(
        self,
        preds: Dict[str, torch.Tensor],
        gt_map_pts: torch.Tensor,
        gt_map_labels: torch.Tensor,
        gt_map_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            gt_map_pts: ``[bs, max_vec, num_orders, num_pts, 2]`` normalised to
                ``[0, 1]`` over the BEV extent, all equivalent point orderings
            gt_map_labels: ``[bs, max_vec]``
            gt_map_valid: ``[bs, max_vec]`` bool
        """
        all_cls = preds["all_map_cls_scores"]
        all_pts = preds["all_map_pts_preds"]
        num_layers = all_cls.size(0)
        device = all_cls.device
        losses: Dict[str, torch.Tensor] = {}

        for lvl in range(num_layers):
            cls_scores = torch.nan_to_num(
                all_cls[lvl], nan=0.0, posinf=0.0, neginf=0.0
            )
            pts_preds = torch.nan_to_num(
                all_pts[lvl], nan=0.0, posinf=0.0, neginf=0.0
            )
            bs = cls_scores.size(0)

            labels_all, pts_target_all, pts_weight_all = [], [], []
            num_pos_total = 0
            for b in range(bs):
                valid = gt_map_valid[b].bool()
                gt_pts_b = gt_map_pts[b][valid]
                gt_lab_b = gt_map_labels[b][valid].long()

                q_idx, g_idx, o_idx = hungarian_assign_map(
                    cls_scores[b],
                    pts_preds[b],
                    gt_pts_b,
                    gt_lab_b,
                    cls_weight=self.assigner_cls_weight,
                    pts_weight=self.assigner_pts_weight,
                )
                num_pos_total += len(q_idx)

                labels = torch.full(
                    (self.map_num_vec,), self.map_num_classes, dtype=torch.long, device=device
                )
                pt = pts_preds.new_zeros(
                    (self.map_num_vec, self.map_num_pts_per_vec, 2)
                )
                pw = pts_preds.new_zeros(
                    (self.map_num_vec, self.map_num_pts_per_vec, 2)
                )
                if len(q_idx):
                    labels[q_idx] = gt_lab_b[g_idx]
                    pt[q_idx] = gt_pts_b[g_idx, o_idx]
                    pw[q_idx] = 1.0
                labels_all.append(labels)
                pts_target_all.append(pt)
                pts_weight_all.append(pw)

            labels = torch.cat(labels_all)
            pts_target = torch.cat(pts_target_all)
            pts_weight = torch.cat(pts_weight_all)

            cls_avg_factor = max(float(num_pos_total), 1.0)
            if self.sync_cls_avg_factor:
                cls_avg_factor = distributed_avg_factor(num_pos_total, cls_scores)
            loss_cls = sigmoid_focal_loss(
                cls_scores.reshape(-1, self.map_num_classes),
                labels,
                avg_factor=max(cls_avg_factor, 1.0),
            ) * self.loss_map_cls_weight

            pts_pred_flat = pts_preds.reshape(-1, self.map_num_pts_per_vec, 2)
            num_pos_factor = distributed_avg_factor(num_pos_total, pts_pred_flat)
            loss_pts = pts_l1_loss(
                pts_pred_flat,
                pts_target,
                pts_weight,
                # VAD's PtsL1Loss sums every point coordinate and divides by
                # the positive-instance count.
                avg_factor=num_pos_factor,
            ) * self.loss_map_pts_weight

            d = self.map_dir_interval
            # Cosine direction must be computed in metric BEV space.  The x/y
            # extents are anisotropic (30 m vs 60 m by default), so directions
            # in [0, 1] coordinates have different angles.
            metric_scale = pts_pred_flat.new_tensor(
                [
                    self.pc_range[3] - self.pc_range[0],
                    self.pc_range[4] - self.pc_range[1],
                ]
            )
            pred_dir = (
                pts_pred_flat[:, d:] - pts_pred_flat[:, :-d]
            ) * metric_scale
            tgt_dir = (pts_target[:, d:] - pts_target[:, :-d]) * metric_scale
            dir_weight = pts_weight[:, :-d, 0]
            loss_dir = pts_dir_cos_loss(
                pred_dir,
                tgt_dir,
                dir_weight,
                # As in the original PtsDirCosLoss, sum directions and divide
                # by the positive-instance count.
                avg_factor=num_pos_factor,
            ) * self.loss_map_dir_weight

            suffix = "" if lvl == num_layers - 1 else f"_d{lvl}"
            losses[f"loss_map_cls{suffix}"] = torch.nan_to_num(loss_cls) * self.loss_weight
            losses[f"loss_map_pts{suffix}"] = torch.nan_to_num(loss_pts) * self.loss_weight
            losses[f"loss_map_dir{suffix}"] = torch.nan_to_num(loss_dir) * self.loss_weight

        return losses
