"""Parallel detection + motion head (``ParaDetMotionHead``), mmdet-free.

One head produces both supervisions from a single forward over the shared BEV,
which is why detection and motion share a single shared-BEV gradient valve.
Nothing here feeds the planner or the map head -- the only coupling is through
``bev_embed`` itself.

Layout follows the nuScenes config: 300 object queries, a 3-layer deformable
DETR decoder with iterative reference-point refinement, then a 1-layer
agent<->agent decoder producing ``fut_mode`` trajectory hypotheses per agent.
There is deliberately no map cross-attention (PARA-Drive Fig. 4 edge (1)).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .losses import (
    distributed_avg_factor,
    hungarian_assign_det,
    normalize_bbox,
    sigmoid_focal_loss,
    weighted_l1_loss,
)
from .ms_deform_attn import CustomMSDeformableAttention
from .transformer_blocks import (
    BaseTransformerLayer,
    MultiheadAttention,
    build_self_attn_decoder,
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
    """Original SSR classification branch: Linear-LN-ReLU blocks."""
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


def _last_valid_fde(
    pred_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-mode FDE at each agent's last annotated future step.

    Both prediction and target tensors contain per-step offsets, so FDE is
    measured only after accumulating them into positions.  Rows without any
    valid future return zero FDE together with ``has_future=False``; callers
    must exclude those rows from trajectory regression and classification.
    """
    pred_positions = pred_offsets.cumsum(dim=-2)
    target_positions = target_offsets.cumsum(dim=-2)
    distances = torch.linalg.vector_norm(
        pred_positions - target_positions[:, None], dim=-1
    )

    valid = valid_mask.bool()
    steps = torch.arange(valid.size(-1), device=valid.device)
    last_valid = torch.where(valid, steps, steps.new_full((), -1)).amax(dim=-1)
    has_future = last_valid >= 0
    safe_last = last_valid.clamp(min=0)
    fde = distances.gather(
        dim=-1,
        index=safe_last[:, None, None].expand(-1, distances.size(1), 1),
    ).squeeze(-1)
    fde = torch.where(has_future[:, None], fde, torch.zeros_like(fde))
    return fde, has_future


class DetectionTransformerDecoder(nn.Module):
    """Deformable DETR decoder with iterative box refinement."""

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
        intermediate = []
        intermediate_reference_points = []
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
            new_reference_points = torch.zeros_like(reference_points)
            new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
            new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
            reference_points = new_reference_points.sigmoid().detach()

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)
        return torch.stack(intermediate), torch.stack(intermediate_reference_points)


class ParaDetMotionHead(nn.Module):
    def __init__(
        self,
        num_query: int = 300,
        num_classes: int = 7,
        embed_dims: int = 256,
        bev_h: int = 100,
        bev_w: int = 100,
        pc_range: Sequence[float] = (-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
        code_size: int = 10,
        code_weights: Optional[Sequence[float]] = None,
        num_reg_fcs: int = 2,
        fut_ts: int = 8,
        fut_mode: int = 6,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        use_pe: bool = True,
        sync_cls_avg_factor: bool = True,
        loss_weight: float = 1.0,
        loss_cls_weight: float = 2.0,
        loss_bbox_weight: float = 0.25,
        loss_traj_weight: float = 0.2,
        loss_traj_cls_weight: float = 0.2,
        assigner_cls_weight: float = 2.0,
        assigner_reg_weight: float = 0.25,
    ):
        super().__init__()
        self.num_query = num_query
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = list(pc_range)
        self.code_size = code_size
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.use_pe = use_pe
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.loss_weight = loss_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_traj_weight = loss_traj_weight
        self.loss_traj_cls_weight = loss_traj_cls_weight
        self.assigner_cls_weight = assigner_cls_weight
        self.assigner_reg_weight = assigner_reg_weight

        if code_weights is None:
            code_weights = [1.0] * 8 + [0.2, 0.2]
        self.register_buffer(
            "code_weights", torch.tensor(code_weights[:code_size], dtype=torch.float32)
        )

        self.query_embedding = nn.Embedding(num_query, embed_dims * 2)
        self.reference_points = nn.Linear(embed_dims, 3)

        self.decoder = DetectionTransformerDecoder(
            num_decoder_layers, embed_dims, num_heads, feedforward_channels
        )

        cls_branch = _cls_mlp(embed_dims, embed_dims, num_classes, num_reg_fcs)
        reg_branch = _mlp(embed_dims, embed_dims, code_size, num_reg_fcs)
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(cls_branch) for _ in range(num_decoder_layers)]
        )
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(num_decoder_layers)]
        )

        # agent <-> agent interaction only
        self.motion_decoder = build_self_attn_decoder(
            1,
            embed_dims,
            num_heads,
            feedforward_channels,
            ("cross_attn", "norm", "ffn", "norm"),
            attn_dropout=0.1,
            ffn_dropout=0.1,
        )
        self.motion_mode_query = nn.Embedding(fut_mode, embed_dims)
        self.pos_mlp_sa = nn.Linear(2, embed_dims) if use_pe else None
        self.traj_branch = _mlp(
            embed_dims, embed_dims, fut_ts * 2, num_reg_fcs
        )
        self.traj_cls_branch = _cls_mlp(
            embed_dims, embed_dims, 1, num_reg_fcs
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Match the BaseModule init hook used by the original heads.  Xavier on
        # the whole decoder overwrites deformable attention's special zero/radial
        # initialization, so restore that module immediately afterwards.
        for decoder in (self.decoder, self.motion_decoder):
            for parameter in decoder.parameters():
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
        """
        Args:
            bev_embed: ``[bs, bev_h * bev_w, embed_dims]``
        """
        bs = bev_embed.size(0)
        device = bev_embed.device

        query_embed = self.query_embedding.weight.to(bev_embed.dtype)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()

        # ``use_pe`` belongs to the motion box-centre PE below.  The original
        # detection decoder consumes the shared BEV directly.
        value = bev_embed

        spatial_shapes = torch.tensor(
            [[self.bev_h, self.bev_w]], dtype=torch.long, device=device
        )
        level_start_index = torch.tensor([0], dtype=torch.long, device=device)

        inter_states, inter_references = self.decoder(
            query=query.permute(1, 0, 2),
            key=value.permute(1, 0, 2),
            value=value.permute(1, 0, 2),
            query_pos=query_pos.permute(1, 0, 2),
            reference_points=reference_points,
            reg_branches=self.reg_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        outputs_classes, outputs_coords, outputs_coords_bev = [], [], []
        for lvl in range(inter_states.shape[0]):
            hidden = inter_states[lvl].permute(1, 0, 2)  # [bs, num_query, C]
            reference = (
                reference_points if lvl == 0 else inter_references[lvl - 1]
            )
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hidden)
            tmp = self.reg_branches[lvl](hidden)
            tmp = tmp.clone()
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            # Motion PE uses normalized box centres and deliberately detaches
            # them, as in SSR, so trajectory loss does not update bbox centres.
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            # Canonical raw code:
            # [x, y, logW, logL, z, logH, sin(yaw), cos(yaw), vx, vy].
            # Only x/y/z are reference-point-normalised and need conversion to
            # metric coordinates.  Every other channel is already decoder code.
            coord = tmp.clone()
            coord[..., 0:1] = tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1:2] = tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            coord[..., 4:5] = tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outputs_classes.append(outputs_class)
            outputs_coords.append(coord)

        # One token per (agent query, motion mode), q-major/mode-minor.  This is
        # materially different from predicting every mode with one final MLP:
        # the original decoder lets modes interact as separate tokens.
        motion_query = inter_states[-1]  # [Q, B, C]
        mode_query = self.motion_mode_query.weight.to(motion_query.dtype)
        motion_query = (
            motion_query[:, None, :, :] + mode_query[None, :, None, :]
        ).flatten(0, 1)  # [Q*M, B, C]

        if self.pos_mlp_sa is not None:
            motion_pos = self.pos_mlp_sa(outputs_coords_bev[-1])  # [B, Q, C]
            motion_pos = (
                motion_pos.unsqueeze(2)
                .repeat(1, 1, self.fut_mode, 1)
                .flatten(1, 2)
                .permute(1, 0, 2)
            )
        else:
            motion_pos = None

        motion_hs = self.motion_decoder(
            query=motion_query,
            key=motion_query,
            value=motion_query,
            query_pos=motion_pos,
            key_pos=motion_pos,
        )
        motion_hs = motion_hs.permute(1, 0, 2).reshape(
            bs, self.num_query, self.fut_mode, self.embed_dims
        )
        traj = self.traj_branch(motion_hs).reshape(
            bs, self.num_query, self.fut_mode, self.fut_ts, 2
        )
        traj_cls = self.traj_cls_branch(motion_hs).squeeze(-1)

        return {
            "all_cls_scores": torch.stack(outputs_classes),
            "all_bbox_preds": torch.stack(outputs_coords),
            "traj_preds": traj,
            "traj_cls_preds": traj_cls,
        }

    # ------------------------------------------------------------------ #
    # loss
    # ------------------------------------------------------------------ #
    def loss(
        self,
        preds: Dict[str, torch.Tensor],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_valid: torch.Tensor,
        gt_fut_trajs: Optional[torch.Tensor] = None,
        gt_fut_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            gt_bboxes: ``[bs, max_agents, 9]`` ``(x, y, z, w, l, h, yaw, vx, vy)``
            gt_labels: ``[bs, max_agents]`` class index
            gt_valid: ``[bs, max_agents]`` bool
            gt_fut_trajs: ``[bs, max_agents, fut_ts, 2]`` future offsets
            gt_fut_masks: ``[bs, max_agents, fut_ts]``
        """
        all_cls_scores = preds["all_cls_scores"]
        all_bbox_preds = preds["all_bbox_preds"]
        num_layers = all_cls_scores.size(0)
        device = all_cls_scores.device

        losses: Dict[str, torch.Tensor] = {}
        last_assign: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for lvl in range(num_layers):
            cls_scores = torch.nan_to_num(
                all_cls_scores[lvl], nan=0.0, posinf=0.0, neginf=0.0
            )
            bbox_preds = torch.nan_to_num(
                all_bbox_preds[lvl], nan=0.0, posinf=0.0, neginf=0.0
            )
            bs = cls_scores.size(0)

            labels_list, weights_list = [], []
            bbox_targets, bbox_weights = [], []
            num_pos_total = 0
            assigns = []
            for b in range(bs):
                valid = gt_valid[b].bool()
                gt_b = gt_bboxes[b][valid]
                gtl_b = gt_labels[b][valid].long()
                gt_norm = normalize_bbox(gt_b) if gt_b.numel() else gt_b.new_zeros((0, self.code_size))

                q_idx, g_idx = hungarian_assign_det(
                    cls_scores[b],
                    bbox_preds[b],
                    gt_norm,
                    gtl_b,
                    cls_weight=self.assigner_cls_weight,
                    reg_weight=self.assigner_reg_weight,
                )
                assigns.append((q_idx, g_idx))
                num_pos_total += len(q_idx)

                labels = torch.full(
                    (self.num_query,), self.num_classes, dtype=torch.long, device=device
                )
                label_w = torch.ones(self.num_query, device=device)
                bt = bbox_preds.new_zeros((self.num_query, self.code_size))
                bw = bbox_preds.new_zeros((self.num_query, self.code_size))
                if len(q_idx):
                    labels[q_idx] = gtl_b[g_idx]
                    bt[q_idx] = gt_norm[g_idx][:, : self.code_size]
                    bw[q_idx] = 1.0
                labels_list.append(labels)
                weights_list.append(label_w)
                bbox_targets.append(bt)
                bbox_weights.append(bw)

            if lvl == num_layers - 1:
                last_assign = assigns

            labels = torch.cat(labels_list)
            label_weights = torch.cat(weights_list)
            bbox_targets_t = torch.cat(bbox_targets)
            bbox_weights_t = torch.cat(bbox_weights)

            cls_avg_factor = max(float(num_pos_total), 1.0)
            if self.sync_cls_avg_factor:
                cls_avg_factor = distributed_avg_factor(num_pos_total, cls_scores)
            loss_cls = sigmoid_focal_loss(
                cls_scores.reshape(-1, self.num_classes),
                labels,
                label_weights,
                avg_factor=max(cls_avg_factor, 1.0),
            ) * self.loss_cls_weight

            # Decoder predictions already use the canonical regression code.
            # Encoding them again would log size/position channels a second
            # time and move z away from code index 4.
            pred_code_all = bbox_preds.reshape(-1, self.code_size)
            bbox_weights_t = bbox_weights_t * self.code_weights.to(bbox_weights_t.dtype)
            bbox_avg_factor = distributed_avg_factor(num_pos_total, pred_code_all)
            loss_bbox = weighted_l1_loss(
                torch.nan_to_num(pred_code_all),
                torch.nan_to_num(bbox_targets_t),
                bbox_weights_t,
                avg_factor=bbox_avg_factor,
            ) * self.loss_bbox_weight

            suffix = "" if lvl == num_layers - 1 else f"_d{lvl}"
            losses[f"loss_cls{suffix}"] = torch.nan_to_num(loss_cls) * self.loss_weight
            losses[f"loss_bbox{suffix}"] = torch.nan_to_num(loss_bbox) * self.loss_weight

        # ---- motion, on the final layer's matching -----------------------
        # Regression uses matched queries.  Classification follows the original
        # SSR head and uses every B*Q query: unmatched detection queries are the
        # motion background class, while matched queries without any annotated
        # future are the only entries whose classification weight is zero.
        traj_preds = torch.nan_to_num(
            preds["traj_preds"], nan=0.0, posinf=0.0, neginf=0.0
        )
        traj_cls_preds = torch.nan_to_num(
            preds["traj_cls_preds"], nan=0.0, posinf=0.0, neginf=0.0
        )
        loss_traj_sum = traj_preds.sum() * 0.0
        traj_labels = torch.full(
            traj_cls_preds.shape[:2],
            self.fut_mode,
            dtype=torch.long,
            device=device,
        )
        traj_cls_weights = traj_cls_preds.new_ones(traj_cls_preds.shape[:2])
        num_detection_pos = sum(int(q_idx.numel()) for q_idx, _ in last_assign)

        for b, (q_idx, g_idx) in enumerate(last_assign):
            if len(q_idx) == 0:
                continue
            if gt_fut_trajs is None or gt_fut_masks is None:
                # A matched query without future supervision is neither a mode
                # target nor background; exclude it from motion classification.
                traj_cls_weights[b, q_idx] = 0.0
                continue

            # g_idx indexes the compact, valid-GT list used by Hungarian.
            valid_gt = gt_valid[b].bool()
            gt_traj = gt_fut_trajs[b][valid_gt][g_idx]  # [n, fut_ts, 2]
            gt_mask = gt_fut_masks[b][valid_gt][g_idx]  # [n, fut_ts]
            pred = traj_preds[b][q_idx]                 # [n, mode, fut_ts, 2]
            fde, has_future = _last_valid_fde(pred, gt_traj, gt_mask)
            best = fde.argmin(dim=-1)
            rows = torch.arange(best.numel(), device=device)
            best_pred = pred[rows, best]
            weights = gt_mask[..., None].expand_as(best_pred)

            # Zero-future positives contribute a zero numerator but remain in
            # the original detection-positive denominator.
            loss_traj_sum = loss_traj_sum + weighted_l1_loss(
                best_pred, gt_traj, weights, avg_factor=1.0
            )
            traj_labels[b, q_idx[has_future]] = best[has_future]
            traj_cls_weights[b, q_idx[~has_future]] = 0.0

        motion_avg_factor = distributed_avg_factor(num_detection_pos, traj_preds)
        loss_traj_cls_sum = sigmoid_focal_loss(
            traj_cls_preds.reshape(-1, self.fut_mode),
            traj_labels.reshape(-1),
            traj_cls_weights.reshape(-1),
            avg_factor=1.0,
        )
        losses["loss_traj"] = torch.nan_to_num(
            loss_traj_sum / motion_avg_factor
            * self.loss_traj_weight
            * self.loss_weight
        )
        losses["loss_traj_cls"] = torch.nan_to_num(
            loss_traj_cls_sum / motion_avg_factor
            * self.loss_traj_cls_weight
            * self.loss_weight
        )

        return losses
