"""BEVFormer encoder for PARA-SSR on navsim, without mmcv.

Structurally identical to the nuScenes ``SSRPerceptionTransformer`` /
``BEVFormerEncoder``: N surround images -> per-camera deformable spatial
cross-attention into a BEV grid, with temporal self-attention against the
previous frame's BEV.

Two things change because navsim is not nuScenes:

1. **No CAN bus.**  The nuScenes path packs ego motion into an 18-vector and
   re-derives the BEV shift from ``can_bus[0], can_bus[1], can_bus[-2]``.
   navsim gives history ego poses already expressed in the *current* ego frame,
   so the shift is computed directly in the feature builder and arrives as an
   explicit ``bev_shift`` tensor.  An ``ego_motion`` vector still feeds the
   ``can_bus_mlp`` so the query-conditioning path is preserved.

2. **No ``img_metas`` dicts.**  ``lidar2img`` and image shapes are ordinary
   batched tensors, built once in the feature builder.

Coordinate frames
-----------------
SSR's ``point_cloud_range = [-15, -30, -2, 15, 30, 2]`` is VAD's ego frame:
**x is lateral (+right), y is longitudinal (+forward)**.  navsim/nuPlan lidar is
**x forward, y left**.  The rotation between them is folded into ``lidar2img``
by the feature builder, so everything below stays in the SSR frame.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .ms_deform_attn import (
    CustomMSDeformableAttention,
    MSDeformableAttention3D,
    TemporalSelfAttention,
)
from .transformer_blocks import FFN


class SpatialCrossAttention(nn.Module):
    """Per-camera deformable cross-attention from BEV queries into image features.

    Only BEV queries whose 3D pillar projects inside a given camera take part in
    that camera's attention; the results are averaged over the cameras that saw
    each query.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        pc_range: Optional[Sequence[float]] = None,
        dropout: float = 0.1,
        deformable_attention: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.pc_range = list(pc_range) if pc_range is not None else None
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = deformable_attention
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        reference_points_cam: Optional[torch.Tensor] = None,
        bev_mask: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        inp_residual = query if residual is None else residual
        slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        D = reference_points_cam.size(3)

        # Visibility is per (sample, camera) -- never reuse sample 0's indexes.
        indexes: List[List[torch.Tensor]] = []
        for batch_idx in range(bs):
            per_sample = []
            for camera_idx in range(self.num_cams):
                visible = bev_mask[camera_idx, batch_idx].sum(-1) > 0
                per_sample.append(visible.nonzero().squeeze(-1))
            indexes.append(per_sample)
        max_len = max(len(idx) for per_sample in indexes for idx in per_sample)
        max_len = max(max_len, 1)

        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2]
        )
        for j in range(bs):
            for i in range(self.num_cams):
                index_query_per_img = indexes[j][i]
                n = len(index_query_per_img)
                if n == 0:
                    continue
                queries_rebatch[j, i, :n] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :n] = reference_points_cam[i, j, index_query_per_img]

        num_cams, l, bs_, embed_dims = key.shape
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i in range(self.num_cams):
                index_query_per_img = indexes[j][i]
                n = len(index_query_per_img)
                if n == 0:
                    continue
                slots[j, index_query_per_img] += queries[j, i, :n]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual


class BEVFormerLayer(nn.Module):
    """``('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')``."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_cams: int,
        pc_range: Sequence[float],
        feedforward_channels: int,
        num_points_sca: int = 8,
        num_levels: int = 1,
        ffn_dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_self_attn = TemporalSelfAttention(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=1, dropout=attn_dropout
        )
        self.spatial_cross_attn = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_cams=num_cams,
            pc_range=pc_range,
            dropout=attn_dropout,
            deformable_attention=MSDeformableAttention3D(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points_sca,
                dropout=attn_dropout,
            ),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])
        self.ffn = FFN(embed_dims, feedforward_channels, ffn_drop=ffn_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bev_pos: Optional[torch.Tensor] = None,
        ref_2d: Optional[torch.Tensor] = None,
        ref_3d: Optional[torch.Tensor] = None,
        bev_h: Optional[int] = None,
        bev_w: Optional[int] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        reference_points_cam: Optional[torch.Tensor] = None,
        bev_mask: Optional[torch.Tensor] = None,
        prev_bev: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        bev_spatial_shapes = torch.tensor(
            [[bev_h, bev_w]], dtype=torch.long, device=query.device
        )
        bev_level_start_index = torch.tensor([0], dtype=torch.long, device=query.device)

        query = self.temporal_self_attn(
            query,
            prev_bev,
            prev_bev,
            identity=query,
            query_pos=bev_pos,
            reference_points=ref_2d,
            spatial_shapes=bev_spatial_shapes,
            level_start_index=bev_level_start_index,
        )
        query = self.norms[0](query)

        query = self.spatial_cross_attn(
            query,
            key,
            value,
            residual=query,
            query_pos=None,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norms[1](query)

        query = self.ffn(query)
        query = self.norms[2](query)
        return query


class BEVFormerEncoder(nn.Module):
    """Stack of ``BEVFormerLayer``s over a fixed BEV grid."""

    def __init__(
        self,
        num_layers: int = 3,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_cams: int = 6,
        pc_range: Sequence[float] = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        num_points_in_pillar: int = 4,
        num_points_sca: int = 8,
        num_levels: int = 1,
        feedforward_channels: int = 512,
        ffn_dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.pc_range = list(pc_range)
        self.num_points_in_pillar = num_points_in_pillar
        self.layers = nn.ModuleList(
            [
                BEVFormerLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_cams=num_cams,
                    pc_range=pc_range,
                    feedforward_channels=feedforward_channels,
                    num_points_sca=num_points_sca,
                    num_levels=num_levels,
                    ffn_dropout=ffn_dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def get_reference_points(
        H: int,
        W: int,
        Z: float = 8.0,
        num_points_in_pillar: int = 4,
        dim: str = "3d",
        bs: int = 1,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float,
    ) -> torch.Tensor:
        if dim == "3d":
            zs = (
                torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            return ref_3d[None].repeat(bs, 1, 1, 1)

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        return ref_2d.repeat(bs, 1, 1).unsqueeze(2)

    def point_sampling(
        self,
        reference_points: torch.Tensor,
        pc_range: Sequence[float],
        lidar2img: torch.Tensor,
        image_hw: torch.Tensor,
    ):
        """Project BEV pillar points into every camera.

        Args:
            reference_points: ``[bs, D, num_query, 3]`` normalised BEV coords
            lidar2img: ``[bs, num_cam, 4, 4]``
            image_hw: ``[bs, num_cam, 2]`` as (H, W)
        """
        lidar2img = lidar2img.to(reference_points.dtype)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        )
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        image_hw = image_hw.to(reference_points_cam.dtype)
        reference_points_cam[..., 0] /= image_hw[None, :, :, None, 1]
        reference_points_cam[..., 1] /= image_hw[None, :, :, None, 0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        return reference_points_cam, bev_mask

    def forward(
        self,
        bev_query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bev_h: int,
        bev_w: int,
        bev_pos: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        lidar2img: torch.Tensor,
        image_hw: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs = bev_query.size(1)
        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bs,
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim="2d", bs=bs, device=bev_query.device, dtype=bev_query.dtype
        )
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, lidar2img, image_hw
        )

        shift_ref_2d = ref_2d.clone()
        if shift is not None:
            shift_ref_2d = shift_ref_2d + shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape

        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )

        output = bev_query
        for layer in self.layers:
            output = layer(
                output,
                key,
                value,
                bev_pos=bev_pos,
                ref_2d=hybrid_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
            )
        return output


class SSRPerceptionTransformer(nn.Module):
    """Wraps the encoder with camera/level embeddings and ego-motion conditioning."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        num_feature_levels: int = 1,
        ego_motion_dims: int = 18,
        use_ego_motion: bool = True,
        ego_motion_norm: bool = True,
        use_shift: bool = True,
        use_cams_embeds: bool = True,
        encoder: Optional[BEVFormerEncoder] = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_feature_levels = num_feature_levels
        self.use_ego_motion = use_ego_motion
        self.use_shift = use_shift
        self.use_cams_embeds = use_cams_embeds
        self.encoder = encoder

        self.level_embeds = nn.Parameter(torch.empty(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.empty(num_cams, embed_dims))
        ego_motion_layers = [
            nn.Linear(ego_motion_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
        ]
        if ego_motion_norm:
            # Original SSR default: can_bus_norm=True.  NAVSIM replaces the
            # CAN vector, not the conditioning network's output contract.
            ego_motion_layers.append(nn.LayerNorm(embed_dims))
        self.ego_motion_mlp = nn.Sequential(*ego_motion_layers)
        self.init_weights()

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # The global Xavier pass above also reaches deformable-attention
        # predictors, whose zero weights and radial offset biases are part of
        # their required initialization.  Restore those specialized values,
        # matching the original SSR transformer initialization order.
        for module in self.modules():
            if isinstance(
                module,
                (
                    MSDeformableAttention3D,
                    TemporalSelfAttention,
                    CustomMSDeformableAttention,
                ),
            ):
                module.init_weights()
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

    def get_bev_features(
        self,
        mlvl_feats: Sequence[torch.Tensor],
        bev_queries: torch.Tensor,
        bev_h: int,
        bev_w: int,
        bev_pos: torch.Tensor,
        lidar2img: torch.Tensor,
        image_hw: torch.Tensor,
        ego_motion: torch.Tensor,
        bev_shift: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mlvl_feats: list of ``[bs, num_cam, C, H, W]``
            bev_queries: ``[bev_h * bev_w, embed_dims]``
            bev_pos: ``[bs, embed_dims, bev_h, bev_w]``
            lidar2img: ``[bs, num_cam, 4, 4]``
            image_hw: ``[bs, num_cam, 2]``
            ego_motion: ``[bs, ego_motion_dims]``
            bev_shift: ``[bs, 2]`` normalised (shift_x, shift_y)
        Returns:
            ``[bs, bev_h * bev_w, embed_dims]``
        """
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        shift = bev_shift.to(bev_queries.dtype)
        if not self.use_shift:
            shift = torch.zeros_like(shift)

        if prev_bev is not None and prev_bev.shape[1] == bev_h * bev_w:
            prev_bev = prev_bev.permute(1, 0, 2)

        if self.use_ego_motion:
            motion = self.ego_motion_mlp(ego_motion.to(bev_queries.dtype))[None, :, :]
            bev_queries = bev_queries + motion

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append((h, w))
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        return self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            lidar2img=lidar2img,
            image_hw=image_hw,
            prev_bev=prev_bev,
            shift=shift,
        )
