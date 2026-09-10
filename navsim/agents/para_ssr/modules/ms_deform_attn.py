"""Deformable attention, reimplemented without mmcv.

The nuScenes PARA-SSR builds every attention out of mmcv's
``MultiScaleDeformableAttention`` and its compiled CUDA op.  navsim's
environment has no mmcv, so the three variants BEVFormer needs are re-derived
here on top of ``F.grid_sample``:

``MSDeformableAttention3D``   3D pillar sampling for the spatial cross-attention
``TemporalSelfAttention``     BEV(t) <-> BEV(t-1) self-attention
``CustomMSDeformableAttention``  plain 2D deformable attention for the decoders

The sampling kernel is mathematically the same as mmcv's
``multi_scale_deformable_attn_pytorch`` fallback, so weights and behaviour stay
comparable to the mmdet3d implementation.  It is kept in fp32 on purpose: the
polynomial accumulation over ``num_points * num_levels`` bilinear taps is the
term the original config forces to fp32, and autocast is disabled around it.
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch multi-scale deformable attention.

    Args:
        value: ``[bs, num_keys, num_heads, head_dims]``
        value_spatial_shapes: ``[num_levels, 2]`` of (H, W)
        sampling_locations: ``[bs, num_queries, num_heads, num_levels, num_points, 2]``
            in normalised ``[0, 1]`` coordinates
        attention_weights: ``[bs, num_queries, num_heads, num_levels, num_points]``

    Returns:
        ``[bs, num_queries, num_heads * head_dims]``
    """
    bs, _, num_heads, head_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([int(H * W) for H, W in value_spatial_shapes], dim=1)
    # grid_sample expects [-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H, W) in enumerate(value_spatial_shapes):
        H, W = int(H), int(W)
        # [bs, H*W, num_heads, head_dims] -> [bs*num_heads, head_dims, H, W]
        value_l = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, head_dims, H, W)
        )
        # [bs, num_queries, num_heads, num_points, 2] -> [bs*num_heads, num_queries, num_points, 2]
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_list.append(
            F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )

    # [bs*num_heads, 1, num_queries, num_levels*num_points]
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * head_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def _constant_(tensor: torch.Tensor, val: float) -> None:
    nn.init.constant_(tensor, val)


def _xavier_(module: nn.Module) -> None:
    if hasattr(module, "weight") and module.weight is not None and module.weight.dim() > 1:
        nn.init.xavier_uniform_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


def _init_sampling_offsets(
    sampling_offsets: nn.Linear, num_heads: int, num_levels: int, num_points: int
) -> None:
    """mmcv's grid init: offsets fan out evenly over the unit circle."""
    _constant_(sampling_offsets.weight, 0.0)
    thetas = torch.arange(num_heads, dtype=torch.float32) * (2.0 * math.pi / num_heads)
    grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
    grid_init = grid_init.view(num_heads, 1, 1, 2).repeat(1, num_levels, num_points, 1)
    for i in range(num_points):
        grid_init[:, :, i, :] *= i + 1
    with torch.no_grad():
        sampling_offsets.bias.copy_(grid_init.view(-1))


class MSDeformableAttention3D(nn.Module):
    """Deformable attention used inside BEVFormer's spatial cross-attention.

    Differs from the plain 2D variant in that the caller supplies one reference
    point per (query, pillar-height) sample and the module returns the sampled
    features without an output projection -- ``SpatialCrossAttention`` owns that.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims {embed_dims} must divide num_heads {num_heads}")
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        _init_sampling_offsets(self.sampling_offsets, self.num_heads, self.num_levels, self.num_points)
        _constant_(self.attention_weights.weight, 0.0)
        _constant_(self.attention_weights.bias, 0.0)
        _xavier_(self.value_proj)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            # reference_points arrives as [bs, num_query, num_Z_anchors, xy]
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy
            )
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy
            )
        else:
            raise ValueError(
                f"reference_points last dim must be 2, got {reference_points.shape[-1]}"
            )

        # fp32 only -- see module docstring.
        with torch.cuda.amp.autocast(enabled=False):
            output = multi_scale_deformable_attn_pytorch(
                value.float(), spatial_shapes, sampling_locations.float(), attention_weights.float()
            )
        output = output.to(query.dtype)

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output


class TemporalSelfAttention(nn.Module):
    """BEVFormer temporal self-attention: BEV(t) attends to [BEV(t), BEV(t-1)]."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        num_bev_queue: int = 2,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims {embed_dims} must divide num_heads {num_heads}")
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.im2col_step = im2col_step

        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        _constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels * self.num_bev_queue, self.num_points, 1
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))
        _constant_(self.attention_weights.weight, 0.0)
        _constant_(self.attention_weights.bias, 0.0)
        _xavier_(self.value_proj)
        _xavier_(self.output_proj)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if value is None:
            # No history yet: attend to the current BEV twice.
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert self.num_bev_queue * bs == value.shape[0]

        # The encoder packs the temporal queue in batch-major order:
        # [history_0, current_0, history_1, current_1, ...].  Taking
        # ``value[:bs]`` therefore mixes samples as soon as bs > 1.  Restore
        # the queue axis before selecting each sample's history tensor.
        history_value = value.reshape(
            bs, self.num_bev_queue, num_value, embed_dims
        )[:, 0]
        query = torch.cat([history_value, query], -1)
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels * self.num_points,
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points
        )

        attention_weights = (
            attention_weights.permute(0, 3, 1, 2, 4, 5)
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points)
            .contiguous()
        )
        sampling_offsets = (
            sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(
                f"reference_points last dim must be 2, got {reference_points.shape[-1]}"
            )

        with torch.cuda.amp.autocast(enabled=False):
            output = multi_scale_deformable_attn_pytorch(
                value.float(), spatial_shapes, sampling_locations.float(), attention_weights.float()
            )
        output = output.to(query.dtype)

        # fuse the two BEV-queue halves back together
        output = output.permute(1, 2, 0)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue).mean(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


class CustomMSDeformableAttention(nn.Module):
    """Plain 2D deformable attention for the detection / map decoders."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims {embed_dims} must divide num_heads {num_heads}")
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        _init_sampling_offsets(self.sampling_offsets, self.num_heads, self.num_levels, self.num_points)
        _constant_(self.attention_weights.weight, 0.0)
        _constant_(self.attention_weights.bias, 0.0)
        _xavier_(self.value_proj)
        _xavier_(self.output_proj)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            identity = identity.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}"
            )

        with torch.cuda.amp.autocast(enabled=False):
            output = multi_scale_deformable_attn_pytorch(
                value.float(), spatial_shapes, sampling_locations.float(), attention_weights.float()
            )
        output = output.to(query.dtype)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)
            identity = identity.permute(1, 0, 2)
        return self.dropout(output) + identity
