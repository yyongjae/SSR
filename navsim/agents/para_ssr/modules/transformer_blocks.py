"""Plain-PyTorch replacements for the mmcv transformer bricks PARA-SSR uses.

The nuScenes model builds its decoders from mmcv's ``BaseTransformerLayer`` /
``DetrTransformerDecoderLayer`` with an ``operation_order`` tuple.  That
mechanism is reproduced here directly so the ported layers keep the same
operation ordering, residual placement and pre/post-norm behaviour as the
configs specify.

One deliberate deviation: mmcv's deprecated ``dropout=`` / ``ffn_dropout=``
kwargs mutate class-level default dicts, which is the footgun the nuScenes
config documents at length.  Here every dropout is an explicit constructor
argument with no shared state, so build order cannot change the model.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class FFN(nn.Module):
    """Position-wise feed-forward with residual, matching mmcv's ``FFN``."""

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        ffn_drop: float = 0.0,
        add_identity: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.layers(x)
        if not self.add_identity:
            return out
        return (x if identity is None else identity) + out


class MultiheadAttention(nn.Module):
    """``nn.MultiheadAttention`` with mmcv's residual/positional conventions.

    Operates on ``[num_query, bs, embed_dims]`` (``batch_first=False``), which
    is what every decoder in the PARA-SSR configs uses.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout_layer: float = 0.0,
        batch_first: bool = False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Dropout(dropout_layer)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None and query_pos is not None and key is query:
            key_pos = query_pos

        q = query if query_pos is None else query + query_pos
        k = key if key_pos is None else key + key_pos
        v = value

        if self.batch_first:
            q, k, v = (t.transpose(0, 1) for t in (q, k, v))

        out = self.attn(
            query=q, key=k, value=v, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class BaseTransformerLayer(nn.Module):
    """One transformer layer driven by an ``operation_order`` tuple.

    ``attentions`` are consumed in order as ``self_attn`` / ``cross_attn``
    operations appear.  This mirrors mmcv exactly, including the quirk the
    nuScenes config relies on for the motion decoder: a ``cross_attn`` op whose
    query, key and value all happen to be the motion queries is an
    self-attention in effect, but it still takes the ``cross_attn`` slot and its
    ``key_padding_mask``.
    """

    def __init__(
        self,
        attentions: Sequence[nn.Module],
        embed_dims: int,
        feedforward_channels: int,
        operation_order: Sequence[str],
        ffn_dropout: float = 0.0,
        ffn_num_fcs: int = 2,
    ):
        super().__init__()
        self.operation_order = tuple(operation_order)
        self.attentions = nn.ModuleList(attentions)
        num_norms = sum(1 for op in self.operation_order if op == "norm")
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_norms)])
        num_ffns = sum(1 for op in self.operation_order if op == "ffn")
        self.ffns = nn.ModuleList(
            [
                FFN(embed_dims, feedforward_channels, num_fcs=ffn_num_fcs, ffn_drop=ffn_dropout)
                for _ in range(num_ffns)
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[List[Optional[torch.Tensor]]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        num_attn = sum(1 for op in self.operation_order if op in ("self_attn", "cross_attn"))
        if attn_masks is None:
            attn_masks = [None] * num_attn
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [attn_masks] * num_attn

        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    query,
                    query,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query

    # Every PARA-SSR decoder is post-norm.
    pre_norm = False


class CustomTransformerDecoder(nn.Module):
    """A stack of identical ``BaseTransformerLayer``s."""

    def __init__(self, layers: Sequence[nn.Module], return_intermediate: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.return_intermediate = return_intermediate

    def forward(self, query: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                intermediate.append(query)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return query


def build_self_attn_decoder(
    num_layers: int,
    embed_dims: int,
    num_heads: int,
    feedforward_channels: int,
    operation_order: Sequence[str],
    attn_dropout: float = 0.0,
    ffn_dropout: float = 0.0,
    return_intermediate: bool = False,
) -> CustomTransformerDecoder:
    """Build the plain multi-head decoders (latent / waypoint / motion)."""
    layers = []
    num_attn = sum(1 for op in operation_order if op in ("self_attn", "cross_attn"))
    for _ in range(num_layers):
        attentions = [
            MultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_dropout,
                # mmcv's deprecated ``dropout=x`` maps to attention-weight
                # dropout plus residual dropout; projection dropout remains 0.
                proj_drop=0.0,
                dropout_layer=attn_dropout,
            )
            for _ in range(num_attn)
        ]
        layers.append(
            BaseTransformerLayer(
                attentions=attentions,
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                operation_order=operation_order,
                ffn_dropout=ffn_dropout,
            )
        )
    return CustomTransformerDecoder(layers, return_intermediate=return_intermediate)


class LearnedPositionalEncoding(nn.Module):
    """BEV positional encoding: separate row / column embedding tables."""

    def __init__(self, num_feats: int = 128, row_num_embed: int = 100, col_num_embed: int = 100):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """``mask``: ``[bs, h, w]`` (values unused, shape and device only)."""
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat(
                (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)),
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
