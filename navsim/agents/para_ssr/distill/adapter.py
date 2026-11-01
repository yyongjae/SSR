"""Per-cell residual MLP adapter and BEV layout helpers.

Ported from ``projects/mmdet3d_plugin/SSR/utils/planning_distill.py`` for the
navsim PARA-SSR agent.  The module itself is framework-agnostic; only the
surrounding plumbing differs between the mmdet3d and navsim hosts.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def bev_tokens_to_map(tokens: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
    """``[B, H*W, C]`` -> ``[B, C, H, W]``.

    navsim's ``bev_embed`` is row-major over ``(bev_h, bev_w)``, the same
    convention the mmdet3d SSR head uses, so this is a plain reshape.
    """
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be BNC, got {tuple(tokens.shape)}")
    height, width = (int(v) for v in spatial_size)
    batch, num, channels = tokens.shape
    if num != height * width:
        raise ValueError(
            f"expected {height * width} tokens for {height}x{width}, got {num}"
        )
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def bev_map_to_tokens(feature: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` -> ``[B, H*W, C]``."""
    if feature.dim() != 4:
        raise ValueError(f"feature must be BCHW, got {tuple(feature.shape)}")
    return feature.flatten(2).transpose(1, 2)


class PlanningBEVAdapter(nn.Module):
    """A deliberately small, per-cell residual MLP.

    Spatial alignment is fixed and happens outside this module, so the same
    learned adapter applies to a teacher's cached grid and to the student grid.
    During stage-2 distillation its weights are frozen: that stops the adapter
    from rotating or collapsing the feature space merely to make the feature
    loss easy, which is the whole point of the frozen-adapter contract.
    """

    def __init__(self, channels: int = 256, hidden_channels: int = 256,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.pre_norm = nn.LayerNorm(self.channels)
        self.fc1 = nn.Linear(self.channels, int(hidden_channels))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_channels), self.channels)
        self.out_norm = nn.LayerNorm(self.channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Start as a normalised identity so the planning decoder sees a sensible
        # BEV on iteration zero instead of a random projection.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() == 4:
            feature = bev_map_to_tokens(feature)
        if feature.dim() != 3 or feature.size(-1) != self.channels:
            raise ValueError(
                f"expected BNC/BCHW with C={self.channels}, got "
                f"{tuple(feature.shape)}"
            )
        residual = feature
        feature = self.pre_norm(feature)
        feature = self.fc2(self.dropout(self.act(self.fc1(feature))))
        return self.out_norm(residual + feature)
