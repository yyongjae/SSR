"""DINOv2 ViT-S image encoder with LoRA adaptation: the image backbone of DrivoR, feeding BEVFormer.

Ported from ``navsim/agents/drivoR/layers/image_encoder/dinov2_lora.py`` of DrivoR (github.com/valeoai/DrivoR @ fc6e5aa,
Apache-2.0), which adapts github.com/mrabiabrn/robustbev.

Same as DrivoR's ``agent.config.image_backbone`` (drivoR.yaml):
  - timm ``vit_small_patch14_reg4_dinov2.lvd142m``: 12 blocks, 384 dims, 4 pretrained register tokens;
  - every pretrained weight frozen; each block's ``attn.qkv`` gets low-rank terms on q and v, none on k; rank 32;
  - A matrices Kaiming-uniform, B matrices zero, so an untrained adapter returns exactly the pretrained output.

Different from DrivoR, because this model keeps BEVFormer and DrivoR has no BEV:
  - no scene tokens: DrivoR prepends 16 learned registers per camera and keeps only those; BEVFormer reads image positions,
    so this encoder returns the patch tokens as a feature map and drops the class and register tokens;
  - the feature map goes through ``SingleLevelFPN`` in ParaSSRModel, not through DrivoR's linear neck.

The input size is fixed at construction (timm interpolates the position embedding once, while loading the weights) and
timm's patch embedding asserts it on every call, so a wrong image size stops the run instead of resampling silently.
There is no BatchNorm in this encoder, so ``backbone_bn_mode`` does nothing to it, and no dropout or stochastic depth, so
train and eval mode give the same output (GridMask, applied before this module, is the only training-time difference).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import timm
import torch
import torch.nn as nn


def is_dinov2(image_architecture: str) -> bool:
    """True for the timm tags this encoder builds; every other tag stays on timm's ``features_only`` path."""
    return "dinov2" in str(image_architecture)


class LoRAQKV(nn.Module):
    """``qkv`` of one timm Attention plus trainable low-rank terms on q and v.  k keeps the pretrained projection."""

    def __init__(self, qkv: nn.Linear, rank: int):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.linear_a_q = nn.Linear(self.dim, rank, bias=False)
        self.linear_b_q = nn.Linear(rank, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, rank, bias=False)
        self.linear_b_v = nn.Linear(rank, self.dim, bias=False)
        for a in (self.linear_a_q, self.linear_a_v):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (self.linear_b_q, self.linear_b_v):
            nn.init.zeros_(b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)                       # [B, N, 3 * dim], ordered q, k, v
        qkv[:, :, : self.dim] += self.linear_b_q(self.linear_a_q(x))
        qkv[:, :, -self.dim :] += self.linear_b_v(self.linear_a_v(x))
        return qkv


class Dinov2LoRAEncoder(nn.Module):
    """``[B, 3, H, W]`` -> ``[[B, num_features, H / 14, W / 14]]``: the last layer's normalized patch tokens."""

    def __init__(self, architecture: str, image_hw: Tuple[int, int], rank: int, pretrained: bool):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"vit_lora_rank must be positive, got {rank}")
        self.vit = timm.create_model(architecture, pretrained=pretrained, img_size=tuple(image_hw), num_classes=0)
        patch_h, patch_w = self.vit.patch_embed.patch_size
        if image_hw[0] % patch_h or image_hw[1] % patch_w:
            raise ValueError(f"image (H, W) {tuple(image_hw)} is not a multiple of the patch {(patch_h, patch_w)}: "
                             "set image_size (and crop_top) so that it is")
        self.vit.requires_grad_(False)
        for block in self.vit.blocks:
            block.attn.qkv = LoRAQKV(block.attn.qkv, rank)
        self.num_features = int(self.vit.num_features)

    @classmethod
    def from_config(cls, cfg) -> "Dinov2LoRAEncoder":
        size = getattr(cfg, "image_size", None)
        if not size:
            raise ValueError(f"{cfg.image_architecture} needs image_size (W, H): its input size is fixed at construction")
        width, height = (int(v) for v in size)
        return cls(cfg.image_architecture, (height - int(cfg.crop_top), width), int(cfg.vit_lora_rank),
                   bool(getattr(cfg, "backbone_pretrained", True)))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        tokens = self.vit.forward_features(x)[:, self.vit.num_prefix_tokens :]      # after the final LayerNorm
        height, width = self.vit.patch_embed.grid_size
        return [tokens.transpose(1, 2).reshape(x.shape[0], self.num_features, height, width)]
