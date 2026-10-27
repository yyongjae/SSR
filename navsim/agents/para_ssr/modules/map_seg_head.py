"""km 10/27: raster (BEV segmentation) map head, the ``map_repr='seg'`` replacement of the vector ``ParaMapHead``.

Head: bev_embed [B, H*W, C] (row = y_forward from pc_range y0, col = x_right from pc_range x0, as the map decoder's
normalised reference points address it) -> conv trunk -> 2x upsample ->
  map_seg_logits [B, K, 2H, 2W]   per-class sigmoid logits (classes overlap, e.g. centerline / crosswalk on road)
Classes = MAP_CLASS_NAMES (road, walkway, centerline, crosswalk); GT = para_ssr_targets.ParaSSRTargetBuilder
._compute_map_seg_targets (same map layers as the vector GT, rasterised at the 2x grid).

Planner memory (return_hidden=True): the trunk feature average-pooled to a G x G grid of BEV blocks, one token per block,
in place of the vector head's polyline instances:
  map_point_hidden [B, G*G, 1, C]  (one 'point' per token, so the planner's point-mean is the identity)
  map_token_xy     [B, G*G, 1, 2]  block centre, normalised to [0, 1] over pc_range (x_right, y_forward)
  map_token_score  [B, G*G, 1]     max class probability inside the block (detached), the foreground confidence
Loss: per-class BCE + soft dice, both averaged over classes.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _blk(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class ParaMapSegHead(nn.Module):
    def __init__(self, cin: int, bev_h: int, bev_w: int, num_classes: int = 4, hid: int = 128, tokens: int = 20,
                 bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bev_h, self.bev_w, self.num_classes, self.tokens = bev_h, bev_w, num_classes, tokens
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.inp = nn.Sequential(nn.Conv2d(cin, hid, 1), nn.BatchNorm2d(hid), nn.ReLU(inplace=True))
        self.low = nn.Sequential(_blk(hid, hid), _blk(hid, hid))
        self.up = nn.Sequential(nn.ConvTranspose2d(hid, hid // 2, 2, stride=2), nn.BatchNorm2d(hid // 2),
                                nn.ReLU(inplace=True), _blk(hid // 2, hid // 2))
        self.out = nn.Conv2d(hid // 2, num_classes, 1)
        self.tok = nn.Conv2d(hid, cin, 1)
        nn.init.constant_(self.out.bias, -2.0)     # most pixels are background for every class but road
        c = (torch.arange(tokens, dtype=torch.float32) + 0.5) / tokens
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("token_xy", torch.stack([gx, gy], -1).reshape(-1, 1, 2), persistent=False)

    def forward(self, bev_embed: torch.Tensor, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        b, n, c = bev_embed.shape
        x = bev_embed.transpose(1, 2).reshape(b, c, self.bev_h, self.bev_w)
        z = self.low(self.inp(x))
        logits = self.out(self.up(z))
        out = {"map_seg_logits": logits}
        if return_hidden:
            g = self.tokens
            t = F.adaptive_avg_pool2d(self.tok(z), g).flatten(2).transpose(1, 2)            # [B, G*G, C]
            score = F.adaptive_max_pool2d(logits.detach().sigmoid(), g).amax(1).flatten(1)  # [B, G*G]
            out.update(map_point_hidden=t[:, :, None], map_token_score=score[..., None],
                       map_token_xy=self.token_xy[None].expand(b, -1, -1, -1).to(t.dtype))
        return out

    def loss(self, preds: Dict[str, torch.Tensor], gt_map_seg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """gt_map_seg [B, K, 2H, 2W] (0 / 1)"""
        logits = preds["map_seg_logits"].float()
        gt = gt_map_seg.to(logits.device).float()
        if gt.shape != logits.shape:
            raise ValueError(f"gt_map_seg {tuple(gt.shape)} does not match map_seg_logits {tuple(logits.shape)}")
        bce = F.binary_cross_entropy_with_logits(logits, gt)
        p = logits.sigmoid()
        inter = (p * gt).sum((0, 2, 3))
        dice = 1.0 - (2.0 * inter + 1.0) / (p.sum((0, 2, 3)) + gt.sum((0, 2, 3)) + 1.0)
        return {"loss_map_seg_bce": self.bce_weight * bce, "loss_map_seg_dice": self.dice_weight * dice.mean()}
