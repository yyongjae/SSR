"""BEV segmentation of the drivable AREA, next to (not instead of) the vector map head.

The vector map head emits ROAD as 20-point OPEN polylines.  Measured on navtest: 0.2% of predicted
road instances close on themselves, and the median gap between a polyline's two ends is 32 m -- the
full width of the BEV.  A polyline says where an edge runs, never which side of it is legal, and
there are only ~3.9 road instances per scene, so the region cannot be recovered from them either.

DAC is the opposite kind of question.  ``pdm_scorer.py:318`` asks
``drivable_area_map.points_in_polygons(ego_coords)`` -- are the four corners of the ego footprint
inside a polygon.  Nothing the model currently produces answers that.

NAVSIM's own baseline prices the gap: removing TransFuser's BEV segmentation drops DAC from 92.8 to
90.5 (arXiv 2406.15349, Table 2).  This module adds that supervision.  The vector head is untouched
and both run together; with ``bev_seg_weight = 0`` (the default) nothing here is constructed, no
target is built, and training is byte-identical to before.

GRID.  The target lives on the SHARED BEV grid, not on ``plan_map``'s finer one, so the head is a
readout of ``bev_embed`` rather than a decoder asked to invent resolution:

    H = ``bev_h`` rows along SSR y (forward), ``pc_range[1] .. pc_range[4]``
    W = ``bev_w`` cols along SSR x (lateral), ``pc_range[0] .. pc_range[3]``
    flattened ``h * W + w`` -- the order ``get_reference_points`` builds (``bevformer.py:391-399``)

LAYERS.  Exactly ``PDMDrivableMap.from_simulation``'s, reused from
``DrivableAreaTargetBuilder.drivable_polygon``: ROADBLOCK polygons and their lanes, the lane
connectors of ROADBLOCK_CONNECTORs, INTERSECTION and CARPARK_AREA.  Polygons arrive in NAVSIM ego
axes and are swapped into SSR BEV axes exactly as the vector targets are
(``para_ssr_targets.py:583``: ``SSR x = -navsim y``, ``SSR y = navsim x``).
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.para_ssr.cache_key import cache_key
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


def bev_extent(pc_range: Sequence[float]) -> Tuple[float, float, float, float]:
    """``pc_range`` -> (y0, y1, x0, x1) in SSR axes: rows along y forward, cols along x lateral."""
    return float(pc_range[1]), float(pc_range[4]), float(pc_range[0]), float(pc_range[3])


def rasterize_bev_mask(polygon, pc_range: Sequence[float], bev_h: int, bev_w: int) -> np.ndarray:
    """Shapely (multi)polygon in NAVSIM ego axes -> uint8 ``[bev_h, bev_w]`` drivable mask.

    A cell is drivable when its CENTRE is inside, matching ``plan_map.rasterize_sdf`` -- cv2.fillPoly
    would also take every cell an edge merely clips, a one-cell outward bias on a 0.64 m grid.
    """
    y0, y1, x0, x1 = bev_extent(pc_range)
    if polygon is None or polygon.is_empty:
        return np.zeros((bev_h, bev_w), dtype=np.uint8)
    import shapely

    # cell centres in SSR axes, then back to NAVSIM axes for the containment test
    ssr_y = y0 + (np.arange(bev_h) + 0.5) * (y1 - y0) / bev_h        # forward
    ssr_x = x0 + (np.arange(bev_w) + 0.5) * (x1 - x0) / bev_w        # lateral
    gy, gx = np.meshgrid(ssr_y, ssr_x, indexing="ij")                # [H, W]
    nav_x, nav_y = gy, -gx                                           # inverse of para_ssr_targets:583
    shapely.prepare(polygon)
    inside = shapely.contains_xy(polygon, nav_x.ravel(), nav_y.ravel())
    return inside.reshape(bev_h, bev_w).astype(np.uint8)


class DrivableSegTargetBuilder(AbstractTargetBuilder):
    """``drivable_seg`` uint8 ``[bev_h, bev_w]``, 1 where the DAC metric would accept a footprint."""

    def __init__(self, config):
        self._config = config

    def get_unique_name(self) -> str:
        cfg = self._config
        return cache_key(
            "para_ssr_drivable_seg",
            (
                ("pc_range", tuple(float(v) for v in cfg.map_pc_range)),
                ("bev", (int(cfg.bev_h), int(cfg.bev_w))),
                ("layers", "pdm_drivable_v1"),
            ),
        )

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        # the polygon query is identical to plan_map's; reuse it rather than drift from the metric
        from navsim.agents.para_ssr.plan_map import DrivableAreaTargetBuilder

        cfg = self._config
        cur_idx = scene.scene_metadata.num_history_frames - 1
        poly = DrivableAreaTargetBuilder(cfg).drivable_polygon(scene, cur_idx)
        mask = rasterize_bev_mask(poly, cfg.map_pc_range, int(cfg.bev_h), int(cfg.bev_w))
        return {"drivable_seg": torch.from_numpy(mask)}


class BevSegHead(nn.Module):
    """``bev_embed`` [B, H*W, C] -> drivable logits [B, H, W], plus a token view of the same map.

    Deliberately shallow.  The question is whether the BEV already carries the region and the model
    simply never had to say so; a deep decoder would blur that into "can a decoder learn it".

    TOKEN VIEW.  When this head REPLACES the vector map head (``map_head_type = "seg"``) the
    planner's task interaction still needs map memories.  ``prepare_task_memories`` reads
    ``map_point_hidden`` [B, V, P, C], ``all_map_pts_preds`` [L, B, V, P, 2] in [0, 1] and
    ``all_map_cls_scores`` [L, B, V, n]; the head produces the same keys from the segmentation
    itself so that function is untouched: the hidden map is average-pooled to a coarse grid
    (``token_pool`` x ``token_pool`` cells -> one token each, P = 1), the token's position is its
    cell centre normalised over the BEV, and its confidence is the mean drivable probability of the
    cell.  So the planner reads "where is road, how sure" on a grid instead of "where are
    boundary polylines".  These keys go to the planner only, never to ``predictions``, so the
    auxiliary evaluator does not mistake 200 grid cells for map instances.
    """

    def __init__(self, embed_dims: int, bev_h: int, bev_w: int, hidden: int = 128,
                 token_pool: int = 5):
        super().__init__()
        self.bev_h, self.bev_w = int(bev_h), int(bev_w)
        self.token_pool = int(token_pool)
        if self.bev_h % self.token_pool or self.bev_w % self.token_pool:
            raise ValueError(f"token_pool={token_pool} must divide the BEV grid {bev_h}x{bev_w}")
        self.trunk = nn.Sequential(
            nn.Conv2d(embed_dims, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, 1, 1)
        self.token_proj = nn.Conv2d(hidden, embed_dims, 1)

    def forward(self, bev_embed: torch.Tensor, return_hidden: bool = False):
        bs, n, c = bev_embed.shape
        if n != self.bev_h * self.bev_w:
            raise ValueError(f"bev_embed has {n} cells, expected {self.bev_h}x{self.bev_w}")
        x = bev_embed.transpose(1, 2).reshape(bs, c, self.bev_h, self.bev_w)   # h * W + w
        feat = self.trunk(x.float())
        logits = self.out(feat).squeeze(1)                                      # [B, H, W]
        if not return_hidden:
            return logits
        p = self.token_pool
        tok = F.avg_pool2d(self.token_proj(feat), p)                            # [B, C, H/p, W/p]
        conf = F.avg_pool2d(torch.sigmoid(logits)[:, None], p)                  # [B, 1, H/p, W/p]
        hh, ww = tok.shape[-2:]
        # cell centres in the BEV's own normalised frame: x over W (lateral), y over H (forward)
        ys = (torch.arange(hh, device=tok.device, dtype=tok.dtype) + 0.5) / hh
        xs = (torch.arange(ww, device=tok.device, dtype=tok.dtype) + 0.5) / ww
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([gx, gy], -1).reshape(1, hh * ww, 1, 2).expand(bs, -1, -1, -1)
        tokens = tok.flatten(2).transpose(1, 2)                                 # [B, N, C]
        return logits, {
            "map_point_hidden": tokens[:, :, None, :].to(bev_embed.dtype),     # [B, N, 1, C]
            "all_map_pts_preds": pos[None].to(bev_embed.dtype),                 # [1, B, N, 1, 2]
            "all_map_cls_scores": torch.logit(conf.flatten(2).transpose(1, 2).clamp(1e-4, 1 - 1e-4))[None]
            .to(bev_embed.dtype),                                               # [1, B, N, 1]
        }


def bev_seg_loss(logits: torch.Tensor, target: torch.Tensor,
                 pos_weight: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """BCE + soft dice.  Dice is there because the drivable area covers most of the front ROI and
    plain BCE is happy to predict "all road"; the metrics below say which failure mode is happening.
    """
    t = target.to(logits.dtype)
    w = logits.new_tensor(pos_weight)
    bce = F.binary_cross_entropy_with_logits(logits, t, pos_weight=w)
    p = torch.sigmoid(logits)
    num = 2.0 * (p * t).sum(dim=(1, 2)) + 1.0
    den = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + 1.0
    dice = 1.0 - (num / den).mean()
    with torch.no_grad():
        hard = p > 0.5
        tp = (hard & (t > 0.5)).sum().float()
        iou = tp / ((hard | (t > 0.5)).sum().float() + 1e-6)
        metrics = {
            "bev_seg/bce": bce.detach(),
            "bev_seg/dice": dice.detach(),
            "bev_seg/iou": iou,
            "bev_seg/pred_pos": hard.float().mean(),
            "bev_seg/true_pos": (t > 0.5).float().mean(),
        }
    return bce + dice, metrics
