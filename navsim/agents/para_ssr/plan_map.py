"""Planning-side map consistency: keep the planned ego footprint on the drivable area.

Map distillation can only fix DAC failures where the student misreads the map.
A planner that reads the map correctly can still leave the road; nothing in the
L1 imitation loss says the road edge is a hard boundary.  This term says it, so
the two causes can be told apart with a 2x2 design (distillation on/off x this
term on/off).

``DrivableAreaTargetBuilder``
    Signed distance field (metres, > 0 inside) of the drivable area around ego,
    built from exactly the layers of ``PDMDrivableMap.from_simulation`` -- the
    map the DAC metric checks: ROADBLOCK polygons and their lanes, the lane
    connectors of ROADBLOCK_CONNECTORs, INTERSECTION and CARPARK_AREA.  The grid
    lives in the current ego frame of NAVSIM's trajectories (x forward, y left,
    rear axle), rows along x, columns along y.

``plan_map_loss``
    The four corners of the ego footprint (Pacifica, rear-axle frame, as the
    PDM scorer builds it) at every step of the COMMANDED predicted trajectory are
    looked up bilinearly in the SDF; each costs ``relu(margin - sdf)``.  A corner
    is only charged where the ground-truth footprint's corner at that step is on
    the drivable area itself, so the term never argues with the demonstration
    (map gaps, a car park the query missed).  Corners outside the grid are skipped.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from shapely.ops import unary_union

from navsim.agents.para_ssr.cache_key import cache_key
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder

# nuPlan's get_pacifica_parameters(): width 2.297, front 4.049 / rear 1.127 of the rear axle
EGO_HALF_WIDTH = 2.297 / 2
EGO_FRONT = 4.049
EGO_REAR = 1.127
EGO_CORNERS = ((EGO_FRONT, EGO_HALF_WIDTH), (EGO_FRONT, -EGO_HALF_WIDTH),
               (-EGO_REAR, EGO_HALF_WIDTH), (-EGO_REAR, -EGO_HALF_WIDTH))


def grid_shape(extent: Tuple[float, float, float, float], res: float) -> Tuple[int, int]:
    x0, x1, y0, y1 = extent
    return int(round((x1 - x0) / res)), int(round((y1 - y0) / res))


def rasterize_sdf(polygon, extent: Tuple[float, float, float, float], res: float, clip: float) -> np.ndarray:
    """Shapely (multi)polygon in the ego frame -> float32 SDF ``[nx, ny]`` (> 0 inside).

    A cell is drivable when its CENTRE is inside (cv2.fillPoly would also take the
    cells the edge merely crosses, a one-cell bias).  The distance transform
    measures to the nearest opposite cell centre; the edge lies half a cell
    before it, which is subtracted.
    """
    import shapely

    x0, x1, y0, y1 = extent
    nx, ny = grid_shape(extent, res)
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    if polygon is None or polygon.is_empty:
        mask = np.zeros((nx, ny), dtype=np.uint8)
    else:
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        shapely.prepare(polygon)
        mask = shapely.contains_xy(polygon, gx.ravel(), gy.ravel()).reshape(nx, ny).astype(np.uint8)
    if mask.all() or not mask.any():
        return np.full((nx, ny), clip if mask.all() else -clip, dtype=np.float32)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sdf = np.where(mask > 0, inside - 0.5, -(outside - 0.5)) * res
    return np.clip(sdf, -clip, clip).astype(np.float32)


class DrivableAreaTargetBuilder(AbstractTargetBuilder):
    """``drivable_sdf`` [nx, ny] float16 for ``plan_map_loss``."""

    def __init__(self, config):
        self._config = config

    def get_unique_name(self) -> str:
        cfg = self._config
        return cache_key(
            "para_ssr_drivable_sdf",
            (
                ("extent", tuple(cfg.plan_map_extent)),
                ("res", cfg.plan_map_res),
                ("clip", cfg.plan_map_clip),
                ("layers", "pdm_drivable_v1"),
            ),
        )

    def drivable_polygon(self, scene, cur_idx: int):
        from nuplan.common.actor_state.state_representation import StateSE2
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        from navsim.agents.para_ssr.para_ssr_targets import _geometry_local_coords

        x0, x1, y0, y1 = self._config.plan_map_extent
        radius = 1.05 * max(math.hypot(x, y) for x in (x0, x1) for y in (y0, y1))
        pose = scene.frames[cur_idx].ego_status.ego_pose
        origin = StateSE2(float(pose[0]), float(pose[1]), float(pose[2]))
        rb, rbc = SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR
        other = (SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA)
        objects = scene.map_api.get_proximal_map_objects(origin.point, radius, [rb, rbc, *other])
        polys = [o.polygon for o in objects.get(rb, [])]
        for layer in (rb, rbc):
            for block in objects.get(layer, []):
                polys.extend(e.polygon for e in block.interior_edges)
        for layer in other:
            polys.extend(o.polygon for o in objects.get(layer, []))
        polys = [p for p in polys if p is not None and not p.is_empty]
        if not polys:
            return None
        return _geometry_local_coords(unary_union(polys), origin)

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        cfg = self._config
        cur_idx = scene.scene_metadata.num_history_frames - 1
        sdf = rasterize_sdf(self.drivable_polygon(scene, cur_idx), tuple(cfg.plan_map_extent),
                            float(cfg.plan_map_res), float(cfg.plan_map_clip))
        return {"drivable_sdf": torch.from_numpy(sdf).half()}


def footprint_corners(poses: torch.Tensor) -> torch.Tensor:
    """``[..., 3]`` rear-axle poses (x, y, heading) -> ``[..., 4, 2]`` footprint corners."""
    c = torch.tensor(EGO_CORNERS, dtype=poses.dtype, device=poses.device)       # [4, 2]
    cos, sin = torch.cos(poses[..., 2:3]), torch.sin(poses[..., 2:3])           # [..., 1]
    x = poses[..., None, 0] + cos * c[:, 0] - sin * c[:, 1]
    y = poses[..., None, 1] + sin * c[:, 0] + cos * c[:, 1]
    return torch.stack([x, y], dim=-1)


def sample_sdf(sdf: torch.Tensor, pts: torch.Tensor, extent) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear SDF at ``pts`` [B, N, 2] (x, y) -> (values [B, N], inside-grid mask [B, N])."""
    x0, x1, y0, y1 = extent
    u = (pts[..., 1] - y0) / (y1 - y0) * 2 - 1          # columns <- y
    v = (pts[..., 0] - x0) / (x1 - x0) * 2 - 1          # rows <- x
    grid = torch.stack([u, v], dim=-1).unsqueeze(1)      # [B, 1, N, 2]
    val = F.grid_sample(sdf.unsqueeze(1).to(pts.dtype), grid, mode="bilinear",
                        padding_mode="border", align_corners=False)[:, 0, 0]
    on_grid = (u.abs() <= 1) & (v.abs() <= 1)
    return val, on_grid


def plan_map_loss(
    ego_fut_preds: torch.Tensor,
    command: torch.Tensor,
    gt_offsets: torch.Tensor,
    gt_mask: torch.Tensor,
    sdf: torch.Tensor,
    extent,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Hinge on the commanded branch's footprint corners; see the module doc."""
    B, _, T, _ = ego_fut_preds.shape
    branch = command.argmax(dim=1)
    offsets = ego_fut_preds[torch.arange(B, device=branch.device), branch]          # [B, T, 3]
    pred = footprint_corners(offsets.cumsum(dim=1)).reshape(B, T * 4, 2)
    gt = footprint_corners(gt_offsets.to(offsets.dtype).cumsum(dim=1)).reshape(B, T * 4, 2)
    s_pred, on_pred = sample_sdf(sdf, pred, extent)
    with torch.no_grad():
        s_gt, on_gt = sample_sdf(sdf, gt, extent)
        step = gt_mask.to(offsets.dtype).repeat_interleave(4, dim=1) > 0
        charge = step & on_pred & on_gt & (s_gt >= 0)
    hinge = torch.relu(margin - s_pred) * charge
    loss = hinge.sum() / charge.sum().clamp(min=1)
    with torch.no_grad():
        valid_pred = step & on_pred
        valid_gt = step & on_gt
        metrics = {
            "plan_map/pred_outside": ((s_pred < 0) & valid_pred).sum() / valid_pred.sum().clamp(min=1),
            "plan_map/gt_outside": ((s_gt < 0) & valid_gt).sum() / valid_gt.sum().clamp(min=1),
            "plan_map/charged_frac": charge.float().mean(),
        }
    return loss, metrics
