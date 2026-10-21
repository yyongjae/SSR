"""Targets of the factorized planner (fact/): GT rasters and the route centerline.

Everything is expressed in the NAVSIM ego frame of the current frame (x forward,
y left, rear axle).  Two grids are produced:

* the *field grid* of ``plan_map`` (``plan_map_extent`` at ``plan_map_res``,
  default (-8, 72, -32, 32) m at 0.25 m -> [320, 256], rows along x, columns
  along y): ``fact_occ`` [T, nx, ny] uint8, the footprint of every annotated
  object at future step t = 0.5 .. 4.0 s, used by the proxy PDM labels
  (fact/proxy.py) together with ``drivable_sdf`` (plan_map.DrivableAreaTargetBuilder);
* the *BEV grid* of the model (SSR layout, rows = forward 0..32 m, columns =
  right -32..32 m, 50 x 100 at 0.64 m): ``fact_occ_bev`` [T, 50, 100] uint8 and
  ``fact_sdf_bev`` [50, 100] float16, targets of the two auxiliary BEV heads.

``fact_route`` [R, 2] is the PDM-Closed route centerline (the one the ego
progress metric projects onto), resampled at 1 m from the ego projection
forward; ``fact_route_valid`` [R] masks stations beyond its end and
``fact_route_ok`` is 0 when the route could not be built.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from navsim.agents.para_ssr.cache_key import cache_key
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# geometry helpers
# ---------------------------------------------------------------------------- #
def _box_corners(cx: np.ndarray, cy: np.ndarray, length: np.ndarray, width: np.ndarray,
                 heading: np.ndarray) -> np.ndarray:
    """``[A]`` box params -> ``[A, 4, 2]`` corners (counter-clockwise)."""
    c, s = np.cos(heading), np.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    lx = np.stack([hl, hl, -hl, -hl], axis=-1)
    ly = np.stack([hw, -hw, -hw, hw], axis=-1)
    x = cx[:, None] + c[:, None] * lx - s[:, None] * ly
    y = cy[:, None] + s[:, None] * lx + c[:, None] * ly
    return np.stack([x, y], axis=-1)


def _future_boxes_in_current_frame(scene, cur_idx: int, step: int) -> Optional[np.ndarray]:
    """Annotated boxes of frame ``cur_idx + step`` as ``[A, 5]`` (x, y, length, width, heading)
    in the CURRENT ego frame, or ``None`` when that frame does not exist."""
    f_idx = cur_idx + step
    if f_idx >= len(scene.frames):
        return None
    boxes = np.asarray(scene.frames[f_idx].annotations.boxes, dtype=np.float64).reshape(-1, 7)
    if boxes.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float64)
    cur = np.asarray(scene.frames[cur_idx].ego_status.ego_pose, dtype=np.float64)
    fut = np.asarray(scene.frames[f_idx].ego_status.ego_pose, dtype=np.float64)
    # future-frame local -> global
    fc, fs = math.cos(fut[2]), math.sin(fut[2])
    gx = fc * boxes[:, 0] - fs * boxes[:, 1] + fut[0]
    gy = fs * boxes[:, 0] + fc * boxes[:, 1] + fut[1]
    gh = boxes[:, 6] + fut[2]
    # global -> current local
    c, s = math.cos(cur[2]), math.sin(cur[2])
    dx, dy = gx - cur[0], gy - cur[1]
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    lh = gh - cur[2]
    return np.stack([lx, ly, boxes[:, 3], boxes[:, 4], lh], axis=-1)


def _fill(mask: np.ndarray, polys_px: np.ndarray) -> None:
    """Fill ``[A, 4, 2]`` polygons given in (col, row) pixel coordinates into ``mask`` (in place)."""
    if polys_px.shape[0] == 0:
        return
    pts = np.round(polys_px).astype(np.int32)
    cv2.fillPoly(mask, list(pts), 1)


def rasterize_field_grid(boxes: np.ndarray, extent, res: float) -> np.ndarray:
    """Boxes ``[A, 5]`` (nav frame) -> uint8 ``[nx, ny]`` on the field grid (rows x, cols y)."""
    x0, x1, y0, y1 = extent
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    mask = np.zeros((nx, ny), dtype=np.uint8)
    if boxes.shape[0]:
        corners = _box_corners(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4])
        px = np.stack([(corners[..., 1] - y0) / res, (corners[..., 0] - x0) / res], axis=-1)  # (col, row)
        _fill(mask, px)
    return mask


def rasterize_bev_grid(boxes: np.ndarray, pc_range, bev_h: int, bev_w: int) -> np.ndarray:
    """Boxes ``[A, 5]`` (nav frame) -> uint8 ``[bev_h, bev_w]`` in SSR layout
    (row = forward = nav x, col = right = -nav y)."""
    xr0, yf0, _, xr1, yf1, _ = pc_range
    mask = np.zeros((bev_h, bev_w), dtype=np.uint8)
    if boxes.shape[0]:
        corners = _box_corners(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4])
        xr, yf = -corners[..., 1], corners[..., 0]
        col = (xr - xr0) / (xr1 - xr0) * bev_w
        row = (yf - yf0) / (yf1 - yf0) * bev_h
        _fill(mask, np.stack([col, row], axis=-1))
    return mask


def sdf_on_bev_grid(sdf: np.ndarray, extent, res: float, pc_range, bev_h: int, bev_w: int) -> np.ndarray:
    """Nearest-cell lookup of the field-grid SDF at the BEV cell centres -> ``[bev_h, bev_w]``."""
    x0, x1, y0, y1 = extent
    xr0, yf0, _, xr1, yf1, _ = pc_range
    rows = np.arange(bev_h)
    cols = np.arange(bev_w)
    nav_x = yf0 + (rows + 0.5) * (yf1 - yf0) / bev_h                 # [bev_h]
    nav_y = -(xr0 + (cols + 0.5) * (xr1 - xr0) / bev_w)             # [bev_w]
    ix = np.clip(((nav_x - x0) / res).astype(np.int64), 0, sdf.shape[0] - 1)
    iy = np.clip(((nav_y - y0) / res).astype(np.int64), 0, sdf.shape[1] - 1)
    return sdf[ix[:, None], iy[None, :]]


# ---------------------------------------------------------------------------- #
# route centerline (PDM-Closed's, the one ego progress is measured along)
# ---------------------------------------------------------------------------- #
class _RouteOnlyPlanner:
    """Holds the AbstractPDMPlanner route helpers without the planner interface."""

    _impl = None

    @classmethod
    def build(cls, map_radius: float):
        if cls._impl is None:
            from navsim.planning.simulation.planner.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner

            class _Impl(AbstractPDMPlanner):
                def __init__(self, r):
                    super().__init__(map_radius=r)

                # AbstractPlanner interface (unused)
                def name(self):
                    return "fact_route"

                def observation_type(self):
                    return None

                def initialize(self, initialization):
                    pass

                def compute_planner_trajectory(self, current_input):
                    raise NotImplementedError

            cls._impl = _Impl
        return cls._impl(map_radius)


def route_centerline_local(scene, cur_idx: int, map_radius: float = 50.0) -> Optional[np.ndarray]:
    """PDM-Closed centerline as ``[M, 2]`` in the current ego frame, or ``None``."""
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap

    scenario = NavSimScenario(scene, os.environ["NUPLAN_MAPS_ROOT"], os.environ["NUPLAN_MAP_VERSION"])
    ego_state = scenario.initial_ego_state
    planner = _RouteOnlyPlanner.build(map_radius)
    planner._map_api = scenario.map_api
    planner._load_route_dicts(scenario.get_route_roadblock_ids())
    planner._route_roadblock_correction(ego_state)
    planner._drivable_area_map = PDMDrivableMap.from_simulation(planner._map_api, ego_state, map_radius)
    lane = planner._get_starting_lane(ego_state)
    if lane is None:
        return None
    states = planner._get_discrete_centerline(lane)
    if not states:
        return None
    g = np.asarray([[st.x, st.y] for st in states], dtype=np.float64)
    cur = np.asarray(scene.frames[cur_idx].ego_status.ego_pose, dtype=np.float64)
    c, s = math.cos(cur[2]), math.sin(cur[2])
    dx, dy = g[:, 0] - cur[0], g[:, 1] - cur[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def resample_from_projection(poly: np.ndarray, num: int, spacing: float):
    """Resample ``poly [M, 2]`` at ``spacing`` from the projection of the origin forward.

    Returns stations ``[num, 2]`` and ``valid [num]`` (0 beyond the end of the polyline).
    """
    poly = np.asarray(poly, dtype=np.float64)
    if poly.shape[0] < 2:
        return np.zeros((num, 2), np.float32), np.zeros(num, np.float32)
    seg = poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(seg, axis=-1)
    keep = seg_len > 1e-6
    if not keep.all():
        poly = np.concatenate([poly[:1], poly[1:][keep]], axis=0)
        seg = poly[1:] - poly[:-1]
        seg_len = np.linalg.norm(seg, axis=-1)
    if poly.shape[0] < 2:
        return np.zeros((num, 2), np.float32), np.zeros(num, np.float32)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    # projection of the origin
    rel = -poly[:-1]
    t = np.clip((rel * seg).sum(-1) / np.maximum(seg_len ** 2, 1e-12), 0.0, 1.0)
    proj = poly[:-1] + t[:, None] * seg
    d = np.linalg.norm(proj, axis=-1)
    k = int(np.argmin(d))
    s0 = cum[k] + t[k] * seg_len[k]
    targets = s0 + np.arange(num) * spacing
    valid = (targets <= cum[-1]).astype(np.float32)
    tc = np.clip(targets, 0.0, cum[-1])
    idx = np.clip(np.searchsorted(cum, tc, side="right") - 1, 0, len(seg_len) - 1)
    w = (tc - cum[idx]) / np.maximum(seg_len[idx], 1e-6)
    pts = poly[idx] + w[:, None] * seg[idx]
    return pts.astype(np.float32), valid


# ---------------------------------------------------------------------------- #
class FactTargetBuilder(AbstractTargetBuilder):
    """``fact_occ``, ``fact_occ_bev``, ``fact_sdf_bev``, ``fact_route``/``_valid``/``_ok``."""

    def __init__(self, config):
        self._config = config
        self._vocab = None

    def _open_vocab(self):
        """Lazily load the vocabulary (the builder is pickled into dataloader workers)."""
        if self._vocab is None:
            cfg = self._config
            paths = np.load(cfg.fact_path_file).astype(np.float64)              # [P, S+1, 2]
            speeds = np.maximum.accumulate(np.clip(np.load(cfg.fact_speed_file).astype(np.float64), 0, None), axis=1)
            self._vocab = (paths, speeds)
        return self._vocab

    def nearest_vocab(self, trajectory: np.ndarray):
        """Human ``[T, 3]`` -> (path index, speed index) under the same distances fact/losses.py uses."""
        cfg = self._config
        paths, speeds = self._open_vocab()
        S = paths.shape[1] - 1
        st_m = float(cfg.fact_station_m)
        pts = np.concatenate([np.zeros((1, 2)), trajectory[:, :2].astype(np.float64)], axis=0)
        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=-1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        length = cum[-1]
        last = np.array([1.0, 0.0])
        for t in range(len(seg_len)):
            if seg_len[t] > 1e-3:
                last = seg[t] / seg_len[t]
        tgt = np.arange(S + 1) * st_m
        tc = np.minimum(tgt, length)
        idx = np.clip(np.searchsorted(cum, tc, side="right") - 1, 0, len(seg_len) - 1)
        w = (tc - cum[idx]) / np.maximum(cum[idx + 1] - cum[idx], 1e-6)
        human = pts[idx] + w[:, None] * seg[idx] + np.maximum(tgt - length, 0)[:, None] * last
        n_use = int(min(max(np.ceil(length / st_m), 5), S))
        d_path = ((paths[:, 1:n_use + 1] - human[None, 1:n_use + 1]) ** 2).sum(-1).mean(-1)
        h_speed = cum[1:]
        d_speed = ((speeds - h_speed[None]) ** 2).mean(-1)
        return int(np.argmin(d_path)), int(np.argmin(d_speed))

    def get_unique_name(self) -> str:
        cfg = self._config
        return cache_key(
            "para_ssr_fact_targets",
            (
                ("extent", tuple(cfg.plan_map_extent)),
                ("res", cfg.plan_map_res),
                ("clip", cfg.plan_map_clip),
                ("pc_range", tuple(cfg.pc_range)),
                ("bev", (cfg.bev_h, cfg.bev_w)),
                ("fut_ts", cfg.fut_ts),
                ("route", (int(cfg.fact_route_stations), float(cfg.fact_route_spacing))),
                ("vocab", (str(cfg.fact_path_file).rsplit("/", 1)[-1], str(cfg.fact_speed_file).rsplit("/", 1)[-1])),
                ("version", "fact_v1"),
            ),
        )

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        cfg = self._config
        cur_idx = scene.scene_metadata.num_history_frames - 1
        extent, res = tuple(cfg.plan_map_extent), float(cfg.plan_map_res)
        T = int(cfg.fut_ts)
        x0, x1, y0, y1 = extent
        nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
        occ = np.zeros((T, nx, ny), dtype=np.uint8)
        occ_bev = np.zeros((T, cfg.bev_h, cfg.bev_w), dtype=np.uint8)
        for t in range(T):
            boxes = _future_boxes_in_current_frame(scene, cur_idx, t + 1)
            if boxes is None:
                break
            occ[t] = rasterize_field_grid(boxes, extent, res)
            occ_bev[t] = rasterize_bev_grid(boxes, cfg.pc_range, cfg.bev_h, cfg.bev_w)

        # SDF on the BEV grid, from the same polygon plan_map uses
        from navsim.agents.para_ssr.plan_map import DrivableAreaTargetBuilder, rasterize_sdf
        sdf = rasterize_sdf(DrivableAreaTargetBuilder(cfg).drivable_polygon(scene, cur_idx),
                            extent, res, float(cfg.plan_map_clip))
        sdf_bev = sdf_on_bev_grid(sdf, extent, res, cfg.pc_range, cfg.bev_h, cfg.bev_w)

        R, spacing = int(cfg.fact_route_stations), float(cfg.fact_route_spacing)
        route_ok = 0.0
        route = np.zeros((R, 2), np.float32)
        route_valid = np.zeros(R, np.float32)
        try:
            poly = route_centerline_local(scene, cur_idx)
            if poly is not None and poly.shape[0] >= 2:
                route, route_valid = resample_from_projection(poly, R, spacing)
                route_ok = 1.0 if route_valid.sum() >= 2 else 0.0
        except Exception as exc:  # a broken route must not kill a dataloader worker
            logger.warning("fact route centerline failed for %s: %r", scene.scene_metadata.initial_token, exc)
        human = np.asarray(scene.get_future_trajectory(num_trajectory_frames=T).poses, dtype=np.float32)
        gt_path, gt_speed = self.nearest_vocab(human)
        return {
            "fact_gt_path": torch.tensor(gt_path, dtype=torch.int64),
            "fact_gt_speed": torch.tensor(gt_speed, dtype=torch.int64),
            "fact_occ": torch.from_numpy(occ),
            "fact_occ_bev": torch.from_numpy(occ_bev),
            "fact_sdf_bev": torch.from_numpy(sdf_bev.astype(np.float32)).half(),
            "fact_route": torch.from_numpy(route),
            "fact_route_valid": torch.from_numpy(route_valid),
            "fact_route_ok": torch.tensor(route_ok, dtype=torch.float32),
        }
