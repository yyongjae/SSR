"""Target builder: navsim ``Scene`` -> planning / detection / motion / map GT.

Planning and detection targets come straight out of navsim.  The other two are
built here because navsim ships no builder for them:

**motion** -- future agent trajectories, recovered by matching ``track_tokens``
across the scene's future frames and expressing each future box centre in the
*current* ego frame.

**vector map** -- VAD-style polylines.  navsim only rasterises map geometry
(``transfuser_features._compute_bev_semantic_map``), but the underlying nuPlan
map API is fully vector, so the polylines are extracted directly:

===============  =========================================================
road             contour (exterior + holes) of the union of ``LANE`` and
                 ``INTERSECTION`` polygons -- what TransFuser rasterises as
                 its ``road`` class
walkway          ``WALKWAYS`` polygon rings
centerline       ``Lane`` / ``LaneConnector`` ``baseline_path`` (open)
crosswalk        ``SemanticMapLayer.CROSSWALK`` polygon exterior (closed)
===============  =========================================================

Earlier revisions used every lane and lane-connector *edge* as a "divider" and
the roadblock union as "boundary".  nuPlan has no divider labels, so that was an
interpretation of an undocumented ``boundary_type_fid`` (the TODO said as much);
the classes above are layers the map API serves as labelled objects, and they
match the ReSMap teacher this model is distilled from.

``SemanticMapLayer.BOUNDARIES`` is deliberately *not* used: nuPlan's
``get_proximal_map_objects`` only serves LANE, LANE_CONNECTOR, ROADBLOCK,
ROADBLOCK_CONNECTOR, STOP_LINE, CROSSWALK, INTERSECTION, WALKWAYS and
CARPARK_AREA, so lane boundaries have to be reached through the lane objects.

Equivalent orderings
--------------------
VAD matches a predicted polyline against every ordering of the GT polyline that
describes the same geometry and keeps the cheapest.  ``num_orders`` slots are
filled with canonical-direction cyclic shifts for closed shapes and with
forward/reversed repeats for open lines, matching the direction semantics of
``map_gt_shift_pts_pattern='v2'``.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Annotations, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


logger = logging.getLogger(__name__)

# nuPlan tracked-object names -> detection class index
DET_CLASS_NAMES: Tuple[str, ...] = (
    "vehicle",
    "pedestrian",
    "bicycle",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
)
DET_NAME_TO_INDEX = {name: i for i, name in enumerate(DET_CLASS_NAMES)}

# NAVSIM-convention map classes, shared with the ReSMap teacher.  nuPlan has no
# divider/boundary labels -- the old three were an interpretation of an
# undocumented boundary_type_fid (see the TODO that used to sit in
# _compute_map_targets).  These four are layers the map API serves as labelled
# classes: the ones TransFuser rasterises, plus CROSSWALK.
MAP_CLASS_NAMES: Tuple[str, ...] = ("road", "walkway", "centerline", "crosswalk")


def navsim_box_to_ssr(
    box: npt.NDArray[np.floating],
    velocity: npt.NDArray[np.floating],
) -> npt.NDArray[np.float32]:
    """Convert one NAVSIM annotation box to PARA-SSR's physical box code.

    NAVSIM stores ``(x_forward, y_left, z, length, width, height, heading)``.
    PARA-SSR follows the SECOND/VAD convention
    ``(x_right, y_forward, z, width, length, height, yaw, vx, vy)``.

    Keeping this conversion in one public helper prevents the training target
    builder and the held-out auxiliary evaluator from drifting apart.
    """

    box = np.asarray(box, dtype=np.float64).reshape(7)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    if velocity.size < 2:
        raise ValueError(
            f"NAVSIM velocity must have at least two components, got {velocity.shape}"
        )
    syaw_raw = -float(box[6]) - np.pi
    syaw = float(np.arctan2(np.sin(syaw_raw), np.cos(syaw_raw)))
    return np.asarray(
        [
            -box[1],
            box[0],
            box[2],
            box[4],
            box[3],
            box[5],
            syaw,
            -velocity[1],
            velocity[0],
        ],
        dtype=np.float32,
    )


def _geometry_local_coords(geometry, origin: StateSE2):
    """Shapely geometry from global frame into ``origin``'s local frame."""
    a, b = np.cos(origin.heading), np.sin(origin.heading)
    d, e = -np.sin(origin.heading), np.cos(origin.heading)
    translated = affinity.affine_transform(geometry, [1, 0, 0, 1, -origin.x, -origin.y])
    return affinity.affine_transform(translated, [a, b, d, e, 0, 0])


def _resample(points: npt.NDArray[np.float64], num_pts: int) -> npt.NDArray[np.float64]:
    """Resample a polyline to ``num_pts`` points at uniform arclength."""
    if len(points) < 2:
        return np.repeat(points[:1], num_pts, axis=0)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-6:
        return np.repeat(points[:1], num_pts, axis=0)
    targets = np.linspace(0.0, total, num_pts)
    out = np.empty((num_pts, 2), dtype=np.float64)
    out[:, 0] = np.interp(targets, cum, points[:, 0])
    out[:, 1] = np.interp(targets, cum, points[:, 1])
    return out


def _equivalent_orders(
    pts: npt.NDArray[np.float64],
    num_orders: int,
    closed: bool,
    num_pts: Optional[int] = None,
) -> npt.NDArray[np.float64]:
    """All orderings of ``pts`` that describe the same geometry, padded to ``num_orders``."""
    if num_orders < 1:
        raise ValueError(f"num_orders must be positive, got {num_orders}")

    output_num_pts = len(pts) if num_pts is None else num_pts
    orders: List[npt.NDArray[np.float64]] = []
    if closed:
        # A closed input includes both endpoints.  Rolling that duplicated endpoint
        # creates a jump through the interior and no longer represents the same
        # closed geometry.  VAD v2 instead shifts the unique cycle, closes it
        # again, and re-samples each shifted line.  It also preserves the
        # canonical polygon direction (closed shapes do not get reversed).
        is_closed = len(pts) >= 2 and np.allclose(
            pts[0], pts[-1], rtol=0.0, atol=1e-6
        )
        cycle = pts[:-1] if is_closed else pts
        if len(cycle) < 2:
            sampled = _resample(pts, output_num_pts)
            return np.repeat(sampled[None], num_orders, axis=0)

        # Use a deterministic, approximately uniform subset when a contour has
        # more vertices than target order slots.  Repeating is only necessary
        # for very small polygons with fewer unique starts than ``num_orders``.
        unique_order_count = min(len(cycle), num_orders)
        shift_indices = np.floor(
            np.arange(unique_order_count, dtype=np.float64)
            * len(cycle)
            / unique_order_count
        ).astype(np.int64)
        for i in range(num_orders):
            shift_idx = int(shift_indices[i % unique_order_count])
            shifted = np.roll(cycle, -shift_idx, axis=0)
            shifted_closed = np.concatenate([shifted, shifted[:1]], axis=0)
            orders.append(_resample(shifted_closed, output_num_pts))
    else:
        sampled = _resample(pts, output_num_pts)
        orders.append(sampled)
        orders.append(sampled[::-1])
    while len(orders) < num_orders:
        orders.append(orders[len(orders) % 2])
    return np.stack(orders[:num_orders])


class ParaSSRTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config, trajectory_sampling: TrajectorySampling):
        self._config = config
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "para_ssr_target"

    # ------------------------------------------------------------------ #
    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        cfg = self._config
        num_poses = self._trajectory_sampling.num_poses

        trajectory = np.asarray(
            scene.get_future_trajectory(num_trajectory_frames=num_poses).poses,
            dtype=np.float32,
        )  # [num_poses, 3] in the current ego frame
        # SSR regresses per-step OFFSETS, not absolute poses.
        offsets = np.diff(
            np.concatenate([np.zeros((1, 3), dtype=np.float32), trajectory]), axis=0
        )
        # wrap the heading increments
        offsets[:, 2] = np.arctan2(np.sin(offsets[:, 2]), np.cos(offsets[:, 2]))

        cur_idx = scene.scene_metadata.num_history_frames - 1
        cur_frame = scene.frames[cur_idx]
        command = np.asarray(cur_frame.ego_status.driving_command, dtype=np.float32)

        targets: Dict[str, torch.Tensor] = {
            "trajectory": torch.tensor(trajectory),
            "trajectory_offsets": torch.tensor(offsets),
            "trajectory_mask": torch.ones(num_poses, dtype=torch.float32),
            "command": torch.tensor(command),
        }

        if cfg.use_det_motion_head:
            targets.update(self._compute_agent_targets(scene, cur_idx))
        if cfg.use_map_head:
            targets.update(self._compute_map_targets(scene, cur_idx))
        return targets

    # ------------------------------------------------------------------ #
    def _compute_agent_targets(self, scene: Scene, cur_idx: int) -> Dict[str, torch.Tensor]:
        cfg = self._config
        max_agents = cfg.max_agents
        fut_ts = cfg.fut_ts

        ann: Annotations = scene.frames[cur_idx].annotations
        boxes = np.asarray(ann.boxes, dtype=np.float32).reshape(-1, 7)
        vel = np.asarray(ann.velocity_3d, dtype=np.float32).reshape(-1, 3)
        names = list(ann.names)
        tracks = list(ann.track_tokens)

        x0, y0, x1, y1 = cfg.pc_range[0], cfg.pc_range[1], cfg.pc_range[3], cfg.pc_range[4]

        gt_boxes = np.zeros((max_agents, 9), dtype=np.float32)
        gt_labels = np.zeros(max_agents, dtype=np.int64)
        gt_valid = np.zeros(max_agents, dtype=bool)
        gt_fut = np.zeros((max_agents, fut_ts, 2), dtype=np.float32)
        gt_fut_mask = np.zeros((max_agents, fut_ts), dtype=np.float32)

        # future ego poses (global) for re-expressing future agent centres
        ego_poses = [
            np.asarray(f.ego_status.ego_pose, dtype=np.float64) for f in scene.frames
        ]
        cur_pose = ego_poses[cur_idx]

        candidates: List[int] = []
        for i, (box, name) in enumerate(zip(boxes, names)):
            if name not in DET_NAME_TO_INDEX:
                continue
            # navsim boxes are (x fwd, y left); SSR BEV is (x right, y fwd)
            sx, sy = -box[1], box[0]
            if not (x0 <= sx <= x1 and y0 <= sy <= y1):
                continue
            candidates.append(i)

        # NAVSIM's annotation order is not a relevance or distance order.  A
        # source-first cap can therefore discard a nearby agent while retaining
        # a farther one.  Stable sorting keeps source order only for exact ties.
        if candidates:
            candidate_idx = np.asarray(candidates, dtype=np.int64)
            candidate_xy = boxes[candidate_idx, :2]
            distance_sq = np.square(candidate_xy).sum(axis=-1)
            nearest_order = np.argsort(distance_sq, kind="stable")[:max_agents]
            kept = candidate_idx[nearest_order].tolist()
        else:
            kept = []

        for slot, i in enumerate(kept):
            box = boxes[i]
            gt_boxes[slot] = navsim_box_to_ssr(box, vel[i])
            gt_labels[slot] = DET_NAME_TO_INDEX[names[i]]
            gt_valid[slot] = True

            traj, mask = self._track_future(
                scene, tracks[i], cur_idx, cur_pose, ego_poses, fut_ts
            )
            gt_fut[slot] = traj
            gt_fut_mask[slot] = mask

        return {
            "gt_boxes": torch.tensor(gt_boxes),
            "gt_labels": torch.tensor(gt_labels),
            "gt_valid": torch.tensor(gt_valid),
            "gt_fut_trajs": torch.tensor(gt_fut),
            "gt_fut_masks": torch.tensor(gt_fut_mask),
        }

    def compute_detection_evaluation_targets(
        self, scene: Scene
    ) -> Dict[str, torch.Tensor]:
        """Return uncapped current-frame detection GT for held-out mAP.

        Training retains only the nearest ``max_agents`` objects because it
        needs fixed-size tensors.  Evaluation must not inherit that cap: an
        omitted annotation would turn a geometrically correct prediction into
        a false positive.  Filtering and coordinates otherwise exactly match
        :meth:`_compute_agent_targets`.
        """

        cur_idx = scene.scene_metadata.num_history_frames - 1
        ann: Annotations = scene.frames[cur_idx].annotations
        boxes = np.asarray(ann.boxes, dtype=np.float32).reshape(-1, 7)
        velocities = np.asarray(ann.velocity_3d, dtype=np.float32).reshape(-1, 3)
        names = list(ann.names)
        if not (len(boxes) == len(velocities) == len(names)):
            raise ValueError(
                "NAVSIM annotation boxes/names/velocities count mismatch for "
                f"token={scene.frames[cur_idx].token!r}: "
                f"{len(boxes)}/{len(names)}/{len(velocities)}"
            )
        x0, y0, x1, y1 = (
            self._config.pc_range[0],
            self._config.pc_range[1],
            self._config.pc_range[3],
            self._config.pc_range[4],
        )

        eval_boxes: List[npt.NDArray[np.float32]] = []
        eval_labels: List[int] = []
        for box, velocity, name in zip(boxes, velocities, names):
            if name not in DET_NAME_TO_INDEX:
                continue
            converted = navsim_box_to_ssr(box, velocity)
            sx, sy = float(converted[0]), float(converted[1])
            if not (x0 <= sx <= x1 and y0 <= sy <= y1):
                continue
            if not np.isfinite(converted).all() or np.any(converted[3:6] <= 0):
                raise ValueError(
                    "non-finite or non-positive NAVSIM detection GT for "
                    f"token={scene.frames[cur_idx].token!r}, class={name!r}"
                )
            eval_boxes.append(converted)
            eval_labels.append(DET_NAME_TO_INDEX[name])

        if eval_boxes:
            box_array = np.stack(eval_boxes).astype(np.float32, copy=False)
        else:
            box_array = np.zeros((0, 9), dtype=np.float32)
        return {
            "gt_boxes": torch.from_numpy(box_array),
            "gt_labels": torch.tensor(eval_labels, dtype=torch.long),
        }

    def _track_future(
        self,
        scene: Scene,
        track_token: str,
        cur_idx: int,
        cur_pose: npt.NDArray[np.float64],
        ego_poses: List[npt.NDArray[np.float64]],
        fut_ts: int,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Future centres of one agent, as per-step offsets in the current ego frame."""
        traj = np.zeros((fut_ts, 2), dtype=np.float32)
        mask = np.zeros(fut_ts, dtype=np.float32)

        c, s = np.cos(cur_pose[2]), np.sin(cur_pose[2])
        prev = None
        # the agent's current centre, in the current ego frame (SSR axes)
        ann = scene.frames[cur_idx].annotations
        try:
            j = list(ann.track_tokens).index(track_token)
        except ValueError:
            return traj, mask
        b = np.asarray(ann.boxes, dtype=np.float64)[j]
        prev = np.array([-b[1], b[0]], dtype=np.float64)

        for k in range(fut_ts):
            f_idx = cur_idx + 1 + k
            if f_idx >= len(scene.frames):
                break
            fann = scene.frames[f_idx].annotations
            ftokens = list(fann.track_tokens)
            if track_token not in ftokens:
                break
            fb = np.asarray(fann.boxes, dtype=np.float64)[ftokens.index(track_token)]
            # future-frame local -> global
            fp = ego_poses[f_idx]
            fc, fs = np.cos(fp[2]), np.sin(fp[2])
            gx = fc * fb[0] - fs * fb[1] + fp[0]
            gy = fs * fb[0] + fc * fb[1] + fp[1]
            # global -> current ego local (navsim axes), then to SSR axes
            dx, dy = gx - cur_pose[0], gy - cur_pose[1]
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            cur = np.array([-ly, lx], dtype=np.float64)
            traj[k] = (cur - prev).astype(np.float32)
            mask[k] = 1.0
            prev = cur

        return traj, mask

    # ------------------------------------------------------------------ #
    def _compute_map_targets(
        self, scene: Scene, cur_idx: int, *, uncapped: bool = False
    ) -> Dict[str, torch.Tensor]:
        cfg = self._config
        num_pts = cfg.map_num_pts_per_vec
        num_orders = cfg.map_num_orders

        pose = scene.frames[cur_idx].ego_status.ego_pose
        origin = StateSE2(float(pose[0]), float(pose[1]), float(pose[2]))
        map_api: AbstractMap = scene.map_api
        radius = float(max(abs(v) for v in cfg.pc_range[:2] + cfg.pc_range[3:5])) * 1.5
        context = (
            f"token={scene.frames[cur_idx].token!r}, "
            f"map={scene.scene_metadata.map_name!r}"
        )

        polylines: List[Tuple[int, npt.NDArray[np.float64]]] = []

        layers = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CROSSWALK,
        ]
        try:
            objects = map_api.get_proximal_map_objects(
                point=origin.point, radius=radius, layers=layers
            )
        except Exception as exc:
            # Empty GT silently turns every map query into background.  In
            # particular, map-schema/version mismatches must stop the run rather
            # than train for an epoch before somebody notices an empty map head.
            logger.warning("Map query failed for %s: %s", context, exc, exc_info=True)
            raise RuntimeError(f"Map query failed for {context}") from exc

        if not isinstance(objects, Mapping):
            error = TypeError(
                "get_proximal_map_objects must return a mapping, "
                f"got {type(objects).__name__}"
            )
            logger.warning("Invalid map-query result for %s: %s", context, error)
            raise RuntimeError(f"Invalid map-query result for {context}") from error
        missing_layers = [layer for layer in layers if layer not in objects]
        if missing_layers:
            names = [getattr(layer, "name", str(layer)) for layer in missing_layers]
            error = KeyError(f"map query omitted requested layers: {names}")
            logger.warning("Invalid map-query result for %s: %s", context, error)
            raise RuntimeError(f"Invalid map-query result for {context}") from error

        def _polygon_of(obj, layer):
            try:
                return obj.polygon
            except Exception as exc:
                logger.warning("Polygon extraction failed for %s, layer=%s, object=%r: %s",
                               context, getattr(layer, "name", str(layer)),
                               getattr(obj, "id", None), exc, exc_info=True)
                raise RuntimeError(f"Polygon extraction failed for {context}") from exc

        def _rings(poly, label):
            """Exterior and holes of one polygon as closed polylines."""
            if poly is None or poly.is_empty:
                return
            parts = [poly] if poly.geom_type == "Polygon" else [
                g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
            for g in parts:
                polylines.append((label, np.asarray(g.exterior.coords, dtype=np.float64)))
                for interior in g.interiors:
                    polylines.append((label, np.asarray(interior.coords, dtype=np.float64)))

        # 0 road: union of LANE + INTERSECTION so adjacent lanes leave no seams.
        road_polygons = [
            _polygon_of(obj, layer)
            for layer in (SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION)
            for obj in objects.get(layer, [])
        ]
        road_polygons = [p for p in road_polygons if p is not None and not p.is_empty]
        if road_polygons:
            try:
                _rings(unary_union(road_polygons), 0)
            except Exception as exc:
                logger.warning("Road polygon union failed for %s: %s", context, exc, exc_info=True)
                raise RuntimeError(f"Road polygon union failed for {context}") from exc

        # 1 walkway
        for obj in objects.get(SemanticMapLayer.WALKWAYS, []):
            _rings(_polygon_of(obj, SemanticMapLayer.WALKWAYS), 1)

        # 2 centerline: baseline paths of LANE and LANE_CONNECTOR (open)
        for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
            for obj in objects.get(layer, []):
                try:
                    ls: LineString = obj.baseline_path.linestring
                except Exception as exc:
                    logger.warning("Baseline path extraction failed for %s, object=%r: %s",
                                   context, getattr(obj, "id", None), exc, exc_info=True)
                    raise RuntimeError(f"Baseline path extraction failed for {context}") from exc
                polylines.append((2, np.asarray(ls.coords, dtype=np.float64)))

        # 3 crosswalk (closed)
        for obj in objects.get(SemanticMapLayer.CROSSWALK, []):
            _rings(_polygon_of(obj, SemanticMapLayer.CROSSWALK), 3)

        x0, y0, x1, y1 = cfg.pc_range[0], cfg.pc_range[1], cfg.pc_range[3], cfg.pc_range[4]
        patch = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

        # Clip and resample everything first, bucketed by class.
        # Filling slots in source order would be wrong: LANE + LANE_CONNECTOR
        # contribute ~34 objects x 2 boundaries per scene against a median of
        # 1 crosswalk and 4 roadblocks, so dividers alone exhaust ``max_vec``
        # and the other two classes never reach the GT tensor at all.
        buckets: Dict[int, List[npt.NDArray[np.float32]]] = {
            i: [] for i in range(len(MAP_CLASS_NAMES))
        }
        for label, coords in polylines:
            if len(coords) < 2:
                continue
            geom = LineString(coords[:, :2])
            local = _geometry_local_coords(geom, origin)
            arr = np.asarray(local.coords, dtype=np.float64)[:, :2]
            # navsim ego axes -> SSR BEV axes
            arr = np.stack([-arr[:, 1], arr[:, 0]], axis=-1)

            try:
                clipped = LineString(arr).intersection(patch)
            except Exception as exc:
                logger.warning(
                    "Map geometry clipping failed for %s, class=%s: %s",
                    context,
                    MAP_CLASS_NAMES[label],
                    exc,
                    exc_info=True,
                )
                continue
            if clipped.is_empty:
                continue

            pieces = []
            if clipped.geom_type == "LineString":
                pieces = [np.asarray(clipped.coords, dtype=np.float64)]
            elif clipped.geom_type in ("MultiLineString", "GeometryCollection"):
                for g in clipped.geoms:
                    if g.geom_type == "LineString" and len(g.coords) >= 2:
                        pieces.append(np.asarray(g.coords, dtype=np.float64))

            for piece in pieces:
                if len(piece) < 2 or LineString(piece).length < cfg.map_min_length:
                    continue
                # A closed source contour becomes open when the BEV patch cuts
                # it.  Equivalent-order generation must use the clipped
                # instance, not the source polygon's old ``closed`` flag.
                piece_closed = bool(
                    len(piece) >= 3
                    and np.allclose(piece[0, :2], piece[-1, :2], rtol=0.0, atol=1e-6)
                )
                orders = _equivalent_orders(
                    piece[:, :2], num_orders, piece_closed, num_pts=num_pts
                )
                # normalise to [0, 1] over the BEV extent
                orders[..., 0] = (orders[..., 0] - x0) / (x1 - x0)
                orders[..., 1] = (orders[..., 1] - y0) / (y1 - y0)
                buckets[label].append(orders.astype(np.float32))

        max_vec = (
            sum(len(instances) for instances in buckets.values())
            if uncapped
            else cfg.map_max_vec
        )
        gt_pts = np.zeros((max_vec, num_orders, num_pts, 2), dtype=np.float32)
        gt_labels = np.zeros(max_vec, dtype=np.int64)
        gt_valid = np.zeros(max_vec, dtype=bool)

        # round-robin across classes so every present class is represented
        slot = 0
        cursors = {label: 0 for label in buckets}
        while slot < max_vec:
            progressed = False
            for label in sorted(buckets):
                items = buckets[label]
                c = cursors[label]
                if c >= len(items):
                    continue
                gt_pts[slot] = items[c]
                gt_labels[slot] = label
                gt_valid[slot] = True
                cursors[label] = c + 1
                slot += 1
                progressed = True
                if slot >= max_vec:
                    break
            if not progressed:
                break

        return {
            "gt_map_pts": torch.tensor(gt_pts),
            "gt_map_labels": torch.tensor(gt_labels),
            "gt_map_valid": torch.tensor(gt_valid),
        }

    def compute_map_evaluation_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Return every in-patch vector-map instance for held-out mAP.

        The training tensor is class-balanced and capped at ``map_max_vec``.
        Chamfer mAP instead uses the complete geometry produced by the exact
        same map extraction and coordinate conversion.  The returned arrays
        are variable-length and contain no padding.
        """

        cur_idx = scene.scene_metadata.num_history_frames - 1
        result = self._compute_map_targets(scene, cur_idx, uncapped=True)
        if not bool(result["gt_map_valid"].all()):
            raise RuntimeError("uncapped map evaluation targets unexpectedly contain padding")
        return {
            "gt_map_pts": result["gt_map_pts"],
            "gt_map_labels": result["gt_map_labels"],
        }
