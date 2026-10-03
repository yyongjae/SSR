# H7 COPY of report/perception_reliability/pdm_attr/rescore_attr.py with paths parameterised by env
# (H7_OUT, H7_TRAJ_PKL, H7_TOKENS); scoring / recording logic unchanged.
#!/usr/bin/env python
"""PDM failure attribution table for PARA-SSR interaction_final (CPU re-scoring, no GPU).

What it does, per navtest token
  1. Re-scores the dumped model trajectory with the *official* ``navsim.evaluate.pdm_score.pdm_score``
     (unchanged function; same PDMSimulator / PDMScorer config as the archived eval hydra config).
     The simulator and scorer are subclassed only to *record* internals:
       - RecSimulator keeps the reference states (linearly interpolated prediction, 0.1 s) and the
         LQR+bicycle simulated states.
       - RecScorer calls the original ``_calculate_no_at_fault_collision`` / ``_calculate_ttc`` first
         (official values) and then re-runs a verbatim copy of their loops that additionally records
         the track token / time idx / collision type of every event.  The copy's scores are asserted
         to equal the official scores (flag ``rec_nc_consistent`` / ``rec_ttc_consistent``).
     pdm_score() scores proposals [PDM-Closed, prediction]; the submitted trajectory is proposal
     index **1** (pred_idx = 1 in navsim/evaluate/pdm_score.py), proposal 0 is PDM-Closed.
  2. Cause objects of NC / TTC (earliest event) are expressed at t=0 in the aux "record frame"
     (x_right, y_forward, origin = ego rear axle) and linked to the aux record ``det_gt_boxes``
     (nearest centre <= 1.5 m) or classified why they are not in the GT (behind / far / side / fov / new).
  3. DAC: first off-road time idx, ego pose / corners (record frame), penetration depth, nearest PDM
     drivable-area boundary point, exit side, and whether the exit is already visible in the reference
     (pre-LQR) trajectory.
  4. The expert (human) trajectory is scored separately as [PDM-Closed, human] (control: does the
     expert also fail?).  This does not touch the prediction's scores.

Run:  see run.sh in this directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import lzma
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))

OUT = Path(os.environ["H7_OUT"])  # H7 copy: output dir from env
MC_ROOT = ROOT / "data/exp/metric_cache"
EVAL = ROOT / "work_dirs/eval"
TABLE = ROOT / "report/head_ablation_scenes/table.npz"

W_E = 1.15  # ego half width [m] (common definition)
ROI_X = 32.0
ROI_Y = 32.0
FOV_HALF_DEG = 80.0
GT_LINK_M = 1.5

# ------------------------------------------------------------------------------------------------
# globals initialised per worker
G = {}


def _init_worker(arm: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer

    cfg = OmegaConf.load(EVAL / "para_ssr_interaction_final" / "code/hydra/config.yaml")  # H7: same scorer config
    sub = OmegaConf.create(
        {"proposal_sampling": cfg.proposal_sampling, "simulator": cfg.simulator, "scorer": cfg.scorer}
    )
    rec_sim_cls, rec_scorer_cls = _make_classes(PDMSimulator, PDMScorer)
    base_sim = instantiate(sub.simulator)
    base_scorer = instantiate(sub.scorer)
    G["sim"] = rec_sim_cls(base_sim.proposal_sampling)
    G["scorer"] = rec_scorer_cls(base_scorer.proposal_sampling, base_scorer._config, base_scorer._vehicle_parameters)
    G["sim_h"] = rec_sim_cls(base_sim.proposal_sampling)
    G["scorer_h"] = rec_scorer_cls(base_scorer.proposal_sampling, base_scorer._config, base_scorer._vehicle_parameters)
    G["scorer_ref"] = PDMScorer(base_scorer.proposal_sampling, base_scorer._config, base_scorer._vehicle_parameters)
    G["scorer_cfg"] = repr(base_scorer._config)

    traj = pickle.load(open(os.environ["H7_TRAJ_PKL"], "rb"))  # H7: trajectory pkl from env
    G["traj"] = traj["trajectories"]
    t = np.load(TABLE, allow_pickle=True)
    G["human"] = {tok: (h, v) for tok, h, v in zip(t["tokens"], t["human"], t["human_valid"])}
    G["aux_dir"] = EVAL / "para_ssr_interaction_final_aux/records"  # H7: identical V3 GT (only GT is read)
    G["arm"] = arm


def _make_classes(PDMSimulator, PDMScorer):
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.metrics.utils.collision_utils import CollisionType
    from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
    from shapely import creation
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        BBCoordsIndex, EgoAreaIndex, MultiMetricIndex, StateIndex, WeightedMetricIndex,
    )

    class RecSimulator(PDMSimulator):
        def simulate_proposals(self, states, initial_ego_state):
            self.last_ref = np.array(states, dtype=np.float64, copy=True)
            out = super().simulate_proposals(states, initial_ego_state)
            self.last_sim = np.array(out, dtype=np.float64, copy=True)
            return out

    class RecScorer(PDMScorer):
        """Records NC / TTC events; official metric values are computed by the parent methods."""

        def _calculate_no_at_fault_collision(self) -> None:
            super()._calculate_no_at_fault_collision()
            official = self._multi_metrics[MultiMetricIndex.NO_COLLISION].copy()
            events = []
            no_collision_scores = np.ones(self._num_proposals, dtype=np.float64)
            proposal_collided_track_ids = {
                p: copy.deepcopy(self._observation.collided_track_ids) for p in range(self._num_proposals)
            }
            for time_idx in range(self.proposal_sampling.num_poses + 1):
                ego_polygons = self._ego_polygons[:, time_idx]
                intersecting = self._observation[time_idx].query(ego_polygons, predicate="intersects")
                if len(intersecting) == 0:
                    continue
                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._observation[time_idx].tokens[geometry_idx]
                    if (self._observation.red_light_token in token) or (
                        token in proposal_collided_track_ids[proposal_idx]
                    ):
                        continue
                    multi_or_nondrivable = bool(
                        self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                        or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                    )
                    tracked_object = self._observation.unique_objects[token]
                    collision_type = get_collision_type(
                        self._states[proposal_idx, time_idx],
                        self._ego_polygons[proposal_idx, time_idx],
                        tracked_object,
                        self._observation[time_idx][token],
                    )
                    front_or_stopped = collision_type in [
                        CollisionType.ACTIVE_FRONT_COLLISION,
                        CollisionType.STOPPED_TRACK_COLLISION,
                    ]
                    lateral = collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                    at_fault = bool(front_or_stopped or (multi_or_nondrivable and lateral))
                    value = np.nan
                    if at_fault:
                        value = 0.0 if tracked_object.tracked_object_type in AGENT_TYPES else 0.5
                        no_collision_scores[proposal_idx] = min(no_collision_scores[proposal_idx], value)
                    else:
                        proposal_collided_track_ids[proposal_idx].append(token)
                    events.append(
                        dict(
                            proposal=int(proposal_idx), time_idx=int(time_idx), track=token,
                            collision_type=collision_type.name, obj_type=tracked_object.tracked_object_type.name,
                            at_fault=at_fault, value=float(value), multi_or_nondrivable=multi_or_nondrivable,
                        )
                    )
            self.rec_nc_events = events
            self.rec_nc_consistent = bool(np.array_equal(no_collision_scores, official))

        def _calculate_ttc(self):
            super()._calculate_ttc()
            official = self._weighted_metrics[WeightedMetricIndex.TTC].copy()
            events = []
            ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
            temp_collided_track_ids = {
                p: copy.deepcopy(self._observation.collided_track_ids) for p in range(self._num_proposals)
            }
            future_time_idcs = np.arange(0, 10, 3)
            n_future_steps = len(future_time_idcs)
            coords_exterior = self._ego_coords.copy()
            coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[:, :, BBCoordsIndex.FRONT_LEFT, :]
            coords_exterior_time_steps = np.repeat(coords_exterior[:, :, None], n_future_steps, axis=2)
            speeds = np.hypot(self._states[..., StateIndex.VELOCITY_X], self._states[..., StateIndex.VELOCITY_Y])
            dxy_per_s = np.stack(
                [np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                 np.sin(self._states[..., StateIndex.HEADING]) * speeds], axis=-1,
            )
            for idx, future_time_idx in enumerate(future_time_idcs):
                delta_t = float(future_time_idx) * self.proposal_sampling.interval_length
                coords_exterior_time_steps[:, :, idx] = coords_exterior_time_steps[:, :, idx] + dxy_per_s[:, :, None] * delta_t
            polygons = creation.polygons(coords_exterior_time_steps)
            for time_idx in range(self.proposal_sampling.num_poses + 1):
                for step_idx, future_time_idx in enumerate(future_time_idcs):
                    current_time_idx = time_idx + future_time_idx
                    intersecting = self._observation[current_time_idx].query(
                        polygons[:, time_idx, step_idx], predicate="intersects"
                    )
                    if len(intersecting) == 0:
                        continue
                    for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                        token = self._observation[current_time_idx].tokens[geometry_idx]
                        if (
                            (self._observation.red_light_token in token)
                            or (token in temp_collided_track_ids[proposal_idx])
                            or (speeds[proposal_idx, time_idx] < self._config.stopped_speed_threshold)
                        ):
                            continue
                        multi_or_nondrivable = bool(
                            self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                            or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                        )
                        ego_rear_axle = StateSE2(*self._states[proposal_idx, time_idx, StateIndex.STATE_SE2])
                        centroid = self._observation[current_time_idx][token].centroid
                        track_heading = self._observation.unique_objects[token].box.center.heading
                        track_state = StateSE2(centroid.x, centroid.y, track_heading)
                        ahead = is_agent_ahead(ego_rear_axle, track_state)
                        in_intersection = self._drivable_area_map.is_in_layer(
                            ego_rear_axle.point, layer=SemanticMapLayer.INTERSECTION
                        )
                        if ahead or ((multi_or_nondrivable or in_intersection) and not is_agent_behind(ego_rear_axle, track_state)):
                            ttc_scores[proposal_idx] = min(ttc_scores[proposal_idx], 0.0)
                            events.append(
                                dict(
                                    proposal=int(proposal_idx), time_idx=int(time_idx),
                                    future_idx=int(future_time_idx), track=token,
                                    obj_type=self._observation.unique_objects[token].tracked_object_type.name,
                                    ahead=bool(ahead), in_intersection=bool(in_intersection),
                                    multi_or_nondrivable=multi_or_nondrivable,
                                )
                            )
                        else:
                            temp_collided_track_ids[proposal_idx].append(token)
            self.rec_ttc_events = events
            self.rec_ttc_consistent = bool(np.array_equal(ttc_scores, official))

    return RecSimulator, RecScorer


# ------------------------------------------------------------------------------------------------
# geometry helpers


def g2rec(xy, ra):
    """global (x, y) -> record frame (x_right, y_forward) about the t=0 ego rear axle ``ra`` (StateSE2)."""
    xy = np.asarray(xy, dtype=np.float64)
    dx, dy = xy[..., 0] - ra.x, xy[..., 1] - ra.y
    c, s = np.cos(ra.heading), np.sin(ra.heading)
    x_fwd = c * dx + s * dy
    y_left = -s * dx + c * dy
    return np.stack([-y_left, x_fwd], axis=-1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def expert_path_rec(human):
    """NAVSIM ego (x fwd, y left) (8,3) -> record-frame polyline with origin prepended."""
    h = np.asarray(human, dtype=np.float64)
    pts = np.concatenate([[[0.0, 0.0]], np.stack([-h[:, 1], h[:, 0]], -1)], 0)
    return pts


def polyline_len(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def signed_offset(path, p):
    """Signed lateral offset of point p from polyline ``path`` (record frame, + = right of the path
    direction), and arc length of the projection (unclamped beyond the ends along end segments)."""
    p = np.asarray(p, dtype=np.float64)
    seg = np.diff(path, axis=0)
    L = np.linalg.norm(seg, axis=1)
    keep = L > 1e-6
    if not keep.any():
        return np.nan, np.nan
    path_k = np.concatenate([path[:1], path[1:][keep]], 0)
    seg = np.diff(path_k, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    best = (np.inf, 0.0, 0.0)
    n = len(seg)
    for i in range(n):
        t = np.dot(p - path_k[i], seg[i]) / L[i] ** 2
        lo = -np.inf if i == 0 else 0.0
        hi = np.inf if i == n - 1 else 1.0
        tc = min(max(t, lo), hi)
        q = path_k[i] + tc * seg[i]
        d = np.linalg.norm(p - q)
        if d < best[0]:
            v = p - q
            cross = seg[i][0] * v[1] - seg[i][1] * v[0]
            best = (d, -np.sign(cross) * d, cum[i] + tc * L[i])
    return float(best[1]), float(best[2])


def tier_of(poly_rec, path, path_len):
    """Common-definition tier of a box footprint polygon (record frame) w.r.t. a path polyline."""
    import shapely.geometry as sg
    if path_len < 1.0:
        line = sg.Point(0.0, 0.0)
    else:
        line = sg.LineString(path)
    d_lat = float(poly_rec.distance(line))
    s = float(line.project(poly_rec.centroid)) if path_len >= 1.0 else 0.0
    if d_lat <= W_E + 0.5 and s <= path_len + 5.0:
        tier = "T0"
    elif d_lat <= W_E + 3.0:
        tier = "T1"
    else:
        tier = "T2"
    return tier, d_lat, s


def vis_class(xr, yf):
    """Why an object at record-frame centre (xr, yf) is or is not inside the aux det GT population."""
    if yf < 0.0:
        return "behind"
    if abs(xr) > ROI_X:
        return "side"
    if yf > ROI_Y:
        return "far"
    bearing = np.degrees(np.arctan2(xr, yf))
    if abs(bearing) > FOV_HALF_DEG + 1e-4:
        return "fov"
    return "roi"


# ------------------------------------------------------------------------------------------------


def cause_object_info(prefix, token, event_time_idx, ego_time_idx, mc, ra, gt_boxes, gt_labels, ep, ep_len, own, own_len,
                      ref_polys, sim_states_pred):
    """Describe one cause track (NC / TTC) in the t=0 record frame and link it to aux det GT."""
    import shapely.geometry as sg
    from shapely import affinity
    obs = mc.observation
    uo = obs.unique_objects[token]
    r = {}
    r[f"{prefix}_obj_type"] = uo.tracked_object_type.name
    r[f"{prefix}_obj_det_cls"] = int(uo.tracked_object_type.value)
    grp = {0: "vehicle", 1: "VRU", 2: "VRU"}.get(int(uo.tracked_object_type.value), "static")
    r[f"{prefix}_obj_group"] = grp
    r[f"{prefix}_obj_length"] = float(uo.box.length)
    r[f"{prefix}_obj_width"] = float(uo.box.width)
    present0 = token in obs[0].token_to_idx
    first_idx = 0
    if not present0:
        first_idx = -1
        for k in range(len(obs._occupancy_maps)):
            if token in obs[k].token_to_idx:
                first_idx = k
                break
    r[f"{prefix}_obj_present_t0"] = bool(present0)
    r[f"{prefix}_obj_first_idx"] = int(first_idx)
    poly_g = obs[max(first_idx, 0)][token]
    c = poly_g.centroid
    xr, yf = g2rec([c.x, c.y], ra)
    r[f"{prefix}_obj_xr"] = float(xr)
    r[f"{prefix}_obj_yf"] = float(yf)
    r[f"{prefix}_obj_dist"] = float(np.hypot(xr, yf))
    r[f"{prefix}_obj_bearing_deg"] = float(np.degrees(np.arctan2(xr, yf)))
    r[f"{prefix}_obj_heading_rel"] = float(wrap(uo.box.center.heading - ra.heading))  # NAVSIM conv. (CCW from ego fwd), at first appearance
    vel = getattr(uo, "velocity", None)
    if vel is not None:
        c_, s_ = np.cos(ra.heading), np.sin(ra.heading)
        vx_f = c_ * vel.x + s_ * vel.y
        vy_l = -s_ * vel.x + c_ * vel.y
        r[f"{prefix}_obj_vxr"] = float(-vy_l)
        r[f"{prefix}_obj_vyf"] = float(vx_f)
        r[f"{prefix}_obj_speed"] = float(np.hypot(vel.x, vel.y))
    else:
        r[f"{prefix}_obj_vxr"] = r[f"{prefix}_obj_vyf"] = np.nan
        r[f"{prefix}_obj_speed"] = 0.0
    # observed displacement over the horizon (tracks are GT-interpolated)
    idxs = [k for k in range(41) if token in obs[k].token_to_idx]
    if len(idxs) >= 2:
        a = obs[idxs[0]][token].centroid
        b = obs[idxs[-1]][token].centroid
        r[f"{prefix}_obj_mean_speed_obs"] = float(np.hypot(b.x - a.x, b.y - a.y) / ((idxs[-1] - idxs[0]) * 0.1))
    else:
        r[f"{prefix}_obj_mean_speed_obs"] = 0.0
    r[f"{prefix}_obj_moving"] = bool(max(r[f"{prefix}_obj_speed"], r[f"{prefix}_obj_mean_speed_obs"]) > 0.5)
    # position at event time
    if token in obs[event_time_idx].token_to_idx:
        ce = obs[event_time_idx][token].centroid
        xe, ye = g2rec([ce.x, ce.y], ra)
    else:
        xe = ye = np.nan
    r[f"{prefix}_obj_xr_event"] = float(xe)
    r[f"{prefix}_obj_yf_event"] = float(ye)
    exr, eyf = g2rec(sim_states_pred[ego_time_idx, :2], ra)
    r[f"{prefix}_ego_xr_event"] = float(exr)
    r[f"{prefix}_ego_yf_event"] = float(eyf)
    # visibility / GT link (at t=0 if present; objects appearing later are "new")
    vc = vis_class(xr, yf) if present0 else "new"
    gt_idx, gt_dist, gt_lab = -1, np.nan, -1
    if present0 and len(gt_boxes):
        d = np.hypot(gt_boxes[:, 0] - xr, gt_boxes[:, 1] - yf)
        j = int(np.argmin(d))
        gt_dist = float(d[j])
        if d[j] <= GT_LINK_M:
            gt_idx, gt_lab = j, int(gt_labels[j])
    if present0:
        if gt_idx >= 0:
            vc = "in_gt" if vc == "roi" else f"in_gt_{vc}"  # in_gt_<x>: linked although centre just outside
        elif vc == "roi":
            vc = "roi_no_ann"  # inside ROI/FOV but no aux GT box within 1.5 m (absent from NAVSIM annotations)
    r[f"{prefix}_vis"] = vc
    r[f"{prefix}_vis_at_first"] = vis_class(xr, yf)  # ROI class at t=0 or, for new tracks, at first appearance
    r[f"{prefix}_gt_idx"] = gt_idx
    r[f"{prefix}_gt_dist"] = gt_dist
    r[f"{prefix}_gt_label"] = gt_lab
    # tier w.r.t. expert path and own (predicted) path, footprint at first appearance
    coords = np.asarray(poly_g.exterior.coords)
    poly_rec = sg.Polygon(g2rec(coords, ra))
    if ep is not None:
        t, dl, s = tier_of(poly_rec, ep, ep_len)
        r[f"{prefix}_tier_expert"], r[f"{prefix}_dlat_expert"], r[f"{prefix}_s_expert"] = t, dl, s
    else:
        r[f"{prefix}_tier_expert"], r[f"{prefix}_dlat_expert"], r[f"{prefix}_s_expert"] = "na", np.nan, np.nan
    t, dl, s = tier_of(poly_rec, own, own_len)
    r[f"{prefix}_tier_own"], r[f"{prefix}_dlat_own"], r[f"{prefix}_s_own"] = t, dl, s
    # does the reference (pre-LQR, linear-interpolated prediction) footprint already overlap this track?
    first_ov = -1
    for k in range(41):
        if token in obs[k].token_to_idx and ref_polys[k].intersects(obs[k][token]):
            first_ov = k
            break
    r[f"{prefix}_ref_overlap_idx"] = first_ov
    return r


def process_token(tok: str):
    import shapely
    import shapely.vectorized
    import shapely.geometry as sg
    from shapely.ops import unary_union, nearest_points
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        state_array_to_coords_array, coords_array_to_polygon_array,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        EgoAreaIndex, MultiMetricIndex, WeightedMetricIndex,
    )

    sim, scorer = G["sim"], G["scorer"]
    log = G["t2l"][tok]
    with lzma.open(MC_ROOT / log / "unknown" / tok / "metric_cache.pkl", "rb") as f:
        mc = pickle.load(f)
    ra = mc.ego_state.rear_axle
    row = {"token": tok, "log": log}

    # ---------------- 1. official re-score of the prediction ----------------
    poses = np.asarray(G["traj"][tok], dtype=np.float32)
    traj = Trajectory(poses, TrajectorySampling(num_poses=8, interval_length=0.5))
    res = pdm_score(mc, traj, sim.proposal_sampling, sim, scorer)
    for k in ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
              "ego_progress", "time_to_collision_within_bound", "comfort", "score"]:
        row["re_" + k] = float(getattr(res, k))
    P = 1  # submitted trajectory proposal index inside pdm_score()
    row["pdmc_nc"] = float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 0])
    row["pdmc_dac"] = float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 0])
    row["pdmc_ttc"] = float(scorer._weighted_metrics[WeightedMetricIndex.TTC, 0])
    row["rec_nc_consistent"] = scorer.rec_nc_consistent
    row["rec_ttc_consistent"] = scorer.rec_ttc_consistent
    row["raw_progress_m"] = float(scorer._progress_raw[P])
    row["raw_progress_pdmc_m"] = float(scorer._progress_raw[0])

    sim_states = sim.last_sim[P].copy()  # (41, 11) LQR-tracked states, global frame
    ref_states = sim.last_ref[P].copy()  # (41, 11) linear-interpolated prediction (reference)
    ego_coords = scorer._ego_coords[P].copy()  # (41, 5, 2) FL, RL, RR, FR, CENTER
    ego_areas = scorer._ego_areas[P].copy()
    nondriv = ego_areas[:, EgoAreaIndex.NON_DRIVABLE_AREA]
    dev = np.hypot(*(sim_states[:, :2] - ref_states[:, :2]).T)
    row["lqr_max_dev_m"] = float(dev.max())

    # paths (record frame)
    hv = G["human"].get(tok)
    ep = None
    ep_len = np.nan
    if hv is not None and bool(hv[1]):
        ep = expert_path_rec(hv[0])
        ep_len = polyline_len(ep)
    row["expert_path_len"] = ep_len
    row["expert_path_stationary"] = bool(ep is not None and ep_len < 1.0)
    own = expert_path_rec(poses)
    own_len = polyline_len(own)
    row["own_path_len"] = own_len

    # reference-trajectory footprints (pre-LQR) for DAC / overlap checks
    ref_coords = state_array_to_coords_array(sim.last_ref[P:P + 1], scorer._vehicle_parameters)[0]
    ref_polys = coords_array_to_polygon_array(ref_coords[None])[0]

    # aux record GT
    aux = np.load(G["aux_dir"] / f"{tok}.npz")
    gt_boxes = aux["det_gt_boxes"].astype(np.float64)
    gt_labels = aux["det_gt_labels"]

    # ---------------- alignment check: t=0 tracks in ROI vs aux det GT ----------------
    obs0 = mc.observation[0]
    cen = np.array([[g.centroid.x, g.centroid.y] for g in obs0._geometries]) if len(obs0) else np.zeros((0, 2))
    rec0 = g2rec(cen, ra) if len(cen) else cen
    in_roi = np.array([vis_class(x, y) == "roi" for x, y in rec0], dtype=bool) if len(rec0) else np.zeros(0, bool)
    row["align_n_obs_roi"] = int(in_roi.sum())
    row["align_n_gt"] = int(len(gt_boxes))
    if in_roi.any() and len(gt_boxes):
        d = np.hypot(rec0[in_roi, None, 0] - gt_boxes[None, :, 0], rec0[in_roi, None, 1] - gt_boxes[None, :, 1])
        j = d.argmin(1)
        dm = d[np.arange(len(j)), j]
        m = dm <= GT_LINK_M
        row["align_n_linked"] = int(m.sum())
        row["align_med_dist"] = float(np.median(dm))
        row["align_mean_dxr"] = float(np.mean(rec0[in_roi][m, 0] - gt_boxes[j[m], 0])) if m.any() else np.nan
        row["align_mean_dyf"] = float(np.mean(rec0[in_roi][m, 1] - gt_boxes[j[m], 1])) if m.any() else np.nan
    else:
        row["align_n_linked"] = 0
        row["align_med_dist"] = row["align_mean_dxr"] = row["align_mean_dyf"] = np.nan

    # ---------------- 2. NC events ----------------
    events = []
    nc_ev = [e for e in scorer.rec_nc_events if e["proposal"] == P]
    for e in nc_ev:
        events.append({"token": tok, "metric": "NC", **e})
    af = [e for e in nc_ev if e["at_fault"]]
    row["nc_n_events_atfault"] = len(af)
    row["nc_n_tracks_atfault"] = len({e["track"] for e in af})
    row["nc_n_tracks_nonfault"] = len({e["track"] for e in nc_ev if not e["at_fault"]})
    if af:
        tmin = min(e["time_idx"] for e in af)
        cand = [e for e in af if e["time_idx"] == tmin]
        cand.sort(key=lambda e: e["value"])  # agents (0.0) before static (0.5)
        e0 = cand[0]
        row["nc_track"] = e0["track"]
        row["nc_time_idx"] = e0["time_idx"]
        row["nc_collision_type"] = e0["collision_type"]
        row["nc_value"] = e0["value"]
        row["nc_multi_or_nondrivable"] = e0["multi_or_nondrivable"]
        row["nc_all_tracks"] = ";".join(sorted({e["track"] for e in af}))
        row.update(cause_object_info("nc", e0["track"], e0["time_idx"], e0["time_idx"], mc, ra, gt_boxes, gt_labels,
                                     ep, ep_len, own, own_len, ref_polys, sim_states))
    # ---------------- 3. TTC events ----------------
    ttc_ev = [e for e in scorer.rec_ttc_events if e["proposal"] == P]
    for e in ttc_ev:
        events.append({"token": tok, "metric": "TTC", **e})
    row["ttc_n_events"] = len(ttc_ev)
    row["ttc_n_tracks"] = len({e["track"] for e in ttc_ev})
    if ttc_ev:
        tmin = min(e["time_idx"] for e in ttc_ev)
        cand = [e for e in ttc_ev if e["time_idx"] == tmin]
        cand.sort(key=lambda e: e["future_idx"])
        e0 = cand[0]
        row["ttc_track"] = e0["track"]
        row["ttc_time_idx"] = e0["time_idx"]
        row["ttc_future_idx"] = e0["future_idx"]
        row["ttc_ahead"] = e0["ahead"]
        row["ttc_in_intersection"] = e0["in_intersection"]
        row["ttc_multi_or_nondrivable"] = e0["multi_or_nondrivable"]
        row["ttc_all_tracks"] = ";".join(sorted({e["track"] for e in ttc_ev}))
        row["ttc_same_as_nc"] = bool("nc_track" in row and row["nc_track"] == e0["track"])
        row.update(cause_object_info("ttc", e0["track"], e0["time_idx"] + e0["future_idx"], e0["time_idx"], mc, ra,
                                     gt_boxes, gt_labels, ep, ep_len, own, own_len, ref_polys, sim_states))

    # ---------------- 4. DAC ----------------
    dm = mc.drivable_area_map
    d_idcs = dm.get_indices_of_map_type(
        [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA,
         SemanticMapLayer.CARPARK_AREA]
    )
    geoms = [dm._geometries[i] for i in d_idcs]

    def corner_inside(coords):  # coords (T, 5, 2) -> (T, 4) bool, exactly as the scorer (per polygon)
        pts = coords[:, :4].reshape(-1, 2)
        inside = np.zeros(len(pts), bool)
        for g in geoms:
            inside |= shapely.vectorized.contains(g, pts[:, 0], pts[:, 1])
        return inside.reshape(-1, 4)

    ref_inside = corner_inside(ref_coords)
    ref_off = ~ref_inside.all(1)
    row["ref_dac"] = float(not ref_off.any())
    row["ref_first_offroad_idx"] = int(np.argmax(ref_off)) if ref_off.any() else -1
    row["dac_n_offroad_steps"] = int(nondriv.sum())
    row["dac_first_idx"] = int(np.argmax(nondriv)) if nondriv.any() else -1
    union = unary_union([g.buffer(0) for g in geoms]) if geoms else sg.Polygon()
    # clearance of all corners (inside) to the drivable boundary: min over horizon (for all tokens)
    sim_inside = corner_inside(ego_coords)
    assert np.array_equal(~sim_inside.all(1), nondriv), "corner test disagrees with scorer"
    pts = ego_coords[:, :4].reshape(-1, 2)
    dist_b = shapely.distance(union.boundary, shapely.points(pts)).reshape(-1, 4) if not union.is_empty else np.full((41, 4), np.nan)
    signed = np.where(sim_inside, dist_b, -dist_b)  # + inside (clearance), - outside (penetration)
    row["dac_min_signed_clear"] = float(np.nanmin(signed))
    pts_r = ref_coords[:, :4].reshape(-1, 2)
    dist_r = shapely.distance(union.boundary, shapely.points(pts_r)).reshape(-1, 4) if not union.is_empty else np.full((41, 4), np.nan)
    signed_r = np.where(ref_inside, dist_r, -dist_r)
    row["ref_min_signed_clear"] = float(np.nanmin(signed_r))
    row["ref_depth_max"] = float(max(0.0, -np.nanmin(signed_r)))
    if nondriv.any():
        t = row["dac_first_idx"]
        out_c = ~sim_inside[t]
        row["dac_first_time_s"] = t * 0.1
        exy = g2rec(sim_states[t, :2], ra)
        row["dac_ego_x"], row["dac_ego_y"], row["dac_ego_h"] = map(float, sim_states[t, :3])
        row["dac_ego_xr"], row["dac_ego_yf"] = float(exy[0]), float(exy[1])
        row["dac_ego_heading_rel"] = float(wrap(sim_states[t, 2] - ra.heading))
        crec = g2rec(ego_coords[t], ra)  # (5,2) FL, RL, RR, FR, CENTER
        for ci, nm in enumerate(["fl", "rl", "rr", "fr", "c"]):
            row[f"dac_corner_{nm}_xr"], row[f"dac_corner_{nm}_yf"] = float(crec[ci, 0]), float(crec[ci, 1])
        row["dac_corners_out"] = ",".join(nm for nm, o in zip(["FL", "RL", "RR", "FR"], out_c) if o)
        left = out_c[0] or out_c[1]
        right = out_c[2] or out_c[3]
        row["dac_side_ego"] = "both" if (left and right) else ("left" if left else "right")
        depth_c = np.where(out_c, dist_b[t], 0.0)
        k = int(np.argmax(depth_c))
        row["dac_depth_first"] = float(depth_c[k])
        row["dac_depth_max"] = float(max(0.0, -np.nanmin(signed)))
        deep_g = ego_coords[t, k]
        bp = nearest_points(union.boundary, sg.Point(*deep_g))[0]
        brec = g2rec([bp.x, bp.y], ra)
        drec = g2rec(deep_g, ra)
        row["dac_deep_corner"] = ["FL", "RL", "RR", "FR"][k]
        row["dac_deep_xr"], row["dac_deep_yf"] = float(drec[0]), float(drec[1])
        row["dac_bnd_xr"], row["dac_bnd_yf"] = float(brec[0]), float(brec[1])
        row["dac_lqr_dev_at_first"] = float(dev[t])
        if ep is not None and ep_len >= 1.0:
            o, s = signed_offset(ep, drec)
            row["dac_deep_lat_vs_expert"], row["dac_deep_s_expert"] = o, s
            o2, s2 = signed_offset(ep, exy + np.array([0.0, 0.0]))
            row["dac_ego_lat_vs_expert"] = o2
            row["dac_side_expert"] = "right" if o > 0 else "left"
        else:
            row["dac_deep_lat_vs_expert"] = row["dac_deep_s_expert"] = row["dac_ego_lat_vs_expert"] = np.nan
            row["dac_side_expert"] = "na"
        # own reference path offset at the same time (did LQR drift to one side?)
        rr = g2rec(ref_states[t, :2], ra)
        row["dac_ref_ego_xr"], row["dac_ref_ego_yf"] = float(rr[0]), float(rr[1])
        row["dac_stage"] = "pred_stage" if ref_off.any() else "lqr_only"

    # ---------------- 5. expert control: [PDM-Closed, human] ----------------
    if hv is not None and bool(hv[1]):
        htraj = Trajectory(np.asarray(hv[0], dtype=np.float32), TrajectorySampling(num_poses=8, interval_length=0.5))
        hres = pdm_score(mc, htraj, G["sim_h"].proposal_sampling, G["sim_h"], G["scorer_h"])
        row["human_nc"] = float(hres.no_at_fault_collisions)
        row["human_dac"] = float(hres.drivable_area_compliance)
        row["human_ttc"] = float(hres.time_to_collision_within_bound)
        row["human_score"] = float(hres.score)
        hnc = [e for e in G["scorer_h"].rec_nc_events if e["proposal"] == 1 and e["at_fault"]]
        httc = [e for e in G["scorer_h"].rec_ttc_events if e["proposal"] == 1]
        row["human_nc_tracks"] = ";".join(sorted({e["track"] for e in hnc}))
        row["human_ttc_tracks"] = ";".join(sorted({e["track"] for e in httc}))
    return row, events, sim_states.astype(np.float64), ref_states[:, :3].astype(np.float64)


def run_chunk(args):
    ci, toks = args
    path = OUT / "shards" / f"shard_{ci:04d}.pkl"
    if path.exists():
        return ci, len(toks), 0, "cached"
    rows, evs, S, R, errs = [], [], [], [], []
    t0 = time.time()
    for tok in toks:
        try:
            row, ev, s, r = process_token(tok)
            rows.append(row)
            evs.extend(ev)
            S.append(s)
            R.append(r)
        except Exception:
            errs.append((tok, traceback.format_exc()))
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"rows": rows, "events": evs, "sim": np.stack(S) if S else None,
                     "ref": np.stack(R) if R else None, "tokens": [r["token"] for r in rows], "errors": errs}, f)
    os.replace(tmp, path)
    return ci, len(toks), len(errs), f"{time.time() - t0:.1f}s"


def _pool_init(arm, t2l):
    _init_worker(arm)
    G["t2l"] = t2l


def main():
    import pandas as pd
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="para_ssr_interaction_final")
    ap.add_argument("--csv", default=str(EVAL / "para_ssr_interaction_final/2026.09.17.00.09.41.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--n_pass", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scope", choices=["sample", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N tokens of the ordered list")
    a = ap.parse_args()

    (OUT / "shards").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv, index_col=0)
    df = df[df.token != "average"].reset_index(drop=True)
    fail = df[(df.no_at_fault_collisions < 1) | (df.time_to_collision_within_bound < 1) | (df.drivable_area_compliance < 1)]
    passing = df.drop(fail.index)
    rng = np.random.default_rng(a.seed)
    pass_sample = sorted(rng.choice(passing.token.values, size=a.n_pass, replace=False).tolist())
    fail_toks = sorted(fail.token.tolist())
    if not os.environ.get("H7_TOKENS"): json.dump({"seed": a.seed, "csv": a.csv, "n_fail": len(fail_toks), "n_pass_sample": len(pass_sample),
               "fail_tokens": fail_toks, "pass_sample_tokens": pass_sample},
              open(OUT / "sample_tokens.json", "w"), indent=1)
    ordered = fail_toks + pass_sample
    if os.environ.get("H7_TOKENS"):
        ordered = json.load(open(os.environ["H7_TOKENS"]))
        a.scope = "given"
    if a.scope == "all":
        rest = sorted(set(df.token) - set(ordered))
        ordered = ordered + rest
    if a.limit:
        ordered = ordered[: a.limit]

    t2l = {}
    for lg in os.listdir(MC_ROOT):
        d = MC_ROOT / lg / "unknown"
        if d.is_dir():
            for t in os.listdir(d):
                t2l[t] = lg
    missing = [t for t in ordered if t not in t2l]
    if missing:
        print(f"WARNING: {len(missing)} tokens without metric cache, skipped", flush=True)
        ordered = [t for t in ordered if t in t2l]

    chunks = [(i, ordered[s:s + a.chunk]) for i, s in enumerate(range(0, len(ordered), a.chunk))]
    print(f"{len(ordered)} tokens in {len(chunks)} chunks, workers={a.workers}", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_pool_init, initargs=(a.arm, t2l)) as pool:
        done = 0
        for ci, n, ne, msg in pool.imap_unordered(run_chunk, chunks):
            done += n
            print(f"chunk {ci} ({n} tok, {ne} err, {msg}) -- {done}/{len(ordered)} in {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
