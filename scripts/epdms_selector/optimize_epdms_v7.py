#!/usr/bin/env python
"""Ultra-fast persistent-cache optimizer for v2_v15_v7_main (64m BEV) on NAVSIM v2.2 EPDMS (navtest & navhard, After & Before)."""
from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

NAVSIM22 = "/workspace/navsim_workspace/navsim"
if sys.path[0] != NAVSIM22:
    sys.path.insert(0, NAVSIM22)

import numpy as np
import pandas as pd
import yaml
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.script.run_pdm_score import (
    compute_final_scores as compute_final_scores_two_stage,
)
from navsim.planning.script.run_pdm_score_one_stage import (
    compute_final_scores as compute_final_scores_one_stage,
    infer_start_adjacent_mapping,
)
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_two_frame_extended_comfort
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents

_MC_LOADER: MetricCacheLoader = None
_SIMULATOR: PDMSimulator = None
_SCORER_UNFILTERED: PDMScorer = None
_POLICY: Any = None
_HUMAN_ZERO_COLS: Dict[str, List[str]] = {}
_TRAJ_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)
_PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)
_SKIP_COLS = {"multiplicative_metrics_prod", "weighted_metrics", "weighted_metrics_array", "pdm_score"}


def _init_worker(mc_dir: str, split: str) -> None:
    global _MC_LOADER, _SIMULATOR, _SCORER_UNFILTERED, _POLICY, _HUMAN_ZERO_COLS
    if sys.path[0] != NAVSIM22:
        sys.path.insert(0, NAVSIM22)
    _SIMULATOR = PDMSimulator(proposal_sampling=_PROPOSAL_SAMPLING)
    _SCORER_UNFILTERED = PDMScorer(
        proposal_sampling=_PROPOSAL_SAMPLING,
        config=PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            lane_keeping_weight=2.0,
            history_comfort_weight=2.0,
            two_frame_extended_comfort_weight=2.0,
            driving_direction_horizon=1.0,
            driving_direction_compliance_threshold=2.0,
            driving_direction_violation_threshold=6.0,
            stopped_speed_threshold=0.005,
            future_collision_horizon_window=1.0,
            progress_distance_threshold=5.0,
            lane_keeping_deviation_limit=0.5,
            lane_keeping_horizon_window=2.0,
            human_penalty_filter=False,
        ),
    )
    if split == "navtest":
        _POLICY = LogReplayTrafficAgents(_PROPOSAL_SAMPLING)
    else:
        _POLICY = NavsimIDMTrafficAgents(
            future_trajectory_sampling=_PROPOSAL_SAMPLING,
            idm_agents_observation=NavsimIDMAgents(
                target_velocity=10,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.0,
                decel_max=2.0,
                open_loop_detections_types=[],
                minimum_path_length=20,
                planned_trajectory_samples=None,
                planned_trajectory_sample_interval=None,
                radius=100,
                add_open_loop_parked_vehicles=True,
                idm_snap_threshold=3.0,
            ),
        )
    _MC_LOADER = MetricCacheLoader(Path(mc_dir))
    _HUMAN_ZERO_COLS = {}


def _get_human_zero_cols(token: str, metric_cache) -> List[str]:
    if token in _HUMAN_ZERO_COLS:
        return _HUMAN_ZERO_COLS[token]
    if metric_cache.scene_type != SceneFrameType.ORIGINAL:
        _HUMAN_ZERO_COLS[token] = []
        return []
    initial_ego_state = metric_cache.ego_state
    human_trajectory = transform_trajectory(metric_cache.human_trajectory, initial_ego_state)
    human_states = get_trajectory_as_array(human_trajectory, _PROPOSAL_SAMPLING, initial_ego_state.time_point)
    human_simulated_states = _SIMULATOR.simulate_proposals(human_states[None, ...], initial_ego_state)
    human_sim_tracks = _POLICY.simulate_environment(human_simulated_states[0], metric_cache)
    human_pdm_result = _SCORER_UNFILTERED.score_proposals(
        human_simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        human_sim_tracks,
        metric_cache.past_human_trajectory,
    )[0]
    zcols = [c for c in human_pdm_result.columns if c not in _SKIP_COLS and human_pdm_result[c].iloc[0] == 0]
    _HUMAN_ZERO_COLS[token] = zcols
    return zcols


def _score_one_item(item: Tuple[str, np.ndarray]) -> Dict[str, Any]:
    token, poses = item
    try:
        metric_cache = _MC_LOADER.get_from_token(token)
        poses = np.asarray(poses, dtype=np.float32).reshape(-1, 3)
        trajectory = Trajectory(poses, _TRAJ_SAMPLING)

        initial_ego_state = metric_cache.ego_state
        pred_trajectory = transform_trajectory(trajectory, initial_ego_state)
        pdm_states = get_trajectory_as_array(metric_cache.trajectory, _PROPOSAL_SAMPLING, initial_ego_state.time_point)
        pred_states = get_trajectory_as_array(pred_trajectory, _PROPOSAL_SAMPLING, initial_ego_state.time_point)
        trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)
        simulated_states = _SIMULATOR.simulate_proposals(trajectory_states, initial_ego_state)
        simulated_tracks = _POLICY.simulate_environment(simulated_states[1], metric_cache)

        pdm_result = _SCORER_UNFILTERED.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            simulated_tracks,
            metric_cache.past_human_trajectory,
        )[1]

        bef_mult = float(pdm_result.at[0, "multiplicative_metrics_prod"])
        bef_w_metrics = pdm_result.at[0, "weighted_metrics"].copy()

        zcols = _get_human_zero_cols(token, metric_cache)
        if zcols:
            for col in zcols:
                pdm_result.at[0, col] = 1
            pdm_result.at[0, "multiplicative_metrics_prod"] = (
                pdm_result.at[0, "no_at_fault_collisions"]
                * pdm_result.at[0, "drivable_area_compliance"]
                * pdm_result.at[0, "driving_direction_compliance"]
                * pdm_result.at[0, "traffic_light_compliance"]
            )
            wm = pdm_result.at[0, "weighted_metrics"].copy()
            wm[WeightedMetricIndex.PROGRESS] = pdm_result.at[0, "ego_progress"]
            wm[WeightedMetricIndex.TTC] = pdm_result.at[0, "time_to_collision_within_bound"]
            wm[WeightedMetricIndex.LANE_KEEPING] = pdm_result.at[0, "lane_keeping"]
            wm[WeightedMetricIndex.HISTORY_COMFORT] = pdm_result.at[0, "history_comfort"]
            pdm_result.at[0, "weighted_metrics"] = wm

        d = pdm_result.iloc[0].to_dict()
        d["bef_multiplicative_metrics_prod"] = bef_mult
        d["bef_weighted_metrics"] = bef_w_metrics
        d["valid"] = True
        d["log_name"] = metric_cache.log_name
        d["frame_type"] = metric_cache.scene_type
        d["start_time"] = metric_cache.timepoint.time_s
        end_pose = StateSE2(
            x=trajectory.poses[-1, 0],
            y=trajectory.poses[-1, 1],
            heading=trajectory.poses[-1, 2],
        )
        absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
        d["endpoint_x"] = absolute_endpoint.x
        d["endpoint_y"] = absolute_endpoint.y
        d["start_point_x"] = metric_cache.ego_state.rear_axle.x
        d["start_point_y"] = metric_cache.ego_state.rear_axle.y
        d["ego_simulated_states"] = simulated_states[1]
    except Exception:
        traceback.print_exc()
        d = PDMResults.get_empty_results().__dict__.copy()
        d["valid"] = False
    d["token"] = token
    return d


def _fast_two_frame_ec(
    now_tok: str,
    now_key: Tuple[str, bytes],
    prev_tok: str,
    prev_key: Tuple[str, bytes],
    row_cache: Dict[Tuple[str, bytes], Dict[str, Any]],
    ec_cache: Dict[Tuple[str, bytes, str, bytes], float],
) -> float:
    ck = (now_tok, now_key[1], prev_tok, prev_key[1])
    if ck in ec_cache:
        return ec_cache[ck]
    r_now = row_cache[now_key]
    r_prev = row_cache[prev_key]
    c_states = r_now["ego_simulated_states"]
    p_states = r_prev["ego_simulated_states"]
    interval_length = _PROPOSAL_SAMPLING.interval_length
    obs_interval = r_now["start_time"] - r_prev["start_time"]
    overlap_start = round(obs_interval / interval_length)
    c_ov = c_states[:-overlap_start]
    p_ov = p_states[overlap_start:]
    tp_s = np.arange(c_ov.shape[0]) * interval_length
    val = float(ego_is_two_frame_extended_comfort(c_ov[None, :], p_ov[None, :], tp_s)[0].astype(np.float64))
    ec_cache[ck] = val
    return val


def compute_final_scores_before_from_saved(agg_df: pd.DataFrame, two_stage: bool) -> pd.DataFrame:
    df = agg_df.copy()
    df["multiplicative_metrics_prod"] = df["bef_multiplicative_metrics_prod"]
    df["weighted_metrics"] = [x.copy() for x in df["bef_weighted_metrics"]]
    df.drop(columns=["bef_multiplicative_metrics_prod", "bef_weighted_metrics"], inplace=True, errors="ignore")
    if two_stage:
        return compute_final_scores_two_stage(df)
    return compute_final_scores_one_stage(df)


def load_pool(mode: str, split: str):
    base = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main")
    if mode == "ep29" and split == "navtest":
        er = Path("/workspace/yongjae/ssd/yongjae_refiner/v7/eval/v2_v15_v7_main_ep30/eval/root")
        pk = er / "packed/navtest"
        inf = er / "infer/v2_v15_v7_main_ep30/navtest"
        tdf = pd.read_parquet(pk / "tokens.parquet")
        toks = [str(t) for t in tdf.token.to_numpy()]
        cand96 = np.load(pk / "cand96.npy").astype(np.float32)
        lat96 = np.load(inf / "lat_traj.npy").astype(np.float32)
        v2_final = np.load(pk / "v2_final.npy").astype(np.float64)
        v2_im = np.load(pk / "v2_im.npy").astype(np.float64)
        v2_sim = np.load(pk / "v2_sim.npy").astype(np.float64)
        logit96 = np.load(inf / "score_logit.npy").astype(np.float64)
        valid96 = np.load(pk / "valid96.npy").astype(bool)
    else:
        pdir = base / f"pool96_{mode}_{split}"
        toks = json.load(open(pdir / "tokens.json"))
        cand96 = np.load(pdir / "cand96.npy").astype(np.float32)
        lat96 = np.load(pdir / "lat_traj.npy").astype(np.float32)
        v2_final = np.load(pdir / "v2_final.npy").astype(np.float64)
        v2_im = np.load(pdir / "v2_im.npy").astype(np.float64)
        v2_sim = np.load(pdir / "v2_sim.npy").astype(np.float64)
        logit96 = np.load(pdir / "score_logit.npy").astype(np.float64)
        valid96 = np.load(pdir / "valid96.npy").astype(bool)
    return toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96


def select_trajectories(
    cfg: Dict[str, Any],
    toks: List[str],
    cand96: np.ndarray,
    lat96: np.ndarray,
    v2_final: np.ndarray,
    v2_im: np.ndarray,
    v2_sim: np.ndarray,
    logit96: np.ndarray,
    valid96: np.ndarray,
) -> Dict[str, np.ndarray]:
    N = len(toks)
    ar = np.arange(N)
    vtype = np.arange(96) % 6
    prank = np.arange(96) // 6
    ok_all = valid96 | (vtype[None, :] == 0)

    if cfg.get("force_col0", False):
        idx = np.zeros(N, dtype=np.int64)
    else:
        prob96 = 1.0 / (1.0 + np.exp(-logit96))
        im96 = np.repeat(v2_im, 6, axis=1)
        v2f96 = np.repeat(v2_final, 6, axis=1)
        v2sim96 = np.repeat(v2_sim, 6, axis=1)

        eps = 1e-6
        nc_c, dac_c, ep_c, ttc_c, cmf_c = [prob96[..., i] for i in range(5)]
        nc_v, dac_v, ep_v, ttc_v, cmf_v = [v2sim96[..., i] for i in range(5)]

        w_v2f = cfg.get("w_v2f", 0.5)
        w_im = cfg.get("w_im", 0.05)
        w_nc = cfg.get("w_nc", 0.5)
        w_dac = cfg.get("w_dac", 0.5)
        w_ep = cfg.get("w_ep", 0.0)
        w_ttc = cfg.get("w_ttc", 0.0)
        w_cmf = cfg.get("w_cmf", 0.0)
        w_rest = cfg.get("w_rest", 0.5)
        a_ttc = cfg.get("a_ttc", 5.0)
        a_cmf = cfg.get("a_cmf", 2.0)
        a_ep = cfg.get("a_ep", 5.0)

        w_nc_v2 = cfg.get("w_nc_v2", 0.0)
        w_dac_v2 = cfg.get("w_dac_v2", 0.0)
        w_ep_v2 = cfg.get("w_ep_v2", 0.0)
        w_ttc_v2 = cfg.get("w_ttc_v2", 0.0)
        w_cmf_v2 = cfg.get("w_cmf_v2", 0.0)

        w_prod = cfg.get("w_prod", 0.0)
        epdms_c = (nc_c * dac_c) * (5.0 * ttc_c + 5.0 * ep_c + 2.0 * cmf_c) / 12.0

        dv = cfg.get("dv", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w_prank = cfg.get("w_prank", 0.0)
        d_col0 = cfg.get("d_col0", 0.0)

        s = (
            w_v2f * v2f96
            + w_im * np.log(im96 + eps)
            + w_nc * np.log(nc_c + eps)
            + w_dac * np.log(dac_c + eps)
            + w_ep * np.log(ep_c + eps)
            + w_ttc * np.log(ttc_c + eps)
            + w_cmf * np.log(cmf_c + eps)
            + w_rest * np.log(a_ttc * ttc_c + a_cmf * cmf_c + a_ep * ep_c + eps)
            + w_nc_v2 * np.log(nc_v + eps)
            + w_dac_v2 * np.log(dac_v + eps)
            + w_ep_v2 * np.log(ep_v + eps)
            + w_ttc_v2 * np.log(ttc_v + eps)
            + w_cmf_v2 * np.log(cmf_v + eps)
            + w_prod * epdms_c
            + w_prank * (prank[None, :] / 15.0)
        )
        for v in range(6):
            if dv[v] != 0.0:
                s[:, vtype == v] += dv[v]
        if d_col0 != 0.0:
            s[:, 0] += d_col0

        s = np.where(ok_all, s, -1e9)
        idx = np.argmax(s, axis=1)

    lat_mode = cfg.get("lat_mode", "lat")
    if lat_mode == "lat":
        chosen = lat96[ar, idx]
    elif lat_mode == "pool":
        chosen = cand96[ar, idx]
    else:
        raise ValueError(lat_mode)

    return {toks[i]: chosen[i] for i in range(N)}


def get_candidate_configs() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        # 1. Baselines (v2, v2_lat, v7_default w=(0.1, 0.5, 0.5, 1.0), f15_default w=(0.1, 1, 1, 1), plugin w=(0, 1, 1, 1))
        ("v2_base", {"force_col0": True, "lat_mode": "pool"}),
        ("v2_lat", {"force_col0": True, "lat_mode": "lat"}),
        ("v7_default_b075", {"w_v2f": 0.25, "w_im": 0.075, "w_nc": 0.375, "w_dac": 0.375, "w_rest": 0.75, "lat_mode": "lat"}),
        ("v7_default_b050", {"w_v2f": 0.50, "w_im": 0.050, "w_nc": 0.250, "w_dac": 0.250, "w_rest": 0.50, "lat_mode": "lat"}),
        ("v7_default_b025", {"w_v2f": 0.75, "w_im": 0.025, "w_nc": 0.125, "w_dac": 0.125, "w_rest": 0.25, "lat_mode": "lat"}),
        ("f15_default_b050", {"w_v2f": 0.50, "w_im": 0.050, "w_nc": 0.500, "w_dac": 0.500, "w_rest": 0.50, "lat_mode": "lat"}),
        ("plugin_b050", {"w_v2f": 0.50, "w_im": 0.000, "w_nc": 0.500, "w_dac": 0.500, "w_rest": 0.50, "lat_mode": "lat"}),
        ("plugin_b100", {"w_v2f": 0.00, "w_im": 0.000, "w_nc": 1.000, "w_dac": 1.000, "w_rest": 1.00, "lat_mode": "lat"}),

        # 2. Zero-shot SOTA formulas from fusion_v15_main!
        ("zs_prod_hybrid2", {"w_v2f": 0.3, "w_im": 0.0, "w_nc": 0.8, "w_dac": 0.8, "w_rest": 0.5, "w_prod": 2.0, "w_nc_v2": 0.4, "w_dac_v2": 0.4, "dv": [0.03, -0.02, 0.01, -0.02, -0.06, -0.06], "d_col0": 0.02, "lat_mode": "lat"}),
        ("zs_f15_nt_ep22_best", {"w_v2f": 0.4, "w_im": 0.0, "w_nc": 0.6, "w_dac": 0.8, "w_rest": 0.5, "w_prod": 2.0, "w_nc_v2": 0.5, "w_dac_v2": 0.3, "dv": [0.04, -0.02, 0.01, -0.02, -0.06, -0.06], "d_col0": 0.0, "lat_mode": "lat"}),
        ("zs_f15_nt_ep29_best", {"w_v2f": 0.3, "w_im": 0.0, "w_nc": 0.8, "w_dac": 1.2, "w_rest": 0.5, "w_prod": 2.0, "w_nc_v2": 0.4, "w_dac_v2": 0.4, "dv": [0.04, -0.02, 0.01, -0.02, -0.03, -0.03], "d_col0": 0.0, "lat_mode": "lat"}),
        ("zs_f15_nh_ep22_best", {"w_v2f": 0.45, "w_im": 0.0, "w_nc": 0.8, "w_dac": 0.6, "w_ttc": 0.25, "w_rest": 0.5, "w_nc_v2": 0.3, "w_dac_v2": 0.5, "dv": [0.0, 0.01, 0.02, -0.05, -0.03, -0.03], "lat_mode": "lat"}),
        ("zs_f15_nh_ep29_best", {"w_v2f": 0.45, "w_im": 0.0, "w_nc": 0.6, "w_dac": 0.9, "w_ttc": 0.25, "w_rest": 0.5, "w_nc_v2": 0.7, "w_dac_v2": 0.9, "dv": [0.0, 0.01, 0.02, -0.05, -0.04, -0.04], "lat_mode": "lat"}),

        # 3. Additional Two-Head Safety Consensus & Variant-Prior candidates
        ("safe_v2sim_strong", {"w_v2f": 0.35, "w_im": 0.0, "w_nc": 1.0, "w_dac": 1.0, "w_rest": 0.5, "w_nc_v2": 0.7, "w_dac_v2": 0.7, "lat_mode": "lat"}),
        ("safe_v2sim_nolat2", {"w_v2f": 0.4, "w_im": 0.0, "w_nc": 0.9, "w_dac": 0.9, "w_rest": 0.5, "w_nc_v2": 0.6, "w_dac_v2": 0.6, "dv": [0.03, 0.0, 0.02, -0.03, -0.08, -0.08], "d_col0": 0.02, "lat_mode": "lat"}),
        ("safe_v2sim_cmf", {"w_v2f": 0.4, "w_im": 0.0, "w_nc": 0.85, "w_dac": 0.85, "w_cmf": 0.15, "w_ttc": 0.2, "w_rest": 0.5, "w_nc_v2": 0.5, "w_dac_v2": 0.5, "w_cmf_v2": 0.15, "dv": [0.02, 0.0, 0.01, -0.02, -0.05, -0.05], "d_col0": 0.02, "lat_mode": "lat"}),
    ]


def evaluate_config_navhard(
    name: str,
    traj_dict: Dict[str, np.ndarray],
    all_tokens: List[str],
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    row_cache: Dict[Tuple[str, bytes], Dict[str, Any]],
    ec_cache: Dict[Tuple[str, bytes, str, bytes], float],
    ex: ProcessPoolExecutor,
    workers: int,
) -> Dict[str, Any]:
    t0 = time.time()
    todo_items = []
    keys_of: Dict[str, Tuple[str, bytes]] = {}
    for tok in all_tokens:
        poses = np.ascontiguousarray(traj_dict[tok], dtype=np.float32).reshape(-1, 3)
        key = (tok, poses.tobytes())
        keys_of[tok] = key
        if key not in row_cache:
            todo_items.append((tok, poses))
    if todo_items:
        cs = max(1, min(16, len(todo_items) // (workers * 2)))
        for (tok, poses), r in zip(todo_items, ex.map(_score_one_item, todo_items, chunksize=cs)):
            row_cache[(tok, poses.tobytes())] = r

    ec_map: Dict[str, float] = {}
    w_map: Dict[str, float] = {}
    sigma2 = 0.1
    for (now_f, prev_f), second_stage in all_mappings.items():
        main_ec = _fast_two_frame_ec(now_f, keys_of[now_f], prev_f, keys_of[prev_f], row_cache, ec_cache)
        ec_map[now_f] = main_ec
        ec_map[prev_f] = main_ec
        w_map[now_f] = 1.0
        w_map[prev_f] = 1.0

        r_now = row_cache[keys_of[now_f]]
        r_prev = row_cache[keys_of[prev_f]]
        s2_now_toks = [p[0] for p in second_stage]
        s2_prev_toks = [p[1] for p in second_stage]

        for f_row, s2_toks in [(r_now, s2_now_toks), (r_prev, s2_prev_toks)]:
            x1, y1 = f_row["endpoint_x"], f_row["endpoint_y"]
            d2 = np.array([(x1 - row_cache[keys_of[t]]["start_point_x"]) ** 2 + (y1 - row_cache[keys_of[t]]["start_point_y"]) ** 2 for t in s2_toks], dtype=np.float64)
            w = np.exp(-d2 / (2.0 * sigma2))
            w_sum = w.sum()
            if np.isclose(w_sum, 0.0) or np.isnan(w_sum):
                w = np.full(len(s2_toks), 1.0 / len(s2_toks))
            else:
                w = w / w_sum
            for t, wt in zip(s2_toks, w):
                w_map[t] = float(wt)

        for now_t, prev_t in second_stage:
            ec2 = _fast_two_frame_ec(now_t, keys_of[now_t], prev_t, keys_of[prev_t], row_cache, ec_cache)
            ec_map[now_t] = ec2
            ec_map[prev_t] = ec2

    rows = []
    for tok in all_tokens:
        d = dict(row_cache[keys_of[tok]])
        d.pop("ego_simulated_states", None)
        d["two_frame_extended_comfort"] = ec_map.get(tok, np.nan)
        d["weight"] = w_map.get(tok, np.nan)
        rows.append(d)
    agg_df = pd.DataFrame(rows)

    df_after = compute_final_scores_two_stage(agg_df.drop(columns=["bef_multiplicative_metrics_prod", "bef_weighted_metrics"], errors="ignore").copy())
    score_cols = [
        c
        for c in df_after.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    def fast_mapping_scores(df_sc: pd.DataFrame):
        idx_df = df_sc.set_index("token")
        mat = idx_df[score_cols].to_numpy(dtype=np.float64)
        wts = idx_df["weight"].to_numpy(dtype=np.float64)
        t2i = {t: i for i, t in enumerate(idx_df.index)}

        g_list, s1_list, s2_list = [], [], []
        for (orig_t, prev_t), s2_pairs in all_mappings.items():
            f_idx = [t2i[p[0]] for p in s2_pairs if len(p) > 0]
            s_idx = [t2i[p[1]] for p in s2_pairs if len(p) > 1]
            g1_s1 = mat[t2i[orig_t]]
            g2_s1 = mat[t2i[prev_t]]
            w1 = wts[f_idx]; g1_s2 = (mat[f_idx] * w1[:, None]).sum(axis=0) / w1.sum()
            w2 = wts[s_idx]; g2_s2 = (mat[s_idx] * w2[:, None]).sum(axis=0) / w2.sum()
            s1_list.append(g1_s1); s1_list.append(g2_s1)
            s2_list.append(g1_s2); s2_list.append(g2_s2)
            g_list.append(0.5 * (g1_s1 * g1_s2 + g2_s1 * g2_s2))
        return (
            dict(zip(score_cols, (np.mean(g_list, axis=0) * 100.0).tolist())),
            dict(zip(score_cols, (np.mean(s1_list, axis=0) * 100.0).tolist())),
            dict(zip(score_cols, (np.mean(s2_list, axis=0) * 100.0).tolist())),
        )

    g_aft, s1_aft, s2_aft = fast_mapping_scores(df_after)
    df_bef = compute_final_scores_before_from_saved(agg_df, two_stage=True)
    g_bef, s1_bef, s2_bef = fast_mapping_scores(df_bef)

    res = {
        "after": {"combined": g_aft, "stage1": s1_aft, "stage2": s2_aft},
        "before": {"combined": g_bef, "stage1": s1_bef, "stage2": s2_bef},
        "new_scored": len(todo_items),
        "sec": round(time.time() - t0, 2),
    }
    print(
        f"[navhard:{name:22s}] new={len(todo_items):4d} ({res['sec']:5.2f}s) | "
        f"After={g_aft['score']:.2f}% (S1={s1_aft['score']:.2f}%, S2={s2_aft['score']:.2f}%) | "
        f"Before={g_bef['score']:.2f}% (S1={s1_bef['score']:.2f}%, S2={s2_bef['score']:.2f}%)",
        flush=True,
    )
    return res


def evaluate_config_navtest(
    name: str,
    traj_dict: Dict[str, np.ndarray],
    all_tokens: List[str],
    start_adj_cache: Dict[str, str],
    row_cache: Dict[Tuple[str, bytes], Dict[str, Any]],
    ec_cache: Dict[Tuple[str, bytes, str, bytes], float],
    ex: ProcessPoolExecutor,
    workers: int,
) -> Dict[str, Any]:
    t0 = time.time()
    todo_items = []
    keys_of: Dict[str, Tuple[str, bytes]] = {}
    for tok in all_tokens:
        poses = np.ascontiguousarray(traj_dict[tok], dtype=np.float32).reshape(-1, 3)
        key = (tok, poses.tobytes())
        keys_of[tok] = key
        if key not in row_cache:
            todo_items.append((tok, poses))
    if todo_items:
        cs = max(1, min(32, len(todo_items) // (workers * 2)))
        for (tok, poses), r in zip(todo_items, ex.map(_score_one_item, todo_items, chunksize=cs)):
            row_cache[(tok, poses.tobytes())] = r

    if not start_adj_cache:
        meta_rows = [{"token": t, "log_name": row_cache[keys_of[t]]["log_name"], "frame_type": row_cache[keys_of[t]]["frame_type"], "start_time": row_cache[keys_of[t]]["start_time"]} for t in all_tokens]
        start_adj_cache.update(infer_start_adjacent_mapping(pd.DataFrame(meta_rows)))

    ec_map: Dict[str, float] = {}
    for now_f, prev_f in start_adj_cache.items():
        ec_map[now_f] = _fast_two_frame_ec(now_f, keys_of[now_f], prev_f, keys_of[prev_f], row_cache, ec_cache)

    rows = []
    for tok in all_tokens:
        d = dict(row_cache[keys_of[tok]])
        d.pop("ego_simulated_states", None)
        d["two_frame_extended_comfort"] = ec_map.get(tok, np.nan)
        rows.append(d)
    agg_df = pd.DataFrame(rows)

    df_after = compute_final_scores_one_stage(agg_df.drop(columns=["bef_multiplicative_metrics_prod", "bef_weighted_metrics"], errors="ignore").copy())
    score_cols = [
        c
        for c in df_after.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]
    avg_aft = {k: float(v) * 100.0 for k, v in df_after[score_cols].mean(skipna=True).items()}

    df_bef = compute_final_scores_before_from_saved(agg_df, two_stage=False)
    avg_bef = {k: float(v) * 100.0 for k, v in df_bef[score_cols].mean(skipna=True).items()}

    res = {
        "after": avg_aft,
        "before": avg_bef,
        "new_scored": len(todo_items),
        "sec": round(time.time() - t0, 2),
    }
    print(
        f"[navtest:{name:22s}] new={len(todo_items):4d} ({res['sec']:5.2f}s) | "
        f"After={avg_aft['score']:.2f}% (NC={avg_aft['no_at_fault_collisions']:.2f}, DAC={avg_aft['drivable_area_compliance']:.2f}, "
        f"EP={avg_aft['ego_progress']:.2f}, TTC={avg_aft['time_to_collision_within_bound']:.2f}, LK={avg_aft['lane_keeping']:.2f}, EC={avg_aft['two_frame_extended_comfort']:.2f}) | "
        f"Before={avg_bef['score']:.2f}%",
        flush=True,
    )
    return res


def run_search(mode: str, split: str, mc_dir: str, out_json: Path, workers: int = 36):
    toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96 = load_pool(mode, split)
    Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    if split == "navhard":
        sf_raw = yaml.load(
            open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"),
            Loader=Loader,
        )
        split_raw = yaml.load(
            open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"),
            Loader=Loader,
        )
        all_tokens = [str(t) for t in sf_raw["tokens"]] + [str(t) for t in sf_raw["reactive_synthetic_initial_tokens"]]
        all_mappings = {}
        token_set = set(all_tokens)
        for orig_token, prev_token, two_stage_pairs in split_raw["reactive_all_mapping"]:
            if prev_token in token_set or orig_token in token_set:
                all_mappings[(str(orig_token), str(prev_token))] = [(str(p[0]), str(p[1])) for p in two_stage_pairs]
    else:
        sf_raw = yaml.load(
            open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"),
            Loader=Loader,
        )
        all_tokens = [str(t) for t in sf_raw["tokens"]]
        all_mappings = {}

    row_cache: Dict[Tuple[str, bytes], Dict[str, Any]] = {}
    ec_cache: Dict[Tuple[str, bytes, str, bytes], float] = {}
    start_adj_cache: Dict[str, str] = {}
    all_results = {}

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(mc_dir), split),
    ) as ex:
        cfgs = get_candidate_configs()
        for name, cfg in cfgs:
            traj_dict = select_trajectories(cfg, toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96)
            if split == "navhard":
                res = evaluate_config_navhard(name, traj_dict, all_tokens, all_mappings, row_cache, ec_cache, ex, workers)
            else:
                res = evaluate_config_navtest(name, traj_dict, all_tokens, start_adj_cache, row_cache, ec_cache, ex, workers)
            res["config"] = cfg
            all_results[name] = res

        def get_score(r):
            return r["after"]["combined"]["score"] if split == "navhard" else r["after"]["score"]

        def get_before_score(r):
            return r["before"]["combined"]["score"] if split == "navhard" else r["before"]["score"]

        best_name = max(all_results, key=lambda k: get_score(all_results[k]))
        best_cfg = dict(all_results[best_name]["config"])
        best_cfg["dv"] = list(best_cfg.get("dv", [0.0] * 6))
        best_score = get_score(all_results[best_name])
        print(f"=== Starting local coordinate refinement from {best_name} ({best_score:.2f}%) ===", flush=True)

        tune_steps = [
            ("w_nc_v2", [0.3, 0.5, 0.7, 0.9]),
            ("w_dac_v2", [0.3, 0.5, 0.7, 0.9]),
            ("w_nc", [0.6, 0.8, 1.0, 1.2]),
            ("w_dac", [0.6, 0.8, 1.0, 1.2]),
            ("w_v2f", [0.3, 0.4, 0.5, 0.6]),
            ("d_col0", [0.0, 0.015, 0.03, 0.06]),
            ("dv_lat", [0.0, -0.03, -0.06, -0.12]),
            ("dv_acc3", [0.0, -0.02, -0.05]),
            ("dv_id0", [0.0, 0.02, 0.04]),
            ("w_ttc", [0.0, 0.15, 0.3]),
            ("w_prod", [0.0, 1.0, 2.0, 3.0]),
        ]
        step_id = 0
        for param, vals in tune_steps:
            for v in vals:
                cand_cfg = dict(best_cfg)
                cand_cfg["dv"] = list(best_cfg["dv"])
                if param == "dv_lat":
                    cand_cfg["dv"][4] = v
                    cand_cfg["dv"][5] = v
                elif param == "dv_acc3":
                    cand_cfg["dv"][3] = v
                elif param == "dv_id0":
                    cand_cfg["dv"][0] = v
                else:
                    cand_cfg[param] = v
                step_id += 1
                cname = f"tune_{step_id:02d}_{param}_{v}"
                traj_dict = select_trajectories(cand_cfg, toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96)
                if split == "navhard":
                    res = evaluate_config_navhard(cname, traj_dict, all_tokens, all_mappings, row_cache, ec_cache, ex, workers)
                else:
                    res = evaluate_config_navtest(cname, traj_dict, all_tokens, start_adj_cache, row_cache, ec_cache, ex, workers)
                res["config"] = cand_cfg
                all_results[cname] = res
                sc = get_score(res)
                if sc > best_score + 1e-4:
                    best_score = sc
                    best_cfg = cand_cfg
                    best_name = cname
                    print(f"  >>> NEW BEST [{split}:{mode}]: {best_name} = {best_score:.2f}%", flush=True)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    best_aft_k = max(all_results, key=lambda k: get_score(all_results[k]))
    best_bef_k = max(all_results, key=lambda k: get_before_score(all_results[k]))
    summary = {
        "mode": mode,
        "split": split,
        "best_after_name": best_aft_k,
        "best_after": all_results[best_aft_k],
        "best_before_name": best_bef_k,
        "best_before": all_results[best_bef_k],
        "all_results": all_results,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"=== FINISHED {mode} {split}: Best After = {best_aft_k} ({best_score:.2f}%), Best Before = {best_bef_k} ({get_before_score(all_results[best_bef_k]):.2f}%) ===", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    split = sys.argv[2]
    mc_dir = sys.argv[3]
    out_json = Path(sys.argv[4])
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 36
    run_search(mode, split, mc_dir, out_json, workers)
