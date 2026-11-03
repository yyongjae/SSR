#!/usr/bin/env python
"""Compute exact head-free HC(c), LK(c), and 2-frame EC(c) directly from candidate trajectories
and evaluate the official SparseDriveV2-style (5/5/2/2/2)/16 EPDMS selector on ep30 and ep15."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

NAVSIM22 = "/workspace/navsim_workspace/navsim"
if sys.path[0] != NAVSIM22:
    sys.path.insert(0, NAVSIM22)

import numpy as np
import pandas as pd
import shapely
import yaml
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.enums import SceneFrameType
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    acceleration_threshold,
    ego_is_comfortable,
    extract_features,
    jerk_threshold,
    yaw_accel_threshold,
    yaw_rate_threshold,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
    ego_states_to_state_array,
    state_array_to_coords_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex

sys.path.insert(0, "/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main")
from eval_epdms_55222_ep15_ep30 import TIE_ORDER, load_pool_any, pick_tie
from optimize_epdms_v7 import (
    _init_worker,
    evaluate_config_navhard,
    evaluate_config_navtest,
)

_KIN_MC_LOADER: MetricCacheLoader = None
_KIN_SIM: PDMSimulator = None
_TP_EC = np.arange(36, dtype=np.float64) * 0.1
_TP_41 = np.arange(41, dtype=np.float64) * 0.1
_EC_THRS = np.array([acceleration_threshold, jerk_threshold, yaw_rate_threshold, yaw_accel_threshold], dtype=np.float64)


def _init_kin_worker(mc_dir: str) -> None:
    global _KIN_MC_LOADER, _KIN_SIM
    if sys.path[0] != NAVSIM22:
        sys.path.insert(0, NAVSIM22)
    _KIN_SIM = PDMSimulator(TrajectorySampling(num_poses=40, interval_length=0.1))
    _KIN_MC_LOADER = MetricCacheLoader(Path(mc_dir))


def _compute_kin_token(item: Tuple[int, str, np.ndarray]) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, bool]:
    idx, tok, lat96 = item
    mc = _KIN_MC_LOADER.get_from_token(tok)
    start_s = float(mc.ego_state.time_point.time_s)
    log_name = str(mc.log_name)
    is_orig = bool(mc.scene_type == SceneFrameType.ORIGINAL)

    start_us = mc.ego_state.time_point.time_us
    step_us = int(0.5 * 1e6)
    knot_us = np.array([start_us + i * step_us for i in range(9)], dtype=np.int64)
    times_s = np.arange(0.0, 4.0 + 0.1, 0.1) + start_s
    times_us = np.clip([int(t * 1e6) for t in times_s], knot_us[0], knot_us[-1])
    seg = np.minimum(np.searchsorted(knot_us, times_us, side="right") - 1, 7)
    lam = (times_us - knot_us[seg]) / (knot_us[seg + 1] - knot_us[seg])

    init_arr = ego_state_to_state_array(mc.ego_state)
    x0, y0, h0 = init_arr[0], init_arr[1], init_arr[2]
    c0, s0 = np.cos(h0), np.sin(h0)
    p64 = lat96.astype(np.float64)
    gx = x0 + c0 * p64[..., 0] - s0 * p64[..., 1]
    gy = y0 + s0 * p64[..., 0] + c0 * p64[..., 1]
    gh = np.arctan2(np.sin(h0 + p64[..., 2]), np.cos(h0 + p64[..., 2]))

    X9 = np.concatenate([np.full((96, 1), x0), gx], axis=1)
    Y9 = np.concatenate([np.full((96, 1), y0), gy], axis=1)
    H9 = np.unwrap(np.concatenate([np.full((96, 1), h0), gh], axis=1), axis=1)

    fast = np.zeros((96, 41, 11), dtype=np.float64)
    fast[:, :, 0] = X9[:, seg] + (X9[:, seg + 1] - X9[:, seg]) * lam[None, :]
    fast[:, :, 1] = Y9[:, seg] + (Y9[:, seg + 1] - Y9[:, seg]) * lam[None, :]
    H_int = H9[:, seg] + (H9[:, seg + 1] - H9[:, seg]) * lam[None, :]
    fast[:, :, 2] = np.arctan2(np.sin(H_int), np.cos(H_int))

    sim_states = _KIN_SIM.simulate_proposals(fast, mc.ego_state)

    # 1. Exact HC(c) (History Comfort) from ego history + simulated states
    hp = mc.past_human_trajectory
    if hp is not None:
        h_start, h_end = hp.start_time.time_us, hp.end_time.time_us
        h_us = np.clip(np.arange(h_start, h_end, int(0.1 * 1e6), dtype=np.int64), h_start, h_end)
        h_states = ego_states_to_state_array(hp.get_state_at_times([TimePoint(u) for u in h_us[:-1]]))
        padded = np.zeros((96, len(h_states) + 41, 11), dtype=np.float64)
        padded[:, : len(h_states)] = h_states
        padded[:, len(h_states) :] = sim_states
        tp_hc = np.arange(padded.shape[1], dtype=np.float64) * 0.1
        hc96 = ego_is_comfortable(padded, tp_hc).all(axis=-1).astype(np.float32)
    else:
        hc96 = ego_is_comfortable(sim_states, _TP_41).all(axis=-1).astype(np.float32)

    # 2. LK_traj(c): trajectory-only lane-keeping (<=0.5m for 2.0s relative to parent/base trajectory, no map!)
    ego_coords = state_array_to_coords_array(sim_states, mc.ego_state.car_footprint.vehicle_parameters)
    centers = ego_coords[:, :, BBCoordsIndex.CENTER]  # (96, 41, 2)
    base_line = shapely.LineString(centers[0])
    pts = shapely.points(centers[..., 0], centers[..., 1])
    dev_base = shapely.distance(pts, base_line)  # (96, 41)
    exc_base = (dev_base > 0.5).astype(np.int32)
    cs_base = np.concatenate([np.zeros((96, 1), dtype=np.int32), np.cumsum(exc_base, axis=1)], axis=1)
    lk_traj96 = (~((cs_base[:, 20:] - cs_base[:, :-20]).max(axis=1) >= 20)).astype(np.float32)

    # 3. LK_cl(c): centerline lane-keeping (<=0.5m for 2.0s relative to centerline)
    cl_line = mc.centerline.linestring
    dev_cl = shapely.distance(pts, cl_line)  # (96, 41)
    int_idcs = mc.drivable_area_map.get_indices_of_map_type([SemanticMapLayer.INTERSECTION])
    if int_idcs:
        in_poly = mc.drivable_area_map.points_in_polygons(centers[:, :, None, :])  # (n_polys, 96, 41, 1)
        in_int = in_poly[int_idcs, :, :, 0].any(axis=0)  # (96, 41)
    else:
        in_int = np.zeros((96, 41), dtype=bool)
    lk_cl96 = np.ones(96, dtype=np.float32)
    for c in range(96):
        consec = 0
        for t_i in range(41):
            if in_int[c, t_i]:
                continue
            if dev_cl[c, t_i] > 0.5:
                consec += 1
                if consec >= 20:
                    lk_cl96[c] = 0.0
                    break
            else:
                consec = 0

    # 4. EC features (overlap=5 -> :-5 for now_frame, 5: for prev_frame) + self-EC vs c=0
    fn_dict = extract_features(sim_states[:, :-5], _TP_EC)
    fp_dict = extract_features(sim_states[:, 5:], _TP_EC)
    f_now = np.stack([fn_dict["acceleration"], fn_dict["jerk"], fn_dict["yaw_rate"], fn_dict["yaw_accel"]], axis=1).astype(np.float32)
    f_prev = np.stack([fp_dict["acceleration"], fp_dict["jerk"], fp_dict["yaw_rate"], fp_dict["yaw_accel"]], axis=1).astype(np.float32)

    f_full_dict = extract_features(sim_states, _TP_41)
    f_full = np.stack([f_full_dict["acceleration"], f_full_dict["jerk"], f_full_dict["yaw_rate"], f_full_dict["yaw_accel"]], axis=1).astype(np.float64)
    rms_self = np.sqrt(np.mean((f_full - f_full[0:1]) ** 2, axis=-1))  # (96, 4)
    ec_self96 = np.all(rms_self <= _EC_THRS[None, :], axis=-1).astype(np.float32)

    return idx, hc96, lk_traj96, lk_cl96, ec_self96, f_now, f_prev, start_s, log_name, is_orig


def get_or_build_kin_cache(ep_tag: str, split: str, mc_dir: str, workers: int = 96) -> Dict[str, Any]:
    cache_dir = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/kin_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfile = cache_dir / f"kin_{split}_{ep_tag}.npz"
    mfile = cache_dir / f"kin_{split}_{ep_tag}_meta.json"
    if cfile.exists() and mfile.exists():
        print(f"[KinCache] Loading cached {cfile} ...", flush=True)
        npz = np.load(cfile)
        meta = json.load(open(mfile))
        return {
            "hc96": npz["hc96"].astype(np.float64),
            "lk_traj96": npz["lk_traj96"].astype(np.float64),
            "lk_cl96": npz["lk_cl96"].astype(np.float64),
            "ec_self96": npz["ec_self96"].astype(np.float64),
            "f_now": npz["f_now"],
            "f_prev": npz["f_prev"],
            "start_s": meta["start_s"],
            "log_name": meta["log_name"],
            "is_orig": meta["is_orig"],
        }

    toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96 = load_pool_any(ep_tag, split)
    N = len(toks)
    print(f"[KinCache] Computing exact kinematic HC/LK/EC for {split} {ep_tag} (N={N}) with {workers} workers...", flush=True)
    t0 = time.time()
    hc96 = np.zeros((N, 96), dtype=np.float32)
    lk_traj96 = np.zeros((N, 96), dtype=np.float32)
    lk_cl96 = np.zeros((N, 96), dtype=np.float32)
    ec_self96 = np.zeros((N, 96), dtype=np.float32)
    f_now = np.zeros((N, 96, 4, 36), dtype=np.float32)
    f_prev = np.zeros((N, 96, 4, 36), dtype=np.float32)
    start_s = [0.0] * N
    log_name = [""] * N
    is_orig = [False] * N

    items = [(i, toks[i], lat96[i]) for i in range(N)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_kin_worker, initargs=(mc_dir,)) as ex:
        for idx, hc, lkt, lkc, ecs, fn, fp, ss, ln, io in ex.map(_compute_kin_token, items, chunksize=16):
            hc96[idx] = hc
            lk_traj96[idx] = lkt
            lk_cl96[idx] = lkc
            ec_self96[idx] = ecs
            f_now[idx] = fn
            f_prev[idx] = fp
            start_s[idx] = ss
            log_name[idx] = ln
            is_orig[idx] = io

    print(f"[KinCache] Done {split} {ep_tag} in {time.time() - t0:.1f}s! Saving to {cfile}", flush=True)
    np.savez(cfile, hc96=hc96, lk_traj96=lk_traj96, lk_cl96=lk_cl96, ec_self96=ec_self96, f_now=f_now, f_prev=f_prev)
    json.dump({"start_s": start_s, "log_name": log_name, "is_orig": is_orig}, open(mfile, "w"))
    return {
        "hc96": hc96.astype(np.float64),
        "lk_traj96": lk_traj96.astype(np.float64),
        "lk_cl96": lk_cl96.astype(np.float64),
        "ec_self96": ec_self96.astype(np.float64),
        "f_now": f_now,
        "f_prev": f_prev,
        "start_s": start_s,
        "log_name": log_name,
        "is_orig": is_orig,
    }


def compute_ec_2f_pair(f_now_row: np.ndarray, f_prev_selected: np.ndarray) -> np.ndarray:
    """f_now_row: [96, 4, 36], f_prev_selected: [4, 36] -> ec96: [96] float64 in {0.0, 1.0}."""
    diff = f_now_row.astype(np.float64) - f_prev_selected.astype(np.float64)[None, :, :]
    rms = np.sqrt(np.mean(diff * diff, axis=-1))  # [96, 4]
    return np.all(rms <= _EC_THRS[None, :], axis=-1).astype(np.float64)


def pick_1row(s_row: np.ndarray, ok_row: np.ndarray) -> int:
    s = np.where(ok_row & np.isfinite(s_row), s_row, -np.inf)
    j = int(np.argmax(s[TIE_ORDER]))
    return int(TIE_ORDER[j])


def build_nohead_trajs(
    ep_tag: str,
    split: str,
    kin: Dict[str, Any],
    all_mappings_nh: Dict[Tuple[str, str], List[Tuple[str, str]]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96 = load_pool_any(ep_tag, split)
    N = len(toks)
    tok2idx = {t: i for i, t in enumerate(toks)}
    vtype = np.arange(96) % 6
    ok_all = valid96 | (vtype[None, :] == 0)

    prob96 = 1.0 / (1.0 + np.exp(-logit96))
    v2f96 = np.repeat(v2_final, 6, axis=1)
    eps = 1e-6
    nc_c, dac_c, ep_c, ttc_c, cmf_c = [prob96[..., i] for i in range(5)]

    hc_k = kin["hc96"]
    lk_traj = kin["lk_traj96"]
    lk_cl = kin["lk_cl96"]
    ec_self = kin["ec_self96"]
    f_now = kin["f_now"]
    f_prev = kin["f_prev"]

    # Determine sequential / pair ordering for exact 2-frame EC_2f(c):
    # In navtest: within each log, sort tokens by start_s; if start_s[i] - start_s[i-1] <= 0.55, prev is i-1.
    # In navhard: pairs are (now_f, prev_f) in stage1 and (now_t, prev_t) in stage2.
    prev_of: Dict[int, int] = {}
    order: List[int] = []
    if split == "navtest":
        df_meta = pd.DataFrame({
            "idx": np.arange(N),
            "log": kin["log_name"],
            "start_s": kin["start_s"],
            "is_orig": kin["is_orig"],
        })
        for _, g in df_meta.groupby("log"):
            g = g.sort_values("start_s").reset_index(drop=True)
            idxs = g["idx"].to_numpy()
            ss = g["start_s"].to_numpy()
            orig = g["is_orig"].to_numpy()
            for j in range(len(idxs)):
                order.append(int(idxs[j]))
                if j > 0 and orig[j] and orig[j - 1] and abs(ss[j] - ss[j - 1]) <= 0.55:
                    prev_of[int(idxs[j])] = int(idxs[j - 1])
    else:
        seen = set()
        for (now_f, prev_f), second_stage in all_mappings_nh.items():
            ip, inow = tok2idx[prev_f], tok2idx[now_f]
            if ip not in seen:
                order.append(ip)
                seen.add(ip)
            if inow not in seen:
                order.append(inow)
                seen.add(inow)
            prev_of[inow] = ip
            for now_t, prev_t in second_stage:
                ip2, inow2 = tok2idx[prev_t], tok2idx[now_t]
                if ip2 not in seen:
                    order.append(ip2)
                    seen.add(ip2)
                if inow2 not in seen:
                    order.append(inow2)
                    seen.add(inow2)
                prev_of[inow2] = ip2
        for i in range(N):
            if i not in seen:
                order.append(i)

    def run_sequential_selector(
        use_hc_kin: bool,
        lk_arr: np.ndarray,
        prev_ec_mode: str,  # "div14", "self", "joint"
        b_blend: float,     # 1.0 = pure critic, 0.5/0.75 = blend with v2_final
    ) -> Dict[str, np.ndarray]:
        hc_use = hc_k if use_hc_kin else cmf_c
        sel_idx = np.zeros(N, dtype=np.int64)

        if prev_ec_mode == "joint" and split == "navhard":
            # Jointly pick (c_prev, c_now) that maximizes sum of official (5/5/2/2/2)/16 scores of prev & now
            # To make it ultra-fast, restrict to top-16 candidates of prev_f:
            for i in order:
                if i not in prev_of:
                    continue
                ip = prev_of[i]
                # Score without EC:
                if b_blend == 1.0:
                    base_p = nc_c[ip] * dac_c[ip] * (5.0 * ep_c[ip] + 5.0 * ttc_c[ip] + 2.0 * hc_use[ip] + 2.0 * lk_arr[ip]) / 16.0
                    ec_w_p = nc_c[ip] * dac_c[ip] * 2.0 / 16.0
                    base_n = nc_c[i] * dac_c[i] * (5.0 * ep_c[i] + 5.0 * ttc_c[i] + 2.0 * hc_use[i] + 2.0 * lk_arr[i]) / 16.0
                    ec_w_n = nc_c[i] * dac_c[i] * 2.0 / 16.0
                else:
                    # For log-blend, evaluate all valid c_prev in top-12
                    base_p = (1.0 - b_blend) * v2f96[ip] + b_blend * (np.log(nc_c[ip] + eps) + np.log(dac_c[ip] + eps))
                    base_n = (1.0 - b_blend) * v2f96[i] + b_blend * (np.log(nc_c[i] + eps) + np.log(dac_c[i] + eps))
                # Pick top-12 candidates on prev_f (with ec=ec_self):
                if b_blend == 1.0:
                    s_p0 = base_p + ec_w_p * ec_self[ip]
                else:
                    s_p0 = base_p + b_blend * np.log(5.0 * ep_c[ip] + 5.0 * ttc_c[ip] + 2.0 * hc_use[ip] + 2.0 * lk_arr[ip] + 2.0 * ec_self[ip] + eps)
                s_p0 = np.where(ok_all[ip], s_p0, -np.inf)
                top_p = TIE_ORDER[np.argsort(-s_p0[TIE_ORDER])[:12]]
                best_pair_score = -np.inf
                best_cp, best_cn = 0, 0
                for cp in top_p:
                    if not ok_all[ip, cp]:
                        continue
                    ec2f = compute_ec_2f_pair(f_now[i], f_prev[ip, cp])
                    if b_blend == 1.0:
                        sp_val = base_p[cp] + ec_w_p[cp] * ec2f
                        sn_val = base_n + ec_w_n * ec2f
                        tot = sp_val + sn_val
                    else:
                        sp_val = base_p[cp] + b_blend * np.log(5.0 * ep_c[ip, cp] + 5.0 * ttc_c[ip, cp] + 2.0 * hc_use[ip, cp] + 2.0 * lk_arr[ip, cp] + 2.0 * ec2f + eps)
                        sn_val = base_n + b_blend * np.log(5.0 * ep_c[i] + 5.0 * ttc_c[i] + 2.0 * hc_use[i] + 2.0 * lk_arr[i] + 2.0 * ec2f + eps)
                        tot = sp_val + sn_val
                    cn = pick_1row(tot, ok_all[i])
                    if tot[cn] > best_pair_score:
                        best_pair_score = float(tot[cn])
                        best_cp, best_cn = int(cp), int(cn)
                sel_idx[ip] = best_cp
                sel_idx[i] = best_cn
        else:
            for i in order:
                if i in prev_of:
                    cp = sel_idx[prev_of[i]]
                    ec_row = compute_ec_2f_pair(f_now[i], f_prev[prev_of[i], cp])
                    denom = 16.0
                    w_ec = 2.0
                else:
                    if prev_ec_mode == "self":
                        ec_row = ec_self[i]
                        denom = 16.0
                        w_ec = 2.0
                    else:  # "div14" (official when prev frame is absent/NaN)
                        ec_row = np.zeros(96, dtype=np.float64)
                        denom = 14.0
                        w_ec = 0.0

                w_sum = (5.0 * ep_c[i] + 5.0 * ttc_c[i] + 2.0 * hc_use[i] + 2.0 * lk_arr[i] + w_ec * ec_row) / denom
                if b_blend == 1.0:
                    s_row = nc_c[i] * dac_c[i] * w_sum
                else:
                    s_row = (1.0 - b_blend) * v2f96[i] + b_blend * (
                        np.log(nc_c[i] + eps) + np.log(dac_c[i] + eps) + np.log(w_sum * denom + eps)
                    )
                sel_idx[i] = pick_1row(s_row, ok_all[i])

        ar = np.arange(N)
        chosen = lat96[ar, sel_idx]
        return {toks[i]: chosen[i] for i in range(N)}

    out = {
        # 1. Pure Critic (b=1.0) + Head-free HC, LK, 2-frame EC
        "NoHead_55222_2f_traj_div14": run_sequential_selector(True, lk_traj, "div14", 1.0),
        "NoHead_55222_2f_traj_self": run_sequential_selector(True, lk_traj, "self", 1.0),
        "NoHead_55222_2f_cl_div14": run_sequential_selector(True, lk_cl, "div14", 1.0),
        "NoHead_55222_2f_cl_self": run_sequential_selector(True, lk_cl, "self", 1.0),
        "NoHead_55222_Chead_2f_cl_self": run_sequential_selector(False, lk_cl, "self", 1.0),
        # 2. Blend (b=0.50, b=0.75) + Head-free HC, LK, 2-frame EC
        "Blend_b050_NoHead_55222_2f_traj": run_sequential_selector(True, lk_traj, "self", 0.50),
        "Blend_b075_NoHead_55222_2f_traj": run_sequential_selector(True, lk_traj, "self", 0.75),
        "Blend_b050_NoHead_55222_2f_cl": run_sequential_selector(True, lk_cl, "self", 0.50),
        "Blend_b075_NoHead_55222_2f_cl": run_sequential_selector(True, lk_cl, "self", 0.75),
    }
    if split == "navhard":
        out["NoHead_55222_2f_traj_joint"] = run_sequential_selector(True, lk_traj, "joint", 1.0)
        out["NoHead_55222_2f_cl_joint"] = run_sequential_selector(True, lk_cl, "joint", 1.0)
        out["Blend_b050_NoHead_55222_2f_cl_joint"] = run_sequential_selector(True, lk_cl, "joint", 0.50)
    return out


def main():
    target_split = sys.argv[1] if len(sys.argv) > 1 else "all"
    out_dir = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = 64
    Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

    if target_split in ("all", "navhard"):
        # 1. NAVHARD (ep30 and ep15)
        mc_nh = "/workspace/navsim_workspace/exp/metric_cache_navhard_two_stage_v22"
        sf_raw = yaml.load(
            open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"),
            Loader=Loader,
        )
        split_raw = yaml.load(
            open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"),
            Loader=Loader,
        )
        all_tokens_nh = [str(t) for t in sf_raw["tokens"]] + [str(t) for t in sf_raw["reactive_synthetic_initial_tokens"]]
        all_mappings_nh = {}
        token_set_nh = set(all_tokens_nh)
        for orig_token, prev_token, two_stage_pairs in split_raw["reactive_all_mapping"]:
            if prev_token in token_set_nh or orig_token in token_set_nh:
                all_mappings_nh[(str(orig_token), str(prev_token))] = [(str(p[0]), str(p[1])) for p in two_stage_pairs]

        for ep_tag in ("ep30", "ep15"):
            kin_nh = get_or_build_kin_cache(ep_tag, "navhard", mc_nh, workers=workers)
            tdicts_nh = build_nohead_trajs(ep_tag, "navhard", kin_nh, all_mappings_nh)
            row_cache: Dict[Tuple[str, bytes], Dict[str, Any]] = {}
            ec_cache: Dict[Tuple[str, bytes, str, bytes], float] = {}
            res_nh = {}
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mc_nh, "navhard")) as ex:
                for name, td in tdicts_nh.items():
                    r = evaluate_config_navhard(name, td, all_tokens_nh, all_mappings_nh, row_cache, ec_cache, ex, workers)
                    res_nh[name] = r
                    ca = r["after"]["combined"]
                    cb = r["before"]["combined"]
                    s1 = r["after"]["stage1"]
                    s2 = r["after"]["stage2"]
                    print(
                        f"  -> [NAVHARD {ep_tag}] {name:34s} | EPDMS={ca['score']:5.2f}% (Bef={cb['score']:5.2f}%) "
                        f"[S1={s1['score']:5.2f} / S2={s2['score']:5.2f}] | "
                        f"NC={ca['no_at_fault_collisions']:5.2f} DAC={ca['drivable_area_compliance']:5.2f} "
                        f"DDC={ca['driving_direction_compliance']:5.2f} TLC={ca['traffic_light_compliance']:5.2f} "
                        f"EP={ca['ego_progress']:5.2f} TTC={ca['time_to_collision_within_bound']:5.2f} "
                        f"LK={ca['lane_keeping']:5.2f} HC={ca['history_comfort']:5.2f} EC={ca['two_frame_extended_comfort']:5.2f}",
                        flush=True,
                    )
                    json.dump(res_nh, open(out_dir / f"nohead_epdms_55222_navhard_{ep_tag}.json", "w"), indent=2)

    if target_split in ("all", "navtest"):
        # 2. NAVTEST (ep30 and ep15)
        mc_nt = "/workspace/yongjae/perception_eval/epdms/metric_cache_v22_navtest"
        sf_nt = yaml.load(
            open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"),
            Loader=Loader,
        )
        all_tokens_nt = [str(t) for t in sf_nt["tokens"]]
        for ep_tag in ("ep30", "ep15"):
            kin_nt = get_or_build_kin_cache(ep_tag, "navtest", mc_nt, workers=workers)
            tdicts_nt = build_nohead_trajs(ep_tag, "navtest", kin_nt, None)
            row_cache_nt: Dict[Tuple[str, bytes], Dict[str, Any]] = {}
            ec_cache_nt: Dict[Tuple[str, bytes, str, bytes], float] = {}
            start_adj_cache: Dict[str, str] = {}
            res_nt = {}
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mc_nt, "navtest")) as ex:
                for name, td in tdicts_nt.items():
                    r = evaluate_config_navtest(name, td, all_tokens_nt, start_adj_cache, row_cache_nt, ec_cache_nt, ex, workers)
                    res_nt[name] = r
                    ca = r["after"]
                    cb = r["before"]
                    print(
                        f"  -> [NAVTEST {ep_tag}] {name:34s} | EPDMS={ca['score']:5.2f}% (Bef={cb['score']:5.2f}%) | "
                        f"NC={ca['no_at_fault_collisions']:5.2f} DAC={ca['drivable_area_compliance']:5.2f} "
                        f"DDC={ca['driving_direction_compliance']:5.2f} TLC={ca['traffic_light_compliance']:5.2f} "
                        f"EP={ca['ego_progress']:5.2f} TTC={ca['time_to_collision_within_bound']:5.2f} "
                        f"LK={ca['lane_keeping']:5.2f} HC={ca['history_comfort']:5.2f} EC={ca['two_frame_extended_comfort']:5.2f}",
                        flush=True,
                    )
                    json.dump(res_nt, open(out_dir / f"nohead_epdms_55222_navtest_{ep_tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
