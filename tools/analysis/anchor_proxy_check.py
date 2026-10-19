#!/usr/bin/env python
"""Direction-D premise check: can GT-geometry proxies reproduce PDM anchor labels on navtrain?

For a random navtrain token sample, every one of the 256 WoTE anchors is scored four ways:
  wote_*   the WoTE labels used as the v2 training target (pdm_score_256.npy)
  off_*    the official PDMScorer re-run here (metric cache built on the fly, LQR-tracked, 256 in one batch)
  raw_*    the official scorer on the UNTRACKED anchor (geometry only; NC needs velocities -> not reported)
  proxies  continuous GT-geometry quantities computable at training time:
             dac_margin     min signed distance (m) of the ego footprint corners to the drivable-area boundary
                            (+ inside, - outside), over the 4 s horizon
             dac_margin_32  the same restricted to time steps whose anchor pose lies in the BEV map range
                            (x_forward <= 32 m, |y| <= 32 m); nan if no step is in range
             nc_dist        min distance (m) between the ego footprint and any GT agent polygon over the horizon
Writes one row per (token, anchor) and prints agreement / AUC summaries.

usage: python anchor_proxy_check.py --n-tokens 1000 --seed 0 --workers 24 --out <csv>
"""
from __future__ import annotations

import argparse
import lzma
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("SSR_REPO", Path.home() / "SSR"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(REPO / "data/dataset/maps"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(REPO / "data/dataset"))
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(REPO))
os.environ.setdefault("NAVSIM_EXP_ROOT", str(REPO / "work_dirs"))

ANCHORS = REPO / "data/planning_vb/trajectory_anchors_256.npy"
LABELS = REPO / "data/planning_vb/pdm_score_256"
SIM_KEYS = ("nc", "dac", "ep", "ttc", "c")   # pdm_score_256.npy order (anchor_planner.SIM_KEYS)
MAP_X, MAP_Y = 32.0, 32.0                    # PROTOCOL_MAP_PC_RANGE: x_forward 0..32, y +-32

_G = {}


def _init(cache_dir: str):
    import torch
    torch.set_num_threads(1)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor

    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/pdm_scoring"), version_base="1.2"):
        cfg = compose(config_name="default_run_pdm_score")
    _G["sim"] = instantiate(cfg.simulator)
    _G["scorer"] = instantiate(cfg.scorer)
    _G["proc"] = MetricCacheProcessor(cache_path=cache_dir, force_feature_computation=False)
    _G["anchors"] = np.load(ANCHORS).astype(np.float64)


def _one(args):
    token, scene_dict, wote = args
    import shapely
    from shapely.ops import unary_union
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score_multi_trajs, transform_trajectory
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex as M

    sim, scorer, proc, anchors = _G["sim"], _G["scorer"], _G["proc"], _G["anchors"]
    try:
        scene = Scene.from_scene_dict_list(scene_dict, None, num_history_frames=4, num_future_frames=10,
                                           sensor_config=SensorConfig.build_no_sensors())
        scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
        proc.compute_metric_cache(scenario)
        path = proc._cache_path / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl"
        with lzma.open(path, "rb") as f:
            mc = pickle.load(f)
        ego0 = mc.ego_state
        ps = sim.proposal_sampling

        # official (tracked, 256 in one batch like WoTE)
        off = pdm_score_multi_trajs(mc, [a.astype(np.float32) for a in anchors], ps, sim, scorer)
        off_arr = {"nc": off.no_at_fault_collisions, "dac": off.drivable_area_compliance, "ddc": off.driving_direction_compliance,
                   "ep": off.ego_progress, "ttc": off.time_to_collision_within_bound, "c": off.comfort, "score": off.score}

        # raw (untracked) states + scorer geometry
        states = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(a.astype(np.float32)), ego0), ps, ego0.time_point)
                           for a in anchors])
        scorer.score_proposals(states.copy(), mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
        raw_dac = scorer._multi_metrics[M.DRIVABLE_AREA].copy()
        raw_ddc = scorer._multi_metrics[M.DRIVING_DIRECTION].copy()
        coords = scorer._ego_coords.copy()        # [K, T, 5, 2]
        polys = scorer._ego_polygons.copy()       # [K, T]
        K, T = polys.shape

        # drivable-area margin
        dm = mc.drivable_area_map
        idx = dm.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                                          SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
        U = unary_union([dm._geometries[i] for i in idx]) if idx else None
        corners = coords[:, :, :4, :]                                   # [K, T, 4, 2]
        if U is None or U.is_empty:
            signed = np.full(corners.shape[:3], -np.inf)
        else:
            pts = shapely.points(corners.reshape(-1, 2))
            d = shapely.distance(pts, U.boundary)
            inside = shapely.contains_xy(U, corners[..., 0].ravel(), corners[..., 1].ravel())
            signed = np.where(inside, d, -d).reshape(corners.shape[:3])
        step_margin = signed.min(axis=2)                                 # [K, T]
        dac_margin = step_margin.min(axis=1)

        # BEV-range restriction, from the anchor poses (ego frame, x forward) interpolated to 0.1 s
        t_anchor = np.arange(1, 9) * 0.5
        t_step = np.arange(T) * ps.interval_length
        ax = np.stack([np.interp(t_step, np.r_[0, t_anchor], np.r_[0, a[:, 0]]) for a in anchors])
        ay = np.stack([np.interp(t_step, np.r_[0, t_anchor], np.r_[0, a[:, 1]]) for a in anchors])
        in_range = (ax <= MAP_X) & (np.abs(ay) <= MAP_Y)
        m32 = np.where(in_range, step_margin, np.inf).min(axis=1)
        m32 = np.where(np.isinf(m32), np.nan, m32)
        out_of_range = (~in_range).any(axis=1)
        viol_beyond = (dac_margin <= 0) & ((np.where(in_range, step_margin, np.inf).min(axis=1)) > 0)

        # GT-agent distance
        red = mc.observation.red_light_token
        nc_dist = np.full(K, np.inf)
        for t in range(T):
            occ = mc.observation[t]
            geoms = [g for tok, g in zip(occ.tokens, occ._geometries) if red not in tok]
            if not geoms:
                continue
            dmat = shapely.distance(np.asarray(polys[:, t], dtype=object)[:, None], np.asarray(geoms, dtype=object)[None, :])
            nc_dist = np.minimum(nc_dist, dmat.min(axis=1))

        rows = []
        for k in range(K):
            r = {"token": token, "anchor": k}
            for i, key in enumerate(SIM_KEYS):
                r[f"wote_{key}"] = float(wote[i, k])
            for key, arr in off_arr.items():
                r[f"off_{key}"] = float(arr[k])
            r.update(raw_dac=float(raw_dac[k]), raw_ddc=float(raw_ddc[k]), dac_margin=float(dac_margin[k]),
                     dac_margin_32=float(m32[k]), out_of_range=bool(out_of_range[k]), viol_beyond_range=bool(viol_beyond[k]),
                     nc_dist=float(nc_dist[k]) if np.isfinite(nc_dist[k]) else 99.0)
            rows.append(r)
        return rows
    except Exception as exc:  # noqa: BLE001
        import traceback
        return [{"token": token, "anchor": -1, "error": repr(exc), "tb": traceback.format_exc()[-800:]}]


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    return float(roc_auc_score(label, score))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tokens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=REPO / "work_dirs/eval/navtrain_cache_sample")
    a = ap.parse_args()

    import json
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    tokens_all = json.load(open(str(LABELS) + ".tokens.json"))
    labels = np.load(str(LABELS) + ".npy", mmap_mode="r")
    rng = np.random.default_rng(a.seed)
    sel = sorted(rng.choice(len(tokens_all), size=a.n_tokens, replace=False).tolist())
    sel_tokens = [tokens_all[i] for i in sel]

    scene_filter = instantiate(OmegaConf.load(REPO / "navsim/planning/script/config/common/scene_filter/navtrain.yaml"))
    scene_filter.tokens = sel_tokens
    loader = SceneLoader(sensor_blobs_path=None, data_path=REPO / "data/dataset/navsim_logs/trainval",
                         scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors())
    found = [t for t in sel_tokens if t in loader.scene_frames_dicts]
    print(f"sampled {len(sel_tokens)} WoTE tokens, {len(found)} found in navtrain logs", flush=True)
    jobs = [(t, loader.scene_frames_dicts[t], np.asarray(labels[tokens_all.index(t)], dtype=np.float32)) for t in found]

    a.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(str(a.cache_dir),)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=1), 1):
            rows.extend(r)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    err = df[df.anchor == -1]
    if len(err):
        print(f"{len(err)} tokens failed; first: {err.iloc[0]['error']}\n{err.iloc[0]['tb']}")
    df = df[df.anchor >= 0].drop(columns=[c for c in ("error", "tb") if c in df.columns])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    n_tok = df.token.nunique()
    print(f"\n{len(df)} anchor rows from {n_tok} tokens\n")

    print("--- 1. official re-scoring vs WoTE label (sanity of the pipeline) ---")
    for key in ("nc", "dac", "ttc", "c"):
        agree = (np.isclose(df[f"off_{key}"], df[f"wote_{key}"])).mean()
        print(f"  {key:4s} agreement {agree:.4f}   (WoTE fail rate {(df[f'wote_{key}'] < 1).mean():.3f}, official {(df[f'off_{key}'] < 1).mean():.3f})")
    print(f"  ep   corr {np.corrcoef(df.off_ep, df.wote_ep)[0, 1]:.4f}, mean |diff| {np.abs(df.off_ep - df.wote_ep).mean():.4f}")

    y_dac = df.off_dac < 1
    y_nc = df.off_nc < 1
    print("\n--- 2. untracked anchor vs tracked (official) DAC ---")
    print(f"  raw DAC fail {int((df.raw_dac < 1).sum())} | tracked fail {int(y_dac.sum())} | agreement {(np.isclose(df.raw_dac, df.off_dac)).mean():.4f}"
          f" | tracked-only {int((y_dac & (df.raw_dac == 1)).sum())} | raw-only {int((~y_dac & (df.raw_dac < 1)).sum())}")

    print("\n--- 3. DAC proxies vs official DAC (label = fail) ---")
    print(f"  dac_margin     AUC {auc(-df.dac_margin, y_dac):.4f} | acc@0 {((df.dac_margin <= 0) == y_dac).mean():.4f}")
    oor = df.out_of_range.mean()
    m32ok = df.dac_margin_32.notna()
    print(f"  anchors leaving the 32 m map range: {oor:.1%}; violations only beyond range: {df.viol_beyond_range.mean():.2%} of anchors, "
          f"{df.viol_beyond_range.sum() / max(1, y_dac.sum()):.1%} of DAC failures")
    print(f"  dac_margin_32  AUC {auc(-df.dac_margin_32[m32ok], y_dac[m32ok]):.4f} | acc@0 {((df.dac_margin_32[m32ok] <= 0) == y_dac[m32ok]).mean():.4f}")
    for thr in (0.0, 0.3, 0.5):
        print(f"  margin <= {thr:.1f} m: {(df.dac_margin <= thr).mean():.3f} of anchors;  among DAC pass: {(df.dac_margin[~y_dac] <= thr).mean():.3f} (scraping)")
    print(f"  DAC-fail anchors with shallow violation (margin > -0.3 m): {(df.dac_margin[y_dac] > -0.3).mean():.3f}")

    print("\n--- 4. NC proxy vs official NC (label = any collision) ---")
    print(f"  nc_dist AUC {auc(-df.nc_dist, y_nc):.4f} | acc@0 {((df.nc_dist <= 0) == y_nc).mean():.4f} | "
          f"overlap-but-not-at-fault {int(((df.nc_dist <= 0) & ~y_nc).sum())} | at-fault-without-overlap {int(((df.nc_dist > 0) & y_nc).sum())}")
    print(f"  among NC pass: nc_dist < 0.5 m {(df.nc_dist[~y_nc] < 0.5).mean():.3f}, < 1.0 m {(df.nc_dist[~y_nc] < 1.0).mean():.3f}")
    ttc_fail = df.off_ttc < 1
    print(f"  TTC fail vs nc_dist AUC {auc(-df.nc_dist, ttc_fail):.4f}")

    print("\n--- 5. per-token: how many of the 256 anchors are safe (NC=DAC=1) ---")
    safe = df.groupby("token").apply(lambda g: int(((g.off_nc == 1) & (g.off_dac == 1)).sum()))
    print(f"  median {safe.median():.0f}, p10 {safe.quantile(0.1):.0f}, tokens with 0 safe anchors {(safe == 0).mean():.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
