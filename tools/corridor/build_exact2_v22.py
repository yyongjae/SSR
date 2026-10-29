#!/usr/bin/env python
"""Build exact2 (NAVSIM v2.2 EPDMS tables) with 100% bit-exact v2.2 agent_rows, pdm, red_seg, red_t, hum, past.

Static map / initial ego tables (drv_seg, lane_seg, int_seg, route_seg, cl, ego) are identical between
NAVSIM v1 and v2.2 (verified 0.0% mismatch across 12,146 navtest scenes) and are symlinked from <exact_dir>/<split>/.
Dynamic / v2.2 tables (agent_rows, red_seg, red_t, pdm, hum, past) are computed using NAVSIM v2.2's
MetricCacheProcessor + PDMClosedPlanner + PDMObservation (reusing the cached drivable_area_map from v1 metric_cache).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
NV2 = "/workspace/navsim_workspace/navsim"
if CK not in sys.path:
    sys.path.insert(0, CK)

from navsim.agents.para_ssr.ck.constants import SPLITS  # noqa: E402
from tools.corridor.build_corridor_dt01 import load_mc  # noqa: E402
import tools.corridor.build_exact as BE  # noqa: E402

if Path(NV2).exists():
    for k in [k for k in sys.modules if k == "navsim" or k.startswith("navsim.")]:
        del sys.modules[k]
    sys.path.insert(0, NV2)

SHARED_TABLES = ("drv_seg", "lane_seg", "int_seg", "route_seg", "cl", "ego")
V2_TABLES = ("agent_rows", "red_seg", "red_t", "pdm", "hum", "past")
V2_WIDTHS = {"agent_rows": 9, "red_seg": 5, "red_t": 41, "pdm": 9, "hum": 3, "past": 9}

_PROC = None
_SCFG = None
_MAP_ROOT = "/home/external-user/yongjae/SSR/data/dataset/maps"
if not Path(_MAP_ROOT).exists():
    _MAP_ROOT = os.environ.get("NUPLAN_MAPS_ROOT", "/workspace/navsim_workspace/dataset/maps")


def _preload_maps():
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    layers = ("lane_groups_polygons", "lane_group_connectors", "lanes_polygons", "lane_connectors", "baseline_paths")
    for m in ("sg-one-north", "us-ma-boston", "us-nv-las-vegas-strip", "us-pa-pittsburgh-hazelwood"):
        ma = get_maps_api(_MAP_ROOT, "nuplan-maps-v1.0", m)
        for lyr in layers:
            ma._load_vector_map_layer(lyr)


def _extract_v2_tables(mc):
    import shapely
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from nuplan.planning.simulation.observation.idm.utils import is_track_stopped

    ra = mc.ego_state.rear_axle
    c, s = np.cos(ra.heading), np.sin(ra.heading)

    def ego(xy):
        dx, dy = xy[:, 0] - ra.x, xy[:, 1] - ra.y
        return np.stack([c * dx + s * dy, -s * dx + c * dy], -1)

    cx = np.array([BE.CROP_X[0], BE.CROP_X[1], BE.CROP_X[1], BE.CROP_X[0]])
    cy = np.array([BE.CROP_Y[0], BE.CROP_Y[0], BE.CROP_Y[1], BE.CROP_Y[1]])
    crop = shapely.Polygon(np.stack([ra.x + c * cx - s * cy, ra.y + s * cx + c * cy], -1))

    obs = mc.observation
    red, coll = obs.red_light_token, set(obs.collided_track_ids)
    uo = obs.unique_objects
    nt = min(BE.T_DYN, len(obs._global_to_local_idcs) - 1)

    red_geo, red_t = {}, {}
    for t in range(0, min(40, nt) + 1):
        om = obs[t]
        for i in om.query(crop):
            tk = om._tokens[i]
            if tk.startswith(red):
                if tk in red_geo and not red_geo[tk].equals_exact(om._geometries[i], 1e-6):
                    tk = f"{tk}@{t}"
                red_geo.setdefault(tk, om._geometries[i])
                red_t.setdefault(tk, np.zeros(41, np.float32))[t] = 1.0

    keys = list(red_geo)
    segs, pid, rt = [], 0, []
    for k in keys:
        g = shapely.simplify(shapely.intersection(red_geo[k], crop), BE.SIMPLIFY, preserve_topology=True)
        if g.is_empty:
            continue
        parts = [q for q in (g.geoms if hasattr(g, "geoms") else [g]) if q.geom_type == "Polygon" and not q.is_empty]
        for p in parts:
            for ring in [p.exterior, *p.interiors]:
                xy = ego(np.asarray(ring.coords)[:, :2])
                segs.append(np.concatenate([xy[:-1], xy[1:], np.full((len(xy) - 1, 1), pid)], -1))
            pid += 1
        rt += [red_t[k]] * len(parts)
    red_seg = np.concatenate(segs, 0).astype(np.float32) if segs else np.zeros((0, 5), np.float32)
    red_tm = np.asarray(rt, np.float32).reshape(-1, 41)

    per = {}
    for t in range(1, nt + 1):
        om = obs[t]
        for i in om.query(crop):
            tok = om._tokens[i]
            if red in tok or tok in coll:
                continue
            per.setdefault(tok, []).append((t, ego(shapely.get_coordinates(om._geometries[i])[:4]).ravel()))

    rows = []
    for tid, (tok, lst) in enumerate(per.items()):
        kind = 1.0 if uo[tok].tracked_object_type in AGENT_TYPES else 0.0
        stop = 1.0 if is_track_stopped(uo[tok]) else 0.0
        if len(lst) == nt and all(np.allclose(x, lst[0][1], atol=1e-4) for _, x in lst):
            lst = [(0, lst[0][1])]
        rows += [np.concatenate([[t], BE._box(x), [kind, tid, stop]]) for t, x in lst]
    ag = np.asarray(rows, np.float32).reshape(-1, 9)

    pdm, hum, past, _ = BE.extras(mc)
    return ag, red_seg, red_tm, pdm, hum, past


def _process_log(args):
    global _PROC, _SCFG
    log_path, items = args  # items: list of (row_idx, token, mc_path)
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    if _PROC is None:
        _PROC = MetricCacheProcessor(
            cache_path=None,
            force_feature_computation=True,
            proposal_sampling=TrajectorySampling(num_poses=40, interval_length=0.1),
        )
        _SCFG = SensorConfig.build_no_sensors()

    frames = pickle.load(open(log_path, "rb"))
    tok_to_idx = {f["token"]: i for i, f in enumerate(frames)}
    out = []
    for r, tok, mc_path in items:
        try:
            i = tok_to_idx[tok]
            mc = load_mc(mc_path)
            scene = Scene.from_scene_dict_list(
                frames[i - 3 : i + 11],
                None,
                num_history_frames=4,
                num_future_frames=10,
                sensor_config=_SCFG,
            )
            scen = NavSimScenario(
                scene,
                map_root=_MAP_ROOT,
                map_version="nuplan-maps-v1.0",
            )
            planner_input, planner_init = _PROC._get_planner_inputs(scen)
            _PROC._pdm_closed._iteration = 0
            _PROC._pdm_closed._map_api = planner_init.map_api
            _PROC._pdm_closed._load_route_dicts(planner_init.route_roadblock_ids)
            ego_state, _ = planner_input.history.current_state
            _PROC._pdm_closed._route_roadblock_correction(ego_state)
            _PROC._pdm_closed._drivable_area_map = mc.drivable_area_map
            pdm_closed_traj = _PROC._pdm_closed._get_closed_loop_trajectory(planner_input)

            _, route_lane_dict = _PROC._load_route_dicts(scen, planner_init.route_roadblock_ids)
            interp_tracks = _PROC._interpolate_gt_observation(scen)
            interp_tl = _PROC._interpolate_traffic_light_status(scen)
            obs = _PROC._build_pdm_observation(interp_tracks, interp_tl, route_lane_dict)

            mc.trajectory = pdm_closed_traj
            mc.observation = obs
            mc.human_trajectory = _PROC._extract_ego_future_trajectory(scen) if len(scen.token) != 17 else None
            mc.past_human_trajectory = InterpolatedTrajectory(list(scen.get_ego_past_trajectory(0, 1.5)))

            ag, rseg, rt, pdm, hum, past = _extract_v2_tables(mc)
            out.append((r, ag, rseg, rt, pdm, hum, past, True))
        except Exception as e:  # noqa: BLE001
            out.append((r, None, None, None, None, None, None, repr(e)[:200]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", default=os.environ.get("KM_CORR_ROOT", f"{CK}/resources/km_corridor"))
    ap.add_argument("--exact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd

    df = pd.read_parquet(Path(a.root) / "packed" / a.split / "tokens.parquet")
    toks = df.token.astype(str).to_numpy()
    logs = df.log_name.astype(str).to_numpy() if "log_name" in df.columns else df.log.astype(str).to_numpy()
    n = len(toks)
    mc_root = Path(SPLITS[a.split]["metric_cache"])
    log_dir = Path(SPLITS[a.split]["navsim_logs"])

    d1 = Path(a.exact) / a.split
    d2 = Path(a.out) / a.split
    d2.mkdir(parents=True, exist_ok=True)
    ok1 = np.load(d1 / "ok.npy")

    by_log = {}
    count_scheduled = 0
    for r in range(n):
        if not ok1[r]:
            continue
        lg = logs[r]
        mc_p = mc_root / lg / "unknown" / toks[r] / "metric_cache.pkl"
        lg_p = log_dir / f"{lg}.pkl"
        if mc_p.exists() and lg_p.exists():
            by_log.setdefault(str(lg_p), []).append((r, toks[r], str(mc_p)))
            count_scheduled += 1
            if a.limit and count_scheduled >= a.limit:
                break

    jobs = list(by_log.items())
    print(f"{a.split}: preloading maps in parent process...", flush=True)
    t_map = time.time()
    _preload_maps()
    print(f"{a.split}: maps preloaded in {time.time() - t_map:.1f}s; {count_scheduled}/{n} tokens across {len(jobs)} logs to build exact2_v22", flush=True)

    tabs = {nm: [np.zeros((0, w), np.float32)] * n for nm, w in V2_WIDTHS.items()}
    ok = np.zeros(n, bool)
    t0, cnt, errs = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(_process_log, jobs)):
            for r, ag, rseg, rt, pdm, hum, past, st in res:
                if st is True:
                    tabs["agent_rows"][r] = ag
                    tabs["red_seg"][r] = rseg
                    tabs["red_t"][r] = rt
                    tabs["pdm"][r] = pdm
                    tabs["hum"][r] = hum
                    tabs["past"][r] = past
                    ok[r] = True
                else:
                    errs += 1
                    if errs <= 5:
                        print(f"err row {r}: {st}", flush=True)
                cnt += 1
            if k % 20 == 0 or k + 1 == len(jobs):
                print(f"logs {k + 1}/{len(jobs)} | {cnt}/{count_scheduled} ({cnt / max(time.time() - t0, 1e-3):.1f} tok/s) | err {errs}", flush=True)

    for nm in SHARED_TABLES:
        for suf in (".npy", "_off.npy"):
            dst = d2 / f"{nm}{suf}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to((d1 / f"{nm}{suf}").resolve())

    for nm, arrs in tabs.items():
        off = np.zeros(n + 1, np.int64)
        off[1:] = np.cumsum([len(x) for x in arrs])
        np.save(d2 / f"{nm}_off.npy", off)
        np.save(d2 / f"{nm}.npy", np.concatenate(arrs, 0) if off[-1] > 0 else np.zeros((0, V2_WIDTHS[nm]), np.float32))
    np.save(d2 / "ok.npy", ok & ok1)
    meta = (
        f"split {a.split} n {n} crop x {BE.CROP_X} y {BE.CROP_Y} T_dyn 40 cl full "
        f"v2 True builder build_exact2_v22.py ok {int((ok & ok1).sum())}/{n}\n"
    )
    (d2 / "meta.txt").write_text(meta)
    print(f"done {cnt} in {time.time() - t0:.1f}s, ok {int((ok & ok1).sum())}/{n}, err {errs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
