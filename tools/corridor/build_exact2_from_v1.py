#!/usr/bin/env python
"""Build exact2 (NAVSIM v2.2 EPDMS tables) directly from v1 exact + navsim_logs + v1 metric_cache.

Reuses the identical geometry tables (agent_rows, drv_seg, lane_seg, int_seg, route_seg, cl, pdm, ego)
via symlinks from <exact_dir>/<split>/ and computes the v2-specific tables (hum, past, red_seg, red_t)
from the raw navsim_logs and metric_cache drivable_area_map (verified bit-exact against NAVSIM v2.2
metric_cache on navtest).
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

SHARED_TABLES = ("agent_rows", "drv_seg", "lane_seg", "int_seg", "route_seg", "cl", "pdm", "ego")


def _process_log(args):
    log_path, items = args  # items: list of (row_idx, token, mc_path)
    import shapely
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.geometry.convert import absolute_to_relative_poses
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array

    frames = pickle.load(open(log_path, "rb"))
    tok_to_idx = {f["token"]: i for i, f in enumerate(frames)}
    scfg = SensorConfig.build_no_sensors()
    out = []
    for r, tok, mc_path in items:
        try:
            i = tok_to_idx[tok]
            mc = load_mc(mc_path)
            ies = mc.ego_state
            ra = ies.rear_axle
            c, s = np.cos(ra.heading), np.sin(ra.heading)

            def ego(xy):
                dx, dy = xy[:, 0] - ra.x, xy[:, 1] - ra.y
                return np.stack([c * dx + s * dy, -s * dx + c * dy], -1)

            # 1) hum & past via NavSimScenario
            scene = Scene.from_scene_dict_list(
                frames[i - 3 : i + 11],
                None,
                num_history_frames=4,
                num_future_frames=10,
                sensor_config=scfg,
            )
            scen = NavSimScenario(
                scene,
                map_root="/home/external-user/yongjae/SSR/data/dataset/maps",
                map_version="nuplan-maps-v1.0",
            )
            fut = list(scen.get_ego_future_trajectory(0, 4.0, 8))
            if fut[0].time_point != ies.time_point:
                fut = [ies] + fut
            rel = absolute_to_relative_poses([st.rear_axle for st in fut])[1:]
            hum = np.array([[q.x, q.y, q.heading] for q in rel], np.float32).reshape(-1, 3)

            pht = InterpolatedTrajectory(list(scen.get_ego_past_trajectory(0, 1.5)))
            t0_us, t1_us = pht.start_time.time_us, pht.end_time.time_us
            tu = np.clip(np.arange(t0_us, t1_us, int(0.1 * 1e6), dtype=np.int64), t0_us, t1_us)
            arr = ego_states_to_state_array(pht.get_state_at_times([TimePoint(int(x)) for x in tu[:-1]]))
            past = BE._states_ego(arr, ra).astype(np.float32)

            # 2) red_seg & red_t over 41 steps (0..40)
            dm = mc.drivable_area_map
            rl = set(mc.route_lane_ids)
            vx, vy = BE.CROP_X, BE.CROP_Y
            corners = np.array([[vx[0], vy[0]], [vx[1], vy[0]], [vx[1], vy[1]], [vx[0], vy[1]]])
            crop = shapely.polygons(
                np.stack([ra.x + c * corners[:, 0] - s * corners[:, 1], ra.y + s * corners[:, 0] + c * corners[:, 1]], -1)
            )
            red_geo, red_t = {}, {}
            for k in range(9):
                if i + k >= len(frames):
                    break
                tls = frames[i + k]["traffic_lights"]
                steps = range(k * 5, min((k + 1) * 5, 41))
                for lc_id, is_red in tls:
                    s_id = str(lc_id)
                    if is_red and s_id in rl and s_id in dm.token_to_idx:
                        poly = dm[s_id]
                        if crop.intersects(poly):
                            tk = f"red_light_{s_id}"
                            red_geo.setdefault(tk, poly)
                            for st in steps:
                                red_t.setdefault(tk, np.zeros(41, np.float32))[st] = 1.0
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
            out.append((r, red_seg, red_tm, hum, past, True))
        except Exception as e:  # noqa: BLE001
            out.append((r, None, None, None, None, repr(e)[:200]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", default=os.environ.get("KM_CORR_ROOT", f"{CK}/resources/km_corridor"))
    ap.add_argument("--exact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
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
    for r in range(n):
        if not ok1[r]:
            continue
        lg = logs[r]
        mc_p = mc_root / lg / "unknown" / toks[r] / "metric_cache.pkl"
        lg_p = log_dir / f"{lg}.pkl"
        if mc_p.exists() and lg_p.exists():
            by_log.setdefault(str(lg_p), []).append((r, toks[r], str(mc_p)))

    jobs = list(by_log.items())
    print(f"{a.split}: {n} tokens across {len(jobs)} logs to build exact2", flush=True)

    tabs = {
        "red_seg": [np.zeros((0, 5), np.float32)] * n,
        "red_t": [np.zeros((0, 41), np.float32)] * n,
        "hum": [np.zeros((0, 3), np.float32)] * n,
        "past": [np.zeros((0, 9), np.float32)] * n,
    }
    ok = np.zeros(n, bool)
    t0, cnt, errs = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(_process_log, jobs)):
            for r, rseg, rt, hum, past, st in res:
                if st is True:
                    tabs["red_seg"][r] = rseg
                    tabs["red_t"][r] = rt
                    tabs["hum"][r] = hum
                    tabs["past"][r] = past
                    ok[r] = True
                else:
                    errs += 1
                    if errs <= 5:
                        print(f"err row {r}: {st}", flush=True)
                cnt += 1
            if k % 25 == 0:
                print(f"logs {k + 1}/{len(jobs)} | {cnt}/{n} ({cnt / (time.time() - t0):.1f} tok/s) | err {errs}", flush=True)

    for nm in SHARED_TABLES:
        for suf in (".npy", "_off.npy"):
            dst = d2 / f"{nm}{suf}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(d1 / f"{nm}{suf}")

    for nm, arrs in tabs.items():
        off = np.zeros(n + 1, np.int64)
        off[1:] = np.cumsum([len(x) for x in arrs])
        np.save(d2 / f"{nm}_off.npy", off)
        np.save(d2 / f"{nm}.npy", np.concatenate(arrs, 0))
    np.save(d2 / "ok.npy", ok & ok1)
    print(f"done {cnt} in {time.time() - t0:.1f}s, ok {int((ok & ok1).sum())}/{n}, err {errs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
