#!/usr/bin/env python
"""km 10/19 (C): exact ground truth for the candidate verdict targets, from the METRIC CACHE (what the official PDMS
scorer reads).  Geometry, not rasters: the targets of the model's own tracked candidates are computed on the GPU in
modules/exact_verdict.py from these primitives.  All coordinates in the current ego frame (x fwd, y left, rear axle).
Crop: x in [-15, 90] m, y in [-40, 40] m (navtest candidates: x <= 60 m, |y| <= 30 m at 4 s; + the footprint
and the TTC shift of <= 0.9 s stay inside; out-of-crop polygons cannot be hit).
Per token (rows = <root>/packed/<split>/tokens.parquet order), ragged tables with offsets [N + 1]:
  agent_rows  f32 [M, 9]   (t index 1..49 = t * 0.1 s, box centre x, y, heading of the polygon's first edge, half
                            lengths along / across that edge, kind: 1 = AGENT_TYPES (NC 0), track id (0.. within the
                            token), stopped (is_track_stopped of observation.unique_objects: speed <= 0.05 or not an
                            Agent) -- what get_collision_type / the memory of the scorer need;
                            0 = static object (NC 0.5)); red-light polygons and the tracks the ego already overlaps at
                            t = 0 (observation.collided_track_ids) are left out, as the scorer ignores them.  An object
                            with the same polygon at every t 1..T is stored once with t = 0 (= all t).  The observation polygons are boxes
                            (navtest: edges orthogonal to |cos| < 4e-5), so the box is exact
  drv_seg     f32 [S, 5]   (x1, y1, x2, y2, polygon id within the token) edges (exterior + holes) of the drivable-area
                            polygons (ROADBLOCK, INTERSECTION, DRIVABLE_AREA, CARPARK_AREA = the DAC layers)
  lane_seg    f32 [S, 5]   the same for LANE + LANE_CONNECTOR (the scorer's MULTIPLE_LANES area)
  int_seg     f32 [S, 5]   the same for INTERSECTION (is_in_layer(rear axle, INTERSECTION) of the TTC rule)
  route_seg   f32 [S, 5]   the same for the on-route LANE / LANE_CONNECTOR polygons (route_lane_ids; v2 ONCOMING_TRAFFIC
                            area of the driving-direction metric)
  red_seg     f32 [S, 5]   (x1, y1, x2, y2, polygon id) the red-light polygons of the observation (one per red-light
                            token = lane connector, v2 traffic-light compliance), red_t f32 [P, 41] the steps 0..40 at
                            which polygon P is red
  cl          f32 [L, 3]   (x, y, s) route centerline vertices (metric_cache.centerline, the EP path), s = arc length
                            along the full path, vertices with s in [s_ego - 40, s_ego + 200]
  pdm         f32 [41, 9]  the official simulated PDM-Closed states (get_trajectory_as_array(metric_cache.trajectory) ->
                            PDMSimulator), (x, y, heading, vx, vy, ax, ay, angular vel / accel) in the ego frame: the EP normaliser and, in v1,
                            the row the driving-direction sum adds (pdm_score.py proposals [PDM-Closed, prediction])
  hum         f32 [8, 3]   (v2) metric_cache.human_trajectory poses (the human_penalty_filter proposal)
  past        f32 [n, 9]   (v2) the history-comfort padding states exactly as PDMScorer._calculate_history_comfort
                            samples metric_cache.past_human_trajectory, ego frame
  ego         f32 [1, 9]   the initial state array (vx, vy, ax, ay, steering angle / rate, angular velocity / accel, 0)
  ok          bool [N]
  $PY tools/corridor/build_exact.py --split navtrain_val --out <dir> [--workers 4]
  --v2 --mc-root <v2 metric cache>   NAVSIM v2 cache, imported with the navsim_v2 package (EPDMS targets); the centerline
                            is stored whole (--cl-full, the default with --v2) as LK / EP project onto the whole path
  --only pdm                write only the given tables (v1: the PDM-Closed states next to an existing build)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
NV2 = "/home/external-user/yongjae/navsim_v2"
if sys.path[0] != CK:
    sys.path.insert(0, CK)
if "--v2" in sys.argv:              # the v2 cache unpickles only with the navsim_v2 package (SceneFrameType, ...)
    sys.path.insert(0, NV2)

from tools.corridor.build_corridor_dt01 import load_mc  # noqa: E402

if "--v2" in sys.argv and sys.path[0] != NV2:   # build_corridor_dt01 puts this checkout first again
    sys.path.insert(0, NV2)

CROP_X, CROP_Y = (-15.0, 90.0), (-40.0, 40.0)
T_DYN = 49
CL_BACK, CL_FWD = 40.0, 200.0
SIMPLIFY = 0.005           # m
CL_FULL = "--v2" in sys.argv or "--cl-full" in sys.argv
NAMES = ("agent_rows", "drv_seg", "lane_seg", "int_seg", "route_seg", "red_seg", "red_t", "cl", "pdm", "hum", "past", "ego")
WIDTH = (9, 5, 5, 5, 5, 5, 41, 3, 9, 3, 9, 9)


def _box(c8):
    """4 corners (x, y) flattened -> (cx, cy, heading of edge 0-1, half |e01|, half |e12|)"""
    p = c8.reshape(4, 2)
    e01, e12 = p[1] - p[0], p[2] - p[1]
    return np.array([*p.mean(0), np.arctan2(e01[1], e01[0]), np.hypot(*e01) / 2, np.hypot(*e12) / 2])


GEOMETRY = True                    # False with --only pdm[,hum,past,ego,cl]: the extras alone (no polygon work)


def one(path):
    import shapely
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from nuplan.common.maps.abstract_map import SemanticMapLayer

    mc = load_mc(path)
    if not GEOMETRY:
        z = [np.zeros((0, w), np.float32) for w in WIDTH[:7]]
        return tuple(z) + (_cl(mc),) + extras(mc)
    ra = mc.ego_state.rear_axle
    c, s = np.cos(ra.heading), np.sin(ra.heading)

    def ego(xy):
        dx, dy = xy[:, 0] - ra.x, xy[:, 1] - ra.y
        return np.stack([c * dx + s * dy, -s * dx + c * dy], -1)

    cx = np.array([CROP_X[0], CROP_X[1], CROP_X[1], CROP_X[0]])
    cy = np.array([CROP_Y[0], CROP_Y[0], CROP_Y[1], CROP_Y[1]])
    crop = shapely.Polygon(np.stack([ra.x + c * cx - s * cy, ra.y + s * cx + c * cy], -1))
    # drivable polygons
    dm = mc.drivable_area_map
    def poly_edges(geoms, pid0=0):
        # clipped to the crop (no candidate corner lies outside it, so containment inside is unchanged) and
        # simplified by SIMPLIFY m (lane polygons carry up to ~28k vertices per token otherwise)
        segs, pid = [], pid0
        for g0 in geoms:
            g = shapely.simplify(shapely.intersection(g0, crop), SIMPLIFY, preserve_topology=True)
            if g.is_empty:
                continue
            parts = [q for q in (g.geoms if hasattr(g, "geoms") else [g]) if q.geom_type == "Polygon" and not q.is_empty]
            for p in parts:                                          # a MultiPolygon part is its own polygon
                for ring in [p.exterior, *p.interiors]:
                    xy = ego(np.asarray(ring.coords)[:, :2])
                    segs.append(np.concatenate([xy[:-1], xy[1:], np.full((len(xy) - 1, 1), pid)], -1))
                pid += 1
        return (np.concatenate(segs, 0).astype(np.float32) if segs else np.zeros((0, 5), np.float32)), pid

    near = sorted(int(j) for j in dm.query(crop))

    def edges(types, route_only=False):
        want = set(dm.get_indices_of_map_type(types))
        rl = set(mc.route_lane_ids)
        return poly_edges([dm._geometries[i] for i in near if i in want and (not route_only or dm.tokens[i] in rl)])[0]

    seg = edges([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA,
                 SemanticMapLayer.CARPARK_AREA])
    lane = edges([SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    inter = edges([SemanticMapLayer.INTERSECTION])
    route = edges([SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR], route_only=True)
    # agents
    obs = mc.observation
    red, coll = obs.red_light_token, set(obs.collided_track_ids)
    uo = obs.unique_objects
    nt = min(T_DYN, len(obs._global_to_local_idcs) - 1)
    red_geo, red_t = {}, {}                  # one polygon per red-light token (a lane connector), its red steps 0..40
    for t in range(0, min(40, nt) + 1):
        om = obs[t]
        for i in om.query(crop):
            tk = om._tokens[i]
            if tk.startswith(red):
                if tk in red_geo and not red_geo[tk].equals_exact(om._geometries[i], 1e-6):
                    tk = f"{tk}@{t}"                  # (never seen: the same token with another polygon)
                red_geo.setdefault(tk, om._geometries[i])
                red_t.setdefault(tk, np.zeros(41, np.float32))[t] = 1.0
    keys = list(red_geo)
    red_seg, npoly = poly_edges([red_geo[k] for k in keys])
    # poly_edges numbers the non-empty clipped parts; map every part back to its token's step mask
    rt = []
    for k in keys:
        g = shapely.simplify(shapely.intersection(red_geo[k], crop), SIMPLIFY, preserve_topology=True)
        if g.is_empty:
            continue
        parts = [q for q in (g.geoms if hasattr(g, "geoms") else [g]) if q.geom_type == "Polygon" and not q.is_empty]
        rt += [red_t[k]] * len(parts)
    red_tm = np.asarray(rt, np.float32).reshape(-1, 41)
    assert len(red_tm) == npoly, (len(red_tm), npoly)
    per = {}
    for t in range(1, nt + 1):
        om = obs[t]
        for i in om.query(crop):
            tok = om._tokens[i]
            if red in tok or tok in coll:
                continue
            per.setdefault(tok, []).append((t, ego(shapely.get_coordinates(om._geometries[i])[:4]).ravel()))
    from nuplan.planning.simulation.observation.idm.utils import is_track_stopped
    rows = []
    for tid, (tok, lst) in enumerate(per.items()):
        kind = 1.0 if uo[tok].tracked_object_type in AGENT_TYPES else 0.0
        stop = 1.0 if is_track_stopped(uo[tok]) else 0.0
        if len(lst) == nt and all(np.allclose(x, lst[0][1], atol=1e-4) for _, x in lst):
            lst = [(0, lst[0][1])]                                    # constant at every t: one row, t = 0
        rows += [np.concatenate([[t], _box(x), [kind, tid, stop]]) for t, x in lst]
    ag = np.asarray(rows, np.float32).reshape(-1, 9)
    return (ag, seg, lane, inter, route, red_seg, red_tm, _cl(mc)) + extras(mc)


def _cl(mc):
    """route centerline (x, y, s) in the ego frame"""
    import shapely
    ra = mc.ego_state.rear_axle
    c, s = np.cos(ra.heading), np.sin(ra.heading)
    ls = mc.centerline.linestring
    xy = np.asarray(ls.coords)[:, :2]
    sv = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    s_ego = float(shapely.line_locate_point(ls, shapely.Point(ra.x, ra.y)))
    keep = (sv >= s_ego - CL_BACK) & (sv <= s_ego + CL_FWD)
    k = np.flatnonzero(keep)
    if len(k):                                # one vertex of margin on both sides keeps the end segments
        k = np.arange(max(k[0] - 1, 0), min(k[-1] + 2, len(xy)))
    if CL_FULL:
        k = np.arange(len(xy))
    dx, dy = xy[k, 0] - ra.x, xy[k, 1] - ra.y
    return np.stack([c * dx + s * dy, -s * dx + c * dy, sv[k]], -1).astype(np.float32)


def _states_ego(arr, ra):
    """official state array [n, 11] (global) -> [n, 9] (x, y, heading, vx, vy, ax, ay, angular velocity, angular
    acceleration), positions / heading in the ego frame; velocities and accelerations are rear-axle (body) quantities
    in the state array and stay as they are (the v2 comfort shifts them to the box centre with the angular terms)"""
    c, s = np.cos(ra.heading), np.sin(ra.heading)
    dx, dy = arr[:, 0] - ra.x, arr[:, 1] - ra.y
    h = arr[:, 2] - ra.heading
    return np.stack([c * dx + s * dy, -s * dx + c * dy, np.arctan2(np.sin(h), np.cos(h)), *arr[:, 3:7].T,
                     arr[:, 9], arr[:, 10]], -1)


def extras(mc):
    """(pdm [41, 9], hum [8, 3], past [n, 9], ego [1, 9]) with the scoring package's own code (v1 or v2); ego = the
    initial state array of metric_cache.ego_state (vx, vy, ax, ay, steering angle, steering rate, angular velocity,
    angular acceleration, heading 0) -- what the official simulation starts from"""
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    fs = TrajectorySampling(num_poses=40, interval_length=0.1)
    ies = mc.ego_state
    pdm_arr = get_trajectory_as_array(mc.trajectory, fs, ies.time_point)
    pdm = _states_ego(PDMSimulator(fs).simulate_proposals(pdm_arr[None], ies)[0], ies.rear_axle).astype(np.float32)
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
    e = ego_states_to_state_array([ies])[0]
    ego = np.array([[*e[3:11], 0.0]], np.float32)
    hum = np.zeros((0, 3), np.float32)
    past = np.zeros((0, 9), np.float32)
    if getattr(mc, "human_trajectory", None) is not None:
        hum = np.asarray(mc.human_trajectory.poses, np.float32).reshape(-1, 3)
    pht = getattr(mc, "past_human_trajectory", None)
    if pht is not None:                       # PDMScorer._calculate_history_comfort, line by line
        from nuplan.common.actor_state.state_representation import TimePoint
        t0, t1 = pht.start_time.time_us, pht.end_time.time_us
        tu = np.clip(np.arange(t0, t1, int(0.1 * 1e6), dtype=np.int64), t0, t1)
        arr = ego_states_to_state_array(pht.get_state_at_times([TimePoint(int(x)) for x in tu[:-1]]))
        past = _states_ego(arr, ies.rear_axle).astype(np.float32)
    return pdm, hum, past, ego


def work(args):
    rows, paths = args
    out = []
    for r, p in zip(rows, paths):
        try:
            out.append((r,) + one(p) + (True,))
        except Exception as e:  # noqa: BLE001
            out.append((r,) + (None,) * len(NAMES) + (repr(e)[:200],))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", default=os.environ.get("KM_CORR_ROOT", "/home/external-user/ssd/kyungmin_ck/km_runs/ck2e2e_bevkd/eval/root"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--cl-full", action="store_true")
    ap.add_argument("--mc-root", default="")
    ap.add_argument("--only", default="", help="comma list of tables to write (default all)")
    ap.add_argument("--part", default="", help="write <out>/<split>/parts/<part>/ for the rows not yet in another part "
                    "(tokens whose cache exists now); merge_exact_parts.py joins the parts")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd
    toks = pd.read_parquet(Path(a.root) / "packed" / a.split / "tokens.parquet").token.astype(str).to_numpy()
    n = len(toks)
    if a.mc_root:
        mc_root = Path(a.mc_root)
    else:
        from tools.ck.data.common import MC_ROOT
        mc_root = MC_ROOT[a.split]
    import navsim
    print("navsim package:", navsim.__file__, flush=True)
    write = set(a.only.split(",")) if a.only else set(NAMES)
    global GEOMETRY
    GEOMETRY = bool(write - {"pdm", "hum", "past", "ego", "cl"})
    where = {}
    for log in os.scandir(mc_root):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = Path(t.path) / "metric_cache.pkl"
    d = Path(a.out) / a.split
    if a.part:
        prev = [np.load(q / "ok.npy") for q in (d / "parts").glob("*/") if (q / "ok.npy").exists() and q.name != a.part]
        taken = np.any(prev, 0) if prev else np.zeros(n, bool)
        d = d / "parts" / a.part
    d.mkdir(parents=True, exist_ok=True)
    meta = (f"split {a.split} n {n} crop x {CROP_X} y {CROP_Y} T_dyn {T_DYN} cl "
            f"{'full' if CL_FULL else f'window [-{CL_BACK}, +{CL_FWD}]'} mc_root {mc_root} v2 {a.v2} "
            f"navsim {navsim.__file__} tables {sorted(write)}\n")
    (d / ("meta.txt" if write == set(NAMES) else f"meta_{'_'.join(sorted(write))}.txt")).write_text(meta)
    rows = np.array([r for r in range(n) if toks[r] in where and not (a.part and taken[r])], np.int64)
    if a.limit:
        rows = rows[: a.limit]
    print(f"{a.split}: {n} tokens, {len(rows)} to build, {n - len(rows)} without metric cache", flush=True)
    jobs = [(rows[i:i + a.chunk], [str(where[toks[r]]) for r in rows[i:i + a.chunk]])
            for i in range(0, len(rows), a.chunk)]
    names = NAMES
    tabs = {k: [np.zeros((0, w), np.float32)] * n for k, w in zip(names, WIDTH)}
    ok = np.zeros(n, bool)
    t0, cnt, errs = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(work, jobs)):
            for r, *vals, st in res:
                if st is True:
                    for nm, v in zip(names, vals):
                        tabs[nm][r] = v
                    ok[r] = True
                else:
                    errs += 1
                    if errs <= 5:
                        print(f"err row {r}: {st}", flush=True)
                cnt += 1
            if k % 50 == 0:
                print(f"{cnt}/{len(rows)} | {cnt / (time.time() - t0):.1f} tok/s | err {errs}", flush=True)
    for key in names:
        if key not in write:
            continue
        off = np.zeros(n + 1, np.int64)
        off[1:] = np.cumsum([len(x) for x in tabs[key]])
        np.save(d / f"{key}_off.npy", off)
        np.save(d / f"{key}.npy", np.concatenate(tabs[key], 0))
    np.save(d / ("ok.npy" if write == set(NAMES) else f"ok_{'_'.join(sorted(write))}.npy"), ok)
    per = {k: np.array([len(x) for x, o in zip(tabs[k], ok) if o]) for k in tabs}
    print(f"done {cnt} in {time.time() - t0:.0f}s, ok {int(ok.sum())}/{n}, err {errs} | per token max/p99: "
          + ", ".join(f"{nm} {v.max() if len(v) else 0}/{int(np.percentile(v, 99)) if len(v) else 0}" for nm, v in per.items()),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
