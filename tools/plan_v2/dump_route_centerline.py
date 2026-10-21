"""The scorer's ROUTE centerline for every scene of a split, in the current ego frame.

The ego-progress metric projects the start and end of the (tracked) plan onto the route centerline
the PDM-Closed planner builds (pdm_scorer._calculate_progress).  A geometric EP label needs that same
line at training time, and the navtrain metric cache exists for only 5,139 of 103,288 scenes.  This
runs the PDM-Closed planner object the metric cache uses (MetricCacheProcessor._pdm_closed, same
constructor arguments) through exactly the steps that produce its centerline -- route roadblock
correction, drivable-area map, starting lane, Dijkstra over the lane graph -- and skips the IDM
proposals, simulation and scoring it does not need.

    python tools/plan_v2/dump_route_centerline.py --split navtest --limit 200 --verify   # control
    python tools/plan_v2/dump_route_centerline.py --split navtrain --workers 12

Output work_dirs/route_centerline_<split>.npz: tokens, line [N, P, 2] float32 (ego frame, NAVSIM
x forward / y left, 1 m spacing from 10 m behind the ego's projection to 150 m ahead, NaN padded),
n [N] number of valid points.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/readout"))
import _env  # noqa: E402,F401  -- before any navsim import: map root is read at import time

P = 161                      # points: -10 .. +150 m at 1 m
BACK, AHEAD, STEP = 10.0, 150.0, 1.0
_CTX = {}


def _init(split):
    # NOT a SceneLoader over the whole split: on navtrain that is 10.5 GB per worker, and 12 of them
    # OOM-killed a training on this shared host (10-01 18:22).  Each task builds a loader over its
    # own few logs instead, the way navsim's metric caching distributes work.
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    _CTX["split"] = split
    _CTX["proc"] = MetricCacheProcessor(cache_path=None, force_feature_computation=False)


def _loader_for(log_names, tokens):
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from _env import scene_filter, split_dirs
    split = _CTX["split"]
    logs, blobs = split_dirs("test" if split == "navtest" else "trainval")
    f = scene_filter(split)
    f.log_names = list(log_names); f.tokens = list(tokens)
    return SceneLoader(logs, blobs, f, sensor_config=SensorConfig.build_no_sensors())


def centerline_global(token):
    """-> (initial ego state, [M, 2] global centerline points) exactly as PDM-Closed builds them."""
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    scene = _CTX["loader"].get_scene_from_token(token)
    scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
    proc = _CTX["proc"]; planner = proc._pdm_closed
    _, init = proc._get_planner_inputs(scenario)
    planner.initialize(init)
    ego = scenario.initial_ego_state
    planner._route_roadblock_correction(ego)                                   # pdm_closed_planner.py:85
    planner._drivable_area_map = PDMDrivableMap.from_simulation(planner._map_api, ego, planner._map_radius)
    lane = planner._get_starting_lane(ego)
    path = planner._get_discrete_centerline(lane)                               # abstract_pdm_closed_planner.py:116
    return ego, np.array([[p.x, p.y] for p in path], dtype=np.float64)


def to_ego_resampled(ego, pts):
    """global [M, 2] -> ego-frame polyline resampled every STEP m over [s_ego - BACK, s_ego + AHEAD]."""
    x0, y0, h = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
    c, s = np.cos(h), np.sin(h)
    d = pts - [x0, y0]
    loc = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
    seg = np.linalg.norm(np.diff(loc, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    loc = loc[keep]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loc, axis=0), axis=1))])
    s_ego = cum[np.argmin(np.linalg.norm(loc, axis=1))]                       # nearest vertex to the ego
    grid = np.arange(max(0.0, s_ego - BACK), min(cum[-1], s_ego + AHEAD) + 1e-9, STEP)
    out = np.full((P, 2), np.nan, np.float32)
    k = min(len(grid), P)
    out[:k, 0] = np.interp(grid[:k], cum, loc[:, 0])
    out[:k, 1] = np.interp(grid[:k], cum, loc[:, 1])
    return out, k


def _task(item):
    """(log names, tokens) -> list of (token, line, k, err) -- one loader per task, then dropped."""
    log_names, tokens = item
    out = []
    try:
        _CTX["loader"] = _loader_for(log_names, tokens)
    except Exception as exc:
        return [(t, None, 0, "loader: " + repr(exc)[:140]) for t in tokens]
    for token in tokens:
        try:
            ego, pts = centerline_global(token)
            line, k = to_ego_resampled(ego, pts)
            out.append((token, line, k, None))
        except Exception as exc:
            out.append((token, None, 0, repr(exc)[:160]))
    _CTX.pop("loader", None)
    return out


def _avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def verify(tokens, lines, ns):
    """Control: against the centerline the scorer itself stored in the navtest metric cache."""
    import lzma
    import pickle
    from shapely.geometry import LineString, Point
    from navsim.common.dataloader import MetricCacheLoader
    mc = MetricCacheLoader(ROOT / "data/exp/metric_cache")
    dists, prog_err = [], []
    for t, line, n in zip(tokens, lines, ns):
        if n < 2 or t not in mc.metric_cache_paths:
            continue
        with lzma.open(mc.metric_cache_paths[t], "rb") as f:
            cache = pickle.load(f)
        ego = cache.ego_state
        x0, y0, h = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        c, s = np.cos(h), np.sin(h)
        g = np.stack([x0 + c * line[:n, 0] - s * line[:n, 1], y0 + s * line[:n, 0] + c * line[:n, 1]], -1)
        ref = cache.centerline._linestring
        dists.append(max(ref.distance(Point(*p)) for p in g))
        # progress of a point 20 m ahead along OUR line, measured on both lines
        mine = LineString(g)
        q = Point(*g[min(n - 1, 30)])
        prog_err.append(abs((ref.project(q) - ref.project(Point(x0, y0))) - (mine.project(q) - mine.project(Point(x0, y0)))))
    dists, prog_err = np.array(dists), np.array(prog_err)
    print(f"대조군 {len(dists)} 장면: 우리 중심선 점이 채점기 중심선에서 떨어진 최대 거리 "
          f"median {np.median(dists):.4f} m, p99 {np.percentile(dists, 99):.3f} m;  "
          f"진행량 차이 median {np.median(prog_err):.4f} m, p99 {np.percentile(prog_err, 99):.3f} m", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain", choices=["navtrain", "navtest"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=60.0)
    ap.add_argument("--abort-free-gb", type=float, default=40.0)
    a = ap.parse_args()

    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from _env import scene_filter, split_dirs
    if _avail_gb() < a.min_free_gb:
        raise SystemExit(f"사용 가능 RAM {_avail_gb():.0f} GB < {a.min_free_gb} GB -- 시작하지 않음")
    logs, blobs = split_dirs("test" if a.split == "navtest" else "trainval")
    full = SceneLoader(logs, blobs, scene_filter(a.split), sensor_config=SensorConfig.build_no_sensors())
    per_log = full.get_tokens_list_per_log()
    toks = sorted(full.tokens)
    if a.limit:
        toks = toks[: a.limit]
    keep = set(toks)
    tasks = []
    for log, tl in sorted(per_log.items()):
        tl = [t for t in tl if t in keep]
        for i in range(0, len(tl), 64):                       # <= 64 scenes per task, one log each
            tasks.append(([log], tl[i:i + 64]))
    del full, per_log
    print(f"{a.split}: {len(toks)} 장면, 작업 {len(tasks)}개, 사용 가능 RAM {_avail_gb():.0f} GB", flush=True)

    lines = np.full((len(toks), P, 2), np.nan, np.float32); ns = np.zeros(len(toks), np.int32)
    at = {t: i for i, t in enumerate(toks)}; bad = []; t0 = time.time(); j = 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.split,)) as ex:
        for res in ex.map(_task, tasks):
            for t, line, k, err in res:
                j += 1
                if err:
                    bad.append((t, err))
                else:
                    lines[at[t]], ns[at[t]] = line, k
            if _avail_gb() < a.abort_free_gb:
                ex.shutdown(cancel_futures=True)
                raise SystemExit(f"사용 가능 RAM {_avail_gb():.0f} GB < {a.abort_free_gb} GB -- 중단 (공유 서버)")
            if j // 2000 != (j - len(res)) // 2000:
                el = time.time() - t0
                print(f"{j}/{len(toks)}  {el / j:.3f} s/scene  남은 {(len(toks) - j) * el / j / 60:.0f}분  "
                      f"RAM 여유 {_avail_gb():.0f} GB", flush=True)
    tag = f"{a.split}" + (f"_limit{a.limit}" if a.limit else "")
    out = ROOT / f"work_dirs/route_centerline_{tag}.npz"
    np.savez_compressed(out, tokens=np.array(toks), line=lines, n=ns)
    print(f"저장 {out}  실패 {len(bad)}  점 수 median {int(np.median(ns))}  ({(time.time() - t0) / 60:.1f}분)", flush=True)
    for t, e in bad[:5]:
        print(f"  {t}: {e}")
    if a.verify:
        verify(toks, lines, ns)


if __name__ == "__main__":
    main()
