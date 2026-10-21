#!/usr/bin/env python3
"""Build a FACTORIZED trajectory vocabulary (SparseDriveV2-style) from NAVSIM
human future trajectories.

A trajectory is factorized into

  * a PATH    -- the geometric route, sampled every ``--station-spacing`` m of
                 arc length: ``paths_{P}.npy`` float32 [P, S+1, 2], station 0
                 is the origin (0, 0), station s is the point ``s * spacing`` m
                 along the route (ego rear-axle frame of the current frame,
                 x forward, y left);
  * a SPEED   -- the progress profile along that route: ``speeds_{V}.npy``
                 float32 [V, T], the cumulative arc length s(t) reached at
                 t = 0.5, 1.0, ..., 4.0 s (T = 8), non-decreasing, s(0) = 0
                 implicit.

A composed trajectory is ``traj(t) = path(s(t))`` (linear interpolation along
the stations, straight-line extrapolation past the last station); see
:func:`compose`.  Any (path, speed) pair is a candidate, so P x V trajectories
are representable from P + V prototypes.

Source data: the human future trajectories (8 poses, 0.5 s steps = 4 s) of
the scenes selected by the hydra scene filter (``navtrain`` by default), read
with the same filtering as ``navsim.common.dataloader.filter_scenes`` and the
same pose conversion as ``Scene.get_future_trajectory``.  Both vocabularies
are k-means (k-means++ init) centroids; the speed sample set is augmented
with "faster" copies (x1.15, x1.3) so faster-than-human profiles are covered.

Outputs (``--out``, default ``data/fact_vocab``):

  paths_{P}.npy          float32 [P, S+1, 2]
  paths_{P}_len.npy      float32 [P]       mean real (non-extrapolated) length
                                           of the member trajectories
  speeds_{V}.npy         float32 [V, T]
  source_trajectories.npz  tokens, log_names + the raw [N, T, 3] human poses
  meta.json              counts, spacing, timestamps, subsampling, coverage

Example::

  python tools/fact/build_vocab.py --split trainval --scene-filter navtrain \
      --paths 1024 --speeds 256 --workers 32 --out data/fact_vocab
  python tools/fact/build_vocab.py --limit 2000 --out /tmp/fact_vocab_test
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.common.dataclasses import SceneFilter  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (  # noqa: E402
    convert_absolute_to_relative_se2_array,
)
from nuplan.common.actor_state.state_representation import StateSE2  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

NUM_FUTURE_POSES = 8            # 4 s at 0.5 s
INTERVAL_LENGTH = 0.5           # s
SPEED_AUG_SCALES = (1.15, 1.3)  # "faster" copies added to the speed samples
SPEED_MAX_DISTANCE = 75.0       # m, clip of the augmented final distance
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Geometry helpers (pure numpy)
# --------------------------------------------------------------------------- #
def cumulative_arc_length(xy: np.ndarray) -> np.ndarray:
    """Cumulative arc length [..., K] of a polyline [..., K, 2] (first = 0)."""
    seg = np.linalg.norm(np.diff(xy, axis=-2), axis=-1)
    return np.concatenate([np.zeros_like(seg[..., :1]), np.cumsum(seg, axis=-1)], axis=-1)


def resample_path(poses_xy: np.ndarray, num_stations: int, spacing: float
                  ) -> Tuple[np.ndarray, float]:
    """Resample one trajectory at fixed arc-length stations.

    Args:
        poses_xy: [T, 2] future positions (origin excluded).
        num_stations: S, number of stations after the origin.
        spacing: station spacing in metres.

    Returns:
        stations: [S, 2] points at arc length spacing * (1..S); past the real
            end of the trajectory the path is extrapolated straight along the
            heading of the last segment (along +x if the trajectory is shorter
            than ``spacing``).
        length: real (non-extrapolated) length of the trajectory in metres.
    """
    pts = np.concatenate([np.zeros((1, 2)), poses_xy.astype(np.float64)], axis=0)
    # Drop (near-)duplicate consecutive points so interpolation is well-defined.
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > EPS])
    pts = pts[keep]
    arc = cumulative_arc_length(pts)
    length = float(arc[-1])

    targets = spacing * np.arange(1, num_stations + 1, dtype=np.float64)
    if length < spacing or len(pts) < 2:
        direction = np.array([1.0, 0.0])
    else:
        last = pts[-1] - pts[-2]
        direction = last / np.linalg.norm(last)

    x = np.interp(targets, arc, pts[:, 0])
    y = np.interp(targets, arc, pts[:, 1])
    stations = np.stack([x, y], axis=-1)
    beyond = targets > length
    if beyond.any():
        stations[beyond] = pts[-1] + (targets[beyond] - length)[:, None] * direction
    return stations.astype(np.float32), length


def compose(paths: np.ndarray, speeds: np.ndarray, path_idx: np.ndarray,
            speed_idx: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Compose (path, speed) pairs into trajectories.

    Args:
        paths: [P, S+1, 2] path stations (station 0 = origin).
        speeds: [V, T] cumulative arc length at each future timestep.
        path_idx, speed_idx: [N] integer indices (broadcastable).
        spacing: station spacing in metres.

    Returns:
        [N, T, 3] float32 trajectories (x, y, heading), where the point at
        time t is the path point at arc length s(t) (linear interpolation
        between stations, straight extrapolation past the last station) and
        the heading is the tangent of the path segment at that arc length.
    """
    path_idx = np.asarray(path_idx)
    speed_idx = np.asarray(speed_idx)
    num_seg = paths.shape[1] - 1
    s = np.asarray(speeds, dtype=np.float64)[speed_idx] / spacing        # [N, T]
    seg = np.clip(np.floor(s).astype(np.int64), 0, num_seg - 1)         # [N, T]
    frac = s - seg                                                      # [N, T]
    p = np.asarray(paths, dtype=np.float64)[path_idx]                   # [N, S+1, 2]
    p0 = np.take_along_axis(p, seg[..., None], axis=1)                  # [N, T, 2]
    p1 = np.take_along_axis(p, seg[..., None] + 1, axis=1)
    d = p1 - p0
    xy = p0 + frac[..., None] * d
    heading = np.arctan2(d[..., 1], d[..., 0])
    return np.concatenate([xy, heading[..., None]], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Scene loading
# --------------------------------------------------------------------------- #
def load_scene_filter(name: str) -> SceneFilter:
    """Instantiate the hydra scene-filter yaml without hydra."""
    cfg_path = (REPO_ROOT / "navsim/planning/script/config/common/scene_filter"
                / f"{name}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    kwargs = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return SceneFilter(**kwargs)


def future_poses_from_frames(frames: Sequence[Dict[str, Any]], num_history_frames: int,
                             num_poses: int) -> np.ndarray:
    """Same computation as ``Scene.get_future_trajectory(num_poses).poses``
    (``_build_ego_status`` + ``convert_absolute_to_relative_se2_array``),
    without building the Scene (no map API, no sensors)."""
    start = num_history_frames - 1
    global_poses = []
    for frame in frames[start:start + num_poses + 1]:
        t = frame["ego2global_translation"]
        yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
        global_poses.append([t[0], t[1], yaw])
    global_poses = np.array(global_poses, dtype=np.float64)
    local = convert_absolute_to_relative_se2_array(StateSE2(*global_poses[0]), global_poses[1:])
    return np.asarray(local, dtype=np.float32)


_WORKER: Dict[str, Any] = {}


def _worker_init(scene_filter: SceneFilter, tokens: Optional[set]) -> None:
    _WORKER["filter"] = scene_filter
    _WORKER["tokens"] = tokens


def _process_log(log_path: str) -> List[Tuple[str, str, np.ndarray]]:
    """Replicates ``navsim.common.dataloader.filter_scenes`` for one log and
    returns (token, log_name, [T, 3] future poses) for each selected scene."""
    sf: SceneFilter = _WORKER["filter"]
    tokens = _WORKER["tokens"]
    with open(log_path, "rb") as f:
        frames = pickle.load(f)
    out = []
    for i in range(0, len(frames), sf.frame_interval):
        scene = frames[i:i + sf.num_frames]
        if len(scene) < sf.num_frames:
            continue
        cur = scene[sf.num_history_frames - 1]
        if sf.has_route and len(cur["roadblock_ids"]) == 0:
            continue
        token = cur["token"]
        if tokens is not None and token not in tokens:
            continue
        out.append((token, cur["log_name"],
                    future_poses_from_frames(scene, sf.num_history_frames, NUM_FUTURE_POSES)))
    return out


def collect_trajectories(split: str, scene_filter: SceneFilter, workers: int,
                         limit: Optional[int]) -> Tuple[List[str], List[str], np.ndarray]:
    """Returns (tokens, log_names, poses [N, T, 3]) sorted by token."""
    data_path = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / split
    log_files = sorted(data_path.iterdir())
    if scene_filter.log_names is not None:
        wanted = set(scene_filter.log_names)
        log_files = [p for p in log_files if p.name.replace(".pkl", "") in wanted]
    tokens = set(scene_filter.tokens) if scene_filter.tokens is not None else None
    print(f"[load] {len(log_files)} log files from {data_path}, "
          f"{'all' if tokens is None else len(tokens)} tokens allowed", flush=True)

    all_tokens: List[str] = []
    all_logs: List[str] = []
    all_poses: List[np.ndarray] = []
    t0 = time.time()
    with mp.get_context("fork").Pool(workers, initializer=_worker_init,
                                     initargs=(scene_filter, tokens)) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_log, [str(p) for p in log_files],
                                                    chunksize=2)):
            for token, log_name, poses in res:
                all_tokens.append(token)
                all_logs.append(log_name)
                all_poses.append(poses)
            if (i + 1) % 100 == 0 or i + 1 == len(log_files):
                print(f"[load] {i + 1}/{len(log_files)} logs, {len(all_tokens)} scenes, "
                      f"{time.time() - t0:.0f}s", flush=True)
            if limit is not None and len(all_tokens) >= limit:
                pool.terminate()
                break
    # Deterministic order regardless of worker scheduling.
    order = np.argsort(np.array(all_tokens))
    all_tokens = [all_tokens[i] for i in order]
    all_logs = [all_logs[i] for i in order]
    poses = np.stack([all_poses[i] for i in order]).astype(np.float32)
    if limit is not None:
        all_tokens, all_logs, poses = all_tokens[:limit], all_logs[:limit], poses[:limit]
    return all_tokens, all_logs, poses


def verify_against_scene_api(split: str, scene_filter: SceneFilter, tokens: List[str],
                             log_names: List[str], poses: np.ndarray, n: int = 5) -> float:
    """Cross-check the fast pose extraction with ``Scene.get_future_trajectory``."""
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    sf = SceneFilter(num_history_frames=scene_filter.num_history_frames,
                     num_future_frames=scene_filter.num_future_frames,
                     frame_interval=scene_filter.frame_interval,
                     has_route=scene_filter.has_route,
                     log_names=sorted(set(log_names[:n])), tokens=tokens[:n])
    loader = SceneLoader(root / "navsim_logs" / split, root / "sensor_blobs" / split, sf,
                         sensor_config=SensorConfig.build_no_sensors())
    worst = 0.0
    for i, token in enumerate(tokens[:n]):
        ref = np.asarray(loader.get_scene_from_token(token)
                         .get_future_trajectory(num_trajectory_frames=NUM_FUTURE_POSES).poses,
                         dtype=np.float32)
        worst = max(worst, float(np.abs(ref - poses[i]).max()))
    return worst


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def kmeans(x: np.ndarray, k: int, iters: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """k-means++ init + Lloyd iterations. Returns (centroids [k, D], labels [N])."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=iters, tol=0.0,
                    random_state=seed, algorithm="lloyd").fit(x)
        return km.cluster_centers_, km.labels_
    except ImportError:
        pass
    # Fallback: plain numpy implementation (used only if sklearn is missing).
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = np.empty((k, x.shape[1]))
    centers[0] = x[rng.integers(n)]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for j in range(1, k):
        centers[j] = x[rng.choice(n, p=d2 / d2.sum())]
        d2 = np.minimum(d2, ((x - centers[j]) ** 2).sum(1))
    xx = (x ** 2).sum(1)
    labels = None
    for _ in range(iters):
        dist = xx[:, None] - 2 * x @ centers.T + (centers ** 2).sum(1)[None]
        labels = dist.argmin(1)
        for j in range(k):
            m = labels == j
            centers[j] = x[m].mean(0) if m.any() else x[rng.integers(n)]
    return centers, labels


# --------------------------------------------------------------------------- #
# Coverage statistics
# --------------------------------------------------------------------------- #
def coverage_composed(paths: np.ndarray, speeds: np.ndarray, spacing: float,
                      heldout: np.ndarray, chunk: int = 8) -> Dict[str, float]:
    """min over all (path, speed) pairs of the ADE (xy) to each held-out trajectory."""
    P, V = len(paths), len(speeds)
    pi, vi = np.meshgrid(np.arange(P), np.arange(V), indexing="ij")
    table = compose(paths, speeds, pi.ravel(), vi.ravel(), spacing)[..., :2]  # [P*V, T, 2]
    table = table.astype(np.float32)
    best = np.empty(len(heldout), dtype=np.float32)
    for i in range(0, len(heldout), chunk):
        h = heldout[i:i + chunk, :, :2].astype(np.float32)                 # [c, T, 2]
        ade = np.linalg.norm(table[None] - h[:, None], axis=-1).mean(-1)   # [c, P*V]
        best[i:i + chunk] = ade.min(1)
    return summarize(best)


def coverage_anchors(anchors: np.ndarray, heldout: np.ndarray) -> Dict[str, float]:
    a = anchors[None, :, :, :2].astype(np.float32)             # [1, A, T, 2]
    h = heldout[:, None, :, :2].astype(np.float32)             # [N, 1, T, 2]
    best = np.linalg.norm(a - h, axis=-1).mean(-1).min(1)
    return summarize(best)


def summarize(best: np.ndarray) -> Dict[str, float]:
    return {"mean": float(best.mean()), "p50": float(np.percentile(best, 50)),
            "p90": float(np.percentile(best, 90)), "max": float(best.max())}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--scene-filter", default="navtrain", help="hydra scene_filter yaml name")
    ap.add_argument("--paths", type=int, default=1024, help="P, number of path prototypes")
    ap.add_argument("--speeds", type=int, default=256, help="V, number of speed prototypes")
    ap.add_argument("--stations", type=int, default=50, help="S, stations after the origin")
    ap.add_argument("--station-spacing", type=float, default=1.0, help="metres between stations")
    ap.add_argument("--limit", type=int, default=None, help="stop after N scenes (quick test)")
    ap.add_argument("--max-trajectories", type=int, default=None,
                    help="uniformly subsample the loaded trajectories to this many")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--heldout", type=int, default=2000, help="trajectories held out for coverage stats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "data/fact_vocab"))
    ap.add_argument("--trajectories-npz", default=None,
                    help="reuse a previously written source_trajectories.npz instead of loading scenes")
    ap.add_argument("--anchors", default=str(REPO_ROOT / "data/planning_vb/trajectory_anchors_256.npy"))
    ap.add_argument("--verify", type=int, default=0,
                    help="cross-check this many trajectories against Scene.get_future_trajectory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    meta: Dict[str, Any] = {"command": " ".join(sys.argv), "args": vars(args)}

    # ---- 1. source trajectories ------------------------------------------- #
    if args.trajectories_npz:
        src = np.load(args.trajectories_npz)
        tokens, log_names, poses = list(src["tokens"]), list(src["log_names"]), src["poses"]
        print(f"[load] reused {len(tokens)} trajectories from {args.trajectories_npz}")
    else:
        scene_filter = load_scene_filter(args.scene_filter)
        tokens, log_names, poses = collect_trajectories(args.split, scene_filter, args.workers, args.limit)
        if args.verify:
            worst = verify_against_scene_api(args.split, scene_filter, tokens, log_names, poses, args.verify)
            print(f"[verify] max |fast - Scene.get_future_trajectory| over {args.verify} scenes: {worst:.2e}")
            meta["verify_max_abs_diff"] = worst
            assert worst < 1e-4, "fast pose extraction disagrees with the Scene API"
    t_load = time.time() - t_start
    num_loaded = len(tokens)
    print(f"[load] {num_loaded} trajectories in {t_load:.0f}s, poses {poses.shape}")

    subsampled = False
    if args.max_trajectories is not None and num_loaded > args.max_trajectories:
        keep = np.sort(rng.choice(num_loaded, args.max_trajectories, replace=False))
        tokens, log_names, poses = [tokens[i] for i in keep], [log_names[i] for i in keep], poses[keep]
        subsampled = True
        print(f"[load] subsampled uniformly to {len(tokens)} trajectories")
    np.savez(out / "source_trajectories.npz", tokens=np.array(tokens), log_names=np.array(log_names),
             poses=poses)

    # held-out split for coverage stats
    n_held = min(args.heldout, len(poses) // 10)
    perm = rng.permutation(len(poses))
    held_idx, fit_idx = perm[:n_held], perm[n_held:]
    heldout, fit = poses[held_idx], poses[fit_idx]

    # ---- 2. path vocabulary ---------------------------------------------- #
    S, spacing = args.stations, args.station_spacing
    stations = np.empty((len(fit), S, 2), dtype=np.float32)
    lengths = np.empty(len(fit), dtype=np.float32)
    for i, p in enumerate(fit):
        stations[i], lengths[i] = resample_path(p[:, :2], S, spacing)
    t0 = time.time()
    centers, labels = kmeans(stations.reshape(len(fit), -1), args.paths, args.kmeans_iters, args.seed)
    paths = np.concatenate([np.zeros((args.paths, 1, 2)), centers.reshape(args.paths, S, 2)],
                           axis=1).astype(np.float32)
    counts = np.bincount(labels, minlength=args.paths)
    path_len = np.zeros(args.paths, dtype=np.float32)
    np.add.at(path_len, labels, lengths)
    path_len = (path_len / np.maximum(counts, 1)).astype(np.float32)
    print(f"[paths] k-means P={args.paths} on {len(fit)}x{S * 2} in {time.time() - t0:.0f}s; "
          f"member counts min/median/max = {counts.min()}/{int(np.median(counts))}/{counts.max()}")
    np.save(out / f"paths_{args.paths}.npy", paths)
    np.save(out / f"paths_{args.paths}_len.npy", path_len)

    # ---- 3. speed vocabulary --------------------------------------------- #
    xy0 = np.concatenate([np.zeros((len(fit), 1, 2)), fit[:, :, :2]], axis=1)
    s_human = cumulative_arc_length(xy0)[:, 1:]                           # [N, T]
    s_aug = [s_human] + [np.minimum(s_human * k, SPEED_MAX_DISTANCE) for k in SPEED_AUG_SCALES]
    s_all = np.concatenate(s_aug, axis=0)
    t0 = time.time()
    centers, labels = kmeans(s_all, args.speeds, args.kmeans_iters, args.seed)
    speeds = np.maximum.accumulate(np.maximum(centers, 0.0), axis=1).astype(np.float32)
    speeds = speeds[np.argsort(speeds[:, -1])]            # sort by final distance (cosmetic)
    counts_v = np.bincount(labels, minlength=args.speeds)
    print(f"[speeds] k-means V={args.speeds} on {len(s_all)}x{s_all.shape[1]} in {time.time() - t0:.0f}s; "
          f"final distance range {speeds[:, -1].min():.1f}..{speeds[:, -1].max():.1f} m; "
          f"member counts min/max = {counts_v.min()}/{counts_v.max()}")
    np.save(out / f"speeds_{args.speeds}.npy", speeds)

    # ---- 4. coverage ------------------------------------------------------ #
    t0 = time.time()
    cov_vocab = coverage_composed(paths, speeds, spacing, heldout)
    cov_vocab_fit = coverage_composed(paths, speeds, spacing, fit[rng.choice(len(fit), n_held, replace=False)])
    cov = {"heldout_trajectories": int(n_held),
           f"factorized_P{args.paths}xV{args.speeds}": cov_vocab,
           f"factorized_P{args.paths}xV{args.speeds}_train_subset": cov_vocab_fit}
    anchors_path = Path(args.anchors)
    if anchors_path.exists():
        anchors = np.load(anchors_path)
        cov[f"anchors_{len(anchors)}"] = coverage_anchors(anchors, heldout)
        cov[f"anchors_{len(anchors)}_train_subset"] = coverage_anchors(
            anchors, fit[rng.choice(len(fit), n_held, replace=False)])
        cov["anchors_file"] = str(anchors_path)
    print(f"[coverage] ({time.time() - t0:.0f}s) " + json.dumps(cov, indent=1))

    # ---- 5. meta ---------------------------------------------------------- #
    meta.update({
        "split": args.split, "scene_filter": args.scene_filter,
        "num_paths": args.paths, "num_speeds": args.speeds,
        "num_stations": S, "station_spacing_m": spacing,
        "path_shape": list(paths.shape), "speed_shape": list(speeds.shape),
        "num_future_poses": NUM_FUTURE_POSES, "interval_length_s": INTERVAL_LENGTH,
        "timestamps_s": [round(INTERVAL_LENGTH * (k + 1), 2) for k in range(NUM_FUTURE_POSES)],
        "frame": "ego rear-axle frame at the current frame, x forward, y left, heading = path tangent",
        "num_source_trajectories_loaded": int(num_loaded),
        "num_source_trajectories_used": int(len(poses)),
        "num_fit_trajectories": int(len(fit)),
        "subsampled": subsampled, "limit": args.limit,
        "speed_augmentation": {"scales": list(SPEED_AUG_SCALES), "max_distance_m": SPEED_MAX_DISTANCE,
                               "num_speed_samples": int(len(s_all))},
        "kmeans": {"init": "k-means++", "iters": args.kmeans_iters, "seed": args.seed},
        "path_member_counts": {"min": int(counts.min()), "median": float(np.median(counts)),
                               "max": int(counts.max())},
        "path_len_m": {"min": float(path_len.min()), "mean": float(path_len.mean()),
                       "max": float(path_len.max())},
        "coverage_ade_m": cov,
        "runtime_s": {"load": round(t_load, 1), "total": round(time.time() - t_start, 1)},
        "files": sorted(p.name for p in out.iterdir()),
    })
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] wrote {out} in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
