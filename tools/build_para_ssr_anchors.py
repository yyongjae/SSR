#!/usr/bin/env python3
"""Build a train-only, fixed NAVSIM trajectory vocabulary for PARA-SSR.

Run from any directory, with the project's NAVSIM Python environment::

    python tools/build_para_ssr_anchors.py --data-root /path/to/dataset \
        --output /path/to/anchors_k16.npz

``data-root`` contains ``navsim_logs/trainval``, ``sensor_blobs`` and ``maps``.
The checked-in navtrain token allowlist AND the training-log split are always
applied. There is deliberately no validation/test split option. A bounded,
deterministic hash-priority sample is collected while loading one log at a
time, so vocabulary generation does not keep the entire dataset in memory.

Anchors are real representative trajectories, selected nearest to each XY
KMeans center within its cluster. Keeping the representative's complete SE(2)
trajectory avoids invalid arithmetic averaging of headings around +/-pi.
Coordinates are NAVSIM current-ego-frame absolute poses (x-forward, y-left,
heading), NOT SSR coordinates or displacement increments. The eight poses
are at 0.5, 1.0, ..., 4.0 seconds; the current/origin pose is excluded.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "navsim/planning/script/config"
NUM_POSES = 8
INTERVAL_LENGTH = 0.5


def cluster_trajectory_medoids(
    trajectories: np.ndarray, num_candidates: int = 16, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return [K,8,3] float32 representatives and their source row indices.

    Clustering uses XY only. The representative is nearest to the center
    among that cluster's assigned members, not an independently averaged yaw.
    """
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    trajectories = np.asarray(trajectories, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (NUM_POSES, 3):
        raise ValueError(f"Expected trajectories [N,{NUM_POSES},3], got {trajectories.shape}")
    if not np.isfinite(trajectories).all():
        raise ValueError("Trajectories contain NaN or infinity")
    if num_candidates < 1 or len(trajectories) < num_candidates:
        raise ValueError("num_candidates must be positive and <= the trajectory count")
    if seed < 0 or seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    xy = trajectories[..., :2].reshape(len(trajectories), -1)
    if len(np.unique(xy, axis=0)) < num_candidates:
        raise ValueError("Fewer distinct XY trajectories than requested candidates")

    # A single BLAS/OpenMP thread avoids oversubscription on data servers and
    # makes the reduction order independent of their thread-count settings.
    with threadpool_limits(limits=1):
        clustering = KMeans(n_clusters=num_candidates, random_state=seed, n_init=10).fit(xy)
    representatives = []
    for cluster_index, center in enumerate(clustering.cluster_centers_):
        member_indices = np.flatnonzero(clustering.labels_ == cluster_index)
        if not len(member_indices):
            raise ValueError("KMeans produced an empty cluster; use fewer candidates")
        distances = np.square(xy[member_indices] - center).sum(axis=1)
        representatives.append(member_indices[np.argmin(distances)])
    representative_indices = np.asarray(representatives, dtype=np.int64)
    anchors = trajectories[representative_indices].astype(np.float32)
    # NAVSIM convention uses a wrapped relative heading; preserve it even if a
    # caller supplied an equivalent angle outside the canonical range.
    anchors[..., 2] = np.arctan2(np.sin(anchors[..., 2]), np.cos(anchors[..., 2]))
    return anchors, representative_indices


def train_source_spec(
    navtrain: Mapping[str, Any], log_split: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the split and retain the official navtrain token allowlist."""
    if not {"train_logs", "val_logs", "test_logs"}.issubset(log_split):
        raise ValueError("Require explicit train_logs, val_logs and test_logs split definitions")
    train_logs = set(log_split.get("train_logs") or [])
    heldout_logs = set(log_split.get("val_logs") or []) | set(log_split.get("test_logs") or [])
    if not train_logs:
        raise ValueError("The training-log allowlist is empty")
    if train_logs & heldout_logs:
        raise ValueError("Training logs overlap validation/test logs; refusing possible leakage")
    navtrain_logs = set(navtrain.get("log_names") or [])
    tokens = navtrain.get("tokens")
    if not navtrain_logs or not tokens:
        raise ValueError("navtrain must contain both log_names and its token allowlist")
    selected_logs = sorted(navtrain_logs & train_logs)
    if not selected_logs:
        raise ValueError("navtrain and the training-log split do not intersect")
    spec = {
        "num_history_frames": int(navtrain["num_history_frames"]),
        "num_future_frames": int(navtrain["num_future_frames"]),
        "frame_interval": int(navtrain["frame_interval"]),
        "has_route": bool(navtrain["has_route"]),
        "max_scenes": None,
        "log_names": selected_logs,
        "tokens": sorted(set(tokens)),
    }
    if spec["num_future_frames"] < NUM_POSES:
        raise ValueError(f"navtrain must provide at least {NUM_POSES} future frames")
    return spec


def _token_priority(token: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{token}".encode()).digest(), "big")


def _sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def collect_train_trajectories(
    data_root: Path,
    source_spec: Mapping[str, Any],
    max_scenes: int = 4096,
    seed: int = 0,
    *,
    loader_cls: Any = None,
) -> tuple[np.ndarray, list[str], list[str], int]:
    """Stream train logs and return a deterministic, globally sampled subset.

    The sample is invariant to file iteration order. SceneLoader retains the
    complete navtrain token filter for every log. ``loader_cls`` exists solely
    to permit testing the leakage checks without accessing a dataset.
    """
    if max_scenes < 1:
        raise ValueError("max_scenes must be positive")
    if seed < 0 or seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    from navsim.common.dataclasses import SceneFilter, SensorConfig

    if loader_cls is None:
        from navsim.common.dataloader import SceneLoader

        loader_cls = SceneLoader
    log_path = Path(data_root) / "navsim_logs/trainval"
    if not log_path.is_dir():
        raise FileNotFoundError(f"Expected the trainval log directory at {log_path}")
    available_logs = {path.stem for path in log_path.glob("*.pkl")}
    source_logs = sorted(set(source_spec["log_names"]) & available_logs)
    if not source_logs:
        raise ValueError("No allowed training logs are present in the dataset")
    allowed_tokens = set(source_spec["tokens"])
    if not allowed_tokens:
        raise ValueError("An explicit navtrain token allowlist is required")

    # Negated priorities implement a bounded max-heap; only pose arrays are
    # retained, not complete scenes or image/map data.
    sample_heap: list[tuple[int, str, str, np.ndarray]] = []
    seen_tokens: set[str] = set()
    for log_index, log_name in enumerate(source_logs):
        filter_kwargs = dict(source_spec)
        filter_kwargs.update(log_names=[log_name], max_scenes=None)
        # SceneLoader prints a pair of start/end messages for every log; the
        # periodic progress below is more useful for a streaming operation.
        with redirect_stdout(io.StringIO()):
            loader = loader_cls(
                data_path=log_path,
                sensor_blobs_path=Path(data_root) / "sensor_blobs/trainval",
                scene_filter=SceneFilter(**filter_kwargs),
                sensor_config=SensorConfig.build_no_sensors(),
            )
        for token in sorted(loader.tokens):
            if token not in allowed_tokens:
                raise ValueError("SceneLoader returned a token outside the navtrain allowlist")
            if token in seen_tokens:
                raise ValueError(f"Duplicate token in multiple log files: {token}")
            seen_tokens.add(token)
            priority = _token_priority(token, seed)
            if len(sample_heap) >= max_scenes and priority >= -sample_heap[0][0]:
                continue
            scene = loader.get_scene_from_token(token)
            if scene.scene_metadata.log_name != log_name:
                raise ValueError("Scene metadata does not match its allowed training log")
            if scene.scene_metadata.initial_token != token:
                raise ValueError("Scene metadata does not match its selected token")
            trajectory = scene.get_future_trajectory(num_trajectory_frames=NUM_POSES)
            poses = np.asarray(trajectory.poses, dtype=np.float32)
            if poses.shape != (NUM_POSES, 3) or not np.isfinite(poses).all():
                raise ValueError(f"Invalid future trajectory for training token {token}")
            if not np.isclose(trajectory.trajectory_sampling.interval_length, INTERVAL_LENGTH):
                raise ValueError(f"Expected {INTERVAL_LENGTH}s future sampling for {token}")
            entry = (-priority, token, log_name, poses.copy())
            if len(sample_heap) < max_scenes:
                heapq.heappush(sample_heap, entry)
            else:
                heapq.heapreplace(sample_heap, entry)
        if (log_index + 1) % 50 == 0 or log_index + 1 == len(source_logs):
            print(
                f"Loaded {log_index + 1}/{len(source_logs)} training logs; "
                f"eligible scenes={len(seen_tokens)}, sampled={len(sample_heap)}",
                flush=True,
            )
        # Release this log before loading the next one.
        del loader
    if not sample_heap:
        raise ValueError("No training scenes pass the official navtrain filter")
    sample = sorted(sample_heap, key=lambda entry: entry[1])
    poses = np.stack([entry[3] for entry in sample])
    return poses, [entry[1] for entry in sample], [entry[2] for entry in sample], len(seen_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--max-scenes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_candidates < 1 or args.max_scenes < args.num_candidates:
        parser.error("Require 1 <= --num-candidates <= --max-scenes")
    if args.seed < 0 or args.seed > np.iinfo(np.uint32).max:
        parser.error("--seed must fit in an unsigned 32-bit integer")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        parser.error(f"Output already exists; choose a new path: {args.output}")

    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(args.data_root.resolve() / "maps"))
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(args.data_root.resolve()))
    import yaml

    navtrain_path = CONFIG_ROOT / "common/scene_filter/navtrain.yaml"
    split_path = CONFIG_ROOT / "training/default_train_val_test_log_split.yaml"
    yaml_loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    navtrain = yaml.load(navtrain_path.read_text(), Loader=yaml_loader)
    log_split = yaml.load(split_path.read_text(), Loader=yaml_loader)
    source_spec = train_source_spec(navtrain, log_split)
    trajectories, tokens, logs, eligible_count = collect_train_trajectories(
        args.data_root, source_spec, max_scenes=args.max_scenes, seed=args.seed
    )
    anchors, representative_indices = cluster_trajectory_medoids(
        trajectories, num_candidates=args.num_candidates, seed=args.seed
    )
    metadata = {
        "format_version": 1,
        "source_split": "train",
        "source_scene_filter": "navtrain",
        "source_log_split": str(split_path.relative_to(REPO_ROOT)),
        "navtrain_yaml_sha256": hashlib.sha256(navtrain_path.read_bytes()).hexdigest(),
        "log_split_yaml_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "source_token_sha256": _sequence_sha256(tokens),
        "source_log_sha256": _sequence_sha256(sorted(set(logs))),
        "source_count": len(tokens),
        "source_log_count": len(set(logs)),
        "eligible_scene_count": eligible_count,
        "max_scenes": args.max_scenes,
        "seed": args.seed,
        "sampling": "lowest_sha256(seed\\0token)",
        "clustering": "xy_kmeans_nearest_assigned_member",
        "kmeans_n_init": 10,
        "num_candidates": args.num_candidates,
        "coordinates": "navsim_current_ego_x_forward_y_left_heading",
        "pose_representation": "absolute_in_current_ego_frame",
        "includes_current_pose": False,
        "pose_times_seconds": [INTERVAL_LENGTH * step for step in range(1, NUM_POSES + 1)],
        "interval_length": INTERVAL_LENGTH,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an accidental replacement after a long run.
    with args.output.open("xb") as output_file:
        np.savez_compressed(
            output_file,
            anchors=anchors,
            interval_length=np.asarray(INTERVAL_LENGTH, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            source_tokens=np.asarray(tokens),
            source_log_names=np.asarray(logs),
            anchor_source_tokens=np.asarray(tokens)[representative_indices],
        )
    print(f"Saved {anchors.shape} train-only anchors to {args.output}", flush=True)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
