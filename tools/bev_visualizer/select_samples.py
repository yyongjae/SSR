#!/usr/bin/env python3
"""Create a deterministic, auditable 80-sample nuScenes val selection.

The selection is scene-aware.  We first choose diverse scenes, then choose five
diverse frames in each scene.  Exporters must still run every frame from each
selected scene's beginning through its last target, so recurrent ``prev_bev``
is identical to ordered single-GPU evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import mmcv
import numpy as np


COMMANDS = ("right", "left", "straight")
LOCATIONS = (
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(here / "inputs.json"))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def condition_tags(description):
    text = (description or "").lower()
    tags = []
    if any(word in text for word in ("night", "dark")):
        tags.append("night")
    else:
        tags.append("day")
    if any(word in text for word in ("rain", "wet")):
        tags.append("rain")
    else:
        tags.append("dry")
    if "construction" in text:
        tags.append("construction")
    if any(word in text for word in ("dense", "heavy traffic", "many vehicle")):
        tags.append("dense-description")
    return tags


def safe_norm(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    return np.sqrt(np.square(x).sum(axis=axis))


def sample_record(index, info, scene_meta):
    boxes = np.asarray(info.get("gt_boxes", np.empty((0, 7))))
    names = np.asarray(info.get("gt_names", []), dtype=str)
    valid = np.asarray(info.get("valid_flag", np.ones(len(boxes), dtype=bool))).astype(bool)
    if len(boxes):
        in_range = ((np.abs(boxes[:, 0]) <= 15.0) &
                    (np.abs(boxes[:, 1]) <= 30.0))
        keep = valid & in_range
    else:
        keep = np.zeros(0, dtype=bool)
    speeds = safe_norm(info.get("gt_velocity", np.zeros((len(boxes), 2))))
    dynamic_names = {"car", "truck", "construction_vehicle", "bus", "trailer",
                     "motorcycle", "bicycle", "pedestrian"}
    dynamic = np.array([name in dynamic_names for name in names], dtype=bool)
    moving = keep & dynamic & np.isfinite(speeds) & (speeds > 0.5)
    ego_steps = np.asarray(info.get("gt_ego_fut_trajs", np.zeros((6, 2))), dtype=np.float32)
    ego_mask = np.asarray(info.get("gt_ego_fut_masks", np.ones(len(ego_steps))), dtype=np.float32)
    if len(ego_steps):
        cumulative = np.cumsum(ego_steps * ego_mask[:, None], axis=0)
        ego_distance = float(safe_norm(cumulative[-1]))
        lateral_excursion = float(np.max(np.abs(cumulative[:, 0])))
    else:
        ego_distance = lateral_excursion = 0.0
    cmd_arr = np.asarray(info.get("gt_ego_fut_cmd", [0, 0, 1]))
    cmd_idx = int(cmd_arr.argmax()) if cmd_arr.size else 2
    frame_idx = int(info.get("frame_idx", 0))
    scene_length = int(scene_meta["length"])
    phase = frame_idx / max(scene_length - 1, 1)
    return {
        "index": int(index),
        "token": info["token"],
        "scene_token": info["scene_token"],
        "scene_name": scene_meta.get("name", info["scene_token"][:8]),
        "frame_idx": frame_idx,
        "scene_length": scene_length,
        "scene_phase": round(float(phase), 6),
        "timestamp": int(info["timestamp"]),
        "location": info.get("map_location", scene_meta.get("location", "unknown")),
        "description": scene_meta.get("description", ""),
        "conditions": condition_tags(scene_meta.get("description", "")),
        "command": COMMANDS[cmd_idx] if cmd_idx < len(COMMANDS) else str(cmd_idx),
        "command_index": cmd_idx,
        "objects_in_bev": int(keep.sum()),
        "moving_objects": int(moving.sum()),
        "pedestrians": int((keep & (names == "pedestrian")).sum()),
        "vehicles": int((keep & np.isin(names, list(dynamic_names - {"pedestrian"}))).sum()),
        "ego_3s_distance_m": round(ego_distance, 4),
        "ego_lateral_excursion_m": round(lateral_excursion, 4),
        "future_valid": bool(info.get("fut_valid_flag", True)),
    }


def normalize_matrix(rows):
    x = np.asarray(rows, dtype=np.float64)
    lo = np.nanpercentile(x, 5, axis=0)
    hi = np.nanpercentile(x, 95, axis=0)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-8), 0.0, 1.0)


def one_hot(values):
    categories = sorted(set(values))
    lookup = {value: i for i, value in enumerate(categories)}
    out = np.zeros((len(values), len(categories)), dtype=np.float64)
    for row, value in enumerate(values):
        out[row, lookup[value]] = 1.0
    return out


def farthest_indices(features, count, seed, required_groups=None, max_per_group=None):
    """Deterministic farthest-point selection with optional group cap."""
    rng = np.random.RandomState(seed)
    n = len(features)
    if count >= n:
        return list(range(n))
    jitter = rng.uniform(0, 1e-9, size=n)
    centre = features.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.square(features - centre).sum(axis=1) + jitter))
    selected = [first]
    group_counts = Counter()
    if required_groups is not None:
        group_counts[required_groups[first]] += 1
    min_dist = np.square(features - features[first]).sum(axis=1)
    while len(selected) < count:
        scores = min_dist + jitter
        scores[selected] = -np.inf
        if required_groups is not None and max_per_group is not None:
            for i, group in enumerate(required_groups):
                if group_counts[group] >= max_per_group:
                    scores[i] = -np.inf
        nxt = int(np.argmax(scores))
        if not np.isfinite(scores[nxt]):
            for i in range(n):
                if i not in selected:
                    nxt = i
                    break
        selected.append(nxt)
        if required_groups is not None:
            group_counts[required_groups[nxt]] += 1
        min_dist = np.minimum(
            min_dist, np.square(features - features[nxt]).sum(axis=1))
    return selected


def main():
    args = parse_args()
    inputs = load_json(args.inputs)
    out_path = Path(args.output or Path(inputs["output_root"]) / "selection.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = mmcv.load(inputs["val_infos"])
    infos = payload["infos"] if isinstance(payload, dict) else payload
    version_dir = Path(inputs["nuscenes_root"]) / "v1.0-trainval"
    scenes_raw = load_json(version_dir / "scene.json")
    logs_raw = load_json(version_dir / "log.json")
    logs = {x["token"]: x for x in logs_raw}

    indices_by_scene = defaultdict(list)
    for idx, info in enumerate(infos):
        indices_by_scene[info["scene_token"]].append(idx)
    scene_table = {}
    for scene in scenes_raw:
        if scene["token"] not in indices_by_scene:
            continue
        log = logs.get(scene["log_token"], {})
        scene_table[scene["token"]] = {
            "name": scene.get("name", scene["token"][:8]),
            "description": scene.get("description", ""),
            "location": log.get("location", "unknown"),
            "length": len(indices_by_scene[scene["token"]]),
        }
    for token, idxs in indices_by_scene.items():
        scene_table.setdefault(token, {
            "name": token[:8], "description": "", "location": "unknown",
            "length": len(idxs)})

    records = [sample_record(i, info, scene_table[info["scene_token"]])
               for i, info in enumerate(infos)]
    records_by_scene = defaultdict(list)
    for record in records:
        records_by_scene[record["scene_token"]].append(record)

    scene_tokens = sorted(records_by_scene)
    scene_num = []
    scene_locations = []
    scene_conditions = []
    for token in scene_tokens:
        rows = records_by_scene[token]
        scene_num.append([
            np.mean([r["objects_in_bev"] for r in rows]),
            np.max([r["objects_in_bev"] for r in rows]),
            np.mean([r["moving_objects"] for r in rows]),
            np.mean([r["pedestrians"] for r in rows]),
            np.mean([r["ego_3s_distance_m"] for r in rows]),
            np.mean([r["ego_lateral_excursion_m"] for r in rows]),
            len(rows),
        ])
        scene_locations.append(rows[0]["location"])
        scene_conditions.append("+".join(rows[0]["conditions"][:2]))
    scene_features = np.concatenate([
        normalize_matrix(scene_num),
        one_hot(scene_locations) * 1.4,
        one_hot(scene_conditions) * 1.1,
    ], axis=1)
    spec = inputs["selection"]
    chosen_scene_rows = farthest_indices(
        scene_features, int(spec["scene_count"]), int(spec["seed"]),
        required_groups=scene_locations,
        max_per_group=max(1, math.ceil(int(spec["scene_count"]) / len(LOCATIONS))))
    chosen_scenes = [scene_tokens[i] for i in chosen_scene_rows]

    targets = []
    context_indices = []
    scene_summaries = []
    per_scene = int(spec["samples_per_scene"])
    for scene_rank, token in enumerate(chosen_scenes):
        rows = sorted(records_by_scene[token], key=lambda r: r["index"])
        numeric = [[
            r["scene_phase"], r["objects_in_bev"], r["moving_objects"],
            r["pedestrians"], r["ego_3s_distance_m"],
            r["ego_lateral_excursion_m"], float(r["future_valid"]),
        ] for r in rows]
        frame_features = np.concatenate([
            normalize_matrix(numeric),
            one_hot([r["command"] for r in rows]) * 1.25,
        ], axis=1)
        chosen_rows = farthest_indices(
            frame_features, per_scene, int(spec["seed"]) + scene_rank + 1)
        selected = sorted((rows[i] for i in chosen_rows), key=lambda r: r["index"])
        for r in selected:
            r["selection_reason"] = "scene-aware farthest-point sample"
            r["scene_rank"] = scene_rank
        targets.extend(selected)
        last_index = max(r["index"] for r in selected)
        context = [r["index"] for r in rows if r["index"] <= last_index]
        context_indices.extend(context)
        scene_summaries.append({
            "scene_token": token,
            "scene_name": rows[0]["scene_name"],
            "location": rows[0]["location"],
            "conditions": rows[0]["conditions"],
            "description": rows[0]["description"],
            "scene_length": len(rows),
            "target_indices": [r["index"] for r in selected],
            "context_indices": context,
        })

    targets.sort(key=lambda r: r["index"])
    context_indices = sorted(set(context_indices))
    summary = {
        "locations": dict(Counter(r["location"] for r in targets)),
        "commands": dict(Counter(r["command"] for r in targets)),
        "conditions": dict(Counter(tag for r in targets for tag in r["conditions"])),
        "object_count": {
            "min": min(r["objects_in_bev"] for r in targets),
            "median": float(np.median([r["objects_in_bev"] for r in targets])),
            "max": max(r["objects_in_bev"] for r in targets),
        },
        "moving_count": {
            "min": min(r["moving_objects"] for r in targets),
            "median": float(np.median([r["moving_objects"] for r in targets])),
            "max": max(r["moving_objects"] for r in targets),
        },
    }
    output = {
        "schema_version": 1,
        "method": "scene-aware farthest-point sampling",
        "seed": int(spec["seed"]),
        "dataset_size": len(infos),
        "target_count": len(targets),
        "selected_scene_count": len(chosen_scenes),
        "context_frame_count": len(context_indices),
        "temporal_policy": (
            "Process every frame from each selected scene start through its "
            "last target; save target frames only."),
        "summary": summary,
        "scenes": scene_summaries,
        "targets": targets,
        "context_indices": context_indices,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "output": str(out_path),
        "targets": len(targets),
        "scenes": len(chosen_scenes),
        "context_frames_per_model": len(context_indices),
        "summary": summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
