#!/usr/bin/env python3
"""Validate and precompute lightweight views of one teacher cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_ssr_bev import derived_maps, dump_json  # noqa: E402


TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
DET_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
MAP_CLASSES = ["divider", "ped_crossing", "boundary"]


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=("bevfusion_teacher", "maptrv2_teacher"))
    parser.add_argument("--inputs", default=str(here / "inputs.json"))
    parser.add_argument("--selection", default=None)
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                return h.hexdigest()
            h.update(block)


def cache_path(root, token):
    if not TOKEN_RE.fullmatch(token):
        raise ValueError(f"invalid token {token!r}")
    path = (Path(root) / "samples" / token[:2] / f"{token}.npz").resolve()
    samples_root = (Path(root) / "samples").resolve()
    if samples_root not in path.parents:
        raise ValueError(f"cache path escaped samples root: {path}")
    return path


def align_feature(feature, source):
    """Return [y,x,C], numeric dataset x/y, common ±15m × ±30m ROI."""
    f = np.asarray(feature, dtype=np.float32).reshape(100, 100, 256)
    # Teacher manifests store [x_index, y_index, C].
    yx = f.transpose(1, 0, 2)
    if source == "maptrv2_teacher":
        return yx
    centres = -54.0 + (np.arange(100, dtype=np.float32) + 0.5) * 1.08
    x_keep = np.abs(centres) <= 15.0
    y_keep = np.abs(centres) <= 30.0
    crop = yx[np.ix_(y_keep, x_keep)]
    return cv2.resize(crop, (100, 100), interpolation=cv2.INTER_LINEAR)


def box_corners_xy(boxes):
    output = []
    for row in boxes:
        x, y, _z, dx, dy, _dz, yaw = [float(v) for v in row[:7]]
        local = np.asarray([
            [-dx / 2, -dy / 2], [dx / 2, -dy / 2],
            [dx / 2, dy / 2], [-dx / 2, dy / 2],
        ], dtype=np.float64)
        # The cached tensor is the native mmdet3d LiDAR box tensor.  Match
        # LiDARInstance3DBoxes.corners, which applies the stored yaw with the
        # opposite sign in xy.
        c, s = np.cos(-yaw), np.sin(-yaw)
        rot = np.asarray([[c, -s], [s, c]])
        output.append((local @ rot.T + [x, y]).tolist())
    return output


def serialize_prediction(archive, source):
    if source == "bevfusion_teacher":
        boxes = archive["pred_boxes_3d"].astype(np.float32)
        scores = archive["pred_scores_3d"].astype(np.float32)
        labels = archive["pred_labels_3d"].astype(np.int64)
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError("BEVFusion final prediction lengths disagree")
        return {
            "detection": {
                "corners": box_corners_xy(boxes),
                "scores": scores.astype(float).tolist(),
                "labels": labels.astype(int).tolist(),
                "class_names": DET_CLASSES,
                "box_schema_assumption": "[x,y,z,dx,dy,dz,yaw,vx,vy]",
            }
        }
    points = archive["pred_pts_3d"].astype(np.float32)
    scores = archive["pred_scores_3d"].astype(np.float32)
    labels = archive["pred_labels_3d"].astype(np.int64)
    if not (len(points) == len(scores) == len(labels)):
        raise ValueError("MapTRv2 final prediction lengths disagree")
    return {
        "map": {
            "points": points.astype(float).tolist(),
            "scores": scores.astype(float).tolist(),
            "labels": labels.astype(int).tolist(),
            "class_names": MAP_CLASSES,
        }
    }


def main():
    args = parse_args()
    inputs = read_json(args.inputs)
    source = inputs["sources"][args.source]
    selection = read_json(
        args.selection or Path(inputs["output_root"]) / "selection.json")
    cache_manifest = read_json(source["manifest"])
    expected_format = args.source.replace("_teacher", "_prediction_bev_cache")
    if cache_manifest.get("state") != "complete":
        raise RuntimeError(f"teacher cache is not complete: {cache_manifest}")
    if cache_manifest.get("format") != expected_format:
        raise RuntimeError(
            f"wrong cache format for {args.source}: "
            f"{cache_manifest.get('format')!r} != {expected_format!r}")
    if cache_manifest.get("format_version") != 1:
        raise RuntimeError(
            f"unsupported cache format version: "
            f"{cache_manifest.get('format_version')!r}")
    if cache_manifest.get("dataset_size") != selection["dataset_size"]:
        raise RuntimeError("teacher cache dataset size differs from selection")
    if cache_manifest.get("bev_feature", {}).get("stored_shape") != [10000, 256]:
        raise RuntimeError("unexpected teacher BEV shape in manifest")

    root = Path(inputs["output_root"]) / "exports" / args.source
    derived_dir = root / "derived"
    pred_dir = root / "predictions"
    derived_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for i, row in enumerate(selection["targets"], 1):
        token = row["token"]
        path = cache_path(source["cache"], token)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            feature = align_feature(archive["bev_feature"], args.source)
            np.savez_compressed(derived_dir / f"{token}.npz",
                                **derived_maps(feature))
            dump_json(pred_dir / f"{token}.json",
                      serialize_prediction(archive, args.source))
        completed.append(token)
        if i % 10 == 0:
            print(f"[{args.source}] {i}/{len(selection['targets'])}", flush=True)
    dump_json(root / "manifest.json", {
        "schema_version": 1,
        "source_key": args.source,
        "label": source["label"],
        "cache": source["cache"],
        "cache_manifest": source["manifest"],
        "cache_manifest_sha256": sha256(source["manifest"]),
        "cache_format": cache_manifest.get("format"),
        "cache_format_version": cache_manifest.get("format_version"),
        "target_count": len(completed),
        "completed_tokens": completed,
        "common_coordinate_frame": {
            "horizontal": "local x (right/lateral)",
            "vertical": "local y (forward)",
            "x_range_m": [-15, 15],
            "y_range_m": [-30, 30],
        },
        "transform": (
            "reshape[x,y,C]->transpose[y,x,C]" if args.source == "maptrv2_teacher"
            else "reshape[x,y,C]->transpose[y,x,C]->crop common ROI->bilinear 100x100"),
        "native_manifest": cache_manifest,
    })


if __name__ == "__main__":
    main()
