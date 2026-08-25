#!/usr/bin/env python3
"""Fail-fast integrity checks for the generated BEV explorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import mmcv
from PIL import Image
from mmdet3d.core import LiDARInstance3DBoxes


TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256(path, block_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def sorted_xy(points):
    points = np.unique(np.round(np.asarray(points, dtype=np.float64), 5), axis=0)
    return points[np.lexsort((points[:, 1], points[:, 0]))]


def lidar_xy_corners(row):
    row = np.asarray(row, dtype=np.float32)
    box = LiDARInstance3DBoxes(
        row[None], box_dim=row.shape[-1], origin=(0.5, 0.5, 0.5))
    return sorted_xy(box.corners[0].numpy()[:, :2])


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(here / "inputs.json"))
    args = parser.parse_args()
    inputs = read(args.inputs)
    root = Path(inputs["output_root"])
    selection = read(root / "selection.json")
    info_payload = mmcv.load(inputs["val_infos"])
    infos = info_payload["infos"] if isinstance(info_payload, dict) else info_payload
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    targets = selection["targets"]
    tokens = [x["token"] for x in targets]
    indices = [x["index"] for x in targets]
    check(50 <= len(targets) <= 100, f"target count outside 50..100: {len(targets)}")
    check(len(tokens) == len(set(tokens)), "duplicate target token")
    check(len(indices) == len(set(indices)), "duplicate target index")
    check(all(TOKEN_RE.fullmatch(x) for x in tokens), "malformed target token")
    check(len(set(x["scene_token"] for x in targets)) == selection["selected_scene_count"],
          "selected scene count mismatch")
    locations = Counter(x["location"] for x in targets)
    check(len(locations) == 4, f"not all four locations represented: {locations}")
    conditions = Counter(tag for x in targets for tag in x["conditions"])
    check(conditions["night"] > 0 and conditions["rain"] > 0,
          f"night/rain missing: {conditions}")
    commands = Counter(x["command"] for x in targets)
    check(all(commands[x] > 0 for x in ("left", "right", "straight")),
          f"command missing: {commands}")

    for scene in selection["scenes"]:
        context = scene["context_indices"]
        target = scene["target_indices"]
        check(context == list(range(min(context), max(context) + 1)),
              f"non-contiguous temporal context: {scene['scene_name']}")
        check(set(target).issubset(context), f"target outside context: {scene['scene_name']}")
        scene_indices = [i for i, info in enumerate(infos)
                         if info["scene_token"] == scene["scene_token"]]
        check(context and context[0] == scene_indices[0],
              f"temporal replay does not start at scene origin: {scene['scene_name']}")
    check(sorted(set(i for scene in selection["scenes"]
                     for i in scene["context_indices"])) == selection["context_indices"],
          "global temporal context differs from scene contexts")

    for key, source in inputs["sources"].items():
        export = root / "exports" / key
        manifest_path = export / "manifest.json"
        check(manifest_path.is_file(), f"missing export manifest: {key}")
        if not manifest_path.is_file():
            continue
        manifest = read(manifest_path)
        check(manifest.get("source_key") == key, f"source key mismatch: {key}")
        check(manifest.get("target_count") == len(tokens), f"target count mismatch: {key}")
        check(manifest.get("completed_tokens", []) == tokens,
              f"token coverage mismatch: {key}")
        if source["kind"] == "ssr_checkpoint":
            check(manifest.get("checkpoint") == source["checkpoint"],
                  f"checkpoint path mismatch: {key}")
            check(manifest.get("config") == source["config"],
                  f"config path mismatch: {key}")
            check(manifest.get("checkpoint_sha256") == sha256(source["checkpoint"]),
                  f"checkpoint hash mismatch: {key}")
            check(manifest.get("config_sha256") == sha256(source["config"]),
                  f"config hash mismatch: {key}")
        else:
            check(manifest.get("cache") == source["cache"],
                  f"teacher cache path mismatch: {key}")
            check(manifest.get("cache_manifest") == source["manifest"],
                  f"teacher manifest path mismatch: {key}")
            check(manifest.get("cache_manifest_sha256") == sha256(source["manifest"]),
                  f"teacher manifest hash mismatch: {key}")
        for token in tokens:
            derived_path = export / "derived" / f"{token}.npz"
            pred_path = export / "predictions" / f"{token}.json"
            check(derived_path.is_file(), f"missing derived {key}/{token}")
            check(pred_path.is_file(), f"missing prediction {key}/{token}")
            if derived_path.is_file():
                with np.load(derived_path, allow_pickle=False) as archive:
                    for required in ("rms", "spatial_contrast", "channel_std",
                                     "abs_mean", "ego_cosine"):
                        check(required in archive.files,
                              f"{key}/{token} lacks {required}")
                        if required in archive.files:
                            array = archive[required]
                            check(array.shape == (100, 100),
                                  f"bad {required} shape {key}/{token}: {array.shape}")
                            check(np.isfinite(array).all(),
                                  f"non-finite {required}: {key}/{token}")
            if source["kind"] == "ssr_checkpoint":
                raw = export / "raw" / f"{token}.npy"
                check(raw.is_file(), f"missing raw BEV {key}/{token}")
                if raw.is_file():
                    array = np.load(raw, mmap_mode="r", allow_pickle=False)
                    check(array.shape == (100, 100, 256),
                          f"bad raw shape {key}/{token}: {array.shape}")
                    check(array.dtype == np.float16,
                          f"bad raw dtype {key}/{token}: {array.dtype}")

    site = root / "site"
    template = Path(__file__).resolve().parent / "site_template"
    for static in ("index.html", "app.js", "styles.css"):
        check((site / static).is_file(), f"site static file missing: {static}")
        if (site / static).is_file():
            check(sha256(site / static) == sha256(template / static),
                  f"generated/template static file mismatch: {static}")
    check((site / "data" / "provenance.json").is_file(), "site provenance missing")
    manifest_path = site / "data" / "manifest.json"
    check(manifest_path.is_file(), "site manifest missing")
    if manifest_path.is_file():
        manifest = read(manifest_path)
        check(len(manifest.get("samples", [])) == len(tokens), "site sample count mismatch")
        check([x.get("token") for x in manifest.get("samples", [])] == tokens,
              "site sample token/order mismatch")
        check(len(manifest.get("sources", [])) == len(inputs["sources"]),
              "site source count mismatch")
        check([x.get("key") for x in manifest.get("sources", [])] == list(inputs["sources"]),
              "site source key/order mismatch")
        for sample in manifest.get("samples", []):
            camera = site / sample["camera_mosaic"]
            overlay = site / sample["overlay"]
            check(camera.is_file(), f"camera mosaic missing: {sample['token']}")
            check(overlay.is_file(), f"overlay missing: {sample['token']}")
            if overlay.is_file():
                overlay_data = read(overlay)
                check(overlay_data.get("token") == sample["token"],
                      f"overlay token mismatch: {sample['token']}")
                check(list(overlay_data.get("sources", {})) == list(inputs["sources"]),
                      f"overlay source mismatch: {sample['token']}")
            if camera.is_file():
                with Image.open(camera) as image:
                    check(image.size == (960, 360),
                          f"camera mosaic size wrong: {sample['token']} {image.size}")
        for source in manifest.get("sources", []):
            for token in tokens:
                pca = site / source["pca_template"].replace("{token}", token)
                if "pca" in source["modes"]:
                    check(pca.is_file(), f"PCA image missing: {source['key']}/{token}")
                    if pca.is_file():
                        with Image.open(pca) as image:
                            check(image.size == (100, 100),
                                  f"PCA image size wrong: {source['key']}/{token}")
                for mode in source["modes"]:
                    if mode == "pca":
                        continue
                    for normalization in ("global", "local"):
                        relative = source["asset_template"].replace(
                            "{token}", token).replace("{mode}", mode).replace(
                                "{normalization}", normalization)
                        image_path = site / relative
                        check(image_path.is_file(),
                              f"heatmap missing: {source['key']}/{token}/{mode}/{normalization}")
                        if image_path.is_file():
                            with Image.open(image_path) as image:
                                check(image.size == (100, 100),
                                      f"heatmap size wrong: {source['key']}/{token}/{mode}/{normalization}")

        # Guard the easy-to-miss nuScenes LiDAR yaw sign convention against
        # both GT and the cached BEVFusion tensors. Corner order is ignored.
        for sample in manifest.get("samples", []):
            info = infos[sample["index"]]
            boxes = np.asarray(info.get("gt_boxes", np.empty((0, 7))))
            valid = np.asarray(info.get("valid_flag", np.ones(len(boxes), dtype=bool))).astype(bool)
            keep = valid & (np.abs(boxes[:, 0]) <= 15) & (np.abs(boxes[:, 1]) <= 30) \
                if len(boxes) else np.zeros(0, dtype=bool)
            if keep.any():
                overlay = read(site / sample["overlay"])
                rendered = sorted_xy(overlay["ground_truth"]["boxes"][0])
                expected = lidar_xy_corners(boxes[np.flatnonzero(keep)[0]])
                check(np.allclose(rendered, expected, atol=1e-4),
                      f"GT LiDAR yaw/corner mismatch: {sample['token']}")
                break
        first = manifest.get("samples", [None])[0]
        if first is not None:
            token = first["token"]
            cache_root = Path(inputs["sources"]["bevfusion_teacher"]["cache"])
            with np.load(cache_root / "samples" / token[:2] / f"{token}.npz",
                         allow_pickle=False) as archive:
                expected = lidar_xy_corners(archive["pred_boxes_3d"][0])
            overlay = read(site / first["overlay"])
            rendered = sorted_xy(
                overlay["sources"]["bevfusion_teacher"]["detection"]["corners"][0])
            check(np.allclose(rendered, expected, atol=1e-4),
                  f"BEVFusion LiDAR yaw/corner mismatch: {token}")

    if failures:
        print("BEV VISUALIZER VERIFICATION FAILED")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print(json.dumps({
        "status": "PASS",
        "targets": len(tokens),
        "scenes": selection["selected_scene_count"],
        "locations": locations,
        "conditions": conditions,
        "commands": commands,
        "sources": list(inputs["sources"]),
        "site": str(site),
    }, indent=2, default=dict))


if __name__ == "__main__":
    main()
