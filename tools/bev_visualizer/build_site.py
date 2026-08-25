#!/usr/bin/env python3
"""Render the static BEV comparison site from completed source exports."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import mmcv
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from matplotlib import cm
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_teacher_cache import align_feature, cache_path  # noqa: E402


DET_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
MAP_CLASSES = ["divider", "ped_crossing", "boundary"]
CAMERAS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
)
MODES = (
    "spatial_contrast", "rms", "ego_cosine", "channel_std", "abs_mean",
    "token_max", "token_entropy",
)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(here / "inputs.json"))
    parser.add_argument("--selection", default=None)
    parser.add_argument("--skip-pca", action="store_true")
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(str(temp), str(path))


def sha256(path, block_size=4 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                return h.hexdigest()
            h.update(block)


def box_corners(row):
    x, y, _z, dx, dy, _dz, yaw = [float(v) for v in row[:7]]
    local = np.asarray([
        [-dx / 2, -dy / 2], [dx / 2, -dy / 2],
        [dx / 2, dy / 2], [-dx / 2, dy / 2],
    ])
    # nuScenes annotations are consumed as LiDARInstance3DBoxes.  Its yaw
    # convention rotates xy corners by -yaw (verified elementwise against
    # LiDARInstance3DBoxes(...).corners); using the usual +yaw CCW matrix
    # mirrors every displayed box orientation.
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot = np.asarray([[c, -s], [s, c]])
    return (local @ rot.T + [x, y]).astype(float).tolist()


def load_gt(info, map_vectors):
    boxes = np.asarray(info.get("gt_boxes", np.empty((0, 7))))
    names = np.asarray(info.get("gt_names", []), dtype=str)
    valid = np.asarray(info.get("valid_flag", np.ones(len(boxes), dtype=bool))).astype(bool)
    if len(boxes):
        keep = valid & (np.abs(boxes[:, 0]) <= 15) & (np.abs(boxes[:, 1]) <= 30)
    else:
        keep = np.zeros(0, dtype=bool)
    ego = np.asarray(info.get("gt_ego_fut_trajs", np.zeros((6, 2))), dtype=float)
    mask = np.asarray(info.get("gt_ego_fut_masks", np.ones(len(ego))), dtype=float)
    return {
        "boxes": [box_corners(row) for row in boxes[keep]],
        "box_labels": names[keep].tolist(),
        "map": [{
            "points": vector["pts"],
            "label": vector.get("cls_name", MAP_CLASSES[int(vector.get("type", 0))]),
        } for vector in map_vectors],
        "ego_trajectory": np.cumsum(ego * mask[:, None], axis=0).tolist(),
        "ego_mask": mask.astype(bool).tolist(),
    }


def resolve_camera_path(path, nuscenes_root):
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    marker = "/samples/"
    if marker in path:
        relative = "samples/" + path.split(marker, 1)[1]
        candidate = Path(nuscenes_root) / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def camera_mosaic(info, nuscenes_root, output):
    tile_size = (320, 180)
    canvas = Image.new("RGB", (tile_size[0] * 3, tile_size[1] * 2), "#08111d")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, name in enumerate(CAMERAS):
        path = resolve_camera_path(info["cams"][name]["data_path"], nuscenes_root)
        with Image.open(path) as image:
            tile = ImageOps.fit(image.convert("RGB"), tile_size,
                                method=Image.Resampling.LANCZOS)
        x = (i % 3) * tile_size[0]
        y = (i // 3) * tile_size[1]
        canvas.paste(tile, (x, y))
        label = name.replace("CAM_", "")
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((x + 5, y + 5, x + 13 + bbox[2], y + 22),
                       fill=(4, 12, 23, 210))
        draw.text((x + 9, y + 9), label, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "JPEG", quality=82, optimize=True, progressive=True)


def robust_limits(values, low=1.0, high=99.0):
    flat = np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1)
                           for value in values])
    finite = flat[np.isfinite(flat)]
    if not len(finite):
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def colorize(array, limits, cmap_name):
    lo, hi = limits
    normalized = np.clip((array.astype(np.float32) - lo) / (hi - lo), 0, 1)
    rgba = cm.get_cmap(cmap_name)(normalized, bytes=True)
    # Stored arrays have rows from y-min to y-max. Browser row 0 is top, so
    # flip once here to put +dataset-y at the top of every card.
    return Image.fromarray(np.flipud(rgba[..., :3]), "RGB")


def load_feature(inputs, source_key, token):
    source = inputs["sources"][source_key]
    if source["kind"] == "ssr_checkpoint":
        path = Path(inputs["output_root"]) / "exports" / source_key / "raw" / f"{token}.npy"
        return np.load(path, allow_pickle=False).astype(np.float32)
    path = cache_path(source["cache"], token)
    with np.load(path, allow_pickle=False) as archive:
        return align_feature(archive["bev_feature"], source_key)


def fit_pca(inputs, source_key, tokens, seed):
    rng = np.random.RandomState(seed)
    vectors = []
    for token in tokens:
        feature = load_feature(inputs, source_key, token).reshape(-1, 256)
        indices = rng.choice(len(feature), size=min(160, len(feature)), replace=False)
        vectors.append(feature[indices])
    fit = np.concatenate(vectors, axis=0)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(fit)
    # PCA signs are arbitrary. Make the largest-magnitude loading positive so
    # repeated builds produce stable RGB semantics for this source.
    signs = np.ones(3, dtype=np.float32)
    for i, component in enumerate(pca.components_):
        signs[i] = 1.0 if component[np.argmax(np.abs(component))] >= 0 else -1.0
    limits = []
    scores *= signs
    for i in range(3):
        limits.append(robust_limits([scores[:, i]], 1.0, 99.0))
    return pca, signs, limits


def render_pca(feature, pca, signs, limits):
    shape = feature.shape[:2]
    scores = pca.transform(feature.reshape(-1, 256)) * signs
    rgb = np.empty((len(scores), 3), dtype=np.uint8)
    for i, (lo, hi) in enumerate(limits):
        rgb[:, i] = np.clip((scores[:, i] - lo) / (hi - lo), 0, 1) * 255
    return Image.fromarray(np.flipud(rgb.reshape(*shape, 3)), "RGB")


def source_capabilities(key):
    if key in ("para_nonstaging", "para_stage1", "para_stage2"):
        return ["planning", "detection", "motion", "map", "token_attention"]
    if key == "ssr_noffp":
        return ["planning", "token_attention"]
    if key == "bevfusion_teacher":
        return ["detection"]
    return ["map"]


def main():
    args = parse_args()
    inputs = read_json(args.inputs)
    output_root = Path(inputs["output_root"])
    selection_path = Path(args.selection or output_root / "selection.json")
    selection = read_json(selection_path)
    source_keys = list(inputs["sources"])
    target_tokens = [row["token"] for row in selection["targets"]]
    target_by_index = {row["index"]: row for row in selection["targets"]}

    export_manifests = {}
    for key in source_keys:
        source = inputs["sources"][key]
        path = output_root / "exports" / key / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"source export is missing: {path}")
        manifest = read_json(path)
        if manifest.get("source_key") != key:
            raise RuntimeError(f"{key} export has the wrong source key")
        if manifest["completed_tokens"] != target_tokens:
            raise RuntimeError(f"{key} token coverage differs from selection")
        if source["kind"] == "ssr_checkpoint":
            if manifest.get("checkpoint") != source["checkpoint"] or \
                    manifest.get("config") != source["config"]:
                raise RuntimeError(f"{key} export was made from stale paths")
            if manifest.get("checkpoint_sha256") != sha256(source["checkpoint"]):
                raise RuntimeError(f"{key} checkpoint changed after export")
            if manifest.get("config_sha256") != sha256(source["config"]):
                raise RuntimeError(f"{key} config changed after export")
        else:
            if manifest.get("cache") != source["cache"] or \
                    manifest.get("cache_manifest") != source["manifest"]:
                raise RuntimeError(f"{key} export was made from a stale cache")
            if manifest.get("cache_manifest_sha256") != sha256(source["manifest"]):
                raise RuntimeError(f"{key} cache manifest changed after export")
        export_manifests[key] = manifest

    info_payload = mmcv.load(inputs["val_infos"])
    infos = info_payload["infos"] if isinstance(info_payload, dict) else info_payload
    for row in selection["targets"]:
        if not 0 <= row["index"] < len(infos):
            raise RuntimeError(f"selection index out of bounds: {row['index']}")
        info = infos[row["index"]]
        if info["token"] != row["token"] or \
                info["scene_token"] != row["scene_token"]:
            raise RuntimeError(
                f"stale selection at val index {row['index']}: "
                f"expected {row['token']}, found {info['token']}")

    map_payload = read_json(inputs["map_gt"])["GTs"]
    map_by_token = {row["sample_token"]: row["vectors"] for row in map_payload}
    missing_map = sorted(set(target_tokens) - set(map_by_token))
    if missing_map:
        raise RuntimeError(f"map GT lacks selected tokens: {missing_map[:5]}")

    site = output_root / "site"
    if site.exists():
        shutil.rmtree(site)
    (site / "assets" / "bev").mkdir(parents=True)
    (site / "assets" / "cameras").mkdir(parents=True)
    (site / "data" / "overlays").mkdir(parents=True)
    template = Path(__file__).resolve().parent / "site_template"
    for filename in ("index.html", "app.js", "styles.css"):
        shutil.copy2(template / filename, site / filename)

    # Load the very small derived arrays and establish fixed, per-source ranges.
    derived = {}
    limits = {}
    supported_modes = {}
    for key in source_keys:
        derived[key] = {}
        for token in target_tokens:
            path = output_root / "exports" / key / "derived" / f"{token}.npz"
            with np.load(path, allow_pickle=False) as archive:
                derived[key][token] = {name: archive[name].astype(np.float32)
                                       for name in archive.files
                                       if name != "token_attention"}
        supported = sorted(set.intersection(*[
            set(items) for items in derived[key].values()
        ]))
        supported_modes[key] = [mode for mode in MODES if mode in supported]
        limits[key] = {}
        for mode in supported_modes[key]:
            limits[key][mode] = robust_limits(
                [derived[key][token][mode] for token in target_tokens])

    pca_models = {}
    if not args.skip_pca:
        for source_index, key in enumerate(source_keys):
            print(f"Fitting PCA for {key} ...", flush=True)
            pca_models[key] = fit_pca(
                inputs, key, target_tokens,
                int(selection["seed"]) + 1000 + source_index)

    # Render source images with both fixed-global and explicitly local contrast.
    for key in source_keys:
        source_dir = site / "assets" / "bev" / key
        source_dir.mkdir(parents=True)
        for sample_index, token in enumerate(target_tokens, 1):
            for mode in supported_modes[key]:
                array = derived[key][token][mode]
                cmap_name = "coolwarm" if mode == "ego_cosine" else "magma"
                global_image = colorize(array, limits[key][mode], cmap_name)
                local_image = colorize(array, robust_limits([array]), cmap_name)
                global_image.save(source_dir / f"{token}_{mode}_global.png", optimize=True)
                local_image.save(source_dir / f"{token}_{mode}_local.png", optimize=True)
            if key in pca_models:
                feature = load_feature(inputs, key, token)
                image = render_pca(feature, *pca_models[key])
                image.save(source_dir / f"{token}_pca.png", optimize=True)
            if sample_index % 20 == 0:
                print(f"Rendered {key}: {sample_index}/{len(target_tokens)}", flush=True)

    sample_rows = []
    for row in selection["targets"]:
        token = row["token"]
        info = infos[row["index"]]
        camera_rel = f"assets/cameras/{token}.jpg"
        camera_mosaic(info, inputs["nuscenes_root"], site / camera_rel)
        overlay = {
            "token": token,
            "coordinate_frame": {
                "x_range_m": [-15, 15], "y_range_m": [-30, 30],
                "screen": "+local-y forward/up, +local-x right",
            },
            "ground_truth": load_gt(info, map_by_token[token]),
            "sources": {},
        }
        for key in source_keys:
            pred_path = output_root / "exports" / key / "predictions" / f"{token}.json"
            overlay["sources"][key] = read_json(pred_path)
        dump_json(site / "data" / "overlays" / f"{token}.json", overlay)
        site_row = dict(row)
        site_row.update({
            "camera_mosaic": camera_rel,
            "overlay": f"data/overlays/{token}.json",
        })
        sample_rows.append(site_row)

    sources = []
    for key, source in inputs["sources"].items():
        manifest = export_manifests[key]
        sources.append({
            "key": key,
            "label": source["label"],
            "color": source["color"],
            "kind": source["kind"],
            "epoch": manifest.get("checkpoint_meta_epoch"),
            "weight_type": manifest.get("weight_type"),
            "capabilities": source_capabilities(key),
            "modes": (["pca"] if key in pca_models else []) + supported_modes[key],
            "limits": limits[key],
            "asset_template": f"assets/bev/{key}/{{token}}_{{mode}}_{{normalization}}.png",
            "pca_template": f"assets/bev/{key}/{{token}}_pca.png",
            "provenance": manifest,
            "warning": (
                "Planner output is untrained in stage 1; hidden by default."
                if key == "para_stage1" else None),
        })
    manifest = {
        "schema_version": 1,
        "title": "SSR BEV Representation Explorer",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "qualitative_warning": (
            "This is a deliberately stratified qualitative subset, not an "
            "unbiased estimate of full-val metrics."),
        "feature_warning": (
            "PCA bases and channel identities are source-specific. Similar "
            "colors across different models do not imply equivalent features. "
            "The taps also differ: SSR transformer BEV, BEVFusion fused BEV "
            "before its decoder backbone, and MapTRv2 encoder BEV."),
        "coordinate_warning": (
            "The common local model frame uses +x right/lateral and +y "
            "forward. BEVFusion manifest prose names the native axes "
            "differently, but prediction/GT and feature-hotspot calibration "
            "supports the numeric identity alignment used here."),
        "selection": {
            key: selection[key] for key in (
                "method", "seed", "dataset_size", "target_count",
                "selected_scene_count", "context_frame_count", "summary",
                "temporal_policy")
        },
        "sources": sources,
        "samples": sample_rows,
        "classes": {"detection": DET_CLASSES, "map": MAP_CLASSES},
        "defaults": {
            "mode": "pca" if not args.skip_pca else "spatial_contrast",
            "normalization": "global",
            "det_threshold": 0.2,
            "map_threshold": 0.2,
        },
    }
    dump_json(site / "data" / "manifest.json", manifest)
    provenance = {
        "inputs_file": str(Path(args.inputs).resolve()),
        "inputs_sha256": sha256(args.inputs),
        "selection_file": str(selection_path.resolve()),
        "selection_sha256": sha256(selection_path),
        "val_infos": inputs["val_infos"],
        "val_infos_sha256": sha256(inputs["val_infos"]),
        "map_gt": inputs["map_gt"],
        "map_gt_sha256": sha256(inputs["map_gt"]),
        "exports": export_manifests,
        "render": {
            "orientation": "arrays [y_min..y_max,x_min..x_max], PNG vertical flip",
            "extent_m": {"x": [-15, 15], "y": [-30, 30]},
            "global_percentile": [1, 99],
            "local_percentile": [1, 99],
            "pca": "source-specific 3-component PCA; deterministic sign",
            "coordinate_calibration": (
                "Local +x=right, +y=forward; checked against ego command "
                "trajectories, GT/prediction geometry, and feature hotspots. "
                "BEVFusion native manifest prose is retained in its export "
                "manifest because it conflicts with the numeric evidence."),
        },
    }
    dump_json(site / "data" / "provenance.json", provenance)
    print(json.dumps({
        "site": str(site),
        "samples": len(sample_rows),
        "sources": source_keys,
        "serve": f"python -m http.server 8766 --bind 127.0.0.1 --directory {site}",
    }, indent=2))


if __name__ == "__main__":
    main()
