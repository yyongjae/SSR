#!/usr/bin/env python3
"""Replay selected nuScenes scenes and export SSR-family BEV representations.

This is deliberately single-GPU and chronological.  Splitting a scene across
workers would reset recurrent ``prev_bev`` at the shard boundary.  Run separate
model sources on GPU 0 and GPU 1 instead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Running ``python tools/bev_visualizer/...`` otherwise puts only this script's
# directory on sys.path, not the repository root that owns the ``projects``
# package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_UNEXPECTED = {
    "det_motion_head.code_weights",
    "map_head.map_code_weights",
}


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="source key from inputs.json")
    parser.add_argument("--inputs", default=str(here / "inputs.json"))
    parser.add_argument("--selection", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-raw", action="store_true",
                        help="do not retain the 256-channel float16 BEV")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(str(tmp), str(path))


def sha256(path, block_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_snapshot(root):
    def run(*args):
        return subprocess.check_output(args, cwd=root, text=True).strip()
    try:
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain")),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def import_plugin(cfg):
    if cfg.get("custom_imports"):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get("plugin", False):
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        parts = Path(plugin_dir.rstrip("/")).parts
        importlib.import_module(".".join(parts))


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return checkpoint, state


def load_model_and_dataset(source, device):
    cfg = Config.fromfile(source["config"])
    import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if cfg.model.get("type") == "ParaSSR":
        cfg.model.test_aux_heads = True
    test_cfg = cfg.data.test
    test_cfg.test_mode = True
    # The old noFFP dump may point at an obsolete map cache name.  This script
    # never invokes dataset.evaluate(), but using the real cache keeps dataset
    # construction and future extensions deterministic.
    if isinstance(test_cfg, dict) and "map_ann_file" in test_cfg:
        candidate = Path(str(test_cfg.map_ann_file))
        if not candidate.exists():
            fallback = Path("data/nuscenes/nuscenes_map_anns_val.json")
            if fallback.exists():
                test_cfg.map_ann_file = str(fallback)
    dataset = build_dataset(test_cfg)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint, state = checkpoint_state(source["checkpoint"])
    incompatible = model.load_state_dict(state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    disallowed = [key for key in unexpected if key not in ALLOWED_UNEXPECTED]
    if missing or disallowed:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={missing}, unexpected={unexpected}")
    meta = checkpoint.get("meta", {})
    model.CLASSES = meta.get("CLASSES", getattr(dataset, "CLASSES", None))
    model.to(device)
    model.eval()
    return cfg, dataset, model, meta, missing, unexpected


def unwrap_tensor(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, "data") and not torch.is_tensor(value):
        value = value.data
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
    return value


def as_numpy(value):
    value = unwrap_tensor(value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def ordered_xy_corners(boxes):
    if boxes is None or len(boxes) == 0:
        return []
    corners = as_numpy(boxes.corners)[..., :2]
    result = []
    for item in corners:
        # Eight 3-D corners reduce to four repeated xy points.  Rounding avoids
        # tiny z-face numerical differences before unique().
        points = np.unique(np.round(item, decimals=5), axis=0)
        centre = points.mean(axis=0)
        order = np.argsort(np.arctan2(points[:, 1] - centre[1],
                                     points[:, 0] - centre[0]))
        result.append(points[order].astype(float).tolist())
    return result


def safe_float_list(value):
    arr = as_numpy(value)
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0,
                        posinf=1e9, neginf=-1e9)
    return arr.tolist()


def serialize_prediction(result):
    if isinstance(result, list):
        if len(result) != 1:
            raise ValueError(f"expected batch one, got {len(result)}")
        result = result[0]
    pred = result.get("pts_bbox", result)
    output = {}
    if "ego_fut_preds" in pred:
        ego = as_numpy(pred["ego_fut_preds"])
        if ego.ndim == 4 and ego.shape[0] == 1:
            ego = ego[0]
        output["planning"] = {
            "trajectories": np.cumsum(ego, axis=-2).astype(float).tolist(),
        }
        if "ego_fut_cmd" in pred:
            cmd = as_numpy(pred["ego_fut_cmd"]).reshape(-1, 3)[0]
            output["planning"]["command_index"] = int(cmd.argmax())
    if all(key in pred for key in ("boxes_3d", "scores_3d", "labels_3d")):
        boxes = pred["boxes_3d"]
        output["detection"] = {
            "corners": ordered_xy_corners(boxes),
            "scores": safe_float_list(pred["scores_3d"]),
            "labels": as_numpy(pred["labels_3d"]).astype(int).tolist(),
        }
        if "trajs_3d" in pred:
            traj = as_numpy(pred["trajs_3d"])
            centres = as_numpy(boxes.gravity_center)[:, None, None, :2]
            # Current VAD-style decoder emits [N, modes, T*2]. Some older
            # heads emit [N,T,2]. Normalize both to [N,modes,T,2].
            if traj.ndim == 3 and traj.shape[-1] != 2 and traj.shape[-1] % 2 == 0:
                traj = traj.reshape(traj.shape[0], traj.shape[1], -1, 2)
            elif traj.ndim == 3 and traj.shape[-1] == 2:
                traj = traj[:, None, :, :]
            elif traj.ndim == 2 and traj.shape[-1] % 2 == 0:
                traj = traj.reshape(traj.shape[0], 1, -1, 2)
            if traj.ndim != 4 or traj.shape[-1] != 2:
                raise ValueError(f"unsupported motion trajectory shape {traj.shape}")
            absolute = np.cumsum(traj, axis=-2) + centres
            output["detection"]["trajectories"] = absolute.astype(float).tolist()
    map_keys = ("map_pts_3d", "map_scores_3d", "map_labels_3d")
    if all(key in pred for key in map_keys):
        output["map"] = {
            "points": safe_float_list(pred["map_pts_3d"]),
            "scores": safe_float_list(pred["map_scores_3d"]),
            "labels": as_numpy(pred["map_labels_3d"]).astype(int).tolist(),
        }
    return output


def normalize_bev(bev):
    bev = as_numpy(bev)
    if bev.ndim == 3 and bev.shape[0] == 1:
        bev = bev[0]
    if bev.shape == (10000, 256):
        return bev.reshape(100, 100, 256)
    if bev.shape == (256, 100, 100):
        return np.moveaxis(bev, 0, -1)
    raise ValueError(f"unexpected BEV shape {bev.shape}")


def derived_maps(bev, token_attention=None):
    f = bev.astype(np.float32)
    centred = f - f.mean(axis=(0, 1), keepdims=True)
    ego_proto = f[47:53, 47:53].mean(axis=(0, 1))
    ego_proto_norm = np.linalg.norm(ego_proto)
    cell_norm = np.linalg.norm(f, axis=-1)
    ego_cosine = np.einsum("hwc,c->hw", f, ego_proto) / np.maximum(
        cell_norm * ego_proto_norm, 1e-8)
    maps = {
        "rms": np.sqrt(np.mean(np.square(f), axis=-1)),
        "spatial_contrast": np.sqrt(np.mean(np.square(centred), axis=-1)),
        "channel_std": np.std(f, axis=-1),
        "abs_mean": np.mean(np.abs(f), axis=-1),
        "ego_cosine": ego_cosine,
    }
    if token_attention is not None:
        attn = as_numpy(token_attention)
        while attn.ndim > 3 and attn.shape[0] == 1:
            attn = attn[0]
        if attn.ndim == 3 and attn.shape[-1] == 10000:
            # [B, token, HW] -> [token, H, W]
            if attn.shape[0] == 1:
                attn = attn[0]
            attn = attn.reshape(attn.shape[0], 100, 100)
            maps["token_max"] = attn.max(axis=0)
            probs = np.maximum(attn, 0)
            probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-8)
            maps["token_entropy"] = -np.sum(
                probs * np.log(np.maximum(probs, 1e-8)), axis=0)
            maps["token_attention"] = attn
    return {key: np.asarray(value, dtype=np.float16)
            for key, value in maps.items()}


def main():
    args = parse_args()
    started = time.time()
    inputs = load_json(args.inputs)
    if args.source not in inputs["sources"]:
        raise KeyError(f"unknown source {args.source!r}")
    source = inputs["sources"][args.source]
    if source["kind"] != "ssr_checkpoint":
        raise ValueError(f"{args.source} is not an SSR checkpoint source")
    selection_path = args.selection or str(Path(inputs["output_root"]) / "selection.json")
    selection = load_json(selection_path)
    targets = {int(row["index"]): row for row in selection["targets"]}
    contexts = [int(x) for x in selection["context_indices"]]
    source_root = Path(inputs["output_root"]) / "exports" / args.source
    manifest_path = source_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"completed export already exists: {manifest_path}; "
            "pass --overwrite to replace it explicitly")
    derived_dir = source_root / "derived"
    raw_dir = source_root / "raw"
    pred_dir = source_root / "predictions"
    for path in (derived_dir, pred_dir):
        path.mkdir(parents=True, exist_ok=True)
    if not args.no_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    device_index = int(str(args.device).rsplit(":", 1)[-1])
    torch.cuda.set_device(device_index)
    cfg, dataset, model, ckpt_meta, missing, unexpected = \
        load_model_and_dataset(source, args.device)
    if len(dataset) != selection["dataset_size"]:
        raise RuntimeError(
            f"dataset size changed: {len(dataset)} != {selection['dataset_size']}")

    capture = {"token_attention": None}
    tokenlearner = getattr(model.pts_bbox_head, "tokenlearner", None)
    hook = None
    if tokenlearner is not None:
        def token_hook(_module, _args, output):
            if isinstance(output, (list, tuple)) and len(output) >= 2:
                capture["token_attention"] = output[1].detach()
        hook = tokenlearner.register_forward_hook(token_hook)

    completed = []
    frame_times = []
    last_scene = None
    with torch.no_grad():
        for position, index in enumerate(contexts, 1):
            info = dataset.data_infos[index]
            scene = info["scene_token"]
            if scene != last_scene:
                # The model also resets when it sees the token, but an explicit
                # reset makes scene replay safe if a future model changes that.
                if hasattr(model, "prev_frame_info"):
                    model.prev_frame_info["prev_bev"] = None
                    model.prev_frame_info["scene_token"] = None
                last_scene = scene
            is_target = index in targets
            if hasattr(model, "test_aux_heads"):
                model.test_aux_heads = bool(is_target)
            capture["token_attention"] = None
            sample = dataset[index]
            data = collate([sample], samples_per_gpu=1)
            data = scatter(data, [device_index])[0]
            tick = time.time()
            result = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize(torch.device(args.device))
            frame_times.append(time.time() - tick)
            if not is_target:
                if position % 50 == 0:
                    print(f"[{args.source}] {position}/{len(contexts)} context frames", flush=True)
                continue

            token = targets[index]["token"]
            if not TOKEN_RE.fullmatch(token):
                raise ValueError(f"invalid token {token!r}")
            bev = normalize_bev(model.prev_frame_info["prev_bev"])
            np.savez_compressed(
                derived_dir / f"{token}.npz",
                **derived_maps(bev, capture["token_attention"]))
            if not args.no_raw:
                np.save(raw_dir / f"{token}.npy", bev.astype(np.float16),
                        allow_pickle=False)
            dump_json(pred_dir / f"{token}.json", serialize_prediction(result))
            completed.append(token)
            print(
                f"[{args.source}] target {len(completed)}/{len(targets)} "
                f"idx={index} token={token[:8]} frame={frame_times[-1]:.3f}s",
                flush=True)

    if hook is not None:
        hook.remove()
    if set(completed) != {row["token"] for row in selection["targets"]}:
        missing_tokens = sorted(
            {row["token"] for row in selection["targets"]} - set(completed))
        raise RuntimeError(f"target export incomplete: {missing_tokens}")
    manifest = {
        "schema_version": 1,
        "source_key": args.source,
        "label": source["label"],
        "checkpoint": source["checkpoint"],
        "checkpoint_sha256": sha256(source["checkpoint"]),
        "config": source["config"],
        "config_sha256": sha256(source["config"]),
        "checkpoint_meta_epoch": ckpt_meta.get("epoch"),
        "checkpoint_meta_iter": ckpt_meta.get("iter"),
        "weight_type": "raw",
        "state_dict_missing": missing,
        "state_dict_unexpected_allowlisted": unexpected,
        "git": git_snapshot(inputs["project_root"]),
        "device": args.device,
        "target_count": len(completed),
        "context_frame_count": len(contexts),
        "mean_forward_seconds": float(np.mean(frame_times)),
        "p95_forward_seconds": float(np.percentile(frame_times, 95)),
        "wall_seconds": time.time() - started,
        "bev_shape": [100, 100, 256],
        "bev_dtype_on_disk": "float16",
        "raw_bev_saved": not args.no_raw,
        "temporal_policy": selection["temporal_policy"],
        "completed_tokens": completed,
    }
    dump_json(source_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
