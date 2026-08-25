"""Collect paired teacher/P BEV statistics for linear stitching.

The planning-only SSR checkpoint (P) is replayed chronologically. BEVFusion
and MapTRv2 features are read from the complete val caches. For every source
and spatial-alignment candidate this worker streams the sufficient statistics
needed to fit a per-cell 256x256 affine map to P's BEV. Only a deterministic
held-out stride is cached for the later frozen-planner evaluation.

Scenes, rather than frames, are split: scene_index % 5 == 0 is calibration;
the other 120 nuScenes val scenes are held out.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/yongjae/e2e/SSR')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from mmcv import Config
from mmcv.parallel import collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader, Subset

importlib.import_module('projects.mmdet3d_plugin')


DEV = 'cuda:0'
torch.backends.cuda.matmul.allow_tf32 = False
P_CONFIG = ROOT / 'work_dirs/ssr_noffp_2gpu_b4/SSR_noffp_e2e_2gpu_b4.py'
P_CHECKPOINT = ROOT / 'work_dirs/ssr_noffp_2gpu_b4/epoch_12.pth'
FIT_EVERY = 5
TEACHERS = {
    'bevfusion': Path('/data1/yong/teacher_cache/bevfusion_cache/cache_val_100x100'),
    'maptrv2': Path('/data1/yong/teacher_cache/maptrv2_cache/cache_val_100x100'),
}

# Selection is made using calibration scenes only; all candidates are then
# reported on held-out scenes. ``roll_control`` is deliberately wrong and
# checks that a channel map cannot silently repair spatial misalignment.
CANDIDATES = {
    'maptrv2': (
        'visualizer_current',
        'flip_lateral',
        'flip_longitudinal',
        'flip_both',
        'roll_control',
    ),
    'bevfusion': (
        'visualizer_current',
        'visualizer_flip_lateral',
        'manifest_physical',
        'manifest_no_lateral_flip',
        'roll_control',
    ),
}
GROUP = {
    ('maptrv2', name): 'maptrv2_common' for name in CANDIDATES['maptrv2']
}
GROUP.update({
    ('bevfusion', 'visualizer_current'): 'bevfusion_visualizer',
    ('bevfusion', 'visualizer_flip_lateral'): 'bevfusion_visualizer',
    ('bevfusion', 'roll_control'): 'bevfusion_visualizer',
    ('bevfusion', 'manifest_physical'): 'bevfusion_manifest',
    ('bevfusion', 'manifest_no_lateral_flip'): 'bevfusion_manifest',
})


def read_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def validate_caches(dataset_size: int):
    manifests = {}
    expected = {
        'bevfusion': 'bevfusion_prediction_bev_cache',
        'maptrv2': 'maptrv2_prediction_bev_cache',
    }
    for key, root in TEACHERS.items():
        manifest = read_json(root / 'manifest.json')
        if manifest.get('state') != 'complete':
            raise RuntimeError(f'{key} cache is incomplete')
        if manifest.get('format') != expected[key]:
            raise RuntimeError(f'{key} cache format mismatch: {manifest.get("format")}')
        if manifest.get('dataset_size') != dataset_size:
            raise RuntimeError(
                f'{key} dataset size {manifest.get("dataset_size")} != {dataset_size}')
        if manifest.get('bev_feature', {}).get('stored_shape') != [10000, 256]:
            raise RuntimeError(f'{key} BEV shape mismatch')
        manifests[key] = manifest
    return manifests


def cache_path(source: str, token: str) -> Path:
    return TEACHERS[source] / 'samples' / token[:2] / f'{token}.npz'


def load_raw(source: str, token: str) -> np.ndarray:
    path = cache_path(source, token)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        value = archive['bev_feature']
    if value.shape != (10000, 256):
        raise ValueError(f'{source} {token}: unexpected feature shape {value.shape}')
    if not np.isfinite(value).all():
        raise ValueError(f'{source} {token}: non-finite teacher BEV')
    return value


def load_teacher_pair(token: str) -> dict[str, np.ndarray]:
    return {source: load_raw(source, token) for source in TEACHERS}


def resize100(value: np.ndarray) -> np.ndarray:
    return cv2.resize(np.ascontiguousarray(value), (100, 100),
                      interpolation=cv2.INTER_LINEAR)


def aligned_variants(source: str, raw: np.ndarray) -> dict[str, np.ndarray]:
    """Convert cached [native_axis0,native_axis1,C] to P's [y,x,C]."""
    f = raw.astype(np.float32, copy=False).reshape(100, 100, 256)
    if source == 'maptrv2':
        # Manifest: axis0 is lateral x, axis1 is longitudinal y.
        current = np.ascontiguousarray(f.transpose(1, 0, 2))
        return {
            'visualizer_current': current,
            'flip_lateral': np.ascontiguousarray(current[:, ::-1]),
            'flip_longitudinal': np.ascontiguousarray(current[::-1]),
            'flip_both': np.ascontiguousarray(current[::-1, ::-1]),
            'roll_control': np.ascontiguousarray(
                np.roll(current, shift=(17, 23), axis=(0, 1))),
        }

    centres = -54.0 + (np.arange(100, dtype=np.float32) + 0.5) * 1.08
    lateral = np.abs(centres) <= 15.0
    longitudinal = np.abs(centres) <= 30.0

    # Existing visualizer calibration. It treats cached axis0 as numeric x
    # and axis1 as numeric y despite the native manifest's prose labels.
    yx = f.transpose(1, 0, 2)
    visual = resize100(yx[np.ix_(longitudinal, lateral)])

    # Native manifest interpretation: axis0=forward, axis1=left. P rows are
    # forward and columns are right, hence the lateral flip in ``physical``.
    manifest = resize100(f[np.ix_(longitudinal, lateral)])
    physical = np.ascontiguousarray(manifest[:, ::-1])
    return {
        'visualizer_current': np.ascontiguousarray(visual),
        'visualizer_flip_lateral': np.ascontiguousarray(visual[:, ::-1]),
        'manifest_physical': physical,
        'manifest_no_lateral_flip': np.ascontiguousarray(manifest),
        'roll_control': np.ascontiguousarray(
            np.roll(visual, shift=(17, 23), axis=(0, 1))),
    }


def build_p():
    cfg = Config.fromfile(str(P_CONFIG))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, str(P_CHECKPOINT), map_location='cpu', strict=False)
    model.to(DEV).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def zeros_y():
    return {
        'n': 0,
        'sy': torch.zeros(256, dtype=torch.float64, device=DEV),
        'YtY': torch.zeros(256, 256, dtype=torch.float64, device=DEV),
    }


def zeros_x():
    return {
        'n': 0,
        'sx': torch.zeros(256, dtype=torch.float64, device=DEV),
        'XtX': torch.zeros(256, 256, dtype=torch.float64, device=DEV),
    }


def zeros_xy():
    return {
        'n': 0,
        'XtY': torch.zeros(256, 256, dtype=torch.float64, device=DEV),
    }


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--nshard', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache-stride', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--teacher-workers', type=int, default=8)
    parser.add_argument('--teacher-prefetch', type=int, default=16)
    parser.add_argument('--limit-scenes', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(P_CONFIG))
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    dataset = build_dataset(cfg.data.test)
    manifests = validate_caches(len(dataset))

    scenes = list(dict.fromkeys(info['scene_token'] for info in dataset.data_infos))
    scene_index = {token: i for i, token in enumerate(scenes)}
    mine = {token for token in scenes
            if scene_index[token] % args.nshard == args.shard}
    if args.limit_scenes:
        mine = set([token for token in scenes if token in mine][:args.limit_scenes])
    indices = [i for i, info in enumerate(dataset.data_infos)
               if info['scene_token'] in mine]
    tokens = [dataset.data_infos[index]['token'] for index in indices]
    print(f'[teacher shard {args.shard}] {len(mine)} scenes, '
          f'{len(indices)} frames', flush=True)

    loader = torch.utils.data.DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(collate, samples_per_gpu=1))
    model = build_p()

    y_acc = {split: zeros_y() for split in ('fit', 'test')}
    x_acc = {
        source: {
            split: {group: zeros_x() for group in sorted(set(
                GROUP[(source, candidate)] for candidate in CANDIDATES[source]))}
            for split in ('fit', 'test')}
        for source in TEACHERS
    }
    xy_acc = {
        source: {
            split: {candidate: zeros_xy() for candidate in CANDIDATES[source]}
            for split in ('fit', 'test')}
        for source in TEACHERS
    }

    state = None
    current_scene = None
    frame_in_scene = 0
    previous_position = None
    previous_angle = None
    cache = []
    started = time.time()

    # npz decompression dominates if done serially. Keep a bounded window so
    # CPU decompression overlaps the image encoder without retaining the whole
    # 60 GB teacher split in memory.
    teacher_pool = ThreadPoolExecutor(max_workers=args.teacher_workers)
    prefetch = max(1, args.teacher_prefetch)
    teacher_futures = {
        index: teacher_pool.submit(load_teacher_pair, tokens[index])
        for index in range(min(prefetch, len(tokens)))
    }

    for local_n, data in enumerate(loader):
        _, kwargs = scatter_kwargs([], data, [0])
        item = kwargs[0]
        metas, image = item['img_metas'][0], item['img'][0]
        global_index = indices[local_n]
        info = dataset.data_infos[global_index]
        scene = metas[0]['scene_token']
        token = info['token']
        raw_teachers = teacher_futures.pop(local_n).result()
        next_prefetch = local_n + prefetch
        if next_prefetch < len(tokens):
            teacher_futures[next_prefetch] = teacher_pool.submit(
                load_teacher_pair, tokens[next_prefetch])
        if scene != info['scene_token']:
            raise RuntimeError(f'scene mismatch at index {global_index}')

        if scene != current_scene:
            current_scene = scene
            frame_in_scene = 0
            state = None
        position_now = copy.deepcopy(metas[0]['can_bus'][:3])
        angle_now = copy.deepcopy(metas[0]['can_bus'][-1])
        if frame_in_scene == 0:
            metas[0]['can_bus'][:3] = 0
            metas[0]['can_bus'][-1] = 0
        else:
            metas[0]['can_bus'][:3] -= previous_position
            metas[0]['can_bus'][-1] -= previous_angle
        previous_position, previous_angle = position_now, angle_now
        frame_in_scene += 1

        split = 'fit' if scene_index[scene] % FIT_EVERY == 0 else 'test'
        command_raw = item['ego_fut_cmd'][0]
        with torch.no_grad():
            features = model.extract_feat(img=image.clone(),
                                          img_metas=copy.deepcopy(metas))
            p_bev = model.pts_bbox_head(
                features, copy.deepcopy(metas), prev_bev=state,
                only_bev=True, cmd=command_raw)
            state = p_bev
            # Inputs are float16 caches / float32 encoder outputs.  Full
            # float64 GEMMs are ~60x slower on Ampere and add no source
            # precision.  Compute each frame's products in true FP32 (TF32 is
            # disabled above), then accumulate those products in float64.
            target = p_bev.reshape(-1, 256).float()

            ya = y_acc[split]
            ya['n'] += target.shape[0]
            ya['sy'] += target.sum(0).double()
            ya['YtY'] += (target.T @ target).double()

            should_cache = split == 'test' and global_index % args.cache_stride == 0
            raw_cache = {}
            for source in TEACHERS:
                raw = raw_teachers[source]
                if should_cache:
                    raw_cache[source] = torch.from_numpy(raw.copy()).half()
                variants = aligned_variants(source, raw)
                tensors = {
                    name: torch.from_numpy(value.reshape(-1, 256)).to(DEV).float()
                    for name, value in variants.items()
                }
                seen_groups = set()
                for candidate in CANDIDATES[source]:
                    tensor = tensors[candidate]
                    group = GROUP[(source, candidate)]
                    if group not in seen_groups:
                        xa = x_acc[source][split][group]
                        xa['n'] += tensor.shape[0]
                        xa['sx'] += tensor.sum(0).double()
                        xa['XtX'] += (tensor.T @ tensor).double()
                        seen_groups.add(group)
                    xya = xy_acc[source][split][candidate]
                    xya['n'] += tensor.shape[0]
                    xya['XtY'] += (tensor.T @ target).double()

            if should_cache:
                cache.append({
                    'idx': global_index,
                    'scene': scene,
                    'token': token,
                    'P': p_bev.half().cpu(),
                    'gt': item['ego_fut_trajs'][0][:, 0].float().cpu(),
                    'msk': item['ego_fut_masks'][0].reshape(1, -1).float().cpu(),
                    'cmd': command_raw.reshape(1, 3).float().cpu(),
                    **raw_cache,
                })

        if (local_n + 1) % 100 == 0:
            elapsed = time.time() - started
            eta = (len(indices) - local_n - 1) * elapsed / (local_n + 1) / 60
            print(f'[teacher shard {args.shard}] {local_n + 1}/{len(indices)} '
                  f'{elapsed / (local_n + 1) * 1000:.0f} ms/frame '
                  f'eta {eta:.1f} min', flush=True)

    payload = {
        'schema_version': 1,
        'shard': args.shard,
        'nshard': args.nshard,
        'models': list(TEACHERS),
        'candidates': CANDIDATES,
        'groups': {f'{source}::{candidate}': GROUP[(source, candidate)]
                   for source in TEACHERS for candidate in CANDIDATES[source]},
        'y_acc': cpu_tree(y_acc),
        'x_acc': cpu_tree(x_acc),
        'xy_acc': cpu_tree(xy_acc),
        'scenes': sorted(mine, key=lambda value: scene_index[value]),
        'scene_index': scene_index,
        'indices': indices,
        'cache_manifests': manifests,
        'p_config': str(P_CONFIG),
        'p_checkpoint': str(P_CHECKPOINT),
    }
    torch.save(payload, out / f'shard_{args.shard}.pt')
    torch.save(cache, out / f'cache_{args.shard}.pt')
    teacher_pool.shutdown(wait=True)
    print(f'[teacher shard {args.shard}] DONE {len(indices)} frames, '
          f'{len(cache)} cached, {(time.time() - started) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
