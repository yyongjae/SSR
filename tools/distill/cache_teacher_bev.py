#!/usr/bin/env python3
"""Cache aligned task-specific BEVs from frozen BEVDepth or HDMapNet.

Run this script in each teacher's native environment.  The cache is the stable
interface between those environments and SSR's legacy MMCV environment.

Feature taps:

* BEVDepth: detection head FPN output, ``[B, 256, 128, 128]``.
* HDMapNet: shared map-decoder feature before its semantic/instance/direction
  output heads, ``[B, 256, 100, 200]``.

Both are physically aligned to SSR's right-x / forward-y rectangle and
downsampled to 25x25 before being stored as fp16.  nuScenes teacher features
use forward-x / left-y, so the default transform is
``source_x=target_y, source_y=-target_x`` (swap x/y and flip source y).  The
adapter remains an MLP; the coordinate transform is fixed and carries no
trainable parameters.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SSR_ROOT = Path(__file__).resolve().parents[2]
UTIL_PATH = SSR_ROOT / 'projects/mmdet3d_plugin/SSR/utils/planning_distill.py'
_spec = importlib.util.spec_from_file_location(
    '_ssr_planning_distill_utils', str(UTIL_PATH))
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
align_bev_feature = _utils.align_bev_feature


class AttrDict(dict):
    """Minimal config object supporting both ``cfg.key`` and ``key in cfg``.

    P-MapNet's ``get_model`` uses both access styles.  ``SimpleNamespace`` only
    supports the former and fails before model construction.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


def sha256(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(root, teacher, token):
    token = str(token)
    return Path(root) / teacher / token[:2] / (token + '.pt')


def save_feature(root, teacher, token, feature, valid_mask, metadata,
                 overwrite=False):
    path = cache_path(root, teacher, token)
    if path.is_file() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    item = dict(
        token=str(token),
        teacher=teacher,
        feature=feature.detach().to('cpu', torch.float16),
        valid_mask=valid_mask.detach().to('cpu', torch.bool),
        metadata=metadata)
    temporary = path.with_suffix(path.suffix + f'.tmp-{os.getpid()}')
    torch.save(item, temporary)
    os.replace(temporary, path)
    return True


class FeatureHook:
    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        while isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError(f'feature hook received {type(output)}')
        self.value = output.detach()

    def pop(self):
        if self.value is None:
            raise RuntimeError('teacher feature hook was not called')
        value, self.value = self.value, None
        return value

    def close(self):
        self.handle.remove()


def load_checkpoint_state(path, strip_prefix):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if strip_prefix:
        state = {
            key[len(strip_prefix):] if key.startswith(strip_prefix) else key:
            value for key, value in state.items()
        }
    return state


def save_aligned_batch(args, tokens, native_feature, source_bounds, tap):
    if native_feature.size(1) != 256:
        raise ValueError(
            f'{tap} must expose 256 channels, got {tuple(native_feature.shape)}')
    aligned, valid = align_bev_feature(
        native_feature.float(), source_bounds, args.target_bounds,
        args.cache_size, swap_xy=args.swap_xy,
        flip_x=args.flip_x, flip_y=args.flip_y)
    saved = 0
    metadata = dict(
        tap=tap,
        native_shape=list(native_feature.shape[1:]),
        source_bounds=list(source_bounds),
        target_bounds=list(args.target_bounds),
        swap_xy=args.swap_xy,
        flip_x=args.flip_x,
        flip_y=args.flip_y)
    for index, token in enumerate(tokens):
        saved += int(save_feature(
            args.cache_root, args.teacher, token, aligned[index], valid[0],
            metadata, overwrite=args.overwrite))
    return saved


def cache_bevdepth(args):
    repo = os.path.abspath(args.teacher_repo)
    sys.path.insert(0, repo)
    from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
    from bevdepth.exps.nuscenes import base_exp
    from bevdepth.models.base_bev_depth import BaseBEVDepth

    backbone = copy.deepcopy(base_exp.backbone_conf)
    backbone['use_da'] = True
    # The released checkpoint contains the complete image backbone.  Suppress
    # an unnecessary network download during construction.
    backbone['img_backbone_conf']['init_cfg'] = None
    head = copy.deepcopy(base_exp.head_conf)
    head['bev_backbone_conf']['in_channels'] = 160  # current + previous key
    head['bev_neck_conf']['in_channels'] = [160, 160, 320, 640]
    head['train_cfg']['code_weights'] = [1.0] * 10
    model = BaseBEVDepth(backbone, head, is_train_depth=False)
    incompatible = model.load_state_dict(
        load_checkpoint_state(args.checkpoint, 'model.'), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'BEVDepth checkpoint mismatch: {incompatible}')
    model.to(args.device).eval().requires_grad_(False)
    hook = FeatureHook(model.head.neck)

    dataset = NuscDetDataset(
        ida_aug_conf=base_exp.ida_aug_conf,
        bda_aug_conf=base_exp.bda_aug_conf,
        classes=base_exp.CLASSES,
        data_root=args.data_root,
        info_paths=args.info_path,
        is_train=False,                 # deterministic image/BDA transforms
        use_cbgs=False,
        num_sweeps=1,
        img_conf=base_exp.img_conf,
        return_depth=False,
        sweep_idxes=[],
        key_idxes=[-1],
        use_fusion=False)
    first = dataset.infos[0]
    required = {'sample_token', 'cam_infos', 'lidar_infos'}
    if not required.issubset(first):
        raise ValueError(
            f'{args.info_path} is not a BEVDepth info file; missing '
            f'{sorted(required - set(first))}. Generate it with the official '
            f'BEVDepth scripts/gen_info.py under a distinct filename.')
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda batch: collate_fn(batch, is_return_depth=False))

    seen = saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='BEVDepth'):
            images, mats, timestamps, metas = batch[:4]
            tokens = [meta['token'] for meta in metas]
            if not args.overwrite and all(
                    cache_path(args.cache_root, args.teacher, token).is_file()
                    for token in tokens):
                seen += len(tokens)
                if args.limit and seen >= args.limit:
                    break
                continue
            images = images.to(args.device, non_blocking=True)
            mats = {key: value.to(args.device, non_blocking=True)
                    for key, value in mats.items()}
            timestamps = timestamps.to(args.device, non_blocking=True)
            model(images, mats, timestamps)
            feature = hook.pop()
            saved += save_aligned_batch(
                args, tokens, feature, (-51.2, -51.2, 51.2, 51.2),
                'model.head.neck')
            seen += len(tokens)
            if args.limit and seen >= args.limit:
                break
    hook.close()
    return seen, saved, dict(native_shape=[256, 128, 128],
                             source_bounds=[-51.2, -51.2, 51.2, 51.2])


class HDMapFeatureDataset(Dataset):
    """Only the inputs needed by HDMapNet; map-label rasterisation is skipped."""

    def __init__(self, version, dataroot, split):
        import numpy as np
        from nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
        self.np = np
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        scene_split = {'train': 'train', 'val': 'val'}[split]
        scenes = set(create_splits_scenes()[scene_split])
        self.samples = [sample for sample in self.nusc.sample
                        if self.nusc.get('scene', sample['scene_token'])['name']
                        in scenes]
        self.samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image
        from pyquaternion import Quaternion
        from data_osm.const import CAMS, IMG_ORIGIN_H, IMG_ORIGIN_W
        from data_osm.image import img_transform, normalize_img
        from data_osm.lidar import get_lidar_data
        from model.utils.voxel import pad_or_trim_to_np

        rec = self.samples[index]
        images, translations, rotations, intrinsics = [], [], [], []
        post_translations, post_rotations = [], []
        final_h, final_w = 128, 352
        resize = (final_w / IMG_ORIGIN_W, final_h / IMG_ORIGIN_H)
        for camera in CAMS:
            sample_data = self.nusc.get('sample_data', rec['data'][camera])
            image = Image.open(os.path.join(
                self.nusc.dataroot, sample_data['filename']))
            image, post_rotation, post_translation = img_transform(
                image, resize, (final_w, final_h))
            images.append(normalize_img(image))
            post_rotations.append(post_rotation)
            post_translations.append(post_translation)
            sensor = self.nusc.get(
                'calibrated_sensor', sample_data['calibrated_sensor_token'])
            translations.append(torch.tensor(sensor['translation']))
            rotations.append(torch.tensor(
                Quaternion(sensor['rotation']).rotation_matrix))
            intrinsics.append(torch.tensor(sensor['camera_intrinsic']))

        lidar = get_lidar_data(
            self.nusc, rec, nsweeps=3, min_distance=2.2).T
        count = lidar.shape[0]
        lidar = pad_or_trim_to_np(lidar, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920, dtype='float32')
        lidar_mask[count:] = 0
        lidar_sample = self.nusc.get(
            'sample_data', rec['data']['LIDAR_TOP'])
        pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        rotation = Quaternion(pose['rotation'])
        return (
            torch.stack(images), torch.stack(translations),
            torch.stack(rotations), torch.stack(intrinsics),
            torch.stack(post_translations), torch.stack(post_rotations),
            torch.from_numpy(lidar), torch.from_numpy(lidar_mask),
            torch.tensor(pose['translation']),
            torch.tensor(rotation.yaw_pitch_roll), rec['token'])


def cache_hdmapnet(args):
    repo = os.path.abspath(args.teacher_repo)
    sys.path.insert(0, repo)

    # HDMapNet constructs EfficientNet with from_pretrained even though its
    # complete weights are in the supplied checkpoint.  Avoid an unrelated
    # Internet download in a cache job.
    from efficientnet_pytorch import EfficientNet
    EfficientNet.from_pretrained = classmethod(
        lambda cls, model_name, *a, **kw: cls.from_name(model_name))
    from model import get_model

    cfg = AttrDict(
        model='HDMapNet_fusion', dataset='nuScenes', instance_seg=True,
        embedding_dim=16, direction_pred=True, angle_class=36)
    data_conf = dict(
        num_channels=4,
        image_size=[128, 352],
        xbound=[-30.0, 30.0, 0.15],
        ybound=[-15.0, 15.0, 0.15],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 45.0, 1.0],
        thickness=5,
        angle_class=36,
        patch_w=20,
        patch_h=20,
        mask_ratio=-1,
        mask_flag=False,
        sd_map_path='')
    model = get_model(cfg, data_conf, True, 16, True, 36)
    incompatible = model.load_state_dict(
        load_checkpoint_state(args.checkpoint, 'module.'), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'HDMapNet checkpoint mismatch: {incompatible}')
    model.to(args.device).eval().requires_grad_(False)
    hook = FeatureHook(model.bevencode.up1)
    dataset = HDMapFeatureDataset('v1.0-trainval', args.data_root, args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    seen = saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='HDMapNet'):
            *inputs, tokens = batch
            if not args.overwrite and all(
                    cache_path(args.cache_root, args.teacher, token).is_file()
                    for token in tokens):
                seen += len(tokens)
                if args.limit and seen >= args.limit:
                    break
                continue
            inputs = [value.to(args.device, non_blocking=True)
                      for value in inputs]
            model(*inputs, None)
            feature = hook.pop()
            saved += save_aligned_batch(
                args, tokens, feature, (-30.0, -15.0, 30.0, 15.0),
                'model.bevencode.up1')
            seen += len(tokens)
            if args.limit and seen >= args.limit:
                break
    hook.close()
    return seen, saved, dict(native_shape=[256, 100, 200],
                             source_bounds=[-30.0, -15.0, 30.0, 15.0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', required=True,
                        choices=('bevdepth', 'hdmapnet'))
    parser.add_argument('--teacher-repo', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--info-path',
                        help='BEVDepth-format info pkl (BEVDepth only)')
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--target-bounds', type=float, nargs=4,
                        default=(-15.0, -30.0, 15.0, 30.0),
                        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'))
    parser.add_argument('--cache-size', type=int, nargs=2, default=(25, 25),
                        metavar=('H', 'W'))
    # Teacher: x=forward, y=left.  SSR's LIDAR_TOP-aligned BEV: x=right,
    # y=forward.  Therefore source_x=target_y and source_y=-target_x.
    parser.set_defaults(swap_xy=True, flip_y=True)
    parser.add_argument('--no-swap-xy', dest='swap_xy', action='store_false')
    parser.add_argument('--flip-x', action='store_true')
    parser.add_argument('--flip-y', dest='flip_y', action='store_true')
    parser.add_argument('--no-flip-y', dest='flip_y', action='store_false')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0,
                        help='0 caches the complete split')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.teacher == 'bevdepth' and not args.info_path:
        parser.error('--info-path is required for BEVDepth')
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith('cuda'):
        raise RuntimeError('teacher feature extraction requires a CUDA device')
    if args.teacher == 'bevdepth':
        seen, saved, extra = cache_bevdepth(args)
    else:
        seen, saved, extra = cache_hdmapnet(args)

    manifest = dict(
        version=1,
        teacher=args.teacher,
        split=args.split,
        checkpoint=os.path.abspath(args.checkpoint),
        checkpoint_sha256=sha256(args.checkpoint),
        teacher_repo=os.path.abspath(args.teacher_repo),
        data_root=os.path.abspath(args.data_root),
        target_bounds=list(args.target_bounds),
        cache_size=list(args.cache_size),
        swap_xy=args.swap_xy,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        seen=seen,
        newly_saved=saved,
        **extra)
    manifest_path = Path(args.cache_root) / args.teacher / \
        f'manifest_{args.split}.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
