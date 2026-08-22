"""Pre-build the vectorised-map GT cache used by chamfer mAP evaluation.

``VADCustomNuScenesDataset._format_gt`` writes ``map_ann_file`` the first time
map mAP is computed, which takes ~30 minutes over the 6019 val samples. Doing
that inside the first in-training validation stalls the run, so build it once
up front. CPU only -- safe to run while GPUs are busy.

    python tools/build_map_gt_cache.py [config]
"""
import argparse
import importlib
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'config', nargs='?',
        default='projects/configs/SSR/PARA_SSR_e2e_12ep.py')
    parser.add_argument('--split', default='test', choices=['val', 'test'],
                        help='which data split entry to read map_ann_file from')
    args = parser.parse_args()

    from mmcv import Config
    importlib.import_module('projects.mmdet3d_plugin')
    from mmdet3d.datasets import build_dataset

    cfg = Config.fromfile(args.config)
    ds_cfg = cfg.data[args.split]
    ann_file = ds_cfg.get('map_ann_file', None)
    if ann_file is None:
        raise SystemExit(f'data.{args.split} has no map_ann_file; nothing to do')
    if os.path.exists(ann_file):
        print(f'{ann_file} already exists -- nothing to do')
        return

    print(f'building map GT cache for data.{args.split} -> {ann_file}')
    ds = build_dataset(ds_cfg)
    ds._format_gt()
    print(f'done: {ann_file}')


if __name__ == '__main__':
    main()
