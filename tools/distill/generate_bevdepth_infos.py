#!/usr/bin/env python3
"""Generate BEVDepth-native infos without overwriting MMDetection infos."""
import argparse
import os
from pathlib import Path
import sys

import mmcv
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-repo', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f'{output} already exists; pass --overwrite explicitly')

    sys.path.insert(0, os.path.abspath(args.teacher_repo))
    from scripts.gen_info import generate_info
    nusc = NuScenes(
        version='v1.0-trainval', dataroot=args.data_root, verbose=True)
    scene_names = splits.train if args.split == 'train' else splits.val
    infos = generate_info(nusc, scene_names)
    output.parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump(infos, str(output))
    print(f'wrote {len(infos)} {args.split} samples to {output}')


if __name__ == '__main__':
    main()
