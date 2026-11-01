import importlib, sys, os, time
sys.path.insert(0, os.getcwd())
from mmcv import Config
from mmdet3d.datasets import build_dataset

cfg = Config.fromfile('projects/configs/SSR/PARA_SSR_e2e_12ep.py')
importlib.import_module('projects.mmdet3d_plugin')

for split in ('val', 'train'):
    c = cfg.data[split].copy()
    c.pop('samples_per_gpu', None)
    if split == 'train':
        c['test_mode'] = False
    ds = build_dataset(c)
    print(f'{split:5s} dataset: {len(ds)} samples')
    t = time.time()
    s = ds[0]
    print(f'  sample 0 loaded in {time.time()-t:.1f}s; keys={sorted(s.keys())[:6]}')
    if split == 'val':
        ds._format_gt()
        print('  map GT written ->', ds.map_ann_file)
