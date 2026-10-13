import importlib, sys, os
sys.path.insert(0, os.getcwd())
from mmcv import Config
from mmdet3d.models import build_model
import torch

cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'projects/configs/SSR/DISTILL_SSR_student_bevfusion_maptrv2.py'
cfg = Config.fromfile(cfg_path)
if cfg.get('plugin'):
    importlib.import_module('projects.mmdet3d_plugin')
print('config loaded :', cfg_path)
print('dataset_type  :', cfg.data.train.get('type'))
print('data_root     :', cfg.data.train.get('data_root'))
print('feature_root  :', cfg.model.get('distill', {}).get('feature_root'))
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
n = sum(p.numel() for p in model.parameters())
print('model built   :', type(model).__name__, f'{n/1e6:.1f}M params')
print('torch         :', torch.__version__, '| cuda', torch.cuda.is_available())
