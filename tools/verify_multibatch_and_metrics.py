"""Regression tests for the P0 issues raised in review.

Scope, stated plainly because an earlier version printed "ALL P0 FIXES
VERIFIED" and implied more than it checked. These are UNIT checks:

* P0-1 greps the source of ``evaluate()`` for the dict unwrap. It does not
  spin up a two-rank ``EvalHook``. The real distributed path is only exercised
  by ``tools/verify_dist_eval.sh``.
* P0-3 calls ``compute_motion_metric_vip3d`` directly rather than going through
  ``simple_test_pts``.
* P0-4 feeds the head's ``train_metrics`` synthetic logits rather than running
  inference and reading the result dict.

Exits non-zero if any check fails.
"""
import os
import sys, os, warnings, importlib, copy
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from mmcv import Config
importlib.import_module('projects.mmdet3d_plugin')
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import (
    VADCustomNuScenesDataset as DS)
from mmdet3d.models import build_model
from mmdet3d.core.bbox import LiDARInstance3DBoxes

fails = []

# ---------------------------------------------------------------- P0-1 ----
print('=== P0-1: distributed eval passes a dict, not a list ===')
src = open('projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py').read()
i_eval = src.index('    def evaluate(self,\n                 results,')
body = src[i_eval:i_eval + 4000]
ok = "results = results['bbox_results']" in body
print(f"  evaluate() unwraps {{'bbox_results': ...}}: {ok}")
fails += [] if ok else ['P0-1']

# ---------------------------------------------------------------- P0-3 ----
print('\n=== P0-3: motion metric must not mutate detection labels ===')
cfg = Config.fromfile('projects/configs/SSR/PARA_SSR_e2e_2gpu_b4.py')
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'))
model.CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                 'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                 'traffic_cone')
fut_ts = model.det_motion_head.fut_ts
fut_mode = model.det_motion_head.fut_mode

n = 6
boxes = LiDARInstance3DBoxes(torch.randn(n, 9) * 3, box_dim=9)
labels = torch.tensor([0, 1, 3, 4, 8, 8])          # car truck bus trailer ped ped
scores = torch.tensor([0.9, 0.9, 0.9, 0.2, 0.9, 0.1])
res = dict(boxes_3d=boxes, scores_3d=scores, labels_3d=labels.clone(),
           trajs_3d=torch.zeros(n, fut_mode * fut_ts * 2))
before = res['labels_3d'].clone()

mot = copy.deepcopy(res)
keep = mot['scores_3d'] > model.motion_score_thresh
for k in ('boxes_3d', 'scores_3d', 'labels_3d', 'trajs_3d'):
    mot[k] = mot[k][keep]
gt_b = LiDARInstance3DBoxes(torch.randn(3, 9) * 3, box_dim=9)
gt_l = torch.tensor([0, 1, 8])
matched = model.assign_pred_to_gt_vip3d(mot, gt_b, gt_l.clone())
model.compute_motion_metric_vip3d(gt_b, gt_l.clone(),
                                  torch.zeros(3, fut_ts * 4 + 10),
                                  mot, matched, model.CLASSES)
same = torch.equal(before, res['labels_3d'])
print(f'  labels before      : {before.tolist()}')
print(f'  labels after motion: {res["labels_3d"].tolist()}')
print(f'  detection labels preserved: {same}')
print(f'  score filter (>{model.motion_score_thresh}) kept '
      f'{int(keep.sum())}/{n} boxes: {bool(keep.sum() < n)}')
fails += [] if (same and keep.sum() < n) else ['P0-3']

# ---------------------------------------------------------------- P0-4 ----
print('\n=== P0-4: occupancy metric must honour gt_occ_valid ===')
T, C, H, W = 7, 1, 100, 100
gt = torch.zeros(1, T, C, H, W)
gt[:, :, :, 40:60, 40:60] = 1.0                     # same blob every frame
valid = torch.zeros(1, T, dtype=torch.bool)
valid[0, :3] = True                                 # only frames 0..2 usable
pred = gt.clone()
pred[:, 3:] = 0.0                                   # wrong on invalid frames only

r = [dict(pts_bbox=dict(occ_scores=pred, occ_seg=pred.long(),
                        gt_occ_seg=gt, gt_occ_valid=valid))]
d_valid = DS.evaluate_occ(DS, r)
r_novalid = [dict(pts_bbox=dict(occ_scores=pred, occ_seg=pred.long(),
                                gt_occ_seg=gt))]
d_all = DS.evaluate_occ(DS, r_novalid)
iou_v = d_valid['occ_iou_thr0.5/mean']
iou_a = d_all['occ_iou_thr0.5/mean']
print(f'  valid-aware IoU : {iou_v:.6f}   (expect 1.0 -- perfect on frames 0..2)')
print(f'  all-frames  IoU : {iou_a:.6f}   (expect < 1.0 -- penalised on 3..6)')
ok4 = abs(iou_v - 1.0) < 1e-6 and iou_a < 0.9
fails += [] if ok4 else ['P0-4']

# training-time metric
print('\n  training-time train_metrics:')
from mmdet.models import build_head
head = build_head(cfg.model.occ_head)
logits = torch.where(pred.bool(), torch.tensor(10.), torch.tensor(-10.))
m_v = head.train_metrics(logits, gt, valid)
m_a = head.train_metrics(logits, gt, None)
print(f'    valid-aware IoU@0.5: {float(m_v["occ_iou0.5/mean"]):.6f}')
print(f'    all-frames  IoU@0.5: {float(m_a["occ_iou0.5/mean"]):.6f}')
ok4b = abs(float(m_v['occ_iou0.5/mean']) - 1.0) < 1e-6 and \
    float(m_a['occ_iou0.5/mean']) < 0.9
fails += [] if ok4b else ['P0-4-train']

# ---------------------------------------------------------------- P0-2 ----
print('\n=== P0-2: the 12-epoch config must stay 12 epochs ===')
c12 = Config.fromfile('projects/configs/SSR/PARA_SSR_e2e_2gpu_b4.py')
c60 = Config.fromfile('projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py')
print(f'  2gpu_b4        : {c12.runner["max_epochs"]} ep, '
      f'eval/{c12.evaluation["interval"]}, aux={c12.model.test_aux_heads}')
print(f'  2gpu_b4_60ep   : {c60.runner["max_epochs"]} ep, '
      f'eval/{c60.evaluation["interval"]}, aux={c60.model.test_aux_heads}')
ok2 = (c12.runner['max_epochs'] == 12 and c12.model.test_aux_heads is False
       and c60.runner['max_epochs'] == 60 and c60.evaluation['interval'] == 10
       and c60.model.test_aux_heads is True)
fails += [] if ok2 else ['P0-2']

print('\n' + ('ALL P0 UNIT CHECKS PASS' if not fails
              else f'STILL FAILING: {fails}'))
sys.exit(1 if fails else 0)
