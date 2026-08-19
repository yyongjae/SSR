"""Regression tests for the training-time aux QUALITY metrics.

These are the numbers used to decide, mid-run, whether an aux head is learning
or has quietly died. Three of them were wrong:

* ``occ_sep/*`` masked unsupervised frames out of the numerator but left them
  in the negative denominator, deflating ``p_off`` and inflating the
  separation ratio -- the one number whose entire job is to say "this head is
  outputting a constant".
* ``map_pts_err_m`` averaged ``|dx|`` and ``|dy|`` instead of taking the
  Euclidean norm, under-reporting a single-axis error by 2x.
* ``map_pos_frac`` was documented as a query-collapse detector but is
  identically ``sum(n_gt) / (B * n_query)`` -- Hungarian matching has no cost
  threshold, so it does not depend on the prediction at all.

Exits non-zero on failure.
"""
import importlib
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from mmcv import Config

importlib.import_module('projects.mmdet3d_plugin')
from mmdet3d.models import build_model  # noqa: E402

fails = []
CFG = 'projects/configs/SSR/PARA_SSR_e2e_2gpu_b4.py'
cfg = Config.fromfile(CFG)
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'))

# ------------------------------------------------------------------ occ ----
# The occupancy head is disabled in every config (report #07: it never reached
# a usable operating point). Its definition is preserved at the base as
# `_occ_head` so these checks keep running -- the separation metric was WRONG
# and got fixed, and that fix has to stay correct for whenever occupancy is
# switched back on.
print('=== occ_sep must ignore unsupervised frames on BOTH sides ===')
from mmdet.models import build_head                              # noqa: E402
occ = model.occ_head
if occ is None:
    occ_cfg = cfg.get('_occ_head')
    assert occ_cfg is not None, 'no occupancy head definition to test against'
    occ = build_head(occ_cfg)
    print('  (occ_head is off in the config; testing the preserved definition)')
B, T, C, H, W = 2, occ.num_frames, occ.num_classes, occ.bev_h, occ.bev_w
gt = torch.zeros(B, T, C, H, W)
gt[:, :, :, :H // 2, :W // 2] = 1                    # 25% positive
valid = torch.ones(B, T, dtype=torch.float32)
valid[:, T // 2:] = 0                                # tail frames unsupervised

# a head that outputs a constant 0.30 everywhere except 0.80 on the GT
p = torch.full((B, T, C, H, W), 0.30)
p[gt.bool()] = 0.80
logits = torch.log(p / (1 - p))                      # inverse sigmoid

m = occ.train_metrics(logits, gt, valid, thresholds=(0.5,))
p_on, p_off = float(m['occ_sep/p_on']), float(m['occ_sep/p_off'])
ratio = float(m['occ_sep/ratio'])
print(f'  p_on {p_on:.4f} (want 0.80)   p_off {p_off:.4f} (want 0.30)   '
      f'ratio {ratio:.4f} (want 2.667)')
ok = abs(p_on - 0.8) < 1e-3 and abs(p_off - 0.3) < 1e-3 and abs(ratio - 8 / 3) < 1e-2
fails += [] if ok else ['occ_sep-values']

# and a genuinely constant head must read exactly 1.0 whatever the mask is
flat = torch.zeros(B, T, C, H, W)                    # sigmoid(0) = 0.5 everywhere
mc = occ.train_metrics(flat, gt, valid, thresholds=(0.5,))
print(f'  constant head ratio {float(mc["occ_sep/ratio"]):.4f} (want 1.0000)')
fails += [] if abs(float(mc['occ_sep/ratio']) - 1.0) < 1e-4 else ['occ_sep-constant']

# IoU must be unchanged by the fix: valid-only, and 1.0 for a perfect head
print(f'  IoU@0.5 {float(m["occ_iou0.5/mean"]):.4f} (want 1.0000)   '
      f'valid_frac {float(m["occ_sep/valid_frac"]):.4f} '
      f'(want {float(valid.mean()):.4f})')
ok = abs(float(m['occ_iou0.5/mean']) - 1.0) < 1e-6 and \
    abs(float(m['occ_sep/valid_frac']) - float(valid.mean())) < 1e-6
fails += [] if ok else ['occ_iou-or-validfrac']

# ------------------------------------------------------------------ map ----
print('\n=== map_pts_err_m must be Euclidean, and collapse must be visible ===')
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import (  # noqa: E402
    LiDARInstanceLines)
from projects.mmdet3d_plugin.SSR.utils.map_utils import (  # noqa: E402
    normalize_2d_pts)

head = model.map_head
Q, P = head.map_num_vec, head.map_num_pts_per_vec
pc = head.pc_range
span = torch.tensor([pc[3] - pc[0], pc[4] - pc[1]])
n_gt = 4
rng = np.random.RandomState(0)


def make_gt(n):
    """n straight polylines at random offsets, in the pipeline's container."""
    lines = []
    for i in range(n):
        x0 = rng.uniform(pc[0] + 2, pc[3] - 2)
        y0 = rng.uniform(pc[1] + 2, pc[4] - 2)
        t = np.linspace(0, 4, P)
        lines.append(np.stack([np.full(P, x0), y0 + t], axis=1))
    from shapely.geometry import LineString
    inst = LiDARInstanceLines(
        [LineString(l) for l in lines], sample_dist=1, num_samples=P,
        padding=False, fixed_num=P, padding_value=-10000,
        patch_size=(pc[4] - pc[1], pc[3] - pc[0]))
    return inst, lines


gt_inst, gt_lines = make_gt(n_gt)
gt_pts = getattr(gt_inst, 'fixed_num_sampled_points')       # [n, P, 2] metres
gt_labels = [torch.zeros(n_gt, dtype=torch.long)]


def run(pred_pts_m, cls_logit=8.0):
    """pred_pts_m: [Q, P, 2] in metres -> the metric dict."""
    norm = normalize_2d_pts(pred_pts_m[None].clone(), pc)   # -> (0,1)
    cls = torch.full((1, Q, head.map_cls_out_channels), -8.0)
    cls[0, :n_gt, 0] = cls_logit
    bbox = torch.cat([norm.min(2).values, norm.max(2).values], -1)
    preds = dict(map_all_cls_scores=cls[None], map_all_bbox_preds=bbox[None],
                 map_all_pts_preds=norm[None])
    return head.train_metrics(preds, [gt_inst], gt_labels)


# perfect prediction on the first n_gt queries, junk far away on the rest
base = torch.full((Q, P, 2), 0.0)
base[:, :, 0] = pc[0] + 1.0
base[:, :, 1] = torch.linspace(pc[1] + 1, pc[1] + 5, P)
base[:n_gt] = torch.as_tensor(np.asarray(gt_pts), dtype=torch.float32)

m0 = run(base)
print(f'  exact match          : err {float(m0["map_pts_err_m"]):.4f} m '
      '(want 0.0000)')
fails += [] if float(m0['map_pts_err_m']) < 1e-3 else ['map-err-exact']

for name, off, want in (('shifted x by 0.75 m', (0.75, 0.0), 0.75),
                        ('shifted y by 0.75 m', (0.0, 0.75), 0.75),
                        ('shifted 45deg 0.75 m',
                         (0.75 / 2 ** .5, 0.75 / 2 ** .5), 0.75)):
    shifted = base.clone()
    shifted[:n_gt, :, 0] += off[0]
    shifted[:n_gt, :, 1] += off[1]
    got = float(run(shifted)['map_pts_err_m'])
    ok = abs(got - want) < 0.02
    print(f'  {name:21s}: err {got:.4f} m (want {want:.4f}) '
          f'{"ok" if ok else "FAIL"}')
    fails += [] if ok else [f'map-err-{name}']

print('\n=== map_spread_m must detect query collapse (map_pos_frac cannot) ===')
collapsed = base.clone()
collapsed[:] = base[0]                                   # every query identical
m_norm, m_coll = run(base), run(collapsed)
print(f'  spread: healthy {float(m_norm["map_spread_m"]):.3f} m  '
      f'collapsed {float(m_coll["map_spread_m"]):.3f} m')
ok = float(m_coll['map_spread_m']) < 1e-3 and float(m_norm['map_spread_m']) > 1.0
fails += [] if ok else ['map-spread']

# the retired metric: identical for both, which is exactly why it is gone
n_pos = sum(int((torch.as_tensor(gt_pts).shape[0])) for _ in range(1))
print(f'  map_pos_frac would have been {n_pos / Q:.4f} for BOTH '
      '(it is sum(n_gt)/(B*Q), independent of the prediction)')
fails += [] if 'map_pos_frac' not in m_norm else ['map_pos_frac-still-present']
print(f'  map_n_gt {float(m_norm["map_n_gt"]):.2f} (want {n_gt}.00)')
fails += [] if abs(float(m_norm['map_n_gt']) - n_gt) < 1e-6 else ['map_n_gt']

print('\n=== map_conf_* must separate matched from unmatched queries ===')
print(f'  conf_pos {float(m_norm["map_conf_pos"]):.4f}   '
      f'conf_neg {float(m_norm["map_conf_neg"]):.4f}')
ok = float(m_norm['map_conf_pos']) > 0.9 and float(m_norm['map_conf_neg']) < 0.1
fails += [] if ok else ['map-conf']

print('\n' + ('ALL AUX-METRIC CHECKS PASS' if not fails
              else f'STILL FAILING: {fails}'))
sys.exit(1 if fails else 0)
