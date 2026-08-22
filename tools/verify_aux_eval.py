"""Positive / negative controls for the auxiliary-head evaluation metrics.

Run this after touching evaluate_map / evaluate_occ / the motion metrics. It
does not need a trained checkpoint: it feeds ground truth back in as the
prediction, then perturbs it by a known amount and checks the metric responds
the way the metric definition says it should.

    python tools/verify_aux_eval.py [--samples 12]

Earlier versions of these controls were wrong in two ways, so note the fixes:
  * the map shift used to add the same offset to x AND y, so a "0.75 m" shift
    was really 0.75*sqrt(2) = 1.06 m and said nothing precise about the chamfer
    thresholds. Shifts here are along a single axis.
  * the occupancy shift used torch.roll, which wraps cells round the far edge
    instead of moving them out of view. Here the tensor is padded and sliced.
"""
import argparse
import copy
import importlib
import os
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def shift_occ(t, cells, axis):
    """Translate along a BEV axis, filling vacated cells with zeros."""
    pad = [0, 0, 0, 0]                 # (W_left, W_right, H_top, H_bottom)
    if axis == 'row':
        pad[2 if cells > 0 else 3] = abs(cells)
    else:
        pad[0 if cells > 0 else 1] = abs(cells)
    out = F.pad(t, pad)
    if axis == 'row':
        return out[..., :t.shape[-2], :] if cells > 0 else out[..., -t.shape[-2]:, :]
    return out[..., :, :t.shape[-1]] if cells > 0 else out[..., :, -t.shape[-1]:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',
                    default='projects/configs/SSR/PARA_SSR_e2e_12ep.py')
    ap.add_argument('--samples', type=int, default=12)
    args = ap.parse_args()

    from mmcv import Config
    importlib.import_module('projects.mmdet3d_plugin')
    from mmdet3d.datasets import build_dataset
    from mmdet.datasets import build_dataloader

    cfg = Config.fromfile(args.config)
    tmpdir = tempfile.mkdtemp(prefix='auxctl_')
    cfg.data.test.map_ann_file = os.path.join(tmpdir, 'gt_subset.json')
    ds = build_dataset(cfg.data.test)
    ds.data_infos = ds.data_infos[:args.samples]
    dx = (ds.pc_range[3] - ds.pc_range[0]) / cfg.bev_w_    # 0.30 m per column
    dy = (ds.pc_range[4] - ds.pc_range[1]) / cfg.bev_h_    # 0.60 m per row
    failures = []

    # -------------------------------------------------------------- map ----
    print('=== map chamfer mAP: GT as prediction, shifted along +x only ===')
    print(f'    (BEV cell = {dx:.2f} m across x, {dy:.2f} m along y)')

    def gt_as_map_results(shift_m):
        out = []
        for i in range(len(ds)):
            d = ds.vectormap_pipeline({}, ds.data_infos[i])
            lines = d['map_gt_bboxes_3d'].data
            pts = lines.fixed_num_sampled_points.clone()
            pts[..., 0] += shift_m                      # x only
            out.append(dict(pts_bbox=dict(
                map_pts_3d=pts,
                map_labels_3d=d['map_gt_labels_3d'].data.clone(),
                map_scores_3d=torch.ones(pts.shape[0]),
                map_boxes_3d=lines.bbox.clone())))
        return out

    rows = []
    for shift in (0.0, 0.75, 1.25, 2.5):
        det = ds.evaluate_map(gt_as_map_results(shift),
                              jsonfile_prefix=os.path.join(tmpdir, f's{shift}'))
        rows.append((shift, det['NuscMap_chamfer/mAP'],
                     [det[f'NuscMap_chamfer/divider_AP_thr_{t}']
                      for t in (0.5, 1.0, 1.5)]))
    print(f'\n  {"x shift":>8} | {"mAP":>7} | divider AP @ thr 0.5 / 1.0 / 1.5')
    for shift, mAP, thr in rows:
        print(f'  {shift:6.2f} m | {mAP:7.4f} | '
              + '  '.join(f'{t:.3f}' for t in thr))
    exact = rows[0][1]
    print(f'\n  exact match mAP = {exact:.4f} (expect 1.0)')
    print('  a shift of s metres should zero the thresholds below s and leave '
          'the ones above it near 1.0')
    if abs(exact - 1.0) > 1e-6:
        failures.append('map exact-match mAP != 1.0')
    if rows[1][2][0] > 0.05 or rows[1][2][1] < 0.5:
        failures.append('0.75 m shift did not straddle the 0.5/1.0 thresholds')

    # -------------------------------------------------------------- occ ----
    print('\n=== occupancy IoU: GT as prediction, translated (no wrap-around) ===')
    dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=2, num_gpus=1,
                          dist=False, shuffle=False)
    gts, valids = [], []
    for i, data in enumerate(dl):
        if i >= len(ds):
            break
        gts.append(torch.as_tensor(data['gt_occ_seg'][0].data[0]).float())
        valids.append(torch.as_tensor(data['gt_occ_valid'][0].data[0]))

    print(f'  {"row shift":>10} | {"metres":>7} | {"IoU@0.5":>8}')
    occ_rows = []
    for cells in (0, 1, 3):
        res = [dict(pts_bbox=dict(occ_scores=shift_occ(g, cells, 'row'),
                                  gt_occ_seg=g, gt_occ_valid=v))
               for g, v in zip(gts, valids)]
        d = ds.evaluate_occ(res)
        occ_rows.append((cells, d['occ_iou_thr0.5/mean']))
        print(f'  {cells:10d} | {cells*dy:6.2f} m | {occ_rows[-1][1]:8.4f}')
    if abs(occ_rows[0][1] - 1.0) > 1e-6:
        failures.append('occ exact-match IoU != 1.0')
    if not (occ_rows[0][1] > occ_rows[1][1] > occ_rows[2][1]):
        failures.append('occ IoU is not monotonically decreasing with shift')

    # valid-frame masking
    g = gts[0].clone()
    v = torch.zeros_like(valids[0])
    v[..., :3] = True
    p = g.clone()
    p[:, 3:] = 0.0                       # wrong only where GT is unusable
    masked = ds.evaluate_occ([dict(pts_bbox=dict(
        occ_scores=p, gt_occ_seg=g, gt_occ_valid=v))])['occ_iou_thr0.5/mean']
    unmasked = ds.evaluate_occ([dict(pts_bbox=dict(
        occ_scores=p, gt_occ_seg=g))])['occ_iou_thr0.5/mean']
    print(f'\n  valid-frame masking: masked {masked:.4f} (expect 1.0), '
          f'unmasked {unmasked:.4f} (expect < 1.0)')
    if abs(masked - 1.0) > 1e-6 or unmasked >= 0.99:
        failures.append('occ valid-frame masking not applied')

    print('\n' + ('ALL CONTROLS PASSED' if not failures
                  else 'FAILURES:\n  - ' + '\n  - '.join(failures)))
    print(f'temp dir: {tmpdir}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
