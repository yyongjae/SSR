"""Shared plumbing for the LMD experiments: checkpoints, model build, sample loop.

The sample loop deliberately drives the model's own `forward_test` rather than
re-implementing inference. `forward_test` owns the temporal bookkeeping -- the
per-scene reset of `prev_bev` and the can_bus pose deltas (para_ssr.py:695-728)
-- and getting that subtly wrong would change every BEV it produces. A tap on
`pts_bbox_head.forward` captures exactly the tensors the head saw.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

ROOT = '/data2/byounggun/rideflux'

# name -> (work_dir, config basename, epoch, condition)
CKPTS = {
    'plan_only':  (f'{ROOT}/ssr_noffp_2gpu_b4', 'SSR_noffp_e2e_2gpu_b4.py', 12, 'planning만 (SSR-noFFP)'),
    'aux_only':   (f'{ROOT}/para_ssr_stage1',   'PARA_SSR_stage1_detmap.py', 48, 'aux만 (plan=0)'),
    'staged':     (f'{ROOT}/para_ssr_stage2',   'PARA_SSR_stage2_all.py',    12, 'stage1 fork + planning'),
    'both':       (f'{ROOT}/para_ssr_60ep',     'PARA_SSR_e2e_60ep.py',      60, '둘 다 (monolithic)'),
}


def resolve(name, epoch=None):
    work_dir, cfg_name, ep, desc = CKPTS[name]
    ep = epoch if epoch is not None else ep
    cfg = os.path.join(work_dir, cfg_name)
    if not os.path.exists(cfg):                      # stage1's dump is named for stage2
        cands = [f for f in os.listdir(work_dir) if f.endswith('.py')]
        cfg = os.path.join(work_dir, sorted(cands)[0])
    ckpt = os.path.join(work_dir, f'epoch_{ep}_ema.pth')
    return cfg, ckpt, desc


def build(cfg_path, ckpt_path, device='cuda:0'):
    import mmcv
    from mmcv.runner import load_checkpoint
    from mmcv.cnn import fuse_conv_bn  # noqa: F401  (import order matters for mmcv ops)
    import projects.mmdet3d_plugin  # noqa: F401  registers SSR/ParaSSR and the datasets
    from mmdet3d.models import build_model

    cfg = mmcv.Config.fromfile(cfg_path)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    model.to(device).eval()
    model.fp16_enabled = False
    return model, cfg


def val_loader(cfg, n_samples=None):
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)
    loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=2,
                              dist=False, shuffle=False)
    return ds, loader


class HeadTap:
    """Capture the planner's inputs and outputs without changing what it does.

    Only the `only_bev=False` call is kept: `obtain_history_bev` rolls the BEV
    over the previous frames with `only_bev=True`, and those are not the frame
    being analysed.
    """

    def __init__(self, head):
        self.head = head
        self._orig = head.forward
        self.last = None
        head.forward = self._tap

    def _tap(self, *args, **kwargs):
        out = self._orig(*args, **kwargs)
        if not kwargs.get('only_bev', False):
            self.last = dict(args=args, kwargs=kwargs, out=out)
        return out

    def close(self):
        self.head.forward = self._orig


def iter_samples(model, loader, n, device='cuda:0'):
    """Yield (sample_index, data, tap.last) for the first `n` val samples.

    Runs the real `forward_test` under no_grad; the analysis re-runs the head
    afterwards with gradients on the captured tensors.
    """
    from mmcv.parallel import MMDataParallel

    tap = HeadTap(model.pts_bbox_head)
    wrapped = MMDataParallel(model, device_ids=[int(device.split(':')[-1])])
    try:
        for i, data in enumerate(loader):
            if n is not None and i >= n:
                break
            with torch.no_grad():
                wrapped(return_loss=False, rescale=True, **data)
            if tap.last is None:
                raise RuntimeError('the head was never called with only_bev=False')
            yield i, data, tap.last
    finally:
        tap.close()


# ------------------------------------------------------------------ BEV geometry
def bev_grid_xy(cfg):
    """Metric centre of every BEV cell, in flat index order.

    encoder.py:44-80 builds reference points as meshgrid(H, W) flattened
    row-major, so index = h * W + w; point_sampling (:95-98) maps
    x <- w-fraction over pc_range[0:3] and y <- h-fraction over pc_range[1:4].
    """
    h = cfg.model.pts_bbox_head.bev_h
    w = cfg.model.pts_bbox_head.bev_w
    pc = list(cfg.point_cloud_range) if hasattr(cfg, 'point_cloud_range') \
        else list(cfg.model.pts_bbox_head.transformer.encoder.pc_range)
    x = (np.arange(w) + 0.5) / w * (pc[3] - pc[0]) + pc[0]
    y = (np.arange(h) + 0.5) / h * (pc[4] - pc[1]) + pc[1]
    yy, xx = np.meshgrid(y, x, indexing='ij')        # [H, W]
    return xx.reshape(-1), yy.reshape(-1), h, w, pc


def gt_box_mask(data, xs, ys):
    """Which BEV cells fall inside a ground-truth box footprint. For PPA."""
    def unwrap(v):
        while isinstance(v, (list, tuple)):
            v = v[0]
        return v
    try:
        boxes = unwrap(data['gt_bboxes_3d'][0].data)
    except Exception:
        return None
    bev = boxes.bev.numpy() if hasattr(boxes, 'bev') else None   # [N, 5] cx cy dx dy yaw
    if bev is None or len(bev) == 0:
        return np.zeros(xs.shape, dtype=bool)
    m = np.zeros(xs.shape, dtype=bool)
    for cx, cy, dx, dy, yaw in bev:
        c, s = np.cos(-yaw), np.sin(-yaw)
        px, py = xs - cx, ys - cy
        rx, ry = c * px - s * py, s * px + c * py
        m |= (np.abs(rx) <= dx / 2) & (np.abs(ry) <= dy / 2)
    return m
