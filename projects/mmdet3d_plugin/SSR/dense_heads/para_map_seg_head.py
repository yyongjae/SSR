"""Parallel dense semantic-map head for PARA-SSR.

PARA-Drive's mapping module is a pixel-wise semantic BEV map (Panoptic
SegFormer flavoured). This head follows that representation: 3 channels
(divider, ped_crossing, boundary) predicted densely on SSR's 100x100 BEV grid.
It exists alongside the vectorised ``ParaMapHead`` (VAD-style) as a config
switch; the dense variant is the default because

* its per-pixel logits are a clean knowledge-distillation target -- no
  Hungarian matching, so teacher/student outputs align pixel-for-pixel;
* it is ~14x smaller and saves ~3 GB of training activations versus the
  2000-query vector decoder.

GT is rasterised at forward time from the ``LiDARInstanceLines`` the VAD
pipeline already produces for every sample (``map_gt_bboxes_3d``); this cannot
be a pipeline transform because ``vectormap_pipeline`` runs *after* the image
pipeline in ``prepare_train_data``. The cv2 work costs ~1 ms per sample and
overlaps GPU compute.

Note nuScenes' drivable-area polygon (PARA-Drive's 4th channel) is not part of
``VectorizedLocalMap``; adding it would require extending the map API.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS, build_loss

from .para_occ_head import conv_trunk


@HEADS.register_module()
class ParaMapSegHead(BaseModule):
    """BEV feature -> per-class semantic map masks ``[B, C, H, W]``."""

    def __init__(self,
                 bev_h=100,
                 bev_w=100,
                 in_channels=256,
                 mid_channels=128,
                 feat_channels=64,
                 num_classes=3,
                 num_trunk_blocks=2,
                 pc_range=(-15.0, -30.0, -2.0, 15.0, 30.0, 2.0),
                 line_thickness=2,
                 loss_mask=dict(
                     type='OccBinarySegmentationLoss',
                     use_top_k=True,
                     top_k_ratio=0.25,
                     future_discount=1.0,
                     loss_weight=5.0),
                 loss_dice=dict(
                     type='OccDiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=1.0),
                 loss_weight=1.0,
                 test_thresh=0.5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pc_range = list(pc_range)
        self.line_thickness = line_thickness
        self.loss_weight = loss_weight
        self.test_thresh = test_thresh
        self.fp16_enabled = False

        # metres per cell (anisotropic: 0.3 m lateral, 0.6 m longitudinal)
        self.dx = (self.pc_range[3] - self.pc_range[0]) / bev_w
        self.dy = (self.pc_range[4] - self.pc_range[1]) / bev_h

        self.trunk = conv_trunk(in_channels, mid_channels, feat_channels,
                                num_trunk_blocks)
        self.pred_head = nn.Conv2d(feat_channels, num_classes, kernel_size=1)

        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def forward(self, bev_embed):
        bs = bev_embed.size(0)
        x = bev_embed.view(bs, self.bev_h, self.bev_w, self.in_channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.pred_head(self.trunk(x))  # [B, C, H, W]

    @torch.no_grad()
    def rasterize_gt(self, map_gt_bboxes_3d, map_gt_labels_3d, device):
        """LiDARInstanceLines -> ``[B, C, H, W]`` binary masks."""
        bsz = len(map_gt_bboxes_3d)
        seg = np.zeros((bsz, self.num_classes, self.bev_h, self.bev_w),
                       dtype=np.uint8)
        for b in range(bsz):
            lines = map_gt_bboxes_3d[b].fixed_num_sampled_points.cpu().numpy()
            labels = map_gt_labels_3d[b].cpu().numpy()
            for i in range(lines.shape[0]):
                cls = int(labels[i])
                if not 0 <= cls < self.num_classes:
                    continue
                cols = (lines[i, :, 0] - self.pc_range[0]) / self.dx
                rows = (lines[i, :, 1] - self.pc_range[1]) / self.dy
                pts = np.round(np.stack([cols, rows], axis=1)).astype(np.int32)
                cv2.polylines(seg[b, cls], [pts], isClosed=False, color=1,
                              thickness=self.line_thickness)
        return torch.from_numpy(seg).to(device=device, dtype=torch.float32)

    @force_fp32(apply_to=('map_preds', ))
    def loss(self, map_preds, map_gt_seg):
        """``map_preds``/``map_gt_seg``: ``[B, C, H, W]``."""
        b, c, h, w = map_preds.shape
        # the shared losses treat dim 0 as instance and dim 1 as time; a static
        # map is a single "frame" per (sample, class)
        pred = map_preds.reshape(b * c, 1, h, w)
        target = map_gt_seg.reshape(b * c, 1, h, w)
        loss_dict = dict()
        loss_dict['loss_map_mask'] = self.loss_weight * self.loss_mask(
            pred, target)
        loss_dict['loss_map_dice'] = self.loss_weight * self.loss_dice(
            pred, target, avg_factor=b * c)
        return loss_dict

    def forward_train(self, bev_embed, img_metas, map_gt_bboxes_3d,
                      map_gt_labels_3d, **kwargs):
        map_preds = self(bev_embed)
        map_gt_seg = self.rasterize_gt(map_gt_bboxes_3d, map_gt_labels_3d,
                                       map_preds.device)
        return self.loss(map_preds, map_gt_seg)

    def forward_test(self, bev_embed, img_metas=None, rescale=False, **kwargs):
        scores = self(bev_embed).sigmoid()
        return dict(
            map_seg_scores=scores,
            map_seg=(scores > self.test_thresh).long())
