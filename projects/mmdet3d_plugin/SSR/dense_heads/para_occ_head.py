"""Parallel occupancy head for PARA-SSR.

PARA-Drive (CVPR'24, Fig. 5) attaches occupancy prediction to the shared BEV
feature *in parallel* with mapping, motion and planning -- no module consumes
another module's output. (The paper's own occupancy module is query-based with
Hungarian matching; here it is deliberately query-free, predicting straight
from the BEV feature, per the project requirement and to keep a clean dense
logit interface for knowledge-distillation experiments.)

Two decoder variants share the same losses, GT and output format
``[B, T, C, H, W]``:

* ``temporal_decoder=False`` (default): **single-shot** -- a plain conv trunk
  at full BEV resolution predicts all ``T * C`` channels with one 1x1 conv.
  Cheapest, and the per-pixel logits are a clean KD target.
* ``temporal_decoder=True``: UniAD-style recurrence (downscale to 25x25, roll
  a state forward one block per frame, CVT decoder back up). Heavier; useful
  as a capacity ablation / stronger-teacher variant.

Either way the supervision is A-style: pipeline-rasterised GT
(``GenerateSSROccLabels``: pre-range-filter, worker-parallel), top-k
hard-pixel BCE with future discount, dice, and per-frame validity masking.

Because SSR's BEV grid (100x100 over x in [-15, 15], y in [-30, 30]) is also
the occupancy GT grid, no resampling is required.
"""
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS, build_loss

from .occ_modules import Bottleneck, CVT_Decoder, SimpleConv2d, UpsamplingAdd
from ..utils.occ_metric import OccIoU, near_range_crop


def conv_trunk(in_channels, mid_channels, feat_channels, num_blocks=2):
    """BN-ReLU 3x3 conv stack at constant resolution."""
    layers = [
        nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
    ]
    ch = mid_channels
    for _ in range(num_blocks):
        layers += [
            nn.Conv2d(ch, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        ]
        ch = feat_channels
    return nn.Sequential(*layers)


@HEADS.register_module()
class ParaOccHead(BaseModule):
    """BEV feature -> current + future occupancy, with no query dependency.

    Args:
        bev_h, bev_w (int): BEV grid size; output resolution is identical.
        in_channels (int): BEV embedding dim (256 for SSR).
        mid_channels (int): first trunk width (single-shot decoder).
        feat_channels (int): working width of the decoder.
        n_future (int): number of future steps (0.5 s each).  ``n_future + 1``
            frames are predicted, frame 0 being the present.
        num_classes (int): occupancy channels (2 = vehicle, pedestrian).
        temporal_decoder (bool): False = single-shot (default), True =
            UniAD-style temporal recurrence.
        loss_mask (dict): BCE-style loss config.
        loss_dice (dict): dice loss config.
        loss_weight (float): scales the whole head's contribution.
    """

    def __init__(self,
                 bev_h=100,
                 bev_w=100,
                 in_channels=256,
                 mid_channels=128,
                 feat_channels=64,
                 n_future=6,
                 num_classes=2,
                 temporal_decoder=False,
                 num_trunk_blocks=2,
                 num_bev_proj_conv=3,
                 loss_mask=dict(
                     type='OccBinarySegmentationLoss',
                     use_top_k=True,
                     top_k_ratio=0.25,
                     future_discount=0.95,
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
        self.feat_channels = feat_channels
        self.n_future = n_future
        self.num_frames = n_future + 1
        self.num_classes = num_classes
        self.temporal_decoder = temporal_decoder
        self.loss_weight = loss_weight
        self.test_thresh = test_thresh
        self.fp16_enabled = False

        if not temporal_decoder:
            # single-shot: full-resolution trunk, all frames in one projection
            self.trunk = conv_trunk(in_channels, mid_channels, feat_channels,
                                    num_trunk_blocks)
            self.pred_head = nn.Conv2d(
                feat_channels, self.num_frames * num_classes, kernel_size=1)
        else:
            # UniAD-style temporal recurrence (kept for capacity ablations)
            self.bev_proj = SimpleConv2d(
                in_channels=in_channels,
                conv_channels=in_channels,
                out_channels=feat_channels,
                num_conv=num_bev_proj_conv)
            self.base_downscale = nn.Sequential(
                Bottleneck(in_channels=feat_channels, downsample=True),
                Bottleneck(in_channels=feat_channels, downsample=True))
            self.downscale_convs = nn.ModuleList()
            self.upsample_adds = nn.ModuleList()
            for _ in range(self.num_frames):
                self.downscale_convs.append(
                    Bottleneck(in_channels=feat_channels, downsample=True))
                self.upsample_adds.append(
                    UpsamplingAdd(feat_channels, feat_channels, scale_factor=2))
            self.dense_decoder = CVT_Decoder(
                dim=feat_channels, blocks=[feat_channels, feat_channels])
            self.pred_head = nn.Conv2d(feat_channels, num_classes,
                                       kernel_size=1)

        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def forward(self, bev_embed):
        """
        Args:
            bev_embed (Tensor): ``[B, bev_h * bev_w, C]``.  Rows of the implicit
                grid index y (longitudinal), columns index x (lateral) -- see
                ``BEVFormerEncoder.get_reference_points``.
        Returns:
            Tensor: occupancy logits ``[B, T, num_classes, bev_h, bev_w]``.
        """
        bs = bev_embed.size(0)
        x = bev_embed.view(bs, self.bev_h, self.bev_w, self.in_channels)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        if not self.temporal_decoder:
            logits = self.pred_head(self.trunk(x))
            return logits.view(bs, self.num_frames, self.num_classes,
                               self.bev_h, self.bev_w)

        x = self.bev_proj(x)
        state = self.base_downscale(x)
        states = []
        for t in range(self.num_frames):
            down = self.downscale_convs[t](state)
            state = self.upsample_adds[t](down, state)
            states.append(state)
        x = torch.stack(states, dim=1)  # [B, T, C, H/4, W/4]
        x = self.dense_decoder(x)       # [B, T, C, H, W]
        b, t, c, h, w = x.shape
        logits = self.pred_head(x.reshape(b * t, c, h, w))
        return logits.reshape(b, t, self.num_classes, h, w)

    @force_fp32(apply_to=('occ_preds', ))
    def loss(self, occ_preds, gt_occ_seg, gt_occ_valid=None):
        """
        Args:
            occ_preds (Tensor): ``[B, T, C, H, W]`` logits.
            gt_occ_seg (Tensor): ``[B, T, C, H, W]`` binary GT.
            gt_occ_valid (Tensor | None): ``[B, T]`` per-frame validity.
        """
        gt_occ_seg = gt_occ_seg[:, :self.num_frames].to(occ_preds.dtype)
        assert gt_occ_seg.shape == occ_preds.shape, \
            f'{gt_occ_seg.shape} vs {occ_preds.shape}'

        # (B, T, C, H, W) -> (B * C, T, H, W): the losses treat dim 0 as the
        # "instance" axis, which for scene-level occupancy is batch x class.
        b, t, c, h, w = occ_preds.shape
        pred = occ_preds.permute(0, 2, 1, 3, 4).reshape(b * c, t, h, w)
        target = gt_occ_seg.permute(0, 2, 1, 3, 4).reshape(b * c, t, h, w)

        frame_mask = None
        if gt_occ_valid is not None:
            # One mask per row of the instance axis. dim 0 above is batch-major
            # (sample 0's classes, then sample 1's, ...), so repeating each
            # sample's [T] mask ``c`` times lines the two up. Sharing a single
            # [T] mask across the batch would supervise frames that ran off the
            # end of one sample's scene as "empty" whenever another sample in
            # the same batch still had valid GT there.
            frame_mask = gt_occ_valid[:, :self.num_frames].bool()
            frame_mask = frame_mask.repeat_interleave(c, dim=0)

        loss_dict = dict()
        loss_dict['loss_occ_mask'] = self.loss_weight * self.loss_mask(
            pred, target, frame_mask=frame_mask)
        loss_dict['loss_occ_dice'] = self.loss_weight * self.loss_dice(
            pred, target, avg_factor=b * c, frame_mask=frame_mask)
        return loss_dict

    def forward_train(self, bev_embed, gt_occ_seg, gt_occ_valid=None,
                      metrics_out=None, **kwargs):
        occ_preds = self(bev_embed)
        if metrics_out is not None:
            metrics_out.update(
                self.train_metrics(occ_preds, gt_occ_seg, gt_occ_valid))
        return self.loss(occ_preds, gt_occ_seg, gt_occ_valid)

    @torch.no_grad()
    def train_metrics(self, occ_preds, gt_occ_seg, gt_occ_valid=None,
                      thresholds=(0.1, 0.5)):
        """IoU on the current training batch -- cheap and diagnostic.

        The BCE/Dice pair can fall for 12 epochs while the head converges to a
        spatially constant output (that is what happened to the dense map head:
        loss -18% but a 1.06x positive/negative separation). IoU cannot be
        faked that way, so it is the number to watch to catch a dead head early
        rather than after a full training run. Reported at 0.1 as well because
        the usable operating point depends on class prevalence -- UniAD
        evaluates occupancy at 0.1, not 0.5 (report #01 6.1).
        """
        probs = occ_preds.detach().sigmoid()
        gt = gt_occ_seg[:, :self.num_frames].to(probs.device).bool()
        # Restrict to the frames the loss actually supervised. Without this the
        # metric scores frames past the end of a scene as "everything empty",
        # which both flatters a dead head and disagrees with the loss.
        #
        # `valid` has to gate the NEGATIVE set explicitly. Zeroing the
        # probabilities of unsupervised cells (the first version of this) keeps
        # them out of the numerator but leaves them in the `~gt` denominator, so
        # p_off is diluted towards zero and the separation ratio is inflated --
        # measured 5.33 against a true 2.67 on a synthetic constant head.
        if gt_occ_valid is not None:
            v = gt_occ_valid[:, :self.num_frames].to(probs.device).bool()
            valid = v[:, :, None, None, None].expand_as(gt)
        else:
            valid = torch.ones_like(gt)
        gt = gt & valid
        out = {}
        for thr in thresholds:
            pb = (probs > thr) & valid
            inter = (pb & gt).sum(dim=(0, 1, 3, 4)).float()
            union = (pb | gt).sum(dim=(0, 1, 3, 4)).float().clamp(min=1)
            iou = inter / union
            for c in range(iou.numel()):
                out[f'occ_iou{thr}/cls{c}'] = iou[c]
            out[f'occ_iou{thr}/mean'] = iou.mean()
        # separation of predicted probability on positive vs negative cells:
        # a constant-output head sits at 1.0 no matter what the loss does
        pos, neg = gt, (~gt) & valid
        p_on = (probs * pos).sum() / pos.sum().clamp(min=1)
        p_off = (probs * neg).sum() / neg.sum().clamp(min=1)
        out['occ_sep/p_on'] = p_on
        out['occ_sep/p_off'] = p_off
        out['occ_sep/ratio'] = p_on / p_off.clamp(min=1e-6)
        # Fraction of the batch the metrics above are computed on. If every
        # frame were invalid the keys would all read 0; the key set stays fixed
        # either way because _parse_losses all-reduces log_vars across ranks and
        # would hang on a rank-dependent key set.
        out['occ_sep/valid_frac'] = valid.float().mean()
        return out

    def forward_test(self, bev_embed, **kwargs):
        occ_preds = self(bev_embed)
        scores = occ_preds.sigmoid()
        return dict(
            occ_scores=scores,
            occ_seg=(scores > self.test_thresh).long())

    def build_metrics(self, near_ratio=0.5):
        """Convenience factory for evaluation: full-range and near-range IoU."""
        return dict(
            occ_iou_full=OccIoU(self.num_classes),
            occ_iou_near=OccIoU(
                self.num_classes,
                crop=near_range_crop(self.bev_h, self.bev_w, near_ratio)))
