"""Occupancy losses for PARA-SSR.

Ported from UniAD (``projects/mmdet3d_plugin/losses/occflow_loss.py``, itself
derived from FIERY).  The instance/query-matching machinery is not needed here:
PARA-SSR predicts scene-level occupancy straight from the BEV feature, so the
losses are applied to a ``(n, T, H, W)`` tensor where ``n`` is simply
``batch * num_classes``.

``frame_mask`` marks which future frames carry usable GT (frames past the end of
a scene do not), mirroring UniAD's ``gt_occ_img_is_valid`` handling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


def _expand_frame_mask(frame_mask, n_gt, s):
    """Normalise a frame-validity mask to a broadcastable ``[n, T]`` float.

    Accepts ``[T]`` (one mask shared by the whole batch, correct only when
    every sample agrees) or ``[n, T]`` (per row of the instance axis, which is
    what ``batch x class`` occupancy needs once ``samples_per_gpu > 1``).
    """
    mask = frame_mask.float()
    if mask.dim() == 1:
        assert mask.size(0) == s, f'{tuple(frame_mask.shape)} vs T={s}'
        return mask.view(1, s)
    assert tuple(mask.shape) == (n_gt, s), \
        f'{tuple(frame_mask.shape)} vs {(n_gt, s)}'
    return mask


@LOSSES.register_module()
class OccBinarySegmentationLoss(nn.Module):
    """BCE with temporal discounting and optional top-k hard pixel mining."""

    def __init__(self,
                 use_top_k=False,
                 top_k_ratio=1.0,
                 future_discount=1.0,
                 loss_weight=1.0):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount
        self.loss_weight = loss_weight

    def forward(self, prediction, target, frame_mask=None):
        n_gt, s, h, w = prediction.size()
        assert prediction.size() == target.size(), \
            f'{prediction.size()}, {target.size()}'

        loss = F.binary_cross_entropy_with_logits(
            prediction, target.float(), reduction='none')

        if frame_mask is not None:
            frame_mask = _expand_frame_mask(frame_mask, n_gt, s)
            if frame_mask.sum().item() == 0:
                return prediction.sum() * 0.
            loss = loss * frame_mask.unsqueeze(-1).unsqueeze(-1)

        future_discounts = self.future_discount ** torch.arange(
            s, device=loss.device, dtype=loss.dtype)
        loss = loss * future_discounts.view(1, s, 1, 1)

        loss = loss.view(n_gt, s, -1)
        if self.use_top_k:
            # penalise only the hardest pixels -- occupancy is very imbalanced
            k = max(1, int(self.top_k_ratio * loss.shape[2]))
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return self.loss_weight * torch.mean(loss)


def occ_dice_loss(pred,
                  target,
                  weight=None,
                  eps=1e-3,
                  reduction='mean',
                  naive_dice=False,
                  avg_factor=None,
                  frame_mask=None):
    """Dice loss over a ``(n, T, H, W)`` occupancy tensor."""
    n, s, h, w = pred.size()
    assert pred.size() == target.size(), f'{pred.size()}, {target.size()}'

    if frame_mask is not None:
        fm = _expand_frame_mask(frame_mask, n, s).unsqueeze(-1).unsqueeze(-1)
        if fm.sum().item() == 0:
            return pred.sum() * 0.
        target = target * fm
        pred = pred * fm

    inputs = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(inputs * target, 1)
    if naive_dice:
        b = torch.sum(inputs, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(inputs * inputs, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


@LOSSES.register_module()
class OccDiceLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=True,
                 loss_weight=1.0,
                 eps=1e-3):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None,
                frame_mask=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        return self.loss_weight * occ_dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
            frame_mask=frame_mask)
