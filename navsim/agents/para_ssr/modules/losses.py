"""Losses and Hungarian matching for the PARA-SSR auxiliary heads, mmdet-free.

The nuScenes heads pull ``FocalLoss``, ``L1Loss``, ``PtsL1Loss``,
``PtsDirCosLoss``, ``HungarianAssigner3D`` and ``MapHungarianAssigner3D`` out of
mmdet / the mmdet3d plugin.  Each is reimplemented here against plain torch and
``scipy.optimize.linear_sum_assignment``, keeping the same reductions and the
same cost definitions so the configured loss weights carry over unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# elementwise losses
# --------------------------------------------------------------------------- #
def weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    """L1 with mmdet's ``reduction='mean'`` + ``avg_factor`` semantics."""
    loss = torch.abs(pred - target)
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        return loss.mean()
    return loss.sum() / max(avg_factor, 1e-6)


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    """Focal loss on sigmoid logits.

    ``target`` holds class indices; ``num_classes`` means background.
    """
    num_classes = pred.size(-1)
    target_onehot = F.one_hot(
        target.clamp(max=num_classes), num_classes=num_classes + 1
    )[..., :num_classes].to(pred.dtype)

    pred_sigmoid = pred.sigmoid()
    pt = (1 - pred_sigmoid) * target_onehot + pred_sigmoid * (1 - target_onehot)
    focal_weight = (alpha * target_onehot + (1 - alpha) * (1 - target_onehot)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target_onehot, reduction="none") * focal_weight

    if weight is not None:
        if weight.dim() == 1:
            weight = weight.view(-1, 1)
        loss = loss * weight
    if avg_factor is None:
        return loss.mean()
    return loss.sum() / max(avg_factor, 1e-6)


def pts_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    """L1 over polyline point sets, ``[..., num_pts, 2]``."""
    return weighted_l1_loss(pred, target, weight, avg_factor)


def pts_dir_cos_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    """1 - cosine similarity between consecutive-point direction vectors.

    ``pred`` / ``target``: ``[N, num_pts-1, 2]`` direction vectors.
    """
    loss = 1.0 - torch.cosine_similarity(pred, target, dim=-1)
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        return loss.mean()
    return loss.sum() / max(avg_factor, 1e-6)


# --------------------------------------------------------------------------- #
# matching costs
# --------------------------------------------------------------------------- #
def focal_cost(
    cls_pred: torch.Tensor, gt_labels: torch.Tensor, weight: float = 2.0,
    alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-12,
) -> torch.Tensor:
    """mmdet's ``FocalLossCost``: ``[num_query, num_gt]``."""
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    return (pos_cost[:, gt_labels] - neg_cost[:, gt_labels]) * weight


def bbox_l1_cost(
    bbox_pred: torch.Tensor, gt_bboxes: torch.Tensor, weight: float = 0.25
) -> torch.Tensor:
    return torch.cdist(bbox_pred, gt_bboxes, p=1) * weight


def ordered_pts_l1_cost(
    pts_pred: torch.Tensor, gt_pts: torch.Tensor, weight: float = 1.0
) -> torch.Tensor:
    """VAD's ``OrderedPtsL1Cost`` before minimising over GT orderings.

    The original implementation flattens every point coordinate and applies
    ``torch.cdist(..., p=1)``.  It is therefore a *sum* over all point
    coordinates, not a per-point mean.

    Returns:
        Cost tensor with shape ``[num_query, num_gt, num_orders]``.
    """
    num_query = pts_pred.size(0)
    num_gt, num_orders = gt_pts.shape[:2]
    return torch.cdist(
        pts_pred.reshape(num_query, -1),
        gt_pts.reshape(num_gt * num_orders, -1),
        p=1,
    ).view(num_query, num_gt, num_orders) * weight


# --------------------------------------------------------------------------- #
# assigners
# --------------------------------------------------------------------------- #
def hungarian_assign_det(
    cls_pred: torch.Tensor,
    bbox_pred: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    cls_weight: float = 2.0,
    reg_weight: float = 0.25,
    code_weights: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-to-one matching for the detection head.

    Args:
        cls_pred: ``[num_query, num_classes]`` logits
        bbox_pred: ``[num_query, code_size]`` in normalised box space
        gt_bboxes: ``[num_gt, code_size]`` in the same space
    Returns:
        ``(matched_query_idx, matched_gt_idx)``
    """
    num_gt = gt_bboxes.size(0)
    device = cls_pred.device
    if num_gt == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    cls_cost = focal_cost(cls_pred, gt_labels, weight=cls_weight)
    # HungarianAssigner3D matches on the first 8 code dims (position, size,
    # rotation) and leaves velocity out of the cost.
    n = min(8, bbox_pred.size(-1), gt_bboxes.size(-1))
    reg_cost = bbox_l1_cost(bbox_pred[:, :n], gt_bboxes[:, :n], weight=reg_weight)

    cost = cls_cost + reg_cost
    cost = torch.nan_to_num(cost, nan=1e5, posinf=1e5, neginf=-1e5)
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(row, dtype=torch.long, device=device),
        torch.as_tensor(col, dtype=torch.long, device=device),
    )


def hungarian_assign_map(
    cls_pred: torch.Tensor,
    pts_pred: torch.Tensor,
    gt_pts: torch.Tensor,
    gt_labels: torch.Tensor,
    cls_weight: float = 2.0,
    pts_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-to-one matching for the vector map head, with permutation search.

    VAD matches a predicted polyline against every equivalent ordering of the
    ground-truth polyline (``map_gt_shift_pts_pattern='v2'`` supplies those
    orderings) and keeps the cheapest.  The winning order index is returned so
    the regression loss can use the same one.

    Args:
        pts_pred: ``[num_query, num_pts, 2]`` normalised to ``[0, 1]``
        gt_pts: ``[num_gt, num_orders, num_pts, 2]``
    Returns:
        ``(matched_query_idx, matched_gt_idx, matched_order_idx)``
    """
    num_gt = gt_pts.size(0)
    device = cls_pred.device
    if num_gt == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    cls_cost = focal_cost(cls_pred, gt_labels, weight=cls_weight)

    # [num_query, num_gt, num_orders]
    pts_cost, order_idx = ordered_pts_l1_cost(
        pts_pred, gt_pts, weight=pts_weight
    ).min(dim=-1)

    cost = cls_cost + pts_cost
    cost = torch.nan_to_num(cost, nan=1e5, posinf=1e5, neginf=-1e5)
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    row_t = torch.as_tensor(row, dtype=torch.long, device=device)
    col_t = torch.as_tensor(col, dtype=torch.long, device=device)
    return row_t, col_t, order_idx[row_t, col_t]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar across DDP ranks (identity when not distributed)."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return tensor
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor.div_(torch.distributed.get_world_size()))
    return tensor


def distributed_avg_factor(count: int, reference: torch.Tensor) -> float:
    """Original mmdet DDP positive-count reduction.

    Every rank divides its local loss sum by the rank-mean positive count, and
    DDP subsequently averages gradients.  Together these operations equal a
    global sum divided by the global positive count.
    """
    factor = reduce_mean(reference.new_tensor(float(count)))
    return max(float(factor.clamp(min=1.0).item()), 1.0)


def normalize_bbox(bboxes: torch.Tensor) -> torch.Tensor:
    """Encode physical boxes into the canonical 10-dim decoder code.

    Input physical layout is ``(cx, cy, cz, w, l, h, yaw, vx, vy)`` and the
    decoder code is ``(cx, cy, log(w), log(l), cz, log(h), sin, cos, vx, vy)``.

    Only ground truth boxes pass through this function.  Decoder predictions
    are already in this code space and must never be encoded a second time.
    """
    cx, cy, cz = bboxes[..., 0:1], bboxes[..., 1:2], bboxes[..., 2:3]
    width, length, height = bboxes[..., 3:4], bboxes[..., 4:5], bboxes[..., 5:6]
    rot = bboxes[..., 6:7]
    out = [
        cx,
        cy,
        width.clamp(min=1e-3).log(),
        length.clamp(min=1e-3).log(),
        cz,
        height.clamp(min=1e-3).log(),
        rot.sin(),
        rot.cos(),
    ]
    if bboxes.size(-1) > 7:
        out.append(bboxes[..., 7:8])
        out.append(bboxes[..., 8:9])
    return torch.cat(out, dim=-1)


def denormalize_bbox(normalized_bboxes: torch.Tensor) -> torch.Tensor:
    """Decode canonical code to ``(cx, cy, cz, w, l, h, yaw, vx, vy)``."""
    rot_sine, rot_cosine = normalized_bboxes[..., 6:7], normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    cx, cy = normalized_bboxes[..., 0:1], normalized_bboxes[..., 1:2]
    width = normalized_bboxes[..., 2:3].exp()
    length = normalized_bboxes[..., 3:4].exp()
    cz = normalized_bboxes[..., 4:5]
    height = normalized_bboxes[..., 5:6].exp()
    out = [cx, cy, cz, width, length, height, rot]
    if normalized_bboxes.size(-1) > 8:
        out.append(normalized_bboxes[..., 8:9])
        out.append(normalized_bboxes[..., 9:10])
    return torch.cat(out, dim=-1)
