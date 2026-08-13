"""Occupancy IoU metric for PARA-SSR.

UniAD's ``occ_head_plugin/metrics.py`` depends on ``pytorch_lightning.metrics``
(the pre-1.0 API), which is not installed in the SSR environment.  Binary /
multi-class IoU is a handful of lines, so it is reimplemented here with plain
tensors.  Instance-level VPQ is intentionally not ported: PARA-SSR predicts
scene-level occupancy and has no instance ids.
"""
import torch


class OccIoU(object):
    """Accumulates per-class intersection/union over a dataset.

    Args:
        num_classes (int): number of occupancy channels (each binary).
        crop (tuple[slice] | None): optional ``(row_slice, col_slice)`` applied
            to both prediction and target so a near-range IoU can be reported
            alongside the full-range one.
    """

    def __init__(self, num_classes, crop=None):
        self.num_classes = num_classes
        self.crop = crop
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes, dtype=torch.double)
        self.union = torch.zeros(self.num_classes, dtype=torch.double)

    @torch.no_grad()
    def update(self, pred, target, frame_mask=None):
        """
        Args:
            pred (Tensor): ``[B, T, C, H, W]`` binary predictions (0/1).
            target (Tensor): ``[B, T, C, H, W]`` binary GT.
            frame_mask (Tensor | None): ``[B, T]`` bool, invalid frames skipped.
        """
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        if self.crop is not None:
            rs, cs = self.crop
            pred = pred[..., rs, cs]
            target = target[..., rs, cs]

        if frame_mask is not None:
            fm = frame_mask.detach().cpu().bool().view(
                frame_mask.shape[0], frame_mask.shape[1], 1, 1, 1)
            pred = pred * fm
            target = target * fm

        p = pred.bool()
        t = target.bool()
        # sum over batch, time, H, W -> per class
        dims = (0, 1, 3, 4)
        self.intersection += (p & t).sum(dim=dims).double()
        self.union += (p | t).sum(dim=dims).double()

    def compute(self):
        """Returns per-class IoU as a ``[num_classes]`` tensor."""
        return self.intersection / torch.clamp(self.union, min=1.0)


def near_range_crop(bev_h, bev_w, ratio=0.5):
    """Centre crop covering ``ratio`` of the BEV extent along each axis."""
    dh = int(bev_h * (1 - ratio) / 2)
    dw = int(bev_w * (1 - ratio) / 2)
    return (slice(dh, bev_h - dh), slice(dw, bev_w - dw))
