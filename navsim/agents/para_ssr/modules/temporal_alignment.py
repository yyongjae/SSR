"""Parameter-free SE(2) alignment of previous BEV features to the current ego.

Like SafeDrive's ``shift_feature``, this uses a backward sampling grid and
bilinear ``grid_sample`` to warp the complete history feature map before
temporal fusion. Our BEV reference points describe cell centers, so the grid
uses ``align_corners=False`` rather than SafeDrive's endpoint convention.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def warp_previous_bev(
    prev_bev: torch.Tensor,
    bev_shift: torch.Tensor,
    bev_yaw: torch.Tensor,
    pc_range: Sequence[float],
) -> torch.Tensor:
    """Resample ``[B, C, H, W]`` previous BEV into the current ego frame.

    ``bev_shift[B, 2]`` is ego displacement expressed in the *current* SSR
    frame (x right, y forward), normalized by the metric ROI spans.
    ``bev_yaw[B]`` is current heading minus previous heading, in radians.
    Consequently each current metric cell center samples the previous map at

    ``p_previous = R(bev_yaw) @ (p_current + translation_current)``.

    Rotation is about the ego origin in metric coordinates, independent of
    ROI shape, offset, and cell size. Metadata is detached; feature gradients
    are retained. Geometry and sampling run in FP32 even under autocast (or
    FP64 for double inputs), and the output preserves the feature dtype.
    Coordinates outside the previous ROI use zero padding, without clamping.
    """
    if prev_bev.ndim != 4 or not prev_bev.is_floating_point():
        raise ValueError("prev_bev must be a floating [B, C, H, W] tensor")
    batch_size, _, height, width = prev_bev.shape
    if height < 1 or width < 1:
        raise ValueError("prev_bev must have nonempty spatial dimensions")
    if tuple(bev_shift.shape) != (batch_size, 2):
        raise ValueError("bev_shift must have shape [B, 2]")
    if tuple(bev_yaw.shape) != (batch_size,):
        raise ValueError("bev_yaw must have shape [B] in radians")
    if len(pc_range) != 6:
        raise ValueError("pc_range must contain six metric bounds")
    x_min, y_min = float(pc_range[0]), float(pc_range[1])
    x_span = float(pc_range[3]) - x_min
    y_span = float(pc_range[4]) - y_min
    if x_span <= 0 or y_span <= 0:
        raise ValueError("pc_range must have positive x and y spans")

    geometry_dtype = torch.float64 if prev_bev.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=prev_bev.device.type, enabled=False):
        shifts = bev_shift.detach().to(device=prev_bev.device, dtype=geometry_dtype)
        yaw = bev_yaw.detach().to(device=prev_bev.device, dtype=geometry_dtype)
        x = (torch.arange(width, device=prev_bev.device, dtype=geometry_dtype) + 0.5)
        y = (torch.arange(height, device=prev_bev.device, dtype=geometry_dtype) + 0.5)
        y, x = torch.meshgrid(y * (y_span / height) + y_min,
                              x * (x_span / width) + x_min, indexing="ij")
        x = x[None] + shifts[:, 0, None, None] * x_span
        y = y[None] + shifts[:, 1, None, None] * y_span
        cosine = yaw.cos()[:, None, None]
        sine = yaw.sin()[:, None, None]
        x_previous = cosine * x - sine * y
        y_previous = sine * x + cosine * y
        # align_corners=False maps ROI boundaries to -1/+1; therefore cell
        # centers map exactly to pixel centers, including rectangular grids.
        grid = torch.stack((2 * (x_previous - x_min) / x_span - 1,
                            2 * (y_previous - y_min) / y_span - 1), dim=-1)
        aligned = F.grid_sample(
            prev_bev.to(geometry_dtype), grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
    return aligned.to(prev_bev.dtype)
