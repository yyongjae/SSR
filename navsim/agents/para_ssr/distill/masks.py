"""Planning-aware spatial weights for Stage-2 BEV distillation.

BEV geometry is SSR: ``x`` right, ``y`` forward, matching ``pc_range``.
NAVSIM trajectories are ``(x_forward, y_left, heading)`` and must be converted
before they are splatted onto that grid.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def bev_cell_centers(
    pc_range: Sequence[float],
    bev_h: int,
    bev_w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grid_y, grid_x)`` of shape ``[H, W]`` in metres."""
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    ys = torch.linspace(
        y_min + (y_max - y_min) / (2 * bev_h),
        y_max - (y_max - y_min) / (2 * bev_h),
        bev_h,
        device=device,
        dtype=dtype,
    )
    xs = torch.linspace(
        x_min + (x_max - x_min) / (2 * bev_w),
        x_max - (x_max - x_min) / (2 * bev_w),
        bev_w,
        device=device,
        dtype=dtype,
    )
    return torch.meshgrid(ys, xs, indexing="ij")


def navsim_trajectory_to_ssr(trajectories: torch.Tensor) -> torch.Tensor:
    """Convert NAVSIM ``(x_forward, y_left, heading)`` to SSR ``(x_right, y_forward)``.

    Heading is rotated by ``+pi/2`` so that NAVSIM 0 (forward) becomes SSR
    ``+y``.  Extra trailing channels are dropped.
    """
    if trajectories.dim() != 3 or trajectories.size(-1) < 2:
        raise ValueError(
            f"trajectories must be [B, T, >=2], got {tuple(trajectories.shape)}"
        )
    x_fwd = trajectories[..., 0]
    y_left = trajectories[..., 1]
    xy = torch.stack((-y_left, x_fwd), dim=-1)
    if trajectories.size(-1) < 3:
        return xy
    heading = torch.atan2(
        torch.sin(trajectories[..., 2] + 0.5 * torch.pi),
        torch.cos(trajectories[..., 2] + 0.5 * torch.pi),
    )
    return torch.cat((xy, heading.unsqueeze(-1)), dim=-1)


def _heading_from_waypoints(xy: torch.Tensor) -> torch.Tensor:
    """``[B, T, 2] -> [B, T]`` SSR heading, estimated from consecutive points."""
    delta = xy[:, 1:] - xy[:, :-1]
    step_heading = torch.atan2(delta[..., 1], delta[..., 0])
    first = step_heading[:, :1]
    return torch.cat((first, step_heading), dim=1)


def compute_corridor_mask(
    trajectories: torch.Tensor,
    pc_range: Tuple[float, ...],
    bev_h: int,
    bev_w: int,
    sigma_base: float = 2.5,
    sigma_growth: float = 0.1,
    base_weight: float = 0.1,
    sigma_along: Optional[float] = None,
    sigma_cross: Optional[float] = None,
    sigma_along_growth: float = 0.15,
    sigma_cross_growth: float = 0.05,
) -> torch.Tensor:
    """Gaussian driving corridor along future waypoints.

    ``trajectories`` must already be in SSR metres.  When ``sigma_along`` and
    ``sigma_cross`` are both set, the Gaussian is heading-aligned (long along
    track, tight across track).  Otherwise the original isotropic
    ``sigma_base + sigma_growth * t`` recipe is used.
    """
    B, T, _ = trajectories.shape
    device = trajectories.device
    dtype = trajectories.dtype
    grid_y, grid_x = bev_cell_centers(
        pc_range, bev_h, bev_w, device=device, dtype=dtype
    )
    grid_pts = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
    pts = trajectories[:, :, :2].unsqueeze(2).unsqueeze(2)
    delta = grid_pts - pts
    ts = torch.arange(T, device=device, dtype=dtype)

    anisotropic = sigma_along is not None and sigma_cross is not None
    if anisotropic:
        if trajectories.size(-1) >= 3:
            heading = trajectories[:, :, 2]
        else:
            heading = _heading_from_waypoints(trajectories[:, :, :2])
        cos_h = torch.cos(heading).view(B, T, 1, 1)
        sin_h = torch.sin(heading).view(B, T, 1, 1)
        along = delta[..., 0] * cos_h + delta[..., 1] * sin_h
        cross = -delta[..., 0] * sin_h + delta[..., 1] * cos_h
        sig_along = (float(sigma_along) + float(sigma_along_growth) * ts).view(1, T, 1, 1)
        sig_cross = (float(sigma_cross) + float(sigma_cross_growth) * ts).view(1, T, 1, 1)
        w = torch.exp(
            -0.5 * ((along / sig_along) ** 2 + (cross / sig_cross) ** 2)
        )
    else:
        d2 = (delta ** 2).sum(dim=-1)
        sigmas = (sigma_base + sigma_growth * ts).view(1, T, 1, 1)
        w = torch.exp(-0.5 * d2 / (sigmas ** 2))

    corridor_max, _ = w.max(dim=1, keepdim=True)
    return base_weight + (1.0 - base_weight) * corridor_max


def rasterize_agent_mask(
    gt_boxes: torch.Tensor,
    gt_valid: torch.Tensor,
    pc_range: Tuple[float, ...],
    bev_h: int,
    bev_w: int,
    inflate: float = 1.5,
    gt_fut_trajs: Optional[torch.Tensor] = None,
    gt_fut_masks: Optional[torch.Tensor] = None,
    future_sigma: float = 1.5,
) -> torch.Tensor:
    """Soft occupancy of current (and future) agents on the SSR BEV grid.

    ``gt_boxes`` is PARA-SSR code ``(x_right, y_forward, z, width, length, ...)``.
    """
    B = gt_boxes.size(0)
    device = gt_boxes.device
    dtype = gt_boxes.dtype
    grid_y, grid_x = bev_cell_centers(
        pc_range, bev_h, bev_w, device=device, dtype=dtype
    )
    mask = torch.zeros(B, 1, bev_h, bev_w, device=device, dtype=dtype)
    valid = gt_valid.bool()
    inflate = float(inflate)

    for b in range(B):
        boxes = gt_boxes[b][valid[b]]
        if boxes.numel() == 0:
            continue
        cx = boxes[:, 0][:, None, None]
        cy = boxes[:, 1][:, None, None]
        half_w = (boxes[:, 3] * 0.5 * inflate)[:, None, None]
        half_l = (boxes[:, 4] * 0.5 * inflate)[:, None, None]
        yaw = boxes[:, 6][:, None, None]
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        dx = grid_x.unsqueeze(0) - cx
        dy = grid_y.unsqueeze(0) - cy
        local_x = cos_y * dx + sin_y * dy
        local_y = -sin_y * dx + cos_y * dy
        inside = (local_x.abs() <= half_l) & (local_y.abs() <= half_w)
        current = inside.any(dim=0).to(dtype)
        mask[b, 0] = current

        if gt_fut_trajs is None or gt_fut_masks is None:
            continue
        fut = gt_fut_trajs[b][valid[b]]
        fut_m = gt_fut_masks[b][valid[b]]
        if fut.numel() == 0:
            continue
        origin = boxes[:, :2]
        fut_xy = origin[:, None, :] + fut.cumsum(dim=1)
        pts = fut_xy[fut_m > 0.5]
        if pts.numel() == 0:
            continue
        d2 = (grid_x.unsqueeze(0) - pts[:, 0, None, None]) ** 2 + (
            grid_y.unsqueeze(0) - pts[:, 1, None, None]
        ) ** 2
        splat = torch.exp(-0.5 * d2 / (future_sigma ** 2)).max(dim=0).values
        mask[b, 0] = torch.maximum(mask[b, 0], splat)

    return mask.clamp(0.0, 1.0)


def rasterize_map_class_mask(
    gt_map_pts: torch.Tensor,
    gt_map_labels: torch.Tensor,
    gt_map_valid: torch.Tensor,
    class_id: int,
    pc_range: Tuple[float, ...],
    bev_h: int,
    bev_w: int,
    sigma: float,
) -> torch.Tensor:
    """Splat normalised map polylines of one class onto the BEV grid."""
    B, max_vec, _orders, num_pts, _ = gt_map_pts.shape
    device = gt_map_pts.device
    dtype = gt_map_pts.dtype
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    grid_y, grid_x = bev_cell_centers(
        pc_range, bev_h, bev_w, device=device, dtype=dtype
    )
    mask = torch.zeros(B, 1, bev_h, bev_w, device=device, dtype=dtype)
    valid = gt_map_valid.bool() & (gt_map_labels == class_id)
    # order-0 is the canonical polyline; coordinates are in [0, 1] over pc_range
    pts = gt_map_pts[:, :, 0]
    metres = torch.stack(
        (
            pts[..., 0] * (x_max - x_min) + x_min,
            pts[..., 1] * (y_max - y_min) + y_min,
        ),
        dim=-1,
    )
    sigma = max(float(sigma), 1e-3)

    for b in range(B):
        keep = valid[b]
        if not bool(keep.any()):
            continue
        xy = metres[b, keep].reshape(-1, 2)
        d2 = (grid_x.unsqueeze(0) - xy[:, 0, None, None]) ** 2 + (
            grid_y.unsqueeze(0) - xy[:, 1, None, None]
        ) ** 2
        splat = torch.exp(-0.5 * d2 / (sigma ** 2)).max(dim=0).values
        mask[b, 0] = splat
    return mask.clamp(0.0, 1.0)


def morphological_boundary(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Morphological gradient (dilate − erode) of a ``[B, 1, H, W]`` splat.

    Peaks on the contour of a thick road splat so DAC-critical kerb cells stay
    in the ReSMap distill mask even when the interior is already high.
    """
    if mask.dim() != 4 or mask.size(1) != 1:
        raise ValueError(f"expected [B,1,H,W], got {tuple(mask.shape)}")
    k = int(kernel)
    if k < 1:
        raise ValueError(f"boundary kernel must be >= 1, got {kernel}")
    if k % 2 == 0:
        k += 1
    pad = k // 2
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(
        (1.0 - mask).clamp(0.0, 1.0), kernel_size=k, stride=1, padding=pad
    )
    return (dilated - eroded).clamp(0.0, 1.0)


def combine_planning_prior(
    corridor: torch.Tensor,
    *,
    agent: Optional[torch.Tensor] = None,
    road: Optional[torch.Tensor] = None,
    centerline: Optional[torch.Tensor] = None,
    crosswalk: Optional[torch.Tensor] = None,
    boundary: Optional[torch.Tensor] = None,
    base_weight: float = 0.1,
) -> torch.Tensor:
    """Union of corridor + aux foreground for planner-BEV look distillation.

    Walkway is intentionally omitted: sidewalk interior is not a cell the
    planner should light up, and suppressing it punched road edges in v2.
    """
    base = float(base_weight)

    def _lift(focus: torch.Tensor) -> torch.Tensor:
        return base + (1.0 - base) * focus.clamp(0.0, 1.0)

    weight = corridor
    for extra in (agent, road, centerline, crosswalk, boundary):
        if extra is not None:
            weight = torch.maximum(weight, _lift(extra))
    return weight.clamp(min=base, max=1.0)


def combine_role_mask(
    branch: str,
    corridor: torch.Tensor,
    *,
    agent: Optional[torch.Tensor] = None,
    road: Optional[torch.Tensor] = None,
    centerline: Optional[torch.Tensor] = None,
    crosswalk: Optional[torch.Tensor] = None,
    walkway: Optional[torch.Tensor] = None,
    boundary: Optional[torch.Tensor] = None,
    base_weight: float = 0.1,
    walkway_suppress: float = 0.0,
) -> torch.Tensor:
    """Teacher-specific spatial prior on top of the driving corridor.

    * ``bevfusion``: agents (and their futures) plus the corridor.
    * ``resmap``: road / centerline / crosswalk / boundary ring plus corridor.
      Walkway suppression is off by default; a positive factor punches kerb
      cells that overlap the sidewalk splat and hurt DAC.
    * any other name: corridor only.
    """
    base = float(base_weight)

    def _lift(focus: torch.Tensor) -> torch.Tensor:
        return base + (1.0 - base) * focus.clamp(0.0, 1.0)

    # ``corridor`` already includes ``base_weight``; extras are 0-1 splats.
    weight = corridor
    if branch == "bevfusion" and agent is not None:
        weight = torch.maximum(weight, _lift(agent))
    elif branch == "resmap":
        for extra in (road, centerline, crosswalk, boundary):
            if extra is not None:
                weight = torch.maximum(weight, _lift(extra))
        suppress = float(walkway_suppress)
        if walkway is not None and suppress > 0.0:
            weight = weight * (1.0 - suppress * walkway.clamp(0.0, 1.0))
    return weight.clamp(min=base, max=1.0)


def tokens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """``[B, 1, H, W] -> [B, H*W, 1]`` aligned with BEV tokens."""
    return mask.flatten(2).transpose(1, 2)
