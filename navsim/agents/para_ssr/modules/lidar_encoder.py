"""LiDAR -> BEV feature for PARA-SSR, without spconv.

SafeDrive feeds its BEVFormer a LiDAR BEV twice: as the *initial BEV queries*
in place of a learned embedding, and as the value of a deformable
``lidar_cross_attn`` inside every encoder layer.  This module produces that
LiDAR BEV; the two injection points live in ``bevformer.py``.

Two encoders share one contract -- ``forward(points [B, N, 5], num_points [B])
-> [B, embed_dims, bev_h, bev_w]`` -- and the rest of the model never sees
which one it got:

* ``LidarSparseEncoder`` (default, ``lidar_encoder="sparse"``): SafeDrive's
  path verbatim -- mean-pooled voxels at ``lidar_voxel_size`` and its
  ``SpMiddleResNetFHD`` (SECOND-style sparse 3D residual backbone, spconv 2.x,
  stride 8 in x/y, z collapsed into channels).  Needs ``spconv``; the prebuilt
  ``spconv-cu126`` wheel runs on this rig (RTX 5090, torch 2.8+cu128).
* ``LidarPillarEncoder`` (``lidar_encoder="pillar"``): an spconv-free
  fallback -- dynamic pillars (PointPillars / MVF) max-pooled into a dense
  canvas at ``lidar_pillar_size``, SECOND's 2D backbone and FPN neck.

Grid alignment
--------------
Both encoders cover ``pc_range`` exactly, in SSR axes (x right, y forward).
The sparse encoder's voxel grid is ``8 * bev_h`` rows over y by ``8 * bev_w``
columns over x (its stride is fixed at 8, as in SafeDrive); the pillar canvas
is ``bev_h * ratio`` by ``bev_w * ratio`` with ``ratio = cell / pillar`` a
power of two.  Either way the output ``[B, C, bev_h, bev_w]`` flattens
row-major into exactly ``bev_embed``'s query order (``h * bev_w + w``), and
each output cell is the physical BEV cell the camera pillars of that query
project from.
"""
from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LIDAR_ENCODERS: Tuple[str, ...] = ("sparse", "pillar")

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _batchnorm_eval(module: nn.Module) -> Iterator[None]:
    """Run BatchNorm layers on their running statistics for one call."""
    norms = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
    for norm in norms:
        norm.eval()
    try:
        yield
    finally:
        for norm in norms:
            norm.train()


# One stride-2 stage per entry; the FPN fuses them back at the BEV resolution.
LIDAR_BACKBONE_STRIDES: Tuple[int, ...] = (2, 2, 2)


def pillar_downsample_ratio(
    pc_range: Sequence[float],
    bev_h: int,
    bev_w: int,
    pillar_size: Sequence[float],
) -> int:
    """``BEV cell / pillar`` as one integer, or ``ValueError``.

    The 2D backbone only knows power-of-two strides, and it applies the same
    stride to both axes, so the pillar grid must divide the BEV grid by the
    same power of two longitudinally and laterally.  With square BEV cells
    that just means one square pillar size.
    """
    if len(pillar_size) != 2:
        raise ValueError(
            f"lidar_pillar_size must be (longitudinal, lateral) metres, got {tuple(pillar_size)}"
        )
    cells = (
        ("longitudinal", (pc_range[4] - pc_range[1]) / bev_h, float(pillar_size[0])),
        ("lateral", (pc_range[3] - pc_range[0]) / bev_w, float(pillar_size[1])),
    )
    ratios = []
    for axis, cell, pillar in cells:
        if not math.isfinite(pillar) or pillar <= 0.0:
            raise ValueError(f"{axis} lidar pillar size must be positive, got {pillar}")
        ratio = cell / pillar
        if not math.isclose(ratio, round(ratio), abs_tol=1e-6) or round(ratio) < 1:
            raise ValueError(
                f"{axis} BEV cell {cell:.4f} m is not an integer multiple of the "
                f"{pillar} m pillar"
            )
        ratios.append(int(round(ratio)))
    if ratios[0] != ratios[1]:
        raise ValueError(
            "lidar pillars must divide the BEV cell by the same factor on both axes, "
            f"got longitudinal {ratios[0]} and lateral {ratios[1]}"
        )
    ratio = ratios[0]
    max_ratio = 2 ** len(LIDAR_BACKBONE_STRIDES)
    if ratio & (ratio - 1) or ratio > max_ratio:
        raise ValueError(
            f"BEV cell / pillar must be a power of two up to {max_ratio}, got {ratio}"
        )
    return ratio


class DynamicPillarFeatureNet(nn.Module):
    """Per-point ``Linear -> BN -> ReLU`` max-pooled into pillars (no point cap)."""

    def __init__(self, in_dims: int, channels: int):
        super().__init__()
        self.linear = nn.Linear(in_dims, channels, bias=False)
        self.norm = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.channels = channels

    def forward(
        self, point_feats: torch.Tensor, pillar_index: torch.Tensor, num_pillars: int
    ) -> torch.Tensor:
        """``[M, in_dims]`` + ``[M]`` pillar ids -> ``[num_pillars, C]``."""
        x = self.linear(point_feats)
        if self.training and x.shape[0] < 2:
            # BatchNorm cannot estimate batch statistics from < 2 points; fall
            # back to the running statistics without updating them.
            x = F.batch_norm(
                x, self.norm.running_mean, self.norm.running_var,
                self.norm.weight, self.norm.bias, False, 0.0, self.norm.eps,
            )
        else:
            x = self.norm(x)
        x = F.relu(x)
        pooled = x.new_zeros(num_pillars, self.channels)
        # Empty pillars keep their zeros; occupied ones take the per-channel max.
        return pooled.scatter_reduce_(
            0, pillar_index[:, None].expand_as(x), x, reduce="amax", include_self=False
        )


def _conv_block(in_channels: int, out_channels: int, num_layers: int, stride: int) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    ]
    for _ in range(num_layers):
        layers += [
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class SECONDBackbone2D(nn.Module):
    """SECOND / PointPillars dense backbone: one stride-2 stage per entry."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        layer_nums: Sequence[int],
        strides: Sequence[int] = LIDAR_BACKBONE_STRIDES,
    ):
        super().__init__()
        if not len(out_channels) == len(layer_nums) == len(strides):
            raise ValueError("lidar backbone channels, layers and strides must align")
        blocks = []
        channels = in_channels
        for out, num, stride in zip(out_channels, layer_nums, strides):
            blocks.append(_conv_block(channels, out, num, stride))
            channels = out
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs


class SECONDFPN(nn.Module):
    """SECOND's neck, as in SafeDrive: per-stage resample, concat, final conv.

    ``upsample_strides`` > 1 are transposed convolutions, == 1 a 1x1 transposed
    convolution, < 1 a strided convolution -- the same three cases SafeDrive's
    ``SECONDFPN`` distinguishes.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        upsample_strides: Sequence[float],
        final_channels: int,
    ):
        super().__init__()
        if len(in_channels) != len(upsample_strides):
            raise ValueError("lidar FPN needs one upsample stride per input stage")
        deblocks = []
        for channels, stride in zip(in_channels, upsample_strides):
            if stride >= 1.0:
                k = int(round(stride))
                resample: nn.Module = nn.ConvTranspose2d(channels, out_channels, k, stride=k, bias=False)
            else:
                k = int(round(1.0 / stride))
                resample = nn.Conv2d(channels, out_channels, k, stride=k, bias=False)
            deblocks.append(
                nn.Sequential(
                    resample,
                    nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                )
            )
        self.deblocks = nn.ModuleList(deblocks)
        fused = out_channels * len(in_channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(fused, fused // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fused // 2, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused // 2, final_channels, 1),
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        ups = [deblock(feat) for deblock, feat in zip(self.deblocks, feats)]
        return self.final_conv(torch.cat(ups, dim=1))


class LidarPillarEncoder(nn.Module):
    """Padded point clouds in SSR axes -> ``[B, embed_dims, bev_h, bev_w]``.

    Inputs come straight from ``ParaSSRFeatureBuilder``: ``points`` is
    ``[B, N, 5]`` = (x_right, y_forward, z, intensity, ring) zero-padded to a
    fixed ``N``, and ``num_points`` ``[B]`` says how many rows are real.  The
    builder has already clipped the cloud to ``pc_range`` x ``lidar_z_range``.
    """

    # (x, y, z, intensity, ring) + offset from the pillar's point mean (xyz)
    # + offset from the pillar centre (xy): PointPillars' decoration.
    POINT_DIMS = 5
    DECORATED_DIMS = POINT_DIMS + 3 + 2

    def __init__(
        self,
        pc_range: Sequence[float],
        bev_h: int,
        bev_w: int,
        embed_dims: int,
        pillar_size: Sequence[float] = (0.16, 0.16),
        pillar_channels: int = 64,
        backbone_channels: Sequence[int] = (64, 128, 256),
        backbone_layers: Sequence[int] = (3, 5, 5),
        neck_channels: int = 128,
    ):
        super().__init__()
        ratio = pillar_downsample_ratio(pc_range, bev_h, bev_w, pillar_size)
        self.pc_range = [float(v) for v in pc_range]
        self.bev_h, self.bev_w = int(bev_h), int(bev_w)
        self.pillar_size = (float(pillar_size[0]), float(pillar_size[1]))
        self.canvas_shape = (self.bev_h * ratio, self.bev_w * ratio)
        self.ratio = ratio

        self.pfn = DynamicPillarFeatureNet(self.DECORATED_DIMS, pillar_channels)
        self.backbone = SECONDBackbone2D(pillar_channels, backbone_channels, backbone_layers)
        # Stage i sits at canvas stride 2^(i+1); bring every stage to the BEV grid.
        stage_strides = [
            math.prod(LIDAR_BACKBONE_STRIDES[: i + 1]) for i in range(len(LIDAR_BACKBONE_STRIDES))
        ]
        self.neck = SECONDFPN(
            backbone_channels,
            neck_channels,
            [stride / ratio for stride in stage_strides],
            embed_dims,
        )

    # ------------------------------------------------------------------ #
    def pillarize(
        self, points: torch.Tensor, num_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decorated per-point features, flat pillar ids and the batch size."""
        if points.dim() != 3 or points.shape[-1] != self.POINT_DIMS:
            raise ValueError(
                f"lidar points must be [B, N, {self.POINT_DIMS}], got {tuple(points.shape)}"
            )
        batch, count, _ = points.shape
        rows, cols = self.canvas_shape
        valid = torch.arange(count, device=points.device)[None, :] < num_points[:, None]
        pts = points[valid]  # [M, 5]
        batch_index = torch.arange(batch, device=points.device)[:, None].expand(batch, count)[valid]

        x0, y0 = self.pc_range[0], self.pc_range[1]
        py, px = self.pillar_size
        col = ((pts[:, 0] - x0) / px).floor().long().clamp_(0, cols - 1)
        row = ((pts[:, 1] - y0) / py).floor().long().clamp_(0, rows - 1)
        pillar = (batch_index * rows + row) * cols + col

        num_pillars = batch * rows * cols
        xyz = pts[:, :3]
        sums = xyz.new_zeros(num_pillars, 3).index_add_(0, pillar, xyz)
        counts = xyz.new_zeros(num_pillars).index_add_(0, pillar, torch.ones_like(pillar, dtype=xyz.dtype))
        mean = sums / counts.clamp_min(1.0)[:, None]
        centre_x = x0 + (col.to(pts.dtype) + 0.5) * px
        centre_y = y0 + (row.to(pts.dtype) + 0.5) * py
        decorated = torch.cat(
            (
                pts,
                xyz - mean[pillar],
                (pts[:, 0] - centre_x)[:, None],
                (pts[:, 1] - centre_y)[:, None],
            ),
            dim=-1,
        )
        return decorated, pillar, num_pillars

    def canvas(self, points: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """Dense pillar canvas ``[B, pillar_channels, rows, cols]``."""
        decorated, pillar, num_pillars = self.pillarize(points, num_points)
        batch = points.shape[0]
        rows, cols = self.canvas_shape
        pooled = self.pfn(decorated, pillar, num_pillars)
        return pooled.view(batch, rows, cols, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, points: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        bev = self.neck(self.backbone(self.canvas(points, num_points)))
        if bev.shape[-2:] != (self.bev_h, self.bev_w):
            raise RuntimeError(
                f"lidar BEV {tuple(bev.shape[-2:])} does not match the BEV grid "
                f"{(self.bev_h, self.bev_w)}"
            )
        return bev


# =========================================================================== #
# SafeDrive's sparse encoder (spconv 2.x)
# =========================================================================== #
# SafeDrive's SpMiddleResNetFHD reduces x/y by 8 (three stride-2 stages) and
# folds what is left of z into the channel dimension.
SPARSE_DOWNSAMPLE = 8


def _import_spconv() -> Any:
    try:
        import spconv.pytorch as spconv
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "lidar_encoder='sparse' runs SafeDrive's SpMiddleResNetFHD and needs "
            "spconv 2.x.  The prebuilt wheel works on this rig: "
            "`pip install spconv-cu126` (verified on RTX 5090 / torch 2.8.0+cu128, "
            "see report/12).  Or set lidar_encoder='pillar' for the spconv-free "
            "encoder."
        ) from exc
    return spconv


def sparse_grid(
    pc_range: Sequence[float],
    bev_h: int,
    bev_w: int,
    z_range: Sequence[float],
    voxel_size: Sequence[float],
) -> Tuple[int, int, int, int]:
    """``(nx, ny, nz, depth_out)`` of the sparse encoder, or ``ValueError``.

    Pure arithmetic (no spconv), so config validation can call it anywhere.
    The x/y voxel counts must be exactly ``8 * bev_w`` / ``8 * bev_h`` because
    the backbone's stride is fixed; ``depth_out`` is what the z axis shrinks to
    after the four stride-2 convolutions (SafeDrive's 41 -> 21 -> 11 -> 5 -> 2).
    """
    if len(voxel_size) != 3:
        raise ValueError(
            f"lidar_voxel_size must be (x_right, y_forward, z) metres, got {tuple(voxel_size)}"
        )
    vx, vy, vz = (float(v) for v in voxel_size)
    for axis, value in (("x_right", vx), ("y_forward", vy), ("z", vz)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{axis} lidar voxel size must be positive, got {value}")
    extents = (
        ("x_right", pc_range[3] - pc_range[0], vx, bev_w),
        ("y_forward", pc_range[4] - pc_range[1], vy, bev_h),
    )
    counts = []
    for axis, extent, voxel, cells in extents:
        count = extent / voxel
        if not math.isclose(count, round(count), abs_tol=1e-6):
            raise ValueError(
                f"{axis} extent {extent} m is not an integer number of {voxel} m voxels"
            )
        count = int(round(count))
        if count != SPARSE_DOWNSAMPLE * cells:
            raise ValueError(
                f"the sparse LiDAR encoder has stride {SPARSE_DOWNSAMPLE}, so {axis} needs "
                f"{SPARSE_DOWNSAMPLE * cells} voxels for {cells} BEV cells; {voxel} m voxels "
                f"give {count} (use {extent / (SPARSE_DOWNSAMPLE * cells):.4g} m)"
            )
        counts.append(count)
    z_extent = float(z_range[1]) - float(z_range[0])
    nz = z_extent / vz
    if not math.isclose(nz, round(nz), abs_tol=1e-6) or round(nz) < 1:
        raise ValueError(
            f"lidar_z_range extent {z_extent} m is not a positive integer number of {vz} m voxels"
        )
    nz = int(round(nz))
    depth = nz + 1  # SafeDrive: sparse_shape = shape[::-1] + [1, 0, 0]
    depth = (depth + 2 - 3) // 2 + 1  # conv2, k3 s2 p1
    depth = (depth + 2 - 3) // 2 + 1  # conv3
    depth = (depth - 3) // 2 + 1      # conv4, z padding 0
    depth = (depth - 3) // 2 + 1      # extra_conv, (3,1,1) stride (2,1,1)
    if depth < 1:
        raise ValueError(
            f"{nz} z voxels collapse to nothing in SpMiddleResNetFHD; use finer z voxels "
            "or a taller lidar_z_range (the default 8 m / 0.2 m -> 41 -> 2)"
        )
    return counts[0], counts[1], nz, depth


class _SparseBasicBlock(nn.Module):
    """SafeDrive ``SparseBasicBlock``: two SubM 3x3 convs with a residual."""

    def __init__(self, spconv: Any, inplanes: int, planes: int, indice_key: str):
        super().__init__()
        # SafeDrive: bias = (norm_cfg is not None) -> True for these convs.
        self.conv1 = spconv.SubMConv3d(inplanes, planes, 3, stride=1, padding=1, bias=True, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(planes, eps=1e-3, momentum=0.01)
        self.conv2 = spconv.SubMConv3d(planes, planes, 3, stride=1, padding=1, bias=True, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(planes, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        return out.replace_feature(self.relu(out.features + x.features))


class _SparseDown(nn.Module):
    """``SparseConv3d -> BN -> ReLU`` stage head."""

    def __init__(self, spconv: Any, cin: int, cout: int, kernel, stride, padding):
        super().__init__()
        self.conv = spconv.SparseConv3d(cin, cout, kernel, stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(cout, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        return out.replace_feature(self.relu(self.bn(out.features)))


class SpMiddleResNetFHD(nn.Module):
    """SafeDrive's ``SpMiddleResNetFHD`` (``ds_factor=8``), layer for layer.

    ``forward`` takes mean-pooled voxel features ``[V, C_in]``, ``[V, 4]``
    ``(batch, z, y, x)`` coordinates, the batch size and the ``(z, y, x)``
    sparse shape, and returns the dense ``[B, C * D, H, W]`` map that
    SafeDrive hands its BEVFormer.
    """

    def __init__(self, spconv: Any, num_input_features: int = 5):
        super().__init__()
        self.conv_input = _SparseDown(spconv, num_input_features, 16, 3, 1, 1)
        # SafeDrive's conv_input is a SubM conv; _SparseDown would be a regular
        # sparse conv, so build the SubM version explicitly.
        self.conv_input.conv = spconv.SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0")
        self.conv1 = nn.ModuleList([_SparseBasicBlock(spconv, 16, 16, "res0") for _ in range(2)])
        self.conv2 = _SparseDown(spconv, 16, 32, 3, 2, 1)
        self.res2 = nn.ModuleList([_SparseBasicBlock(spconv, 32, 32, "res1") for _ in range(2)])
        self.conv3 = _SparseDown(spconv, 32, 64, 3, 2, 1)
        self.res3 = nn.ModuleList([_SparseBasicBlock(spconv, 64, 64, "res2") for _ in range(2)])
        self.conv4 = _SparseDown(spconv, 64, 128, 3, 2, [0, 1, 1])
        self.res4 = nn.ModuleList([_SparseBasicBlock(spconv, 128, 128, "res3") for _ in range(2)])
        self.extra_conv = _SparseDown(spconv, 128, 128, (3, 1, 1), (2, 1, 1), 0)
        self._spconv = spconv

    def forward(self, voxel_features, coors, batch_size: int, sparse_shape: Sequence[int]):
        x = self._spconv.SparseConvTensor(voxel_features, coors.int(), list(sparse_shape), batch_size)
        x = self.conv_input(x)
        for block in self.conv1:
            x = block(x)
        x = self.conv2(x)
        for block in self.res2:
            x = block(x)
        x = self.conv3(x)
        for block in self.res3:
            x = block(x)
        x = self.conv4(x)
        for block in self.res4:
            x = block(x)
        dense = self.extra_conv(x).dense()  # [B, C, D, H, W]
        n, c, d, h, w = dense.shape
        return dense.view(n, c * d, h, w)


class LidarSparseEncoder(nn.Module):
    """SafeDrive's LiDAR path on the padded point clouds: voxels -> sparse BEV.

    Points arrive in SSR axes already clipped to ``pc_range`` x ``z_range``.
    Voxels are the per-voxel mean of the raw 5-dim points (SafeDrive's
    ``voxelization``), coordinates are ``(batch, z, y_forward, x_right)`` and
    the sparse shape is ``(nz + 1, ny, nx)``, so the dense output's rows are
    ``y_forward`` and its columns ``x_right``: the BEV grid, no transpose.

    SafeDrive's output has ``128 * depth_out`` channels; with the default
    41-deep grid that is 256 = ``embed_dims`` and no projection exists.  Any
    other combination gets a 1x1 projection, which SafeDrive does not have.
    """

    POINT_DIMS = 5

    def __init__(
        self,
        pc_range: Sequence[float],
        bev_h: int,
        bev_w: int,
        embed_dims: int,
        z_range: Sequence[float] = (-3.0, 5.0),
        voxel_size: Sequence[float] = (0.08, 0.08, 0.2),
    ):
        super().__init__()
        nx, ny, nz, depth = sparse_grid(pc_range, bev_h, bev_w, z_range, voxel_size)
        spconv = _import_spconv()
        self.pc_range = [float(v) for v in pc_range]
        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.grid = (nx, ny, nz)
        self.sparse_shape = (nz + 1, ny, nx)
        self.bev_h, self.bev_w = int(bev_h), int(bev_w)
        self.backbone = SpMiddleResNetFHD(spconv, self.POINT_DIMS)
        out_channels = 128 * depth
        self.embed_dims = int(embed_dims)
        self.proj: Optional[nn.Module] = (
            None if out_channels == embed_dims else nn.Conv2d(out_channels, embed_dims, 1)
        )

    # ------------------------------------------------------------------ #
    def voxelize(
        self, points: torch.Tensor, num_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean-pooled voxel features ``[V, 5]`` and ``[V, 4]`` (b, z, y, x) coords."""
        if points.dim() != 3 or points.shape[-1] != self.POINT_DIMS:
            raise ValueError(
                f"lidar points must be [B, N, {self.POINT_DIMS}], got {tuple(points.shape)}"
            )
        batch, count, _ = points.shape
        nx, ny, nz = self.grid
        valid = torch.arange(count, device=points.device)[None, :] < num_points[:, None]
        pts = points[valid]
        batch_index = torch.arange(batch, device=points.device)[:, None].expand(batch, count)[valid]

        vx, vy, vz = self.voxel_size
        ix = ((pts[:, 0] - self.pc_range[0]) / vx).floor().long().clamp_(0, nx - 1)
        iy = ((pts[:, 1] - self.pc_range[1]) / vy).floor().long().clamp_(0, ny - 1)
        iz = ((pts[:, 2] - self.z_range[0]) / vz).floor().long().clamp_(0, nz - 1)
        key = ((batch_index * nz + iz) * ny + iy) * nx + ix
        unique, inverse = torch.unique(key, return_inverse=True)
        num_voxels = unique.shape[0]
        sums = pts.new_zeros(num_voxels, pts.shape[1]).index_add_(0, inverse, pts)
        counts = pts.new_zeros(num_voxels).index_add_(0, inverse, torch.ones_like(inverse, dtype=pts.dtype))
        features = sums / counts.clamp_min(1.0)[:, None]

        ix = unique % nx
        rest = unique // nx
        iy = rest % ny
        rest = rest // ny
        iz = rest % nz
        b = rest // nz
        coords = torch.stack((b, iz, iy, ix), dim=1)
        return features, coords

    def forward(self, points: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        features, coords = self.voxelize(points, num_points)
        num_voxels = features.shape[0]
        if num_voxels == 0:
            # spconv cannot launch a kernel over zero voxels (cumm asserts N > 0).
            # A batch whose every cloud is empty is a broken sensor, not a
            # modelling case: return an all-zero BEV instead of crashing a
            # multi-day run.  The zero-weighted parameter sum keeps every
            # parameter in the graph so DDP still sees a gradient for it.
            logger.warning("LidarSparseEncoder received a batch with no LiDAR points; emitting a zero BEV")
            edge = sum(p.sum() for p in self.parameters()) * 0.0
            return points.new_zeros(points.shape[0], self.embed_dims, self.bev_h, self.bev_w) + edge
        if self.training and num_voxels < 2:
            # BatchNorm1d needs >= 2 voxels for batch statistics.
            with _batchnorm_eval(self):
                bev = self.backbone(features, coords, points.shape[0], self.sparse_shape)
        else:
            bev = self.backbone(features, coords, points.shape[0], self.sparse_shape)
        if self.proj is not None:
            bev = self.proj(bev)
        if bev.shape[-2:] != (self.bev_h, self.bev_w):
            raise RuntimeError(
                f"sparse LiDAR BEV {tuple(bev.shape[-2:])} does not match the BEV grid "
                f"{(self.bev_h, self.bev_w)}"
            )
        return bev


def build_lidar_encoder(cfg) -> nn.Module:
    """The encoder ``cfg.lidar_encoder`` names, with the model's grid."""
    kind = str(cfg.lidar_encoder)
    if kind == "sparse":
        return LidarSparseEncoder(
            pc_range=cfg.pc_range,
            bev_h=cfg.bev_h,
            bev_w=cfg.bev_w,
            embed_dims=cfg.embed_dims,
            z_range=cfg.lidar_z_range,
            voxel_size=cfg.lidar_voxel_size,
        )
    if kind == "pillar":
        return LidarPillarEncoder(
            pc_range=cfg.pc_range,
            bev_h=cfg.bev_h,
            bev_w=cfg.bev_w,
            embed_dims=cfg.embed_dims,
            pillar_size=cfg.lidar_pillar_size,
            pillar_channels=cfg.lidar_pillar_channels,
            backbone_channels=cfg.lidar_backbone_channels,
            backbone_layers=cfg.lidar_backbone_layers,
            neck_channels=cfg.lidar_neck_channels,
        )
    raise ValueError(f"lidar_encoder must be one of {LIDAR_ENCODERS}, got {kind!r}")
