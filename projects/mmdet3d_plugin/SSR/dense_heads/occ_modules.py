"""Convolutional BEV blocks for PARA-SSR's occupancy head.

Ported from UniAD (``projects/mmdet3d_plugin/uniad/dense_heads/occ_head_plugin/
modules.py``), keeping only the parts that are *pure BEV convolution* and have
no dependency on track / motion instance queries.  ``einops`` is not installed
in the SSR environment, so ``rearrange`` calls are replaced with plain reshapes.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule


class SimpleConv2d(BaseModule):
    """A few 3x3 convs followed by a 1x1 projection, keeping H/W unchanged."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_channels=64,
                 num_conv=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels
        if num_conv == 1:
            conv_channels = in_channels

        conv_layers = []
        c_in = in_channels
        for _ in range(num_conv - 1):
            conv_layers.append(
                ConvModule(
                    c_in,
                    conv_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            c_in = conv_channels
        # No norm and relu in last conv
        conv_layers.append(
            build_conv_layer(
                conv_cfg,
                conv_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))
        self.conv_layers = nn.Sequential(*conv_layers)

        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        b, _, h_in, w_in = x.size()
        out = self.conv_layers(x)
        assert out.size() == (b, self.out_channels, h_in, w_in)
        return out


class CVT_DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor,
                 upsample, with_relu=True):
        super().__init__()
        dim = out_channels // factor

        layers = []
        if upsample:
            layers.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers += [
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.conv = nn.Sequential(*layers)

        self.up = nn.Conv2d(skip_dim, out_channels, 1) if residual else None
        self.with_relu = with_relu
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)
        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])
            x = x + up
        if self.with_relu:
            return self.relu(x)
        return x


class CVT_Decoder(BaseModule):
    """Upsamples a ``[B, T, C, H, W]`` stack by 2x per block."""

    def __init__(self, dim, blocks, residual=True, factor=2, upsample=True,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        layers = []
        channels = dim
        for i, out_channels in enumerate(blocks):
            with_relu = i < len(blocks) - 1  # no relu on the last block
            layers.append(
                CVT_DecoderBlock(channels, out_channels, dim, residual, factor,
                                 upsample, with_relu=with_relu))
            channels = out_channels

        self.layers = nn.ModuleList(layers)
        self.out_channels = channels
        if init_cfg is None:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        y = x
        for layer in self.layers:
            y = layer(y, x)
        return y.reshape(b, t, y.shape[1], y.shape[2], y.shape[3])


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                        align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        # `x_skip` may be one pixel larger when the input size is odd
        if x.shape[-2:] != x_skip.shape[-2:]:
            x = F.interpolate(x, size=x_skip.shape[-2:], mode='bilinear',
                              align_corners=False)
        return x + x_skip


class Interpolate(nn.Module):
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self._interpolate = nn.functional.interpolate
        self._scale_factor = scale_factor

    def forward(self, x):
        return self._interpolate(x, scale_factor=self._scale_factor,
                                 mode='bilinear', align_corners=False)


class Bottleneck(nn.Module):
    """Bottleneck residual block with optional 2x up/down sampling."""

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_size=3,
                 dilation=1,
                 groups=1,
                 upsample=False,
                 downsample=False,
                 dropout=0.0):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        assert dilation == 1
        if upsample:
            assert not downsample, \
                'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(
                bottleneck_channels, bottleneck_channels,
                kernel_size=kernel_size, bias=False, dilation=1, stride=2,
                output_padding=padding_size, padding=padding_size, groups=groups)
        elif downsample:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels, bottleneck_channels,
                kernel_size=kernel_size, bias=False, dilation=dilation,
                stride=2, padding=padding_size, groups=groups)
        else:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels, bottleneck_channels,
                kernel_size=kernel_size, bias=False, dilation=dilation,
                padding=padding_size, groups=groups)

        self.layers = nn.Sequential(
            OrderedDict([
                ('conv_down_project',
                 nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                           bias=False)),
                ('abn_down_project',
                 nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
                               nn.ReLU(inplace=True))),
                ('conv', bottleneck_conv),
                ('abn',
                 nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
                               nn.ReLU(inplace=True))),
                ('conv_up_project',
                 nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                           bias=False)),
                ('abn_up_project',
                 nn.Sequential(nn.BatchNorm2d(out_channels),
                               nn.ReLU(inplace=True))),
                ('dropout', nn.Dropout2d(p=dropout)),
            ]))

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update(
                    {'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update({
                'conv_skip_proj':
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                'bn_skip_proj': nn.BatchNorm2d(out_channels),
            })
            self.projection = nn.Sequential(projection)

    def forward(self, *args):
        (x, ) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad odd h/w so the residual shapes line up
                x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x
