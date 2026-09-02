"""Small, framework-independent building blocks for planning distillation.

The expensive teacher encoders are frozen and evaluated once.  Training then
only reads their aligned BEV tensors.  Keeping the cache boundary here has two
useful properties:

* the teachers may live in their original, mutually incompatible Python
  environments; and
* student training does not pay for extra image encoders every iteration.

This module intentionally imports only PyTorch/NumPy so its coordinate
transform and gradient behaviour can be regression-tested outside the legacy
MMCV runtime.
"""
from collections import OrderedDict
from collections.abc import Mapping
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# SSR planning BEV: LIDAR_TOP ``x=right, y=forward``, columns along x.
SSR_BEV_BOUNDS = (-15.0, -30.0, 15.0, 30.0)

# Offline BEVFusion / MapTRv2 caches store ``token = x_index * Y + y_index``.
# Alignment is applied at load time so the adapter always sees SSR CHW maps.
TEACHER_CACHE_PRESETS = {
    'bevdepth': dict(layout='legacy_pt'),
    'hdmapnet': dict(layout='legacy_pt'),
    'bevfusion': dict(
        layout='npz_xy',
        grid_size=(100, 100),
        source_bounds=(-54.0, -54.0, 54.0, 54.0),
        target_bounds=SSR_BEV_BOUNDS,
        swap_xy=True,
        flip_x=False,
        flip_y=True,
        output_size=(100, 100),
        train_subdir='cache_train_100x100',
        val_subdir='cache_val_100x100'),
    'maptrv2': dict(
        layout='npz_xy',
        grid_size=(100, 100),
        source_bounds=SSR_BEV_BOUNDS,
        target_bounds=SSR_BEV_BOUNDS,
        swap_xy=False,
        flip_x=False,
        flip_y=False,
        output_size=(100, 100),
        train_subdir='cache_train_100x100',
        val_subdir='cache_val_100x100'),
}

_STORE_CFG_KEYS = {
    'cache_layout': 'layout',
    'cache_grid': 'grid_size',
    'source_bounds': 'source_bounds',
    'target_bounds': 'target_bounds',
    'swap_xy': 'swap_xy',
    'flip_x': 'flip_x',
    'flip_y': 'flip_y',
    'output_size': 'output_size',
    'train_subdir': 'train_subdir',
    'val_subdir': 'val_subdir',
}


def align_bev_feature(feature, source_bounds, target_bounds, output_size,
                      swap_xy=False, flip_x=False, flip_y=False):
    """Sample a physical BEV rectangle onto a canonical target grid.

    Args:
        feature: ``[B, C, H, W]`` with rows representing source ``y`` and
            columns representing source ``x``.
        source_bounds: ``(x_min, y_min, x_max, y_max)`` in metres.
        target_bounds: same convention for the desired grid.
        output_size: target ``(H, W)``.
        swap_xy: map target forward ``y`` to source forward ``x`` and target
            lateral ``x`` to source lateral ``y``.  The released teachers use
            ego ``x=forward, y=left`` while SSR's LIDAR_TOP frame uses
            ``x=right, y=forward``; their complete transform is therefore
            ``swap_xy=True, flip_y=True``.

    Returns:
        Aligned feature and a ``[1, 1, H, W]`` validity mask.  The mask is
        important when a teacher covers less physical space than the student;
        padded zeros must not become distillation targets.
    """
    if feature.dim() != 4:
        raise ValueError(f'feature must be BCHW, got {tuple(feature.shape)}')
    if len(source_bounds) != 4 or len(target_bounds) != 4:
        raise ValueError('source_bounds and target_bounds must have 4 values')
    out_h, out_w = map(int, output_size)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f'invalid output_size: {output_size}')

    sx0, sy0, sx1, sy1 = map(float, source_bounds)
    tx0, ty0, tx1, ty1 = map(float, target_bounds)
    if not (sx1 > sx0 and sy1 > sy0 and tx1 > tx0 and ty1 > ty0):
        raise ValueError('BEV bounds must be strictly increasing')

    # Cell centres, not rectangle edges.  With align_corners=False the
    # normalised formula below maps the first centre to pixel 0 exactly.
    dtype, device = feature.dtype, feature.device
    xs = torch.arange(out_w, dtype=dtype, device=device)
    ys = torch.arange(out_h, dtype=dtype, device=device)
    xs = tx0 + (xs + 0.5) * ((tx1 - tx0) / out_w)
    ys = ty0 + (ys + 0.5) * ((ty1 - ty0) / out_h)
    # No ``indexing=`` argument: the project pins PyTorch 1.9, where
    # torch.meshgrid still uses ij indexing by default and does not accept the
    # newer keyword.
    yy, xx = torch.meshgrid(ys, xs)

    source_x, source_y = (yy, xx) if swap_xy else (xx, yy)
    if flip_x:
        source_x = -source_x
    if flip_y:
        source_y = -source_y

    valid = ((source_x >= sx0) & (source_x <= sx1) &
             (source_y >= sy0) & (source_y <= sy1))
    grid_x = 2.0 * (source_x - sx0) / (sx1 - sx0) - 1.0
    grid_y = 2.0 * (source_y - sy0) / (sy1 - sy0) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = grid.unsqueeze(0).expand(feature.size(0), -1, -1, -1)
    aligned = F.grid_sample(
        feature, grid, mode='bilinear', padding_mode='zeros',
        align_corners=False)
    return aligned, valid[None, None]


def bev_tokens_to_map(tokens, spatial_size):
    """Convert ``[B, H*W, C]`` tokens to ``[B, C, H, W]``."""
    if tokens.dim() != 3:
        raise ValueError(f'tokens must be BNC, got {tuple(tokens.shape)}')
    h, w = map(int, spatial_size)
    if tokens.size(1) != h * w:
        raise ValueError(
            f'{tokens.size(1)} tokens cannot form spatial size {(h, w)}')
    return tokens.reshape(tokens.size(0), h, w, tokens.size(2)).permute(
        0, 3, 1, 2).contiguous()


def bev_map_to_tokens(feature):
    """Convert ``[B, C, H, W]`` to ``[B, H*W, C]``."""
    if feature.dim() != 4:
        raise ValueError(f'feature must be BCHW, got {tuple(feature.shape)}')
    return feature.flatten(2).transpose(1, 2).contiguous()


def resize_bev_tokens(tokens, source_size, target_size):
    """Bilinearly resize BEV tokens while preserving channel layout."""
    feature = bev_tokens_to_map(tokens, source_size)
    feature = F.interpolate(
        feature, size=tuple(target_size), mode='bilinear',
        align_corners=False)
    return bev_map_to_tokens(feature)


class PlanningBEVAdapter(nn.Module):
    """A deliberately small, per-cell residual MLP.

    Spatial alignment/downsampling is fixed and happens outside this module.
    Consequently the exact same learned adapter can be applied to a teacher's
    cached grid and to a downsampled student grid.  During distillation its
    weights are frozen: this prevents the adapter from rotating/collapsing the
    feature space merely to make the feature loss easy.
    """

    def __init__(self, channels=256, hidden_channels=256, dropout=0.0):
        super().__init__()
        self.channels = int(channels)
        self.pre_norm = nn.LayerNorm(self.channels)
        self.fc1 = nn.Linear(self.channels, int(hidden_channels))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_channels), self.channels)
        self.out_norm = nn.LayerNorm(self.channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Start as normalised identity.  This gives the planning decoder a
        # sensible BEV on iteration zero instead of a random projection.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feature):
        if feature.dim() == 4:
            feature = bev_map_to_tokens(feature)
        if feature.dim() != 3 or feature.size(-1) != self.channels:
            raise ValueError(
                f'expected BNC/BCHW with C={self.channels}, got '
                f'{tuple(feature.shape)}')
        residual = feature
        feature = self.pre_norm(feature)
        feature = self.fc2(self.dropout(self.act(self.fc1(feature))))
        return self.out_norm(residual + feature)


def xy_tokens_to_map(tokens, spatial_size):
    """Convert ``token = x_index * Y + y_index`` arrays to SSR CHW maps.

    Offline BEVFusion / MapTRv2 caches store ``[X*Y, C]`` with axis0 = x and
    axis1 = y.  SSR maps are ``[C, H=y, W=x]``.
    """
    if tokens.dim() != 2:
        raise ValueError(f'tokens must be NC, got {tuple(tokens.shape)}')
    width, height = map(int, spatial_size)
    if tokens.size(0) != width * height:
        raise ValueError(
            f'{tokens.size(0)} tokens cannot form xy size {(width, height)}')
    return tokens.reshape(width, height, tokens.size(1)).permute(
        2, 1, 0).contiguous()


def _as_bool_mask(mask, height, width):
    if mask is None:
        return torch.ones(1, height, width, dtype=torch.bool)
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if mask.dim() == 2:
        mask = mask[None]
    elif mask.dim() == 4 and mask.size(0) == 1:
        mask = mask[0]
    if tuple(mask.shape[-2:]) != (height, width):
        raise ValueError(
            f'valid_mask has shape {tuple(mask.shape)}, expected (*, {height}, '
            f'{width})')
    return mask.bool()


def _load_legacy_pt(path):
    item = torch.load(path, map_location='cpu')
    if torch.is_tensor(item):
        feature, mask = item, None
    else:
        feature = item['feature']
        mask = item.get('valid_mask')
    if feature.dim() == 4 and feature.size(0) == 1:
        feature = feature[0]
    if feature.dim() != 3:
        raise ValueError(
            f'{path}: cached feature must be CHW, got {tuple(feature.shape)}')
    return feature, _as_bool_mask(mask, feature.size(-2), feature.size(-1))


def _load_npz_xy(path, grid_size, source_bounds, target_bounds, swap_xy,
                 flip_x, flip_y, output_size):
    with np.load(path, allow_pickle=False) as packed:
        if 'bev_feature' not in packed.files:
            raise KeyError(f'{path}: missing bev_feature')
        tokens = torch.from_numpy(np.array(packed['bev_feature'], copy=True))
        mask = (np.array(packed['valid_mask'], copy=True)
                if 'valid_mask' in packed.files else None)
    feature = xy_tokens_to_map(tokens, grid_size)
    identity = source_bounds is None or (
        tuple(source_bounds) == tuple(target_bounds)
        and not swap_xy and not flip_x and not flip_y
        and tuple(output_size) == (feature.size(-2), feature.size(-1)))
    if identity:
        return feature, _as_bool_mask(
            None if mask is None else torch.as_tensor(mask),
            feature.size(-2), feature.size(-1))
    aligned, valid = align_bev_feature(
        feature[None].float(), source_bounds, target_bounds, output_size,
        swap_xy=swap_xy, flip_x=flip_x, flip_y=flip_y)
    if mask is not None:
        mask_map = _as_bool_mask(torch.as_tensor(mask), *grid_size[::-1])
        extra, _ = align_bev_feature(
            mask_map[None].float(), source_bounds, target_bounds, output_size,
            swap_xy=swap_xy, flip_x=flip_x, flip_y=flip_y)
        valid = valid & (extra > 0.5)
    return aligned[0], valid[0]


class TeacherFeatureStore:
    """Token-addressed on-disk store written by a frozen teacher cache.

    Legacy BEVDepth / HDMapNet caches live at
    ``<root>/<teacher>/<token[:2]>/<token>.pt`` as already-aligned CHW maps.
    Offline BEVFusion / MapTRv2 caches live at
    ``<root>/<teacher>/cache_{train,val}_100x100/samples/<token[:2]>/<token>.npz``
    and are converted to SSR CHW maps on load.
    """

    def __init__(self, root, teacher, **overrides):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.teacher = teacher
        preset = dict(TEACHER_CACHE_PRESETS.get(teacher, dict(layout='legacy_pt')))
        unknown = [key for key in overrides if key not in preset and key not in (
            'layout', 'grid_size', 'source_bounds', 'target_bounds', 'swap_xy',
            'flip_x', 'flip_y', 'output_size', 'train_subdir', 'val_subdir')]
        if unknown:
            raise TypeError(
                f'unused TeacherFeatureStore options for {teacher}: {unknown}')
        preset.update(overrides)
        self.layout = preset.get('layout', 'legacy_pt')
        self.grid_size = tuple(preset['grid_size']) if 'grid_size' in preset else None
        self.source_bounds = preset.get('source_bounds')
        self.target_bounds = preset.get('target_bounds', SSR_BEV_BOUNDS)
        self.swap_xy = bool(preset.get('swap_xy', False))
        self.flip_x = bool(preset.get('flip_x', False))
        self.flip_y = bool(preset.get('flip_y', False))
        self.output_size = tuple(preset.get('output_size', (100, 100)))
        self.train_subdir = preset.get('train_subdir')
        self.val_subdir = preset.get('val_subdir')
        if self.layout not in ('legacy_pt', 'npz_xy'):
            raise ValueError(f'unknown cache layout: {self.layout}')
        if self.layout == 'npz_xy' and self.grid_size is None:
            raise ValueError(f'{teacher}: npz_xy caches require grid_size')

    def path_for(self, token):
        token = str(token)
        shard = token[:2]
        if self.layout == 'npz_xy':
            candidates = []
            for subdir in (self.train_subdir, self.val_subdir):
                if subdir:
                    candidates.append(os.path.join(
                        self.root, self.teacher, subdir, 'samples', shard,
                        token + '.npz'))
            candidates.extend((
                os.path.join(self.root, self.teacher, 'samples', shard,
                             token + '.npz'),
                os.path.join(self.root, self.teacher, shard, token + '.npz'),
                os.path.join(self.root, self.teacher, token + '.npz'),
            ))
            for path in candidates:
                if os.path.isfile(path):
                    return path
            return candidates[0]

        sharded = os.path.join(
            self.root, self.teacher, shard, token + '.pt')
        if os.path.isfile(sharded):
            return sharded
        # Backward-compatible flat layout is useful for small smoke caches.
        flat = os.path.join(self.root, self.teacher, token + '.pt')
        return flat if os.path.isfile(flat) else sharded

    def _load_one(self, path):
        if self.layout == 'npz_xy':
            return _load_npz_xy(
                path, self.grid_size, self.source_bounds, self.target_bounds,
                self.swap_xy, self.flip_x, self.flip_y, self.output_size)
        return _load_legacy_pt(path)

    def load_batch(self, tokens, device, dtype):
        features, masks, missing = [], [], []
        for token in tokens:
            path = self.path_for(token)
            if not os.path.isfile(path):
                missing.append(str(token))
                continue
            feature, mask = self._load_one(path)
            features.append(feature)
            masks.append(mask.bool())
        if missing:
            preview = ', '.join(missing[:3])
            raise FileNotFoundError(
                f'{self.teacher} cache misses {len(missing)} sample(s), e.g. '
                f'{preview}. Generate both train and val caches first.')
        return (torch.stack(features).to(device=device, dtype=dtype,
                                          non_blocking=True),
                torch.stack(masks).to(device=device, non_blocking=True))


def current_sample_tokens(img_metas):
    """Extract current-frame nuScenes tokens from train/test metadata nests."""
    metas = img_metas
    # Test-time augmentation adds one list level.
    while isinstance(metas, (list, tuple)) and len(metas) == 1 and \
            isinstance(metas[0], (list, tuple)):
        metas = metas[0]
    if isinstance(metas, dict):
        metas = [metas]

    tokens = []
    for meta in metas:
        if isinstance(meta, dict) and 'sample_idx' not in meta:
            # Temporal training metadata is {queue_index: frame_meta}.
            numeric = [k for k in meta if isinstance(k, int)]
            if numeric:
                meta = meta[max(numeric)]
        if not isinstance(meta, dict) or 'sample_idx' not in meta:
            raise KeyError('img_metas does not contain current sample_idx')
        tokens.append(meta['sample_idx'])
    return tokens


def _checkpoint_state(path):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, (dict, OrderedDict)):
        raise TypeError(f'{path}: checkpoint does not contain a state dict')
    return state


def load_prefixed_module(module, state, prefix):
    """Strictly load one module from an MMCV checkpoint prefix."""
    candidates = (prefix, 'module.' + prefix)
    for candidate in candidates:
        selected = {
            key[len(candidate):]: value for key, value in state.items()
            if key.startswith(candidate)
        }
        if selected:
            module.load_state_dict(selected, strict=True)
            return candidate
    raise KeyError(
        f'checkpoint has no parameters under {prefix!r}; expected a stage-1 '
        f'teacher-adapter checkpoint')


class PlanningDistillation(nn.Module):
    """Frozen teacher adapters and feature losses attached to ParaSSR.

    A branch uses one *shared module instance* for the teacher and student
    forward.  Its parameters are frozen, but autograd still differentiates the
    student input, which is exactly the intended path:

        feature loss -> frozen task adapter -> student BEV encoder
    """

    def __init__(self, feature_root, adapter_checkpoint, branches,
                 student_bev_size=(100, 100), cache_size=(25, 25)):
        super().__init__()
        self.student_bev_size = tuple(student_bev_size)
        self.cache_size = tuple(cache_size)
        self.adapters = nn.ModuleDict()
        self.stores = {}
        self.loss_weights = {}

        if isinstance(adapter_checkpoint, (str, os.PathLike)):
            checkpoint_by_branch = {
                name: os.fspath(adapter_checkpoint) for name in branches}
        elif isinstance(adapter_checkpoint, Mapping):
            checkpoint_by_branch = dict(adapter_checkpoint)
            missing = [name for name in branches
                       if name not in checkpoint_by_branch]
            unused = [name for name in checkpoint_by_branch
                      if name not in branches]
            if missing or unused:
                raise KeyError(
                    'adapter checkpoint mapping mismatch: '
                    f'missing={missing}, unused={unused}')
        else:
            raise TypeError(
                'adapter_checkpoint must be one path or a teacher-to-path '
                f'mapping, got {type(adapter_checkpoint)}')

        states = {}
        loaded = []
        loaded_checkpoints = {}
        for name, cfg in branches.items():
            cfg = dict(cfg)
            adapter_cfg = dict(cfg.pop('adapter', {}))
            adapter = PlanningBEVAdapter(**adapter_cfg)
            prefix = cfg.pop(
                'adapter_prefix', f'branches.{name}.adapter.')
            checkpoint_path = os.path.abspath(os.path.expanduser(
                os.fspath(checkpoint_by_branch[name])))
            if checkpoint_path not in states:
                states[checkpoint_path] = _checkpoint_state(checkpoint_path)
            state = states[checkpoint_path]
            loaded.append(load_prefixed_module(adapter, state, prefix))
            loaded_checkpoints[name] = checkpoint_path
            adapter.requires_grad_(False)
            adapter.eval()
            self.adapters[name] = adapter
            store_kwargs = {}
            for src, dst in _STORE_CFG_KEYS.items():
                if src in cfg:
                    store_kwargs[dst] = cfg.pop(src)
            self.stores[name] = TeacherFeatureStore(
                feature_root, cfg.pop('cache_name', name), **store_kwargs)
            self.loss_weights[name] = float(cfg.pop('loss_weight', 1.0))
            if cfg:
                raise TypeError(f'unused distillation options for {name}: {cfg}')
        self.loaded_adapter_prefixes = tuple(loaded)
        self.loaded_adapter_checkpoints = loaded_checkpoints

    def train(self, mode=True):
        super().train(mode)
        # Parent model.train() must never turn the teacher feature space into a
        # moving target (Dropout is configurable even though the default is 0).
        for adapter in self.adapters.values():
            adapter.eval()
        return self

    def forward_train(self, student_bev, img_metas):
        student_map = bev_tokens_to_map(student_bev, self.student_bev_size)
        student_map = F.interpolate(
            student_map, size=self.cache_size, mode='bilinear',
            align_corners=False)
        tokens = current_sample_tokens(img_metas)
        losses, metrics = {}, {}
        for name, adapter in self.adapters.items():
            teacher_map, valid = self.stores[name].load_batch(
                tokens, student_map.device, student_map.dtype)
            if tuple(teacher_map.shape[-2:]) != self.cache_size:
                raise ValueError(
                    f'{name} cache has {tuple(teacher_map.shape[-2:])}, '
                    f'expected {self.cache_size}')
            with torch.no_grad():
                teacher = adapter(teacher_map)
            student = adapter(student_map)
            mask = valid.flatten(2).transpose(1, 2).to(student.dtype)
            denom = (mask.sum() * student.size(-1)).clamp(min=1.0)
            mse = ((student - teacher).square() * mask).sum() / denom
            losses[f'loss_distill_{name}'] = mse * self.loss_weights[name]

            with torch.no_grad():
                cosine = F.cosine_similarity(student, teacher, dim=-1)
                token_mask = mask.squeeze(-1)
                n = token_mask.sum().clamp(min=1.0)
                metrics[f'distill_cos/{name}'] = \
                    (cosine * token_mask).sum() / n
                metrics[f'distill_rmse/{name}'] = mse.detach().sqrt()
                metrics[f'distill_valid/{name}'] = token_mask.mean()
        return losses, metrics
