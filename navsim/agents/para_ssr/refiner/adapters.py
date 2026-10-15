"""M2 input adapters for the stage-T refiner (IMPL_SPEC §3.8): scene feature -> common [64, 50, 100] map on the S grid.

Arms (identical code everywhere else; ONLY this module differs between arms)
  'T'    : teacher BEVFusion 50x100 bev_feature [256, 50, 100] (float16 cache, post-ReLU)
           -> S grid (lateral flip, below) -> per-channel z-score (mean / std over the run's TRAIN tokens, stored in
           runs/<run>/norm.npz, frozen buffers, not parameters) -> 1x1 conv 256->128, GELU, 1x1 conv 128->64.
           Parameters: 256*128 + 128 + 128*64 + 64 = 41,152.
  'none' : no scene feature: zeros [64, 50, 100] (0 parameters).  The rest of the network (corridor sampling of the
           zero map, global tokens = learned positional embedding only) is the same module with the same parameters.
  'M'    : (PRESTATED_DECISION_RULE AMENDMENT 6, run 4) ReSMap map-teacher neck BEV [256, 50, 100] on the S grid
           (resmap_cache.ResmapCache.load_bev = transpose of the last two axes, origin rear axle) -> per-channel z-score
           (train tokens, runs/<run>/norm_map.npz) -> 1x1 256->128, GELU, 1x1 128->64: the SAME form as 'T' (AdapterM).
           Parameters 41,152.
  'TM'   : two branches, det = AdapterT form on channels 0..255 (BEVFusion, norm.npz), map = AdapterM form on channels
           256..511 (ReSMap, norm_map.npz), each with its own ChannelZScore; outputs concatenated (128 ch) -> 1x1
           128->64 ('fuse', no activation after it, as the amendment states: "concat 128 -> 1x1 128->64").  Input = ONE
           tensor [T, 512, 50, 100] (det first, then map) so RefinerNet.forward keeps its signature.
           Parameters 2 x 41,152 + 128*64 + 64 = 90,560.
           Branch drop (evaluation analysis only): drop_branch = 'det' | 'map' sets that branch's NORMALISED input to 0
           (= the per-channel training-mean feature); None (default) = normal forward.  Not a parameter / buffer (never
           saved in a checkpoint).

Adapter initialisation (build_adapter(..., seed=s), s = RefinerNet seed + ADAPTER_SEED_OFFSET; the caller has already
  called torch.manual_seed(s))
  T  : built directly from the caller's RNG state (unchanged from runs 1-3).
  M  : torch.manual_seed(s + MAP_SEED_OFFSET) first.
  TM : det branch under torch.manual_seed(s)  (== the T adapter of the same seed, parameter for parameter),
       map branch under torch.manual_seed(s + MAP_SEED_OFFSET)  (== the M adapter of the same seed),
       fuse under torch.manual_seed(s + FUSE_SEED_OFFSET).
  So a TM net's branches start exactly where the single-teacher nets start (tests/test_stageT4.py).

Grids (IMPL_SPEC §2)
  S grid [C, 50, 100]: row r -> x = (r + 0.5) * 0.64 m (0..32 m forward); col c -> y_left = 32 - (c + 0.5) * 0.64 m
          (col 0 = 32 m LEFT).  N frame = NAVSIM ego frame at t0 (rear axle, x forward, y left).
  Teacher cache bev_feature / dense_heatmap: (C, H = x_forward, W = y_left) with col j -> y_left = -32 + (j + 0.5) * 0.64
          (mmdet3d LiDAR frame, lidar2ego = identity => origin = rear axle).  Hence S = bev[:, :, ::-1]
          (teacher_to_s_grid); verified in tools/refiner/tests/test_adapters.py with the cached dense_heatmap vs the
          cached pred_boxes_3d centres (class logit at the box cell vs at the mirrored cell).

Normalisation
  z = (x - mean_c) / std_c per channel c over all cells of all sampled train tokens (float64 accumulation,
  ChannelStats).  std is floored at STD_FLOOR.  The same statistic definition would be used for any other feature arm
  (critic_logic.md issue 5: per-cell LayerNorm is replaced by per-channel z-score for every arm).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------------- constants
BEV_H, BEV_W = 50, 100       # S grid rows (x forward), cols (y left, col 0 = left)
CELL = 0.64                  # [m]
IN_CH_T = 256                # teacher bev_feature channels
IN_CH_M = 256                # ReSMap neck bev channels (map arm)
HID_CH = 128
OUT_CH = 64                  # common feature channels after the adapter
STD_FLOOR = 1e-3
ARMS = ("T", "none", "M", "TM")
MAP_SEED_OFFSET = 104729     # M adapter / TM map branch: seed + MAP_SEED_OFFSET (see module docstring)
FUSE_SEED_OFFSET = 1299709   # TM fuse conv
DROP_BRANCHES = ("det", "map")


# ----------------------------------------------------------------------------------------------- grid
def teacher_to_s_grid(x):
    """Teacher cache layout (C, H = x_fwd, W = y_left ascending) -> S grid (col 0 = left).  numpy (contiguous copy)
    or torch (flip of the last dim).  Works for bev_feature, dense_heatmap and batched [..., H, W] arrays."""
    if isinstance(x, torch.Tensor):
        return torch.flip(x, dims=[-1])
    return np.ascontiguousarray(np.asarray(x)[..., ::-1])


def s_grid_cell(x, y):
    """(x, y_left) [m] -> (row, col) float cell coordinates of the S grid (cell centres at integers)."""
    return np.asarray(x) / CELL - 0.5, (32.0 - np.asarray(y)) / CELL - 0.5


# ----------------------------------------------------------------------------------------------- statistics
class ChannelStats:
    """Streaming per-channel mean / std over [C, H, W] arrays (float64 sums)."""

    def __init__(self, n_ch: int = IN_CH_T):
        self.n_ch = n_ch
        self.s1 = np.zeros(n_ch, np.float64)
        self.s2 = np.zeros(n_ch, np.float64)
        self.n = 0            # cells per channel
        self.n_maps = 0
        self.frac_zero = np.zeros(n_ch, np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float64).reshape(self.n_ch, -1)
        self.s1 += x.sum(1)
        self.s2 += (x * x).sum(1)
        self.frac_zero += (x == 0).sum(1)
        self.n += x.shape[1]
        self.n_maps += 1

    def result(self):
        if self.n == 0:
            raise ValueError("ChannelStats: no data")
        mean = self.s1 / self.n
        var = np.maximum(self.s2 / self.n - mean * mean, 0.0)
        std = np.maximum(np.sqrt(var), STD_FLOOR)
        return mean.astype(np.float32), std.astype(np.float32)


def save_norm(path: Union[str, Path], mean: np.ndarray, std: np.ndarray, meta: Optional[Dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez(tmp, mean=np.asarray(mean, np.float32), std=np.asarray(std, np.float32),
             meta=np.array(json.dumps(meta or {}, sort_keys=True)))
    tmp.replace(path)


def load_norm(path: Union[str, Path]):
    """-> (mean [C] f32, std [C] f32, meta dict)."""
    with np.load(path, allow_pickle=False) as z:
        return z["mean"].astype(np.float32), z["std"].astype(np.float32), json.loads(str(z["meta"]))


def compute_norm(bev_iter: Iterable[np.ndarray], n_ch: int = IN_CH_T):
    """Per-channel z-score statistics of an iterable of [C, H, W] maps -> (mean, std, info)."""
    st = ChannelStats(n_ch)
    for x in bev_iter:
        st.update(x)
    mean, std = st.result()
    info = dict(n_maps=st.n_maps, n_cells=int(st.n), frac_zero_mean=float((st.frac_zero / st.n).mean()))
    return mean, std, info


# ----------------------------------------------------------------------------------------------- modules
class ChannelZScore(nn.Module):
    """(x - mean_c) / std_c with frozen buffers (not parameters, excluded from the parameter count)."""

    def __init__(self, n_ch: int, mean=None, std=None):
        super().__init__()
        m = torch.zeros(n_ch) if mean is None else torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        s = torch.ones(n_ch) if std is None else torch.as_tensor(np.asarray(std), dtype=torch.float32)
        if m.shape != (n_ch,) or s.shape != (n_ch,):
            raise ValueError(f"norm stats shape {tuple(m.shape)} / {tuple(s.shape)} != ({n_ch},)")
        self.register_buffer("mean", m.clone())
        self.register_buffer("inv_std", 1.0 / torch.clamp(s.clone(), min=STD_FLOOR))
        self.register_buffer("is_set", torch.tensor(mean is not None and std is not None))

    def set_stats(self, mean, std) -> None:
        self.mean.copy_(torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.inv_std.copy_(1.0 / torch.clamp(torch.as_tensor(np.asarray(std), dtype=torch.float32), min=STD_FLOOR))
        self.is_set.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)[:, None, None]) * self.inv_std.to(x.dtype)[:, None, None]


class AdapterT(nn.Module):
    """Teacher BEV adapter: S-grid bev [T, 256, 50, 100] -> z-score -> 1x1 256->128, GELU, 1x1 128->64."""

    arm = "T"
    in_ch = IN_CH_T
    needs_bev = True

    def __init__(self, mean=None, std=None, in_ch: int = IN_CH_T, out_ch: int = OUT_CH):
        super().__init__()
        self.in_ch = in_ch
        self.norm = ChannelZScore(in_ch, mean, std)
        self.conv1 = nn.Conv2d(in_ch, HID_CH, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(HID_CH, out_ch, 1)
        self.out_ch = out_ch

    def forward(self, bev: torch.Tensor, n_tokens: Optional[int] = None) -> torch.Tensor:
        if bev is None:
            raise ValueError("AdapterT needs the teacher BEV")
        if bev.shape[-3:] != (self.in_ch, BEV_H, BEV_W):
            raise ValueError(f"bev shape {tuple(bev.shape)} != [T, {self.in_ch}, {BEV_H}, {BEV_W}]")
        if not bool(self.norm.is_set):
            raise RuntimeError("AdapterT: normalisation statistics not set (load runs/<run>/norm.npz)")
        x = self.norm(bev.float())
        return self.conv2(self.act(self.conv1(x)))


class AdapterNone(nn.Module):
    """No scene feature: zeros [T, 64, 50, 100] (dtype / device of the dummy buffer).  0 parameters."""

    arm = "none"
    in_ch = 0
    needs_bev = False

    def __init__(self, out_ch: int = OUT_CH):
        super().__init__()
        self.out_ch = out_ch
        self.register_buffer("_ref", torch.zeros(()), persistent=False)

    def forward(self, bev: Optional[torch.Tensor] = None, n_tokens: Optional[int] = None) -> torch.Tensor:
        T = int(n_tokens) if n_tokens is not None else int(bev.shape[0])
        return self._ref.new_zeros(T, self.out_ch, BEV_H, BEV_W)


class AdapterM(AdapterT):
    """ReSMap map-teacher adapter (arm 'M'): the AdapterT form on the ReSMap S-grid bev [T, 256, 50, 100]."""

    arm = "M"
    in_ch = IN_CH_M

    def __init__(self, mean=None, std=None, in_ch: int = IN_CH_M, out_ch: int = OUT_CH):
        super().__init__(mean, std, in_ch=in_ch, out_ch=out_ch)

    def forward(self, bev: torch.Tensor, n_tokens: Optional[int] = None) -> torch.Tensor:
        if bev is not None and not bool(self.norm.is_set):
            raise RuntimeError("AdapterM: normalisation statistics not set (load runs/<run>/norm_map.npz)")
        return super().forward(bev, n_tokens)


class AdapterTM(nn.Module):
    """Det (BEVFusion, AdapterT form) + map (ReSMap, AdapterM form) branches -> concat 128 -> 1x1 128->64.
    Input [T, 512, 50, 100]: channels 0..255 det, 256..511 map.  drop_branch: see module docstring."""

    arm = "TM"
    in_ch = IN_CH_T + IN_CH_M
    needs_bev = True

    def __init__(self, det_mean=None, det_std=None, map_mean=None, map_std=None, out_ch: int = OUT_CH,
                 seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.det = AdapterT(det_mean, det_std)
        if seed is not None:
            torch.manual_seed(seed + MAP_SEED_OFFSET)
        self.map = AdapterM(map_mean, map_std)
        if seed is not None:
            torch.manual_seed(seed + FUSE_SEED_OFFSET)
        self.fuse = nn.Conv2d(self.det.out_ch + self.map.out_ch, out_ch, 1)
        self.out_ch = out_ch
        self.drop_branch: Optional[str] = None

    def set_drop_branch(self, which: Optional[str]) -> None:
        if which is not None and which not in DROP_BRANCHES:
            raise ValueError(f"drop_branch {which!r} not in {DROP_BRANCHES}")
        self.drop_branch = which

    @staticmethod
    def _branch(ad: AdapterT, x: torch.Tensor, drop: bool) -> torch.Tensor:
        z = ad.norm(x.float())
        if drop:
            z = torch.zeros_like(z)          # normalised input 0 == the per-channel training-mean feature
        return ad.conv2(ad.act(ad.conv1(z)))

    def forward(self, bev: torch.Tensor, n_tokens: Optional[int] = None) -> torch.Tensor:
        if bev is None:
            raise ValueError("AdapterTM needs the concatenated [det, map] BEV")
        if bev.shape[-3:] != (self.in_ch, BEV_H, BEV_W):
            raise ValueError(f"bev shape {tuple(bev.shape)} != [T, {self.in_ch}, {BEV_H}, {BEV_W}]")
        if not bool(self.det.norm.is_set):
            raise RuntimeError("AdapterTM: det normalisation statistics not set (load runs/<run>/norm.npz)")
        if not bool(self.map.norm.is_set):
            raise RuntimeError("AdapterTM: map normalisation statistics not set (load runs/<run>/norm_map.npz)")
        d = self._branch(self.det, bev[:, :IN_CH_T], self.drop_branch == "det")
        m = self._branch(self.map, bev[:, IN_CH_T:], self.drop_branch == "map")
        return self.fuse(torch.cat([d, m], 1))


def build_adapter(arm: str, mean=None, std=None, map_mean=None, map_std=None, seed: Optional[int] = None) -> nn.Module:
    """mean / std: BEVFusion z-score stats (arms T, TM); map_mean / map_std: ReSMap stats (arms M, TM).  seed: the
    adapter seed (module docstring; arm T ignores it and uses the caller's RNG state exactly as in runs 1-3)."""
    if arm == "T":
        return AdapterT(mean, std)
    if arm == "none":
        return AdapterNone()
    if arm == "M":
        if seed is not None:
            torch.manual_seed(seed + MAP_SEED_OFFSET)
        return AdapterM(map_mean, map_std)
    if arm == "TM":
        return AdapterTM(mean, std, map_mean, map_std, seed=seed)
    raise ValueError(f"arm {arm!r} not in {ARMS}")
