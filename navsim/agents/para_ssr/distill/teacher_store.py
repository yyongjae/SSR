"""Token-addressed on-disk store for cached frozen-teacher BEV features.

The NAVSIM caches are written by the BEVFusion/MapTRv2 extraction on another
machine and land at::

    <root>/<teacher>/cache_{train,val}_100x100/samples/<token[:2]>/<token>.npz

Each ``.npz`` holds ``bev_feature`` already in CHW at the cache resolution --
unlike the nuScenes ``npz_xy`` caches described in
``docs/PLANNING_DISTILLATION.md``, which store ``[X*Y, C]`` xy-major tokens.
The teacher grid is the mmdet3d LiDAR frame ``(C, H=x_forward, W=y_left)`` and
the student grid is the PARA-SSR frame ``(C, bev_h=y_forward, bev_w=x_right)``.
Rows already agree; the columns run in opposite directions, so the conversion
is the manifest's ``student_bev = teacher_bev[:, :, ::-1]``.

Geometry is *verified*, not assumed.  A cache whose extent or class set does not
match the student config is the failure that trains for days and produces
numbers nobody can interpret, so :meth:`TeacherFeatureStore.validate_manifest`
compares the recorded manifest against the live config and refuses a mismatch.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

BEV_FEATURE_KEY = "bev_feature"
EXPECTED_TRANSFORM = "student_bev = teacher_bev[:, :, ::-1]"


class TeacherCacheMismatch(RuntimeError):
    """The cache on disk does not describe the model being trained."""


class TeacherFeatureStore:
    """Loads one teacher's cached BEV maps by NAVSIM frame token."""

    def __init__(
        self,
        root: str,
        teacher: str,
        cache_subdirs: Sequence[str] = (
            "cache_train_50x100",
            "cache_val_50x100",
            "cache_train_100x100",
            "cache_val_100x100",
        ),
        feature_key: str = BEV_FEATURE_KEY,
        flip_w: bool = True,
    ) -> None:
        self.root = os.path.abspath(os.path.expanduser(root))
        self.teacher = str(teacher)
        self.cache_subdirs = tuple(cache_subdirs)
        self.feature_key = str(feature_key)
        self.flip_w = bool(flip_w)
        self._manifests: Optional[Dict[str, dict]] = None
        self._meta: Optional[dict] = None
        self._index: Optional[Dict[str, list]] = None
        self._mmap_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def teacher_root(self) -> str:
        return os.path.join(self.root, self.teacher)

    def is_sharded(self) -> bool:
        return os.path.isfile(os.path.join(self.teacher_root(), "index.json"))

    def meta(self) -> Optional[dict]:
        if self._meta is None:
            path = os.path.join(self.teacher_root(), "meta.json")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    self._meta = json.load(handle)
        return self._meta

    def _get_index(self) -> Dict[str, list]:
        if self._index is None:
            path = os.path.join(self.teacher_root(), "index.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{self.teacher}: index.json not found at {path}")
            with open(path, "r", encoding="utf-8") as handle:
                self._index = json.load(handle)
        return self._index

    def manifests(self) -> Dict[str, dict]:
        """Read every split manifest once; missing ones are simply absent."""
        if self._manifests is None:
            found = {}
            for subdir in self.cache_subdirs:
                path = os.path.join(self.teacher_root(), subdir, "manifest.json")
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as handle:
                        found[subdir] = json.load(handle)
            self._manifests = found
        return self._manifests

    def spatial_size(self) -> Tuple[int, int]:
        """``(H, W)`` of tensors returned by :meth:`load_batch`.

        Stage-1's planner positional encoding must match this grid.  The student
        default is 100x100; ReSMap's sharded cache is 50x100 after the
        ``(C, lateral, forward) -> (C, forward, lateral)`` transpose.
        """
        if self.is_sharded():
            meta = self.meta()
            if meta is None:
                raise TeacherCacheMismatch(
                    f"{self.teacher}: index.json exists but meta.json missing under {self.teacher_root()}"
                )
            return self._hw_from_sharded_meta(meta)
        for subdir in self.cache_subdirs:
            manifest = self.manifests().get(subdir)
            if not manifest:
                continue
            got_hw = tuple(int(v) for v in manifest.get("target_bev_shape", ()))
            if len(got_hw) == 2:
                return got_hw
        raise TeacherCacheMismatch(
            f"{self.teacher}: cannot infer BEV grid from manifests under {self.teacher_root()}"
        )

    @staticmethod
    def _hw_from_sharded_meta(meta: dict) -> Tuple[int, int]:
        bev_info = meta.get("tensors", {}).get("bev", {})
        bev_shape = bev_info.get("shape", [])
        if len(bev_shape) == 3:
            # ReSMap raw axes: (C, lateral, forward) e.g. [256, 100, 50].
            # load_batch transposes to (C, forward, lateral) when dim1 > dim2.
            height, width = int(bev_shape[1]), int(bev_shape[2])
            if height > width:
                return (width, height)
            return (height, width)
        grid = meta.get("bev_grid", [])
        if len(grid) == 2:
            # bev_grid is [lateral, forward] = [100, 50] for ReSMap.
            lat, fwd = int(grid[0]), int(grid[1])
            if lat > fwd:
                return (fwd, lat)
            return (lat, fwd)
        raise TeacherCacheMismatch("sharded meta has neither tensors.bev.shape nor bev_grid")

    def validate_manifest(self, config) -> None:
        """Fail loudly when the cache and the student disagree.

        Checks the two things that silently corrupt a distillation run: the BEV
        grid the tensor was written on, and the physical extent that grid
        covers.  A shape-only check passes for a cache of the wrong dataset.
        """
        want_hws = [
            (int(config.bev_h), int(config.bev_w)),
            (50, 100),  # Standard 0.64m square-cell front-only cache grid
        ]
        if hasattr(config, "distill_cache_size"):
            want_hws.append(tuple(int(v) for v in config.distill_cache_size))

        sx0, sy0, _, sx1, sy1, _ = (float(v) for v in config.pc_range)
        want_lon = (sy0, sy1)
        want_lat = (sx0, sx1)

        if self.is_sharded():
            meta = self.meta()
            if meta is None:
                raise TeacherCacheMismatch(
                    f"{self.teacher}: index.json exists but meta.json missing under {self.teacher_root()}"
                )
            where = f"{self.teacher} (sharded)"
            bev_info = meta.get("tensors", {}).get("bev", {})
            bev_shape = bev_info.get("shape", [])
            channels = int(bev_shape[0]) if len(bev_shape) == 3 else int(config.embed_dims)
            got_hw = self._hw_from_sharded_meta(meta)

            if got_hw not in want_hws:
                raise TeacherCacheMismatch(
                    f"{where}: cache grid {got_hw} not in allowable student/cache grids {want_hws}"
                )
            if channels != int(config.embed_dims):
                raise TeacherCacheMismatch(
                    f"{where}: cache has {channels} channels, student expects {int(config.embed_dims)}"
                )
            pcr = meta.get("pc_range")
            if pcr is not None:
                tx0, ty0, _, tx1, ty1, _ = (float(v) for v in pcr)
                got_lon, got_lat = (tx0, tx1), (ty0, ty1)
                if not _close(got_lon, want_lon) or not _close(got_lat, want_lat):
                    raise TeacherCacheMismatch(
                        f"{where}: cache covers longitudinal {got_lon} lateral {got_lat} m, "
                        f"student covers longitudinal {want_lon} lateral {want_lat} m."
                    )
            return

        manifests = self.manifests()
        if not manifests:
            raise TeacherCacheMismatch(
                f"{self.teacher}: no manifest.json or meta.json under {self.teacher_root()}; "
                "expected one per cache_<split>_<HxW> directory or root meta.json"
            )

        for subdir, manifest in manifests.items():
            where = f"{self.teacher}/{subdir}"
            got_hw = tuple(int(v) for v in manifest.get("target_bev_shape", ()))
            if got_hw not in want_hws:
                raise TeacherCacheMismatch(
                    f"{where}: cache grid {got_hw} not in allowable student/cache grids {want_hws}"
                )
            channels = int(manifest.get("bev_channels", -1))
            if channels != int(config.embed_dims):
                raise TeacherCacheMismatch(
                    f"{where}: cache has {channels} channels, student expects "
                    f"{int(config.embed_dims)}"
                )
            pcr = manifest.get("point_cloud_range")
            if pcr is not None:
                tx0, ty0, _, tx1, ty1, _ = (float(v) for v in pcr)
                got_lon, got_lat = (tx0, tx1), (-ty1, -ty0)
                if not _close(got_lon, want_lon) or not _close(got_lat, want_lat):
                    raise TeacherCacheMismatch(
                        f"{where}: cache covers longitudinal {got_lon} lateral "
                        f"{got_lat} m, student covers longitudinal {want_lon} "
                        f"lateral {want_lat} m. The grids have the same shape "
                        "but different extents, so their cells are not the "
                        "same places -- this is almost always a nuScenes cache "
                        "being fed to the NAVSIM student, or vice versa."
                    )
            transform = manifest.get("to_student_transform")
            if self.flip_w and transform is not None:
                if EXPECTED_TRANSFORM not in transform and "teacher_bev[:, :, ::-1]" not in transform:
                    raise TeacherCacheMismatch(
                        f"{where}: manifest records to_student_transform "
                        f"{transform!r}, this loader implements "
                        f"{EXPECTED_TRANSFORM!r}"
                    )

    # ------------------------------------------------------------------ #
    def has_token(self, token: str) -> bool:
        """True if :meth:`load_batch` would find this frame."""
        token = str(token)
        if self.is_sharded():
            return token in self._get_index()
        return os.path.isfile(self.path_for(token))

    def path_for(self, token: str) -> str:
        token = str(token)
        if self.is_sharded():
            idx = self._get_index()
            if token in idx:
                shard, _ = idx[token]
                return os.path.join(self.teacher_root(), "bev", f"{shard}.npy")
            return ""
        shard = token[:2]
        candidates = [
            os.path.join(self.teacher_root(), subdir, "samples", shard, token + ".npz")
            for subdir in self.cache_subdirs
        ]
        candidates.append(
            os.path.join(self.teacher_root(), "samples", shard, token + ".npz")
        )
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def _load_one(self, path: str) -> np.ndarray:
        with np.load(path) as data:
            if self.feature_key not in data:
                raise KeyError(
                    f"{path}: no {self.feature_key!r} array; found "
                    f"{sorted(data.keys())}"
                )
            feature = data[self.feature_key]
        if feature.ndim != 3:
            raise ValueError(
                f"{path}: {self.feature_key} must be CHW, got {feature.shape}"
            )
        if self.flip_w:
            # Negative strides are not accepted by torch.from_numpy.
            feature = feature[:, :, ::-1].copy()
        return feature

    def _load_sharded_one(self, token: str) -> np.ndarray:
        idx = self._get_index()
        if token not in idx:
            raise FileNotFoundError(f"{self.teacher}: token {token} not found in index.json")
        shard, row = idx[token]
        if shard not in self._mmap_cache:
            shard_path = os.path.join(self.teacher_root(), "bev", f"{shard}.npy")
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Missing shard file: {shard_path}")
            self._mmap_cache[shard] = np.load(shard_path, mmap_mode="r")
        raw = np.asarray(self._mmap_cache[shard][row])
        # In ReSMap, axes are (C, lateral, forward) = (256, 100, 50)
        # Transpose to (C, forward=50, lateral=100)
        if raw.ndim == 3 and raw.shape[1] > raw.shape[2]:
            raw = raw.transpose(0, 2, 1)
        if self.flip_w:
            raw = raw[:, :, ::-1].copy()
        return raw

    def load_batch(
        self, tokens: Sequence[str], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Stack the cached maps for ``tokens`` into ``[B, C, H, W]``."""
        features, missing = [], []
        is_sharded = self.is_sharded()
        for token in tokens:
            token_str = str(token)
            if is_sharded:
                try:
                    features.append(torch.from_numpy(self._load_sharded_one(token_str)))
                except FileNotFoundError:
                    missing.append(token_str)
            else:
                path = self.path_for(token_str)
                if not os.path.isfile(path):
                    missing.append(token_str)
                    continue
                features.append(torch.from_numpy(self._load_one(path)))
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{self.teacher} cache misses {len(missing)} sample(s), e.g. "
                f"{preview}. Cache both splits before training; a partially "
                "rsynced cache must not be trained on."
            )
        return torch.stack(features).to(device=device, dtype=dtype, non_blocking=True)


def _close(a: Tuple[float, float], b: Tuple[float, float], tol: float = 1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def restrict_dataset_to_stores(dataset, stores: Sequence[TeacherFeatureStore], name: str = "dataset") -> int:
    """Drop samples that any teacher cache is missing. Returns how many were dropped.

    ReSMap's published cache is navtrain ``train_logs`` only; NAVSIM val tokens
    are not in it. Filtering here is what lets stage 1/2 train instead of
    crashing on the first validation batch.
    """
    stores = tuple(stores)
    if not stores:
        return 0

    def cached(token: str) -> bool:
        return all(store.has_token(token) for store in stores)

    if hasattr(dataset, "_scene_loader"):
        frames = dataset._scene_loader.scene_frames_dicts
        before = len(frames)
        dataset._scene_loader.scene_frames_dicts = {
            token: scene for token, scene in frames.items() if cached(token)
        }
        dropped = before - len(dataset._scene_loader.scene_frames_dicts)
        remaining = len(dataset._scene_loader.scene_frames_dicts)
    elif hasattr(dataset, "tokens"):
        before_tokens = list(dataset.tokens)
        dataset.tokens = [token for token in before_tokens if cached(token)]
        dropped = len(before_tokens) - len(dataset.tokens)
        remaining = len(dataset.tokens)
        if hasattr(dataset, "_valid_cache_paths"):
            keep = set(dataset.tokens)
            dataset._valid_cache_paths = {
                token: path
                for token, path in dataset._valid_cache_paths.items()
                if token in keep
            }
    else:
        raise TypeError(f"cannot filter teacher-cache tokens on {type(dataset)!r}")

    logger.info(
        "%s: teacher cache kept %d sample(s), dropped %d missing from %s",
        name,
        remaining,
        dropped,
        ",".join(store.teacher for store in stores),
    )
    return dropped
