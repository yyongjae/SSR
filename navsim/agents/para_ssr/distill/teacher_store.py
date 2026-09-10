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
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

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
        cache_subdirs: Sequence[str] = ("cache_train_100x100", "cache_val_100x100"),
        feature_key: str = BEV_FEATURE_KEY,
        flip_w: bool = True,
    ) -> None:
        self.root = os.path.abspath(os.path.expanduser(root))
        self.teacher = str(teacher)
        self.cache_subdirs = tuple(cache_subdirs)
        self.feature_key = str(feature_key)
        self.flip_w = bool(flip_w)
        self._manifests: Optional[Dict[str, dict]] = None

    # ------------------------------------------------------------------ #
    def teacher_root(self) -> str:
        return os.path.join(self.root, self.teacher)

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

    def validate_manifest(self, config) -> None:
        """Fail loudly when the cache and the student disagree.

        Checks the two things that silently corrupt a distillation run: the BEV
        grid the tensor was written on, and the physical extent that grid
        covers.  A shape-only check passes for a cache of the wrong dataset.
        """
        manifests = self.manifests()
        if not manifests:
            raise TeacherCacheMismatch(
                f"{self.teacher}: no manifest.json under {self.teacher_root()}; "
                "expected one per cache_<split>_<HxW> directory"
            )

        want_hw = (int(config.bev_h), int(config.bev_w))
        # Student pc_range is (x_min, y_min, z_min, x_max, y_max, z_max) with
        # x to the right and y forward.  The teacher records the mmdet3d LiDAR
        # frame, where x is forward and y is left, so the axes are swapped and
        # the lateral bounds are negated.
        sx0, sy0, _, sx1, sy1, _ = (float(v) for v in config.pc_range)
        want_lon = (sy0, sy1)
        want_lat = (sx0, sx1)

        for subdir, manifest in manifests.items():
            where = f"{self.teacher}/{subdir}"
            got_hw = tuple(int(v) for v in manifest.get("target_bev_shape", ()))
            if got_hw != want_hw:
                raise TeacherCacheMismatch(
                    f"{where}: cache grid {got_hw} != student ({want_hw[0]}, "
                    f"{want_hw[1]})"
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
            if self.flip_w and transform is not None and transform != EXPECTED_TRANSFORM:
                raise TeacherCacheMismatch(
                    f"{where}: manifest records to_student_transform "
                    f"{transform!r}, this loader implements "
                    f"{EXPECTED_TRANSFORM!r}"
                )

    # ------------------------------------------------------------------ #
    def path_for(self, token: str) -> str:
        token = str(token)
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

    def load_batch(
        self, tokens: Sequence[str], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Stack the cached maps for ``tokens`` into ``[B, C, H, W]``."""
        features, missing = [], []
        for token in tokens:
            path = self.path_for(token)
            if not os.path.isfile(path):
                missing.append(str(token))
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
