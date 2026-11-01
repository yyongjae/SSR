"""Token-addressed reader of the cached frozen-teacher BEV maps used by v7 distillation.

Ported (reduced) from origin/aux_distill ``distill/teacher_store.py``.  Two on-disk layouts under
``<root>/<teacher>``:

    per-token  (BEVFusion)  cache_{train,val}_50x100/samples/<token[:2]>/<token>.npz, key bev_feature,
                            (C, H = x_forward, W = y_left) -> student (C, y_forward, x_right):
                            ``student_bev = teacher_bev[:, :, ::-1]`` as the manifest records
    sharded    (ReSMap)     index.json token -> [shard, row], bev/<shard>.npy (C, lateral, forward)
                            -> student by a plain transpose, NO flip

The ReSMap orientation deviates from the teammate's loader, which transposes AND flips.  Measured
on this cache (tools/readout/verify_teacher_alignment.py, 60 navtrain frames, 2026-10-11): the
teacher's own segmentation logits correlate with PARA-SSR map GT at 0.370 under the plain
transpose and 0.054 under transpose + left/right flip -- the flip mirrors the map.  ``flip_w`` is
kept per teacher so the teammate's convention can still be reproduced.

The geometry is validated against the student config (grid, channels, metric extent) and a
mismatch refuses to train, as in the teammate's store.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

BEV_FEATURE_KEY = "bev_feature"
EXPECTED_TRANSFORM = "teacher_bev[:, :, ::-1]"


class TeacherCacheMismatch(RuntimeError):
    """The cache on disk does not describe the model being trained."""


class TeacherFeatureStore:
    def __init__(self, root: str, teacher: str, flip_w: bool,
                 cache_subdirs: Sequence[str] = ("cache_train_50x100", "cache_val_50x100"),
                 feature_key: str = BEV_FEATURE_KEY) -> None:
        self.root = os.path.abspath(os.path.expanduser(root))
        self.teacher = str(teacher)
        self.flip_w = bool(flip_w)
        self.cache_subdirs = tuple(cache_subdirs)
        self.feature_key = str(feature_key)
        self._index: Optional[Dict[str, list]] = None
        self._meta: Optional[dict] = None
        self._mmap: Dict[str, np.ndarray] = {}

    # pickled into dataloader workers: drop open memory maps
    def __getstate__(self):
        state = dict(self.__dict__)
        state["_mmap"] = {}
        return state

    def teacher_root(self) -> str:
        return os.path.join(self.root, self.teacher)

    def is_sharded(self) -> bool:
        return os.path.isfile(os.path.join(self.teacher_root(), "index.json"))

    def meta(self) -> dict:
        if self._meta is None:
            with open(os.path.join(self.teacher_root(), "meta.json"), "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta

    def index(self) -> Dict[str, list]:
        if self._index is None:
            with open(os.path.join(self.teacher_root(), "index.json"), "r", encoding="utf-8") as f:
                self._index = json.load(f)
        return self._index

    def manifests(self) -> Dict[str, dict]:
        out = {}
        for sub in self.cache_subdirs:
            p = os.path.join(self.teacher_root(), sub, "manifest.json")
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    out[sub] = json.load(f)
        return out

    # ------------------------------------------------------------------ #
    def validate_manifest(self, config) -> None:
        want_hw = (int(config.bev_h), int(config.bev_w))
        sx0, sy0, _, sx1, sy1, _ = (float(v) for v in config.pc_range)
        want_lon, want_lat = (sy0, sy1), (sx0, sx1)
        if self.is_sharded():
            meta = self.meta()
            shape = meta.get("tensors", {}).get("bev", {}).get("shape", [])
            if len(shape) != 3:
                raise TeacherCacheMismatch(f"{self.teacher}: meta.json has no tensors.bev.shape")
            c, a, b = (int(v) for v in shape)
            got_hw = (b, a) if a > b else (a, b)
            pcr = meta.get("pc_range")
            got = None if pcr is None else ((float(pcr[0]), float(pcr[3])), (float(pcr[1]), float(pcr[4])))
            where = f"{self.teacher} (sharded)"
        else:
            mans = self.manifests()
            if not mans:
                raise TeacherCacheMismatch(f"{self.teacher}: no manifest.json under {self.teacher_root()}")
            for sub, man in mans.items():
                where = f"{self.teacher}/{sub}"
                got_hw = tuple(int(v) for v in man.get("target_bev_shape", ()))
                c = int(man.get("bev_channels", -1))
                pcr = man.get("point_cloud_range")
                # teacher LiDAR frame: x forward, y left -> student lateral = (-y1, -y0)
                got = None if pcr is None else ((float(pcr[0]), float(pcr[3])), (-float(pcr[4]), -float(pcr[1])))
                t = man.get("to_student_transform")
                if self.flip_w and t is not None and EXPECTED_TRANSFORM not in t:
                    raise TeacherCacheMismatch(f"{where}: to_student_transform {t!r}, loader flips W")
                self._check(where, got_hw, c, got, want_hw, want_lon, want_lat, config)
            return
        self._check(where, got_hw, c, got, want_hw, want_lon, want_lat, config)

    @staticmethod
    def _check(where, got_hw, c, got, want_hw, want_lon, want_lat, config):
        if tuple(got_hw) != want_hw:
            raise TeacherCacheMismatch(f"{where}: cache grid {got_hw}, student {want_hw}")
        if c != int(config.embed_dims):
            raise TeacherCacheMismatch(f"{where}: cache has {c} channels, student {int(config.embed_dims)}")
        if got is not None:
            close = lambda a, b: all(abs(x - y) <= 1e-6 for x, y in zip(a, b))  # noqa: E731
            if not close(got[0], want_lon) or not close(got[1], want_lat):
                raise TeacherCacheMismatch(f"{where}: cache covers lon {got[0]} lat {got[1]} m, "
                                           f"student lon {want_lon} lat {want_lat} m")

    # ------------------------------------------------------------------ #
    def path_for(self, token: str) -> str:
        for sub in self.cache_subdirs:
            p = os.path.join(self.teacher_root(), sub, "samples", token[:2], token + ".npz")
            if os.path.isfile(p):
                return p
        return ""

    def has_token(self, token: str) -> bool:
        token = str(token)
        return token in self.index() if self.is_sharded() else bool(self.path_for(token))

    def load(self, token: str) -> Optional[np.ndarray]:
        """(C, bev_h, bev_w) float16 in the student layout, or None if the cache lacks the frame."""
        token = str(token)
        if self.is_sharded():
            hit = self.index().get(token)
            if hit is None:
                return None
            shard, row = hit
            if shard not in self._mmap:
                self._mmap[shard] = np.load(os.path.join(self.teacher_root(), "bev", f"{shard}.npy"), mmap_mode="r")
            raw = np.asarray(self._mmap[shard][row])
            if raw.shape[1] > raw.shape[2]:                    # (C, lateral, forward) -> (C, forward, lateral)
                raw = raw.transpose(0, 2, 1)
        else:
            path = self.path_for(token)
            if not path:
                return None
            with np.load(path) as data:
                raw = data[self.feature_key]
        if raw.ndim != 3:
            raise ValueError(f"{self.teacher}/{token}: bev must be CHW, got {raw.shape}")
        if self.flip_w:
            raw = raw[:, :, ::-1]
        return np.ascontiguousarray(raw, dtype=np.float16)


def v7_stores(config) -> Tuple[TeacherFeatureStore, TeacherFeatureStore]:
    root = str(config.distill_v7_root)
    return (TeacherFeatureStore(root, "bevfusion", flip_w=True),
            TeacherFeatureStore(root, "resmap", flip_w=bool(getattr(config, "distill_v7_resmap_flip", False))))
