"""Feeds the two cached teachers of v7 distillation (BEVFusion, ReSMap) into training as targets.

Added to the agent's target builders only when ``distill_v7`` is on.  Loading here (dataloader
workers) instead of in the loss, as the teammate does, keeps ~5 MB of disk reads per frame off
the training process.

    v7_teacher_bevfusion  [256, 50, 100] float16, student layout
    v7_teacher_resmap     [256, 50, 100] float16, student layout
    v7_teacher_valid      []  1.0 if BOTH caches have the frame, else 0.0 (zeros, masked in the loss)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from navsim.common.dataclasses import Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder

from ..cache_key import cache_key
from .teacher_store import v7_stores


class V7TeacherTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config):
        self._config = config
        if not getattr(config, "distill_v7_root", None):
            raise ValueError("distill_v7 needs distill_v7_root (teacher cache root)")
        self._stores = v7_stores(config)
        for s in self._stores:
            s.validate_manifest(config)

    def get_unique_name(self) -> str:
        cfg = self._config
        return cache_key("para_ssr_v7_teacher", (("root", str(cfg.distill_v7_root)),
                                                 ("resmap_flip", bool(getattr(cfg, "distill_v7_resmap_flip", False))),
                                                 ("bev", (cfg.bev_h, cfg.bev_w))))

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        cfg = self._config
        token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
        maps = [s.load(token) for s in self._stores]
        ok = all(m is not None for m in maps)
        zero = np.zeros((cfg.embed_dims, cfg.bev_h, cfg.bev_w), np.float16)
        maps = [m if ok else zero for m in maps]
        return {"v7_teacher_bevfusion": torch.from_numpy(maps[0]),
                "v7_teacher_resmap": torch.from_numpy(maps[1]),
                "v7_teacher_valid": torch.tensor(float(ok))}
