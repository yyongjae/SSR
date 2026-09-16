"""Feeds the cached ReSMap teacher into PARA-SSR training as extra targets.

Added to the agent's target builders only when ``kd_mode != "none"`` or
``map_label_source == "teacher"``, so the default model is untouched.

    teacher_bev        [256, 50, 100] float16, student layout  (kd_mode != none)
    teacher_valid      []  1.0 if the token is in the cache, else 0.0
    teacher_map_pts    [max_vec, orders, pts, 2]  \
    teacher_map_labels [max_vec]                   } map_label_source == "teacher"
    teacher_map_valid  [max_vec]                  /

A token missing from the cache yields zeros and ``teacher_valid = 0``; the loss
masks it out (distillation) or falls back to GT (pseudo labels) instead of
failing a long run on one frame.
"""
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
import torch

from navsim.common.dataclasses import Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder

from ..cache_key import cache_key
from .bev_cache import BevCache, pseudo_map_targets


class ResMapTeacherTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config):
        self._config = config
        self._need_bev = config.kd_mode != "none"
        self._need_map = config.map_label_source == "teacher"
        if not config.kd_teacher_cache:
            raise ValueError("kd_teacher_cache must point at the ReSMap cache root")
        self._root = str(config.kd_teacher_cache)
        self._cache = None  # opened lazily: the builder is pickled into workers

    @property
    def cache(self) -> BevCache:
        if self._cache is None:
            self._cache = BevCache(self._root)
        return self._cache

    def get_unique_name(self) -> str:
        cfg = self._config
        digest = self.cache.meta.get("checkpoint_sha256", self._root)
        return cache_key(
            "para_ssr_resmap_teacher",
            (
                ("teacher", hashlib.sha1(str(digest).encode()).hexdigest()[:12]),
                ("bev", self._need_bev),
                ("map", self._need_map),
                ("score_thr", cfg.map_pseudo_score_thr),
                ("map_max_vec", cfg.map_max_vec),
                ("map_num_orders", cfg.map_num_orders),
                ("map_num_pts_per_vec", cfg.map_num_pts_per_vec),
                ("pc_range", cfg.pc_range),
            ),
        )

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        cfg = self._config
        cur = scene.scene_metadata.num_history_frames - 1
        token = scene.frames[cur].token
        hit = token in self.cache
        out: Dict[str, torch.Tensor] = {"teacher_valid": torch.tensor(float(hit))}
        if self._need_bev:
            if hit:
                bev = self.cache.bev(token)
            else:
                bev = np.zeros((cfg.embed_dims, cfg.bev_h, cfg.bev_w), np.float16)
            out["teacher_bev"] = torch.from_numpy(bev)
        if self._need_map:
            shape = (cfg.map_max_vec, cfg.map_num_orders, cfg.map_num_pts_per_vec, 2)
            if hit:
                t = pseudo_map_targets(
                    self.cache, token,
                    score_thr=cfg.map_pseudo_score_thr,
                    max_vec=cfg.map_max_vec,
                    num_orders=cfg.map_num_orders,
                    num_pts=cfg.map_num_pts_per_vec,
                    pc_range=cfg.pc_range,
                )
            else:
                t = {
                    "gt_map_pts": np.zeros(shape, np.float32),
                    "gt_map_labels": np.zeros(cfg.map_max_vec, np.int64),
                    "gt_map_valid": np.zeros(cfg.map_max_vec, bool),
                }
            out["teacher_map_pts"] = torch.from_numpy(t["gt_map_pts"])
            out["teacher_map_labels"] = torch.from_numpy(t["gt_map_labels"])
            out["teacher_map_valid"] = torch.from_numpy(t["gt_map_valid"])
        return out
