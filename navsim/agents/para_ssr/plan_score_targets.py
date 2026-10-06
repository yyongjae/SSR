"""``sim_reward`` target for the v2 anchor planner: PDM scores of the 256 anchors.

WoTE's ``WoTE_targets.py`` reads ``formatted_pdm_score_256.npy`` (token ->
per-anchor NC / DAC / EP / TTC / comfort from the PDM scorer).  Same labels here,
from the packed memory-mapped copy written by ``tools/plan_v2/pack_pdm_scores.py``
(``<plan_score_file>.npy`` + ``.tokens.json``), so a dataloader worker does not
unpickle 1.5 GB.  Tokens without labels (navtest) get ``sim_reward_valid = 0``.
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np
import torch

from navsim.agents.para_ssr.cache_key import cache_key
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


class AnchorScoreTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config):
        if not config.plan_score_file:
            raise ValueError("plan_anchor=true needs plan_score_file (packed PDM scores, without the .npy suffix)")
        self._path = str(config.plan_score_file)
        self._scores = None
        self._index = None

    def get_unique_name(self) -> str:
        return cache_key("para_ssr_anchor_pdm_score", (("file", self._path.rsplit("/", 1)[-1]),))

    def _open(self):
        if self._scores is None:  # lazily: the builder is pickled into workers
            self._scores = np.load(self._path + ".npy", mmap_mode="r")
            self._index = {t: i for i, t in enumerate(json.load(open(self._path + ".tokens.json")))}

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        self._open()
        token = scene.frames[scene.scene_metadata.num_history_frames - 1].token
        i = self._index.get(token)
        if i is None:
            return {"sim_reward": torch.zeros(self._scores.shape[1:], dtype=torch.float32),
                    "sim_reward_valid": torch.tensor(0.0)}
        return {"sim_reward": torch.from_numpy(np.asarray(self._scores[i], dtype=np.float32)),
                "sim_reward_valid": torch.tensor(1.0)}
