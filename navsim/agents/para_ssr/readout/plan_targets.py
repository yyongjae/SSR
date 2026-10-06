"""Per-token planning inputs/targets, extracted once so readouts train off-sensor.

A readout epoch over cached BEVs must not touch nuPlan logs.  This file holds,
for every token of a split, exactly what PARA-SSR's own builders produce:

    command   [4]      one-hot driving command         (features["command"])
    ego       [4]      vx, vy, ax, ay, NAVSIM ego axes  (status_feature[4:])
    offsets   [8, 3]   per-step (x, y, heading) deltas  (targets["trajectory_offsets"])
    mask      [8]

Built by ``tools/readout/build_plan_targets.py`` through ``ParaSSRFeatureBuilder``
and ``ParaSSRTargetBuilder`` code paths, so a readout sees the same numbers the
student does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

FIELDS = ("command", "ego", "offsets", "mask")


def plan_arrays_from_scene(scene, num_poses: int) -> Dict[str, np.ndarray]:
    """The four arrays for one scene, mirroring PARA-SSR's builders line by line."""
    trajectory = np.asarray(
        scene.get_future_trajectory(num_trajectory_frames=num_poses).poses, dtype=np.float32
    )
    offsets = np.diff(np.concatenate([np.zeros((1, 3), np.float32), trajectory]), axis=0)
    offsets[:, 2] = np.arctan2(np.sin(offsets[:, 2]), np.cos(offsets[:, 2]))
    status = scene.get_agent_input().ego_statuses[-1]
    return {
        "command": np.asarray(status.driving_command, dtype=np.float32),
        "ego": np.concatenate(
            [np.asarray(status.ego_velocity, np.float32), np.asarray(status.ego_acceleration, np.float32)]
        ),
        "offsets": offsets,
        "mask": np.ones(num_poses, dtype=np.float32),
    }


class PlanTargetStore:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.tokens: List[str] = [str(t) for t in data["tokens"]]
        self.logs: List[str] = [str(t) for t in data["logs"]]
        self.arrays = {k: data[k] for k in FIELDS}
        self.row = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def get(self, token: str) -> Dict[str, np.ndarray]:
        i = self.row[token]
        return {k: v[i] for k, v in self.arrays.items()}

    @staticmethod
    def save(path: Path, tokens, logs, rows: List[Dict[str, np.ndarray]]) -> None:
        arrays = {k: np.stack([r[k] for r in rows]) for k in FIELDS}
        np.savez(path, tokens=np.asarray(tokens), logs=np.asarray(logs), **arrays)


def log_holdout(logs: List[str], fraction: float, seed: int = 0) -> set:
    """Deterministic log-level hold-out (frames of one log never straddle splits)."""
    uniq = sorted(set(logs))
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(uniq) * fraction))) if fraction > 0 else 0
    return set(rng.choice(uniq, size=n, replace=False).tolist()) if n else set()


def load_log_split(name: str, repo_root: Optional[Path] = None) -> Dict[str, List[str]]:
    """``train_logs`` / ``val_logs`` from the navsim training split config."""
    import yaml

    root = repo_root or Path(__file__).resolve().parents[4]
    cfg = yaml.safe_load(
        (root / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml").read_text()
    )
    return {k: cfg[k] for k in ("train_logs", "val_logs") if k in cfg}
