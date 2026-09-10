"""Training-only labels from the NAVSIM v1 PDM evaluation protocol.

Candidate poses are in NAVSIM's ego frame: x forward, y left, heading CCW.
World caches contain privileged future annotations; this class must only be
called while computing training/validation losses, never by inference.

Every candidate is evaluated together with the cached PDM reference trajectory,
exactly as ``navsim.evaluate.pdm_score.pdm_score`` does. Scoring all candidates in
one call would change progress normalization and therefore change the label of
an unchanged trajectory when a different candidate is added to the set.
"""

from collections import OrderedDict
import csv
import lzma
import math
from pathlib import Path
import pickle
from typing import Any, Dict, Sequence, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


METRIC_NAMES = ("NC", "DAC", "DDC", "EP", "TTC", "COMFORT", "SCORE")
_RESULT_FIELDS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "score",
)


class CandidateMetricSupervisor:
    """Lazily load NAVSIM world caches and label current predicted candidates.

    The returned tensor is float32 on the candidates' device and has no autograd
    history. The fixed 4-second, 10-Hz rollout and scorer parameters match this
    repository's ``pdm_scoring/default_scoring_parameters.yaml``. The prediction
    sampling can be different from 10 Hz but must span exactly four seconds.

    Instances are local to a training process and are not thread-safe: the PDM
    simulator/scorer maintain mutable working arrays. No Ray workers or GPU
    resources are created. Cache files must be trusted NAVSIM-generated pickles.
    """

    def __init__(
        self,
        metric_cache_path: str,
        trajectory_sampling: "TrajectorySampling",
        cache_size: int = 8,
    ) -> None:
        if not str(metric_cache_path or "").strip():
            raise ValueError("metric_cache_path must name a NAVSIM metric world cache")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        horizon = float(trajectory_sampling.time_horizon)
        interval = float(trajectory_sampling.interval_length)
        count = int(trajectory_sampling.num_poses)
        if (
            not math.isfinite(horizon)
            or not math.isfinite(interval)
            or count < 1
            or interval <= 0.0
            or not math.isclose(horizon, 4.0, rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(count * interval, horizon, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError("NAVSIM v1 metric supervision requires a consistent 4-second trajectory sampling")

        self.metric_cache_path = Path(metric_cache_path).expanduser()
        self.trajectory_sampling = trajectory_sampling
        self.cache_size = cache_size
        self._paths: Dict[str, Path] | None = None
        self._world_cache: OrderedDict[str, Any] = OrderedDict()
        self._pdm_score = None
        self._simulator = None
        self._scorer = None
        self._proposal_sampling = None
        self._trajectory_type = None

    def _resolve_path(self, recorded_path: str) -> Path:
        recorded = Path(recorded_path).expanduser()
        # Cache roots are commonly moved or symlinked after generation. Prefer
        # a file under the requested root to stale absolute paths in metadata.
        suffix = Path(*recorded.parts[-4:])
        choices = [self.metric_cache_path / suffix]
        if not recorded.is_absolute():
            choices.extend((self.metric_cache_path / recorded, self.metric_cache_path.parent / recorded))
        choices.append(recorded)
        for candidate in choices:
            if candidate.is_file():
                return candidate.resolve()
        # Keep an actionable path; existence is checked when tokens are used.
        return choices[0]

    def _ensure_index(self) -> None:
        if self._paths is not None:
            return
        metadata_dir = self.metric_cache_path / "metadata"
        metadata_files = sorted(metadata_dir.glob("*.csv"))
        if not metadata_files:
            raise FileNotFoundError(
                f"No NAVSIM metric cache metadata/*.csv found under {self.metric_cache_path}. "
                "Generate metric world caches for the training/validation split before enabling metric supervision."
            )
        paths: Dict[str, Path] = {}
        for metadata_path in metadata_files:
            with metadata_path.open(newline="") as source:
                reader = csv.DictReader(source)
                if not reader.fieldnames or "file_name" not in reader.fieldnames:
                    raise ValueError(f"Metric cache metadata lacks the file_name column: {metadata_path}")
                for row in reader:
                    recorded_path = str(row.get("file_name") or "").strip()
                    if not recorded_path:
                        continue
                    recorded = Path(recorded_path)
                    if recorded.name != "metric_cache.pkl" or recorded.parent.name in ("", ".", ".."):
                        raise ValueError(f"Malformed metric cache path in {metadata_path}: {recorded_path}")
                    token = recorded.parent.name
                    path = self._resolve_path(recorded_path)
                    if token in paths and paths[token] != path:
                        raise ValueError(f"Conflicting metric cache entries for token {token}: {paths[token]} and {path}")
                    paths[token] = path
        if not paths:
            raise ValueError(f"Metric cache metadata contains no token entries: {metadata_dir}")
        self._paths = paths

    @property
    def tokens(self) -> list[str]:
        """Available indexed tokens; reading this does not deserialize worlds."""
        self._ensure_index()
        return list(self._paths)

    def validate_tokens(self, tokens: Sequence[str]) -> None:
        """Check split coverage and files without deserializing any world cache."""
        if isinstance(tokens, (str, bytes)):
            raise TypeError("tokens must be a sequence of scene-frame token strings")
        self._ensure_index()
        missing = []
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise ValueError("Each metric cache token must be a non-empty string")
            if token not in self._paths:
                missing.append(token)
            elif not self._paths[token].is_file():
                raise FileNotFoundError(f"Metric cache file missing for token {token}: {self._paths[token]}")
        if missing:
            examples = ", ".join(missing[:5])
            raise KeyError(
                f"Metric cache missing {len(missing)} requested token(s), including {examples}. "
                "Use world caches for the same split; a navtest-only cache cannot supervise navtrain."
            )

    def _get_world(self, token: str) -> Any:
        if token in self._world_cache:
            self._world_cache.move_to_end(token)
            return self._world_cache[token]
        path = self._paths[token]
        try:
            with lzma.open(path, "rb") as source:
                world = pickle.load(source)
        except Exception as error:
            raise RuntimeError(f"Cannot load metric world cache for token {token}: {path}") from error
        required = ("trajectory", "ego_state", "observation", "centerline", "route_lane_ids", "drivable_area_map")
        if any(not hasattr(world, field) for field in required):
            raise ValueError(f"Invalid NAVSIM metric world cache for token {token}: {path}")
        if self.cache_size:
            self._world_cache[token] = world
            while len(self._world_cache) > self.cache_size:
                self._world_cache.popitem(last=False)
        return world

    def _ensure_scorer(self) -> None:
        if self._pdm_score is not None:
            return
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

        self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self._simulator = PDMSimulator(self._proposal_sampling)
        # The class default is 0.1 m; official evaluation YAML uses 5.0 m.
        # Using the class default silently changes low-speed progress labels.
        scorer_config = PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            comfortable_weight=2.0,
            driving_direction_horizon=1.0,
            driving_direction_compliance_threshold=2.0,
            driving_direction_violation_threshold=6.0,
            stopped_speed_threshold=5e-3,
            progress_distance_threshold=5.0,
        )
        self._scorer = PDMScorer(self._proposal_sampling, config=scorer_config)
        self._trajectory_type = Trajectory
        self._pdm_score = pdm_score

    @torch.no_grad()
    def score(self, tokens: Sequence[str], candidates: torch.Tensor) -> torch.Tensor:
        """Return ``[B, K, 7]`` labels for current ``[B, K, T, 3]`` candidates.

        Timesteps start at the first future instant, with no initial ego pose in
        the supplied tensor. All candidate coordinates must be finite. Errors
        include the token/candidate so failures cannot silently yield fake GT.
        """
        if not isinstance(candidates, torch.Tensor) or not candidates.is_floating_point():
            raise TypeError("candidates must be a floating-point torch tensor")
        expected_steps = int(self.trajectory_sampling.num_poses)
        if (
            candidates.ndim != 4
            or candidates.shape[0] < 1
            or candidates.shape[1] < 1
            or candidates.shape[2:] != (expected_steps, 3)
        ):
            raise ValueError(f"candidates must have shape [B, K, {expected_steps}, 3] with B,K > 0")
        if isinstance(tokens, (str, bytes)) or len(tokens) != candidates.shape[0]:
            raise ValueError("One scene-frame token is required for each candidate batch item")
        poses = candidates.detach().to(device="cpu", dtype=torch.float64).numpy()
        if not np.isfinite(poses).all():
            raise ValueError("Candidate trajectory poses contain NaN or infinity")
        self.validate_tokens(tokens)
        self._ensure_scorer()

        labels = np.empty((*poses.shape[:2], len(METRIC_NAMES)), dtype=np.float32)
        for batch_index, token in enumerate(tokens):
            world = self._get_world(token)
            for candidate_index, candidate in enumerate(poses[batch_index]):
                trajectory = self._trajectory_type(candidate, self.trajectory_sampling)
                try:
                    result = self._pdm_score(
                        world, trajectory, self._proposal_sampling, self._simulator, self._scorer
                    )
                    metric_values = np.asarray([getattr(result, field) for field in _RESULT_FIELDS], dtype=np.float64)
                    if metric_values.shape != (len(METRIC_NAMES),) or not np.isfinite(metric_values).all():
                        raise ValueError("PDM evaluator returned malformed or non-finite labels")
                    if np.any(metric_values < -1e-6) or np.any(metric_values > 1.0 + 1e-6):
                        raise ValueError(f"PDM labels outside [0,1]: {metric_values}")
                    labels[batch_index, candidate_index] = np.clip(metric_values, 0.0, 1.0)
                except Exception as error:
                    raise RuntimeError(f"PDM metric labeling failed for token {token}, candidate {candidate_index}") from error
        return torch.from_numpy(labels).to(device=candidates.device)
