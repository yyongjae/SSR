"""Metric supervision contracts, plus opt-in real NAVSIM evaluator parity.

Run the real-cache test with PARA_SSR_METRIC_TEST_CACHE=/path/to/metric_cache.
The default tests create tiny synthetic cache metadata and do not need data.
"""

import csv
import lzma
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.para_ssr.metric_supervision import CandidateMetricSupervisor, METRIC_NAMES


def _sampling():
    return TrajectorySampling(time_horizon=4.0, interval_length=0.5)


def _write_cache(root, token, *, shard=0, recorded_root=None, world=None):
    path = root / "a_log" / "unknown" / token / "metric_cache.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if world is None:
        world = SimpleNamespace(
            trajectory=None, ego_state=None, observation=None,
            centerline=None, route_lane_ids=[], drivable_area_map=None,
        )
    with lzma.open(path, "wb") as target:
        pickle.dump(world, target)
    metadata = root / "metadata" / f"metric_cache_metadata_node_{shard}.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    create_header = not metadata.exists()
    with metadata.open("a", newline="") as target:
        writer = csv.writer(target)
        if create_header:
            writer.writerow(["file_name"])
        recorded = path if recorded_root is None else Path(recorded_root) / path.relative_to(root)
        writer.writerow([str(recorded)])
    return path


def test_supervisor_does_not_load_worlds_or_import_scorer_on_construction(tmp_path):
    supervisor = CandidateMetricSupervisor(str(tmp_path / "not_created"), _sampling())
    assert supervisor._paths is None
    assert supervisor._scorer is None
    assert not supervisor._world_cache
    with pytest.raises(FileNotFoundError, match="metadata"):
        supervisor.score(["a"], torch.zeros(1, 1, 8, 3))


@pytest.mark.parametrize("cache_size", [-1, 1.5, True])
def test_invalid_cache_size_fails_at_construction(tmp_path, cache_size):
    with pytest.raises(ValueError, match="cache_size"):
        CandidateMetricSupervisor(str(tmp_path), _sampling(), cache_size=cache_size)


def test_four_second_protocol_and_nonempty_path_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="metric_cache_path"):
        CandidateMetricSupervisor("", _sampling())
    with pytest.raises(ValueError, match="4-second"):
        CandidateMetricSupervisor(str(tmp_path), TrajectorySampling(time_horizon=3, interval_length=0.5))


@pytest.mark.parametrize(
    "tokens,candidates,error,match",
    [
        (["a"], torch.zeros(1, 8, 3), ValueError, "shape"),
        (["a"], torch.zeros(1, 1, 9, 3), ValueError, "shape"),
        (["a"], torch.zeros(1, 0, 8, 3), ValueError, "shape"),
        ([], torch.zeros(1, 1, 8, 3), ValueError, "token"),
        ("a", torch.zeros(1, 1, 8, 3), ValueError, "token"),
        (["a"], torch.zeros(1, 1, 8, 3, dtype=torch.long), TypeError, "floating"),
        (["a"], torch.full((1, 1, 8, 3), float("nan")), ValueError, "NaN"),
        (["a"], torch.full((1, 1, 8, 3), float("inf")), ValueError, "infinity"),
    ],
)
def test_invalid_candidates_fail_before_cache_io(tmp_path, tokens, candidates, error, match):
    supervisor = CandidateMetricSupervisor(str(tmp_path), _sampling())
    with pytest.raises(error, match=match):
        supervisor.score(tokens, candidates)
    assert supervisor._paths is None


def test_all_metadata_shards_and_relocated_root_are_respected(tmp_path):
    _write_cache(tmp_path, "a", shard=0, recorded_root="/old/machine/cache")
    _write_cache(tmp_path, "b", shard=1)
    supervisor = CandidateMetricSupervisor(str(tmp_path), _sampling())
    assert set(supervisor.tokens) == {"a", "b"}
    supervisor.validate_tokens(["b", "a", "a"])
    assert not supervisor._world_cache
    assert supervisor._paths["a"].is_relative_to(tmp_path)
    with pytest.raises(KeyError, match="navtest-only cache cannot supervise navtrain"):
        supervisor.validate_tokens(["missing_train_token"])
    supervisor._paths["a"].unlink()
    with pytest.raises(FileNotFoundError, match="token a"):
        supervisor.validate_tokens(["a"])


def test_bad_metadata_and_duplicate_tokens_fail_clearly(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    first = metadata / "one.csv"
    first.write_text("wrong_column\nentry\n")
    with pytest.raises(ValueError, match="file_name column"):
        CandidateMetricSupervisor(str(tmp_path), _sampling()).validate_tokens(["a"])
    first.write_text("file_name\n/log_a/unknown/a/metric_cache.pkl\n")
    (metadata / "two.csv").write_text("file_name\n/log_b/unknown/a/metric_cache.pkl\n")
    with pytest.raises(ValueError, match="Conflicting"):
        CandidateMetricSupervisor(str(tmp_path), _sampling()).validate_tokens(["a"])


def test_bounded_world_lru_and_corrupt_cache_errors(tmp_path):
    a_path = _write_cache(tmp_path, "a")
    _write_cache(tmp_path, "b")
    supervisor = CandidateMetricSupervisor(str(tmp_path), _sampling(), cache_size=1)
    supervisor.validate_tokens(["a", "b"])
    first_a = supervisor._get_world("a")
    assert supervisor._get_world("a") is first_a
    supervisor._get_world("b")
    assert list(supervisor._world_cache) == ["b"]
    assert supervisor._get_world("a") is not first_a
    supervisor._world_cache.clear()
    a_path.write_bytes(b"corrupt cache")
    with pytest.raises(RuntimeError, match="token a"):
        supervisor._get_world("a")


def test_label_order_device_and_candidate_detach(tmp_path):
    _write_cache(tmp_path, "a")
    supervisor = CandidateMetricSupervisor(str(tmp_path), _sampling())
    supervisor._trajectory_type = lambda poses, sampling: SimpleNamespace(poses=poses, sampling=sampling)
    calls = []

    def fake_pdm_score(world, trajectory, sampling, simulator, scorer):
        calls.append(trajectory)
        values = trajectory.poses[-1, 0] / 100 + np.arange(7) / 10
        return SimpleNamespace(**dict(zip(
            ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
             "ego_progress", "time_to_collision_within_bound", "comfort", "score"), values
        )))

    supervisor._pdm_score = fake_pdm_score
    candidates = torch.zeros(1, 2, 8, 3, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        candidates[0, 1, -1, 0] = 5.0
    labels = supervisor.score(["a"], candidates)
    assert METRIC_NAMES == ("NC", "DAC", "DDC", "EP", "TTC", "COMFORT", "SCORE")
    assert labels.shape == (1, 2, 7)
    assert labels.dtype == torch.float32 and labels.device == candidates.device
    assert not labels.requires_grad and labels.grad_fn is None
    assert len(calls) == 2
    assert calls[1].poses.dtype == np.float64
    np.testing.assert_allclose(labels.numpy()[0, 0], np.arange(7) / 10, atol=1e-7)
    np.testing.assert_allclose(labels.numpy()[0, 1], np.arange(7) / 10 + 0.05, atol=1e-7)


@pytest.mark.parametrize("value", [float("nan"), -0.01, 1.01])
def test_invalid_simulator_output_never_becomes_training_gt(tmp_path, value):
    _write_cache(tmp_path, "a")
    supervisor = CandidateMetricSupervisor(str(tmp_path), _sampling())
    supervisor._trajectory_type = lambda poses, sampling: poses
    supervisor._pdm_score = lambda *args: SimpleNamespace(
        no_at_fault_collisions=value, drivable_area_compliance=1.0,
        driving_direction_compliance=1.0, ego_progress=1.0,
        time_to_collision_within_bound=1.0, comfort=1.0, score=1.0,
    )
    with pytest.raises(RuntimeError, match="token a, candidate 0"):
        supervisor.score(["a"], torch.zeros(1, 1, 8, 3))


def test_supervisor_scoring_settings_match_official_evaluation_yaml(tmp_path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    supervisor = CandidateMetricSupervisor(str(tmp_path), _sampling())
    supervisor._ensure_scorer()
    yaml_path = Path(__file__).resolve().parents[1] / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
    settings = OmegaConf.load(yaml_path)
    official = instantiate(settings.scorer)
    assert supervisor._proposal_sampling == instantiate(settings.proposal_sampling)
    assert vars(supervisor._scorer._config) == vars(official._config)


@pytest.mark.skipif(not os.environ.get("PARA_SSR_METRIC_TEST_CACHE"), reason="set PARA_SSR_METRIC_TEST_CACHE for real cache parity")
def test_real_cache_matches_official_score_and_is_candidate_set_invariant():
    from dataclasses import astuple
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array

    supervisor = CandidateMetricSupervisor(os.environ["PARA_SSR_METRIC_TEST_CACHE"], _sampling())
    token = supervisor.tokens[0]
    supervisor.validate_tokens([token])
    world = supervisor._get_world(token)
    states = get_trajectory_as_array(world.trajectory, _sampling(), world.ego_state.time_point)
    reference = convert_absolute_to_relative_se2_array(world.ego_state.rear_axle, states[1:, :3])
    candidates = np.stack((reference, np.zeros_like(reference), reference.copy()))
    candidates[2, :, 1] += np.linspace(0.0, 4.0, len(reference))
    candidates = torch.tensor(candidates, dtype=torch.float64).unsqueeze(0).requires_grad_()
    actual = supervisor.score([token], candidates)

    yaml_path = Path(__file__).resolve().parents[1] / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
    settings = OmegaConf.load(yaml_path)
    scorer, simulator = instantiate(settings.scorer), instantiate(settings.simulator)
    expected = []
    for poses in candidates.detach().numpy()[0]:
        result = pdm_score(world, Trajectory(poses, _sampling()), instantiate(settings.proposal_sampling), simulator, scorer)
        expected.append(astuple(result))
    np.testing.assert_allclose(actual.numpy()[0], expected, atol=1e-6, rtol=0.0)
    alone = supervisor.score([token], candidates[:, :1])
    permuted = supervisor.score([token], candidates[:, [2, 0, 1]])
    torch.testing.assert_close(alone[:, 0], actual[:, 0], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(permuted[:, [1, 2, 0]], actual, atol=1e-6, rtol=0.0)
    assert not actual.requires_grad
