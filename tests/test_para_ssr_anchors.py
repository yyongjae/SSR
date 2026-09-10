"""Geometry, determinism and train/validation separation for anchor generation."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.build_para_ssr_anchors import (
    _sequence_sha256,
    cluster_trajectory_medoids,
    collect_train_trajectories,
    train_source_spec,
)


def _trajectories(count=20):
    times = np.arange(1, 9, dtype=np.float32) * 0.5
    speeds = np.linspace(1.0, 20.0, count, dtype=np.float32)
    poses = np.zeros((count, 8, 3), dtype=np.float32)
    poses[..., 0] = speeds[:, None] * times[None]
    poses[..., 1] = (speeds[:, None] - 10.0) * times[None] ** 2 / 20.0
    poses[..., 2] = np.arctan2(poses[..., 1], poses[..., 0])
    return poses


def _split():
    return {"train_logs": ["train_a", "train_b"], "val_logs": ["val"], "test_logs": ["test"]}


def _navtrain():
    return {
        "num_history_frames": 4,
        "num_future_frames": 10,
        "frame_interval": 1,
        "has_route": True,
        "max_scenes": 1,
        "log_names": ["train_a", "val", "train_b"],
        "tokens": ["token_b", "token_a", "token_val"],
    }


def test_medoids_are_real_full_trajectories_and_deterministic():
    trajectories = _trajectories()
    anchors, indices = cluster_trajectory_medoids(trajectories, num_candidates=4, seed=0)
    repeated, repeated_indices = cluster_trajectory_medoids(trajectories, num_candidates=4, seed=0)
    assert anchors.shape == (4, 8, 3)
    assert anchors.dtype == np.float32
    assert len(set(indices)) == 4
    np.testing.assert_allclose(anchors, trajectories[indices], atol=1e-7)
    np.testing.assert_array_equal(anchors, repeated)
    np.testing.assert_array_equal(indices, repeated_indices)
    # Future-only absolute positions: first sample at t=0.5, not t=0 and
    # not constant displacement increments for constant-speed trajectories.
    assert np.all(anchors[:, 0, 0] > 0)
    np.testing.assert_allclose(anchors[:, -1, 0], 8 * anchors[:, 0, 0])


def test_heading_near_pi_does_not_arithmetically_average_to_zero():
    poses = _trajectories(3)
    poses[:, :, :2] = poses[0, :, :2]
    poses[0, :, 2] = np.pi - 0.01
    poses[1, :, 2] = -np.pi + 0.01
    poses[2, :, 2] = np.pi - 0.02
    anchors, indices = cluster_trajectory_medoids(poses, num_candidates=1)
    assert np.all(np.abs(anchors[..., 2]) > 3.0)
    np.testing.assert_allclose(anchors[..., 2], poses[indices, :, 2], atol=1e-6)


@pytest.mark.parametrize(
    "poses,candidates,match",
    [
        (np.zeros((8, 3)), 1, "Expected trajectories"),
        (np.zeros((3, 9, 3)), 1, "Expected trajectories"),
        (np.full((3, 8, 3), np.nan), 1, "NaN or infinity"),
        (np.zeros((3, 8, 3)), 0, "num_candidates"),
        (np.zeros((3, 8, 3)), 4, "num_candidates"),
        (np.zeros((3, 8, 3)), 2, "distinct XY"),
    ],
)
def test_invalid_vocabularies_fail_loudly(poses, candidates, match):
    with pytest.raises(ValueError, match=match):
        cluster_trajectory_medoids(poses, candidates)


def test_train_spec_intersects_logs_and_preserves_official_tokens():
    spec = train_source_spec(_navtrain(), _split())
    assert spec["log_names"] == ["train_a", "train_b"]
    assert spec["tokens"] == ["token_a", "token_b", "token_val"]
    assert spec["max_scenes"] is None  # cap is applied after uniform hash sampling
    assert spec["num_history_frames"] == 4


@pytest.mark.parametrize("heldout_split", ["val_logs", "test_logs"])
def test_conflicting_training_split_is_rejected(heldout_split):
    split = _split()
    split[heldout_split].append("train_a")
    with pytest.raises(ValueError, match="overlap validation/test"):
        train_source_spec(_navtrain(), split)


def test_missing_heldout_split_definition_is_rejected():
    split = _split()
    del split["val_logs"]
    with pytest.raises(ValueError, match="explicit train_logs"):
        train_source_spec(_navtrain(), split)


@pytest.mark.parametrize("missing", ["tokens", "log_names"])
def test_missing_navtrain_allowlist_is_rejected(missing):
    navtrain = _navtrain()
    del navtrain[missing]
    with pytest.raises(ValueError, match="token allowlist"):
        train_source_spec(navtrain, _split())


def test_collection_never_opens_heldout_logs_and_samples_independently_of_order(tmp_path):
    log_path = tmp_path / "navsim_logs/trainval"
    log_path.mkdir(parents=True)
    for log_name in ["train_a", "val", "train_b", "test"]:
        (log_path / f"{log_name}.pkl").touch()
    opened = []
    reverse_tokens = False
    allowed = [f"token_{index}" for index in range(10)]
    spec = train_source_spec({**_navtrain(), "tokens": allowed}, _split())
    poses = _trajectories(10)

    class FakeLoader:
        def __init__(self, *, data_path, sensor_blobs_path, scene_filter, sensor_config):
            self.log_name = scene_filter.log_names[0]
            assert self.log_name not in {"val", "test"}
            opened.append(self.log_name)
            assert set(scene_filter.tokens) == set(allowed)
            assert scene_filter.max_scenes is None
            assert sensor_config.get_sensors_at_iteration(3) == []
            indices = range(0, 5) if self.log_name == "train_a" else range(5, 10)
            self.tokens = [f"token_{index}" for index in indices]
            if reverse_tokens:
                self.tokens.reverse()

        def get_scene_from_token(self, token):
            index = int(token.split("_")[-1])

            def future(num_trajectory_frames):
                assert num_trajectory_frames == 8
                return SimpleNamespace(
                    poses=poses[index], trajectory_sampling=SimpleNamespace(interval_length=0.5)
                )

            return SimpleNamespace(
                scene_metadata=SimpleNamespace(log_name=self.log_name, initial_token=token),
                get_future_trajectory=future,
            )

    actual, tokens, logs, count = collect_train_trajectories(
        tmp_path, spec, max_scenes=4, seed=42, loader_cls=FakeLoader
    )
    reverse_tokens = True
    again, repeated_tokens, repeated_logs, repeated_count = collect_train_trajectories(
        tmp_path, {**spec, "log_names": spec["log_names"][::-1]},
        max_scenes=4, seed=42, loader_cls=FakeLoader,
    )
    assert count == repeated_count == 10
    assert actual.shape == (4, 8, 3)
    assert tokens == repeated_tokens
    assert logs == repeated_logs
    assert set(opened) == {"train_a", "train_b"}
    np.testing.assert_array_equal(actual, again)
    for pose, token in zip(actual, tokens):
        np.testing.assert_array_equal(pose, poses[int(token.split("_")[-1])])


def test_navsim_future_pose_convention_excludes_origin_and_rotates_to_current_ego():
    from navsim.common.dataclasses import Scene

    # Ego is facing world +y. Forward movement in the world must become
    # positive local x; positive local y is left (world -x).
    origin = np.array([100.0, 200.0, np.pi / 2])
    frames = [SimpleNamespace(ego_status=SimpleNamespace(ego_pose=origin.copy())) for _ in range(4)]
    for step in range(1, 9):
        pose = origin + np.array([-0.1 * step, float(step), 0.02 * step])
        frames.append(SimpleNamespace(ego_status=SimpleNamespace(ego_pose=pose)))
    scene = Scene(
        scene_metadata=SimpleNamespace(num_history_frames=4, num_future_frames=8),
        map_api=None,
        frames=frames,
    )
    trajectory = scene.get_future_trajectory(num_trajectory_frames=8)
    expected = np.stack([np.arange(1, 9), 0.1 * np.arange(1, 9), 0.02 * np.arange(1, 9)], axis=-1)
    np.testing.assert_allclose(trajectory.poses, expected, atol=1e-6)
    assert trajectory.trajectory_sampling.interval_length == 0.5


@pytest.mark.parametrize("bad_field", ["token", "log_name", "initial_token", "interval", "poses"])
def test_collection_rejects_inconsistent_loaded_data(tmp_path, bad_field):
    log_path = tmp_path / "navsim_logs/trainval"
    log_path.mkdir(parents=True)
    (log_path / "train_a.pkl").touch()
    spec = train_source_spec(_navtrain(), _split())

    class BadLoader:
        def __init__(self, **kwargs):
            self.tokens = ["not_in_navtrain" if bad_field == "token" else "token_a"]

        def get_scene_from_token(self, token):
            return SimpleNamespace(
                scene_metadata=SimpleNamespace(
                    log_name="val" if bad_field == "log_name" else "train_a",
                    initial_token="wrong" if bad_field == "initial_token" else token,
                ),
                get_future_trajectory=lambda **kwargs: SimpleNamespace(
                    poses=np.full((8, 3), np.nan) if bad_field == "poses" else _trajectories(1)[0],
                    trajectory_sampling=SimpleNamespace(interval_length=1.0 if bad_field == "interval" else 0.5),
                ),
            )

    with pytest.raises(ValueError):
        collect_train_trajectories(tmp_path, spec, max_scenes=1, loader_cls=BadLoader)


def test_token_digest_is_order_independent_and_changes_with_membership():
    assert _sequence_sha256(["a", "b"]) == _sequence_sha256(["b", "a"])
    assert _sequence_sha256(["a", "b"]) != _sequence_sha256(["a", "c"])
