from types import SimpleNamespace

import numpy as np
import pytest

from navsim.agents.para_ssr.para_ssr_targets import (
    ParaSSRTargetBuilder,
    _equivalent_orders,
    detection_box_in_roi,
    navsim_box_to_ssr,
)


def test_equivalent_orders_preserve_open_and_closed_geometry() -> None:
    closed_ring = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [5.0, 1.0],
            [4.5, 3.0],
            [2.0, 4.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    closed_orders = _equivalent_orders(
        closed_ring, num_orders=20, closed=True, num_pts=20
    )

    assert closed_orders.shape == (20, 20, 2)
    # A cyclic shift must remove the duplicated endpoint, shift the unique
    # cycle, then close it again.  Rolling all 20 points would fail this for
    # every non-zero shift and introduce an interior jump.
    np.testing.assert_allclose(closed_orders[:, 0], closed_orders[:, -1])
    signed_areas = 0.5 * np.sum(
        closed_orders[:, :-1, 0] * closed_orders[:, 1:, 1]
        - closed_orders[:, 1:, 0] * closed_orders[:, :-1, 1],
        axis=1,
    )
    assert np.all(signed_areas > 0.0), "closed v2 orders must not be reversed"

    open_line = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 3.0]], dtype=np.float64)
    open_orders = _equivalent_orders(open_line, num_orders=7, closed=False, num_pts=20)

    np.testing.assert_allclose(open_orders[0], open_orders[1, ::-1])
    np.testing.assert_allclose(open_orders[0], open_orders[2])
    np.testing.assert_allclose(open_orders[1], open_orders[3])


def test_agent_targets_convert_conventions_and_keep_nearest_top_k() -> None:
    boxes = np.array(
        [
            # x_fwd, y_left, z_center, length, width, height, heading
            [20.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.1],
            [2.0, 0.0, 0.6, 5.0, 2.2, 1.7, -0.2],
            [3.0, 4.0, 0.7, 6.0, 2.4, 1.8, 1.0],
            [1.0, 0.0, 0.8, 7.0, 2.6, 1.9, 3.0],
        ],
        dtype=np.float32,
    )
    velocity = np.array(
        [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0], [7.0, 8.0, 0.0]],
        dtype=np.float32,
    )
    annotations = SimpleNamespace(
        boxes=boxes,
        velocity_3d=velocity,
        names=["vehicle"] * len(boxes),
        track_tokens=["far", "second", "middle", "nearest"],
    )
    scene = SimpleNamespace(
        frames=[
            SimpleNamespace(
                annotations=annotations,
                ego_status=SimpleNamespace(ego_pose=np.zeros(3, dtype=np.float64)),
            )
        ]
    )
    config = SimpleNamespace(
        max_agents=2,
        fut_ts=0,
        pc_range=(-30.0, 0.0, -2.0, 30.0, 30.0, 2.0),
        det_fov_half_angle_deg=80.0,
    )
    builder = ParaSSRTargetBuilder(config, trajectory_sampling=SimpleNamespace())

    targets = builder._compute_agent_targets(scene, cur_idx=0)
    actual = targets["gt_boxes"].numpy()[targets["gt_valid"].numpy()]

    # Nearest source indices are 3 then 1, regardless of annotation order.
    expected = np.array(
        [
            [
                0.0,
                1.0,
                0.8,
                2.6,
                7.0,
                1.9,
                np.arctan2(np.sin(-3.0 - np.pi), np.cos(-3.0 - np.pi)),
                -8.0,
                7.0,
            ],
            [
                0.0,
                2.0,
                0.6,
                2.2,
                5.0,
                1.7,
                np.arctan2(np.sin(0.2 - np.pi), np.cos(0.2 - np.pi)),
                -4.0,
                3.0,
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_detection_roi_is_front_only_and_clips_at_eighty_degrees() -> None:
    config = SimpleNamespace(
        pc_range=(-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        det_fov_half_angle_deg=80.0,
    )

    def converted(distance: float, bearing_deg: float):
        angle = np.deg2rad(bearing_deg)
        # NAVSIM x is forward and y is left.  Sign is immaterial for this
        # symmetric boundary test, but conversion itself is exercised too.
        nav_box = np.array(
            [distance * np.cos(angle), distance * np.sin(angle), 0.5,
             4.0, 2.0, 1.5, 0.0],
            dtype=np.float32,
        )
        return navsim_box_to_ssr(nav_box, np.zeros(3, dtype=np.float32))

    assert detection_box_in_roi(converted(10.0, 0.0), config)
    assert detection_box_in_roi(converted(10.0, 80.0), config)
    assert not detection_box_in_roi(converted(10.0, 80.1), config)
    assert not detection_box_in_roi(converted(10.0, -80.1), config)
    assert not detection_box_in_roi(converted(10.0, 180.0), config)


def test_future_track_is_reexpressed_through_moving_rotating_ego_frames() -> None:
    def annotations(x: float, y: float):
        return SimpleNamespace(
            boxes=np.array([[x, y, 0.5, 4.0, 2.0, 1.5, 0.0]], dtype=np.float32),
            track_tokens=["target"],
        )

    # Current ego is the global origin. The tracked agent moves globally
    # (1, 2) -> (2, 2) -> (2, 3). Future annotations are deliberately
    # expressed in two different, rotating future-ego frames:
    #   ego (1, 0, +pi/2): global (2, 2) -> local (2, -1)
    #   ego (1, 1, -pi/2): global (2, 3) -> local (-2, 1)
    ego_poses = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, np.pi / 2], dtype=np.float64),
        np.array([1.0, 1.0, -np.pi / 2], dtype=np.float64),
    ]
    scene = SimpleNamespace(
        frames=[
            SimpleNamespace(annotations=annotations(1.0, 2.0)),
            SimpleNamespace(annotations=annotations(2.0, -1.0)),
            SimpleNamespace(annotations=annotations(-2.0, 1.0)),
        ]
    )
    builder = ParaSSRTargetBuilder(SimpleNamespace(), SimpleNamespace())

    offsets, mask = builder._track_future(
        scene=scene,
        track_token="target",
        cur_idx=0,
        cur_pose=ego_poses[0],
        ego_poses=ego_poses,
        fut_ts=2,
    )

    # NAVSIM current-frame global positions become SSR positions
    # (-2, 1) -> (-2, 2) -> (-3, 2), hence these step offsets.
    np.testing.assert_allclose(
        offsets, np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32), atol=1e-6
    )
    np.testing.assert_array_equal(mask, np.ones(2, dtype=np.float32))


def test_map_query_failure_is_contextual_and_not_empty_gt(caplog) -> None:
    class BrokenMap:
        def get_proximal_map_objects(self, **kwargs):
            raise ValueError("schema mismatch")

    scene = SimpleNamespace(
        frames=[
            SimpleNamespace(
                token="token-123",
                ego_status=SimpleNamespace(ego_pose=np.zeros(3, dtype=np.float64)),
            )
        ],
        scene_metadata=SimpleNamespace(map_name="map-xyz"),
        map_api=BrokenMap(),
    )
    config = SimpleNamespace(
        map_max_vec=1,
        map_num_pts_per_vec=20,
        map_num_orders=20,
        pc_range=(-32.0, -32.0, -2.0, 32.0, 32.0, 2.0),
        map_pc_range=(-32.0, 0.0, -2.0, 32.0, 32.0, 2.0),
        map_min_length=1.0,
    )
    builder = ParaSSRTargetBuilder(config, trajectory_sampling=SimpleNamespace())

    with pytest.raises(RuntimeError, match="token-123.*map-xyz"):
        builder._compute_map_targets(scene, cur_idx=0)
    assert "schema mismatch" in caplog.text
