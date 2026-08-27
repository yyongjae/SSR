import numpy as np

from navsim.agents.para_ssr.para_ssr_features import (
    T_LIDAR_FROM_SSR,
    build_lidar2img,
)


def test_ssr_to_lidar_basis_uses_literal_navsim_convention():
    # SSR: x right, y forward. NAVSIM lidar: x forward, y left.
    ssr_basis = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],  # right
            [0.0, 1.0, 0.0, 1.0],  # forward
            [0.0, 0.0, 1.0, 1.0],  # up
        ]
    )
    expected_lidar = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],  # right -> -left
            [1.0, 0.0, 0.0, 1.0],   # forward -> forward
            [0.0, 0.0, 1.0, 1.0],   # up -> up
        ]
    )

    actual_lidar = (T_LIDAR_FROM_SSR @ ssr_basis.T).T
    np.testing.assert_array_equal(actual_lidar, expected_lidar)


def test_build_lidar2img_matches_direct_navsim_projection_for_nonidentity_pose():
    angle = 0.37
    c, s = np.cos(angle), np.sin(angle)
    sensor2lidar_rotation = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    sensor2lidar_translation = np.array([1.25, -0.4, 0.7])
    intrinsics = np.array(
        [[820.0, 0.0, 960.0], [0.0, 815.0, 540.0], [0.0, 0.0, 1.0]]
    )
    scale = (0.4, 0.35)
    crop_top = 17
    projection = build_lidar2img(
        intrinsics,
        sensor2lidar_rotation,
        sensor2lidar_translation,
        scale,
        crop_top,
    )

    ssr_points = np.array(
        [[2.0, 14.0, 1.5, 1.0], [-3.0, 20.0, 0.2, 1.0], [0.5, 8.0, 2.0, 1.0]]
    )
    matrix_homogeneous = (projection @ ssr_points.T).T

    # Independent literal convention and official inverse-extrinsic formula:
    # p_lidar=(y_forward, -x_right, z), p_camera=R.T @ (p_lidar-t).
    lidar_points = np.column_stack(
        (ssr_points[:, 1], -ssr_points[:, 0], ssr_points[:, 2])
    )
    camera_points = (
        sensor2lidar_rotation.T
        @ (lidar_points - sensor2lidar_translation).T
    ).T
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[0, :] *= scale[0]
    scaled_intrinsics[1, :] *= scale[1]
    scaled_intrinsics[1, 2] -= crop_top
    direct_homogeneous = (scaled_intrinsics @ camera_points.T).T

    np.testing.assert_allclose(matrix_homogeneous[:, :3], direct_homogeneous)
    np.testing.assert_allclose(
        matrix_homogeneous[:, :2] / matrix_homogeneous[:, 2:3],
        direct_homogeneous[:, :2] / direct_homogeneous[:, 2:3],
    )


def test_build_lidar2img_supports_exact_per_axis_resize_scale():
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]]
    )
    result = build_lidar2img(
        intrinsics=intrinsics,
        sensor2lidar_rotation=np.eye(3),
        sensor2lidar_translation=np.zeros(3),
        scale=(0.5, 0.25),
        crop_top=5,
    )

    scaled_intrinsics = np.array(
        [[50.0, 0.0, 25.0], [0.0, 50.0, 10.0], [0.0, 0.0, 1.0]]
    )
    view = np.eye(4)
    view[:3, :3] = scaled_intrinsics
    np.testing.assert_allclose(result, view @ T_LIDAR_FROM_SSR)
