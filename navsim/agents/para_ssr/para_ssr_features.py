"""Feature builder: navsim ``AgentInput`` -> the tensors BEVFormer needs.

Produces, per sample:

``camera_feature``  ``[T, N_cam, 3, H, W]``   normalised selected camera images
``lidar2img``       ``[N_cam, 4, 4]``         SSR-BEV frame -> image pixels
``image_hw``        ``[N_cam, 2]``            (H, W) after resize/crop
``bev_shift``       ``[T, 2]``                normalised ego motion per step
``ego_motion``      ``[T, ego_motion_dims]``  legacy vector; relative yaw at [:,2] aligns BEV
``command``         ``[4]``                   driving command one-hot
``status_feature``  ``[8]``                   command + current planner velocity/acceleration
``lidar_points``    ``[T, N_max, 5]``         (use_lidar) front-ROI points in SSR
                                              axes, zero-padded
``lidar_num_points`` ``[T]``                  (use_lidar) real rows per frame

Coordinate frames
-----------------
SSR's BEV is VAD's ego frame: **x lateral (+right), y longitudinal (+forward)**.
navsim/nuPlan lidar is **x forward, y left**.  ``T_LIDAR_FROM_SSR`` below folds
that rotation into ``lidar2img`` so the model itself never sees the difference.
LiDAR points have no matrix to hide behind, so ``_get_lidar`` rotates them into
SSR axes explicitly: ``x_right = -y_left``, ``y_forward = x_forward``.

Ego motion
----------
There is no CAN bus.  navsim already expresses history ego poses in the
*current* frame (``AgentInput.from_scene_dict_list`` calls
``convert_absolute_to_relative_se2_array``), so the BEV shift between two
consecutive frames is read straight off those poses -- no angle round-trip.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import torch

from navsim.agents.abstract_agent import AbstractAgent  # noqa: F401  (typing aid)
from navsim.common.dataclasses import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder

from .cache_key import cache_key

# SSR-BEV (x right, y forward, z up) -> nuPlan lidar (x forward, y left, z up)
T_LIDAR_FROM_SSR = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Per-point layout of ``lidar_points``; part of the feature cache identity.
# NAVSIM's merged cloud is (x, y, z, intensity, ring, lidar_id) per LidarIndex;
# the sensor id is dropped, intensity (0..255) and ring (0..39 on this rig)
# are scaled to roughly unit range for the pillar MLP.
LIDAR_POINT_LAYOUT: Tuple[str, ...] = ("x_right", "y_forward", "z", "intensity", "ring")
LIDAR_INTENSITY_SCALE = 255.0
LIDAR_RING_SCALE = 40.0


def build_lidar2img(
    intrinsics: npt.NDArray[np.float64],
    sensor2lidar_rotation: npt.NDArray[np.float64],
    sensor2lidar_translation: npt.NDArray[np.float64],
    scale: Union[float, Tuple[float, float]],
    crop_top: int,
) -> npt.NDArray[np.float64]:
    """SSR-BEV homogeneous point -> image pixel, as a 4x4 matrix.

    ``sensor2lidar_*`` maps camera coordinates into the lidar frame, so the
    lidar->camera transform is its inverse.
    """
    R = np.asarray(sensor2lidar_rotation, dtype=np.float64)
    t = np.asarray(sensor2lidar_translation, dtype=np.float64).reshape(3)

    lidar2cam = np.eye(4, dtype=np.float64)
    lidar2cam[:3, :3] = R.T
    lidar2cam[:3, 3] = -R.T @ t

    K = np.asarray(intrinsics, dtype=np.float64).copy()
    if isinstance(scale, tuple):
        scale_x, scale_y = scale
    else:
        scale_x = scale_y = scale
    K[0, :] *= scale_x
    K[1, :] *= scale_y
    K[1, 2] -= crop_top

    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:3, :3] = K

    return viewpad @ lidar2cam @ T_LIDAR_FROM_SSR


class ParaSSRFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config):
        self._config = config

    def get_unique_name(self) -> str:
        """Cache name, invalidated by any config that changes the tensors."""
        cfg = self._config
        fields = [
            ("bev_h", cfg.bev_h),
            ("bev_w", cfg.bev_w),
            ("camera_names", cfg.camera_names),
            ("crop_top", cfg.crop_top),
            ("ego_motion_dims", cfg.ego_motion_dims),
            ("frame_indices", cfg.frame_indices),
            ("image_scale", cfg.image_scale),
            ("pc_range", cfg.pc_range),
            ("use_lidar", cfg.use_lidar),
        ]
        if cfg.use_lidar:
            # Only what reaches the tensor: padding length, z clip and layout.
            # Pillar/backbone sizes are model-side and do not touch the cache.
            fields += [
                ("lidar_layout", LIDAR_POINT_LAYOUT),
                ("lidar_max_points", cfg.lidar_max_points),
                ("lidar_z_range", cfg.lidar_z_range),
            ]
        return cache_key("para_ssr_feature", fields)

    # ------------------------------------------------------------------ #
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        cfg = self._config
        frame_indices = cfg.frame_indices  # e.g. (2, 3): one history + current

        images, l2i, hw = [], None, None
        for t in frame_indices:
            imgs_t, l2i_t, hw_t = self._get_cameras(agent_input, t)
            images.append(imgs_t)
            if l2i is None:  # extrinsics are ego-fixed; take them once
                l2i, hw = l2i_t, hw_t

        camera_feature = torch.stack(images)  # [T, N_cam, 3, H, W]

        bev_shift, ego_motion = self._get_ego_motion(agent_input, frame_indices)

        command = torch.tensor(
            np.asarray(agent_input.ego_statuses[-1].driving_command, dtype=np.float32)
        )
        status_feature = torch.cat(
            [
                command,
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ]
        )

        features = {
            "camera_feature": camera_feature,
            "lidar2img": l2i,
            "image_hw": hw,
            "bev_shift": bev_shift,
            "ego_motion": ego_motion,
            "command": command,
            "status_feature": status_feature,
        }
        if cfg.use_lidar:
            clouds = [self._get_lidar(agent_input, t) for t in frame_indices]
            features["lidar_points"] = torch.stack([points for points, _ in clouds])
            features["lidar_num_points"] = torch.tensor(
                [count for _, count in clouds], dtype=torch.int64
            )
        return features

    # ------------------------------------------------------------------ #
    def _get_lidar(self, agent_input: AgentInput, frame_idx: int) -> Tuple[torch.Tensor, int]:
        """Front-ROI point cloud in SSR axes, padded to ``lidar_max_points``.

        Returns the ``[N_max, 5]`` tensor and how many leading rows are real.
        """
        cfg = self._config
        pc = getattr(agent_input.lidars[frame_idx], "lidar_pc", None)
        if pc is None:
            raise ValueError(
                f"lidar_pc is missing at frame {frame_idx}; check use_lidar against "
                "get_sensor_config()"
            )
        pc = np.asarray(pc, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[0] < 5:
            raise ValueError(
                "lidar_pc must be a (>=5, n) array (x, y, z, intensity, ring, ...), "
                f"got {pc.shape}"
            )
        x_forward, y_left, z, intensity, ring = pc[0], pc[1], pc[2], pc[3], pc[4]
        x_right = -y_left
        y_forward = x_forward

        x0, y0, _, x1, y1, _ = cfg.pc_range
        z0, z1 = cfg.lidar_z_range
        # Half-open bounds: a point exactly on the far edge would otherwise map
        # to a pillar one past the canvas.
        keep = (
            (x_right >= x0) & (x_right < x1)
            & (y_forward >= y0) & (y_forward < y1)
            & (z >= z0) & (z < z1)
            & np.isfinite(x_right) & np.isfinite(y_forward) & np.isfinite(z)
        )
        index = np.flatnonzero(keep)
        if index.size > cfg.lidar_max_points:
            # Uniform thinning along scan order: deterministic (the cache must
            # not depend on a RNG) and spatially unbiased, since the merged scan
            # interleaves the five sensors rather than sorting by position.
            picks = np.linspace(0, index.size - 1, cfg.lidar_max_points).round().astype(np.int64)
            index = index[picks]

        points = np.zeros((cfg.lidar_max_points, len(LIDAR_POINT_LAYOUT)), dtype=np.float32)
        count = int(index.size)
        points[:count, 0] = x_right[index]
        points[:count, 1] = y_forward[index]
        points[:count, 2] = z[index]
        points[:count, 3] = intensity[index] / LIDAR_INTENSITY_SCALE
        points[:count, 4] = ring[index] / LIDAR_RING_SCALE
        return torch.from_numpy(points), count

    # ------------------------------------------------------------------ #
    def _get_cameras(
        self, agent_input: AgentInput, frame_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self._config
        cameras = agent_input.cameras[frame_idx]

        imgs, mats, shapes = [], [], []
        for name in cfg.camera_names:
            cam = getattr(cameras, name)
            if cam.image is None:
                raise ValueError(
                    f"camera {name!r} has no image at frame {frame_idx}; "
                    f"check ParaSSRConfig.camera_names against get_sensor_config()"
                )
            img = cam.image
            h, w = img.shape[:2]
            new_w = int(round(w * cfg.image_scale))
            new_h = int(round(h * cfg.image_scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if cfg.crop_top:
                img = img[cfg.crop_top :]
            img = img.astype(np.float32) / 255.0
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            imgs.append(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))))
            shapes.append([img.shape[0], img.shape[1]])
            mats.append(
                build_lidar2img(
                    cam.intrinsics,
                    cam.sensor2lidar_rotation,
                    cam.sensor2lidar_translation,
                    (new_w / w, new_h / h),
                    cfg.crop_top,
                )
            )

        return (
            torch.stack(imgs),
            torch.tensor(np.stack(mats), dtype=torch.float32),
            torch.tensor(shapes, dtype=torch.float32),
        )

    # ------------------------------------------------------------------ #
    def _get_ego_motion(
        self, agent_input: AgentInput, frame_indices: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BEV shift and the legacy ego vector, per queue step.

        ``ego_pose`` is ``(x, y, heading)`` in the *current* frame, x forward,
        y left.  For queue step ``k`` the shift is the ego displacement from
        frame ``k-1`` to frame ``k``, expressed in frame ``k``'s own axes.
        Temporal feature alignment also reads relative yaw at ``motion[2]``;
        the remaining vector entries are retained for cache compatibility.
        """
        cfg = self._config
        poses = [
            np.asarray(agent_input.ego_statuses[i].ego_pose, dtype=np.float64)
            for i in frame_indices
        ]

        grid_x = (cfg.pc_range[3] - cfg.pc_range[0]) / cfg.bev_w  # metres per cell, lateral
        grid_y = (cfg.pc_range[4] - cfg.pc_range[1]) / cfg.bev_h  # metres per cell, longitudinal

        shifts, motions = [], []
        for k, idx in enumerate(frame_indices):
            if k == 0:
                d_fwd = d_left = d_yaw = 0.0
            else:
                prev, cur = poses[k - 1], poses[k]
                # previous pose expressed in the current frame's axes
                dx, dy = prev[0] - cur[0], prev[1] - cur[1]
                c, s = np.cos(-cur[2]), np.sin(-cur[2])
                px, py = c * dx - s * dy, s * dx + c * dy
                d_fwd, d_left = -px, -py
                d_yaw = float(np.arctan2(np.sin(cur[2] - prev[2]), np.cos(cur[2] - prev[2])))

            # SSR axes: x = right = -left, y = forward
            shift_x = (-d_left) / grid_x / cfg.bev_w
            shift_y = d_fwd / grid_y / cfg.bev_h
            shifts.append([shift_x, shift_y])

            status = agent_input.ego_statuses[idx]
            motion = np.zeros(cfg.ego_motion_dims, dtype=np.float32)
            motion[0] = d_fwd
            motion[1] = d_left
            motion[2] = d_yaw
            motion[3] = np.sin(d_yaw)
            motion[4] = np.cos(d_yaw)
            motion[5:7] = np.asarray(status.ego_velocity, dtype=np.float32)[:2]
            motion[7:9] = np.asarray(status.ego_acceleration, dtype=np.float32)[:2]
            motion[9] = float(np.hypot(d_fwd, d_left))
            cmd = np.asarray(status.driving_command, dtype=np.float32).reshape(-1)
            motion[10 : 10 + min(4, cmd.size)] = cmd[:4]
            motions.append(motion)

        return (
            torch.tensor(np.asarray(shifts), dtype=torch.float32),
            torch.tensor(np.asarray(motions), dtype=torch.float32),
        )
