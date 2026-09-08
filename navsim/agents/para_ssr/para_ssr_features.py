"""Feature builder: navsim ``AgentInput`` -> the tensors BEVFormer needs.

Produces, per sample:

``camera_feature``  ``[T, N_cam, 3, H, W]``   normalised selected camera images
``lidar2img``       ``[N_cam, 4, 4]``         SSR-BEV frame -> image pixels
``image_hw``        ``[N_cam, 2]``            (H, W) after resize/crop
``bev_shift``       ``[T, 2]``                normalised ego motion per step
``ego_motion``      ``[T, ego_motion_dims]``  query conditioning vector
``command``         ``[4]``                   driving command one-hot
``status_feature``  ``[8]``                   command + velocity + acceleration

Coordinate frames
-----------------
SSR's BEV is VAD's ego frame: **x lateral (+right), y longitudinal (+forward)**.
navsim/nuPlan lidar is **x forward, y left**.  ``T_LIDAR_FROM_SSR`` below folds
that rotation into ``lidar2img`` so the model itself never sees the difference.

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
        return cache_key(
            "para_ssr_feature",
            (
                ("bev_h", cfg.bev_h),
                ("bev_w", cfg.bev_w),
                ("camera_names", cfg.camera_names),
                ("crop_top", cfg.crop_top),
                ("ego_motion_dims", cfg.ego_motion_dims),
                ("frame_indices", cfg.frame_indices),
                ("image_scale", cfg.image_scale),
                ("pc_range", cfg.pc_range),
            ),
        )

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

        return {
            "camera_feature": camera_feature,
            "lidar2img": l2i,
            "image_hw": hw,
            "bev_shift": bev_shift,
            "ego_motion": ego_motion,
            "command": command,
            "status_feature": status_feature,
        }

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
        """BEV shift and the query-conditioning vector, per queue step.

        ``ego_pose`` is ``(x, y, heading)`` in the *current* frame, x forward,
        y left.  For queue step ``k`` the shift is the ego displacement from
        frame ``k-1`` to frame ``k``, expressed in frame ``k``'s own axes.
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
