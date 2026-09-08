from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
from navsim.agents.para_ssr.para_ssr_features import ParaSSRFeatureBuilder
from navsim.agents.para_ssr.para_ssr_model import ParaSSRModel


FRONT_CAMERAS = ("cam_f0", "cam_l0", "cam_r0")
SURROUND_CAMERAS = (
    "cam_f0", "cam_l0", "cam_l1", "cam_l2",
    "cam_r0", "cam_r1", "cam_r2", "cam_b0",
)


def _front_input():
    # Camera optical axes face forward. Only the selected front cameras exist;
    # reading any other camera or either unused history frame must fail.
    camera_frames = [SimpleNamespace(), SimpleNamespace()]
    for frame in (2, 3):
        cameras = {}
        for index, name in enumerate(FRONT_CAMERAS):
            cameras[name] = SimpleNamespace(
                image=np.full((64, 96, 3), 30 + index * 50 + frame, dtype=np.uint8),
                intrinsics=np.array([
                    [45.0 + index, 0.0, 48.0],
                    [0.0, 46.0 + index, 32.0],
                    [0.0, 0.0, 1.0],
                ]),
                sensor2lidar_rotation=np.array([
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ]),
                sensor2lidar_translation=np.array([0.0, float(index - 1), 1.5]),
            )
        camera_frames.append(SimpleNamespace(**cameras))
    statuses = [
        SimpleNamespace(
            ego_pose=np.array([0.5 * (frame - 3), 0.0, 0.0]),
            ego_velocity=np.array([1.0, 0.0]),
            ego_acceleration=np.zeros(2),
            driving_command=np.array([1, 0, 0, 0]),
        )
        for frame in range(4)
    ]
    return SimpleNamespace(cameras=camera_frames, ego_statuses=statuses)


def test_python_and_hydra_defaults_use_the_same_three_front_cameras():
    config = ParaSSRConfig()
    yaml_path = (
        Path(__file__).resolve().parents[1]
        / "navsim/planning/script/config/common/agent/para_ssr_agent.yaml"
    )
    hydra_config = instantiate(OmegaConf.load(yaml_path).config)

    for candidate in (config, hydra_config):
        assert tuple(candidate.camera_names) == FRONT_CAMERAS
        assert candidate.num_cams == 3
        assert tuple(candidate.frame_indices) == (2, 3)
        assert candidate.queue_length == 2
        ParaSSRAgent._validate_config(candidate, candidate.trajectory_sampling)


def test_default_sensor_config_loads_only_front_cameras_at_selected_history_frames():
    agent = ParaSSRAgent.__new__(ParaSSRAgent)
    torch.nn.Module.__init__(agent)
    agent._config = ParaSSRConfig()
    sensors = agent.get_sensor_config()

    for frame in range(4):
        expected = set(FRONT_CAMERAS) if frame in (2, 3) else set()
        assert set(sensors.get_sensors_at_iteration(frame)) == expected
    for name in set(SURROUND_CAMERAS) - set(FRONT_CAMERAS):
        assert getattr(sensors, name) is False
    assert sensors.lidar_pc is False


@pytest.mark.parametrize("camera_names", [FRONT_CAMERAS, FRONT_CAMERAS[::-1]])
def test_features_need_no_rear_cameras_and_preserve_image_and_calibration_order(camera_names):
    config = ParaSSRConfig(camera_names=camera_names, image_scale=0.5, crop_top=2)
    agent_input = _front_input()
    features = ParaSSRFeatureBuilder(config).compute_features(agent_input)

    assert features["camera_feature"].shape == (2, 3, 3, 30, 48)
    assert features["lidar2img"].shape == (3, 4, 4)
    torch.testing.assert_close(
        features["image_hw"], torch.tensor([[30.0, 48.0]] * 3)
    )

    for camera_index, name in enumerate(camera_names):
        for history_index, frame in enumerate((2, 3)):
            rgb = getattr(agent_input.cameras[frame], name).image[0, 0] / 255.0
            expected = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            actual = features["camera_feature"][history_index, camera_index]
            torch.testing.assert_close(
                actual,
                torch.tensor(expected, dtype=torch.float32)[:, None, None].expand_as(actual),
            )

        # Independent direct projection in NAVSIM coordinates, including the
        # camera's distinct intrinsic/extrinsic values and resize/crop.
        camera = getattr(agent_input.cameras[2], name)
        point_ssr = np.array([2.0, 14.0, 0.5, 1.0])
        point_lidar = np.array([14.0, -2.0, 0.5])
        point_camera = camera.sensor2lidar_rotation.T @ (
            point_lidar - camera.sensor2lidar_translation
        )
        projected = camera.intrinsics @ point_camera
        expected_pixel = projected[:2] / projected[2] * 0.5 - [0.0, 2.0]
        actual = features["lidar2img"][camera_index].numpy() @ point_ssr
        np.testing.assert_allclose(actual[:2] / actual[2], expected_pixel, rtol=1e-6)


def test_feature_cache_separates_front_surround_and_reordered_cameras():
    config = ParaSSRConfig()
    names = {
        ParaSSRFeatureBuilder(replace(config, camera_names=cameras)).get_unique_name()
        for cameras in (FRONT_CAMERAS, SURROUND_CAMERAS, FRONT_CAMERAS[::-1])
    }
    assert len(names) == 3
    assert ParaSSRFeatureBuilder(config).get_unique_name() == (
        ParaSSRFeatureBuilder(replace(config, camera_names=list(FRONT_CAMERAS)))
        .get_unique_name()
    )


def test_front_camera_model_has_finite_backward_on_front_only_bev():
    torch.manual_seed(9)
    config = ParaSSRConfig(
        image_architecture="resnet18",
        backbone_pretrained=False,
        image_scale=1.0,
        crop_top=0,
        bev_h=8,
        bev_w=8,
        embed_dims=16,
        num_heads=2,
        ffn_channels=32,
        encoder_num_layers=1,
        encoder_num_points_in_pillar=2,
        encoder_num_points_sca=2,
        encoder_attn_dropout=0.0,
        encoder_ffn_dropout=0.0,
        latent_num_layers=1,
        num_scenes=2,
        num_query=4,
        max_agents=4,
        det_num_decoder_layers=1,
        map_num_vec=3,
        map_num_pts_per_vec=4,
        map_num_decoder_layers=1,
        use_grid_mask=False,
    )
    model = ParaSSRModel(config).train()
    features = {
        name: tensor.unsqueeze(0)
        for name, tensor in ParaSSRFeatureBuilder(config).compute_features(_front_input()).items()
    }
    transformer = model.pts_bbox_head.transformer
    assert transformer.cams_embeds.shape == (3, config.embed_dims)
    encoder = transformer.encoder
    reference = encoder.get_reference_points(
        config.bev_h, config.bev_w,
        config.pc_range[5] - config.pc_range[2],
        config.encoder_num_points_in_pillar,
        device=torch.device("cpu"),
    )
    _, mask = encoder.point_sampling(
        reference, config.pc_range, features["lidar2img"], features["image_hw"]
    )
    visible = mask.any(dim=-1).any(dim=0).reshape(config.bev_h, config.bev_w)
    assert config.pc_range[1] == 0.0
    assert config.map_pc_range == config.pc_range
    assert visible.any()

    predictions = model(features)
    assert predictions["trajectory"].shape == (1, 8, 3)
    assert predictions["bev_embed"].shape == (1, 64, 16)
    assert "all_bbox_preds" in predictions
    assert "all_map_pts_preds" in predictions
    for prediction in predictions.values():
        assert torch.isfinite(prediction).all()
    sum(prediction.square().mean() for prediction in predictions.values()).backward()

    camera_gradient = transformer.cams_embeds.grad
    assert camera_gradient is not None
    assert torch.isfinite(camera_gradient).all()
    assert camera_gradient.abs().sum() > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
