import numpy as np
import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig


class _FeatureBuilder:
    def compute_features(self, agent_input):
        return {"feature": torch.ones(2)}


class _DummyAgent(AbstractAgent):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def name(self):
        return "dummy"

    def get_sensor_config(self):
        return SensorConfig.build_no_sensors()

    def initialize(self):
        return None

    def get_feature_builders(self):
        return [_FeatureBuilder()]

    def forward(self, features):
        assert features["feature"].device == self.anchor.device
        trajectory = torch.zeros((1, 8, 3), device=self.anchor.device)
        return {"trajectory": trajectory}


def test_compute_trajectory_gpu_follows_model_device_on_cpu():
    agent = _DummyAgent().cpu()
    trajectory = agent.compute_trajectory_gpu(object())
    assert trajectory.poses.shape == (8, 3)
    assert trajectory.poses.dtype == np.float32
