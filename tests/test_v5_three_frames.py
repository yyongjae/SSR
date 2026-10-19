"""refiner v5 student, 3-frame input (t-1.0, t-0.5, t0 = NAVSIM history frames 1, 2, 3; fusion_v13_v5.extra).

The 2-frame default (2, 3) stays the code default; these tests check that (1, 2, 3) is accepted, loads the right
sensors, chains the ego motion frame by frame and runs the history BEV recursion in order with no gradient.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
from navsim.agents.para_ssr.para_ssr_features import ParaSSRFeatureBuilder
from navsim.agents.para_ssr.para_ssr_model import ParaSSRModel

REPO = Path(__file__).resolve().parents[1]
EXTRA = REPO / "tools/ck/e2e2/fusion_v13_v5.extra"


def _status(pose, vel=(5.0, 0.1), acc=(0.3, -0.05), cmd=(0, 1, 0, 0)):
    return SimpleNamespace(ego_pose=np.asarray(pose, dtype=np.float64), ego_velocity=np.asarray(vel),
                           ego_acceleration=np.asarray(acc), driving_command=np.asarray(cmd))


def _agent_input():
    # NAVSIM ego poses (x forward, y left, heading) in the CURRENT frame (frame 3 = origin); a gentle left curve
    poses = [(-7.6, -0.9, -0.12), (-5.1, -0.45, -0.08), (-2.55, -0.12, -0.035), (0.0, 0.0, 0.0)]
    return SimpleNamespace(ego_statuses=[_status(p, vel=(5.0 + 0.1 * i, 0.02 * i)) for i, p in enumerate(poses)])


def _rel(prev, cur):
    """prev pose seen from cur (cur's axes): (d_fwd, d_left, d_yaw) of the step prev -> cur, independent formula."""
    R = np.array([[np.cos(cur[2]), -np.sin(cur[2])], [np.sin(cur[2]), np.cos(cur[2])]])
    p = R.T @ (np.asarray(prev[:2]) - np.asarray(cur[:2]))
    dyaw = np.arctan2(np.sin(cur[2] - prev[2]), np.cos(cur[2] - prev[2]))
    return -p[0], -p[1], dyaw


def test_config_accepts_three_frames_and_extra_sets_them():
    cfg = replace(ParaSSRConfig(), frame_indices=(1, 2, 3))
    ParaSSRAgent._validate_config(cfg, cfg.trajectory_sampling)
    assert cfg.queue_length == 3
    lines = [ln.strip() for ln in EXTRA.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert "++agent.config.frame_indices=[1,2,3]" in lines
    # the Hydra override arrives as a list: the validator converts it (tuple(config.frame_indices)) and accepts it
    cfg_list = replace(ParaSSRConfig(), frame_indices=[1, 2, 3])
    ParaSSRAgent._validate_config(cfg_list, cfg_list.trajectory_sampling)
    for bad in ((1, 3, 2), (1, 2), (1, 1, 3)):
        with pytest.raises(ValueError):
            ParaSSRAgent._validate_config(replace(ParaSSRConfig(), frame_indices=bad), cfg.trajectory_sampling)


def test_sensor_config_loads_frame_one():
    cfg = replace(ParaSSRConfig(), frame_indices=(1, 2, 3))
    sc = ParaSSRAgent.get_sensor_config(SimpleNamespace(_config=cfg))
    for name in cfg.camera_names:
        assert getattr(sc, name) == [1, 2, 3]
    assert sc.cam_b0 is False and sc.cam_l1 is False
    assert sc.lidar_pc == ([1, 2, 3] if cfg.use_lidar else False)


def test_ego_motion_chains_frame_by_frame():
    ai = _agent_input()
    poses = [s.ego_pose for s in ai.ego_statuses]
    b3 = ParaSSRFeatureBuilder(replace(ParaSSRConfig(), frame_indices=(1, 2, 3)))
    b2 = ParaSSRFeatureBuilder(ParaSSRConfig())
    s3, m3 = b3._get_ego_motion(ai, (1, 2, 3))
    s2, m2 = b2._get_ego_motion(ai, (2, 3))
    assert s3.shape == (3, 2) and m3.shape[0] == 3
    # first queue entry has no predecessor: zero shift (its prev_bev is None)
    assert torch.count_nonzero(s3[0]) == 0 and float(m3[0, 2]) == 0.0
    # the t0 step (2 -> 3) is the same in both queues; frame 1 -> 2 is new
    torch.testing.assert_close(s3[2], s2[1])
    torch.testing.assert_close(m3[2], m2[1])
    cfg = b3._config
    gx = (cfg.pc_range[3] - cfg.pc_range[0]) / cfg.bev_w
    gy = (cfg.pc_range[4] - cfg.pc_range[1]) / cfg.bev_h
    for k, (a, b) in enumerate(((1, 2), (2, 3)), start=1):
        d_fwd, d_left, d_yaw = _rel(poses[a], poses[b])
        exp = torch.tensor([(-d_left) / gx / cfg.bev_w, d_fwd / gy / cfg.bev_h], dtype=torch.float32)
        torch.testing.assert_close(s3[k], exp)
        assert abs(float(m3[k, 0]) - d_fwd) < 1e-5 and abs(float(m3[k, 2]) - d_yaw) < 1e-6
        # motion vector status entries come from the queue frame itself
        np.testing.assert_allclose(m3[k, 5:7].numpy(), ai.ego_statuses[b].ego_velocity[:2], rtol=1e-6)


class _StubModel(nn.Module):
    """The attributes ParaSSRModel.obtain_history_bev touches; records every BEV call."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.lidar_encoder = None
        self.calls = []

    def extract_img_feat(self, img):
        return img.mean(dim=(1, 2, 3, 4))[:, None] * self.w            # [B, 1]

    def lidar_bev(self, features, t):
        return None

    def pts_bbox_head(self, feats, *, lidar2img, image_hw, ego_motion, bev_shift, bev_yaw, prev_bev, only_bev,
                      lidar_bev):
        out = feats * 10.0 + bev_shift.sum(-1, keepdim=True) + (0.0 if prev_bev is None else prev_bev)
        self.calls.append({"feats": feats.detach().clone(), "bev_shift": bev_shift.clone(), "bev_yaw": bev_yaw.clone(),
                           "prev": None if prev_bev is None else prev_bev.detach().clone(),
                           "out": out.detach().clone(), "training": self.training, "grad": torch.is_grad_enabled()})
        return out


def test_history_bev_recursion_three_frames_in_order_no_grad():
    torch.manual_seed(0)
    B, T = 2, 3
    feats = {"camera_feature": torch.randn(B, T, 3, 3, 4, 5), "lidar2img": torch.zeros(B, 3, 4, 4),
             "image_hw": torch.zeros(B, 3, 2), "bev_shift": torch.randn(B, T, 2), "ego_motion": torch.randn(B, T, 18)}
    m = _StubModel().train()
    prev = ParaSSRModel.obtain_history_bev(m, feats)
    assert len(m.calls) == T - 1                       # frames 1 and 2 (t-1.0, t-0.5); t0 is the caller's pass
    c0, c1 = m.calls
    assert c0["prev"] is None                          # the oldest frame starts the chain
    torch.testing.assert_close(c0["feats"], feats["camera_feature"][:, 0].mean(dim=(1, 2, 3, 4))[:, None])
    torch.testing.assert_close(c0["bev_shift"], feats["bev_shift"][:, 0])
    torch.testing.assert_close(c1["prev"], c0["out"])  # frame 2 attends to frame 1's BEV
    torch.testing.assert_close(c1["bev_shift"], feats["bev_shift"][:, 1])
    torch.testing.assert_close(c1["bev_yaw"], feats["ego_motion"][:, 1, 2])
    torch.testing.assert_close(prev, c1["out"])        # the current frame gets frame 2's (recursive) BEV
    assert not prev.requires_grad and not any(c["grad"] for c in m.calls)
    assert not any(c["training"] for c in m.calls)     # history runs in eval mode ...
    assert m.training                                  # ... and the training flag is restored
    # 2-frame queue: one history call, frame 2 alone (no prev)
    m2 = _StubModel().train()
    f2 = {k: (v[:, 1:] if k in ("camera_feature", "bev_shift", "ego_motion") else v) for k, v in feats.items()}
    ParaSSRModel.obtain_history_bev(m2, f2)
    assert len(m2.calls) == 1 and m2.calls[0]["prev"] is None
