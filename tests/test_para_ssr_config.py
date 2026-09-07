from types import SimpleNamespace

import pytest
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
from navsim.agents.para_ssr.para_ssr_targets import MAP_CLASS_NAMES


def _config(**overrides):
    values = {
        "frame_indices": (2, 3),
        "ego_motion_dims": 18,
        "num_navi_cmd": 4,
        "ego_fut_mode": 4,
        "traj_dims": 3,
        "num_feature_levels": 1,
        "backbone_out_indices": (4,),
        "num_det_classes": 7,
        "det_code_size": 10,
        "det_code_weights": (1.0,) * 10,
        "map_num_classes": len(MAP_CLASS_NAMES),
        "map_num_orders": 20,
        "camera_names": ("cam_f0", "cam_l0", "cam_r0"),
        "max_agents": 100,
        "num_query": 300,
        "map_dir_interval": 1,
        "map_num_pts_per_vec": 20,
        "fut_ts": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sampling():
    return TrajectorySampling(time_horizon=4.0, interval_length=0.5)


def test_default_config_contract_is_valid():
    ParaSSRAgent._validate_config(_config(), _sampling())


@pytest.mark.parametrize(
    "override, message",
    [
        ({"frame_indices": (1, 2)}, "current history frame"),
        ({"frame_indices": (-1, 3)}, "indices in"),
        ({"frame_indices": (2.5, 3)}, "indices in"),
        ({"frame_indices": (3, 2, 3)}, "unique and chronological"),
        ({"ego_motion_dims": 13}, "at least 14"),
        ({"ego_fut_mode": 3}, "must match"),
        ({"num_navi_cmd": 3, "ego_fut_mode": 3}, "fixed at 4"),
        ({"traj_dims": 2}, "traj_dims"),
        ({"num_feature_levels": 2}, "single-level"),
        ({"backbone_out_indices": (3, 4)}, "single-level"),
        ({"num_det_classes": 8}, "num_det_classes"),
        ({"det_code_size": 9}, "fixed 10D"),
        ({"det_code_weights": (1.0,) * 9}, "fixed 10D"),
        ({"map_num_classes": 2}, "map_num_classes"),
        ({"map_num_orders": 0}, "map_num_orders"),
        ({"camera_names": ("cam_f0", "cam_f0")}, "camera_names"),
        ({"camera_names": ("cam_front",)}, "camera_names"),
        ({"max_agents": 301}, "cannot exceed"),
        ({"map_dir_interval": 0}, "map_dir_interval"),
        ({"map_dir_interval": 20}, "map_dir_interval"),
        ({"fut_ts": 6}, "disagree"),
    ],
)
def test_invalid_config_contract_fails_fast(override, message):
    with pytest.raises(ValueError, match=message):
        ParaSSRAgent._validate_config(_config(**override), _sampling())


def test_non_navsim_sampling_interval_fails_fast():
    sampling = TrajectorySampling(time_horizon=2.0, interval_length=0.25)
    with pytest.raises(ValueError, match="0.5 second"):
        ParaSSRAgent._validate_config(_config(), sampling)
