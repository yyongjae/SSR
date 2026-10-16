"""km 10/15: exact comfort of tracked candidates = the official ego_is_comfortable on the torch tracker's states."""
import numpy as np
import torch

from navsim.agents.para_ssr.modules.torch_tracker import track
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

TP = np.arange(41, dtype=np.float64) * 0.1


def comfort(traj: np.ndarray, status: np.ndarray) -> np.ndarray:
    """traj [K, 8, 3] ego frame, status (vx, vy, ax, ay) -> bool [K] official comfort of the tracked rollout."""
    k = traj.shape[0]
    init = torch.zeros(k, 7, dtype=torch.float64)
    init[:, :4] = torch.as_tensor(np.asarray(status, np.float64))[None]
    st = track(torch.as_tensor(np.asarray(traj, np.float64)), init, full=True).numpy()       # [K, 41, 7]
    S = np.zeros((k, 41, StateIndex.size()), np.float64)
    S[..., StateIndex.X], S[..., StateIndex.Y], S[..., StateIndex.HEADING] = st[..., 0], st[..., 1], st[..., 2]
    S[..., StateIndex.VELOCITY_X], S[..., StateIndex.VELOCITY_Y] = st[..., 3], st[..., 4]
    S[..., StateIndex.ACCELERATION_X], S[..., StateIndex.ACCELERATION_Y] = st[..., 5], st[..., 6]
    return ego_is_comfortable(S, TP).all(-1)
