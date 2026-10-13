"""SE(2) history warp matches the interaction-branch grid_sample."""
import torch

from navsim.agents.para_ssr.modules.temporal_alignment import warp_previous_bev

PC = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)


def test_zero_motion_keeps_every_cell():
    torch.manual_seed(0)
    prev = torch.randn(2, 3, 4, 8)
    shift = torch.zeros(2, 2)
    yaw = torch.zeros(2)
    aligned = warp_previous_bev(prev, shift, yaw, PC)
    assert torch.allclose(aligned, prev, atol=1e-5)


def test_lateral_shift_moves_a_cell_and_yaw_is_detached():
    prev = torch.zeros(1, 1, 4, 8)
    prev[0, 0, 1, 3] = 1.0
    prev.requires_grad_(True)
    shift = torch.tensor([[1.0 / 8.0, 0.0]])
    yaw = torch.zeros(1, requires_grad=True)
    aligned = warp_previous_bev(prev, shift, yaw, PC)
    assert aligned[0, 0, 1, 2].item() > 0.9
    assert aligned[0, 0, 1, 3].item() < 0.1
    aligned.sum().backward()
    assert prev.grad.abs().sum() > 0
    assert yaw.grad is None
