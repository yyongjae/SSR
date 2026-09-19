"""Kinematic-bicycle output layer, ported from TOAD (github.com/valeoai/TOAD).

Source: ``navsim/agents/drivoR/drivor_model.py`` of TOAD -- ``_bicycle_rollout``,
``_proposals_to_controls`` (inverse kinematics) and ``_clamp_controls`` with the
same comfort limits.  TOAD uses them at test time (CEM in control space); here
they form the planner's OUTPUT LAYER during training as well: the head predicts
controls (lon. acceleration a, yaw rate omega) per 0.5 s step and the poses are
their rollout, so heading and path agree by construction and every trajectory
is one a car can drive.

Why: the PDM scorer tracks position AND heading with an LQR controller.  v1
regresses the two independently (per-step deltas, cumsum'ed); its heading drifts
off the path direction with the horizon (median 0.28 -> 0.98 deg, p90 0.83 ->
3.91 deg over the 8 steps on navtest) and overwriting it with the path tangent
at test time lifts the control run's PDMS 84.74 -> 85.61.
"""
from __future__ import annotations

import torch

# TOAD's constants (the PDM comfort thresholds)
MAX_LON_ACCEL = 2.40        # [m/s^2]
MIN_LON_ACCEL = -4.05       # [m/s^2]
MAX_ABS_YAW_RATE = 0.95     # [rad/s]
CLAMP_MARGIN = 1.5          # TOAD: "slightly-relaxed kinematic envelopes"


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def bicycle_rollout(controls: torch.Tensor, init_speed: torch.Tensor, dt: float):
    """Forward-Euler kinematic rollout with mid-point evaluation (TOAD ``_bicycle_rollout``).

    controls [..., P, 2] (a, omega); init_speed broadcastable to controls[..., 0, 0].
    Returns poses [..., P, 3] (x, y, heading) in the ego frame at t=0, speeds [..., P].
    """
    a, omega = controls[..., 0], controls[..., 1]
    v0 = init_speed.unsqueeze(-1).expand_as(a[..., :1])
    speeds_end = (v0 + torch.cumsum(a * dt, dim=-1)).clamp_min(0.0)
    speeds_start = torch.cat([v0, speeds_end[..., :-1]], dim=-1)
    v_mid = 0.5 * (speeds_start + speeds_end)
    headings_end = torch.cumsum(omega * dt, dim=-1)
    headings_start = torch.cat([torch.zeros_like(headings_end[..., :1]), headings_end[..., :-1]], dim=-1)
    theta_mid = 0.5 * (headings_start + headings_end)
    x = torch.cumsum(v_mid * torch.cos(theta_mid) * dt, dim=-1)
    y = torch.cumsum(v_mid * torch.sin(theta_mid) * dt, dim=-1)
    return torch.stack([x, y, headings_end], dim=-1), speeds_end


def poses_to_controls(poses: torch.Tensor, init_speed: torch.Tensor, dt: float) -> torch.Tensor:
    """Inverse kinematics (TOAD ``_proposals_to_controls``): poses [..., P, 3] -> (a, omega) [..., P, 2]."""
    zero = torch.zeros_like(poses[..., :1, :])
    full = torch.cat([zero, poses], dim=-2)
    dx, dy = torch.diff(full[..., 0], dim=-1), torch.diff(full[..., 1], dim=-1)
    dh = wrap_angle(torch.diff(full[..., 2], dim=-1))
    speed = torch.sqrt(dx * dx + dy * dy + 1e-12) / dt
    v0 = init_speed.unsqueeze(-1).expand_as(speed[..., :1])
    a = torch.diff(torch.cat([v0, speed], dim=-1), dim=-1) / dt
    return torch.stack([a, dh / dt], dim=-1)


def clamp_controls(controls: torch.Tensor) -> torch.Tensor:
    """TOAD ``_clamp_controls``: (a, omega) inside the relaxed comfort envelope."""
    a = controls[..., 0].clamp(MIN_LON_ACCEL * CLAMP_MARGIN, MAX_LON_ACCEL * CLAMP_MARGIN)
    omega = controls[..., 1].clamp(-MAX_ABS_YAW_RATE * CLAMP_MARGIN, MAX_ABS_YAW_RATE * CLAMP_MARGIN)
    return torch.stack([a, omega], dim=-1)


def bezier_xyyaw(xy: torch.Tensor, fallback_yaw: torch.Tensor, min_speed: float = 1e-3) -> torch.Tensor:
    """Heading from the path, ported from DiffusionDriveV2 (``diffusiondrivev2_model_sel.bezier_xyyaw``).

    The origin and the T predicted points are the control points of a degree-T
    Bezier curve; the heading at point k is the direction of its derivative at
    t = k / T.  The regressed heading is not used, so heading and path agree by
    construction.

    xy [..., T, 2]; fallback_yaw [..., T] is used where the derivative is ~0 (a car
    standing still has no direction of travel; DiffusionDriveV2 would return
    atan2(0, 0) = 0, whose gradient is NaN).  Returns [..., T, 3].
    """
    import math

    n = xy.shape[-2]
    ctrl = torch.cat([torch.zeros_like(xy[..., :1, :]), xy], dim=-2)          # [..., T+1, 2]
    delta = ctrl[..., 1:, :] - ctrl[..., :-1, :]                              # [..., T, 2]
    binom = torch.tensor([math.comb(n - 1, i) for i in range(n)], device=xy.device, dtype=xy.dtype)
    t = torch.arange(1, n + 1, device=xy.device, dtype=xy.dtype) / n
    powers = torch.arange(0, n, device=xy.device, dtype=xy.dtype)
    basis = binom * t.view(-1, 1) ** powers * (1 - t).view(-1, 1) ** powers.flip(0)   # [T(t_k), T(i)]
    deriv = n * torch.einsum("ki,...ic->...kc", basis, delta)                  # [..., T, 2]
    moving = deriv.norm(dim=-1) > min_speed
    safe = torch.where(moving.unsqueeze(-1), deriv, torch.ones_like(deriv))   # keep atan2's backward finite
    yaw = torch.where(moving, torch.atan2(safe[..., 1], safe[..., 0]), fallback_yaw)
    return torch.cat([xy, yaw.unsqueeze(-1)], dim=-1)
