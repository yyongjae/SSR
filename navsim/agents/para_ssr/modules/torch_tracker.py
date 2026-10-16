"""km 10/15: differentiable, batched torch port of the official PDM simulation (PDMSimulator = BatchLQRTracker +
BatchKinematicBicycleModel, navsim/planning/simulation/planner/pdm_planner/simulation/), in the current ego frame.

  track(poses [N, 8, 3] ego frame (x fwd, y left, heading), 0.5 s steps; v0 [N] (vx, vy, ax, ay, steer, steer_rate,
        yaw_rate) as `init` [N, 7]) -> tracked states [N, 41, 3] (x, y, heading) at 0.1 s, index 0 = now

Same steps as the official pipeline:
  1. the proposal = [current pose] + 8 poses, linearly interpolated to 0.1 s (nuPlan InterpolatedTrajectory)
  2. velocity / curvature profiles by the two regularised least squares (batch_lqr_utils)
  3. 40 steps of: one-step longitudinal LQR (or the stopping P controller) + one-step lateral LQR over a 10-step
     horizon -> first-order-lagged accel / steering -> Euler kinematic bicycle (Pacifica wheel base)
Everything is differentiable w.r.t. the poses except the stop / no-stop switch.
"""
from __future__ import annotations

import math

import torch

WHEEL_BASE = 3.089                 # get_pacifica_parameters().wheel_base
DT = 0.1
HORIZON = 10
Q_LON, R_LON = 10.0, 1.0
Q_LAT = (1.0, 10.0, 0.0)
R_LAT = 1.0
JERK, CURV_RATE, CURV0 = 1e-4, 1e-2, 1e-10
STOP_GAIN, STOP_V = 0.5, 0.2
ACC_TAU, STEER_TAU, MAX_STEER = 0.2, 0.05, math.pi / 3


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def interpolate(poses: torch.Tensor) -> torch.Tensor:
    """[N, 8, 3] at 0.5 s (+ the origin at t = 0) -> [N, 41, 3] at 0.1 s, headings unwrapped then re-wrapped."""
    n = poses.shape[0]
    p = torch.cat([poses.new_zeros(n, 1, 3), poses], 1)                               # [N, 9, 3]
    h = p[..., 2]
    h = torch.cat([h[:, :1], h[:, :1] + torch.cumsum(_wrap(h[:, 1:] - h[:, :-1]), 1)], 1)
    p = torch.cat([p[..., :2], h[..., None]], -1)
    t = torch.arange(41, device=poses.device, dtype=poses.dtype) * DT / 0.5
    i0 = torch.clamp(t.floor().long(), max=7)
    f = (t - i0.to(t.dtype))[None, :, None]
    out = p[:, i0] * (1 - f) + p[:, i0 + 1] * f
    return torch.cat([out[..., :2], _wrap(out[..., 2:])], -1)


def _profiles(poses: torch.Tensor):
    """batch_lqr_utils.get_velocity_curvature_profiles_with_derivatives_from_poses -> (velocity [N, 41], curvature [N, 41])"""
    d = poses[:, 1:] - poses[:, :-1]
    xy, dh = d[..., :2], _wrap(d[..., 2])
    head = poses[:, :-1, 2]
    n, m = xy.shape[:2]                                                                 # m = 40 displacements
    y = xy.reshape(n, -1)
    col = torch.stack([torch.cos(head), torch.sin(head)], -1).reshape(n, -1)           # [N, 2m]
    A = col[..., None].expand(n, 2 * m, m) * DT ** 2
    A = torch.cat([col[..., None] * DT, A[..., 1:]], -1)
    rows = torch.arange(2 * m, device=poses.device) // 2
    mask = torch.arange(m, device=poses.device)[None, :] > rows[:, None]              # upper triangle (k = 1), x2 rows
    mask[:, 0] = False
    A = A.masked_fill(mask[None], 0.0)
    band = torch.zeros(m - 2, m - 1, dtype=poses.dtype, device=poses.device)
    idx = torch.arange(m - 2, device=poses.device)
    band[idx, idx + 1] = 1.0
    band[idx, idx] = -1.0
    R = torch.cat([torch.zeros(m - 2, 1, dtype=poses.dtype, device=poses.device), band], 1)   # [m-2, m]
    x = torch.linalg.solve(A.transpose(1, 2) @ A + JERK * (R.T @ R)[None], (A.transpose(1, 2) @ y[..., None]))[..., 0]
    v = torch.cat([x[:, :1], x[:, :1] + torch.cumsum(x[:, 1:] * DT, 1)], 1)            # [N, m] (v_0 .. v_{m-1})
    # curvature
    tri = torch.tril(torch.ones(m, m, dtype=poses.dtype, device=poses.device))
    vel2 = v * DT ** 2
    rs = torch.cat([torch.ones_like(vel2[:, :1]), vel2[:, 1:]], 1)                     # row 0 of the rest is unscaled
    B = torch.cat([(v * DT)[..., None], tri[None, :, 1:] * rs[..., None]], -1)          # [N, m, m]
    Q = CURV_RATE * torch.eye(m, dtype=poses.dtype, device=poses.device)
    Q[0, 0] = CURV0
    c = torch.linalg.solve(B.transpose(1, 2) @ B + Q[None], B.transpose(1, 2) @ dh[..., None])[..., 0]
    k = torch.cat([c[:, :1], c[:, :1] + torch.cumsum(c[:, 1:] * DT, 1)], 1)            # [N, m]
    return v, k


def track(poses: torch.Tensor, init: torch.Tensor, full: bool = False) -> torch.Tensor:
    """poses [N, 8, 3] ego frame; init [N, 7] = (vx, vy, ax, ay, steering angle, steering rate, yaw rate) of the ego
    rear axle now -> tracked (x, y, heading) [N, 41, 3];  full=True: (x, y, heading, vx, vy, ax, ay) [N, 41, 7] as
    the official simulated states carry them (index 0 = init; afterwards vy = ay = 0, ax = the lagged command)."""
    dt64 = torch.float64
    ref = interpolate(poses.to(dt64))                                                  # [N, 41, 3]
    vprof, kprof = _profiles(ref)                                                       # [N, 40]
    n_ref = vprof.shape[1]                                                              # 40 = num_poses
    init = init.to(dt64)
    x = ref.new_zeros(ref.shape[0]); y = ref.new_zeros(ref.shape[0]); h = ref.new_zeros(ref.shape[0])
    vx, ax, steer = init[:, 0], init[:, 2], init[:, 4]
    q = torch.tensor(Q_LAT, dtype=dt64, device=ref.device)
    out = [torch.stack([x, y, h], -1)]
    dyn = [torch.stack([vx, init[:, 1], ax, init[:, 3]], -1)]
    for i in range(40):
        # reference (velocity at the lookahead, curvature profile over the horizon)
        ridx = min(i + HORIZON, n_ref - 1)
        vref = vprof[:, ridx]
        L = ridx - i
        kp = kprof[:, i:ridx]
        if L < HORIZON:
            kp = torch.cat([kp, kprof[:, ridx:ridx + 1].expand(-1, HORIZON - L)], 1)
        # tracking errors w.r.t. the proposal state at this index
        r = ref[:, i]
        ex, ey = x - r[:, 0], y - r[:, 1]
        lat = -ex * torch.sin(r[:, 2]) + ey * torch.cos(r[:, 2])
        herr = _wrap(h - r[:, 2])
        stop = (vref <= STOP_V) & (vx <= STOP_V)
        # longitudinal one-step LQR
        Bl = HORIZON * DT
        a_lqr = -1.0 / (Bl * Q_LON * Bl + R_LON) * Bl * Q_LON * (vx - vref)
        a_stop = -STOP_GAIN * (vx - vref)
        acc = torch.where(stop, a_stop, a_lqr)
        # lateral one-step LQR over the horizon with the velocity profile v0 + a t
        vp = vx[:, None] + acc[:, None] * DT * torch.arange(HORIZON, dtype=dt64, device=ref.device)[None]
        # The lateral step matrices are unipotent upper-triangular (I + a_s E01 + b_s E12), so the horizon products
        # of the official loop have a closed form: z = A x0 + g and B = sum of products applied to (0, 0, dt).
        a_s, b_s = vp * DT, vp * DT / WHEEL_BASE                                        # [N, H]
        h_s = -vp * kp * DT
        cb = torch.cumsum(b_s, 1) - b_s                                                 # sum_{t<s} b_t
        ch = torch.cumsum(h_s, 1) - h_s
        z1_s = herr[:, None] + cb * steer[:, None] + ch                                 # z_1 before step s
        z0 = lat + (a_s * z1_s).sum(1)
        z1 = herr + b_s.sum(1) * steer + h_s.sum(1)
        tt = torch.arange(HORIZON, dtype=dt64, device=ref.device)[None] * DT             # B_2 before step s = s dt
        cbt = torch.cumsum(b_s * tt, 1) - b_s * tt                                      # B_1 before step s
        Bm = torch.stack([(a_s * cbt).sum(1), (b_s * tt).sum(1), torch.full_like(z0, HORIZON * DT)], -1)
        e0 = torch.stack([z0, _wrap(z1), _wrap(steer)], -1)
        sr_lqr = -1.0 / ((Bm * q * Bm).sum(-1) + R_LAT) * (Bm * q * e0).sum(-1)
        srate = torch.where(stop, torch.zeros_like(sr_lqr), sr_lqr)
        # motion model: first-order lag, then Euler
        ideal_steer = DT * srate + steer
        acc_u = DT / (DT + ACC_TAU) * (acc - ax) + ax
        steer_u = DT / (DT + STEER_TAU) * (ideal_steer - steer) + steer
        srate_u = (steer_u - steer) / DT
        # (the official _update_commands leaves the steering ANGLE of the propagating state unchanged: the heading
        #  rate uses the previous angle, the new angle = previous + lagged rate * dt)
        x = x + vx * torch.cos(h) * DT
        y = y + vx * torch.sin(h) * DT
        h = _wrap(h + vx * torch.tan(steer) / WHEEL_BASE * DT)
        vx = vx + acc_u * DT
        steer = torch.clamp(steer + srate_u * DT, -MAX_STEER, MAX_STEER)
        ax = acc_u
        out.append(torch.stack([x, y, h], -1))
        dyn.append(torch.stack([vx, torch.zeros_like(vx), ax, torch.zeros_like(vx)], -1))
    if full:
        return torch.cat([torch.stack(out, 1), torch.stack(dyn, 1)], -1).to(poses.dtype)
    return torch.stack(out, 1).to(poses.dtype)
