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
import os

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


_CONSTS = {}


def _consts(m: int, dtype, device):
    """the input-independent matrices of _profiles, built once per (m, dtype, device) (in-place writes into fresh
    tensors are not allowed inside a CUDA-graph capture): A's mask, the jerk difference matrix R, tri, the curvature Q"""
    key = (m, dtype, device)
    if key not in _CONSTS:
        # report 129 A28: never inference tensors (a first call inside val's inference_mode would cache them, and the
        # next training call with grad cannot save them for backward); same values in every mode
        with torch.inference_mode(False):
            rows = torch.arange(2 * m, device=device) // 2
            mask = torch.arange(m, device=device)[None, :] > rows[:, None]            # upper triangle (k = 1), x2 rows
            mask[:, 0] = False
            band = torch.zeros(m - 2, m - 1, dtype=dtype, device=device)
            idx = torch.arange(m - 2, device=device)
            band[idx, idx] = -1.0
            band[m - 3, m - 2] = 1.0
            R = torch.cat([torch.zeros(m - 2, 1, dtype=dtype, device=device), band], 1)    # [m-2, m]
            tri = torch.tril(torch.ones(m, m, dtype=dtype, device=device))
            Q = CURV_RATE * torch.eye(m, dtype=dtype, device=device)
            Q[0, 0] = CURV0
        _CONSTS[key] = (mask, R, tri, Q)
    return _CONSTS[key]


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
    mask, R, tri, Q = _consts(m, poses.dtype, poses.device)
    A = A.masked_fill(mask[None], 0.0)
    # km 10/19: as the official _make_banded_difference_matrix, which writes +I into [:, 1:] and THEN -I into [:, :-1]:
    # the second write overwrites every +1 except the last row's, so the official 'jerk' penalty is in fact (almost) a
    # penalty on the accelerations themselves.  The 10/15 port built the documented difference matrix instead: up to
    # 2 m/s off in the velocity profile, 0.02-0.2 m at 4 s (navtest), ~1 % of the candidate verdicts flipped.
    # (R = [0 | band] with band[i, i] = -1 and band[m - 3, m - 2] = +1: _consts)
    x = torch.linalg.solve(A.transpose(1, 2) @ A + JERK * (R.T @ R)[None], (A.transpose(1, 2) @ y[..., None]))[..., 0]
    v = torch.cat([x[:, :1], x[:, :1] + torch.cumsum(x[:, 1:] * DT, 1)], 1)            # [N, m] (v_0 .. v_{m-1})
    # curvature
    vel2 = v * DT ** 2
    rs = torch.cat([torch.ones_like(vel2[:, :1]), vel2[:, 1:]], 1)                     # row 0 of the rest is unscaled
    B = torch.cat([(v * DT)[..., None], tri[None, :, 1:] * rs[..., None]], -1)          # [N, m, m]
    c = torch.linalg.solve(B.transpose(1, 2) @ B + Q[None], B.transpose(1, 2) @ dh[..., None])[..., 0]
    k = torch.cat([c[:, :1], c[:, :1] + torch.cumsum(c[:, 1:] * DT, 1)], 1)            # [N, m]
    return v, k


def _track(poses: torch.Tensor, init: torch.Tensor, full: bool = False, angular: bool = False) -> torch.Tensor:
    """poses [N, 8, 3] ego frame; init [N, 7] = (vx, vy, ax, ay, steering angle, steering rate, yaw rate) of the ego
    rear axle now (or [N, 8] + angular acceleration) -> tracked (x, y, heading) [N, 41, 3];  full=True: (x, y, heading,
    vx, vy, ax, ay) [N, 41, 7] as the official simulated states carry them (index 0 = init; afterwards vy = ay = 0, ax =
    the lagged command); angular=True (with full) appends the official ANGULAR_VELOCITY = vx tan(steering) / wheel base
    and ANGULAR_ACCELERATION = its backward difference / dt of batch_kinematic_bicycle -> [N, 41, 9]."""
    ref = interpolate(poses.to(torch.float64))                                         # [N, 41, 3]
    vprof, kprof = _profiles(ref)                                                       # [N, 40]
    return _loop(ref, vprof, kprof, init, full, angular).to(poses.dtype)


def _loop(ref: torch.Tensor, vprof: torch.Tensor, kprof: torch.Tensor, init: torch.Tensor, full: bool, angular: bool):
    """the 40 tracking steps of _track (float64 out; no linear solves -> CUDA-graph capturable)"""
    dt64 = torch.float64
    n_ref = vprof.shape[1]                                                              # 40 = num_poses
    init = init.to(dt64)
    x = ref.new_zeros(ref.shape[0]); y = ref.new_zeros(ref.shape[0]); h = ref.new_zeros(ref.shape[0])
    vx, ax, steer = init[:, 0], init[:, 2], init[:, 4]
    key = ("q_lat", ref.device)
    if key not in _CONSTS:                                                              # (built before any capture)
        with torch.inference_mode(False):                                               # not an inference tensor (A28)
            _CONSTS[key] = torch.tensor(Q_LAT, dtype=dt64, device=ref.device)
    q = _CONSTS[key]
    out = [torch.stack([x, y, h], -1)]
    dyn = [torch.stack([vx, init[:, 1], ax, init[:, 3]], -1)]
    w_prev = init[:, 6]
    ang = [torch.stack([w_prev, init[:, 7] if init.shape[1] > 7 else torch.zeros_like(w_prev)], -1)]
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
        w = vx * torch.tan(steer) / WHEEL_BASE
        ang.append(torch.stack([w, (w - w_prev) / DT], -1))
        w_prev = w
    if full and angular:
        return torch.cat([torch.stack(out, 1), torch.stack(dyn, 1), torch.stack(ang, 1)], -1)
    if full:
        return torch.cat([torch.stack(out, 1), torch.stack(dyn, 1)], -1)
    return torch.stack(out, 1)


# km 10/20: the 40-step loop is ~2000 small kernels per call (CPU launch bound: ~35 ms per call vs ~10 ms of GPU work),
# so on CUDA _loop is replayed from a CUDA graph captured once per (shape, dtype, flags, grad): the same kernels on the
# same inputs -> bitwise the eager result (tests/km/test_track_graph.py).  interpolate + _profiles (the two batched
# linear solves are not capturable) stay eager.  With grad (corridor_plan_loss) the forward + backward pair of _loop
# is graphed by torch.cuda.make_graphed_callables.  KM_TRACK_GRAPH=0 -> eager.
_GRAPHS = {}


class _ThreadLocalGraph(torch.cuda.graph):
    """capture_error_mode 'thread_local': only the capturing thread is restricted, so the DataLoader pin-memory
    thread (cudaHostAlloc / copies, pin_memory=True) cannot invalidate a capture (the 'global' default can)"""

    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode: str = "thread_local"):
        super().__init__(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode)


def _graphed(key, args, full, angular, grad):
    g = _GRAPHS.get(key)
    if g is not None:
        return g
    try:
        g = _capture(args, full, angular, grad)
    except Exception as e:                                                       # never fail a step: eager for this key
        import warnings
        torch.cuda.synchronize()
        warnings.warn(f"torch_tracker: CUDA-graph capture failed ({type(e).__name__}: {e}); eager for {key}")
        g = lambda *a, full=full, angular=angular: _loop(*a, full, angular)
    _GRAPHS[key] = g
    return g


def _capture(args, full, angular, grad):
    fn = lambda r, v, k, i: _loop(r, v, k, i, full, angular)
    sargs = tuple(a.detach().clone() for a in args)
    if grad:
        sargs = tuple(a.requires_grad_(a_.requires_grad) for a, a_ in zip(sargs, args))
        orig = torch.cuda.graph                       # make_graphed_callables captures through torch.cuda.graph
        torch.cuda.graph = _ThreadLocalGraph
        try:
            g = torch.cuda.make_graphed_callables(fn, sargs)
        finally:
            torch.cuda.graph = orig
    else:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):                                                          # warm-up before the capture
                fn(*sargs)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with _ThreadLocalGraph(graph), torch.no_grad():
            so = fn(*sargs)

        def g(*a, graph=graph, sargs=sargs, so=so):
            for d, x in zip(sargs, a):
                d.copy_(x)
            graph.replay()
            return so.clone()
    return g


def track(poses: torch.Tensor, init: torch.Tensor, full: bool = False, angular: bool = False,
          slot: int = 0) -> torch.Tensor:
    """slot (km 10/25): a graphed callable's backward reads its static buffers, so two grad calls of the same shape in
    one forward (the cross-fit planners' plan loss next to the main one) need separate graphs: distinct slots."""
    # report 129 A28: eager under inference_mode (Lightning's val / eval): a graph first captured there holds inference
    # tensors as static inputs and the next training replay of that key crashes; eager is bitwise the graph's result
    # and val adds no graph memory (refiner_v6c.hook._trk did this already).  Training never runs in inference_mode.
    if (not poses.is_cuda or os.environ.get("KM_TRACK_GRAPH", "1") == "0" or torch.cuda.is_current_stream_capturing()
            or torch.is_inference_mode_enabled()):
        return _track(poses, init, full, angular)
    ref = interpolate(poses.to(torch.float64))
    vprof, kprof = _profiles(ref)
    args = (ref, vprof, kprof, init.detach())
    grad = torch.is_grad_enabled() and any(a.requires_grad for a in args)
    key = tuple((tuple(a.shape), a.dtype, a.requires_grad and grad) for a in args) + (poses.device, full, angular, grad,
                                                                                    slot)
    return _graphed(key, args, full, angular, grad)(*args).to(poses.dtype)


track.__doc__ = _track.__doc__
