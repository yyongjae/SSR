"""Official NAVSIM PDM simulation (PDMSimulator: reference interpolation -> BatchLQRTracker ->
BatchKinematicBicycleModel) as a batched torch function -- refiner v4_3 M1 (docs/refiner_v4/V4_3_SPEC.md).
0 parameters, no buffers.

What it reproduces (navsim/evaluate/pdm_score.py transform_trajectory + get_trajectory_as_array, navsim/planning/
simulation/planner/pdm_planner/simulation/{pdm_simulator, batch_lqr, batch_lqr_utils, batch_kinematic_bicycle}.py with
the default_scoring_parameters.yaml sampling: 40 steps of 0.1 s)
  1. reference (`reference_states`): origin (= t0 ego rear axle) + the 8 submitted poses (t = 0.5 k s) -> nuplan
     InterpolatedTrajectory -> 41 states at t = 0.1 n s: x, y linear in time, heading np.unwrap-ed over the 9 knots then
     linear, principal value (= refiner.geometry.dense_reference + wrap).
  2. profiles (`fit_profiles`, BatchLQRTracker.update, once per trajectory): least-squares velocity profile v_k
     (k = 0..39; jerk penalty 1e-4) and curvature profile kappa_k (k = 0..39; curvature-rate penalty 1e-2, initial
     curvature penalty 1e-10) of the 41 reference poses.
  3. 40 steps n -> n + 1 (`simulate`), rear-axle state (x, y, heading, v, a, steer):
       r = min(n + 10, 39), v_ref = v_r; kappa window kappa_n .. kappa_{n+9} (held at kappa_39 past the end)
       stopping (v_ref <= 0.2 and v <= 0.2 m/s): a_cmd = -0.5 (v - v_ref), steer_rate_cmd = 0
       else lon LQR, one step over the 1 s horizon (Q 10, R 1, B = 10 x 0.1 s): a_cmd = -(B Q (v - v_ref)) / (B Q B + R)
            -- a speed reference only, no position feedback;
            lat LQR, one step over the 10 x 0.1 s linear time-varying model of [e_lat, e_head, steer]:
              e_lat' = v_k e_head, e_head' = v_k (steer / L - kappa_k), steer' = steer_rate, v_k = v + a_cmd 0.1 k
              (Euler, zero-order hold), errors to reference state n: e_lat = -dx sin h_n + dy cos h_n,
              e_head = wrap(heading - h_n); zero-input prediction x_N = A x_0 + g (heading / steer rows wrapped),
              steer_rate_cmd = -(B^T Q x_N) / (B^T Q B + R), Q diag(1, 10, 0), R 1
       bicycle (wheelbase 3.089 m, first-order lags accel 0.2 s / steer 0.05 s, Euler dt = 100000 us * 1e-6,
              |steer| <= pi / 3):
              a' = a + dt / (dt + 0.2) (a_cmd - a)
              steer_rate' = (dt / (dt + 0.05)) (steer + dt steer_rate_cmd - steer) / dt
              x += v cos(heading) dt, y += v sin(heading) dt, heading = pv(heading + v tan(steer) / L dt)
              (OLD v, steer in the position / heading update)
              v += a' dt;  steer = clip(steer + steer_rate' dt, -pi / 3, pi / 3)
     (tracker dt = proposal_sampling.interval_length = 0.1; bicycle dt = TimePoint(100000).time_s = 0.09999999999999999,
     both kept as in the official code.)
  Initial state (ego_state_to_state_array of the metric-cache ego state, in N): pose (0, 0, 0), v0 =
  rear_axle_velocity_2d.x [m/s], a0 = rear_axle_acceleration_2d.x [m/s^2], steer0 = tire_steering_angle [rad] (0 in
  every navsim metric cache: F1 init column 2 is 0 on all 11,500 sampled val / train / navtest tokens).  The other
  state fields (vy, ay, steering rate, yaw rate, yaw acceleration) never reach (x, y, heading, v, a, steer): vy / ay are
  overwritten with 0 at step 1, the steering rate is rebuilt from the command every step, yaw rate / acceleration are
  outputs only.  Hence ego = (v0, a0[, steer0]) -- in the packed data v0 = status[..., 4], a0 = status[..., 6] (the
  ego-frame rear-axle velocity / acceleration x; equal to F1 init on val, max diff 0).  NOT hypot(vx, vy).

Frame
  The official simulates in the global (map) frame, this port in the N frame (NAVSIM ego frame at t0: rear-axle
  origin, x forward, y left, heading CCW from +x, rad; metres, seconds).  Every step is equivariant under the rigid
  motion global -> N (pose differences, wrapped heading differences, rotation-invariant least squares), so both agree up
  to float rounding and the official's microsecond rounding of the knot / query time stamps (int(t_s * 1e6) of a
  ~1.6e9 s epoch: <= 1 us -> <= 3e-5 m at 30 m/s).

Least squares (fit='reduced', default)
  The official solves the normal equations in (v0, a_0..a_38) and (kappa_0, c_0..c_38) with torch.inverse; they are
  badly scaled (dt / dt^2 columns, 1e-10 regulariser) and lose most digits in float32.  'reduced' is the SAME minimiser
  written in the profile values themselves (a bijective change of variables, so not an approximation):
    velocity : u_k = v_k dt is the step length along the reference heading h_k; with p_k = cos h_k dx_k + sin h_k dy_k
               the objective is |u - p|^2 + (jerk / dt^4) |Rb D1 u|^2 + const (a_k = (D1 u)_k / dt^2)  ->
               u = (I + lam (Rb D1)^T Rb D1)^-1 p, a CONSTANT 40 x 40 matrix (lam = 1 for dt 0.1, jerk 1e-4;
               condition number 7.25).
               Official quirk kept: navsim's vectorised _make_banded_difference_matrix(38) assigns the +1 band first and
               then overwrites it with -eye, so Rb = -I (38 x 39) with a single +1 at (37, 38): the "jerk" penalty is
               in fact 1e-4 sum_{k<=36} a_k^2 + 1e-4 (a_38 - a_37)^2 (an acceleration-magnitude penalty, unlike the
               nuplan-devkit loop).  Rb is built the same way here (tests check it against the navsim function).
    curvature: y_k = wrap(h_{k+1} - h_k) ~ w_k kappa_k, w_k = v_k dt; objective |y - W kappa|^2
               + (crp / dt^2) |D1 kappa|^2 + 1e-10 kappa_0^2  ->  tridiagonal SPD system
               (W^2 + lam1 D1^T D1 + 1e-10 e0 e0^T) kappa = W y.
  fit='official' is the literal normal-equation port (inverse included), for parity checks.  Both always run in float64
  (B x 40 x 40, cheap) whatever the rollout dtype; the 40-step rollout runs in `dtype`.

Parity (2026-10-16; tests/test_v4_lqr.py prints the sampled numbers with -s)
  * vs the official numpy chain (transform_trajectory + get_trajectory_as_array + PDMSimulator) on 13 synthetic edge
    cases (standstill, creep with heading jitter, reverse / v0 < 0, U-turn through +-pi, steer saturation, lateral
    shift, hard brake, steer0 != 0) at an ego pose 664 km / 3,997 km from the map origin: float64 max |dxy| 2.8e-9 m,
    |dheading| 3.8e-10 rad (both fits); float32 1.3e-5 m / 9.7e-7 rad.
  * vs F1 out/{val,train,navtest}/run.npz, ALL rows (official PDMSimulator on the metric-cache ego states, stored as
    float32 in N; par16 + var80 + latT16 (+ latM16); 1,376,000 trajectories, 116,366 of them use the stopping
    controller, 12,707 saturate the steering): float64 max |dxy| 2.4e-5 m (val / train / navtest 2.38 / 2.43 / 2.24e-5,
    p99 1.4e-5), max |dheading| 3.7e-6 rad, max |dv| 5.6e-5 m/s, 0 trajectories > 1 mm -- the float32 storage +
    microsecond time-stamp floor.  float32 rollout: max 3.4e-5 m on train / navtest; on val 1 of 672,000 trajectories
    is off by 8.4 mm / 1.7e-3 rad (its speed passes within 3e-8 m/s of the 0.2 m/s stopping threshold, float32 takes
    the other controller branch at one step), all others <= 3.4e-5 m.  Use float64 where exact labels matter.
  * CPU, 1 thread: B = 1 ~15-20 ms, B = 600 ~35-40 ms per call (float32 ~ float64; 40 sequential steps of
    elementwise ops over B, profile fit B x 40 x 40).

Differentiability
  Differentiable w.r.t. traj and ego except at the stopping-controller switch (hard mask), the steering clip (zero
  gradient while saturated) and the heading unwrap (piecewise constant).  Returned headings are principal values
  (as the official); wrap heading differences.  Non-finite inputs are not handled (the scorer callers replace a
  non-finite trajectory by zeros, f1_common.process).

Helpers
  executed_progress(states) -> executed 4 s rear-axle arc length S_exec [B] and cumulative s [B, 41];
  executed_geometry(states, sdf) -> the same + the 4 footprint corners' SDF per state (refiner.sdf.sample_sdf).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from ...refiner import sdf as SDF
from ...refiner.geometry import dense_reference, safe_norm

# ----------------------------------------------------------------------------------------- official constants
NT = 41                                  # simulated states, t = 0.1 n s (n = 0..40)
N_STEPS = NT - 1                         # proposal_sampling.num_poses
N_PROF = NT - 1                          # velocity / curvature profile length (one per reference displacement)
DT = 0.1                                 # proposal_sampling.interval_length = tracker._discretization_time [s]
DT_SIM = 100000 * 1e-6                   # bicycle sampling_time.time_s (TimePoint diff 100000 us) = 0.09999999999999999
WHEEL_BASE = 3.089                       # get_pacifica_parameters().wheel_base [m] (tracker + bicycle)
MAX_STEER = math.pi / 3                  # BatchKinematicBicycleModel max_steering_angle [rad]
TAU_ACCEL = 0.2                          # accel_time_constant [s]
TAU_STEER = 0.05                         # steering_angle_time_constant [s]
Q_LON, R_LON = 10.0, 1.0                 # BatchLQRTracker q_longitudinal / r_longitudinal
Q_LAT = (1.0, 10.0, 0.0)                 # q_lateral (lateral error, heading error, steering angle)
R_LAT = 1.0                              # r_lateral
HORIZON = 10                             # tracking_horizon [steps of DT]
JERK_PENALTY = 1e-4
CURV_RATE_PENALTY = 1e-2
INITIAL_CURVATURE_PENALTY = 1e-10        # batch_lqr_utils.INITIAL_CURVATURE_PENALTY
STOP_GAIN = 0.5                          # stopping_proportional_gain
STOP_V = 0.2                             # stopping_velocity [m/s]
FITS = ("reduced", "official")
STATE_COLS = ("x", "y", "heading", "v", "a", "steer")
TWO_PI = 2.0 * math.pi


def principal_value(a: torch.Tensor) -> torch.Tensor:
    """nuplan principal_value: wrap to [-pi, pi) as (a + pi) mod 2 pi - pi (gradient 1)."""
    return torch.remainder(a + math.pi, TWO_PI) - math.pi


def normalize_angle(a: torch.Tensor) -> torch.Tensor:
    """pdm_geometry_utils.normalize_angle: atan2(sin a, cos a) in [-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


# ----------------------------------------------------------------------------------------- reference
def reference_states(traj: torch.Tensor) -> torch.Tensor:
    """Official tracker reference: traj [..., 8, 3] (x, y, heading at t = 0.5 k s, N frame) -> [..., 41, 3] at
    t = 0.1 n s (index 5 k = pose k, index 0 = the origin), heading principal-valued.  Same dtype as traj."""
    lead = traj.shape[:-2]
    d = dense_reference(traj.reshape(-1, 8, 3))
    d = torch.cat([d[..., :2], principal_value(d[..., 2:3])], -1)
    return d.reshape(lead + (NT, 3))


# ----------------------------------------------------------------------------------------- profiles
_K_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


def official_banded(number_rows: int) -> np.ndarray:
    """batch_lqr_utils._make_banded_difference_matrix as navsim implements it ([number_rows, number_rows + 1]):
    the +1 band is written first and then overwritten by the -eye block, leaving -I plus one +1 at the last row."""
    band = np.zeros((number_rows, number_rows + 1), dtype=np.float64)
    eye = np.eye(number_rows, dtype=np.float64)
    band[:, 1:] = eye
    band[:, :-1] = -eye
    return band


def _velocity_gain(device) -> torch.Tensor:
    """(I + lam (Rb D1)^T (Rb D1))^-1 [40, 40] float64 (symmetric), Rb = official_banded(38), D1 the 39 x 40 first
    difference, lam = jerk / dt^4 (cached per device)."""
    key = (str(device), N_PROF)
    if key not in _K_CACHE:
        D1 = np.diff(np.eye(N_PROF), axis=0)                                        # [39, 40]: (D1 u)_k = u_{k+1} - u_k
        D = official_banded(N_PROF - 2) @ D1                                        # [38, 40]
        lam = JERK_PENALTY / DT ** 4
        K = np.linalg.inv(np.eye(N_PROF) + lam * D.T @ D)
        _K_CACHE[key] = torch.from_numpy(0.5 * (K + K.T)).to(device)
    return _K_CACHE[key]


def _displacements(ref: torch.Tensor):
    """ref [B, 41, 3] float64 -> xy displacements [B, 40, 2], wrapped heading displacements [B, 40], headings
    h_0..h_39 [B, 40]."""
    dxy = ref[:, 1:, :2] - ref[:, :-1, :2]
    dh = normalize_angle(ref[:, 1:, 2] - ref[:, :-1, 2])
    return dxy, dh, ref[:, :-1, 2]


def _fit_reduced(ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dxy, dh, h = _displacements(ref)
    p = torch.cos(h) * dxy[..., 0] + torch.sin(h) * dxy[..., 1]                    # [B, 40] step along the heading
    v = (p @ _velocity_gain(ref.device)) / DT                                       # K symmetric -> u = K p
    w = v * DT                                                                      # [B, 40]
    lam1 = CURV_RATE_PENALTY / DT ** 2
    n = N_PROF
    diag = w * w + lam1 * torch.cat([w.new_ones(1), 2.0 * w.new_ones(n - 2), w.new_ones(1)])
    diag = diag + torch.cat([w.new_full((1,), INITIAL_CURVATURE_PENALTY), w.new_zeros(n - 1)])
    M = torch.diag_embed(diag)
    off = torch.full((n - 1,), -lam1, dtype=w.dtype, device=w.device)
    M = M + torch.diag(off, 1) + torch.diag(off, -1)
    kappa = torch.linalg.solve(M, (w * dh)[..., None])[..., 0]
    return v, kappa


def _fit_official(ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Literal port of batch_lqr_utils.get_velocity_curvature_profiles_with_derivatives_from_poses (float64)."""
    B = ref.shape[0]
    dxy, dh, h = _displacements(ref)
    M = dxy.shape[1]                                                                # 40 displacements
    y = dxy.reshape(B, -1)                                                          # [B, 80] (dx_0, dy_0, ...)
    col = torch.stack([torch.cos(h), torch.sin(h)], -1).reshape(B, -1)              # [B, 80]
    A = (col[..., None] * DT ** 2).repeat(1, 1, M)
    A[..., 0] = col * DT
    mask = torch.triu(torch.ones(M, M, dtype=torch.bool, device=ref.device), 1).repeat_interleave(2, 0)
    A = A.masked_fill(mask[None], 0.0)
    band = torch.from_numpy(official_banded(M - 2)).to(device=ref.device, dtype=ref.dtype)   # [38, 39]
    R = torch.cat([torch.zeros(M - 2, 1, dtype=ref.dtype, device=ref.device), band], 1)   # [38, 40]
    AT = A.transpose(1, 2)
    x = torch.bmm(torch.bmm(torch.inverse(torch.bmm(AT, A) + JERK_PENALTY * (R.T @ R)[None]), AT), y[..., None])[..., 0]
    v = x[:, :1] + torch.cat([x.new_zeros(B, 1), torch.cumsum(x[:, 1:] * DT, -1)], -1)   # [B, 40]
    Ac = torch.tril(torch.ones(M, M, dtype=ref.dtype, device=ref.device)).repeat(B, 1, 1)
    Ac[:, :, 0] = v * DT
    Ac[:, 1:, 1:] = Ac[:, 1:, 1:] * (v * DT ** 2)[:, 1:, None]
    Q = CURV_RATE_PENALTY * torch.eye(M, dtype=ref.dtype, device=ref.device)
    Q[0, 0] = INITIAL_CURVATURE_PENALTY
    AcT = Ac.transpose(1, 2)
    xc = torch.bmm(torch.bmm(torch.inverse(torch.bmm(AcT, Ac) + Q[None]), AcT), dh[..., None])[..., 0]
    kappa = xc[:, :1] + torch.cat([xc.new_zeros(B, 1), torch.cumsum(xc[:, 1:] * DT, -1)], -1)
    return v, kappa


def fit_profiles(ref: torch.Tensor, fit: str = "reduced") -> Tuple[torch.Tensor, torch.Tensor]:
    """ref [B, 41, 3] (reference_states) -> (velocity v_k [B, 40] m/s, curvature kappa_k [B, 40] 1/m), float64,
    k = 0..39 (the official tracker's _velocity_profile / _curvature_profile)."""
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
    r = ref.to(torch.float64)
    return _fit_reduced(r) if fit == "reduced" else _fit_official(r)


# ----------------------------------------------------------------------------------------- simulation
def _excl_cumsum(x: torch.Tensor) -> torch.Tensor:
    """[B, H] -> [B, H] sum_{j < k} x_j (sequential order, as the official step-by-step recurrence)."""
    return torch.cat([torch.zeros_like(x[:, :1]), torch.cumsum(x[:, :-1], 1)], 1)


def _ego_columns(ego, lead, dtype, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e = torch.as_tensor(ego, device=device).to(dtype)
    if e.dim() == len(lead):                                                        # v0 only
        e = e[..., None]
    if e.shape[-1] > 3:
        raise ValueError(f"ego must be [..., k] with k <= 3 columns (v0, a0, steer0), got {tuple(e.shape)}")
    e = torch.broadcast_to(e, tuple(lead) + (e.shape[-1],)).reshape(-1, e.shape[-1])
    zero = torch.zeros_like(e[:, 0])
    v0 = e[:, 0]
    a0 = e[:, 1] if e.shape[1] > 1 else zero
    s0 = e[:, 2] if e.shape[1] > 2 else zero
    return v0, a0, s0


def simulate(traj: torch.Tensor, ego, dtype: Optional[torch.dtype] = None, fit: str = "reduced",
             return_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Official PDM simulation of submitted trajectories, batched (no loop over B; 40 sequential steps).

    traj  : [..., 8, 3] (x, y, heading) at t = 0.5 k s, k = 1..8, N frame (finite).
    ego   : [..., k] k = 1..3 columns (v0 m/s, a0 m/s^2, steer0 rad; missing = 0), or [...] = v0 only; broadcast to
            traj's leading dims (e.g. [T, 1, 2] for traj [T, K, 8, 3]).  v0 / a0 = rear-axle longitudinal velocity /
            acceleration in the ego frame (status[..., 4] / status[..., 6]).
    dtype : rollout dtype (default traj.dtype if floating, float32 otherwise); profiles are fitted in float64.
    fit   : 'reduced' (default) | 'official' (literal normal equations), see the module docstring.
    returns states [..., 41, 6] (x, y, heading, v, a, steer) at t = 0.1 n s in N; row 0 = (0, 0, 0, v0, a0, steer0).
    return_info adds a dict: ref [.., 41, 3], v_prof / kappa_prof [.., 40] (float64), a_cmd / steer_rate_cmd [.., 40]
    (commands of step n -> n + 1), stop [.., 40] bool (stopping controller used).
    """
    if traj.shape[-2:] != (8, 3):
        raise ValueError(f"traj must be [..., 8, 3], got {tuple(traj.shape)}")
    lead = traj.shape[:-2]
    if dtype is None:
        dtype = traj.dtype if traj.is_floating_point() else torch.float32
    dev = traj.device
    tr = traj.reshape(-1, 8, 3).to(dtype)
    Bn = tr.shape[0]
    ref = reference_states(tr.to(torch.float64))                                    # [B, 41, 3] float64
    vp64, kp64 = fit_profiles(ref, fit)
    ref_d = ref.to(dtype)
    vp, kp = vp64.to(dtype), kp64.to(dtype)
    kp_pad = torch.cat([kp, kp[:, -1:].expand(Bn, HORIZON)], 1)                     # held at kappa_39 past the end
    v0, a0, s0 = _ego_columns(ego, lead, dtype, dev)

    L = WHEEL_BASE
    b2_before = _excl_cumsum(torch.full((1, HORIZON), DT, dtype=dtype, device=dev))  # [1, 10] k dt (official sum)
    B_lon = HORIZON * DT
    k_lon = -1.0 / (B_lon * Q_LON * B_lon + R_LON)
    g_acc = DT_SIM / (DT_SIM + TAU_ACCEL)
    g_str = DT_SIM / (DT_SIM + TAU_STEER)
    q0, q1 = Q_LAT[0], Q_LAT[1]

    x = torch.zeros(Bn, dtype=dtype, device=dev)
    y = torch.zeros_like(x)
    h = torch.zeros_like(x)
    v, a, st = v0, a0, s0
    out = [torch.stack([x, y, h, v, a, st], -1)]
    a_cmds, sr_cmds, stops = [], [], []
    for n in range(N_STEPS):
        # ---- tracker (BatchLQRTracker.track_trajectory at iteration n)
        xr, yr, hr = ref_d[:, n, 0], ref_d[:, n, 1], ref_d[:, n, 2]
        dx, dy = x - xr, y - yr
        e_lat = -dx * torch.sin(hr) + dy * torch.cos(hr)
        e_head = normalize_angle(h - hr)
        r = min(n + HORIZON, N_PROF - 1)
        v_ref = vp[:, r]
        kap = kp_pad[:, n:n + HORIZON]                                              # [B, 10]
        stop = (v_ref <= STOP_V) & (v <= STOP_V)
        a_lqr = k_lon * B_lon * Q_LON * (v - v_ref)                                 # _solve_one_step_longitudinal_lqr
        a_cmd = torch.where(stop, -STOP_GAIN * (v - v_ref), a_lqr)
        vk = v[:, None] + _excl_cumsum(a_lqr[:, None].expand(Bn, HORIZON) * DT)     # tracker velocity over the horizon
        m01 = vk * DT                                                               # state matrix (0, 1)
        m12 = vk * DT / L                                                           # state matrix (1, 2)
        aff1 = -vk * kap * DT                                                       # affine term (heading row)
        # products of the unit upper-triangular step matrices, A = M_9 .. M_0, B, g (official recurrence, closed form)
        m12_ex = _excl_cumsum(m12)                                                  # A[1, 2] before step k
        A01 = m01.sum(1)
        A12 = m12.sum(1)
        A02 = (m01 * m12_ex).sum(1)
        b1_ex = _excl_cumsum(m12 * b2_before)                                       # B[1] before step k
        B0 = (m01 * b1_ex).sum(1)
        B1 = (m12 * b2_before).sum(1)
        g1_ex = _excl_cumsum(aff1)
        G0 = (m01 * g1_ex).sum(1)
        G1 = aff1.sum(1)
        err0 = e_lat + A01 * e_head + A02 * st + G0
        err1 = normalize_angle(e_head + A12 * st + G1)
        # err2 = wrap(steer) has weight Q_LAT[2] = 0
        bqb = B0 * q0 * B0 + B1 * q1 * B1
        sr_lqr = -1.0 / (bqb + R_LAT) * (B0 * q0 * err0 + B1 * q1 * err1)
        sr_cmd = torch.where(stop, torch.zeros_like(sr_lqr), sr_lqr)
        # ---- motion model (BatchKinematicBicycleModel.propagate_state)
        a_new = g_acc * (a_cmd - a) + a
        st_target = DT_SIM * sr_cmd + st
        st_upd = g_str * (st_target - st) + st
        sr_new = (st_upd - st) / DT_SIM
        x_n = x + v * torch.cos(h) * DT_SIM
        y_n = y + v * torch.sin(h) * DT_SIM
        h_n = principal_value(h + v * torch.tan(st) / L * DT_SIM)
        v_n = v + a_new * DT_SIM
        st_n = torch.clamp(st + sr_new * DT_SIM, -MAX_STEER, MAX_STEER)
        x, y, h, v, a, st = x_n, y_n, h_n, v_n, a_new, st_n
        out.append(torch.stack([x, y, h, v, a, st], -1))
        if return_info:
            a_cmds.append(a_cmd)
            sr_cmds.append(sr_cmd)
            stops.append(stop)
    states = torch.stack(out, 1).reshape(tuple(lead) + (NT, 6))
    if not return_info:
        return states
    info = dict(ref=ref_d.reshape(tuple(lead) + (NT, 3)),
                v_prof=vp64.reshape(tuple(lead) + (N_PROF,)), kappa_prof=kp64.reshape(tuple(lead) + (N_PROF,)),
                a_cmd=torch.stack(a_cmds, 1).reshape(tuple(lead) + (N_STEPS,)),
                steer_rate_cmd=torch.stack(sr_cmds, 1).reshape(tuple(lead) + (N_STEPS,)),
                stop=torch.stack(stops, 1).reshape(tuple(lead) + (N_STEPS,)))
    return states, info


def reference_rollout(traj: torch.Tensor, ego, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Tracker-free stand-in of `simulate` (exec_model 'ref'; refiner v5 student, user 2026-10-20: the student reads the
    plan, the teachers keep the LQR): the tracker's own 41-state reference (`reference_states`: origin + the 8 poses,
    positions linear in time, heading unwrapped then linear) as the states, so every 41-state reader / loss runs
    unchanged.  Report 89 §3-2 measured this read against the executed one (GT fields, held-out 12,179).
      x, y   : the reference positions;
      heading: the reference heading changes clamped to the bicycle's turn limit |dh_n| <= len_n tan(MAX_STEER) /
               WHEEL_BASE (len_n = |p_n - p_{n-1}|): equal to the reference wherever its curvature is drivable (radius
               >= 1.78 m), but a near-stop plan whose knot headings flip (v2 stop parents, ~2.9 rad) keeps the ego
               pointing forward instead of a mirrored footprint (review 2026-10-20); principal-valued;
      v      : v_0 = v0, v_n = (p_{n+1} - p_n) . (cos h_n, sin h_n) / DT for n = 1..39 (signed; the speed that moves
               state n, as the bicycle's), v_40 = v_39;
      a      : a_0 = a0, a_n = (v_n - v_{n-1}) / DT -- impulses at the knots, NOT a tracked acceleration (comfort
               cannot be judged without the tracker: refine_loss drops its comfort factor for 'ref');
      steer  : steer_0 = steer0, 0 after (no reader uses it).
    traj [..., 8, 3], ego as `simulate` -> states [..., 41, 6] in `dtype` (default traj.dtype if floating, else
    float32); differentiable w.r.t. traj (finite gradient at standstill).  Tens of GPU launches against `simulate`'s
    4,741 (2026-10-20 profile)."""
    if traj.shape[-2:] != (8, 3):
        raise ValueError(f"traj must be [..., 8, 3], got {tuple(traj.shape)}")
    lead = traj.shape[:-2]
    if dtype is None:
        dtype = traj.dtype if traj.is_floating_point() else torch.float32
    ref = reference_states(traj.reshape(-1, 8, 3).to(dtype))                        # [B, 41, 3]
    v0, a0, s0 = _ego_columns(ego, lead, dtype, traj.device)
    d = ref[:, 1:, :2] - ref[:, :-1, :2]                                           # [B, 40, 2] step n -> n + 1
    lim = safe_norm(d) * (math.tan(MAX_STEER) / WHEEL_BASE)
    dh = principal_value(ref[:, 1:, 2] - ref[:, :-1, 2])
    dh = torch.maximum(torch.minimum(dh, lim), -lim)
    h = torch.cat([ref[:, :1, 2], ref[:, :1, 2] + torch.cumsum(dh, 1)], 1)         # [B, 41] (ref heading 0 at n = 0)
    vs = (d[..., 0] * torch.cos(h[:, :-1]) + d[..., 1] * torch.sin(h[:, :-1])) / DT  # [B, 40] speed of step n -> n + 1
    v = torch.cat([v0[:, None], vs[:, 1:], vs[:, -1:]], 1)                          # [B, 41]
    a = torch.cat([a0[:, None], (v[:, 1:] - v[:, :-1]) / DT], 1)
    st = torch.cat([s0[:, None], torch.zeros_like(v[:, 1:])], 1)
    out = torch.cat([ref[..., :2], principal_value(h)[..., None], v[..., None], a[..., None], st[..., None]], -1)
    return out.reshape(tuple(lead) + (NT, 6))


EXECUTORS = {"lqr": "simulate", "ref": "reference_rollout"}


def executor(exec_model: str):
    """exec_model 'lqr' -> simulate, 'ref' -> reference_rollout (both (traj, ego, dtype=) -> [..., 41, 6])."""
    if exec_model not in EXECUTORS:
        raise ValueError(f"exec_model {exec_model!r} has no 41-state executor (one of {tuple(EXECUTORS)})")
    return globals()[EXECUTORS[exec_model]]


class LQRSimulator(nn.Module):
    """nn.Module wrapper of `simulate` (no parameters, no buffers: attaching it leaves state dicts unchanged)."""

    def __init__(self, fit: str = "reduced", dtype: Optional[torch.dtype] = None):
        super().__init__()
        if fit not in FITS:
            raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
        self.fit = fit
        self.dtype = dtype

    def forward(self, traj: torch.Tensor, ego, return_info: bool = False):
        return simulate(traj, ego, dtype=self.dtype, fit=self.fit, return_info=return_info)


# ----------------------------------------------------------------------------------------- executed-state helpers
def executed_progress(states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """states [..., 41, >=2] -> (S_exec [...] = rear-axle path length over the 4 s (m, unsigned; a reversing segment
    counts positive), s [..., 41] cumulative arc length, s[..., 0] = 0).  Finite (zero) gradient at standstill."""
    seg = safe_norm(states[..., 1:, :2] - states[..., :-1, :2])                     # [..., 40]
    s = torch.cat([torch.zeros_like(seg[..., :1]), torch.cumsum(seg, -1)], -1)
    return s[..., -1], s


def executed_geometry(states: torch.Tensor, sdf: Optional[torch.Tensor] = None,
                      sdf_index: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Executed-path summary of simulated states [B, 41, >=3] (N frame).

    returns progress [B] (S_exec, m), s [B, 41] (cumulative arc length, m) and, with an SDF field (refiner.sdf E grid,
    [S, 320, 256] or [320, 256]; sdf_index [B] as in sample_sdf), the official footprint's 4 corners per state
    (refiner.sdf.ego_corners: FL, RL, RR, FR; half length 2.588, half width 1.1485, centre 1.461 m ahead of the rear
    axle): corner_sdf [B, 41, 4] (m, + inside the DAC drivable union), corner_valid [B, 41, 4] (in grid and finite),
    min_corner_sdf [B, 41] (min over the valid corners, +inf where none is valid; DAC fails iff min over the states of
    min_corner_sdf < 0 up to grid resolution, sdf.py)."""
    prog, s = executed_progress(states)
    out = dict(progress=prog, s=s)
    if sdf is not None:
        val, ok = SDF.sample_sdf(sdf, SDF.ego_corners(states[..., :3]), sdf_index)
        out["corner_sdf"] = val
        out["corner_valid"] = ok
        out["min_corner_sdf"] = torch.where(ok, val, torch.full_like(val, float("inf"))).amin(-1)
    return out


__all__ = ["simulate", "reference_rollout", "executor", "EXECUTORS", "reference_states", "fit_profiles", "executed_progress", "executed_geometry", "LQRSimulator",
           "official_banded", "principal_value", "normalize_angle", "STATE_COLS", "NT", "FITS"]
