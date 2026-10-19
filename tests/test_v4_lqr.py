"""refiner v4_3 M1: batched torch port of the official PDM simulation (navsim/agents/para_ssr/ck/v4/lqr.py).

  * constants / the navsim banded "jerk" matrix are the official objects' values;
  * synthetic edge cases (standstill, creep with heading jitter, reverse, U-turn through +-pi, steer saturation,
    lateral offset, hard brake, v0 < 0, a0 != 0, steer0 != 0) vs the official numpy chain (transform_trajectory +
    get_trajectory_as_array + PDMSimulator.simulate_proposals) run at a far-from-origin global ego pose and mapped to N;
  * F1 parity: official LQR-tracked states stored by CK/ck2/analysis/detmap_path_ideas/F1 (out/<split>/run.npz,
    float32 in N; par16 + var80 + latT16 (+ latM16) per token) on 32 tokens per split (24 seeded random + the 4 lowest
    and 4 highest v0, i.e. stopping-controller rows included);
  * shapes / broadcasting / module without parameters, gradients (gradcheck), executed progress + corner SDF,
    CPU runtime (B = 1, 600).
CPU only, read-only on every input; real-data tests skip when F1 / the trajectory sources are absent.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_lqr.py
  (-s prints the parity / runtime numbers)
"""
from __future__ import annotations

import math
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.ck.v4 import lqr as LQ
from navsim.agents.para_ssr.refiner import sdf as SDF

CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
F1 = CK / "ck2/analysis/detmap_path_ideas/F1/out"
_A = CK / "ck2/analysis/refiner_dm_traits/infer"
_LAD = CK / "ck2/eval/sd_plugin/ladder/ck2e2e_main"
# = f1_common.sources(split)['traj'] (group order par, var, latT, latM as in F1 run.npz 'groups')
SOURCES = {
    "val": dict(par=CK / "packed/navtrain_val/cand.npy", var=_A / "navtrain_val/r34_variants/traj80.npy",
                latT=_A / "navtrain_val/ck2T10/r34/lat_traj16.npy"),
    "train": dict(par=CK / "packed/navtrain_train/cand.npy", var=_A / "navtrain_train/r34_variants/traj80.npy",
                  latT=_A / "navtrain_train/ck2T10/r34/lat_traj16.npy",
                  latM=_A / "navtrain_train/ck2M10/r34/lat_traj16.npy"),
    "navtest": dict(par=_LAD / "root_navtest/packed/navtest/cand.npy", var=_LAD / "navtest/r34_variants/traj80.npy",
                    latT=_LAD / "navtest/ck2T10/r34/lat_traj16.npy", latM=_LAD / "navtest/ck2M10/r34/lat_traj16.npy"),
}
N_RANDOM, N_EXTREME = 24, 4
TOL_XY_F64, TOL_H_F64 = 1e-3, 1e-3          # spec M1 (float64)
TOL_XY_F32, TOL_H_F32 = 1e-3, 1e-3          # float32 on these samples (measured max 3.1e-5 m; over all F1 rows 1 of
                                             # 1,376,000 is 8.4 mm off: a stopping-threshold branch flip, lqr.py)


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ----------------------------------------------------------------------------------------- official chain (numpy)
def _official_states(traj: np.ndarray, v0: float, a0: float, steer0: float = 0.0,
                     ego_pose=(664_000.25, 3_997_000.75, 2.5), t0_us: int = 0) -> np.ndarray:
    """traj [K, 8, 3] float32 (N frame) -> official tracked states [K, 41, 6] (x, y, heading, v, a, steer) in N,
    simulated in the global frame of an ego at ego_pose (the scorer's own functions)."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex as SI
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    samp = TrajectorySampling(num_poses=40, interval_length=0.1)
    es = EgoState.build_from_rear_axle(StateSE2(*ego_pose), StateVector2D(float(v0), 0.0),
                                       StateVector2D(float(a0), 0.0), float(steer0), TimePoint(t0_us),
                                       get_pacifica_parameters())
    samp8 = TrajectorySampling(num_poses=8, interval_length=0.5)
    ref = np.concatenate([get_trajectory_as_array(transform_trajectory(Trajectory(d, samp8), es), samp,
                                                  es.time_point)[None] for d in traj], 0)
    sim = PDMSimulator(samp).simulate_proposals(ref, es)
    x0, y0, h0 = ego_pose
    c, s = math.cos(h0), math.sin(h0)
    dx, dy = sim[..., SI.X] - x0, sim[..., SI.Y] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy, _wrap(sim[..., SI.HEADING] - h0), sim[..., SI.VELOCITY_X],
                     sim[..., SI.ACCELERATION_X], sim[..., SI.STEERING_ANGLE]], -1)


def _synthetic():
    """[K, 8, 3] float32 edge-case trajectories + per-row (v0, a0, steer0)."""
    t = np.arange(1, 9) * 0.5
    rows, ego = [], []

    def arc(s, kappa, y_off=0.0):
        s = np.asarray(s, np.float64)
        if abs(kappa) < 1e-9:
            x, y, h = s, np.zeros_like(s), np.zeros_like(s)
        else:
            h = kappa * s
            x, y = np.sin(h) / kappa, (1 - np.cos(h)) / kappa
        return np.stack([x, y + y_off * np.minimum(t / 2.0, 1.0), h], -1)

    rows.append(np.zeros((8, 3))); ego.append((0.0, 0.0, 0.0))                       # standstill, stopping controller
    rows.append(np.zeros((8, 3))); ego.append((0.15, -0.1, 0.0))                     # creeping ego, stopped plan
    rows.append(np.zeros((8, 3))); ego.append((5.0, 0.0, 0.0))                       # moving ego, stopped plan
    jit = np.random.default_rng(1)
    c = np.stack([np.linspace(0.01, 0.08, 8), jit.normal(0, 0.01, 8), jit.normal(0, 0.3, 8)], -1)
    rows.append(c); ego.append((0.05, 0.2, 0.0))                                     # creep with heading jitter
    rows.append(arc(-1.0 * t, 0.0)); ego.append((-0.5, 0.0, 0.0))                    # reverse, v0 < 0
    rows.append(arc(8.0 * t, 0.0)); ego.append((8.0, 0.0, 0.0))                      # straight cruise
    rows.append(arc(8.0 * t - 0.8 * t ** 2, 0.0)); ego.append((8.0, -1.0, 0.0))      # hard brake
    rows.append(arc(2.0 * t + 1.2 * t ** 2, 0.0)); ego.append((2.0, 1.5, 0.0))       # strong acceleration
    rows.append(arc(6.0 * t, 1 / 6.0)); ego.append((6.0, 0.0, 0.0))                  # U-turn: heading through pi
    rows.append(arc(4.0 * t, -0.6)); ego.append((4.0, 0.3, 0.0))                     # tight right turn, steer saturates
    rows.append(arc(10.0 * t, 0.0, y_off=0.5)); ego.append((10.0, 0.0, 0.0))         # l+0.5-like lateral shift
    rows.append(arc(10.0 * t, 0.02, y_off=-1.5)); ego.append((10.0, 0.0, 0.05))      # lateral shift + steer0 != 0
    rows.append(arc(12.0 * t, 0.01)); ego.append((14.0, -2.0, -0.02))               # plan slower than ego
    return np.stack(rows).astype(np.float32), np.asarray(ego, np.float64)


@pytest.fixture(scope="module")
def synthetic_official():
    traj, ego = _synthetic()
    off = np.stack([_official_states(traj[k:k + 1], *ego[k])[0] for k in range(len(traj))])
    return traj, ego, off


def _errs(st: np.ndarray, ref: np.ndarray):
    dxy = np.hypot(st[..., 0] - ref[..., 0], st[..., 1] - ref[..., 1])
    dh = np.abs(_wrap(st[..., 2] - ref[..., 2]))
    return dxy, dh


# ----------------------------------------------------------------------------------------- constants
def test_constants_match_official():
    from navsim.planning.simulation.planner.pdm_planner.simulation import batch_lqr_utils as U
    from navsim.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import (
        BatchKinematicBicycleModel,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTracker
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from omegaconf import OmegaConf

    trk, bic = BatchLQRTracker(), BatchKinematicBicycleModel()
    assert (trk._q_longitudinal, trk._r_longitudinal) == (LQ.Q_LON, LQ.R_LON)
    assert np.array_equal(np.diag(trk._q_lateral), np.asarray(LQ.Q_LAT)) and trk._r_lateral[0, 0] == LQ.R_LAT
    assert trk._tracking_horizon == LQ.HORIZON and trk._discretization_time == LQ.DT
    assert (trk._jerk_penalty, trk._curvature_rate_penalty) == (LQ.JERK_PENALTY, LQ.CURV_RATE_PENALTY)
    assert (trk._stopping_proportional_gain, trk._stopping_velocity) == (LQ.STOP_GAIN, LQ.STOP_V)
    assert trk._wheel_base == LQ.WHEEL_BASE == get_pacifica_parameters().wheel_base == bic._vehicle.wheel_base
    assert bic._max_steering_angle == LQ.MAX_STEER
    assert (bic._accel_time_constant, bic._steering_angle_time_constant) == (LQ.TAU_ACCEL, LQ.TAU_STEER)
    assert U.INITIAL_CURVATURE_PENALTY == LQ.INITIAL_CURVATURE_PENALTY
    assert (TimePoint(200000) - TimePoint(100000)).time_s == LQ.DT_SIM
    assert np.array_equal(U._make_banded_difference_matrix(38), LQ.official_banded(38))
    # the navsim vectorisation differs from the nuplan-devkit loop (a true first difference): the quirk is official
    from nuplan.planning.simulation.controller.tracker import tracker_utils as NU
    assert not np.array_equal(NU._make_banded_difference_matrix(38), LQ.official_banded(38))
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1]
                         / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    assert cfg.proposal_sampling.num_poses == LQ.N_STEPS and cfg.proposal_sampling.interval_length == LQ.DT


def test_profiles_reduced_equals_official_and_navsim():
    """Both fits == the navsim least squares (float64) on synthetic references; the reduced one is well-conditioned."""
    from navsim.planning.simulation.planner.pdm_planner.simulation import batch_lqr_utils as U

    traj, _ = _synthetic()
    ref = LQ.reference_states(torch.from_numpy(traj).double())
    vn, _, kn, _ = U.get_velocity_curvature_profiles_with_derivatives_from_poses(LQ.DT, ref.numpy(), LQ.JERK_PENALTY,
                                                                                 LQ.CURV_RATE_PENALTY)
    for fit in LQ.FITS:
        v, k = LQ.fit_profiles(ref, fit)
        assert v.dtype == torch.float64 and v.shape == (len(traj), 40) and k.shape == (len(traj), 40)
        np.testing.assert_allclose(v.numpy(), vn, atol=1e-9, rtol=0)
        # standstill rows (w ~ 0): kappa is set by the 1e-10 regulariser only; both fits agree where it matters (w > 0)
        moving = np.abs(vn) > 0.05
        np.testing.assert_allclose(k.numpy()[moving], kn[moving], atol=1e-7, rtol=1e-7)


def test_reference_matches_official_interpolation():
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    traj, _ = _synthetic()
    es = EgoState.build_from_rear_axle(StateSE2(0.0, 0.0, 0.0), StateVector2D(1.0, 0.0), StateVector2D(0.0, 0.0), 0.0,
                                       TimePoint(0), get_pacifica_parameters())
    off = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(d, TrajectorySampling(num_poses=8,
                                                                                                 interval_length=0.5)),
                                                                    es), TrajectorySampling(num_poses=40,
                                                                                            interval_length=0.1),
                                            es.time_point)[:, :3] for d in traj])
    mine = LQ.reference_states(torch.from_numpy(traj).double()).numpy()
    assert np.abs(mine[..., :2] - off[..., :2]).max() < 1e-12
    assert np.abs(_wrap(mine[..., 2] - off[..., 2])).max() < 1e-12
    assert (mine[..., 2] >= -math.pi).all() and (mine[..., 2] < math.pi).all()


# ----------------------------------------------------------------------------------------- synthetic parity
@pytest.mark.parametrize("fit", LQ.FITS)
def test_synthetic_vs_official(synthetic_official, fit):
    traj, ego, off = synthetic_official
    t = torch.from_numpy(traj).double()
    e = torch.from_numpy(ego)
    st, info = LQ.simulate(t, e, dtype=torch.float64, fit=fit, return_info=True)
    st = st.numpy()
    dxy, dh = _errs(st, off)
    print(f"\n[synthetic {fit} f64] max dxy {dxy.max():.3e} m  dh {dh.max():.3e} rad"
          f"  dv {np.abs(st[..., 3] - off[..., 3]).max():.3e}  da {np.abs(st[..., 4] - off[..., 4]).max():.3e}"
          f"  dsteer {np.abs(st[..., 5] - off[..., 5]).max():.3e}")
    assert dxy.max() < 1e-7 and dh.max() < 1e-8
    np.testing.assert_allclose(st[..., 3:], off[..., 3:], atol=1e-7, rtol=0)
    # the edge cases are exercised: stopping controller, steering saturation, heading through +-pi
    assert bool(info["stop"][0].all()) and not bool(info["stop"][5].any())
    assert np.isclose(np.abs(st[9, :, 5]).max(), LQ.MAX_STEER)
    assert np.abs(np.diff(np.unwrap(st[8, :, 2]))).max() < 0.5 and np.unwrap(st[8, :, 2]).max() > math.pi
    s32 = LQ.simulate(t.float(), e.float(), dtype=torch.float32, fit=fit).double().numpy()
    dxy32, dh32 = _errs(s32, off)
    print(f"[synthetic {fit} f32] max dxy {dxy32.max():.3e} m  dh {dh32.max():.3e} rad")
    assert dxy32.max() < 1e-3 and dh32.max() < 1e-4


# ----------------------------------------------------------------------------------------- F1 parity
def _npz_member(path: Path, key: str) -> np.ndarray:
    """Read-only memmap of an uncompressed (np.savez) npz member (no full load of the 0.4-0.7 GB arrays)."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(key + ".npy")
    assert info.compress_type == zipfile.ZIP_STORED, f"{path}:{key} is compressed"
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        hdr = f.read(30)
        n_name, n_extra = struct.unpack("<HH", hdr[26:30])
        f.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")


def _f1_sample(split: str):
    run = F1 / split / "run.npz"
    if not run.is_file() or not all(p.is_file() for p in SOURCES[split].values()):
        pytest.skip(f"F1 run / trajectory sources for {split} not found")
    import json

    with zipfile.ZipFile(run) as z:
        groups = json.loads(str(np.load(z.open("groups.npy"))))
    assert [g[0] for g in groups] == list(SOURCES[split])
    init = np.asarray(_npz_member(run, "init"))
    rows = np.asarray(_npz_member(run, "rows"))
    rng = np.random.default_rng(0)
    order = np.argsort(init[:, 0])
    pick = np.unique(np.concatenate([rng.choice(len(rows), N_RANDOM, replace=False), order[:N_EXTREME],
                                     order[-N_EXTREME:]]))
    sim = np.asarray(_npz_member(run, "sim")[pick], np.float64)                    # [n, K, 41, 6]
    fin = np.asarray(_npz_member(run, "fin")[pick])
    srcs = {g: np.load(p, mmap_mode="r") for g, p in SOURCES[split].items()}
    traj = np.concatenate([np.asarray(srcs[g][rows[pick]], np.float32) for g, _, _ in groups], 1)   # [n, K, 8, 3]
    fin_mine = np.isfinite(traj).reshape(traj.shape[0], traj.shape[1], -1).all(-1)
    assert np.array_equal(fin, fin_mine)
    traj = np.where(fin[..., None, None], traj, 0.0).astype(np.float32)            # = f1_common.process
    return traj, init[pick], sim, groups


@pytest.mark.parametrize("split", ["val", "train", "navtest"])
def test_f1_parity(split):
    traj, init, ref, groups = _f1_sample(split)
    n, K = traj.shape[:2]
    assert np.all(init[:, 2] == 0.0)                                               # tire steering angle 0 in navsim
    assert (ref[:, :, 0, :3] == 0.0).all()                                         # row 0 = ego origin
    np.testing.assert_allclose(ref[:, :, 0, 3], np.broadcast_to(init[:, None, 0], (n, K)), atol=0)
    t = torch.from_numpy(traj)
    ego = torch.from_numpy(init[:, None, :2])                                      # [n, 1, 2] broadcast over K
    res = {}
    for dt in (torch.float64, torch.float32):
        st = LQ.simulate(t.to(dt), ego.to(dt), dtype=dt).double().numpy()
        dxy, dh = _errs(st, ref)
        dv = np.abs(st[..., 3] - ref[..., 3])
        res[dt] = (dxy, dh)
        per_g = "  ".join(f"{g} {dxy[:, s:s + m].max():.2e}" for g, s, m in groups)
        print(f"\n[F1 {split} {str(dt)[6:]}] {n} tok x {K} = {n * K} traj  max dxy {dxy.max():.3e} m  "
              f"p99 {np.quantile(dxy.max(-1), 0.99):.3e}  max dh {dh.max():.3e} rad  max dv {dv.max():.3e}  [{per_g}]")
    dxy, dh = res[torch.float64]
    assert dxy.max() <= TOL_XY_F64 and dh.max() <= TOL_H_F64
    dxy, dh = res[torch.float32]
    assert dxy.max() <= TOL_XY_F32 and dh.max() <= TOL_H_F32


def test_f1_official_fit_matches_reduced():
    """fit='official' (literal normal equations) reproduces F1 val too and stays within 1e-6 m of 'reduced'."""
    traj, init, ref, _ = _f1_sample("val")
    t = torch.from_numpy(traj[:8]).double()
    ego = torch.from_numpy(init[:8, None, :2]).double()
    a = LQ.simulate(t, ego, fit="reduced").numpy()
    b = LQ.simulate(t, ego, fit="official").numpy()
    assert np.abs(a[..., :2] - b[..., :2]).max() < 1e-6
    assert _errs(b, ref[:8])[0].max() <= TOL_XY_F64


# ----------------------------------------------------------------------------------------- API
def test_shapes_broadcast_and_module():
    traj, ego = _synthetic()
    t = torch.from_numpy(traj).double()
    e = torch.from_numpy(ego)
    full = LQ.simulate(t, e)
    assert full.shape == (len(traj), 41, 6) and full.dtype == torch.float64
    np.testing.assert_array_equal(full[:, 0].numpy(), np.concatenate([np.zeros((len(traj), 3)), ego], 1))
    # [T, K] candidate sets with ego [T, 1, k]; v0-only ego; float32 default dtype from a float32 traj
    tk = t[:12].reshape(3, 4, 8, 3)
    ek = e[:12].reshape(3, 4, 3)[:, :1, :2].expand(3, 1, 2)
    out = LQ.simulate(tk, ek)
    ref = LQ.simulate(tk.reshape(12, 8, 3), ek.expand(3, 4, 2).reshape(12, 2))
    assert out.shape == (3, 4, 41, 6)
    assert torch.equal(out.reshape(12, 41, 6), ref)
    v_only = LQ.simulate(t, e[:, 0])
    assert torch.equal(v_only, LQ.simulate(t, torch.stack([e[:, 0], 0 * e[:, 0]], 1)))
    assert LQ.simulate(t.float(), e.float()).dtype == torch.float32
    with pytest.raises(ValueError):
        LQ.simulate(t, torch.zeros(len(traj), 8, dtype=torch.float64))             # e.g. the full status vector
    with pytest.raises(ValueError):
        LQ.simulate(t, e, fit="nope")
    m = LQ.LQRSimulator()
    assert len(list(m.parameters())) == 0 and len(m.state_dict()) == 0
    assert torch.equal(m(t, e), full)
    _, info = LQ.simulate(t, e, return_info=True)
    assert info["ref"].shape == (len(traj), 41, 3) and info["v_prof"].shape == (len(traj), 40)
    assert info["a_cmd"].shape == info["steer_rate_cmd"].shape == info["stop"].shape == (len(traj), 40)


def test_gradients():
    traj, ego = _synthetic()
    sel = [5, 10, 11]                                                              # moving, no stop / saturation
    t = torch.from_numpy(traj[sel]).double().requires_grad_(True)
    e = torch.from_numpy(ego[sel]).double().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: LQ.simulate(a, b)[:, ::5], (t, e), eps=1e-6, atol=1e-5)
    # every row (incl. standstill / saturation) yields finite gradients, also through the helpers
    ta = torch.from_numpy(traj).float().requires_grad_(True)
    st = LQ.simulate(ta, torch.from_numpy(ego).float())
    field = torch.zeros(320, 256)
    g = LQ.executed_geometry(st, field)
    (st.sum() + g["progress"].sum() + g["corner_sdf"].sum()).backward()
    assert torch.isfinite(ta.grad).all()


def test_executed_geometry():
    # straight constant-speed plan matched by the ego: progress = 4 s x v, corner SDF of a linear field y -> 3 - |y|
    t = torch.arange(1, 9, dtype=torch.float64) * 0.5
    v = 8.0
    traj = torch.stack([v * t, torch.zeros_like(t), torch.zeros_like(t)], -1)[None]
    st = LQ.simulate(traj, torch.tensor([[v, 0.0]], dtype=torch.float64))
    xs, ys = SDF.grid_centres()
    field = torch.from_numpy(np.broadcast_to(3.0 - np.abs(ys)[None, :], (320, 256)).copy())
    g = LQ.executed_geometry(st, field)
    assert g["s"].shape == (1, 41) and abs(float(g["progress"][0]) - 4.0 * v) < 1e-6
    assert torch.allclose(g["s"][0], torch.arange(41, dtype=torch.float64) * 0.1 * v, atol=1e-6)
    assert g["corner_sdf"].shape == (1, 41, 4) and bool(g["corner_valid"].all())
    assert torch.allclose(g["corner_sdf"], torch.full_like(g["corner_sdf"], 3.0 - SDF.EGO_HALF_WID), atol=1e-9)
    assert torch.allclose(g["min_corner_sdf"], torch.full((1, 41), 3.0 - SDF.EGO_HALF_WID, dtype=torch.float64))
    # a state far outside the grid: no valid corner -> +inf min; standstill: progress 0 with a finite gradient
    far = st.clone()
    far[0, 5, 0] = 500.0
    gf = LQ.executed_geometry(far, field)
    assert not bool(gf["corner_valid"][0, 5].any()) and math.isinf(float(gf["min_corner_sdf"][0, 5]))
    z = torch.zeros(1, 41, 6, dtype=torch.float64, requires_grad=True)
    p, _ = LQ.executed_progress(z)
    p.sum().backward()
    assert float(p) == 0.0 and torch.isfinite(z.grad).all()
    assert set(LQ.executed_geometry(st)) == {"progress", "s"}


def test_runtime_cpu():
    traj, ego = _synthetic()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(traj), 600)
    t = torch.from_numpy(traj[idx] + rng.normal(0, 0.05, (600, 8, 3)).astype(np.float32))
    e = torch.from_numpy(ego[idx]).float()
    msg = []
    for B in (1, 600):
        for dt in (torch.float32, torch.float64):
            LQ.simulate(t[:B].to(dt), e[:B].to(dt))                                 # warm-up (K cache)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                with torch.no_grad():
                    LQ.simulate(t[:B].to(dt), e[:B].to(dt))
                ts.append(time.perf_counter() - t0)
            msg.append(f"B={B} {str(dt)[6:]} {1e3 * min(ts):.1f} ms")
            assert min(ts) < 10.0
    print("\n[runtime CPU, 1 thread] " + "  ".join(msg))
