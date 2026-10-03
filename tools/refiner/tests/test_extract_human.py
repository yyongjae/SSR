"""Tests for tools/refiner/extract_human.py (CPU, < 1 min): synthetic log (circle drive with a frame gap and a
> pi heading change), real tokens vs navsim Scene.get_future_trajectory, the packed files and the validation report.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest \
          -q tools/refiner/tests/test_extract_human.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path("/home/external-user/yongjae/SSR")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(ROOT / "data/dataset/maps"))  # read by navsim at import time
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))
import extract_human as X  # noqa: E402

V, W = 5.0, 0.5            # m/s, rad/s  -> 8 s turn = 4 rad > pi
YAW0 = 2.9                 # global yaw at t0 (global yaw wraps past pi soon after t0)
G0 = np.array([100.0, -50.0])


def _frames(n_future=16, gap_at=6):
    """Synthetic log: 3 history frames, t0 frame (index 3), n_future future frames on a circle (speed V, yaw rate W).
    Future step gap_at (1-based) is 1.0 s instead of 0.5 s.  Returns frames and the true times of all frames."""
    from pyquaternion import Quaternion
    steps = [0.5] * 3 + [1.0 if k == gap_at else 0.5 for k in range(1, n_future + 1)]
    t = np.concatenate([[0.0], np.cumsum(steps)]) - 1.5          # t0 frame at t = 0
    R = V / W
    frames = []
    for i, ti in enumerate(t):
        xl, yl = R * np.sin(W * ti), R * (1 - np.cos(W * ti))    # N frame of t0
        c, s = np.cos(YAW0), np.sin(YAW0)
        gx, gy = G0 + [c * xl - s * yl, s * xl + c * yl]
        q = Quaternion(axis=[0, 0, 1], angle=YAW0 + W * ti)
        frames.append(dict(token=f"tok{i}", timestamp=int(round((ti + 10) * 1e6)), frame_idx=i,
                           ego2global_translation=np.array([gx, gy, 0.0]), ego2global_rotation=q.elements,
                           ego_dynamic_state=[V, 0.1, 0.3, V * W], driving_command=np.array([1, 0, 0, 0])))
    return frames, t


def test_synthetic_circle_with_gap():
    fr, t = _frames()
    d = X.extract_token(fr, 3)
    tf = t[4:]                                                   # true times of future frames
    R = V / W
    exp_xy = np.stack([R * np.sin(W * tf), R * (1 - np.cos(W * tf))], -1)
    assert d["traj"].dtype == np.float32 and d["traj"].shape == (8, 3)
    assert np.abs(d["traj"][:, :2] - exp_xy[:8]).max() < 1e-4
    assert np.abs(d["path"][:, :2] - exp_xy).max() < 1e-4
    h_true = W * tf
    assert np.abs(d["path"][:, 2] - h_true).max() < 1e-5               # unwrapped, reaches > pi
    assert d["path"][-1, 2] > np.pi
    assert np.all(np.abs(d["traj"][:, 2]) <= np.pi + 1e-6)
    wrapped = np.arctan2(np.sin(h_true[:8]), np.cos(h_true[:8]))
    assert np.abs(d["traj"][:, 2] - wrapped).max() < 1e-5
    chord = 2 * R * np.sin(W * np.diff(np.concatenate([[0.0], tf])) / 2)
    assert np.abs(d["path_s"] - np.concatenate([[0.0], np.cumsum(chord)])).max() < 1e-4
    assert d["n_avail"] == 16 and d["n_reg"] == 5
    assert abs(d["dt"][5] - 1.0) < 1e-6 and np.allclose(np.delete(d["dt"], 5), 0.5)
    assert d["frame_gap"] and d["gap4"] and not d["gap_hist"]
    assert abs(d["v0"] - np.hypot(V, 0.1)) < 1e-6 and abs(d["a0"] - 0.3) < 1e-7
    assert d["cmd"] == 0 and list(d["cmd_raw"]) == [1, 0, 0, 0]
    assert np.allclose(d["ego_global"][:2], G0) and d["log_pos"] == 3


def test_short_log_and_late_gap():
    fr, _ = _frames(n_future=12, gap_at=11)                     # gap outside the 5 s window, log ends at 6 s
    d = X.extract_token(fr, 3)
    assert d["n_avail"] == 12 and d["n_reg"] == 10
    assert not d["frame_gap"] and not d["gap4"]
    assert np.isnan(d["path"][12:]).all() and np.isnan(d["path_s"][13:]).all() and np.isnan(d["dt"][12:]).all()
    assert np.isfinite(d["path"][:12]).all()
    with pytest.raises(ValueError):
        X.extract_token(fr, 2)                                  # no full history
    with pytest.raises(ValueError):
        X.extract_token(fr, 6)                                  # < 10 future frames


@pytest.mark.skipif(not (X.SPLITS / "dev.parquet").exists(), reason="split files not built")
def test_real_tokens_match_scene_future_trajectory():
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    dv = pd.read_parquet(X.SPLITS / "dev.parquet")
    dv = dv[dv.map_location == "us-pa-pittsburgh-hazelwood"].iloc[[0, 50, 100]]
    for r in dv.itertuples():
        d = X.extract_log((r.log, [r.token], str(X.TRAIN_LOGS)))[0][1]
        sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                         log_names=[r.log], tokens=[r.token])
        sc = SceneLoader(data_path=X.TRAIN_LOGS, sensor_blobs_path=None, scene_filter=sf,
                         sensor_config=SensorConfig.build_no_sensors()).get_scene_from_token(r.token)
        ref = sc.get_future_trajectory(8).poses
        assert np.array_equal(d["traj"], ref.astype(np.float32))       # f32 cast of navsim's f64 poses
        assert np.abs(d["path"][:10, :2] - sc.get_future_trajectory(10).poses[:, :2]).max() < 1e-5
        assert d["frame_idx"] == r.frame_idx


@pytest.mark.skipif(not (X.OUT / "dev.npz").exists(), reason="human files not built")
@pytest.mark.parametrize("split", ["train", "dev"])
def test_packed_files(split):
    d = X.load_human(split)
    sp = pd.read_parquet(X.SPLITS / f"{split}.parquet", columns=["token", "log", "frame_idx"])
    n = len(sp)
    assert (d["tokens"] == sp.token.to_numpy()).all() and (d["logs"] == sp.log.to_numpy()).all()
    assert (d["frame_idx"].astype(np.int64) == sp.frame_idx.to_numpy()).all()
    shapes = dict(traj=(n, 8, 3), path=(n, 16, 3), path_s=(n, 17), dt=(n, 16), eds=(n, 4), cmd_raw=(n, 4),
                  ego_global=(n, 3), v0=(n,), a0=(n,), frame_gap=(n,))
    for k, s in shapes.items():
        assert d[k].shape == s, k
    assert np.isfinite(d["traj"]).all()
    assert np.array_equal(d["traj"][:, :, :2], d["path"][:, :8, :2])
    assert (d["n_avail"] >= 10).all() and (d["n_reg"] <= d["n_avail"]).all()
    ds = np.diff(d["path_s"], axis=1)
    assert np.all(ds[np.isfinite(ds)] >= 0) and np.all(d["path_s"][:, 0] == 0)
    reg = np.abs(d["dt"][:, :10] - 0.5) <= 0.1
    assert np.array_equal(d["frame_gap"], ~reg.all(1))
    assert np.allclose(d["v0"], np.hypot(d["eds"][:, 0], d["eds"][:, 1]), atol=1e-5)
    assert np.array_equal(d["a0"], d["eds"][:, 2])
    assert d["frame_gap"].mean() < 0.02
    assert len(d["index"]) == n


def test_validation_report():
    p = X.REPORT / "human_validation.json"
    if not p.exists():
        pytest.skip("validation not run")
    s = json.load(open(p))
    assert s["navtrain"]["n"] >= 50 and s["navtrain"]["scene_cmd_all_equal"]
    assert max(s["navtrain"]["max_abs"].values()) < 1e-5
    nt = s["navtest_vs_head_ablation_table"]
    assert nt["n"] >= 50 and nt["cmd_all_equal"] and max(nt["traj_xy_err"], nt["traj_h_err"], nt["a0_err"]) < 1e-6
