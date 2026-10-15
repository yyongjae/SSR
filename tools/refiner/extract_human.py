#!/usr/bin/env python
"""Human (expert) trajectory, human path up to 8 s and t0 ego state per token, from the raw navsim logs
(IMPL_SPEC §3.6 inputs).

Frames and units: N frame = NAVSIM ego frame at t0 (rear-axle origin, x forward, y left, heading CCW from +x, rad);
metres, seconds, m/s, m/s^2.  The t0 frame c is the token's position in the log list (navsim filter_scenes windows
[c-3, c+10], frame_interval 1); frames are taken BY INDEX exactly like Scene.get_future_trajectory (frame k after t0 is
nominally t = 0.5 k s; with a frame gap the nominal time is wrong -> gap flags below).
Global ego pose of a frame = (ego2global_translation[0], [1], Quaternion(ego2global_rotation).yaw_pitch_roll[0]),
converted to N with navsim's convert_absolute_to_relative_se2_array (float64), as navsim does.

Per token (row i of the packed arrays; rows in the order of splits/<split>.parquet):
  traj       [8,3]  f32  (x, y, heading) at t = 0.5..4.0 s (frames c+1..c+8); heading wrapped by normalize_angle
                         -> identical to Scene.get_future_trajectory(8).poses cast to f32 (validated).
  path       [16,3] f32  frames c+1..c+16 (<= 8 s) in N; x, y as traj; heading UNWRAPPED (continuous from 0 at t0);
                         NaN beyond the end of the log.  path[:8, :2] == traj[:, :2].
  path_s     [17]   f32  arc length (m) along the polyline [origin, path_1..path_16] at its vertices; path_s[0] = 0;
                         NaN beyond the log.  (chords between 0.5 s frames; a DraftPath/C2 fit is the consumer's job.)
  n_avail    i1     number of future frames present in the log (0..16); scene tokens always have >= 10.
  n_reg      i1     number of leading future frames whose timestamp step is 0.5 s +- 0.1 s (regular grid, <= n_avail).
  dt         [16]   f32  timestamp step (s) of future frame k from frame k-1 (dt[0] = t(c+1) - t(c)); NaN beyond.
  frame_gap  bool   any |step - 0.5| > 0.1 s among future frames c+1..c+10 (the 5 s window used by the metric cache,
                    objects and scorer).  IMPL_SPEC §3.6 "tokens with 1.0 s frame gaps are skipped" uses this flag.
  gap4       bool   same, frames c+1..c+8 only (the trajectory);  gap_hist: same for history frames c-3..c.
  v0         f32    |(vx, vy)| at t0 (rear-axle speed; = nuPlan DynamicCarState.speed of the metric-cache initial state)
  a0         f32    ax at t0: signed longitudinal rear-axle acceleration (ego frame) (NOT the nuPlan magnitude)
  eds        [4]    f32  raw ego_dynamic_state (vx, vy, ax, ay), ego frame at t0
  cmd        i1     argmax of driving_command, order (left, straight, right, unknown) as in PARA-SSR;  cmd_raw [4] i1
  ego_global [3]    f64  global (x, y, yaw) at t0 (map frame; = metric cache ego_state.rear_axle)
  frame_idx  i2     the log frame's frame_idx field (cross-checked against the split table)
  log_pos    i4     c (position in the log list)
Packed file: /home/external-user/ssd/yongjae_refiner/human/<split>.npz with the arrays above (+ `tokens`, `logs`,
  `meta_json`).  load_human(split) returns the dict plus `index` = {token: row}.

  python tools/refiner/extract_human.py extract --splits train,dev --workers 2
  python tools/refiner/extract_human.py validate --n 50        # -> report/refiner_T/human_validation.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
TRAIN_LOGS = ROOT / "data/dataset/navsim_logs/trainval"
TEST_LOGS = Path("/home/external-user/navsim/download/test_navsim_logs/test")
DATA = Path("/home/external-user/ssd/yongjae_refiner")
SPLITS = DATA / "splits"
OUT = DATA / "human"
MC = DATA / "metric_cache"
E_MC = ROOT / "report/cause_and_correction_tests/E_train_split_feasibility/metric_cache"
NAVTEST_HUMAN = ROOT / "report/head_ablation_scenes/table.npz"
REPORT = ROOT / "report/refiner_T"
MAXF = 16          # future frames kept for the path (8 s)
NTRAJ = 8          # trajectory poses (4 s)
NSCENE = 10        # scene future frames (5 s)
DT = 0.5
DT_TOL = 0.1
FIELDS = ["traj", "path", "path_s", "n_avail", "n_reg", "dt", "frame_gap", "gap4", "gap_hist", "v0", "a0", "eds",
          "cmd", "cmd_raw", "ego_global", "frame_idx", "log_pos"]


# ----------------------------------------------------------------------------------------------- core
def ego_pose(frame: dict) -> np.ndarray:
    """Global (x, y, yaw) of a log frame, as navsim Scene._build_ego_status."""
    from pyquaternion import Quaternion
    t = frame["ego2global_translation"]
    return np.array([t[0], t[1], Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]], np.float64)


def extract_token(frames: Sequence[dict], c: int, maxf: int = MAXF) -> Dict[str, np.ndarray]:
    """All per-token fields for the t0 frame index c of a log frame list (see module docstring)."""
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )
    n = len(frames)
    n_av = min(n - 1 - c, maxf)
    if c < 3 or n_av < NSCENE:
        raise ValueError(f"frame {c} has no full scene window ({n} frames)")
    g0 = ego_pose(frames[c])
    G = np.stack([ego_pose(frames[c + k]) for k in range(1, n_av + 1)])
    rel = convert_absolute_to_relative_se2_array(StateSE2(*g0), G)          # [n_av,3] f64, wrapped heading
    path = np.full((maxf, 3), np.nan, np.float64)
    path[:n_av] = rel
    path[:n_av, 2] = np.unwrap(np.concatenate([[0.0], rel[:, 2]]))[1:]
    xy = np.concatenate([np.zeros((1, 2)), rel[:, :2]])
    s = np.full(maxf + 1, np.nan, np.float64)
    s[: n_av + 1] = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    ts = np.array([frames[k]["timestamp"] for k in range(c - 3, c + n_av + 1)], np.int64)
    steps = np.diff(ts) / 1e6                                               # [3 + n_av]
    hist, fut = steps[:3], steps[3:]
    dt = np.full(maxf, np.nan, np.float64)
    dt[:n_av] = fut
    reg = np.abs(fut - DT) <= DT_TOL
    eds = np.asarray(frames[c]["ego_dynamic_state"], np.float64)
    cmd_raw = np.asarray(frames[c]["driving_command"]).astype(np.int8)
    return dict(
        traj=rel[:NTRAJ].astype(np.float32), path=path.astype(np.float32), path_s=s.astype(np.float32),
        n_avail=np.int8(n_av), n_reg=np.int8(int(np.cumprod(reg).sum())), dt=dt.astype(np.float32),
        frame_gap=bool(~reg[:NSCENE].all()), gap4=bool(~reg[:NTRAJ].all()), gap_hist=bool((np.abs(hist - DT) > DT_TOL).any()),
        v0=np.float32(np.hypot(eds[0], eds[1])), a0=np.float32(eds[2]), eds=eds.astype(np.float32),
        cmd=np.int8(np.argmax(cmd_raw)), cmd_raw=cmd_raw, ego_global=g0, frame_idx=np.int16(frames[c]["frame_idx"]),
        log_pos=np.int32(c))


def extract_log(a) -> List[tuple]:
    """a = (log, tokens, logs_dir) -> [(token, fields | None, error)]"""
    log, toks, logs_dir = a
    frames = pickle.load(open(Path(logs_dir) / f"{log}.pkl", "rb"))
    pos = {f["token"]: i for i, f in enumerate(frames)}
    out = []
    for t in toks:
        try:
            out.append((t, extract_token(frames, pos[t]), ""))
        except Exception as e:  # pragma: no cover
            out.append((t, None, repr(e)))
    return out


def pack(tokens: Sequence[str], logs: Sequence[str], rows: Dict[str, dict]) -> Dict[str, np.ndarray]:
    d = {k: np.stack([np.asarray(rows[t][k]) for t in tokens]) for k in FIELDS}
    d["tokens"] = np.asarray(tokens)
    d["logs"] = np.asarray(logs)
    return d


def extract_split(df: pd.DataFrame, logs_dir: Path, workers: int = 2) -> Dict[str, np.ndarray]:
    """df: token, log (row order kept)."""
    jobs = [(lg, g.token.tolist(), str(logs_dir)) for lg, g in df.groupby("log", sort=True)]
    rows, errs = {}, []
    with Pool(workers) as p:
        for res in p.imap_unordered(extract_log, jobs, chunksize=4):
            for t, r, e in res:
                if r is None:
                    errs.append((t, e))
                else:
                    rows[t] = r
    if errs:
        raise RuntimeError(f"{len(errs)} tokens failed, e.g. {errs[:3]}")
    return pack(df.token.tolist(), df.log.tolist(), rows)


def load_human(split: str, root: Path = OUT) -> Dict[str, np.ndarray]:
    """Packed arrays of <root>/<split>.npz + index {token: row}."""
    z = np.load(Path(root) / f"{split}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["index"] = {t: i for i, t in enumerate(d["tokens"].tolist())}
    return d


# ----------------------------------------------------------------------------------------------- CLI: extract
def cmd_extract(a):
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for split in a.splits.split(","):
        t0 = time.time()
        df = pd.read_parquet(Path(a.splits_dir) / f"{split}.parquet", columns=["token", "log", "frame_idx"])
        if a.limit:
            df = df.iloc[: a.limit]
        d = extract_split(df, TRAIN_LOGS, a.workers)
        assert (d["tokens"] == df.token.to_numpy()).all()
        fi_ok = bool((d["frame_idx"].astype(np.int64) == df.frame_idx.to_numpy()).all())
        meta = dict(split=split, n=len(df), created=time.strftime("%F %T"), logs_dir=str(TRAIN_LOGS),
                    frame_idx_matches_split=fi_ok, frame_gap=int(d["frame_gap"].sum()), gap4=int(d["gap4"].sum()),
                    n_avail_ge16=int((d["n_avail"] >= 16).sum()), n_reg_ge16=int((d["n_reg"] >= 16).sum()),
                    seconds=time.time() - t0)
        d["meta_json"] = np.asarray(json.dumps(meta))
        tmp = out / f".{split}.tmp.npz"
        np.savez(tmp, **d)
        tmp.replace(out / f"{split}.npz")
        print(json.dumps(meta), flush=True)
        if not fi_ok:
            raise SystemExit("frame_idx mismatch between log and split table")


# ----------------------------------------------------------------------------------------------- CLI: validate
def _hdiff(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _mc_file(log: str, tok: str) -> Optional[Path]:
    for root in (MC, E_MC):
        p = root / log / "unknown" / tok / "metric_cache.pkl"
        if p.exists():
            return p
    return None


def cmd_validate(a):
    """(1) traj / 5 s path vs navsim SceneLoader Scene.get_future_trajectory on n navtrain tokens (those with a metric
    cache, incl. up to n/10 frame-gap tokens); (2) t0 pose / velocity / acceleration vs the metric cache ego_state;
    (3) n navtest tokens extracted from the test logs vs report/head_ablation_scenes/table.npz (human, speed, accel,
    command)."""
    import lzma
    import os
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(ROOT / "data/dataset/maps"))  # read at navsim import time
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    rng = np.random.default_rng(a.seed)
    H = {s: load_human(s, Path(a.out)) for s in ("train", "dev")}
    df = pd.concat([pd.read_parquet(Path(a.splits_dir) / f"{s}.parquet", columns=["token", "log"]).assign(split=s)
                    for s in ("train", "dev")]).reset_index(drop=True)
    df["mc"] = [_mc_file(lg, t) is not None for t, lg in zip(df.token, df.log)]
    df["gap"] = [bool(H[s]["frame_gap"][H[s]["index"][t]]) for t, s in zip(df.token, df.split)]
    cand = df[df.mc]
    ng = min(a.n // 10, int(cand.gap.sum()))
    pick = pd.concat([cand[cand.gap].sample(ng, random_state=a.seed), cand[~cand.gap].sample(a.n - ng, random_state=a.seed)])
    rows = []
    for r in pick.itertuples():
        h = H[r.split]
        i = h["index"][r.token]
        sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                         log_names=[r.log], tokens=[r.token])
        sl = SceneLoader(data_path=TRAIN_LOGS, sensor_blobs_path=None, scene_filter=sf,
                         sensor_config=SensorConfig.build_no_sensors())
        sc = sl.get_scene_from_token(r.token)
        f8 = sc.get_future_trajectory(8).poses
        f10 = sc.get_future_trajectory(10).poses
        with lzma.open(_mc_file(r.log, r.token), "rb") as f:
            mc = pickle.load(f)
        es = mc.ego_state
        ra = es.rear_axle
        dcs = es.dynamic_car_state
        rows.append(dict(
            token=r.token, split=r.split, frame_gap=bool(h["frame_gap"][i]),
            traj_xy_err=float(np.abs(h["traj"][i, :, :2] - f8[:, :2]).max()),
            traj_h_err=float(_hdiff(h["traj"][i, :, 2], f8[:, 2]).max()),
            path5_xy_err=float(np.abs(h["path"][i, :10, :2] - f10[:, :2]).max()),
            path5_h_err=float(_hdiff(h["path"][i, :10, 2], f10[:, 2]).max()),
            t0_pose_err=float(max(abs(ra.x - h["ego_global"][i, 0]), abs(ra.y - h["ego_global"][i, 1]),
                                  _hdiff(ra.heading, h["ego_global"][i, 2]))),
            t0_vel_err=float(max(abs(dcs.rear_axle_velocity_2d.x - h["eds"][i, 0]),
                                 abs(dcs.rear_axle_velocity_2d.y - h["eds"][i, 1]))),
            t0_acc_err=float(max(abs(dcs.rear_axle_acceleration_2d.x - h["eds"][i, 2]),
                                 abs(dcs.rear_axle_acceleration_2d.y - h["eds"][i, 3]))),
            v0_err=float(abs(dcs.speed - h["v0"][i])),
            scene_v0_err=float(abs(np.linalg.norm(sc.frames[3].ego_status.ego_velocity) - h["v0"][i])),
            scene_a0_err=float(abs(sc.frames[3].ego_status.ego_acceleration[0] - h["a0"][i])),
            scene_cmd_eq=bool(int(np.argmax(sc.frames[3].ego_status.driving_command)) == int(h["cmd"][i])),
        ))
    R = pd.DataFrame(rows)
    # (3) navtest cross-check against the existing human table
    T = np.load(NAVTEST_HUMAN, allow_pickle=True)
    ok = np.where(T["human_valid"])[0]
    sel = rng.choice(ok, a.n, replace=False)
    nt = pd.DataFrame(dict(token=T["tokens"][sel], log=T["logs"][sel], row=sel))
    got = {}
    for lg, g in nt.groupby("log"):
        for t, r, e in extract_log((lg, g.token.tolist(), str(TEST_LOGS))):
            got[t] = r
    nt_xy = max(float(np.abs(got[t]["traj"][:, :2] - T["human"][j][:, :2]).max()) for t, j in zip(nt.token, nt.row))
    nt_h = max(float(_hdiff(got[t]["traj"][:, 2], T["human"][j][:, 2]).max()) for t, j in zip(nt.token, nt.row))
    nt_v = max(float(abs(got[t]["v0"] - T["speed"][j])) for t, j in zip(nt.token, nt.row))
    nt_a = max(float(abs(got[t]["a0"] - T["accel"][j])) for t, j in zip(nt.token, nt.row))
    nt_c = all(int(got[t]["cmd"]) == int(T["command"][j]) for t, j in zip(nt.token, nt.row))
    num = [c for c in R.columns if c.endswith("_err")]
    summ = dict(
        navtrain=dict(n=len(R), n_frame_gap=int(R.frame_gap.sum()), max_abs=R[num].max().to_dict(),
                      scene_cmd_all_equal=bool(R.scene_cmd_eq.all())),
        navtest_vs_head_ablation_table=dict(n=len(nt), traj_xy_err=nt_xy, traj_h_err=nt_h, v0_err=nt_v, a0_err=nt_a,
                                            cmd_all_equal=nt_c),
        notes="traj is f32; errors ~1e-6 m come from the f32 cast of poses that are f64 in navsim "
              "(Trajectory stores float32 as well). metric-cache ego velocity/acceleration are float32 (EgoStatus).",
        rows=rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(REPORT / "human_validation.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["extract", "validate"])
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--splits-dir", default=str(SPLITS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.workers > 2:
        raise SystemExit("at most 2 workers for extraction")
    {"extract": cmd_extract, "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    main()
