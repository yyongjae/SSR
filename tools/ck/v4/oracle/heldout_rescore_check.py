#!/usr/bin/env python
"""refiner v4_3 Phase 0b -- held-out data check against the OFFICIAL simulator / scorer (CPU; writes HO/checks only).

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/v4/oracle/heldout_rescore_check.py [--n 100] [--workers 28]

On --n held-out tokens (seeded, seed 0) the 96 + the DET head go through ONE official score_token call
(tools/refiner/score_trajectories, plain navsim classes, return_states=True) on the navtrain_val metric cache:
  labels  : official values vs ho_exec (= the stored label files): max |diff| per EXEC column; time indices exact.
  states  : official PDMSimulator states (global -> N frame at the metric-cache rear axle) vs ho_sim (ck/v4/lqr
            float64): max |dxy| (m), |dheading| (rad), |dv| (m/s) over the 96.
-> HO/checks/rescore.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_common as H  # noqa: E402
import oracle_common as OC  # noqa: E402
import numpy as np  # noqa: E402

_G: dict = {}


def init():
    try:
        os.nice(10)
    except Exception:
        pass
    _G["D"] = H.HeldoutData()
    _G["S"] = OC.OfficialScorer()
    _G["var"] = np.load(H.TRAJ80, mmap_mode="r")
    _G["lat"] = np.load(H.LAT16, mmap_mode="r")


def to_n(states_g: np.ndarray, x0: float, y0: float, h0: float) -> np.ndarray:
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex as SI
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = states_g[..., SI.X] - x0, states_g[..., SI.Y] - y0
    h = np.arctan2(np.sin(states_g[..., SI.HEADING] - h0), np.cos(states_g[..., SI.HEADING] - h0))
    return np.stack([c * dx + s * dy, -s * dx + c * dy, h, states_g[..., SI.VELOCITY_X]], -1)


def process(i: int) -> dict:
    D, S = _G["D"], _G["S"]
    row, tok = int(D.rows[i]), str(D.tokens[i])
    tr96, _ = H.token_trajs96(D.cand[row], _G["var"][row])
    trajs = np.concatenate([tr96, np.asarray(_G["lat"][row, :1], np.float32)], 0)
    mc = S.ST.load_metric_cache(OC.CM.mc_path(OC.SPLIT, str(D.logs[row]), tok))
    res = S.ST.score_token(mc, trajs, S.sim, S.scorer, return_states=True)
    lab = np.array([[float(r[c]) for c in OC.SCORE_COLS] for r in res], np.float64)          # [97, 11]
    ra = mc.ego_state.rear_axle
    Xo = to_n(np.stack([r["states"] for r in res[:OC.N96]]), ra.x, ra.y, ra.heading)      # [96, 41, 4]
    Xl = np.asarray(D.sim[i], np.float64)
    E = np.asarray(D.exec[i], np.float64)
    return dict(i=i, lab_diff=np.abs(lab - E).max(0), dxy=float(np.hypot(*(Xo[..., :2] - Xl[..., :2]).transpose(2, 0, 1)).max()),
                dh=float(np.abs(np.arctan2(np.sin(Xo[..., 2] - Xl[..., 2]), np.cos(Xo[..., 2] - Xl[..., 2]))).max()),
                dv=float(np.abs(Xo[..., 3] - Xl[..., 3]).max()),
                n_time_idx_mismatch=int((lab[:, 8:] != E[:, 8:]).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    assert a.workers <= 28
    out = H.HO / "checks"
    out.mkdir(parents=True, exist_ok=True)
    n_all = len(np.load(H.HO_PREP / "ho_rows.npy"))
    pick = np.sort(np.random.default_rng(0).choice(n_all, a.n, replace=False))
    t0 = time.time()
    H.HeldoutData()                      # fail fast here (a failing Pool initializer respawns forever)
    with Pool(a.workers, initializer=init) as pool:
        res = pool.map(process, pick.tolist(), chunksize=1)
    ld = np.stack([r["lab_diff"] for r in res]).max(0)
    rep = dict(n_tokens=a.n, n_trajectories=a.n * H.N97, heldout_index=pick.tolist(),
               label_max_abs_diff={c: float(ld[k]) for k, c in enumerate(OC.EXEC_COLS)},
               time_idx_mismatches=int(sum(r["n_time_idx_mismatch"] for r in res)),
               states_max_dxy_m=float(max(r["dxy"] for r in res)), states_max_dheading_rad=float(max(r["dh"] for r in res)),
               states_max_dv=float(max(r["dv"] for r in res)), sec=time.time() - t0, finished=H.kst())
    (out / "rescore.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "heldout_index"}, indent=1))


if __name__ == "__main__":
    main()
