#!/usr/bin/env python
"""Raw-trajectory vs LQR-tracked PDM sub-scores on navtest.

For every token, scores the model trajectory twice with the unmodified PDMScorer:
  tracked : after PDMSimulator (LQR + bicycle), i.e. the official PDMS path
  raw     : the interpolated model trajectory fed straight into the scorer (no simulator)
and records the max / final XY deviation between the two.
NC / DAC / DDC are geometry-only and meaningful for `raw`; TTC / comfort need velocities
(zero in the raw array) and are reported for `tracked` only.

usage: python dac_raw_vs_tracked.py --traj <dump.pkl> --cache <metric_cache dir> --out <csv>
       [--ref-csv <official PDMS csv for a sanity check>] [--workers N] [--limit N]
"""
from __future__ import annotations

import argparse
import lzma
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("SSR_REPO", Path.home() / "SSR"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(REPO / "data/dataset/maps"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(REPO / "data/dataset"))
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(REPO))
os.environ.setdefault("NAVSIM_EXP_ROOT", str(REPO / "work_dirs"))

_SIM = _SCORER = _SAMPLING = None


def _init(num_poses: int, interval: float) -> None:
    global _SIM, _SCORER, _SAMPLING
    import torch
    torch.set_num_threads(1)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/pdm_scoring"), version_base="1.2"):
        cfg = compose(config_name="default_run_pdm_score")
    _SIM = instantiate(cfg.simulator)
    _SCORER = instantiate(cfg.scorer)
    assert _SIM.proposal_sampling == _SCORER.proposal_sampling
    _SAMPLING = TrajectorySampling(num_poses=num_poses, interval_length=interval)


def _score(args):
    token, cache_path, poses = args
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex as M, WeightedMetricIndex as W

    try:
        with lzma.open(cache_path, "rb") as f:
            mc = pickle.load(f)
        ego0 = mc.ego_state
        ps = _SIM.proposal_sampling
        pdm_states = get_trajectory_as_array(mc.trajectory, ps, ego0.time_point)
        pred_states = get_trajectory_as_array(
            transform_trajectory(Trajectory(np.asarray(poses, np.float32), _SAMPLING), ego0), ps, ego0.time_point)
        states = np.stack([pdm_states, pred_states])

        simulated = _SIM.simulate_proposals(states.copy(), ego0)
        s_tr = _SCORER.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
        mm, wm = _SCORER._multi_metrics.copy(), _SCORER._weighted_metrics.copy()
        row = dict(token=token, valid=True,
                   score=float(s_tr[1]),
                   nc=float(mm[M.NO_COLLISION, 1]), dac=float(mm[M.DRIVABLE_AREA, 1]), ddc=float(mm[M.DRIVING_DIRECTION, 1]),
                   ep=float(wm[W.PROGRESS, 1]), ttc=float(wm[W.TTC, 1]), comfort=float(wm[W.COMFORTABLE, 1]))

        _SCORER.score_proposals(states.copy(), mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
        mm = _SCORER._multi_metrics
        row.update(raw_nc=float(mm[M.NO_COLLISION, 1]), raw_dac=float(mm[M.DRIVABLE_AREA, 1]), raw_ddc=float(mm[M.DRIVING_DIRECTION, 1]))

        dev = np.linalg.norm(simulated[1, :, :2] - states[1, :, :2], axis=-1)
        row.update(dev_max=float(dev.max()), dev_final=float(dev[-1]),
                   dev_t_max=float(dev.argmax() * ps.interval_length))
        return row
    except Exception as exc:  # noqa: BLE001
        return dict(token=token, valid=False, error=repr(exc))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=REPO / "data/exp/metric_cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref-csv", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    with open(a.traj, "rb") as f:
        d = pickle.load(f)
    trajs, meta = d["trajectories"], d["meta"]
    cache_paths = {p.parent.name: p for p in a.cache.rglob("metric_cache.pkl")}
    tokens = sorted(set(trajs) & set(cache_paths))
    missing = len(trajs) - len(tokens)
    if a.limit:
        tokens = tokens[: a.limit]
    print(f"{len(tokens)} tokens to score ({missing} without cache); num_poses={meta['num_poses']} dt={meta['interval_length']}", flush=True)

    jobs = [(t, cache_paths[t], trajs[t]) for t in tokens]
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(int(meta["num_poses"]), float(meta["interval_length"]))) as pool:
        for i, row in enumerate(pool.imap_unordered(_score, jobs, chunksize=8), 1):
            rows.append(row)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows).sort_values("token")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    ok = df[df.valid == True]  # noqa: E712
    print(f"\nvalid {len(ok)}/{len(df)}  tracked PDMS {ok.score.mean():.4f}")
    if a.ref_csv is not None:
        ref = pd.read_csv(a.ref_csv)
        ref = ref[ref.token.notna()].set_index("token")
        j = ok.set_index("token").join(ref[["score", "drivable_area_compliance", "no_at_fault_collisions"]], rsuffix="_ref", how="inner")
        print(f"sanity vs official csv: n={len(j)} max|score diff|={np.abs(j.score - j.score_ref).max():.2e} "
              f"DAC mismatches={(j.dac != j.drivable_area_compliance).sum()} NC mismatches={(j.nc != j.no_at_fault_collisions).sum()}")

    def table(name, tr, raw):
        t0, r0 = ok[tr] == 0, ok[raw] == 0
        print(f"\n[{name}] tracked fail {int(t0.sum())} | raw fail {int(r0.sum())} | both {int((t0 & r0).sum())} | "
              f"tracked-only (tracking-induced) {int((t0 & ~r0).sum())} | raw-only (tracking rescued) {int((~t0 & r0).sum())}")
        if t0.sum():
            print(f"   share of tracked failures induced by tracking: {(t0 & ~r0).sum() / t0.sum():.1%}")

    table("DAC", "dac", "raw_dac")
    table("NC", "nc", "raw_nc")
    table("DDC", "ddc", "raw_ddc")
    q = ok.dev_max.quantile([0.5, 0.9, 0.99])
    print(f"\ntracking deviation dev_max: median {q[0.5]:.2f} m, p90 {q[0.9]:.2f} m, p99 {q[0.99]:.2f} m")
    for name, m in [("DAC tracked-only fail", (ok.dac == 0) & (ok.raw_dac == 1)), ("DAC both fail", (ok.dac == 0) & (ok.raw_dac == 0)), ("DAC pass", ok.dac == 1)]:
        if m.sum():
            print(f"   dev_max median | {name}: {ok.dev_max[m].median():.2f} m (n={int(m.sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
