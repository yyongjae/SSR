"""True per-candidate PDMS for all 256 candidates, from the real scorer.

Every selection-ceiling number so far came from an ORACLE OVER A PRECOMPUTED LABEL
(``data/planning_vb/navtest_refined_scores.npz``), not over what the scorer returns.  Measured, that
label is not the scorer: at top-1 it promises 77.06 where PDMS gives 88.12, and over the full 256 it
promises 92.52 where PDMS gives 84.90.  The sign of the gap FLIPS, because a larger pool lets the
argmax find candidates the label overstates -- the optimizer's curse.  So the label oracle cannot
state a ceiling; it states where the label is most wrong.

This removes the proxy.  ``pdm_score_multi_trajs`` simulates and scores all 256 candidates against
one metric cache in a single call, so the result is the scorer's own per-candidate score.  With it,
the oracle at any k is an argmax over a table -- no further scoring, no proxy, and the full-256
number is the real ceiling.

    python tools/plan_v2/pdms_all_candidates.py --workers 10 [--limit 200]

Writes work_dirs/eval_epdms/candidate_pdms_<exp>.npz with `scores` [N, 256, 7]
(NC, DAC, DDC, EP, TTC, C, PDMS) in the scorer's own column order.
"""
import argparse
import lzma
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))

_CTX = {}


def _init(cache_path: str, traj_path: str):
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from navsim.common.dataloader import MetricCacheLoader

    with initialize_config_module("navsim.planning.script.config.pdm_scoring", version_base="1.2"):
        cfg = compose("default_scoring_parameters")
    _CTX["sim"] = instantiate(cfg.simulator)
    _CTX["scorer"] = instantiate(cfg.scorer)
    _CTX["loader"] = MetricCacheLoader(Path(cache_path))
    _CTX["traj"] = np.load(traj_path, mmap_mode="r")


def _score(item):
    """(row, token) -> [256, 7] in the scorer's own numbers, or None if the scene fails.

    NOT one batched call over 256 proposals.  ``PDMScorer`` normalises progress by the MAXIMUM raw
    progress in the batch (pdm_scorer.py:178), so a candidate's score depends on who else is in the
    call: batching all 256 made the deployed pick score 54.46 against its published 88.14.  The
    single-trajectory ``pdm_score`` scores the pair [PDM reference, candidate] and reads index 1, so
    that pair is what each candidate must be scored in.  Simulation is per-proposal and independent,
    so it still runs once for all 257; only ``score_proposals`` repeats, on two rows at a time.
    """
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.metric_caching.metric_cache import MetricCache
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        MultiMetricIndex, WeightedMetricIndex)

    row, token = item
    sim, scorer = _CTX["sim"], _CTX["scorer"]
    try:
        with lzma.open(_CTX["loader"].metric_cache_paths[token], "rb") as f:
            cache: MetricCache = pickle.load(f)
        ego0 = cache.ego_state
        fs = sim.proposal_sampling
        states = [get_trajectory_as_array(cache.trajectory, fs, ego0.time_point)]
        for traj in np.asarray(_CTX["traj"][row], dtype=np.float64):
            states.append(get_trajectory_as_array(
                transform_trajectory(Trajectory(traj), ego0), fs, ego0.time_point))
        simulated = sim.simulate_proposals(np.stack(states), ego0)        # [257, T, D]

        out = np.empty((len(states) - 1, 7), np.float32)
        for i in range(1, len(states)):
            sc = scorer.score_proposals(simulated[[0, i]], cache.observation, cache.centerline,
                                        cache.route_lane_ids, cache.drivable_area_map)
            out[i - 1] = (
                scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 1],
                scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 1],
                scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, 1],
                scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 1],
                scorer._weighted_metrics[WeightedMetricIndex.TTC, 1],
                scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, 1],
                sc[1])
        return row, out
    except Exception as exc:
        return row, repr(exc)[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="para_ssr_v2_r34")
    ap.add_argument("--dump", default="")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", default=str(ROOT / "data/exp/metric_cache"))
    a = ap.parse_args()

    from navsim.common.dataloader import MetricCacheLoader
    dump = Path(a.dump) if a.dump else ROOT / f"work_dirs/eval_epdms/diag_{a.exp}/dump.npz"
    z = np.load(dump)
    toks = [str(t) for t in z["tokens"]]
    anchors = np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy").astype(np.float32)
    have = set(MetricCacheLoader(Path(a.cache)).tokens)
    rows = [i for i, t in enumerate(toks) if t in have]
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} 장면 x 256 후보", flush=True)

    # materialise anchor+offset once; the workers mmap it rather than pickling 256 trajectories each
    tmp = ROOT / f"work_dirs/eval_epdms/_cand_{a.exp}.npy"
    off = z["trajectory_offset"]
    all_traj = np.empty((len(toks), 256, off.shape[2], 3), np.float32)
    for s in range(0, len(toks), 512):
        e = min(s + 512, len(toks))
        all_traj[s:e] = anchors[None] + off[s:e].astype(np.float32)
    np.save(tmp, all_traj)
    del all_traj

    out = np.full((len(toks), 256, 7), np.nan, np.float32)
    bad, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.cache, str(tmp))) as ex:
        for i, (row, r) in enumerate(ex.map(_score, [(r, toks[r]) for r in rows], chunksize=4), 1):
            if isinstance(r, str):
                bad.append((toks[row], r))
            else:
                out[row] = r
            if i % 500 == 0:
                el = time.time() - t0
                print(f"{i}/{len(rows)}  {el / i:.3f} s/scene  남은 {(len(rows) - i) * el / i / 60:.0f}분",
                      flush=True)

    f = ROOT / f"work_dirs/eval_epdms/candidate_pdms_{a.exp}.npz"
    np.savez_compressed(f, tokens=np.asarray(toks), scores=out,
                        columns=np.asarray(["NC", "DAC", "DDC", "EP", "TTC", "C", "PDMS"]))
    tmp.unlink(missing_ok=True)
    print(f"\n저장 {f}   실패 {len(bad)}")
    for t, e in bad[:5]:
        print(f"  {t}: {e}")

    # control: the model's own pick, scored here, must reproduce the published PDMS
    if "plan_topk_index" in z.files:
        pick = z["plan_topk_index"][:, 0]
        ok = np.array([r for r in rows if not np.isnan(out[r, 0, 0])])
        print(f"\n대조군: 배포 선택의 PDMS = {100 * out[ok, pick[ok], 6].mean():.2f}  (공표 88.14)")


if __name__ == "__main__":
    main()
