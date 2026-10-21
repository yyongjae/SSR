"""navtest PDMS (v1 scorer) for a trajectory dump, on CPU, with no GPU pass.

``epdms_from_npz.sh`` already scores a dump with the v2.2 scorer, but PDMS only had
``run_pdm_score_gpu.py``, which runs the model.  That made every selection-rule experiment -- the
oracles, best-of-top-k, and now the inference field gate -- cost one GPU pass per variant, and made
it tempting to report EPDMS alone.  Both scorers must appear together, so this closes the pair: the
metric cache holds everything PDMS needs, the model is irrelevant once the trajectory exists.

``run_pdm_score_from_submission.py`` exists but is single-threaded and discards its own result
without writing a csv, so it is not the same thing.

    python tools/plan_v2/pdms_from_npz.py --name <label> --npz <dump.npz> [--workers 12]

Writes work_dirs/eval/<label>/<timestamp>.csv with the same columns and ``average`` row as
run_pdm_score_gpu.py, so tools/plan_v2/collect_results.py picks it up unchanged.
"""
import argparse
import lzma
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))

_CTX = {}


def _init(cache_path: str, traj_path: str):
    """One simulator/scorer per worker process.  They hold no cross-token state, and instantiating
    them once per token dominated the runtime."""
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from navsim.common.dataloader import MetricCacheLoader

    with initialize_config_module("navsim.planning.script.config.pdm_scoring", version_base="1.2"):
        cfg = compose("default_scoring_parameters")
    _CTX["sim"] = instantiate(cfg.simulator)
    _CTX["scorer"] = instantiate(cfg.scorer)
    assert _CTX["sim"].proposal_sampling == _CTX["scorer"].proposal_sampling
    _CTX["loader"] = MetricCacheLoader(Path(cache_path))
    _CTX["traj"] = pickle.load(open(traj_path, "rb"))


def _score(token: str):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.metric_caching.metric_cache import MetricCache

    row = {"token": token, "valid": True}
    try:
        with lzma.open(_CTX["loader"].metric_cache_paths[token], "rb") as f:
            cache: MetricCache = pickle.load(f)
        res = pdm_score(metric_cache=cache,
                        model_trajectory=Trajectory(_CTX["traj"][token]),
                        future_sampling=_CTX["sim"].proposal_sampling,
                        simulator=_CTX["sim"], scorer=_CTX["scorer"])
        row.update(asdict(res))
    except Exception as exc:                       # one bad token must not void the run
        row["valid"] = False
        row["error"] = repr(exc)[:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache", default=str(ROOT / "data/exp/metric_cache"))
    a = ap.parse_args()

    from navsim.common.dataloader import MetricCacheLoader
    z = np.load(a.npz, allow_pickle=True)
    traj = {str(t): np.asarray(p, dtype=np.float64)
            for t, p in zip(z["tokens"], z["trajectory"])}
    tokens = sorted(set(traj) & set(MetricCacheLoader(Path(a.cache)).tokens))
    print(f"{len(traj)} 궤적, metric cache 와 겹치는 토큰 {len(tokens)}", flush=True)
    if not tokens:
        raise SystemExit("metric cache 와 겹치는 토큰이 없음")

    out = ROOT / "work_dirs/eval" / a.name
    out.mkdir(parents=True, exist_ok=True)
    tp = out / "trajectories.pkl"
    pickle.dump(traj, open(tp, "wb"))

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.cache, str(tp))) as ex:
        for i, r in enumerate(ex.map(_score, tokens, chunksize=16), 1):
            rows.append(r)
            if i % 2000 == 0:
                print(f"{i}/{len(tokens)}  {(time.time() - t0) / i:.3f} s/scene", flush=True)

    df = pd.DataFrame(rows)
    bad = int((~df["valid"]).sum())
    avg = df.drop(columns=[c for c in ("token", "valid", "error") if c in df], errors="ignore").mean(skipna=True)
    avg["token"], avg["valid"] = "average", bool(df["valid"].all())
    df.loc[len(df)] = avg
    csv = out / f"{datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}.csv"
    df.to_csv(csv)
    print(f"\n{a.name}: PDMS {100 * float(avg['score']):.2f}   실패 {bad}   -> {csv}")
    for c in ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
              "ego_progress", "time_to_collision_within_bound", "comfort"):
        if c in avg:
            print(f"    {c:32s} {100 * float(avg[c]):6.2f}")


if __name__ == "__main__":
    main()
