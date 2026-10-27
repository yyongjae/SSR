#!/usr/bin/env python
"""Paired official PDMS (NAVSIM v1) of refiner v6c on a dump (tools/corridor/dump_corr_model.py --v6), with no field
dump and no verdict file (corr_select.py scores only candidate indices of traj.npy, report 125).

Per token it scores three trajectories with the official simulator + scorer (tools/refiner/score_trajectories):
  base  tau_base  (v6_base.npy, f32; the cand_official pick = msel)
  v6    tau_v6    (v6_traj.npy)
  out   the submitted one (mode B: v6_switch ? v6 : base; the dump records the forward's switch)
and reports the pre-registered endpoints of report 124 §5: mean PDMS of each, the paired difference v6 - base and
out - base with a token bootstrap CI, the safety layer (tokens where base passes NC, DAC and TTC) difference, the EP
change on the safety layer (2nd endpoint), and the switch rate.

  $PY tools/refiner_v6c/val_score.py --dump <dir> --split navtrain_val [--n 0] [--workers 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TERMS = ("pdms", "nc", "dac", "ep", "ttc", "comfort")


def work(job):
    from tools.ck.data import label_cands as LC
    ST = LC.ST
    ST.get_simulator_scorer()
    out = []
    for path, trajs in job:
        try:
            res = ST.score_token(ST.load_metric_cache(path), trajs.astype(np.float32))
            out.append([[d[c] for c in TERMS] for d in res])
        except Exception as e:  # noqa: BLE001
            out.append(repr(e)[:200])
    return out


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return float("nan"), float("nan")
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--split", default="navtrain_val")
    ap.add_argument("--n", type=int, default=0, help="0 = every dumped token, else a rng(0) sample")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd
    from tools.ck.data import common as CM
    root = os.environ.get("KM_CORR_ROOT", str(REPO / "resources/km_corridor"))
    d = Path(a.dump) / a.split
    done = np.zeros(len(np.load(d / "v6_base.npy", mmap_mode="r")), bool)
    for f in d.glob("done_*.npy"):
        done |= np.load(f)
    rows = np.flatnonzero(done)
    if a.n and a.n < len(rows):
        rows = np.sort(np.random.default_rng(0).choice(rows, a.n, replace=False))
    base = np.asarray(np.load(d / "v6_base.npy", mmap_mode="r")[rows], np.float32)
    v6 = np.asarray(np.load(d / "v6_traj.npy", mmap_mode="r")[rows], np.float32)
    sw = np.asarray(np.load(d / "v6_switch.npy", mmap_mode="r")[rows]).astype(bool)
    tdf = pd.read_parquet(f"{root}/packed/{a.split}/tokens.parquet")
    toks, logs = tdf.token.astype(str).to_numpy(), tdf.log.astype(str).to_numpy()
    items = [(str(CM.mc_path(a.split, logs[r], toks[r])), np.stack([b, v])) for r, b, v in zip(rows, base, v6)]
    jobs = [items[i:i + 20] for i in range(0, len(items), 20)]
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for x in ex.map(work, jobs):
            res += x
    ok = np.array([not isinstance(x, str) for x in res])
    R = np.array([x for x in res if not isinstance(x, str)], np.float64)                 # [n_ok, 2, 6]
    swo = sw[ok]
    out = np.where(swo[:, None], R[:, 1], R[:, 0])
    summ = {"split": a.split, "tokens": int(len(rows)), "scored": int(ok.sum()), "switch_rate": float(swo.mean())}
    for nm, M in (("base", R[:, 0]), ("v6", R[:, 1]), ("out", out)):
        summ[nm] = dict(zip(TERMS, np.round(100 * M.mean(0), 3).tolist()))
    for nm, M in (("v6_minus_base", R[:, 1, 0] - R[:, 0, 0]), ("out_minus_base", out[:, 0] - R[:, 0, 0])):
        lo, hi = boot_ci(M)
        summ[nm] = {"mean": round(100 * float(M.mean()), 3), "ci95": [round(100 * lo, 3), round(100 * hi, 3)]}
    safe = (R[:, 0, 1] == 1) & (R[:, 0, 2] == 1) & (R[:, 0, 4] == 1)                     # base passes NC, DAC, TTC
    dsafe = out[safe, 0] - R[safe, 0, 0]
    summ["safety_layer"] = {"frac": float(safe.mean()), "out_minus_base_pdms": round(100 * float(dsafe.mean()), 3)
                            if safe.any() else None,
                            "out_minus_base_ep": round(100 * float((out[safe, 3] - R[safe, 0, 3]).mean()), 3)
                            if safe.any() else None}
    if (~ok).any():
        summ["first_error"] = [x for x in res if isinstance(x, str)][0]
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(d / f"v6_val_{a.n}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
