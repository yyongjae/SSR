#!/usr/bin/env python
"""Stage E navtest comparison: per-arm official sub-scores and paired differences with a log-cluster paired bootstrap.

  python tools/refiner/stageE_compare.py --arm E1=<csv> --arm E1_tau0=<csv> --arm E2=<csv> --arm E2_tau0=<csv> \
      --arm E0N=<csv> --contrast E2-E1 --contrast E1-E1_tau0 --contrast E2-E2_tau0 --contrast E2-E0N [--out <json>]

<csv> = the run_pdm_score_gpu.py output (columns token, valid, no_at_fault_collisions, drivable_area_compliance,
driving_direction_compliance, ego_progress, time_to_collision_within_bound, comfort, score; row 'average' dropped).
Contrasts use the tokens valid in both arms.  CI: resample whole logs (splits/navtest.parquet token -> log) with
replacement, 10,000 draws, seed 0, percentile 95 % interval of the difference of token means.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOGMAP = Path("/home/external-user/ssd/yongjae_refiner/splits/navtest.parquet")
COLS = {"PDMS": "score", "NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
        "DDC": "driving_direction_compliance", "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
        "C": "comfort"}


def load(csv) -> pd.DataFrame:
    d = pd.read_csv(csv)
    d = d[d.token != "average"]
    d = d[d.valid.astype(str).str.lower() == "true"]
    return d.set_index("token")[list(COLS.values())].astype(float)


def boot(diff: np.ndarray, logs: np.ndarray, n: int = 10000, seed: int = 0):
    codes, inv = np.unique(logs, return_inverse=True)
    s = np.bincount(inv, weights=diff)
    c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(codes), size=(n, len(codes)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=<csv>")
    ap.add_argument("--contrast", action="append", default=[], help="A-B")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    arms = {k: load(v) for k, v in (s.split("=", 1) for s in a.arm)}
    logmap = pd.read_parquet(LOGMAP, columns=["token", "log"]).drop_duplicates("token").set_index("token").log
    res = {"arms": {k: dict(n=int(len(d)), **{m: float(d[c].mean()) for m, c in COLS.items()}) for k, d in arms.items()},
           "contrasts": {}}
    for con in a.contrast:
        A, B = con.split("-", 1)
        tok = arms[A].index.intersection(arms[B].index)
        lg = logmap.reindex(tok).fillna("unknown").values
        r = {"n_tokens": int(len(tok)), "n_logs": int(len(set(lg)))}
        for m, c in COLS.items():
            d = arms[A].loc[tok, c].values - arms[B].loc[tok, c].values
            lo, hi = boot(d, lg, a.n_boot)
            r[m] = {"diff": float(d.mean()), "ci95": [lo, hi]}
        res["contrasts"][con] = r
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        Path(a.out).write_text(s)


if __name__ == "__main__":
    main()
