#!/usr/bin/env python
"""Paired official PDMS (NAVSIM v1) of refiner v6c on a dump (tools/corridor/dump_corr_model.py --v6), with no field
dump and no verdict file (corr_select.py scores only candidate indices of traj.npy, report 125).

Per token it scores three trajectories with the official simulator + scorer (tools/refiner/score_trajectories):
  base  tau_base  (v6_base.npy, f32; the cand_official pick = msel)
  v6    tau_v6    (v6_traj.npy)
  out   the submitted one (mode B: v6_switch ? v6 : base; the dump records the forward's switch)
and reports the pre-registered endpoints of report 124 §5: mean PDMS of each, the paired difference v6 - base and
out - base with a log-cluster bootstrap CI (tokens of one log are correlated: val_epdms.log_bootstrap resamples logs;
the i.i.d. token CI is kept as ci95_token), the safety layer (tokens where base passes NC, DAC and TTC) difference, the
EP change on the safety layer (2nd endpoint), and the switch rate.  Also the official v1 driving direction compliance
(DDC, a multiplicative PDMS term) of each trajectory and how many tokens change DDC between base and v6 / out: R-J's
expected PDMS assumes DDC = 1 (report 126 §2-7), so this measures that assumption.  EPDMS / EC: val_epdms.py.

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

from tools.refiner_v6c.val_epdms import log_bootstrap  # noqa: E402  (numpy / pandas only)

TERMS = ("pdms", "nc", "dac", "ep", "ttc", "comfort", "ddc")       # ddc appended: the indices 0..5 above are kept


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


def paired_ci(diff: np.ndarray, logs: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """mean and 95 % CI of a paired difference, x 100 (A52): ci95 = log-cluster bootstrap over the tokens' logs
    (val_epdms.log_bootstrap), ci95_token = the i.i.d. token bootstrap (too narrow: tokens of a log are correlated)"""
    b = log_bootstrap(diff, logs, n, seed)
    lo, hi = boot_ci(np.asarray(diff, np.float64), n, seed)
    return {"mean": round(100 * b["mean"], 3), "ci95": [round(100 * b["lo"], 3), round(100 * b["hi"], 3)],
            "ci_method": "log-cluster bootstrap", "n_logs": b["n_logs"],
            "ci95_token": [round(100 * lo, 3), round(100 * hi, 3)]}


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
    R = np.array([x for x in res if not isinstance(x, str)], np.float64)                 # [n_ok, 2, 7]
    swo = sw[ok]
    lg = logs[rows][ok]                                                                  # the clusters of paired_ci
    out = np.where(swo[:, None], R[:, 1], R[:, 0])
    summ = {"split": a.split, "tokens": int(len(rows)), "scored": int(ok.sum()), "switch_rate": float(swo.mean())}
    for nm, M in (("base", R[:, 0]), ("v6", R[:, 1]), ("out", out)):
        summ[nm] = dict(zip(TERMS, np.round(100 * M.mean(0), 3).tolist()))
    for nm, M in (("v6_minus_base", R[:, 1, 0] - R[:, 0, 0]), ("out_minus_base", out[:, 0] - R[:, 0, 0])):
        summ[nm] = paired_ci(M, lg)
    safe = (R[:, 0, 1] == 1) & (R[:, 0, 2] == 1) & (R[:, 0, 4] == 1)                     # base passes NC, DAC, TTC
    dsafe = out[safe, 0] - R[safe, 0, 0]
    summ["safety_layer"] = {"frac": float(safe.mean()), "out_minus_base_pdms": round(100 * float(dsafe.mean()), 3)
                            if safe.any() else None,
                            "out_minus_base_ep": round(100 * float((out[safe, 3] - R[safe, 0, 3]).mean()), 3)
                            if safe.any() else None}
    di = TERMS.index("ddc")                                                              # DDC = 1 assumption of R-J
    summ["ddc"] = {"base_lt1_frac": float((R[:, 0, di] < 1).mean()), "v6_lt1_frac": float((R[:, 1, di] < 1).mean()),
                   "out_lt1_frac": float((out[:, di] < 1).mean())}
    for nm, M in (("v6", R[:, 1, di]), ("out", out[:, di])):
        dd = M - R[:, 0, di]
        summ["ddc"][f"{nm}_minus_base"] = {**paired_ci(dd, lg), "n_differ": int((dd != 0).sum()),
                                           "n_lower": int((dd < 0).sum()), "n_higher": int((dd > 0).sum())}
    if (~ok).any():
        summ["first_error"] = [x for x in res if isinstance(x, str)][0]
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(d / f"v6_val_{a.n}.json", "w"), indent=1)
    if (~ok).any():                                                                      # like score_npz.py: a mean
        print(f"ERROR: {int((~ok).sum())} tokens failed to score", file=sys.stderr)       # over a subset fails loudly
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
