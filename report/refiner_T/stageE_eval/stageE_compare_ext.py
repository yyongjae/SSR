#!/usr/bin/env python
"""Stage E navtest comparison, extended: stageE_compare.py (same load / log-cluster bootstrap) + per-city
(Las Vegas vs others, and each city) arm means and contrasts, and fixed / broken token counts (PDMS up / down).

  python report/refiner_T/stageE_eval/stageE_compare_ext.py --arm E0=<csv> --arm E2=<csv> --arm E2_tau0=<csv> \
      --contrast E2-E0 --contrast E2_tau0-E0 --contrast E2-E2_tau0 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "refiner"))
from stageE_compare import COLS, LOGMAP, boot, load  # noqa: E402

LV = "us-nv-las-vegas-strip"


def arm_stats(d):
    return dict(n=int(len(d)), **{m: float(d[c].mean()) for m, c in COLS.items()})


def contrast(A, B, tok, lg, n_boot):
    r = {"n_tokens": int(len(tok)), "n_logs": int(len(set(lg)))}
    for m, c in COLS.items():
        d = A.loc[tok, c].values - B.loc[tok, c].values
        lo, hi = boot(d, lg, n_boot)
        r[m] = {"diff": float(d.mean()), "ci95": [lo, hi]}
    d = A.loc[tok, "score"].values - B.loc[tok, "score"].values
    eps = 1e-9
    r["pdms_token_changes"] = {
        "up": int((d > eps).sum()), "down": int((d < -eps).sum()), "tie": int((np.abs(d) <= eps).sum()),
        "up_ge_0.1": int((d >= 0.1).sum()), "down_le_-0.1": int((d <= -0.1).sum()),
        # binary flips of the multiplicative gate (PDMS 0 <-> > 0)
        "fixed_from_zero": int(((B.loc[tok, "score"].values == 0) & (A.loc[tok, "score"].values > 0)).sum()),
        "broken_to_zero": int(((B.loc[tok, "score"].values > 0) & (A.loc[tok, "score"].values == 0)).sum()),
        "sum_up": float(d[d > eps].sum()), "sum_down": float(d[d < -eps].sum()),
    }
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--contrast", action="append", default=[])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = dict(s.split("=", 1) for s in a.arm)
    arms = {k: load(v) for k, v in src.items()}
    meta = pd.read_parquet(LOGMAP, columns=["token", "log", "city"]).drop_duplicates("token").set_index("token")
    city = meta.city
    groups = {"all": None, "las_vegas": lambda c: c == LV, "non_las_vegas": lambda c: c != LV}
    for cname in sorted(city.unique()):
        groups[cname] = (lambda cn: (lambda c: c == cn))(cname)
    res = {"csv": src, "n_boot": a.n_boot, "arms": {}, "contrasts": {}}
    for k, d in arms.items():
        res["arms"][k] = {}
        for g, f in groups.items():
            sub = d if f is None else d[f(city.reindex(d.index).fillna("unknown").values)]
            res["arms"][k][g] = arm_stats(sub)
    for con in a.contrast:
        An, Bn = con.split("-", 1)
        tok_all = arms[An].index.intersection(arms[Bn].index)
        res["contrasts"][con] = {}
        for g, f in groups.items():
            tok = tok_all if f is None else tok_all[f(city.reindex(tok_all).fillna("unknown").values)]
            lg = meta.log.reindex(tok).fillna("unknown").values
            res["contrasts"][con][g] = contrast(arms[An], arms[Bn], tok, lg, a.n_boot)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v["all"] for k, v in res["contrasts"].items()}, indent=1))


if __name__ == "__main__":
    main()
