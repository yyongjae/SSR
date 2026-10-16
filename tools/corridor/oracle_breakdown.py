#!/usr/bin/env python
"""km 10/15: why the GT-corridor oracle stops at ~90: official scores of ALL 256 candidates on a val subset.
Official scoring of the model's own candidates = evaluation only (never a label).
  $PY tools/corridor/oracle_breakdown.py --dump <eval_dir> --n 400 --workers 6
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)
from tools.corridor import corr_select as CS   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--split", default="navtrain_val")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import pandas as pd
    from tools.ck.data.common import MC_ROOT
    d = Path(a.dump) / a.split
    D = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("traj", "sim", "im", "final")}
    V = dict(np.load(d / "verdict.npz"))
    rows = V["rows"]
    rows = np.sort(np.random.default_rng(0).choice(rows, 3000, replace=False))      # the same 3000 as corr_select
    rows = rows[np.isfinite(V["gt_sdf_min"][rows]).all(1)]
    rows = np.sort(np.random.default_rng(1).choice(rows, a.n, replace=False))
    toks = pd.read_parquet(f"{CS.ROOT}/packed/{a.split}/tokens.parquet").token.astype(str).to_numpy()
    where = {}
    for log in os.scandir(MC_ROOT[a.split]):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in os.scandir(u):
                where[t.name] = str(Path(t.path) / "metric_cache.pkl")
    items = [(where[toks[r]], np.asarray(D["traj"][r], np.float32)) for r in rows]
    from concurrent.futures import ProcessPoolExecutor
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for x in ex.map(CS.work, [items[i:i + 4] for i in range(0, len(items), 4)]):
            res += x
    ok = np.array([not isinstance(x, str) for x in res])
    O = np.array([res[i] for i in range(len(res)) if ok[i]], np.float64)           # [n, 256, 6]
    rows = rows[ok]
    np.savez(d / f"official_all_{a.n}.npz", rows=rows, O=O)
    print(f"{len(rows)} tokens x 256 scored")


if __name__ == "__main__":
    main()
