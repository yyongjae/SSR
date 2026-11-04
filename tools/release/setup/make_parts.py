#!/usr/bin/env python
"""Split token tables into K parquet parts so K builder processes work on disjoint tokens.

  make_parts.py --tables A.parquet [B.parquet ...] --k K --out-prefix PREFIX [--skip-mc METRIC_CACHE_ROOT]

Whole logs stay together and go to the smallest part so far (a metric cache chunk is per log).  Output <PREFIX><i>.parquet with
token, log, i = 1..K.  --skip-mc leaves out tokens whose metric cache exists, so RESUME=1 needs no second pass over them.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def split_parts(df: pd.DataFrame, k: int) -> List[pd.DataFrame]:
    sizes = df.groupby("log").size().sort_values(ascending=False, kind="stable")
    parts: List[List[str]] = [[] for _ in range(k)]
    n = [0] * k
    for log, c in sizes.items():
        i = int(np.argmin(n))
        parts[i].append(log)
        n[i] += int(c)
    return [df[df.log.isin(p)][["token", "log"]].reset_index(drop=True) for p in parts]


def load_tables(tables: Sequence[str], skip_mc: Optional[str] = None) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(t, columns=["token", "log"]) for t in tables], ignore_index=True)
    if df.token.duplicated().any():
        raise SystemExit("the tables share tokens")
    if skip_mc:
        root = Path(skip_mc)
        keep = [not (root / g / "unknown" / t / "metric_cache.pkl").exists() for t, g in zip(df.token, df.log)]
        df = df[keep].reset_index(drop=True)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--skip-mc", default="")
    a = ap.parse_args(argv)
    if a.k < 1:
        ap.error("--k must be >= 1")
    df = load_tables(a.tables, a.skip_mc or None)
    parts = split_parts(df, a.k)
    for i, p in enumerate(parts, 1):
        p.to_parquet(f"{a.out_prefix}{i}.parquet", index=False)
    print(f"{len(df)} tokens, {df.log.nunique()} logs -> {a.k} parts, sizes {[len(p) for p in parts]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
