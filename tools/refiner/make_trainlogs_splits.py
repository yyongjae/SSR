"""train_logs-only token subsets for the stage-T map arm (all arms use them).  New files only; train/dev.parquet unchanged.

  splits/train_trainlogs.parquet = rows of splits/train.parquet whose log is in NAVSIM train_logs (same columns, order)
  splits/dev_trainlogs.parquet   = rows of splits/dev.parquet   whose log is in NAVSIM train_logs
  splits/trainlogs_summary.json  = counts per fold / city, teacher coverage (ReSMap index + BEVFusion npz), packed
                                   trainable rows, navtest coverage of both caches.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python tools/refiner/make_trainlogs_splits.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache, restrict_to_train_logs, train_logs  # noqa: E402


def packed_trainable(split: str, tokens: set):
    """Rows of packed/<split> that are trainable (all parts done, no frame gap) restricted to `tokens`."""
    try:
        ps = RD.PackedSplit(split)
    except FileNotFoundError:
        return None
    rows = ps.rows_with()
    tk = ps.index.token.values[rows]
    return int(np.isin(tk, list(tokens)).sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RD.DATA_ROOT / "splits"))
    ap.add_argument("--force", action="store_true", help="overwrite existing *_trainlogs.parquet")
    a = ap.parse_args(argv)
    root = Path(a.root)
    t0 = time.time()
    tl = train_logs()
    rs = {s: ResmapCache.for_subset(s) for s in ("navtrain", "navtest")}
    bf = {s: RD.TeacherCache.for_subset(s) for s in ("navtrain", "navtest")}
    summ = dict(train_logs_yaml_n=len(tl), resmap_sha256=rs["navtrain"].sha256, bevfusion_sha_head=bf["navtrain"].sha_head)
    for split in ("train", "dev"):
        full = pd.read_parquet(root / f"{split}.parquet")
        sub = restrict_to_train_logs(full, tl)
        has_r = sub.token.map(rs["navtrain"].has)
        has_b = sub.token.map(bf["navtrain"].has)
        dropped = full[~full.log.isin(tl)]
        out = root / f"{split}_trainlogs.parquet"
        if out.exists() and not a.force:
            old = pd.read_parquet(out)
            if not old.equals(sub):
                raise SystemExit(f"{out} exists with different content (use --force)")
        else:
            sub.to_parquet(out, index=False)
        city = "map_location"
        s = dict(
            n_full=len(full), n=len(sub), n_logs=int(sub.log.nunique()), n_logs_full=int(full.log.nunique()),
            dropped_rows=len(dropped), dropped_logs=int(dropped.log.nunique()),
            dropped_part_counts=dropped.part.value_counts().to_dict(), kept_part_counts=sub.part.value_counts().to_dict(),
            resmap_has=int(has_r.sum()), bevfusion_has=int(has_b.sum()),
            per_fold=sub.fold.value_counts().sort_index().astype(int).to_dict(),
            per_fold_logs=sub.groupby("fold").log.nunique().astype(int).to_dict(),
            per_city=sub[city].value_counts().astype(int).to_dict(),
            per_city_full=full[city].value_counts().astype(int).to_dict(),
            per_fold_city=pd.crosstab(sub.fold, sub[city]).astype(int).to_dict(orient="index"),
            packed_trainable_full=packed_trainable(split, set(full.token)),
            packed_trainable_trainlogs=packed_trainable(split, set(sub.token)),
            path=str(out))
        s["per_fold"] = {int(k): v for k, v in s["per_fold"].items()}
        s["per_fold_logs"] = {int(k): v for k, v in s["per_fold_logs"].items()}
        s["per_fold_city"] = {int(k): v for k, v in s["per_fold_city"].items()}
        if s["resmap_has"] != s["n"] or s["bevfusion_has"] != s["n"]:
            raise SystemExit(f"{split}: coverage resmap {s['resmap_has']} bevfusion {s['bevfusion_has']} of {s['n']}")
        summ[split] = s
        print(split, json.dumps(s, default=int), flush=True)
    nt = pd.read_parquet(root / "navtest.parquet")
    summ["navtest"] = dict(n=len(nt), resmap_has=int(nt.token.map(rs["navtest"].has).sum()),
                           bevfusion_has=int(nt.token.map(bf["navtest"].has).sum()),
                           resmap_index_n=len(rs["navtest"].index),
                           resmap_root_has_any_navtest=int(nt.token.map(rs["navtrain"].has).sum()))
    print("navtest", summ["navtest"], flush=True)
    summ["sec"] = round(time.time() - t0, 1)
    (root / "trainlogs_summary.json").write_text(json.dumps(summ, indent=1, default=int))


if __name__ == "__main__":
    main()
