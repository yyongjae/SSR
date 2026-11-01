#!/usr/bin/env python
"""CK Phase 1: lead-vehicle flags per split -> CK_DATA/lead/<split>.parquet (the 'lead-decel scene' of report 43).

Definition (fixed before any outcome; direction_analysis/uncertainty/scripts/extract_lead.py, used UNCHANGED):
  lead      vehicle | bicycle at t0, |lat| <= 1.8 m from the metric-cache route centerline, |dh| <= 60 deg,
            gap = s - L/2 - 4.049 m in (-1, 80] m; lead = smallest s
  D_1       lead v0 - min v over 0.5..4 s > 1 m/s ('lead decelerates within 4 s')
  D_strong  v0 > 2 m/s and min v < 0.5 v0;  D_hard  >= 3 m/s drop within 1 s
  censored  lead not annotated at >= 6 of the 8 future keyframes or not at k >= 6 -> D_* = NaN
Lead-decel scene (eval metric): has_lead == 1 and D_1 == 1 (censored excluded).  Convenience column lead_decel =
  1.0 / 0.0, NaN when has_lead and censored.

Sources
  navtrain_train  direction_analysis/uncertainty/out/lead_navtrain.parquet (85,109 rows, same extract_lead run)
  navtest         direction_analysis/uncertainty/out/lead_navtest.parquet (12,146)
  navtrain_val    built here: extract_lead.py imported unchanged with its SPLIT / LOGS / MC / OUT tables extended
                  (needs the val metric cache, tools/ck/data/build_val.sh a2) -> CK_DATA/lead/_build/
Columns: token, log, has_lead, D_1, D_strong, D_hard, lead_v, lead_gap, thw, ttc0, censored, lead_decel, error.
  Row order = split tokens parquet order; every split token must be present.

  lead_split.py --split {navtrain_train, navtrain_val, navtest} [--root R] [--workers 24]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import common as CM  # noqa: E402  (pins navsim before extract_lead -> score_trajectories)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXTRACT = Path("/workspace/yongjae/ssd/yongjae_refiner/direction_analysis/uncertainty/scripts")
OUT_COLS = ("token", "log", "has_lead", "D_1", "D_strong", "D_hard", "lead_v", "lead_gap", "thw", "ttc0", "censored",
            "lead_decel", "error")


def build_val(root, workers: int) -> Path:
    """extract_lead for navtrain_val -> <root>/lead/_build/lead_navtrain_val.parquet (resumable per log)."""
    assert workers <= 48
    sys.path.insert(0, str(EXTRACT))
    import extract_lead as XL
    import navsim
    assert navsim.__file__.startswith(CM.CK), navsim.__file__
    sp = CM.split("navtrain_val")
    out = CM.root_dir(root) / "lead" / "_build"
    out.mkdir(parents=True, exist_ok=True)
    XL.SPLIT["navtrain_val"] = Path(sp["tokens"])
    XL.LOGS["navtrain_val"] = Path(sp["navsim_logs"])
    XL.MC["navtrain_val"] = Path(sp["metric_cache"])
    XL.OUT = out
    argv = sys.argv
    sys.argv = [XL.__file__, "--split", "navtrain_val", "--workers", str(workers)]
    try:
        XL.main()
    finally:
        sys.argv = argv
    return out / "lead_navtrain_val.parquet"


def finalize(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    toks = CM.split_tokens(split_name)
    df = df.drop_duplicates("token").set_index("token")
    miss = sorted(set(toks.token) - set(df.index))
    assert not miss, f"{len(miss)} split tokens without lead row, e.g. {miss[:3]}"
    df = df.loc[toks.token].reset_index()
    out = pd.DataFrame({"token": df.token.astype(str), "log": toks.log.values})
    has = df["has_lead"].astype("float64").fillna(0.0) if "has_lead" in df else pd.Series(0.0, index=df.index)
    out["has_lead"] = has.values
    for c in ("D_1", "D_strong", "D_hard", "lead_v", "lead_gap", "thw", "ttc0"):
        out[c] = df[c].astype("float64").values if c in df else np.nan
    out["censored"] = df["censored"].astype("float64").values if "censored" in df else np.nan
    ld = np.where(out.has_lead.values == 1.0, out.D_1.values, 0.0)   # NaN stays NaN for censored leads
    out["lead_decel"] = ld.astype("float64")
    out["error"] = df["error"].fillna("").astype(str).values if "error" in df else ""
    return out[list(OUT_COLS)]


def run(split_name: str, root=None, workers: int = 24) -> dict:
    t0 = time.time()
    if split_name == "navtrain_val":
        src = build_val(root, workers)
    else:
        src = CM.LEAD_SRC[split_name]
    df = finalize(pd.read_parquet(src), split_name)
    dst = CM.lead_path(split_name, root)
    CM.atomic_to_parquet(df, dst)
    rep = dict(split=split_name, src=str(src), n=len(df), has_lead=int((df.has_lead == 1).sum()),
               D_1=int((df.D_1 == 1).sum()), lead_decel=int((df.lead_decel == 1).sum()),
               censored_leads=int(((df.has_lead == 1) & df.lead_decel.isna()).sum()),
               errors=int((df.error != "").sum()), seconds=round(time.time() - t0, 1))
    CM.atomic_write_json(dst.with_suffix(".json"), rep)
    print(rep, flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=sorted(CM.SPLITS))
    ap.add_argument("--root", default="")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(argv)
    run(a.split, a.root or None, a.workers)


if __name__ == "__main__":
    main()
