#!/usr/bin/env python
"""Attribution-lite: join per-token perception quality (aux-eval records) with PDMS failures.

Perception proxies per token (ego frame, PARA-SSR convention x_right / y_forward):
  det_recall_near : fraction of GT boxes within --near m of ego matched by a prediction
                    (score >= --det-score, center distance <= --det-dist, class-agnostic)
  det_missed_near : number of such GT boxes left unmatched
  map_cd_road     : mean over GT 'road' polylines of the symmetric Chamfer distance to the nearest
                    predicted 'road' polyline (score >= --map-score); --map-cap if no prediction
  map_cd_center   : same for 'centerline'
Joined with the raw-vs-tracked PDM csv (dac / raw_dac / nc / raw_nc / ttc / ep / score).

usage: python failure_attribution_lite.py --records <aux records dir> --pdm <raw_vs_tracked.csv> --out <csv>
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

MAP_CLASSES = ("road", "walkway", "centerline", "crosswalk")
_A = None


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def _one(path: Path) -> dict:
    a = _A
    d = np.load(path, allow_pickle=True)
    row = {"token": str(d["token"])}
    # detection
    gt, pr, sc = d["det_gt_boxes"], d["det_pred_boxes"], d["det_pred_scores"]
    pr = pr[sc >= a.det_score]
    if len(gt):
        near = np.hypot(gt[:, 0], gt[:, 1]) <= a.near
        gtn = gt[near]
        if len(gtn):
            if len(pr):
                dist = np.linalg.norm(gtn[:, None, :2] - pr[None, :, :2], axis=-1)
                hit = dist.min(1) <= a.det_dist
            else:
                hit = np.zeros(len(gtn), bool)
            row.update(det_gt_near=int(len(gtn)), det_recall_near=float(hit.mean()), det_missed_near=int((~hit).sum()))
        else:
            row.update(det_gt_near=0, det_recall_near=np.nan, det_missed_near=0)
    else:
        row.update(det_gt_near=0, det_recall_near=np.nan, det_missed_near=0)
    row["det_n_pred"] = int(len(pr))
    # map
    gp, gl = d["map_gt_points"], d["map_gt_labels"]
    pp, pl, ps = d["map_pred_points"], d["map_pred_labels"], d["map_pred_scores"]
    keep = ps >= a.map_score
    pp, pl = pp[keep], pl[keep]
    for ci, cname in ((0, "road"), (2, "center")):
        g = gp[gl == ci]
        p = pp[pl == ci]
        if len(g) == 0:
            row[f"map_cd_{cname}"] = np.nan
            row[f"map_gt_{cname}"] = 0
            continue
        row[f"map_gt_{cname}"] = int(len(g))
        if len(p) == 0:
            row[f"map_cd_{cname}"] = a.map_cap
            continue
        row[f"map_cd_{cname}"] = float(np.mean([min(_chamfer(gi, pj) for pj in p) for gi in g]))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--pdm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--near", type=float, default=15.0)
    ap.add_argument("--det-score", type=float, default=0.3)
    ap.add_argument("--det-dist", type=float, default=2.0)
    ap.add_argument("--map-score", type=float, default=0.3)
    ap.add_argument("--map-cap", type=float, default=10.0)
    ap.add_argument("--map-bad", type=float, default=1.5, help="map_cd above this counts as a bad map")
    ap.add_argument("--workers", type=int, default=16)
    global _A
    _A = a = ap.parse_args()

    files = sorted(a.records.glob("*.npz"))
    print(f"{len(files)} records", flush=True)
    with Pool(a.workers) as pool:
        rows = pool.map(_one, files, chunksize=64)
    per = pd.DataFrame(rows)
    pdm = pd.read_csv(a.pdm)
    pdm = pdm[pdm.valid == True]  # noqa: E712
    df = pdm.merge(per, on="token", how="inner")
    df.to_csv(a.out, index=False)
    print(f"joined {len(df)} tokens\n")

    def grp(name, mask):
        s = df[mask]
        if len(s) == 0:
            return
        bad_map = (s.map_cd_road > a.map_bad).mean()
        miss = (s.det_missed_near > 0).mean()
        print(f"{name:34s} n={len(s):5d} | map_cd_road med {s.map_cd_road.median():5.2f} m, bad-map {bad_map:5.1%} | "
              f"det_recall_near med {s.det_recall_near.median():.2f}, any-miss {miss:5.1%}")

    print("--- DAC ---")
    grp("DAC pass", df.dac == 1)
    grp("DAC fail (all)", df.dac == 0)
    grp("DAC fail, raw also off-road", (df.dac == 0) & (df.raw_dac == 0))
    grp("DAC fail, tracking-induced", (df.dac == 0) & (df.raw_dac == 1))
    print("--- NC / TTC ---")
    grp("NC pass", df.nc == 1)
    grp("NC fail", df.nc < 1)
    grp("TTC pass", df.ttc == 1)
    grp("TTC fail", df.ttc == 0)
    print("--- EP ---")
    grp("EP < 0.5 (score>0)", (df.ep < 0.5) & (df.score > 0))
    grp("EP >= 0.9", df.ep >= 0.9)
    print("\n--- rough split of DAC failures ---")
    f = df[df.dac == 0]
    bm = f.map_cd_road > a.map_bad
    ti = f.raw_dac == 1
    print(f"total {len(f)}: bad map {int(bm.sum())} ({bm.mean():.1%}), tracking-induced with ok map {int((ti & ~bm).sum())}, "
          f"planned off-road with ok map {int((~ti & ~bm).sum())} ({(~ti & ~bm).mean():.1%})")
    f = df[df.nc < 1]
    mm = f.det_missed_near > 0
    print(f"NC failures {len(f)}: with a missed GT agent within {a.near:.0f} m {int(mm.sum())} ({mm.mean():.1%}); "
          f"perception ok {int((~mm).sum())} ({(~mm).mean():.1%})")
    f = df[df.ttc == 0]
    mm = f.det_missed_near > 0
    print(f"TTC failures {len(f)}: with a missed GT agent within {a.near:.0f} m {int(mm.sum())} ({mm.mean():.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
