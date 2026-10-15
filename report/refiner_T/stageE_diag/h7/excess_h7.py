#!/usr/bin/env python3
"""H7 gap share: how much of T-S (R_T4(E2 tau0) - E2 tau_final, 0.65 pts) and M-S sits on tokens where the student's
own perception misses what the teacher needed, BEYOND what the student loses on comparable tokens where its perception
was fine.  Token-level (no double counting across NC / TTC / DAC).

Per A-failing token (A = E2 tau0 fails NC, TTC or DAC):
  obj_miss   : NC (else TTC) cause object is in the V3 GT, BEVFusion detects it (2 m, tau .3) and E2 does not
  obj_ok     : cause object in GT and E2 detects it
  road_miss  : DAC exit boundary point in ROI and E2's nearest predicted road boundary > 1 m (strict: > 0.5 m)
  road_ok    : ... <= 0.5 m
  perception_miss = obj_miss | road_miss ; perception_ok = (obj_ok or no NC/TTC failure) & (road_ok or no DAC failure)
Statistics (log-cluster bootstrap, 2000):
  upper   = sum over perception_miss tokens of d / N              (all of the gap on those tokens)
  excess  = n_miss * (mean d | miss - mean d | ok) / N             (the part beyond the 'perception fine' rate)
Also: fix rates of S, T, M by E2 road-distance bin and by E0 road-distance bin (is the dependence specific to the
student's own BEV or a difficulty shared by all arms?).
Writes: report/refiner_T/stageE_diag/perception_excess.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_h7 import Boot, load_arms  # noqa: E402

TAB = HERE / "tables"


def main():
    X = load_arms().set_index("token")
    N = len(X)
    bt = Boot(X["log"].values)
    nc = pd.read_csv(TAB / "cause_nc_A.csv", index_col=0)
    ttc = pd.read_csv(TAB / "cause_ttc_A.csv", index_col=0)
    dac = pd.read_csv(TAB / "dac_A.csv", index_col=0)
    tok = X.index
    A_fail = ((X["A.nc"] < 1) | (X["A.ttc"] < 1) | (X["A.dac"] < 1)).values

    def objflags(c, thr_col="e2.matched_agn@2"):
        c = c.reindex(tok)
        ing = c["in_gt"].fillna(False).astype(bool).values
        e2 = c[thr_col].fillna(False).astype(bool).values
        te = c["teacher.matched_agn@2"].fillna(False).astype(bool).values
        has = c["in_gt"].notna().values
        return has, ing & e2, ing & ~e2 & te, ing & ~e2 & ~te, has & ~ing

    res = {}
    for variant, objcol, road_thr in (("main (2 m det, road > 1 m)", "e2.matched_agn@2", 1.0),
                                      ("strict (1 m det, road > 0.5 m)", "e2.matched_agn@1", 0.5)):
        h_nc, ok_nc, miss_nc, both_nc, out_nc = objflags(nc, objcol)
        h_tt, ok_tt, miss_tt, both_tt, out_tt = objflags(ttc, objcol)
        # NC cause takes precedence (a token failing both is classified by its NC cause)
        obj_has = h_nc | h_tt
        obj_ok = np.where(h_nc, ok_nc, ok_tt)
        obj_miss = np.where(h_nc, miss_nc, miss_tt)
        d = dac.reindex(tok)
        d_has = d["d_e2"].notna().values & d["dac_bnd_in_roi"].fillna(False).astype(bool).values
        de2 = d["d_e2"].fillna(np.inf).values
        road_miss = d_has & (de2 > road_thr)
        road_ok = d_has & (de2 <= 0.5)
        pmiss = obj_miss | road_miss
        pok = A_fail & ~pmiss & (~obj_has | obj_ok) & (~(X["A.dac"] < 1).values | road_ok)
        r = {"n_A_fail": int(A_fail.sum()), "n_obj_miss": int(obj_miss.sum()), "n_road_miss": int(road_miss.sum()),
             "n_perception_miss": int(pmiss.sum()), "n_perception_ok": int(pok.sum())}
        for arm in ("T", "M"):
            dv = 100 * (X[f"{arm}.pdms"] - X["S.pdms"]).values
            acc = {}
            for name, miss, ok in (("obj", obj_miss, obj_ok), ("road", road_miss & ~obj_miss, road_ok & ~obj_miss)):
                # reference = same failure type, student's perception fine (obj: cause detected; road: <= 0.5 m)
                num_m = np.bincount(bt.idx, np.where(miss, dv, 0), minlength=len(bt.u))
                den_m = np.bincount(bt.idx, miss.astype(float), minlength=len(bt.u))
                num_o = np.bincount(bt.idx, np.where(ok, dv, 0), minlength=len(bt.u))
                den_o = np.bincount(bt.idx, ok.astype(float), minlength=len(bt.u))
                den_all = np.bincount(bt.idx, minlength=len(bt.u)).astype(float)

                def stat(w):
                    Wm, Wo, Wn, Wd = w @ num_m, w @ num_o, w @ den_m, w @ den_o
                    Wa = w @ den_all
                    up = Wm / Wa
                    ex = (Wm - Wn * Wo / np.maximum(Wd, 1e-9)) / Wa
                    return up, ex, Wm / np.maximum(Wn, 1e-9), Wo / np.maximum(Wd, 1e-9)

                pt = stat(np.ones(len(bt.u)))
                bs = np.array(stat(bt.W))
                acc[name] = (np.array(pt), bs)
                q = np.percentile(bs, [2.5, 97.5], axis=1)
                r[f"{arm}-S|{name}"] = {
                    "upper_pts": [pt[0], q[0][0], q[1][0]], "excess_pts": [pt[1], q[0][1], q[1][1]],
                    "mean_d_miss": [pt[2], q[0][2], q[1][2]], "mean_d_ok": [pt[3], q[0][3], q[1][3]],
                    "n_miss": int(miss.sum()), "n_ok_ref": int(ok.sum())}
            pt = acc["obj"][0] + acc["road"][0]
            bs = acc["obj"][1] + acc["road"][1]
            q = np.percentile(bs, [2.5, 97.5], axis=1)
            r[f"{arm}-S|obj+road"] = {"upper_pts": [pt[0], q[0][0], q[1][0]], "excess_pts": [pt[1], q[0][1], q[1][1]]}
        res[variant] = r
    # fix rates by road-distance bin: student vs teachers, E2 bins and E0 bins
    d = dac.copy()
    d = d[d["dac_bnd_in_roi"].astype(bool) & d["d_e2"].notna()]
    cb = Boot(d["log"].values)
    fr = {}
    for src in ("d_e2", "d_e0"):
        v = d[src].fillna(np.inf).values
        for lab, m in (("<=0.5", v <= 0.5), ("0.5-1", (v > 0.5) & (v <= 1)), (">1", v > 1)):
            fr[f"{src} {lab}"] = {"n": int(m.sum())}
            for arm in ("S", "T", "M"):
                fr[f"{src} {lab}"][f"P({arm} fixes DAC)"] = cb.mean(d[f"{arm}_fix"].astype(float).values, m)
    v = d["d_e2"].fillna(np.inf).values
    ok, mi = v <= 0.5, v > 1
    for arm in ("M", "T"):
        yS, yA = d["S_fix"].astype(float).values, d[f"{arm}_fix"].astype(float).values
        def did(w):
            def m(y, k):
                return (w @ np.bincount(cb.idx[k], y[k], minlength=len(cb.u))) / np.maximum(
                    w @ np.bincount(cb.idx[k], minlength=len(cb.u)), 1e-9)
            return (m(yA, ok) - m(yS, ok)) - (m(yA, mi) - m(yS, mi))
        pt = did(np.ones(len(cb.u)))
        bs = did(cb.W)
        fr[f"DiD fix rate ({arm}-S | road ok) - ({arm}-S | road >1 m)"] = [pt, *np.percentile(bs, [2.5, 97.5])]
        # same on the relative scale: ratio of fix rates miss/ok for S vs arm
    res["dac_fix_rate_by_road_bin"] = fr
    out = HERE.parent / "perception_excess.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda o: round(float(o), 3)))


if __name__ == "__main__":
    main()
