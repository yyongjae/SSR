#!/usr/bin/env python3
"""H7 analysis: does the student BEV (E2's own camera BEV) lack the obstacle / road information that the teacher BEVs
carry, so that the student refiner cannot fix what R_T4 (BEVFusion) / R_M4 (ReSMap) fix on the SAME drafts?

Arms (all on E2's own tau0 drafts, batched official scorer):  A = E2 tau0, S = E2 tau_final (student refiner),
T = R_T4(A), M = R_M4(A), N = R_none4(A).  E0 arms from e0_teacher_refine for the gap decomposition.
Perception proxies:
  NC / TTC : cause object of A's failure (pdm_attr copy on A) -> aux V3 GT box (gt_idx) -> matched by
             E2 det head (this dump), E0 det head (recomputed; checked vs build_det), BEVFusion (build_det 'teacher',
             = the cache R_T4 reads).  'detected' = class-agnostic greedy 1:1, 2 m, score >= 0.3 (build_det).
  DAC      : exit point of A (nearest PDM drivable boundary point to the deepest outside corner at first exit) ->
             distance to nearest predicted road polyline (score >= 0.3, class road) of E2 / E0 map heads.
Outputs: report/refiner_T/stageE_diag/perception.json (+ h7/tables/*.csv)
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_h7 import Boot, H7, MET, load_arms  # noqa: E402

BD = Path("/home/external-user/yongjae/SSR/report/perception_reliability/build_det")
E0_REC = Path("/home/external-user/yongjae/SSR/work_dirs/eval/para_ssr_interaction_final_aux/records")
E2_REC = H7 / "e2_aux/records"
OUTJ = HERE.parent / "perception.json"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)
R = {}


def load_attr(arm):
    rows = []
    for f in sorted(glob.glob(str(H7 / f"attr_{arm}/shards/*.pkl"))):
        d = pickle.load(open(f, "rb"))
        assert not d["errors"], d["errors"][:1]
        rows += d["rows"]
    return pd.DataFrame(rows).set_index("token")


def ci3(t, nd=1, s=100.0):
    return f"{s * t[0]:.{nd}f} [{s * t[1]:.{nd}f}, {s * t[2]:.{nd}f}]"


def rate(bt, y, m):
    """cluster-bootstrap mean of y over mask m (tuple pt, lo, hi) + n"""
    m = np.asarray(m, bool)
    if m.sum() == 0:
        return (np.nan, np.nan, np.nan), 0
    return bt.mean(np.asarray(y, float), m), int(m.sum())


def diff_rate(bt, y, m1, m2):
    y = np.asarray(y, float)
    n1 = np.bincount(bt.idx[m1], y[m1], minlength=len(bt.u)); d1 = np.bincount(bt.idx[m1], minlength=len(bt.u))
    n2 = np.bincount(bt.idx[m2], y[m2], minlength=len(bt.u)); d2 = np.bincount(bt.idx[m2], minlength=len(bt.u))
    pt = n1.sum() / max(d1.sum(), 1) - n2.sum() / max(d2.sum(), 1)
    bs = (bt.W @ n1) / np.maximum(bt.W @ d1, 1e-9) - (bt.W @ n2) / np.maximum(bt.W @ d2, 1e-9)
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def share(bt, v, m, N):
    """sum of v over mask m / N (per-token average contribution), cluster bootstrap"""
    v = np.where(m, np.asarray(v, float), 0.0)
    num = np.bincount(bt.idx, v, minlength=len(bt.u))
    den = np.bincount(bt.idx, minlength=len(bt.u)).astype(float)
    return num.sum() / den.sum(), *np.percentile((bt.W @ num) / (bt.W @ den), [2.5, 97.5])


def pt_to_polylines(q, polys):
    """min distance point q (2,) -> polylines (K,P,2)"""
    if len(polys) == 0:
        return np.inf
    A = polys[:, :-1].reshape(-1, 2).astype(np.float64)
    B = polys[:, 1:].reshape(-1, 2).astype(np.float64)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((q - A) * AB).sum(1) / L2, 0, 1)
    return float(np.sqrt((((A + t[:, None] * AB) - q) ** 2).sum(1)).min())


def main():
    X = load_arms()
    X = X.set_index("token")
    toks = X.index.values
    bt = Boot(X["log"].values)
    N = len(X)
    # ---------------------------------------------------------------- 0. arm table and the gap decomposition
    arms = {}
    for a in ["E0", "E0T", "E0M", "E0N", "A", "S", "T", "M", "N"]:
        arms[a] = {m: 100 * float(X[f"{a}.{m}"].mean()) for m in MET}
    R["arms_x100"] = arms
    con = {}
    for a, b in [("T", "S"), ("M", "S"), ("T", "A"), ("M", "A"), ("S", "A"), ("N", "A"), ("E0T", "S"), ("E0T", "T"),
                 ("A", "E0"), ("E0T", "E0"), ("T", "M")]:
        con[f"{a}-{b}"] = {m: bt.mean(100 * (X[f"{a}.{m}"] - X[f"{b}.{m}"]).values) for m in MET}
    R["contrasts_pts"] = con
    # ---------------------------------------------------------------- 1. token sets
    sets = {}
    for m, t in [("nc", "T"), ("ttc", "T"), ("dac", "T"), ("nc", "M"), ("ttc", "M"), ("dac", "M")]:
        Af = X[f"A.{m}"] < 1
        tf = X[f"{t}.{m}"] == 1
        sf = X[f"S.{m}"] == 1
        sets[f"{m}_{t}"] = dict(A_fail=int(Af.sum()), both_fix=int((Af & tf & sf).sum()),
                                teacher_only=int((Af & tf & ~sf).sum()), student_only=int((Af & ~tf & sf).sum()),
                                neither=int((Af & ~tf & ~sf).sum()))
    R["fix_sets"] = sets

    # ---------------------------------------------------------------- 2. cause objects (A) + detection
    at = load_attr("A").reindex(toks)
    ats = load_attr("S")
    # consistency of the attribution copy with the batched scorer
    chk = {}
    for m, col in [("nc", "re_no_at_fault_collisions"), ("dac", "re_drivable_area_compliance"),
                   ("ttc", "re_time_to_collision_within_bound")]:
        ok = at[col].notna()
        chk[m] = int((np.abs(at.loc[ok, col].values - X.loc[ok, f"A.{m}"].values) > 1e-9).sum())
    chk["n_attr"] = int(at["re_score"].notna().sum())
    R["attr_consistency_A_vs_batched_mismatch"] = chk

    gto = pd.read_parquet(BD / "gt_objects.parquet",
                          columns=["token", "gt_idx", "class", "dist", "speed", "moving", "tier",
                                   "inter.matched_agn@2", "inter.matched_agn@1", "inter.err_vel", "inter.nn_dist",
                                   "teacher.matched_agn@2", "teacher.matched_agn@1", "teacher.err_vel",
                                   "teacher.nn_dist", "teacher.matched_any@2"])
    e2d = pd.read_parquet(H7 / "det_gt_e2.parquet")
    have_e2 = set(e2d.token.unique())
    g = gto.merge(e2d, on=["token", "gt_idx"], how="inner")
    # check: recomputed E0 == build_det inter
    R["check_e0_recompute_vs_build_det"] = {
        "n_obj": int(len(g)),
        "matched_agn@2_mismatch": int((g["e0re.matched_agn@2"] != g["inter.matched_agn@2"]).sum()),
        "matched_agn@1_mismatch": int((g["e0re.matched_agn@1"] != g["inter.matched_agn@1"]).sum()),
        "nn_dist_maxabs": float(np.nanmax(np.abs(np.where(np.isfinite(g["inter.nn_dist"]),
                                                          g["e0re.nn_dist"] - g["inter.nn_dist"], 0)))),
    }
    # global perception, E2 vs E0 vs teacher on the same objects (tokens with an E2 dump)
    glob_r = {}
    lg = X.loc[g.token.values, "log"].values
    btg = Boot(lg)
    for name, mask in [("all", np.ones(len(g), bool)), ("T0", g.tier.values == "T0"),
                       ("T0T1_0-16m_vehicle", (np.isin(g.tier.values, ["T0", "T1"])) & (g.dist.values < 16) &
                        (g["class"].values == 0)),
                       ("VRU", np.isin(g["class"].values, [1, 2])), ("static", g["class"].values >= 3)]:
        glob_r[name] = {"n": int(mask.sum())}
        for tag, col in [("E2", "e2.matched_agn@2"), ("E0", "inter.matched_agn@2"), ("BEVFusion", "teacher.matched_agn@2"),
                         ("E2@1m", "e2.matched_agn@1"), ("E0@1m", "inter.matched_agn@1"),
                         ("BEVFusion@1m", "teacher.matched_agn@1")]:
            glob_r[name][tag] = btg.mean(g[col].values.astype(float), mask)
        glob_r[name]["E2-E0"] = btg.mean((g["e2.matched_agn@2"].astype(float) - g["inter.matched_agn@2"].astype(float)).values, mask)
        glob_r[name]["BEVF-E2"] = btg.mean((g["teacher.matched_agn@2"].astype(float) - g["e2.matched_agn@2"].astype(float)).values, mask)
    R["global_recall_2m_tau0.3"] = glob_r
    R["n_tokens_with_e2_det"] = len(have_e2)

    # per-token cause-object table for NC and TTC failures of A
    def cause_rows(kind):
        m = X[f"A.{kind}"] < 1
        c = at.loc[m, [f"{kind}_track", f"{kind}_obj_group", f"{kind}_obj_dist", f"{kind}_obj_speed",
                       f"{kind}_obj_moving", f"{kind}_vis", f"{kind}_gt_idx", f"{kind}_obj_bearing_deg",
                       f"{kind}_tier_own", f"{kind}_obj_yf", f"{kind}_obj_xr"]].copy()
        c.columns = [k.replace(f"{kind}_", "") for k in c.columns]
        c = c.rename_axis(None); c["token"] = c.index.values
        c["gt_idx"] = c["gt_idx"].fillna(-1).astype(int)
        c = c.merge(g[["token", "gt_idx", "e2.matched_agn@2", "e2.matched_agn@1", "inter.matched_agn@2",
                       "teacher.matched_agn@2", "teacher.matched_agn@1", "e2.err_vel", "inter.err_vel",
                       "teacher.err_vel", "e2.err_pos", "e2.nn_dist", "teacher.nn_dist", "e2.max_score_2m"]],
                    on=["token", "gt_idx"], how="left")
        c = c.set_index("token")
        c["has_e2"] = c.index.isin(have_e2)
        c["in_gt"] = c["gt_idx"] >= 0
        for t in ("T", "M"):
            c[f"{t}_fix"] = X.loc[c.index, f"{t}.{kind}"].values == 1
        c["S_fix"] = X.loc[c.index, f"S.{kind}"].values == 1
        c["S_cause_same"] = [ (ats.loc[t, f"{kind}_track"] == c.loc[t, "track"]) if (t in ats.index and X.loc[t, f"S.{kind}"] < 1) else np.nan for t in c.index]
        c["dT_S"] = 100 * (X.loc[c.index, "T.pdms"] - X.loc[c.index, "S.pdms"]).values
        c["dM_S"] = 100 * (X.loc[c.index, "M.pdms"] - X.loc[c.index, "S.pdms"]).values
        c["log"] = X.loc[c.index, "log"].values
        # perception category for the cause object
        e2 = c["e2.matched_agn@2"].fillna(False).astype(bool)
        te = c["teacher.matched_agn@2"].fillna(False).astype(bool)
        cat = np.where(~c["in_gt"], "not_in_GT(" + c["vis"].astype(str) + ")",
                       np.where(e2 & te, "E2+BEVF", np.where(e2 & ~te, "E2_only",
                                np.where(~e2 & te, "BEVF_only", "neither"))))
        c["cat"] = cat
        return c

    out_c = {}
    for kind in ("nc", "ttc"):
        c = cause_rows(kind)
        c.to_csv(TAB / f"cause_{kind}_A.csv")
        cb = Boot(c["log"].values)
        r = {"n_A_fail": int(len(c)), "has_e2_det": int(c["has_e2"].sum())}
        grp = {"teacher_only": c.T_fix & ~c.S_fix, "both_fix": c.T_fix & c.S_fix, "neither": ~c.T_fix & ~c.S_fix,
               "student_only": ~c.T_fix & c.S_fix}
        desc = {}
        for gname, gm in grp.items():
            gm = gm.values
            sub = c[gm]
            gi = gm & c.in_gt.values
            d = {"n": int(gm.sum()),
                 "class_group": sub.obj_group.value_counts(normalize=True).round(3).to_dict(),
                 "vis": sub.vis.value_counts(normalize=True).round(3).to_dict(),
                 "dist_t0_median": float(np.nanmedian(sub.obj_dist)), "dist_t0_p25_p75":
                     [float(np.nanpercentile(sub.obj_dist, 25)), float(np.nanpercentile(sub.obj_dist, 75))],
                 "speed_median": float(np.nanmedian(sub.obj_speed)),
                 "moving_frac": float(np.nanmean(sub.obj_moving.astype(float))),
                 "front_fov_frac": float(np.mean((sub.obj_yf > 0) & (np.abs(sub.obj_bearing_deg) <= 80))),
                 "in_gt_frac": float(sub.in_gt.mean()),
                 "tier_own": sub.tier_own.value_counts(normalize=True).round(3).to_dict()}
            for tag, col in [("E2_det2m", "e2.matched_agn@2"), ("E0_det2m", "inter.matched_agn@2"),
                             ("BEVF_det2m", "teacher.matched_agn@2"), ("E2_det1m", "e2.matched_agn@1"),
                             ("BEVF_det1m", "teacher.matched_agn@1")]:
                y = c[col].fillna(False).astype(float).values
                d[tag + "_among_in_gt"] = rate(cb, y, gi)
            vm = gi & c.obj_moving.fillna(False).astype(bool).values
            for tag, col in [("E2", "e2.err_vel"), ("E0", "inter.err_vel"), ("BEVF", "teacher.err_vel")]:
                v = c[col].values[vm]
                d[f"{tag}_err_vel_moving_median"] = float(np.nanmedian(v)) if np.isfinite(v).any() else None
            d["cat"] = sub.cat.value_counts().to_dict()
            desc[gname] = d
        r["by_fix_group"] = desc
        # key test: among tokens where the teacher fixes (fixable with teacher BEV), does the student's fix rate depend
        # on whether its own det head saw the cause object?
        tf = c.T_fix.values
        seen = c["e2.matched_agn@2"].fillna(False).astype(bool).values & c.in_gt.values
        miss = c.in_gt.values & ~seen
        out_gt = ~c.in_gt.values
        y = c.S_fix.values.astype(float)
        r["P(S_fix|T_fix, E2 detects)"] = rate(cb, y, tf & seen)
        r["P(S_fix|T_fix, E2 misses, in GT)"] = rate(cb, y, tf & miss)
        r["P(S_fix|T_fix, not in GT)"] = rate(cb, y, tf & out_gt)
        r["diff detects-misses"] = diff_rate(cb, y, tf & seen, tf & miss) if (tf & miss).sum() else None
        # same for the teacher: does R_T4 fix more where BEVFusion sees the object?
        tseen = c["teacher.matched_agn@2"].fillna(False).astype(bool).values & c.in_gt.values
        r["P(T_fix|BEVF detects)"] = rate(cb, tf.astype(float), tseen)
        r["P(T_fix|BEVF misses, in GT)"] = rate(cb, tf.astype(float), c.in_gt.values & ~tseen)
        r["P(S_fix|E2 detects) all A-fail"] = rate(cb, y, seen)
        r["P(S_fix|E2 misses, in GT) all A-fail"] = rate(cb, y, miss)
        # E2-vs-BEVF detection on teacher-only vs both-fix, paired within token
        for gname in ("teacher_only", "both_fix"):
            gm = grp[gname].values & c.in_gt.values
            r[f"BEVF-E2 det gap on {gname}"] = rate(cb, (c["teacher.matched_agn@2"].fillna(False).astype(float) -
                                                         c["e2.matched_agn@2"].fillna(False).astype(float)).values, gm)
        # contribution to the T-S gap by perception category (per navtest token, points)
        contrib = {}
        for k in sorted(c.cat.unique()):
            mm = (c.cat == k).values
            full = np.zeros(N)
            full[X.index.get_indexer(c.index[mm])] = c.dT_S.values[mm]
            contrib[k] = {"n": int(mm.sum()), "sum_dT_S_per_navtest_token": share(bt, full, np.ones(N, bool), N),
                          "teacher_only_n": int((mm & grp["teacher_only"].values).sum())}
        r["T-S_contribution_by_cat"] = contrib
        out_c[kind] = r
    R["cause"] = out_c

    # ---------------------------------------------------------------- 3. DAC exits (A) vs predicted road boundaries
    m = (X["A.dac"] < 1).values
    dac = at.loc[X.index[m], ["dac_bnd_xr", "dac_bnd_yf", "dac_depth_max", "dac_stage", "dac_side_expert",
                              "dac_first_time_s"]].copy()
    # as pdm_attr/merge_and_check.link_map: exit boundary point in ROI; distance to GT road polylines (class 0)
    inroi, dgt = [], []
    for t, row in dac.iterrows():
        bx, by = row.dac_bnd_xr, row.dac_bnd_yf
        if not np.isfinite([bx, by]).all():
            inroi.append(False); dgt.append(np.nan); continue
        inroi.append(bool(-32 <= bx <= 32 and 0 <= by <= 32))
        with np.load(E0_REC / f"{t}.npz") as z:
            dgt.append(pt_to_polylines(np.array([bx, by]), z["map_gt_points"][z["map_gt_labels"] == 0]))
    dac["dac_bnd_in_roi"] = inroi
    dac["dac_bnd_to_gt_road_m"] = dgt
    d_e2, d_e0 = [], []
    for t, row in dac.iterrows():
        q = np.array([row.dac_bnd_xr, row.dac_bnd_yf], float)
        vals = []
        for rec in (E2_REC, E0_REC):
            f = rec / f"{t}.npz"
            if not f.exists() or not np.isfinite(q).all():
                vals.append(np.nan)
                continue
            with np.load(f) as z:
                P, s, l = z["map_pred_points"], z["map_pred_scores"], z["map_pred_labels"]
            vals.append(pt_to_polylines(q, P[(s >= 0.3) & (l == 0)]))
        d_e2.append(vals[0])
        d_e0.append(vals[1])
    dac["d_e2"] = d_e2
    dac["d_e0"] = d_e0
    for t in ("T", "M"):
        dac[f"{t}_fix"] = X.loc[dac.index, f"{t}.dac"].values == 1
    dac["S_fix"] = X.loc[dac.index, "S.dac"].values == 1
    dac["dM_S"] = 100 * (X.loc[dac.index, "M.pdms"] - X.loc[dac.index, "S.pdms"]).values
    dac["dT_S"] = 100 * (X.loc[dac.index, "T.pdms"] - X.loc[dac.index, "S.pdms"]).values
    dac["log"] = X.loc[dac.index, "log"].values
    dac.to_csv(TAB / "dac_A.csv")
    rd = {"n_A_fail": int(m.sum()), "in_roi_frac": float(dac.dac_bnd_in_roi.mean()),
          "n_with_e2_map": int(np.isfinite(dac.d_e2).sum())}
    cb = Boot(dac["log"].values)
    use = np.isfinite(dac.d_e2.values) & dac.dac_bnd_in_roi.fillna(False).astype(bool).values
    grp = {"M_only": dac.M_fix & ~dac.S_fix, "both_fix_M": dac.M_fix & dac.S_fix, "neither_M": ~dac.M_fix & ~dac.S_fix,
           "S_only_vs_M": ~dac.M_fix & dac.S_fix, "T_only": dac.T_fix & ~dac.S_fix}
    by = {}
    for gname, gm in grp.items():
        gm = gm.values & use
        d = {"n": int(gm.sum())}
        for tag, col in (("E2", "d_e2"), ("E0", "d_e0")):
            v = dac[col].values
            d[f"{tag}_d_median"] = float(np.nanmedian(v[gm])) if gm.any() else None
            d[f"{tag}_d<=0.5"] = rate(cb, (v <= 0.5).astype(float), gm)[0]
            d[f"{tag}_d>1"] = rate(cb, (v > 1.0).astype(float), gm)[0]
        d["gt_road_dist_median"] = float(np.nanmedian(dac.dac_bnd_to_gt_road_m.values[gm])) if gm.any() else None
        d["depth_max_median"] = float(np.nanmedian(dac.dac_depth_max.values[gm])) if gm.any() else None
        d["stage"] = dac.dac_stage[gm].value_counts(normalize=True).round(3).to_dict()
        by[gname] = d
    rd["by_fix_group"] = by
    y = dac.S_fix.values.astype(float)
    mf = dac.M_fix.values
    seen = use & (dac.d_e2.values <= 0.5)
    miss = use & (dac.d_e2.values > 1.0)
    mid = use & (dac.d_e2.values > 0.5) & (dac.d_e2.values <= 1.0)
    rd["P(S_fix|M_fix, E2 road<=0.5m)"] = rate(cb, y, mf & seen)
    rd["P(S_fix|M_fix, E2 road 0.5-1m)"] = rate(cb, y, mf & mid)
    rd["P(S_fix|M_fix, E2 road>1m)"] = rate(cb, y, mf & miss)
    rd["P(S_fix|M_fix, outside ROI / no dump)"] = rate(cb, y, mf & ~use)
    rd["diff seen-miss"] = diff_rate(cb, y, mf & seen, mf & miss)
    rd["P(S_fix|E2 road<=0.5m) all A-fail"] = rate(cb, y, seen)
    rd["P(S_fix|E2 road>1m) all A-fail"] = rate(cb, y, miss)
    contrib = {}
    for k, mm in (("E2_road<=0.5", seen), ("E2_road0.5-1", mid), ("E2_road>1", miss), ("outside_roi_or_nodump", ~use)):
        for col in ("dM_S", "dT_S"):
            full = np.zeros(N)
            full[X.index.get_indexer(dac.index[mm])] = dac[col].values[mm]
            contrib[f"{k}|{col}"] = {"n": int(mm.sum()), "per_navtest_token": share(bt, full, np.ones(N, bool), N)}
    rd["contribution_by_E2_road_dist"] = contrib
    # E2 vs E0 map at the same points (paired)
    both = use & np.isfinite(dac.d_e0.values)
    rd["paired_E2_minus_E0_d_median_all_A_fail"] = float(np.nanmedian(dac.d_e2.values[both] - dac.d_e0.values[both]))
    rd["E2_d<=0.5_all"] = rate(cb, (dac.d_e2.values <= 0.5).astype(float), both)
    rd["E0_d<=0.5_all"] = rate(cb, (dac.d_e0.values <= 0.5).astype(float), both)
    R["dac"] = rd

    # ---------------------------------------------------------------- 4. overall gap attribution summary
    Afail_any = ((X["A.nc"] < 1) | (X["A.ttc"] < 1) | (X["A.dac"] < 1)).values
    dTS = (100 * (X["T.pdms"] - X["S.pdms"])).values
    R["T-S_split"] = {"A_fail_any": share(bt, dTS, Afail_any, N), "A_pass": share(bt, dTS, ~Afail_any, N)}
    json.dump(R, open(OUTJ, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:20000])


if __name__ == "__main__":
    main()
