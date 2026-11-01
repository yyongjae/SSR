"""Descriptive (NOT decisive) breakdown of the AMENDMENT 5 navtest comparison at the frozen run-3 theta (0 for both arms).
Adapted from report/refiner_T/run2/diag.py: reads runs/stageT3_{T,none}_fold0_seed0/eval_navtest/report_rows.parquet
(+ pred.npz), adds per-city (Las Vegas vs others, and each city) and per-family paired log-cluster bootstrap CIs.
Writes report/refiner_T/navtest/diag.json.   Run from the repo root."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "tools/refiner")
import stageT_decision as SD
from navsim.agents.para_ssr.refiner.decoder import FAMILY_NAME

RUNS = Path("/home/external-user/ssd/yongjae_refiner/runs")
TAG = "stageT3"
EVAL = "eval_navtest"
SELF = "report/refiner_T/run3/selection.json"
OUT = "report/refiner_T/navtest/diag.json"
SPLIT = "/home/external-user/ssd/yongjae_refiner/splits/navtest.parquet"
NB = 10000
sel = json.load(open(SELF))
F = {}
for arm in ("T", "none"):
    th = sel["arms"][arm]["theta"]
    assert th == sel["arms"][arm]["candidates"][TAG]["theta"]
    F[arm] = SD.final_outcomes(SD.load_rows(RUNS / f"{TAG}_{arm}_fold0_seed0" / EVAL), th).set_index(["token", "k"])
idx = F["T"].index.intersection(F["none"].index)
T, N = F["T"].loc[idx], F["none"].loc[idx]
city_of = pd.read_parquet(SPLIT).set_index("token").city
out = {"eval": EVAL, "theta": {a: sel["arms"][a]["theta"] for a in ("T", "none")}, "n": int(len(idx)),
       "n_tokens": int(T.index.get_level_values(0).nunique()), "n_logs": int(T.log.nunique())}


def rate(x): return round(100 * float(np.mean(x)), 3)


for m in ("nc", "ttc", "dac", "ddc", "comfort"):
    out[f"fail_{m}_pp"] = dict(orig=rate(T[f"{m}_orig"] < 1), T=rate(T[f"{m}_final"] < 1), none=rate(N[f"{m}_final"] < 1))
out["fail_ncttc_pp"] = dict(orig=rate(T.fail_ncttc_orig), T=rate(T.fail_ncttc), none=rate(N.fail_ncttc))
out["ep_mean"] = dict(orig=float(T.ep_orig.mean()), T=float(T.ep_final.mean()), none=float(N.ep_final.mean()))
out["pdms_mean"] = dict(orig=float(T.pdms_orig.mean()), T=float(T.pdms_final.mean()), none=float(N.pdms_final.mean()))
mult = lambda D, s: (D[f"nc_{s}"] >= 1) & (D[f"dac_{s}"] >= 1) & (D[f"ddc_{s}"] >= 1)
both = mult(T, "orig") & mult(T, "final") & mult(N, "final")
out["ep_loss_points_on_always_passing"] = dict(n=int(both.sum()), T=float(100 * (T.ep_orig - T.ep_final)[both].mean()),
                                               none=float(100 * (N.ep_orig - N.ep_final)[both].mean()))
fo = T.fail_ncttc_orig > 0
out["ncttc_orig_fail_n"] = int(fo.sum())
out["ncttc_fixed"] = dict(T=int((fo & (T.fail_ncttc == 0)).sum()), none=int((fo & (N.fail_ncttc == 0)).sum()),
                          both=int((fo & (T.fail_ncttc == 0) & (N.fail_ncttc == 0)).sum()),
                          T_only=int((fo & (T.fail_ncttc == 0) & (N.fail_ncttc > 0)).sum()),
                          none_only=int((fo & (T.fail_ncttc > 0) & (N.fail_ncttc == 0)).sum()))
out["ncttc_new"] = dict(T=int((~fo & (T.fail_ncttc > 0)).sum()), none=int((~fo & (N.fail_ncttc > 0)).sum()))
dfo = T.dac_orig < 1
out["dac_fixed"] = dict(orig_fail=int(dfo.sum()), T=int((dfo & (T.dac_final >= 1)).sum()), none=int((dfo & (N.dac_final >= 1)).sum()))
out["dac_new"] = dict(T=int((~dfo & (T.dac_final < 1)).sum()), none=int((~dfo & (N.dac_final < 1)).sum()))
out["new_fail_any"] = dict(T=int(T.new_fail.sum()), none=int(N.new_fail.sum()))

# paired per-draft differences (T - none unless named as a reduction), log-cluster bootstrap
D = pd.DataFrame({"log": T.log.values, "family": T.family.values,
                  "city": city_of.reindex(T.index.get_level_values(0)).values,
                  "P1": 100 * (T.pdms_final.values - N.pdms_final.values),
                  "P2": 100 * (N.fail_ncttc.values - T.fail_ncttc.values),
                  "nc": 100 * ((T.nc_final < 1).astype(float).values - (N.nc_final < 1).astype(float).values),
                  "ttc": 100 * ((T.ttc_final < 1).astype(float).values - (N.ttc_final < 1).astype(float).values),
                  "dac": 100 * ((T.dac_final < 1).astype(float).values - (N.dac_final < 1).astype(float).values),
                  "new": 100 * (T.new_fail.values - N.new_fail.values),
                  "dT": 100 * T.d_pdms.values, "dN": 100 * N.d_pdms.values})
assert D.city.notna().all()
B = lambda g, c, nb=NB: SD.cluster_boot(g[c].to_numpy(np.float64), g.log.to_numpy(), nb)
out["paired_ci"] = {"nc_fail_T_minus_none_pp": B(D, "nc"), "ttc_fail_T_minus_none_pp": B(D, "ttc"),
                    "dac_fail_T_minus_none_pp": B(D, "dac"), "P1_d_pdms_points": B(D, "P1"),
                    "P2_ncttc_reduction_pp": B(D, "P2"), "P3_new_fail_excess_pp": B(D, "new")}


def block(g, nb=NB):
    return dict(n_drafts=int(len(g)), n_tokens=None, n_logs=int(g.log.nunique()),
                P1_d_pdms_points=B(g, "P1", nb), P2_ncttc_reduction_pp=B(g, "P2", nb),
                nc_fail_T_minus_none_pp=B(g, "nc", nb), ttc_fail_T_minus_none_pp=B(g, "ttc", nb),
                dac_fail_T_minus_none_pp=B(g, "dac", nb), P3_new_fail_excess_pp=B(g, "new", nb),
                d_pdms_vs_orig_points=dict(T=float(g.dT.mean()), none=float(g.dN.mean())))


tok = T.index.get_level_values(0)
city = {}
for c, g in D.groupby("city"):
    city[c] = block(g); city[c]["n_tokens"] = int(pd.Index(tok[g.index]).nunique())
lv = D.city == "us-nv-las-vegas-strip"
city["las_vegas"] = block(D[lv]); city["las_vegas"]["n_tokens"] = int(pd.Index(tok[np.flatnonzero(lv)]).nunique())
city["others"] = block(D[~lv]); city["others"]["n_tokens"] = int(pd.Index(tok[np.flatnonzero(~lv)]).nunique())
# fail rates per city group
for nm, msk in (("las_vegas", lv.values), ("others", ~lv.values)):
    Tg, Ng = T[msk], N[msk]
    city[nm]["fail_pp"] = {m: dict(orig=rate(Tg[f"{m}_orig"] < 1), T=rate(Tg[f"{m}_final"] < 1), none=rate(Ng[f"{m}_final"] < 1))
                           for m in ("nc", "ttc", "dac")}
    city[nm]["pdms"] = dict(orig=float(Tg.pdms_orig.mean()), T=float(Tg.pdms_final.mean()), none=float(Ng.pdms_final.mean()))
out["by_city"] = city
D.index = pd.RangeIndex(len(D))

fam = {}
for f, g in T.groupby("family"):
    n = N.loc[g.index]
    Dg = D.iloc[np.flatnonzero(T.family.values == f)]
    fam[FAMILY_NAME.get(int(f), str(f))] = dict(
        n=len(g), ncttc_orig=rate(g.fail_ncttc_orig), ncttc_T=rate(g.fail_ncttc), ncttc_none=rate(n.fail_ncttc),
        nc_orig=rate(g.nc_orig < 1), nc_T=rate(g.nc_final < 1), nc_none=rate(n.nc_final < 1),
        ttc_orig=rate(g.ttc_orig < 1), ttc_T=rate(g.ttc_final < 1), ttc_none=rate(n.ttc_final < 1),
        dac_orig=rate(g.dac_orig < 1), dac_T=rate(g.dac_final < 1), dac_none=rate(n.dac_final < 1),
        ep_T=round(float(g.ep_final.mean()), 4), ep_none=round(float(n.ep_final.mean()), 4),
        pdms_orig=round(float(g.pdms_orig.mean()), 4), pdms_T=round(float(g.pdms_final.mean()), 4),
        pdms_none=round(float(n.pdms_final.mean()), 4),
        P1_d_pdms_points=B(Dg, "P1", 2000), P2_ncttc_reduction_pp=B(Dg, "P2", 2000))
out["by_family"] = fam
Path(OUT).write_text(json.dumps(out, indent=1, default=float))

# ---- part 2: what the arms output (displacement of tau1 vs tau0)
P = {a: np.load(RUNS / f"{TAG}_{a}_fold0_seed0" / EVAL / "pred.npz") for a in ("T", "none")}
out2 = {}
for a, d in P.items():
    v = d["draft_valid"]
    t0, t1 = d["tau0"][v], d["tau1"][v]
    fm = d["family"][v]
    arc = lambda t: np.linalg.norm(np.diff(np.concatenate([np.zeros_like(t[:, :1, :2]), t[:, :, :2]], 1), axis=1), axis=-1).sum(1)
    short = arc(t0) - arc(t1)
    lat = np.linalg.norm(t1[:, :, :2] - t0[:, :, :2], axis=-1).max(1)
    rec = dict(mean_arc_short_m=float(short.mean()), frac_short_gt_1m=float((short > 1).mean()),
               mean_max_disp_m=float(lat.mean()), frac_disp_gt_0p3m=float((lat > 0.3).mean()), by_family={})
    for f in np.unique(fm):
        m = fm == f
        rec["by_family"][FAMILY_NAME.get(int(f), str(int(f)))] = dict(arc_short_m=round(float(short[m].mean()), 3),
                                                                      max_disp_m=round(float(lat[m].mean()), 3))
    out2[a] = rec
out["part2"] = out2
Path(OUT).write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k not in ("by_family", "by_city", "part2")}, indent=1, default=float))
