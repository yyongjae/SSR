"""Robustness / sensitivity lens on the stage-T run-3 PASS (descriptive; read-only on runs/)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/external-user/yongjae/SSR"); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools/refiner"))
from stageT_decision import load_rows, final_outcomes, cluster_boot, sweep, pick_theta, THETAS
RUNS = Path("/home/external-user/ssd/yongjae_refiner/runs")
OUT = Path(__file__).resolve().parent
FAM = {0: 'identity', 1: 'small', 2: 'lconst', 3: 'ignore_brake', 4: 'creep', 5: 'lat', 6: 'combined', 7: 'cv', 8: 'hdrift'}
NB = 10000
cities = pd.concat([pd.read_parquet(f"/home/external-user/ssd/yongjae_refiner/splits/{s}.parquet")[["log", "map_location"]]
                    for s in ("train", "dev")]).drop_duplicates("log").set_index("log").map_location

def rows(tag, arm, ev):
    return load_rows(RUNS / f"{tag}_{arm}_fold0_seed0" / ev)

def paired(rT, rN, thT, thN):
    fT, fN = final_outcomes(rT, thT), final_outcomes(rN, thN)
    m = fT.merge(fN, on=["token", "k", "log", "family"], suffixes=("_T", "_none"), validate="one_to_one")
    m["city"] = m.log.map(cities)
    m["nc_fail_T"] = (m.nc_final_T < 1).astype(float); m["nc_fail_none"] = (m.nc_final_none < 1).astype(float)
    m["ttc_fail_T"] = (m.ttc_final_T < 1).astype(float); m["ttc_fail_none"] = (m.ttc_final_none < 1).astype(float)
    return m

def ends(m, nb=NB):
    if len(m) == 0: return None
    L = m.log.to_numpy()
    B = lambda d: {k: (round(100 * v, 3) if k in ("mean", "lo", "hi") else v) for k, v in cluster_boot(np.asarray(d, float), L, nb).items()}
    return dict(P1=B(m.pdms_final_T - m.pdms_final_none), P2=B(m.fail_ncttc_none - m.fail_ncttc_T),
                P3=B(m.new_fail_T - m.new_fail_none), DAC=B(m.fail_dac_T - m.fail_dac_none),
                NC_red=B(m.nc_fail_none - m.nc_fail_T), TTC_red=B(m.ttc_fail_none - m.ttc_fail_T),
                sanT=B(m.d_pdms_T), sanN=B(m.d_pdms_none), modT=round(float(m.modified_T.mean()), 4),
                modN=round(float(m.modified_none.mean()), 4))

def verdict(e):
    if e is None: return None
    return "PASS" if (e["P1"]["lo"] > 0 and e["P2"]["lo"] > 0 and e["P3"]["hi"] <= 0.5) else "not-PASS"

def fmt(e):
    if e is None: return "n/a"
    f = lambda k: f"{e[k]['mean']:+.2f} [{e[k]['lo']:+.2f},{e[k]['hi']:+.2f}]"
    return f"n={e['P1']['n']} logs={e['P1']['n_logs']} P1 {f('P1')} P2 {f('P2')} P3 {f('P3')} DAC {f('DAC')} NCred {f('NC_red')} TTCred {f('TTC_red')} -> {verdict(e)}"

def block(m, label, res):
    out = {}
    out["all"] = ends(m)
    for c in sorted(m.family.unique()):
        out[f"fam={FAM[c]}"] = ends(m[m.family == c])
    for c in sorted(m.family.unique()):
        out[f"LOFO-{FAM[c]}"] = ends(m[m.family != c])
    ex = lambda fs: m[~m.family.isin([k for k, v in FAM.items() if v in fs])]
    inc = lambda fs: m[m.family.isin([k for k, v in FAM.items() if v in fs])]
    out["excl cv"] = ends(ex({"cv"}))
    out["excl cv,lat,combined"] = ends(ex({"cv", "lat", "combined"}))
    out["excl cv,lat,combined,creep"] = ends(ex({"cv", "lat", "combined", "creep"}))
    out["only lconst (rule: L-const)"] = ends(inc({"lconst"}))
    out["only cv (rule: const-velocity)"] = ends(inc({"cv"}))
    out["rule real-draft-like: cv+lconst"] = ends(inc({"cv", "lconst"}))
    out["longitudinal only: lconst,ignore_brake,creep"] = ends(inc({"lconst", "ignore_brake", "creep"}))
    out["lateral synthetic: lat,combined"] = ends(inc({"lat", "combined"}))
    for c in sorted(m.city.dropna().unique()):
        out[f"city={c}"] = ends(m[m.city == c])
        out[f"LOCO-{c}"] = ends(m[m.city != c])
    res[label] = out
    print(f"\n===== {label}"); [print(f"{k:48s} {fmt(v)}") for k, v in out.items()]
    return out

def per_log(m, label, res):
    g = m.assign(d1=m.pdms_final_T - m.pdms_final_none, d2=m.fail_ncttc_none - m.fail_ncttc_T).groupby("log")
    pl = pd.DataFrame(dict(n=g.size(), p1=100 * g.d1.mean(), p2=100 * g.d2.mean(), p2_cnt=g.d2.sum(), p1_sum=g.d1.sum(),
                           city=g.city.first()))
    tot2, tot1 = pl.p2_cnt.sum(), pl.p1_sum.sum()
    s2 = pl.sort_values("p2_cnt", ascending=False); s1 = pl.sort_values("p1_sum", ascending=False)
    top10 = s2.index[:10]
    rest = m[~m.log.isin(top10)]
    top10_p1 = s1.index[:10]
    r = dict(n_logs=len(pl), P2_total_net_drafts=float(tot2), P2_top10_share=float(s2.p2_cnt[:10].sum() / tot2),
             P2_top1_share=float(s2.p2_cnt[:1].sum() / tot2), P2_top20_share=float(s2.p2_cnt[:20].sum() / tot2),
             P2_pos_logs=int((pl.p2_cnt > 0).sum()), P2_zero_logs=int((pl.p2_cnt == 0).sum()), P2_neg_logs=int((pl.p2_cnt < 0).sum()),
             P2_gross_pos=float(pl.p2_cnt[pl.p2_cnt > 0].sum()), P2_gross_neg=float(pl.p2_cnt[pl.p2_cnt < 0].sum()),
             P2_perlog_pp_quantiles=pl.p2.quantile([0, .05, .1, .25, .5, .75, .9, .95, 1]).round(3).to_dict(),
             P2_perlog_pp_mean_unweighted=float(pl.p2.mean()),
             P1_top10_share=float(s1.p1_sum[:10].sum() / tot1), P1_pos_logs=int((pl.p1 > 0).sum()), P1_neg_logs=int((pl.p1 < 0).sum()),
             P1_perlog_pts_quantiles=pl.p1.quantile([0, .05, .1, .25, .5, .75, .9, .95, 1]).round(3).to_dict(),
             top10_P2_logs=s2.head(10)[["n", "p2", "p2_cnt", "city"]].reset_index().to_dict(orient="records"),
             P2_by_city_share=(pl.groupby("city").p2_cnt.sum() / tot2).round(3).to_dict(),
             excl_top10_P2_logs=ends(rest), excl_top10_P1_logs=ends(m[~m.log.isin(top10_p1)]),
             excl_top20_P2_logs=ends(m[~m.log.isin(s2.index[:20])]))
    # sign test over logs with nonzero P2
    from scipy.stats import binomtest
    r["P2_sign_test_p"] = float(binomtest(r["P2_pos_logs"], r["P2_pos_logs"] + r["P2_neg_logs"]).pvalue)
    r["P1_sign_test_p"] = float(binomtest(r["P1_pos_logs"], r["P1_pos_logs"] + r["P1_neg_logs"]).pvalue)
    res[label] = r
    print(f"\n===== {label}")
    for k, v in r.items():
        if k.startswith("excl"): print(f"{k:30s} {fmt(v)}")
        else: print(f"{k:30s} {v}")

if __name__ == "__main__":
    res = {}
    dT, dN = rows("stageT3", "T", "eval_dev"), rows("stageT3", "none", "eval_dev")
    oT, oN = rows("stageT3", "T", "eval_train_fold0"), rows("stageT3", "none", "eval_train_fold0")
    # (a) theta: OOF sweep under the OLD 'all' budget, and 0.5
    th = {}
    for arm, o in (("T", oT), ("none", oN)):
        sw = sweep([o], THETAS, "all")
        th[arm] = dict(all=pick_theta(sw, 0.5), max_ep_loss_all=float(sw.ep_loss_points.max()),
                       mod_at_05=float(sw[sw.theta == 0.5].modified_frac.iloc[0]))
    res["theta_all_budget"] = th; print("theta under 'all' budget:", th)
    res["dev_theta0"] = ends(paired(dT, dN, 0.0, 0.0)); print("dev theta 0/0  ", fmt(res["dev_theta0"]))
    res["dev_theta0.5"] = ends(paired(dT, dN, 0.5, 0.5)); print("dev theta .5/.5", fmt(res["dev_theta0.5"]))
    res["dev_theta_allbudget"] = ends(paired(dT, dN, th["T"]["all"], th["none"]["all"])); print("dev theta all ", fmt(res["dev_theta_allbudget"]))
    # theta grid descriptive (common theta)
    res["dev_theta_grid"] = {}
    for t in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9):
        e = ends(paired(dT, dN, t, t), 2000); res["dev_theta_grid"][t] = e; print(f"dev theta {t}  ", fmt(e))
    # (b)(c) dev at selected theta 0
    m = paired(dT, dN, 0.0, 0.0)
    block(m, "dev_theta0_blocks", res)
    per_log(m, "dev_theta0_perlog", res)
    m5 = paired(dT, dN, 0.5, 0.5)
    block(m5, "dev_theta0.5_blocks", res)
    # (d) OOF train fold 0
    mo = paired(oT, oN, 0.0, 0.0)
    block(mo, "oof_fold0_theta0_blocks", res)
    per_log(mo, "oof_fold0_theta0_perlog", res)
    res["oof_theta0.5"] = ends(paired(oT, oN, 0.5, 0.5)); print("oof theta .5", fmt(res["oof_theta0.5"]))
    # runs 1/2/3 side by side (dev + oof, theta 0 = what each run selected)
    res["runs"] = {}
    for tag in ("stageT", "stageT2", "stageT3"):
        for ev in ("eval_dev", "eval_train_fold0"):
            mm = paired(rows(tag, "T", ev), rows(tag, "none", ev), 0.0, 0.0)
            arm = {}
            for a in ("T", "none"):
                arm[a] = dict(pdms=round(float(mm[f"pdms_final_{a}"].mean()), 4), d_pdms_pts=round(100 * float(mm[f"d_pdms_{a}"].mean()), 3),
                              ncttc_fail_pp=round(100 * float(mm[f"fail_ncttc_{a}"].mean()), 3), nc_fail_pp=round(100 * float(mm[f"nc_fail_{a}"].mean()), 3),
                              ttc_fail_pp=round(100 * float(mm[f"ttc_fail_{a}"].mean()), 3), dac_fail_pp=round(100 * float(mm[f"fail_dac_{a}"].mean()), 3),
                              new_fail_pp=round(100 * float(mm[f"new_fail_{a}"].mean()), 3))
            e = ends(mm, 2000)
            res["runs"][f"{tag}/{ev}"] = dict(arms=arm, orig=dict(pdms=round(float(mm.pdms_orig_T.mean()), 4),
                ncttc_fail_pp=round(100 * float(mm.fail_ncttc_orig_T.mean()), 3)), endpoints=e)
            print(f"\n{tag}/{ev} orig pdms {mm.pdms_orig_T.mean():.4f} ncttc {100*mm.fail_ncttc_orig_T.mean():.2f}")
            for a in arm: print(f"   {a:5s}", arm[a])
            print("   ", fmt(e))
            # per-family P2 per run (to see which families flipped)
            fam = {FAM[c]: round(100 * float((g.fail_ncttc_none - g.fail_ncttc_T).mean()), 3) for c, g in mm.groupby("family")}
            res["runs"][f"{tag}/{ev}"]["fam_P2"] = fam; print("    fam P2:", fam)
    (OUT / "robust.json").write_text(json.dumps(res, indent=1, default=float))
