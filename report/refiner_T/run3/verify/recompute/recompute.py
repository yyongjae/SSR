"""Independent recomputation of stage-T run 3 dev endpoints (does NOT import stageT_decision.py). Read-only on runs/."""
import json, sys
import numpy as np, pandas as pd
R = "/home/external-user/ssd/yongjae_refiner/runs/stageT3_{}_fold0_seed0/eval_dev/"
OUT = "/home/external-user/yongjae/SSR/report/refiner_T/run3/verify/recompute/"
M = ["nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms"]
res = {}
d = {}
for a in ("T", "none"):
    x = pd.read_parquet(R.format(a) + "report_rows.parquet")
    tk = pd.read_parquet(R.format(a) + "tokens.parquet")
    assert tk.token.is_unique
    x = x.merge(tk, on="token", how="left")
    assert x.log.notna().all()
    d[a] = x
T, N = d["T"], d["none"]
chk = {}
chk["rows"] = [len(T), len(N)]
chk["same_order_token_k_family_valid"] = bool((T[["token", "k", "family", "valid", "log"]].values == N[["token", "k", "family", "valid", "log"]].values).all())
chk["tokens_T_eq_none"] = bool(pd.read_parquet(R.format("T") + "tokens.parquet").equals(pd.read_parquet(R.format("none") + "tokens.parquet")))
for m in M:
    o1, o2 = T[f"{m}_orig"].to_numpy(), N[f"{m}_orig"].to_numpy()
    chk[f"orig_{m}_identical"] = bool(np.array_equal(o1, o2, equal_nan=True))
chk["dup_token_k"] = int(T.duplicated(["token", "k"]).sum())
chk["k_per_token"] = T.groupby("token").k.nunique().value_counts().to_dict()
for a, x in d.items():
    v = x.valid.to_numpy()
    chk[f"{a}_valid"] = int(v.sum()); chk[f"{a}_invalid"] = int((~v).sum())
    chk[f"{a}_pg_nan"] = int(x.p_g.isna().sum()); chk[f"{a}_pg_min_valid"] = float(x.p_g[v].min())
    chk[f"{a}_pg_ge0_all_valid"] = bool((x.p_g[v] >= 0).all())
    fin_o = np.isfinite(x[[f"{m}_orig" for m in M]].to_numpy()).all(1)
    fin_t = np.isfinite(x[[f"{m}_tau1" for m in M]].to_numpy()).all(1)
    chk[f"{a}_valid_nonfinite_orig"] = int((v & ~fin_o).sum()); chk[f"{a}_valid_nonfinite_tau1"] = int((v & ~fin_t).sum())
    chk[f"{a}_invalid_finite_orig"] = int((~v & fin_o).sum())
    chk[f"{a}_invalid_by_family"] = x[~v].family.value_counts().to_dict()
    chk[f"{a}_alpha_unique"] = np.unique(x.alpha).tolist()[:5]
    # PDMS recomputation (fork: prod(NC,DAC,DDC) * (5 EP + 5 TTC + 2 comfort)/12)
    for s in ("orig", "tau1"):
        rec = x[f"nc_{s}"] * x[f"dac_{s}"] * x[f"ddc_{s}"] * (5 * x[f"ep_{s}"] + 5 * x[f"ttc_{s}"] + 2 * x[f"comfort_{s}"]) / 12
        err = (rec - x[f"pdms_{s}"])[v].abs()
        chk[f"{a}_pdms_{s}_maxabs_err"] = float(err.max())
        chk[f"{a}_pdms_{s}_n_err_gt1e-9"] = int((err > 1e-9).sum())
    for s in ("orig", "tau1"):
        for m in ("nc", "dac", "ddc", "ttc", "comfort"):
            chk[f"{a}_{m}_{s}_values"] = sorted(np.unique(x[f"{m}_{s}"][v]).round(6).tolist())[:6]
common_valid = T.valid.to_numpy() & N.valid.to_numpy()
fin = np.ones(len(T), bool)
for x in (T, N):
    for s in ("orig", "tau1"):
        fin &= np.isfinite(x[[f"{m}_{s}" for m in M]].to_numpy()).all(1)
keep = common_valid & fin
chk["n_keep"] = int(keep.sum()); chk["n_logs_keep"] = int(T.log[keep].nunique()); chk["n_tokens_keep"] = int(T.token[keep].nunique())
chk["n_logs_all"] = int(T.log.nunique())
res["checks"] = chk

theta = 0.0
def final(x):
    mod = (x.p_g.to_numpy() >= theta)
    f = {m: np.where(mod, x[f"{m}_tau1"], x[f"{m}_orig"]).astype(np.float64) for m in M}
    o = {m: x[f"{m}_orig"].to_numpy(np.float64) for m in M}
    return o, f, mod
T, N = T[keep].reset_index(drop=True), N[keep].reset_index(drop=True)
logs = T.log.to_numpy()
oT, fT, modT = final(T); oN, fN, modN = final(N)
res["modified_frac"] = dict(T=float(modT.mean()), none=float(modN.mean()))
def failany(dct, ks): return np.any(np.stack([dct[k] < 1 for k in ks]), 0)
allk = ["nc", "dac", "ddc", "ttc", "comfort"]
per = {}
for a, o, f in (("T", oT, fT), ("none", oN, fN)):
    per[a] = dict(pdms=f["pdms"], d_pdms=f["pdms"] - o["pdms"], ncttc=failany(f, ["nc", "ttc"]).astype(float),
                  nc=(f["nc"] < 1).astype(float), ttc=(f["ttc"] < 1).astype(float),
                  dac=(f["dac"] < 1).astype(float), ddc=(f["ddc"] < 1).astype(float),
                  newf=(~failany(o, allk) & failany(f, allk)).astype(float),
                  passed_before=(~failany(o, allk)).astype(float))
res["n_passed_all_before"] = int(per["T"]["passed_before"].sum())
res["rates_pct"] = {a: {k: 100 * float(v.mean()) for k, v in p.items() if k not in ("pdms", "d_pdms")} for a, p in per.items()}
res["rates_pct"]["orig"] = {"ncttc": 100 * float(failany(oT, ["nc", "ttc"]).mean()), "nc": 100 * float((oT["nc"] < 1).mean()),
                            "ttc": 100 * float((oT["ttc"] < 1).mean()), "dac": 100 * float((oT["dac"] < 1).mean()), "ddc": 100 * float((oT["ddc"] < 1).mean())}
res["pdms_mean"] = dict(T=float(fT["pdms"].mean()), none=float(fN["pdms"].mean()), orig=float(oT["pdms"].mean()))

# independent bootstrap: explicit resampling of log labels, 10,000, different RNG/seed, draft-weighted mean
ul, inv = np.unique(logs, return_inverse=True)
L = len(ul); cnt = np.bincount(inv, minlength=L).astype(float)
rng = np.random.Generator(np.random.PCG64(20260929))
idx = rng.integers(0, L, size=(10000, L))
W = np.zeros((10000, L))
for b in range(10000):
    W[b] = np.bincount(idx[b], minlength=L)
def boot(diff, scale=100):
    s = np.bincount(inv, weights=diff, minlength=L)
    st = (W @ s) / (W @ cnt)
    lo, hi = np.percentile(st, [2.5, 97.5])
    # log-equal-weight variant
    lm = s / cnt
    st2 = (W @ lm) / L
    lo2, hi2 = np.percentile(st2, [2.5, 97.5])
    return dict(mean=scale * float(diff.mean()), lo=scale * float(lo), hi=scale * float(hi), se=scale * float(st.std()),
                p_le0=float((st <= 0).mean()), p_ge0=float((st >= 0).mean()),
                logeq_mean=scale * float(lm.mean()), logeq_lo=scale * float(lo2), logeq_hi=scale * float(hi2))
pT, pN = per["T"], per["none"]
E = {}
E["P1_d_pdms_points"] = boot(pT["pdms"] - pN["pdms"])
E["P2_ncttc_reduction_pp"] = boot(pN["ncttc"] - pT["ncttc"])
E["P2_nc_only_reduction_pp"] = boot(pN["nc"] - pT["nc"])
E["P2_ttc_only_reduction_pp"] = boot(pN["ttc"] - pT["ttc"])
E["P2ni_dac_excess_pp"] = boot(pT["dac"] - pN["dac"])
E["P2ni_ddc_excess_pp"] = boot(pT["ddc"] - pN["ddc"])
E["P3_new_fail_excess_pp"] = boot(pT["newf"] - pN["newf"])
pb = pT["passed_before"] > 0
# P3 conditional on passing-before drafts (rate among them), bootstrap over logs with ratio
s_num = np.bincount(inv, weights=(pT["newf"] - pN["newf"]), minlength=L); s_den = np.bincount(inv, weights=pb.astype(float), minlength=L)
st = (W @ s_num) / (W @ s_den)
E["P3_conditional_on_passing_pp"] = dict(mean=100 * float((pT["newf"] - pN["newf"])[pb].mean()), lo=100 * float(np.percentile(st, 2.5)), hi=100 * float(np.percentile(st, 97.5)))
E["sanity_T_vs_orig_points"] = boot(pT["d_pdms"]); E["sanity_none_vs_orig_points"] = boot(pN["d_pdms"])
res["endpoints"] = E
# EP loss (budget def 'passing': NC, DAC, DDC >= 1 before and after)
for a, o, f in (("T", oT, fT), ("none", oN, fN)):
    core = lambda z: (z["nc"] >= 1) & (z["dac"] >= 1) & (z["ddc"] >= 1)
    pbo = core(o) & core(f)
    res[f"ep_loss_passing_points_{a}"] = 100 * float((o["ep"] - f["ep"])[pbo].mean()); res[f"n_passing_{a}"] = int(pbo.sum())
    res[f"ep_loss_all_points_{a}"] = 100 * float((o["ep"] - f["ep"]).mean())
p1, p2, p3 = E["P1_d_pdms_points"], E["P2_ncttc_reduction_pp"], E["P3_new_fail_excess_pp"]
san = E["sanity_T_vs_orig_points"]["lo"] > 0 or E["sanity_none_vs_orig_points"]["lo"] > 0
res["outcome_recomputed"] = "PASS" if (san and p1["lo"] > 0 and p2["lo"] > 0 and p3["hi"] <= 0.5) else "NOT PASS"
# per-log contribution: how concentrated is P1 / P2?
s1 = np.bincount(inv, weights=pT["pdms"] - pN["pdms"], minlength=L)
order = np.argsort(-s1)
res["P1_top5_logs_share_of_sum"] = float(s1[order[:5]].sum() / s1.sum())
res["P1_frac_logs_positive"] = float((s1 > 0).mean())
# leave-top-k-logs-out for P1 and P2
s2 = np.bincount(inv, weights=pN["ncttc"] - pT["ncttc"], minlength=L)
for kk in (1, 5, 10):
    m1 = np.ones(L, bool); m1[order[:kk]] = False
    o2 = np.argsort(-s2); m2 = np.ones(L, bool); m2[o2[:kk]] = False
    res[f"P1_drop_top{kk}_logs"] = 100 * float(s1[m1].sum() / cnt[m1].sum())
    res[f"P2_drop_top{kk}_logs"] = 100 * float(s2[m2].sum() / cnt[m2].sum())
dj = json.load(open("/home/external-user/yongjae/SSR/report/refiner_T/run3/decision_dev.json"))
cmp = {}
for k in ("P1_d_pdms_points", "P2_ncttc_reduction_pp", "P2ni_dac_excess_pp", "P2ni_ddc_excess_pp", "P3_new_fail_excess_pp"):
    cmp[k] = dict(stored=[dj["endpoints"][k][z] for z in ("mean", "lo", "hi")], mine=[E[k][z] for z in ("mean", "lo", "hi")])
cmp["sanity_T"] = dict(stored=[dj["sanity"]["d_pdms_vs_orig_points_T"][z] for z in ("mean", "lo", "hi")], mine=[E["sanity_T_vs_orig_points"][z] for z in ("mean", "lo", "hi")])
cmp["sanity_none"] = dict(stored=[dj["sanity"]["d_pdms_vs_orig_points_none"][z] for z in ("mean", "lo", "hi")], mine=[E["sanity_none_vs_orig_points"][z] for z in ("mean", "lo", "hi")])
cmp["ep_loss_T"] = [dj["arms"]["T"]["ep_loss_points"], res["ep_loss_passing_points_T"]]
cmp["ep_loss_none"] = [dj["arms"]["none"]["ep_loss_points"], res["ep_loss_passing_points_none"]]
cmp["n_passing"] = [dj["arms"]["T"]["n_passing"], res["n_passing_T"], dj["arms"]["none"]["n_passing"], res["n_passing_none"]]
cmp["n"] = [dj["n_paired_drafts"], chk["n_keep"], dj["n_logs"], chk["n_logs_keep"]]
res["compare_with_decision_dev"] = cmp
json.dump(res, open(OUT + "recompute.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
