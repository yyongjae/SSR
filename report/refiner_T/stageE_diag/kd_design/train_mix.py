#!/usr/bin/env python
"""H6: training draft mix of E2 from stageE_steps.jsonl (rank-0 micro-batches, batch 4).  Per draft type estimates
from (a) pure micro-batches (frac_draft_tau0 == 1 -> all sg(tau0) [incl. invalid-perturbation fallbacks], == 0 -> all
perturbed GT human) and (b) OLS metric ~ a + b * frac_draft_tau0 over all micro-batches (tau0 = a + b, human = a).
Per-sample rates of 'surrogate active' from the pure batches: p = 1 - (1 - P(batch any > 0))^(1/n_ok)."""
import json, sys
import numpy as np, pandas as pd
RUN = "/home/external-user/yongjae/SSR/work_dirs/stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0/stageE_steps.jsonl"
OUT = sys.argv[1]
df = pd.read_json(RUN, lines=True)
M = ["kd/teacher_live_0", "kd/teacher_live_1", "ref/live", "kd/l1_lon_0", "kd/l1_lat_0", "kd/l1_lon_1", "kd/l1_lat_1",
     "kd/l1_0", "kd/l1_1", "ref/t_col", "ref/t_dac", "ref/t_ttc", "ref/L_sur", "ref/P1_minus_P0", "ref/short_m",
     "ref/lat_m", "ref/dist_final_tau0"]
rng = np.random.default_rng(0)

def boot_mean(x, B=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return [np.nan] * 3
    # block bootstrap over contiguous chunks of 50 micro-batches (serial correlation of the training state)
    nb = max(1, len(x) // 50); ch = np.array_split(x, nb); mu = np.array([c.mean() for c in ch]); w = np.array([len(c) for c in ch])
    bs = [np.average(mu[i], weights=w[i]) for i in (rng.integers(0, nb, nb) for _ in range(B))]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

res = {"n_microbatches": int(len(df)), "note": __doc__}
res["mix_by_phase"] = {}
for name, sel in {"ep0-4 (human only)": df.epoch < 5, "ep5-29": df.epoch >= 5, "ep25-29": df.epoch >= 25}.items():
    d = df[sel]
    res["mix_by_phase"][name] = {"n": int(len(d)), "frac_draft_human": float(d["ref/frac_draft_human"].mean()),
                                 "frac_draft_tau0": float(d["ref/frac_draft_tau0"].mean()),
                                 "perturb_invalid_rate": float(d["ref/n_perturb_invalid"].sum() / max(1, d["ref/n_perturbed"].sum() + d["ref/n_perturb_invalid"].sum())),
                                 "kd_w_ema": float(d["kd/w_ema"].mean()), "vshare_kd": float(d["ref/vshare_kd"].mean()),
                                 "vshare_sur": float(d["ref/vshare_sur"].mean())}
res["by_type"] = {}
for name, sel in {"ep5-29": df.epoch >= 5, "ep25-29": df.epoch >= 25}.items():
    d = df[sel]
    pt, ph = d[d["ref/frac_draft_tau0"] == 1.0], d[d["ref/frac_draft_tau0"] == 0.0]
    o = {"n_pure_tau0": int(len(pt)), "n_pure_human": int(len(ph)), "metrics": {}}
    X = np.c_[np.ones(len(d)), d["ref/frac_draft_tau0"].values]
    for m in M:
        y = d[m].values.astype(float); ok = np.isfinite(y)
        beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        o["metrics"][m] = {"pure_tau0": boot_mean(pt[m]), "pure_human": boot_mean(ph[m]),
                           "ols_tau0": float(beta.sum()), "ols_human": float(beta[0])}
    for m in ("ref/t_col", "ref/t_dac", "ref/t_ttc"):
        for lab, dd in (("tau0", pt), ("human", ph)):
            pb = float((dd[m] > 0).mean()); n = float(dd["ref/n_gt_ok"].mean())
            o["metrics"][m][f"per_sample_active_{lab}"] = 1 - (1 - pb) ** (1 / n) if pb < 1 else 1.0
    res["by_type"][name] = o
json.dump(res, open(OUT, "w"), indent=1)
for name, o in res["by_type"].items():
    print(name, o["n_pure_tau0"], o["n_pure_human"])
    for m, v in o["metrics"].items():
        print(f"  {m:22s} tau0 {v['pure_tau0'][0]:.4f} [{v['pure_tau0'][1]:.4f},{v['pure_tau0'][2]:.4f}] ols {v['ols_tau0']:.4f} | human {v['pure_human'][0]:.4f} [{v['pure_human'][1]:.4f},{v['pure_human'][2]:.4f}] ols {v['ols_human']:.4f}",
              {k: round(x, 4) for k, x in v.items() if k.startswith("per_sample")})
print(json.dumps(res["mix_by_phase"], indent=1))
