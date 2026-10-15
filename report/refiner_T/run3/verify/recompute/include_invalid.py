"""Sensitivity: endpoints if the 5,704 invalid drafts (finite scores, same in both arms) were included."""
import numpy as np, pandas as pd, json
R = "/home/external-user/ssd/yongjae_refiner/runs/stageT3_{}_fold0_seed0/eval_dev/"
d = {a: pd.read_parquet(R.format(a) + "report_rows.parquet").merge(pd.read_parquet(R.format(a) + "tokens.parquet"), on="token") for a in ("T", "none")}
T, N = d["T"], d["none"]
ul, inv = np.unique(T.log, return_inverse=True); L = len(ul); cnt = np.bincount(inv, minlength=L).astype(float)
rng = np.random.Generator(np.random.PCG64(7)); W = np.stack([np.bincount(rng.integers(0, L, L), minlength=L) for _ in range(10000)]).astype(float)
def b(x, mask):
    s = np.bincount(inv[mask], weights=x[mask], minlength=L); c = np.bincount(inv[mask], minlength=L).astype(float)
    st = (W @ s) / (W @ c); return [round(100 * float(x[mask].mean()), 4), round(100 * np.percentile(st, 2.5), 4), round(100 * np.percentile(st, 97.5), 4)]
f = lambda X, ks: np.any(np.stack([X[f"{k}_tau1"] < 1 for k in ks]), 0)
fo = lambda X, ks: np.any(np.stack([X[f"{k}_orig"] < 1 for k in ks]), 0)
allk = ["nc", "dac", "ddc", "ttc", "comfort"]
out = {}
for name, mask in (("valid_only", T.valid.values), ("all_rows", np.ones(len(T), bool)), ("invalid_only", ~T.valid.values)):
    out[name] = dict(n=int(mask.sum()),
        P1=b((T.pdms_tau1 - N.pdms_tau1).values, mask),
        P2=b((f(N, ["nc", "ttc"]).astype(float) - f(T, ["nc", "ttc"])).astype(float), mask),
        DAC=b((f(T, ["dac"]).astype(float) - f(N, ["dac"])).astype(float), mask),
        P3=b(((~fo(T, allk) & f(T, allk)).astype(float) - (~fo(N, allk) & f(N, allk))).astype(float), mask))
fam = T.family.values
out["by_family_P1_P2"] = {int(k): dict(n=int((fam == k).sum()), P1=b((T.pdms_tau1 - N.pdms_tau1).values, T.valid.values & (fam == k)),
                                       P2=b((f(N, ["nc", "ttc"]).astype(float) - f(T, ["nc", "ttc"])).astype(float), T.valid.values & (fam == k)))
                          for k in np.unique(fam) if (T.valid.values & (fam == k)).sum() > 50}
print(json.dumps(out, indent=1)); json.dump(out, open("/home/external-user/yongjae/SSR/report/refiner_T/run3/verify/recompute/include_invalid.json", "w"), indent=1)
