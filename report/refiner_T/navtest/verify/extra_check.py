"""Extra: DAC fixed/new counts, per-family P1/P2 means, and P1/P2 with the 8,796 invalid drafts included (robustness)."""
import json, numpy as np, pandas as pd
R = '/home/external-user/ssd/yongjae_refiner/runs'
T = pd.read_parquet(f'{R}/stageT3_T_fold0_seed0/eval_navtest/report_rows.parquet'); N = pd.read_parquet(f'{R}/stageT3_none_fold0_seed0/eval_navtest/report_rows.parquet')
v = T.valid.to_numpy(bool); out = {}
d0 = T.dac_orig.to_numpy() < 1; dT = T.dac_tau1.to_numpy() < 1; dN = N.dac_tau1.to_numpy() < 1
out['dac'] = dict(orig_fail=int((d0 & v).sum()), fixed_T=int((d0 & ~dT & v).sum()), fixed_none=int((d0 & ~dN & v).sum()), new_T=int((~d0 & dT & v).sum()), new_none=int((~d0 & dN & v).sum()))
F = {0: 'identity', 1: 'small', 2: 'lconst', 3: 'ignore_brake', 4: 'creep', 5: 'lat', 6: 'combined', 7: 'cv', 8: 'hdrift'}
nt = lambda df: ((df.nc_tau1 < 1) | (df.ttc_tau1 < 1)).to_numpy().astype(float)
p1 = 100 * (T.pdms_tau1 - N.pdms_tau1).to_numpy(); p2 = 100 * (nt(N) - nt(T))
out['family'] = {F[f]: dict(n=int(((T.family == f) & v).sum()), P1=float(p1[(T.family == f).to_numpy() & v].mean()), P2=float(p2[(T.family == f).to_numpy() & v].mean())) for f in sorted(T.family.unique())}
fin = np.isfinite(T.pdms_tau1.to_numpy()) & np.isfinite(N.pdms_tau1.to_numpy()) & np.isfinite(T.pdms_orig.to_numpy())
out['include_invalid'] = dict(n=int(fin.sum()), P1=float(p1[fin].mean()), P2=float(p2[fin].mean()),
                              invalid_family_counts={F[int(k)]: int(c) for k, c in T[~v].family.value_counts().items()})
print(json.dumps(out, indent=1))
