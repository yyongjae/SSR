"""Independent recomputation of P1/P2/P3 from report_rows (no stageT_decision import), bootstrap seed 7."""
import json, numpy as np, pandas as pd
R='/home/external-user/ssd/yongjae_refiner/runs'
def rows(a):
    d=pd.read_parquet(f'{R}/stageT3_{a}_fold0_seed0/eval_dev/report_rows.parquet')
    tk=pd.read_parquet(f'{R}/stageT3_{a}_fold0_seed0/eval_dev/tokens.parquet')[['token','log']].drop_duplicates('token')
    return d.merge(tk,on='token')
T,N=rows('T'),rows('none')
M=['nc','dac','ddc','ep','ttc','comfort','pdms']
out={'rows':[len(T),len(N)]}
out['orig_identical']=bool(all(np.allclose(T.sort_values(['token','k'])[f'{m}_orig'].values,N.sort_values(['token','k'])[f'{m}_orig'].values,equal_nan=True) for m in M))
out['valid_identical']=bool((T.sort_values(['token','k']).valid.values==N.sort_values(['token','k']).valid.values).all())
out['p_g_min']=[float(T.p_g.min()),float(N.p_g.min())]
m=T.merge(N,on=['token','k','log'],suffixes=('_T','_N'))
ok=m.valid_T.astype(bool)&m.valid_N.astype(bool)
for s in ('T','N'):
    for x in M: ok&=np.isfinite(m[f'{x}_orig_{s}'])&np.isfinite(m[f'{x}_tau1_{s}'])
m=m[ok]; out['n_paired']=int(len(m)); out['invalid_rows_by_family']=None
fail=lambda s: ((m[f'nc_tau1_{s}']<1)|(m[f'ttc_tau1_{s}']<1)).astype(float)
d=dict(P1=100*(m.pdms_tau1_T-m.pdms_tau1_N),P2=100*(fail('N')-fail('T')),
       DAC=100*((m.dac_tau1_T<1).astype(float)-(m.dac_tau1_N<1).astype(float)))
logs=m.log.values; u,inv=np.unique(logs,return_inverse=True); rng=np.random.default_rng(7)
for k,v in d.items():
    v=v.to_numpy(); s=np.bincount(inv,v); n=np.bincount(inv).astype(float)
    c=rng.multinomial(len(u),np.full(len(u),1/len(u)),size=10000)
    b=(c@s)/(c@n); out[k]=[float(v.mean()),*np.percentile(b,[2.5,97.5]).tolist()]
# token-cluster and half-split robustness
print(json.dumps(out,indent=1)); json.dump(out,open('recompute.json','w'),indent=1)
