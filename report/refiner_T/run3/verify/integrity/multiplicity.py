import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/home/external-user/yongjae/SSR/tools/refiner'); sys.path.insert(0,'/home/external-user/yongjae/SSR')
from stageT_decision import load_rows, final_outcomes
R='/home/external-user/ssd/yongjae_refiner/runs'; S='/home/external-user/ssd/yongjae_refiner/splits'
dv=pd.read_parquet(f'{S}/dev.parquet'); tr=pd.read_parquet(f'{S}/train.parquet')
fT=final_outcomes(load_rows(f'{R}/stageT3_T_fold0_seed0/eval_dev'),0.0); fN=final_outcomes(load_rows(f'{R}/stageT3_none_fold0_seed0/eval_dev'),0.0)
m=fT.merge(fN,on=['token','k','log','family'],suffixes=('_T','_N')).merge(dv[['token','map_location']],on='token')
m['P1']=100*(m.pdms_final_T-m.pdms_final_N); m['P2']=100*(m.fail_ncttc_N-m.fail_ncttc_T)
def boot(g,col,n=10000,seed=3):
    s=g.groupby('log')[col].agg(['sum','count']); rng=np.random.default_rng(seed)
    c=rng.multinomial(len(s),np.full(len(s),1/len(s)),size=n); return (c@s['sum'].values)/(c@s['count'].values)
out={}
for col in ('P1','P2'):
    b=boot(m,col); out[col]=dict(mean=float(m[col].mean()),ci95=np.percentile(b,[2.5,97.5]).tolist(),
        ci_bonf3=np.percentile(b,[100*0.05/6,100-100*0.05/6]).tolist(),ci_bonf6=np.percentile(b,[100*0.05/12,100-100*0.05/12]).tolist(),
        frac_boot_le0=float((b<=0).mean()))
td=set(tr.log.str.rsplit('_',n=2).str[0]); m['sib']=m.log.str.rsplit('_',n=2).str[0].isin(td)
for nm,g in (('LV_heldout_drive',m[(~m.sib)&(m.map_location=='us-nv-las-vegas-strip')]),('LV_sibling',m[m.sib&(m.map_location=='us-nv-las-vegas-strip')]),('heldout_drive_all',m[~m.sib])):
    out[nm]={col:dict(mean=float(g[col].mean()),ci95=np.percentile(boot(g,col,4000),[2.5,97.5]).tolist(),n_logs=int(g.log.nunique())) for col in ('P1','P2')}
print(json.dumps(out,indent=1)); json.dump(out,open('multiplicity.json','w'),indent=1)
