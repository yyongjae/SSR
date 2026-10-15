import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/home/external-user/yongjae/SSR/tools/refiner'); sys.path.insert(0,'/home/external-user/yongjae/SSR')
from stageT_decision import load_rows, final_outcomes
R='/home/external-user/ssd/yongjae_refiner/runs'; S='/home/external-user/ssd/yongjae_refiner/splits'
dv=pd.read_parquet(f'{S}/dev.parquet'); tr=pd.read_parquet(f'{S}/train.parquet')
fT=final_outcomes(load_rows(f'{R}/stageT3_T_fold0_seed0/eval_dev'),0.0); fN=final_outcomes(load_rows(f'{R}/stageT3_none_fold0_seed0/eval_dev'),0.0)
m=fT.merge(fN,on=['token','k','log','family'],suffixes=('_T','_N')).merge(dv[['token','map_location']],on='token')
m['dp']=100*(m.pdms_final_T-m.pdms_final_N); m['dn']=100*(m.fail_ncttc_N-m.fail_ncttc_T)
td=set(tr.log.str.rsplit('_',n=2).str[0]); m['sib']=m.log.str.rsplit('_',n=2).str[0].isin(td)
out={}
out['city_x_sibling']=m.groupby(['map_location','sib']).agg(n_logs=('log','nunique'),P1=('dp','mean'),P2=('dn','mean')).reset_index().to_dict(orient='records')
nt=pd.read_parquet('/home/external-user/ssd/yongjae_refiner/objects/_validation/navtest_all_token_log.parquet')
out['navtest_cols']=list(nt.columns)
if 'map_location' in nt.columns:
    w=nt.map_location.value_counts(normalize=True)
else:
    w=None
out['navtest_city_share']=None if w is None else w.to_dict()
out['dev_city_share']=m.drop_duplicates('token').map_location.value_counts(normalize=True).to_dict()
# city-stratified log bootstrap of reweighted P1/P2
if w is not None:
    rng=np.random.default_rng(0); per={}
    for c,g in m.groupby('map_location'):
        s=g.groupby('log').agg(sp=('dp','sum'),sn=('dn','sum'),n=('dp','size')); per[c]=s
    bs=[]
    for _ in range(4000):
        p1=p2=0
        for c,s in per.items():
            i=rng.integers(0,len(s),len(s)); ss=s.iloc[i]
            p1+=w.get(c,0)*ss.sp.sum()/ss.n.sum(); p2+=w.get(c,0)*ss.sn.sum()/ss.n.sum()
        bs.append((p1,p2))
    bs=np.array(bs)
    pt1=sum(w.get(c,0)*s.sp.sum()/s.n.sum() for c,s in per.items()); pt2=sum(w.get(c,0)*s.sn.sum()/s.n.sum() for c,s in per.items())
    out['navtest_city_reweighted']=dict(P1=[pt1,*np.percentile(bs[:,0],[2.5,97.5])],P2=[pt2,*np.percentile(bs[:,1],[2.5,97.5])])
print(json.dumps(out,indent=1,default=float)); json.dump(out,open('city_adjust.json','w'),indent=1,default=float)
