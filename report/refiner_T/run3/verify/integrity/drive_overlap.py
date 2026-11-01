"""Dev segments whose parent drive (date_vehicle) also has train segments: gap to nearest train segment, city, and
R_T - R_none by stratum (+ bootstrap of the stratum difference, logs resampled within strata), per city."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/home/external-user/yongjae/SSR/tools/refiner'); sys.path.insert(0,'/home/external-user/yongjae/SSR')
from stageT_decision import load_rows, final_outcomes, cluster_boot
R='/home/external-user/ssd/yongjae_refiner/runs'; S='/home/external-user/ssd/yongjae_refiner/splits'
tr=pd.read_parquet(f'{S}/train.parquet'); dv=pd.read_parquet(f'{S}/dev.parquet')
la=pd.read_parquet(f'{S}/log_assignment.parquet')
def parse(l):
    p=l.rsplit('_',2); return p[0],int(p[1]),int(p[2])
la[['drive','s','e']]=la.log.apply(lambda l: pd.Series(parse(l)))
trl=la[la.side=='train']; 
gaps={}
for _,r in la[la.side=='dev'].iterrows():
    t=trl[trl.drive==r.drive]
    if len(t)==0: gaps[r.log]=np.nan; continue
    g=np.minimum(np.abs(t.s.values-r.e),np.abs(r.s-t.e.values)); gaps[r.log]=float(g.min())
gp=pd.Series(gaps)
out=dict(side_counts=la.side.value_counts().to_dict(),
 dev_logs_with_train_sibling=int(gp.notna().sum()),dev_logs=int(len(gp)),
 gap_frames_quantiles=gp.dropna().quantile([0,.25,.5,.75,1]).to_dict(),
 frac_dev_logs_gap_le_20frames=float((gp<=20).mean()))
fT=final_outcomes(load_rows(f'{R}/stageT3_T_fold0_seed0/eval_dev'),0.0)
fN=final_outcomes(load_rows(f'{R}/stageT3_none_fold0_seed0/eval_dev'),0.0)
m=fT.merge(fN,on=['token','k','log','family'],suffixes=('_T','_N'),validate='one_to_one')
m['dp']=100*(m.pdms_final_T-m.pdms_final_N); m['dn']=100*(m.fail_ncttc_N-m.fail_ncttc_T); m['dd']=100*(m.fail_dac_T-m.fail_dac_N)
m['sib']=m.log.map(gp).notna()
m=m.merge(dv[['token','map_location']].drop_duplicates(),on='token')
out['heldout_drive_logs_city']=m[~m.sib].groupby('map_location').log.nunique().to_dict()
out['heldout_drive_logs_gap']=None
rng=np.random.default_rng(1)
def strat_boot(a,b,col,n=4000):
    def agg(g):
        s=g.groupby('log')[col].agg(['sum','count']); return s['sum'].to_numpy(),s['count'].to_numpy()
    sa,na=agg(a); sb,nb=agg(b); r=[]
    for _ in range(n):
        i=rng.integers(0,len(sa),len(sa)); j=rng.integers(0,len(sb),len(sb))
        r.append(sa[i].sum()/na[i].sum()-sb[j].sum()/nb[j].sum())
    return dict(diff=float(a[col].mean()-b[col].mean()),lo=float(np.percentile(r,2.5)),hi=float(np.percentile(r,97.5)))
out['sibling_minus_heldout']={c:strat_boot(m[m.sib],m[~m.sib],c) for c in ('dp','dn','dd')}
out['dac_excess_by_sibling']={str(k):cluster_boot(g.dd.to_numpy(),g.log.to_numpy(),2000) for k,g in m.groupby('sib')}
out['by_city']={c:dict(n_logs=g.log.nunique(),P1=cluster_boot(g.dp.to_numpy(),g.log.to_numpy(),2000),P2=cluster_boot(g.dn.to_numpy(),g.log.to_numpy(),2000)) for c,g in m.groupby('map_location')}
# sibling-only within city composition check: gap-stratified among sibling logs
m['gap']=m.log.map(gp)
q=m[m.sib].drop_duplicates('log').gap.median()
out['by_gap_among_sibling']={f'gap<={q}':cluster_boot(m[m.sib&(m.gap<=q)].dp.to_numpy(),m[m.sib&(m.gap<=q)].log.to_numpy(),2000),
                              f'gap>{q}':cluster_boot(m[m.sib&(m.gap>q)].dp.to_numpy(),m[m.sib&(m.gap>q)].log.to_numpy(),2000)}
print(json.dumps(out,indent=1,default=float)); json.dump(out,open('drive_overlap.json','w'),indent=1,default=float)
