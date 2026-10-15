"""Extra navtest tables for SUMMARY.md (read-only over existing score files).
Arms are batched-scorer parquet files; bootstrap = paired log-cluster, 10,000 draws, seed 0."""
import json, numpy as np, pandas as pd
R='/home/external-user/ssd/yongjae_refiner'
D=R+'/stageE_diag'
def ld(p,k=None):
    s=pd.read_parquet(p)
    if k is not None: s=s[s.k==k]
    return s.set_index('token')
arms={
 'E0':ld(R+'/e0_teacher_refine/E0/scores.parquet'),
 'E0+R_T4':ld(R+'/e0_teacher_refine/R_T4/scores.parquet'),
 'E0+R_M4':ld(R+'/e0_teacher_refine/R_M4/scores.parquet'),
 'E2_tau0':ld(D+'/scores.parquet',0),
 'E2':ld(D+'/scores.parquet',1),
 'E2t0+R_T4':ld(D+'/same_draft/R_T4/scores.parquet'),
 'E2t0+R_M4':ld(D+'/same_draft/R_M4/scores.parquet'),
 'E2t0+R_none4':ld(D+'/same_draft/R_none4/scores.parquet'),
}
cons=pd.read_parquet(D+'/kd_design/consensus/scores.parquet')
sw=pd.read_parquet(D+'/teacher_on_e2/scores.parquet')
cm=json.load(open(D+'/kd_design/consensus/consensus_meta.json'))['variants']
smeta=json.load(open(D+'/teacher_on_e2/swaps_meta.json'))['arms']
arms['E2t0+MID']=cons[cons.k==cm.index('MID')].set_index('token')
arms['E2 student x2']=sw[sw.k==smeta.index('Sx2')].set_index('token')
tok=arms['E0'].index
for a in arms: arms[a]=arms[a].loc[tok]
assert np.allclose(arms['E2 student x2'].pdms.mean()*100,87.00,atol=0.02), arms['E2 student x2'].pdms.mean()
assert abs(arms['E2t0+MID'].pdms.mean()*100-87.09)<0.02
logs=arms['E0'].log.values
city_map=pd.read_csv('/home/external-user/yongjae/SSR/report/refiner_T/stageE_diag/summary/city.csv').set_index('token').city if False else None
M=['pdms','nc','dac','ttc','ep']
ul,li=np.unique(logs,return_inverse=True)
rng=np.random.default_rng(0)
W=rng.multinomial(len(ul),np.ones(len(ul))/len(ul),size=10000).astype(float)
def boot(x,mask=None):
    x=np.asarray(x,float); m=np.ones_like(x,bool) if mask is None else mask
    s=np.bincount(li[m],x[m],len(ul)); n=np.bincount(li[m],None,len(ul)).astype(float)
    b=(W@s)/np.maximum(W@n,1)
    return [round(100*x[m].mean(),3),round(100*np.percentile(b,2.5),3),round(100*np.percentile(b,97.5),3)]
out={'means':{a:{m:round(100*arms[a][m].mean(),2) for m in M} for a in arms}}
fail=lambda a:(arms[a][['nc','dac','ddc','ttc']].min(axis=1)<1).values
out['fail_counts']={a:{'nc0':int((arms[a].nc==0).sum()),'nc_half':int((arms[a].nc==0.5).sum()),'dac0':int((arms[a].dac==0).sum()),'ttc0':int((arms[a].ttc==0).sum()),'pdms0':int((arms[a].pdms==0).sum()),'fail_any':int(fail(a).sum())} for a in arms}
out['nc_obj_type_E2_fail']=arms['E2'][arms['E2'].nc<1].nc_obj_type.value_counts().to_dict()
out['nc_obj_type_E2t0RT4_fail']=arms['E2t0+R_T4'][arms['E2t0+R_T4'].nc<1].nc_obj_type.value_counts().to_dict()
def d(a,b,mask=None): return {m:boot(arms[a][m].values-arms[b][m].values,mask) for m in M}
out['decomp']={'gap E0+R_T4 - E2':d('E0+R_T4','E2'),'draft E0+R_T4 - E2t0+R_T4':d('E0+R_T4','E2t0+R_T4'),'refiner E2t0+R_T4 - E2':d('E2t0+R_T4','E2'),
 'gap E0+R_M4 - E2':d('E0+R_M4','E2'),'draft E0+R_M4 - E2t0+R_M4':d('E0+R_M4','E2t0+R_M4'),'refiner E2t0+R_M4 - E2':d('E2t0+R_M4','E2'),
 'E2 - E0':d('E2','E0'),'E2_tau0 - E0':d('E2_tau0','E0'),'E2 - E2_tau0':d('E2','E2_tau0'),'E2t0+R_T4 - E2_tau0':d('E2t0+R_T4','E2_tau0'),
 'E2t0+MID - E2':d('E2t0+MID','E2'),'E2x2 - E2':d('E2 student x2','E2'),'E2x2 - E0+R_T4':d('E2 student x2','E0+R_T4')}
city=np.array([l.split('_')[0] for l in logs])
# city from log name is not the city; use nuplan log -> city via E0 teacher json per_city token lists unavailable; use metric-cache-free mapping below
out_city=None
# fixed/broken A vs B
def fb(a,b):
    fa,fbb=fail(a),fail(b); r={'fixed':int((fbb&~fa).sum()),'broken':int((~fbb&fa).sum())}
    for m in ['nc','dac','ttc']:
        r[m+'_fixed']=int(((arms[b][m]<1)&(arms[a][m]==1)).sum()); r[m+'_broken']=int(((arms[b][m]==1)&(arms[a][m]<1)).sum())
    return r
out['fixed_broken']={f'{a} vs {b}':fb(a,b) for a,b in [('E2','E2_tau0'),('E2t0+R_T4','E2_tau0'),('E2t0+R_M4','E2_tau0'),('E2_tau0','E0'),('E2','E0'),('E2','E0+R_T4'),('E2','E2t0+R_T4'),('E2','E2t0+R_M4')]}
# gap split by E2_tau0 fail/pass
f0=fail('E2_tau0')
out['refiner_part_by_tau0_status']={'E2t0 fails (n=%d)'%f0.sum():boot(arms['E2t0+R_T4'].pdms.values-arms['E2'].pdms.values,f0),
 'E2t0 passes (n=%d)'%(~f0).sum():boot(arms['E2t0+R_T4'].pdms.values-arms['E2'].pdms.values,~f0)}
# contributions in points of the whole mean
out['refiner_part_contrib_pts']={'fail':round(100*(arms['E2t0+R_T4'].pdms.values-arms['E2'].pdms.values)[f0].sum()/len(tok),3),
 'pass':round(100*(arms['E2t0+R_T4'].pdms.values-arms['E2'].pdms.values)[~f0].sum()/len(tok),3)}
# EP on passing tokens
p=~fail('E2')&~fail('E2t0+R_T4')
out['ep_both_pass']={'n':int(p.sum()),'E2t0+R_T4 - E2 ep':boot(arms['E2t0+R_T4'].ep.values-arms['E2'].ep.values,p)}
out['_city_note']='per-city in stageE_compare_ext.json / e0_teacher_refine.json'
json.dump(out,open('/home/external-user/yongjae/SSR/report/refiner_T/stageE_diag/summary/summary_tables.json','w'),indent=1)
print(json.dumps(out,indent=1))

# ---- per city: gap / draft / refiner parts (added) ----
cm_=pd.read_parquet('/home/external-user/ssd/yongjae_refiner/splits/navtest.parquet',columns=['token','city']).drop_duplicates('token').set_index('token').city.reindex(tok).values
pc={}
for c in sorted(set(cm_)):
    m=cm_==c
    pc[f'{c} (n={m.sum()})']={k:boot(arms[a].pdms.values-arms[b].pdms.values,m) for k,(a,b) in
      {'gap E0+R_T4-E2':('E0+R_T4','E2'),'draft':('E0+R_T4','E2t0+R_T4'),'refiner':('E2t0+R_T4','E2'),'E2-E0':('E2','E0'),'E2_tau0-E0':('E2_tau0','E0'),'E2-E2_tau0':('E2','E2_tau0'),'E2t0+R_T4-E2_tau0':('E2t0+R_T4','E2_tau0'),'refiner vs R_M4':('E2t0+R_M4','E2')}.items()}
    pc[f'{c} (n={m.sum()})']['E2_pdms']=round(100*arms['E2'].pdms.values[m].mean(),2)
out['per_city']=pc
json.dump(out,open('/home/external-user/yongjae/SSR/report/refiner_T/stageE_diag/summary/summary_tables.json','w'),indent=1)
for k,v in pc.items(): print(k, v)

# ---- top logs by contribution to gap / refiner part (added) ----
cc=pd.Series(cm_,index=tok)
def toplogs(a,b,n=8):
    d=(arms[a].pdms-arms[b].pdms)*100/len(tok)
    g=d.groupby(arms['E0'].log).agg(['sum','size']).sort_values('sum',ascending=False)
    g['city']=[cc[arms['E0'].log==l].iloc[0] for l in g.index]
    g['sum']=g['sum'].round(3)
    return g.head(n).reset_index().values.tolist(), round(float(g['sum'].head(10).sum()),3)
out['top_logs']={'gap E0+R_T4-E2':toplogs('E0+R_T4','E2'),'refiner E2t0+R_T4-E2':toplogs('E2t0+R_T4','E2'),'draft E0+R_T4-E2t0+R_T4':toplogs('E0+R_T4','E2t0+R_T4')}
json.dump(out,open('/home/external-user/yongjae/SSR/report/refiner_T/stageE_diag/summary/summary_tables.json','w'),indent=1)
for k,v in out['top_logs'].items():
    print(k,'top10 sum',v[1])
    for r in v[0]: print('  ',r)
