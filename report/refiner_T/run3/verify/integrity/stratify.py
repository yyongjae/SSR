"""R_T - R_none (dev, theta 0 as selected) stratified by teacher per-token detection quality and by whether the
dev segment's parent drive also has train segments. Log-cluster bootstrap (2,000) per stratum."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/home/external-user/yongjae/SSR/tools/refiner'); sys.path.insert(0,'/home/external-user/yongjae/SSR')
from stageT_decision import load_rows, final_outcomes, cluster_boot
R='/home/external-user/ssd/yongjae_refiner/runs'
fT=final_outcomes(load_rows(f'{R}/stageT3_T_fold0_seed0/eval_dev'),0.0)
fN=final_outcomes(load_rows(f'{R}/stageT3_none_fold0_seed0/eval_dev'),0.0)
m=fT.merge(fN,on=['token','k','log','family'],suffixes=('_T','_N'),validate='one_to_one')
m['dp']=100*(m.pdms_final_T-m.pdms_final_N); m['dncttc']=100*(m.fail_ncttc_N-m.fail_ncttc_T)
m['ddac']=100*(m.fail_dac_T-m.fail_dac_N)
q=pd.read_parquet('teacher_quality_tokens.parquet')
nt_rand=set(pd.read_parquet('/home/external-user/ssd/yongjae_refiner/objects/_validation/navtest_rand500.parquet').token)
out={}
# navtest rand500 only
qn=q[(q.split=='navtest')&q.token.isin(nt_rand)]; qd=q[q.split=='dev']
for nm,g in (('dev',qd),('navtest_rand500',qn)):
    rec=np.where(g.n_gt>0,g.tp2/g.n_gt.clip(lower=1),1.0); miss=(g.n_gt-g.tp2)
    out[f'quality_{nm}']=dict(n=len(g),recall2_pooled=g.tp2.sum()/g.n_gt.sum(),precision2_pooled=g.tp2.sum()/g.n_pred.sum(),
        mean_missed_per_tok=miss.mean(),frac_tok_recall_ge_0p95=float((rec>=0.95).mean()),frac_tok_zero_miss=float((miss==0).mean()))
qd=qd.assign(miss=qd.n_gt-qd.tp2, fp=qd.n_pred-qd.tp2, rec=np.where(qd.n_gt>0,qd.tp2/qd.n_gt.clip(lower=1),1.0))
m=m.merge(qd[['token','miss','fp','rec','n_gt']],on='token',how='left')
def strat(col,bins,labels):
    r={}
    c=pd.cut(m[col],bins=bins,labels=labels,include_lowest=True)
    for lab in labels:
        g=m[c==lab]
        if len(g)==0: continue
        r[lab]=dict(n_drafts=len(g),n_tok=g.token.nunique(),
                    P1=cluster_boot(g.dp.to_numpy(),g.log.to_numpy(),2000),
                    P2=cluster_boot(g.dncttc.to_numpy(),g.log.to_numpy(),2000),
                    dac_excess=float(g.ddac.mean()))
    return r
out['by_teacher_recall2m']=strat('rec',[-0.01,0.8,0.95,1.0],['<0.8','0.8-0.95','>=0.95'])
out['by_teacher_missed_objects']=strat('miss',[-0.5,0.5,2.5,1e9],['0','1-2','>=3'])
out['by_teacher_false_pos']=strat('fp',[-0.5,0.5,1e9],['0','>=1'])
# reweight dev per-token advantage to navtest miss-distribution (miss bins)
bins=[-0.5,0.5,2.5,1e9]; labs=['0','1-2','>=3']
qn2=qn.assign(miss=qn.n_gt-qn.tp2)
wn=pd.cut(qn2.miss,bins,labels=labs).value_counts(normalize=True).reindex(labs)
wd=pd.cut(m.drop_duplicates('token').miss,bins,labels=labs).value_counts(normalize=True).reindex(labs)
mu={l:out['by_teacher_missed_objects'][l]['P1']['mean'] for l in labs}
mu2={l:out['by_teacher_missed_objects'][l]['P2']['mean'] for l in labs}
out['reweight_to_navtest_miss_dist']=dict(w_dev=wd.to_dict(),w_navtest=wn.to_dict(),
    P1_dev=float(sum(wd[l]*mu[l] for l in labs)),P1_reweighted=float(sum(wn[l]*mu[l] for l in labs)),
    P2_dev=float(sum(wd[l]*mu2[l] for l in labs)),P2_reweighted=float(sum(wn[l]*mu2[l] for l in labs)))
# drive overlap
tr=pd.read_parquet('/home/external-user/yongjae/../ssd/yongjae_refiner/splits/train.parquet') if False else pd.read_parquet('/home/external-user/ssd/yongjae_refiner/splits/train.parquet')
td=set(tr.log.str.rsplit('_',n=2).str[0])
m['drive_in_train']=m.log.str.rsplit('_',n=2).str[0].isin(td)
out['by_drive_in_train']={str(k):dict(n_drafts=len(g),n_logs=g.log.nunique(),P1=cluster_boot(g.dp.to_numpy(),g.log.to_numpy(),2000),
    P2=cluster_boot(g.dncttc.to_numpy(),g.log.to_numpy(),2000)) for k,g in m.groupby('drive_in_train')}
# spearman per-token
tokagg=m.groupby('token').agg(dp=('dp','mean'),miss=('miss','first'),rec=('rec','first'))
out['spearman_tok_dp_vs_miss']=float(tokagg[['dp','miss']].corr('spearman').iloc[0,1])
print(json.dumps(out,indent=1,default=float)); json.dump(out,open('stratify.json','w'),indent=1,default=float)
