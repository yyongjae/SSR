"""Teacher (BEVFusion 50x100, trained on all navtrain) detection quality per token: dev (in-sample for the teacher)
vs navtest (out-of-sample), class-agnostic, t0 boxes inside the teacher range (x 0..32, |y| < 32).
Greedy score-ordered centre matching at 1 m and 2 m. GT = objects annotated at keyframe 0 (gt_future npz)."""
import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
TC={'dev':'/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100/samples',
    'navtest':'/home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100/samples'}
OB={'dev':'/home/external-user/ssd/yongjae_refiner/objects/dev','navtest':'/home/external-user/ssd/yongjae_refiner/objects/_validation/navtest'}
THR=0.3
def one(a):
    split,tok=a
    try:
        o=np.load(f'{OB[split]}/{tok}.npz'); t=np.load(f'{TC[split]}/{tok[:2]}/{tok}.npz')
    except FileNotFoundError: return None
    kf=o['kf']; pres=kf[:,0,5]>0.5; g=kf[pres,0,:2].astype(np.float64)
    ing=(g[:,0]>=0)&(g[:,0]<32)&(np.abs(g[:,1])<32); g=g[ing]
    b=t['pred_boxes_3d']; s=t['pred_scores_3d']; keep=s>=THR; p=b[keep,:2].astype(np.float64); ps=s[keep]
    p=p[np.argsort(-ps)]
    res=dict(token=tok,split=split,n_gt=len(g),n_pred=len(p))
    for r in (1.0,2.0):
        used=np.zeros(len(g),bool); tp=0; err=[]
        for q in p:
            if len(g)==0: break
            d=np.hypot(*(g-q).T); d[used]=np.inf; j=int(np.argmin(d))
            if d[j]<=r: used[j]=True; tp+=1; err.append(d[j])
        res[f'tp{int(r)}']=tp
        res[f'err{int(r)}']=float(np.mean(err)) if err else np.nan
    return res
if __name__=='__main__':
    dv=pd.read_parquet('/home/external-user/ssd/yongjae_refiner/runs/stageT3_T_fold0_seed0/eval_dev/tokens.parquet').token.unique()
    import os
    nt=[f[:-4] for f in os.listdir(OB['navtest']) if f.endswith('.npz')]
    jobs=[('dev',t) for t in dv]+[('navtest',t) for t in nt]
    with Pool(4) as P: rs=[r for r in P.map(one,jobs,chunksize=64) if r is not None]
    df=pd.DataFrame(rs); df.to_parquet('teacher_quality_tokens.parquet')
    out={}
    for sp,g in df.groupby('split'):
        d=dict(n_tokens=len(g),gt_per_tok=g.n_gt.mean(),pred_per_tok=g.n_pred.mean())
        for r in (1,2):
            d[f'recall@{r}m_pooled']=g[f'tp{r}'].sum()/g.n_gt.sum(); d[f'precision@{r}m_pooled']=g[f'tp{r}'].sum()/g.n_pred.sum()
            rec=np.where(g.n_gt>0,g[f'tp{r}']/g.n_gt.clip(lower=1),1.0)
            d[f'frac_tokens_recall1_@{r}m']=float((rec>=1).mean())
            d[f'mean_centre_err@{r}m']=float(np.nanmean(g[f'err{r}']))
        out[sp]=d
    print(json.dumps(out,indent=1,default=float)); json.dump(out,open('teacher_quality.json','w'),indent=1,default=float)
