import os; os.environ['OMP_NUM_THREADS']='1'
import json, sys, numpy as np, pandas as pd
from multiprocessing import Pool
R='/home/external-user/datasets/teacher_cache/resmap'; D='/home/external-user/ssd/yongjae_refiner'
r=(np.arange(100)+.5)*.32; c=32-(np.arange(200)+.5)*.32
I=np.clip(np.round((r+8-0.125)/0.25).astype(int),0,319); J=np.clip(np.round((32-0.125-c)/0.25).astype(int),0,255)
G={}
def init():
    G['idx']={'navtrain':json.load(open(R+'/index.json')),'navtest':json.load(open(R+'/navtest/index.json'))}
    G['mm']={}; G['sdf']={p:np.load(f'{D}/packed/{p}/sdf.npy',mmap_mode='r') for p in ('train','dev')}
def mm(sub,f,sh):
    k=(sub,f,sh)
    if k not in G['mm']: G['mm'][k]=np.load((R if sub=='navtrain' else R+'/navtest')+f'/{f}/{sh}.npy',mmap_mode='r')
    return G['mm'][k]
def work(a):
    t,sub,part,row,bevchk=a
    sh,rw=G['idx'][sub][t]
    seg=np.asarray(mm(sub,'seg',sh)[rw],np.float32)
    if sub=='navtrain': sdf=np.asarray(G['sdf'][part][row],np.float32)
    else:
        with np.load(f'{D}/sdf/navtest/{t}.npz') as z: sdf=z['sdf'].astype(np.float32)
    gt=sdf[I][:,J]>0
    S=np.swapaxes(seg,-1,-2)
    out=[t,sub,bool(np.isfinite(seg).all()), float(np.abs(seg).max())]
    for X in (S, S[...,::-1], S[...,::-1,:]):
        rd=X[0]>0; out.append(float((rd&gt).sum()/max((rd|gt).sum(),1)))
    if bevchk:
        b=np.asarray(mm(sub,'bev',sh)[rw],np.float32); out+= [bool(np.isfinite(b).all()), float((b==0).mean()), float(b.std())]
    else: out+=[None,None,None]
    return out
if __name__=='__main__':
    jobs=[]
    for p in ('train','dev'):
        sub=pd.read_parquet(f'{D}/splits/{p}_trainlogs.parquet'); ix=pd.read_parquet(f'{D}/packed/{p}/index.parquet').set_index('token')
        rng=np.random.default_rng(1); chk=set(rng.choice(sub.token.values,1000,replace=False))
        jobs+=[(t,'navtrain',p,int(ix.loc[t,'row']),t in chk) for t in sub.token]
    nav=pd.read_parquet(f'{D}/splits/navtest.parquet'); rng=np.random.default_rng(2); chk=set(rng.choice(nav.token.values,500,replace=False))
    jobs+=[(t,'navtest',None,-1,t in chk) for t in nav.token]
    with Pool(4,initializer=init) as P: res=P.map(work,jobs,chunksize=64)
    df=pd.DataFrame(res,columns=['token','subset','seg_finite','seg_absmax','iou','iou_mirror','iou_fflip','bev_finite','bev_zero_frac','bev_std'])
    df.to_parquet(sys.argv[1]); 
    for s,d in df.groupby('subset'):
        print(s,len(d),'seg nonfinite',(~d.seg_finite).sum(),'iou median',d.iou.median(),'p1',d.iou.quantile(.01),
              '<0.3',(d.iou<0.3).sum(),'<0.5',(d.iou<0.5).sum(),'mirror>chosen',(d.iou_mirror>d.iou).sum(),'fflip>chosen',(d.iou_fflip>d.iou).sum(),
              'bev checked',d.bev_finite.notna().sum(),'bev nonfinite',(d.bev_finite==False).sum(),'bev zero frac max',d.bev_zero_frac.max(),'bev std min',d.bev_std.min())
        print(d[(d.iou_mirror>d.iou)|(d.iou_fflip>d.iou)].sort_values('iou').head(12).to_string())
