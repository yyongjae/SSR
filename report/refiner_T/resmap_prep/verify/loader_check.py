import sys, json, pickle, os, numpy as np, pandas as pd, torch
sys.path.insert(0,'/home/external-user/yongjae/SSR')
from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache, resmap_to_s_grid, resmap_vectors_to_n
from navsim.agents.para_ssr.refiner import adapters
R='/home/external-user/datasets/teacher_cache/resmap'
jobs=pd.read_parquet(sys.argv[1]+'/_jobs.parquet')
for sub in ('navtrain','navtest'):
    c=ResmapCache.for_subset(sub); root=R if sub=='navtrain' else R+'/navtest'
    idx=json.load(open(root+'/index.json')); bad=0
    for t in jobs[jobs.subset==sub].token.values[:60]:
        sh,row=idx[t]
        b=np.load(f'{root}/bev/{sh}.npy',mmap_mode='r')[row]; s=np.load(f'{root}/seg/{sh}.npy',mmap_mode='r')[row]
        lb=c.load_bev(t); ls=c.load_seg(t,s_grid=True)
        ok = lb.shape==(256,50,100) and lb.dtype==np.float16 and lb.flags.c_contiguous and np.array_equal(lb, b.transpose(0,2,1)) \
             and ls.shape==(4,100,200) and np.array_equal(ls, s.transpose(0,2,1)) and np.array_equal(c.load(t,np.float32), lb.astype(np.float32))
        # S-grid cell semantics: S[r,c] must be raw[a=c, b=r]
        ok &= lb[5,10,70]==b[5,70,10]
        bad += not ok
    print(sub,'loader mismatches',bad, 'split meta', c.meta.get('split'), 'n', len(c.index))
    # torch path
    tt=torch.from_numpy(np.ascontiguousarray(b)); print(' torch equal', torch.equal(resmap_to_s_grid(tt), torch.from_numpy(b.transpose(0,2,1).copy())))
# pickle round trip drops memmaps
c=ResmapCache.for_subset('navtrain'); _=c.load_bev(jobs.token.values[0]); p=pickle.loads(pickle.dumps(c)); print('pickled mm', len(p._mm), 'orig mm', len(c._mm))
# s_grid_cell consistency: point (x=10, y_left=+5) -> S cell (row, col); raw cell a=(32-5)/.64-.5, b=10/.64-.5
r,cc=adapters.s_grid_cell(10.0,5.0); print('s_grid_cell', r, cc, 'raw a,b', (32-5)/.64-.5, 10/.64-.5)
# vectors
v=c.load_vectors(jobs.token.values[0]); print('vec range m', v['vectors'][...,0].min(), v['vectors'][...,0].max(), v['vectors'][...,1].min(), v['vectors'][...,1].max())
# has() on val_logs tokens
tr=pd.read_parquet('/home/external-user/ssd/yongjae_refiner/splits/train.parquet'); dv=pd.read_parquet('/home/external-user/ssd/yongjae_refiner/splits/dev.parquet')
val=pd.concat([tr,dv]); val=val[val.part=='val']
print('val tokens', len(val), 'in resmap root', sum(c.has(t) for t in val.token))
# nan/inf scan of seg+scores on all subset tokens (seg only)
