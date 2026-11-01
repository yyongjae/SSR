import os
os.environ.setdefault('NUPLAN_MAPS_ROOT','/home/external-user/yongjae/SSR/data/dataset/maps')
os.environ.setdefault('NUPLAN_MAP_VERSION','nuplan-maps-v1.0')
os.environ.setdefault('OPENSCENE_DATA_ROOT','/home/external-user/yongjae/SSR/data/dataset')
import json, numpy as np, torch
from pathlib import Path
from navsim.agents.para_ssr.refiner import e2e as E
from navsim.agents.para_ssr.refiner.data import TeacherCache
from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
root=Path('/home/external-user/ssd/yongjae_refiner')
rng=np.random.default_rng(20260929)
with np.load(root/E.HUMAN_NPZ,allow_pickle=False) as z:
 ix=rng.choice(len(z['tokens']),64,replace=False)
 H={k:z[k][ix] for k in ['traj','path','n_reg','v0','a0','eds','cmd','tokens']}
tau=torch.from_numpy(H['traj']); ego=[torch.from_numpy(H[k]) for k in ['v0','a0','eds','cmd']]
draft,pert,inval=E.perturb_batch(tau,ego[0],ego[1],1.0,rng,src=tau,src_path=torch.from_numpy(H['path']),src_nreg=torch.from_numpy(H['n_reg']))
cs=[]; raws=[]; nets=[]
tr=E.train_refiner_module()
for arm,cache in [('T',TeacherCache.for_subset('navtrain')),('M',ResmapCache.for_subset('navtrain'))]:
 net,_=tr.load_run_model(root/f'stageE/teachers/stageT4_{arm}_fold0_seed0','best','cpu')
 net.eval(); net.requires_grad_(False); oo=[]; zz=[]
 with torch.no_grad():
  for j in range(0,len(tau),4):
   bev=torch.from_numpy(np.stack([cache.load_bev(t) for t in H['tokens'][j:j+4]])).float()
   eo=[x[j:j+4] for x in ego]; d=draft[j:j+4]
   out=net(bev,d[:,None],*eo)
   dec=E.StageE._decode(None,d,out,eo[0],0.0)
   oo.append(E.kd_controls(out,dec,'decoded'));zz.append(out['z_lon'][:,0])
 cs.append(torch.cat(oo));raws.append(torch.cat(zz))
ct,cm=cs
zero=torch.zeros_like(ct,requires_grad=True)
loss,_=E.kd_loss(zero,cs,[torch.ones(len(tau),dtype=torch.bool)]*2);loss.backward()
res={'n':len(tau),'perturbed':int(pert.sum()),'invalid':int(inval.sum()),'lat_opposite_sign_frac':float(((ct[:,6:]*cm[:,6:])<0).float().mean()),'lat_grad_zero_frac_at_student_identity':float((zero.grad[:,6:]==0).float().mean()),'lon_grad_zero_frac_at_student_identity':float((zero.grad[:,:6]==0).float().mean()),'loss_at_identity':float(loss)}
for arm,c,r in zip(['T','M'],cs,raws):
 res[arm]={'live':float((c[:,:6]<0).any(-1).float().mean()),'mean_abs_lon':float(c[:,:6].abs().mean()),'mean_abs_lat':float(c[:,6:].abs().mean()),'z_quantiles':torch.quantile(r.flatten(),torch.tensor([0.,.5,.9,1.])).tolist(),'frac_z_saturated_above_5':float((r>5).float().mean())}
print(json.dumps(res,indent=2)); Path('/tmp/refiner_audit_teacher.json').write_text(json.dumps(res,indent=2))
