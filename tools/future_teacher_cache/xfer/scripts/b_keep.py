# read-only: download-frame counts per set/horizon (k up to 10 = 5 s) and keep-size estimates
import json, yaml, pandas as pd, numpy as np
S="/home/external-user/ssd/yongjae_refiner/splits"
df=pd.read_parquet("frames_avail.parquet").sort_values(["split","log","idx"]).reset_index(drop=True)
df["disk"]=df.d_lidar&df.d_CAM_F0&df.d_CAM_L0&df.d_CAM_R0
pos=df.index.values; logn=df.groupby(["split","log"]).idx.transform("size").values; idx=df.idx.values
def shift(k):
    ok=(idx+k>=0)&(idx+k<logn); return np.where(ok,pos+k,-1)
tok2row=pd.Series(df.index.values,index=df.token)
sets={"navtest":tok2row[pd.read_parquet(f"{S}/navtest.parquet").token].values,
 "stageT":np.union1d(tok2row[pd.read_parquet(f"{S}/train_trainlogs.parquet").token].values,tok2row[pd.read_parquet(f"{S}/dev_trainlogs.parquet").token].values),
 "e2e":tok2row[pd.read_parquet(f"{S}/e2e_train_trainlogs.parquet").token].values,
 "navtrain":df.index[df.c_bev_train].values}
disk=df.disk.values; bev=df.c_bev.values; cam3=(df.d_CAM_F0&df.d_CAM_L0&df.d_CAM_R0).values; lid=df.d_lidar.values
SH={k:shift(k) for k in range(-2,11)}
out={}
for n,rows in sets.items():
    o={}
    for H in (8,10):
        F=set()
        for k in range(1,H+1):
            r=SH[k][rows]; F.update(r[r>=0].tolist())
        F=np.array(sorted(F)); run=F[~bev[F]]
        o[f"h{H/2}"]=dict(future=len(F),run=len(run),run_dl=int((~disk[run]).sum()),all_dl=int((~disk[F]).sum()),
                          dl_cam3_missing=int((~cam3[F]).sum()))
    # t+0.5 only (SSR-style next-frame), all future frames needing cams (not just run)
    r=SH[1][rows]; F1=np.unique(r[r>=0]); o["k1_only"]=dict(future=len(F1),cam3_missing=int((~cam3[F1]).sum()))
    out[n]=o
print(json.dumps(out,indent=1))
json.dump(out,open("b_keep.json","w"),indent=1)
# sweep (f-1,f-2 lidar) frames of run frames that are neither in F nor have lidar on disk, at 5 s
for n,rows in sets.items():
    F=set()
    for k in range(1,11):
        r=SH[k][rows]; F.update(r[r>=0].tolist())
    Fa=np.array(sorted(F)); run=Fa[~bev[Fa]]; sw=set()
    for k in (1,2):
        r=SH[-k][run]; sw.update(r[r>=0].tolist())
    sw=np.array(sorted(sw-F),dtype=int)
    print(n,"h5 sweep frames outside F:",len(sw),"of which lidar missing:",int((~lid[sw]).sum()) if len(sw) else 0)
