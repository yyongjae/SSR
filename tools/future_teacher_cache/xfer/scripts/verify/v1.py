import glob, json, os, pickle, random, collections, numpy as np, pandas as pd
S="/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad"
LOGD="/home/external-user/navsim/download/trainval_navsim_logs/trainval"
BLOB="/home/external-user/navsim/download/trainval_sensor_blobs/trainval"
CACHE="/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100"
SP="/home/external-user/ssd/yongjae_refiner/splits"
W="/home/external-user/yongjae/SSR/tools/future_teacher_cache"
LF=pickle.load(open(f"{S}/logframes.pkl","rb"))
# spot check logframes vs raw pkl
random.seed(1)
for log in random.sample(sorted(LF),4):
    fr=pickle.load(open(f"{LOGD}/{log}.pkl","rb"))
    ok = len(fr)==len(LF[log]) and all(f['token']==g[0] and (os.path.basename(f['lidar_path']) if f.get('lidar_path') else None)==g[1] and os.path.basename(f['cams']['CAM_F0']['data_path'])==g[2]['CAM_F0'] for f,g in zip(fr,LF[log]))
    print('spot',log,len(fr),ok)
print('logs',len(LF),'pkl files',len(glob.glob(f"{LOGD}/*.pkl")))
amap=json.load(open(f"{S}/hf/map_trainval.json"))
allm=[l for v in amap.values() for l in v]
print('map logs',len(allm),len(set(allm)),set(allm)==set(LF), collections.Counter(len(v) for v in amap.values()))
log2arc={l:int(k.rsplit('_',1)[1]) for k,v in amap.items() for l in v}
tot=sum(len(v) for v in LF.values()); print('frames all',tot)
cached=set(os.path.basename(p)[:-4] for p in glob.glob(f"{CACHE}/samples/*/*.npz")); print('cached',len(cached))
idx=json.load(open(f"{W}/future_index_train.json"))
print('idx', len(idx), collections.Counter(len(v) for v in idx.values()).most_common(3))
tok2={}
for log,fr in LF.items():
    for i,f in enumerate(fr): tok2[f[0]]=(log,i)
disk={}
def od(log):
    if log not in disk:
        d={}
        for sub in ("CAM_F0","CAM_L0","CAM_R0","MergedPointCloud"):
            try: d[sub]=set(os.listdir(f"{BLOB}/{log}/{sub}"))
            except FileNotFoundError: d[sub]=set()
        disk[log]=d
    return disk[log]
def camok(log,i): f=LF[log][i]; d=od(log); return all(f[2][c] in d[c] for c in ("CAM_F0","CAM_L0","CAM_R0"))
def lidok(log,i): f=LF[log][i]; return f[1] is not None and f[1] in od(log)["MergedPointCloud"]
sets={"stageT":set(pd.read_parquet(f"{SP}/train_trainlogs.parquet").token)|set(pd.read_parquet(f"{SP}/dev_trainlogs.parquet").token),
      "e2e":set(pd.read_parquet(f"{SP}/e2e_train_trainlogs.parquet").token),"navtrain":set(idx)}
# per-archive total frames
afr=np.zeros(200); 
for l,fr in LF.items(): afr[log2arc[l]]+=len(fr)
sz={m:{int(x['path'].rsplit('_',1)[1][:-4]):x['size'] for x in json.load(open(f'tree_{m}.json')) if x['path'].endswith('.tgz')} for m in ('camera','lidar')}
cs=np.array([sz['camera'][i] for i in range(200)]); ls=np.array([sz['lidar'][i] for i in range(200)])
for nm,s in (('cam',cs),('lid',ls)):
    r=s/afr; print(nm,'corr',np.corrcoef(s,afr)[0,1],'B/frame mean',s.sum()/afr.sum(),'per-arc ratio rel min/max',(r/np.median(r)).min(),(r/np.median(r)).max())
pair=cs+ls; print('pair mean',pair.mean()/1e9,'max',pair.max()/1e9,'argmax',pair.argmax())
out={}
F3=2028814; CAM3=625726; PCD=1403088
for name,toks in sets.items():
  for H in (4,5):
    run=set()
    for t in toks:
        for f in (idx.get(t) or [])[:2*H]:
            if f is not None and f not in cached: run.add(f)
    ondisk={f for f in run if camok(*tok2[f]) and lidok(*tok2[f])}
    dl=run-ondisk
    # S1 files needed
    camN=np.zeros(200); lidN=np.zeros(200); lidN2=np.zeros(200); camN2=np.zeros(200)
    sweepS1=set(); sweepS2=set()
    for f in run:
        log,i=tok2[f]; a=log2arc[log]
        if not camok(log,i): camN[a]+=1
        if not lidok(log,i): lidN[a]+=1
        camN2[a]+=1; lidN2[a]+=1
        for k in (1,2):
            j=i-k
            if j<0 or LF[log][j][1] is None: break
            if LF[log][j][0] in run: continue
            sweepS2.add((log,j))
            if not lidok(log,j): sweepS1.add((log,j))
    for (log,j) in sweepS1: lidN[log2arc[log]]+=1
    for (log,j) in sweepS2: lidN2[log2arc[log]]+=1
    perS1=(camN*CAM3+lidN*PCD)/1e9; perS2=(camN2*CAM3+lidN2*PCD)/1e9
    def peaks(per):
        r={}
        for g in (1,10,40):
            grp=[per[k:k+g].sum() for k in range(0,200,g)]
            r[g]=(round(max(grp),1), round(max(grp[k]+grp[k+1] for k in range(len(grp)-1)),1))
        return r
    perdl=np.zeros(200)
    for f in dl: perdl[log2arc[tok2[f][0]]]+=1
    # compressed bytes needed S1 -> discarded fraction
    comp=(camN*CAM3/1.003+lidN*PCD/1.134).sum()/1e9
    out[f"{name}_{H}s"]=dict(run=len(run),ondisk=len(ondisk),dl=len(dl),
       cam_missing=int(camN.sum()),lid_missing=int(lidN.sum()),sweepS1=len(sweepS1),sweepS2=len(sweepS2),
       S1GB=round(perS1.sum(),1),S2GB=round(perS2.sum(),1),peakS1=peaks(perS1),peakS2=peaks(perS2),
       arcs_zero_dl=int((perdl==0).sum()),max_dl_arc=int(perdl.max()),
       logs=len({tok2[f][0] for f in run}), discard_frac_S1=round(1-comp/2124.35,4))
    print(name,H,out[f"{name}_{H}s"],flush=True)
# all-frame scopes
navlogs={tok2[t][0] for t in idx}
for nm,logs in (('trainval_all',set(LF)),('navtrain_logs',navlogs)):
    fr=[(l,i) for l in logs for i in range(len(LF[l]))]
    runf=[x for x in fr if LF[x[0]][x[1]][0] not in cached]
    camN=sum(1 for x in runf if not camok(*x)); lidN=sum(1 for x in fr if not lidok(*x))  # all lidar in logs needed for sweeps
    notdisk3=sum(1 for x in fr if not camok(*x))
    out[nm]=dict(logs=len(logs),frames=len(fr),run=len(runf),S1GB=round((camN*CAM3+lidN*PCD)/1e9,1),front3_notdisk_GB=round(notdisk3*CAM3/1e9,1))
    print(nm,out[nm])
json.dump(out,open('v1.json','w'),indent=1,default=str)
