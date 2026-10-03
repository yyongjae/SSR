exec(open('v1.py').read().split("out={}")[0].replace("for log in random.sample(sorted(LF),4)","for log in []"))
# check idx == next 8 frames in same log
bad=0
for t,fut in list(idx.items()):
    log,i=tok2[t]
    exp=[LF[log][i+k][0] if i+k<len(LF[log]) else None for k in range(1,9)]
    if exp!=fut: bad+=1
print('idx mismatches vs next-8-in-log',bad)
F3=2028814; CAM3=625726; PCD=1403088
res={}
for name,toks in sets.items():
    run=set()
    for t in toks:
        log,i=tok2[t]
        for k in range(1,11):
            if i+k<len(LF[log]):
                f=LF[log][i+k][0]
                if f not in cached: run.add(f)
    ondisk={f for f in run if camok(*tok2[f]) and lidok(*tok2[f])}
    dl=run-ondisk
    camN=sum(1 for f in run if not camok(*tok2[f])); lidN=sum(1 for f in run if not lidok(*tok2[f]))
    sw1=set();sw2=set()
    for f in run:
        log,i=tok2[f]
        for k in (1,2):
            j=i-k
            if j<0 or LF[log][j][1] is None: break
            if LF[log][j][0] in run: continue
            sw2.add((log,j))
            if not lidok(log,j): sw1.add((log,j))
    S1=(camN*CAM3+(lidN+len(sw1))*PCD)/1e9
    comp=(camN*CAM3/1.003+(lidN+len(sw1))*PCD/1.134)/1e9
    res[name]=dict(run=len(run),dl=len(dl),camN=camN,lidN=lidN,sw1=len(sw1),sw2=len(sw2),S1GB=round(S1,1),discard=round(1-comp/2124.35,4),
                   keep3_GB=round(camN*CAM3/1e9,1),npz_drop=round(len(run)*87886/1e9,2),npz_bev=round(len(run)*2648014/1e9,1),min_in_GB=round(len(dl)*F3/1e9,1))
    print(name,'5s',res[name],flush=True)
# navtest 5s run
