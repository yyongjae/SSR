exec(open('v1.py').read().split("out={}")[0].replace("for log in random.sample(sorted(LF),4)","for log in []"))
CAM3=625726; PCD=1403088
navlogs={tok2[t][0] for t in idx}
for nm,logs in (('trainval_all',set(LF)),('navtrain_logs',navlogs)):
    per=np.zeros(200)
    for l in logs:
        a=log2arc[l]
        for i,f in enumerate(LF[l]):
            if f[0] not in cached and not camok(l,i): per[a]+=CAM3
            if not lidok(l,i): per[a]+=PCD
    per/=1e9
    r={}
    for g in (1,10,40):
        grp=[per[k:k+g].sum() for k in range(0,200,g)]
        r[g]=(round(max(grp),1), round(max(grp[k]+grp[k+1] for k in range(len(grp)-1)),1))
    print(nm,round(per.sum(),1),r)
