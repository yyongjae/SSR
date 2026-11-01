import json, glob, pandas as pd
S='/home/external-user/ssd/yongjae_refiner/splits'; D='/home/external-user/ssd/yongjae_refiner'
tr=pd.read_parquet(f'{S}/train.parquet'); dv=pd.read_parquet(f'{S}/dev.parquet'); T=set(tr.token); Dv=set(dv.token); DL=set(dv.log)
out={}
def chk(name,toks,logs=None):
    toks=set(toks); d=dict(n=len(toks),in_train=len(toks&T),in_dev=len(toks&Dv))
    if logs is not None: d['logs_in_dev']=len(set(logs)&DL)
    out[name]=d
t=pd.read_parquet(f'{D}/m8_recheck/tokens.parquet'); chk('m8_sweep_pool',t.token,t.get('log'))
c=pd.read_csv('/home/external-user/yongjae/SSR/report/refiner_T/m8_recheck/confirm/tokens_confirm.csv'); chk('m8_confirm_pool',c.token,c.get('log'))
r=pd.read_parquet(f'{D}/ttc_select/rows.parquet',columns=['token']); chk('ttc_select_rows',r.token)
lp=pd.read_parquet(f'{D}/m8_recheck/labels_pool.parquet',columns=['token']); chk('m8_labels_pool',lp.token)
for p in sorted(glob.glob(f'{D}/runs/pilot2V*/config.json'))+sorted(glob.glob(f'{D}/runs/stageT*_fold0_seed0/config.json')):
    c=json.load(open(p)); out[p.split('/')[-2]]=dict(split=c.get('split'),fold=c.get('fold'),packed=c.get('packed'),epochs=c.get('epochs'),lr=c.get('lr'),patience=c.get('patience'),
        lam=c.get('lon_st_slope'),w=c.get('w'),m_ttc=c.get('m_ttc'),n_train=c.get('n_train_tokens'),n_ival=c.get('n_ival_tokens'),code_train=c.get('code',{}).get('train_refiner.py'),code_sur=c.get('code',{}).get('surrogate.py'))
# inner-val vs confirm pool
print(json.dumps(out,indent=1)); json.dump(out,open('check_selection_sources.json','w'),indent=1)
