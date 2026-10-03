"""Integrity: train/dev log disjointness, eval token membership, drive-level (parent log) overlap."""
import json, glob
import pandas as pd, numpy as np
S='/home/external-user/ssd/yongjae_refiner/splits'; R='/home/external-user/ssd/yongjae_refiner/runs'
tr=pd.read_parquet(f'{S}/train.parquet'); dv=pd.read_parquet(f'{S}/dev.parquet')
out={}
out['n_tok']=dict(train=len(tr),dev=len(dv)); out['token_overlap']=len(set(tr.token)&set(dv.token))
out['log_overlap']=len(set(tr.log)&set(dv.log)); out['n_logs']=dict(train=tr.log.nunique(),dev=dv.log.nunique())
drive=lambda s: s.str.rsplit('_',n=2).str[0]
tr['drive']=drive(tr.log); dv['drive']=drive(dv.log)
dd=set(dv.drive); td=set(tr.drive)
out['drives']=dict(train=len(td),dev=len(dd),shared=len(dd&td))
out['dev_tokens_whose_drive_in_train_frac']=float(dv.drive.isin(td).mean())
for tag in ['stageT3']:
  for arm in ['T','none']:
    rd=f'{R}/{tag}_{arm}_fold0_seed0'
    for ev in ['eval_dev','eval_train_fold0']:
      tk=pd.read_parquet(f'{rd}/{ev}/tokens.parquet'); rows=pd.read_parquet(f'{rd}/{ev}/report_rows.parquet')
      toks=set(tk.token); rtoks=set(rows.token)
      d=dict(n_tok=len(toks), rows=len(rows), row_tok_not_in_tokens=len(rtoks-toks),
             in_dev=len(toks&set(dv.token)), in_train=len(toks&set(tr.token)),
             in_train_fold0=len(toks&set(tr[tr.fold==0].token)), in_train_folds1_4=len(toks&set(tr[tr.fold!=0].token)),
             log_mismatch=int((tk.merge(pd.concat([tr,dv])[['token','log']],on='token',suffixes=('','_s')).pipe(lambda x:(x.log!=x.log_s).sum()))))
      out[f'{tag}_{arm}_{ev}']=d
    cfg=json.load(open(f'{rd}/config.json'))
    out[f'{tag}_{arm}_cfg']=dict(n_train_tokens=cfg['n_train_tokens'],n_ival_tokens=cfg['n_ival_tokens'],split=cfg['split'],packed=cfg['packed'])
out['train_fold_tokens']=tr.fold.value_counts().sort_index().to_dict()
print(json.dumps(out,indent=1,default=int)); json.dump(out,open('check_splits.json','w'),indent=1,default=int)
