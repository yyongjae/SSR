import json, yaml, numpy as np, pandas as pd
from pathlib import Path
S='/home/external-user/ssd/yongjae_refiner/splits/'
y=yaml.safe_load(open('/home/external-user/yongjae/SSR/navsim/planning/script/config/training/default_train_val_test_log_split.yaml'))
TL,VL,TE=set(y['train_logs']),set(y['val_logs']),set(y.get('test_logs',[]))
print('yaml sizes', len(TL),len(VL),len(TE), 'TL&VL', len(TL&VL))
ridx=json.load(open('/home/external-user/datasets/teacher_cache/resmap/index.json'))
nidx=json.load(open('/home/external-user/datasets/teacher_cache/resmap/navtest/index.json'))
nav=pd.read_parquet(S+'navtest.parquet')
BF=Path('/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100/samples')
out={}
for s in ('train','dev'):
    full=pd.read_parquet(S+f'{s}.parquet'); sub=pd.read_parquet(S+f'{s}_trainlogs.parquet')
    exp=full[full.log.isin(TL)].reset_index(drop=True)
    same=exp.equals(sub.reset_index(drop=True))
    print(s, 'full',len(full),'sub',len(sub),'expected',len(exp),'identical_to_filter',same,
          'dup tokens',sub.token.duplicated().sum(),'logs',sub.log.nunique(),
          'part counts',sub.part.value_counts().to_dict(),
          'dropped part', full[~full.log.isin(TL)].part.value_counts().to_dict(),
          'part=train but log not in TL', ((full.part=='train')&~full.log.isin(TL)).sum(),
          'part=val but log in TL', ((full.part=='val')&full.log.isin(TL)).sum(),
          'in resmap', sub.token.isin(set(ridx)).sum(), 'in navtest idx', sub.token.isin(set(nidx)).sum(),
          'bevfusion has', sum((BF/t[:2]/f'{t}.npz').is_file() for t in sub.token))
    print('  folds', sub.fold.value_counts().sort_index().to_dict(), 'city', sub.map_location.value_counts().to_dict())
    out[s]=sub
print('train/dev log overlap', len(set(out['train'].log)&set(out['dev'].log)), 'token overlap', len(set(out['train'].token)&set(out['dev'].token)))
print('navtest logs in TL', nav.log.isin(TL).sum(), 'navtest tokens in resmap root', nav.token.isin(set(ridx)).sum(), 'navtest in navtest idx', nav.token.isin(set(nidx)).sum(), len(nav))
# fold log-disjointness within train
g=out['train'].groupby('log').fold.nunique(); print('logs spanning >1 fold', (g>1).sum())
sm=json.load(open(S+'trainlogs_summary.json')); print('summary train n',sm['train']['n'], 'dev n', sm['dev']['n'], sm['dev'].get('per_city'))
# packed trainable
for s,p in (('train','train'),('dev','dev')):
    ix=pd.read_parquet(f'/home/external-user/ssd/yongjae_refiner/packed/{p}/index.parquet')
    print(s,'packed rows',len(ix),'sub tokens in packed', out[s].token.isin(set(ix.token)).sum())
