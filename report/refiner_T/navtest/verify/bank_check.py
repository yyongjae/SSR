"""Draft bank dev vs navtest: family-by-slot layout, valid rates, official failure rates per family; cfg hash of every navtest file."""
import json, os, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
R = '/home/external-user/ssd/yongjae_refiner'
FAM = {0: 'identity', 1: 'small', 2: 'lconst', 3: 'ignore_brake', 4: 'creep', 5: 'lat', 6: 'combined', 7: 'cv', 8: 'hdrift'}
def meta(p):
    d = np.load(p, allow_pickle=True); return str(d['cfg_hash']), str(d['version']), d['family'].tolist(), d['valid'].tolist(), str(d['split'])
out = {}
for sp in ['dev', 'navtest']:
    files = sorted(f for f in os.listdir(f'{R}/drafts/{sp}') if len(f) == 20 and f.endswith('.npz'))
    with ProcessPoolExecutor(6) as ex: M = list(ex.map(meta, [f'{R}/drafts/{sp}/{f}' for f in files], chunksize=200))
    fam = np.array([m[2] for m in M]); val = np.array([m[3] for m in M])
    S = pd.read_parquet(f'{R}/scores/{sp}.parquet')
    r = pd.read_parquet(f'{R}/runs/stageT3_T_fold0_seed0/eval_{sp}/report_rows.parquet')
    S = S.merge(r[['token', 'k', 'valid']], on=['token', 'k'], how='inner')
    Sv = S[S.valid]
    fail = lambda d, c: float(100 * (d[c] < 1).mean())
    out[sp] = dict(n_files=len(files), cfg_hashes=sorted(set(m[0] for m in M)), versions=sorted(set(m[1] for m in M)),
                   split_field=sorted(set(m[4] for m in M)), valid_rate_pct=float(100 * val.mean()),
                   n_tokens_eval=int(r.token.nunique()),
                   family_by_slot={int(k): {FAM[f]: int(c) for f, c in zip(*np.unique(fam[:, k], return_counts=True))} for k in range(13)},
                   valid_pct_by_family={FAM[f]: float(100 * val[fam == f].mean()) for f in np.unique(fam)},
                   report_valid_eq_npz=None,
                   fail_by_family={FAM[int(f)]: dict(n=int(len(g)), any=float(100 * ((g.nc < 1) | (g.dac < 1) | (g.ddc < 1)).mean()), nc=fail(g, 'nc'), dac=fail(g, 'dac'), ttc=fail(g, 'ttc'))
                                   for f, g in Sv.groupby('family')},
                   pdms_orig=float(Sv.pdms.mean()), nc=fail(Sv, 'nc'), dac=fail(Sv, 'dac'), ttc=fail(Sv, 'ttc'))
    # valid flag in report rows == draft npz valid (for evaluated tokens)
    vm = {f[:-4]: v for f, v in zip(files, val)}
    rv = r.groupby('token').valid.apply(lambda s: s.to_numpy(bool)).to_dict()
    out[sp]['report_valid_eq_npz'] = int(sum(np.array_equal(vm[t], rv[t]) for t in rv)); out[sp]['n_report_tokens'] = len(rv)
print(json.dumps(out, indent=1))
