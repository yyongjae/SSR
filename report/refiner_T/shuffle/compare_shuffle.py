"""AMENDMENT 5 teacher-shuffle control (descriptive, dev only): R_T(true BEV) vs R_T(shuffled BEV) vs R_none, theta 0,
same drafts (paired by token, k), log-cluster paired bootstrap 10,000 (stageT_decision.cluster_boot, seed 0).
  python report/refiner_T/shuffle/compare_shuffle.py
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/external-user/yongjae/SSR/tools/refiner'); sys.path.insert(0, '/home/external-user/yongjae/SSR')
from stageT_decision import load_rows, final_outcomes, cluster_boot

R = '/home/external-user/ssd/yongjae_refiner/runs'; S = '/home/external-user/ssd/yongjae_refiner/splits'
OUT = '/home/external-user/yongjae/SSR/report/refiner_T/shuffle/shuffle.json'
NB = 10000
dirs = dict(true=f'{R}/stageT3_T_fold0_seed0/eval_dev', shuf=f'{R}/stageT3_T_fold0_seed0/eval_dev_shuffle',
            none=f'{R}/stageT3_none_fold0_seed0/eval_dev')
F = {a: final_outcomes(load_rows(d), 0.0) for a, d in dirs.items()}
for a, f in F.items():
    for k in ('nc', 'ttc', 'dac'):
        f[f'fail_{k}'] = (f[f'{k}_final'] < 1).astype(float)
        f[f'fail_{k}_o'] = (f[f'{k}_orig'] < 1).astype(float)
cols = ['pdms_final', 'pdms_orig', 'fail_nc', 'fail_ttc', 'fail_dac', 'fail_ncttc', 'new_fail', 'modified',
        'fail_nc_o', 'fail_ttc_o', 'fail_dac_o']
key = ['token', 'k', 'log', 'family']
m = F['true'][key + cols].merge(F['shuf'][key + cols], on=key, suffixes=('', '_shuf'), validate='one_to_one')
m = m.merge(F['none'][key + cols].add_suffix('_none').rename(columns={f'{c}_none': c for c in key}), on=key,
            validate='one_to_one')
m.columns = [c if (c in key or c.endswith(('_shuf', '_none'))) else c + '_true' for c in m.columns]
assert (m.pdms_orig_true == m.pdms_orig_shuf).all() and (m.pdms_orig_true == m.pdms_orig_none).all()
dv = pd.read_parquet(f'{S}/dev.parquet')[['token', 'map_location']]; tr = pd.read_parquet(f'{S}/train.parquet')
m = m.merge(dv, on='token', how='left')
m['sib'] = m.log.str.rsplit('_', n=2).str[0].isin(set(tr.log.str.rsplit('_', n=2).str[0]))

def rates(sfx):
    return dict(pdms=float(m[f'pdms_final_{sfx}'].mean()),
                **{f'fail_{k}_pct': 100 * float(m[f'fail_{k}_{sfx}'].mean()) for k in ('nc', 'ttc', 'dac', 'ncttc')},
                new_fail_pct=100 * float(m[f'new_fail_{sfx}'].mean()))

def contrast(g, a, b, nb=NB):
    """a - b; PDMS in points (+ = a better); failure rates in pp (+ = a fails MORE)."""
    L = g.log.to_numpy(); r = {}
    sc = lambda d: {k: (100 * v if k in ('mean', 'lo', 'hi') else v) for k, v in cluster_boot(d.to_numpy(np.float64), L, nb).items()}
    r['pdms_points'] = sc(g[f'pdms_final_{a}'] - g[f'pdms_final_{b}'])
    for k in ('nc', 'ttc', 'dac', 'ncttc'):
        r[f'fail_{k}_pp'] = sc(g[f'fail_{k}_{a}'] - g[f'fail_{k}_{b}'])
    r['new_fail_pp'] = sc(g[f'new_fail_{a}'] - g[f'new_fail_{b}'])
    return r

pairs = [('true', 'shuf'), ('shuf', 'none'), ('true', 'none')]
out = dict(
    what='AMENDMENT 5 teacher-shuffle control (descriptive, dev only). R_T run 3 frozen ckpt_best, BEV lookup remapped by '
         'a seed-0 cyclic derangement of the 7930 dev tokens (each token reads another token\'s BEV); drafts, tokens, '
         'model, theta (0 for both arms) unchanged. Paired by (token, k); log-cluster bootstrap 10,000, seed 0, 95% pct CI.',
    sign='a_minus_b: pdms_points + = a better; fail_*_pp + = a fails more',
    shuffle_meta=json.load(open(f"{dirs['shuf']}/predict_meta.json")),
    n_paired_drafts=int(len(m)), n_logs=int(m.log.nunique()),
    orig=dict(pdms=float(m.pdms_orig_true.mean()), **{f'fail_{k}_pct': 100 * float(m[f'fail_{k}_o_true'].mean()) for k in ('nc', 'ttc', 'dac')}),
    rates={a: rates(a) for a in ('true', 'shuf', 'none')},
    contrasts={f'{a}_minus_{b}': contrast(m, a, b) for a, b in pairs},
    share_of_true_minus_none_retained_by_shuffle={},
    strata={},
)
for k, lab in (('pdms_points', 'pdms'), ('fail_ncttc_pp', 'ncttc')):
    tn = out['contrasts']['true_minus_none'][k]['mean']; sn = out['contrasts']['shuf_minus_none'][k]['mean']
    out['share_of_true_minus_none_retained_by_shuffle'][lab] = sn / tn if tn else None
for name, sel in (('las_vegas', m.map_location == 'us-nv-las-vegas-strip'), ('other_cities', m.map_location != 'us-nv-las-vegas-strip'),
                  ('drive_sibling_in_train', m.sib), ('drive_heldout', ~m.sib)):
    g = m[sel.to_numpy()]
    out['strata'][name] = dict(n=int(len(g)), n_logs=int(g.log.nunique()),
                               **{f'{a}_minus_{b}': {kk: vv for kk, vv in contrast(g, a, b, 2000).items() if kk in ('pdms_points', 'fail_ncttc_pp')}
                                  for a, b in pairs})
json.dump(out, open(OUT, 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ('rates', 'orig', 'share_of_true_minus_none_retained_by_shuffle')}, indent=1))
for p, v in out['contrasts'].items():
    print(p, {k: [round(x['mean'], 3), round(x['lo'], 3), round(x['hi'], 3)] for k, x in v.items()})
for s, v in out['strata'].items():
    print(s, v['n_logs'], {p: {k: [round(x['mean'], 2), round(x['lo'], 2), round(x['hi'], 2)] for k, x in d.items()} for p, d in v.items() if isinstance(d, dict)})
