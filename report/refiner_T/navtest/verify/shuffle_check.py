"""Independent recompute of the teacher-shuffle control (dev) for PDMS and NC+TTC; shuffle-map sanity."""
import json, numpy as np, pandas as pd
R = '/home/external-user/ssd/yongjae_refiner/runs'
def load(run, ev):
    r = pd.read_parquet(f'{R}/{run}/{ev}/report_rows.parquet'); tk = pd.read_parquet(f'{R}/{run}/{ev}/tokens.parquet')
    return r.merge(tk, on='token', validate='many_to_one').sort_values(['token', 'k']).reset_index(drop=True)
Tt, Ts, Nn = load('stageT3_T_fold0_seed0', 'eval_dev'), load('stageT3_T_fold0_seed0', 'eval_dev_shuffle'), load('stageT3_none_fold0_seed0', 'eval_dev')
out = {}
out['pairing_ok'] = bool((Tt[['token', 'k']].values == Ts[['token', 'k']].values).all() and (Tt[['token', 'k']].values == Nn[['token', 'k']].values).all())
out['orig_identical'] = bool(all(Tt[c].equals(Ts[c]) and Tt[c].equals(Nn[c]) for c in Tt.columns if c.endswith('_orig') or c in ('valid', 'log')))
mets = ['nc', 'dac', 'ddc', 'ep', 'ttc', 'comfort']
v = Tt.valid.to_numpy(bool)
for df in (Tt, Ts, Nn): v &= np.all([np.isfinite(df[f'{m}_tau1']) for m in mets], 0)
out['n_valid'] = int(v.sum()); out['n_tokens'] = int(Tt.token.nunique())
pd_ = lambda df: (df.nc_tau1 * df.dac_tau1 * df.ddc_tau1 * (5 * df.ep_tau1 + 5 * df.ttc_tau1 + 2 * df.comfort_tau1) / 12).to_numpy()[v]
nt = lambda df: ((df.nc_tau1 < 1) | (df.ttc_tau1 < 1)).to_numpy()[v].astype(float)
dac = lambda df: (df.dac_tau1 < 1).to_numpy()[v].astype(float)
po = (Tt.nc_orig * Tt.dac_orig * Tt.ddc_orig * (5 * Tt.ep_orig + 5 * Tt.ttc_orig + 2 * Tt.comfort_orig) / 12).to_numpy()[v]
out['pdms'] = dict(orig=float(po.mean()), T=float(pd_(Tt).mean()), shuffled=float(pd_(Ts).mean()), none=float(pd_(Nn).mean()))
out['dac_fail_pct'] = dict(orig=float(100 * (Tt.dac_orig < 1).to_numpy()[v].mean()), shuffled=float(100 * dac(Ts).mean()))
logs = Tt.log.to_numpy()[v]; ul, inv = np.unique(logs, return_inverse=True); L = len(ul); cnt = np.bincount(inv).astype(float)
IDX = np.random.Generator(np.random.PCG64(123)).integers(0, L, size=(10000, L))
def boot(d):
    s = np.bincount(inv, weights=d, minlength=L); b = s[IDX].sum(1) / cnt[IDX].sum(1)
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out['PDMS_true_minus_shuf_points'] = boot(100 * (pd_(Tt) - pd_(Ts)))
out['PDMS_shuf_minus_none_points'] = boot(100 * (pd_(Ts) - pd_(Nn)))
out['PDMS_shuf_minus_orig_points'] = boot(100 * (pd_(Ts) - po))
out['NCTTC_true_minus_shuf_pp'] = boot(100 * (nt(Tt) - nt(Ts)))
out['NCTTC_shuf_minus_none_pp'] = boot(100 * (nt(Ts) - nt(Nn)))
out['P1_T_minus_none_dev'] = boot(100 * (pd_(Tt) - pd_(Nn)))
# shuffle map sanity
mp = json.load(open(f'{R}/stageT3_T_fold0_seed0/eval_dev_shuffle/teacher_shuffle_map.json'))
lg = dict(zip(Tt.token, Tt.log))
out['map'] = dict(n=len(mp), fixed_points=int(sum(k == s for k, s in mp.items())), is_permutation=bool(sorted(mp.values()) == sorted(mp.keys())),
                  covers_eval_tokens=bool(set(mp) == set(Tt.token)), same_log_frac=float(np.mean([lg[k] == lg[s] for k, s in mp.items()])))
out['predict_meta'] = json.load(open(f'{R}/stageT3_T_fold0_seed0/eval_dev_shuffle/predict_meta.json'))
print(json.dumps(out, indent=1))
