"""Independent recompute of navtest stage-T endpoints (does NOT import stageT_decision.py).
Reads report_rows.parquet + tokens.parquet of both frozen run-3 arms, cross-checks against the raw score parquets."""
import json, sys, numpy as np, pandas as pd
R = '/home/external-user/ssd/yongjae_refiner'
EV = sys.argv[1] if len(sys.argv) > 1 else 'eval_navtest'
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'navtest'
out = {}
arms = {}
for a in ['T', 'none']:
    d = f'{R}/runs/stageT3_{a}_fold0_seed0/{EV}/'
    r = pd.read_parquet(d + 'report_rows.parquet'); tk = pd.read_parquet(d + 'tokens.parquet')
    r = r.merge(tk, on='token', how='left', validate='many_to_one')
    arms[a] = r
T, N = arms['T'], arms['none']
# ---- pairing checks
key = ['token', 'k']
assert (T[key].values == N[key].values).all(), 'row order differs'
orig_cols = [c for c in T.columns if c.endswith('_orig')] + ['valid', 'family', 'log']
out['orig_identical_across_arms'] = bool(all(T[c].equals(N[c]) for c in orig_cols))
# ---- cross-check orig vs scores/<split>.parquet and tau1 vs scores_tau1.parquet
S = pd.read_parquet(f'{R}/scores/{SPLIT}.parquet')
m = T[key + ['nc_orig', 'dac_orig', 'ddc_orig', 'ep_orig', 'ttc_orig', 'comfort_orig', 'pdms_orig', 'log']].merge(
    S[key + ['nc', 'dac', 'ddc', 'ep', 'ttc', 'comfort', 'pdms', 'log', 'error', 'rec_ok']], on=key, how='left', validate='one_to_one')
out['orig_vs_scores_parquet_maxabs'] = float(max(np.nanmax(np.abs(m[f'{c}_orig'] - m[c])) for c in ['nc', 'dac', 'ddc', 'ep', 'ttc', 'comfort', 'pdms']))
out['orig_missing_in_scores'] = int(m.nc.isna().sum())
out['log_mismatch_scores_vs_tokens'] = int((m.log_x != m.log_y).sum())
out['scores_errors'] = int((S.error.fillna('') != '').sum()); out['scores_rec_not_ok'] = int((~S.rec_ok.astype(bool)).sum())
for a, df in arms.items():
    t1 = pd.read_parquet(f'{R}/runs/stageT3_{a}_fold0_seed0/{EV}/scores_tau1.parquet')
    mm = df[key + [f'{c}_tau1' for c in ['nc', 'dac', 'ddc', 'ep', 'ttc', 'comfort', 'pdms']]].merge(t1[key + ['nc', 'dac', 'ddc', 'ep', 'ttc', 'comfort', 'pdms', 'error']], on=key, how='left')
    out[f'tau1_vs_scores_tau1_maxabs_{a}'] = float(max(np.nanmax(np.abs(mm[f'{c}_tau1'] - mm[c])) for c in ['nc', 'dac', 'ddc', 'ep', 'ttc', 'comfort', 'pdms']))
    out[f'tau1_missing_{a}'] = int(mm.nc.isna().sum()); out[f'tau1_errors_{a}'] = int((t1.error.fillna('') != '').sum())
    # PDMS formula check
    for s in ['orig', 'tau1']:
        f = df[f'nc_{s}'] * df[f'dac_{s}'] * df[f'ddc_{s}'] * (5 * df[f'ep_{s}'] + 5 * df[f'ttc_{s}'] + 2 * df[f'comfort_{s}']) / 12
        out[f'pdms_formula_maxabs_{a}_{s}'] = float(np.nanmax(np.abs(f - df[f'pdms_{s}'])))
# ---- valid mask: valid & all 14 metrics finite in both arms
mets = ['nc', 'dac', 'ddc', 'ep', 'ttc', 'comfort']
def fin(df):
    return np.all([np.isfinite(df[f'{c}_{s}'].to_numpy(float)) for c in mets for s in ['orig', 'tau1']], axis=0)
valid = T.valid.to_numpy(bool) & N.valid.to_numpy(bool) & fin(T) & fin(N)
out['n_rows'] = len(T); out['n_valid_paired'] = int(valid.sum()); out['n_valid_flag'] = int(T.valid.sum())
out['n_invalid_but_finite'] = int((~T.valid.to_numpy(bool) & fin(T)).sum())
# theta = 0 => final = tau1 for every valid draft (p_g >= 0 always)
out['p_g_min'] = {a: float(df.p_g[valid].min()) for a, df in arms.items()}
logs = T.log.to_numpy()[valid]
def P(df, s):  # recompute pdms from components (formula), final = tau1
    return (df[f'nc_{s}'] * df[f'dac_{s}'] * df[f'ddc_{s}'] * (5 * df[f'ep_{s}'] + 5 * df[f'ttc_{s}'] + 2 * df[f'comfort_{s}']) / 12).to_numpy(float)[valid]
def fail(df, s, c): return (df[f'{c}_{s}'].to_numpy(float)[valid] < 1)
def fail_any(df, s): return fail(df, s, 'nc') | fail(df, s, 'dac') | fail(df, s, 'ddc') | fail(df, s, 'ttc') | fail(df, s, 'comfort')
pT, pN, pO = P(T, 'tau1'), P(N, 'tau1'), P(T, 'orig')
ncttc = lambda df, s: fail(df, s, 'nc') | fail(df, s, 'ttc')
newf = lambda df: (~fail_any(df, 'orig')) & fail_any(df, 'tau1')
diffs = {
    'P1_d_pdms_points': 100 * (pT - pN),
    'P2_ncttc_reduction_pp': 100 * (ncttc(N, 'tau1').astype(float) - ncttc(T, 'tau1')),
    'P2ni_dac_excess_pp': 100 * (fail(T, 'tau1', 'dac').astype(float) - fail(N, 'tau1', 'dac')),
    'P2ni_ddc_excess_pp': 100 * (fail(T, 'tau1', 'ddc').astype(float) - fail(N, 'tau1', 'ddc')),
    'P3_new_fail_excess_pp': 100 * (newf(T).astype(float) - newf(N)),
    'sanity_T_vs_orig_points': 100 * (pT - pO),
    'sanity_none_vs_orig_points': 100 * (pN - pO),
}
# independent bootstrap: per-log sums, resample log indices with replacement (different RNG path than the decision script)
ul, inv = np.unique(logs, return_inverse=True); L = len(ul)
cnt = np.bincount(inv, minlength=L).astype(float)
rng = np.random.Generator(np.random.PCG64(20260929))
IDX = rng.integers(0, L, size=(10000, L))
res = {}
for k, v in diffs.items():
    sm = np.bincount(inv, weights=v, minlength=L)
    bs = sm[IDX].sum(1) / cnt[IDX].sum(1)
    res[k] = dict(mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
out['endpoints'] = res; out['n_logs'] = int(L)
# EP loss on drafts passing NC, DAC, DDC before and after (budget def 'passing'), per arm
for a, df in arms.items():
    core = lambda s: ~(fail(df, s, 'nc') | fail(df, s, 'dac') | fail(df, s, 'ddc'))
    pb = core('orig') & core('tau1')
    ep0 = df.ep_orig.to_numpy(float)[valid]; ep1 = df.ep_tau1.to_numpy(float)[valid]
    out[f'ep_loss_passing_points_{a}'] = float(100 * (ep0 - ep1)[pb].mean()); out[f'n_passing_{a}'] = int(pb.sum())
    out[f'pdms_{a}'] = float(P(df, 'tau1').mean())
    out[f'rates_{a}'] = {c: float(100 * fail(df, 'tau1', c).mean()) for c in ['nc', 'ttc', 'dac', 'ddc', 'comfort']}
    out[f'rates_{a}']['ncttc'] = float(100 * ncttc(df, 'tau1').mean())
out['pdms_orig'] = float(pO.mean())
out['rates_orig'] = {c: float(100 * fail(T, 'orig', c).mean()) for c in ['nc', 'ttc', 'dac', 'ddc', 'comfort']}
out['rates_orig']['ncttc'] = float(100 * ncttc(T, 'orig').mean())
# fixed/new counts
o = ncttc(T, 'orig'); t = ncttc(T, 'tau1'); n = ncttc(N, 'tau1')
out['ncttc_counts'] = dict(orig_fail=int(o.sum()), fixed_T=int((o & ~t).sum()), fixed_none=int((o & ~n).sum()),
                           new_T=int((~o & t).sum()), new_none=int((~o & n).sum()))
out['new_any_counts'] = dict(T=int(newf(T).sum()), none=int(newf(N).sum()))
# city split (P1, P2) with the same bootstrap
if SPLIT == 'navtest':
    sp = pd.read_parquet(f'{R}/splits/navtest.parquet')[['token', 'city', 'log']]
    cc = T[['token']].merge(sp, on='token', how='left').city.to_numpy()[valid]
    out['city_token_log_consistent'] = bool((T[['token']].merge(sp, on='token', how='left').log.to_numpy() == T.log.to_numpy()).all())
    cr = {}
    for city in list(np.unique(cc)) + ['non_lv']:
        msk = (cc != 'us-nv-las-vegas-strip') if city == 'non_lv' else (cc == city)
        li = np.unique(inv[msk]); sub = {}
        for k in ['P1_d_pdms_points', 'P2_ncttc_reduction_pp']:
            v = diffs[k][msk]; ii = inv[msk]; _, ii2 = np.unique(ii, return_inverse=True); Lc = ii2.max() + 1
            sm = np.bincount(ii2, weights=v, minlength=Lc); cn = np.bincount(ii2, minlength=Lc).astype(float)
            IX = rng.integers(0, Lc, size=(10000, Lc)); bs = sm[IX].sum(1) / cn[IX].sum(1)
            sub[k] = [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        sub['n_tokens'] = int(len(np.unique(T.token.to_numpy()[valid][msk]))); sub['n_logs'] = int(len(li))
        cr[str(city)] = sub
    out['city'] = cr
print(json.dumps(out, indent=1))
