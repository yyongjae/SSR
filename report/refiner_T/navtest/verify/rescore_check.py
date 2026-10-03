"""Re-score navtest drafts with the OFFICIAL single-trajectory navsim.evaluate.pdm_score.pdm_score (plain navsim
simulator/scorer instantiated from default_scoring_parameters.yaml), metric cache from the official navtest cache dir."""
import json, lzma, pickle, glob, os, sys, numpy as np, pandas as pd
ROOT = '/home/external-user/yongjae/SSR'; sys.path.insert(0, ROOT)
os.environ.setdefault("NUPLAN_MAPS_ROOT", ROOT + "/data/dataset/maps"); os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.evaluate.pdm_score import pdm_score
from navsim.common.dataclasses import Trajectory
R = '/home/external-user/ssd/yongjae_refiner'
cfg = OmegaConf.load(ROOT + '/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml')
sim = instantiate(cfg.simulator); scorer = instantiate(cfg.scorer)
S = pd.read_parquet(f'{R}/scores/navtest.parquet')
rng = np.random.default_rng(5)
failing = S[(S.nc < 1) | (S.dac < 1) | (S.ttc < 1) | (S.ddc < 1)]
pick = pd.concat([failing.sample(25, random_state=5), S.drop(failing.index).sample(25, random_state=5)])
jobs = [('orig', r.token, r.log, int(r.k), None) for r in pick.itertuples()]
for arm, n in [('T', 15), ('none', 15)]:
    rr = pd.read_parquet(f'{R}/runs/stageT3_{arm}_fold0_seed0/eval_navtest/report_rows.parquet')
    rr = rr[rr.valid & ((rr.nc_orig < 1) | (rr.ttc_orig < 1) | (rr.dac_orig < 1))].sample(n, random_state=7)
    lg = pd.read_parquet(f'{R}/runs/stageT3_{arm}_fold0_seed0/eval_navtest/tokens.parquet').set_index('token').log
    for r in rr.itertuples(): jobs.append((arm, r.token, lg[r.token], int(r.k), None))
ref = {a: np.load(f'{R}/runs/stageT3_{a}_fold0_seed0/eval_navtest/refined.npz') for a in ['T', 'none']}
refidx = {a: {t: i for i, t in enumerate(ref[a]['tokens'])} for a in ref}
refdr = {a: ref[a]['drafts'] for a in ref}
tau1 = {a: pd.read_parquet(f'{R}/runs/stageT3_{a}_fold0_seed0/eval_navtest/scores_tau1.parquet').set_index(['token', 'k']) for a in ['T', 'none']}
Si = S.set_index(['token', 'k'])
rows = []
for kind, t, lg, k, _ in jobs:
    mc = pickle.load(lzma.open(glob.glob(f'{ROOT}/data/exp/metric_cache/{lg}/*/{t}/metric_cache.pkl')[0], 'rb'))
    if kind == 'orig':
        tr = np.load(f'{R}/drafts/navtest/{t}.npz', allow_pickle=True)['drafts'][k]; lab = Si.loc[(t, k)]
    else:
        tr = refdr[kind][refidx[kind][t], k]; lab = tau1[kind].loc[(t, k)]
    res = pdm_score(mc, Trajectory(np.asarray(tr, np.float32)), sim.proposal_sampling, sim, scorer)
    off = dict(nc=res.no_at_fault_collisions, dac=res.drivable_area_compliance, ddc=res.driving_direction_compliance,
               ep=res.ego_progress, ttc=res.time_to_collision_within_bound, comfort=res.comfort, pdms=res.score)
    rows.append(dict(kind=kind, token=t, k=k, **{f'd_{m}': float(abs(off[m] - lab[m])) for m in off}, pdms_label=float(lab['pdms'])))
df = pd.DataFrame(rows); dcols = [c for c in df.columns if c.startswith('d_')]
print(json.dumps(dict(n=len(df), by_kind=df.kind.value_counts().to_dict(), maxabs=df[dcols].max().to_dict(),
      n_any_mismatch_gt_1e6=int((df[dcols].max(1) > 1e-6).sum()), n_failing_label=int((df.pdms_label < 1e-9).sum())), indent=1))
df.to_csv('rescore_check_rows.csv', index=False)
