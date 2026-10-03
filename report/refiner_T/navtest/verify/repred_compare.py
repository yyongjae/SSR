"""Re-predict the first 96 navtest tokens with each frozen ckpt_best (eval_refiner.py predict --limit 96 --out scratchpad)
and compare with the published eval_navtest/pred.npz rows; also load ckpt_best and compare its config to config.json."""
import json, sys, numpy as np, torch
R = '/home/external-user/ssd/yongjae_refiner/runs'; SP = '/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad/repred'
out = {}
for a in ['T', 'none']:
    A = np.load(f'{SP}/{a}/pred.npz'); B = np.load(f'{R}/stageT3_{a}_fold0_seed0/eval_navtest/pred.npz')
    n = len(A['tokens']); idx = {t: i for i, t in enumerate(B['tokens'])}; j = np.array([idx[t] for t in A['tokens']])
    o = {k: float(np.nanmax(np.abs(A[k].astype(np.float64) - B[k][j].astype(np.float64)))) for k in ['tau0', 'tau1', 'p_g', 'z_lon', 'w_lat']}
    o['n_tokens'] = int(n); o['draft_valid_eq'] = bool(np.array_equal(A['draft_valid'], B['draft_valid'][j]))
    ck = torch.load(f'{R}/stageT3_{a}_fold0_seed0/ckpt_best.pt', map_location='cpu')
    o['ckpt_keys'] = [k for k in ck.keys()] if isinstance(ck, dict) else str(type(ck))
    for k in ('epoch', 'step', 'val_loss', 'best_val_loss'):
        if isinstance(ck, dict) and k in ck: o[f'ckpt_{k}'] = float(ck[k]) if not isinstance(ck[k], dict) else str(ck[k])[:80]
    cfg = json.load(open(f'{R}/stageT3_{a}_fold0_seed0/config.json'))
    if isinstance(ck, dict) and 'config' in ck:
        c2 = ck['config']; o['ckpt_config_diff_keys'] = sorted(k for k in set(cfg) | set(c2) if cfg.get(k) != c2.get(k))
    out[a] = o
print(json.dumps(out, indent=1, default=str))
