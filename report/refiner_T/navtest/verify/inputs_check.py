"""Independent spot-checks of navtest inputs (20 tokens): human traj vs raw test log (own transform, no navsim code),
t0 vs metric cache, objects npz corners vs official metric-cache occupancy polygons, identity draft == human."""
import json, lzma, pickle, glob, sys, numpy as np, pandas as pd
R = '/home/external-user/ssd/yongjae_refiner'; MC = '/home/external-user/yongjae/SSR/data/exp/metric_cache'
RAW = '/home/external-user/navsim/download/test_navsim_logs/test'
sys.path.insert(0, '/home/external-user/yongjae/SSR')
H = np.load(f'{R}/human/navtest.npz', allow_pickle=True)
hidx = {t: i for i, t in enumerate(H['tokens'])}
rows = pd.read_parquet(f'{R}/runs/stageT3_T_fold0_seed0/eval_navtest/tokens.parquet')
rng = np.random.default_rng(11); pick = rng.choice(len(rows), 20, replace=False)
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
def pose(fr):  # NAVSIM convention: translation + pyquaternion yaw_pitch_roll[0] = atan2(2(wz - xy), 1 - 2(y^2 + z^2))
    w, x, y, z = fr['ego2global_rotation']; n = np.sqrt(w*w + x*x + y*y + z*z); w, x, y, z = w/n, x/n, y/n, z/n
    return fr['ego2global_translation'][0], fr['ego2global_translation'][1], np.arctan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z))
def to_local(x, y, h, o):
    c, s = np.cos(o[2]), np.sin(o[2]); dx, dy = x - o[0], y - o[1]
    return c * dx + s * dy, -s * dx + c * dy, wrap(h - o[2])
logcache = {}; out = []
for j in pick:
    t, lg = rows.token.iloc[j], rows.log.iloc[j]
    if lg not in logcache: logcache = {lg: pickle.load(open(f'{RAW}/{lg}.pkl', 'rb'))}
    L = logcache[lg]; fi = [f['token'] for f in L].index(t); o = pose(L[fi])
    fut = np.array([to_local(*pose(L[fi + k]), o) for k in range(1, 9)])
    dts = np.diff([L[fi + k]['timestamp'] for k in range(0, 9)]) / 1e6
    i = hidx[t]; ht = H['traj'][i]
    r = dict(token=t, dt_max=float(dts.max()), human_xy_err=float(np.abs(fut[:, :2] - ht[:, :2]).max()),
             human_h_err=float(np.abs(wrap(fut[:, 2] - ht[:, 2])).max()))
    mc = pickle.load(lzma.open(glob.glob(f'{MC}/{lg}/*/{t}/metric_cache.pkl')[0], 'rb'))
    ra = mc.ego_state.rear_axle
    r['t0_raw_vs_mc'] = float(max(abs(ra.x - o[0]), abs(ra.y - o[1]), abs(wrap(ra.heading - o[2]))))
    r['v0_err'] = float(abs(mc.ego_state.dynamic_car_state.speed - H['v0'][i]))
    r['cmd_raw_vs_npz'] = bool(int(np.argmax(L[fi]['driving_command'])) == int(H['cmd'][i]))
    d = np.load(f'{R}/drafts/navtest/{t}.npz'); r['identity_eq_human'] = bool(np.array_equal(d['drafts'][0], ht)) and int(d['family'][0]) == 0
    r['cfg_hash'] = str(d['cfg_hash'])
    # objects vs metric cache polygons at keyframes k=0..10 (dense index 5k)
    ob = np.load(f'{R}/objects/navtest/{t}.npz'); tr = list(ob['track']); kf = ob['kf']; first = ob['first']
    o_mc = (ra.x, ra.y, ra.heading); obs = mc.observation; red = obs._red_light_token
    cerr, n_cmp, miss_npz, miss_mc = 0.0, 0, 0, 0
    for k in range(11):
        om = obs._occupancy_maps[obs._global_to_local_idcs[5 * k]]
        mc_tok = {tk: g for tk, g in zip(om._tokens, om._geometries) if red not in tk}
        for a, tk in enumerate(tr):
            single = bool(ob['meta'][a, 4]); kk = int(ob['meta'][a, 2]) if single else k
            present = (kf[a, k, 5] > 0.5) or single
            if present and tk not in mc_tok: miss_mc += 1
            if not present and tk in mc_tok: miss_npz += 1
            if present and tk in mc_tok:
                xy = np.array(mc_tok[tk].exterior.coords)[:4]
                lx, ly, _ = to_local(xy[:, 0], xy[:, 1], 0 * xy[:, 0], o_mc)
                P = np.stack([lx, ly], 1)
                cx, cy, hh = kf[a, kk, 0], kf[a, kk, 1], kf[a, kk, 2]; Lh, Wh = first[a, 0] / 2, first[a, 1] / 2
                c, s = np.cos(hh), np.sin(hh)
                Q = np.array([[cx + c * u * Lh - s * v * Wh, cy + s * u * Lh + c * v * Wh] for u, v in [(1, 1), (1, -1), (-1, -1), (-1, 1)]])
                e = np.linalg.norm(P[:, None] - Q[None], axis=2).min(1).max(); cerr = max(cerr, float(e)); n_cmp += 1
        # mc tracks near ego (within R-10 m) that are not in the npz at all
        for tk, g in mc_tok.items():
            if tk not in tr:
                c0 = g.centroid; lx, ly, _ = to_local(c0.x, c0.y, 0, o_mc)
                if np.hypot(lx, ly) < float(ob['R']) - 10: miss_npz += 1
    r.update(obj_corner_err=cerr, obj_boxes_compared=n_cmp, obj_present_npz_absent_mc=miss_mc, obj_present_mc_absent_npz=miss_npz)
    out.append(r)
df = pd.DataFrame(out)
summ = dict(n=len(df), max_dt=float(df.dt_max.max()), human_xy_err_max=float(df.human_xy_err.max()), human_h_err_max=float(df.human_h_err.max()),
            t0_raw_vs_mc_max=float(df.t0_raw_vs_mc.max()), v0_err_max=float(df.v0_err.max()), cmd_all_eq=bool(df.cmd_raw_vs_npz.all()),
            identity_eq_human_all=bool(df.identity_eq_human.all()), cfg_hashes=sorted(set(df.cfg_hash)),
            obj_corner_err_max=float(df.obj_corner_err.max()), obj_boxes_compared=int(df.obj_boxes_compared.sum()),
            obj_presence_mismatch=int(df.obj_present_npz_absent_mc.sum() + df.obj_present_mc_absent_npz.sum()))
print(json.dumps(dict(summary=summ, rows=out), indent=1, default=float))
