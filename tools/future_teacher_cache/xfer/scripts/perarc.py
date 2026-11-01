import json, pickle, glob, os, numpy as np, pandas as pd
S = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{S}/needed.py").read().split("res = {}")[0])  # reuse defs
fr = {i: sum(len(LF[l]) for l in amap[f'openscene_sensor_trainval_{i}']) for i in range(200)}
out = {}
for name, toks in sets.items():
    need = set()
    for t in toks:
        for f in (idx.get(t) or [])[:8]:
            if f is not None and f not in cached: need.add(f)
    per = np.zeros(200, int)
    for f in need:
        log, i = tok2[f]
        if not (cams_ok(log, i) and lidar_ok(log, i)): per[log2arc[log]] += 1
    frac = per / np.array([fr[i] for i in range(200)])
    out[name] = dict(dl=int(per.sum()), per_arc_min=int(per.min()), per_arc_med=float(np.median(per)), per_arc_max=int(per.max()),
                     arcs_with_zero=int((per == 0).sum()), frac_min=float(frac.min()), frac_med=float(np.median(frac)), frac_max=float(frac.max()),
                     max_arc_needed_GB_min_inputs=float(per.max() * 2.023e-3), max_arc_needed_GB_all9=float(per.max() * 3.116e-3),
                     overall_frac=float(per.sum() / sum(fr.values())))
    print(name, out[name])
# logs per set and logs not in navtrain
nav_logs = {tok2[t][0] for t in idx}
print('navtrain logs', len(nav_logs), 'archives with any non-navtrain log', sum(1 for i in range(200) if set(amap[f'openscene_sensor_trainval_{i}']) - nav_logs))
json.dump(out, open(f"{S}/perarc_result.json", "w"), indent=1)
