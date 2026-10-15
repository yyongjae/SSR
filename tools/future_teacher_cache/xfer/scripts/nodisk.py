import os
S = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{S}/needed.py").read().split("res = {}")[0])
for name, toks in sets.items():
    run = set()
    for t in toks:
        for f in (idx.get(t) or [])[:8]:
            if f is not None and f not in cached: run.add(f)
    sw = set()
    for f in run:
        log, i = tok2[f]
        for k in (1, 2):
            j = i - k
            if j >= 0 and LF[log][j][1] and LF[log][j][0] not in run: sw.add(LF[log][j][0])
    print(name, 'run', len(run), 'sweep-only lidar frames if nothing on disk', len(sw),
          'GB min inputs', round(len(run) * 2.0288e-3 + len(sw) * 1.4031e-3, 1), 'GB 8cam', round(len(run) * 3.0915e-3 + len(sw) * 1.4031e-3, 1))
