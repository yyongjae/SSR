import sys, json; sys.path.insert(0, "."); sys.argv=["x"]
import numpy as np, pandas as pd
from robust import rows, paired, ends, fmt, FAM
res = {}
def interact(m, nb=10000, seed=1):
    """bootstrap (stratified by group, logs resampled within group) of P2(Vegas) - P2(non-Vegas)."""
    rng = np.random.default_rng(seed); out = {}
    for name, col in (("P2", None), ("P1", None)):
        d = (m.fail_ncttc_none - m.fail_ncttc_T) if name == "P2" else (m.pdms_final_T - m.pdms_final_none)
        g = m.assign(d=d.values).groupby(["vegas", "log"]).d.agg(["sum", "count"]).reset_index()
        st = {}
        for v in (True, False):
            gg = g[g.vegas == v]; s, n = gg["sum"].to_numpy(), gg["count"].to_numpy(float); L = len(gg)
            c = rng.multinomial(L, np.full(L, 1 / L), size=nb).astype(float)
            st[v] = (c @ s) / (c @ n)
        diff = 100 * (st[True] - st[False])
        out[name] = dict(mean=float(100 * (m[m.vegas].pipe(lambda x: 0) if False else 0)), lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)))
    return out
for tag in ("stageT", "stageT2", "stageT3"):
    for ev in ("eval_dev", "eval_train_fold0"):
        m = paired(rows(tag, "T", ev), rows(tag, "none", ev), 0.0, 0.0); m["vegas"] = m.city == "us-nv-las-vegas-strip"
        print(f"\n### {tag}/{ev}")
        for v in (True, False):
            s = m[m.vegas == v]
            print(f"  vegas={v!s:5s} logs={s.log.nunique()} n={len(s)} orig ncttc {100*s.fail_ncttc_orig_T.mean():.2f}% "
                  f"T {100*s.fail_ncttc_T.mean():.2f} none {100*s.fail_ncttc_none.mean():.2f} | {fmt(ends(s, 2000))}")
        it = interact(m); print("  Vegas-minus-rest CI: P2", {k: round(v, 3) for k, v in it['P2'].items() if k != 'mean'}, "P1", {k: round(v, 3) for k, v in it['P1'].items() if k != 'mean'})
        if tag == "stageT3":
            tab = m.assign(d2=100*(m.fail_ncttc_none - m.fail_ncttc_T), o=100*m.fail_ncttc_orig_T, d1=100*(m.pdms_final_T-m.pdms_final_none)).groupby(["family", "vegas"]).agg(n=("d2", "size"), orig_fail=("o", "mean"), P2=("d2", "mean"), P1=("d1", "mean")).round(2)
            tab.index = tab.index.set_levels([[FAM[i] for i in tab.index.levels[0]], tab.index.levels[1]]); print(tab.to_string())
            # contribution of Vegas lconst to total P2 net count
            cnt = m.assign(d2=m.fail_ncttc_none - m.fail_ncttc_T).groupby(["vegas", "family"]).d2.sum()
            cnt.index = [(v, FAM[f]) for v, f in cnt.index]; print("  P2 net draft counts:", cnt.to_dict(), "total", cnt.sum())
