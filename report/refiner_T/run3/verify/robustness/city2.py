import sys; sys.path.insert(0, ".")
import numpy as np
from robust import rows, paired, ends, fmt, FAM
inv = {v: k for k, v in FAM.items()}
for ev in ("eval_dev", "eval_train_fold0"):
    m = paired(rows("stageT3", "T", ev), rows("stageT3", "none", ev), 0.0, 0.0); nv = m[m.city != "us-nv-las-vegas-strip"]
    print("##", ev)
    for lab, fs in (("nonVegas lconst", ["lconst"]), ("nonVegas cv+lconst", ["cv", "lconst"]),
                    ("nonVegas excl cv,lat,combined", [f for f in FAM.values() if f not in ("cv", "lat", "combined")]),
                    ("nonVegas lat+combined", ["lat", "combined"])):
        print(f"  {lab:32s}", fmt(ends(nv[nv.family.isin([inv[f] for f in fs])], 2000)))
    fr = lambda s, a: 100 * (s.fail_ncttc_orig_T.sum() - s[f"fail_ncttc_{a}"].sum()) / s.fail_ncttc_orig_T.sum()
    for v, s in (("Vegas", m[m.city == "us-nv-las-vegas-strip"]), ("nonVegas", nv)):
        print(f"  {v}: net share of orig NC|TTC failures removed  T {fr(s,'T'):.1f}%  none {fr(s,'none'):.1f}%")
    n = m.groupby("log").size(); print("  drafts/log quantiles", n.quantile([0, .5, .9, 1]).to_dict())
