"""navtest token list for the stage-T frozen evaluation: every token with an official navtest metric cache AND a
teacher cache_val_50x100 entry. Columns like splits/dev.parquet (token, log, frame_idx, map_location, part, split, fold,
e_cached, order) + city (= map_location)."""
import json, os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/external-user/yongjae/SSR")
MC = Path("/home/external-user/yongjae/SSR/data/exp/metric_cache")
TC = Path("/home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100")
LOGS = Path("/home/external-user/navsim/download/test_navsim_logs/test")
TABLE = Path("/home/external-user/yongjae/SSR/report/head_ablation_scenes/table.npz")
OUT = Path("/home/external-user/ssd/yongjae_refiner/splits/navtest.parquet")
rows = []
for lg in sorted(os.listdir(MC)):
    d = MC / lg
    if lg == "metadata" or not d.is_dir():
        continue
    for st in os.listdir(d):
        for tk in os.listdir(d / st):
            if (d / st / tk / "metric_cache.pkl").is_file():
                rows.append((tk, lg, st))
mc = pd.DataFrame(rows, columns=["token", "log", "scenario_type"])
csv = pd.read_csv(MC / "metadata/metric_cache_metadata_node_0.csv")
man = json.loads((TC / "manifest.json").read_text())
mc["teacher"] = [(TC / "samples" / t[:2] / f"{t}.npz").is_file() for t in mc.token]
T = np.load(TABLE, allow_pickle=True)
tab = pd.DataFrame(dict(token=T["tokens"], log_tab=T["logs"], city_tab=T["cities"]))
info = dict(n_mc=len(mc), n_mc_csv=len(csv), mc_dup_tokens=int(mc.token.duplicated().sum()),
            n_teacher=int(mc.teacher.sum()), teacher_manifest_written=man.get("num_samples_written"),
            teacher_sha=man.get("checkpoint_sha256_head"), scenario_types=mc.scenario_type.value_counts().to_dict())
keep = mc[mc.teacher].copy()
# frame_idx / map_location from the raw test logs (frame dict fields); position check
fi, city, missing_log = {}, {}, []
for lg, g in keep.groupby("log"):
    p = LOGS / f"{lg}.pkl"
    if not p.exists():
        missing_log.append(lg); continue
    fr = pickle.load(open(p, "rb"))
    pos = {f["token"]: i for i, f in enumerate(fr)}
    for t in g.token:
        if t in pos:
            fi[t] = int(fr[pos[t]]["frame_idx"]); city[t] = str(fr[pos[t]]["map_location"])
keep["frame_idx"] = keep.token.map(fi)
keep["map_location"] = keep.token.map(city)
info.update(n_teacher_and_mc=len(keep), missing_logs=missing_log, n_not_in_log=int(keep.frame_idx.isna().sum()))
keep = keep[keep.frame_idx.notna()].copy()
m = keep.merge(tab, on="token", how="left")
info.update(n_in_head_ablation_table=int(m.log_tab.notna().sum()), table_n=len(tab),
            log_mismatch_vs_table=int((m.log_tab.notna() & (m.log_tab != m.log)).sum()),
            city_mismatch_vs_table=int((m.city_tab.notna() & (m.city_tab != m.map_location)).sum()),
            table_tokens_not_kept=int((~tab.token.isin(keep.token)).sum()))
keep = keep.sort_values(["log", "frame_idx", "token"]).reset_index(drop=True)
df = pd.DataFrame(dict(token=keep.token.astype(str), log=keep.log.astype(str), frame_idx=keep.frame_idx.astype(np.int64),
                       map_location=keep.map_location.astype(str), part="navtest", split="navtest",
                       fold=np.int8(-1), e_cached=False, order=np.nan, city=keep.map_location.astype(str)))
df["fold"] = df.fold.astype(np.int8)
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    raise SystemExit(f"{OUT} exists")
df.to_parquet(OUT, index=False)
info.update(n=len(df), n_logs=int(df.log.nunique()), city=df.city.value_counts().to_dict(), created=time.strftime("%F %T"))
Path(str(OUT).replace(".parquet", "_summary.json")).write_text(json.dumps(info, indent=1, default=str))
print(json.dumps(info, indent=1, default=str))
