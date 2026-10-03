"""Objects vs metric cache on the 100 sampled new tokens: read-only (process_log with existing npz, no index write)."""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/home/external-user/yongjae/SSR/tools/refiner")
import build_future_objects as BF

HERE = Path(__file__).parent
s = pd.read_parquet(HERE / "sample100.parquet")
cfg = dict(out="/home/external-user/ssd/yongjae_refiner/objects/navtrain",
           logs="/home/external-user/yongjae/SSR/data/dataset/navsim_logs/trainval",
           overwrite=False, mc_root="/home/external-user/ssd/yongjae_refiner/metric_cache")
tasks = [(lg, [(r.token, int(r.frame_idx)) for r in g.itertuples()], cfg) for lg, g in s.groupby("log")]
t0 = time.time()
rows = []
with Pool(4, initializer=BF._init) as p:
    for r in p.imap_unordered(BF.process_log, tasks):
        rows += r
df = pd.DataFrame(rows)
print(df.status.value_counts().to_dict())
df.to_parquet(HERE / "objects_rows.parquet", index=False)
sm = BF.summarize(df)
sm["wall_s"] = time.time() - t0
sm["status_counts"] = df.status.value_counts().to_dict()
sm["frame_idx_mismatch"] = int(df.frame_idx_mismatch.sum())
sm["mc_missing"] = int(df.mc_missing.sum())
json.dump(sm, open(HERE / "objects_summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in sm.items() if k != "unknown"}, indent=1, default=float))
