"""extract_human.py extract for split navtest: identical code path (extract_split / extract_token / pack), only the
log directory is TEST_LOGS (the tool's cmd_extract hard-codes TRAIN_LOGS)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/external-user/yongjae/SSR/tools/refiner")
import extract_human as EH
split = "navtest"
out = EH.OUT
t0 = time.time()
df = pd.read_parquet(EH.SPLITS / f"{split}.parquet", columns=["token", "log", "frame_idx"])
d = EH.extract_split(df, EH.TEST_LOGS, 2)
assert (d["tokens"] == df.token.to_numpy()).all()
fi_ok = bool((d["frame_idx"].astype(np.int64) == df.frame_idx.to_numpy()).all())
meta = dict(split=split, n=len(df), created=time.strftime("%F %T"), logs_dir=str(EH.TEST_LOGS),
            frame_idx_matches_split=fi_ok, frame_gap=int(d["frame_gap"].sum()), gap4=int(d["gap4"].sum()),
            n_avail_ge16=int((d["n_avail"] >= 16).sum()), n_reg_ge16=int((d["n_reg"] >= 16).sum()),
            seconds=time.time() - t0)
d["meta_json"] = np.asarray(json.dumps(meta))
if (out / f"{split}.npz").exists():
    raise SystemExit("exists")
tmp = out / f".{split}.tmp.npz"
np.savez(tmp, **d)
tmp.replace(out / f"{split}.npz")
print(json.dumps(meta), flush=True)
with open(out / "logs" / "extract.log", "a") as f:
    f.write(json.dumps(meta) + "\n")
