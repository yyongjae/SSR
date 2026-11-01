"""Re-cache N sampled new tokens with build_metric_cache's own worker (_init/run_chunk, SAVED_CFG) into a scratch
root and compare with the stored E2E caches via compare_pair (official scores of probe trajectories + all fields)."""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools/refiner"))
import build_metric_cache as BMC

HERE = Path(__file__).parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
if __name__ == "__main__":
    for k, v in BMC.ENV.items():
        os.environ.setdefault(k, v)
    assert BMC.sha256(BMC.SAVED_CFG) == BMC.SAVED_CFG_SHA256
    s = pd.read_parquet(HERE / "sample100.parquet").iloc[:N]
    out = HERE / "recache"
    out.mkdir(exist_ok=True)
    chunks = [(r.log, [r.token], str(out), str(BMC.TRAIN_LOGS)) for r in s.itertuples()]
    t0 = time.time()
    with Pool(4, initializer=BMC._init, initargs=(str(out), str(BMC.TRAIN_LOGS))) as p:
        res = list(p.imap_unordered(BMC.run_chunk, chunks))
    print("recached", sum(r[2] for r in res), "fail", sum(r[3] for r in res), f"{time.time()-t0:.0f}s", flush=True)
    sys.path.insert(0, str(BMC.CF_DIR))
    import cf_common as CF
    CF.init_worker()
    rows = []
    for r in s.itertuples():
        mo = BMC.load_mc(BMC.mc_path(BMC.OUT, r.log, r.token))
        mn = BMC.load_mc(BMC.mc_path(out, r.log, r.token))
        row = dict(token=r.token, log=r.log, **BMC.compare_pair(mo, mn))
        rows.append(row)
        print(r.token, row["max_score_diff"], row["max_field_diff"],
              {k: v for k, v in row.items() if k.startswith("pdms_")}, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "recache_rows.parquet", index=False)
    sm = dict(n=len(df), all_scores_identical=bool((df.max_score_diff == 0).all()),
              all_fields_identical=bool((df.max_field_diff == 0).all()),
              max_score_diff=float(df.max_score_diff.max()), max_field_diff=float(df.max_field_diff.max()),
              n_traj=int(sum(c.startswith("score_diff_") for c in df.columns) * len(df)),
              n_pdms_lt1=int(sum((df[c] < 1).sum() for c in df.columns if c.startswith("pdms_"))),
              field_max={c: float(df[c].max()) for c in df.columns if c.startswith("field_")})
    json.dump(sm, open(HERE / "recache_summary.json", "w"), indent=1)
    print(json.dumps(sm, indent=1))
