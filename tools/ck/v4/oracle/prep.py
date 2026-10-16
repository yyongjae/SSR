#!/usr/bin/env python
"""refiner v4_3 Phase 0 oracle -- step 0: inputs for the workers (CPU, read-only on every input).

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/v4/oracle/prep.py

Writes OUT/prep (OUT = CK/v4/analysis/range_oracle):
  f1_{rows,tokens,fin,exec,plan,sim,init}.npy   the F1 val run.npz members as plain .npy (np.load cannot mmap an npz
                                                member; 28 workers would each hold the 0.66 GB sim array otherwise);
                                                byte copies, checked equal after writing.
  tracks.parquet   row, col (0..95: par k / var 16 + 5 k + v - 1), nc, ttc, nc_track, nc_obj_type, ttc_track,
                   nc_time_idx, ttc_time_idx of every official NC / TTC failure among the F1 tokens' 96 candidates,
                   from the official label shards (labels/navtrain_val/cand, r34_variants/labels); used to put a field
                   miss whose official object is absent from the GT store into the UNKNOWN bucket.
  prep.json        provenance (sources, counts, KST time).
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_common as OC  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def kst() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime(time.time() + 9 * 3600))


def main():
    OC.PREP.mkdir(parents=True, exist_ok=True)
    z = np.load(OC.F1_DIR / "run.npz")
    groups = json.loads(str(z["groups"]))
    assert groups == [["par", 0, 16], ["var", 16, 80], ["latT", 96, 16]], groups
    rep = {"src": str(OC.F1_DIR / "run.npz"), "groups": groups, "started": kst()}
    for k in ("rows", "tokens", "fin", "exec", "plan", "sim", "init"):
        a = z[k]
        p = OC.PREP / f"f1_{k}.npy"
        if not p.is_file():
            np.save(p, a)
        b = np.load(p, mmap_mode="r")
        assert b.shape == a.shape and b.dtype == a.dtype and np.array_equal(np.asarray(b), a), k
        rep[k] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    rows = set(int(r) for r in z["rows"])
    parts = []
    for grp, d in OC.LAB_SHARDS.items():
        for f in sorted(glob.glob(str(d / "c*.parquet"))):
            t = pd.read_parquet(f, columns=["row", "k", "nc", "ttc", "nc_track", "nc_obj_type", "ttc_track",
                                            "nc_time_idx", "ttc_time_idx"])
            t = t[t["row"].isin(rows) & ((t["nc"] < 1) | (t["ttc"] < 1))]
            if len(t):
                t = t.assign(col=t["k"].astype(int) + (0 if grp == "par" else OC.N_PAR))
                parts.append(t)
    tr = pd.concat(parts, ignore_index=True).drop(columns=["k"])
    tr.to_parquet(OC.PREP / "tracks.parquet", index=False)
    rep.update(tracks_rows=int(len(tr)), tracks_tokens=int(tr["row"].nunique()), finished=kst())
    (OC.PREP / "prep.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
