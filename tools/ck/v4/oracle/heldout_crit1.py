#!/usr/bin/env python
"""refiner v4_3 Phase 0b -- criterion 1 (R2) + the per-row tube flags of criterion S, on the 12,179 held-out
navtrain_val tokens (executed states of the 96 = par16 + valid var80; heldout_common.HeldoutData).  CPU only, no
scorer call (official labels = the stored label files, heldout_prep / heldout_rescore_check).

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/v4/oracle/heldout_crit1.py [--workers 28] [--limit N]

Per token:
  * run_crit1.process unchanged on the held-out row (every Phase 0 flag: grid main / ablations / post hoc reads,
    exact proxies, official-rule emulation, UNKNOWN bucket, scene descriptors) -- the Phase 0 code path, so the
    Phase 0 representations can be read on held-out tokens too (context only; no verdict uses them).
  * R2 through its own path (heldout_common.r2_from / r2_eval on freshly built parent fields): r2_nc, r2_ttc, r2_dac
    (hit = slack < 0), r2_slack_{nc,ttc,dac}, r2_noog; r2_code_mismatch = rows where R2 differs from Phase 0's
    'grid stopped-aware NC + read any4' flags (g25sa_nc, g25a_ttc, g25a_dac) -- must be 0.
Shards HO/crit1/shard_XXX.npz (resumable) -> HO/crit1/rows.npz + run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_common as H  # noqa: E402
import oracle_common as OC  # noqa: E402
import run_crit1 as C1  # noqa: E402  (process() only; its main / OUTD are not used)
import numpy as np  # noqa: E402

OUTD = H.HO / "crit1"
R2_BOOL = ("r2_nc", "r2_ttc", "r2_dac")
R2_FLOAT = ("r2_slack_nc", "r2_slack_ttc", "r2_slack_dac", "r2_noog")
BOOL_KEYS = C1.BOOL_KEYS + R2_BOOL
FLOAT_KEYS = C1.FLOAT_KEYS + R2_FLOAT
TOK_KEYS = C1.TOK_KEYS + ("r2_code_mismatch",)
_G: dict = {}


def init():
    try:
        os.nice(10)
    except Exception:
        pass
    D = H.HeldoutData()
    C1._G["D"] = D
    _G["D"] = D


def process(i: int) -> dict:
    o = C1.process(i)
    T = OC.token_inputs(_G["D"], i)
    for k in R2_BOOL:
        o[k] = np.zeros(OC.N96, bool)
    for k in R2_FLOAT:
        o[k] = np.full(OC.N96, np.nan)
    o["r2_code_mismatch"] = 0
    if T["sdf"] is None:
        return o
    cols = np.flatnonzero(T["valid96"])
    idx = OC.parent_of(cols)
    F2 = H.r2_from(OC.parent_fields(T["par"], T["gt"], T["sdf"], point=False), T["gt"])
    g = H.r2_eval(T["X"][cols], F2, idx)
    for k in ("nc", "ttc", "dac"):
        o[f"r2_{k}"][cols] = g[f"hit_{k}"]
        o[f"r2_slack_{k}"][cols] = g[f"slack_{k}"]
    o["r2_noog"][cols] = g["n_oog"]
    o["r2_code_mismatch"] = int((o["r2_nc"] != o["g25sa_nc"]).sum() + (o["r2_ttc"] != o["g25a_ttc"]).sum()
                                + (o["r2_dac"] != o["g25a_dac"]).sum())
    return o


def stack(res: list) -> dict:
    res = sorted(res, key=lambda d: d["i"])
    o = {"i": np.array([d["i"] for d in res]), "row": np.array([d["row"] for d in res])}
    for k in BOOL_KEYS + FLOAT_KEYS:
        o[k] = np.stack([d[k] for d in res])
    for k in TOK_KEYS:
        o[k] = np.array([d[k] for d in res], np.float64)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=250)
    a = ap.parse_args()
    assert a.workers <= 28
    OUTD.mkdir(parents=True, exist_ok=True)
    n = len(np.load(H.HO_PREP / "ho_rows.npy"))
    items = list(range(n if a.limit <= 0 else min(a.limit, n)))
    shards = [items[k:k + a.shard] for k in range(0, len(items), a.shard)]
    tag = "" if a.limit <= 0 else f"_lim{a.limit}"
    todo = [(j, s) for j, s in enumerate(shards) if not (OUTD / f"shard{tag}_{j:03d}.npz").is_file()]
    H.HeldoutData()                      # fail fast here (a failing Pool initializer respawns forever)
    t0 = time.time()
    print(f"{time.strftime('%F %T')} UTC tokens {len(items)} shards {len(shards)} todo {len(todo)}", flush=True)
    with Pool(a.workers, initializer=init, maxtasksperchild=100) as pool:
        for j, s in todo:
            res = pool.map(process, s, chunksize=1)
            np.savez(OUTD / f"shard{tag}_{j:03d}.npz", **stack(res))
            print(f"{time.strftime('%F %T')} shard {j} done ({time.time() - t0:.0f} s)", flush=True)
    parts = [dict(np.load(OUTD / f"shard{tag}_{j:03d}.npz")) for j in range(len(shards))]
    cat = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
    np.savez(OUTD / f"rows{tag}.npz", **cat)
    (OUTD / f"run{tag}.json").write_text(json.dumps(dict(
        n=len(items), sec=time.time() - t0, mean_sec_token=float(np.mean(cat["sec"])),
        gt_missing=int((cat["gt_ok"] == 0).sum()), sdf_missing=int((cat["sdf_ok"] == 0).sum()),
        r2_code_mismatch_rows=int(cat["r2_code_mismatch"].sum()), finished=H.kst()), indent=1))
    print("done", len(items), f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
