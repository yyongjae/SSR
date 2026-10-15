#!/usr/bin/env python
"""Stage E side store: per token the route centerline crop and PDM-Closed's effective progress, so the E2E dataloader
never unpickles a metric cache (0.05-0.7 s each).

  python tools/refiner/build_e2e_side.py build  [--tokens <parquet>] [--workers 6] [--limit N]
  python tools/refiner/build_e2e_side.py check  [--n 50]     # p_pdm / centerline vs scores/train.parquet / packed train

Output <DATA_ROOT>/e2e_side/<tok[:2]>/<tok>.npz (atomic write; existing files are skipped, so it is resumable and can
follow the metric-cache build): cl_xy [CL_MAX, 2] f32, cl_valid [CL_MAX] bool, cl_n i32 (= data.centerline_samples,
identical to the packed 'centerline' part), p_pdm f64 (= score_trajectories.score_token(...)[0]['pdm_progress_eff'],
PDM-Closed raw progress x PDM-Closed multiplier; it does not depend on the submitted trajectory, a standing-still
trajectory is submitted).  Tokens without a metric cache are skipped (retried on the next run).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(REPO / "data/dataset/maps"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402

TOKENS = RD.DATA_ROOT / "splits/e2e_train_trainlogs.parquet"
OUT = RD.DATA_ROOT / "e2e_side"
_ST = None


def _scorer_module():
    global _ST
    if _ST is None:
        spec = importlib.util.spec_from_file_location("stageE_score_trajectories", REPO / "tools/refiner/score_trajectories.py")
        _ST = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ST)
    return _ST


def side_path(token: str, root: Path = OUT) -> Path:
    return Path(root) / token[:2] / f"{token}.npz"


def compute_side(mc) -> dict:
    cl = RD.centerline_samples(mc)
    p = _scorer_module().score_token(mc, np.zeros((1, 8, 3), np.float32))[0]["pdm_progress_eff"]
    return dict(cl, p_pdm=np.float64(p))


def _one(args):
    token, log, root = args
    out = side_path(token, Path(root))
    if out.is_file():
        return token, "exists"
    mcp = RD.locate_metric_cache(token, log)
    if mcp is None:
        return token, "no_mc"
    try:
        d = compute_side(RD.load_metric_cache(mcp))
    except Exception as e:  # noqa: BLE001
        return token, f"error:{type(e).__name__}:{e}"[:200]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{token}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(out)
    return token, "ok"


def build(a) -> None:
    df = pd.read_parquet(a.tokens, columns=["token", "log"])
    if a.limit:
        df = df.iloc[: a.limit]
    todo = [(t, lg, a.out) for t, lg in zip(df.token, df.log) if not side_path(t, Path(a.out)).is_file()]
    print(f"{len(df)} tokens, {len(todo)} to build -> {a.out}", flush=True)
    counts, t0 = {}, time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        for i, (tok, st) in enumerate(pool.imap_unordered(_one, todo, chunksize=8)):
            key = st.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "error" and counts[key] <= 5:
                print(tok, st, flush=True)
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(todo)} {counts} {time.time() - t0:.0f}s", flush=True)
    n_done = sum(side_path(t, Path(a.out)).is_file() for t in df.token)
    rec = dict(counts=counts, n_tokens=len(df), n_with_side=int(n_done), sec=round(time.time() - t0, 1),
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "status.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


def check(a) -> None:
    """p_pdm vs scores/train.parquet (pdm_progress_eff) and the centerline vs the packed train split, on n tokens that
    are in both the stage-T train split and the E2E list."""
    e2e = pd.read_parquet(a.tokens, columns=["token", "log"])
    sc = pd.read_parquet(RD.DATA_ROOT / "scores/train.parquet", columns=["token", "pdm_progress_eff"]).drop_duplicates("token")
    both = e2e.merge(sc, on="token").sample(frac=1.0, random_state=0)
    packed = RD.PackedSplit("train")
    pidx = {t: i for i, t in enumerate(packed.index.token.values)}
    n = bad = 0
    for r in both.itertuples():
        if n >= a.n:
            break
        mcp = RD.locate_metric_cache(r.token, r.log)
        if mcp is None:
            continue
        d = compute_side(RD.load_metric_cache(mcp))
        row = packed.row(pidx[r.token]) if r.token in pidx else None
        eq_p = float(d["p_pdm"]) == float(r.pdm_progress_eff)
        eq_c = row is None or (np.array_equal(row["cl_xy"], d["cl_xy"]) and int(row["cl_n"]) == int(d["cl_n"]))
        bad += int(not (eq_p and eq_c))
        n += 1
    print(json.dumps(dict(n_checked=n, n_mismatch=bad)))
    if bad or n == 0:
        raise SystemExit(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--tokens", default=str(TOKENS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args(argv)
    build(a) if a.cmd == "build" else check(a)


if __name__ == "__main__":
    main()
