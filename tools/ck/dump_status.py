#!/usr/bin/env python
"""v2 dump progress / ETA (run_phase1.sh dump + status).  Counts the tokens listed in v2dump/<split>/done_shard*.txt
(appended by tools/ck/data/dump_v2.py per batch), estimates the rate since runs/dump_start_epoch.txt and writes
runs/dump_status.json + runs/dump_eta.txt.  --check: exact file check (bev .npy + plan .npz for every split token),
exit 1 if anything is missing.

  PYTHONPATH=/workspace/yongjae/SSR-ck2 /venv/ssr/bin/python tools/ck/dump_status.py [--check] [--splits a,b]
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[2])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402

ORDER = ("navtrain_val", "navtest", "navtrain_train")


def done_tokens(d: Path) -> set:
    s = set()
    for f in d.glob("done_shard*.txt"):
        s.update(x.strip() for x in f.read_text().splitlines() if x.strip())
    return s


def main(argv=None):
    from navsim.agents.para_ssr.ck import constants as Cn
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--splits", default=",".join(ORDER))
    a = ap.parse_args(argv)
    root = U.ck_data()
    runs = root / "runs"
    splits = [s for s in a.splits.split(",") if s]
    rep = {"time": U.now(), "splits": {}}
    tot_n = tot_done = 0
    missing_all = 0
    for s in splits:
        d = root / "v2dump" / s
        n = int(Cn.SPLITS[s]["n"])
        dn = done_tokens(d) if d.is_dir() else set()
        r = {"n": n, "done_listed": len(dn)}
        if a.check:
            toks = pd.read_parquet(Cn.SPLITS[s]["tokens"]).token.astype(str).tolist()
            miss = [t for t in toks if not ((d / "bev" / t[:2] / f"{t}.npy").is_file()
                                            and (d / "plan" / t[:2] / f"{t}.npz").is_file())]
            r["missing_files"] = len(miss)
            r["missing_example"] = miss[:5]
            missing_all += len(miss)
        rep["splits"][s] = r
        tot_n += n
        tot_done += min(len(dn), n)
    rep.update(total=tot_n, done=tot_done)
    st = runs / "dump_start_epoch.txt"
    if st.is_file():
        t0 = float(st.read_text().strip())
        el = time.time() - t0
        rep["elapsed_min"] = round(el / 60, 1)
        if tot_done > 0 and el > 0:
            rate = tot_done / el
            rep["tok_per_s_total"] = round(rate, 2)
            rem = max(0, tot_n - tot_done) / max(rate, 1e-9)
            rep["eta_min"] = round(rem / 60, 1)
            rep["eta_clock"] = time.strftime("%F %T", time.localtime(time.time() + rem))
    try:
        du = os.statvfs(str(root))
        rep["disk_free_tb"] = round(du.f_bavail * du.f_frsize / 1e12, 3)
    except Exception:
        pass
    if runs.is_dir():
        U.write_json(runs / "dump_status.json", rep)
        (runs / "dump_eta.txt").write_text(
            f"{rep['time']} done {tot_done}/{tot_n} rate {rep.get('tok_per_s_total')} tok/s "
            f"ETA {rep.get('eta_clock')} ({rep.get('eta_min')} min)\n")
    for s, r in rep["splits"].items():
        print(f"{s}: {r['done_listed']}/{r['n']}" + (f" missing_files {r['missing_files']}" if a.check else ""))
    print(f"total {tot_done}/{tot_n} rate {rep.get('tok_per_s_total')} tok/s ETA {rep.get('eta_clock')} "
          f"({rep.get('eta_min')} min) disk free {rep.get('disk_free_tb')} TB")
    if a.check and missing_all:
        sys.exit(1)


if __name__ == "__main__":
    main()
