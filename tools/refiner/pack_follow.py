#!/usr/bin/env python
"""Pack the stage-T train / dev splits while the upstream builds are still running (resumable, follow mode).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 python tools/refiner/pack_follow.py \
      --splits train,dev --workers 2 --interval 900 --max-h 12 > <data>/packed/logs/follow.log 2>&1 &

Each pass runs data.pack_split(split) for every part (human, drafts, labels, objects, sdf, centerline) of every split.
pack_split is resumable per (row, part): rows whose source does not exist yet (draft npz, 13 scored label rows in
scores/<split>.parquet, objects npz, SDF npz, metric cache for the centerline) stay undone and are picked up by a later
pass; a source that is being written (truncated metric-cache pickle) raises inside the worker, is recorded as an error
and is retried next pass.  Status per pass -> <out_root>/<split>/follow_status.json and stdout.

Stops when
  * every row without a frame gap has all parts (complete), or
  * no upstream job is running any more (build_metric_cache.py build, build_sdf.py, make_draft_bank.py,
    build_future_objects.py; detected with pgrep) AND the pass packed nothing new (nothing left to wait for), or
  * --max-h hours have passed.
Caveat: a packed part is final.  If an upstream product is REGENERATED (e.g. a new draft-bank config), repack with
  python -m navsim.agents.para_ssr.refiner.data --split <s> --redo drafts,labels
CPU only; --workers <= 2 by default (shared machine; the pack reads ~0.4 MB/token and loads one metric cache per token
for the centerline, ~0.2 s).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402

UPSTREAM = ("build_metric_cache.py build", "build_sdf.py", "make_draft_bank.py", "build_future_objects.py")


def upstream_alive(patterns: Sequence[str] = UPSTREAM) -> Dict[str, bool]:
    """pattern -> True if a process whose command line contains it is running (pgrep -f)."""
    out = {}
    for p in patterns:
        r = subprocess.run(["pgrep", "-f", p], capture_output=True, text=True)
        out[p] = bool(r.stdout.strip())
    return out


def split_status(packed_root: Path, split: str) -> Dict:
    """Per part: rows done among the rows without a frame gap (the rows training can use), + fully ready rows."""
    P = RD.PackedSplit(split, packed_root)
    done = np.asarray(P.done)
    human = done[:, RD.PART_ID["human"]] == 1
    gap = np.asarray(P.arrays["frame_gap"], bool) & human
    use = ~gap
    st = dict(split=split, n=int(P.N), n_frame_gap=int(gap.sum()), n_usable=int(use.sum()),
              parts={p: int((done[:, RD.PART_ID[p]] == 1)[use].sum()) for p in RD.PARTS},
              ready=int(len(P.rows_with())))
    st["complete"] = bool(human.all() and st["ready"] == st["n_usable"])
    return st


def follow(splits: Sequence[str], packed_root: Path, workers: int = 2, interval: float = 900.0, max_h: float = 12.0,
           sources_fn: Optional[Callable] = None, tokens_fn: Optional[Callable] = None,
           alive_fn: Callable[[], Dict[str, bool]] = upstream_alive, sleep_fn: Callable = time.sleep,
           log_fn: Callable = print, chunk: int = 64) -> Dict:
    """Pack passes until complete / upstream finished with no progress / timeout.  -> final status per split."""
    t_start = time.time()
    n_pass = 0
    last_ready = None
    reason = None
    status = {}
    while True:
        n_pass += 1
        alive_before = alive_fn()
        ready = {}
        for s in splits:
            src = sources_fn(s) if sources_fn else None
            tok = tokens_fn(s) if tokens_fn else None
            summ = RD.pack_split(s, tok, packed_root, sources=src, workers=workers, chunk=chunk, log_fn=log_fn)
            st = split_status(packed_root, s)
            st.update(pass_=n_pass, time=time.strftime("%Y-%m-%dT%H:%M:%S"), upstream_alive=alive_before,
                      pass_summary={p: {k: v for k, v in d.items() if k != "error_examples"} for p, d in summ.items()},
                      error_examples={p: d.get("error_examples", [])[:2] for p, d in summ.items() if d.get("errors")})
            status[s] = st
            (Path(packed_root) / s / "follow_status.json").write_text(json.dumps(st, indent=1, default=str))
            log_fn(f"[pack_follow] pass {n_pass} {s}: ready {st['ready']}/{st['n_usable']} parts {st['parts']}")
            ready[s] = st["ready"]
        if all(status[s]["complete"] for s in splits):
            reason = "complete"
            break
        # the upstream state is read BEFORE the pass: if nothing was running then, this pass saw every final source
        if not any(alive_before.values()) and ready == last_ready:
            reason = "upstream finished, no progress"
            break
        if time.time() - t_start > max_h * 3600:
            reason = "timeout"
            break
        last_ready = ready
        sleep_fn(interval)
    for s in splits:
        status[s]["stop_reason"] = reason
        (Path(packed_root) / s / "follow_status.json").write_text(json.dumps(status[s], indent=1, default=str))
    log_fn(f"[pack_follow] stop: {reason} after {n_pass} passes, {time.time() - t_start:.0f}s")
    return dict(reason=reason, passes=n_pass, status=status)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--packed-root", default=str(RD.DATA_ROOT / "packed"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--interval", type=float, default=900.0, help="seconds between passes")
    ap.add_argument("--max-h", type=float, default=12.0)
    a = ap.parse_args(argv)
    if a.workers > 4:
        raise SystemExit("--workers <= 4 (shared machine)")
    res = follow([s for s in a.splits.split(",") if s], Path(a.packed_root), a.workers, a.interval, a.max_h,
                 log_fn=lambda m: print(m, flush=True), chunk=a.chunk)
    print(json.dumps({"reason": res["reason"], "passes": res["passes"],
                      **{s: {k: v for k, v in st.items() if k in ("ready", "n_usable", "parts")}
                         for s, st in res["status"].items()}}, indent=1), flush=True)


if __name__ == "__main__":
    main()
