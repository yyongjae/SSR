#!/usr/bin/env python
"""Official PDM labels of an FPD teacher's DiT bank on CPU (the second half of train_fpd_teacher.py --stage bank).

  label_fpd_bank.py --run-dir <FPD teacher run dir> [--splits navtrain_train,navtrain_val] [--workers 48]
                    [--chunk 2048] [--retry-errors] [--fresh] [--limit N] [--scorer official|fake]

Input   run_dir/bank/<split>/ written by train_fpd_teacher.py --stage bank (tools/ck/fpd/bank.py layout): traj.npy
        [N, 32, 8, 3], sampled.npy (1 = sampled), tokens.parquet (token, log), sample_meta.json and sample_done.json
        (sampling finished; traj_sha16).  --splits default: every bank/<split> with a sample_meta.json.
Scoring the e2e labeler's per-token function (tools/ck/e2e/labeler.py score_one, which the on-policy labeler2 runs):
        ONE score_trajectories.score_token call (official PDM simulate / score with pairwise-official EP / DDC) on the
        token's finite trajectories with its navtrain metric cache, located like the labeler does
        (tools/ck/data/common.mc_path(split, log, token) = <SPLITS[split]['metric_cache']>/<log>/unknown/<token>/
        metric_cache.pkl; navtrain_val shares the navtrain cache); score dicts -> the 9 LABEL_COLS (nc, dac, ep, ttc,
        comfort, ddc, pdms, raw_progress, pdm_progress_eff), ok = SCORE_COLS finite (label_cands rule); non-finite
        trajectories stay unscored.  Fork process pool (labeler._init_worker), OMP_NUM_THREADS=1, <= 48 workers.
        --scorer fake = labeler.fake_scores (tests only).
Output  (same dir) labels.npy f32 [N, 32, 9] (NaN = not scored), ok.npy bool [N, 32], label_state.npy uint8 [N]
        (0 to do, 1 labelled, 2 error), errors.jsonl, label_plan.json (the traj_sha16 / scorer the labels are for),
        meta.json once no sampled row is left to do (n_sampled, n_labelled, n_err, n_ok_cands, traj_sha16, scorer
        source sha256, wall time).  Label / ok rows are flushed before their state (resume never sees half a row);
        labelled rows are skipped on a rerun, error rows are re-scored with --retry-errors; a bank re-sampled since the
        labels started (another traj_sha16) or another --scorer is refused unless --fresh (wipes the label files).
        One labeler per split (flock bank/<split>/.label.lock).  Exit 0 = every split has its meta.json, 1 = rows left.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.e2e import labeler as L1  # noqa: E402  (pins navsim + the official scorer; score_one)
from tools.ck.data import common as CM  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402
from tools.ck.fpd import bank as BK  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABEL_COLS = tuple(L1.LABEL_COLS)
SCORE_COLS = tuple(L1.SCORE_COLS)
MAX_WORKERS = int(L1.MAX_WORKERS)
WRITE_BATCH = 64
SCORERS = ("official", "fake")
LABEL_FILES = (BK.F_LABELS, BK.F_OK, BK.F_LSTATE, BK.F_LPLAN, BK.F_LMETA, BK.F_ERRORS)


def _t() -> str:
    return time.strftime("%F %T")


def task_mc_path(split: str, log: str, token: str, scorer: str) -> str:
    """The labeler's metric-cache path of a token ('fake:<token>' for the fake scorer)."""
    if scorer == "fake":
        return f"fake:{token}"
    return str(CM.mc_path(split, log, token))


def _write(labels, okm, state, buf: List) -> None:
    """data rows, flush, then their state 1, flush (labeler.write_generation_rows order)."""
    if not buf:
        return
    r = np.array([b[0] for b in buf], np.int64)
    labels[r] = np.stack([b[1] for b in buf]).astype(np.float32)
    okm[r] = np.stack([b[2] for b in buf]).astype(bool)
    labels.flush()
    okm.flush()
    state[r] = 1
    state.flush()


def default_splits(run_dir) -> List[str]:
    root = Path(run_dir) / "bank"
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / BK.F_SAMPLE_META).is_file())


def label_split(bdir, workers: int = 48, chunk: int = 2048, scorer: str = "official", retry_errors: bool = False,
                fresh: bool = False, limit: int = 0, log=print) -> Optional[Dict]:
    """Label one bank/<split> dir -> its meta.json dict once no sampled row is left to do, else None."""
    import fcntl
    import multiprocessing as mp

    bdir = Path(bdir)
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise SystemExit(f"--workers {workers} not in [1, {MAX_WORKERS}]")
    if scorer not in SCORERS:
        raise SystemExit(f"--scorer {scorer!r} not in {SCORERS}")
    smeta, sdone = BK.read_json(bdir / BK.F_SAMPLE_META), BK.read_json(bdir / BK.F_SAMPLE_DONE)
    if smeta is None or sdone is None:
        raise SystemExit(f"{bdir}: sampling not finished (no {BK.F_SAMPLE_DONE}): train_fpd_teacher.py --stage bank")
    if fresh and (bdir.parent.parent / "done.json").is_file():
        raise SystemExit(f"{bdir.parent.parent} is a finished FPD teacher (done.json): its bank labels are not wiped")
    split, N, K = str(smeta["split"]), int(smeta["n_packed"]), BK.BANK_K
    lockf = open(bdir / ".label.lock", "a+")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lockf.close()
        raise SystemExit(f"another labeler holds {bdir / '.label.lock'}")
    try:
        plan = dict(split=split, n_packed=N, k=K, traj_sha16=sdone["traj_sha16"],
                    dit_ckpt_sha16=sdone.get("dit_ckpt_sha16"), label_cols=list(LABEL_COLS),
                    ok_rule="SCORE_COLS finite: " + ",".join(SCORE_COLS), scorer=scorer)
        old = BK.read_json(bdir / BK.F_LPLAN)
        stale = old is not None and any(old.get(k) != plan[k] for k in ("traj_sha16", "scorer", "n_packed", "k"))
        if stale and not fresh:
            raise SystemExit(f"{bdir}: labels started for traj {old.get('traj_sha16')} / scorer {old.get('scorer')}, "
                             f"now {plan['traj_sha16']} / {scorer}: --fresh relabels from scratch")
        if fresh or stale:
            for f in LABEL_FILES:
                (bdir / f).unlink(missing_ok=True)
            old = None
        if old is None:
            CM.atomic_write_json(bdir / BK.F_LPLAN, dict(plan, created=_t()))
        labels = U.open_memmap(bdir / BK.F_LABELS, (N, K, len(LABEL_COLS)), np.float32, fill=np.nan)
        okm = U.open_memmap(bdir / BK.F_OK, (N, K), np.bool_, fill=False)
        state = U.open_memmap(bdir / BK.F_LSTATE, (N,), np.uint8, fill=0)
        traj = np.load(bdir / BK.F_TRAJ, mmap_mode="r")
        sampled = np.asarray(np.load(bdir / BK.F_SAMPLED, mmap_mode="r"))
        tt = pd.read_parquet(bdir / BK.F_TOKENS)
        if traj.shape != (N, K, 8, 3) or sampled.shape != (N,) or len(tt) != N:
            raise SystemExit(f"{bdir}: traj {traj.shape} / sampled {sampled.shape} / tokens {len(tt)} != {N} rows")
        toks, logs = tt.token.astype(str).to_numpy(), tt.log.astype(str).to_numpy()
        rows = np.flatnonzero(sampled == 1)
        st = np.asarray(state)
        todo = rows[(st[rows] == 0) | ((st[rows] == 2) & bool(retry_errors))]
        if limit:
            todo = todo[: int(limit)]
        log(f"[{_t()}] {bdir}: {len(rows)} sampled rows, {int((st[rows] == 1).sum())} labelled, "
            f"{int((st[rows] == 2).sum())} errors, {len(todo)} to do (workers {workers}, scorer {scorer})")
        t0, n_ok, n_err = time.time(), 0, 0
        if len(todo):
            if scorer == "official":
                missing = [int(r) for r in todo[:2000] if not Path(task_mc_path(split, logs[r], toks[r], scorer)).is_file()]
                if missing:
                    log(f"[{_t()}] warning: {len(missing)} of the first {min(len(todo), 2000)} rows have no metric cache "
                        f"(e.g. {task_mc_path(split, logs[missing[0]], toks[missing[0]], scorer)}): they will error")
            pool = mp.get_context("fork").Pool(min(int(workers), len(todo)), initializer=L1._init_worker,
                                                initargs=(scorer,), maxtasksperchild=2560)
            clean, buf = False, []
            try:
                with open(bdir / BK.F_ERRORS, "a") as ef:
                    for lo in range(0, len(todo), int(chunk)):
                        part = todo[lo:lo + int(chunk)]
                        tasks = [(int(r), toks[r], np.array(traj[r], np.float32),
                                  task_mc_path(split, logs[r], toks[r], scorer)) for r in part]
                        for key, lab, ok, err, s_load, s_score in pool.imap_unordered(L1.score_one, tasks):
                            if lab is None:
                                n_err += 1
                                state[key] = 2
                                state.flush()
                                ef.write(json.dumps(dict(row=int(key), token=str(toks[key]), error=err, time=_t())) + "\n")
                                ef.flush()
                            else:
                                buf.append((key, lab, ok))
                                n_ok += 1
                                if len(buf) >= WRITE_BATCH:
                                    _write(labels, okm, state, buf)
                                    buf = []
                            n = n_ok + n_err
                            if n % 2000 == 0:
                                el = time.time() - t0
                                log(f"[{_t()}] {split}: {n}/{len(todo)} ok {n_ok} err {n_err} | {el:.0f}s "
                                    f"{n / max(el, 1e-9):.1f} tok/s")
                        _write(labels, okm, state, buf)
                        buf = []
                clean = True
            finally:
                _write(labels, okm, state, buf)
                if clean:
                    pool.close()
                    pool.join()
                else:
                    pool.terminate()
        st = np.asarray(state)
        left = int((st[rows] == 0).sum())
        if left:
            log(f"[{_t()}] {bdir}: {left} sampled rows still to label (rerun)")
            return None
        meta = BK.read_json(bdir / BK.F_LMETA)
        if meta is not None and n_ok + n_err == 0:
            log(f"[{_t()}] {bdir}: labels complete ({meta.get('n_labelled')} rows, {meta.get('n_err')} errors)")
            return meta
        err_rows = np.flatnonzero(st == 2)
        meta = dict(plan, n_rows=int(smeta.get("n_rows", len(rows))), n_sampled=int(len(rows)),
                    n_labelled=int((st[rows] == 1).sum()), n_err=int((st[rows] == 2).sum()),
                    n_ok_cands=int(np.asarray(okm)[rows].sum()), err_rows_head=[int(r) for r in err_rows[:20]],
                    scorer_sha256=({str(p): L1.ST._sha256(p) for p in L1.ST.SOURCES} if scorer == "official" else None),
                    scorer_yaml=str(L1.ST.SCORING_YAML), navsim=L1.LCD.navsim.__file__, workers=int(workers),
                    n_scored_last_session=n_ok, n_err_last_session=n_err, wall_s_last_session=round(time.time() - t0, 1),
                    finished=_t())
        CM.atomic_write_json(bdir / BK.F_LMETA, meta)
        log(f"[{_t()}] {bdir}: labels complete: {meta['n_labelled']} rows, {meta['n_err']} errors, "
            f"{meta['n_ok_cands']} ok candidates")
        return meta
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, help="FPD teacher run dir (holds bank/<split>/)")
    ap.add_argument("--splits", default="", help="comma list (default: every bank/<split> with a sample_meta.json)")
    ap.add_argument("--workers", type=int, default=48, help=f"pool processes (<= {MAX_WORKERS})")
    ap.add_argument("--chunk", type=int, default=2048, help="rows per pool round (bounds the queued trajectories)")
    ap.add_argument("--retry-errors", action="store_true", help="re-score rows of errors.jsonl (label_state 2)")
    ap.add_argument("--fresh", action="store_true", help="wipe the label files of the split(s) first")
    ap.add_argument("--limit", type=int, default=0, help="label at most N rows per split this session (tests)")
    ap.add_argument("--scorer", default="official", choices=SCORERS, help="'fake' = tests only")
    a = ap.parse_args(argv)
    splits = BK.parse_list(a.splits) if a.splits else default_splits(a.run_dir)
    if not splits:
        raise SystemExit(f"{a.run_dir}: no bank/<split>/{BK.F_SAMPLE_META} (run train_fpd_teacher.py --stage bank)")
    print("navsim", L1.LCD.navsim.__file__, "| score_trajectories", L1.ST.__file__, "| splits", splits, flush=True)
    rc = 0
    for split in splits:
        if label_split(BK.bank_dir(a.run_dir, split), a.workers, a.chunk, a.scorer, a.retry_errors, a.fresh, a.limit,
                       log=lambda s: print(s, flush=True)) is None:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
