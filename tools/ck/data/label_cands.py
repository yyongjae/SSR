#!/usr/bin/env python
"""CK Phase 1: official PDM labels of K candidate trajectories per token (score_trajectories.score_token, LQR).

  label_cands.py --split S --name {cand|kd_corr|corr_<run>} [--traj <npy [N,K,8,3] rows = packed/<split>/tokens.parquet>]
                 [--rows <npy int>] [--root R] [--workers 48] [--limit N] [--chunk 64]
                 [--from-dump [--watch --poll 60]] [--assemble-only] [--retry-errors] [--fresh] [--out DIR]

Trajectory source
  --traj NPY         rows aligned to <root>/packed/<split>/tokens.parquet (kd_corr_traj, corr_traj, ...).
  (default, --name cand) <root>/packed/<split>/cand.npy.
  --from-dump        cand_top16 straight from <root>/v2dump/<split>/plan/<tok[:2]>/<tok>.npz (token list = packed tokens if
                     packed exists, else split order); a chunk is scored once all its tokens are dumped, so labelling
                     can run while the dump is still writing (--watch polls until every chunk is done).
Work unit = chunk of --chunk tokens -> <labels>/shards/c<id>.parquet (token, row, k, crc32 of the token's trajectories,
  every score_token OUT_KEY, error, sec_load, sec_score).  Existing shards are skipped (resume); a chunk whose metric
  caches are not all present is left for a later run (not written).  Scoring exceptions are written as k = -1 rows and
  retried with --retry-errors.  shards/_plan.json pins (token list sha, chunk size, K); a different plan needs --fresh.
Assembly (when every chunk is done and packed/<split>/tokens.parquet exists):
  labels.npy f32 [N,K,9] (LABEL_COLS = nc, dac, ep, ttc, comfort, ddc, pdms, raw_progress, pdm_progress_eff; NaN if
  unscored), ok.npy bool [N,K], meta.json (traj source + sha16, n_scored, n_err, scorer source sha256, wall time).
  The crc of every scored token is checked against the assembly trajectory source (stale-shard guard).
Output dir: <root>/labels/<split>/<name>/ (default) or --out DIR (used as is; packed inputs still come from <root>).
Metric caches: core SPLITS[split]['metric_cache']/<log>/unknown/<token>/metric_cache.pkl.
CPU only; OMP_NUM_THREADS=1; <= 48 worker processes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import common as CM  # noqa: E402  (pins navsim BEFORE score_trajectories)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, CM.CK + "/tools/refiner")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import score_trajectories as ST  # noqa: E402

import navsim  # noqa: E402

assert navsim.__file__.startswith(CM.CK), navsim.__file__
LABEL_COLS = CM.LABEL_COLS
SCORE_COLS = ("nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms")   # must be finite for ok

_W: Dict[str, object] = {}


# ----------------------------------------------------------------------------------------------- traj sources
class TrajSource:
    """token -> [K, 8, 3] f32.  kind 'npy' (rows aligned to a token list) or 'dump' (plan npz cand_top<K>)."""

    def __init__(self, kind: str, path: str, tokens: List[str], k: int = CM.K_CAND):
        self.kind, self.path, self.k = kind, path, k
        self.row = {t: i for i, t in enumerate(tokens)}
        self.arr = None

    def open(self):
        if self.kind == "npy" and self.arr is None:
            self.arr = np.load(self.path, mmap_mode="r")
        return self

    def has(self, tok: str) -> bool:
        if self.kind == "npy":
            return tok in self.row
        return CM.plan_path(Path(self.path), tok).is_file()

    def get(self, tok: str) -> np.ndarray:
        if self.kind == "npy":
            self.open()
            return np.asarray(self.arr[self.row[tok]], np.float32)
        with np.load(CM.plan_path(Path(self.path), tok), allow_pickle=False) as z:
            return np.asarray(z[f"cand_top{self.k}"], np.float32)


def _init(src: TrajSource, split_name: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["src"] = src.open()
    _W["split"] = split_name
    ST.get_simulator_scorer()


def score_chunk(task):
    """task = (cid, shard_path, [(row, token, log)]) -> (cid, n_ok, n_err, seconds)."""
    cid, shard, items = task
    t0 = time.time()
    rows, n_ok, n_err = [], 0, 0
    src: TrajSource = _W["src"]
    for row, tok, log in items:
        try:
            trajs = src.get(tok)
            crc = CM.traj_crc(trajs)
            ts = time.time()
            mc = ST.load_metric_cache(CM.mc_path(_W["split"], log, tok))
            tl = time.time()
            res = ST.score_token(mc, trajs)
            te = time.time()
            for k, r in enumerate(res):
                rows.append(dict(token=tok, row=int(row), k=k, crc=crc, **r, error="", sec_load=tl - ts,
                                 sec_score=te - tl))
            n_ok += 1
        except Exception as e:
            rows.append(dict(token=tok, row=int(row), k=-1, crc=-1,
                             error=(repr(e) + " | " + traceback.format_exc()[-400:])[:800]))
            n_err += 1
    CM.atomic_to_parquet(pd.DataFrame(rows), shard)
    return cid, n_ok, n_err, time.time() - t0


# ----------------------------------------------------------------------------------------------- planning
def token_table(split_name: str, root, from_dump: bool, limit: int) -> pd.DataFrame:
    pk = CM.packed_dir(split_name, root) / "tokens.parquet"
    if pk.is_file():
        df = pd.read_parquet(pk)[["token", "log", "row"]]
    elif from_dump:
        df = CM.split_tokens(split_name)[["token", "log"]]
        df["row"] = np.arange(len(df))
    else:
        raise SystemExit(f"{pk} missing: run pack_v2.py --split {split_name} first (or --from-dump)")
    if limit:
        df = df.iloc[:limit]
    return df.reset_index(drop=True)


def shard_done(shard_dir: Path, retry_errors: bool) -> set:
    done = set()
    for f in shard_dir.glob("c*.parquet"):
        if retry_errors:
            d = pd.read_parquet(f, columns=["k"])
            if (d.k < 0).any():
                continue
        done.add(int(f.stem[1:]))
    return done


def check_plan(shard_dir: Path, plan: dict, fresh: bool):
    pf = shard_dir / "_plan.json"
    if fresh and shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    if pf.is_file():
        old = json.loads(pf.read_text())
        keys = ("token_sha16", "chunk", "k", "source_id")
        if any(old.get(k) != plan.get(k) for k in keys):
            raise SystemExit(f"shard plan changed ({ {k: (old.get(k), plan.get(k)) for k in keys} }); use --fresh")
    else:
        CM.atomic_write_json(pf, plan)


# ----------------------------------------------------------------------------------------------- assembly
def assemble(split_name: str, name: str, root, src: TrajSource, k: int, meta_extra: dict,
             out_dir=None) -> Optional[dict]:
    out = Path(out_dir) if out_dir else CM.labels_dir(split_name, name, root)
    pk = CM.packed_dir(split_name, root) / "tokens.parquet"
    if not pk.is_file():
        print(f"assemble deferred: {pk} missing (run pack_v2.py, then label_cands.py --assemble-only)", flush=True)
        return None
    toks = pd.read_parquet(pk)
    n = len(toks)
    rowmap = {t: i for i, t in enumerate(toks.token)}
    parts = [pd.read_parquet(f) for f in sorted((out / "shards").glob("c*.parquet"))]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["token", "k"])
    lab = np.full((n, k, len(LABEL_COLS)), np.nan, np.float32)
    ok = np.zeros((n, k), bool)
    good = df[df.k >= 0]
    err_tok = sorted(set(df.loc[df.k < 0, "token"]) - set(good.token))
    # stale-shard guard: crc of the scored trajectories == assembly source (dump source: checked against packed
    # cand.npy when it has the same K, which also proves packed cand == dumped cand bit for bit)
    chk = src
    pc = CM.packed_dir(split_name, root) / "cand.npy"
    if src.kind == "dump" and pc.is_file() and np.load(pc, mmap_mode="r").shape[1] == k:
        chk = TrajSource("npy", str(pc), toks.token.tolist(), k).open()
    n_stale = 0
    crc = good.groupby("token").crc.first()
    for t, c in crc.items():
        if t in rowmap and chk.has(t) and CM.traj_crc(chk.get(t)) != int(c):
            n_stale += 1
    if n_stale:
        raise SystemExit(f"{n_stale} tokens scored on different trajectories than the source; rerun with --fresh")
    good = good[good.token.isin(rowmap)]
    r = good.token.map(rowmap).to_numpy()
    kk = good.k.to_numpy().astype(int)
    assert (kk < k).all(), "k out of range"
    vals = good[list(LABEL_COLS)].to_numpy(np.float32)
    lab[r, kk] = vals
    ok[r, kk] = np.isfinite(good[list(SCORE_COLS)].to_numpy(np.float64)).all(1)
    CM.atomic_save_npy(out / "labels.npy", lab)
    CM.atomic_save_npy(out / "ok.npy", ok)
    sec = good.groupby("token")[["sec_load", "sec_score"]].first() if len(good) else pd.DataFrame()
    meta = dict(split=split_name, name=name, n=n, k=k, label_cols=list(LABEL_COLS),
                n_scored_tokens=int(ok.any(1).sum()), n_scored_cands=int(ok.sum()), n_err_tokens=len(err_tok),
                err_tokens_head=err_tok[:20], n_rec_not_ok=int((~good.rec_ok.astype(bool)).sum()) if len(good) else 0,
                traj_kind=src.kind, traj_path=src.path,
                mean_sec_load=float(sec.sec_load.mean()) if len(sec) else None,
                mean_sec_score=float(sec.sec_score.mean()) if len(sec) else None,
                scorer_sha256={str(p): ST._sha256(p) for p in ST.SOURCES}, scorer_yaml=str(ST.SCORING_YAML),
                navsim=navsim.__file__, assembled=time.strftime("%F %T"), **meta_extra)
    if src.kind == "npy":
        meta["traj_sha16"] = CM.sha16_array(np.load(src.path, mmap_mode="r"))
    CM.atomic_write_json(out / "meta.json", meta)
    print(f"assembled {out}: tokens {n} scored {meta['n_scored_tokens']} cands {meta['n_scored_cands']} "
          f"err {len(err_tok)}", flush=True)
    return meta


# ----------------------------------------------------------------------------------------------- main
def run(split_name: str, name: str, traj: str = "", rows: str = "", root=None, workers: int = 48, limit: int = 0,
        chunk: int = 64, from_dump: bool = False, watch: bool = False, poll: int = 60, assemble_only: bool = False,
        retry_errors: bool = False, fresh: bool = False, k: int = 0, max_wait: int = 4 * 3600,
        out_dir=None) -> Optional[dict]:
    assert 1 <= workers <= 48, "at most 48 worker processes per agent"
    t_start = time.time()
    out = Path(out_dir) if out_dir else CM.labels_dir(split_name, name, root)
    shard_dir = out / "shards"
    df = token_table(split_name, root, from_dump and not traj, limit)
    toks = df.token.tolist()
    if traj:
        kind, path = "npy", str(Path(traj).resolve())
    elif from_dump:
        kind, path = "dump", str(CM.dump_dir(split_name, root))
    else:
        assert name == "cand", "--traj is required for names other than 'cand'"
        kind, path = "npy", str(CM.packed_dir(split_name, root) / "cand.npy")
    if kind == "npy":
        shp = np.load(path, mmap_mode="r").shape
        pk_n = len(pd.read_parquet(CM.packed_dir(split_name, root) / "tokens.parquet"))
        assert shp[0] == pk_n and shp[2:] == (8, 3), (shp, pk_n)
        k = shp[1]
    k = k or CM.K_CAND
    src = TrajSource(kind, path, pd.read_parquet(CM.packed_dir(split_name, root) / "tokens.parquet").token.tolist()
                     if kind == "npy" else toks, k)
    sel = np.arange(len(df))
    if rows:
        sel = np.unique(np.load(rows).astype(np.int64))
        sel = sel[sel < len(df)]
    # the dumped cand_top16 and packed/<split>/cand.npy are the same trajectories (crc-checked at assembly), so a
    # --from-dump run and a packed run of 'cand' share shards; any other source is identified by its path
    pc = str(CM.packed_dir(split_name, root) / "cand.npy")
    source_id = f"v2cand:{split_name}:k{k}" if (name == "cand" and (kind == "dump" or path == pc)) else path
    plan = dict(split=split_name, name=name, token_sha16=CM.sha16_array(np.array(toks, "U16")), chunk=chunk, k=k,
                source_id=source_id, traj_kind=kind, traj_path=path, rows=rows, n_rows=int(len(sel)))
    if not assemble_only:
        check_plan(shard_dir, plan, fresh)
        chunks = [(i // chunk, sel[i:i + chunk]) for i in range(0, len(sel), chunk)]
        print(f"[{time.strftime('%F %T')}] {split_name}/{name}: {len(sel)} tokens, {len(chunks)} chunks, K {k}, "
              f"traj {kind}:{path}, workers {workers}", flush=True)
        ready_tok, mc_ok = set(), set()
        pool = None
        tot = dict(ok=0, err=0)
        try:
            while True:
                done = shard_done(shard_dir, retry_errors)
                todo = [(cid, rr) for cid, rr in chunks if cid not in done]
                if not todo:
                    break
                tasks = []
                n_wait_dump = n_wait_mc = 0
                for cid, rr in todo:
                    items = [(int(r), toks[r], df.log.iat[r]) for r in rr]
                    wait = False
                    for _, t, lg in items:
                        if t not in ready_tok:
                            if src.has(t):
                                ready_tok.add(t)
                            else:
                                wait = True
                                n_wait_dump += 1
                                break
                        if t not in mc_ok:
                            if CM.mc_path(split_name, lg, t).is_file():
                                mc_ok.add(t)
                            else:
                                wait = True
                                n_wait_mc += 1
                                break
                    if not wait:
                        tasks.append((cid, str(shard_dir / f"c{cid:05d}.parquet"), items))
                if tasks:
                    if pool is None:
                        pool = Pool(workers, initializer=_init, initargs=(src, split_name), maxtasksperchild=40)
                    t0 = time.time()
                    for j, (cid, n_ok, n_err, dt) in enumerate(pool.imap_unordered(score_chunk, tasks)):
                        tot["ok"] += n_ok
                        tot["err"] += n_err
                        if (j + 1) % 20 == 0 or j + 1 == len(tasks):
                            el = time.time() - t0
                            rate = tot["ok"] / max(time.time() - t_start, 1e-9)
                            print(f"[{time.strftime('%T')}] chunks {j + 1}/{len(tasks)} | tokens ok {tot['ok']} err "
                                  f"{tot['err']} | {el:.0f}s {rate:.2f} tok/s", flush=True)
                    continue
                if watch and max_wait and time.time() - t_start > max_wait:
                    print(f"max wait {max_wait}s reached with {len(todo)} chunks not ready", flush=True)
                    break
                if not watch:
                    print(f"{len(todo)} chunks not ready (waiting for dump {n_wait_dump}, metric cache {n_wait_mc}); "
                          f"rerun later or use --watch", flush=True)
                    break
                print(f"[{time.strftime('%T')}] waiting: {len(todo)} chunks (dump {n_wait_dump}, mc {n_wait_mc})",
                      flush=True)
                time.sleep(poll)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        if len(shard_done(shard_dir, False)) < len(chunks):
            return None
    return assemble(split_name, name, root, src.open() if kind == "npy" else src, k,
                    dict(rows=rows, n_rows=int(len(sel)), chunk=chunk, workers=workers,
                         wall_s=round(time.time() - t_start, 1)), out_dir=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=sorted(CM.SPLITS))
    ap.add_argument("--name", required=True)
    ap.add_argument("--traj", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--root", default="")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--from-dump", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-wait", type=int, default=4 * 3600, help="--watch gives up after this many seconds")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--out", default="", help="output dir (default <root>/labels/<split>/<name>)")
    a = ap.parse_args(argv)
    print("navsim", navsim.__file__, "| score_trajectories", ST.__file__, flush=True)
    run(a.split, a.name, a.traj, a.rows, a.root or None, a.workers, a.limit, a.chunk, a.from_dump, a.watch, a.poll,
        a.assemble_only, a.retry_errors, a.fresh, max_wait=a.max_wait, out_dir=a.out or None)


if __name__ == "__main__":
    main()
