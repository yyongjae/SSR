#!/usr/bin/env python
"""CK Phase 2 background labeler: official PDM labels (score_token, LQR) of the student candidates recorded in training.

  labeler.py --io-dir <RUN>/ck_e2e --world-size 4 [--workers 24] [--poll 30] [--max-epoch 28] [--exit-when-done]
             [--train-pid-file <RUN>/launch/train.pid] [--retry-errors] [--once] [--limit-chunks N]
             [--pid-file F] [--exit-file F]

Input   rec/ep{E}/r{R}/c_<attempt>_<seq>.npz (training; e2e_data.list_rec_chunks / read_rec_chunk, tmp names ignored).
        Per token the 32 trajectories cat(cand 16, kd_corr 16) are scored in ONE score_token call on the
        navtrain_train metric cache (= tools/ck/data/label_cands.score_chunk per token; K=32 at once == 2 x K=16 bit
        for bit).  Non-finite trajectories are left unscored (labels NaN, cand_ok False); the finite ones are scored
        together.  ok = SCORE_COLS finite (label_cands rule).
Output  lab/ep{E}/{traj,labels,cand_ok,row_state}.npy via e2e_data.open_generation / write_generation_rows (this
        process is the only writer; traj/labels/cand_ok flushed, then row_state = 1 flushed; a row that already has
        row_state 1 is never overwritten = the first scored record wins).  Rows are written in small batches (<= 64
        rows or 2 s), always before the chunk marker.
        Chunk marker lab/ep{E}/chunks/r{R}_<chunk stem>.json {n, n_ok, n_err, n_skip, sec, tok_s (= labeler aggregate
        recent tok/s at that time), chunk_tok_s (that chunk alone; ~1/6 of the aggregate), lag_s (now - chunk mtime)}.
        Errors -> lab/ep{E}/errors.jsonl (row_state stays 0; --retry-errors re-scores them from the chunk file).
        Generation DONE lab/ep{E}/DONE.json once every rank 0..world_size-1 has a rec DONE_<attempt>.json whose chunks
        all carry markers and no chunk of E (orphans of interrupted attempts included) is pending.
        lab/status.json (per epoch chunks / marked / pending / rows done, tok/s, backlog, lag) every --status-every s.
Restart idempotent: markers skip chunks, generations reopen r+, rows with row_state 1 are skipped.  One labeler per
        io_dir (flock lab/.labeler.lock; a second instance exits 3).
Exit    --once: after the current backlog (0).  --limit-chunks N: after N chunks (0).  --exit-when-done: io_dir/TRAIN_DONE
        exists (0) or the train pid is dead (2; a pid file absent for > --pid-grace s counts as dead), and the backlog
        is 0.  SIGTERM: 143.  Code -> --exit-file (default <io_dir>/../labeler/labeler.exit when that dir exists).
        Abnormal end (SIGTERM, exception, tasks in flight): the pool workers are SIGKILLed and the process ends with
        os._exit after the final writes -- never Pool.terminate()/join(): a worker that died while holding the pool's
        task-queue lock (e.g. a TERM to the whole process group) makes them deadlock (smoke S4, 2026-10-07).
CPU only, OMP_NUM_THREADS=1, workers <= 48 (default 24), os.nice(5).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import label_cands as LCD  # noqa: E402  (pins navsim, imports score_trajectories)
from tools.ck.data import common as CM  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data as ED  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ST = LCD.ST
LABEL_COLS = tuple(LCD.LABEL_COLS)
SCORE_COLS = tuple(LCD.SCORE_COLS)
PACKED_TOKENS = str(Path(ED.PHASE1_DEFAULT["packed"]) / "tokens.parquet")
SPLIT = "navtrain_train"
MAX_WORKERS = 48
G_K = ED.G_K
GEN_KEYS = ("traj", "labels", "cand_ok", "row_state")
WRITE_BATCH, WRITE_AGE = 64, 2.0
RETRY = "retry:"

_W: Dict[str, object] = {}


# ----------------------------------------------------------------------------------------------- helpers
def _t() -> str:
    return time.strftime("%F %T")


def atomic_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=CM._json_default))
    os.replace(tmp, path)


def marker_name(rank: int, chunk_name: str) -> str:
    """Chunk marker name; the rank is part of it (chunk names alone are only unique within a rank dir)."""
    return f"r{int(rank)}_{Path(chunk_name).stem}.json"


def generation_ready(io_dir, epoch: int) -> bool:
    return (ED.gen_dir(io_dir, epoch) / "meta.json").is_file()


def list_chunks(io_dir, max_epoch: Optional[int]) -> List[dict]:
    return [c for c in ED.list_rec_chunks(io_dir) if max_epoch is None or c["epoch"] <= max_epoch]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except Exception:
        return True


def _sleep(sec: float, stop) -> None:
    end = time.time() + sec
    while time.time() < end and not stop():
        time.sleep(min(0.5, max(0.0, end - time.time())))


# ----------------------------------------------------------------------------------------------- worker side
def _init_worker(mode: str):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _W["mode"] = mode
    if mode == "official":
        ST.get_simulator_scorer()


def fake_scores(trajs: np.ndarray) -> np.ndarray:
    """Deterministic stand-in for score_token (tests only): [K, 9] f32 from the trajectory values."""
    s = np.asarray(trajs, np.float64).reshape(len(trajs), -1).sum(1)
    return np.stack([np.sin(s + c) for c in range(len(LABEL_COLS))], 1).astype(np.float32)


def score_one(task) -> tuple:
    """task = (key, token, trajs [K,8,3] f32, metric-cache path) -> (key, labels [K,9] f32 | None, ok [K] | None,
    error, sec_load, sec_score).  Official path = label_cands.score_chunk for one token."""
    key, tok, trajs, mcp = task
    t0 = time.time()
    try:
        trajs = np.asarray(trajs, np.float32)
        k = len(trajs)
        lab = np.full((k, len(LABEL_COLS)), np.nan, np.float32)
        ok = np.zeros(k, bool)
        fin = np.isfinite(trajs).reshape(k, -1).all(1)
        if _W.get("mode", "official") == "official":
            mc = ST.load_metric_cache(mcp)
            tl = time.time()
            if fin.any():
                res = ST.score_token(mc, trajs[fin])
                lab[fin] = np.array([[r[c] for c in LABEL_COLS] for r in res], np.float64).astype(np.float32)
                ok[fin] = np.isfinite(np.array([[r[c] for c in SCORE_COLS] for r in res], np.float64)).all(1)
        else:  # 'fake' (tests): a missing metric cache fails like the official loader
            if not str(mcp).startswith("fake:") and not Path(mcp).is_file():
                raise FileNotFoundError(mcp)
            fail = os.environ.get("CK_LABELER_FAKE_FAIL", "")
            if fail and Path(fail).is_file() and tok in json.loads(Path(fail).read_text()):
                raise RuntimeError(f"fake failure {tok}")
            tl = time.time()
            if fin.any():
                lab[fin] = fake_scores(trajs[fin])
                ok[fin] = True
        return key, lab, ok, "", tl - t0, time.time() - tl
    except Exception as e:
        return key, None, None, (repr(e) + " | " + traceback.format_exc()[-400:])[:800], time.time() - t0, 0.0


# ----------------------------------------------------------------------------------------------- labeler
class Labeler:
    def __init__(self, io_dir, world_size: int, workers: int = 24, poll: float = 30, max_epoch: Optional[int] = 28,
                 exit_when_done: bool = False, train_pid_file: str = "", retry_errors: bool = False,
                 once: bool = False, limit_chunks: int = 0, n_rows: int = ED.N_ROWS, tokens: str = PACKED_TOKENS,
                 split: str = SPLIT, scorer: str = "official", inflight: int = 0, pid_grace: float = 600,
                 task_timeout: float = 900, status_every: float = 30, log=print):
        self.io = Path(io_dir)
        assert self.io.is_absolute() and "=" not in str(self.io), f"io_dir must be absolute without '=': {io_dir}"
        assert 1 <= workers <= MAX_WORKERS, f"workers {workers} not in [1, {MAX_WORKERS}]"
        assert scorer in ("official", "fake"), scorer
        self.world_size, self.workers, self.poll = int(world_size), int(workers), float(poll)
        self.max_epoch = max_epoch
        self.exit_when_done, self.train_pid_file = exit_when_done, train_pid_file
        self.retry_errors, self.once, self.limit_chunks = retry_errors, once, int(limit_chunks)
        self.n_rows, self.split, self.scorer = int(n_rows), split, scorer
        self.inflight_max = int(inflight) or self.workers * 4
        self.pid_grace, self.task_timeout, self.status_every = pid_grace, task_timeout, status_every
        self.log = log
        tt = pd.read_parquet(tokens)
        if "row" in tt.columns:
            assert (tt["row"].to_numpy() == np.arange(len(tt))).all(), "tokens.parquet rows must be 0..N-1"
        assert len(tt) >= self.n_rows, (len(tt), self.n_rows)
        self.tok = tt.token.astype(str).to_numpy()
        self.logs = tt.log.astype(str).to_numpy()
        self.lab = self.io / "lab"
        self.gens: Dict[int, dict] = {}
        self.chunks: Dict[str, dict] = {}               # chunk key (path or retry:path) -> state while open
        self.tasks: deque = deque()                      # (key, token, trajs, mc path) waiting for the pool
        self.inflight: Dict[tuple, float] = {}           # key -> submit time
        self.rows_busy: set = set()                      # (epoch, row) queued / in flight / waiting to be written
        self.traj: Dict[tuple, np.ndarray] = {}          # key -> [32,8,3] of queued / in-flight / unwritten tasks
        self.wbuf: Dict[int, list] = {}                  # epoch -> [(key, labels, ok)] scored, not yet written
        self.wbuf_t: Dict[int, float] = {}
        self.results: "queue.Queue" = queue.Queue()
        self.pool = None
        self.n_chunks_taken = 0
        self.t_start = time.time()
        self.tot = dict(tok_ok=0, tok_err=0, tok_skip=0, chunks=0)
        self.recent: deque = deque(maxlen=400)           # finish times of scored tokens
        self.lags: deque = deque(maxlen=50)
        self.pid_missing_since: Optional[float] = None
        self.exit_reason = ""
        self._last_status = 0.0
        self._stop = False
        self.hard_exit = False                           # set when the pool was killed (main() then os._exit's)
        self._redo_done: set = set()                     # epochs whose DONE.json must be rewritten (retry)

    # ------------------------------------------------------------------- files
    def gen(self, epoch: int) -> dict:
        if epoch not in self.gens:
            mode = "r+" if generation_ready(self.io, epoch) else "w+"
            g = ED.open_generation(self.io, epoch, self.n_rows, mode)
            self.gens[epoch] = g
            self.log(f"[{_t()}] generation ep{epoch:03d} opened ({mode}) rows done "
                     f"{int(np.count_nonzero(np.asarray(g['row_state']) == 1))}")
        return self.gens[epoch]

    def marker_dir(self, epoch: int) -> Path:
        return ED.gen_dir(self.io, epoch) / "chunks"

    def marked(self, epoch: int) -> set:
        d = self.marker_dir(epoch)
        return {f.name for f in os.scandir(d) if f.name.endswith(".json")} if d.is_dir() else set()

    def scan(self) -> List[dict]:
        """Unmarked record chunks not open in this process, oldest epoch first."""
        mk: Dict[int, set] = {}
        out = []
        for c in list_chunks(self.io, self.max_epoch):
            if str(c["path"]) in self.chunks:
                continue
            if c["epoch"] not in mk:
                mk[c["epoch"]] = self.marked(c["epoch"])
            if marker_name(c["rank"], c["name"]) not in mk[c["epoch"]]:
                out.append(c)
        return out

    def _error(self, ep: int, rec: dict):
        p = ED.gen_dir(self.io, ep) / "errors.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(dict(rec, time=_t(), epoch=ep), default=CM._json_default) + "\n")

    # ------------------------------------------------------------------- task flow
    @staticmethod
    def _new_state(ep, rank, name, key, mtime, retry=False) -> dict:
        return dict(epoch=ep, rank=rank, name=name, key=key, mtime=mtime, n=0, n_tasks=0, n_done=0, n_ok=0,
                    n_err=0, n_skip=0, n_bad=0, kd_ok=0, t0=time.time(), cpu=0.0, retry=retry, bad=False)

    def take_chunk(self, c: dict):
        ep, path = int(c["epoch"]), str(c["path"])
        st = self.chunks[path] = self._new_state(ep, int(c["rank"]), c["name"], path, c["mtime"])
        self.n_chunks_taken += 1
        try:
            d = ED.read_rec_chunk(path)
            trajs = np.concatenate([np.asarray(d["cand"], np.float32), np.asarray(d["kd_corr"], np.float32)], 1)
            assert trajs.shape[1] == G_K, f"{trajs.shape[1]} trajectories per token != {G_K}"
        except Exception as e:
            self.log(f"[{_t()}] BAD chunk {path}: {e!r}")
            self._error(ep, dict(chunk=path, rank=st["rank"], i=-1, row=-1, token="", error=f"bad chunk: {e!r}"))
            st["bad"] = True
            self._finish_chunk(st)
            return
        g = self.gen(ep)
        rows = np.asarray(d["row"], np.int64)
        st["n"], st["kd_ok"] = len(rows), int(np.asarray(d["kd_ok"]).sum())
        for i, r in enumerate(rows.tolist()):
            if r < 0 or r >= self.n_rows:
                st["n_bad"] += 1
            elif g["row_state"][r] == 1 or (ep, r) in self.rows_busy:
                st["n_skip"] += 1
                self.tot["tok_skip"] += 1
            else:
                self._queue(ep, path, i, r, trajs[i])
                st["n_tasks"] += 1
        if st["n_tasks"] == 0:
            self._finish_chunk(st)

    def _queue(self, ep: int, ckey: str, i: int, r: int, trajs: np.ndarray):
        key = (ep, ckey, i, r)
        self.rows_busy.add((ep, r))
        self.traj[key] = np.ascontiguousarray(trajs, np.float32)
        lg = self.logs[r]
        mcp = f"fake:{self.tok[r]}" if (self.scorer == "fake" and lg == "__fake__") else \
            str(CM.mc_path(self.split, lg, self.tok[r]))
        self.tasks.append((key, self.tok[r], self.traj[key], mcp))

    def _submit(self):
        while self.tasks and len(self.inflight) < self.inflight_max:
            t = self.tasks.popleft()
            self.inflight[t[0]] = time.time()
            self.pool.apply_async(score_one, (t,), callback=self.results.put,
                                  error_callback=lambda e, k=t[0]: self.results.put(
                                      (k, None, None, f"pool error {e!r}", 0.0, 0.0)))

    def _handle(self, res):
        key, lab, ok, err, s_load, s_score = res
        if key not in self.inflight:
            return                                         # late result of a timed-out task
        del self.inflight[key]
        ep, ckey, i, r = key
        st = self.chunks[ckey]
        st["n_done"] += 1
        st["cpu"] += s_load + s_score
        if lab is None:
            st["n_err"] += 1
            self.tot["tok_err"] += 1
            src = ckey[len(RETRY):] if ckey.startswith(RETRY) else ckey
            self._error(ep, dict(chunk=src, rank=st["rank"], i=i, row=r, token=str(self.tok[r]), error=err,
                                 retry=bool(st["retry"])))
            self.traj.pop(key, None)
            self.rows_busy.discard((ep, r))
        else:
            self.recent.append(time.time())
            self.wbuf.setdefault(ep, []).append((key, lab, ok))
            self.wbuf_t.setdefault(ep, time.time())
        if st["n_done"] == st["n_tasks"]:
            self._write(ep)
            self._finish_chunk(st)
        elif len(self.wbuf.get(ep, ())) >= WRITE_BATCH:
            self._write(ep)

    def _write(self, ep: int):
        """Scored rows of epoch ep -> generation (e2e_data.write_generation_rows: data, flush, row_state, flush)."""
        buf = self.wbuf.pop(ep, [])
        self.wbuf_t.pop(ep, None)
        if not buf:
            return
        rows = np.array([k[3] for k, _, _ in buf], np.int64)
        w = ED.write_generation_rows(self.gen(ep), rows, np.stack([self.traj[k] for k, _, _ in buf]),
                                     np.stack([b[1] for b in buf]), np.stack([b[2] for b in buf]))
        for (key, _, _), wrote in zip(buf, w):
            st = self.chunks[key[1]]
            if wrote:
                st["n_ok"] += 1
                self.tot["tok_ok"] += 1
                if st["retry"]:
                    self._redo_done.add(ep)
            else:
                st["n_skip"] += 1
                self.tot["tok_skip"] += 1
            self.traj.pop(key, None)
            self.rows_busy.discard((ep, key[3]))

    def _write_old(self, max_age: float = WRITE_AGE):
        now = time.time()
        for ep in [e for e, t in self.wbuf_t.items() if now - t > max_age]:
            self._write(ep)

    def _finish_chunk(self, st: dict):
        self.chunks.pop(st["key"], None)
        if st["retry"]:
            return
        ep, now = st["epoch"], time.time()
        sec, lag = now - st["t0"], now - st["mtime"]
        self.lags.append(lag)
        n_sc = st["n_ok"] + st["n_err"]
        chunk_rate = n_sc / max(sec, 1e-9)               # one chunk's own rate (several chunks run concurrently)
        rate = self.tok_s_recent()
        mk = dict(chunk=st["key"], rank=st["rank"], epoch=ep, n=st["n"], n_ok=st["n_ok"], n_err=st["n_err"],
                  n_skip=st["n_skip"], n_bad=st["n_bad"], kd_ok=st["kd_ok"], bad=st["bad"], sec=round(sec, 3),
                  tok_s=round(rate if rate is not None else chunk_rate, 3), chunk_tok_s=round(chunk_rate, 3),
                  cpu_sec=round(st["cpu"], 3), lag_s=round(lag, 1),
                  t_start=st["t0"], t_end=now, pid=os.getpid(), scorer=self.scorer)
        atomic_json(self.marker_dir(ep) / marker_name(st["rank"], st["name"]), mk)
        self.tot["chunks"] += 1

    # ------------------------------------------------------------------- retry
    def queue_retries(self) -> int:
        """Re-score error rows whose row_state is still 0 (errors.jsonl of every generation)."""
        n = 0
        for ed in sorted(self.lab.glob("ep[0-9][0-9][0-9]")):
            ep = int(ed.name[2:])
            ef = ed / "errors.jsonl"
            if not ef.is_file() or (self.max_epoch is not None and ep > self.max_epoch):
                continue
            todo: Dict[str, set] = {}
            for line in ef.read_text().splitlines():
                try:
                    e = json.loads(line)
                    i, r = int(e.get("i", -1)), int(e.get("row", -1))
                except Exception:
                    continue
                if i >= 0 and 0 <= r < self.n_rows:
                    todo.setdefault(e["chunk"], set()).add((i, r, int(e.get("rank", -1))))
            g = self.gen(ep) if todo else None
            for path, items in todo.items():
                if not Path(path).is_file():
                    self.log(f"[{_t()}] retry: chunk {path} gone")
                    continue
                d = ED.read_rec_chunk(path)
                trajs = np.concatenate([np.asarray(d["cand"], np.float32), np.asarray(d["kd_corr"], np.float32)], 1)
                ckey = RETRY + path
                st = self._new_state(ep, sorted(items)[0][2], Path(path).name, ckey, time.time(), retry=True)
                for i, r, _ in sorted(items):
                    if g["row_state"][r] == 1 or (ep, r) in self.rows_busy or int(d["row"][i]) != r:
                        continue
                    self.chunks[ckey] = st
                    self._queue(ep, ckey, i, r, trajs[i])
                    st["n_tasks"] += 1
                    n += 1
        self.log(f"[{_t()}] retry-errors: {n} rows queued")
        return n

    # ------------------------------------------------------------------- generations
    def check_generations(self):
        recs = list_chunks(self.io, self.max_epoch)
        busy = {st["epoch"] for st in self.chunks.values()}
        for ep in sorted({c["epoch"] for c in recs}):
            gd = ED.gen_dir(self.io, ep)
            done_f = gd / "DONE.json"
            if (done_f.is_file() and ep not in self._redo_done) or ep in busy:
                continue
            dones = ED.read_rec_done(self.io, ep)
            if any(r not in dones for r in range(self.world_size)):
                continue
            mk = self.marked(ep)
            if any(marker_name(c["rank"], c["name"]) not in mk for c in recs if c["epoch"] == ep):
                continue                                     # every chunk of E (orphans included) first
            if not all(any(all(marker_name(r, n) in mk for n in dn.get("chunks", [])) for dn in dones[r])
                       for r in range(self.world_size)):
                continue
            g = self.gen(ep) if generation_ready(self.io, ep) else None
            rs = np.asarray(g["row_state"]) if g is not None else np.zeros(self.n_rows, np.uint8)
            ms = [json.loads((self.marker_dir(ep) / m).read_text()) for m in sorted(mk)]
            t0 = min((m.get("t_start", time.time()) for m in ms), default=time.time())
            n_sc = sum(m.get("n_ok", 0) + m.get("n_err", 0) for m in ms)
            err_rows = set()
            ef = gd / "errors.jsonl"
            if ef.is_file():
                for line in ef.read_text().splitlines():
                    try:
                        r = int(json.loads(line).get("row", -1))
                    except Exception:
                        continue
                    if 0 <= r < self.n_rows and rs[r] != 1:
                        err_rows.add(r)
            wall = time.time() - t0
            n_done = int(np.count_nonzero(rs == 1))
            atomic_json(done_f, dict(epoch=ep, n_rows_done=n_done, n_err=len(err_rows), n_chunks=len(mk),
                                     n_rec_tokens=sum(m.get("n", 0) for m in ms), wall_s=round(wall, 1),
                                     tok_s=round(n_sc / max(wall, 1e-9), 2), finished=_t(),
                                     world_size=self.world_size,
                                     rec_done={str(r): [d.get("attempt") for d in dones[r]] for r in sorted(dones)}))
            self._redo_done.discard(ep)
            self.log(f"[{_t()}] generation ep{ep:03d} DONE: rows {n_done} err {len(err_rows)} wall {wall:.0f}s")

    # ------------------------------------------------------------------- status / exit
    def tok_s_recent(self) -> Optional[float]:
        if len(self.recent) < 2:
            return None
        return (len(self.recent) - 1) / max(self.recent[-1] - self.recent[0], 1e-9)

    def write_status(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_status < self.status_every:
            return
        self._last_status = now
        by_ep: Dict[int, list] = {}
        for c in list_chunks(self.io, self.max_epoch):
            by_ep.setdefault(c["epoch"], []).append(c)
        eps: Dict[int, dict] = {ep: dict(chunks=len(cs)) for ep, cs in by_ep.items()}
        for ep, e in eps.items():
            mk = self.marked(ep)
            pend = [c for c in by_ep[ep] if marker_name(c["rank"], c["name"]) not in mk]
            e.update(marked=len(mk), pending=len(pend),
                     oldest_pending_age_s=round(max((now - c["mtime"] for c in pend), default=0.0), 1))
            g = self.gens.get(ep)
            if g is None and generation_ready(self.io, ep):
                try:
                    g = ED.open_generation(self.io, ep, self.n_rows, "r")
                except Exception:
                    g = None
            e["rows_done"] = int(np.count_nonzero(np.asarray(g["row_state"]) == 1)) if g is not None else 0
            e["done"] = (ED.gen_dir(self.io, ep) / "DONE.json").is_file()
        backlog = sum(e["pending"] for e in eps.values())
        unread = max(0, backlog - sum(1 for k in self.chunks if not k.startswith(RETRY)))
        rate = self.tok_s_recent()
        st = dict(time=_t(), pid=os.getpid(), scorer=self.scorer, workers=self.workers,
                  uptime_s=round(now - self.t_start, 1), recent_tok_s=None if rate is None else round(rate, 2),
                  total_tok_s=round(self.tot["tok_ok"] / max(now - self.t_start, 1e-9), 2),
                  tokens_ok=self.tot["tok_ok"], tokens_err=self.tot["tok_err"], tokens_skip=self.tot["tok_skip"],
                  chunks_marked_this_run=self.tot["chunks"], backlog_chunks=backlog,
                  backlog_tokens_est=len(self.tasks) + len(self.inflight) + unread * 64,
                  inflight=len(self.inflight), queued_tasks=len(self.tasks),
                  lag_s_recent=round(float(np.mean(self.lags)), 1) if self.lags else None,
                  train_done=(self.io / "TRAIN_DONE").is_file(), epochs={str(k): v for k, v in sorted(eps.items())})
        atomic_json(self.lab / "status.json", st)
        self.log(f"[{_t()}] status: ok {st['tokens_ok']} err {st['tokens_err']} skip {st['tokens_skip']} | "
                 f"{st['recent_tok_s']} tok/s | backlog {backlog} chunks | inflight {len(self.inflight)} | "
                 f"lag {st['lag_s_recent']} s | ep marked/chunks: "
                 + " ".join(f"{k}:{v['marked']}/{v['chunks']}{'D' if v['done'] else ''}" for k, v in sorted(eps.items())))

    def train_finished(self) -> Optional[int]:
        """0 = TRAIN_DONE, 2 = train pid dead (or its pid file missing beyond --pid-grace), None = running."""
        if (self.io / "TRAIN_DONE").is_file():
            return 0
        if not self.train_pid_file:
            return None
        p = Path(self.train_pid_file)
        if not p.is_file():
            self.pid_missing_since = self.pid_missing_since or time.time()
            return 2 if time.time() - self.pid_missing_since > self.pid_grace else None
        self.pid_missing_since = None
        try:
            pid = int(p.read_text().split()[0])
        except Exception:
            return None
        return None if _pid_alive(pid) else 2

    # ------------------------------------------------------------------- main loop
    def _drain(self, timeout: float):
        try:
            self._handle(self.results.get(timeout=timeout))
        except queue.Empty:
            pass
        while True:
            try:
                self._handle(self.results.get_nowait())
            except queue.Empty:
                break
        now = time.time()
        for k, t in list(self.inflight.items()):
            if now - t > self.task_timeout:
                self._handle((k, None, None, f"task timeout {self.task_timeout}s", 0.0, 0.0))
        self._write_old()

    def _kill_pool(self) -> None:
        """Stop the pool without Pool.terminate()/join() (deadlock when a worker died holding the task-queue lock):
        stop the worker-maintenance thread, then SIGKILL every child process of this process and reap it."""
        try:
            from multiprocessing.pool import TERMINATE
            self.pool._state = TERMINATE
            self.pool._worker_handler._state = TERMINATE
        except Exception:                       # pragma: no cover
            pass
        time.sleep(0.3)                         # let the maintenance loop see the state (it polls every 0.1 s)
        me = os.getpid()
        kids = set()
        for p in list(getattr(self.pool, "_pool", []) or []):
            kids.add(int(p.pid))
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                st = (d / "stat").read_text()
                if int(st.rsplit(")", 1)[1].split()[1]) == me:
                    kids.add(int(d.name))
            except Exception:
                continue
        for pid in kids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in kids:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        self.log(f"[{_t()}] pool workers killed ({len(kids)} children)")

    def run(self) -> int:
        import fcntl
        import multiprocessing as mp
        self.lab.mkdir(parents=True, exist_ok=True)
        lockf = open(self.lab / ".labeler.lock", "a+")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lockf.close()
            self.log(f"[{_t()}] another labeler holds {self.lab / '.labeler.lock'}; exiting")
            self.exit_reason = "locked"
            return 3
        lockf.seek(0)
        lockf.truncate()
        lockf.write(f"{os.getpid()}\n")
        lockf.flush()
        try:
            os.nice(5)
        except Exception:
            pass
        self.log(f"[{_t()}] labeler pid {os.getpid()} io_dir {self.io} world {self.world_size} workers {self.workers} "
                 f"scorer {self.scorer} max_epoch {self.max_epoch} once {self.once} exit_when_done "
                 f"{self.exit_when_done}")
        self.pool = mp.get_context("fork").Pool(self.workers, initializer=_init_worker, initargs=(self.scorer,),
                                                maxtasksperchild=2560)
        code, last_scan = 0, 0.0
        try:
            if self.retry_errors:
                self.queue_retries()
            while not self._stop:
                now = time.time()
                room = not (self.limit_chunks and self.n_chunks_taken >= self.limit_chunks)
                if room and len(self.tasks) < self.inflight_max and \
                        (now - last_scan > min(self.poll, 10.0) or not self.inflight):
                    for c in self.scan():
                        if (self.limit_chunks and self.n_chunks_taken >= self.limit_chunks) or \
                                len(self.tasks) >= 4 * self.inflight_max:
                            break
                        self.take_chunk(c)
                    last_scan = now
                self._submit()
                if self.inflight:
                    self._drain(1.0)
                    self.write_status()
                    continue
                if self.tasks:
                    continue
                for ep in list(self.wbuf):
                    self._write(ep)
                for st in [s for s in self.chunks.values() if s["n_done"] == s["n_tasks"]]:
                    self._finish_chunk(st)
                self.check_generations()
                self.write_status(force=True)
                if self.once or not room:
                    self.exit_reason = "once" if self.once else "limit-chunks"
                    break
                if self.exit_when_done:
                    f = self.train_finished()
                    if f is not None and not self.scan():
                        code, self.exit_reason = f, ("train done" if f == 0 else "train pid dead")
                        break
                _sleep(self.poll, lambda: self._stop)
            if self._stop:
                code, self.exit_reason = 143, "SIGTERM"
        finally:
            if not self.inflight and not self._stop and code in (0, 2):
                self.pool.close()
                self.pool.join()
            else:
                self.hard_exit = True
                self._kill_pool()
            for ep in list(self.wbuf):          # scored rows are kept (their chunks stay unmarked -> rescanned)
                try:
                    self._write(ep)
                except Exception as e:          # pragma: no cover
                    self.log(f"final write ep{ep} failed {e!r}")
            try:
                self.write_status(force=True)
            except Exception as e:              # pragma: no cover
                self.log(f"status write failed {e!r}")
            fcntl.flock(lockf, fcntl.LOCK_UN)
            lockf.close()
        self.log(f"[{_t()}] labeler exit {code} ({self.exit_reason}); ok {self.tot['tok_ok']} err "
                 f"{self.tot['tok_err']} skip {self.tot['tok_skip']} chunks {self.tot['chunks']} "
                 f"wall {time.time() - self.t_start:.0f}s")
        return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--io-dir", required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--max-epoch", type=int, default=28, help="ignore record epochs above this (-1 = no limit)")
    ap.add_argument("--exit-when-done", action="store_true")
    ap.add_argument("--train-pid-file", default="")
    ap.add_argument("--pid-grace", type=float, default=600)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--inflight", type=int, default=0, help="max tasks in the pool (default 4 x workers)")
    ap.add_argument("--task-timeout", type=float, default=900)
    ap.add_argument("--status-every", type=float, default=30)
    ap.add_argument("--n-rows", type=int, default=ED.N_ROWS)
    ap.add_argument("--tokens", default=PACKED_TOKENS, help="row -> token/log table (packed tokens.parquet)")
    ap.add_argument("--split", default=SPLIT, help="metric-cache split (common.mc_path)")
    ap.add_argument("--scorer", default="official", choices=("official", "fake"), help="'fake' = tests only")
    ap.add_argument("--pid-file", default="")
    ap.add_argument("--exit-file", default="")
    a = ap.parse_args(argv)
    io_dir = Path(a.io_dir)
    exit_file = Path(a.exit_file) if a.exit_file else io_dir.parent / "labeler" / "labeler.exit"
    if a.pid_file:
        Path(a.pid_file).parent.mkdir(parents=True, exist_ok=True)
        Path(a.pid_file).write_text(f"{os.getpid()}\n")
    lab = Labeler(io_dir, a.world_size, a.workers, a.poll, None if a.max_epoch < 0 else a.max_epoch,
                  a.exit_when_done, a.train_pid_file, a.retry_errors, a.once, a.limit_chunks, a.n_rows, a.tokens,
                  a.split, a.scorer, a.inflight, a.pid_grace, a.task_timeout, a.status_every,
                  log=lambda s: print(s, flush=True))

    def _term(signum, frame):
        lab._stop = True
    signal.signal(signal.SIGTERM, _term)
    print("navsim", LCD.navsim.__file__, "| score_trajectories", ST.__file__, "| e2e_data", ED.__file__, flush=True)
    code = 1
    try:
        code = lab.run()
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        if a.exit_file or exit_file.parent.is_dir():
            exit_file.parent.mkdir(parents=True, exist_ok=True)
            exit_file.write_text(f"{code}\n")
    if lab.hard_exit:      # skip multiprocessing's atexit pool finaliser (it would block on the dead workers' lock)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


if __name__ == "__main__":
    sys.exit(main())
