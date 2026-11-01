#!/usr/bin/env python
"""CK2 e2e background labeler: official PDM labels (score_token, LQR) of the 96 trajectories per token recorded in training.

  labeler2.py --io-dir <RUN>/ck_e2e2 [--world-size 2] [--workers 32] [--poll 30] [--max-epoch 28] [--exit-when-done]
              [--train-pid-file <RUN>/launch/train.pid] [--retry-errors] [--once] [--limit-chunks N] [--skip-invalid]
              [--pid-file F] [--exit-file F]

= tools/ck/e2e/labeler.py (class Labeler subclassed; lock, markers, errors.jsonl, DONE.json, status.json, exit codes,
  hard-exit pool kill, fake scorer and score_one are inherited unchanged) with the CK2 formats:
Input   rec/ep{E}/r{R}/c_<attempt>_<seq>.npz written by e2e_data2.CandRecorder2 (fmt 2): traj f32 [n, 96, 8, 3]
        (c = k * 6 + v: the student's top-16 x (id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5), exactly the f32 values of the
        training step), valid bool [n, 96].  A CK1 chunk (fmt != 2) is a BAD chunk (marker bad, errors.jsonl).
        Per token ONE score_token call on its finite trajectories (all 96 -- invalid lateral duplicates included,
        user "label ALL 96"; --skip-invalid leaves the valid == False columns unscored, off by default, NU23).
Output  lab/ep{E}/{traj, labels, cand_ok, valid, row_state}.npy + meta.json (format 'ck2_96') via
        e2e_data2.open_generation2 / write_generation_rows2: traj = the recorded trajectories (bitwise), valid = the
        recorded validity, labels / cand_ok from score_one (ok = SCORE_COLS finite), row_state 1 written last.
Defaults --workers 32 (user), --world-size 2 (2-GPU run), --max-epoch 28.  CPU only, OMP_NUM_THREADS=1, os.nice(5).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.e2e import labeler as L1  # noqa: E402  (pins navsim + the official scorer)
from navsim.agents.para_ssr.ck import e2e_data as ED  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402

import numpy as np  # noqa: E402

G_K = D2.G_K
GEN_KEYS = D2.GEN2_KEYS
DEFAULT_WORKERS = 32
DEFAULT_WORLD = 2
DEFAULT_MAX_EPOCH = 28
_t = L1._t


class Labeler2(L1.Labeler):
    """Labeler for 96-column records / generations (see the module docstring)."""

    def __init__(self, io_dir, world_size: int = DEFAULT_WORLD, workers: int = DEFAULT_WORKERS,
                 skip_invalid: bool = False, **kw):
        kw.setdefault("max_epoch", DEFAULT_MAX_EPOCH)
        super().__init__(io_dir, world_size, workers, **kw)
        self.skip_invalid = bool(skip_invalid)
        self.valid: Dict[tuple, np.ndarray] = {}         # key -> [96] bool, alongside self.traj

    # ------------------------------------------------------------------- files
    def gen(self, epoch: int) -> dict:
        if epoch not in self.gens:
            mode = "r+" if L1.generation_ready(self.io, epoch) else "w+"
            g = D2.open_generation2(self.io, epoch, self.n_rows, mode)
            self.gens[epoch] = g
            self.log(f"[{_t()}] generation ep{epoch:03d} opened ({mode}, {D2.GEN2_FORMAT}) rows done "
                     f"{int(np.count_nonzero(np.asarray(g['row_state']) == 1))}")
        return self.gens[epoch]

    # ------------------------------------------------------------------- task flow
    def _queue(self, ep: int, ckey: str, i: int, r: int, trajs: np.ndarray, valid: Optional[np.ndarray] = None):
        key = (ep, ckey, i, r)
        self.rows_busy.add((ep, r))
        self.traj[key] = np.ascontiguousarray(trajs, np.float32)            # written to the generation as recorded
        self.valid[key] = (np.ones(len(trajs), bool) if valid is None else np.asarray(valid, bool).copy())
        task_traj = self.traj[key]
        if self.skip_invalid and not self.valid[key].all():
            task_traj = np.where(self.valid[key][:, None, None], task_traj, np.float32(np.nan)).astype(np.float32)
        lg = self.logs[r]
        mcp = f"fake:{self.tok[r]}" if (self.scorer == "fake" and lg == "__fake__") else \
            str(L1.CM.mc_path(self.split, lg, self.tok[r]))
        self.tasks.append((key, self.tok[r], task_traj, mcp))

    @staticmethod
    def _read(path):
        d = D2.read_rec2_chunk(path)
        trajs = np.asarray(d["traj"], np.float32)
        assert trajs.ndim == 4 and trajs.shape[1:] == (G_K, 8, 3), f"{trajs.shape}: not [n, {G_K}, 8, 3]"
        return d, trajs, np.asarray(d["valid"], bool)

    def take_chunk(self, c: dict):
        ep, path = int(c["epoch"]), str(c["path"])
        st = self.chunks[path] = self._new_state(ep, int(c["rank"]), c["name"], path, c["mtime"])
        self.n_chunks_taken += 1
        try:
            d, trajs, valid = self._read(path)
        except Exception as e:
            self.log(f"[{_t()}] BAD chunk {path}: {e!r}")
            self._error(ep, dict(chunk=path, rank=st["rank"], i=-1, row=-1, token="", error=f"bad chunk: {e!r}"))
            st["bad"] = True
            self._finish_chunk(st)
            return
        g = self.gen(ep)
        rows = np.asarray(d["row"], np.int64)
        st["n"], st["kd_ok"] = len(rows), int(valid.sum())          # marker 'kd_ok' = number of valid columns here
        for i, r in enumerate(rows.tolist()):
            if r < 0 or r >= self.n_rows:
                st["n_bad"] += 1
            elif g["row_state"][r] == 1 or (ep, r) in self.rows_busy:
                st["n_skip"] += 1
                self.tot["tok_skip"] += 1
            else:
                self._queue(ep, path, i, r, trajs[i], valid[i])
                st["n_tasks"] += 1
        if st["n_tasks"] == 0:
            self._finish_chunk(st)

    def _handle(self, res):
        key = res[0]
        super()._handle(res)
        if key not in self.traj:                    # error path (or written): drop the companion validity
            self.valid.pop(key, None)

    def _write(self, ep: int):
        """Scored rows of epoch ep -> generation (write_generation_rows2: data, flush, row_state, flush)."""
        buf = self.wbuf.pop(ep, [])
        self.wbuf_t.pop(ep, None)
        if not buf:
            return
        rows = np.array([k[3] for k, _, _ in buf], np.int64)
        w = D2.write_generation_rows2(self.gen(ep), rows, np.stack([self.traj[k] for k, _, _ in buf]),
                                      np.stack([b[1] for b in buf]), np.stack([b[2] for b in buf]),
                                      np.stack([self.valid[k] for k, _, _ in buf]))
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
            self.valid.pop(key, None)
            self.rows_busy.discard((ep, key[3]))

    # ------------------------------------------------------------------- retry
    def queue_retries(self) -> int:
        """Re-score error rows whose row_state is still 0 (errors.jsonl of every generation)."""
        import json
        n = 0
        for edir in sorted(self.lab.glob("ep[0-9][0-9][0-9]")):
            ep = int(edir.name[2:])
            ef = edir / "errors.jsonl"
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
                d, trajs, valid = self._read(path)
                ckey = L1.RETRY + path
                st = self._new_state(ep, sorted(items)[0][2], Path(path).name, ckey, time.time(), retry=True)
                for i, r, _ in sorted(items):
                    if g["row_state"][r] == 1 or (ep, r) in self.rows_busy or int(d["row"][i]) != r:
                        continue
                    self.chunks[ckey] = st
                    self._queue(ep, ckey, i, r, trajs[i], valid[i])
                    st["n_tasks"] += 1
                    n += 1
        self.log(f"[{_t()}] retry-errors: {n} rows queued")
        return n

    # ------------------------------------------------------------------- status
    def write_status(self, force: bool = False):
        """Base status, with every ready generation opened through open_generation2 (the base reader would open
        it with the 32-column CK1 layout and report rows_done 0)."""
        if force or time.time() - self._last_status >= self.status_every:
            eps = {c["epoch"] for c in L1.list_chunks(self.io, self.max_epoch)}
            for ep in sorted(eps):
                if ep not in self.gens and L1.generation_ready(self.io, ep):
                    try:
                        self.gen(ep)
                    except Exception as e:      # noqa: BLE001
                        self.log(f"[{_t()}] status: cannot open generation ep{ep:03d}: {e!r}")
        return super().write_status(force)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--io-dir", required=True)
    ap.add_argument("--world-size", type=int, default=DEFAULT_WORLD)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--max-epoch", type=int, default=DEFAULT_MAX_EPOCH, help="ignore record epochs above this (-1 = no limit)")
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
    ap.add_argument("--tokens", default=L1.PACKED_TOKENS, help="row -> token/log table (packed tokens.parquet)")
    ap.add_argument("--split", default=L1.SPLIT, help="metric-cache split (common.mc_path)")
    ap.add_argument("--scorer", default="official", choices=("official", "fake"), help="'fake' = tests only")
    ap.add_argument("--skip-invalid", action="store_true",
                    help="leave valid == False (lateral duplicate) columns unscored (default: label all 96)")
    ap.add_argument("--pid-file", default="")
    ap.add_argument("--exit-file", default="")
    a = ap.parse_args(argv)
    io_dir = Path(a.io_dir)
    exit_file = Path(a.exit_file) if a.exit_file else io_dir.parent / "labeler" / "labeler.exit"
    if a.pid_file:
        Path(a.pid_file).parent.mkdir(parents=True, exist_ok=True)
        Path(a.pid_file).write_text(f"{os.getpid()}\n")
    lab = Labeler2(io_dir, a.world_size, a.workers, skip_invalid=a.skip_invalid, poll=a.poll,
                   max_epoch=None if a.max_epoch < 0 else a.max_epoch, exit_when_done=a.exit_when_done,
                   train_pid_file=a.train_pid_file, retry_errors=a.retry_errors, once=a.once,
                   limit_chunks=a.limit_chunks, n_rows=a.n_rows, tokens=a.tokens, split=a.split, scorer=a.scorer,
                   inflight=a.inflight, pid_grace=a.pid_grace, task_timeout=a.task_timeout,
                   status_every=a.status_every, log=lambda s: print(s, flush=True))

    def _term(signum, frame):
        lab._stop = True
    signal.signal(signal.SIGTERM, _term)
    print("navsim", L1.LCD.navsim.__file__, "| score_trajectories", L1.ST.__file__, "| e2e_data2", D2.__file__,
          "| labeler2 skip_invalid", lab.skip_invalid, flush=True)
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
