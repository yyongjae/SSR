"""Shared helpers of the CK pipeline entry points (tools/ck/*.py).  Pipeline-owned; no model code here.

pin()          : navsim pinned to the SSR-ck worktree (contract env.navsim_pin); call before importing navsim users.
ck_data()      : CK data root (core constants.CK_DATA; env CK_DATA_ROOT overrides it for smoke / tests ONLY when the
                 core constants do not already honour it -- see resolve note in ck_data()).
gpu_guard()    : refuse any physical GPU outside 0-3 (CUDA_VISIBLE_DEVICES must be one of 0,1,2,3 when CUDA is used).
gpu_guard_ddp(): train_ck2 (DDP): 1..2 distinct GPUs of 0-3, one process per visible GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

CK_REPO = str(Path(__file__).resolve().parents[2])   # repo root of this worktree
ALLOWED_GPUS = ("0", "1", "2", "3")


def pin() -> None:
    if sys.path[0] != CK_REPO:
        sys.path.insert(0, CK_REPO)
    import navsim  # noqa: F401
    assert navsim.__file__.startswith(CK_REPO), f"navsim not pinned to {CK_REPO}: {navsim.__file__}"
    import tools.ck  # noqa: F401
    assert tools.ck.__file__.startswith(CK_REPO), tools.ck.__file__


def ck_data() -> Path:
    """CK_DATA root.  The core constant is the authority; CK_DATA_ROOT (env) is honoured for smoke/test isolation."""
    env = os.environ.get("CK_DATA_ROOT")
    if env:
        return Path(env)
    from navsim.agents.para_ssr.ck import constants as C
    return Path(C.CK_DATA)


def gpu_guard(device: str) -> None:
    """device 'cuda' needs CUDA_VISIBLE_DEVICES to be exactly one allowed GPU (single-GPU processes, no DDP)."""
    if not str(device).startswith("cuda"):
        return
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [x.strip() for x in vis.split(",") if x.strip()]
    if len(ids) != 1 or ids[0] not in ALLOWED_GPUS:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={vis!r}: CK jobs run on exactly one GPU of {ALLOWED_GPUS}")


def gpu_guard_ddp(device: str, world_size: int, max_gpus: int = 2) -> None:
    """DDP variant (train_ck2): CUDA_VISIBLE_DEVICES = 1 .. max_gpus distinct GPUs of ALLOWED_GPUS and exactly one
    process per visible GPU (world_size == number of visible GPUs).  gpu_guard above is unchanged (single-GPU jobs)."""
    if not str(device).startswith("cuda"):
        return
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [x.strip() for x in vis.split(",") if x.strip()]
    if not (1 <= len(ids) <= max_gpus) or len(set(ids)) != len(ids) or any(i not in ALLOWED_GPUS for i in ids):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={vis!r}: CK2 jobs run on 1..{max_gpus} distinct GPUs of {ALLOWED_GPUS}")
    if len(ids) != int(world_size):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={vis!r} has {len(ids)} GPU(s) but WORLD_SIZE={world_size}: "
                         "launch one process per visible GPU (torchrun --nproc_per_node=<#GPUs>)")


def sha256_file(p, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", CK_REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def write_json(path, obj, indent: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=indent, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def read_json(path, default=None):
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text())


def log_jsonl(path, rec: Dict, echo: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=_json_default)
    with open(path, "a") as f:
        f.write(line + "\n")
    if echo:
        print(line, flush=True)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def open_memmap(path, shape, dtype, fill=None):
    """Create (atomically) or open r+ an .npy memmap of the given shape / dtype; a mismatch raises."""
    import numpy as np
    path = Path(path)
    if path.is_file():
        m = np.load(path, mmap_mode="r+")
        if tuple(m.shape) != tuple(shape) or m.dtype != np.dtype(dtype):
            raise ValueError(f"{path}: existing {m.shape} {m.dtype} != {tuple(shape)} {np.dtype(dtype)}")
        return m
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npy")
    m = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=tuple(shape))
    if fill is not None:
        m[...] = fill
    m.flush()
    del m
    try:
        os.link(tmp, path)          # atomic create-if-absent (another shard may have won the race)
    except FileExistsError:
        pass
    finally:
        tmp.unlink(missing_ok=True)
    return open_memmap(path, shape, dtype)


class FileLock:
    """fcntl advisory lock on <path> (shards creating shared outputs)."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a+")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
        return False


def shard_rows(n: int, shard: int, nshard: int):
    import numpy as np
    if not (0 <= shard < max(1, nshard)):
        raise SystemExit(f"--shard {shard} not in [0, {nshard})")
    return np.arange(n, dtype=np.int64)[shard::max(1, nshard)]


def log_stratified_rows(logs, n: int, seed: int = 0):
    """<= n row indices, the same fraction of every log (at least 1 per log while budget lasts), seeded; sorted."""
    import numpy as np
    logs = np.asarray(logs)
    N = len(logs)
    if n <= 0 or n >= N:
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    frac = n / N
    out = []
    for lg in np.unique(logs):
        idx = np.flatnonzero(logs == lg)
        k = int(round(frac * len(idx)))
        if k > 0:
            out.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    rows = np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)
    if len(rows) > n:
        rows = np.sort(rng.choice(rows, size=n, replace=False))
    return rows.astype(np.int64)
