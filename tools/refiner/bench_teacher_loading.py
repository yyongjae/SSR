"""Teacher-BEV loading throughput: ReSMap (sharded .npy memmap, per-worker shard cache) vs BEVFusion (npz per token).

DataLoader-like: torch DataLoader, `--workers` worker processes, batch 8 tokens, dataset item = S-grid float16 bev.
Pass 1 reads tokens not read before in this run (disk-bound unless the OS page cache already holds them); pass 2
repeats the same tokens (page-cache warm).  Disjoint random token sets per cache.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python tools/refiner/bench_teacher_loading.py --n 480 --workers 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache  # noqa: E402


class BevDS:
    def __init__(self, cache, tokens):
        self.cache, self.tokens = cache, list(tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        import torch

        return torch.from_numpy(self.cache.load_bev(self.tokens[i], s_grid=True))


def run(cache, tokens, workers, bs=8):
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(BevDS(cache, tokens), batch_size=bs, shuffle=False, num_workers=workers,
                    prefetch_factor=2 if workers else None, persistent_workers=False)
    t = time.perf_counter()
    n = 0
    for b in dl:
        assert b.shape[1:] == (256, 50, 100) and b.dtype == torch.float16
        n += b.shape[0]
    return (time.perf_counter() - t) / n * 1e3


def single(cache, tokens):
    t = time.perf_counter()
    for tk in tokens:
        cache.load_bev(tk)
    return (time.perf_counter() - t) / len(tokens) * 1e3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.workers > 4:
        raise SystemExit("--workers <= 4")
    tk = pd.read_parquet(RD.DATA_ROOT / "splits" / "train_trainlogs.parquet").token.values
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(tk, 4 * a.n, replace=False)
    sets = {"resmap": pick[: a.n], "bevfusion": pick[a.n: 2 * a.n]}
    single_sets = {"resmap": pick[2 * a.n: 2 * a.n + a.n // 4], "bevfusion": pick[3 * a.n: 3 * a.n + a.n // 4]}
    caches = {"resmap": ResmapCache.for_subset("navtrain"), "bevfusion": RD.TeacherCache.for_subset("navtrain")}
    t = time.perf_counter()
    _ = len(caches["resmap"].index)                  # index.json parsed once, before timing (inherited by workers)
    res = {"resmap_index_load_s": round(time.perf_counter() - t, 3)}
    for name, c in caches.items():
        r = dict(single_process_cold_ms=single(c, single_sets[name]),
                 single_process_warm_ms=single(c, single_sets[name]))
        r[f"loader_w{a.workers}_pass1_ms"] = run(c, sets[name], a.workers)
        r[f"loader_w{a.workers}_pass2_ms"] = run(c, sets[name], a.workers)
        res[name] = {k: round(v, 2) for k, v in r.items()}
        print(name, res[name], flush=True)
    res["config"] = dict(n=a.n, workers=a.workers, batch=8, seed=a.seed, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
