#!/usr/bin/env python
"""Token table of a split, from the official scene filter and log split in this repo.

  derive_tokens.py --split {navtrain_train,navtrain_val,navtest} --out FILE [--logs DIR] [--workers 6]

navtrain_train and navtrain_val: the navtrain filter on the logs in train_logs and val_logs, which is the set the training
dataset builds.  navtest: the navtest filter on the test logs.  Columns token, log, frame_idx, city; rows by sorted log, then
log order.  The result must equal the pins and the copy tracked in resources/km_corridor/packed, or nothing is written.
An existing file is never overwritten.  CPU only, ~1 min.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from navsim.common import ssr_paths as SP  # noqa: E402

os.environ.setdefault("NUPLAN_MAPS_ROOT", SP.NUPLAN_MAPS_ROOT)        # read when nuplan is imported
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

SF_DIR = ROOT / "navsim/planning/script/config/common/scene_filter"
LOG_SPLIT = ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
TRACKED = ROOT / "resources/km_corridor/packed/{split}/tokens.parquet"
SPLITS = {"navtrain_train": ("navtrain.yaml", "train_logs"), "navtrain_val": ("navtrain.yaml", "val_logs"),
          "navtest": ("navtest.yaml", None)}
# n, sha256("\n".join(sorted(tokens)))[:16], sha256(np.array(tokens, "U16"))[:16]
PINS = {"navtrain_train": (85109, "c3fa309cbde971e8", "0b56b9aecbe1f094"),
        "navtrain_val": (18179, "05b78ec43b95d954", "0fea22c41a4106b0"),
        "navtest": (12146, None, None)}


def filter_of(split: str) -> Tuple[dict, List[str]]:
    """Scene filter dict and the sorted logs of this split."""
    sf = yaml.safe_load((SF_DIR / SPLITS[split][0]).read_text())
    key = SPLITS[split][1]
    logs = set(sf["log_names"])
    if key is not None:
        logs &= set(yaml.safe_load(LOG_SPLIT.read_text())[key])
    return sf, sorted(logs)


def rows_of_log(log: str, sf: dict, logs_dir: str) -> List[tuple]:
    from navsim.common.dataclasses import SceneFilter
    from navsim.common import dataloader as DL
    DL.tqdm = lambda x, **k: x
    f = SceneFilter(num_history_frames=sf["num_history_frames"], num_future_frames=sf["num_future_frames"],
                    frame_interval=sf["frame_interval"], has_route=sf["has_route"], max_scenes=None,
                    log_names=[log], tokens=sf.get("tokens"))
    out = []
    for t, fl in DL.filter_scenes(Path(logs_dir), f).items():
        fr = fl[sf["num_history_frames"] - 1]
        out.append((t, log, int(fr.get("frame_idx", -1)), fr.get("map_location")))
    return out


def derive(split: str, logs_dir: Path, workers: int = 6) -> pd.DataFrame:
    sf, logs = filter_of(split)
    with Pool(workers) as p:
        res = p.map(functools.partial(rows_of_log, sf=sf, logs_dir=str(logs_dir)), logs, chunksize=4)
    df = pd.DataFrame([r for rr in res for r in rr], columns=["token", "log", "frame_idx", "city"])
    if df.token.duplicated().any():
        raise SystemExit(f"{split}: duplicate tokens")
    return df


def compare(split: str, df: pd.DataFrame, pins: Optional[tuple] = None, tracked: Optional[Path] = None) -> List[str]:
    n, sorted_sha, order_sha = PINS[split] if pins is None else pins
    toks = df.token.astype(str).tolist()
    bad = []
    if len(toks) != n:
        bad.append(f"{len(toks)} tokens, expected {n}")
    if sorted_sha is not None:
        got = hashlib.sha256("\n".join(sorted(toks)).encode()).hexdigest()[:16]
        if got != sorted_sha:
            bad.append(f"sorted token sha16 {got} != {sorted_sha}")
    if order_sha is not None:
        got = hashlib.sha256(np.ascontiguousarray(np.array(toks, "U16")).tobytes()).hexdigest()[:16]
        if got != order_sha:
            bad.append(f"token_sha16 {got} != {order_sha}")
    t = Path(str(TRACKED).format(split=split)) if tracked is None else tracked
    if t.is_file():
        old = pd.read_parquet(t, columns=["token", "log"])
        if old.token.astype(str).tolist() != toks or old.log.astype(str).tolist() != df.log.astype(str).tolist():
            bad.append(f"token or log order differs from {t}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", default="")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    default = (Path(SP.SSR_TEST_DATA_ROOT) / "navsim_logs/test/test" if a.split == "navtest"
               else Path(SP.OPENSCENE_DATA_ROOT) / "navsim_logs/trainval")
    df = derive(a.split, Path(a.logs) if a.logs else default, a.workers)
    bad = compare(a.split, df)
    if bad:
        raise SystemExit(f"{a.split}: " + "; ".join(bad))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    print(f"{a.split}: {len(df)} tokens, {df.log.nunique()} logs, equal to the pins -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
