"""CK data helpers: navsim pin, split tables, on-disk layout under CK_DATA, atomic writers.

Shared constants (CK_DATA, DATA_ROOT, SPLITS, K_CAND, CK_KEYS, LABEL_COLS) come from the core package
navsim/agents/para_ssr/ck/constants.py and are never redefined here.  Data-only paths (metric-cache roots, test
logs ...) live here.

Layout (root = CK_DATA by default; the smoke uses CK_DATA/smoke with the same sub-layout):
  <root>/v2dump/<split>/{bev,plan}/<tok[:2]>/<tok>.{npy,npz}, stats_shard<i>.jsonl, meta_shard<i>.json
  <root>/packed/<split>/            pack_v2.py
  <root>/labels/<split>/<name>/     label_cands.py
  <root>/lead/<split>.parquet       lead_split.py
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

CK = str(Path(__file__).resolve().parents[3])   # repo root of this worktree


def pin_navsim():
    """Import navsim from the SSR-ck worktree (must run before tools/refiner modules that prepend the main checkout)."""
    if CK not in sys.path:
        sys.path.insert(0, CK)
    import navsim
    assert navsim.__file__.startswith(CK), f"navsim not pinned to {CK}: {navsim.__file__}"
    return navsim


pin_navsim()
from navsim.agents.para_ssr.ck import constants as C  # noqa: E402  (core-owned shared constants)

CK_DATA = Path(C.CK_DATA)
DATA_ROOT = Path(C.DATA_ROOT)
SPLITS = C.SPLITS
K_CAND = int(C.K_CAND)
CK_KEYS = tuple(C.CK_KEYS)
LABEL_COLS = tuple(C.LABEL_COLS)

# taken from core constants (never redefined)
MC_ROOT = {k: Path(v["metric_cache"]) for k, v in SPLITS.items()}
NAVTEST_CSV = Path(C.V2_NAVTEST_CSV)
NAVTEST_CSV_PDMS = float(C.V2_NAVTEST_PDMS)
NAVTEST_CSV_TOL = float(C.V2_NAVTEST_TOL)
V2_CKPT = Path(C.V2_CKPT)
V2_TRAIN_CFG = Path(C.V2_TRAIN_CFG)
V2_CKPT_SHA16 = str(C.V2_CKPT_SHA16)
LEAD_SRC = {k: Path(v) for k, v in C.LEAD_SRC.items()}
LEAD_COLS = ("token", "log", "has_lead", "D_1", "D_strong", "D_hard", "lead_v", "lead_gap", "thw", "ttc0", "censored")


# ----------------------------------------------------------------------------------------------- splits
SCENE_FILTER_DIR = "navsim/planning/script/config/common/scene_filter"


def split(name: str) -> dict:
    """SPLITS[name] with paths normalised: 'same as <other>' values resolved, bare scene-filter file names prefixed
    with the repo scene_filter dir (the contract text uses both shorthands)."""
    if name not in SPLITS:
        raise KeyError(f"unknown split {name!r}; known {sorted(SPLITS)}")
    s = dict(SPLITS[name])
    for k, v in list(s.items()):
        if isinstance(v, str) and v.startswith("same as "):
            s[k] = split(v[len("same as "):].strip())[k]
    sf = s.get("scene_filter")
    if isinstance(sf, str) and "/" not in sf:
        s["scene_filter"] = f"{SCENE_FILTER_DIR}/{sf}"
    return s


def split_tokens(name: str):
    """DataFrame (token, log, city) in the split parquet's row order."""
    import pandas as pd
    s = split(name)
    df = pd.read_parquet(s["tokens"])
    if "city" not in df.columns:
        df["city"] = df["map_location"] if "map_location" in df.columns else ""
    df = df[["token", "log", "city"]].reset_index(drop=True)
    df["token"] = df.token.astype(str)
    assert not df.token.duplicated().any(), name
    return df


def resolve_path(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(CK) / p


# ----------------------------------------------------------------------------------------------- layout
def root_dir(root: Optional[Union[str, Path]] = None) -> Path:
    """Explicit root > env CK_DATA_ROOT (smoke / test isolation, same rule as tools/ck/ckutil.ck_data) > CK_DATA."""
    if root:
        return Path(root)
    env = os.environ.get("CK_DATA_ROOT")
    return Path(env) if env else CK_DATA


def dump_dir(split_name: str, root=None) -> Path:
    return root_dir(root) / "v2dump" / split_name


def bev_path(dump: Path, tok: str) -> Path:
    return Path(dump) / "bev" / tok[:2] / f"{tok}.npy"


def plan_path(dump: Path, tok: str) -> Path:
    return Path(dump) / "plan" / tok[:2] / f"{tok}.npz"


def packed_dir(split_name: str, root=None) -> Path:
    return root_dir(root) / "packed" / split_name


def labels_dir(split_name: str, name: str, root=None) -> Path:
    return root_dir(root) / "labels" / split_name / name


def lead_path(split_name: str, root=None) -> Path:
    return root_dir(root) / "lead" / f"{split_name}.parquet"


def mc_path(split_name: str, log: str, tok: str) -> Path:
    return MC_ROOT[split_name] / log / "unknown" / tok / "metric_cache.pkl"


# ----------------------------------------------------------------------------------------------- io
def _tmp(p: Path) -> Path:
    return p.with_name(f".{p.name}.tmp{os.getpid()}")


def atomic_save_npy(path: Union[str, Path], arr: np.ndarray):
    """np.save to <path> via tmp + os.replace (np.lib.format, loadable with mmap_mode='r')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.lib.format.write_array(f, np.ascontiguousarray(arr), allow_pickle=False)
    os.replace(tmp, path)


def atomic_savez(path: Union[str, Path], **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def atomic_write_json(path: Union[str, Path], obj, indent: int = 1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    tmp.write_text(json.dumps(obj, indent=indent, default=_json_default))
    os.replace(tmp, path)


def atomic_to_parquet(df, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


class MemmapWriter:
    """Writes an .npy through np.lib.format.open_memmap on a tmp file; finalize() flushes and renames atomically."""

    def __init__(self, path: Union[str, Path], dtype, shape, fill=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = _tmp(self.path)
        self.arr = np.lib.format.open_memmap(self.tmp, mode="w+", dtype=dtype, shape=tuple(shape))
        if fill is not None:
            self.arr[...] = fill

    def finalize(self):
        self.arr.flush()
        del self.arr
        os.replace(self.tmp, self.path)


def load_npy(path: Union[str, Path], mmap: bool = True) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


# ----------------------------------------------------------------------------------------------- hashing
def traj_crc(arr: np.ndarray) -> int:
    """crc32 of the float32 bytes of one token's trajectories (shard staleness check)."""
    return int(zlib.crc32(np.ascontiguousarray(np.asarray(arr, np.float32)).tobytes()))


def sha256_file(path: Union[str, Path], block: int = 1 << 24) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha16_array(arr: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def git_head() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", CK, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""
