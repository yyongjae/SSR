"""ReSMap map-teacher cache access for the stage-T refiner (map arm).  Read-only; new module, data.py is unchanged.

Source: /home/external-user/datasets/teacher_cache/resmap (README.md, meta.json; generator
kyungmin/SSR/tools/readout/resmap/cache_teacher_kd.py, sha256 46884c2f...3bb3ff6e, read for reference only).
  root      : NAVSIM navtrain, train_logs ONLY (126,032 frames; every train_logs token of navtrain, no val_logs token)
  navtest/  : the 12,146 navtest tokens
  Each: index.json token -> [shard, row]; <field>/<shard>.npy shards (~2000 frames), read with mmap_mode='r'.
  Fields: bev (256, 100, 50) f16 neck output; seg (4, 200, 100) f16 raw logits (road, walkway, centerline, crosswalk);
          vectors (100, 20, 2) f16 in [0, 1]; scores (100,) f16; labels (100,) i8; props (100,) i16.
  The teacher is camera (CAM_L0/F0/R0) + satellite, temporal (memory bank through each scene), front ROI only.
  NOTE: the teacher was TRAINED on navtrain train_logs, i.e. on every token of the stage-T train and dev pools that
  this arm uses; only navtest outputs are held-out predictions (see report/refiner_T/resmap_axes_{train,navtest}.json).

Axis convention (MEASURED, tools/refiner/validate_resmap_axes.py -> report/refiner_T/resmap_axes_*.json; 300
train_logs tokens incl. 100 curves + 300 navtest tokens; all 8 swap/flip hypotheses vs the stage-T drivable SDF and
route centerline, which are in the N frame = rear axle at t0):
  bev[c, a, b] / seg[k, a, b]: axis a = lateral, a = 0 at y_left = +32 m (LEFT), increasing to the right;
                               axis b = forward, b = 0 at x = 0 (the rear axle), increasing forward.
      bev cell 0.64 m: y_left = 32 - (a + 0.5) * 0.64, x = (b + 0.5) * 0.64
      seg cell 0.32 m: y_left = 32 - (a + 0.5) * 0.32, x = (b + 0.5) * 0.32
  => S grid [C, 50 rows = x forward, 100 cols = y_left descending (col 0 = +32 m)] = bev.transpose(0, 2, 1)
     (pure transpose, no flip, no resample; resmap_to_s_grid).  Same array convention for bev and seg (bev -> seg
     ridge probe, and bev -> drivable-area probe under the 8 hypotheses).
  vectors[..., 0] = x / 32, vectors[..., 1] = (y_left + 32) / 64 (resmap_vectors_to_n).
  Origin = the REAR AXLE (N frame), no offset: scanning a shift of the teacher's metric coordinates (dx -3..+3 m, dy
  -1.5..+1.5 m, 0.1 m steps) every metric peaks at 0.0 (seg road IoU, centerline logit at GT, road-vector |sdf|,
  GT-centerline -> predicted-centerline distance); a parabola through the 3 best vector-metric points puts the optimum
  at +0.002..+0.003 m (dx) / +0.002 m (dy), i.e. |offset| well under 0.05 m.  A camera- or LiDAR-centred origin would
  have shown up as a >1 m peak shift.

Interface mirrors data.TeacherCache: for_subset('navtrain'|'navtest'), has(token), load_bev(token, s_grid=True)
(-> float16 [256, 50, 100] S grid), load(token, dtype) alias, sha_head, root.  It can be passed wherever a TeacherCache
is used (TokenDataset(teacher=...), data.compute_teacher_norm) since those only call load_bev / root / sha_head.
Memmaps are opened lazily and cached per (field, shard) PER PROCESS (the cache is dropped when pickled and rebuilt
when the pid changes, so fork / spawn DataLoader workers each hold their own).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np

# ----------------------------------------------------------------------------------------------- constants
RESMAP_ROOT = Path("/home/external-user/datasets/teacher_cache/resmap")
RESMAP_ROOTS = {"navtrain": RESMAP_ROOT, "navtest": RESMAP_ROOT / "navtest"}
RESMAP_SHA256 = "0eaeda793402a804925b46861a91a3ebe011e608d3b3726604f400be6f296f1a"   # meta.json checkpoint_sha256
RESMAP_SHA_HEAD = RESMAP_SHA256[:16]
RESMAP_CLASSES = ("road", "walkway", "centerline", "crosswalk")
RESMAP_FIELDS = {"bev": ((256, 100, 50), "float16"), "seg": ((4, 200, 100), "float16"),
                 "vectors": ((100, 20, 2), "float16"), "scores": ((100,), "float16"), "labels": ((100,), "int8"),
                 "props": ((100,), "int16")}
RESMAP_PC_RANGE = [0, -32, -3, 32, 32, 5]
RESMAP_ROI_SIZE = [32, 64]
S_H, S_W = 50, 100                  # S grid (adapters.BEV_H, BEV_W)
SEG_H, SEG_W = 100, 200             # seg on the S-grid orientation at 0.32 m
BEV_CELL, SEG_CELL = 0.64, 0.32
LOG_SPLIT_YAML = Path(__file__).resolve().parents[3] / \
    "planning/script/config/training/default_train_val_test_log_split.yaml"
RESMAP_ORIGIN_NOTE = "rear axle; measured |offset| < 0.05 m (report/refiner_T/resmap_axes_*.json origin_scan)"


# ----------------------------------------------------------------------------------------------- transforms
def resmap_to_s_grid(x):
    """ReSMap (..., lateral a [row 0 = +32 m left], forward b [0 = ego]) -> S orientation (..., forward, lateral) with
    col 0 = +32 m left: a pure swap of the last two axes.  bev [256,100,50] -> [256,50,100]; seg [4,200,100] ->
    [4,100,200] (0.32 m).  numpy -> contiguous copy (dtype kept); torch -> transpose(-1, -2)."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.transpose(-1, -2)
    except ImportError:  # pragma: no cover
        pass
    return np.ascontiguousarray(np.swapaxes(np.asarray(x), -1, -2))


def resmap_cell_xy(shape, cell: float):
    """Metric centres (x forward, y_left) [A, B, 2] of a RAW ReSMap (lateral A, forward B) array with square cells."""
    A, B = shape
    y = 32.0 - (np.arange(A) + 0.5) * cell
    x = (np.arange(B) + 0.5) * cell
    return np.stack(np.broadcast_arrays(x[None, :], y[:, None]), -1)


def s_grid_xy(h: int = S_H, w: int = S_W, cell: float = BEV_CELL):
    """Metric centres [h, w, 2] of an S-orientation grid: row r -> x = (r+.5) cell, col c -> y = 32 - (c+.5) cell."""
    x = (np.arange(h) + 0.5) * cell
    y = 32.0 - (np.arange(w) + 0.5) * cell
    return np.stack(np.broadcast_arrays(x[:, None], y[None, :]), -1)


def resmap_vectors_to_n(v):
    """vectors [..., 2] in [0, 1] -> N-frame metres [..., 2] (x = 32 u, y_left = -32 + 64 v)."""
    v = np.asarray(v, np.float64)
    return np.stack([32.0 * v[..., 0], -32.0 + 64.0 * v[..., 1]], -1)


# ----------------------------------------------------------------------------------------------- log split helpers
def train_logs(yaml_path: Union[str, Path] = LOG_SPLIT_YAML) -> set:
    """NAVSIM default_train_val_test_log_split.yaml train_logs (the logs the ReSMap root covers)."""
    import yaml

    with open(yaml_path) as f:
        return set(yaml.safe_load(f)["train_logs"])


def restrict_to_train_logs(df, logs: Optional[Iterable[str]] = None):
    """Rows of a split DataFrame (column 'log') whose log is in train_logs; order and columns kept, index reset."""
    logs = set(logs) if logs is not None else train_logs()
    return df[df.log.isin(logs)].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- cache
class ResmapCache:
    """Read-only access to one ReSMap cache directory, pinned to the meta.json checkpoint sha256."""

    def __init__(self, root: Union[str, Path], expect_sha256: str = RESMAP_SHA256,
                 expect_split: Optional[str] = None):
        root = Path(root)
        mf = root / "meta.json"
        if not mf.is_file():
            raise FileNotFoundError(f"ReSMap cache without meta.json: {mf}")
        meta = json.loads(mf.read_text())
        sha = str(meta.get("checkpoint_sha256", ""))
        if sha != expect_sha256:
            raise ValueError(f"ReSMap cache {root}: checkpoint_sha256 {sha!r} != {expect_sha256!r}")
        if expect_split is not None and str(meta.get("split")) != expect_split:
            raise ValueError(f"ReSMap cache {root}: split {meta.get('split')!r} != {expect_split!r}")
        if list(meta.get("classes", [])) != list(RESMAP_CLASSES):
            raise ValueError(f"ReSMap cache {root}: classes {meta.get('classes')}")
        if list(meta.get("pc_range", [])) != RESMAP_PC_RANGE or list(meta.get("roi_size_m", [])) != RESMAP_ROI_SIZE:
            raise ValueError(f"ReSMap cache {root}: pc_range {meta.get('pc_range')} roi {meta.get('roi_size_m')}")
        for f in ("bev", "seg"):
            t = meta.get("tensors", {}).get(f, {})
            if tuple(t.get("shape", ())) != RESMAP_FIELDS[f][0] or t.get("dtype") != RESMAP_FIELDS[f][1]:
                raise ValueError(f"ReSMap cache {root}: {f} {t}")
        self.root = root
        self.meta = meta
        self.manifest = meta                   # TeacherCache attribute name
        self.sha256 = sha
        self.sha_head = sha[:16]
        self._index: Optional[Dict[str, list]] = None
        self._mm: Dict[tuple, np.ndarray] = {}
        self._pid = os.getpid()

    @classmethod
    def for_subset(cls, subset: str) -> "ResmapCache":
        return cls(RESMAP_ROOTS[subset], expect_split={"navtrain": "train", "navtest": "none"}[subset])

    # pickling (DataLoader workers): ship the index, never the memmaps
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_mm"] = {}
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._pid = os.getpid()

    @property
    def index(self) -> Dict[str, list]:
        if self._index is None:
            with open(self.root / "index.json") as f:
                idx = json.load(f)
            if int(self.meta.get("num_frames", len(idx))) != len(idx):
                raise ValueError(f"{self.root}: index.json has {len(idx)} tokens, meta num_frames "
                                 f"{self.meta.get('num_frames')}")
            self._index = idx
        return self._index

    def tokens(self):
        return self.index.keys()

    def has(self, token: str) -> bool:
        return token in self.index

    def _memmap(self, field: str, shard: str) -> np.ndarray:
        if os.getpid() != self._pid:           # forked worker: do not reuse the parent's handles
            self._mm, self._pid = {}, os.getpid()
        key = (field, shard)
        a = self._mm.get(key)
        if a is None:
            a = np.load(self.root / field / f"{shard}.npy", mmap_mode="r")
            tail, dt = RESMAP_FIELDS[field]
            if a.shape[1:] != tail or a.dtype != np.dtype(dt):
                raise ValueError(f"{self.root}/{field}/{shard}.npy: {a.shape} {a.dtype} != (N,)+{tail} {dt}")
            self._mm[key] = a
        return a

    def raw(self, token: str, field: str) -> np.ndarray:
        """One frame of one field in the cache layout (a copy; only this frame's pages are read)."""
        shard, row = self.index[token]
        a = self._memmap(field, shard)
        if not 0 <= int(row) < a.shape[0]:
            raise IndexError(f"{token}: row {row} outside {field}/{shard} ({a.shape[0]})")
        return np.array(a[int(row)])

    def load_bev(self, token: str, s_grid: bool = True) -> np.ndarray:
        """bev float16: S grid [256, 50, 100] (row = x forward, col 0 = +32 m left), or the cache layout
        [256, 100, 50] (lateral, forward) if s_grid=False."""
        x = self.raw(token, "bev")
        return resmap_to_s_grid(x) if s_grid else x

    def load(self, token: str, dtype=np.float16, s_grid: bool = True) -> np.ndarray:
        x = self.load_bev(token, s_grid=s_grid)
        return x if np.dtype(dtype) == x.dtype else x.astype(dtype)

    def load_seg(self, token: str, s_grid: bool = False) -> np.ndarray:
        """seg raw logits float16 (4, 200, 100) (lateral, forward; 0.32 m), or [4, 100, 200] S orientation."""
        x = self.raw(token, "seg")
        return resmap_to_s_grid(x) if s_grid else x

    def load_vectors(self, token: str, metres: bool = True) -> Dict[str, np.ndarray]:
        """vectors [100, 20, 2] (N-frame metres if metres else normalised), scores [100] f32, labels [100] i8,
        props [100] i16."""
        v = self.raw(token, "vectors")
        return {"vectors": resmap_vectors_to_n(v) if metres else v,
                "scores": self.raw(token, "scores").astype(np.float32),
                "labels": self.raw(token, "labels"), "props": self.raw(token, "props")}

    def close(self) -> None:
        self._mm.clear()
