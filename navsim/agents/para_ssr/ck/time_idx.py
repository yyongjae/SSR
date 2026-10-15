"""refiner v4: first-failure time indices of the official scorer as an OPTIONAL per-candidate int8 sidecar.

Source: tools/refiner/score_trajectories.score_token (official PDMScorer arrays of the proposal in the K+1 call):
  nc_time_idx   PDMScorer._collision_time_idcs: first ego state n with an AT-FAULT collision (the one that makes NC
                0 or 0.5); non-at-fault contacts never set it.
  ttc_time_idx  PDMScorer._ttc_time_idcs: first ego state n whose forward-projected ego box (front-left corner as
                centre, constant heading / speed, look-ahead 0, 0.3, 0.6, 0.9 s) intersects an agent that makes TTC 0,
                i.e. the ego time of the FLAGGED STEP -- the projected contact itself is up to 0.9 s later.
  dac_time_idx  first ego state n with a corner outside the drivable layers (EgoAreaIndex.NON_DRIVABLE_AREA).
  n = 0 .. 40 on the 41-state LQR-tracked grid (t = 0.1 n s, 0 .. 4 s after t0); -1 = no event.
Exact equivalences of the official arrays (checked by tools/ck/v4/build_time_idx.py on every ok candidate):
  nc < 1 <=> nc_idx >= 0,   ttc < 1 <=> ttc_idx >= 0,   dac < 1 <=> dac_idx >= 0.

Sidecar layout (never part of labels.npy, which stays 9 columns)
  <dir>/time_idx.npy  int8 [N, K, 3], columns TIDX_COLS = (nc, ttc, dac), row / column aligned with <label dir>/labels.npy
  <dir>/meta.json     version, source label dir(s), labels_path / labels_shape / labels_sha16 (sha256[:16] of the full
                      labels.npy array bytes) / labels_probe_sha16 (probe_sha16: cheap row-sample fingerprint), counts,
                      consistency check
  values  n >= 0 event at state n, NONE (-1) no event, UNKNOWN (-2) candidate not scored / not ok / source shard
          without the time columns / generation written before the sidecar existed.
Default root of the offline sidecars: DEFAULT_ROOT/<split>/<label dir name>/ (tools/ck/v4/build_time_idx.py).

pdm_progress_eff (LABEL_COLS index 8, PDM-Closed raw progress x PDM-Closed multiplier = the EP normaliser candidate;
token-level: constant over the K candidates of one score_token call up to float noise) is exposed as P_ref by
pref_of(labels9) (first finite value along the candidate axis, NaN if none).

numpy only (torch tensors accepted by to_seconds / pref_of); no navsim / file-layout imports beyond constants.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .constants import CK_DATA, LABEL_COLS

PathLike = Union[str, Path]

VERSION = "v4_time_idx_v1"
TIDX_COLS = ("nc", "ttc", "dac")                                   # time_idx.npy column order
TIDX_SRC_COLS = ("nc_time_idx", "ttc_time_idx", "dac_time_idx")    # score_token / label-shard column names
TIDX_LABEL_IDX = tuple(LABEL_COLS.index(c) for c in TIDX_COLS)     # (0, 3, 1): the matching labels.npy columns
N_TIDX = len(TIDX_COLS)
NONE = -1                                                          # scored, no event
UNKNOWN = -2                                                       # not scored / not ok / no source
N_STATES = 41                                                      # LQR grid states 0 .. 40
MAX_IDX = N_STATES - 1
DT = 0.1                                                           # s per state
DTYPE = np.int8
FILE = "time_idx.npy"
META = "meta.json"
PREF_COL = "pdm_progress_eff"
PREF_IDX = LABEL_COLS.index(PREF_COL)                              # 8
DEFAULT_ROOT = Path(CK_DATA) / "v4" / "time_idx"
CONVENTION = ("int8 [N, K, 3] columns (nc, ttc, dac); n = 0..40 first event state (t = 0.1 n s on the 41-state LQR "
              "grid; ttc = ego state of the flagged step, contact up to 0.9 s later); -1 = no event; -2 = unknown "
              "(not scored / not ok / no time columns)")


def default_dir(split: str, name: str, root: Optional[PathLike] = None) -> Path:
    return Path(root or DEFAULT_ROOT) / split / name


def unknown(shape: Sequence[int]) -> np.ndarray:
    """int8 array of UNKNOWN with shape (..., 3) appended."""
    return np.full(tuple(shape) + (N_TIDX,), UNKNOWN, DTYPE)


def sanitize(t, ok=None) -> np.ndarray:
    """any int array [..., 3] -> int8 with values outside {-2, -1, 0 .. 40} -> UNKNOWN; ok (bool [...]) False ->
    UNKNOWN."""
    a = np.asarray(t)
    if a.shape[-1:] != (N_TIDX,):
        raise ValueError(f"time indices must end with {N_TIDX} columns, got {a.shape}")
    a = a.astype(np.int64)
    bad = (a < UNKNOWN) | (a > MAX_IDX)
    out = np.where(bad, UNKNOWN, a).astype(DTYPE)
    if ok is not None:
        out[~np.asarray(ok, bool)] = UNKNOWN
    return out


def to_seconds(t, none_value: float = float("inf"), unknown_value: float = float("nan")):
    """time indices -> seconds after t0 (0.1 n) as float32; NONE -> none_value (default +inf: the event never happens
    within the 4 s horizon), UNKNOWN (and anything < -1) -> unknown_value (default NaN).  numpy in -> numpy out,
    torch in -> torch out (same device)."""
    try:
        import torch
        if torch.is_tensor(t):
            x = t.to(torch.float32) * DT
            x = torch.where(t == NONE, torch.full_like(x, none_value), x)
            return torch.where(t < NONE, torch.full_like(x, unknown_value), x)
    except ImportError:   # pragma: no cover
        pass
    a = np.asarray(t)
    x = a.astype(np.float32) * np.float32(DT)
    x = np.where(a == NONE, np.float32(none_value), x)
    return np.where(a < NONE, np.float32(unknown_value), x).astype(np.float32)


def from_labels_shards(df) -> np.ndarray:
    """label-shard / score_token rows (pandas DataFrame or list of dicts with TIDX_SRC_COLS) -> int64 [n, 3]."""
    if isinstance(df, (list, tuple)):
        return np.array([[int(r[c]) for c in TIDX_SRC_COLS] for r in df], np.int64).reshape(-1, N_TIDX)
    return df[list(TIDX_SRC_COLS)].to_numpy(np.int64)


def consistency(labels9, tidx, ok=None, max_examples: int = 10) -> Dict[str, Any]:
    """For candidates with ok (default: all) and a known index (>= -1): label < 1 <=> idx >= 0 per TIDX_COLS column.
    labels9 [..., 9] (LABEL_COLS), tidx [..., 3] -> {n_checked, violations {col: n}, n_violations, examples
    [(flat index, col, label, idx)]}."""
    L = np.asarray(labels9, np.float64).reshape(-1, len(LABEL_COLS))
    T = np.asarray(tidx, np.int64).reshape(-1, N_TIDX)
    m = np.ones(len(L), bool) if ok is None else np.asarray(ok, bool).reshape(-1)
    out: Dict[str, Any] = {"n_checked": 0, "violations": {}, "n_violations": 0, "examples": []}
    for j, (c, li) in enumerate(zip(TIDX_COLS, TIDX_LABEL_IDX)):
        mm = m & (T[:, j] >= NONE) & np.isfinite(L[:, li])
        bad = mm & ((L[:, li] < 1.0) != (T[:, j] >= 0))
        out["violations"][c] = int(bad.sum())
        out["n_checked"] = max(out["n_checked"], int(mm.sum()))
        for i in np.flatnonzero(bad)[: max(0, max_examples - len(out["examples"]))]:
            out["examples"].append([int(i), c, float(L[i, li]), int(T[i, j])])
    out["n_violations"] = int(sum(out["violations"].values()))
    return out


def merge_consistency(parts: Sequence[Dict[str, Any]], max_examples: int = 10) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_checked": 0, "violations": {c: 0 for c in TIDX_COLS}, "n_violations": 0, "examples": []}
    for p in parts:
        out["n_checked"] += int(p.get("n_checked", 0))
        for c in TIDX_COLS:
            out["violations"][c] += int(p.get("violations", {}).get(c, 0))
        out["examples"].extend(p.get("examples", [])[: max(0, max_examples - len(out["examples"]))])
    out["n_violations"] = int(sum(out["violations"].values()))
    return out


def pref_of(labels9, axis: int = -2):
    """P_ref = first finite pdm_progress_eff along the candidate axis of labels [..., K, 9] -> float32 [...]; NaN where
    no candidate has a finite value.  numpy or torch."""
    try:
        import torch
        if torch.is_tensor(labels9):
            v = labels9.select(-1, PREF_IDX).float()
            ax = axis + 1 if axis < 0 else axis                        # candidate axis of v (the label dim is gone)
            fin = torch.isfinite(v)
            first = torch.argmax(fin.to(torch.uint8), dim=ax, keepdim=True)
            got = v.gather(ax, first).squeeze(ax)
            return torch.where(fin.any(ax), got, torch.full_like(got, float("nan")))
    except ImportError:   # pragma: no cover
        pass
    L = np.asarray(labels9)
    v = np.take(L, PREF_IDX, axis=-1).astype(np.float32)
    ax = axis + 1 if axis < 0 else axis
    fin = np.isfinite(v)
    first = np.argmax(fin, axis=ax)
    got = np.take_along_axis(v, np.expand_dims(first, ax), ax).squeeze(ax)
    return np.where(fin.any(ax), got, np.float32(np.nan)).astype(np.float32)


# ----------------------------------------------------------------------------------------------- fingerprints
def sha16_array(arr, block_rows: int = 256) -> str:
    """sha256[:16] of the C-order bytes of arr (== tools/ck/data/common.sha16_array), streamed over row blocks (a
    memmap is never copied whole)."""
    a = np.asarray(arr) if not isinstance(arr, np.memmap) else arr
    h = hashlib.sha256()
    if a.ndim == 0 or len(a) == 0:
        h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]
    for i in range(0, len(a), block_rows):
        h.update(np.ascontiguousarray(a[i:i + block_rows]).tobytes())
    return h.hexdigest()[:16]


def probe_rows(n: int, n_probe: int = 257) -> np.ndarray:
    return np.unique(np.linspace(0, max(int(n) - 1, 0), min(int(n), int(n_probe))).astype(np.int64))


def probe_sha16(labels, n_probe: int = 257) -> str:
    """cheap fingerprint of a labels array [N, K, 9] (path or array): sha256[:16] of the shape + the bytes of <= n_probe
    evenly spaced rows (row 0 and N-1 included).  Detects a different / re-assembled label file in O(n_probe) rows."""
    a = np.load(labels, mmap_mode="r") if isinstance(labels, (str, Path)) else labels
    h = hashlib.sha256(json.dumps([list(map(int, a.shape)), str(a.dtype)]).encode())
    if len(a):
        h.update(np.ascontiguousarray(np.asarray(a[probe_rows(len(a), n_probe)])).tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- reader
class TimeIdxError(ValueError):
    """A time-index sidecar does not match its label file (shape / fingerprint) or is malformed."""


def read_meta(d: PathLike) -> Dict[str, Any]:
    p = Path(d) / META
    if not p.is_file():
        raise FileNotFoundError(f"time-index sidecar {d}: no {META}")
    return json.loads(p.read_text())


def load_time_idx(d: PathLike, expect_sha16_of_labels: Optional[str] = None, labels=None, mmap: bool = True,
                  check_values: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """-> (time_idx int8 [N, K, 3] (memmap unless mmap False), meta) of the sidecar dir d.

    Checks: meta version / cols / dtype; array shape == meta (n, k, 3) and dtype int8;
      expect_sha16_of_labels given -> == meta labels_sha16 (the full-array sha of the labels.npy it was built for);
      labels given (path of a labels.npy or an array [N, K, 9]) -> its shape [:2] == (n, k) and probe_sha16 == meta
        labels_probe_sha16 (cheap: same label file, row / column aligned);
      check_values -> every value in {-2, -1, 0 .. 40} (reads the whole array).
    TimeIdxError on a mismatch, FileNotFoundError when the files are missing."""
    d = Path(d)
    meta = read_meta(d)
    if meta.get("version") != VERSION:
        raise TimeIdxError(f"{d}: version {meta.get('version')!r} != {VERSION!r}")
    if tuple(meta.get("cols", ())) != TIDX_COLS:
        raise TimeIdxError(f"{d}: cols {meta.get('cols')} != {list(TIDX_COLS)}")
    f = d / FILE
    if not f.is_file():
        raise FileNotFoundError(f"time-index sidecar {d}: no {FILE}")
    a = np.load(f, mmap_mode="r" if mmap else None, allow_pickle=False)
    want = (int(meta["n"]), int(meta["k"]), N_TIDX)
    if a.shape != want or a.dtype != DTYPE:
        raise TimeIdxError(f"{f}: {a.shape} {a.dtype} != {want} int8")
    if expect_sha16_of_labels is not None and meta.get("labels_sha16") != expect_sha16_of_labels:
        raise TimeIdxError(f"{d}: built for labels sha16 {meta.get('labels_sha16')}, expected {expect_sha16_of_labels}")
    if labels is not None:
        L = np.load(labels, mmap_mode="r") if isinstance(labels, (str, Path)) else labels
        if tuple(L.shape[:2]) != want[:2]:
            raise TimeIdxError(f"{d}: time_idx [{want[0]}, {want[1]}] vs labels {tuple(L.shape)}")
        ps = probe_sha16(L)
        if meta.get("labels_probe_sha16") != ps:
            raise TimeIdxError(f"{d}: labels probe sha16 {ps} of {labels if isinstance(labels, (str, Path)) else 'array'}"
                               f" != {meta.get('labels_probe_sha16')} recorded at build (other / re-assembled labels)")
    if check_values:
        v = np.asarray(a)
        if ((v < UNKNOWN) | (v > MAX_IDX)).any():
            raise TimeIdxError(f"{f}: values outside [-2, 40]")
    return a, meta


__all__ = ["VERSION", "TIDX_COLS", "TIDX_SRC_COLS", "TIDX_LABEL_IDX", "N_TIDX", "NONE", "UNKNOWN", "N_STATES",
           "MAX_IDX", "DT", "DTYPE", "FILE", "META", "PREF_COL", "PREF_IDX", "DEFAULT_ROOT", "CONVENTION",
           "default_dir", "unknown", "sanitize", "to_seconds", "from_labels_shards", "consistency",
           "merge_consistency", "pref_of", "sha16_array", "probe_rows", "probe_sha16", "TimeIdxError", "read_meta",
           "load_time_idx"]
