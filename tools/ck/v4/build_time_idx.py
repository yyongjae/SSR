#!/usr/bin/env python
"""refiner v4: per-candidate first-failure time-index sidecar (time_idx.npy int8 [N, K, 3]) of an EXISTING label dir.

  build_time_idx.py --label-dir DIR [--out DIR] [--packed DIR] [--root R] [--traj NPY | --no-crc] [--rows SPEC]
                    [--max-shards N] [--workers 4] [--force]
  build_time_idx.py --merge-accstraight [--split navtrain_train | --merged-dir DIR] [--base-dir DIR] [--alt-dir DIR]
                    [--alt-names a+0.5] [--out DIR] [--packed DIR] [--rows SPEC] [--max-shards N] [--workers 4] [--force]

Source  the label shards <dir>/shards/c*.parquet of tools/ck/data/label_cands.py (one row per (token, k) with every
        score_token key, among them nc_time_idx / ttc_time_idx / dac_time_idx int64: n = 0..40 on the 41-state LQR
        grid, t = 0.1 n s, -1 = no event) -- the columns label_cands.assemble drops.  Nothing is re-scored.
Mapping = label_cands.assemble: row = position of the shard's token in <packed>/tokens.parquet (tokens not in it are
        skipped, as there), column = shard k (k = -1 error rows skipped), ok = SCORE_COLS finite; shards are applied in
        sorted file order (a later shard wins).
Checks (any failure -> SystemExit, nothing written)
  stale-shard guard  crc32 (common.traj_crc) of each scored token's trajectories in the traj source of the label dir's
                     meta.json == the shard crc (label_cands rule: an npy source has rows aligned to the packed tokens;
                     a 'dump' source is read through packed cand.npy when its K matches).  --traj overrides the source,
                     --no-crc skips the guard (the label equality below still pins the alignment).
  label equality     the shard's 9 label columns (as f32) == <dir>/labels.npy at (row, k), bitwise with NaN == NaN: the
                     sidecar is aligned row-for-row / column-for-column with THAT labels.npy.
  coverage           (full builds only) every ok.npy True cell is covered by a shard row.
  --merge-accstraight  var_separate_sampler16_accstraight has no shards: its a+0.5 columns came from
                     var_accel_straight_sampler16 (V = 2: id, a+0.5), the rest from var_separate_sampler16 (meta.json
                     'merged').  Column map from the three index.npz (names; c = k * V + v checked against col_k /
                     col_v); built rows, anchor_idx of every built row and the mapped 'valid' must be equal in all
                     three; both source dirs are checked as above; then per built row the merged traj.npy == the mapped
                     source traj.npy rows (bitwise, and the crc32 of the row as the labeler computes it), merged
                     labels.npy / ok.npy == the mapped source labels.npy / ok.npy.
Reported (meta.json 'consistency', not fatal): on every ok candidate with a known index nc < 1 <=> nc_idx >= 0,
        ttc < 1 <=> ttc_idx >= 0, dac < 1 <=> dac_idx >= 0 (exact for the official scorer arrays; violations counted
        per column with up to 10 examples (row, k, col, label, idx)); shard values for a label dir, the merged
        labels.npy vs the merged sidecar for --merge-accstraight (per-source shard checks in consistency_per_source).
Output  <out>/time_idx.npy int8 [N, K, 3] (navsim/agents/para_ssr/ck/time_idx.py: cols nc, ttc, dac; -1 none, -2 unknown
        = not scored / not ok / shard without the time columns / row not selected) + <out>/meta.json (written last;
        labels_sha16 + labels_probe_sha16 of the labels.npy it is aligned with, counts, checks).  Default
        out = time_idx.DEFAULT_ROOT/<split>/<label dir name>/ (CK/v4/time_idx); an existing time_idx.npy is never
        replaced without --force.
Small test builds  --rows SPEC (a:b | i,j,k | file.npy of packed rows; with an unfiltered label plan only the shards
        holding those rows are read) and/or --max-shards N (first N shards per source dir): meta partial = True,
        unselected rows stay -2, the coverage check is skipped.
CPU only, OMP_NUM_THREADS=1; --workers (default 4, <= 16) fork processes read the shards.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import label_cands as LC  # noqa: E402  (pins navsim; TrajSource / SCORE_COLS of the labeler)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from navsim.agents.para_ssr.ck import time_idx as TI  # noqa: E402

CM = LC.CM
LABEL_COLS = tuple(LC.LABEL_COLS)
SCORE_COLS = tuple(LC.SCORE_COLS)
MAX_WORKERS = 16
DEFAULT_WORKERS = 4
MERGED_NAME = "var_separate_sampler16_accstraight"
VERIFY_BLOCK = 256                     # rows per merged-mapping verification task
HEAD = 10                              # examples kept per failure kind

_W: Dict[str, Any] = {}


def _kst() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def _eq_nan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """element-wise bitwise f32 equality with NaN == NaN."""
    a = np.ascontiguousarray(a, np.float32)
    b = np.ascontiguousarray(b, np.float32)
    return (a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))


def parse_rows(spec: str, n: int) -> Optional[np.ndarray]:
    """'' -> None (all rows); 'a:b' -> a..b-1; 'i,j,k'; a .npy file of ints -> sorted unique rows < n."""
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.endswith(".npy"):
        r = np.load(spec).astype(np.int64).reshape(-1)
    elif ":" in spec:
        a, b = spec.split(":", 1)
        r = np.arange(int(a or 0), int(b) if b else n, dtype=np.int64)
    else:
        r = np.array([int(x) for x in spec.split(",") if x.strip()], np.int64)
    r = np.unique(r)
    return r[(r >= 0) & (r < n)]


# ----------------------------------------------------------------------------------------------- sources
def open_source(label_dir, packed_tokens: Sequence[str], traj: str = "", no_crc: bool = False,
                packed_dir: Optional[Path] = None) -> Dict[str, Any]:
    """A label dir with shards/: labels.npy [N, K, 9], ok.npy, meta.json, sorted shard list, plan, traj source spec of
    the stale-shard guard (None with no_crc)."""
    d = Path(label_dir)
    if not (d / "labels.npy").is_file():
        raise SystemExit(f"{d}: no labels.npy")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
    L = np.load(d / "labels.npy", mmap_mode="r")
    if L.ndim != 3 or L.shape[2] != len(LABEL_COLS):
        raise SystemExit(f"{d}/labels.npy {L.shape}: expected [N, K, {len(LABEL_COLS)}]")
    n, k = int(L.shape[0]), int(L.shape[1])
    if meta.get("n") is not None and (int(meta["n"]), int(meta.get("k", k))) != (n, k):
        raise SystemExit(f"{d}: meta n / k {meta.get('n')} / {meta.get('k')} != labels.npy {L.shape}")
    if len(packed_tokens) != n:
        raise SystemExit(f"{d}: labels.npy has {n} rows, packed tokens {len(packed_tokens)}")
    sd = d / "shards"
    shards = sorted(sd.glob("c*.parquet")) if sd.is_dir() else []
    if not shards:
        raise SystemExit(f"{d}: no label shards (shards/c*.parquet); a merged dir needs --merge-accstraight")
    plan = json.loads((sd / "_plan.json").read_text()) if (sd / "_plan.json").is_file() else {}
    spec = None
    if not no_crc:
        if traj:
            spec = ("npy", str(Path(traj).resolve()), k)
        else:
            kind, path = meta.get("traj_kind", "npy"), meta.get("traj_path", "")
            if kind == "dump":
                pc = Path(packed_dir) / "cand.npy" if packed_dir is not None else None
                if pc is not None and pc.is_file() and np.load(pc, mmap_mode="r").shape[1] == k:
                    spec = ("npy", str(pc), k)
                else:
                    spec = ("dump", str(path), k)
            else:
                spec = ("npy", str(path), k)
        if spec[0] == "npy":
            if not Path(spec[1]).is_file():
                raise SystemExit(f"{d}: traj source {spec[1]} missing (pass --traj or --no-crc)")
            shp = np.load(spec[1], mmap_mode="r").shape
            if shp != (n, k, 8, 3):
                raise SystemExit(f"{d}: traj source {spec[1]} {shp} != ({n}, {k}, 8, 3)")
    return {"dir": d, "meta": meta, "labels": str(d / "labels.npy"), "ok": str(d / "ok.npy"), "n": n, "k": k,
            "split": meta.get("split"), "shards": shards, "plan": plan, "traj": spec}


def select_shards(src: Dict[str, Any], rows: Optional[np.ndarray], max_shards: int) -> List[Path]:
    """shards to read: all, or (rows given and the plan unfiltered: rows = 0..N-1 chunked) the ones holding rows; then
    the first max_shards."""
    sh = list(src["shards"])
    plan = src["plan"]
    if rows is not None and plan.get("chunk") and not plan.get("rows") and int(plan.get("n_rows", src["n"])) == src["n"]:
        ids = set((rows // int(plan["chunk"])).tolist())
        sh = [p for p in sh if int(p.stem[1:]) in ids]
    if max_shards:
        sh = sh[: int(max_shards)]
    return sh


# ----------------------------------------------------------------------------------------------- workers
def _init(srcs: Dict[str, Dict[str, Any]], tokens: Sequence[str], sel: Optional[np.ndarray], merge=None):
    os.environ["OMP_NUM_THREADS"] = "1"
    _W["srcs"] = srcs
    _W["rowmap"] = {t: i for i, t in enumerate(tokens)}
    _W["tokens"] = list(tokens)
    _W["sel"] = sel
    _W["mm"] = {}
    _W["traj"] = {}
    _W["merge"] = merge


def _mm(path: str) -> np.ndarray:
    a = _W["mm"].get(path)
    if a is None:
        a = _W["mm"][path] = np.load(path, mmap_mode="r")
    return a


def _traj_src(tag: str):
    s = _W["srcs"][tag]["traj"]
    if s is None:
        return None
    t = _W["traj"].get(tag)
    if t is None:
        t = _W["traj"][tag] = LC.TrajSource(s[0], s[1], _W["tokens"], int(s[2])).open()
    return t


def read_shard(task) -> Dict[str, Any]:
    """task = (tag, shard path) -> cells (r, k, t int8 [n, 3]) + check counters of one shard (see module doc)."""
    import pyarrow.parquet as pq

    tag, path = task
    S = _W["srcs"][tag]
    have = set(pq.read_schema(path).names)
    has_time = all(c in have for c in TI.TIDX_SRC_COLS)
    cols = [c for c in ("token", "row", "k", "crc", *LABEL_COLS, *TI.TIDX_SRC_COLS) if c in have]
    df = pd.read_parquet(path, columns=cols)
    out: Dict[str, Any] = {"tag": tag, "path": str(path), "has_time": has_time, "n_file_rows": int(len(df)),
                           "n_err_rows": 0, "n_unknown_token": 0, "n_cells": 0, "n_ok": 0, "n_bad_value": 0,
                           "n_tok": 0, "n_crc_checked": 0, "n_stale": 0, "stale_head": [], "n_label_cmp": 0,
                           "n_label_mismatch": 0, "label_mismatch_head": [], "n_row_col_diff": 0,
                           "cons": TI.consistency(np.zeros((0, len(LABEL_COLS))), np.zeros((0, TI.N_TIDX), np.int64)),
                           "r": np.zeros(0, np.int64), "k": np.zeros(0, np.int64), "t": TI.unknown((0,))}
    if not len(df):
        return out
    out["n_err_rows"] = int((df.k < 0).sum())
    good = df[df.k >= 0]
    rr = good.token.astype(str).map(_W["rowmap"])
    known = rr.notna().to_numpy()
    out["n_unknown_token"] = int((~known).sum())
    good = good[known]
    r = rr[known].to_numpy(np.int64)
    if _W["sel"] is not None:
        m = _W["sel"][r]
        good, r = good[m], r[m]
    if not len(good):
        return out
    kk = good.k.to_numpy(np.int64)
    if (kk >= S["k"]).any():
        raise ValueError(f"{path}: k {int(kk.max())} >= K {S['k']}")
    if "row" in good.columns:
        out["n_row_col_diff"] = int((good.row.to_numpy(np.int64) != r).sum())
    lab64 = good[list(LABEL_COLS)].to_numpy(np.float64)
    lab = lab64.astype(np.float32)
    ok = np.isfinite(good[list(SCORE_COLS)].to_numpy(np.float64)).all(1)
    if has_time:
        raw = good[list(TI.TIDX_SRC_COLS)].to_numpy(np.int64)
        out["n_bad_value"] = int((((raw < TI.NONE) | (raw > TI.MAX_IDX)) & ok[:, None]).sum())
        t = TI.sanitize(np.where(raw < TI.NONE, TI.UNKNOWN, raw), ok)
    else:
        t = TI.unknown((len(good),))
    c = TI.consistency(lab64, t, ok, max_examples=HEAD)
    c["examples"] = [[int(r[i]), int(kk[i]), col, lv, iv] for i, col, lv, iv in c["examples"]]
    out["cons"] = c
    # stale-shard guard (label_cands.assemble rule)
    src = _traj_src(tag)
    if src is not None and "crc" in good.columns:
        crc = good.groupby("token", sort=False).crc.first()
        for tok, cc in crc.items():
            out["n_tok"] += 1
            if src.has(tok):
                out["n_crc_checked"] += 1
                if CM.traj_crc(src.get(tok)) != int(cc):
                    out["n_stale"] += 1
                    if len(out["stale_head"]) < HEAD:
                        out["stale_head"].append(str(tok))
    else:
        out["n_tok"] = int(good.token.nunique())
    # label equality with labels.npy (the array the sidecar is aligned with)
    L = _mm(S["labels"])
    ru = np.unique(r)
    block = np.asarray(L[ru], np.float32)
    want = block[np.searchsorted(ru, r), kk]
    eq = _eq_nan(want, lab).all(1)
    out["n_label_cmp"] = int(len(eq))
    out["n_label_mismatch"] = int((~eq).sum())
    out["label_mismatch_head"] = [[int(r[i]), int(kk[i])] for i in np.flatnonzero(~eq)[:HEAD]]
    out["n_cells"], out["n_ok"] = int(len(good)), int(ok.sum())
    out["r"], out["k"], out["t"] = r, kk, t
    return out


def verify_merge_rows(rows) -> Dict[str, Any]:
    """merged traj / labels / ok of these packed rows == the mapped source rows (worker; _W['merge'] = column map)."""
    M = _W["merge"]
    rows = np.asarray(rows, np.int64)
    bcols, acols = M["base_cols"], M["alt_cols"]
    res: Dict[str, Any] = {"n_rows": int(len(rows)), "n_traj_mismatch": 0, "n_crc_mismatch": 0,
                           "n_label_mismatch_rows": 0, "n_ok_mismatch_rows": 0, "head": []}
    for name in ("traj", "labels", "ok"):
        A = np.asarray(_mm(M["merged"][name])[rows])
        B = np.asarray(_mm(M["base"][name])[rows])
        C = np.asarray(_mm(M["alt"][name])[rows])
        exp = np.empty_like(A)
        exp[:, bcols] = B[:, M["src_col"][bcols]]
        exp[:, acols] = C[:, M["src_col"][acols]]
        if name == "traj":                          # bitwise, and the labeler's per-token crc32 of the row
            badr = ~(A.view(np.uint32) == exp.view(np.uint32)).reshape(len(rows), -1).all(1)
            res["n_traj_mismatch"] = int(badr.sum())
            res["n_crc_mismatch"] = int(sum(CM.traj_crc(A[i]) != CM.traj_crc(exp[i]) for i in range(len(rows))))
        elif name == "labels":
            badr = ~_eq_nan(A, exp).reshape(len(rows), -1).all(1)
            res["n_label_mismatch_rows"] = int(badr.sum())
        else:
            badr = ~(A == exp).reshape(len(rows), -1).all(1)
            res["n_ok_mismatch_rows"] = int(badr.sum())
        for i in np.flatnonzero(badr)[:HEAD]:
            if len(res["head"]) < HEAD:
                res["head"].append([name, int(rows[i])])
    return res


def _pool_map(fn, tasks: List, workers: int, initargs: Tuple, what: str, log=print) -> List[Dict[str, Any]]:
    nw = max(1, min(int(workers), len(tasks)))
    out: List[Dict[str, Any]] = []
    t0 = time.time()
    with get_context("fork").Pool(nw, initializer=_init, initargs=initargs) as pool:
        for j, r in enumerate(pool.imap(fn, tasks, chunksize=1)):
            out.append(r)
            if (j + 1) % 100 == 0 or j + 1 == len(tasks):
                log(f"[{time.strftime('%T')}] {what} {j + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    return out


# ----------------------------------------------------------------------------------------------- assembly
def assemble_source(src: Dict[str, Any], results: List[Dict[str, Any]], partial: bool
                    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """shard results (sorted order) -> (time_idx int8 [N, K, 3], covered bool [N, K], stats); SystemExit on a stale
    shard, a label mismatch or (full builds) an ok cell no shard covers."""
    n, k = src["n"], src["k"]
    T = TI.unknown((n, k))
    cov = np.zeros((n, k), bool)
    st: Dict[str, Any] = {"n_shards_read": len(results), "n_shards_no_time": 0, "n_cells": 0, "n_ok": 0,
                          "n_duplicate_cells": 0, "n_err_rows": 0, "n_unknown_token": 0, "n_bad_value": 0,
                          "n_tok": 0, "n_crc_checked": 0, "n_stale": 0, "stale_head": [], "n_label_cmp": 0,
                          "n_label_mismatch": 0, "label_mismatch_head": [], "n_row_col_diff": 0,
                          "no_time_shards_head": []}
    for res in results:
        r, kk = res["r"], res["k"]
        st["n_duplicate_cells"] += int(cov[r, kk].sum())
        T[r, kk] = res["t"]
        cov[r, kk] = True
        if not res["has_time"]:
            st["n_shards_no_time"] += 1
            if len(st["no_time_shards_head"]) < HEAD:
                st["no_time_shards_head"].append(Path(res["path"]).name)
        for key in ("n_cells", "n_ok", "n_err_rows", "n_unknown_token", "n_bad_value", "n_tok", "n_crc_checked",
                    "n_stale", "n_label_cmp", "n_label_mismatch", "n_row_col_diff"):
            st[key] += int(res[key])
        st["stale_head"].extend(res["stale_head"][: HEAD - len(st["stale_head"])])
        st["label_mismatch_head"].extend(res["label_mismatch_head"][: HEAD - len(st["label_mismatch_head"])])
    st["consistency"] = TI.merge_consistency([res["cons"] for res in results], HEAD)
    st["crc_guard"] = "off (--no-crc)" if src["traj"] is None else f"{src['traj'][0]}:{src['traj'][1]}"
    if st["n_stale"]:
        raise SystemExit(f"{src['dir']}: {st['n_stale']} tokens scored on different trajectories than the traj source "
                         f"{st['crc_guard']} (stale shards), e.g. {st['stale_head']}")
    if st["n_label_mismatch"]:
        raise SystemExit(f"{src['dir']}: {st['n_label_mismatch']} shard label cells != labels.npy (row, k) e.g. "
                         f"{st['label_mismatch_head']}: the shards are not the source of this labels.npy")
    if not partial:
        okf = np.load(src["ok"], mmap_mode="r") if Path(src["ok"]).is_file() else None
        if okf is not None:
            miss = np.asarray(okf, bool) & ~cov
            st["n_ok_uncovered"] = int(miss.sum())
            if st["n_ok_uncovered"]:
                rr, kk = np.nonzero(miss)
                raise SystemExit(f"{src['dir']}: {st['n_ok_uncovered']} ok.npy cells covered by no shard, e.g. "
                                 f"{list(zip(rr[:HEAD].tolist(), kk[:HEAD].tolist()))}")
    return T, cov, st


def consistency_vs_labels(labels_path: str, T: np.ndarray, rows: Optional[np.ndarray] = None,
                          block: int = 4096) -> Dict[str, Any]:
    """time_idx.consistency of T against labels.npy itself on every known cell (row blocks; examples as (row, k))."""
    L = np.load(labels_path, mmap_mode="r")
    rows = np.arange(len(T)) if rows is None else np.asarray(rows, np.int64)
    parts = []
    k = T.shape[1]
    for i in range(0, len(rows), block):
        rr = rows[i:i + block]
        t = np.asarray(T[rr])
        c = TI.consistency(np.asarray(L[rr]), t, t[..., 0] != TI.UNKNOWN, max_examples=HEAD)
        c["examples"] = [[int(rr[f // k]), int(f % k), col, lv, iv] for f, col, lv, iv in c["examples"]]
        parts.append(c)
    return TI.merge_consistency(parts, HEAD)


def counts(T: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_cells": int(T.shape[0] * T.shape[1])}
    for j, c in enumerate(TI.TIDX_COLS):
        x = T[..., j]
        out[c] = {"event": int((x >= 0).sum()), "none": int((x == TI.NONE).sum()), "unknown": int((x == TI.UNKNOWN).sum())}
    out["n_known_cells"] = int((T[..., 0] != TI.UNKNOWN).sum())
    return out


def _label_fp(labels_path: str) -> Dict[str, Any]:
    L = np.load(labels_path, mmap_mode="r")
    return {"labels_path": str(labels_path), "labels_shape": list(map(int, L.shape)),
            "labels_sha16": TI.sha16_array(L), "labels_probe_sha16": TI.probe_sha16(L)}


def _write(out: Path, T: np.ndarray, meta: Dict[str, Any], force: bool, log=print) -> Dict[str, Any]:
    f = out / TI.FILE
    if f.exists() and not force:
        raise SystemExit(f"{f} exists (use --force to replace it)")
    CM.atomic_save_npy(f, np.ascontiguousarray(T, TI.DTYPE))
    CM.atomic_write_json(out / TI.META, meta)
    log(f"wrote {f} {T.shape} int8 + meta.json (partial {meta['partial']}, known cells "
        f"{meta['counts']['n_known_cells']} / {meta['counts']['n_cells']}, consistency violations "
        f"{meta['consistency']['n_violations']})")
    return meta


def _base_meta(n: int, k: int, partial: bool, rows_spec: str, max_shards: int, workers: int, t0: float) -> Dict:
    return {"version": TI.VERSION, "n": int(n), "k": int(k), "cols": list(TI.TIDX_COLS),
            "src_cols": list(TI.TIDX_SRC_COLS), "dtype": "int8", "convention": TI.CONVENTION,
            "values": {"none": TI.NONE, "unknown": TI.UNKNOWN, "max_idx": TI.MAX_IDX, "dt_s": TI.DT},
            "partial": bool(partial), "rows_spec": rows_spec, "max_shards": int(max_shards), "workers": int(workers),
            "builder": str(Path(__file__).resolve()), "builder_sha16": CM.sha256_file(__file__)[:16],
            "git_head": CM.git_head(), "built_kst": _kst(), "wall_s": round(time.time() - t0, 1)}


def _tokens(packed: Optional[str], split: Optional[str], root) -> Tuple[List[str], Path]:
    pk = Path(packed) if packed else CM.packed_dir(split, root)
    if not (pk / "tokens.parquet").is_file():
        raise SystemExit(f"{pk}/tokens.parquet missing (pass --packed)")
    toks = pd.read_parquet(pk / "tokens.parquet", columns=["token"]).token.astype(str).tolist()
    return toks, pk


def _sel_mask(rows: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if rows is None:
        return None
    m = np.zeros(n, bool)
    m[rows] = True
    return m


def _plan_check(src: Dict[str, Any], toks: Sequence[str]) -> Dict[str, Any]:
    sha = CM.sha16_array(np.array(list(toks), "U16"))
    want = src["plan"].get("token_sha16")
    return {"packed_token_sha16": sha, "plan_token_sha16": want, "equal": (want == sha) if want else None}


# ----------------------------------------------------------------------------------------------- runs
def run(label_dir, out=None, packed: str = "", root=None, traj: str = "", no_crc: bool = False, rows: str = "",
        max_shards: int = 0, workers: int = DEFAULT_WORKERS, force: bool = False, log=print) -> Dict[str, Any]:
    """Sidecar of one label dir with shards (see module doc) -> meta."""
    assert 1 <= int(workers) <= MAX_WORKERS, f"workers {workers} not in [1, {MAX_WORKERS}]"
    t0 = time.time()
    d = Path(label_dir).resolve()
    m0 = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
    toks, pk = _tokens(packed, m0.get("split"), root)
    src = open_source(d, toks, traj, no_crc, pk)
    sel = parse_rows(rows, src["n"])
    partial = sel is not None or bool(max_shards)
    shards = select_shards(src, sel, max_shards)
    log(f"[{time.strftime('%F %T')}] {d}: N {src['n']} K {src['k']}, {len(shards)}/{len(src['shards'])} shards, "
        f"crc guard {src['traj'][1] if src['traj'] else 'off'}, workers {workers}")
    res = _pool_map(read_shard, [("src", str(p)) for p in shards], workers,
                    ({"src": {k: src[k] for k in ("labels", "traj", "k")}}, toks, _sel_mask(sel, src["n"])),
                    "shards", log)
    T, cov, st = assemble_source(src, res, partial)
    name = d.name
    o = Path(out) if out else TI.default_dir(src["split"] or "unknown_split", name)
    meta = _base_meta(src["n"], src["k"], partial, rows, max_shards, workers, t0)
    meta.update({"mode": "shards", "source_dir": str(d), "name": name, "split": src["split"], "packed": str(pk),
                 **_label_fp(src["labels"]), "n_shards_total": len(src["shards"]),
                 "rows_built": int(cov.any(1).sum()), "plan": _plan_check(src, toks),
                 "label_meta": {k: m0.get(k) for k in ("split", "name", "n", "k", "traj_kind", "traj_path",
                                                      "traj_sha16", "assembled", "scorer_yaml", "merged")},
                 "checks": {k: v for k, v in st.items() if k != "consistency"}, "consistency": st["consistency"],
                 "counts": counts(T)})
    meta["wall_s"] = round(time.time() - t0, 1)
    return _write(o, T, meta, force, log)


def _resolve(md: Path, v: str) -> Path:
    p = Path(v)
    if p.is_absolute():
        return p
    return (md.parent.parent / p) if "/" in v else (md.parent / p)


def merge_spec(md: Path) -> Tuple[Path, Path, List[str]]:
    """(base dir, alt dir, alt names) of a merged variant dir from its meta.json / build.json 'merged' record
    ({"from": base, "<name>_from": alt, ...})."""
    rec: Dict[str, Any] = {}
    for f in ("build.json", "meta.json"):
        if (md / f).is_file():
            rec.update(json.loads((md / f).read_text()).get("merged") or {})
    if "from" not in rec:
        rec["from"] = "var_separate_sampler16"
    alt = {k[: -len("_from")]: v for k, v in rec.items() if k.endswith("_from")}
    if len(set(alt.values())) != 1:
        raise SystemExit(f"{md}: merged record {rec} must name exactly one alt source dir (<name>_from)")
    return _resolve(md, str(rec["from"])), _resolve(md, next(iter(alt.values()))), sorted(alt)


def _index(d: Path) -> Dict[str, np.ndarray]:
    with np.load(d / "index.npz", allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ("names", "col_k", "col_v", "anchor_idx", "valid", "built")}


def column_map(ixm: Dict, ixb: Dict, ixa: Dict, alt_names: Sequence[str]) -> Dict[str, Any]:
    """merged column c -> (source 0 base / 1 alt, source column); SystemExit on any structural mismatch."""
    nm, nb, na = ([str(x) for x in ix["names"]] for ix in (ixm, ixb, ixa))
    if nm != nb:
        raise SystemExit(f"merged names {nm} != base names {nb}")
    for x in alt_names:
        if x not in na or x not in nm:
            raise SystemExit(f"alt variant {x!r} not in merged {nm} / alt {na}")
    K = ixm["anchor_idx"].shape[1]
    for what, ix, names in (("merged", ixm, nm), ("base", ixb, nb), ("alt", ixa, na)):
        V = len(names)
        col = np.arange(K * V)
        if ix["anchor_idx"].shape[1] != K or not (np.array_equal(ix["col_k"], col // V)
                                                  and np.array_equal(ix["col_v"], col % V)):
            raise SystemExit(f"{what} index.npz: K / col_k / col_v not the k-major c = k * {V} + v layout with K {K}")
    Vm, Va = len(nm), len(na)
    src_of = np.zeros(K * Vm, np.int8)
    src_col = np.arange(K * Vm, dtype=np.int64)
    for c in range(K * Vm):
        k, v = divmod(c, Vm)
        if nm[v] in alt_names:
            src_of[c], src_col[c] = 1, k * Va + na.index(nm[v])
    return {"K": int(K), "names": nm, "alt_vnames": na, "alt_names": list(alt_names), "src_of": src_of,
            "src_col": src_col, "base_cols": np.flatnonzero(src_of == 0), "alt_cols": np.flatnonzero(src_of == 1)}


def run_merge(merged_dir, base_dir=None, alt_dir=None, alt_names: Optional[Sequence[str]] = None, out=None,
              packed: str = "", root=None, no_crc: bool = False, rows: str = "", max_shards: int = 0,
              workers: int = DEFAULT_WORKERS, force: bool = False, log=print) -> Dict[str, Any]:
    """Sidecar of the merged accstraight variant dir (no shards) from its two source dirs; refuses on any mapping
    mismatch (see module doc) -> meta."""
    assert 1 <= int(workers) <= MAX_WORKERS, f"workers {workers} not in [1, {MAX_WORKERS}]"
    t0 = time.time()
    md = Path(merged_dir).resolve()
    bd0, ad0, an0 = merge_spec(md)
    bd = Path(base_dir).resolve() if base_dir else bd0
    ad = Path(alt_dir).resolve() if alt_dir else ad0
    alt_names = list(alt_names) if alt_names else an0
    mm = json.loads((md / "meta.json").read_text()) if (md / "meta.json").is_file() else {}
    toks, pk = _tokens(packed, mm.get("split"), root)
    ix = {"merged": _index(md), "base": _index(bd), "alt": _index(ad)}
    cm = column_map(ix["merged"], ix["base"], ix["alt"], alt_names)
    Lm = np.load(md / "labels.npy", mmap_mode="r")
    n, km = int(Lm.shape[0]), int(Lm.shape[1])
    if km != len(cm["src_of"]) or len(toks) != n or np.load(md / "traj.npy", mmap_mode="r").shape != (n, km, 8, 3):
        raise SystemExit(f"{md}: labels {Lm.shape} / traj / packed {len(toks)} rows do not match K {len(cm['src_of'])}")
    # rows / anchors / validity must be the same in all three index files
    built = ix["merged"]["built"].astype(bool)
    for t in ("base", "alt"):
        if not np.array_equal(ix[t]["built"].astype(bool), built):
            raise SystemExit(f"{t} index.npz 'built' != merged 'built'")
        if not np.array_equal(ix[t]["anchor_idx"][built], ix["merged"]["anchor_idx"][built]):
            raise SystemExit(f"{t} index.npz anchor_idx != merged anchor_idx on built rows (different anchor subsets)")
    K, Vm, Va = cm["K"], len(cm["names"]), len(cm["alt_vnames"])
    vm = ix["merged"]["valid"].reshape(n, K * Vm)
    vexp = np.empty_like(vm)
    vexp[:, cm["base_cols"]] = ix["base"]["valid"].reshape(n, -1)[:, cm["src_col"][cm["base_cols"]]]
    vexp[:, cm["alt_cols"]] = ix["alt"]["valid"].reshape(n, K * Va)[:, cm["src_col"][cm["alt_cols"]]]
    n_valid_diff = int((vm != vexp)[built].sum())
    if n_valid_diff:
        raise SystemExit(f"merged index.npz valid differs from the mapped source valid in {n_valid_diff} cells")
    srcs = {"base": open_source(bd, toks, "", no_crc, pk), "alt": open_source(ad, toks, "", no_crc, pk)}
    for t, s in srcs.items():
        if s["k"] != K * (Vm if t == "base" else Va):
            raise SystemExit(f"{t} {s['dir']}: K {s['k']} != {K} x {Vm if t == 'base' else Va}")
    sel = parse_rows(rows, n)
    partial = sel is not None or bool(max_shards)
    tasks = [(t, str(p)) for t in ("base", "alt") for p in select_shards(srcs[t], sel, max_shards)]
    log(f"[{time.strftime('%F %T')}] merge {md.name}: base {bd.name} + alt {ad.name} {alt_names} -> N {n} K {km}, "
        f"{len(tasks)} shards, workers {workers}")
    init = {t: {k: s[k] for k in ("labels", "traj", "k")} for t, s in srcs.items()}
    res = _pool_map(read_shard, tasks, workers, (init, toks, _sel_mask(sel, n)), "shards", log)
    T, cov, st = {}, {}, {}
    for t in ("base", "alt"):
        T[t], cov[t], st[t] = assemble_source(srcs[t], [r for r in res if r["tag"] == t], partial)
    bc, ac, sc = cm["base_cols"], cm["alt_cols"], cm["src_col"]
    Tm = TI.unknown((n, km))
    Tm[:, bc] = T["base"][:, sc[bc]]
    Tm[:, ac] = T["alt"][:, sc[ac]]
    cv = np.zeros((n, km), bool)
    cv[:, bc] = cov["base"][:, sc[bc]]
    cv[:, ac] = cov["alt"][:, sc[ac]]
    vrows = np.flatnonzero(built & (cv.any(1) if partial else True))
    mspec = {"merged": {x: str(md / f"{x}.npy") for x in ("traj", "labels", "ok")},
             "base": {x: str(bd / f"{x}.npy") for x in ("traj", "labels", "ok")},
             "alt": {x: str(ad / f"{x}.npy") for x in ("traj", "labels", "ok")},
             "src_col": sc, "base_cols": bc, "alt_cols": ac}
    blocks = [vrows[i:i + VERIFY_BLOCK] for i in range(0, len(vrows), VERIFY_BLOCK)]
    vres = _pool_map(verify_merge_rows, blocks, workers, ({}, toks, None, mspec), "verify", log) if blocks else []
    ver = {k: int(sum(v[k] for v in vres)) for k in ("n_rows", "n_traj_mismatch", "n_crc_mismatch",
                                                      "n_label_mismatch_rows", "n_ok_mismatch_rows")}
    ver["head"] = [h for v in vres for h in v["head"]][:HEAD]
    if any(ver[k] for k in ("n_traj_mismatch", "n_crc_mismatch", "n_label_mismatch_rows", "n_ok_mismatch_rows")):
        raise SystemExit(f"{md}: column mapping refused (merged vs mapped sources): {ver}")
    if not partial:
        okm = np.asarray(np.load(md / "ok.npy", mmap_mode="r"), bool)
        n_unc = int((okm & ~cv).sum())
        if n_unc:
            raise SystemExit(f"{md}: {n_unc} merged ok.npy cells not covered by the mapped source shards")
    o = Path(out) if out else TI.default_dir(mm.get("split") or "unknown_split", md.name)
    meta = _base_meta(n, km, partial, rows, max_shards, workers, t0)
    cons = consistency_vs_labels(str(md / "labels.npy"), Tm, np.flatnonzero(cv.any(1)))
    meta.update({"mode": "merge_accstraight", "source_dir": str(md), "name": md.name, "split": mm.get("split"),
                 "packed": str(pk), **_label_fp(str(md / "labels.npy")),
                 "merge": {"base_dir": str(bd), "alt_dir": str(ad), "alt_names": alt_names, "names": cm["names"],
                           "alt_vnames": cm["alt_vnames"], "K": K,
                           "map": "merged c = k * V + v; v in alt_names -> alt column k * V_alt + alt_vnames.index(v), "
                                  "else base column c",
                           "n_alt_cols": int(len(ac)), "n_base_cols": int(len(bc)), "verify": ver,
                           "base_meta_traj_sha16": srcs["base"]["meta"].get("traj_sha16"),
                           "alt_meta_traj_sha16": srcs["alt"]["meta"].get("traj_sha16")},
                 "rows_built": int(cv.any(1).sum()),
                 "plan": {t: _plan_check(srcs[t], toks) for t in srcs},
                 "label_meta": {k: mm.get(k) for k in ("split", "name", "n", "k", "traj_path", "traj_sha16",
                                                       "assembled", "merged")},
                 "checks": {t: {k: v for k, v in st[t].items() if k != "consistency"} for t in st},
                 "consistency": cons, "consistency_per_source": {t: st[t]["consistency"] for t in st},
                 "counts": counts(Tm)})
    meta["wall_s"] = round(time.time() - t0, 1)
    return _write(o, Tm, meta, force, log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label-dir", default="", help="label dir with shards/ (tools/ck/data/label_cands.py output)")
    ap.add_argument("--merge-accstraight", action="store_true", help="build the merged accstraight variant sidecar")
    ap.add_argument("--split", default="navtrain_train", help="--merge-accstraight default merged dir split")
    ap.add_argument("--merged-dir", default="", help=f"default CK/ck2/labels/<split>/{MERGED_NAME}")
    ap.add_argument("--base-dir", default="", help="default: the merged record 'from' (var_separate_sampler16)")
    ap.add_argument("--alt-dir", default="", help="default: the merged record '<name>_from' (var_accel_straight_...)")
    ap.add_argument("--alt-names", nargs="*", default=None, help="variant names from the alt dir (default: a+0.5)")
    ap.add_argument("--out", default="", help="output dir (default time_idx.DEFAULT_ROOT/<split>/<name>)")
    ap.add_argument("--packed", default="", help="packed dir with tokens.parquet (default <root>/packed/<split>)")
    ap.add_argument("--root", default="", help="CK data root (default CK_DATA)")
    ap.add_argument("--traj", default="", help="traj source of the crc guard (default: the label meta traj_path)")
    ap.add_argument("--no-crc", action="store_true", help="skip the stale-shard crc guard")
    ap.add_argument("--rows", default="", help="small test build: packed rows a:b | i,j,k | rows.npy")
    ap.add_argument("--max-shards", type=int, default=0, help="small test build: first N shards per source dir")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--force", action="store_true", help="replace an existing time_idx.npy")
    a = ap.parse_args(argv)
    if bool(a.label_dir) == bool(a.merge_accstraight):
        ap.error("exactly one of --label-dir / --merge-accstraight")
    if a.merge_accstraight:
        md = a.merged_dir or str(Path(CM.root_dir(a.root or None)) / "ck2" / "labels" / a.split / MERGED_NAME)
        run_merge(md, a.base_dir or None, a.alt_dir or None, a.alt_names, a.out or None, a.packed, a.root or None,
                  a.no_crc, a.rows, a.max_shards, a.workers, a.force, log=lambda s: print(s, flush=True))
    else:
        run(a.label_dir, a.out or None, a.packed, a.root or None, a.traj, a.no_crc, a.rows, a.max_shards, a.workers,
            a.force, log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
