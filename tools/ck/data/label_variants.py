#!/usr/bin/env python
"""CK2: official PDM labels of the speed / lateral variants of a fixed per-token raw-anchor subset.

  label_variants.py --split S [--combine separate|cross] [--subset sampler|nearest|file] [--n-anchors 16]
                    [--epoch 0] [--seed 0] [--idx-file NPY] [--speeds -1 -0.5 0.5] [--lats -0.5 0.5] [--s-on-frac 0.2]
                    [--name NAME] [--out-root DIR] [--root R] [--limit N] [--workers 48] [--chunk 64]
                    [--build-workers 8] [--build-only | --assemble-only | --check-only] [--retry-errors] [--fresh]

Pipeline (CPU only; resumable)
  1. subset   per packed token (row = <root>/packed/<split>/tokens.parquet) a fixed set of K = --n-anchors raw anchors
              (trajectory_anchors_256.npy, sha16 checked):
                sampler  (default) AnchorSampler(seed).sample(token, gt, official raw256 labels, epoch,
                         shuffle=False)[:K] in its canonical order near, mid, strat; K must be n_near (8),
                         n_near + n_mid (16, default = near + mid of epoch --epoch) or the full K = 32 (adds the
                         strat draw).  The anchor slot order in the outputs is therefore GT-rank ordered: a trainer
                         must permute the slots (or index by anchor_idx) before any order-aware use.
                nearest  the K GT-nearest anchors (v2 WTA metric, ranks 0 .. K-1).
                file     --idx-file int [N, >= K] (rows = packed tokens), first K columns.
              GT = packed/<split>/gt_traj.npy (training-time sampling only).
  2. build    navsim/agents/para_ssr/ck/variants.make_variants(anchors[idx], v0, speeds, lats, combine, s_on_frac)
              (float64 decode, float32 out; v0 = hypot(status[4], status[5]) of packed/<split>/status.npy, NaN -> 0,
              as cf_variants) -> traj.npy f32 [N, K * V, 8, 3], column c = k * V + v (k-major), rows not built
              (--limit) stay 0 and are never scored.  Variant v = 0 is the anchor itself (bitwise).
  3. score    tools/ck/data/label_cands.run(--traj traj.npy, --out this dir): shards/c<id>.parquet (resumable,
              --retry-errors), assembled to labels.npy f32 [N, K * V, 9] (LABEL_COLS) and ok.npy bool [N, K * V];
              labels.reshape(N, K, V, 9) gives [token, anchor slot, variant].
  4. check    check.json: identity (v = 0) labels vs labels/<split>/raw256 (all 9 columns, exact) and vs WoTE
              pdm_score_256 (NC / DAC / TTC / C); lateral-disabled duplicates (valid = False) vs their parent;
              per-variant pass rate / mean PDMS; measured seconds per token (load, score, build).

Output dir <out-root>/labels/<split>/<name>/ (out-root default CK_DATA/ck2; name default var_<combine>_<subset><K>):
  traj.npy      f32 [N, K*V, 8, 3]
  index.npz     anchor_idx int16 [N, K] (-1 = row not built), group int8 [N, K] (0 near / 1 mid / 2 strat region of
                the training sampler), pi f32 [N, K] (training-sampler inclusion probability of that anchor, design of
                AnchorSampler(seed) incl. the far pass / fail strata), rank int16 [N, K], dist f32 [N, K] (GT distance),
                pass bool [N, K] (raw256 NC = DAC = TTC = C = 1 of the anchor), v0 f32 [N], built bool [N],
                valid bool [N, K, V] (False = lateral part requested on an anchor with S_8 < 3 m -> duplicate of its
                speed-only parent, see variants.py), S8 f32 [N, K], col_k / col_v int16 [K*V], speed / lat f32 [V],
                names [V]
  dec.npz       f32 [N, K, V]: dev_xy (max knot xy distance to the anchor), ds4, d_end, alpha, beta, ext_m
  build.json    config (+ cfg_sha16), anchors / label source / token list hashes, build seconds
  shards/, labels.npy, ok.npy, meta.json   label_cands output
  check.json    step 4
A dry run (--limit N) must use its own --name (label_cands pins the token list per name).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import label_cands as LC  # noqa: E402  (pins navsim + the official scorer, as label_cands)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import anchor_sampler as AS  # noqa: E402
from navsim.agents.para_ssr.ck import variants as VV  # noqa: E402

CM = LC.CM
VERSION = "ck2_label_variants_v1"
OUT_ROOT = Path(CM.CK_DATA) / "ck2"
SUBSETS = ("sampler", "nearest", "file")
DEC_KEYS = ("dev_xy", "ds4", "d_end", "alpha", "beta", "ext_m")
LCOL = {c: i for i, c in enumerate(CM.LABEL_COLS)}
PASS_COLS = [LCOL[c] for c in ("nc", "dac", "ttc", "comfort")]
WOTE_KEYS = ("nc", "dac", "ttc", "comfort")             # pdm_score_256 axis 1 = CK_KEYS (nc, dac, ep, ttc, comfort)


# ----------------------------------------------------------------------------------------------- config
def out_dir(split: str, name: str, out_root=OUT_ROOT) -> Path:
    return Path(out_root) / "labels" / split / name


def default_name(combine: str, subset: str, n_anchors: int) -> str:
    return f"var_{combine}_{subset}{n_anchors}"


def make_config(split: str, combine: str = "separate", subset: str = "sampler", n_anchors: int = 16, epoch: int = 0,
                seed: int = 0, idx_file: str = "", speeds=VV.SPEEDS, lats=VV.LATS, s_on_frac: float = VV.S_ON_FRAC,
                limit: int = 0, root=None, ext: str = "const_curv") -> Dict:
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    tab = VV.variant_table(speeds, lats, combine)          # validates speeds / lats / combine
    VV._check(tuple(float(a) for a in speeds), tuple(float(d) for d in lats), combine, s_on_frac)
    smp = AS.AnchorSampler(anchors=AS.load_anchors(), seed=seed)
    if subset == "sampler" and n_anchors not in (smp.n_near, smp.n_near + smp.n_mid, smp.K):
        raise ValueError(f"subset 'sampler' takes n_anchors in {(smp.n_near, smp.n_near + smp.n_mid, smp.K)}")
    if not 1 <= n_anchors <= AS.N_ANCHORS:
        raise ValueError(f"n_anchors {n_anchors} out of range")
    if subset == "file":
        if not idx_file or not Path(idx_file).is_file():
            raise ValueError("subset 'file' needs --idx-file")
        idx_file = str(Path(idx_file).resolve())
    toks = pd.read_parquet(CM.packed_dir(split, root) / "tokens.parquet", columns=["token"]).token.astype(str)
    cfg = dict(version=VERSION, variants_version=VV.VERSION, split=split, combine=combine, subset=subset,
               n_anchors=int(n_anchors), epoch=int(epoch) if subset == "sampler" else None, seed=int(seed),
               idx_file=idx_file if subset == "file" else "",
               idx_file_sha16=CM.sha256_file(idx_file)[:16] if subset == "file" else "",
               speeds=[float(a) for a in speeds], lats=[float(d) for d in lats], s_on_frac=float(s_on_frac),
               n_variants=len(tab), names=[VV.variant_name(a, d) for a, d in tab], compute_dtype="torch.float64",
               anchors_file=str(AS.ANCHORS_FILE), anchors_sha16=AS.ANCHORS_SHA16, sampler=smp.config(),
               packed=str(CM.packed_dir(split, root)), n_tokens=len(toks),
               token_sha16=CM.sha16_array(np.array(toks.tolist(), "U16")), v0="hypot(status[4], status[5]), NaN -> 0")
    if ext != "const_curv":   # accel path extension (variants.make_variants ext); default keeps old cfg_sha16
        if ext not in VV.EXT_MODES:
            raise ValueError(f"ext must be one of {VV.EXT_MODES}")
        cfg["ext"] = ext
    cfg["cfg_sha16"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    cfg["limit"] = int(limit)                                # not part of cfg_sha16 (but pinned by label_cands)
    return cfg


# ----------------------------------------------------------------------------------------------- build (workers)
_B: Dict[str, object] = {}


def _binit(cfg: Dict, root):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    pk = CM.packed_dir(cfg["split"], root)
    _B["cfg"] = cfg
    _B["tokens"] = pd.read_parquet(pk / "tokens.parquet", columns=["token"]).token.astype(str).tolist()
    _B["gt"] = np.load(pk / "gt_traj.npy", mmap_mode="r")
    _B["status"] = np.load(pk / "status.npy", mmap_mode="r")
    _B["labels"] = AS.LabelSource()
    _B["sampler"] = AS.AnchorSampler(anchors=AS.load_anchors(), label_source=_B["labels"], seed=cfg["seed"])
    _B["idx_file"] = np.load(cfg["idx_file"], mmap_mode="r") if cfg["subset"] == "file" else None


def token_subset(row: int) -> Dict:
    """Anchor subset of one packed row -> dict(idx [K], group, pi, rank, dist, pass)."""
    cfg, smp = _B["cfg"], _B["sampler"]
    tok = _B["tokens"][row]
    gt = np.asarray(_B["gt"][row], np.float32)
    lab = _B["labels"].get(tok)
    if lab is None:
        raise KeyError(f"row {row} {tok}: no raw256 labels")
    d = smp.design(gt, lab)                                  # raises on non-finite GT
    n = cfg["n_anchors"]
    if cfg["subset"] == "sampler":
        idx = smp.sample(tok, gt, lab, epoch=cfg["epoch"], shuffle=False)["idx"][:n]     # canonical order: prefix
    elif cfg["subset"] == "nearest":
        idx = d["order"][:n]
    else:
        idx = np.asarray(_B["idx_file"][row][:n], np.int64)
        if len(idx) != n or idx.min() < 0 or idx.max() >= AS.N_ANCHORS or len(np.unique(idx)) != n:
            raise ValueError(f"row {row}: bad idx-file entry {idx}")
    idx = np.asarray(idx, np.int64)
    return dict(idx=idx, group=np.minimum(d["stratum"][idx], AS.GROUP_STRAT).astype(np.int8),
                pi=d["pi"][idx].astype(np.float32), rank=d["rank"][idx].astype(np.int16),
                dist=d["dist"][idx].astype(np.float32), passed=d["pass"][idx])


def build_chunk(rows) -> Dict:
    t0 = time.time()
    cfg = _B["cfg"]
    rows = np.asarray(rows, np.int64)
    subs = [token_subset(int(r)) for r in rows]
    anchors = _B["sampler"].anchors
    tau = np.stack([anchors[s["idx"]] for s in subs]).astype(np.float32)               # [T, K, 8, 3]
    st = np.asarray(_B["status"][rows], np.float64)
    v0 = np.nan_to_num(np.hypot(st[:, 4], st[:, 5]), nan=0.0)                           # [T] (= cf_variants)
    out = VV.make_variants(torch.from_numpy(tau), torch.from_numpy(v0), cfg["speeds"], cfg["lats"], cfg["combine"],
                           cfg["s_on_frac"], compute_dtype=torch.float64, ext=cfg.get("ext", "const_curv"))
    T, K = tau.shape[:2]
    V = out["traj"].shape[2]
    assert np.array_equal(out["traj"][:, :, 0].numpy(), tau), "identity variant must be the anchor bytes"
    m = out["meta"]
    res = dict(rows=rows, traj=out["traj"].numpy().reshape(T, K * V, 8, 3), v0=v0.astype(np.float32),
               valid=out["valid"].numpy(), S8=m["S8"].numpy(),
               **{k: np.stack([s[k] for s in subs]) for k in ("idx", "group", "pi", "rank", "dist", "passed")},
               **{k: m[k].numpy() for k in DEC_KEYS})
    res["sec"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------------------------- build (main)
def build(cfg: Dict, out: Path, root=None, build_workers: int = 8, rows_per_task: int = 256) -> Dict:
    assert 1 <= build_workers <= 16, "<= 16 build workers"
    N, K, V = cfg["n_tokens"], cfg["n_anchors"], cfg["n_variants"]
    rows = np.arange(cfg["limit"] if cfg["limit"] else N)
    t0 = time.time()
    w = CM.MemmapWriter(out / "traj.npy", np.float32, (N, K * V, 8, 3))              # untouched rows stay 0
    ix = dict(anchor_idx=np.full((N, K), -1, np.int16), group=np.full((N, K), -1, np.int8),
              pi=np.full((N, K), np.nan, np.float32), rank=np.full((N, K), -1, np.int16),
              dist=np.full((N, K), np.nan, np.float32), passed=np.zeros((N, K), bool),
              v0=np.full(N, np.nan, np.float32), built=np.zeros(N, bool), valid=np.zeros((N, K, V), bool),
              S8=np.full((N, K), np.nan, np.float32))
    dec = {k: np.full((N, K, V), np.nan, np.float32) for k in DEC_KEYS}
    tasks = [rows[i:i + rows_per_task] for i in range(0, len(rows), rows_per_task)]
    sec = 0.0
    nw = max(1, min(build_workers, len(tasks)))
    with Pool(nw, initializer=_binit, initargs=(cfg, root)) as pool:
        for j, r in enumerate(pool.imap_unordered(build_chunk, tasks)):
            rr = r["rows"]
            w.arr[rr] = r["traj"]
            for a, b in (("anchor_idx", "idx"), ("group", "group"), ("pi", "pi"), ("rank", "rank"),
                         ("dist", "dist"), ("passed", "passed"), ("v0", "v0"), ("valid", "valid"), ("S8", "S8")):
                ix[a][rr] = r[b]
            for k in DEC_KEYS:
                dec[k][rr] = r[k]
            ix["built"][rr] = True
            sec += r["sec"]
            if (j + 1) % 50 == 0 or j + 1 == len(tasks):
                print(f"[{time.strftime('%T')}] build {j + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    w.finalize()
    tab = VV.variant_table(cfg["speeds"], cfg["lats"], cfg["combine"])
    col = np.arange(K * V)
    CM.atomic_savez(out / "index.npz", **{("pass" if k == "passed" else k): v for k, v in ix.items()},
                    col_k=(col // V).astype(np.int16), col_v=(col % V).astype(np.int16),
                    speed=np.array([a for a, _ in tab], np.float32), lat=np.array([d for _, d in tab], np.float32),
                    names=np.array(cfg["names"]))
    CM.atomic_savez(out / "dec.npz", **dec)
    info = dict(cfg, n_built=int(ix["built"].sum()), build_cpu_s=round(sec, 2),
                build_cpu_s_per_token=sec / max(1, len(rows)), build_wall_s=round(time.time() - t0, 1),
                build_workers=nw, built=time.strftime("%F %T"),
                n_invalid=int((~ix["valid"][ix["built"]]).sum()),
                traj_sha16=CM.sha16_array(np.load(out / "traj.npy", mmap_mode="r")))
    CM.atomic_write_json(out / "build.json", info)
    print(f"built {out}/traj.npy: {info['n_built']} rows x {K * V} trajectories, {info['build_wall_s']}s wall "
          f"({info['build_cpu_s_per_token'] * 1e3:.1f} ms CPU / token), invalid {info['n_invalid']}", flush=True)
    return info


def ensure_built(cfg: Dict, out: Path, root, build_workers: int, fresh: bool) -> Dict:
    bj = out / "build.json"
    if fresh and out.exists():
        shutil.rmtree(out)
    if bj.is_file() and (out / "traj.npy").is_file():
        old = json.loads(bj.read_text())
        if old.get("cfg_sha16") == cfg["cfg_sha16"] and old.get("limit") == cfg["limit"]:
            print(f"traj.npy up to date (cfg {cfg['cfg_sha16']}, {old['n_built']} rows)", flush=True)
            return old
        if (out / "shards").exists():
            raise SystemExit(f"{out}: built with a different config / limit ({old.get('cfg_sha16')}, "
                             f"limit {old.get('limit')}); use --fresh or another --name")
    out.mkdir(parents=True, exist_ok=True)
    return build(cfg, out, root, build_workers)


# ----------------------------------------------------------------------------------------------- check
def _eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise exact equality with NaN == NaN."""
    return (a == b) | (np.isnan(a) & np.isnan(b))


def check(out: Path, root=None) -> Dict:
    cfg = json.loads((out / "build.json").read_text())
    split, K, V = cfg["split"], cfg["n_anchors"], cfg["n_variants"]
    lab = np.load(out / "labels.npy", mmap_mode="r")
    ok = np.load(out / "ok.npy")
    ix = np.load(out / "index.npz")
    rows = np.flatnonzero(ix["built"] & ok.any(1))
    L = np.asarray(lab[rows], np.float32).reshape(len(rows), K, V, -1)
    O = ok[rows].reshape(len(rows), K, V)
    aidx = ix["anchor_idx"][rows].astype(np.int64)
    valid = ix["valid"][rows]
    res = dict(name=out.name, split=split, n_rows=int(len(rows)), K=K, V=V, names=cfg["names"],
               n_ok=int(O.sum()), n_expected=int(len(rows) * K * V))
    # identity vs raw256 (same anchors, same scorer): all 9 columns exact
    rd = CM.labels_dir(split, "raw256", root)
    if (rd / "labels.npy").is_file():
        R = np.asarray(np.load(rd / "labels.npy", mmap_mode="r")[rows], np.float32)       # [n, 256, 9]
        Rok = np.load(rd / "ok.npy", mmap_mode="r")[rows]
        Ri = np.take_along_axis(R, aidx[:, :, None], 1)                                     # [n, K, 9]
        m = O[:, :, 0] & np.take_along_axis(np.asarray(Rok), aidx, 1)
        eq = _eq(L[:, :, 0], Ri)[m]                                                        # [n_m, 9]
        res["identity_vs_raw256"] = dict(n_compared=int(m.sum()), all_equal=bool(eq.all()),
                                         n_equal_per_col={c: int(eq[:, i].sum()) for c, i in LCOL.items()},
                                         max_abs_diff=float(np.nanmax(np.abs(L[:, :, 0][m] - Ri[m]))) if m.any()
                                         else None)
    # identity vs WoTE pdm_score_256 (NC / DAC / TTC / C)
    if Path(str(AS.WOTE_LABELS) + ".npy").is_file():
        dis, AS.log.disabled = AS.log.disabled, True           # deliberate WoTE source: no fallback WARNING
        try:
            src = AS.LabelSource(prefer_official=False)
        finally:
            AS.log.disabled = dis
        toks = pd.read_parquet(CM.packed_dir(split, root) / "tokens.parquet", columns=["token"]).token.astype(str)
        W = src.get_rows(toks.iloc[rows].tolist())                                          # [n, 5, 256]
        Wi = np.take_along_axis(W, aidx[:, None, :], 2)                                     # [n, 5, K]
        m = O[:, :, 0] & np.isfinite(Wi[:, list(AS.PASS_IDX)]).all(1)
        per = {}
        for k in WOTE_KEYS:
            per[k] = int(_eq(L[:, :, 0, LCOL[k]], Wi[:, AS.LABEL_KEYS.index(k)])[m].sum())
        res["identity_vs_wote_pdm_score_256"] = dict(n_compared=int(m.sum()), n_equal=per,
                                                     all_equal=all(v == int(m.sum()) for v in per.values()))
    # lateral-disabled duplicates (valid = False) vs their speed-only parent
    tab = VV.variant_table(cfg["speeds"], cfg["lats"], cfg["combine"])
    parent = np.array([tab.index((a, 0.0)) for a, _ in tab])
    Lp = L[:, :, parent]
    mi = ~valid & O & O[:, :, parent]
    res["invalid_vs_parent"] = dict(n_invalid=int((~valid).sum()), n_compared=int(mi.sum()),
                                    all_equal=bool(_eq(L, Lp).all(-1)[mi].all()) if mi.any() else None)
    # per-variant summary over valid, scored entries
    pv = (L[..., PASS_COLS] >= 1.0 - 1e-6).all(-1)
    per_v = {}
    for v, nm in enumerate(cfg["names"]):
        mm = O[:, :, v] & valid[:, :, v]
        same = _eq(L[:, :, v], L[:, :, 0]).all(-1) & mm
        per_v[nm] = dict(n=int(mm.sum()), pass_rate=float(pv[:, :, v][mm].mean()) if mm.any() else None,
                         mean_pdms=float(L[:, :, v, LCOL["pdms"]][mm].mean()) if mm.any() else None,
                         mean_ep=float(L[:, :, v, LCOL["ep"]][mm].mean()) if mm.any() else None,
                         frac_label_eq_identity=float(same.sum() / max(1, mm.sum())))
    res["per_variant"] = per_v
    # timing (label_cands shards: seconds per token, measured inside the workers)
    parts = [pd.read_parquet(f, columns=["token", "k", "sec_load", "sec_score"]) for f in sorted(
        (out / "shards").glob("c*.parquet"))]
    if parts:
        df = pd.concat(parts)
        sec = df[df.k >= 0].groupby("token")[["sec_load", "sec_score"]].first()
        meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").is_file() else {}
        res["timing"] = dict(n_tokens=int(len(sec)), workers=meta.get("workers"), chunk=meta.get("chunk"),
                             sec_load_mean=float(sec.sec_load.mean()), sec_score_mean=float(sec.sec_score.mean()),
                             sec_score_median=float(sec.sec_score.median()),
                             sec_score_per_traj=float(sec.sec_score.mean() / (K * V)),
                             build_cpu_s_per_token=cfg.get("build_cpu_s_per_token"), wall_s=meta.get("wall_s"))
    res["checked"] = time.strftime("%F %T")
    CM.atomic_write_json(out / "check.json", res)
    return res


# ----------------------------------------------------------------------------------------------- main
def run(split: str, combine: str = "separate", subset: str = "sampler", n_anchors: int = 16, epoch: int = 0,
        seed: int = 0, idx_file: str = "", speeds=VV.SPEEDS, lats=VV.LATS, s_on_frac: float = VV.S_ON_FRAC,
        name: str = "", out_root=OUT_ROOT, root=None, limit: int = 0, workers: int = 48, chunk: int = 64,
        build_workers: int = 8, build_only: bool = False, assemble_only: bool = False, check_only: bool = False,
        retry_errors: bool = False, fresh: bool = False, ext: str = "const_curv") -> Optional[Dict]:
    name = name or default_name(combine, subset, n_anchors)
    out = out_dir(split, name, out_root)
    if check_only:
        return check(out, root)
    cfg = make_config(split, combine, subset, n_anchors, epoch, seed, idx_file, speeds, lats, s_on_frac, limit, root,
                      ext=ext)
    print(f"[{time.strftime('%F %T')}] {split}/{name}: subset {subset} K {n_anchors} x V {cfg['n_variants']} "
          f"({combine}) = {n_anchors * cfg['n_variants']} trajectories / token, limit {limit or 'all'} -> {out}",
          flush=True)
    if assemble_only:
        if not (out / "build.json").is_file():
            raise SystemExit(f"{out}/build.json missing")
    else:
        ensure_built(cfg, out, root, build_workers, fresh)
    if build_only:
        return None
    meta = LC.run(split, name, traj=str(out / "traj.npy"), root=root, workers=workers, limit=limit, chunk=chunk,
                  assemble_only=assemble_only, retry_errors=retry_errors, fresh=False, out_dir=str(out))
    if meta is None:
        return None
    res = check(out, root)
    print(json.dumps({k: res.get(k) for k in ("n_rows", "n_ok", "n_expected", "identity_vs_raw256",
                                                "identity_vs_wote_pdm_score_256", "invalid_vs_parent", "timing")},
                     indent=1), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=sorted(CM.SPLITS))
    ap.add_argument("--combine", default="separate", choices=VV.COMBINES)
    ap.add_argument("--subset", default="sampler", choices=SUBSETS)
    ap.add_argument("--n-anchors", type=int, default=16)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--idx-file", default="")
    ap.add_argument("--speeds", type=float, nargs="+", default=list(VV.SPEEDS))
    ap.add_argument("--lats", type=float, nargs="*", default=list(VV.LATS))
    ap.add_argument("--ext", default="const_curv", choices=list(VV.EXT_MODES),
                    help="accel path extension beyond S_8 (variants.make_variants ext); 'straight' for CK2 (user 10-08)")
    ap.add_argument("--s-on-frac", type=float, default=VV.S_ON_FRAC)
    ap.add_argument("--name", default="")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--root", default="", help="CK data root for packed inputs / raw256 (default CK_DATA)")
    ap.add_argument("--limit", type=int, default=0, help="first N packed tokens only (dry run; use its own --name)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--build-workers", type=int, default=8)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--build-only", action="store_true")
    g.add_argument("--assemble-only", action="store_true")
    g.add_argument("--check-only", action="store_true")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete this name's output dir (traj + shards) first")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    print("navsim", LC.navsim.__file__, "| score_trajectories", LC.ST.__file__, flush=True)
    run(a.split, a.combine, a.subset, a.n_anchors, a.epoch, a.seed, a.idx_file, a.speeds, a.lats, a.s_on_frac,
        a.name, a.out_root, a.root or None, a.limit, a.workers, a.chunk, a.build_workers, a.build_only,
        a.assemble_only, a.check_only, a.retry_errors, a.fresh, ext=a.ext)


if __name__ == "__main__":
    main()
