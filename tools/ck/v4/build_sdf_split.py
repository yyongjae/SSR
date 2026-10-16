#!/usr/bin/env python
"""refiner v4_3: official-DAC SDF of every token of a held-out split (navtrain_val, navtest) -- V4_3_SPEC M2 / M6.

  build_sdf_split.py build    --split {navtrain_val,navtest} [--workers 28] [--num-shards S --shard I] [--limit N]
                              [--tokens FILE] [--mc-root DIR] [--out-root DIR] [--force]
  build_sdf_split.py manifest --split S [--workers 28]          # recount / re-check the split -> manifest.json only
  build_sdf_split.py validate --split S [--n-random 400] [--seed 0] [--workers 8]
                                                                # stored field vs official DAC on F1 executed states

Field and file (identical to the navtrain root SDF_ROOT/navtrain, tools/refiner/build_sdf.py, 2026-10-03)
  navsim/agents/para_ssr/refiner/sdf.py: build_sdf_from_metric_cache(mc) -> float32 [320, 256] on the E grid (rows
  x = -8..72 m forward, cols y_left = +32..-32 m, 0.25 m, N frame = t0 rear axle), metres, + inside the union of the
  official DAC layers (ROADBLOCK, INTERSECTION, DRIVABLE_AREA, CARPARK_AREA) of the metric-cache drivable_area_map,
  clipped to +-10 m, exact signed distance at the cell centres; save_sdf -> <out-root>/<split>/<token>.npz (float16
  'sdf' + provenance keys version / token / grid / ego_xyh (global t0 rear axle x, y, heading) / layers / n_polys /
  n_invalid; atomic rename).  Same code path, same bytes: a navtrain token rebuilt here equals its stored file bit for
  bit (tests/test_v4_gt_splits.py).  sdf.py itself is unchanged (its sha256 goes into the manifest).
Tokens / caches  ck.constants.SPLITS[split]: 'tokens' parquet (row order kept; count and, where given, sha16 of the
  sorted tokens checked) and 'metric_cache' root -- the caches the official labels were scored on
  (tools/ck/data/common.mc_path: <root>/<log>/unknown/<token>/metric_cache.pkl; other scenario-type dirs of the log are
  searched as a fallback):  navtrain_val -> ssd/yongjae_refiner/metric_cache (18,179 tokens);
  navtest -> SSR/data/exp/metric_cache (12,146).  --tokens / --mc-root override (tests, smoke).
Output  <out-root>/<split>/<token>.npz; <out-root>/<split>/_build/stats.jsonl (one JSON line per built / rebuilt /
  failed token, the navtrain fields: token, status, log, mc_path, t_load, t_build, bytes, n_polys, n_invalid,
  n_segments, frac_inside), _build/summary_<stamp>_<pid>.json per pass, and _build/manifest.json (written last, over the
  WHOLE split, whatever shard ran): n_tokens, n_valid, complete, counts per check status, failures (token, status, last
  error), provenance (sdf_version, code sha256, tokens file + sha16, mc root, numpy / shapely versions, KST time).
  Log + pid: <out-root>/<split>/_build/build[_s<I>of<S>].{log,pid} (the navtrain build kept them in <out-root>/logs;
  here every new file stays inside the new split dir); validate writes _build/validate.log.
Resumable  an existing file is re-checked (load_sdf version + grid + shape / dtype, token key == file name, finite) and
  kept when valid ('exists'); an invalid one is rebuilt in place ('rebuilt'); --force rebuilds everything.  A token
  without a metric cache is 'no_mc' (counted as a failure in the manifest).  Every written file is re-loaded and
  compared bit for bit with the field before it counts as 'built'.
Shards  --num-shards S --shard I takes tokens I, I + S, ... (several processes / machines); --workers is per process and
  the total over concurrent shards must stay <= 28 (shared machine).  A live pid file of the same shard refuses a
  second start.
Validate (the phase-0 oracle check)  F1 executed states (ck2/analysis/detmap_path_ideas/F1/out/<val|navtest>/run.npz:
  sim [n, K, 41, 6] = the official LQR-tracked states in the N frame, groups par16 + var80 + latT16 [+ latM16]) -> the 4
  official ego corners (sdf.ego_corners: half length 2.588, half width 1.1485, centre 1.461 m ahead of the rear axle)
  -> bilinear stored field (sample_sdf_np, float16 upcast to float64).  Prediction: DAC pass <=> every in-grid corner
  of the 41 states has sdf > 0 (out-of-grid corners excluded and counted).  Official label: navtrain_val
  labels/navtrain_val/cand/labels.npy[row, k, dac] (= F1 exec, cross-checked), navtest F1 exec (official scorer on the
  same states).  Reported for the v2 top-1 (par k = 0) on --n-random seeded tokens (headline, >= 99 % agreement) and
  on the whole F1 sample, all trajectories of the random tokens, and per disagreement the exact signed distance of
  the corners to the uncropped official-layer union from the metric cache (sdf.signed_distance_exact).
  -> <out-root>/<split>/_build/validation_dac.json.
CPU only (CUDA_VISIBLE_DEVICES=), OMP_NUM_THREADS = MKL_NUM_THREADS = 1, run under nice -n 10.  Never touches
SDF_ROOT/navtrain or any input.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import lzma
import os
import pickle
import sys
import time
import traceback
import zipfile
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[3]          # repo root of this worktree
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from navsim.agents.para_ssr.ck import constants as C  # noqa: E402
from navsim.agents.para_ssr.refiner import sdf as SDF  # noqa: E402

SPLITS_SDF = ("navtrain_val", "navtest")
MAX_WORKERS = 28
CK = Path(C.CK_DATA)
F1_OUT = CK / "ck2/analysis/detmap_path_ideas/F1/out"
F1_SPLIT = {"navtrain_val": "val", "navtest": "navtest"}
VAL_LABELS = CK / "labels/navtrain_val/cand/labels.npy"     # [N, 16, 9] official, packed row order
DAC_COL = C.LBL["dac"]                                       # label column of DAC
F1_DAC_COL = 1                                               # F1 exec / plan columns: nc, dac, ddc, ep, ...
TRANSIENT = ("exists",)                                      # statuses not written to stats.jsonl (navtrain rule)
CODE = (REPO / "navsim/agents/para_ssr/refiner/sdf.py", Path(__file__).resolve())


def _kst() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json_safe(v):
    """Recursive: numpy scalars / arrays -> Python, non-finite floats -> None (strict JSON)."""
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def _write_json(p: Path, obj) -> None:
    """Atomic strict-JSON write (tmp + rename)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(obj), indent=1, allow_nan=False))
    os.replace(tmp, p)


# ----------------------------------------------------------------------------------------------- inputs
def tokens_sha16(tokens: Sequence[str]) -> str:
    """sha256('\\n'.join(sorted tokens))[:16] -- the ck.constants SPLITS 'sha16_sorted_tokens' rule."""
    return hashlib.sha256("\n".join(sorted(map(str, tokens))).encode()).hexdigest()[:16]


def split_tokens(split: str, tokens_file: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """-> (DataFrame token, log in file order, info).  The split file is checked against SPLITS n / sha16; a --tokens
    override (parquet / csv with token + log) is only checked for duplicates."""
    sp = C.SPLITS[split]
    f = Path(tokens_file) if tokens_file else Path(sp["tokens"])
    df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
    df = df[["token", "log"]].astype(str).reset_index(drop=True)
    if df.token.duplicated().any():
        raise ValueError(f"{f}: duplicate tokens")
    sha = tokens_sha16(df.token)
    if tokens_file is None:
        if len(df) != int(sp["n"]):
            raise ValueError(f"{f}: {len(df)} tokens != SPLITS[{split!r}]['n'] = {sp['n']}")
        if sp.get("sha16_sorted_tokens") and sha != sp["sha16_sorted_tokens"]:
            raise ValueError(f"{f}: sorted-token sha16 {sha} != {sp['sha16_sorted_tokens']}")
    return df, dict(tokens_file=str(f), n=len(df), sha16_sorted_tokens=sha, override=tokens_file is not None)


def mc_path(mc_root: Path, log: str, token: str) -> Optional[Path]:
    """<root>/<log>/unknown/<token>/metric_cache.pkl (common.mc_path), else the first other scenario-type dir of the
    log that has it, else None."""
    p = Path(mc_root) / log / "unknown" / token / "metric_cache.pkl"
    if p.is_file():
        return p
    d = Path(mc_root) / log
    if d.is_dir():
        for st in sorted(os.scandir(d), key=lambda e: e.name):
            q = Path(st.path) / token / "metric_cache.pkl"
            if st.is_dir() and q.is_file():
                return q
    return None


# ----------------------------------------------------------------------------------------------- per token
def check_file(path: Path, token: str) -> str:
    """'valid' | 'missing' | 'invalid: <reason>' for a stored field (load_sdf checks + token key + finite)."""
    path = Path(path)
    if not path.is_file():
        return "missing"
    try:
        f = SDF.load_sdf(path, check=True)
        with np.load(path, allow_pickle=False) as z:
            tk = str(z["token"])
        if tk != token:
            return f"invalid: token key {tk!r}"
        if not np.isfinite(f).all():
            return "invalid: non-finite"
    except Exception as e:   # noqa: BLE001 -- a broken file is rebuilt, not fatal
        return f"invalid: {type(e).__name__}: {e}"[:300]
    return "valid"


def build_one(job) -> Dict:
    """(token, log, mc_root, out_path, force) -> stats dict (never raises); the navtrain build_sdf.build_one record."""
    tok, log, mc_root, out_path, force = job
    out_path = Path(out_path)
    rec = dict(token=tok, status="", log=log, mc_path="", t_load=np.nan, t_build=np.nan, bytes=0, n_polys=-1,
               n_invalid=-1, n_segments=-1, frac_inside=np.nan)
    try:
        prior = check_file(out_path, tok)
        if prior == "valid" and not force:
            rec.update(status="exists", bytes=out_path.stat().st_size)
            return rec
        p = mc_path(Path(mc_root), log, tok)
        if p is None:
            rec["status"] = "no_mc"
            return rec
        rec["mc_path"] = str(p)
        t0 = time.time()
        with lzma.open(p, "rb") as fh:
            mc = pickle.load(fh)
        t1 = time.time()
        field, info = SDF.build_sdf_from_metric_cache(mc)
        t2 = time.time()
        nbytes = SDF.save_sdf(out_path, field, token=tok, ego_xyh=info["ego_xyh"], info=info)
        back = SDF.load_sdf(out_path, check=True)                            # written bytes == the field
        if not np.array_equal(back.view(np.uint16), np.asarray(field, np.float32).astype(np.float16).view(np.uint16)):
            raise RuntimeError("re-loaded field differs from the built one")
        rec.update(status="built" if prior == "missing" else "rebuilt", t_load=t1 - t0, t_build=t2 - t1,
                   bytes=nbytes, n_polys=info["n_polys"], n_invalid=info["n_invalid"],
                   n_segments=info["n_segments"], frac_inside=info["frac_inside"])
        if prior.startswith("invalid"):
            rec["prior"] = prior
    except Exception as e:  # noqa: BLE001 -- recorded, the run continues
        rec["status"] = f"error: {type(e).__name__}: {e}"[:300]
        rec["traceback"] = traceback.format_exc()[-2000:]
    return rec


def _check_job(job) -> Tuple[str, str]:
    tok, path = job
    return tok, check_file(Path(path), tok)


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _pool(workers: int):
    return get_context("fork").Pool(workers, initializer=_worker_init, maxtasksperchild=500)


def _imap(fn, jobs: List, workers: int, chunksize: int = 8):
    if workers <= 1 or len(jobs) <= 1:
        yield from map(fn, jobs)
        return
    with _pool(workers) as pool:
        yield from pool.imap_unordered(fn, jobs, chunksize=max(1, chunksize))


# ----------------------------------------------------------------------------------------------- logging / pid
class Log:
    """Progress lines to stdout and (append) to the shard's log file."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[build_sdf_split] {time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a") as fh:
                fh.write(line + "\n")


def _pid_alive_builder(pid: int) -> bool:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "build_sdf_split" in cmd


def claim_pid(pid_file: Path) -> None:
    """Write our pid; refuse when the file names another live build_sdf_split process (same shard running)."""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    if pid_file.is_file():
        try:
            old = int(pid_file.read_text().strip() or 0)
        except ValueError:
            old = 0
        if old and old != os.getpid() and _pid_alive_builder(old):
            raise SystemExit(f"{pid_file}: build_sdf_split pid {old} is still running")
    pid_file.write_text(f"{os.getpid()}\n")


# ----------------------------------------------------------------------------------------------- build / manifest
def _stats(recs: List[Dict], wall: float) -> Dict:
    st: Dict[str, int] = {}
    for r in recs:
        k = "error" if r["status"].startswith("error") else r["status"]
        st[k] = st.get(k, 0) + 1
    b = [r for r in recs if r["status"] in ("built", "rebuilt")]

    def q(key):
        v = np.array([r[key] for r in b], np.float64)
        if not len(v):
            return {}
        return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.quantile(v, 0.95)),
                    max=float(v.max()))

    return dict(n=len(recs), status=st, wall_s=wall, t_load_s=q("t_load"), t_build_s=q("t_build"), bytes=q("bytes"),
                n_invalid_polys_total=int(sum(r["n_invalid"] for r in b)), sdf_version=SDF.SDF_VERSION)


def _last_failures(stats_f: Path) -> Dict[str, Dict]:
    """token -> its last stats.jsonl record (to attach the last error to a token that is still not valid)."""
    out: Dict[str, Dict] = {}
    if stats_f.is_file():
        with open(stats_f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[r["token"]] = r
    return out


def write_manifest(a, df: pd.DataFrame, tok_info: Dict, mc_root: Path, log: Log,
                   run_summary: Optional[Dict] = None) -> Dict:
    """Check every token of the split (not only this shard) -> _build/manifest.json (atomic)."""
    out_dir = Path(a.out_root) / a.split
    jobs = [(t, str(SDF.sdf_path(t, a.split, a.out_root))) for t in df.token]
    status: Dict[str, str] = {}
    for tok, s in _imap(_check_job, jobs, a.workers, chunksize=64):
        status[tok] = s
    counts: Dict[str, int] = {}
    for s in status.values():
        k = "invalid" if s.startswith("invalid") else s
        counts[k] = counts.get(k, 0) + 1
    last = _last_failures(out_dir / "_build" / "stats.jsonl")
    bad = [t for t in df.token if status[t] != "valid"]
    fails = []
    for t in bad:
        r = last.get(t, {})
        fails.append(dict(token=t, check=status[t], last_status=r.get("status", ""), mc_path=r.get("mc_path", "")))
    import shapely
    man = dict(
        split=a.split, out_dir=str(out_dir), n_tokens=len(df), n_valid=counts.get("valid", 0),
        complete=counts.get("valid", 0) == len(df), check_counts=counts, n_failures=len(fails), failures=fails[:1000],
        sdf_version=SDF.SDF_VERSION, grid=[SDF.E_X0, SDF.E_Y0, SDF.E_RES, SDF.E_H, SDF.E_W, SDF.SDF_CLIP],
        layers=list(SDF.DAC_LAYER_NAMES), tokens=tok_info, mc_root=str(mc_root),
        code_sha256={str(p): _sha256(p) for p in CODE}, numpy=np.__version__, shapely=shapely.__version__,
        same_as="SDF_ROOT/navtrain (tools/refiner/build_sdf.py): same sdf.py functions and save_sdf layout",
        written=_kst(), argv=sys.argv, pid=os.getpid(), last_run=run_summary)
    _write_json(out_dir / "_build" / "manifest.json", man)
    log(f"manifest {a.split}: {man['n_valid']}/{len(df)} valid, failures {len(fails)} {counts} "
        f"-> {out_dir / '_build' / 'manifest.json'}")
    return man


def _shard_tag(a) -> str:
    return f"_s{a.shard}of{a.num_shards}" if a.num_shards > 1 else ""


def run_build(a) -> Dict:
    df, tok_info = split_tokens(a.split, a.tokens)
    mc_root = Path(a.mc_root or C.SPLITS[a.split]["metric_cache"])
    logs = Path(a.out_root) / a.split / "_build"
    tag = f"build{_shard_tag(a)}"
    log = Log(logs / f"{tag}.log")
    claim_pid(logs / f"{tag}.pid")
    sel = df.iloc[a.shard::a.num_shards]
    if a.limit:
        sel = sel.iloc[: a.limit]
    out_dir = Path(a.out_root) / a.split
    (out_dir / "_build").mkdir(parents=True, exist_ok=True)
    jobs = [(t, lg, str(mc_root), str(SDF.sdf_path(t, a.split, a.out_root)), a.force)
            for t, lg in zip(sel.token, sel.log)]
    log(f"split={a.split} tokens={len(df)} shard={a.shard}/{a.num_shards} jobs={len(jobs)} mc_root={mc_root} "
        f"out={out_dir} workers={a.workers} force={a.force} pid={os.getpid()}")
    stats_f = out_dir / "_build" / "stats.jsonl"
    recs: List[Dict] = []
    t0 = last = time.time()
    n_new = 0
    with open(stats_f, "a") as fo:
        for k, rec in enumerate(_imap(build_one, jobs, a.workers, a.chunksize), 1):
            recs.append(rec)
            if rec["status"] not in TRANSIENT:
                fo.write(json.dumps(_json_safe(rec)) + "\n")
                fo.flush()
            n_new += rec["status"] in ("built", "rebuilt")
            if rec["status"].startswith("error") or rec["status"] == "no_mc":
                log(f"{rec['token']}: {rec['status']}")
            now = time.time()
            if now - last > 60 or k == len(jobs):
                last = now
                rate = n_new / max(now - t0, 1e-9)
                todo = len(jobs) - k
                log(f"{k}/{len(jobs)} done, built {n_new}  {now - t0:.0f}s  {rate:.2f} tok/s  "
                    f"ETA {todo / max(rate, 1e-9) / 60 if n_new else 0:.1f} min")
    summ = _stats(recs, time.time() - t0)
    summ.update(split=a.split, n_tokens=len(df), shard=a.shard, num_shards=a.num_shards, workers=a.workers,
                mc_root=str(mc_root), tokens=tok_info, written=_kst())
    if recs:
        _write_json(out_dir / "_build" / f"summary_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json", summ)
    log("summary " + json.dumps(_json_safe(summ)))
    man = write_manifest(a, df, tok_info, mc_root, log, run_summary=summ)
    return dict(summary=summ, manifest=man)


# ----------------------------------------------------------------------------------------------- validation
def npz_member(path: Path, name: str, mmap: bool = True) -> np.ndarray:
    """One array of an .npz; memory-mapped when the member is stored uncompressed (np.savez), else loaded."""
    path = Path(path)
    if mmap:
        with zipfile.ZipFile(path) as zf:
            zi = zf.getinfo(name + ".npy")
            if zi.compress_type == zipfile.ZIP_STORED:
                with open(path, "rb") as fh:
                    fh.seek(zi.header_offset)
                    h = fh.read(30)
                    if h[:4] != b"PK\x03\x04":
                        raise ValueError(f"{path}:{name}: bad local header")
                    n_name, n_extra = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
                    fh.seek(zi.header_offset + 30 + n_name + n_extra)
                    ver = np.lib.format.read_magic(fh)
                    rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
                    shape, fortran, dtype = rd(fh)
                    off = fh.tell()
                if dtype.hasobject:
                    raise ValueError(f"{path}:{name}: object array")
                return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape,
                                 order="F" if fortran else "C")
    with np.load(path, allow_pickle=False) as z:
        return z[name]


def ego_corners_np(poses: np.ndarray) -> np.ndarray:
    """float64 twin of sdf.ego_corners: rear-axle poses [..., 3] -> corners [..., 4, 2] (FL, RL, RR, FR)."""
    poses = np.asarray(poses, np.float64)
    x, y, h = poses[..., 0], poses[..., 1], poses[..., 2]
    c, s = np.cos(h), np.sin(h)
    cx, cy = x + SDF.EGO_RA2C * c, y + SDF.EGO_RA2C * s
    hl, hw = SDF.EGO_HALF_LEN, SDF.EGO_HALF_WID
    out = [np.stack([cx + sl * hl * c - sw * hw * s, cy + sl * hl * s + sw * hw * c], -1)
           for sl, sw in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))]
    return np.stack(out, -2)


def corner_eval(field: np.ndarray, states: np.ndarray) -> Dict[str, np.ndarray]:
    """field [320, 256] (any float), states [K, T, >= 3] (x, y, heading; N frame) ->
    pred_pass [K] bool (every in-grid corner sdf > 0), min_sdf [K] (min over in-grid corners; +inf if none),
    n_oog [K] (out-of-grid corners), first_t [K] (first state index with an in-grid corner <= 0, -1 if none)."""
    cor = ego_corners_np(states[..., :3])                    # [K, T, 4, 2]
    val, valid = SDF.sample_sdf_np(field, cor)               # [K, T, 4]
    bad = valid & ~(val > 0)
    mn = np.where(valid, val, np.inf).min(axis=(1, 2))
    t_bad = bad.any(-1)
    first = np.where(t_bad.any(1), t_bad.argmax(1), -1)
    return dict(pred_pass=~bad.any(axis=(1, 2)), min_sdf=mn, n_oog=(~valid).sum(axis=(1, 2)), first_t=first)


def _confusion(pred_pass: np.ndarray, off_pass: np.ndarray) -> Dict:
    pred_pass, off_pass = np.asarray(pred_pass, bool), np.asarray(off_pass, bool)
    n = int(len(pred_pass))
    tp = int((~pred_pass & ~off_pass).sum())        # both fail
    tn = int((pred_pass & off_pass).sum())
    fa = int((~pred_pass & off_pass).sum())         # false alarm (sdf fail, official pass)
    ms = int((pred_pass & ~off_pass).sum())         # miss (sdf pass, official fail)
    return dict(n=n, agree=tp + tn, agree_frac=(tp + tn) / max(n, 1), official_fail=int((~off_pass).sum()),
                both_fail=tp, both_pass=tn, false_alarm=fa, miss=ms,
                fail_recall=tp / max(tp + ms, 1) if tp + ms else None,
                pass_recall=tn / max(tn + fa, 1) if tn + fa else None)


def _exact_job(job) -> Dict:
    """Disagreement diagnosis: exact signed distance of the 41 x 4 corners to the UNCROPPED official-layer union."""
    tok, mc_p, states, ego_xyh_file = job
    try:
        with lzma.open(mc_p, "rb") as fh:
            mc = pickle.load(fh)
        ra = mc.ego_state.rear_axle
        xyh = (float(ra.x), float(ra.y), float(ra.heading))
        union, _ = SDF.drivable_union(mc.drivable_area_map, xyh, crop=False)
        cor = ego_corners_np(np.asarray(states, np.float64)[..., :3])        # [T, 4, 2]
        sd = SDF.signed_distance_exact(union, cor)
        return dict(token=tok, exact_min=float(sd.min()), exact_fail=bool((sd <= 0).any()),
                    exact_first_t=int(np.argmax((sd <= 0).any(-1))) if (sd <= 0).any() else -1,
                    ego_xyh_match=bool(np.allclose(xyh, ego_xyh_file, rtol=0, atol=1e-9)))
    except Exception as e:  # noqa: BLE001
        return dict(token=tok, error=f"{type(e).__name__}: {e}"[:300])


def run_validate(a) -> Dict:
    """Stored field vs official DAC on the F1 executed states (module docstring)."""
    split, out_root = a.split, Path(a.out_root)
    f1 = Path(a.f1_run) if a.f1_run else F1_OUT / F1_SPLIT[split] / "run.npz"
    log = Log(out_root / split / "_build" / "validate.log")
    toks = np.asarray(npz_member(f1, "tokens", mmap=False)).astype(str)
    rows = np.asarray(npz_member(f1, "rows", mmap=False), np.int64)
    fin = np.asarray(npz_member(f1, "fin", mmap=False), bool)
    ex = np.asarray(npz_member(f1, "exec", mmap=False))[..., F1_DAC_COL]           # [n, K]
    sim = npz_member(f1, "sim", mmap=True)                                          # [n, K, 41, 6] (mmap)
    groups = json.loads(str(npz_member(f1, "groups", mmap=False)))
    n, K = ex.shape
    log(f"validate {split}: F1 {f1} n={n} K={K} groups={groups}")
    df, _ = split_tokens(split)
    tok2log = dict(zip(df.token, df.log))
    if not all(t in tok2log for t in toks):
        raise SystemExit(f"F1 tokens are not all in {split}")
    lab_src = "F1 exec (official scorer on these LQR-tracked states; F1 verify.json: == stored labels)"
    off_top = ex[:, 0]
    if split == "navtrain_val":
        # F1 rows index the packed order (= the split order); the packed cand labels are the official v2 top-16 labels
        pk = pd.read_parquet(CK / "packed" / split / "tokens.parquet").token.astype(str).to_numpy()
        if not (pk[rows] == toks).all():
            raise SystemExit("F1 rows do not index packed/navtrain_val/tokens.parquet")
        lab = np.load(VAL_LABELS, mmap_mode="r")
        lab_par = np.asarray(lab[rows, :16, DAC_COL], np.float64)
        n_mis = int((lab_par != ex[:, :16]).sum())
        log(f"labels/navtrain_val/cand dac vs F1 exec (par16): {n_mis} mismatches of {lab_par.size}")
        if n_mis:
            raise SystemExit("official labels and F1 exec disagree; F1 states are not the labelled ones")
        off_top, lab_src = lab_par[:, 0], str(VAL_LABELS)
    rng = np.random.default_rng(a.seed)
    rand = np.sort(rng.choice(n, size=min(a.n_random, n), replace=False))

    def per_token(i: int, ks) -> Dict:
        field = SDF.load_sdf(SDF.sdf_path(str(toks[i]), split, out_root))
        return corner_eval(field, np.asarray(sim[i, ks], np.float64))

    # 1) v2 top-1 over the whole F1 sample (headline subset = rand)
    t0 = time.time()
    top = dict(pred_pass=np.zeros(n, bool), min_sdf=np.zeros(n), n_oog=np.zeros(n, np.int64),
               first_t=np.zeros(n, np.int64))
    missing = []
    for i in range(n):
        if not fin[i, 0]:
            continue
        try:
            r = per_token(i, [0])
        except FileNotFoundError:
            missing.append(str(toks[i]))
            continue
        for k in top:
            top[k][i] = r[k][0]
    okm = fin[:, 0] & ~np.isin(toks, missing)
    off_pass = off_top >= 1.0
    log(f"top-1 over {int(okm.sum())} tokens in {time.time() - t0:.0f}s (missing fields {len(missing)})")
    head = _confusion(top["pred_pass"][rand][okm[rand]], off_pass[rand][okm[rand]])
    full = _confusion(top["pred_pass"][okm], off_pass[okm])
    # 2) every trajectory of the random tokens (official = F1 exec; par cols checked == labels for val)
    allk = dict(pred=[], off=[], grp=[])
    names = np.empty(K, object)
    for g, s, ln in groups:
        names[s:s + ln] = g
    for i in rand:
        if not okm[i]:
            continue
        ks = np.nonzero(fin[i])[0]
        r = per_token(int(i), ks)
        allk["pred"].append(r["pred_pass"])
        allk["off"].append(ex[i, ks] >= 1.0)
        allk["grp"].append(names[ks])
    pa, oa, ga = (np.concatenate(allk[k]) for k in ("pred", "off", "grp"))
    by_group = {str(g): _confusion(pa[ga == g], oa[ga == g]) for g in dict.fromkeys(ga.tolist())}
    # 3) disagreements of the top-1 (whole sample): exact distance to the uncropped union
    dis = np.nonzero(okm & (top["pred_pass"] != off_pass))[0]
    mc_root = Path(C.SPLITS[split]["metric_cache"])
    jobs = []
    for i in dis[: a.max_exact]:
        p = mc_path(mc_root, tok2log[str(toks[i])], str(toks[i]))
        with np.load(SDF.sdf_path(str(toks[i]), split, out_root)) as z:
            exyh = np.asarray(z["ego_xyh"], np.float64)
        jobs.append((str(toks[i]), str(p), np.asarray(sim[i, 0], np.float64), exyh))
    exact = {r["token"]: r for r in _imap(_exact_job, jobs, a.workers, chunksize=1)}
    dis_rows = []
    for i in dis:
        e = exact.get(str(toks[i]), {})
        dis_rows.append(dict(token=str(toks[i]), row=int(rows[i]), in_headline=bool(np.isin(i, rand)),
                             official_dac=float(off_top[i]), sdf_pass=bool(top["pred_pass"][i]),
                             min_sdf=float(top["min_sdf"][i]), n_oog=int(top["n_oog"][i]),
                             sdf_first_t=int(top["first_t"][i]), **{k: v for k, v in e.items() if k != "token"}))
    res = dict(
        split=split, f1_run=str(f1), label_source=lab_src, seed=a.seed, n_f1=n,
        rule="DAC pass <=> every in-grid ego corner (official box) of the 41 executed states has stored sdf > 0",
        headline_top1_random=head, top1_all_f1=full, all_traj_random_tokens=dict(
            total=_confusion(pa, oa), by_group=by_group),
        top1_oog_tokens=int((top["n_oog"][okm] > 0).sum()), missing_fields=missing[:100], n_missing=len(missing),
        disagreements_top1=dis_rows, sdf_version=SDF.SDF_VERSION, written=_kst())
    res["pass"] = bool(head["agree_frac"] >= 0.99)
    out = out_root / split / "_build" / "validation_dac.json"
    _write_json(out, res)
    log(f"validate {split}: headline top-1 agree {head['agree']}/{head['n']} = {head['agree_frac']:.4f} "
        f"(official fail {head['official_fail']}, miss {head['miss']}, false alarm {head['false_alarm']}); "
        f"all F1 top-1 {full['agree']}/{full['n']}; all traj {res['all_traj_random_tokens']['total']['agree']}/"
        f"{res['all_traj_random_tokens']['total']['n']} -> {out}")
    return res


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("build", "manifest", "validate"))
    ap.add_argument("--split", required=True, choices=SPLITS_SDF)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tokens of the shard (0 = all)")
    ap.add_argument("--tokens", default=None, help="token file override (parquet / csv: token, log)")
    ap.add_argument("--mc-root", default=None, help="metric-cache root override")
    ap.add_argument("--out-root", default=str(SDF.SDF_ROOT))
    ap.add_argument("--force", action="store_true", help="rebuild valid files too")
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=400, help="validate: headline random tokens of the F1 sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-exact", type=int, default=200, help="validate: disagreements diagnosed with the exact union")
    ap.add_argument("--f1-run", default=None, help="validate: F1 run.npz override")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        ap.error(f"--workers must be in 1..{MAX_WORKERS} (shared machine)")
    if not (a.num_shards >= 1 and 0 <= a.shard < a.num_shards):
        ap.error("need 0 <= --shard < --num-shards")
    if a.mode == "build":
        return run_build(a)
    if a.mode == "manifest":
        df, info = split_tokens(a.split, a.tokens)
        return write_manifest(a, df, info, Path(a.mc_root or C.SPLITS[a.split]["metric_cache"]),
                              Log(Path(a.out_root) / a.split / "_build" / "build.log"))
    return run_validate(a)


if __name__ == "__main__":
    main()
