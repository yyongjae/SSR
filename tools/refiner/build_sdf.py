#!/usr/bin/env python
"""Build per-token drivable-area SDFs from PDM metric caches (IMPL_SPEC section 3.5).

Field: ``navsim.agents.para_ssr.refiner.sdf`` -- E grid [320, 256] (rows x = -8..72 m, cols y_left = +32..-32 m,
0.25 m), float16, metres, positive inside the union of the official DAC layers of the metric cache
``drivable_area_map`` (ROADBLOCK, INTERSECTION, DRIVABLE_AREA, CARPARK_AREA), negative outside, clipped to
+-10 m; exact signed distance at the cell centres (see the sdf.py docstring for why not a distance transform).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python \
      tools/refiner/build_sdf.py --subset navtrain --tokens <tokens.parquet> --workers 4

Tokens   : --tokens FILE (.parquet / .csv with a 'token' column and optionally 'log'; .txt one token per line;
           repeatable) or --all (every token found under the metric-cache roots).
Caches   : <root>/<log>/<scenario_type>/<token>/metric_cache.pkl, first hit in --mc-root order. Defaults:
           navtrain -> stage-T cache (ssd/yongjae_refiner/metric_cache), then E's navtrain cache;
           navtest  -> data/exp/metric_cache.
Output   : <out-root>/<subset>/<token>.npz (sdf.save_sdf, atomic rename) and one JSON line per built (or failed)
           token in <out-root>/<subset>/_build/stats.jsonl (status, log, t_load, t_build, bytes, n_polys, n_invalid,
           n_segments, frac_inside); a summary JSON per pass next to it.
Resumable: existing npz -> 'exists' (not rebuilt; --force rebuilds); no metric cache yet -> 'no_mc';
           metric cache written < --min-age s ago (a caching job may still be writing it) -> 'mc_fresh'.
           Re-run the same command later to fill the gaps, or pass --follow-interval S to keep making passes
           (every S s, up to --follow-max-h) until every token is built -- used to trail build_metric_cache.py.
Workers  : <= 4 processes (shared machine); CPU only.
"""
from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/home/external-user/yongjae/SSR")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from navsim.agents.para_ssr.refiner import sdf as SDF  # noqa: E402

DATA = Path("/home/external-user/ssd/yongjae_refiner")
DEFAULT_MC_ROOTS = {
    "navtrain": [DATA / "metric_cache",
                 ROOT / "report/cause_and_correction_tests/E_train_split_feasibility/metric_cache"],
    "navtest": [ROOT / "data/exp/metric_cache"],
}
MAX_WORKERS = 4


# ----------------------------------------------------------------------------------------- inputs
def read_tokens(files: Sequence[str]) -> Dict[str, Optional[str]]:
    """token -> log (None if the file has no log column); order preserved, duplicates dropped."""
    out: Dict[str, Optional[str]] = {}
    for f in files:
        p = Path(f)
        if p.suffix in (".parquet", ".csv"):
            import pandas as pd
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            logs = df["log"].astype(str).tolist() if "log" in df.columns else [None] * len(df)
            for t, lg in zip(df["token"].astype(str).tolist(), logs):
                out.setdefault(t, lg)
        else:
            for line in p.read_text().split():
                out.setdefault(line.strip(), None)
    return out


def scan_root(root: Path) -> Dict[str, Path]:
    """token -> metric_cache.pkl for every <root>/<log>/<type>/<token>/metric_cache.pkl."""
    idx: Dict[str, Path] = {}
    if not root.is_dir():
        return idx
    for lg in os.scandir(root):
        if not lg.is_dir() or lg.name == "metadata":
            continue
        for st in os.scandir(lg.path):
            if not st.is_dir():
                continue
            for tk in os.scandir(st.path):
                p = Path(tk.path) / "metric_cache.pkl"
                if p.exists():
                    idx.setdefault(tk.name, p)
    return idx


def locate(tokens: Dict[str, Optional[str]], roots: Sequence[Path]) -> Dict[str, Optional[Path]]:
    """token -> metric_cache.pkl (first root that has it) or None.  Uses the log column when given
    (direct path, no directory scan); otherwise scans each root once."""
    res: Dict[str, Optional[Path]] = {t: None for t in tokens}
    need_scan = [t for t, lg in tokens.items() if lg is None]
    for t, lg in tokens.items():
        if lg is None:
            continue
        for r in roots:
            d = Path(r) / lg
            if not d.is_dir():
                continue
            for st in os.scandir(d):
                p = Path(st.path) / t / "metric_cache.pkl"
                if p.exists():
                    res[t] = p
                    break
            if res[t] is not None:
                break
    if need_scan:
        for r in roots:
            idx = scan_root(Path(r))
            for t in need_scan:
                if res[t] is None and t in idx:
                    res[t] = idx[t]
    return res


# ----------------------------------------------------------------------------------------- worker
def build_one(args) -> Dict:
    """(token, mc_path | None, out_path, force, min_age) -> stats dict (never raises)."""
    tok, mc_path, out_path, force, min_age = args
    rec = dict(token=tok, status="", log="", mc_path=str(mc_path) if mc_path else "", t_load=np.nan,
               t_build=np.nan, bytes=0, n_polys=-1, n_invalid=-1, n_segments=-1, frac_inside=np.nan)
    try:
        out_path = Path(out_path)
        if out_path.exists() and not force:
            rec["status"] = "exists"
            rec["bytes"] = out_path.stat().st_size
            return rec
        if mc_path is None or not Path(mc_path).exists():
            rec["status"] = "no_mc"
            return rec
        mc_path = Path(mc_path)
        rec["log"] = mc_path.parent.parent.parent.name
        if min_age > 0 and time.time() - mc_path.stat().st_mtime < min_age:
            rec["status"] = "mc_fresh"
            return rec
        t0 = time.time()
        with lzma.open(mc_path, "rb") as f:
            mc = pickle.load(f)
        t1 = time.time()
        field, info = SDF.build_sdf_from_metric_cache(mc)
        t2 = time.time()
        rec["bytes"] = SDF.save_sdf(out_path, field, token=tok, ego_xyh=info["ego_xyh"], info=info)
        rec.update(status="built", t_load=t1 - t0, t_build=t2 - t1, n_polys=info["n_polys"],
                   n_invalid=info["n_invalid"], n_segments=info["n_segments"], frac_inside=info["frac_inside"])
    except Exception as e:  # recorded, the run continues
        rec["status"] = f"error: {type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    return rec


# ----------------------------------------------------------------------------------------- main
def summarise(recs: List[Dict], wall: float) -> Dict:
    st: Dict[str, int] = {}
    for r in recs:
        k = r["status"] if not r["status"].startswith("error") else "error"
        st[k] = st.get(k, 0) + 1
    b = [r for r in recs if r["status"] == "built"]

    def q(key):
        v = np.array([r[key] for r in b], np.float64)
        if not len(v):
            return {}
        return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.quantile(v, 0.95)),
                    max=float(v.max()))

    return dict(n=len(recs), status=st, wall_s=wall, t_load_s=q("t_load"), t_build_s=q("t_build"),
                bytes=q("bytes"), n_invalid_polys_total=int(sum(r["n_invalid"] for r in b)),
                sdf_version=SDF.SDF_VERSION)


def run_pass(a, toks: Dict[str, Optional[str]], roots: Sequence[Path]) -> Dict:
    """One pass over the tokens: build every missing field whose metric cache is available."""
    names = list(toks)
    out_dir = Path(a.out_root) / a.subset
    (out_dir / "_build").mkdir(parents=True, exist_ok=True)
    todo = [t for t in names if a.force or not SDF.sdf_path(t, a.subset, a.out_root).exists()]
    n_exist = len(names) - len(todo)
    paths = locate({t: toks[t] for t in todo}, roots)
    jobs = [(t, paths[t], SDF.sdf_path(t, a.subset, a.out_root), a.force, a.min_age) for t in todo]
    print(f"[build_sdf] {time.strftime('%H:%M:%S')} subset={a.subset} tokens={len(names)} existing={n_exist} "
          f"todo={len(todo)} with_mc={sum(p is not None for p in paths.values())} "
          f"roots={[str(r) for r in roots]} workers={a.workers} pid={os.getpid()}", flush=True)

    stats_f = out_dir / "_build" / "stats.jsonl"
    recs: List[Dict] = []
    t0 = time.time()
    last = t0
    n_built = 0
    with open(stats_f, "a") as fo:
        pool = None
        if a.workers == 1:
            it = map(build_one, jobs)
        else:
            pool = Pool(a.workers)
            it = pool.imap_unordered(build_one, jobs, chunksize=max(1, a.chunksize))
        try:
            for k, rec in enumerate(it, 1):
                recs.append(rec)
                if rec["status"] not in ("exists", "no_mc", "mc_fresh"):     # transient states are not logged
                    fo.write(json.dumps({kk: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                         for kk, v in rec.items()}) + "\n")
                    fo.flush()
                n_built += rec["status"] == "built"
                if rec["status"].startswith("error"):
                    print(f"[build_sdf] {rec['token']}: {rec['status']}", flush=True)
                now = time.time()
                if now - last > 60 or k == len(jobs):
                    last = now
                    rate = n_built / max(now - t0, 1e-9)
                    left = sum(1 for j in jobs[k:] if j[1] is not None)
                    print(f"[build_sdf] {k}/{len(jobs)} built {n_built}  {now - t0:.0f}s  {rate:.2f} tok/s  "
                          f"ETA(with mc) {left / max(rate, 1e-9) / 60:.1f} min", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    summ = summarise(recs, time.time() - t0)
    summ.update(subset=a.subset, n_tokens=len(names), n_existing_before=n_exist, workers=a.workers,
                mc_roots=[str(r) for r in roots])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if recs:
        (out_dir / "_build" / f"summary_{stamp}_{os.getpid()}.json").write_text(json.dumps(summ, indent=1))
    print("[build_sdf] summary " + json.dumps(summ), flush=True)
    return summ


def main(argv: Optional[Iterable[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, help="output sub-directory, e.g. navtrain | navtest")
    ap.add_argument("--tokens", action="append", default=[], help="token file(s); see module docstring")
    ap.add_argument("--all", action="store_true", help="every token found under the metric-cache roots")
    ap.add_argument("--mc-root", action="append", default=None, help="metric-cache root(s), searched in order")
    ap.add_argument("--out-root", default=str(SDF.SDF_ROOT))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tokens (0 = all)")
    ap.add_argument("--force", action="store_true", help="rebuild existing files")
    ap.add_argument("--min-age", type=float, default=60.0, help="skip metric caches modified < this many s ago")
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--follow-interval", type=float, default=0.0,
                    help="> 0: repeat passes every this many s while tokens still lack a metric cache "
                         "(trails a running metric-cache job)")
    ap.add_argument("--follow-max-h", type=float, default=12.0, help="stop following after this many hours")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not (1 <= a.workers <= MAX_WORKERS):
        ap.error(f"--workers must be in 1..{MAX_WORKERS} (shared machine)")
    roots = [Path(r) for r in (a.mc_root or DEFAULT_MC_ROOTS.get(a.subset, []))]
    if not roots:
        ap.error(f"no default metric-cache root for subset {a.subset!r}; pass --mc-root")
    if not (a.all or a.tokens):
        ap.error("pass --tokens FILE or --all")

    t_start = time.time()
    while True:
        if a.all:
            toks: Dict[str, Optional[str]] = {}
            for r in roots:
                for t in scan_root(r):
                    toks.setdefault(t, None)
        else:
            toks = read_tokens(a.tokens)
        if a.limit:
            toks = dict(list(toks.items())[: a.limit])
        summ = run_pass(a, toks, roots)
        pending = summ["status"].get("no_mc", 0) + summ["status"].get("mc_fresh", 0)
        if a.follow_interval <= 0 or pending == 0 or a.all:
            break
        if time.time() - t_start > a.follow_max_h * 3600:
            print(f"[build_sdf] follow: giving up after {a.follow_max_h} h with {pending} tokens pending", flush=True)
            break
        a.force = False                        # never rebuild in later passes
        print(f"[build_sdf] follow: {pending} tokens pending, next pass in {a.follow_interval:.0f}s", flush=True)
        time.sleep(a.follow_interval)
    return summ


if __name__ == "__main__":
    main()
