#!/usr/bin/env python
"""Stage-T token splits (train / dev) from the navtrain pool (IMPL_SPEC §4).

Pool: report/cause_and_correction_tests/E_train_split_feasibility/tokens/navtrain_token_log.parquet
(103,288 navtrain tokens in 1,192 logs, produced by the repo's own filter_scenes with the navtrain scene filter;
columns token, log, frame_idx, map_location, part in {train, val} = the student's train_logs / val_logs).

Sampling design (two-stage, log = cluster; seed-deterministic):
  1. Log assignment (stage 1).  EVERY pool log goes to exactly one side, dev or train.  Stratified by
     (map_location, part): inside each stratum the logs are randomly permuted and the permuted prefix whose
     cumulative token count is closest to dev_frac * (stratum tokens) becomes dev, dev_frac = n_dev / (n_dev + n_train)
     (a log goes to dev iff its cumulative-count midpoint is below the target).  => dev = held-out logs, and each city
     (and each student part) has the same dev share of tokens up to one log.  `part` is an extra stratification
     variable (not required by the spec): it keeps the student's train_logs / val_logs mix equal on both sides, which
     matters once student drafts / features enter (stage F/E).
  2. Token thinning (stage 2).  Inside each side a constant rate r_side = n_side / (tokens of the side's logs) is
     applied per log; the per-log quota q_l = r_side * n_l is rounded by largest remainder so that the side totals are
     exactly n_train / n_dev.  A uniform within-log rate keeps the token sample self-weighting (every token of a side
     has the same inclusion probability, hence city shares = population shares) while using all 1,192 logs as
     clusters (tokens of one log are 0.5 s apart and strongly correlated, so more clusters = more information and
     tighter log-cluster bootstrap CIs than whole-log sampling of ~370 logs).
  3. E-cache preference (inside stage 2).  Tokens that already have E's metric cache (the 9,000 tokens of E's
     tokens/sample.parquet, found on disk under E/metric_cache/<log>/unknown/<token>/metric_cache.pkl) are taken
     first inside each log: if the log has m_l cached tokens and quota q_l, all m_l are taken when m_l <= q_l and the
     remaining q_l - m_l are drawn uniformly from the uncached tokens; if m_l > q_l a uniform q_l-subset of the cached
     tokens is taken.  E's tokens are a simple random sample of tokens within `part` (seed 0, independent of scene
     content), hence exchangeable inside a log, so the selected set is still a UNIFORM random q_l-subset of the log
     (P(S = A) = C(q,m) / (C(n,m) C(n-m,q-m)) is the same for every q-subset A).  No bias is introduced; it only saves
     metric-cache build time (E measured ~2.3 tok/s with 4 workers).
  4. Cross-fit folds.  The train side's logs get fold ids 0..n_folds-1 (log-level), stratified by
     (map_location, part): logs are visited in random order within each stratum and each goes to the fold with the
     fewest sampled tokens so far in that stratum (ties -> lowest fold id).  dev rows have fold = -1.

Output (one row per token; rows sorted by (log, frame_idx, token) -- deterministic but NOT temporal: the log field
frame_idx restarts at 0 for every OpenScene scene inside a log; the temporal position in the log list is
extract_human's `log_pos`):
  <out>/train.parquet, <out>/dev.parquet: token, log, frame_idx, map_location, part (spec columns) +
     split, fold (int8; -1 for dev), e_cached (bool: E metric cache exists), order (float, random rank / n within the
     split: any prefix sorted by `order` is a uniform subsample of the split)
  <out>/summary.json (+ a copy in report/refiner_T/splits_summary.json): counts per side x city x part, logs, rates,
     E-cached counts, fold sizes, teacher-cache coverage, dev log list.
  <out>/log_assignment.parquet: log, map_location, part, side, n_pool, n_sampled, fold (every pool log, incl. logs whose
     quota rounded to 0 -- they stay on their side if the sample is enlarged later with the same seed/assignment).
Teacher-cache check (default on): every sampled token must have
  /home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100/samples/<token[:2]>/<token>.npz and the
  manifest checkpoint sha head must be cddf943ffec8d6a8.

  python tools/refiner/make_splits.py                      # 24,000 train + 8,000 dev, seed 0, 5 folds
  python tools/refiner/make_splits.py --n-train 12000 --n-dev 4000 --out /tmp/x --no-teacher-check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/external-user/yongjae/SSR")
E_DIR = ROOT / "report/cause_and_correction_tests/E_train_split_feasibility"
POOL = E_DIR / "tokens/navtrain_token_log.parquet"
E_MC = E_DIR / "metric_cache"
DATA = Path("/home/external-user/ssd/yongjae_refiner")
OUT = DATA / "splits"
REPORT = ROOT / "report/refiner_T"
TEACHER = Path("/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100")
TEACHER_SHA_HEAD = "cddf943ffec8d6a8"
SPEC_COLS = ["token", "log", "frame_idx", "map_location", "part"]
STRATA = ["map_location", "part"]


# ----------------------------------------------------------------------------------------------- helpers
def e_cached_tokens(e_mc: Path = E_MC, tokens: Optional[pd.DataFrame] = None) -> Set[str]:
    """Tokens that have E's metric cache on disk (<e_mc>/<log>/unknown/<token>/metric_cache.pkl).
    If ``tokens`` (token, log) is given only those rows are checked (cheap); otherwise the directory tree is scanned."""
    if tokens is not None:
        return {t for t, lg in zip(tokens.token, tokens.log) if (e_mc / lg / "unknown" / t / "metric_cache.pkl").exists()}
    return {p.parent.name for p in e_mc.glob("*/unknown/*/metric_cache.pkl")}


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Integer allocation q_i ~ total * w_i / sum(w) with sum(q) == total exactly (Hamilton / largest remainder).
    Ties in the remainder are broken by index order (deterministic)."""
    w = np.asarray(weights, np.float64)
    if total < 0 or w.sum() <= 0:
        raise ValueError("need total >= 0 and positive weights")
    exact = total * w / w.sum()
    q = np.floor(exact).astype(np.int64)
    rest = total - int(q.sum())
    if rest > 0:
        order = np.lexsort((np.arange(len(w)), -(exact - q)))  # largest remainder first, then index
        q[order[:rest]] += 1
    return q


def assign_dev_logs(logs: pd.DataFrame, dev_frac: float, rng: np.random.Generator) -> Set[str]:
    """Stage 1.  logs: one row per log with columns log, n (tokens), map_location, part.
    Returns the set of dev logs: per (map_location, part) stratum, the random-permutation prefix whose cumulative token
    count is closest to dev_frac * stratum tokens (log in dev iff cumsum - n/2 < target)."""
    dev: Set[str] = set()
    for _, g in logs.sort_values("log").groupby(STRATA, sort=True):
        g = g.iloc[rng.permutation(len(g))]
        target = dev_frac * g.n.sum()
        mid = g.n.cumsum().to_numpy() - g.n.to_numpy() / 2.0
        dev.update(g.log[mid < target].tolist())
    return dev


def thin_within_logs(pool: pd.DataFrame, n_total: int, prefer: Set[str], rng: np.random.Generator) -> pd.DataFrame:
    """Stage 2 (+3).  pool: tokens of one side.  Per-log quota by largest remainder at a constant rate
    n_total / len(pool); inside a log the preferred (E-cached) tokens are taken first (uniform subset if too many),
    the rest uniformly from the other tokens.  Returns the selected rows (pool columns kept)."""
    if n_total > len(pool):
        raise ValueError(f"requested {n_total} tokens but the side has only {len(pool)}")
    per_log = pool.groupby("log", sort=True).size()
    quota = pd.Series(largest_remainder(per_log.to_numpy(), n_total), index=per_log.index)
    keep = []
    for log, g in pool.sort_values(["log", "token"]).groupby("log", sort=True):
        q = int(quota[log])
        if q == 0:
            continue
        is_p = g.token.isin(prefer).to_numpy()
        pref, other = g.index[is_p].to_numpy(), g.index[~is_p].to_numpy()
        if len(pref) >= q:
            keep.append(rng.choice(pref, q, replace=False))
        else:
            keep.append(pref)
            keep.append(rng.choice(other, q - len(pref), replace=False))
    idx = np.concatenate(keep) if keep else np.array([], np.int64)
    return pool.loc[idx]


def assign_folds(train: pd.DataFrame, n_folds: int, rng: np.random.Generator) -> Dict[str, int]:
    """Stage 4.  Log -> fold id (0..n_folds-1), stratified by (map_location, part), greedy token balancing:
    logs visited in random order inside a stratum, each to the fold with the fewest sampled tokens so far."""
    per_log = train.groupby("log", sort=True).agg(n=("token", "size"), map_location=("map_location", "first"),
                                                  part=("part", "first")).reset_index()
    fold: Dict[str, int] = {}
    for _, g in per_log.groupby(STRATA, sort=True):
        g = g.iloc[rng.permutation(len(g))]
        load = np.zeros(n_folds, np.int64)
        for lg, n in zip(g.log, g.n):
            k = int(np.argmin(load))
            fold[lg] = k
            load[k] += n
    return fold


def check_teacher(tokens: Iterable[str], teacher: Path = TEACHER) -> Tuple[int, list, str]:
    """(#tokens with a teacher npz, first missing tokens, manifest checkpoint_sha256_head)."""
    man = json.load(open(teacher / "manifest.json"))
    missing = [t for t in tokens if not (teacher / "samples" / t[:2] / f"{t}.npz").exists()]
    return len(missing), missing[:20], man.get("checkpoint_sha256_head", "")


# ----------------------------------------------------------------------------------------------- main API
def make_splits(pool: pd.DataFrame, n_train: int, n_dev: int, e_tokens: Set[str], seed: int = 0,
                n_folds: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Deterministic (seed) stage-T split.  pool: SPEC_COLS rows (one per token).  Returns (train, dev, summary)."""
    if pool.token.duplicated().any():
        raise ValueError("duplicate tokens in pool")
    if pool.groupby("log")[STRATA].nunique().max().max() > 1:
        raise ValueError("a log spans several cities / parts")
    rng = np.random.default_rng(seed)
    pool = pool[SPEC_COLS].sort_values(["log", "token"]).reset_index(drop=True)
    logs = pool.groupby("log", sort=True).agg(n=("token", "size"), map_location=("map_location", "first"),
                                              part=("part", "first")).reset_index()
    dev_frac = n_dev / float(n_dev + n_train)
    dev_logs = assign_dev_logs(logs, dev_frac, rng)
    is_dev = pool.log.isin(dev_logs)
    out = {}
    for side, mask, n in (("train", ~is_dev, n_train), ("dev", is_dev, n_dev)):
        sel = thin_within_logs(pool[mask], n, e_tokens, rng).copy()
        sel["split"] = side
        sel["e_cached"] = sel.token.isin(e_tokens)
        sel["order"] = rng.permutation(len(sel)) / float(len(sel))
        out[side] = sel
    folds = assign_folds(out["train"], n_folds, rng)
    out["train"]["fold"] = out["train"].log.map(folds).astype(np.int8)
    out["dev"]["fold"] = np.int8(-1)
    cols = SPEC_COLS + ["split", "fold", "e_cached", "order"]
    for side in out:
        out[side] = out[side][cols].sort_values(["log", "frame_idx", "token"]).reset_index(drop=True)
    tr, dv = out["train"], out["dev"]
    assert not set(tr.log) & set(dv.log), "log overlap between train and dev"
    summary = summarize(pool, tr, dv, dev_logs, e_tokens, seed, n_folds, dev_frac)
    return tr, dv, summary


def summarize(pool, tr, dv, dev_logs, e_tokens, seed, n_folds, dev_frac) -> dict:
    def tab(df):
        g = df.groupby(STRATA).agg(tokens=("token", "size"), logs=("log", "nunique"))
        return {f"{a}|{b}": {k: int(v) for k, v in r.items()} for (a, b), r in g.iterrows()}

    def city_share(df):
        return {k: round(float(v), 4) for k, v in df.map_location.value_counts(normalize=True).sort_index().items()}

    in_dev = pool.log.isin(dev_logs)  # side pools = all logs assigned to the side (a small log may get quota 0)
    pool_tr, pool_dv = pool[~in_dev], pool[in_dev]
    return dict(
        seed=seed, n_folds=n_folds, dev_frac_target=dev_frac,
        pool=dict(tokens=len(pool), logs=int(pool.log.nunique()), city_share=city_share(pool),
                  e_cached_in_pool=int(pool.token.isin(e_tokens).sum())),
        train=dict(tokens=len(tr), logs=int(tr.log.nunique()), side_pool_tokens=len(pool_tr),
                   side_pool_logs=int(pool_tr.log.nunique()), rate=len(tr) / len(pool_tr), city_share=city_share(tr),
                   e_cached=int(tr.e_cached.sum()), strata=tab(tr),
                   folds={int(k): dict(tokens=int(len(g)), logs=int(g.log.nunique()),
                                       e_cached=int(g.e_cached.sum()), city_share=city_share(g))
                          for k, g in tr.groupby("fold")}),
        dev=dict(tokens=len(dv), logs=int(dv.log.nunique()), side_pool_tokens=len(pool_dv),
                 side_pool_logs=int(pool_dv.log.nunique()), rate=len(dv) / len(pool_dv), city_share=city_share(dv),
                 e_cached=int(dv.e_cached.sum()), strata=tab(dv)),
        e_cached_total_selected=int(tr.e_cached.sum() + dv.e_cached.sum()),
        e_cached_available=len(e_tokens),
        dev_logs=sorted(dev_logs),
    )


def log_table(pool: pd.DataFrame, tr: pd.DataFrame, dv: pd.DataFrame, dev_logs: Set[str]) -> pd.DataFrame:
    """One row per pool log: log, map_location, part, side (train/dev), n_pool, n_sampled, fold (-1: dev or no sampled
    token).  Records the held-out assignment also for logs whose quota rounded to 0."""
    t = pool.groupby("log", sort=True).agg(map_location=("map_location", "first"), part=("part", "first"),
                                           n_pool=("token", "size")).reset_index()
    t["side"] = np.where(t.log.isin(dev_logs), "dev", "train")
    smp = pd.concat([tr, dv])
    t["n_sampled"] = t.log.map(smp.groupby("log").size()).fillna(0).astype(np.int64)
    t["fold"] = t.log.map(tr.groupby("log").fold.first()).fillna(-1).astype(np.int8)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-dev", type=int, default=8000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--e-mc", default=str(E_MC))
    ap.add_argument("--no-teacher-check", action="store_true")
    ap.add_argument("--no-report-copy", action="store_true")
    a = ap.parse_args()
    pool = pd.read_parquet(a.pool)
    pool = pool[pool.part.isin(["train", "val"])]
    e_tok = e_cached_tokens(Path(a.e_mc), pool[["token", "log"]])
    tr, dv, summary = make_splits(pool, a.n_train, a.n_dev, e_tok, a.seed, a.folds)
    if not a.no_teacher_check:
        n_miss, first, sha = check_teacher(pd.concat([tr.token, dv.token]))
        summary["teacher"] = dict(missing=n_miss, first_missing=first, checkpoint_sha256_head=sha,
                                  sha_ok=sha == TEACHER_SHA_HEAD)
        if n_miss or sha != TEACHER_SHA_HEAD:
            raise SystemExit(f"teacher cache check failed: {summary['teacher']}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(out / "train.parquet", index=False)
    dv.to_parquet(out / "dev.parquet", index=False)
    log_table(pool, tr, dv, set(summary["dev_logs"])).to_parquet(out / "log_assignment.parquet", index=False)
    summary["args"] = vars(a)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    if not a.no_report_copy:
        REPORT.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(REPORT / "splits_summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("pool", "e_cached_total_selected")}, indent=1))
    for side in ("train", "dev"):
        s = summary[side]
        print(side, s["tokens"], "tokens", s["logs"], "logs", f"rate {s['rate']:.4f}", "E-cached", s["e_cached"],
              s["city_share"])
    print("folds", {k: (v["tokens"], v["logs"]) for k, v in summary["train"]["folds"].items()})
    if "teacher" in summary:
        print("teacher", summary["teacher"])


if __name__ == "__main__":
    main()
