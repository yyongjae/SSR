"""Tests for tools/refiner/make_splits.py (CPU, < 30 s): allocation, log-level split, E-preference uniformity, folds,
and the produced split files.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest \
          -q tools/refiner/tests/test_make_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT / "tools/refiner"))
import make_splits as M  # noqa: E402

SPLITS = M.OUT


def _pool(seed=0, n_logs=80):
    rng = np.random.default_rng(seed)
    rows = []
    cities = ["a", "b", "c", "d"]
    for i in range(n_logs):
        city, part = cities[i % 4], ("train" if (i // 4) % 3 else "val")
        n = int(rng.integers(1, 150))
        for k in range(n):
            rows.append((f"t{i:03d}_{k:04d}", f"log{i:03d}", k, city, part))
    return pd.DataFrame(rows, columns=M.SPEC_COLS)


def test_largest_remainder():
    w = np.array([5, 1, 7, 13, 2, 0, 9])
    for tot in (0, 1, 10, 17, 37):
        q = M.largest_remainder(w, tot)
        assert q.sum() == tot
        assert np.all(np.abs(q - tot * w / w.sum()) < 1.0)
        assert q[5] == 0


def test_make_splits_synthetic():
    pool = _pool()
    prefer = set(pool.token.sample(400, random_state=1))
    tr, dv, s = M.make_splits(pool, 1500, 500, prefer, seed=3, n_folds=5)
    assert len(tr) == 1500 and len(dv) == 500
    assert not set(tr.log) & set(dv.log)
    assert not (set(tr.token) & set(dv.token))
    assert list(tr.columns[:5]) == M.SPEC_COLS
    # each train log in exactly one fold, folds 0..4 used, dev fold -1
    assert tr.groupby("log").fold.nunique().max() == 1
    assert set(tr.fold) == set(range(5)) and set(dv.fold) == {-1}
    # constant within-side rate: per-log counts are floor/ceil of rate * n_l
    for side, df in (("train", tr), ("dev", dv)):
        side_logs = set(df.log) | (set(s["dev_logs"]) if side == "dev" else set())
        pl = pool[pool.log.isin(set(s["dev_logs"])) == (side == "dev")].groupby("log").size()
        r = len(df) / pl.sum()
        got = df.groupby("log").size().reindex(pl.index, fill_value=0)
        assert np.all(np.abs(got - r * pl) < 1.0), side
        assert set(df.log) <= side_logs
    # preferred tokens taken first inside each log
    smp = pd.concat([tr, dv])
    for lg, g in pool[pool.log.isin(set(smp.log))].groupby("log"):
        q = (smp.log == lg).sum()
        m = g.token.isin(prefer).sum()
        took = smp[smp.log == lg].token.isin(prefer).sum()
        assert took == min(q, m)
    # e_cached flag and order column
    assert (tr.e_cached == tr.token.isin(prefer)).all()
    assert np.allclose(np.sort(tr.order.to_numpy()), np.arange(len(tr)) / len(tr))
    # determinism
    tr2, dv2, _ = M.make_splits(pool, 1500, 500, prefer, seed=3, n_folds=5)
    pd.testing.assert_frame_equal(tr, tr2)
    pd.testing.assert_frame_equal(dv, dv2)
    tr3, _, _ = M.make_splits(pool, 1500, 500, prefer, seed=4, n_folds=5)
    assert set(tr3.token) != set(tr.token)


def test_dev_logs_stratified():
    pool = _pool(seed=5, n_logs=200)
    _, _, s = M.make_splits(pool, 3000, 1000, set(), seed=0)
    dev = set(s["dev_logs"])
    per = pool.assign(dev=pool.log.isin(dev)).groupby(M.STRATA).dev.mean()
    # every (city, part) stratum gets ~25 % of its tokens on the dev side (whole logs => within ~1 log)
    big = pool.groupby(M.STRATA).size()
    assert np.all(np.abs(per - 0.25) < 150.0 / big + 0.02)


def test_preference_keeps_within_log_uniformity():
    """E-preferred-first selection is a uniform q-subset when the preferred set is itself a uniform random subset."""
    rng = np.random.default_rng(0)
    toks = [f"x{i}" for i in range(10)]
    pool = pd.DataFrame(dict(token=toks, log="L", frame_idx=range(10), map_location="a", part="train"))
    cnt = pd.Series(0.0, index=toks)
    trials = 4000
    for _ in range(trials):
        prefer = set(rng.choice(toks, 2, replace=False))
        sel = M.thin_within_logs(pool, 4, prefer, rng)
        cnt[sel.token] += 1
    freq = cnt / trials
    assert np.all(np.abs(freq - 0.4) < 0.035), freq.to_dict()


@pytest.mark.skipif(not (SPLITS / "train.parquet").exists(), reason="split files not built")
def test_split_files():
    tr = pd.read_parquet(SPLITS / "train.parquet")
    dv = pd.read_parquet(SPLITS / "dev.parquet")
    la = pd.read_parquet(SPLITS / "log_assignment.parquet")
    pool = pd.read_parquet(M.POOL)
    for df in (tr, dv):
        assert set(M.SPEC_COLS) <= set(df.columns)
        assert not df.token.duplicated().any()
        m = df.merge(pool, on="token", suffixes=("", "_p"))
        assert len(m) == len(df)
        for c in ["log", "frame_idx", "map_location", "part"]:
            assert (m[c] == m[c + "_p"]).all(), c
    assert len(tr) == 24000 and len(dv) == 8000
    assert not set(tr.log) & set(dv.log)
    assert set(dv.log) <= set(la.log[la.side == "dev"]) and set(tr.log) <= set(la.log[la.side == "train"])
    assert set(tr.fold) == set(range(5)) and set(dv.fold) == {-1}
    assert tr.groupby("log").fold.nunique().max() == 1
    share_p = pool.map_location.value_counts(normalize=True)
    for df in (tr, dv):
        assert (df.map_location.value_counts(normalize=True) - share_p).abs().max() < 0.01
    # e_cached flags agree with E's files (sample)
    smp = pd.concat([tr, dv]).sample(300, random_state=0)
    on_disk = [(M.E_MC / lg / "unknown" / t / "metric_cache.pkl").exists() for t, lg in zip(smp.token, smp.log)]
    assert (np.array(on_disk) == smp.e_cached.to_numpy()).all()
    assert pd.concat([tr, dv]).e_cached.sum() >= 8900
