"""Tests for tools/refiner/build_metric_cache.py (CPU, ~1 min): chunking, seeding from E, crash repair, the
field / score comparison, one real navtrain re-cache identical to E's cache, and the verification reports.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest \
          -q tools/refiner/tests/test_build_metric_cache.py
"""
from __future__ import annotations

import json
import lzma
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path("/home/external-user/yongjae/SSR")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(ROOT / "data/dataset/maps"))  # read by navsim at import time
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))
import build_metric_cache as B  # noqa: E402

NAVTEST_TABLE = B.NAVTEST_TABLE


def _write_xz(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(p, "wb", preset=0) as f:
        pickle.dump(obj, f)


def test_saved_config_unchanged():
    assert B.sha256(B.SAVED_CFG) == B.SAVED_CFG_SHA256


def test_make_chunks():
    df = pd.DataFrame(dict(token=[f"t{i}" for i in range(23)], log=["a"] * 17 + ["b"] * 6))
    ch = B.make_chunks(df, Path("/o"), Path("/l"), chunk=8, seed=1)
    assert sorted(t for c in ch for t in c[1]) == sorted(df.token)
    assert all(len(c[1]) <= 8 for c in ch)
    assert all(set(c[1]) <= set(df.token[df.log == c[0]]) for c in ch)
    assert [c[1] for c in ch] == [c[1] for c in B.make_chunks(df, Path("/o"), Path("/l"), chunk=8, seed=1)]
    assert len([c for c in ch if c[0] == "a"]) == 3


def test_seed_repair_state(tmp_path):
    e_root, out = tmp_path / "E", tmp_path / "out"
    df = pd.DataFrame(dict(token=["k1", "k2", "k3", "k4"], log=["L1", "L1", "L2", "L2"]))
    _write_xz(B.mc_path(e_root, "L1", "k1"), {"a": 1})
    _write_xz(B.mc_path(e_root, "L2", "k3"), {"a": 3})
    st = B.State(out)
    assert B.seed_from_e(df, out, e_root, st) == 2
    assert B.mc_path(out, "L1", "k1").read_bytes() == B.mc_path(e_root, "L1", "k1").read_bytes()
    assert st.done() == {"k1", "k3"}
    assert B.seed_from_e(df, out, e_root, st) == 0          # idempotent
    # k2: truncated file from a killed worker (not recorded done) -> deleted;  k4: complete but unrecorded -> recorded
    good = B.mc_path(out, "L2", "k4")
    _write_xz(good, {"a": 4})
    bad = B.mc_path(out, "L1", "k2")
    _write_xz(bad, {"a": list(range(20000))})
    bad.write_bytes(bad.read_bytes()[:200])
    tmpf = B.mc_path(out, "L1", "k1").with_name("metric_cache.pkl.tmp123")
    tmpf.write_bytes(b"x")
    assert B.xz_ok(good) and not B.xz_ok(bad)
    assert B.repair(df, out, st) == (2, 1)
    assert not bad.exists() and good.exists() and not tmpf.exists()
    assert st.done() == {"k1", "k3", "k4"}


def _navtest_token(i=11):
    t = pd.read_parquet(NAVTEST_TABLE, columns=["token", "log"])
    return t.iloc[i]


@pytest.mark.skipif(not NAVTEST_TABLE.exists(), reason="navtest table missing")
def test_probe_and_compare_fields():
    r = _navtest_token()
    p = B.mc_path(B.NAVTEST_MC, r.log, r.token)
    mo, mn = B.load_mc(p), B.load_mc(p)
    tr = B.probe_trajectories(mo)
    assert set(tr) == {"cv", "cv_lat", "brake", "pdmc"}
    for v in tr.values():
        assert v.shape == (8, 3) and v.dtype == np.float32 and np.isfinite(v).all()
    assert np.all(tr["cv"][:, 1:] == 0) and np.all(np.diff(tr["brake"][:, 0]) >= -1e-6)
    f = B.compare_fields(mo, mn)
    assert all(v == 0.0 for v in f.values()), f
    mn.centerline._states_se2_array = mn.centerline._states_se2_array + 1e-3
    mn.route_lane_ids = list(mn.route_lane_ids)[:-1]
    f2 = B.compare_fields(mo, mn)
    assert abs(f2["centerline"] - 1e-3) < 1e-9 and f2["route"] == float("inf")


@pytest.mark.skipif(not (B.SPLITS / "dev.parquet").exists(), reason="split files not built")
def test_recache_navtrain_token_identical_to_e(tmp_path):
    """One E-cached dev token re-cached with this module's copied e2_cache logic == E's cache (scores + fields)."""
    sys.path.insert(0, str(B.CF_DIR))
    import cf_common as CF
    dv = pd.read_parquet(B.SPLITS / "dev.parquet")
    r = dv[dv.e_cached].iloc[3]
    B._init(str(tmp_path), str(B.TRAIN_LOGS))
    log, toks, ok, bad, dt, msg = B.run_chunk((r.log, [r.token], str(tmp_path), str(B.TRAIN_LOGS)))
    assert ok == 1 and bad == 0, msg
    CF.init_worker()
    row = B.compare_pair(B.load_mc(B.mc_path(B.E_MC, r.log, r.token)), B.load_mc(B.mc_path(tmp_path, r.log, r.token)))
    assert row["max_score_diff"] == 0.0 and row["max_field_diff"] == 0.0, row


@pytest.mark.parametrize("name,min_n", [("navtest", 20), ("e", 5)])
def test_verification_reports(name, min_n):
    p = B.REPORT / f"metric_cache_verify_{name}.json"
    if not p.exists():
        pytest.skip("verification not run")
    s = json.load(open(p))
    assert s["n_tokens"] >= min_n
    assert s["all_scores_identical"] and s["all_fields_identical"]
    assert s["max_score_diff"] == 0.0 and s["max_field_diff"] == 0.0
    assert s["n_nc_fail"] > 0 and s["n_dac_fail"] > 0          # failure paths exercised
    assert s["saved_cfg_sha256"] == B.SAVED_CFG_SHA256
    if name == "e":
        assert all(r["seeded_copy_bytes_equal_e"] for r in s["rows"])
