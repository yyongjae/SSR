"""Tests for tools/refiner/build_future_objects.py (CPU, < 1 min): real navtest / navtrain tokens against their
official metric caches (corner error < 1e-3 m, presence identical), npz format, resumability, error reporting."""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_future_objects as BF  # noqa: E402
from navsim.agents.para_ssr.refiner import gt_future as G  # noqa: E402

NAVTEST_TABLE = ROOT / "report/perception_reliability/pdm_attr/table.parquet"
NAVTEST_MC = ROOT / "data/exp/metric_cache"
TEST_LOGS = Path("/home/external-user/navsim/download/test_navsim_logs/test")
E_DIR = ROOT / "report/cause_and_correction_tests/E_train_split_feasibility"
TRAIN_LOGS = ROOT / "data/dataset/navsim_logs/trainval"

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _check_val(s, n):
    assert s["n_error"] == 0
    assert s["val_tokens"] == n
    assert s["val_max_corner_err"] < 1e-3
    assert s["val_presence_mismatch"] == 0 and s["val_tokens_presence_identical"] == n
    assert s["val_mine_not_in_mc"] == 0 and s["val_excl_unexplained"] == 0
    assert s["val_red_light_tokens"] == 0 and s["val_pose0_err"] < 1e-9
    assert s["val_first_heading_err"] < 1e-5 and s["val_first_LW_err"] < 1e-5 and s["val_first_v_err"] < 1e-4
    assert s["val_first_class_mismatch"] == 0 and s["val_static_v_max"] == 0.0


@pytest.mark.skipif(not NAVTEST_TABLE.exists() or not NAVTEST_MC.exists(), reason="navtest data missing")
def test_navtest_tokens_match_metric_cache(tmp_path):
    t = pd.read_parquet(NAVTEST_TABLE)
    # 3 tokens incl. one NC failure (cause objects) -> same rule as the metric cache
    rows = pd.concat([t[t.re_no_at_fault_collisions < 1].head(1), t.sample(2, random_state=3)])[["token", "log"]]
    tp = tmp_path / "tok.parquet"
    rows.to_parquet(tp, index=False)
    out = tmp_path / "out"
    s = BF.main(["--tokens", str(tp), "--logs", str(TEST_LOGS), "--out", str(out), "--workers", "1",
                 "--mc-root", str(NAVTEST_MC)])
    _check_val(s, 3)
    # stored format
    for tok in rows.token:
        o = G.load_objects(out / f"{tok}.npz")
        A = o["kf"].shape[0]
        assert o["kf"].shape == (A, 11, 6) and o["kf"].dtype == np.float32
        assert o["first"].shape == (A, 6) and o["first"].dtype == np.float32
        assert o["meta"].shape == (A, 5) and o["meta"].dtype == np.int16
        assert o["track"].shape == (A,) and o["ego_kf"].shape == (11, 3)
        assert float(o["R"]) >= 80.0 and str(o["version"]) == G.VERSION
    idx = pd.read_parquet(out / "index.parquet")
    assert set(idx.token) == set(rows.token)
    # resumable: a second run without validation rebuilds nothing
    mt = {tok: (out / f"{tok}.npz").stat().st_mtime_ns for tok in rows.token}
    s2 = BF.main(["--tokens", str(tp), "--logs", str(TEST_LOGS), "--out", str(out), "--workers", "1"])
    assert s2["n_built"] == 0
    assert all((out / f"{tok}.npz").stat().st_mtime_ns == mt[tok] for tok in rows.token)


@pytest.mark.skipif(not (E_DIR / "tokens/sample.parquet").exists(), reason="E navtrain data missing")
def test_navtrain_E_token_and_frame_idx(tmp_path):
    t = pd.read_parquet(E_DIR / "tokens/sample.parquet").sample(2, random_state=1)
    tp = tmp_path / "tok.parquet"
    t.to_parquet(tp, index=False)
    s = BF.main(["--tokens", str(tp), "--logs", str(TRAIN_LOGS), "--out", str(tmp_path / "out"), "--workers", "1",
                 "--mc-root", str(E_DIR / "metric_cache"), "--report", str(tmp_path / "rep.json")])
    _check_val(s, 2)
    df = pd.read_parquet(tmp_path / "rep.parquet")
    assert (df.frame_idx_mismatch == 0).all(), "table frame_idx must equal the frame dict's own frame_idx"
    assert "human" in "".join(df.columns) and df.unk_human_npts.notna().all()


def test_synthetic_log_and_missing_token(tmp_path):
    from test_gt_future import make_log
    frames, _ = make_log(n_frames=20)
    logs = tmp_path / "logs"
    logs.mkdir()
    with open(logs / "synthlog.pkl", "wb") as f:
        pickle.dump(frames, f)
    tp = tmp_path / "tok.parquet"
    pd.DataFrame(dict(token=["tok000", "tok002", "missing"], log="synthlog", frame_idx=[0, 2, -1])).to_parquet(tp)
    s = BF.main(["--tokens", str(tp), "--logs", str(logs), "--out", str(tmp_path / "out"), "--workers", "1"])
    assert s["n_built"] == 2 and s["n_error"] == 1
    idx = pd.read_parquet(tmp_path / "out/index.parquet").set_index("token")
    assert idx.loc["missing", "status"] == "error" and "token not in log" in idx.loc["missing", "error"]
    assert idx.loc["tok002", "frame_idx_mismatch"] == 0 and idx.loc["tok002", "log_pos"] == 2
    o = G.load_objects(tmp_path / "out/tok000.npz")
    ref = G.build_from_frames(frames[:17])
    assert np.array_equal(o["kf"], ref["kf"]) and np.array_equal(o["meta"], ref["meta"])
    # the unknown-rate reference trajectories work on synthetic data (cv + su14 + human)
    r = BF.unknown_stats(o, frames[:17])
    assert r["unk_human_spec_any"] == 0 and r["unk_cv_npts"] == 4 * 41 * 4
