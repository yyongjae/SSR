"""Tests for tools/refiner/validate_sdf.py helpers (CPU, seconds): confusion / AUC bookkeeping, per-corner
statistics on a synthetic road, and the summary on synthetic rows.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest \
          -q tools/refiner/tests/test_validate_sdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))
import validate_sdf as V  # noqa: E402
from navsim.agents.para_ssr.refiner import sdf as S  # noqa: E402


def test_conf_and_auc():
    y = np.array([1, 1, 0, 0, 0], bool)
    c = V._conf(y, np.array([1, 0, 1, 0, 0], bool))
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 2)
    assert c["recall"] == 0.5 and c["precision"] == 0.5 and abs(c["fpr"] - 1 / 3) < 1e-12 and c["agree"] == 0.6
    assert V._auc(y, np.array([0.9, 0.8, 0.1, 0.2, 0.3])) == 1.0
    assert V._auc(np.zeros(3, bool), np.arange(3)) is None


def test_corner_stats_on_straight_road():
    import shapely
    road = shapely.box(-50, -3.5, 120, 3.5)                        # 7 m wide straight road along x
    f16 = torch.from_numpy(S.rasterize_sdf(road).astype(np.float16))
    x = np.linspace(0, 30, 41)
    for y_off, fail in ((0.0, False), (2.6, True)):                 # half width 1.1485: 2.6 + 1.15 > 3.5
        poses = torch.from_numpy(np.stack([x, np.full(41, y_off), np.zeros(41)], -1))
        corners = S.ego_corners(poses).numpy()
        rec, err = V._corner_stats(corners, f16, road, "tracked")
        assert rec["tracked_exact_fail"] == fail and (rec["tracked_sdf_min"] < 0) == fail
        assert abs(rec["tracked_exact_min"] - (3.5 - y_off - S.EGO_HALF_WID)) < 1e-9
        assert abs(rec["tracked_sdf_min"] - rec["tracked_exact_min"]) < 5e-3
        assert rec["tracked_n_oog"] == 0 and rec["tracked_torch_np_maxdiff"] < 1e-5
        assert (rec["tracked_sdf_first_t"] == 0) == fail and err.shape[1] == 2


def test_summarize_synthetic_rows():
    rows = pd.DataFrame(dict(
        token=list("abcd"), log="L", traj=["orig", "orig", "orig", "human"], random=[True, True, False, False],
        dac_official=[1.0, 0.0, 0.0, 1.0], dac_table=[1.0, 0.0, 0.0, np.nan], ok=True,
        tracked_sdf_min=[0.5, -0.2, 0.01, 0.3], tracked_exact_min=[0.5, -0.2, -0.01, 0.3],
        tracked_exact_fail=[False, True, True, False], tracked_n_oog=0, tracked_torch_np_maxdiff=0.0,
        tracked_x_max=30.0, tracked_absy_max=5.0,
        raw_sdf_min=[0.6, 0.1, 0.2, 0.4], raw_exact_min=[0.6, 0.1, 0.2, 0.4], raw_exact_fail=False,
        raw_n_oog=[0, 0, 2, 0], raw_torch_np_maxdiff=0.0, raw_x_max=31.0, raw_absy_max=5.0))
    errs = np.array([[0.1, 0.11, 0], [-0.3, -0.29, 0], [0.02, -0.01, 1]])
    s = V.summarize(rows, errs)
    assert s["sanity"]["official_vs_table_mismatch"] == 0
    assert s["sanity"]["official_vs_exact_union_on_tracked_mismatch"] == 0
    assert s["sanity"]["raw_n_oog_total"] == 2
    t = s["orig_all"]["tracked"]
    assert t["sdf"]["tp"] == 1 and t["sdf"]["fn"] == 1 and t["exact_union"]["agree"] == 1.0
    assert t["disagreements"][0]["token"] == "c"
    assert s["orig_random"]["n"] == 2 and s["human_all"]["n"] == 1
    assert s["corner_accuracy_abs_d_lt_2m"]["raw"]["sign_mismatch_within_5cm"] == 1
