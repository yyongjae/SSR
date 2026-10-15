"""Tests for tools/refiner/ttc_select.py (AMENDMENT 4 (2) m_ttc selection rule).  CPU, seconds.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python -m pytest -q tools/refiner/tests/test_ttc_select.py

- flags: TTC flag = col_g < 0.15 OR ttc_g < m_ttc (strict; +inf never flags);
- t1_t2: T1 over all bank drafts with ttc < 1 (k = 0 included), T2 over k = 0 with ttc == 1, hand counts;
- feasible / choose: largest feasible m_ttc; none feasible -> fewest failed, then smallest shortfall, then larger margin;
- main refuses > 2 workers and output paths under runs/.
All data synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import ttc_select as TS  # noqa: E402


def _df():
    # 3 tokens / logs; k = 0 human rows + perturbed rows
    return pd.DataFrame(dict(
        token=["a", "a", "a", "b", "b", "c", "c"], log=["la", "la", "la", "lb", "lb", "lc", "lc"],
        k=[0, 1, 2, 0, 3, 0, 5],
        col_g=[1.0, 0.10, 2.0, 0.149, 3.0, np.inf, 1.0],
        ttc_g=[0.07, 1.0, 0.04, 1.0, 0.12, np.inf, -0.2],
        ttc=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]))


def test_flags():
    d = _df()
    assert TS.flags(d, 0.0, "col_only").tolist() == [False, True, False, True, False, False, False]
    assert TS.flags(d, 0.0, "proj_only").tolist() == [False, False, False, False, False, False, True]
    assert TS.flags(d, 0.05).tolist() == [False, True, True, True, False, False, True]
    assert TS.flags(d, 0.12).tolist() == [True, True, True, True, False, False, True]    # 0.12 < 0.12 is False
    assert TS.flags(d, 0.15).tolist() == [True, True, True, True, True, False, True]


def test_t1_t2_counts():
    d = _df()
    s = TS.t1_t2(d, TS.flags(d, 0.05))
    # ttc fails: rows 1, 2, 4, 5 (row 5 is a human k = 0 failure, counted in T1)
    assert (s["T1"]["k"], s["T1"]["n"]) == (2, 4)
    assert (s["T1_perturbed_only"]["k"], s["T1_perturbed_only"]["n"]) == (2, 3)
    # human ttc == 1: rows 0 and 3; row 3 flagged by the collision flag
    assert (s["T2"]["k"], s["T2"]["n"]) == (1, 2)
    assert (s["FA_bank"]["k"], s["FA_bank"]["n"]) == (2, 3)
    lo, hi = s["T1"]["boot_log"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_choose_feasible_largest():
    t = pd.DataFrame(dict(m_ttc=[0.0, 0.05, 0.10, 0.15], T1=[0.72, 0.8, 0.85, 0.9], T2=[0.0, 0.01, 0.02, 0.021]))
    s = TS.choose(t)
    assert s["m_ttc"] == 0.10 and s["feasible_m_ttc"] == [0.0, 0.05, 0.10]


def test_choose_none_feasible():
    # all fail one criterion; smallest shortfall 0.05 (T1 0.65) at 0.0 and 0.05 -> larger margin 0.05
    t = pd.DataFrame(dict(m_ttc=[0.0, 0.05, 0.10, 0.15], T1=[0.65, 0.65, 0.69, 0.5], T2=[0.0, 0.0, 0.10, 0.2]))
    s = TS.choose(t)
    assert s["n_feasible"] == 0 and s["m_ttc"] == 0.05 and s["limitation"]
    # fewest failed wins over a smaller shortfall
    t = pd.DataFrame(dict(m_ttc=[0.0, 0.05], T1=[0.5, 0.69], T2=[0.0, 0.021]))
    assert TS.choose(t)["m_ttc"] == 0.0


def test_feasible_boundaries():
    assert TS.feasible(0.70, 0.02)["all"]
    f = TS.feasible(0.6999, 0.0201)
    assert f["n_failed"] == 2 and abs(f["shortfall"] - (0.0001 + 0.0001)) < 1e-12


def test_main_guards(tmp_path):
    with pytest.raises(SystemExit):
        TS.main(["run", "--workers", "3", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        TS.main(["run", "--workers", "1", "--out", "/home/external-user/ssd/yongjae_refiner/runs/stageT3_x"])
