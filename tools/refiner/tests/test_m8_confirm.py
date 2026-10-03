"""Tests for tools/refiner/m8_confirm.py (disjoint-pool confirmation of the M8 margin selection).  CPU, seconds.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_m8_confirm.py

- pool: capped per-log sampling (cap = smallest c reaching n), determinism, disjointness proof;
- setting ids of S* / D; paired / independent log-bootstrap differences on hand-built data;
- verdict rule (criteria by point estimate + new-failure guard) and the both-pools rule order;
- the real pool files (if present) are disjoint by token and by log from the sweep pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))
import m8_confirm as C  # noqa: E402
import m8_recheck as R  # noqa: E402


def test_sids_and_defaults():
    from navsim.agents.para_ssr.refiner import surrogate as SU
    assert C.sid_of(C.SEL) == 14 and C.sid_of(C.DEF) == 23
    c = SU.SurrogateConfig()
    assert (c.m_col, c.m_dac, c.use_human_mask) == (C.DEF["m_col"], C.DEF["m_dac"], True)


def test_cap_and_pick():
    n = pd.Series([2, 5, 30, 1])
    assert C.cap_for(n, 10) == 4 and C.cap_for(n, 9) == 3 and C.cap_for(n, 8) == 3 and C.cap_for(n, 7) == 2
    assert C.cap_for(n, 1000) == 30
    s = pd.DataFrame({"token": [f"t{i:03d}" for i in range(38)],
                      "log": ["A"] * 2 + ["B"] * 5 + ["C"] * 30 + ["D"]})
    p, c = C.pick_capped(s, 9, 0)
    assert c == 3 and len(p) == 2 + 3 + 3 + 1 and p.token.is_unique
    assert p.groupby("log").size().to_dict() == {"A": 2, "B": 3, "C": 3, "D": 1}
    p2, _ = C.pick_capped(s.sample(frac=1, random_state=1), 9, 0)          # input order does not matter
    assert p.equals(p2)
    assert not p.equals(C.pick_capped(s, 9, 1)[0])


def test_disjoint_proof():
    a = pd.DataFrame({"token": ["x", "y"], "log": ["L1", "L2"]})
    b = pd.DataFrame({"token": ["z"], "log": ["L3"]})
    pr = C.disjoint_proof(a, b)
    assert pr["disjoint_tokens"] and pr["disjoint_logs"] and pr["token_intersection"] == []
    pr = C.disjoint_proof(a, pd.DataFrame({"token": ["z"], "log": ["L2"]}))
    assert pr["disjoint_tokens"] and not pr["disjoint_logs"] and pr["log_intersection"] == ["L2"]


def test_boot_diffs():
    log = np.repeat(np.arange(40), 5)
    rng = np.random.default_rng(0)
    den = np.ones(200, bool)
    na = rng.random(200) < 0.3
    r = C.boot_ratio_diff(log, na, den, na, den)
    assert r["diff"] == 0.0 and r["ci"] == [0.0, 0.0]
    nb = na & (rng.random(200) < 0.5)
    r = C.boot_ratio_diff(log, na, den, nb, den)
    assert r["diff"] == pytest.approx(na.mean() - nb.mean()) and r["ci"][0] > 0      # a strictly contains b
    r = C.boot_indep_diff(log, na, den, log + 100, na, den)
    assert r["diff"] == 0.0 and r["ci"][0] < 0 < r["ci"][1]


def _guard(lo_nc, lo_dac):
    return {"new NC failure rate (all sources)": {"ci": [lo_nc, lo_nc + 0.01]},
            "new DAC failure rate (all sources)": {"ci": [lo_dac, lo_dac + 0.01]}}


def test_verdict_rule():
    ok = {"all": True, "criteria": {"A1": True, "A2": True, "A3": True, "DAC_human_FA": True}}
    bad = {"all": False, "criteria": {"A1": True, "A2": True, "A3": False, "DAC_human_FA": True}}
    assert C.verdict(ok, _guard(-0.01, 0.0))["verdict"] == "CONFIRMED"               # CI touching 0 is not "above"
    assert C.verdict(ok, _guard(0.001, -0.01))["verdict"] == "NOT CONFIRMED"
    assert C.verdict(ok, _guard(-0.01, 0.002))["verdict"] == "NOT CONFIRMED"
    assert C.verdict(bad, _guard(-0.01, -0.01))["verdict"] == "NOT CONFIRMED"


def test_fragile():
    m = {"A1": {"boot_log": [0.8, 0.9]}, "A2": {"boot_log": [0.01, 0.03]}, "A3": {"boot_log": [0.55, 0.65]},
         "DAC_human_FA": {"boot_log": [0.0, 0.01]}}
    assert C.fragile(m) == {"A1": False, "A2": True, "A3": True, "DAC_human_FA": False}


def test_both_pool_choice():
    rows = [dict(sid=s["sid"], mask=s["mask"], m_col=s["m_col"], m_dac=s["m_dac"]) for s in R.SETTINGS]
    ts, tc = pd.DataFrame(rows), pd.DataFrame(rows)
    feas_s = {14, 15, 18, 19, 38}
    feas_c = {14, 18, 38, 10}
    ts["feasible"], tc["feasible"] = ts.sid.isin(feas_s), tc.sid.isin(feas_c)
    r = C.both_pool_choice(ts, tc)
    assert r["feasible_both_sids"] == [18, 14, 38] and r["rule_choice_sid"] == 18
    r = C.both_pool_choice(ts, tc, {18: False})
    assert r["rule_choice_sid"] == 14
    tc["feasible"] = False
    assert C.both_pool_choice(ts, tc)["rule_choice_sid"] is None


@pytest.mark.skipif(not (C.OUT / "tokens.parquet").exists(), reason="confirmation pool not selected")
def test_real_pools_disjoint():
    a = pd.read_parquet(C.SWEEP_OUT / "tokens.parquet")
    b = pd.read_parquet(C.OUT / "tokens.parquet")
    assert not set(a.token) & set(b.token) and not set(a.log) & set(b.log)
    split = pd.read_parquet(R.SPLIT)
    assert set(b.token) <= set(split.token) and set(split.split) == {"train"}
