"""Tests for tools/refiner/validate_surrogate.py (M8).  CPU, < 1 min.

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest \
          -q tools/refiner/tests/test_validate_surrogate.py

Helper statistics (Clopper-Pearson, recall at FPR, pair definitions incl. the pre-stated A3 literal rule), the tracked-
state frame conversion, and one real dev token end to end (skipped when the stage-T data are missing).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))
import validate_surrogate as V  # noqa: E402


def test_clopper_pearson_and_rate():
    from scipy.stats import beta
    assert V._cp(0, 10)[0] == 0.0 and V._cp(10, 10)[1] == 1.0
    lo, hi = V._cp(3, 10)
    assert abs(lo - beta.ppf(0.025, 3, 8)) < 1e-12 and abs(hi - beta.ppf(0.975, 4, 7)) < 1e-12
    r = V._rate(np.array([1, 0, 1, 1], bool), np.array([1, 1, 0, 1], bool))
    assert r["k"] == 2 and r["n"] == 3 and abs(r["p"] - 2 / 3) < 1e-12
    assert math.isnan(V._rate(np.zeros(2, bool), np.zeros(2, bool))["p"])


def test_recall_at_fpr():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    s = np.array([5, 4, 0.5, 3, 1, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0, -1, -2, -3, -4, -5, -6], float)
    assert V._recall_at_fpr(y, s, 0.0) == 2 / 3        # threshold = top negative (3): positives 5, 4
    assert V._recall_at_fpr(y, s, 0.5) == 1.0


def test_states_to_n_roundtrip():
    class RA:
        x, y, heading = 664_123.25, 3_998_001.5, 2.9
    rng = np.random.default_rng(0)
    n = np.cumsum(rng.normal([1, 0, 0.01], [0.2, 0.1, 0.02], (3, 41, 3)), 1)
    c, s = math.cos(RA.heading), math.sin(RA.heading)
    st = np.zeros((3, 41, 11))
    st[..., 0] = RA.x + c * n[..., 0] - s * n[..., 1]
    st[..., 1] = RA.y + s * n[..., 0] + c * n[..., 1]
    st[..., 2] = np.arctan2(np.sin(n[..., 2] + RA.heading), np.cos(n[..., 2] + RA.heading))
    back = V._states_to_n(st, RA)
    assert np.abs(back[..., :2] - n[..., :2]).max() < 1e-8
    assert np.abs(back[..., 2] - n[..., 2] + 2 * np.pi * np.round((back[..., 2] - n[..., 2]) / (2 * np.pi))).max() < 1e-9


def test_pairs_and_pre_stated_A3():
    rows = []
    # token t: source k=1 (fails NC, flagged) -> guided fixes both;  k=2 (passes, flagged = false alarm) -> guided clean;
    # k=3 (fails, not flagged = miss) -> guided unchanged officially, still clean
    src = [(1, 0.0, 0.5), (2, 1.0, 0.2), (3, 0.0, -0.5)]
    for k, nc, viol in src:
        rows.append(dict(token="t", log="L", k=k, pool="bank", src=-1, modified=False, nc=nc, col_viol=viol))
    for j, (k, nc1, viol1) in enumerate([(1, 1.0, -0.1), (2, 1.0, -0.2), (3, 0.0, -0.6)]):
        rows.append(dict(token="t", log="L", k=20 + j, pool="guided", src=k, modified=True, nc=nc1, col_viol=viol1))
    df = pd.DataFrame(rows)
    for c in ("dac", "ddc", "ttc", "comfort", "pdms", "ep", "raw_progress", "P_sur"):
        df[c] = 1.0
    for c in ("col_cost", "dac_viol", "dac_cost"):
        df[c] = -1.0 if c == "dac_viol" else 0.0
    df["col_cost"] = np.where(df.col_viol > 0, 0.3, 0.0)
    df["col_hard"] = df["dac_hard"] = df["t_col_hard"] = df["t_dac_hard"] = False
    df["t_col_viol"] = df["col_viol"]
    df["col_gmin_hard"] = 1.0
    df["fam"] = 1
    pp = V._pairs(df, "guided")
    assert len(pp) == 3 and list(pp.nc_0) == [0.0, 1.0, 0.0]
    f0, f1 = pp.nc_0.to_numpy() < 1, pp.nc_1.to_numpy() < 1
    s0, s1 = pp.col_viol_0.to_numpy() > 0, pp.col_viol_1.to_numpy() > 0
    b = V._pair_block(pp, f0, f1, s0, s1, (pp.col_cost_1 - pp.col_cost_0).to_numpy(), "NC")
    # surrogate says fixed on k=1, k=2; officially fixed only k=1 -> 1/2 (literal A3)
    assert b["A3_literal P(off fixed | sur fixed)"]["k"] == 1 and b["A3_literal P(off fixed | sur fixed)"]["n"] == 2
    # conditioning on official failure: k=1 (fixed) and k=3 (not fixed), both surrogate-clean after -> 1/2
    assert b["P(off pass1 | off fail0, sur clean1)"]["n"] == 2 and b["P(off pass1 | off fail0, sur clean1)"]["k"] == 1
    assert b["P(off pass1 | off fail0 & sur flag0, sur clean1)"]["p"] == 1.0


DATA = Path("/home/external-user/ssd/yongjae_refiner")


@pytest.mark.skipif(not (DATA / "m8/tokens.parquet").exists() or not V.OBJ_DIR.exists(), reason="M8 data missing")
def test_process_one_real_token():
    toks = pd.read_parquet(DATA / "m8/tokens.parquet")
    toks = toks[[(V.OBJ_DIR / f"{t}.npz").exists() for t in toks.token]]
    if not len(toks):
        pytest.skip("no objects")
    V._init()
    tok, log = toks.token.iloc[0], toks.log.iloc[0]
    rows = V.process_token((tok, log))
    df = pd.DataFrame(rows)
    assert (df.error == "").all(), df.error.iloc[0]
    n_er = int((df.pool == "erule").sum())
    assert len(df) == 13 + 1 + n_er + 26
    assert (df.pool == "bank").sum() == 13 and (df.pool == "guided").sum() == 13 and (df.pool == "random").sum() == 13
    assert np.abs(df.P_sur - df.P_shp).max() < 1e-6                 # cropped torch projection == shapely full line
    h = df[(df.pool == "bank") & (df.k == 0)].iloc[0]
    assert h.fam == 0 and not bool(h.col_hard)                       # human never overlaps a counted object
    # unmodified corrections keep the source's official scores
    for _, r in df[df.pool.isin(["guided", "random"]) & ~df.modified].iterrows():
        s = df[df.k == r.src].iloc[0]
        assert r.nc == s.nc and r.pdms == s.pdms
    g = df[df.pool == "guided"]
    assert g.g_finite.all() and (g.g_loss1 <= g.g_loss0 + 1e-12).all()
