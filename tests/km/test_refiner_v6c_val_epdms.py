"""tools/refiner_v6c/val_epdms.py (CPU, synthetic): adjacency = official infer_start_adjacent_mapping rule on the
evaluated subset, v2.2 score recomposition, log bootstrap = tools/ck/eval_ck.log_bootstrap, submission pickles from a
dump (out = switch ? v6 : base, c0 = traj[msel]), quoted subset split (float-like tokens survive OmegaConf), and the
paired analysis on synthetic official csvs (EC NaN <=> no previous frame, EC identical on unchanged pairs)."""
from __future__ import annotations

import hashlib
import json

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.refiner_v6c import val_epdms as VE  # noqa: E402

TOKS = ["3315880386e45927", "0e21222359505469", "aa01", "aa02", "bb01", "bb02"]   # first two parse as floats unquoted
LOGS = ["logA", "logA", "logA", "logA", "logB", "logB"]
TIMES = [0.0, 0.5, 1.0, 2.0, 0.0, 0.5]                                               # aa02: gap 1.0 s -> no previous


@pytest.fixture()
def adj_full(tmp_path, monkeypatch):
    p = tmp_path / "adjacent_full.parquet"
    pd.DataFrame({"token": TOKS, "log_name": LOGS, "t": TIMES, "prev": [None] * len(TOKS)}).to_parquet(p)
    monkeypatch.setattr(VE, "ADJ_FULL", p)
    return p


def test_adjacency_rule_and_subset(adj_full):
    a = VE.adjacency(TOKS).set_index("token")
    assert a.loc["0e21222359505469", "prev"] == "3315880386e45927"
    assert a.loc["aa01", "prev"] == "0e21222359505469"
    assert a.loc["aa02", "prev"] is None and a.loc["3315880386e45927", "prev"] is None   # 1.0 s gap, first frame
    assert a.loc["bb02", "prev"] == "bb01" and a.loc["bb01", "prev"] is None              # other log
    s = VE.adjacency(["3315880386e45927", "aa01"]).set_index("token")                    # the middle frame is not evaluated
    assert s.prev.isna().all()                                                            # 1.0 s apart -> no pair


def _csv_frame(n, ec):
    df = pd.DataFrame({m: np.ones(n) for m in VE.MET})
    df["ego_progress"] = np.linspace(0.5, 1.0, n)
    df["two_frame_extended_comfort"] = ec
    return df


def test_recompose_matches_v22_formula():
    df = _csv_frame(3, [1.0, 0.0, np.nan])
    df.loc[1, "no_at_fault_collisions"] = 0.5
    r = VE.recompose(df)
    ep = df.ego_progress.to_numpy()
    assert np.allclose(r, [(5 * ep[0] + 5 + 2 + 2 + 2) / 16, 0.5 * (5 * ep[1] + 9) / 16, (5 * ep[2] + 9) / 14])
    assert np.allclose(VE.no_ec(df), [(5 * ep[0] + 9) / 14, 0.5 * (5 * ep[1] + 9) / 14, (5 * ep[2] + 9) / 14])


def test_log_bootstrap_equals_eval_ck():
    from tools.ck.eval_ck import log_bootstrap as ref
    rng = np.random.default_rng(3)
    d, lg = rng.normal(size=200), rng.integers(0, 17, 200).astype(str)
    d[5] = np.nan
    assert VE.log_bootstrap(d, lg, 500, 1) == ref(d, lg, 500, 1)


def _fake_dump(tmp_path, monkeypatch):
    root = tmp_path / "km"
    (root / "packed/navtrain_val").mkdir(parents=True)
    pd.DataFrame({"token": TOKS, "log": LOGS, "city": "x", "row": range(len(TOKS))}).to_parquet(
        root / "packed/navtrain_val/tokens.parquet")
    monkeypatch.setenv("KM_CORR_ROOT", str(root))
    d = tmp_path / "dump" / "navtrain_val"
    d.mkdir(parents=True)
    n = len(TOKS)
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n, 8, 3)).astype(np.float32)
    v6 = base + 0.01
    sw = np.array([1, 0, 1, 0, 0, 1], np.uint8)
    traj = rng.normal(size=(n, 256, 8, 3)).astype(np.float32)
    msel = np.array([3, 1, 4, 1, 5, 9], np.int16)
    for k, v in dict(v6_base=base, v6_traj=v6, v6_switch=sw, traj=traj, msel=msel).items():
        np.save(d / f"{k}.npy", v)
    done = np.ones(n, bool)
    done[4] = False                                                                       # one row not dumped
    np.save(d / "done_0.npy", done)
    return d.parent, base, v6, sw.astype(bool), traj, msel, done


def test_pkl_submissions(tmp_path, monkeypatch):
    dump, base, v6, sw, traj, msel, done = _fake_dump(tmp_path, monkeypatch)
    out = tmp_path / "out"
    a = argparse.Namespace(out=str(out), dump=str(dump), split="navtrain_val", c0_dump=str(dump), npz=None)
    assert VE.cmd_pkl(a) == 0
    rows = np.flatnonzero(done)
    P = {s: pickle.load(open(out / "traj" / f"{s}.pkl", "rb")) for s in ("base", "v6", "out", "c0")}
    toks = [TOKS[r] for r in rows]
    for s in P:
        assert list(P[s]) == toks and all(x.dtype == np.float32 and x.shape == (8, 3) for x in P[s].values())
    for i, r in enumerate(rows):
        t = TOKS[r]
        assert np.array_equal(P["base"][t], base[r]) and np.array_equal(P["v6"][t], v6[r])
        assert np.array_equal(P["out"][t], v6[r] if sw[r] else base[r])
        assert np.array_equal(P["c0"][t], traj[r, msel[r]])
    tk = pd.read_parquet(out / "tokens.parquet")
    assert tk.token.tolist() == toks and tk.switch.tolist() == sw[rows].tolist()


def test_subset_split_is_quoted(tmp_path, adj_full):
    name = VE.split_for(tmp_path, TOKS[:3], LOGS[:3])
    assert name != VE.FULL_SPLIT and name.endswith("_q")
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(tmp_path / "config/train_test_split/scene_filter" / f"{name}.yaml")
    assert list(cfg.tokens) == TOKS[:3] and all(isinstance(t, str) for t in cfg.tokens)
    assert VE.split_for(tmp_path, TOKS, LOGS) == VE.FULL_SPLIT


def test_analyze_paired(tmp_path, adj_full):
    out = tmp_path / "out"
    (out / "traj").mkdir(parents=True)
    n = len(TOKS)
    has_prev = np.array([False, True, True, False, False, True])
    base = {t: np.zeros((8, 3), np.float32) for t in TOKS}
    outp = {t: x.copy() for t, x in base.items()}
    outp["aa01"][-1, 0] = 0.5                                                             # one changed token
    md5 = {}
    for s, T in (("base", base), ("out", outp)):
        pickle.dump(T, open(out / "traj" / f"{s}.pkl", "wb"))
        md5[s] = hashlib.md5((out / "traj" / f"{s}.pkl").read_bytes()).hexdigest()
    pickle.dump(base, open(out / "traj" / "c0.pkl", "wb"))                               # stale: not in pkl_info
    json.dump({"subs": {s: {"md5": m} for s, m in md5.items()}}, open(out / "pkl_info.json", "w"))
    pd.DataFrame({"token": TOKS, "log": LOGS, "row": range(n), "switch": [t == "aa01" for t in TOKS]}).to_parquet(
        out / "tokens.parquet")
    for s, ec_changed in (("base", 1.0), ("out", 0.0)):
        ec = np.where(has_prev, 1.0, np.nan)
        ec[TOKS.index("aa01")] = ec_changed                                               # EC differs only where changed
        df = _csv_frame(n, ec)
        df.insert(0, "token", TOKS)
        df.insert(1, "valid", True)
        if s == "out":
            df.loc[TOKS.index("aa01"), "driving_direction_compliance"] = 0.5
        df["score"] = VE.recompose(df)
        avg = {c: df[c].mean() if c not in ("token", "valid") else ("average_all_frames" if c == "token" else True)
               for c in df.columns}
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
        rd = VE.run_dir(out, s, md5[s])
        rd.mkdir(parents=True)
        df.to_csv(rd / "x.csv")
    (out / "exp" / "epdms_val_c0_000000000000").mkdir(parents=True)                       # a stale c0 result
    res = VE.analyze(out, n_boot=200)
    assert "c0" not in res["paired"].get("out_minus_c0", {}) and not any("c0" in k for k in res["paired"])
    assert res["n_with_prev"] == int(has_prev.sum())
    assert all(v for k, v in res["checks"].items() if k.endswith(("_all_valid", "_ec_nan_equals_no_prev")))
    ch = res["change"]["out_minus_base"]
    assert ch["cur_changed"]["n"] == 1 and ch["ec_identical_on_unchanged_pairs"]
    assert res["ddc_flips"]["out_minus_base"] == {"n_differ": 1, "out_lower": 1, "out_higher": 0,
                                                  "pct_differ": pytest.approx(100 / n)}
    assert res["paired"]["out_minus_base"]["EC"]["n"] == int(has_prev.sum())
    assert res["paired"]["out_minus_base"]["EPDMS"]["mean"] < 0


def test_csv_follows_the_current_pickle_and_force_reruns(tmp_path):
    """review 10-28: a csv is reused only for the same pickle (md5), and a --force re-run's newer csv wins"""
    import os
    import time as _t
    out = tmp_path
    json.dump({"subs": {"base": {"md5": "a" * 32}}}, open(out / "pkl_info.json", "w"))
    assert VE.csv_of(out, "base") is None
    d1 = VE.run_dir(out, "base", "a" * 32)
    d1.mkdir(parents=True)
    (d1 / "1.csv").write_text("x")
    d2 = d1.with_name(d1.name + "_20261028010101")
    d2.mkdir()
    (d2 / "2.csv").write_text("y")
    os.utime(d1 / "1.csv", (_t.time() - 100, _t.time() - 100))
    assert VE.csv_of(out, "base") == d2 / "2.csv"
    json.dump({"subs": {"base": {"md5": "b" * 32}}}, open(out / "pkl_info.json", "w"))    # a new dump's pickle
    assert VE.csv_of(out, "base") is None and VE.csv_of(out, "c0") is None
