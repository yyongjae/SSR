"""Tests for tools/refiner/eval_refiner.py (gate mixing, metrics, predict / report on a synthetic split; CPU, < 1 min).
Official scoring of refined drafts (the 'score' sub-command and --direct-check) runs on real data in smoke_refiner.py.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_eval_refiner.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE), str(HERE.parent)):
    sys.path.insert(0, p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
import eval_refiner as EV  # noqa: E402
import refiner_synth as SY  # noqa: E402
import train_refiner as TR  # noqa: E402

M = {m: i for i, m in enumerate(EV.METRICS)}


def row(nc=1, dac=1, ddc=1, ep=1.0, ttc=1, comfort=1):
    pdms = nc * dac * ddc * (5 * ep + 5 * ttc + 2 * comfort) / 12
    return [nc, dac, ddc, ep, ttc, comfort, pdms]


def test_mix_metrics_definitions():
    orig = np.array([[row(nc=0), row(), row(), row(dac=0), row(ep=0.9)]], np.float64)     # [1, 5, 7]
    ref = np.array([[row(), row(ttc=0), row(ep=0.8), row(dac=0), row(ep=0.95)]], np.float64)
    valid = np.array([[True, True, True, True, False]])
    mod = np.array([[True, True, False, True, True]])
    r = EV.mix_metrics(orig, ref, mod, valid, family=np.array([[0, 1, 1, 2, 2]]))
    assert r["n"] == 4 and r["modified"] == 3
    assert r["fixed"] == 1                           # draft 0: NC fail -> pass
    assert r["new_fail"] == 1                        # draft 1: passing -> TTC fail
    assert r["fail_any"] == 2 and r["fail_any_orig"] == 2          # drafts 1 (ttc) and 3 (dac, unchanged)
    assert r["fail_nc"] == 0 and r["fail_nc_orig"] == 1 and r["fail_ttc"] == 1
    assert r["unnecessary_mod"] == 1                 # draft 1 passed nc, dac, ttc and was modified
    assert r["ep_loss_points"] == pytest.approx(0.0)                # draft 2 not modified (ep 0.8 unused)
    po = np.mean([orig[0, i, M["pdms"]] for i in range(4)])
    pf = np.mean([ref[0, 0, M["pdms"]], ref[0, 1, M["pdms"]], orig[0, 2, M["pdms"]], ref[0, 3, M["pdms"]]])
    assert r["pdms"] == pytest.approx(pf) and r["pdms_orig"] == pytest.approx(po)
    assert r["d_pdms_points"] == pytest.approx(100 * (pf - po))
    assert r["harm"] == 1
    assert r["by_family"]["small"]["new_fail"] == 1
    # theta semantics: nothing modified == original
    r0 = EV.mix_metrics(orig, ref, np.zeros_like(mod), valid)
    assert r0["d_pdms_points"] == 0 and r0["modified"] == 0 and r0["fixed"] == 0 and r0["new_fail"] == 0


def test_scores_array():
    df = pd.DataFrame({"token": ["a", "a", "b", "b", "b"], "k": [0, 1, 0, 1, -1], "error": [None, None, None, "x", "y"],
                       **{m: [1.0, 0.5, 0.0, 1.0, 1.0] for m in EV.METRICS}})
    arr = EV._scores_array(df, np.array(["b", "a"]), 2)
    assert arr[1, 0, 0] == 1.0 and arr[1, 1, 0] == 0.5 and arr[0, 0, 0] == 0.0 and np.isnan(arr[0, 1]).all()


def test_predict_and_report_synthetic(tmp_path):
    df, src = SY.make_sources(tmp_path, 10)
    RD.pack_split("train", df, tmp_path / "packed", sources=src, workers=1, log_fn=lambda *a, **k: None)
    a = TR.get_parser().parse_args(["--arm", "none", "--fold", "0", "--gpu", "-1", "--packed-root",
                                    str(tmp_path / "packed"), "--runs", str(tmp_path / "runs"), "--tokens-per-batch",
                                    "3", "--workers", "0", "--surrogate", "stub", "--max-steps", "2",
                                    "--inner-val-frac", "0.3", "--log-every", "100"])
    run = TR.train(a)
    ev = lambda *x: EV.get_parser().parse_args(list(x) + ["--run", str(run), "--split", "train", "--packed-root",
                                                          str(tmp_path / "packed"), "--loader-workers", "0"])
    e = ev("predict")
    EV.resolve_fold(e)
    assert e.fold == 0                                               # the run's own held-out fold
    out = EV.predict(e)
    assert out.name == "eval_train_fold0"
    P = np.load(out / "pred.npz")
    rows = P["rows"]
    assert set(RD.PackedSplit("train", tmp_path / "packed").index.fold.values[rows]) == {0}
    N, K = P["p_g"].shape
    assert K == 13 and P["tau1"].shape == (N, K, 8, 3) and np.isfinite(P["tau1"]).all()
    assert ((P["p_g"] > 0) & (P["p_g"] < 1)).all()
    R = np.load(out / "refined.npz")
    assert np.array_equal(R["drafts"], P["tau1"]) and list(R["tokens"]) == list(P["tokens"])
    assert list(pd.read_parquet(out / "tokens.parquet").token) == list(P["tokens"])
    # fake official scores of tau1: every refined draft fixes NC but loses EP 0.1
    Pk = RD.PackedSplit("train", tmp_path / "packed")
    lab = np.asarray(Pk.arrays["labels"][rows])
    recs = []
    for i, tk in enumerate(P["tokens"]):
        for k in range(K):
            d = {m: float(lab[i, k, RD.LBL[m]]) for m in EV.METRICS}
            d["nc"], d["ep"] = 1.0, max(0.0, d["ep"] - 0.1)
            d["pdms"] = d["nc"] * d["dac"] * d["ddc"] * (5 * d["ep"] + 5 * d["ttc"] + 2 * d["comfort"]) / 12
            recs.append(dict(token=tk, k=k, error=None, **d))
    pd.DataFrame(recs).to_parquet(out / "scores_tau1.parquet", index=False)
    e2 = ev("report", "--theta", "0.0", "1.01", "--sweep", "--budget-ep", "5")
    EV.resolve_fold(e2)
    rep = EV.report(e2)
    valid = P["draft_valid"].astype(bool)
    all_mod = rep["theta"]["0.0000"]
    assert all_mod["modified"] == valid.sum() == all_mod["n"]
    assert all_mod["fail_nc"] == 0 and all_mod["fail_nc_orig"] == int((lab[..., RD.LBL["nc"]][valid] < 1).sum())
    none = rep["theta"]["1.0100"]
    assert none["modified"] == 0 and none["d_pdms_points"] == 0.0
    assert rep["no_correction"]["pdms"] == none["pdms"]
    assert rep["theta_at_budget"] is not None
    tb = rep["theta"][f"{rep['theta_at_budget']:.4f}"]
    assert tb["ep_loss_points"] <= 5.0
    rows_df = pd.read_parquet(out / "report_rows.parquet")
    assert len(rows_df) == N * K and {"nc_orig", "nc_tau1", "p_g", "family"} <= set(rows_df.columns)
    assert json.loads((out / "report.json").read_text())["K"] == 13


# ----------------------------------------------------------------------------------------------- teacher shuffle
def test_derangement():
    toks = [f"t{i:03d}" for i in range(50)]
    m = EV.derangement(toks, 0)
    assert sorted(m) == toks and sorted(m.values()) == toks          # a permutation of the evaluated tokens
    assert all(k != v for k, v in m.items())                          # no fixed point
    assert m == EV.derangement(toks, 0) and m != EV.derangement(toks, 1)
    assert all(k != v for k, v in EV.derangement(toks[:2], 3).items())
    with pytest.raises(ValueError):
        EV.derangement(["a"], 0)


def test_predict_shuffle_teacher_equals_permuted_cache(tmp_path):
    """--shuffle-teacher-seed == an unshuffled predict on a cache whose files were physically permuted by the same map;
    tokens / rows / drafts unchanged; recorded in predict_meta.json."""
    import shutil

    df, src = SY.make_sources(tmp_path, 10)
    RD.pack_split("train", df, tmp_path / "packed", sources=src, workers=1, log_fn=lambda *a, **k: None)
    teacher = SY.make_fake_teacher(tmp_path, df.token)
    a = TR.get_parser().parse_args(["--arm", "T", "--fold", "0", "--gpu", "-1", "--packed-root", str(tmp_path / "packed"),
                                    "--runs", str(tmp_path / "runs"), "--teacher-root", str(teacher), "--tokens-per-batch",
                                    "3", "--workers", "0", "--n-norm", "4", "--surrogate", "stub", "--max-steps", "2",
                                    "--inner-val-frac", "0.3", "--log-every", "100"])
    run = TR.train(a)
    ev = lambda out, root, *x: EV.get_parser().parse_args(
        ["predict", "--run", str(run), "--split", "train", "--fold", "0", "--packed-root", str(tmp_path / "packed"),
         "--loader-workers", "0", "--teacher-root", str(root), "--out", str(tmp_path / out)] + list(x))
    o_true = EV.predict(ev("true", teacher))
    o_shuf = EV.predict(ev("shuf", teacher, "--shuffle-teacher-seed", "0"))
    meta = json.loads((o_shuf / "predict_meta.json").read_text())
    assert meta["teacher_shuffle"]["seed"] == 0 and meta["teacher_shuffle"]["fixed_points"] == 0
    assert json.loads((o_true / "predict_meta.json").read_text())["teacher_shuffle"] is None
    mapping = json.loads((o_shuf / "teacher_shuffle_map.json").read_text())
    # physically permuted cache: file of token t holds the BEV of mapping[t]
    perm = tmp_path / "teacher_perm"
    shutil.copytree(teacher, perm)
    for t, s in mapping.items():
        shutil.copy(teacher / "samples" / s[:2] / f"{s}.npz", perm / "samples" / t[:2] / f"{t}.npz")
    o_perm = EV.predict(ev("perm", perm))
    T_, S_, P_ = (np.load(o / "pred.npz") for o in (o_true, o_shuf, o_perm))
    for k in ("tokens", "rows", "tau0", "draft_valid", "family"):
        assert np.array_equal(T_[k], S_[k])                          # only the BEV lookup changes
    assert np.array_equal(S_["p_g"], P_["p_g"]) and np.array_equal(S_["tau1"], P_["tau1"])
    assert not np.array_equal(T_["p_g"], S_["p_g"])                  # the shuffle reaches the network
    assert set(mapping) == set(map(str, T_["tokens"]))
