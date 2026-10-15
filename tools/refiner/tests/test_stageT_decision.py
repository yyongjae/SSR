"""Tests for tools/refiner/stageT_decision.py (cross-fitted theta selection, dev comparison; synthetic rows, CPU, < 20 s).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_stageT_decision.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/refiner"))

import stageT_decision as SD  # noqa: E402

METRICS = SD.METRICS


def synth_rows(n_logs=30, tok_per_log=4, K=13, seed=0, fix_prob=0.5, harm_ep=0.0, p_shift=0.0):
    """Every draft: orig fails NC with prob 0.2; tau1 fixes a failure with fix_prob and costs harm_ep EP; p_g is higher
    on failing drafts."""
    rng = np.random.default_rng(seed)
    rec, toks = [], []
    for lg in range(n_logs):
        for t in range(tok_per_log):
            tk = f"t{lg:03d}_{t}"
            toks.append((tk, f"log{lg:03d}"))
            for k in range(K):
                fail = rng.random() < 0.2
                fixed = fail and rng.random() < fix_prob
                ep = rng.uniform(0.6, 1.0)
                r = dict(token=tk, k=k, family=k % 9, valid=True,
                         p_g=float(np.clip((0.7 if fail else 0.3) + rng.normal(0, 0.1) + p_shift, 0, 1)), alpha=1.0)
                o = dict(nc=0.0 if fail else 1.0, dac=1.0, ddc=1.0, ep=ep, ttc=1.0, comfort=1.0)
                f = dict(o, nc=1.0 if (fixed or not fail) else 0.0, ep=max(ep - harm_ep, 0.0))
                for m, d in (("orig", o), ("tau1", f)):
                    for kk in ("nc", "dac", "ddc", "ep", "ttc", "comfort"):
                        r[f"{kk}_{m}"] = d[kk]
                    r[f"pdms_{m}"] = d["nc"] * d["dac"] * d["ddc"] * (5 * d["ep"] + 5 * d["ttc"] + 2 * d["comfort"]) / 12
                rec.append(r)
    return pd.DataFrame(rec), pd.DataFrame(toks, columns=["token", "log"])


def write_eval(d: Path, rows: pd.DataFrame, toks: pd.DataFrame):
    d.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(d / "report_rows.parquet", index=False)
    toks.to_parquet(d / "tokens.parquet", index=False)


def test_final_outcomes_gate_and_indicators():
    rows, toks = synth_rows(n_logs=3, seed=1)
    df = rows.merge(toks, on="token")
    f0 = SD.final_outcomes(df, 1.01)                       # nothing modified -> final == orig
    assert not f0.modified.any() and np.array_equal(f0.pdms_final, f0.pdms_orig) and f0.new_fail.sum() == 0
    f1 = SD.final_outcomes(df, 0.0)                        # everything modified -> final == tau1
    assert f1.modified.all() and np.array_equal(f1.pdms_final.to_numpy(), df.pdms_tau1.to_numpy())
    assert np.array_equal(f1.fail_ncttc.to_numpy(), (df.nc_tau1 < 1).to_numpy().astype(float))
    # invalid and non-finite rows are dropped
    df2 = df.copy()
    df2.loc[0, "valid"] = False
    df2.loc[1, "pdms_tau1"] = np.nan
    assert len(SD.final_outcomes(df2, 0.0)) == len(df) - 2
    assert len(SD.final_outcomes(df2, 1.01)) == len(df) - 1      # row 1 keeps its original (finite) scores


def test_cluster_boot_matches_iid_when_one_draft_per_log():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1.0, 2000)
    r = SD.cluster_boot(x, np.arange(2000), 4000)
    se = x.std() / np.sqrt(len(x))
    assert abs(r["mean"] - x.mean()) < 1e-12
    assert r["lo"] == pytest.approx(x.mean() - 1.96 * se, abs=0.4 * se)
    assert r["hi"] == pytest.approx(x.mean() + 1.96 * se, abs=0.4 * se)
    # clustering widens the interval when drafts of a log share their value
    logs = np.repeat(np.arange(100), 20)
    y = np.repeat(rng.normal(0, 1, 100), 20)
    rc = SD.cluster_boot(y, logs, 4000)
    ri = SD.cluster_boot(y, np.arange(len(y)), 4000)
    assert (rc["hi"] - rc["lo"]) > 3 * (ri["hi"] - ri["lo"])


def test_select_and_compare_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    # train pool OOF rows: arm T fixes more failures than arm none; both cost EP when modifying
    for arm, fix in (("T", 0.8), ("none", 0.3)):
        for s in (0, 1):
            for k in range(5):
                rows, toks = synth_rows(n_logs=8, seed=100 * s + k, fix_prob=fix, harm_ep=0.02)
                toks["token"] = toks.token + f"_f{k}"
                rows["token"] = rows.token + f"_f{k}"
                write_eval(SD.run_dir(runs, "wt", arm, k, s) / f"eval_train_fold{k}", rows, toks)
            rows, toks = synth_rows(n_logs=40, seed=1000 + s, fix_prob=fix, harm_ep=0.02)
            write_eval(SD.run_dir(runs, "wt", arm, -1, s) / "eval_dev", rows, toks)
    sel = SD.select(runs, ["wt"], ["T", "none"], [0, 1], [0, 1, 2, 3, 4], budget_ep=0.5)
    for arm in ("T", "none"):
        a = sel["arms"][arm]
        assert a["complete"] and a["wtag"] == "wt" and a["theta"] is not None
        c = a["candidates"]["wt"]
        assert c["at_theta"]["ep_loss_points"] <= 0.5
        sw = pd.DataFrame(c["sweep"])
        # the chosen theta is the smallest one inside the budget
        assert (sw[sw.theta < a["theta"]].ep_loss_points > 0.5).all()
    res = SD.compare(runs, sel, [0, 1], n_boot=2000)
    assert res["n_paired_drafts"] == 40 * 4 * 13 and res["n_logs"] == 40
    p1 = res["endpoints"]["P1_d_pdms_points"]
    assert p1["lo"] <= p1["mean"] <= p1["hi"]
    assert res["endpoints"]["P2_ncttc_reduction_pp"]["mean"] > 0          # T fixes more NC failures
    assert res["outcome"] in ("PASS", "INCONCLUSIVE", "EQUIVALENT", "WORSE") or res["outcome"].startswith("INVALID")
    assert set(res["by_family"]) and res["arms"]["T"]["theta"] == sel["arms"]["T"]["theta"]


def test_compare_identical_arms_is_equivalent(tmp_path):
    runs = tmp_path / "runs"
    for arm in ("T", "none"):
        for s in (0, 1, 2):
            rows, toks = synth_rows(n_logs=60, seed=500 + s, fix_prob=0.6, harm_ep=0.0)   # same rows in both arms
            write_eval(SD.run_dir(runs, "wt", arm, -1, s) / "eval_dev", rows, toks)
    sel = {"budget_ep_points": 0.5, "arms": {a: {"theta": 0.5, "wtag": "wt"} for a in ("T", "none")}}
    res = SD.compare(runs, sel, [0, 1, 2], n_boot=1000)
    ep = res["endpoints"]
    assert ep["P1_d_pdms_points"]["mean"] == 0 and ep["P1_d_pdms_points"]["lo"] == 0 == ep["P1_d_pdms_points"]["hi"]
    assert res["sanity"]["ok"] and res["outcome"] == "EQUIVALENT"
    assert res["p3_non_inferior"] and res["dac_ddc_non_inferior"]


def test_compare_no_improvement_is_invalid(tmp_path):
    runs = tmp_path / "runs"
    for arm in ("T", "none"):
        rows, toks = synth_rows(n_logs=30, seed=7, fix_prob=0.0, harm_ep=0.05)            # corrections only hurt
        write_eval(SD.run_dir(runs, "wt", arm, -1, 0) / "eval_dev", rows, toks)
    sel = {"budget_ep_points": 0.5, "arms": {a: {"theta": 0.0, "wtag": "wt"} for a in ("T", "none")}}
    res = SD.compare(runs, sel, [0], n_boot=500)
    assert not res["sanity"]["ok"] and res["outcome"].startswith("INVALID")
    assert not res["arms"]["T"]["within_budget"]


def test_compare_eval_name_reads_that_subdir(tmp_path):
    """--eval-name picks the eval subdirectory (eval_navtest) instead of eval_dev; default stays eval_dev."""
    runs = tmp_path / "runs"
    for arm, fix in (("T", 0.8), ("none", 0.3)):
        rows, toks = synth_rows(n_logs=20, seed=11, fix_prob=fix)
        write_eval(SD.run_dir(runs, "wt", arm, 0, 0) / "eval_navtest", rows, toks)
        rows, toks = synth_rows(n_logs=10, seed=12, fix_prob=0.5)                        # different dev rows
        write_eval(SD.run_dir(runs, "wt", arm, 0, 0) / "eval_dev", rows, toks)
    sel = {"budget_ep_points": 0.5, "arms": {a: {"theta": 0.0, "wtag": "wt"} for a in ("T", "none")}}
    rn = SD.compare(runs, sel, [0], n_boot=300, final_fold=0, eval_name="eval_navtest")
    rd = SD.compare(runs, sel, [0], n_boot=300, final_fold=0)
    assert rn["eval_name"] == "eval_navtest" and rd["eval_name"] == "eval_dev"
    assert rn["n_logs"] == 20 and rd["n_logs"] == 10
    assert rn["endpoints"]["P2_ncttc_reduction_pp"]["mean"] > 0
    out = tmp_path / "d.json"
    sp = tmp_path / "sel.json"
    import json
    sp.write_text(json.dumps(sel))
    SD.main(["compare", "--runs", str(runs), "--wtags", "wt", "--selection", str(sp), "--seeds", "0", "--final-fold", "0",
             "--n-boot", "300", "--eval-name", "eval_navtest", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["eval_name"] == "eval_navtest" and r["n_logs"] == 20
    with pytest.raises(FileNotFoundError):
        SD.compare(runs, sel, [0], n_boot=10, final_fold=0, eval_name="eval_missing")
