"""Tests for stageT_decision.py --budget-def (PRESTATED_DECISION_RULE AMENDMENT 3 (2), "B+"; synthetic rows, CPU, < 20 s).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python -m pytest -q tools/refiner/tests/test_stageT_budget_def.py

'passing': the EP loss counts only drafts passing NC, DAC and DDC both before (orig) and after (final at theta) the
correction; 'all' (default) is run 1's definition, unchanged.
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/refiner"))

import stageT_decision as SD  # noqa: E402

# (p_g, orig (nc, dac, ddc, ep), tau1 (nc, dac, ddc, ep)); ttc = comfort = 1
DRAFTS = [
    (0.6, (1, 1, 1, 0.9), (1, 1, 1, 0.8)),    # d0 passes both: EP loss 0.1 when modified (theta <= 0.6)
    (0.9, (1, 0, 1, 0.2), (1, 1, 1, 0.9)),    # d1 fails DAC before -> excluded while modified (EP gain 0.7)
    (0.9, (1, 1, 1, 0.9), (1, 0, 1, 0.1)),    # d2 fails DAC after -> excluded while modified (EP loss 0.8)
    (0.9, (1, 1, 1, 1.0), (1, 1, 1, 1.0)),    # d3 passes both, no EP change
    (0.9, (0, 1, 1, 0.3), (0, 1, 1, 0.1)),    # d4 fails NC before and after -> always excluded
    (0.9, (1, 1, 1, 0.8), (1, 1, 0, 0.7)),    # d5 fails DDC after -> excluded while modified
    (0.9, (1, 1, 1, 0.5), (1, 1, 1, 0.9)),    # d6 passes both, EP gain 0.4 (theta <= 0.9)
    (0.2, (1, 0, 1, 0.1), (1, 1, 1, 1.0)),    # d7 fails DAC before, fixed: big EP gain (makes 'all' non-binding)
]


def rows(n_tok=6, n_logs=3, pg_shift=0.0):
    rec, toks = [], []
    for t in range(n_tok):
        tk = f"tok{t}"
        toks.append((tk, f"log{t % n_logs}"))
        for k, (pg, o, f) in enumerate(DRAFTS):
            r = dict(token=tk, k=k, family=k % 9, valid=True, p_g=min(1.0, pg + pg_shift), alpha=1.0)
            for s, v in (("orig", o), ("tau1", f)):
                nc, dac, ddc, ep = v
                r.update({f"nc_{s}": float(nc), f"dac_{s}": float(dac), f"ddc_{s}": float(ddc), f"ep_{s}": float(ep),
                          f"ttc_{s}": 1.0, f"comfort_{s}": 1.0,
                          f"pdms_{s}": nc * dac * ddc * (5 * ep + 5 + 2) / 12})
            rec.append(r)
    return pd.DataFrame(rec), pd.DataFrame(toks, columns=["token", "log"])


def frame():
    r, t = rows()
    return r.merge(t, on="token")


def write_eval(d: Path, r: pd.DataFrame, t: pd.DataFrame):
    d.mkdir(parents=True, exist_ok=True)
    r.to_parquet(d / "report_rows.parquet", index=False)
    t.to_parquet(d / "tokens.parquet", index=False)


def test_pass_both_and_ep_loss_definitions():
    df = frame()
    f = SD.final_outcomes(df, 0.0)                                   # everything modified
    pb = f.groupby("k").pass_both.first().to_dict()
    assert pb == {0: True, 1: False, 2: False, 3: True, 4: False, 5: False, 6: True, 7: False}
    # passing: d0 (0.1), d3 (0), d6 (-0.4) -> mean -0.1 -> -10 points
    assert SD.ep_loss_points(f, "passing") == pytest.approx(100 * (0.1 + 0.0 - 0.4) / 3)
    # all == the run-1 formula
    assert SD.ep_loss_points(f, "all") == 100 * f.ep_loss.mean()
    f_none = SD.final_outcomes(df, 1.01)                             # nothing modified: passing = passing orig
    assert set(f_none.groupby("k").pass_both.first()[lambda s: s].index) == {0, 2, 3, 5, 6}
    assert SD.ep_loss_points(f_none, "passing") == 0.0 and SD.ep_loss_points(f_none, "all") == 0.0
    f7 = SD.final_outcomes(df, 0.7)                                  # d0, d7 unmodified; d0 counts with 0 loss
    assert SD.ep_loss_points(f7, "passing") == pytest.approx(100 * (0.0 + 0.0 - 0.4) / 3)
    with pytest.raises(ValueError):
        SD.ep_loss_points(f, "bogus")


def test_sweep_all_unchanged_and_passing_binding():
    df = frame()
    thetas = (0.0, 0.5, 0.61, 0.9, 0.91, 1.0)
    sa = SD.sweep([df], thetas)
    for th, v in zip(sa.theta, sa.ep_loss_points):                  # 'all' == 100 mean(ep_orig - ep_final), as before
        assert v == pytest.approx(100 * SD.final_outcomes(df, th).ep_loss.mean(), abs=1e-12)
    assert SD.pick_theta(sa, 0.5) == 0.0                             # run-1 situation: fixes make 'all' negative
    sp = SD.sweep([df], thetas, "passing")
    assert np.allclose(sp.ep_loss_points_all, sa.ep_loss_points) and np.allclose(sp.ep_loss_points, sp.ep_loss_points_passing)
    assert np.array_equal(sp.d_pdms_points, sa.d_pdms_points)
    # d6 (p_g 0.9) is a pure-slowdown EP GAIN in this synthetic set; build the binding case from d0 alone
    df2 = df[df.k.isin([0, 3, 7])]
    sp2 = SD.sweep([df2], thetas, "passing")
    assert list(sp2.ep_loss_points.round(9)) == [pytest.approx(5.0), pytest.approx(5.0), 0.0, 0.0, 0.0, 0.0]
    assert SD.pick_theta(sp2, 0.5) == 0.61 and SD.pick_theta(SD.sweep([df2], thetas), 0.5) == 0.0
    assert list(sp2.n_passing) == [12, 12, 12, 12, 12, 12]           # d0 + d3 per token; d7 never (DAC before)


def test_select_records_budget_def(tmp_path):
    runs = tmp_path / "runs"
    r, t = rows()
    r = r[r.k.isin([0, 3, 7])]
    for arm in ("T", "none"):
        write_eval(SD.run_dir(runs, "wt", arm, 0, 0) / "eval_train_fold0", r, t)
    sel_all = SD.select(runs, ["wt"], ["T", "none"], [0], [0], budget_ep=0.5)
    sel_p = SD.select(runs, ["wt"], ["T", "none"], [0], [0], budget_ep=0.5, budget_def="passing")
    assert sel_all["budget_def"] == "all" and sel_p["budget_def"] == "passing"
    for arm in ("T", "none"):
        assert sel_all["arms"][arm]["theta"] == 0.0 and sel_p["arms"][arm]["theta"] == 0.61
        at = sel_p["arms"][arm]["candidates"]["wt"]["at_theta"]
        assert at["ep_loss_points"] == at["ep_loss_points_passing"] <= 0.5 and at["n_passing"] == 12
    with pytest.raises(ValueError):
        SD.select(runs, ["wt"], ["T"], [0], [0], budget_ep=0.5, budget_def="bogus")
    # CLI
    out = tmp_path / "sel.json"
    SD.main(["select", "--runs", str(runs), "--wtags", "wt", "--seeds", "0", "--folds", "0", "--budget-def", "passing",
             "--out", str(out)])
    assert json.loads(out.read_text())["budget_def"] == "passing"
    SD.main(["select", "--runs", str(runs), "--wtags", "wt", "--seeds", "0", "--folds", "0", "--out", str(out)])
    assert json.loads(out.read_text())["budget_def"] == "all"


def test_compare_uses_selection_budget_def(tmp_path):
    runs = tmp_path / "runs"
    for arm, sh in (("T", 0.0), ("none", 0.05)):
        for s in (0, 1):
            r, t = rows(n_tok=12, n_logs=6, pg_shift=sh)
            write_eval(SD.run_dir(runs, "wt", arm, -1, s) / "eval_dev", r, t)
    base = {"budget_ep_points": 0.5, "arms": {a: {"theta": 0.0, "wtag": "wt"} for a in ("T", "none")}}
    legacy = SD.compare(runs, base, [0, 1], n_boot=200)                          # run-1 selection json: 'all'
    df = SD.final_outcomes(SD.load_rows(SD.run_dir(runs, "wt", "T", -1, 0) / "eval_dev"), 0.0)
    assert legacy["budget_def"] == "all" and legacy["arms"]["T"]["budget_def"] == "all"
    assert legacy["arms"]["T"]["ep_loss_points"] == pytest.approx(100 * df.ep_loss.mean())
    assert legacy["arms"]["T"]["within_budget"] is True                          # negative 'all' loss
    selp = dict(base, budget_def="passing")
    rp = SD.compare(runs, selp, [0, 1], n_boot=200)
    a = rp["arms"]["T"]
    assert rp["budget_def"] == "passing" and a["ep_loss_points"] == a["ep_loss_points_passing"]
    assert a["ep_loss_points"] == pytest.approx(100 * (0.1 + 0.0 - 0.4) / 3)
    assert a["n_passing"] == 36 and a["ep_loss_points_all"] == legacy["arms"]["T"]["ep_loss_points"]
    assert rp["endpoints"] == legacy["endpoints"] and rp["outcome"] == legacy["outcome"]   # endpoints untouched
    selp1 = {"budget_ep_points": -20.0, "budget_def": "passing",
             "arms": {x: {"theta": 0.0, "wtag": "wt"} for x in ("T", "none")}}
    assert SD.compare(runs, selp1, [0, 1], n_boot=100)["arms"]["T"]["within_budget"] is False
    with pytest.raises(ValueError):
        SD.compare(runs, selp, [0, 1], n_boot=100, budget_def="all")              # must match the selection
    # CLI: an explicit mismatch is refused
    sp = tmp_path / "selp.json"
    sp.write_text(json.dumps(selp))
    with pytest.raises(SystemExit):
        SD.main(["compare", "--runs", str(runs), "--selection", str(sp), "--seeds", "0,1", "--n-boot", "100",
                 "--budget-def", "all", "--out", str(tmp_path / "d.json")])
    SD.main(["compare", "--runs", str(runs), "--selection", str(sp), "--seeds", "0,1", "--n-boot", "100",
             "--out", str(tmp_path / "d.json")])
    assert json.loads((tmp_path / "d.json").read_text())["budget_def"] == "passing"
