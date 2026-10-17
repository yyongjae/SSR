"""tools/ck/v4/eval_range_teacher.py (refiner v4_3 (c): range teachers as selection evidence): the amended Phase 1c P1
decision (FILTER rule rows; p1_decide PASS / FAIL by (a), (b), (c) / VOID on synthetic arrays, the inclusive bars, the
deciding block = val held-out + ck2T10 only, the md decision lines; the P1 gate: the registered token set
(HO/prep/ho_rows.npy, per-input counts of removed rows), the registered teacher / unchanged checkpoint, the GT tube, and
stage eval on the deciding split with fake v4T10_rng2 outputs: half done -> report only + non-zero exit), the resume
refusal of stage infer (another checkpoint / which, done rows without infer_meta.json), the descriptive NC only / TTC
only filter rows, the old criterion S switch rule (pool-order ties = Phase 0b) and its numbers, the fidelity / bootstrap
helpers, the pair key rule, the blend critics, the Phase 0b R2 GT tube (bitwise equal to
tools/ck/v4/oracle/oracle_common.grid_eval 'grid stopped-aware NC + read any4' on the same states; the LQR states from
the packed status vs the official F1 states) and the readout-rule tube, the --gt-tube rules (gttube refuses it; eval
reads it read-only after the meta / shape checks), the Phase 0b reproduction check logic (switch and filter rows), --tag
report names, the CUDA guard of stage infer (approval, busy GPU; no GPU is touched), and a CPU smoke of every stage on a
few real navtest / navtrain_val tokens with randomly initialised R2 range teachers saved to a scratch dir (MAP refused
on val, non-range teachers refused).
CPU only:

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_eval_range_teacher.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_KEYS, LBL  # noqa: E402
from navsim.agents.para_ssr.ck.model import CKNet, load_arm_norm, save_ck  # noqa: E402
from navsim.agents.para_ssr.ck.select2 import TIE_ORDER  # noqa: E402
from navsim.agents.para_ssr.ck.v4 import V4Cfg  # noqa: E402
from tools.ck.v4 import eval_range_teacher as E  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CK = E.CK
R2 = V4Cfg.range_arm(rg_read="any4", rg_stopped_aware=True)
ORACLE_PREP = CK / "v4" / "analysis" / "range_oracle" / "prep"
have_f1 = pytest.mark.skipif(not ((ORACLE_PREP / "f1_sim.npy").is_file() and (CK / "v3/gt_objects/navtrain_val").is_dir()),
                             reason="Phase 0 oracle prep (F1 arrays) / GT objects absent")


def _navtest_inputs():
    sp = E.SPLITS["navtest"]
    need = [sp["variants"] / "traj80.npy", sp["variants"] / "labels" / "labels.npy", CK / "labels/navtest/cand/labels.npy",
            sp["base_root"] / "ck2T10/r34/prob16.npy", sp["base_root"] / "ck2M10/r34/prob16.npy", sp["gt_obj"], sp["sdf"]]
    return TL.have_inputs() and all(p.exists() for p in need)


def _val_inputs():
    sp = E.SPLITS["navtrain_val"]
    need = [sp["variants"] / "traj80.npy", sp["variants"] / "labels" / "labels.npy",
            CK / "labels/navtrain_val/cand/labels.npy", sp["base_root"] / "ck2T10/r34/prob16.npy", E.F1_ROWS,
            sp["gt_obj"], sp["sdf"]]
    return TL.have_inputs() and all(p.exists() for p in need)


# ----------------------------------------------------------------------------------------------- rules
def _lab(pdms, fail=None):
    """labels [n, 96, 9] with the given PDMS [n, 96] and NC fails where fail."""
    n = pdms.shape[0]
    lab = np.ones((n, 96, 9))
    lab[..., LBL["pdms"]] = pdms
    if fail is not None:
        lab[..., LBL["nc"]] = np.where(fail, 0.0, 1.0)
    return lab


def test_tube_switch_rules():
    """switch / filter / blend picks; ties in the pool order (parent k, then its variants = heldout_analyze.amax), not
    select2's TIE_ORDER (all identity columns first)."""
    n = 4
    score = np.zeros((n, 96))
    score[:, 7] = 3.0                                                       # the blend's favourite (a variant)
    score[:, 12] = 2.0                                                      # identity column of parent 2
    score[:, 13] = 2.0                                                      # its variant (tie: identity first)
    allowed = np.ones((n, 96), bool)
    allowed[3, 7] = False                                                   # token 3: column 7 not selectable
    inside = np.ones((n, 96), bool)
    inside[1, 0] = False                                                    # token 1: top-1 outside, 7 inside
    inside[2, [0, 7]] = False                                               # token 2: top-1 and 7 outside
    inside[3, :] = False                                                    # token 3: nothing inside
    pk = E.tube_switch(score, allowed, inside)
    assert pk["blend"].tolist() == [7, 7, 7, 12]
    assert pk["switch"].tolist() == [0, 7, 12, 12]                          # tie 12 / 13 -> the identity column 12
    assert pk["filter"].tolist() == [7, 7, 12, 12]                          # token 3: none inside -> the blend pick
    s = np.random.default_rng(0).normal(size=(5, 96))
    assert np.array_equal(E.argmax_tie(s), TIE_ORDER[np.argmax(s[:, TIE_ORDER], 1)])
    # a tie between parent 0's variant (column 1) and parent 1 (column 6): Phase 0b's pool order picks 1
    t = np.zeros((1, 96))
    t[0, [1, 6]] = 1.0
    assert E.tube_switch(t, np.ones((1, 96), bool), np.zeros((1, 96), bool))["blend"].tolist() == [1]
    assert E.argmax_tie(t).tolist() == [6]
    sys.path.insert(0, str(REPO / "tools/ck/v4/oracle"))
    import heldout_analyze as HA                                            # the Phase 0b rule on F1 column order
    f1 = np.random.default_rng(3).integers(0, 3, size=(50, 96)).astype(float)   # many ties
    key = HA.POOL96_KEY                                                     # F1 column -> select2 column
    s2 = np.zeros_like(f1)
    s2[:, key] = f1
    assert np.array_equal(key[HA.amax(f1)], E.argmax_pool(s2))


def test_criterion_s_numbers_and_break():
    n = 3
    pd_ = np.full((n, 96), 0.5)
    pd_[:, 0] = 0.6
    pd_[:, 7] = 0.9
    fail = np.zeros((n, 96), bool)
    fail[0, 7] = True                                                       # token 0: the blend pick fails
    fail[2, 0] = True                                                       # token 2: top-1 fails (switch keeps it)
    lab = _lab(pd_, fail)
    score = np.zeros((n, 96))
    score[:, 7] = 1.0
    inside = np.ones((n, 96), bool)
    row, d, pk = E.criterion_s(lab, score, np.ones((n, 96), bool), inside, np.array(["a", "b", "c"]))
    assert pk["switch"].tolist() == [0, 0, 0] and pk["blend"].tolist() == [7, 7, 7]
    assert np.allclose(d, [-30.0, -30.0, -30.0])
    assert row["break_"] == pytest.approx(0.5)                               # tokens 1, 2 pass the blend; 2 breaks
    assert (row["break_n"], row["break_den"]) == (1, 2)
    assert not row["bars_pass"] and row["top1_outside"] == 0.0 and row["changed_vs_top1"] == 0.0


def test_criterion_filter_numbers():
    """the FILTER rule row: filter - blend, break, rows dropped, blend pick dropped, changed vs blend, none left -> the
    blend pick; the picks are tube_switch's filter picks."""
    n = 4
    pd_ = np.ones((n, 96))
    fail = np.zeros((n, 96), bool)
    pd_[:, 7], fail[:, 7] = 0.0, True                                       # the blend's pick fails NC everywhere
    pd_[3, 7], fail[3, 7] = 1.0, False                                      # ... except on token 3
    pd_[2, 12], fail[2, 12] = 0.4, True                                     # token 2: the next best fails too
    lab = _lab(pd_, fail)
    score = np.zeros((n, 96))
    score[:, 7], score[:, 12] = 3.0, 2.0
    allowed = np.ones((n, 96), bool)
    inside = np.ones((n, 96), bool)
    inside[[0, 2, 3], 7] = False                                            # token 1: the tube keeps the failing pick
    inside[3, :] = False                                                    # token 3: nothing left -> the blend pick
    row, d, pk = E.criterion_filter(lab, score, allowed, inside, np.array(["a", "a", "b", "b"]))
    assert pk["filter"].tolist() == [12, 7, 12, 7] and pk["blend"].tolist() == [7, 7, 7, 7]
    assert np.allclose(d, [100.0, 0.0, 40.0, 0.0])
    assert row["d_vs_blend"]["mean"] == pytest.approx(35.0)
    assert (row["break_n"], row["break_den"]) == (0, 1)                     # only token 3's blend pick passes
    assert row["dropped_rows"] == pytest.approx((3 + 95) / (4 * 96))
    assert row["blend_pick_dropped"] == pytest.approx(0.75) and row["changed_vs_blend"] == pytest.approx(0.5)
    assert row["no_inside"] == pytest.approx(0.25) and row["rule"] == "filter"
    assert row["fail3_blend"] == pytest.approx(0.75) and row["fail3_filter"] == pytest.approx(0.5)


def _p1_case(learned_fix_every: int = 1, false_drop_tokens: int = 0, gt: str = "fix"):
    """synthetic P1 arrays: 400 tokens in 40 logs (10 consecutive tokens each); the blend picks column 7, which fails
    NC (PDMS 0) on the even tokens; column 12 (next best) passes there but fails on the odd tokens.
    GT tube: 'fix' drops column 7 on every even token (+100 there); 'none' drops nothing (gain 0); 'harm' drops it on
    the odd tokens only (-100 there).  Learned tube: drops column 7 on the even tokens i with i % (2 *
    learned_fix_every) == 0, and on the first false_drop_tokens odd tokens (-> column 12, a break)."""
    n = 400
    logs = np.array([f"log{i // 10}" for i in range(n)])
    i = np.arange(n)
    even = i % 2 == 0
    pd_ = np.ones((n, 96))
    fail = np.zeros((n, 96), bool)
    pd_[even, 7], fail[even, 7] = 0.0, True
    pd_[~even, 12], fail[~even, 12] = 0.0, True
    lab = _lab(pd_, fail)
    score = np.zeros((n, 96))
    score[:, 7], score[:, 12] = 3.0, 2.0
    allowed = np.ones((n, 96), bool)
    ins_gt = np.ones((n, 96), bool)
    if gt == "fix":
        ins_gt[even, 7] = False
    elif gt == "harm":
        ins_gt[~even, 7] = False
    ins_l = np.ones((n, 96), bool)
    ins_l[even & (i % (2 * learned_fix_every) == 0), 7] = False
    ins_l[np.flatnonzero(~even)[:false_drop_tokens], 7] = False
    return lab, score, allowed, ins_l, ins_gt, logs


def _p1(lab, score, allowed, ins_l, ins_gt, logs):
    lb = E.LogBoot(logs)
    blk, _ = E.s_block(lab, score, allowed, logs, lb, {}, {"learned": ins_l, "gt_r2_same_keys": ins_gt},
                       {"learned": "learned", "gt": "gt_r2_same_keys"})
    return blk["P1"], blk


def test_p1_decision_pass_fail_void_synthetic():
    """the amended P1 decision on synthetic arrays (through s_block's filter rows): PASS; FAIL by (a), by (b) only, by
    (c) only; VOID when the GT reference is zero or negative (whatever the learned tube does)."""
    # PASS: the learned tube = the GT tube
    P, blk = _p1(*_p1_case(1))
    assert P["verdict"] == "PASS" and all(P["parts"].values())
    assert P["d_learned"]["mean"] == pytest.approx(50.0) and P["gain_fraction"] == pytest.approx(1.0)
    assert P["break_n"] == 0 and P["gt_reference"]["valid"] and blk["filter"]["gain_fraction"]["estimate"] == 1.0
    assert P["learned_tube"] == "learned" and P["gt_tube"] == "gt_r2_same_keys" and P["keys"] == ["nc", "ttc"]
    # FAIL (a): the learned tube drops nothing -> 0 gain (and 0 % of the GT gain)
    lab, score, allowed, _, ins_gt, logs = _p1_case(1)
    P, _ = _p1(lab, score, allowed, np.ones((400, 96), bool), ins_gt, logs)
    assert P["verdict"] == "FAIL" and not P["parts"]["a_mean_and_ci"] and not P["parts"]["c_gain_fraction"]
    assert P["d_learned"]["mean"] == 0.0 and P["gain_fraction"] == 0.0 and P["parts"]["b_break"]
    # FAIL (a): one fix (token 0) in 400 tokens: mean +0.25 < +0.30
    P, _ = _p1(*_p1_case(learned_fix_every=10 ** 6))
    assert P["verdict"] == "FAIL" and P["d_learned"]["mean"] == pytest.approx(0.25)
    assert not P["parts"]["a_mean_and_ci"] and not P["parts"]["c_gain_fraction"]
    # FAIL (b) only: GT fixes + 6 false drops of a passing blend pick (break 6 / 200 = 3 %)
    P, _ = _p1(*_p1_case(1, false_drop_tokens=6))
    assert P["verdict"] == "FAIL" and P["parts"]["a_mean_and_ci"] and P["parts"]["c_gain_fraction"]
    assert not P["parts"]["b_break"] and (P["break_n"], P["break_den"]) == (6, 200)
    assert P["d_learned"]["mean"] == pytest.approx(50.0 - 600 / 400) and P["gain_fraction"] == pytest.approx(0.97)
    # FAIL (c) only: the learned tube fixes 1 of the 5 bad tokens of every log: +10 everywhere (CI > 0), 20 % < 25 %
    P, _ = _p1(*_p1_case(learned_fix_every=5))
    assert P["verdict"] == "FAIL" and P["parts"]["a_mean_and_ci"] and P["parts"]["b_break"]
    assert not P["parts"]["c_gain_fraction"] and P["gain_fraction"] == pytest.approx(0.2)
    assert P["d_learned"]["lo"] > 0 and P["gain_fraction_ci"] == pytest.approx([0.2, 0.2])
    # 2 of 5 (40 %) passes (c)
    lab, score, allowed, ins_l, ins_gt, logs = _p1_case(1)
    i = np.arange(400)
    ins_l = np.ones((400, 96), bool)
    ins_l[(i % 10 == 0) | (i % 10 == 2), 7] = False
    P, _ = _p1(lab, score, allowed, ins_l, ins_gt, logs)
    assert P["verdict"] == "PASS" and P["gain_fraction"] == pytest.approx(0.4)
    # VOID: the GT reference has no gain (CI lower bound 0, not > 0) or loses; the learned tube is irrelevant
    for gt in ("none", "harm"):
        P, _ = _p1(*_p1_case(1, gt=gt))
        assert P["verdict"] == "VOID" and not P["gt_reference"]["valid"] and not P["parts"]["gt_reference_valid"]
        assert P["parts"]["a_mean_and_ci"] and not P["parts"]["c_gain_fraction"]
        assert P["gt_reference"]["d_vs_blend"]["mean"] == (0.0 if gt == "none" else pytest.approx(-50.0))


def test_p1_decide_bars_and_edges():
    """p1_decide on hand-made rows: the bars are inclusive (mean 0.30, break 1.3 %, fraction 0.25 pass), the CI lower
    bound must be > 0 (0 fails), NaN never passes, no GT tube -> verdict None, token mismatch -> ValueError; the
    readout-rule ceiling and the all-key reference are descriptive."""
    def row(mean, lo, brk=0.0, n_brk=0, den=100):
        return {"d_vs_blend": {"mean": mean, "lo": lo, "hi": mean + 1}, "break_": brk, "break_n": n_brk,
                "break_den": den}
    d_l = np.full(8, 0.5)                                                   # sums 4 / 16: exactly 25 %
    d_g = np.full(8, 2.0)
    P = E.p1_decide(row(0.30, 0.01, 0.013, 13, 1000), d_l, row(2.0, 0.5), d_g)
    assert P["verdict"] == "PASS" and P["gain_fraction"] == 0.25 and P["gain_fraction_ci"] is None
    assert E.p1_decide(row(0.30, 0.0), d_l, row(2.0, 0.5), d_g)["verdict"] == "FAIL"           # CI lower bound 0
    assert E.p1_decide(row(0.2999, 0.1), d_l, row(2.0, 0.5), d_g)["verdict"] == "FAIL"
    assert E.p1_decide(row(0.5, 0.1, 0.0131), d_l, row(2.0, 0.5), d_g)["verdict"] == "FAIL"
    assert E.p1_decide(row(0.5, 0.1, float("nan")), d_l, row(2.0, 0.5), d_g)["verdict"] == "FAIL"  # no break den
    assert E.p1_decide(row(0.5, 0.1), np.full(8, 0.4999), row(2.0, 0.5), d_g)["verdict"] == "FAIL"  # 24.995 %
    assert E.p1_decide(row(0.5, 0.1), d_l, row(2.0, 0.0), d_g)["verdict"] == "VOID"            # GT CI touches 0
    assert E.p1_decide(row(0.5, 0.1), d_l, row(float("nan"), float("nan")), d_g)["verdict"] == "VOID"
    P = E.p1_decide(row(0.5, 0.1), d_l, None, None)
    assert P["verdict"] is None and "gttube" in P["reason"] and P["parts"]["a_mean_and_ci"]
    with pytest.raises(ValueError, match="different tokens"):
        E.p1_decide(row(0.5, 0.1), d_l, row(2.0, 0.5), d_g[:5])
    P = E.p1_decide(row(0.5, 0.1), d_l, row(2.0, 0.5), d_g, rule=row(2.5, 0.6), d_rule=np.full(8, 2.5))
    assert P["ceiling"]["gain_fraction_vs_ceiling"] == pytest.approx(0.2) and P["verdict"] == "PASS"
    # the deciding token set and the summary
    from types import SimpleNamespace as NS
    assert E.p1_deciding_set(NS(split="navtrain_val", subset="heldout", limit=0))
    for a in (NS(split="navtrain_val", subset="heldout", limit=2), NS(split="navtrain_val", subset="all", limit=0),
              NS(split="navtest", subset="all", limit=0)):
        assert not E.p1_deciding_set(a)
    S = {"criterion_S": {"blend_critic": "ck2T10,ck2M10", "P1": {"verdict": "PASS", "deciding": False}},
         "criterion_S_phase0b_convention": {"blend_critic": "ck2T10", "P1": {"verdict": "FAIL", "deciding": True}}}
    assert E.p1_summary(S) == {"verdict": "FAIL", "deciding": True, "blend_critic": "ck2T10"}
    S["criterion_S_phase0b_convention"]["P1"]["deciding"] = False
    assert E.p1_summary(S) is None
    assert E.DECISION_RULE and "FILTER" in E.DECISION_RULE and "VOID" in E.DECISION_RULE
    assert (E.P1_D_MIN, E.P1_BREAK_MAX, E.P1_FRAC_MIN, E.P1_KEYS, E.P1_BLEND) == (0.30, 0.013, 0.25, ("nc", "ttc"),
                                                                                   "ck2T10")


def test_p1_deciding_block_and_md(tmp_path, monkeypatch):
    """_s_all marks P1 deciding only on the val held-out set (no --limit) in the ck2T10 block; navtest blocks (pair
    and ck2T10 convention) and a --limit run are report only; the md prints the decision header and the P1 lines."""
    from types import SimpleNamespace as NS
    from navsim.agents.para_ssr.ck import select2
    lab, score, allowed, ins_l, ins_gt, logs = _p1_case(1)
    n = len(lab)
    np.save(tmp_path / "v2_final.npy", np.zeros((n, 16)))
    np.save(tmp_path / "v2_im.npy", np.zeros((n, 16)))
    monkeypatch.setattr(select2, "pool_scores", lambda v2f, v2i, p, beta: score)
    prob = lambda r: (np.zeros((len(r), 16, 5)), np.zeros((len(r), 80, 5)))
    tubes = {"learned": ins_l, "gt_r2_same_keys": ins_gt}
    names = {"learned": "learned", "gt": "gt_r2_same_keys"}
    r, lb = np.arange(n), E.LogBoot(logs)

    def run(split, subset, limit, blends):
        a = NS(split=split, subset=subset, limit=limit)
        C = {"pdir": tmp_path, "blends": [{"name": b, "prob": prob} for b in blends], "split": split,
             "subset": subset, "limit": limit}
        return E._s_all(C, r, lab, allowed, logs, lb, tubes, tubes, names, deciding_set=E.p1_deciding_set(a))[0]

    S = run("navtrain_val", "heldout", 0, ["ck2T10"])
    P = S["criterion_S"]["P1"]
    assert P["deciding"] and P["verdict"] == "PASS" and P["context"] == "decision (navtrain_val heldout, blend ck2T10)"
    assert E.p1_summary(S)["verdict"] == "PASS"
    S2 = run("navtest", "all", 0, ["ck2T10,ck2M10", "ck2T10"])
    assert not S2["criterion_S"]["P1"]["deciding"] and not S2["criterion_S_phase0b_convention"]["P1"]["deciding"]
    assert S2["criterion_S_phase0b_convention"]["P1"]["context"] == "report only (navtest all, blend ck2T10)"
    assert E.p1_summary(S2) is None
    S3 = run("navtrain_val", "heldout", 5, ["ck2T10"])
    assert not S3["criterion_S"]["P1"]["deciding"] and E.p1_summary(S3) is None
    S4 = run("navtrain_val", "heldout", 0, ["v4T10_haz", "ck2T10"])                # --blend-critic other: ck2T10 decides
    assert not S4["criterion_S"]["P1"]["deciding"] and S4["criterion_S_phase0b_convention"]["P1"]["deciding"]
    # the gate (stage eval: registered tokens / teacher) failed: report only with its reason; passed: the token note
    C = {"pdir": tmp_path, "blends": [{"name": "ck2T10", "prob": prob}], "split": "navtrain_val", "subset": "heldout",
         "limit": 0}
    S5 = E._s_all(C, r, lab, allowed, logs, lb, tubes, tubes, names, deciding_set=False,
                  gate_reason="token set 4 of 8 registered rows")[0]
    assert not S5["criterion_S"]["P1"]["deciding"] and E.p1_summary(S5) is None
    assert S5["criterion_S"]["P1"]["context"] == "report only (navtrain_val heldout, blend ck2T10): token set 4 of 8 " \
                                                 "registered rows"
    S6 = E._s_all(C, r, lab, allowed, logs, lb, tubes, tubes, names, deciding_set=True,
                  token_note=", tokens 400 / 400 registered")[0]
    assert S6["criterion_S"]["P1"]["context"] == "decision (navtrain_val heldout, blend ck2T10, tokens 400 / 400 " \
                                                 "registered)"
    # md: decision header + the P1 lines of the block
    res = {"stage": "eval", "split": "navtrain_val", "subset": "heldout", "finished": "t", "det": "x", "map": None,
           "keys": ["nc", "ttc"], "n_tokens": n, "n_logs": lb.L, "blend_critic": "ck2T10", "gt_tube": "g",
           "decision_rule": E.DECISION_RULE, "caveats": [], "phase0b_check": {}, "fidelity": {}, **S,
           "decision": {"verdict": "PASS", "deciding": True, "blend_critic": "ck2T10"},
           "selection": {"rule": "DET only", "v2_top1_pdms": 0.0, "range": {}, "vs": {}}}
    md = E.report_md(res)
    assert "**DECISION (P1, amended): PASS** (navtrain_val heldout, blend ck2T10" in md
    assert "**P1 (amended, filter rule) -- PASS** (decision (navtrain_val heldout, blend ck2T10))" in md
    assert "(c) fraction of the GT R2 filter gain (gt_r2_same_keys, same keys and tokens): 1.000" in md


def _ns(split="navtrain_val", subset="heldout", limit=0):
    from types import SimpleNamespace as NS
    return NS(split=split, subset=subset, limit=limit)


def _good_teacher(tmp_path, name=None, **over):
    """teacher_identity of infer_meta of the registered teacher (fake checkpoint file); over: fields to change
    (... = drop the field)."""
    ck = tmp_path / "ckpt_last.pt"
    if not ck.is_file():
        ck.write_bytes(b"fake checkpoint")
    meta = {k: v for k, v in E.P1_TEACHER.items() if k != "name"}
    meta.update(ckpt=str(ck), ckpt_sha16=E.U.sha256_file(ck), n_done=8, n_rows=8, status="finished")
    meta.update(over)
    return E.teacher_identity(name or E.P1_TEACHER["name"], {k: v for k, v in meta.items() if v is not Ellipsis})


def test_p1_gate_tokens_and_teacher(tmp_path, monkeypatch):
    """P1 decides only when the evaluated rows are the registered rows (HO/prep/ho_rows.npy: a row removed by a missing
    teacher output / baseline / GT tube is counted per input) AND the DET outputs are the registered teacher's
    (v4T10_rng2, arm T, --which last, rg_read any4, rg_stopped_aware True) at an unchanged checkpoint."""
    ho = tmp_path / "HO"
    monkeypatch.setattr(E, "HO_DIR", ho)
    reg, why = E.p1_registered_rows()
    assert reg is None and "unavailable" in why
    (ho / "prep").mkdir(parents=True)
    np.save(ho / "prep" / "ho_rows.npy", np.array([9, 3, 5, 7, 11, 13, 15, 17]))
    reg, why = E.p1_registered_rows()
    assert why is None and reg.tolist() == [3, 5, 7, 9, 11, 13, 15, 17]
    (ho / "results.json").write_text(json.dumps({"n_tokens_used": 7}))
    assert "used 7 of its 8" in E.p1_registered_rows()[1]
    (ho / "results.json").write_text(json.dumps({"n_tokens_used": 8}))
    assert E.p1_registered_rows()[1] is None
    # token gate: half the teacher outputs missing; a baseline missing one more row
    N = 20
    tdone = np.zeros(N, bool)
    tdone[[3, 5, 7, 9]] = True
    base = np.ones(N, bool)
    base[3] = False
    filters = {"labels": np.ones(N, bool), "teacher det outputs (done & BEV)": tdone, "baseline b det": base}
    r = np.flatnonzero(tdone & base)
    ts = E.p1_token_gate(r, reg, filters)
    assert (ts["n"], ts["n_registered"], ts["n_missing"], ts["n_extra"], ts["complete"]) == (3, 8, 5, 0, False)
    assert ts["missing_by_input"] == {"teacher det outputs (done & BEV)": 4, "baseline b det": 1}
    assert "3 of 8 registered" in E._token_reason(ts) and "baseline b det 1" in E._token_reason(ts)
    ts = E.p1_token_gate(np.r_[reg, 19], reg, {})
    assert not ts["complete"] and ts["n_extra"] == 1                       # an extra row is not the registered set
    assert E.p1_token_gate(reg[::-1], reg, {})["complete"]
    # teacher identity
    good = _good_teacher(tmp_path)
    assert good["ckpt_current"] is True and E.p1_teacher_problems(good) == []
    for over, msg in (({"rg_read": "center"}, "rg_read 'center'"), ({"rg_read": ...}, "rg_read None"),
                      ({"which": "best"}, "which 'best'"), ({"rg_stopped_aware": False}, "rg_stopped_aware False"),
                      ({"rg_stopped_aware": 1}, "rg_stopped_aware 1"), ({"arm": "M"}, "arm 'M'"),
                      ({"name": "v4T10_rng"}, "name 'v4T10_rng'"), ({"ckpt_sha16": "0" * 16}, "changed since infer"),
                      ({"ckpt_sha16": ...}, "no ckpt_sha16"), ({"ckpt": str(tmp_path / "gone.pt")}, "missing")):
        bad = E.p1_teacher_problems(_good_teacher(tmp_path, **over))
        assert len(bad) == 1 and msg in bad[0], (over, bad)
    assert "no infer_meta.json" in E.p1_teacher_problems(E.teacher_identity("v4T10_rng2", {}))[0]
    assert E.p1_teacher_problems(None) == ["no DET teacher"]
    # the gate
    full = np.ones(N, bool)
    g = E.p1_gate(_ns(), reg, {"labels": full}, good)
    assert g["deciding"] and g["reason"] is None and g["token_set"]["complete"] and not g["incomplete"]
    assert E._token_note(g) == ", tokens 8 / 8 registered"
    g = E.p1_gate(_ns(), r, filters, good)
    assert not g["deciding"] and g["incomplete"] and "3 of 8" in g["reason"]
    g = E.p1_gate(_ns(), reg, {"labels": full}, _good_teacher(tmp_path, which="best"))
    assert not g["deciding"] and not g["incomplete"] and "registered teacher" in g["reason"]
    for a in (_ns(limit=4), _ns(split="navtest", subset="all")):
        g = E.p1_gate(a, reg, {"labels": full}, good)
        assert not g["deciding"] and not g["registered_split"] and not g["incomplete"] and g["token_set"] is None
    g = E.p1_gate(_ns(), reg, {"labels": full}, good, have_gt=False)       # no GT reference: incomplete
    assert not g["deciding"] and g["incomplete"] and "no GT R2 tube" in g["reason"]
    g = E.p1_gate(_ns(), reg, {"labels": full}, need_teacher=False)       # gtcheck: no teacher
    assert g["deciding"] and g["teacher_problems"] == []
    (ho / "prep" / "ho_rows.npy").unlink()
    g = E.p1_gate(_ns(), reg, {"labels": full}, good)
    assert not g["deciding"] and g["incomplete"] and "unavailable" in g["reason"]


def test_infer_resume_check(tmp_path):
    """stage infer resumes only into outputs of the same teacher / checkpoint."""
    td = tmp_path / "t"
    td.mkdir()
    meta = {"teacher": "t", "role": "det", "arm": "T", "which": "last", "ckpt_sha16": "a" * 16, "rg_read": "any4",
            "rg_stopped_aware": True, "corridor_version": 2, "split": "navtest", "sessions": [{"n": 3}]}
    assert E.infer_resume_check(td, meta) == {}                            # fresh dir
    done = np.zeros(10, bool)
    np.save(td / "done.npy", done)
    assert E.infer_resume_check(td, meta) == {}                            # no row done yet, no meta
    done[2] = True
    np.save(td / "done.npy", done)
    with pytest.raises(SystemExit, match="no infer_meta.json"):
        E.infer_resume_check(td, meta)
    (td / "infer_meta.json").write_text(json.dumps(meta))
    assert E.infer_resume_check(td, dict(meta, sessions=None, batch=4))["sessions"] == [{"n": 3}]   # same ckpt: resume
    for k, v in (("ckpt_sha16", "b" * 16), ("which", "best"), ("rg_read", "center"), ("rg_stopped_aware", False)):
        with pytest.raises(SystemExit, match=f"resume refused.*{k}"):
            E.infer_resume_check(td, dict(meta, **{k: v}))
    np.save(td / "done.npy", np.zeros(10, bool))                            # nothing done: a new teacher may start
    assert E.infer_resume_check(td, dict(meta, ckpt_sha16="b" * 16)) == {}


def test_recall_fidelity_and_logboot():
    rng = np.random.default_rng(1)
    n = 40
    fail = rng.random((n, 96, 3)) < 0.2
    out = fail.copy()
    out[:5] = ~out[:5]                                                      # 5 tokens wrong everywhere
    valid = np.ones((n, 96), bool)
    logs = np.array([f"log{i % 7}" for i in range(n)])
    lb = E.LogBoot(logs)
    valid[:, 90:] = False                                                    # excluded rows
    f = E.fidelity(out, fail, valid, ("nc", "ttc", "dac"), lb)
    for k, q in (("nc", 0), ("ttc", 1), ("dac", 2)):
        x, fk, ok = f[k], fail[..., q], out[..., q]
        assert x["fail_recall"] == pytest.approx((ok & fk & valid).sum() / (fk & valid).sum())
        assert x["pass_recall"] == pytest.approx((~ok & ~fk & valid).sum() / (~fk & valid).sum())
        assert x["fail_recall"] == pytest.approx((fk[5:] & valid[5:]).sum() / (fk & valid).sum())
        assert x["fail_recall_ci"][0] <= x["fail_recall"] <= x["fail_recall_ci"][1]
        assert x["pass_recall_ci"][0] <= x["pass_recall"] <= x["pass_recall_ci"][1]
    fj, oj = fail.any(-1), out.any(-1)
    assert f["joint"]["n_fail"] == int((fj & valid).sum())
    assert f["joint"]["fail_recall"] == pytest.approx((oj & fj & valid).sum() / (fj & valid).sum())
    g = E.fidelity(out, fail, valid, ("nc",), None)
    assert set(g) == {"nc", "joint", "keys"} and g["joint"]["n_fail"] == int((fail[..., 0] & valid).sum())


def test_critic_prob_rules():
    from tools.ck.v4.eval_teacher_pair_navtest import pair_prob
    rng = np.random.default_rng(2)
    pd_, pm = rng.random((3, 16, 5)), rng.random((3, 16, 5))
    assert np.array_equal(E.critic_prob(pd_, pm), pair_prob(pd_, pm))
    d = E.critic_prob(pd_, None)
    m = E.critic_prob(None, pm)
    i = {k: CK_KEYS.index(k) for k in CK_KEYS}
    assert (d[..., i["dac"]] == 1).all() and np.array_equal(d[..., i["nc"]], pd_[..., i["nc"]])
    assert np.array_equal(d[..., i["ep"]], pd_[..., i["ep"]])
    assert (m[..., [i["nc"], i["ttc"], i["comfort"]]] == 1).all() and np.array_equal(m[..., i["dac"]], pm[..., i["dac"]])


# ----------------------------------------------------------------------------------------------- GT tube vs oracle
@have_f1
def test_gt_tube_equals_oracle_grid_eval_on_f1_tokens():
    """r2_fields + gt_tube_flags == oracle_common.grid_eval(read 'any4', TTC LON, DAC on D) with B01 :=
    stopped_aware_B01, bitwise on the F1 official executed states of 6 tokens (96 candidates each); executed_states
    (LQR float64 from the packed status) within 1e-4 m of the F1 official states; token_tube runs on the same token."""
    sys.path.insert(0, str(REPO / "tools/ck/v4/oracle"))
    import oracle_common as OC
    D = OC.Data()
    st = np.load(OC.PACK / "status.npy", mmap_mode="r")
    traj80 = np.load(OC.VAR / "traj80.npy", mmap_mode="r")
    n_out = 0
    agree, n_valid = np.zeros(3), 0
    for i in range(6):
        T = OC.token_inputs(D, i)
        if T["gt"] is None:
            continue
        X = T["X"][:96]
        idx = OC.parent_of(np.arange(96))
        F = OC.parent_fields(T["par"], T["gt"], T["sdf"], point=False)
        F["B01"] = OC.stopped_aware_B01(F["frn"], T["gt"])
        o = OC.grid_eval(X, F, idx, tb=0.25, read="any4", ttc_lon=True, dac="grid")
        want = np.stack([o["hit_nc"], o["hit_ttc"], o["hit_dac"]], -1)
        gt = E.load_gt_store(OC.GT_OBJ, T["token"])
        got = E.gt_tube_flags(X, E.r2_fields(T["par"], gt, T["sdf"]), idx)
        assert np.array_equal(got, want), i
        n_out += int(want.sum())
        trajs = np.concatenate([T["par32"], np.asarray(traj80[T["row"]], np.float32)], 0)
        Xm = E.executed_states(trajs, st[T["row"]])
        ok = T["valid96"]
        assert float(np.abs(Xm[ok, :, :3] - X[ok, :, :3]).max()) < 1e-4
        h16, h80 = E.token_tube(T["par32"], traj80[T["row"]], st[T["row"]], gt, T["sdf"])
        assert h16.shape == (16, 3) and h80.shape == (80, 3)
        # the readout-rule tube (float32 teacher states and frames, TTC on Bp / U) next to the R2 tube
        o = E.token_tubes(T["par32"], traj80[T["row"]], st[T["row"]], gt, T["sdf"])
        assert np.array_equal(o["hit16"], h16) and np.array_equal(o["hit80"], h80)
        assert o["rule16"].shape == (16, 3) and o["rule80"].shape == (80, 3)
        ok16 = np.concatenate([np.ones(16, bool), np.asarray(np.load(OC.VAR / "valid80.npy", mmap_mode="r")[T["row"]])])
        h = np.concatenate([h16, h80], 0)[ok16]
        q = np.concatenate([o["rule16"], o["rule80"]], 0)[ok16]
        agree += (h == q).sum(0)
        n_valid += int(ok16.sum())
    assert n_out > 0
    # NC / DAC: the same reductions; TTC: the 2 m switch (0.80 % of the rows over 1,500 F1 tokens; these 6: 2.8 %)
    rate = agree / n_valid
    assert rate[0] >= 0.99 and rate[2] >= 0.99 and rate[1] >= 0.95, rate


# ----------------------------------------------------------------------------------------------- guards / checks
FAKE_SMI = """#!/bin/bash
[ "${FAKE_SMI_FAIL:-0}" = 1 ] && exit 9
echo "${FAKE_SMI_MEM:-0}"
"""


def test_cuda_guard_refusals(tmp_path, monkeypatch):
    """stage infer on CUDA refuses without approval, with a CUDA_VISIBLE_DEVICES outside 0-3 / several GPUs, on a busy
    GPU (>= 1024 MiB) and when nvidia-smi fails -- all before CUDA is touched (fake nvidia-smi)."""
    from types import SimpleNamespace
    smi = tmp_path / "smi.sh"
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    monkeypatch.setenv("V4_NVIDIA_SMI", str(smi))
    a = SimpleNamespace(device="cuda", approved=False)
    monkeypatch.delenv("V4_USER_APPROVED", raising=False)
    with pytest.raises(SystemExit, match="approval"):
        E.cuda_guard(a)
    monkeypatch.setenv("V4_USER_APPROVED", "1")
    for vis in ("", "6", "2,3"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", vis)
        with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
            E.cuda_guard(a)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("FAKE_SMI_MEM", "5000")
    with pytest.raises(SystemExit, match="5000 MiB"):
        E.cuda_guard(a)
    monkeypatch.setenv("FAKE_SMI_FAIL", "1")
    with pytest.raises(SystemExit, match="query of GPU 2 failed"):
        E.cuda_guard(SimpleNamespace(device="cuda", approved=True))
    monkeypatch.setenv("FAKE_SMI_FAIL", "0")
    monkeypatch.setenv("FAKE_SMI_MEM", "12")
    monkeypatch.delenv("V4_USER_APPROVED")
    E.cuda_guard(SimpleNamespace(device="cuda", approved=True))            # --approved, idle: passes (no CUDA call)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_phase0b_check_logic(tmp_path, monkeypatch):
    """comparable only on the Phase 0b token set (navtrain_val heldout, no --limit, the same rows, all of Phase 0b's
    rows used); match = the same mean dPDMS (1e-9) and break counts."""
    from types import SimpleNamespace
    ho = tmp_path / "HO"
    (ho / "prep").mkdir(parents=True)
    rows = np.array([3, 5, 9])
    np.save(ho / "prep" / "ho_rows.npy", rows)
    cs = {"d_vs_blend": {"mean": 1.5, "lo": 1.0, "hi": 2.0}, "break_": {"n": 2, "den": 40}, "verdict": True}
    (ho / "results.json").write_text(json.dumps({"n_tokens_used": 3, "selection": {"criterion_S": cs}}))
    monkeypatch.setattr(E, "HO_DIR", ho)
    a = SimpleNamespace(split="navtrain_val", subset="heldout", limit=0)
    row = {"d_vs_blend": {"mean": 1.5, "lo": 1.0, "hi": 2.0}, "break_n": 2, "break_den": 40}
    c = E.phase0b_check(a, rows, row)
    assert c["comparable"] and c["match"]
    assert not E.phase0b_check(a, rows, dict(row, break_n=3))["match"]
    assert not E.phase0b_check(a, rows, dict(row, d_vs_blend={"mean": 1.5001, "lo": 1.0, "hi": 2.0}))["match"]
    assert not E.phase0b_check(a, rows[:2], row)["comparable"]
    assert not E.phase0b_check(SimpleNamespace(split="navtrain_val", subset="heldout", limit=2), rows, row)["comparable"]
    assert not E.phase0b_check(SimpleNamespace(split="navtest", subset="all", limit=0), rows, row)["comparable"]
    with pytest.raises(SystemExit, match="Phase 0b reproduction FAILED"):
        E._raise_on_mismatch({"phase0b_check": E.phase0b_check(a, rows, dict(row, break_n=3))})
    E._raise_on_mismatch({"phase0b_check": c})
    # the filter row vs HO/results.json selection.picks['R2 tube filter + blend'] (absent -> not comparable)
    frow = {"d_vs_blend": {"mean": 2.75, "lo": 2.3, "hi": 3.2}, "break_n": 7, "break_den": 40}
    c = E.phase0b_check(a, rows, row, frow)
    assert c["match"] and not c["filter"]["comparable"] and "picks" in c["filter"]["reason"]
    pk = {"d_vs_blend": {"mean": 2.75, "lo": 2.3, "hi": 3.2}, "break_vs_blend": {"n": 7, "den": 40}}
    (ho / "results.json").write_text(json.dumps({"n_tokens_used": 3, "selection": {
        "criterion_S": cs, "picks": {E.PHASE0B_FILTER_PICK: pk}}}))
    c = E.phase0b_check(a, rows, row, frow)
    assert c["match"] and c["filter"]["comparable"] and c["filter"]["match"]
    E._raise_on_mismatch({"phase0b_check": c})
    bad = E.phase0b_check(a, rows, row, dict(frow, break_n=8))
    assert bad["match"] and bad["filter"]["comparable"] and not bad["filter"]["match"]
    with pytest.raises(SystemExit, match="FAILED for the filter rule"):
        E._raise_on_mismatch({"phase0b_check": bad})
    nc = E.phase0b_check(SimpleNamespace(split="navtest", subset="all", limit=0), rows, row, frow)
    assert not nc["comparable"] and not nc["filter"]["comparable"]
    assert "filter" not in E.phase0b_check(a, rows, row)                    # no filter row: the old dict


# ----------------------------------------------------------------------------------------------- CPU smoke
def _range_run(root: Path, arm: str, v4: V4Cfg, name: str) -> Path:
    src = TL.SMOKE_T if arm == "T" else TL.SMOKE_M
    rd = Path(root) / name
    rd.mkdir(parents=True, exist_ok=True)
    nf = "norm.npz" if arm == "T" else "norm_map.npz"
    shutil.copy(src / nf, rd / nf)
    torch.manual_seed(5)
    net = CKNet(arm, seed=1 if arm == "T" else 2, norm=load_arm_norm(rd, arm), v4=v4, corridor_version=2)
    net.set_score_prior([0.97, 0.97, 0.9, 0.95, 0.99])                  # a plausible pass rate for the smoke
    with torch.no_grad():
        net.trunk.lon_head[-1].weight.zero_()
        net.trunk.lon_head[-1].bias.zero_()
    cfg = {"arm": arm, "seed": 1 if arm == "T" else 2, "trainer": "train_ck2", "v4": v4.to_dict(), "corridor_version": 2,
           "score_hidden": 256, "lead_aux": 0}
    save_ck(rd / "ckpt_last.pt", net, cfg)
    (rd / "config.json").write_text(json.dumps(cfg))
    (rd / "done.json").write_text("{}")
    return rd


@pytest.mark.skipif(not _navtest_inputs(), reason="navtest eval inputs / smoke teachers absent")
def test_cpu_smoke_navtest_all_stages(tmp_path):
    det, mp = _range_run(tmp_path, "T", R2, "rng2T"), _range_run(tmp_path, "M", R2, "rng2M")
    out = tmp_path / "out"
    base = ["--split", "navtest", "--limit", "3", "--device", "cpu", "--out-root", str(out), "--batch", "2",
            "--workers-io", "0", "--workers", "2"]
    res = E.main(base + ["--det", str(det), "--map", str(mp)])
    root = out / "smoke3" / "navtest"
    rows = np.flatnonzero(np.load(CK / "packed/navtest/ok.npy"))[:3]
    for t in ("rng2T", "rng2M"):
        assert bool(np.load(root / t / "done.npy")[rows].all())
        meta = json.loads((root / t / "infer_meta.json").read_text())
        assert meta["rg_read"] == "any4" and meta["rg_stopped_aware"] is True
    sl = np.load(root / "rng2T" / "slack16.npy")[rows]
    assert np.isfinite(sl).all() and np.isfinite(np.load(root / "rng2T" / "score_logit80.npy")[rows]).all()
    # the stored GT tube = token_tube recomputed for one token
    from navsim.agents.para_ssr.refiner.sdf import load_sdf
    import pandas as pd
    tok = pd.read_parquet(CK / "packed/navtest/tokens.parquet").token.astype(str).to_numpy()
    r0 = int(rows[0])
    sp = E.SPLITS["navtest"]
    h16, h80 = E.token_tube(np.load(CK / "packed/navtest/cand.npy", mmap_mode="r")[r0, :16],
                            np.load(sp["variants"] / "traj80.npy", mmap_mode="r")[r0],
                            np.load(CK / "packed/navtest/status.npy", mmap_mode="r")[r0],
                            E.load_gt_store(sp["gt_obj"], tok[r0]),
                            load_sdf(sp["sdf"] / f"{tok[r0]}.npz").astype(np.float64))
    assert np.array_equal(np.load(root / "gt_tube_r2" / "hit16.npy")[r0], h16)
    assert np.array_equal(np.load(root / "gt_tube_r2" / "hit80.npy")[r0], h80)
    for k in ("rule16", "rule80"):                                           # the readout-rule tube is stored too
        assert np.load(root / "gt_tube_r2" / f"{k}.npy").shape[1:] == ((16, 3) if k == "rule16" else (80, 3))
    meta = json.loads((root / "gt_tube_r2" / "meta.json").read_text())
    assert meta["representation_id"] == "R2" and meta["split"] == "navtest" and meta["format"] == 2
    ev = res["eval"]
    assert ev["n_tokens"] == 3 and ev["keys"] == ["nc", "ttc", "dac"]
    assert set(ev["fidelity"]) >= {"learned", "gt_r2", "gt_rule", "agreement_learned_gt_r2", "gt_r2_all_keys"}
    S = ev["criterion_S"]
    assert set(S) >= {"learned", "gt_r2_same_keys", "gt_r2_all_keys", "gt_rule_same_keys", "gain_fraction",
                      "gain_fraction_vs_rule"}
    assert S["blend_critic"] == "ck2T10,ck2M10"                              # navtest: the deployed pair
    conv = ev["criterion_S_phase0b_convention"]
    assert conv["blend_critic"] == "ck2T10" and "Phase 0b convention" in conv["label"]
    assert ev["phase0b_check"]["comparable"] is False and ev["decision_rule"] == E.DECISION_RULE
    # the amended P1 (filter rule, DET keys only next to a MAP): report only on navtest, for both blend critics
    assert set(S["filter"]) >= {"learned", "learned_nc_ttc", "gt_r2_nc_ttc", "gt_rule_nc_ttc", "gt_r2_same_keys",
                                "gt_r2_all_keys", "gain_fraction", "gain_fraction_vs_rule"}
    for blk in (S, conv):
        P = blk["P1"]
        assert (P["learned_tube"], P["gt_tube"], P["ceiling_tube"]) == ("learned_nc_ttc", "gt_r2_nc_ttc",
                                                                        "gt_rule_nc_ttc")
        assert P["deciding"] is False and P["context"].startswith("report only") and P["verdict"] in (
            "PASS", "FAIL", "VOID")
        assert P["d_learned"] == blk["filter"]["learned_nc_ttc"]["d_vs_blend"]
        assert P["gt_reference"]["d_vs_blend"] == blk["filter"]["gt_r2_nc_ttc"]["d_vs_blend"]
    assert ev["P1"] is None and ev["decision"]["deciding"] is False and ev["decision"]["verdict"] is None
    assert set(ev["selection"]["vs"]) == {"ck2T10,ck2M10", "v4T10_haz,v4M10_haz"}
    assert "pool96_ck_noim" in ev["selection"]["range"]
    tag = "rng2T+rng2M"
    assert (root / f"eval_{tag}.md").is_file() and (root / f"eval_{tag}_picks.npz").is_file()
    md = (root / f"eval_{tag}.md").read_text()
    assert "decision rule (registered)" in md and "Phase 0b convention" in md and "gt_rule_same_keys" in md
    assert "P1 not deciding here" in md and "FILTER rule" in md and "Old SWITCH rule" in md
    # the md header says what the outputs are about; the identity is checked against the checkpoint
    ti = ev["teacher_identity"]["det"]
    assert (ti["which"], ti["rg_read"], ti["rg_stopped_aware"], ti["ckpt_current"]) == ("last", "any4", True, True)
    assert f"Teacher det rng2T: --which last, arm T, rg_read any4, rg_stopped_aware True, ckpt_sha16 " \
           f"{ti['ckpt_sha16']} (checkpoint unchanged)" in md and "Teacher map rng2M" in md
    assert ev["p1_gate"]["registered_split"] is False and any("name 'rng2T'" in x for x in
                                                              ev["p1_gate"]["teacher_problems"])
    # descriptive per-key filter rows and their share of P1's GT gain
    assert {"learned_nc_only", "learned_ttc_only", "gt_r2_nc_only", "gt_r2_ttc_only"} <= set(S["filter"])
    assert S["filter"]["key_share"]["vs"] == "gt_r2_nc_ttc" and "Per-key share" in md
    j = json.loads((root / f"eval_{tag}.json").read_text())
    assert j["n_tokens"] == 3
    # gtcheck: GT rows only, no teacher; --tag names new report files
    g = E.main(base + ["--stage", "gtcheck", "--tag", "t1"])["gtcheck"]
    assert g["n_tokens"] == 3 and set(g["criterion_S"]) >= {"gt_r2_all_keys", "gt_rule_all_keys", "gt_r2_nc_ttc"}
    assert g["criterion_S"]["gt_r2_all_keys"] == S["gt_r2_all_keys"]         # the same row as in eval
    assert g["criterion_S"]["filter"]["gt_r2_nc_ttc"] == S["filter"]["gt_r2_nc_ttc"]
    assert g["criterion_S"]["filter"]["gt_r2_all_keys"] == S["filter"]["gt_r2_all_keys"]
    R = g["criterion_S"]["P1_reference"]
    assert R["tube"] == "gt_r2_nc_ttc" and R["deciding"] is False and R["gt_reference_all_keys"]["tube"] == \
        "gt_r2_all_keys"
    assert R["d_vs_blend"] == S["P1"]["gt_reference"]["d_vs_blend"]
    assert {"gt_r2_nc_only", "gt_r2_ttc_only"} <= set(g["criterion_S"]["filter"]) and not g["p1_gate"]["deciding"]
    assert g["criterion_S"]["filter"]["gt_r2_nc_only"] == S["filter"]["gt_r2_nc_only"]
    assert (root / "gtcheck_t1.md").is_file() and (root / "gtcheck_t1.json").is_file()
    assert not (root / "gtcheck.json").exists()
    E.main(base + ["--stage", "gtcheck"])
    assert (root / "gtcheck.md").is_file()
    with pytest.raises(SystemExit, match="--tag"):
        E.main(base + ["--stage", "gtcheck", "--tag", "../x"])
    # gttube writes only its own dir; eval reads another dir read-only after the meta / shape checks
    with pytest.raises(SystemExit, match="writes only its own"):
        E.main(base + ["--stage", "gttube", "--gt-tube", str(root / "gt_tube_r2")])
    ext = tmp_path / "ext_tube"
    shutil.copytree(root / "gt_tube_r2", ext)
    e2 = E.main(base + ["--det", str(det), "--map", str(mp), "--stage", "eval", "--gt-tube", str(ext)])["eval"]
    assert e2["gt_tube"] == str(ext) and e2["criterion_S"]["gt_r2_all_keys"] == S["gt_r2_all_keys"]
    t = E.load_gt_tube(E.get_parser().parse_args(base + ["--gt-tube", str(ext)]), len(np.load(CK / "packed/navtest/ok.npy")))
    assert not t["hit16"].flags.writeable                                   # opened read-only
    for bad_meta, msg in (({"split": "navtrain_val", "representation_id": "R2"}, "meta.json must name"),
                          ({"split": "navtest", "representation_id": "R1"}, "meta.json must name")):
        (ext / "meta.json").write_text(json.dumps(bad_meta))
        with pytest.raises(SystemExit, match=msg):
            E.main(base + ["--det", str(det), "--stage", "eval", "--gt-tube", str(ext)])
    (ext / "meta.json").write_text(json.dumps({"split": "navtest", "representation_id": "R2"}))
    np.save(ext / "hit16.npy", np.zeros((5, 16, 3), bool))                   # not in packed-row order
    with pytest.raises(SystemExit, match="packed-row"):
        E.main(base + ["--det", str(det), "--stage", "eval", "--gt-tube", str(ext)])
    # resumable: a second infer call has nothing to do
    m2 = E.main(base + ["--det", str(det), "--stage", "infer"])["infer_det"]
    assert m2["sessions"][-1]["n"] == 0 and len(m2["sessions"]) == 2 and m2["status"] == "finished"
    # ... but only into outputs of the same checkpoint (infer_meta.json ckpt_sha16 / which ...)
    mp_ = root / "rng2T" / "infer_meta.json"
    good = mp_.read_text()
    mp_.write_text(json.dumps(dict(json.loads(good), ckpt_sha16="0" * 16)))
    with pytest.raises(SystemExit, match="resume refused.*ckpt_sha16"):
        E.main(base + ["--det", str(det), "--stage", "infer"])
    mp_.unlink()
    with pytest.raises(SystemExit, match="no infer_meta.json"):
        E.main(base + ["--det", str(det), "--stage", "infer"])
    mp_.write_text(good)
    # DET only: keys nc / ttc, the DET-only rule, baselines reduced to their DET member
    e1 = E.main(base + ["--det", str(det), "--stage", "eval"])["eval"]
    assert e1["keys"] == ["nc", "ttc"] and e1["selection"]["rule"].startswith("DET only")
    P = e1["criterion_S"]["P1"]
    assert (P["learned_tube"], P["gt_tube"]) == ("learned", "gt_r2_same_keys") and "learned_nc_ttc" not in \
        e1["criterion_S"]["filter"]
    # MAP only: no DET tube, P1 not evaluable
    e2 = E.main(base + ["--map", str(mp), "--stage", "eval"])["eval"]
    assert "P1" not in e2["criterion_S"] and e2["decision"]["reason"] == "no DET teacher"
    assert any("P1" in c for c in e2["caveats"])


@pytest.mark.skipif(not _val_inputs(), reason="navtrain_val eval inputs / smoke teachers absent")
def test_cpu_smoke_val_det_only_and_refusals(tmp_path):
    det = _range_run(tmp_path, "T", R2, "rng2T")
    out = tmp_path / "out"
    base = ["--split", "navtrain_val", "--limit", "2", "--device", "cpu", "--out-root", str(out), "--batch", "2",
            "--workers-io", "0", "--workers", "1"]
    with pytest.raises(SystemExit, match="ReSMap"):
        E.main(base + ["--det", str(det), "--map", str(det)])
    with pytest.raises(SystemExit, match="not a range teacher"):
        E.main(base + ["--det", str(TL.SMOKE_T), "--stage", "infer"])
    res = E.main(base + ["--det", str(det)])
    ev = res["eval"]
    assert ev["subset"] == "heldout" and ev["keys"] == ["nc", "ttc"] and ev["n_tokens"] >= 1
    f1 = set(np.load(E.F1_ROWS).tolist())
    picks = np.load(out / "smoke2" / "navtrain_val" / "eval_rng2T_picks.npz")
    assert not (set(picks["rows"].tolist()) & f1)                          # held-out tokens only
    assert any("in-sample" in c for c in ev["caveats"]) and list(ev["selection"]["vs"]) == ["ck2T10"]
    # the P1 block on val (ck2T10 blend, DET keys): a --limit smoke is not the deciding token set -> report only
    P = ev["criterion_S"]["P1"]
    assert ev["criterion_S"]["blend_critic"] == "ck2T10" and (P["learned_tube"], P["gt_tube"]) == (
        "learned", "gt_r2_same_keys")
    assert P["deciding"] is False and "limit" in P["context"] and ev["P1"] is None
    assert "criterion_S_phase0b_convention" not in ev


GT_VAL = E.OUT_ROOT / "navtrain_val" / "gt_tube_r2"


def _fake_det_outputs(root: Path, name: str, rows: np.ndarray, N: int, meta: dict) -> Path:
    """stage infer outputs of a fake DET range teacher on the given packed rows: slacks = the GT R2 tube's flags
    (learned tube = the GT tube), score logits 0."""
    td = root / "navtrain_val" / name
    td.mkdir(parents=True, exist_ok=True)
    done = np.zeros(N, bool)
    done[rows] = True
    np.save(td / "done.npy", done)
    np.save(td / "bev_ok.npy", done)
    for k, K in (("16", 16), ("80", 80)):
        hit = np.asarray(np.load(GT_VAL / f"hit{k}.npy", mmap_mode="r"))
        np.save(td / f"slack{k}.npy", np.where(hit, -1.0, 1.0).astype(np.float32))
        np.save(td / f"score_logit{k}.npy", np.zeros((N, K, 5), np.float32))
    (td / "infer_meta.json").write_text(json.dumps(meta))
    return td


@pytest.mark.skipif(not (_val_inputs() and (GT_VAL / "rule16.npy").is_file()),
                    reason="navtrain_val eval inputs / the val GT tube absent")
def test_p1_gate_stage_eval_registered_split(tmp_path, monkeypatch):
    """stage eval on --split navtrain_val --subset heldout without --limit (the deciding split), with the registered
    token set monkeypatched to 8 real held-out rows (HO/prep/ho_rows.npy) and a fake v4T10_rng2 DET output dir (learned
    tube = the GT tube, the real val GT tube read-only via --gt-tube): a half-done teacher -> report only + per-input
    counts, non-zero exit after the outputs are written; complete + registered teacher -> DECISION (tokens 8 / 8);
    no GT tube -> non-zero exit; infer_meta without rg_read, or a changed checkpoint -> report only (exit 0)."""
    out = tmp_path / "out"
    base = ["--split", "navtrain_val", "--out-root", str(out), "--gt-tube", str(GT_VAL), "--stage", "eval"]
    a = E.get_parser().parse_args(base)
    a.subset = "heldout"
    C = E._common(a, need_teachers=False)
    rows = np.flatnonzero(C["keep"])[:8]
    N = int(C["N"])
    assert len(rows) == 8
    ho = tmp_path / "HO"
    (ho / "prep").mkdir(parents=True)
    np.save(ho / "prep" / "ho_rows.npy", rows)
    monkeypatch.setattr(E, "HO_DIR", ho)
    ck = tmp_path / "ckpt_last.pt"
    ck.write_bytes(b"fake v4T10_rng2 checkpoint")
    meta = {"role": "det", "teacher": "v4T10_rng2", "arm": "T", "which": "last", "rg_read": "any4",
            "rg_stopped_aware": True, "corridor_version": 2, "split": "navtrain_val", "ckpt": str(ck),
            "ckpt_sha16": E.U.sha256_file(ck), "n_done": 8, "n_rows": 12179, "status": "finished"}
    root = out / "navtrain_val"
    ev = lambda tag: json.loads((root / f"eval_v4T10_rng2_{tag}.json").read_text())
    md = lambda tag: (root / f"eval_v4T10_rng2_{tag}.md").read_text()
    # half done: report only, non-zero exit, the outputs written
    _fake_det_outputs(out, "v4T10_rng2", rows[:4], N, meta)
    with pytest.raises(SystemExit, match="deciding split is incomplete.*4 of 8"):
        E.main(base + ["--det", "v4T10_rng2", "--tag", "half"])
    j = ev("half")
    assert j["n_tokens"] == 4 and j["P1"] is None and j["decision"]["deciding"] is False and j["decision"]["incomplete"]
    assert j["decision"]["token_set"]["missing_by_input"] == {"teacher det outputs (done & BEV)": 4}
    P = j["criterion_S"]["P1"]
    assert P["deciding"] is False and "report only" in P["context"] and "token set 4 of 8" in P["context"]
    assert "**P1 not deciding here**" in md("half") and "**DECISION" not in md("half")
    # complete, the registered teacher: deciding
    _fake_det_outputs(out, "v4T10_rng2", rows, N, meta)
    j = E.main(base + ["--det", "v4T10_rng2", "--tag", "full"])["eval"]
    assert j["n_tokens"] == 8 and j["decision"]["deciding"] is True and j["P1"]["deciding"]
    assert j["decision"]["verdict"] in ("PASS", "FAIL", "VOID") and j["decision"]["token_set"]["complete"]
    assert j["P1"]["d_learned"] == j["P1"]["gt_reference"]["d_vs_blend"]           # learned tube = the GT tube
    assert j["criterion_S"]["P1"]["context"] == "decision (navtrain_val heldout, blend ck2T10, tokens 8 / 8 registered)"
    assert f"**DECISION (P1, amended): {j['decision']['verdict']}** (navtrain_val heldout, blend ck2T10, tokens 8 / 8 " \
           f"registered; teacher v4T10_rng2 --which last, ckpt_sha16 {meta['ckpt_sha16']}" in md("full")
    # no GT tube (the reference of (c)): incomplete, non-zero exit
    with pytest.raises(SystemExit, match="deciding split is incomplete.*no GT R2 tube"):
        E.main([x for x in base if x not in ("--gt-tube", str(GT_VAL))] + ["--det", "v4T10_rng2", "--tag", "nogt"])
    assert ev("nogt")["decision"]["deciding"] is False and ev("nogt")["gt_tube"] is None
    # infer_meta without rg_read: report only (complete token set: exit 0)
    _fake_det_outputs(out, "v4T10_rng2", rows, N, {k: v for k, v in meta.items() if k != "rg_read"})
    j = E.main(base + ["--det", "v4T10_rng2", "--tag", "noread"])["eval"]
    assert j["decision"]["deciding"] is False and not j["decision"]["incomplete"] and "rg_read None" in \
        j["decision"]["reason"]
    # the checkpoint changed after infer: report only
    _fake_det_outputs(out, "v4T10_rng2", rows, N, meta)
    ck.write_bytes(b"retrained")
    j = E.main(base + ["--det", "v4T10_rng2", "--tag", "newck"])["eval"]
    assert j["decision"]["deciding"] is False and "changed since infer" in j["decision"]["reason"]
    assert "checkpoint CHANGED since infer" in md("newck")
