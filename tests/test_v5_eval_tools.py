"""refiner v5 evaluation tooling: tools/ck/v5/refine_stats.py (break / fix, per-key counts, |mu|, bootstrap),
tools/ck/v5/eval_refine_gate.py (the Stage-1 (3) gate: token selection, run parsing and config refusals, the gate
clauses PASS / FAIL / INCOMPLETE and 'deciding', the arm block numbers, the draft-label check, token completeness,
resume identity, the checkpoint copy), tools/ck/v4/eval_range_teacher.range_forward (rel_objs only when given),
tools/ck/e2e2/eval_e2e2.py + eval_ck2.py (the v5 tau1 dump / label job / eval block; a student without v5 unchanged;
extract + infer_check of a v5 relation-student on CPU).  Synthetic arrays except the two CPU smokes at the end (a few
real navtrain_val tokens with the v5 smoke teacher CK/v5/smoke/v5T_smoke2; skipped when absent).  CPU only:

  cd /workspace/yongjae/SSR-v5 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v5_eval_tools.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from tools.ck.v5 import eval_refine_gate as G  # noqa: E402
from tools.ck.v5 import refine_stats as RS  # noqa: E402

L = Cn.LBL
P = L["pdms"]
CK = G.CK
SMOKE_T = CK / "v5" / "smoke" / "v5T_smoke2"
SMOKE_M = CK / "v5" / "smoke" / "v5M_smoke2"


def _lab(n, pdms=1.0, **keys):
    """labels [n, 9]: every key 1, PDMS / EP given, overrides keys[name] = array or scalar."""
    lab = np.ones((n, 9))
    lab[:, P] = pdms
    for k, v in keys.items():
        lab[:, L[k]] = v
    return lab


# ----------------------------------------------------------------------------------------------- refine_stats
def test_break_fix_rules_and_counts():
    ref = _lab(6, pdms=np.array([1.0, 0.8, 0.0, 0.9, 0.7, 0.0]),
               nc=np.array([1, 1, 0, 1, 1, 0.5]), ttc=np.array([1, 1, 1, 0, 1, 1]))
    new = _lab(6, pdms=np.array([0.0, 0.9, 0.6, 0.9, 0.0, 0.0]),
               nc=np.array([0, 1, 1, 1, 1, 0.5]), ttc=np.array([1, 0, 1, 1, 1, 1]), dac=np.array([1, 1, 1, 1, 0, 1]))
    b = RS.break_fix(ref, new, "pdms0")
    # ref PDMS > 0 on rows 0, 1, 3, 4 -> new PDMS 0 on 0, 4 (break 2 / 4); ref 0 on rows 2, 5 -> new > 0 on 2 (fix 1 / 2)
    assert (b["n_break"], b["den"], b["n_fix"], b["fix_den"]) == (2, 4, 1, 2) and b["rate"] == 0.5
    f = RS.break_fix(ref, new, "fails3")
    # ref passes NC / TTC / DAC on rows 0, 1, 4 -> new fails on 0 (nc), 1 (ttc), 4 (dac): 3 / 3; ref fails 2, 3, 5 ->
    # new passes on 2, 3 (fix 2 / 3)
    assert (f["n_break"], f["den"], f["n_fix"], f["fix_den"]) == (3, 3, 2, 3) and f["n_new_fail"] == 4
    with pytest.raises(ValueError):
        RS.break_fix(ref, new, "sometimes")
    k = RS.key_counts(ref, new)
    assert k["nc"] == {"fail_ref": 2, "fail_new": 2, "break": 1, "fix": 1}
    assert k["ttc"] == {"fail_ref": 1, "fail_new": 1, "break": 1, "fix": 1}
    assert k["dac"]["break"] == 1 and k["pdms0"] == {"fail_ref": 2, "fail_new": 3, "break": 2, "fix": 1}
    assert k["ep"]["loss"] == pytest.approx(0.0)


def test_mu_stats_moved_and_boot100():
    mu = np.zeros((4, 12))
    mu[0, 3] = 0.2
    mu[1, 7] = -0.06
    mu[2, 0] = 0.01
    s = RS.mu_stats(mu)
    assert s["n"] == 4 and s["frac_gt_thr"] == 0.5 and s["linf_max"] == pytest.approx(0.2)
    assert s["linf_mean"] == pytest.approx((0.2 + 0.06 + 0.01) / 4) and len(s["per_control_mean_abs"]) == 12
    assert RS.mu_stats(np.zeros((0, 12)))["n"] == 0
    a = np.zeros((3, 8, 3))
    b = a.copy()
    b[1, 4, 0] = 2e-3
    b[2, 4, 2] = 1.0                                          # heading only: not a move in x / y
    assert RS.moved(a, b).tolist() == [False, True, False]
    from tools.ck.eval_ck import log_bootstrap
    rng = np.random.default_rng(0)
    new, ref = rng.uniform(0, 1, 50), rng.uniform(0, 1, 50)
    logs = np.array([f"l{i % 6}" for i in range(50)])
    d = RS.boot100(new, ref, logs, 200, 0)
    e = log_bootstrap(100 * new - 100 * ref, logs, 200, 0)
    assert d == e and d["n"] == 50 and d["n_logs"] == 6


# ----------------------------------------------------------------------------------------------- gate clauses
def _blk(mean, lo, rate, noedit="draft"):
    return {"vs_noedit": {"mean": mean, "lo": lo, "hi": mean + 1, "n": 100}, "noedit": noedit,
            "break_gate": {"rule": "pdms0", "rate": rate, "n_break": int(rate * 1000), "den": 1000}}


def _ci(mean, lo):
    return {"mean": mean, "lo": lo, "hi": mean + 1, "n": 100}


def test_gate_decide_verdicts_and_bars():
    # T alone: (c) / (d) evaluable only -> INCOMPLETE when both pass, FAIL when one fails; never deciding
    g = G.gate_decide({"T": _blk(0.5, 0.1, 0.01)}, {"T": "T"}, {}, deciding_set=True)
    assert g["verdict"] == "INCOMPLETE" and not g["deciding"] and g["context"].startswith("report only")
    assert not g["clauses"]["a_t_minus_s"]["evaluable"] and g["clauses"]["d_t_vs_noedit"]["pass"]
    assert G.gate_decide({"T": _blk(0.29, 0.1, 0.01)}, {"T": "T"}, {}, True)["verdict"] == "FAIL"
    assert G.gate_decide({"T": _blk(0.5, 0.0, 0.01)}, {"T": "T"}, {}, True)["verdict"] == "FAIL"       # lo > 0 strict
    assert G.gate_decide({"T": _blk(0.5, 0.1, 0.0131)}, {"T": "T"}, {}, True)["verdict"] == "FAIL"
    # all three arms
    blocks = {"T": _blk(0.8, 0.4, 0.013), "S": _blk(0.1, -0.05, 0.01), "GT": _blk(1.2, 0.9, 0.01)}
    roles = {"T": "T", "S": "S", "GT": "GT"}
    pairs = {"T-S": _ci(0.30, 0.05), "GT-S": _ci(0.60, 0.3), "GT-T": _ci(0.4, 0.1)}
    g = G.gate_decide(blocks, roles, pairs, deciding_set=True, context="x")
    assert g["verdict"] == "PASS" and g["deciding"] and g["context"] == "decision (x)"     # bars inclusive (0.30, 1.3 %)
    assert g["gt_minus_t"] == pairs["GT-T"] and g["student_floor"]["pass"]
    assert G.gate_decide(blocks, roles, pairs, deciding_set=False)["deciding"] is False
    p2 = dict(pairs, **{"GT-S": _ci(0.61, 0.3)})                                       # 0.30 < 0.5 x 0.61
    g2 = G.gate_decide(blocks, roles, p2, True)
    assert g2["verdict"] == "FAIL" and g2["clauses"]["b_t_minus_s_vs_half_gt"]["pass"] is False
    p3 = dict(pairs, **{"T-S": _ci(0.5, -0.01)})
    assert G.gate_decide(blocks, roles, p3, True)["clauses"]["a_t_minus_s"]["pass"] is False
    # student floor: CI lower bound must be > -0.10
    b4 = dict(blocks, S=_blk(0.1, -0.11, 0.01))
    assert G.gate_decide(b4, roles, pairs, True)["student_floor"]["pass"] is False
    # an arm without a role is report-only: the clauses ignore it
    g5 = G.gate_decide({"v5T_x": _blk(0.5, 0.1, 0.0)}, {"v5T_x": None}, {}, True)
    assert g5["verdict"] == "INCOMPLETE" and not any(c["evaluable"] for c in g5["clauses"].values())


def test_arm_block_numbers():
    n = 40
    logs = np.array([f"log{i % 5}" for i in range(n)])
    draft = _lab(n, pdms=0.8)
    base = _lab(n, pdms=0.8)
    tau1 = _lab(n, pdms=0.8)
    draft[:4, P], draft[:4, L["nc"]] = 0.0, 0.0              # tau1 fixes 4 NC zeros of the draft ...
    base[:4, P], base[:4, L["nc"]] = 0.0, 0.0
    tau1[4, P], tau1[4, L["dac"]] = 0.0, 0.0                 # ... and breaks one DAC
    lab3 = np.stack([draft, base, tau1], 1)
    mu = np.zeros((n, 12))
    mu[:10, 0] = 0.1
    t3 = np.zeros((n, 3, 8, 3))
    t3[:10, 2, :, 0] = 0.5
    b = G.arm_block(lab3, logs, mu, t3, "draft", "pdms0", n_boot=100)
    assert b["n"] == n and b["vs_noedit"] is b["d"]["tau1-draft"]
    assert b["d"]["tau1-draft"]["mean"] == pytest.approx(100 * (4 * 0.8 - 0.8) / n)
    assert b["d"]["base-draft"]["mean"] == pytest.approx(0.0)
    bg = b["break_gate"]
    assert bg["rule"] == "pdms0" and (bg["n_break"], bg["den"], bg["n_fix"], bg["fix_den"]) == (1, 36, 4, 4)
    assert b["brk"]["tau1_vs_draft"]["fails3"]["n_break"] == 1 and b["keys"]["tau1_vs_draft"]["nc"]["fix"] == 4
    assert b["rows"]["draft"]["pass_nc"] == pytest.approx(36 / 40) and b["rows"]["tau1"]["pdms_pos"] == pytest.approx(
        39 / 40)
    assert b["mu"]["frac_gt_thr"] == 0.25 and b["moved"]["tau1_vs_draft"] == 0.25 and b["moved"]["base_vs_draft"] == 0
    bb = G.arm_block(lab3, logs, None, None, "base", "fails3", n_boot=100)
    assert bb["vs_noedit"] is bb["d"]["tau1-base"] and bb["break_gate"]["rule"] == "fails3" and "mu" not in bb
    with pytest.raises(ValueError):
        G.arm_block(lab3, logs, noedit="v2")


def test_check_draft_labels_and_token_gate():
    ref = _lab(5, pdms=0.7)
    ok = np.ones(5, bool)
    c = G.check_draft_labels(ref.astype(np.float32), ok, ref.astype(np.float32), ok)
    assert c["pass"] and c["max_abs_diff"] == 0.0 and c["n_both_ok"] == 5
    bad = ref.copy()
    bad[2, P] += 1e-4
    assert not G.check_draft_labels(bad, ok, ref, ok)["pass"]
    nan = ref.copy()
    nan[1, L["raw_progress"]] = np.nan
    assert G.check_draft_labels(nan, ok, ref, ok)["n_nan_mismatch"] == 1
    ok2 = ok.copy()
    ok2[0] = False
    assert G.check_draft_labels(ref, ok2, ref, ok)["n_ok_mismatch"] == 1
    N = 10
    sel = np.array([1, 2, 3, 7])
    keep = np.zeros(N, bool)
    keep[[1, 3, 7]] = True
    done = np.ones(N, bool)
    done[2] = False
    t = G.token_gate(sel, keep, {"T: infer done (BEV)": done, "T: labels ok": np.ones(N, bool)}, registered=True)
    assert (t["n"], t["n_selected"], t["n_missing"], t["complete"]) == (3, 4, 1, False)
    assert t["missing_by_input"] == {"T: infer done (BEV)": 1} and t["missing_example"] == [2]


# ----------------------------------------------------------------------------------------------- token selection
def _tdf(n=20):
    return pd.DataFrame({"token": [f"tok{i:013d}" for i in range(n)], "log": [f"log{i % 3}" for i in range(n)]})


def test_select_rows_sets_and_refusals(tmp_path, monkeypatch):
    from tools.ck.v4 import eval_range_teacher as E
    tdf, pok = _tdf(), np.ones(20, bool)
    pok[5] = False
    r, tag, _ = G.select_rows("navtest", "all", tdf, pok)
    assert tag == "all" and 5 not in r and len(r) == 19
    with pytest.raises(SystemExit, match="navtrain_val"):
        G.select_rows("navtest", "heldout12179", tdf, pok)
    ho = tmp_path / "HO"
    (ho / "prep").mkdir(parents=True)
    np.save(ho / "prep" / "ho_rows.npy", np.array([9, 2, 4], np.int64))
    monkeypatch.setattr(E, "HO_DIR", ho)
    with pytest.raises(SystemExit, match="expected 12179"):        # the registered set must have its 12,179 rows
        G.select_rows("navtrain_val", "heldout12179", tdf, pok)
    monkeypatch.setattr(G, "N_REGISTERED", 3)
    r, tag, info = G.select_rows("navtrain_val", "heldout12179", tdf, pok)
    assert tag == "heldout12179" and r.tolist() == [2, 4, 9] and info["n_registered"] == 3
    (ho / "results.json").write_text(json.dumps({"n_tokens_used": 2}))     # Phase 0b used fewer: refused
    with pytest.raises(SystemExit, match="unclear"):
        G.select_rows("navtrain_val", "heldout12179", tdf, pok)
    (ho / "results.json").unlink()
    np.save(ho / "prep" / "ho_rows.npy", np.array([5, 2, 4], np.int64))
    with pytest.raises(SystemExit, match="not packed ok"):
        G.select_rows("navtrain_val", "heldout12179", tdf, pok)
    # files: rows (.npy int), tokens (.npy str / text)
    np.save(tmp_path / "rows.npy", np.array([3, 5, 3, 1]))
    r, tag, info = G.select_rows("navtest", str(tmp_path / "rows.npy"), tdf, pok)
    assert r.tolist() == [1, 3] and info["n_not_packed_ok"] == 1 and tag.startswith("file-rows-")
    (tmp_path / "tok.txt").write_text("tok0000000000004\n\ntok0000000000002\n")
    assert G.select_rows("navtest", str(tmp_path / "tok.txt"), tdf, pok)[0].tolist() == [2, 4]
    np.save(tmp_path / "tok.npy", np.array(["tok0000000000007"]))
    assert G.select_rows("navtest", str(tmp_path / "tok.npy"), tdf, pok)[0].tolist() == [7]
    (tmp_path / "bad.txt").write_text("nope\n")
    with pytest.raises(SystemExit, match="not in packed"):
        G.select_rows("navtest", str(tmp_path / "bad.txt"), tdf, pok)
    np.save(tmp_path / "oor.npy", np.array([25]))
    with pytest.raises(SystemExit, match="out of range"):
        G.select_rows("navtest", str(tmp_path / "oor.npy"), tdf, pok)
    with pytest.raises(SystemExit, match="heldout12179 \\| all"):
        G.select_rows("navtest", "nonexistent_file.npy", tdf, pok)
    assert G.default_tokens("navtrain_val", None) == "heldout12179" and G.default_tokens("navtest", None) == "all"


def test_limit_rows_spreads_over_logs():
    logs = np.array([f"log{i // 5}" for i in range(20)])                # 4 logs x 5 rows
    rows = np.arange(20)
    r4 = G.limit_rows(rows, logs, 4)
    assert r4.tolist() == [0, 5, 10, 15]                                # one row (the first) of every log
    r6 = G.limit_rows(rows, logs, 6)
    assert len(r6) == 6 and set(r4) <= set(r6) and len(set(logs[r6])) == 4 and np.all(np.diff(r6) > 0)
    assert G.limit_rows(rows, logs, 6).tolist() == r6.tolist()          # seeded
    assert G.limit_rows(rows, logs, 0).tolist() == rows.tolist() and G.limit_rows(rows, logs, 99).tolist() == \
        rows.tolist()
    sub = np.array([3, 4, 8, 17, 18])
    assert G.limit_rows(sub, logs, 3).tolist() == [3, 8, 17]            # the selection's rows only


# ----------------------------------------------------------------------------------------------- runs / config
def _run_dir(root: Path, name: str, cfg: dict) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


V5_T = {"relation": "teacher", "unknown_ev": True, "corrector": True}


def test_parse_runs_and_config_refusals(tmp_path):
    a = _run_dir(tmp_path, "runA", {"arm": "T", "v5": V5_T})
    b = _run_dir(tmp_path, "runB", {"arm": "M", "v5": {"unknown_ev": True, "corrector": True}})
    R = G.parse_runs([f"T={a}", str(b)])
    assert [(x["role"], x["label"]) for x in R] == [("T", "T"), (None, "runB")]
    a2 = _run_dir(tmp_path / "other", "runA", {"arm": "T", "v5": V5_T})          # another run, the same dir name
    for bad, msg in (([f"X={a}"], "role 'X'"), ([f"T={a}", f"T={b}"], "role given twice"), ([str(a), f"T={a}"],
                     "same run"), ([str(tmp_path / "missing")], "no such run"),
                     ([f"S={a}"], "role 'S' refused"), ([f"GT={a}"], "role 'GT' refused"),   # no A_S / A_GT inputs
                     ([f"T={a}", f"T={b}", f"S={b}"], "role 'S' refused"),
                     ([f"T={a}", str(a2)], "same dir name")):
        with pytest.raises(SystemExit, match=msg):
            G.parse_runs(bad)
    assert G.check_run_cfg(a, json.loads((a / "config.json").read_text()), "navtrain_val")["arm"] == "T"
    with pytest.raises(SystemExit, match="navtest only"):              # the MAP teacher: no ReSMap cache on val
        G.check_run_cfg(b, {"arm": "M", "v5": {"corrector": True}}, "navtrain_val")
    assert G.check_run_cfg(b, {"arm": "M", "v5": {"corrector": True}}, "navtest")["arm"] == "M"
    for cfg, msg in (({"arm": "T", "v5": {"relation": "teacher"}}, "corrector A"),
                     ({"arm": "T"}, "corrector A"),
                     ({"arm": "S", "v5": V5_T}, "arm 'S'"),
                     ({"arm": "T", "v5": dict(V5_T, relation="student")}, "student"),
                     ({"arm": "T", "v5": V5_T, "v4": {"rg_priv": "gt_t0"}}, "PRIVILEGED"),
                     ({"arm": "T", "v5": {"bogus": 1}}, "unknown key")):
        with pytest.raises(SystemExit, match=msg):
            G.check_run_cfg(a, cfg, "navtrain_val")
    with pytest.raises(SystemExit, match="no relation channel"):
        G.check_run_cfg(b, {"arm": "M", "v5": {"corrector": True}}, "navtest", knockout="objects")
    assert G.check_which("ep3") == "ep3" and G.check_which("last") == "last"
    with pytest.raises(SystemExit):
        G.check_which("best")


def test_freeze_ckpt_and_resume_identity(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "ckpt_last.pt").write_bytes(b"A" * 32)
    ad = tmp_path / "out" / "run@last"
    c = G.freeze_ckpt(run, "last", ad)
    assert c["copied"] and not c["source_changed"] and Path(c["copy"]).read_bytes() == b"A" * 32
    (run / "ckpt_last.pt").write_bytes(b"B" * 32)                       # the training moved on: the copy stays
    c2 = G.freeze_ckpt(run, "last", ad)
    assert not c2["copied"] and c2["source_changed"] and c2["ckpt_sha16"] == c["ckpt_sha16"]
    with pytest.raises(SystemExit, match="missing"):
        G.freeze_ckpt(run, "ep7", ad)
    rec = json.loads((ad / "ckpt_last.source.json").read_text())
    assert rec["source"] == str(run.resolve() / "ckpt_last.pt") and rec["copy_sha16"] == c["ckpt_sha16"]
    other = tmp_path / "elsewhere" / "run"                              # another run with the same dir name
    other.mkdir(parents=True)
    (other / "ckpt_last.pt").write_bytes(b"A" * 32)
    with pytest.raises(SystemExit, match="not this run's"):
        G.freeze_ckpt(other, "last", ad)
    (ad / "ckpt_last.pt").write_bytes(b"C" * 32)                        # an altered copy
    with pytest.raises(SystemExit, match="not this run's"):
        G.freeze_ckpt(run, "last", ad)
    (ad / "ckpt_last.pt").write_bytes(b"A" * 32)
    (ad / "ckpt_last.source.json").unlink()                             # a copy without its record
    with pytest.raises(SystemExit, match="not this run's"):
        G.freeze_ckpt(run, "last", ad)
    ad2 = tmp_path / "out" / "fresh@last"
    assert G.freeze_ckpt(other, "last", ad2)["copied"]                  # a fresh dir: copied with its record
    man = G.infer_code_manifest()
    assert "navsim/agents/para_ssr/ck/model.py" in man and "navsim/agents/para_ssr/ck/v5/corrector.py" in man
    assert "tools/ck/data/obj_pack.py" in man and "tools/ck/v5/eval_refine_gate.py::refine_forward" in man
    assert "code_sha16" in G.INFER_ID_KEYS and G.code_sha16(man) == G.code_sha16(dict(man))
    meta = {k: None for k in G.INFER_ID_KEYS}
    meta.update(run_real=str(run), ckpt_sha16=c["ckpt_sha16"], v4={"ev_tbins": 20, "lat": (1, 2)}, score_k=1)
    assert G.infer_resume_check(ad, meta) == {}                       # fresh
    (ad / "infer_meta.json").write_text(json.dumps(meta))
    np.save(ad / "done.npy", np.array([True, False]))
    assert G.infer_resume_check(ad, meta)["ckpt_sha16"] == c["ckpt_sha16"]      # tuples vs JSON lists: equal
    with pytest.raises(SystemExit, match="score_k"):
        G.infer_resume_check(ad, dict(meta, score_k=16))
    with pytest.raises(SystemExit, match="code_sha16"):                 # a resume across a code change
        G.infer_resume_check(ad, dict(meta, code_sha16="0123456789abcdef"))
    (ad / "infer_meta.json").unlink()
    with pytest.raises(SystemExit, match="no infer_meta.json"):
        G.infer_resume_check(ad, meta)


# ----------------------------------------------------------------------------------------------- eval_range_teacher
def test_range_forward_rel_objs_only_when_given():
    from tools.ck.v4 import eval_range_teacher as E
    calls = []

    class FakeNet:
        class trunk:                                                 # noqa: N801
            @staticmethod
            def scene(bev, T):
                return torch.zeros(T, 1), torch.zeros(T, 1)

        @staticmethod
        def scene_pref(mem, ego):
            return None

        @staticmethod
        def score_set(feat, mem, ego, cand, *args, **kw):
            calls.append(kw)
            T, K = cand.shape[:2]
            h = {k: torch.zeros(T, K) for k in ("slack_nc", "slack_ttc", "slack_dac", "lp_nc", "lp_ttc", "lp_dac")}
            return {"score_logit": torch.zeros(T, K, 5), "haz": h}

    T = 2
    cand, var, st = torch.zeros(T, 16, 8, 3), torch.zeros(T, 80, 8, 3), torch.zeros(T, 8)
    E.range_forward(FakeNet, None, cand, st, var)
    assert [sorted(c) for c in calls] == [["pref", "priv"]] * 2                # the v4_3 calls: no rel_objs keyword
    calls.clear()
    rel = {"obj_in": torch.zeros(T, 3, 96, 17)}
    o = E.range_forward(FakeNet, None, cand, st, var, rel_objs=rel)
    assert all(c["rel_objs"] is rel for c in calls) and len(calls) == 2 and o["slack80"].shape == (T, 80, 3)
    if (E.CK / "v5" / "train" / "v5T_rel").is_dir():                  # resolve_run looks under CK/v5/train too
        assert E.resolve_run("v5T_rel") == E.CK / "v5" / "train" / "v5T_rel"


# ----------------------------------------------------------------------------------------------- e2e2 eval (v5)
def _fake_e2e_root(root: Path, run: str, split: str, n: int = 30, v5: bool = True, seed: int = 0):
    """an eval root (packed / infer / pool96 / lat96 labels) + the v5 dump and its labels: pool PDMS 0.6, the lateral
    columns 0.65, the dump-time pick src = column 6 (k 1 identity) everywhere but row 3 (-1), tau1 PDMS 0.9 except a
    planted break on row 5 (PDMS 0, NC 0)."""
    rng = np.random.default_rng(seed)
    pk, inf, lb = root / "packed" / split, root / "infer" / run / split, root / "labels" / split
    for d in (pk, inf, lb / f"pool96_{run}", lb / f"lat96_{run}"):
        d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"token": [f"t{i:05d}" for i in range(n)], "log": [f"log{i % 5}" for i in range(n)], "city": "x",
                  "row": np.arange(n)}).to_parquet(pk / "tokens.parquet")
    np.save(pk / "valid96.npy", np.ones((n, 96), bool))
    np.save(pk / "v2_final.npy", np.sort(rng.normal(size=(n, 16)), 1)[:, ::-1].astype(np.float32))
    np.save(pk / "v2_im.npy", rng.dirichlet(np.ones(16), size=n).astype(np.float32))
    np.save(pk / "ok.npy", np.ones(n, bool))
    logit = rng.normal(-1.0, 0.5, size=(n, 96, 5)).astype(np.float32)
    np.save(inf / "score_logit.npy", logit)
    np.save(inf / "done.npy", np.ones(n, bool))
    lab = np.ones((n, 96, 9), np.float32)
    lab[..., P] = 0.6
    labl = lab.copy()
    labl[..., P] = 0.65
    for name, arr in ((f"pool96_{run}", lab), (f"lat96_{run}", labl)):
        np.save(lb / name / "labels.npy", arr)
        np.save(lb / name / "ok.npy", np.ones((n, 96), bool))
    if v5:
        src = np.full(n, 6, np.int16)
        src[3] = -1
        np.save(inf / "v5_traj.npy", np.zeros((n, 1, 8, 3), np.float32))
        np.save(inf / "v5_src.npy", src)
        mu = np.zeros((n, 12), np.float32)
        mu[:6, 2] = 0.2
        np.save(inf / "v5_mu.npy", mu)
        l5 = np.ones((n, 1, 9), np.float32)
        l5[:, 0, P] = 0.9
        l5[5, 0, P], l5[5, 0, L["nc"]] = 0.0, 0.0
        (lb / f"v5_{run}").mkdir(parents=True, exist_ok=True)
        np.save(lb / f"v5_{run}" / "labels.npy", l5)
        np.save(lb / f"v5_{run}" / "ok.npy", np.ones((n, 1), bool))


def test_eval_ck2_v5_block_and_unchanged_without(tmp_path):
    from tools.ck.e2e2 import eval_ck2 as EV
    root, run = tmp_path / "root", "e2e_v5_fake"
    _fake_e2e_root(root, run, "navtrain_val")
    m = EV.run_eval(root, run, "navtrain_val", n_boot=50, lead_root=tmp_path / "lead")
    v = m["v5"]
    assert (v["n_tok"], v["n_v5"], v["n_no_src"], v["n_unlabelled"]) == (30, 29, 1, 0)
    assert v["rows"]["v5 pick"]["pdms"] == pytest.approx(0.6) and v["rows"]["v5 pick+lat"]["pdms"] == pytest.approx(0.65)
    assert v["rows"]["v5 tau1"]["pdms"] == pytest.approx(0.9 * 28 / 29)
    assert v["d"]["tau1-pick"]["mean"] == pytest.approx(0.9 * 28 / 29 - 0.6) and v["d"]["tau1-pick"]["n"] == 29
    assert v["d"]["pick-v2"]["mean"] == pytest.approx(0.0)
    bp = v["brk"]["tau1_vs_pick"]
    assert (bp["pdms0"]["n_break"], bp["pdms0"]["den"]) == (1, 29) and bp["fails3"]["n_break"] == 1
    assert v["keys"]["tau1_vs_pick"]["nc"]["break"] == 1 and v["mu"]["frac_gt_thr"] == pytest.approx(5 / 29)
    assert v["pick_rule"] is None and v["src_agree"] is None            # no ck_run/config.json in this fake root
    out = Path(m["out_dir"])
    pt = pd.read_parquet(out / "per_token.parquet")
    assert {"v5_src", "pdms_v5_pick", "pdms_v5_tau1"} <= set(pt.columns) and np.isnan(pt.pdms_v5_tau1[3])
    assert "refiner v5 tau1" in (out / "table.md").read_text()
    # the v5 rows change no choice: the same run without the v5 files gives the same rows / choice, no 'v5' key / column
    root2 = tmp_path / "root2"
    _fake_e2e_root(root2, run, "navtrain_val", v5=False)
    m2 = EV.run_eval(root2, run, "navtrain_val", n_boot=50, lead_root=tmp_path / "lead")
    assert "v5" not in m2 and m2["best"] == m["best"] and [r["pdms"] for r in m2["rows"]] == [r["pdms"] for r in m["rows"]]
    pt2 = pd.read_parquet(Path(m2["out_dir"]) / "per_token.parquet")
    assert not ({"v5_src", "pdms_v5_pick", "pdms_v5_tau1"} & set(pt2.columns))
    assert "refiner v5" not in (Path(m2["out_dir"]) / "table.md").read_text()
    # the pick rule of the extracted CK run -> src agreement (select96 recomputed)
    (root.parent / "ck_run").mkdir(exist_ok=True)
    (root.parent / "ck_run" / "config.json").write_text(json.dumps({"ck_e2e2": {"infer_select": None}}))
    m3 = EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=tmp_path / "lead", out=tmp_path / "o3")
    assert m3["v5"]["pick_rule"]["beta"] == 0.5 and m3["v5"]["pick_rule"]["source"] == "default"
    assert 0.0 <= m3["v5"]["src_agree"] <= 1.0
    # dumped but unlabelled -> refused
    import shutil
    shutil.rmtree(root / "labels" / "navtrain_val" / f"v5_{run}")
    with pytest.raises(SystemExit, match="no labels"):
        EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=tmp_path / "lead")


def _pred(B, v5=True):
    p = {"ck2_cand96": np.zeros((B, 96, 8, 3), np.float32), "ck2_valid96": np.ones((B, 96), bool),
         "ck2_cand_idx": np.zeros((B, 16), np.int16), "ck2_v2_final": np.zeros((B, 16), np.float32),
         "ck2_v2_im": np.ones((B, 16), np.float32), "ck2_v2_sim": np.zeros((B, 16, 5), np.float32),
         "ck2_score_logit": np.zeros((B, 96, 5), np.float32), "ck2_w_lat": np.zeros((B, 96, 6), np.float32),
         "ck2_e_lat": np.zeros((B, 96, 6), np.float32), "ck2_lat_traj": np.zeros((B, 96, 8, 3), np.float32)}
    if v5:
        p.update(ck2_v5_traj=np.ones((B, 8, 3), np.float32), ck2_v5_src=np.array([7, 13]),
                 ck2_v5_mu=np.full((B, 12), 0.5, np.float32))
    return p


def test_writer_v5_and_label_job(tmp_path, monkeypatch):
    from tools.ck.e2e import eval_e2e as E1
    from tools.ck.e2e2 import eval_e2e2 as E2
    tdf = pd.DataFrame({"token": [f"t{i}" for i in range(3)], "log": ["a", "a", "b"], "city": ["c"] * 3})
    P0 = E1.Paths(tmp_path / "runN", "navtrain_val")              # a student without v5: no v5 file
    w0 = E2.EvalRootWriter2(P0, tdf)
    assert w0.write([0, 2], _pred(2, v5=False), np.zeros((2, 8), np.float32), np.zeros((2, 8, 3), np.float32),
                    np.zeros((2, 8, 3), np.float32)).all()
    w0.flush()
    assert not any((P0.infer / f"{k}.npy").exists() for k in E2.V5_SPECS)
    P = E1.Paths(tmp_path / "runV", "navtrain_val")
    w = E2.EvalRootWriter2(P, tdf)
    ok = w.write([0, 2], _pred(2), np.zeros((2, 8), np.float32), np.zeros((2, 8, 3), np.float32),
                 np.zeros((2, 8, 3), np.float32))
    w.flush()
    assert ok.all() and np.load(P.packed / "ok.npy").tolist() == [True, False, True]     # the 96-pool ok unchanged
    t = np.load(P.infer / "v5_traj.npy")
    assert t.shape == (3, 1, 8, 3) and float(t[0].min()) == 1.0 and np.isnan(t[1]).all()
    assert np.load(P.infer / "v5_src.npy").tolist() == [7, -1, 13] and np.load(P.infer / "v5_src.npy").dtype == np.int16
    assert np.load(P.infer / "v5_mu.npy")[2, 0] == 0.5
    # stage label: the v5 job (K = 1) next to pool96 / lat96, only when v5_traj exists
    np.save(P.packed / "ok_rows.npy", np.array([0, 2]))
    np.save(P0.packed / "ok_rows.npy", np.array([0, 2]))
    cmds = []

    def fake_run(cmd, env=None, stdout=None, stderr=None):
        cmds.append(cmd)
        d = Path(cmd[cmd.index("--root") + 1]) / "labels" / cmd[cmd.index("--split") + 1] / cmd[cmd.index("--name") + 1]
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "labels.npy", np.zeros((3, 1, 9), np.float32))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(E2.subprocess, "run", fake_run)
    a = SimpleNamespace(workers_label=2)
    E2.stage_label(P, a)
    names = [c[c.index("--name") + 1] for c in cmds]
    assert names == ["pool96_runV", "lat96_runV", "v5_runV"]
    assert cmds[-1][cmds[-1].index("--traj") + 1] == str(P.infer / "v5_traj.npy")
    cmds.clear()
    E2.stage_label(P0, a)
    assert [c[c.index("--name") + 1] for c in cmds] == ["pool96_runN", "lat96_runN"]


V4_STUDENT = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
              "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": True, "edits": "off",
              "corrector": False,
              "v5": {"enabled": True, "relation": "student", "unknown_ev": True, "corrector": True, "n_obj": 8}}


def test_extract_and_infer_check_v5_student(tmp_path):
    """a randomly initialised v5 relation-student (ck_e2e2.v4.v5) in a fake Lightning checkpoint: extract builds its v5
    modules (strict load, config.json 'v5', load_ck round trip); infer_check replays the v5 branch on stored check rows
    (det inputs in check_bev.npz) and compares v5_traj / v5_mu; a perturbed v5_traj fails, missing det inputs refuse."""
    from navsim.agents.para_ssr.ck import anchor_sampler as AS
    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.ck.online2 import (CKE2E2Config, build_student_ck2, infer_pool, lateral_decode,
                                                   student_rel_rows)
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from tools.ck.e2e import eval_e2e as E1
    from tools.ck.e2e2 import eval_e2e2 as E2
    ck_cfg = {"enabled": True, "seed": 0, "v4": V4_STUDENT}
    cfg = CKE2E2Config.from_any({"v4": V4_STUDENT})
    torch.manual_seed(0)
    net = build_student_ck2(cfg, apply_prior=False)
    with torch.no_grad():                                   # a non-trivial corrector (zero-init otherwise)
        for p in net.corrector_v5.parameters():
            p.add_(0.01 * torch.randn_like(p))
    R = tmp_path / "e2e_v5"
    ckpt = R / "train" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"state_dict": {E1.STUDENT_PREFIX + k: v for k, v in net.state_dict().items()}}, ckpt)
    P = E1.Paths(R, "navtrain_val")
    out = E2.extract_ck2(ckpt, P.ck_run, "e2e_v5", ck_cfg)
    assert out["v5"]["relation"] == "student" and out["v5"]["corrector"] is True
    net2, c2 = load_ck(P.ck_run, "last", "cpu")
    assert net2.relation is not None and net2.corrector_v5 is not None and c2["ck_e2e2"]["v4"]["v5"]["enabled"]
    # a 2-row eval root: pool from the anchors, check rows with BEV + det inputs, infer arrays from the v5 branch ops
    n, Q = 2, 12
    anc = torch.from_numpy(np.asarray(AS.load_anchors(), np.float32))
    top = torch.stack([anc[i * 16:(i + 1) * 16] for i in range(n)])                      # [n, 16, 8, 3]
    scale = torch.tensor([1.0, 0.9, 0.95, 1.05, 1.0, 1.0]).reshape(1, 1, 6, 1, 1)
    raw = (top[:, :, None] * scale).reshape(n, 96, 8, 3).contiguous()
    status = torch.zeros(n, 8)
    status[:, 1], status[:, 4] = 1.0, 6.0
    for k, v in (("cand96", raw), ("status", status), ("valid96", torch.ones(n, 96, dtype=torch.bool)),
                 ("v2_final", torch.linspace(1, 0, 16).expand(n, 16)), ("v2_im", torch.full((n, 16), 1 / 16))):
        P.packed.mkdir(parents=True, exist_ok=True)
        np.save(P.packed / f"{k}.npy", v.numpy())
    g = torch.Generator().manual_seed(1)
    bev = torch.randn(n, 256, 50, 100, generator=g)
    code = torch.zeros(n, Q, 10)
    code[..., 0] = torch.rand(n, Q, generator=g) * 8 - 4                   # x_right
    code[..., 1] = torch.rand(n, Q, generator=g) * 25 + 3                  # y_forward
    code[..., 2:4] = 0.5
    code[..., 7] = 1.0
    cls = torch.randn(n, Q, 7, generator=g) + 1.0
    hid = torch.randn(n, Q, 256, generator=g)
    rows = np.arange(n)
    P.infer.mkdir(parents=True, exist_ok=True)
    np.savez(P.infer / "check_bev.npz", rows=rows, bev=bev.numpy(), det_code=code.numpy(), det_cls=cls.numpy(),
             det_hidden=hid.numpy())
    cand, fin, par = infer_pool(raw, raw[:, ::6])
    ego = ego_inputs(status)
    with torch.no_grad():
        feat, mem = net2.trunk.scene(bev, n)
        pref = net2.scene_pref(mem, ego)
        rel = student_rel_rows({"all_bbox_preds": [code], "all_cls_scores": [cls], "det_hidden": hid}, cfg.v5)
        o = net2.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=True, rel_objs=rel)
        src = torch.tensor([3, 50])
        rf = net2.refine_drafts(o, cand.float(), src[:, None], ego)
        lat = lateral_decode(cand, o["w_lat"], ego[0], 0.0)
    for k, v in (("score_logit", o["score_logit"]), ("w_lat", o["w_lat"]), ("e_lat", lat["e_lat"][..., 2:]),
                 ("lat_traj", lat["traj"]), ("v5_traj", rf["tau1"]), ("v5_mu", rf["mu"][:, 0])):
        np.save(P.infer / f"{k}.npy", v.float().numpy())
    np.save(P.infer / "v5_src.npy", src.numpy().astype(np.int16))
    a = SimpleNamespace(gpus="")
    res = E2.stage_infer_check(P, a)
    assert res["pass"] and {"v5_traj", "v5_mu"} <= set(res["max_abs_err"]) and res["v5"]["relation"] == "student"
    assert res["max_abs_err"]["v5_traj"] < 1e-5 and 0.0 <= res["v5"]["src_replay_agree"] <= 1.0
    t = np.load(P.infer / "v5_traj.npy")
    np.save(P.infer / "v5_traj.npy", t + 0.01)
    with pytest.raises(SystemExit, match="infer_check failed"):
        E2.stage_infer_check(P, a)
    np.save(P.infer / "v5_traj.npy", t)
    np.savez(P.infer / "check_bev.npz", rows=rows, bev=bev.numpy())
    with pytest.raises(SystemExit, match="det inputs"):
        E2.stage_infer_check(P, a)


# ----------------------------------------------------------------------------------------------- CPU smokes (real data)
def _have_gate_inputs() -> bool:
    need = [SMOKE_T / "ckpt_last.pt", SMOKE_T / "config.json", CK / "packed/navtrain_val/cand.npy",
            CK / "labels/navtrain_val/cand/labels.npy",
            CK / "v4/analysis/range_oracle/heldout_r2/prep/ho_rows.npy",
            CK / "v3/objpack_k96/bevfusion/navtrain_val/build.json"]
    return all(p.exists() for p in need)


@pytest.mark.skipif(not _have_gate_inputs(), reason="v5 smoke teacher / packed val / object pack absent")
def test_gate_cpu_smoke_val(tmp_path):
    out = tmp_path / "gate"
    base = ["--split", "navtrain_val", "--limit", "2", "--device", "cpu", "--label-workers", "2", "--batch", "2",
            "--workers-io", "0", "--out", str(out), "--bootstrap", "50"]
    res = G.main(base + ["--run", f"T={SMOKE_T}"])
    root = out / "navtrain_val" / "smoke2_heldout12179"
    ad = root / "v5T_smoke2@last"
    rows = res["report"]["blocks"]["T"]["n"]
    assert rows == 2 and res["report"]["token_set"]["complete"] and not res["report"]["gate"]["deciding"]
    tdf = pd.read_parquet(CK / "packed/navtrain_val/tokens.parquet")
    ho = G.limit_rows(np.unique(np.load(CK / "v4/analysis/range_oracle/heldout_r2/prep/ho_rows.npy")),
                      tdf.log.astype(str).to_numpy(), 2)
    assert res["report"]["n_logs"] == 2                                  # --limit spreads over logs
    t3 = np.load(ad / "traj3.npy")
    cand = np.load(CK / "packed/navtrain_val/cand.npy", mmap_mode="r")
    assert np.array_equal(t3[ho, 0], np.asarray(cand[ho, 0])) and np.isfinite(t3[ho]).all()
    assert json.loads((ad / "label_check.json").read_text())["pass"]
    lab = np.load(ad / "labels/labels.npy")
    ref = np.load(CK / "labels/navtrain_val/cand/labels.npy", mmap_mode="r")
    assert np.array_equal(lab[ho, 0], np.asarray(ref[ho, 0]))                 # the draft column = the stored labels
    meta = json.loads((ad / "infer_meta.json").read_text())
    assert meta["ckpt_sha16"] == G.U.sha256_file(ad / "ckpt_last.pt") and meta["v5"]["relation"] == "teacher"
    assert (root / "gate_T@last.json").is_file() and (root / "gate_T@last.md").is_file() and (
        root / "gate_T@last_per_token.parquet").is_file()
    assert "**Gate (report only): " in (root / "gate_T@last.md").read_text()
    json.loads((root / "gate_T@last.json").read_text(), parse_constant=lambda c: pytest.fail(f"bare {c} in JSON"))
    # resume: nothing to infer, labels reused
    r2 = G.main(base + ["--run", f"T={SMOKE_T}", "--stage", "infer,label"])
    assert r2["infer"]["T"]["sessions"][-1]["n"] == 0
    # refusals before any output: the MAP teacher on val, an unknown role
    if SMOKE_M.is_dir():
        with pytest.raises(SystemExit, match="navtest only"):
            G.main(base + ["--run", str(SMOKE_M)])
    with pytest.raises(SystemExit, match="role"):
        G.main(base + ["--run", f"A={SMOKE_T}"])


@pytest.mark.skipif(not _have_gate_inputs(), reason="v5 smoke teacher / packed val / object pack absent")
def test_range_teacher_infer_v5_relation_teacher(tmp_path):
    from tools.ck.v4 import eval_range_teacher as E
    out = tmp_path / "rt"
    E.main(["--split", "navtrain_val", "--det", str(SMOKE_T), "--stage", "infer", "--device", "cpu", "--limit", "2",
            "--batch", "2", "--workers-io", "0", "--out-root", str(out)])
    td = out / "smoke2" / "navtrain_val" / "v5T_smoke2"
    r = np.flatnonzero(np.load(td / "done.npy"))
    assert len(r) == 2 and np.isfinite(np.load(td / "score_logit80.npy")[r]).all()
    assert np.isfinite(np.load(td / "slack16.npy")[r]).all()
    meta = json.loads((td / "infer_meta.json").read_text())
    assert meta["v5"]["relation"] == "teacher" and meta["objects"]["pack_root"]


# ----------------------------------------------------------------------------------------------- stage report (synthetic)
def _fake_arm(ad: Path, N: int, pdms3, done=None, role="T", run="runX"):
    """an arm dir as stages infer + label leave it: traj3 / mu / done / infer_meta, labels [N, 3, 9] with the given
    PDMS per column (pdms3 [N, 3]; 0 -> NC 0 too), label_check (pass) and label_key (the traj3 sha16)."""
    from tools.ck.data import common as CM
    ad.mkdir(parents=True)
    t3 = np.zeros((N, 3, 8, 3), np.float32)
    t3[:, 2, :, 0] = 0.01
    np.save(ad / "traj3.npy", t3)
    np.save(ad / "mu.npy", np.full((N, 12), 0.1, np.float32))
    np.save(ad / "done.npy", np.ones(N, bool) if done is None else done)
    lab = np.ones((N, 3, 9), np.float32)
    lab[..., P] = pdms3
    lab[..., L["nc"]] = np.where(np.asarray(pdms3) == 0, 0.0, 1.0)
    (ad / "labels").mkdir()
    np.save(ad / "labels" / "labels.npy", lab)
    np.save(ad / "labels" / "ok.npy", np.ones((N, 3), bool))
    (ad / "labels" / "meta.json").write_text(json.dumps({"traj_sha16": CM.sha16_array(t3)}))
    (ad / "label_check.json").write_text(json.dumps({"pass": True}))
    (ad / "label_key.json").write_text(json.dumps({"traj_sha16": CM.sha16_array(t3)}))
    src = ad / "src.pt"
    src.write_bytes(b"x")
    (ad / "infer_meta.json").write_text(json.dumps({"run": run, "which": "last", "ckpt": str(src),
                                                    "ckpt_sha16": G.U.sha256_file(src), "source_ckpt": str(src),
                                                    "v5": {"relation": "teacher"}, "knockout": "none", "amp": "fp32",
                                                    "score_k": 1, "arm": "T"}))
    return {"role": role, "label": role, "run": Path(run), "dir": ad}


def test_stage_report_gate_and_completeness(tmp_path, monkeypatch):
    N = 60
    tdf = pd.DataFrame({"token": [f"t{i:04d}" for i in range(N)], "log": [f"log{i % 6}" for i in range(N)]})
    monkeypatch.setattr(G, "packed", lambda split: (tmp_path, tdf, np.ones(N, bool)))
    rows = np.arange(N)
    base = np.full((N, 3), 0.8)
    pT, pS = base.copy(), base.copy()
    pT[:, 2] = 0.9                                         # tau1^T + 10 points everywhere, tau1^S + 2
    pS[:, 2] = 0.82
    root = tmp_path / "root"
    a = SimpleNamespace(split="navtrain_val", tokens="heldout12179", limit=0, noedit="draft", break_rule="fails3",
                        bootstrap=50, drafts="r34top1", knockout="none", tag="", which="last", score_k=1, fresh=False)
    arms = [_fake_arm(root / "T", N, pT, role="T", run="rT"), _fake_arm(root / "S", N, pS, role="S", run="rS")]
    res = G.stage_report(a, arms, rows, root, {"tokens": "heldout12179"})
    g = res["gate"]
    assert res["token_set"]["complete"] and not g["deciding"] and "not evaluable: b_t_minus_s_vs_half_gt" in g["context"]
    assert g["clauses"]["a_t_minus_s"]["value"]["mean"] == pytest.approx(10.0 - 2.0)
    assert g["clauses"]["d_t_vs_noedit"]["pass"] and g["verdict"] == "INCOMPLETE"     # no GT arm: (b) not evaluable
    assert g["student_floor"]["pass"] and res["pairs"]["T-S"]["mean"] == pytest.approx(8.0)
    assert (root / "gate_T@last+S@last.json").is_file() and (root / "gate_T@last+S@last_per_token.parquet").is_file()
    pt = pd.read_parquet(root / "gate_T@last+S@last_per_token.parquet")
    assert len(pt) == N and "T:pdms_tau1" in pt.columns and "S:mu_linf" in pt.columns
    pG = base.copy()
    pG[:, 2] = 0.95                                        # GT - S = 13 -> 0.5 x 13 = 6.5 <= T - S = 8
    arms3 = arms + [_fake_arm(root / "GT", N, pG, role="GT", run="rGT")]
    g3 = G.stage_report(a, arms3, rows, root, {})["gate"]
    assert g3["verdict"] == "PASS" and g3["deciding"] and g3["context"].startswith("decision")
    assert g3["gt_minus_t"]["mean"] == pytest.approx(5.0) and (root / "gate_T@last+S@last+GT@last.md").is_file()
    assert "**Gate: PASS**" in (root / "gate_T@last+S@last+GT@last.md").read_text()
    # strict JSON: a zero-denominator rate (no draft fails -> fix_rate) is null, never a bare NaN
    txt = (root / "gate_T@last+S@last+GT@last.json").read_text()
    j3 = json.loads(txt, parse_constant=lambda c: pytest.fail(f"bare {c} in JSON"))
    assert j3["blocks"]["T"]["brk"]["tau1_vs_draft"]["fails3"]["fix_rate"] is None
    assert j3["gate"]["clauses"]["c_break_t"]["value"]["fix_rate"] is None
    # the registered break rule / no-edit reference only: another one is report only, in its own report file
    for kw, why, name in ((dict(break_rule="pdms0"), "--break-rule pdms0", "gate_T@last+S@last+GT@last_break-pdms0"),
                          (dict(noedit="base"), "--noedit base", "gate_T@last+S@last+GT@last_noedit-base"),
                          (dict(knockout="objects"), "knock-out objects", "gate_T@last+S@last+GT@last_objoff"),
                          (dict(score_k=16), None, "gate_T@last+S@last+GT@last_k16")):
        gx = G.stage_report(SimpleNamespace(**dict(vars(a), **kw)), arms3, rows, root, {})["gate"]
        assert (root / f"{name}.json").is_file()
        if why:
            assert not gx["deciding"] and why in gx["context"]
    assert json.loads((root / "gate_T@last+S@last+GT@last.json").read_text())["noedit"] == "draft"   # not overwritten
    # --noedit base: between-arm pairs are gain differences (tau1 - base per arm), not tau1_A - tau1_B
    pS2 = pS.copy()
    pS2[:, 1] = 0.7                                        # S's base 0.70 -> gain 0.12; T's gain 0.10
    armsb = [_fake_arm(root / "Tb", N, pT, role="T", run="rT"), _fake_arm(root / "Sb", N, pS2, role="S", run="rS")]
    rb = G.stage_report(SimpleNamespace(**dict(vars(a), noedit="base", tag="b")), armsb, rows, root, {})
    assert rb["pairs"]["T-S"]["mean"] == pytest.approx(-2.0, abs=1e-4)
    assert rb["gate"]["clauses"]["a_t_minus_s"]["pass"] is False
    rd = G.stage_report(SimpleNamespace(**dict(vars(a), tag="d")), armsb, rows, root, {})
    assert rd["pairs"]["T-S"]["mean"] == pytest.approx(8.0, abs=1e-4)            # draft: tau1_T - tau1_S
    # a report of the same name for another run is not overwritten (unless --fresh)
    armsr = [_fake_arm(root / "T3", N, pT, role="T", run="rT_other"), arms[1]]
    with pytest.raises(SystemExit, match="not overwritten"):
        G.stage_report(a, armsr, rows, root, {})
    G.stage_report(SimpleNamespace(**dict(vars(a), fresh=True)), armsr, rows, root, {})
    assert json.loads((root / "gate_T@last+S@last.json").read_text())["arm_info"]["T"]["run"] == "rT_other"
    # a registered row missing for one arm: outputs written, then a non-zero exit; not deciding
    done = np.ones(N, bool)
    done[7] = False
    arms2 = [_fake_arm(root / "T2", N, pT, done=done, role="T", run="rT")]
    a.tag = "half"
    with pytest.raises(SystemExit, match="registered token set is incomplete"):
        G.stage_report(a, arms2, rows, root, {})
    j = json.loads((root / "gate_T@last_half.json").read_text())
    assert j["token_set"]["n"] == N - 1 and j["token_set"]["missing_by_input"] == {"T: infer done (BEV)": 1}
    assert not j["gate"]["deciding"] and "incomplete" in j["gate"]["context"]
    # a --limit / other token set: report only, exit 0
    a.tag, a.limit = "lim", 10
    r3 = G.stage_report(a, arms2, rows[:10], root, {})
    assert not r3["gate"]["deciding"] and "not the registered token set" in r3["gate"]["context"]
    # labels of an older traj3 are refused
    np.save(root / "T" / "traj3.npy", np.ones((N, 3, 8, 3), np.float32))
    a.limit = 0
    with pytest.raises(SystemExit, match="rerun --stage label"):
        G.stage_report(a, arms, rows, root, {})
