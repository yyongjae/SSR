"""tools/ck/v4/eval_range_teacher.py (refiner v4_3 (c): range teachers as selection evidence): the criterion S switch
rule (pool-order ties = Phase 0b) and its numbers, the fidelity / bootstrap helpers, the pair key rule, the blend
critics, the Phase 0b R2 GT tube (bitwise equal to tools/ck/v4/oracle/oracle_common.grid_eval 'grid stopped-aware NC +
read any4' on the same states; the LQR states from the packed status vs the official F1 states) and the readout-rule
tube, the --gt-tube rules (gttube refuses it; eval reads it read-only after the meta / shape checks), the Phase 0b
reproduction check logic, the CUDA guard of stage infer (approval, busy GPU; no GPU is touched), and a CPU smoke of
every stage on a few real navtest / navtrain_val tokens with randomly initialised R2 range teachers saved to a scratch
dir (MAP refused on val, non-range teachers refused).
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
    assert ev["phase0b_check"]["comparable"] is False and ev["decision_rule"] is None
    assert set(ev["selection"]["vs"]) == {"ck2T10,ck2M10", "v4T10_haz,v4M10_haz"}
    assert "pool96_ck_noim" in ev["selection"]["range"]
    tag = "rng2T+rng2M"
    assert (root / f"eval_{tag}.md").is_file() and (root / f"eval_{tag}_picks.npz").is_file()
    md = (root / f"eval_{tag}.md").read_text()
    assert "decision rule: not registered" in md and "Phase 0b convention" in md and "gt_rule_same_keys" in md
    j = json.loads((root / f"eval_{tag}.json").read_text())
    assert j["n_tokens"] == 3
    # gtcheck: GT rows only, no teacher
    g = E.main(base + ["--stage", "gtcheck"])["gtcheck"]
    assert g["n_tokens"] == 3 and set(g["criterion_S"]) >= {"gt_r2_all_keys", "gt_rule_all_keys"}
    assert g["criterion_S"]["gt_r2_all_keys"] == S["gt_r2_all_keys"]         # the same row as in eval
    assert (root / "gtcheck.md").is_file()
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
    assert m2["sessions"][-1]["n"] == 0
    # DET only: keys nc / ttc, the DET-only rule, baselines reduced to their DET member
    e1 = E.main(base + ["--det", str(det), "--stage", "eval"])["eval"]
    assert e1["keys"] == ["nc", "ttc"] and e1["selection"]["rule"].startswith("DET only")


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
