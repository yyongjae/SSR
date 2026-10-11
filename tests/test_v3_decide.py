"""refiner-v3 Step-1 decision + inference tools (SPEC sections 6, 8; builder LOSS+EVAL).  CPU only.

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v3_decide.py

Synthetic: selection / frozen-DAC rule, AUC, pool layout, hazard pairs, the GO gates on designed arms (G1-G4 edges),
validity flags, arm-meta rules, pre-registered constants.  Reference equality with B_ceiling lib.py (loaded by path,
no bytecode).  Real (skipped when absent): decide with a = b = c = ck2T dump -> deltas 0 / CI (0, 0), PDMS 90.814,
1,605 lead-decel tokens, 1,583 hazard pairs, a-1.0 pref .745, the sd_plugin (SafeDrive-candidate) ck2T10 dump is
refused as arm c; infer_val_v3 on 64 ck2T rows on CPU vs the fp16 GPU dump, CKNet.forward == the lead_remedy split
path, reproduce check on the ck2T dump.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[1])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tools.ck.v3 import decide_step1 as DS  # noqa: E402

LIB = Path("/workspace/yongjae/SSR/report/51_teacher_specialization_evidence/verify/B_ceiling/scripts/lib.py")
CK2T = DS.LR / "infer" / "ck2T"
SD_CK2T10 = DS.CK / "ck2" / "eval" / "sd_plugin" / "navtrain_val" / "ck2T10" / "r34"
CK2T_RUN = DS.CK / "ck2" / "train" / "ck2T"


def _have_real():
    return all(p.is_file() for p in (CK2T / "prob16.npy", CK2T / "prob80.npy", DS.POOL, DS.LEAD,
                                     DS.LABELS / "labels.npy", DS.PACK / "tokens.parquet"))


real = pytest.mark.skipif(not _have_real(), reason="real navtrain_val decision inputs absent")


def load_lib():
    if not LIB.is_file():
        pytest.skip("B_ceiling lib.py absent")
    dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("b_ceiling_lib_ref", LIB)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = dwb
    return mod


# ----------------------------------------------------------------------------------------------- synthetic world
def lab_of(nc, ttc, pdms, dac=None):
    L = np.ones(np.shape(nc) + (9,), np.float64)
    L[..., DS.LBL["nc"]] = nc
    L[..., DS.LBL["ttc"]] = ttc
    L[..., DS.LBL["pdms"]] = pdms
    if dac is not None:
        L[..., DS.LBL["dac"]] = dac
    return L


def world(n=120, n_logs=12, seed=0, n_pool=40):
    """Synthetic decision inputs: random labels, half the tokens lead-decel, a 96-pool on n_pool tokens."""
    g = np.random.default_rng(seed)
    nc = (g.random((n, 16)) > 0.1).astype(float)
    ttc = (g.random((n, 16)) > 0.15).astype(float)
    pdms = g.random((n, 16)) * nc * ttc
    lab = lab_of(nc, ttc, pdms)
    logs = np.array([f"log{i % n_logs:02d}" for i in range(n)])
    lead = np.arange(n) % 2 == 0
    pr = np.sort(g.choice(n, n_pool, replace=False))
    nc96 = (g.random((n_pool, 96)) > 0.2).astype(float)
    ttc96 = (g.random((n_pool, 96)) > 0.3).astype(float)
    lab96 = lab_of(nc96, ttc96, g.random((n_pool, 96)))
    data = DS.Data(tokens=np.array([f"t{i}" for i in range(n)]), logs=logs, lab=lab, lead=lead,
                   p_dac=np.full((n, 16), 0.9), v2_dac=g.random((n, 16)), pool_rows=pr, lab96=lab96,
                   ok96=np.ones((n_pool, 96), bool), valid96=g.random((n_pool, 96)) > 0.1, n_total=n)
    return data


def probs_picking(data, pick, g, sharp=0.97):
    """p16 whose noim selection (frozen DAC 0.9 everywhere) is exactly `pick`; p80 random."""
    n = len(data.lab)
    p = np.full((n, 16, 5), 0.5)
    p[..., DS.PK["nc"]] = 0.6
    p[np.arange(n), pick, DS.PK["nc"]] = sharp
    p[..., DS.PK["dac"]] = g.random((n, 16))          # own DAC: irrelevant under the frozen rule
    p80 = g.uniform(0.05, 0.95, (n, 80, 5))
    return p, p80


PROV_OK = {"errors": [], "warnings": []}


# ----------------------------------------------------------------------------------------------- unit
def test_auc_boot_matches_roc_auc_and_brute_force():
    g = np.random.default_rng(3)
    n, L = 240, 12
    y = g.random(n) < 0.3
    logs = np.array([f"l{i % L}" for i in range(n)])
    s1 = np.round(g.random(n) + 0.4 * y, 2)                               # ties on purpose
    s2 = np.round(g.random(n) + 0.1 * y, 2)
    r = DS.auc_boot(y, {"x": s1, "z": s2}, logs, ["x-z"], n_boot=300, seed=0)["x-z"]
    assert r["mean"] == pytest.approx(DS.roc_auc(y, s1) - DS.roc_auc(y, s2), abs=1e-12)
    # brute force: materialise each resample with the same multinomial weights
    ul, inv = np.unique(logs, return_inverse=True)
    W = np.random.default_rng(0).multinomial(L, np.full(L, 1.0 / L), size=300)
    est = []
    for w in W:
        idx = np.concatenate([np.repeat(np.flatnonzero(inv == j), int(w[j])) for j in range(L)])
        est.append(DS.roc_auc(y[idx], s1[idx]) - DS.roc_auc(y[idx], s2[idx]))
    lo, hi = np.quantile(est, [0.025, 0.975])
    assert (r["lo"], r["hi"]) == pytest.approx((lo, hi), abs=1e-9)
    same = DS.auc_boot(y, {"x": s1, "z": s1.copy()}, logs, ["x-z"], n_boot=100)["x-z"]
    assert (same["mean"], same["lo"], same["hi"]) == (0.0, 0.0, 0.0)


def test_decide_d4_reports_and_second_seed_flag():
    data, arms, exp = designed()
    res = DS.decide(data, arms, n_boot=200, arm_check={"a": [], "b": [], "c": []}, prov_check=PROV_OK)
    g = res["gates"]
    assert set(g["G2"]["values"]) == {"a", "b", "c"} and set(g["G2"]["deltas"]) == {"a-b", "a-c"}
    assert set(g["G3"]["deltas"]) == {"a-b", "a-c"} and g["G4"]["vs_b"] is not None
    for pr in ("a-b", "a-c"):
        d = g["G2"]["deltas"][pr]
        assert d["lo"] <= d["mean"] <= d["hi"]
    assert g["G1"]["second_seed_required"] == bool(abs(exp.mean() - 0.20) <= 0.15)
    assert any("single seed" in s for s in res["notes"])
    assert "pdms_ncttc_only" not in res["deltas"]["a-b"]                 # no ck2T probabilities in the synthetic world
    # the G1 band: a designed delta right at the threshold flags a second seed
    r2 = DS.decide(data, {**arms, "b": arms["a"]}, n_boot=100, arm_check={"a": [], "b": [], "c": []},
                   prov_check=PROV_OK)
    assert r2["gates"]["G1"]["value"] == 0.0 and r2["gates"]["G1"]["second_seed_required"] is False
    # provenance errors make the result invalid
    r3 = DS.decide(data, arms, n_boot=100, arm_check={"a": [], "b": [], "c": []},
                   prov_check={"errors": ["code differs"], "warnings": []})
    assert not r3["valid_for_decision"] and r3["decision"].startswith("INVALID")
    r4 = DS.decide(data, arms, n_boot=100, arm_check={"a": [], "b": [], "c": []}, prov_check=None)
    assert not r4["valid_for_decision"]


def test_g4_noninferiority_edges():
    """pref(a) - pref(c) >= -0.01 and CI lo >= -0.03 (D4), on designed per-pair indicators."""
    data, arms, _ = designed(seed=6)
    ii, kk = DS.hazard_pairs(data.lab96, data.ok96, data.valid96)
    n_p = len(ii)
    pr = data.pool_rows

    def arm_with(ind):
        p16, p80 = arms["c"].p16.copy(), arms["c"].p80.copy()
        P = DS.to_pool(p16[pr], p80[pr])
        P[ii, kk * 6, DS.PK["nc"]] = np.where(ind > 0, 0.9, 0.3)
        P[ii, kk * 6 + 1, DS.PK["nc"]] = 0.6
        P[ii, kk * 6, DS.PK["ttc"]] = P[ii, kk * 6 + 1, DS.PK["ttc"]] = 0.9
        Q = P.reshape(len(pr), 16, 6, 5)
        p16[pr], p80[pr] = Q[:, :, 0], Q[:, :, 1:].reshape(len(pr), 80, 5)
        return DS.Arm(p16, p80)

    c = arm_with(np.ones(n_p))
    for flip, ok in ((0, True), (n_p, False)):
        ind = np.ones(n_p)
        ind[:flip] = 0
        r = DS.decide(data, {**arms, "a": arm_with(ind), "c": c}, n_boot=200,
                      arm_check={"a": [], "b": [], "c": []}, prov_check=PROV_OK)
        g4 = r["gates"]["G4"]
        assert g4["pass"] == ok and g4["pass"] == bool(g4["value"] >= -0.01 and g4["lo"] >= -0.03)


def test_preregistered_constants():
    R = DS.GO_RULE
    assert R["G1"]["min_delta_pdms"] == 0.20 and R["G1"]["ci_lo_gt"] == 0.0 and R["G1"]["pair"] == "a-b"
    assert R["G2"]["min_lead_ttc_auc"] == 0.855 and R["G3"]["max_lead_ncttc_fail_pct"] == 6.97
    assert R["G4"]["pair"] == "a-c" and R["G4"]["min_delta"] == -0.01 and R["G4"]["min_ci_lo"] == -0.03
    assert R["G1"]["second_seed_band"] == 0.15 and R["G2"]["report"] == ["a-b", "a-c"] == R["G3"]["report"]
    assert DS.N_BOOT == 2000 and DS.SEED == 0 and DS.W_NOIM == (0.0, 0.5, 0.5, 1.0) and DS.N_VAL == 18179
    assert DS.FROZEN_DAC == DS.CK / "ck2" / "analysis" / "lead_remedy" / "infer" / "ck2T" / "prob16.npy"


def test_select_pick_known_answer_tie_and_nonfinite():
    p = np.full((3, 4, 5), 0.5)
    p[0, 2, DS.PK["nc"]] = 0.9                      # clear winner 2
    p[2, 1, DS.PK["nc"]] = np.nan                   # NaN candidate never picked; tie among the rest -> first (0)
    assert DS.select_pick(p).tolist() == [2, 0, 0]
    p[1, 3, DS.PK["ttc"]] = 0.99
    assert DS.select_pick(p)[1] == 3


def test_frozen_dac_rule_changes_only_dac():
    g = np.random.default_rng(1)
    p = g.random((5, 16, 5))
    fz = g.random((5, 16))
    q = DS.with_dac(p, fz)
    assert np.array_equal(q[..., DS.PK["dac"]], fz)
    keep = [i for i in range(5) if i != DS.PK["dac"]]
    assert np.array_equal(q[..., keep], p[..., keep]) and q is not p
    assert np.array_equal(DS.with_dac(p, None), p)
    # under the frozen rule the arm's own DAC cannot change the pick
    p2 = p.copy()
    p2[..., DS.PK["dac"]] = g.random((5, 16))
    assert np.array_equal(DS.select_pick(DS.with_dac(p, fz)), DS.select_pick(DS.with_dac(p2, fz)))


def test_roc_auc_bruteforce_and_sklearn():
    g = np.random.default_rng(2)
    y = g.random(300) < 0.3
    s = np.round(g.random(300), 1)                  # many ties
    pos, neg = s[y], s[~y]
    bf = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert DS.roc_auc(y, s) == pytest.approx(bf, abs=1e-12)
    sk = pytest.importorskip("sklearn.metrics")
    assert DS.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)
    assert np.isnan(DS.roc_auc(np.zeros(5, bool), np.arange(5)))


def test_to_pool_layout():
    n = 2
    x16 = np.arange(n * 16).reshape(n, 16).astype(float)
    x80 = 1000 + np.arange(n * 80).reshape(n, 80).astype(float)
    P = DS.to_pool(x16, x80)
    for k in range(16):
        assert (P[:, 6 * k] == x16[:, k]).all()
        for v in range(1, 6):
            assert (P[:, 6 * k + v] == x80[:, k * 5 + v - 1]).all()


def test_hazard_pairs_and_indicator():
    n = 2
    nc = np.ones((n, 96))
    ttc = np.ones((n, 96))
    nc[0, 1] = 0          # token 0: parent 0 safe, a-1.0 unsafe -> pair (0, 0)
    ttc[0, 6 * 3 + 1] = 0  # token 0: parent 3 / its a-1.0 unsafe -> pair (0, 3)
    nc[0, 6 * 5] = 0      # token 0: parent 5 unsafe -> no pair even with an unsafe variant
    nc[0, 6 * 5 + 1] = 0
    nc[1, 6 * 2 + 1] = 0  # token 1: pair (1, 2) but the variant is not allowed -> dropped
    nc[1, 6 * 4 + 2] = 0  # a-0.5 unsafe: not an a-1.0 pair
    lab96 = lab_of(nc, ttc, np.ones((n, 96)))
    valid = np.ones((n, 96), bool)
    valid[1, 6 * 2 + 1] = False
    ok = np.ones((n, 96), bool)
    ii, kk = DS.hazard_pairs(lab96, ok, valid)
    assert list(zip(ii.tolist(), kk.tolist())) == [(0, 0), (0, 3)]
    valid[:, ::6] = False                          # identity columns are always allowed
    assert len(DS.hazard_pairs(lab96, ok, valid)[0]) == 2
    p96 = np.full((n, 96, 5), 0.5)
    p96[0, 1, DS.PK["nc"]] = 0.4                   # pair (0, 0): parent preferred
    p96[0, 6 * 3, DS.PK["ttc"]] = 0.3              # pair (0, 3): variant preferred
    assert DS.hazard_indicator(p96, ii, kk).tolist() == [1.0, 0.0]
    p96[0, 6 * 3 + 1, DS.PK["ttc"]] = 0.3          # exact tie -> not preferred (strict)
    assert DS.hazard_indicator(p96, ii, kk).tolist() == [1.0, 0.0]


def test_reference_equality_with_lib():
    lib = load_lib()
    g = np.random.default_rng(3)
    p = g.uniform(1e-4, 1, (400, 16, 5))
    pick_lib = lib.select(lib.ck_final(p, w=(0, .5, .5, 1)), np.ones((400, 16), bool), np.arange(16))
    assert np.array_equal(DS.select_pick(p), pick_lib)
    d = g.normal(size=500)
    logs = np.array([f"l{i % 23}" for i in range(500)])
    assert DS.boot(d, logs) == pytest.approx(lib.boot(d, logs, 2000, 0))
    x16, x80 = g.random((3, 16, 5)), g.random((3, 80, 5))
    assert np.array_equal(DS.to_pool(x16, x80), lib.to_pool(x16, x80))


# ----------------------------------------------------------------------------------------------- decide (synthetic)
def designed(seed=0, gain_tokens=30):
    """a = b except on gain_tokens tokens where a picks the label-best candidate; c = b."""
    data = world(seed=seed)
    g = np.random.default_rng(seed + 10)
    n = len(data.lab)
    pd_ = data.lab[..., DS.LBL["pdms"]]
    pick_b = np.zeros(n, np.int64)
    pick_a = pick_b.copy()
    idx = np.arange(gain_tokens)
    pick_a[idx] = pd_[idx].argmax(1)
    pa, pa80 = probs_picking(data, pick_a, g)
    pb, pb80 = probs_picking(data, pick_b, g)
    arms = {"a": DS.Arm(pa, pa80), "b": DS.Arm(pb, pb80), "c": DS.Arm(pb.copy(), pb80.copy())}
    exp = 100 * (pd_[np.arange(n), pick_a] - pd_[np.arange(n), pick_b])
    return data, arms, exp


def test_decide_synthetic_g1_and_identity():
    data, arms, exp = designed()
    res = DS.decide(data, arms, n_boot=500, arm_check={"a": [], "b": [], "c": []}, prov_check=PROV_OK)
    d = res["deltas"]["a-b"]["pdms_decision"]
    assert d["mean"] == pytest.approx(exp.mean(), abs=1e-12)
    ref = DS.boot(exp, data.logs, 500, 0)
    assert (d["lo"], d["hi"]) == pytest.approx((ref["lo"], ref["hi"]))
    assert res["gates"]["G1"]["pass"] == bool(exp.mean() >= 0.2 and ref["lo"] > 0)
    bc = res["deltas"]["b-c"]
    for k in ("pdms_decision", "pdms_wdac0", "pdms_v2dac", "pdms_owndac", "lead_fail_pp", "hazard_pref"):
        assert (bc[k]["mean"], bc[k]["lo"], bc[k]["hi"]) == (0.0, 0.0, 0.0), k
    assert res["valid_for_decision"] and res["decision"] in ("GO", "NO-GO")
    assert res["n"] == 120 and res["n_lead"] == 60


def test_decide_gate_edges():
    data, arms, _ = designed(seed=1)
    n = len(data.lab)
    ar = np.arange(n)
    # G3: lead fail of a's picks; G2: lead TTC AUC of a's p_ttc
    res = DS.decide(data, arms, n_boot=200, arm_check={"a": [], "b": [], "c": []})
    m = res["arms"]["a"]
    pick_a = DS.select_pick(DS.with_dac(arms["a"].p16, data.p_dac))
    fail = ~DS.safe_of(data.lab[ar, pick_a])
    assert m["lead"]["ncttc_fail_pct"] == pytest.approx(100 * fail[data.lead].mean())
    assert res["gates"]["G3"]["pass"] == bool(100 * fail[data.lead].mean() <= 6.97)
    y = data.lab[data.lead][..., DS.LBL["ttc"]] < 1
    assert m["lead"]["ttc_auc"] == pytest.approx(DS.roc_auc(y, 1 - arms["a"].p16[data.lead][..., DS.PK["ttc"]]))
    # make a's p_ttc a perfect lead ranker -> G2 passes; then a perfect anti-ranker -> fails
    a2 = DS.Arm(arms["a"].p16.copy(), arms["a"].p80)
    a2.p16[..., DS.PK["ttc"]] = np.where(data.lab[..., DS.LBL["ttc"]] < 1, 0.2, 0.8)
    r2 = DS.decide(data, {**arms, "a": a2}, n_boot=200, arm_check={"a": [], "b": [], "c": []})
    assert r2["arms"]["a"]["lead"]["ttc_auc"] == 1.0 and r2["gates"]["G2"]["pass"]
    a2.p16[..., DS.PK["ttc"]] = 1.0 - a2.p16[..., DS.PK["ttc"]]
    r3 = DS.decide(data, {**arms, "a": a2}, n_boot=200, arm_check={"a": [], "b": [], "c": []})
    assert r3["arms"]["a"]["lead"]["ttc_auc"] == 0.0 and not r3["gates"]["G2"]["pass"]
    assert not r3["go"]


def test_decide_g4_reading():
    """G4 = non-inferiority (D4): pref(a) - pref(c) >= -0.01 and CI lo >= -0.03: a identical to c passes; a that
    always prefers the variant on hazard pairs (c prefers the parent) fails."""
    data, arms, _ = designed(seed=2)
    ii, kk = DS.hazard_pairs(data.lab96, data.ok96, data.valid96)
    assert len(ii) > 10
    pr = data.pool_rows

    def with_pref(p16, p80, parent_wins: bool):
        p16, p80 = p16.copy(), p80.copy()
        P96 = DS.to_pool(p16[pr], p80[pr])
        hi, lo = (0.9, 0.3) if parent_wins else (0.3, 0.9)
        P96[ii, kk * 6, DS.PK["nc"]] = hi
        P96[ii, kk * 6 + 1, DS.PK["nc"]] = lo
        P96[ii, kk * 6, DS.PK["ttc"]] = 0.9
        P96[ii, kk * 6 + 1, DS.PK["ttc"]] = 0.9
        Q = P96.reshape(len(pr), 16, 6, 5)
        p80[pr] = Q[:, :, 1:].reshape(len(pr), 80, 5)
        p16[pr] = Q[:, :, 0]
        return p16, p80

    pc16, pc80 = with_pref(arms["c"].p16, arms["c"].p80, True)
    c = DS.Arm(pc16, pc80)
    same = DS.decide(data, {**arms, "a": DS.Arm(pc16.copy(), pc80.copy()), "c": c}, n_boot=200,
                     arm_check={"a": [], "b": [], "c": []})
    assert same["gates"]["G4"]["hi"] == 0.0 and same["gates"]["G4"]["lo"] == 0.0 and same["gates"]["G4"]["pass"]
    assert same["arms"]["c"]["hazard_a-1.0"]["pref"] == 1.0
    pa16, pa80 = with_pref(arms["a"].p16, arms["a"].p80, False)
    worse = DS.decide(data, {**arms, "a": DS.Arm(pa16, pa80), "c": c}, n_boot=200,
                      arm_check={"a": [], "b": [], "c": []})
    g4 = worse["gates"]["G4"]
    assert g4["value"] == -1.0 and g4["hi"] < 0 and not g4["pass"] and not worse["go"]


def test_decide_validity_flags():
    data, arms, _ = designed(seed=3)
    r = DS.decide(data, arms, n_boot=100, arm_check=None)
    assert not r["valid_for_decision"] and r["decision"].startswith("INVALID") and not r["go"]
    r = DS.decide(data, arms, n_boot=100, arm_check={"a": ["temporal off"], "b": [], "c": []})
    assert not r["valid_for_decision"]
    data.n_total = len(data.lab) + 1
    r = DS.decide(data, arms, n_boot=100, arm_check={"a": [], "b": [], "c": []})
    assert not r["valid_for_decision"] and any("token set" in s for s in r["notes"])
    with pytest.raises(ValueError):
        DS.decide(data, {"a": arms["a"], "b": arms["b"]}, n_boot=100)


def test_decide_optional_d_is_descriptive_only():
    data, arms, _ = designed(seed=4)
    d = DS.Arm(arms["a"].p16.copy(), arms["a"].p80.copy())
    r0 = DS.decide(data, arms, n_boot=100, arm_check={"a": [], "b": [], "c": []})
    r1 = DS.decide(data, {**arms, "d": d}, n_boot=100, arm_check={"a": [], "b": [], "c": [], "d": []})
    assert "d-b" in r1["deltas"] and "d-b" not in r0["deltas"]
    assert {g: v["pass"] for g, v in r0["gates"].items()} == {g: v["pass"] for g, v in r1["gates"].items()}


def test_full_history_subset_reported():
    data, arms, exp = designed(seed=5)
    data.full_hist = np.arange(len(data.lab)) % 3 != 0
    r = DS.decide(data, arms, n_boot=100, arm_check={"a": [], "b": [], "c": []})
    fh = r["deltas"]["a-b"]["full_history_pdms"]
    assert fh["mean"] == pytest.approx(exp[data.full_hist].mean()) and fh["n"] == int(data.full_hist.sum())
    assert r["arms"]["a"]["full_history"]["n"] == int(data.full_hist.sum())


def test_check_arm_meta_rules():
    ref = {"cand_sha16": "c", "variants_sha16": "v"}

    def meta(on, fc=None, priv=False, **kw):
        t = {"on": on, "forecaster": fc or "learned", "privileged": priv}
        vp = {"data": {"packs": {"navtrain_val": {"build_sha256": "V"}}}}
        return dict(split="navtrain_val", which="last", temporal=t, privileged=priv, cand_sha16="c",
                    variants_sha16="v", check={"ok": True}, provenance={"train": vp, "infer": vp}, **kw)

    m = meta(True, "learned")
    m["check"] = {"ok": False, "rule": "x"}
    assert any("reproduce check" in x for x in DS.check_arm_meta("a", m, ref))           # D6: check required
    m = meta(True, "ca")
    m["provenance"] = {"train": m["provenance"]["train"], "infer": {"data": {"packs": {"navtrain_val":
                                                                                   {"build_sha256": "OTHER"}}}}}
    assert any("val pack" in x for x in DS.check_arm_meta("b", m, ref))                  # D5: same val pack
    m = meta(True, "ca")
    m["provenance"] = {}
    assert any("provenance" in x for x in DS.check_arm_meta("b", m, ref))                # pre-fix run

    assert DS.check_arm_meta("a", meta(True, "learned"), ref) == []
    assert DS.check_arm_meta("b", meta(True, "ca"), ref) == []
    assert DS.check_arm_meta("c", meta(False), ref) == []
    assert DS.check_arm_meta("d", meta(True, "oracle_gt", True), ref) == []
    assert DS.check_arm_meta("a", meta(True, "ca"), ref)                       # wrong forecaster
    assert DS.check_arm_meta("a", meta(False), ref)                            # off
    assert DS.check_arm_meta("c", meta(True, "ca"), ref)                       # c must be off
    assert DS.check_arm_meta("d", meta(True, "oracle_gt", False), ref)         # d must be privileged
    assert DS.check_arm_meta("a", meta(True, "learned", True), ref)            # a must not be
    m = meta(True, "learned")
    m["which"] = "best"
    assert DS.check_arm_meta("a", m, ref)
    m = meta(True, "learned")
    m["cand_sha16"] = "other"                                                 # e.g. SafeDrive candidates
    assert DS.check_arm_meta("a", m, ref)
    assert DS.check_arm_meta("c", {"teacher": "ck2T10"}, ref)                # no shas -> refused


def test_provenance_check_between_arms():
    def raw(code="c", tr="t"):
        p = {"k_obj": 96, "code": {"train": {"sha256": code}, "worktree": {"sha256": "w"}},
             "data": {"packs": {"navtrain_train": {"build_sha256": tr}, "navtrain_val": {"build_sha256": "v"}},
                      "ca_scales": {"sha256": "s"}}}
        return {"meta": {"provenance": {"train": p}}}
    assert DS.provenance_check({"a": raw(), "b": raw(), "c": {"meta": {}}}) == PROV_OK
    assert DS.provenance_check({"a": raw(), "b": raw(code="x")})["errors"]
    assert DS.provenance_check({"a": raw(), "b": raw(tr="x")})["errors"]


# ----------------------------------------------------------------------------------------------- real
@real
def test_real_ck2T_selfcompare(tmp_path):
    """a = b = c = the ck2T (30-ep all-key) dump: every delta 0 with CI (0, 0); known answers of reports 51 / 53."""
    out = tmp_path / "step1.json"
    res = DS.run({"a": str(CK2T), "b": str(CK2T), "c": str(CK2T)}, out, arm_check=False)
    assert res["n"] == 18179 and res["n_logs"] == 214
    assert res["n_lead"] == 1605
    assert res["n_hazard_pairs"] == 1583
    A = res["arms"]["a"]
    assert A["pdms"] == pytest.approx(90.814, abs=1e-3)               # frozen DAC = ck2T's own DAC here
    assert A["pdms_owndac"] == A["pdms"]
    assert abs(A["hazard_a-1.0"]["pref"] - 0.745) < 0.005
    assert A["lead"]["ttc_auc"] == pytest.approx(0.793, abs=1e-3)      # A_novelty 'none' row
    assert A["lead"]["ncttc_fail_pct"] == pytest.approx(10.34, abs=0.01)
    for pr in ("a-b", "b-c", "a-c"):
        for k, v in res["deltas"][pr].items():
            assert (v["mean"], v["lo"], v["hi"]) == (0.0, 0.0, 0.0), (pr, k)
    assert not res["valid_for_decision"] and res["decision"].startswith("INVALID")
    j = json.load(open(out))
    assert j["n_hazard_pairs"] == 1583 and j["info"]["frozen_dac_sha16"] == DS.sha16(DS.FROZEN_DAC)


@real
def test_real_arm_checks_refuse_wrong_dumps(tmp_path):
    with pytest.raises(SystemExit, match="arm checks failed"):
        DS.run({"a": str(CK2T), "b": str(CK2T), "c": str(CK2T)}, tmp_path / "x.json", arm_check=True)
    if (SD_CK2T10 / "prob16.npy").is_file():
        with pytest.raises(SystemExit, match="cand_sha16"):
            DS.run({"a": str(CK2T), "b": str(CK2T), "c": str(SD_CK2T10)}, tmp_path / "y.json", arm_check=True)
    assert not (tmp_path / "x.json").exists()


@real
def test_real_reproduce_check_on_ck2T_dump():
    from tools.ck.v3 import infer_val_v3 as IV
    p16 = np.load(CK2T / "prob16.npy")
    chk = IV.reproduce_check(CK2T_RUN, p16, np.load(CK2T / "done.npy"))
    assert chk["n_sel"] == 18179 and chk["ok"] and chk["absdiff"] <= 1e-6
    assert chk["pdms_ck_noim"] == pytest.approx(0.908141, abs=1e-6)


def test_infer_guards(monkeypatch):
    from tools.ck.v3 import infer_val_v3 as IV
    for cv in ("2", "3", "0,1", "4", None):
        if cv is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cv)
        with pytest.raises(SystemExit):
            IV.device_guard("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    IV.device_guard("cuda")
    IV.device_guard("cpu")
    with pytest.raises(SystemExit):
        IV.device_guard("mps")


@pytest.mark.skipif(not (_have_real() and (CK2T_RUN / "ckpt_last.pt").is_file()
                         and os.environ.get("V3_SKIP_SLOW") != "1"), reason="real ck2T run absent / slow skipped")
def test_real_infer_cpu_64_rows_vs_gpu_dump(tmp_path_factory):
    """infer_val_v3 (temporal-off path) on the first 64 val rows on CPU fp32 vs the fp16 GPU lead_remedy dump; and
    CKNet.forward == the lead_remedy split path (trunk.scene once + trunk.candidates x 2) bitwise on CPU."""
    import torch

    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    from tools.ck.v3 import infer_val_v3 as IV
    out = Path("/tmp") / f"v3_test_infer_{os.getpid()}"
    try:
        IV.main(["--run", str(CK2T_RUN), "--device", "cpu", "--limit", "64", "--workers", "0", "--out", str(out)])
        mine = np.load(out / "prob16.npy")[:64]
        ref = np.load(CK2T / "prob16.npy")[:64]
        d = np.abs(mine - ref)
        assert np.quantile(d, 0.99) < 2e-3 and d.max() < 0.02          # fp16 (GPU) vs fp32 (CPU) [실측 p99 6e-4]
        d80 = np.abs(np.load(out / "prob80.npy")[:64] - np.load(CK2T / "prob80.npy")[:64])
        assert np.quantile(d80, 0.99) < 2e-3 and d80.max() < 0.03
        assert np.load(out / "done.npy").sum() == 64 and np.isnan(np.load(out / "prob16.npy")[64:]).all()
        meta = json.load(open(out / "infer_meta.json"))
        assert meta["temporal"]["on"] is False and meta["privileged"] is False
        assert meta["cand_sha16"] == DS.sha16(DS.PACK / "cand.npy")
        assert meta["variants_sha16"] == DS.sha16(DS.VARIANTS)
        net, _ = load_ck(CK2T_RUN, which="last", device="cpu")[:2]
        net.eval()
        ds = CKDataset("navtrain_val", bev="T", k=16, labels=None, gt=False, rows=np.arange(4))
        b = collate_ck([ds[i] for i in range(4)])
        var = torch.from_numpy(np.array(np.load(DS.VARIANTS, mmap_mode="r")[:4], np.float32))
        with torch.no_grad():
            o = net(b["bev"], b["cand"].float(), b["status"].float(), extra=var, decode=False)
            v0, a0, eds, cmd = ego_inputs(b["status"].float())
            feat, mem = net.trunk.scene(b["bev"], 4)
            s16 = net.score(net.trunk.candidates(feat, mem, b["cand"].float(), v0, a0, eds, cmd), 4, 16)
            s80 = net.score(net.trunk.candidates(feat, mem, var, v0, a0, eds, cmd), 4, 80)
        assert torch.equal(o["score_logit"], s16) and torch.equal(o["extra_score_logit"], s80)
        assert np.abs(o["score_logit"].numpy() - np.load(out / "score_logit16.npy")[:4]).max() < 1e-4
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
