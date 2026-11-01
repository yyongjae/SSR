"""CK2 e2e selection over the 96-candidate pool (SPEC s8 T11): ck/select2.py and tools/ck/e2e2/eval_ck2.py on synthetic
arrays (val choice -> navtest fixed).  CPU, no model:
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_select.py \
    --basetemp=/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e/pytest/select
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.select import blend, ck_final, select_all  # noqa: E402


def _rand(n=7, seed=0):
    rng = np.random.default_rng(seed)
    v2f = np.sort(rng.normal(size=(n, 16)), 1)[:, ::-1].copy()          # top-16 sorted desc (student_topk order)
    im = rng.dirichlet(np.ones(16), size=n)
    prob = rng.uniform(0.05, 0.95, size=(n, 96, 5))
    valid = rng.random((n, 96)) > 0.2
    valid[:, ::6] = True
    return v2f, im, prob, valid


def test_constants_and_layout():
    assert S2.VARIANT_SETS == {"none": (0,), "decel": (0, 1, 2), "speed": (0, 1, 2, 3), "lat": (0, 4, 5),
                               "all_noaccel": (0, 1, 2, 4, 5), "all": (0, 1, 2, 3, 4, 5)}
    assert S2.LAT_MODES == ("on", "on_except_latvar", "off")
    from navsim.agents.para_ssr.ck.e2e_data2 import VNAMES
    assert S2.VNAMES == tuple(VNAMES)
    assert sorted(S2.TIE_ORDER.tolist()) == list(range(96)) and S2.TIE_ORDER[:16].tolist() == list(range(0, 96, 6))


def test_pool_scores_parent_inheritance():
    v2f, im, prob, _ = _rand()
    s0 = S2.pool_scores(v2f, im, prob, 0.0)
    np.testing.assert_array_equal(s0, np.repeat(v2f, 6, 1))                       # beta 0 = parent's v2 score
    s1 = S2.pool_scores(v2f, im, prob, 1.0)
    np.testing.assert_allclose(s1, ck_final(prob, np.repeat(im, 6, 1)), rtol=0, atol=0)
    sb = S2.pool_scores(v2f, im, prob, 0.3)
    np.testing.assert_allclose(sb, blend(np.repeat(v2f, 6, 1), ck_final(prob, np.repeat(im, 6, 1)), 0.3))
    # torch inputs give the same numbers
    st = S2.pool_scores(torch.tensor(v2f), torch.tensor(im), torch.tensor(prob), 0.3)
    np.testing.assert_allclose(st, sb)
    # offset: zero = OFF, a positive nc offset raises every score
    np.testing.assert_allclose(S2.pool_scores(v2f, im, prob, 1.0, offset=[0] * 5), s1, atol=1e-9)
    assert (S2.pool_scores(v2f, im, prob, 1.0, offset=[2, 0, 0, 0, 0]) > s1).all()


def test_select96_rules():
    v2f, im, prob, valid = _rand(n=50, seed=1)
    # identity-only set == old select_all 'a' over the 16 originals (same ck_final / blend)
    for beta in Cn.BETA_GRID:
        a = select_all(v2f, im, prob[:, ::6], beta=beta)["a"]
        np.testing.assert_array_equal(S2.select96(v2f, im, prob, valid, beta, S2.VARIANT_SETS["none"]), 6 * a)
    # beta 0: variants inherit the parent's v2 score and ties go to the identity -> column 0 (v2's trajectory)
    assert (S2.select96(v2f, im, prob, valid, 0.0) == 0).all()
    # invalid / out-of-set / non-finite columns are never chosen
    for beta in (0.5, 1.0):
        for name, types in S2.VARIANT_SETS.items():
            idx = S2.select96(v2f, im, prob, valid, beta, types)
            assert valid[np.arange(50), idx].all() and np.isin(idx % 6, types).all(), name
    p2 = prob.copy()
    p2[:, 7] = 0.999                                   # column 7 (k 1, a-1.0) best ...
    v2 = valid.copy()
    v2[:, 7] = False                                   # ... but invalid
    assert not (S2.select96(v2f, im, p2, v2, 1.0) == 7).any()
    v2[:, 7] = True
    assert (S2.select96(v2f, im, p2, v2, 1.0) == 7).all()
    p3 = p2.copy()
    p3[:, 7] = np.nan
    assert not (S2.select96(v2f, im, p3, v2, 1.0) == 7).any()
    assert (S2.select96(v2f, im, np.full_like(prob, np.nan), valid, 1.0) == 0).all()     # nothing finite -> 0
    # ties: equal scores of an identity (k 3) and a variant of k 1 -> the identity; two variants -> lower c
    pt = np.full((1, 96, 5), 0.1)
    imt = np.full((1, 16), 1 / 16)
    pt[0, 18] = pt[0, 7] = 0.9                         # identity k 3 (c 18) and a-1.0 of k 1 (c 7)
    assert S2.select96(np.zeros((1, 16)), imt, pt, np.ones((1, 96), bool), 1.0).tolist() == [18]
    pt[0, 18] = 0.1
    pt[0, 13] = 0.9                                    # c 7 and c 13 (both variants)
    assert S2.select96(np.zeros((1, 16)), imt, pt, np.ones((1, 96), bool), 1.0).tolist() == [7]


def test_final_traj_and_labels():
    rng = np.random.default_rng(3)
    cand = rng.normal(size=(5, 96, 8, 3)).astype(np.float32)
    lat = rng.normal(size=(5, 96, 8, 3)).astype(np.float32)
    idx = np.array([0, 4, 5, 6, 95])                   # id, l-0.5, l+0.5, id k 1, l+0.5 k 15
    for mode, want in (("on", [1, 1, 1, 1, 1]), ("off", [0, 0, 0, 0, 0]), ("on_except_latvar", [1, 0, 0, 1, 0])):
        assert S2.use_lat(idx, mode).astype(int).tolist() == want
        t, src = S2.final_traj(cand, lat, idx, mode)
        for i, w in enumerate(want):
            np.testing.assert_array_equal(t[i], (lat if w else cand)[i, idx[i]])
            assert src[i] == ("lat" if w else "pool")
        tt, _ = S2.final_traj(torch.tensor(cand), torch.tensor(lat), torch.tensor(idx), mode)
        assert isinstance(tt, torch.Tensor) and np.array_equal(tt.numpy(), t)
    lp, ll = rng.random((5, 96, 9)), rng.random((5, 96, 9))
    got = S2.chosen_labels(lp, ll, idx, "on_except_latvar")
    np.testing.assert_array_equal(got[1], lp[1, 4])
    np.testing.assert_array_equal(got[3], ll[3, 6])
    with pytest.raises(ValueError):
        S2.use_lat(idx, "sometimes")


# ----------------------------------------------------------------------------------------------- eval_ck2
def _fake_root(root: Path, run: str, split: str, n: int = 40, seed: int = 0):
    """eval root with a planted best: variant column 3 (a+0.5 of k 0) is labelled PDMS 1 and has the highest CK
    probability; v2's column 0 PDMS 0.5; the laterally corrected columns are 0.1 worse."""
    rng = np.random.default_rng(seed)
    pk, inf, lb = root / "packed" / split, root / "infer" / run / split, root / "labels" / split
    for d in (pk, inf, lb / f"pool96_{run}", lb / f"lat96_{run}"):
        d.mkdir(parents=True, exist_ok=True)
    toks = [f"t{split[:3]}{i:04d}" for i in range(n)]
    pd.DataFrame({"token": toks, "log": [f"log{i % 7}" for i in range(n)], "city": "x", "row": np.arange(n)}
                 ).to_parquet(pk / "tokens.parquet")
    valid = rng.random((n, 96)) > 0.1
    valid[:, ::6] = True
    valid[:, 3] = True
    np.save(pk / "valid96.npy", valid)
    np.save(pk / "v2_final.npy", np.sort(rng.normal(size=(n, 16)), 1)[:, ::-1].astype(np.float32))
    np.save(pk / "v2_im.npy", rng.dirichlet(np.ones(16), size=n).astype(np.float32))
    ok = np.ones(n, bool)
    ok[0] = False                                       # packed not ok -> excluded
    np.save(pk / "ok.npy", ok)
    logit = rng.normal(-1.0, 0.5, size=(n, 96, 5)).astype(np.float32)
    logit[:, 3] = 6.0
    np.save(inf / "score_logit.npy", logit)
    np.save(inf / "done.npy", np.ones(n, bool))
    L = Cn.LBL
    lab = np.full((n, 96, 9), 0.5, np.float32)
    lab[..., L["nc"]] = lab[..., L["ttc"]] = 1.0
    lab[:, 3, L["pdms"]] = 1.0
    labl = lab.copy()
    labl[..., L["pdms"]] -= 0.1
    okp = np.ones((n, 96), bool)
    okp[1, 3] = False                                   # a submittable column unlabelled -> token 1 excluded
    okp[2] = valid[2] | (np.arange(96) % 6 == 0)        # unlabelled INVALID columns do not exclude token 2
    okp[2, ~valid[2]] = False
    for name, arr in ((f"pool96_{run}", lab), (f"lat96_{run}", labl)):
        np.save(lb / name / "labels.npy", arr)
        np.save(lb / name / "ok.npy", okp)
    return toks


def test_eval_ck2_val_choice_then_navtest_fixed(tmp_path):
    from tools.ck.e2e2 import eval_ck2 as EV
    root, run = tmp_path / "root", "ck2e2e_fake"
    _fake_root(root, run, "navtrain_val", seed=0)
    _fake_root(root, run, "navtest", seed=1)
    lead = tmp_path / "lead"                            # no lead file -> lead metrics absent
    m = EV.run_eval(root, run, "navtrain_val", n_boot=50, lead_root=lead)
    assert m["n_total"] == 40 and m["n_eval"] == 38      # token 0 (packed) and 1 (unlabelled column) excluded
    keys = {r["key"] for r in m["rows"]}
    assert {"v2", "v2_lat", "oracle16", "oracle96", "oracle96_lat"} <= keys
    assert sum(r["variant"] == "sel" for r in m["rows"]) == 5 * 6 * 3
    rows = {r["key"]: r for r in m["rows"]}
    assert rows["v2"]["pdms"] == pytest.approx(0.5) and rows["oracle96"]["pdms"] == pytest.approx(1.0)
    assert rows["sel b=1 set=none lat=off"]["pdms"] == pytest.approx(0.5)          # column 3 not in 'none'
    # default choose_lat 'on' (user: the learned lateral is applied after selection): only beta / variant set are
    # chosen; the planted column 3 is submitted laterally corrected (PDMS 0.9), although 'off' would score 1.0 on val
    best = m["best"]
    assert m["choose_lat"] == "on" and best["choose_lat"] == "on"
    assert best["set"] == "all" and best["lat_mode"] == "on" and best["pdms"] == pytest.approx(0.9)
    top_betas = [r["beta"] for r in m["rows"] if r["variant"] == "sel" and r["set"] == "all" and r["lat_mode"] == "on"
                 and abs(r["pdms"] - 0.9) < 1e-6]                  # f32 labels
    assert best["beta"] == max(top_betas)
    r = rows[best["key"]]
    assert r["sel_share"]["a+0.5"] == pytest.approx(1.0) and r["frac_lat"] == 1.0
    assert r["d_pdms"]["mean"] == pytest.approx(0.4) and r["d_pdms"]["n"] == 38
    # choose_lat 'any': navtrain_val chooses the lateral mode too -> the uncorrected column (lat off, PDMS 1.0)
    ma = EV.run_eval(root, run, "navtrain_val", out=tmp_path / "val_any", n_boot=50, lead_root=lead, choose_lat="any")
    ba = ma["best"]
    assert ba["set"] == "all" and ba["lat_mode"] == "off" and ba["pdms"] == pytest.approx(1.0)
    ra = {x["key"]: x for x in ma["rows"]}[ba["key"]]
    assert ra["frac_lat"] == 0.0 and ra["d_pdms"]["mean"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        EV.choose_best(m["rows"], "sometimes")
    vm = root / "eval" / run / "navtrain_val" / "metrics.json"
    assert vm.is_file() and (vm.parent / "table.md").is_file() and (vm.parent / "per_token.parquet").is_file()
    # navtest: refused without the val choice; with it only the fixed rows are computed (never chosen on navtest)
    with pytest.raises(SystemExit):
        EV.run_eval(root, run, "navtest", n_boot=50, lead_root=lead)
    t = EV.run_eval(root, run, "navtest", fix_from=str(vm), n_boot=50, lead_root=lead)
    assert "best" not in t and t["fixed_from_val"]["key"] == best["key"]
    sel_rows = [x for x in t["rows"] if x["variant"] == "sel"]
    assert {(x["beta"], x["set"], x["lat_mode"]) for x in sel_rows} <= {(best["beta"], best["set"], best["lat_mode"]),
                                                                        (1.0, "all", "on")}
    assert t["representative"] == best["key"]
    assert json.loads((root / "eval" / run / "navtest" / "metrics.json").read_text())["representative"] == best["key"]


# ----------------------------------------------------------------------------------------------- named weight sets
def test_named_selection_weight_sets(tmp_path):
    """select.SEL_W_SETS: 'default' = SEL_W (every default unchanged), 'plugin' = (0, 1, 1, 1), 'noim'; names are
    accepted wherever weights are; eval_ck2 --sel-w writes <split>__selw_<name> and records the set."""
    from navsim.agents.para_ssr.ck import select as S1
    from tools.ck.e2e2 import eval_ck2 as EV
    assert S1.resolve_w(None) == S1.resolve_w("default") == tuple(Cn.SEL_W) == (0.1, 0.5, 0.5, 1.0)
    assert S1.resolve_w("plugin") == (0.0, 1.0, 1.0, 1.0) and S1.resolve_w("noim") == (0.0, 0.5, 0.5, 1.0)
    assert S1.sel_w_name(Cn.SEL_W) == "default" and S1.sel_w_name((0, 1, 1, 1)) == "plugin"
    assert S1.sel_w_name((0.2, 1, 1, 1)) == "0.2,1,1,1"
    with pytest.raises(ValueError):
        S1.resolve_w("bogus")
    with pytest.raises(ValueError):
        S1.resolve_w((1.0, 2.0))
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, size=(6, 16, 5))
    im = rng.dirichlet(np.ones(16), size=6)
    assert np.array_equal(ck_final(p, im), ck_final(p, im, "default"))
    eps = Cn.SEL_EPS
    want = np.log(p[..., 0] + eps) + np.log(p[..., 1] + eps) + np.log(5 * p[..., 3] + 2 * p[..., 4] + 5 * p[..., 2] + eps)
    np.testing.assert_allclose(ck_final(p, im, "plugin"), want, rtol=1e-12)
    assert torch.allclose(ck_final(torch.from_numpy(p), torch.from_numpy(im), "plugin"), torch.from_numpy(want))
    sa = select_all(rng.normal(size=(6, 16)), im, p, beta=1.0, w="plugin")
    assert np.array_equal(sa["a"], np.argmax(want, -1))
    v2f, p96 = rng.normal(size=(6, 16)), rng.uniform(0.01, 0.99, size=(6, 96, 5))
    valid = np.ones((6, 96), bool)
    assert np.array_equal(S2.select96(v2f, im, p96, valid, 1.0, w="plugin"),
                          S2.select96(v2f, im, p96, valid, 1.0, w=(0.0, 1.0, 1.0, 1.0)))
    assert np.array_equal(S2.select96(v2f, im, p96, valid, 0.5), S2.select96(v2f, im, p96, valid, 0.5, w=Cn.SEL_W))
    # eval_ck2: default output unchanged, plugin to its own directory, navtest needs a val choice of the same set
    root, run = tmp_path / "root", "ck2e2e_fake"
    _fake_root(root, run, "navtrain_val", seed=0)
    _fake_root(root, run, "navtest", seed=1)
    lead = tmp_path / "lead"
    assert EV.out_name("navtrain_val") == "navtrain_val" and EV.out_name("navtest", "plugin") == "navtest__selw_plugin"
    m0 = EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=lead)
    mp = EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=lead, sel_w="plugin")
    assert m0["sel_w"] == "default" and m0["sel_w_values"] == [0.1, 0.5, 0.5, 1.0]
    assert mp["sel_w"] == "plugin" and mp["sel_w_values"] == [0.0, 1.0, 1.0, 1.0]
    d = root / "eval" / run
    assert (d / "navtrain_val" / "metrics.json").is_file() and (d / "navtrain_val__selw_plugin" / "metrics.json").is_file()
    with pytest.raises(SystemExit, match="selection weights"):
        EV.run_eval(root, run, "navtest", fix_from=str(d / "navtrain_val" / "metrics.json"), n_boot=20,
                    lead_root=lead, sel_w="plugin")
    t = EV.run_eval(root, run, "navtest", fix_from=str(d / "navtrain_val__selw_plugin" / "metrics.json"), n_boot=20,
                    lead_root=lead, sel_w="plugin")
    assert t["sel_w"] == "plugin" and (d / "navtest__selw_plugin" / "metrics.json").is_file()
    assert EV.main(["--root", str(root), "--run", run, "--split", "navtrain_val", "--bootstrap", "10",
                    "--sel-w", "noim", "--out", str(tmp_path / "cli")]) == 0
    assert json.loads((tmp_path / "cli" / "metrics.json").read_text())["sel_w"] == "noim"


# ----------------------------------------------------------------------------------------------- frozen eval spec (F3)
def _offset_root(root: Path, run: str, split: str, n: int = 20, meta: dict = None):
    """r48_check/F3 fixture: column 0 (v2) PDMS 0.5, column 6 (identity k 1) PDMS 1.0; without an offset the CK score
    picks column 0, with an NC logit offset +3 it picks column 6 (every other column hopeless).  meta: written to the
    packed / infer meta.json (ckpt_sha16, limit, ...)."""
    pk, inf, lb = root / "packed" / split, root / "infer" / run / split, root / "labels" / split
    for d in (pk, inf, lb / f"pool96_{run}", lb / f"lat96_{run}"):
        d.mkdir(parents=True, exist_ok=True)
    toks = [f"t{split[:3]}{i:04d}" for i in range(n)]
    pd.DataFrame({"token": toks, "log": [f"log{i % 5}" for i in range(n)], "city": "x", "row": np.arange(n)}
                 ).to_parquet(pk / "tokens.parquet")
    valid = np.zeros((n, 96), bool)
    valid[:, ::6] = True
    np.save(pk / "valid96.npy", valid)
    np.save(pk / "v2_final.npy", np.tile(np.linspace(0, -1, 16), (n, 1)).astype(np.float32))
    np.save(pk / "v2_im.npy", np.full((n, 16), 1 / 16, np.float32))
    np.save(pk / "ok.npy", np.ones(n, bool))
    logit = np.full((n, 96, 5), -8.0, np.float32)
    logit[:, 0] = [4.0, 4, 1, 1, 1]
    logit[:, 6] = [-2.0, 6, 6, 6, 6]
    np.save(inf / "score_logit.npy", logit)
    np.save(inf / "done.npy", np.ones(n, bool))
    lab = np.zeros((n, 96, 9), np.float32)
    lab[:, 0, Cn.LBL["pdms"]] = 0.5
    lab[:, 6, Cn.LBL["pdms"]] = 1.0
    for name in (f"pool96_{run}", f"lat96_{run}"):
        np.save(lb / name / "labels.npy", lab)
        np.save(lb / name / "ok.npy", np.ones((n, 96), bool))
    if meta is not None:
        for d in (pk, inf):
            (d / "meta.json").write_text(json.dumps(meta))


def _rep(m):
    return next(r for r in m["rows"] if r["key"] == m["representative"])["pdms"]


def test_frozen_spec_calibration_restored_and_conflicts_refused(tmp_path):
    """report 48 F3: the val result stores the calibration offset VALUES (+ source sha) in a frozen spec; navtest with
    only --fix-from reproduces the val pick (before: offset silently OFF, PDMS 0.5 instead of 1.0); a different or
    'none' offset on navtest is refused, an offset file edited after val is refused (or the stored values are used
    when --calib-offset is omitted); calibrated runs write their own <split>__cal_<sha8> dirs."""
    from tools.ck.e2e2 import eval_ck2 as EV
    root, run, lead = tmp_path / "root", "fakeA", tmp_path / "nolead"
    meta = {"ckpt": "/x/last.ckpt", "ckpt_sha16": "a" * 16, "limit": 0}
    _offset_root(root, run, "navtrain_val", meta=meta)
    _offset_root(root, run, "navtest", meta=meta)
    off = tmp_path / "offset_nc3.json"
    off.write_text(json.dumps({"offset": [3.0, 0, 0, 0, 0]}))
    m0 = EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=lead)                   # OFF -> column 0
    assert _rep(m0) == pytest.approx(0.5) and m0["calib_offset_values"] is None
    mv = EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=lead, calib_offset=str(off))
    tag = EV.offset_tag([3.0, 0, 0, 0, 0])
    assert tag.startswith("__cal_") and Path(mv["out_dir"]).name == "navtrain_val" + tag
    assert _rep(mv) == pytest.approx(1.0) and mv["calib_offset_values"] == [3.0, 0, 0, 0, 0]
    spec = mv["eval_spec"]
    assert spec["calib_offset_values"] == [3.0, 0, 0, 0, 0] and spec["calib_offset_src"]["sha16"] == \
        EV.U.sha256_file(off) and spec["model"]["ckpt_sha16"] == "a" * 16 and spec["spec_sha16"]
    assert json.loads((Path(mv["out_dir"]) / "eval_spec.json").read_text()) == spec
    # the uncalibrated val result is still there (an offset ablation does not overwrite it)
    assert json.loads((root / "eval" / run / "navtrain_val" / "metrics.json").read_text())["calib_offset_values"] is None
    vm = str(Path(mv["out_dir"]) / "metrics.json")
    t = EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead)          # offset inherited
    assert _rep(t) == pytest.approx(1.0) and t["calib_offset_values"] == [3.0, 0, 0, 0, 0]
    assert Path(t["out_dir"]).name == "navtest" + tag and t["eval_spec"]["spec_sha16"] == spec["spec_sha16"]
    assert EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead,
                       calib_offset=str(off))["calib_offset_values"] == [3.0, 0, 0, 0, 0]  # same values: accepted
    bad = tmp_path / "offset_neg.json"
    bad.write_text(json.dumps({"offset": [-3.0, 0, 0, 0, 0]}))
    for cal in (str(bad), "none"):
        with pytest.raises(SystemExit, match="calibration offset"):
            EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead, calib_offset=cal)
    off.write_text(json.dumps({"offset": [0.0, 0, 0, 0, 0]}))                            # file edited after val
    with pytest.raises(SystemExit, match="calibration offset"):
        EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead, calib_offset=str(off))
    assert _rep(EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead)) == pytest.approx(1.0)
    # a val result without a frozen spec / with an edited spec is refused
    d = json.loads(Path(vm).read_text())
    for name, mod in (("nospec", lambda x: x.pop("eval_spec")),
                      ("edited", lambda x: x["eval_spec"]["best"].update(beta=0.0))):
        dd = json.loads(json.dumps(d))
        mod(dd)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(dd))
        with pytest.raises(SystemExit, match="eval_spec"):
            EV.run_eval(root, run, "navtest", fix_from=str(p), n_boot=20, lead_root=lead)


def test_stage_eval_ablation_cannot_replace_the_navtest_spec(tmp_path):
    """report 48 F3-a, through eval_e2e2.stage_eval (the planned commands): val OFF and val ON go to different dirs, the
    OFF result is byte-identical afterwards, navtest without --calib-offset fixes from the OFF result (its pick), with
    the offset file from the ON result (its pick)."""
    import argparse
    import hashlib

    from tools.ck.e2e2 import eval_ck2 as EV
    from tools.ck.e2e2 import eval_e2e2 as E2
    run_dir = tmp_path / "ck2e2e_main"
    Pv = E2.Paths(run_dir, "navtrain_val")
    for split in ("navtrain_val", "navtest"):
        _offset_root(Pv.root, Pv.name, split, meta={"ckpt_sha16": "c" * 16, "limit": 0})
    off = tmp_path / "offset_nc3.json"
    off.write_text(json.dumps({"offset": [3.0, 0, 0, 0, 0]}))
    ns = lambda cal: argparse.Namespace(fix_from="", bootstrap=20, calib_offset=cal, choose_lat="on",  # noqa: E731
                                        sel_w=None)
    a_off = E2.stage_eval(Pv, ns(""))
    h = hashlib.sha256(a_off.read_bytes()).hexdigest()
    a_on = E2.stage_eval(Pv, ns(str(off)))
    assert a_on != a_off and a_on.parent.name == "navtrain_val" + EV.offset_tag([3.0, 0, 0, 0, 0])
    assert hashlib.sha256(a_off.read_bytes()).hexdigest() == h
    Pt = E2.Paths(run_dir, "navtest")
    t_off = json.loads(E2.stage_eval(Pt, ns("")).read_text())
    assert t_off["fix_from"] == str(a_off) and _rep(t_off) == pytest.approx(0.5) and t_off["calib_offset_values"] is None
    t_on = json.loads(E2.stage_eval(Pt, ns(str(off))).read_text())
    assert t_on["fix_from"] == str(a_on) and _rep(t_on) == pytest.approx(1.0)
    assert t_on["calib_offset_values"] == [3.0, 0, 0, 0, 0]
    assert (Pt.out.parent / "navtest").is_dir() and (Pt.out.parent / ("navtest" + EV.offset_tag([3.0, 0, 0, 0, 0]))).is_dir()


def test_frozen_spec_weights_model_and_run_checks(tmp_path, monkeypatch):
    """report 48 F3-b / F3-c / F1-c: the sel_w VALUES are frozen (a named set redefined between val and navtest is
    refused; omitted --sel-w inherits the stored values); navtest refuses another run's choice, a dump of another
    checkpoint sha16, another ep_target of the extracted CK run, and a --limit val choice for a full navtest."""
    from tools.ck.e2e2 import eval_ck2 as EV
    root, run, lead = tmp_path / "root", "runX", tmp_path / "nolead"
    _fake_root(root, run, "navtrain_val", seed=0)
    _fake_root(root, run, "navtest", seed=1)
    for split, sha in (("navtrain_val", "a" * 16), ("navtest", "a" * 16)):
        for d in (root / "packed" / split, root / "infer" / run / split):
            (d / "meta.json").write_text(json.dumps({"ckpt": "/r/last.ckpt", "ckpt_sha16": sha, "limit": 0}))
    (root.parent / "ck_run").mkdir(parents=True, exist_ok=True)
    (root.parent / "ck_run" / "config.json").write_text(json.dumps({"ck_e2e2": {"ep_target": "decoupled"}}))
    mp = EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=lead, sel_w="plugin")
    assert mp["eval_spec"]["model"]["ep_target"] == "decoupled" and mp["eval_spec"]["model"]["ckpt_sha16"] == "a" * 16
    vm = mp["out_dir"] + "/metrics.json"
    t = EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead)           # --sel-w inherited
    assert t["sel_w"] == "plugin" and t["sel_w_values"] == mp["sel_w_values"] == [0.0, 1.0, 1.0, 1.0]
    assert t["model"]["ckpt_sha16"] == "a" * 16
    with monkeypatch.context() as mpatch:              # 'plugin' redefined after the val choice
        mpatch.setitem(Cn.SEL_W_SETS, "plugin", (0.0, 2.0, 2.0, 1.0))
        with pytest.raises(SystemExit, match="selection weights"):
            EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead, sel_w="plugin")
        t2 = EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead, out=tmp_path / "t2")
        assert t2["sel_w_values"] == [0.0, 1.0, 1.0, 1.0]                                # the frozen values
    # another run's choice
    _fake_root(root, "runY", "navtest", seed=1)
    with pytest.raises(SystemExit, match="runX"):
        EV.run_eval(root, "runY", "navtest", fix_from=vm, n_boot=20, lead_root=lead)
    # navtest dumped from another checkpoint
    for d in (root / "packed" / "navtest", root / "infer" / run / "navtest"):
        (d / "meta.json").write_text(json.dumps({"ckpt": "/r/epoch=9.ckpt", "ckpt_sha16": "b" * 16, "limit": 0}))
    with pytest.raises(SystemExit, match="same checkpoint"):
        EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead)
    for d in (root / "packed" / "navtest", root / "infer" / run / "navtest"):
        (d / "meta.json").write_text(json.dumps({"ckpt": "/r/last.ckpt", "ckpt_sha16": "a" * 16, "limit": 0}))
    # ep_target of the extracted CK run changed
    (root.parent / "ck_run" / "config.json").write_text(json.dumps({"ck_e2e2": {"ep_target": "official"}}))
    with pytest.raises(SystemExit, match="ep_target"):
        EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead)
    (root.parent / "ck_run" / "config.json").write_text(json.dumps({"ck_e2e2": {"ep_target": "decoupled"}}))
    # a --limit val choice cannot fix a full navtest
    for d in (root / "packed" / "navtrain_val", root / "infer" / run / "navtrain_val"):
        (d / "meta.json").write_text(json.dumps({"ckpt": "/r/last.ckpt", "ckpt_sha16": "a" * 16, "limit": 50}))
    ml = EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=lead, out=tmp_path / "val_lim")
    with pytest.raises(SystemExit, match="limit"):
        EV.run_eval(root, run, "navtest", fix_from=str(tmp_path / "val_lim" / "metrics.json"), n_boot=20,
                    lead_root=lead)
    assert ml["eval_spec"]["model"]["limit"] == 50
    # packed and infer meta of one split disagreeing (two dumps mixed) is refused
    (root / "packed" / "navtest" / "meta.json").write_text(json.dumps({"ckpt_sha16": "c" * 16}))
    with pytest.raises(SystemExit, match="disagree"):
        EV.run_eval(root, run, "navtest", fix_from=vm, n_boot=20, lead_root=lead)
