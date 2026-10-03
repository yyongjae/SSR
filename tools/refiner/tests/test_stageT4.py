"""Tests for the run-4 changes (PRESTATED_DECISION_RULE AMENDMENT 6): arms M / TM (adapters, net, data, training,
evaluation incl. shuffle / branch drop / eval dirs / token subsets), the multi-pair decision with a CI level, the GPU
script guards and the EPDMS tool's non-scorer parts.  Synthetic data only (fake BEVFusion + fake ReSMap caches); CPU.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_stageT4.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE), str(HERE.parent)):
    sys.path.insert(0, p)

from navsim.agents.para_ssr.refiner import adapters as AD  # noqa: E402
from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner import refiner_net as RN  # noqa: E402
from navsim.agents.para_ssr.refiner import resmap_cache as RM  # noqa: E402
import epdms_navtest as EP  # noqa: E402
import eval_refiner as EV  # noqa: E402
import refiner_synth as SY  # noqa: E402
import stageT_decision as SD  # noqa: E402
import train_refiner as TR  # noqa: E402
from test_stageT_decision import synth_rows, write_eval  # noqa: E402

QUIET = lambda *a, **k: None
RNG = np.random.default_rng(0)
NORM_D = (RNG.uniform(0.1, 0.4, 256).astype(np.float32), RNG.uniform(0.3, 0.9, 256).astype(np.float32))
NORM_M = (RNG.normal(0, 0.2, 256).astype(np.float32), RNG.uniform(0.5, 1.5, 256).astype(np.float32))
N_TOK = 12


# ----------------------------------------------------------------------------------------------- adapters / net
def _nets(seed=0):
    return {"T": RN.RefinerNet("T", *NORM_D, seed=seed), "none": RN.RefinerNet("none", seed=seed),
            "M": RN.RefinerNet("M", seed=seed, map_norm_mean=NORM_M[0], map_norm_std=NORM_M[1]),
            "TM": RN.RefinerNet("TM", *NORM_D, seed=seed, map_norm_mean=NORM_M[0], map_norm_std=NORM_M[1])}


def test_param_counts_and_trunk_identity():
    nets = _nets(0)
    c = {k: n.param_counts() for k, n in nets.items()}
    assert c["T"]["adapter"] == c["M"]["adapter"] == 256 * 128 + 128 + 128 * 64 + 64 == 41152
    assert c["TM"]["adapter"] == 2 * 41152 + 128 * 64 + 64 == 90560 and c["none"]["adapter"] == 0
    assert len({v["trunk"] for v in c.values()}) == 1
    ref = dict(nets["none"].trunk_named_parameters())
    for arm in ("T", "M", "TM"):
        tr = dict(nets[arm].trunk_named_parameters())
        assert tr.keys() == ref.keys() and all(torch.equal(tr[k], ref[k]) for k in ref), arm
    for arm in ("M", "TM"):                                            # normalisation = buffers, not parameters
        assert not any("norm." in n for n, _ in nets[arm].adapter.named_parameters())


@pytest.mark.parametrize("seed", [0, 5])
def test_tm_branch_init_equals_single_teacher_adapters(seed):
    nets = _nets(seed)
    t, m, tm = nets["T"].adapter, nets["M"].adapter, nets["TM"].adapter
    for a, b in ((t, tm.det), (m, tm.map)):
        pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
        assert pa.keys() == pb.keys() and all(torch.equal(pa[k], pb[k]) for k in pa)
    assert not torch.equal(t.conv1.weight, m.conv1.weight)             # M has its own adapter seed
    assert torch.equal(tm.det.norm.mean, torch.as_tensor(NORM_D[0])) and torch.equal(tm.map.norm.mean, torch.as_tensor(NORM_M[0]))
    # T adapter unchanged vs the pre-run-4 construction (torch.manual_seed(seed + offset); AdapterT(...))
    torch.manual_seed(seed + RN.ADAPTER_SEED_OFFSET)
    legacy = AD.AdapterT(*NORM_D)
    assert all(torch.equal(p, q) for p, q in zip(legacy.parameters(), t.parameters()))
    # the global RNG is untouched by building any arm
    torch.manual_seed(3)
    x0 = torch.rand(4)
    torch.manual_seed(3)
    _nets(seed)
    assert torch.equal(torch.rand(4), x0)


def test_tm_forward_and_drop_branch():
    ad = AD.build_adapter("TM", *NORM_D, *NORM_M, seed=11)
    torch.manual_seed(0)
    xd = torch.relu(torch.randn(2, 256, 50, 100))
    xm = torch.randn(2, 256, 50, 100)
    x = torch.cat([xd, xm], 1).half()
    y = ad(x)
    assert y.shape == (2, 64, 50, 100) and torch.isfinite(y).all()
    manual = ad.fuse(torch.cat([ad.det(x[:, :256]), ad.map(x[:, 256:])], 1))
    torch.testing.assert_close(y, manual, rtol=0, atol=0)
    for which, part in (("det", slice(0, 256)), ("map", slice(256, 512))):
        ad.set_drop_branch(which)
        yd = ad(x)
        x2 = x.clone()
        x2[:, part] = torch.randn_like(x2[:, part].float()).half()
        assert torch.equal(ad(x2), yd)                                  # the dropped input does not matter
        assert not torch.allclose(yd, y)
        # == feeding that branch its training-mean feature (normalised input 0), up to float rounding
        x3 = x.float().clone()
        x3[:, part] = torch.as_tensor(NORM_D[0] if which == "det" else NORM_M[0])[None, :, None, None]
        ad.set_drop_branch(None)
        torch.testing.assert_close(ad(x3), yd, rtol=1e-5, atol=1e-5)
    assert not any("drop" in k for k in ad.state_dict())
    with pytest.raises(ValueError):
        ad.set_drop_branch("both")
    with pytest.raises(ValueError):
        ad(x[:, :256])
    with pytest.raises(RuntimeError, match="map normalisation"):
        AD.build_adapter("TM", *NORM_D)(x)
    with pytest.raises(RuntimeError, match="norm_map"):
        AD.build_adapter("M")(x[:, 256:])
    assert AD.build_adapter("M", map_mean=NORM_M[0], map_std=NORM_M[1])(x[:, 256:]).shape == (2, 64, 50, 100)


def test_identity_at_init_for_new_arms():
    z = np.load(HERE / "fixtures" / "decoder_navtest_sample.npz")
    tau = torch.as_tensor(z["tau_h"][:8].reshape(2, 4, 8, 3))
    T = 2
    v0 = torch.full((T,), 5.0)
    e = (v0, torch.zeros(T), torch.stack([v0, torch.zeros(T), torch.zeros(T), torch.zeros(T)], -1), torch.arange(T))
    nets = _nets(0)
    for arm, ch in (("M", 256), ("TM", 512)):
        with torch.no_grad():
            o = nets[arm].eval()(torch.randn(T, ch, 50, 100).half(), tau, *e)
        assert not o["z_lon"].any() and not o["w_lat"].any() and o["gate_logit"].shape == (T, 4)
        assert nets[arm].needs_bev


# ----------------------------------------------------------------------------------------------- synthetic env
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("t4")
    df, src = SY.make_sources(tmp, N_TOK)
    for split in ("train", "dev"):
        RD.pack_split(split, df, tmp / "packed", sources=src, workers=1, log_fn=QUIET)
    teacher = SY.make_fake_teacher(tmp, df.token)
    covered = list(df.token[:-1])                                     # the last token is NOT in the ReSMap cache
    resmap = SY.make_fake_resmap(tmp, covered)
    sub = tmp / "train_sub.parquet"
    df[df.token.isin(covered)].to_parquet(sub, index=False)
    full = tmp / "train_full.parquet"
    df.to_parquet(full, index=False)
    dev_sub = tmp / "dev_sub.parquet"
    df[df.token.isin(covered[3:])].to_parquet(dev_sub, index=False)
    e = dict(tmp=tmp, df=df, teacher=teacher, resmap=resmap, sub=sub, full=full, dev_sub=dev_sub, covered=covered)
    runs = {}
    for arm in ("T", "M", "TM", "none"):
        runs[arm] = TR.train(targs(e, arm, "--token-subset", str(sub), "--dev-token-subset", str(dev_sub)))
    runs["M_nodev"] = TR.train(targs(e, "M", "--token-subset", str(sub), tag="t4nodev"))
    e["runs"] = runs
    return e


def targs(e, arm, *extra, tag="t4"):
    tmp = e["tmp"]
    a = ["--arm", arm, "--fold", "0", "--seed", "0", "--gpu", "-1", "--packed-root", str(tmp / "packed"),
         "--runs", str(tmp / "runs"), "--teacher-root", str(e["teacher"]), "--resmap-root", str(e["resmap"]),
         "--tokens-per-batch", "3", "--workers", "0", "--n-norm", "4", "--log-every", "1", "--surrogate", "stub",
         "--inner-val-frac", "0.3", "--max-steps", "2", "--epochs", "1", "--max-val-batches", "1", "--tag", tag]
    return TR.get_parser().parse_args(a + list(extra))


def test_concat_teacher_and_arm_teachers(env):
    tk = env["covered"][2]
    t, det, mp = RD.arm_teachers("TM", "train", env["teacher"], env["resmap"])
    x = t.load_bev(tk)
    assert x.shape == (512, 50, 100) and x.dtype == np.float16
    assert np.array_equal(x[:256], det.load_bev(tk)) and np.array_equal(x[256:], mp.load_bev(tk))
    assert np.array_equal(x[256:], np.swapaxes(SY.fake_resmap_bev(tk), 1, 2))          # S grid = transpose
    assert isinstance(RD.arm_teachers("M", "train", None, env["resmap"])[0], RM.ResmapCache)
    assert RD.arm_teachers("none", "train") == (None, None, None)
    ds = RD.TokenDataset(RD.PackedSplit("train", env["tmp"] / "packed"), [0, 1], t)
    b = RD.collate_tokens([ds[0], ds[1]])
    assert b["bev"].shape == (2, 512, 50, 100) and b["bev"].dtype == torch.float16
    with pytest.raises(ValueError):
        t.load_bev(tk, s_grid=False)


def test_load_token_subset_and_select_rows(env):
    tok, info = RD.load_token_subset(env["sub"])
    assert info["sha256"] == hashlib.sha256(Path(env["sub"]).read_bytes()).hexdigest()
    assert info["n_rows"] == info["n_tokens"] == N_TOK - 1 and len(tok) == N_TOK - 1
    P = RD.PackedSplit("train", env["tmp"] / "packed")
    a0, b0 = TR.select_rows(P, 0, 0.3)
    a1, b1 = TR.select_rows(P, 0, 0.3, subset_tokens=None)
    assert np.array_equal(a0, a1) and np.array_equal(b0, b1)          # no subset == runs 1-3 exactly
    tr, va = TR.select_rows(P, 0, 0.3, subset_tokens=tok)
    toks = set(P.index.token.values[np.concatenate([tr, va])])
    assert toks <= set(tok) and env["df"].token.iloc[-1] not in toks
    assert 0 not in set(P.index.fold.values[np.concatenate([tr, va])])
    assert not set(P.index.log.values[tr]) & set(P.index.log.values[va])
    tr2, va2 = TR.select_rows(P, 0, 0.3, subset_tokens=tok)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)


def test_train_new_arms_norms_and_config(env):
    R = env["runs"]
    cfg = {k: json.loads((R[k] / "config.json").read_text()) for k in ("T", "M", "TM", "none")}
    assert (R["T"] / "norm.npz").exists() and not (R["T"] / "norm_map.npz").exists()
    assert (R["M"] / "norm_map.npz").exists() and not (R["M"] / "norm.npz").exists()
    assert (R["TM"] / "norm.npz").exists() and (R["TM"] / "norm_map.npz").exists()
    assert not (R["none"] / "norm.npz").exists()
    for a, b, f in (("TM", "T", "norm.npz"), ("TM", "M", "norm_map.npz")):
        ma, sa, ia = AD.load_norm(R[a] / f)
        mb, sb, ib = AD.load_norm(R[b] / f)
        assert np.array_equal(ma, mb) and np.array_equal(sa, sb) and ia["token_hash"] == ib["token_hash"]
    _, _, im = AD.load_norm(R["M"] / "norm_map.npz")
    assert im["sha_head"] == RM.RESMAP_SHA_HEAD and im["branch"] == "map" and im["teacher_root"] == str(env["resmap"])
    sha = hashlib.sha256(Path(env["sub"]).read_bytes()).hexdigest()
    for arm, c in cfg.items():
        ts = c["token_subset"]
        assert ts["sha256"] == sha and ts["n_rows"] == N_TOK - 1
        assert ts["n_train_rows"] == c["n_train_tokens"] and ts["n_ival_rows"] == c["n_ival_tokens"]
        assert c["dev_token_subset"]["sha256"] == hashlib.sha256(Path(env["dev_sub"]).read_bytes()).hexdigest()
    assert cfg["M"]["param_counts"]["adapter"] == 41152 and cfg["TM"]["param_counts"]["adapter"] == 90560
    assert cfg["M"]["resmap_sha_head"] == cfg["TM"]["resmap_sha_head"] == RM.RESMAP_SHA_HEAD
    assert cfg["TM"]["teacher_sha_head"] == RD.TEACHER_SHA_HEAD and cfg["TM"]["norm_files"] == {
        "norm.npz": RD.TEACHER_SHA_HEAD, "norm_map.npz": RM.RESMAP_SHA_HEAD}
    net, c = TR.load_run_model(R["TM"])
    assert c["arm"] == "TM" and bool(net.adapter.det.norm.is_set) and bool(net.adapter.map.norm.is_set)
    log = [json.loads(x) for x in (R["TM"] / "log.jsonl").read_text().splitlines()]
    assert any(r["kind"] == "epoch" for r in log) and all(np.isfinite(r.get("loss", 0.0)) for r in log)


def test_train_refusals(env):
    with pytest.raises(SystemExit, match="needs --token-subset"):
        TR.train(targs(env, "M", tag="bad1"))
    with pytest.raises(SystemExit, match="not in the ReSMap cache"):
        TR.train(targs(env, "TM", "--token-subset", str(env["full"]), tag="bad2"))
    assert not (env["tmp"] / "runs" / "bad2_TM_fold0_seed0" / "config.json").exists()
    # resuming / re-launching a run with another subset (or without one) is refused, even when finished
    with pytest.raises(SystemExit, match="token_subset sha256"):
        TR.check_resume_config(env["runs"]["T"], targs(env, "T", "--token-subset", str(env["full"]),
                                                       "--dev-token-subset", str(env["dev_sub"])))
    with pytest.raises(SystemExit, match="token_subset sha256"):
        TR.check_resume_config(env["runs"]["T"], targs(env, "T", "--dev-token-subset", str(env["dev_sub"])))
    TR.check_resume_config(env["runs"]["T"], targs(env, "T", "--token-subset", str(env["sub"]), "--dev-token-subset",
                                                   str(env["dev_sub"])))


# ----------------------------------------------------------------------------------------------- evaluation
def ev(env, run, *extra, split="dev"):
    a = EV.get_parser().parse_args(["predict", "--run", str(run), "--split", split, "--packed-root",
                                    str(env["tmp"] / "packed"), "--loader-workers", "0", "--teacher-root",
                                    str(env["teacher"]), "--resmap-root", str(env["resmap"])] + list(extra))
    EV.resolve_fold(a)
    return a


def test_eval_subsets(env):
    R = env["runs"]
    P = RD.PackedSplit("train", env["tmp"] / "packed")
    o = EV.predict(ev(env, R["M"], split="train"))
    assert o.name == "eval_train_fold0"
    pred = np.load(o / "pred.npz")
    assert set(pred["tokens"]) <= set(env["covered"]) and set(P.index.fold.values[pred["rows"]]) == {0}
    meta = json.loads((o / "predict_meta.json").read_text())
    assert meta["token_subset"]["source"].startswith("run config.json token_subset")
    assert meta["token_subset"]["n_eval_tokens"] == len(pred["tokens"])
    # an explicit, different train subset is refused
    with pytest.raises(SystemExit, match="sha256"):
        EV.predict(ev(env, R["M"], "--token-subset", str(env["full"]), "--eval-name", "x_train", split="train"))
    # dev: default = the run's dev_token_subset
    o = EV.predict(ev(env, R["TM"]))
    toks = set(np.load(o / "pred.npz")["tokens"])
    assert toks == set(env["covered"][3:])
    assert json.loads((o / "predict_meta.json").read_text())["token_subset"]["source"] == \
        "run config.json dev_token_subset"
    # explicit dev subset
    o = EV.predict(ev(env, R["M_nodev"], "--token-subset", str(env["sub"]), "--eval-name", "eval_dev_sub"))
    assert o.name == "eval_dev_sub" and len(np.load(o / "pred.npz")["tokens"]) == N_TOK - 1
    # no subset and a dev token outside the ReSMap cache -> refused
    with pytest.raises(SystemExit, match="not in the ReSMap cache"):
        EV.predict(ev(env, R["M_nodev"], "--eval-name", "eval_dev_all"))
    with pytest.raises(SystemExit, match="navtest is never subset"):
        EV.check_run4_eval_args(ev(env, R["M"], "--token-subset", str(env["sub"]), split="navtest"), {"arm": "M"})


def test_eval_shuffle_which_equals_permuted_cache(env):
    R = env["runs"]
    o_true = EV.predict(ev(env, R["TM"], "--eval-name", "tm_true"))
    o_shuf = EV.predict(ev(env, R["TM"], "--shuffle-teacher-seed", "0", "--shuffle-which", "map", "--eval-name",
                           "tm_shufmap"))
    meta = json.loads((o_shuf / "predict_meta.json").read_text())
    assert meta["teacher_shuffle"]["which"] == "map" and meta["teacher_shuffle"]["fixed_points"] == 0
    assert meta["eval_name"] == "tm_shufmap"
    mapping = json.loads((o_shuf / "teacher_shuffle_map.json").read_text())
    perm = SY.make_fake_resmap(env["tmp"], env["covered"], name="resmap_perm", source=mapping)
    a = ev(env, R["TM"], "--eval-name", "tm_perm")
    a.resmap_root = str(perm)
    o_perm = EV.predict(a)
    T_, S_, P_ = (np.load(o / "pred.npz") for o in (o_true, o_shuf, o_perm))
    for k in ("tokens", "rows", "tau0"):
        assert np.array_equal(T_[k], S_[k])
    assert np.array_equal(S_["p_g"], P_["p_g"]) and np.array_equal(S_["tau1"], P_["tau1"])
    assert not np.array_equal(T_["p_g"], S_["p_g"])
    # defaults: TM -> both, M -> map
    o = EV.predict(ev(env, R["TM"], "--shuffle-teacher-seed", "0", "--eval-name", "tm_shufboth"))
    assert json.loads((o / "predict_meta.json").read_text())["teacher_shuffle"]["which"] == "both"
    o = EV.predict(ev(env, R["M"], "--shuffle-teacher-seed", "0", "--eval-name", "m_shuf"))
    assert json.loads((o / "predict_meta.json").read_text())["teacher_shuffle"]["which"] == "map"
    o = EV.predict(ev(env, R["T"], "--shuffle-teacher-seed", "0", "--eval-name", "t_shuf"))
    assert "which" not in json.loads((o / "predict_meta.json").read_text())["teacher_shuffle"]   # runs 1-3 format


def test_eval_drop_branch(env):
    R = env["runs"]
    o_true = EV.predict(ev(env, R["TM"], "--eval-name", "tm_true2"))
    o_drop = EV.predict(ev(env, R["TM"], "--drop-branch", "map", "--eval-name", "tm_dropmap"))
    o_drop_shuf = EV.predict(ev(env, R["TM"], "--drop-branch", "map", "--shuffle-teacher-seed", "1", "--shuffle-which",
                                "map", "--eval-name", "tm_dropmap_shuf"))
    D, DS, T_ = (np.load(o / "pred.npz") for o in (o_drop, o_drop_shuf, o_true))
    assert np.array_equal(D["p_g"], DS["p_g"]) and np.array_equal(D["tau1"], DS["tau1"])   # map input irrelevant
    assert not np.array_equal(D["p_g"], T_["p_g"])
    assert json.loads((o_drop / "predict_meta.json").read_text())["drop_branch"] == "map"
    assert "drop_branch" not in json.loads((o_true / "predict_meta.json").read_text())


@pytest.mark.parametrize("arm,extra,msg", [
    ("M", ["--drop-branch", "det", "--eval-name", "z"], "arm TM only"),
    ("T", ["--shuffle-teacher-seed", "0", "--shuffle-which", "map", "--eval-name", "z"], "does not apply"),
    ("M", ["--shuffle-teacher-seed", "0", "--shuffle-which", "det", "--eval-name", "z"], "does not apply"),
    ("none", ["--shuffle-teacher-seed", "0", "--eval-name", "z"], "teacher arm"),
    ("TM", ["--shuffle-teacher-seed", "0"], "needs --eval-name"),
    ("TM", ["--drop-branch", "det"], "needs --eval-name"),
    ("TM", ["--drop-branch", "det", "--eval-name", "eval_dev"], "default evaluation directory"),
    ("TM", ["--shuffle-teacher-seed", "0", "--eval-name", "eval_navtest"], "default evaluation directory"),
    ("TM", ["--shuffle-which", "det", "--eval-name", "z"], "needs --shuffle-teacher-seed"),
    ("TM", ["--eval-name", "z", "--out", "/nonexistent/x"], "mutually exclusive"),
])
def test_eval_refusals(env, arm, extra, msg):
    run = env["runs"][arm]
    before = sorted(p.name for p in run.iterdir())
    with pytest.raises(SystemExit, match=msg):
        EV.predict(ev(env, run, *extra))
    assert sorted(p.name for p in run.iterdir()) == before


# ----------------------------------------------------------------------------------------------- decision
def test_ci_percentiles_and_level():
    assert SD.ci_percentiles(0.95) == [2.5, 97.5] and SD.ci_percentiles(0.9875) == [0.625, 99.375]
    with pytest.raises(ValueError):
        SD.ci_percentiles(1.0)
    x = np.random.default_rng(0).normal(0, 1, 500)
    a = SD.cluster_boot(x, np.arange(500), 2000)
    b = SD.cluster_boot(x, np.arange(500), 2000, level=0.95)
    c = SD.cluster_boot(x, np.arange(500), 2000, level=0.9875)
    assert a == b and c["lo"] < a["lo"] and c["hi"] > a["hi"]


def _four_arm_runs(tmp, same_tm_t=False):
    runs = tmp / "runs"
    fix = {"T": 0.8, "M": 0.6, "TM": 0.8, "none": 0.3}
    for arm, f in fix.items():
        seed_arm = "T" if (same_tm_t and arm == "TM") else arm
        off = {"T": 0, "M": 1, "TM": 2, "none": 3}[seed_arm]
        rows, toks = synth_rows(n_logs=8, seed=10 + off, fix_prob=f, harm_ep=0.02)
        write_eval(SD.run_dir(runs, "wt", arm, 0, 0) / "eval_train_fold0", rows, toks)
        rows, toks = synth_rows(n_logs=30, seed=100 + off, fix_prob=f, harm_ep=0.0)
        # dev rows share tokens / drafts across arms (same draft bank); only the refined outcomes differ
        write_eval(SD.run_dir(runs, "wt", arm, 0, 0) / "eval_dev", rows, toks)
    return runs


def test_select_and_compare_four_arms(tmp_path):
    runs = _four_arm_runs(tmp_path)
    sel = SD.select(runs, ["wt"], ["T", "M", "TM", "none"], [0], [0], 0.5, budget_def="passing")
    assert all(sel["arms"][a]["theta"] is not None for a in ("T", "M", "TM", "none"))
    pairs = "T:none,M:none,TM:T,TM:M"
    r95 = SD.compare_pairs(runs, sel, pairs, [0], n_boot=1000, final_fold=0, ci_level=0.95)
    r = SD.compare_pairs(runs, sel, pairs, [0], n_boot=1000, final_fold=0, ci_level=0.9875)
    assert list(r["outcomes"]) == ["T:none", "M:none", "TM:T", "TM:M"] and r["ci_percentiles"] == [0.625, 99.375]
    assert set(r["arms"]) == {"T", "M", "TM", "none"}
    for p in r["pairs"]:
        a, b = r["pairs"][p], r95["pairs"][p]
        assert a["a"] + ":" + a["b"] == p and "dac_excess_pp" in a
        for k in a["endpoints"]:
            assert a["endpoints"][k]["mean"] == b["endpoints"][k]["mean"]
            assert a["endpoints"][k]["lo"] <= b["endpoints"][k]["lo"] and a["endpoints"][k]["hi"] >= b["endpoints"][k]["hi"]
    # the T:none pair at 0.95 == the old two-arm compare (same endpoints, sanity, outcome, counts, families)
    old = SD.compare(runs, sel, [0], n_boot=1000, final_fold=0)
    new = r95["pairs"]["T:none"]
    assert old["outcome"] == new["outcome"] and old["endpoints"] == new["endpoints"]
    assert old["sanity"] == new["sanity"] and old["n_paired_drafts"] == new["n_paired_drafts"]
    assert old["arms"]["T"] == r95["arms"]["T"] and old["arms"]["none"] == r95["arms"]["none"]
    for fam, d in old["by_family"].items():
        assert d["d_pdms_points_T_minus_none"] == new["by_family"][fam]["d_pdms_points_T_minus_none"]
    # old CLI without --ci-level: no new key; explicit 0.95 only adds 'ci_level'
    old2 = SD.compare(runs, sel, [0], n_boot=1000, final_fold=0, ci_level=0.95)
    assert "ci_level" not in old and old2.pop("ci_level") == 0.95 and json.dumps(old2) == json.dumps(old)


def test_compare_pairs_identical_arms_equivalent_and_cli(tmp_path):
    runs = _four_arm_runs(tmp_path, same_tm_t=True)
    sel = {"budget_ep_points": 0.5, "budget_def": "passing",
           "arms": {a: {"theta": 0.5, "wtag": "wt"} for a in ("T", "M", "TM", "none")}}
    r = SD.compare_pairs(runs, sel, [("TM", "T")], [0], n_boot=500, final_fold=0, ci_level=0.9875)
    p = r["pairs"]["TM:T"]
    assert p["endpoints"]["P1_d_pdms_points"]["mean"] == 0 == p["endpoints"]["P1_d_pdms_points"]["hi"]
    assert p["outcome"] == "EQUIVALENT" and p["n_unpaired_a"] == p["n_unpaired_b"] == 0
    sp = tmp_path / "sel.json"
    sp.write_text(json.dumps(sel))
    out = tmp_path / "d.json"
    SD.main(["compare", "--runs", str(runs), "--selection", str(sp), "--seeds", "0", "--final-fold", "0", "--n-boot",
             "300", "--pairs", "T:none,M:none,TM:T,TM:M", "--ci-level", "0.9875", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["ci_level"] == 0.9875 and len(res["outcomes"]) == 4 and res["pairs"]["TM:T"]["outcome"] == "EQUIVALENT"
    # ARM@EVAL elements: a control directory of the same run vs its true evaluation
    import shutil
    shutil.copytree(SD.run_dir(runs, "wt", "TM", 0, 0) / "eval_dev", SD.run_dir(runs, "wt", "TM", 0, 0) / "eval_dev_ctl")
    rc = SD.compare_pairs(runs, sel, "TM@eval_dev_ctl:TM,TM@eval_dev_ctl:none", [0], n_boot=300, final_fold=0,
                          ci_level=0.9875)
    assert rc["pairs"]["TM@eval_dev_ctl:TM"]["endpoints"]["P1_d_pdms_points"]["mean"] == 0
    assert rc["arms"]["TM@eval_dev_ctl"]["eval_name"] == "eval_dev_ctl"
    ref = SD.compare_pairs(runs, sel, "TM:none", [0], n_boot=300, final_fold=0, ci_level=0.9875)
    assert rc["pairs"]["TM@eval_dev_ctl:none"]["endpoints"] == ref["pairs"]["TM:none"]["endpoints"]
    with pytest.raises(ValueError):
        SD.parse_pairs("T:T")
    with pytest.raises(ValueError):
        SD.parse_pairs("T:none,T:none")
    with pytest.raises(ValueError, match="budget_def"):
        SD.compare_pairs(runs, sel, "T:none", [0], n_boot=10, final_fold=0, budget_def="all")


# ----------------------------------------------------------------------------------------------- GPU script
def _sandbox_script(tmp_path):
    src = (HERE.parent / "stageT_gpu_commands.sh").read_text()
    rec = tmp_path / "called"
    py = tmp_path / "py.sh"
    py.write_text(f"#!/usr/bin/env bash\necho \"$@\" >> {rec}\n")
    py.chmod(0o755)
    lines = [(f"DATA={tmp_path / 'data'}" if ln.startswith("DATA=") else f"PY={py}" if ln.startswith("PY=") else ln)
             for ln in src.splitlines()]
    sh = tmp_path / "gpu.sh"
    sh.write_text("\n".join(lines) + "\n")
    return sh, rec


def test_gpu_script_run4_guards_and_subsets(tmp_path):
    sh, rec = _sandbox_script(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in ("TAG", "DEV_EVAL")}
    for tag in (None, "stageT", "stageT2", "stageT3"):
        e = env if tag is None else {**env, "TAG": tag}
        for arm, extra in (("T", ["--token-subset", "/x.parquet"]), ("none", ["--dev-token-subset=/y.parquet"]),
                           ("M", []), ("TM", []), ("T", ["--resmap-root", "/r"])):
            r = subprocess.run(["bash", str(sh), "0", arm, "0", "0", *extra], env=e, capture_output=True, text=True)
            assert r.returncode == 2 and "refused" in r.stderr, (tag, arm, extra)
    r = subprocess.run(["bash", str(sh), "0", "X", "0", "0"], env={**env, "TAG": "stageT4"}, capture_output=True,
                       text=True)
    assert r.returncode == 2 and "unknown arm" in r.stderr
    assert not rec.exists() and not (tmp_path / "data").exists()
    r = subprocess.run(["bash", str(sh), "1", "TM", "0", "0", "--token-subset", "/s/train.parquet",
                        "--dev-token-subset=/s/dev.parquet", "--m-ttc", "0.15"],
                       env={**env, "TAG": "stageT4"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    calls = rec.read_text().splitlines()
    assert "--arm TM --fold 0 --seed 0 --gpu 1 --workers 2 --tag stageT4 --token-subset /s/train.parquet " \
           "--dev-token-subset=/s/dev.parquet --m-ttc 0.15" in calls[0]
    assert len(calls) == 2 and "--split train --fold 0 --token-subset /s/train.parquet --gpu 1" in calls[1]
    rec.unlink()
    r = subprocess.run(["bash", str(sh), "1", "M", "0", "0", "--token-subset", "/s/train.parquet",
                        "--dev-token-subset", "/s/dev.parquet"], env={**env, "TAG": "stageT4", "DEV_EVAL": "1"},
                       capture_output=True, text=True)
    calls = rec.read_text().splitlines()
    assert r.returncode == 0 and len(calls) == 3 and "--split dev --token-subset /s/dev.parquet --gpu 1" in calls[2]
    rec.unlink()
    # runs 1-3 style call: argv of every call unchanged (no subset -> nothing added)
    r = subprocess.run(["bash", str(sh), "0", "T", "0", "0"], env={**env, "TAG": "stageT3", "DEV_EVAL": "1"},
                       capture_output=True, text=True)
    calls = rec.read_text().splitlines()
    assert r.returncode == 0 and calls[1].endswith("--split train --fold 0 --gpu 0 --workers 4 --theta 0.5 --sweep "
                                                   "--budget-ep 0.5")
    assert calls[2].endswith("--split dev --gpu 0 --workers 4 --theta 0.5 --sweep --budget-ep 0.5")


# ----------------------------------------------------------------------------------------------- EPDMS tool
def test_epdms_pick_tokens_and_env():
    toks = [f"t{i:04d}" for i in range(300)][::-1]
    a = EP.pick_tokens(toks, 50, 0)
    assert a == EP.pick_tokens(list(reversed(toks)), 50, 0) and len(a) == 50 == len(set(a)) and a == sorted(a)
    assert a != EP.pick_tokens(toks, 50, 1) and EP.pick_tokens(toks, None) == sorted(toks)
    assert EP.pick_tokens(toks, 1000) == sorted(toks)
    e = EP.v2_env({"PYTHONPATH": "/home/external-user/yongjae/SSR", "X": "1"})
    assert e["PYTHONPATH"] == str(EP.NAV) and e[EP.V2_FLAG] == "1" and e["CUDA_VISIBLE_DEVICES"] == "" and e["X"] == "1"


def _fake_eval(run: Path, arm: str, tau0, tau1, p_g, tokens, logs):
    (run / "eval_navtest").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"arm": arm}))
    N, K = p_g.shape
    np.savez(run / "eval_navtest" / "pred.npz", tokens=np.array(tokens), tau0=tau0, tau1=tau1, p_g=p_g,
             draft_valid=np.ones((N, K), bool), family=np.tile(np.arange(K) % 9, (N, 1)).astype(np.int8))
    pd.DataFrame({"token": tokens, "log": logs}).to_parquet(run / "eval_navtest" / "tokens.parquet", index=False)


def test_epdms_load_sets_and_report(tmp_path):
    rng = np.random.default_rng(0)
    N, K = 6, 13
    tokens = [f"tok{i}" for i in range(N)]
    logs = [f"log{i // 2}" for i in range(N)]
    tau0 = rng.normal(size=(N, K, 8, 3)).astype(np.float32)
    runs = {}
    for arm in ("T", "none"):
        tau1 = tau0 + rng.normal(size=tau0.shape).astype(np.float32)
        tau1[:, 0] = tau0[:, 0]                                       # draft 0: zero correction
        runs[arm] = tmp_path / f"r_{arm}"
        _fake_eval(runs[arm], arm, tau0, tau1, rng.uniform(size=(N, K)), tokens, logs)
    S = EP.load_sets([runs["T"], runs["none"]], "eval_navtest", EP.pick_tokens(tokens, 4, 0))
    assert list(S["sets"]) == ["orig", "r_T", "r_none"] and S["sets"]["r_T"]["arm"] == "T"
    assert S["sets"]["r_T"]["copy"][:, 0].all() and not S["sets"]["r_T"]["copy"][:, 1:].any()
    assert len(S["tokens"]) == 4 and S["sets"]["orig"]["traj"].shape == (4, K, 8, 3)
    bad = tmp_path / "r_bad"
    _fake_eval(bad, "M", tau0 + 1, tau0, rng.uniform(size=(N, K)), tokens, logs)
    with pytest.raises(ValueError, match="not the same draft bank"):
        EP.load_sets([runs["T"], bad], "eval_navtest")
    # report on synthetic score rows: orig score 0.5, T tau1 0.9, none tau1 0.6
    rows = []
    for name, arm, sc in (("orig", "orig", 0.5), ("r_T", "T", 0.9), ("r_none", "none", 0.6)):
        for i, t in enumerate(S["tokens"]):
            for k in range(K):
                d = {c: 1.0 for c in EP.METRIC_COLS}
                rows.append(dict(d, set=name, token=t, k=k, score=sc, error=None, copied_from_orig=False))
    df = EP.rows_frame(rows, S)
    d = tmp_path / "scores"
    d.mkdir()
    df.to_parquet(d / "rows.parquet", index=False)
    (d / "meta.json").write_text(json.dumps(dict(arms={"r_T": "T", "r_none": "none"}, runs=[str(runs["T"]), str(runs["none"])],
                                                 eval="eval_navtest", n_tokens=4, n_tokens_all=N, subset_seed=0, ec="x")))
    a = EP.get_parser().parse_args(["report", "--scores", str(d), "--theta", "0.0", "--pairs", "T:none", "--n-boot", "200"])
    res = EP.cmd_report(a)
    assert res["arms"]["T"]["epdms"] == pytest.approx(90) and res["arms"]["T"]["epdms_orig"] == pytest.approx(50)
    assert res["arms"]["none"]["d_epdms_points"] == pytest.approx(10)
    assert res["pairs_epdms_points"]["T:none"]["mean"] == pytest.approx(30)
    a = EP.get_parser().parse_args(["report", "--scores", str(d), "--theta", "2.0"])
    assert EP.cmd_report(a)["arms"]["T"]["epdms"] == pytest.approx(50)                    # nothing modified


def test_epdms_report_refined_error_dropped_not_orig(tmp_path):
    """A modified draft whose refined scoring errored is dropped and counted, never scored as the original."""
    import pandas as pd
    tokens = ["tok0"]
    S = dict(tokens=tokens, valid=np.ones((1, 2), bool), family=np.zeros((1, 2), np.int8),
             sets={"orig": dict(arm="orig", p_g=np.full((1, 2), np.nan)), "r_T": dict(arm="T", p_g=np.full((1, 2), 0.9))})
    rows = []
    for name, sc in (("orig", 0.5), ("r_T", 0.9)):
        for k in range(2):
            err = "RuntimeError: boom" if (name == "r_T" and k == 1) else None
            d = {c: (np.nan if err else 1.0) for c in EP.METRIC_COLS}
            rows.append(dict(d, set=name, token="tok0", k=k, score=np.nan if err else sc, error=err,
                             copied_from_orig=False))
    df = EP.rows_frame(rows, S)
    f = EP.final_frame(df, "r_T", 0.5)
    assert len(f) == 1 and f.score.iloc[0] == pytest.approx(0.9) and f.attrs["n_refined_errors"] == 1
    f = EP.final_frame(df, "r_T", 0.95)                    # not modified: the refined error is irrelevant
    assert len(f) == 2 and f.attrs["n_refined_errors"] == 0 and f.score.mean() == pytest.approx(0.5)
    d = tmp_path / "scores"
    d.mkdir()
    df.to_parquet(d / "rows.parquet", index=False)
    (d / "meta.json").write_text(json.dumps(dict(arms={"r_T": "T"}, runs=["x"], eval="eval_navtest", n_tokens=1,
                                                 n_tokens_all=1, subset_seed=0, ec="x")))
    res = EP.cmd_report(EP.get_parser().parse_args(["report", "--scores", str(d), "--theta", "0.5"]))
    t = res["arms"]["T"]
    assert t["n"] == 1 and t["epdms"] == pytest.approx(90) and t["n_refined_errors"] == 1 and t["modified_frac"] == 1.0
    # duplicate arms (several seeds of one arm) are refused rather than silently overwritten
    (d / "meta.json").write_text(json.dumps(dict(arms={"r_T": "T", "r_T_s1": "T"}, runs=["x", "y"], eval="eval_navtest",
                                                 n_tokens=1, n_tokens_all=1, subset_seed=0, ec="x")))
    with pytest.raises(SystemExit, match="same arm"):
        EP.cmd_report(EP.get_parser().parse_args(["report", "--scores", str(d), "--theta", "0.5"]))
    with pytest.raises(SystemExit, match="same arm"):
        EP.check_unique_arms({"a": "T", "b": "T"})
    EP.check_unique_arms({"a": "T", "b": "M"})
