"""Evaluation tooling of refiner v6c (CPU; report 129 F20-F24): select_npz rule v6 = the submitted trajectory and the
'model' warning on a v6 dump, the navtest table of val_epdms (split_for / adjacency / scorer call = km's
run_eval_ep5.sh, navtrain_val names unchanged), val_score's log-cluster paired CI, eval.sh's command routes (EVAL_DRY)
for a v6c config and km's c64 config, dump_corr_model's resume guard and --v6 select-rule refusal, score_npz.py."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.corridor import dump_corr_model as DCM  # noqa: E402
from tools.corridor import score_npz as SN  # noqa: E402
from tools.refiner_v6c import val_epdms as VE  # noqa: E402
from tools.refiner_v6c import val_score as VS  # noqa: E402

PY = sys.executable
TOKS = ["3315880386e45927", "aa01", "aa02", "bb01", "bb02"]
LOGS = ["logA", "logA", "logA", "logB", "logB"]
CFG_A = REPO / "work_dirs/km_runs/corr30/v6cD_main6/train/code/hydra/config.yaml"                 # a v6c run
CFG_B = Path("/workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15_fix/train/code/hydra/config.yaml")  # km
NAVTEST_ADJ = VE.SPLITS["navtest"]["adj"]


def _env(**kw):
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **kw}


# ------------------------------------------------------------------------------------------------ 1. select_npz
@pytest.fixture()
def v6_dump(tmp_path):
    root = tmp_path / "km"
    (root / "packed/navtest").mkdir(parents=True)
    pd.DataFrame({"token": TOKS, "log": LOGS, "city": "x", "row": range(len(TOKS))}).to_parquet(
        root / "packed/navtest/tokens.parquet")
    d = tmp_path / "dump" / "navtest"
    d.mkdir(parents=True)
    n = len(TOKS)
    rng = np.random.default_rng(0)
    traj = rng.normal(size=(n, 256, 8, 3)).astype(np.float32)
    msel = np.array([3, 1, 4, 1, 5], np.int16)
    base = traj[np.arange(n), msel]
    v6 = base + rng.normal(size=base.shape).astype(np.float32) * 0.1
    sw = np.array([1, 0, 1, 0, 1], np.uint8)
    arrs = dict(traj=traj, sim=rng.random((n, 5, 256)).astype(np.float16), im=rng.random((n, 256)).astype(np.float16),
                final=rng.normal(size=(n, 256)).astype(np.float32), msel=msel, v6_base=base, v6_traj=v6, v6_switch=sw,
                done_0=np.ones(n, bool))
    for k, v in arrs.items():
        np.save(d / f"{k}.npy", v)
    return root, d.parent, arrs


def _select(root, dump, rule, out):
    return subprocess.run([PY, str(REPO / "tools/corridor/select_npz.py"), "--dump", str(dump), "--split", "navtest",
                           "--rule", rule, "--out", str(out)], env=_env(KM_CORR_ROOT=str(root)), cwd=REPO,
                          capture_output=True, text=True)


def test_select_rule_v6_is_the_submission_and_model_warns(v6_dump, tmp_path):
    root, dump, A = v6_dump
    r = _select(root, dump, "v6", tmp_path / "v6.npz")
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "v6.npz", allow_pickle=True)
    want = np.where(A["v6_switch"].astype(bool)[:, None, None], A["v6_traj"], A["v6_base"])
    assert z["tokens"].astype(str).tolist() == TOKS and np.array_equal(z["trajectory"], want)
    assert z["trajectory"].dtype == np.float32
    assert "WARNING" not in r.stdout + r.stderr
    for rule, ref in (("v6A", A["v6_traj"]), ("v6base", A["v6_base"])):
        assert _select(root, dump, rule, tmp_path / f"{rule}.npz").returncode == 0
        assert np.array_equal(np.load(tmp_path / f"{rule}.npz", allow_pickle=True)["trajectory"], ref)
    r = _select(root, dump, "model", tmp_path / "model.npz")
    assert r.returncode == 0, r.stderr
    assert "WARNING" in r.stdout and "use rule v6" in r.stdout
    assert np.array_equal(np.load(tmp_path / "model.npz", allow_pickle=True)["trajectory"], A["v6_base"])


def test_select_model_no_warning_without_v6(v6_dump, tmp_path):
    root, dump, _ = v6_dump
    (dump / "navtest/v6_switch.npy").unlink()
    r = _select(root, dump, "model", tmp_path / "model.npz")
    assert r.returncode == 0 and "WARNING" not in r.stdout + r.stderr


# ------------------------------------------------------------------------------------------------ 2. val_epdms
def test_navtest_table():
    T = VE.SPLITS["navtest"]
    assert set(T) == set(VE.SPLITS["navtrain_val"]) == {"metric_cache", "full_split", "logs", "adj", "data_split",
                                                         "searchpath"}
    assert T["full_split"] == "navtest" and T["data_split"] == "test" and T["searchpath"] == []
    assert T["metric_cache"] == Path("/workspace/yongjae/perception_eval/epdms/metric_cache_v22_navtest")
    assert T["logs"] == VE.NAVSIM_DATA / "navsim_logs/test/test"
    V = VE.table("navtrain_val")                                               # the old constants, unchanged
    assert V["metric_cache"] == VE.METRIC_CACHE and V["full_split"] == VE.FULL_SPLIT == "navtrain_val_e2e_q"
    assert V["adj"] == VE.ADJ_FULL and V["data_split"] == "trainval"
    with pytest.raises(KeyError):
        VE.table("navtrain_train")


def test_navtrain_val_names_unchanged(tmp_path):
    assert VE.run_dir(tmp_path, "base", "a" * 32).name == "epdms_val_base_aaaaaaaaaaaa"
    assert VE.run_dir(tmp_path, "base", "a" * 32, "navtrain_val") == VE.run_dir(tmp_path, "base", "a" * 32)
    assert VE.run_dir(tmp_path, "out", "b" * 32, "navtest").name == "epdms_navtest_out_bbbbbbbbbbbb"
    assert VE.split_of(tmp_path) == "navtrain_val"                               # no pkl_info.json
    json.dump({"subs": {}}, open(tmp_path / "pkl_info.json", "w"))
    assert VE.split_of(tmp_path) == "navtrain_val"                               # pkl_info.json of before the table
    json.dump({"subs": {"out": {"md5": "c" * 32}}, "split": "navtest"}, open(tmp_path / "pkl_info.json", "w"))
    assert VE.split_of(tmp_path) == "navtest"
    rd = VE.run_dir(tmp_path, "out", "c" * 32, "navtest")
    rd.mkdir(parents=True)
    (rd / "x.csv").write_text("x")
    assert VE.csv_of(tmp_path, "out") == rd / "x.csv"


def test_scorer_cmd_val_unchanged_and_navtest_is_km_call(tmp_path):
    out = tmp_path
    cmd, env, cwd = VE.scorer_cmd("navtrain_val", f"{out}/traj/base.pkl", out / "exp", "epdms_val_v6c_base",
                                  out / "exp/rd", 16, 10, [f"file://{out}/config"], tts="v6c_val_3_abc_q")
    sp = ["pkg://navsim.planning.script.config.common", f"file://{VE.PE_EPDMS}/config", f"file://{VE.A_EPDMS}/config",
          f"file://{out}/config"]
    old = ["nice", "-n", "10", sys.executable, "navsim/planning/script/run_pdm_score_one_stage.py",
           "train_test_split=v6c_val_3_abc_q", f"navsim_log_path={VE.NAVSIM_DATA}/navsim_logs/trainval/trainval",
           f"metric_cache_path={VE.METRIC_CACHE}", "agent=precomputed_agent", f"agent.traj_pkl={out}/traj/base.pkl",
           "experiment_name=epdms_val_v6c_base", f"output_dir={out / 'exp/rd'}",
           "worker=ray_distributed", "worker.threads_per_node=16", "hydra.searchpath=[" + ",".join(sp) + "]"]
    assert cmd == old and cwd == VE.NAVSIM22                                     # = the cmd_run command of 1205db7
    assert env["NAVSIM_EXP_ROOT"] == str(out / "exp") and env["PYTHONPATH"] == f"{VE.NAVSIM22}:{VE.PE_EPDMS}"
    cmd, env, _ = VE.scorer_cmd("navtest", "/x/sel.pkl", out, "e", out / "o", 32, 5)
    kv = dict(c.split("=", 1) for c in cmd if "=" in c)                          # km's run_eval_ep5.sh [4/4]
    assert cmd[:3] == ["nice", "-n", "5"] and kv["train_test_split"] == "navtest"
    assert kv["navsim_log_path"] == "/workspace/navsim_workspace/dataset/navsim_logs/test/test"
    assert kv["metric_cache_path"] == "/workspace/yongjae/perception_eval/epdms/metric_cache_v22_navtest"
    assert kv["agent"] == "precomputed_agent" and kv["worker"] == "ray_distributed"
    assert kv["hydra.searchpath"] == ("[pkg://navsim.planning.script.config.common,"
                                      "file:///workspace/yongjae/perception_eval/epdms/config]")
    for k, v in dict(NAVSIM_DEVKIT_ROOT="/workspace/navsim_workspace/navsim",
                     OPENSCENE_DATA_ROOT="/workspace/navsim_workspace/dataset",
                     NUPLAN_MAPS_ROOT="/workspace/navsim_workspace/dataset/maps", NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1").items():
        assert env[k] == v, k


def test_cmd_run_navtest_split(tmp_path, monkeypatch):
    """pkl --split navtest -> run: the navtest table, epdms_navtest_* names, a subset gets a quoted split"""
    adj = tmp_path / "adj.parquet"
    pd.DataFrame({"token": TOKS, "log_name": LOGS, "t": [0.0, 0.5, 1.0, 0.0, 0.5], "prev": None}).to_parquet(adj)
    monkeypatch.setitem(VE.SPLITS, "navtest", {**VE.SPLITS["navtest"], "adj": adj})
    out = tmp_path / "out"
    (out / "traj").mkdir(parents=True)
    calls = []

    def fake_run(cmd, cwd, env, stdout, stderr):
        calls.append(cmd)
        od = Path(dict(c.split("=", 1) for c in cmd if "=" in c)["output_dir"])
        od.mkdir(parents=True)
        (od / "r.csv").write_text("x")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(VE.subprocess, "run", fake_run)
    for toks, want in ((TOKS, "navtest"), (TOKS[:3], None)):
        shutil.rmtree(out / "exp", ignore_errors=True)
        T = {t: np.zeros((8, 3), np.float32) for t in toks}
        pickle.dump(T, open(out / "traj/out.pkl", "wb"))
        json.dump({"split": "navtest", "subs": {"out": {"md5": "d" * 32}}}, open(out / "pkl_info.json", "w"))
        pd.DataFrame({"token": toks, "log": LOGS[:len(toks)], "row": range(len(toks)), "switch": False}).to_parquet(
            out / "tokens.parquet")
        calls.clear()
        assert VE.cmd_run(argparse.Namespace(out=str(out), subs="out", threads=4, nice=10, force=False)) == 0
        kv = dict(c.split("=", 1) for c in calls[0] if "=" in c)
        assert kv["experiment_name"] == "epdms_navtest_v6c_out" and "epdms_navtest_out_" in kv["output_dir"]
        assert kv["metric_cache_path"] == str(VE.SPLITS["navtest"]["metric_cache"])
        if want:
            assert kv["train_test_split"] == "navtest" and f"file://{out.resolve()}/config" not in kv["hydra.searchpath"]
        else:
            name = kv["train_test_split"]
            assert name.startswith("v6c_navtest_3_") and name.endswith("_q")
            assert f"file://{out.resolve()}/config" in kv["hydra.searchpath"]
            assert "data_split: test" in (out / "config/train_test_split" / f"{name}.yaml").read_text()
        assert (out / "logs/epdms_navtest_out.log").exists()


@pytest.mark.skipif(not NAVTEST_ADJ.exists(), reason="navtest adjacency table absent")
def test_navtest_split_for_and_adjacency(tmp_path):
    full = pd.read_parquet(NAVTEST_ADJ)
    toks, logs = full.token.astype(str).tolist(), full.log_name.astype(str).tolist()
    assert len(toks) == 12146
    assert VE.split_for(tmp_path, toks, logs, "navtest") == "navtest"
    assert not (tmp_path / "config").exists()
    name = VE.split_for(tmp_path, toks[:3], logs[:3], "navtest")
    assert name != "navtest" and name.endswith("_q")
    assert "data_split: test" in (tmp_path / "config/train_test_split" / f"{name}.yaml").read_text()
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(tmp_path / "config/train_test_split/scene_filter" / f"{name}.yaml")
    assert list(cfg.tokens) == toks[:3]
    a = VE.adjacency(toks, VE.SPLITS["navtest"]["adj"])
    assert set(a.token) == set(toks) and len(a) == len(toks)
    km_tok = pd.read_parquet(REPO / "resources/km_corridor/packed/navtest/tokens.parquet").token.astype(str)
    assert set(km_tok) == set(toks)                                              # = the dumped navtest rows


def test_existing_val_epdms_tests_pass_unchanged():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        str(REPO / "tests/km/test_refiner_v6c_val_epdms.py")], cwd=REPO, env=_env(CUDA_VISIBLE_DEVICES=""),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


# ------------------------------------------------------------------------------------------------ 3. val_score
def test_paired_ci_is_log_cluster_bootstrap():
    rng = np.random.default_rng(0)
    L = 40
    logs = np.repeat([f"log{i}" for i in range(L)], 25)
    diff = np.repeat(rng.normal(0, 0.05, L), 25) + rng.normal(0, 0.01, len(logs))   # log-clustered effects
    c = VS.paired_ci(diff, logs)
    b = VE.log_bootstrap(diff, logs, 2000, 0)
    assert c["mean"] == round(100 * b["mean"], 3) and c["ci95"] == [round(100 * b["lo"], 3), round(100 * b["hi"], 3)]
    assert c["ci_method"] == "log-cluster bootstrap" and c["n_logs"] == L
    lo, hi = VS.boot_ci(diff)
    assert c["ci95_token"] == [round(100 * lo, 3), round(100 * hi, 3)]
    assert c["ci95"][1] - c["ci95"][0] > 2 * (c["ci95_token"][1] - c["ci95_token"][0])


# ------------------------------------------------------------------------------------------------ 4. eval.sh
EVAL = REPO / "scripts/km_corridor/eval.sh"


def test_eval_sh_syntax():
    assert subprocess.run(["bash", "-n", str(EVAL)]).returncode == 0


def _eval(root, spl, run, *extra, **env):
    return subprocess.Popen(["bash", str(EVAL), spl, run, "x.ckpt", "0", "t", *extra], cwd=REPO,
                            env=_env(KM_RUN_ROOT=str(root), EVAL_DRY="1", CAND="1", **env),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _done(p):
    o, e = p.communicate(timeout=600)
    assert p.returncode == 0, e
    return o


@pytest.mark.skipif(not (CFG_A.exists() and CFG_B.exists()), reason="archived run configs absent")
def test_eval_sh_routes(tmp_path):
    for nm, src in (("a", CFG_A), ("b", CFG_B)):
        d = tmp_path / "corr30" / nm / "train/code/hydra"
        d.mkdir(parents=True)
        shutil.copy(src, d / "config.yaml")
    P = {(r, s): _eval(tmp_path, s, r) for r in ("a", "b") for s in ("navtest", "val")}
    O = {k: _done(p) for k, p in P.items()}
    a = O["a", "navtest"]                                                        # v6c: the submission is rule v6
    assert "--v6" in a and "--planes 49 --fwd-rows 200" in a and "--traj32 --cand --no-fields --v6" in a
    for r in ("v6", "v6base", "v6A", "v2"):
        assert f"--rule {r} " in a
    assert "val_epdms.py all --split navtest" in a and "--npz v2=" in a and "val_score.py" in a
    assert "score_npz.py --check --split navtest" in a and "score_npz.sh" not in a and "score_npz.py --npz" not in a
    av = O["a", "val"]
    assert "--split navtrain_val" in av and "--v6" in av and "val_epdms.py all --dump" in av and "corr_verdict" not in av
    b = O["b", "navtest"]                                                        # km's c64 run (no v6c)
    assert "--v6" not in b and "--planes 49 --fwd-rows 200 --traj32 --cand --no-fields" in b
    assert "--rule model " in b and "--rule v2 " in b and "--rule v6" not in b
    assert b.count("score_npz.py --npz") == 2 and "score_npz.py --check --split navtest" in b
    assert "score_npz.sh" not in b and "--name b_t_model" in b
    bv = O["b", "val"]
    assert "--traj32" in bv and "--planes 49 --fwd-rows 200" in bv and "corr_select.py" in bv and "--v6" not in bv


def test_eval_sh_defaults_and_failures(tmp_path):
    o = _done(_eval(tmp_path, "val", "nocfg"))                                    # no archived config: the old defaults
    assert "--planes 8 --fwd-rows 100" in o and "--traj32" in o
    o = _done(_eval(tmp_path, "val", "nocfg", "40", "1", "200"))                  # positional args still override
    assert "--planes 40 --fwd-rows 200 --route --traj32" in o
    d = tmp_path / "corr30/bad/train/code/hydra"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text("agent: [unclosed\n")
    p = _eval(tmp_path, "navtest", "bad")
    p.communicate(timeout=120)
    assert p.returncode == 1                                                      # unreadable config: exit 1
    p = subprocess.run(["bash", str(EVAL), "navtest", "nocfg", "x.ckpt", "", "t"], cwd=REPO, capture_output=True,
                       text=True, env=_env(KM_RUN_ROOT=str(tmp_path), CAND="1", TOKENS_ROOT=str(tmp_path / "none/packed")),
                       timeout=120)
    assert p.returncode == 1 and "score_npz: missing token table" in p.stderr     # scorer pre-check before the dump
    assert not (tmp_path / "corr30/nocfg/eval_t").exists()


# ------------------------------------------------------------------------------------------------ 5. dump_corr_model
def test_resume_guard():
    DCM._resume_guard([], np.ones(4, bool))
    DCM._resume_guard(["v6_base"], np.zeros(4, bool))
    with pytest.raises(SystemExit, match="v6_base, v6_traj had to be created"):
        DCM._resume_guard(["v6_base", "v6_traj"], np.array([0, 1, 0, 0], bool))


def test_v6_refuses_select_rule_before_any_file_access(tmp_path, capsys):
    out = tmp_path / "z"
    assert DCM.main(["--run-dir", str(tmp_path / "x"), "--ckpt", "y", "--out", str(out), "--v6",
                     "--select-rule", "v2"]) == 2
    assert not out.exists() and "--select-rule v2" in capsys.readouterr().err


def test_resume_guard_in_main(tmp_path, monkeypatch):
    """a resume that must create new files (other flags) into rows already done is refused; the same flags resume"""
    root = tmp_path / "km"
    (root / "packed/navtest").mkdir(parents=True)
    pd.DataFrame({"token": TOKS, "log": LOGS}).to_parquet(root / "packed/navtest/tokens.parquet")
    monkeypatch.setattr(DCM, "ROOT", str(root))
    monkeypatch.setattr(DCM, "SPEC", dict(DCM.SPEC))
    out = tmp_path / "out"
    base = ["--run-dir", str(tmp_path / "run"), "--ckpt", "y", "--split", "navtest", "--out", str(out), "--no-fields"]
    with pytest.raises(FileNotFoundError):                                        # fresh: guard passes, no run config
        DCM.main(base)
    np.save(out / "navtest/done_1.npy", np.array([0, 1, 0, 0, 0], bool))           # another part finished a row
    np.save(out / "navtest/done_0.npy", np.zeros(5, bool))
    with pytest.raises(SystemExit, match="v6_base"):
        DCM.main(base + ["--v6"])
    assert not (out / "navtest/v6_base.npy").exists()                              # a refused resume creates nothing
    np.save(out / "navtest/done_0.npy", np.array([1, 0, 1, 0, 1], bool))
    monkeypatch.setattr(DCM, "SPEC", {k: v for k, v in DCM.SPEC.items() if not k.startswith("v6_")})
    assert DCM.main(base + ["--part", "0/2"]) == 0                                 # same flags: rows 0 / 2 / 4 done


# ------------------------------------------------------------------------------------------------ 6. score_npz.py
@pytest.mark.skipif(bool(SN.check("navtest")), reason="navtest scorer inputs absent on this host")
def test_score_npz_check():
    r = subprocess.run([PY, str(REPO / "tools/corridor/score_npz.py"), "--check", "--split", "navtest"], cwd=REPO,
                       env=_env(KM_CORR_ROOT=str(REPO / "resources/km_corridor")), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([PY, str(REPO / "tools/corridor/score_npz.py"), "--check", "--split", "navtest"], cwd=REPO,
                       env=_env(KM_CORR_ROOT="/nonexistent"), capture_output=True, text=True)
    assert r.returncode == 1 and "token table" in r.stderr


def test_score_npz_epdms_parse(tmp_path, monkeypatch):
    """the scorer call is val_epdms.scorer_cmd on the split table; EPDMS / terms = 100 x mean over the valid rows"""
    adj = tmp_path / "adj.parquet"
    pd.DataFrame({"token": TOKS, "log_name": LOGS, "t": [0.0, 0.5, 1.0, 0.0, 0.5], "prev": None}).to_parquet(adj)
    monkeypatch.setitem(VE.SPLITS, "navtest", {**VE.SPLITS["navtest"], "adj": adj})
    df = pd.DataFrame({m: np.ones(len(TOKS)) for m in VE.MET})
    df["ego_progress"] = [0.5, 0.6, 0.7, 0.8, 0.0]
    df["two_frame_extended_comfort"] = [np.nan, 1.0, 0.0, np.nan, 1.0]
    df.insert(0, "token", TOKS)
    df.insert(1, "valid", [True] * 4 + [False])
    df["score"] = VE.recompose(df)
    avg = {c: df[c].mean() for c in VE.MET + ["score"]}
    avg.update(token="average_all_frames", valid=False)
    csv = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    calls = []

    def fake_run(cmd, cwd, env, stdout, stderr):
        calls.append(cmd)
        od = Path(dict(c.split("=", 1) for c in cmd if "=" in c)["output_dir"])
        od.mkdir(parents=True)
        csv.to_csv(od / "2026.10.29.00.00.00.csv")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(SN.subprocess, "run", fake_run)
    a = argparse.Namespace(out=str(tmp_path / "o"), name="t", npz="x.npz", split="navtest", threads=8, nice=5)
    (tmp_path / "o").mkdir()
    traj = np.random.default_rng(0).normal(size=(len(TOKS), 8, 3)).astype(np.float32)
    res = SN.score_epdms(a, TOKS, traj, dict(zip(TOKS, LOGS)))
    assert "error" not in res and res["valid"] == 4 and res["train_test_split"] == "navtest"
    v = df[df.valid]
    assert res["EPDMS"] == pytest.approx(100 * v.score.mean()) and res["EP"] == pytest.approx(100 * 0.65)
    assert res["EC"] == pytest.approx(50.0) and res["n_EC"] == 2
    P = pickle.load(open(tmp_path / "o/sel_t.pkl", "rb"))
    assert list(P) == TOKS and all(np.array_equal(P[t], x) for t, x in zip(TOKS, traj))
    kv = dict(c.split("=", 1) for c in calls[0] if "=" in c)
    assert kv["agent.traj_pkl"] == str((tmp_path / "o/sel_t.pkl").resolve()) and kv["train_test_split"] == "navtest"

    def fail_run(cmd, cwd, env, stdout, stderr):
        return subprocess.CompletedProcess(cmd, 3)

    monkeypatch.setattr(SN.subprocess, "run", fail_run)
    assert "rc 3" in SN.score_epdms(a, TOKS, traj, dict(zip(TOKS, LOGS)))["error"]
