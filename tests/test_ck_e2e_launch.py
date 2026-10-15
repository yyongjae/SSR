"""CK Phase 2 launch tests (owner: launch; contract launch.tests): ck_e2e.yaml -> Hydra overrides -> compose ->
ParaSSRConfig -> CKE2EConfig.from_any; GPU / '=' / approval guards of train_e2e.sh; Lightning ckpt -> extract ->
load_ck strict; synthetic eval root -> eval_ck + summary; infer_check; status / smoke checks / KD table.
CPU only; no training is launched (train_e2e.sh is exercised through its refusals and E2E_DRY_RUN)."""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[1])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from tools.ck.e2e import launch_util as LU  # noqa: E402
from tools.ck.e2e import eval_e2e as EV  # noqa: E402

TE = f"{CK}/tools/ck/e2e/train_e2e.sh"
CONTRACT = Path("/home/external-user/ssd/yongjae_refiner/ck/phase2/spec/contract_e2e.json")
ENV_TRAIN = {"NAVSIM_EXP_ROOT": "/tmp", "OPENSCENE_DATA_ROOT": "/home/external-user/yongjae/SSR/data/dataset",
             "NUPLAN_MAPS_ROOT": "/home/external-user/yongjae/SSR/data/dataset/maps", "NAVSIM_DEVKIT_ROOT": CK,
             "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k, v in ENV_TRAIN.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------- (1) overrides
def _compose(overrides):
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module="navsim.planning.script.config.training", version_base="1.2"):
        return compose("default_training", overrides=list(overrides))


def test_ck_yaml_matches_contract_and_config():
    ck = LU.load_ck_yaml()
    assert ck["enabled"] is True and ck["io_dir"] == ""
    if CONTRACT.is_file():
        keys = json.loads(CONTRACT.read_text())["config"]["ck_e2e_keys"]
        assert set(ck) == set(keys), set(ck) ^ set(keys)
        for k, (default, _) in keys.items():
            if k in ("enabled", "io_dir"):
                continue
            assert ck[k] == default, (k, ck[k], default)
    online = pytest.importorskip("navsim.agents.para_ssr.ck.online")
    d = online.CKE2EConfig.from_any({}).to_dict()
    for k, v in ck.items():
        if k not in ("enabled", "io_dir"):
            assert d[k] == v, (k, d[k], v)


def test_overrides_compose_from_any(tmp_path):
    run = tmp_path / "e2e" / "runA"
    ov, ck = LU.build_overrides(str(run), ["0", "1", "2", "3"], wandb="offline",
                                ck_sets=["kd_ema.start_mb=50", "kd_ramp_epochs=0.5", "teacher_map_run=''"],
                                extra=["trainer.params.max_epochs=4", "trainer.params.limit_train_batches=60"],
                                resume_ckpt=str(tmp_path / "last.ckpt"))
    assert ck["io_dir"] == f"{run}/ck_e2e" and ck["kd_ema"]["start_mb"] == 50 and ck["teacher_map_run"] == ""
    keys = [LU._okey(o) for o in ov]
    assert len([k for k in keys if k == "trainer.params.max_epochs"]) == 1          # extra replaced the recipe entry
    assert "trainer.params.max_epochs=4" in ov and "trainer.params.limit_train_batches=60" in ov
    assert f"output_dir={run}/train/" in ov and "++agent.config.log_sync_dist=false" in ov
    assert "trainer.params.strategy=ddp" in ov and "+trainer.params.devices=4" in ov
    assert "dataloader.params.batch_size=4" in ov and "trainer.params.accumulate_grad_batches=8" in ov
    assert "agent.config.grad_balance_warmup_iters=5300" in ov and "trainer.params.limit_val_batches=0" in ov
    assert "wandb.mode=offline" in ov and "wandb.project=para-ssr-v2" in ov
    cfg = _compose(ov)
    got = dict(cfg.agent.config.ck_e2e)
    assert got["io_dir"] == f"{run}/ck_e2e" and got["kd_ramp_epochs"] == 0.5 and got["teacher_map_run"] == ""
    assert cfg.agent.config.log_sync_dist is False
    assert cfg.resume_checkpoint == str(tmp_path / "last.ckpt")
    assert int(cfg.trainer.params.devices) == 4 and int(cfg.agent.config.max_epochs) == 30
    online = pytest.importorskip("navsim.agents.para_ssr.ck.online")
    from hydra.utils import instantiate
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    import dataclasses
    if "ck_e2e" not in {f.name for f in dataclasses.fields(ParaSSRConfig)}:
        pytest.skip("ParaSSRConfig.ck_e2e not added yet (model owner)")
    pc = instantiate(cfg.agent.config)
    assert pc.log_sync_dist is False
    c = online.CKE2EConfig.from_any(pc.ck_e2e)
    assert c.enabled and c.io_dir == f"{run}/ck_e2e" and c.kd_ema["start_mb"] == 50
    one = LU.build_overrides(str(run), ["2"], wandb="disable")[0]
    assert "trainer.params.strategy=auto" in one and "wandb.enable=false" in one


@pytest.mark.parametrize("v", ["a b", "", "x=y", "/abs/path-1_2.npy", "1e-4", "true", None, True, False, 3, 0.0001,
                               1.0, [1, 2], "it's"])
def test_hydra_value_roundtrip(v):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    o = OverridesParser.create().parse_overrides([f"++a.b={LU.hydra_value(v)}"])[0]
    got = o.value()
    assert got == (list(v) if isinstance(v, list) else v), (v, LU.hydra_value(v), got)


# ----------------------------------------------------------------------------------------------- (2) guards
def test_gpu_guard_and_equals():
    for bad in ("4", "0,5", "7", "1,1"):
        with pytest.raises(LU.LaunchError):
            LU.parse_gpus(bad)
    assert LU.parse_gpus("0,1,2,3") == ["0", "1", "2", "3"]
    used = {"0": 1, "1": 5000, "2": 1, "3": 1, "6": 15000}
    with pytest.raises(LU.LaunchError):
        LU.gpu_check(["0", "1"], used=used)
    assert LU.gpu_check(["0", "2", "3"], used=used) == {"0": 1, "2": 1, "3": 1}
    assert LU.pick_gpus(1, ["1", "2", "3"], used=used) == ["2"]
    with pytest.raises(LU.LaunchError):
        LU.pick_gpus(3, ["1", "2", "3"], used=used)
    with pytest.raises(LU.LaunchError):
        LU.build_overrides("/tmp/x=1/run", ["0"], wandb="disable")
    with pytest.raises(LU.LaunchError):
        LU.build_overrides("/tmp/run", ["0"], wandb="disable", resume_ckpt="/a/epoch=1-step=2.ckpt")
    with pytest.raises(LU.LaunchError):
        LU.build_overrides("/tmp/run", ["0"], wandb="disable", ck_sets=["record_from_epoch=6"])
    with pytest.raises(LU.LaunchError):
        LU.build_overrides("/tmp/run", ["0"], wandb="disable", ck_sets=["no_such_key=1"])


def _te(args, tmp_path, **env):
    e = dict(os.environ, E2E_ROOT=str(tmp_path / "e2e"), E2E_WANDB="disable", RUN="t1")
    e.pop("CK_E2E_APPROVED", None)
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", TE] + args, env=e, capture_output=True, text=True, timeout=120)


def test_train_sh_refusals(tmp_path):
    r = _te(["start"], tmp_path)
    assert r.returncode == 3 and "CK_E2E_APPROVED" in r.stderr
    assert not (tmp_path / "e2e/t1/launch/train.pid").exists()
    r = _te(["start", "--gpus", "4,5"], tmp_path, CK_E2E_APPROVED=1)
    assert r.returncode == 3 and "not allowed" in (r.stderr + r.stdout)
    r = _te(["start"], tmp_path, E2E_SMOKE=1)            # smoke waiver only below the e2e_smoke root
    assert r.returncode == 3 and "E2E_SMOKE" in r.stderr
    r = _te(["start"], tmp_path, CK_E2E_APPROVED=1, RUN="a=b")
    assert r.returncode == 3 and "'='" in r.stderr
    (tmp_path / "e2e/t1/train/lightning_logs").mkdir(parents=True)
    r = _te(["start"], tmp_path, CK_E2E_APPROVED=1, E2E_DRY_RUN=1)
    assert r.returncode == 3 and "--resume" in r.stderr
    r = _te(["start", "--resume"], tmp_path, CK_E2E_APPROVED=1, E2E_DRY_RUN=1)
    assert r.returncode == 3 and "last.ckpt" in r.stderr


def test_train_sh_dry_run(tmp_path):
    (tmp_path / "ck.txt").write_text("kd_ema.start_mb=50\n")
    (tmp_path / "extra.txt").write_text("trainer.params.limit_train_batches=150\n")
    r = _te(["start", "--gpus", "1"], tmp_path, CK_E2E_APPROVED=1, E2E_DRY_RUN=1, MAX_EPOCHS=1,
            E2E_CK_SET=tmp_path / "ck.txt", E2E_EXTRA=tmp_path / "extra.txt")
    assert r.returncode == 0, r.stderr
    ov = (tmp_path / "e2e/t1/launch/overrides.txt").read_text().splitlines()
    assert "++agent.config.ck_e2e.enabled=true" in ov and "++agent.config.ck_e2e.kd_ema.start_mb=50" in ov
    assert "trainer.params.max_epochs=1" in ov and "trainer.params.limit_train_batches=150" in ov
    assert "trainer.params.strategy=auto" in ov and "wandb.enable=false" in ov
    eff = json.loads((tmp_path / "e2e/t1/launch/ck_e2e_effective.json").read_text())
    assert eff["_trainer_max_epochs"] == 1 and eff["io_dir"] == f"{tmp_path}/e2e/t1/ck_e2e"
    assert not (tmp_path / "e2e/t1/launch/train.pid").exists()
    # resume picks the newest last.ckpt
    ck = tmp_path / "e2e/t1/train/lightning_logs/version_1/checkpoints"
    ck.mkdir(parents=True)
    (ck / "last.ckpt").write_bytes(b"x")
    r = _te(["start", "--resume", "--gpus", "1"], tmp_path, CK_E2E_APPROVED=1, E2E_DRY_RUN=1)
    assert r.returncode == 0, r.stderr
    assert f"++resume_checkpoint={ck / 'last.ckpt'}" in r.stdout


def test_find_last_and_epoch_ckpts(tmp_path):
    for v, names in ((0, ["epoch=0-step=10.ckpt", "last.ckpt"]), (1, ["epoch=2-step=30.ckpt", "last.ckpt"])):
        d = tmp_path / f"train/lightning_logs/version_{v}/checkpoints"
        d.mkdir(parents=True)
        for n in names:
            (d / n).write_bytes(b"x")
        os.utime(d / "last.ckpt", (1000 + v, 1000 + v))
    assert LU.find_last_ckpt(str(tmp_path)).parent.parent.name == "version_1"
    assert [e for e, _ in LU.epoch_ckpts(str(tmp_path))] == [0, 2]
    assert EV.resolve_ckpt(tmp_path, "epoch:2").name == "epoch=2-step=30.ckpt"


# ----------------------------------------------------------------------------------------------- (3) extract
def _student():
    from navsim.agents.para_ssr.ck.model import CKNet
    torch.manual_seed(0)
    net = CKNet("S", 0, None, score_hidden=256, lead_aux=False)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.01 * torch.randn_like(p))
    return net


def test_extract_roundtrip(tmp_path):
    from navsim.agents.para_ssr.ck.model import load_ck
    net = _student()
    sd = {f"agent.ck_student.{k}": v for k, v in net.state_dict().items()}
    sd["agent.para_ssr_model.dummy.weight"] = torch.zeros(3)
    ckpt = tmp_path / "epoch_last.ckpt"
    torch.save({"state_dict": sd, "callbacks": {}}, ckpt)
    cfg = EV.extract_ck(ckpt, tmp_path / "ck_run", "runA", {"seed": 0})
    assert cfg["arm"] == "S" and cfg["e2e"] and cfg["n_tensors"] == len(net.state_dict())
    net2, cfg2 = load_ck(tmp_path / "ck_run", "last", "cpu")
    for k, v in net.state_dict().items():
        assert torch.equal(v, net2.state_dict()[k]), k
    torch.save({"state_dict": {"agent.x": torch.zeros(1)}}, tmp_path / "plain.ckpt")
    with pytest.raises(SystemExit):
        EV.extract_ck(tmp_path / "plain.ckpt", tmp_path / "ck_run2")


# ----------------------------------------------------------------------------------------------- (4) eval root
def _synthetic_run(tmp_path, split="navtrain_val", n=4, seed=0):
    """Eval root of n real split tokens written through EvalRootWriter + label_cands-format labels."""
    from navsim.agents.para_ssr.ck import constants as Cn
    from tools.ck.data import common as CM
    rng = np.random.default_rng(seed)
    run_dir = tmp_path / "runA"
    P = EV.Paths(run_dir, split)
    tdf = CM.split_tokens(split).iloc[:n].reset_index(drop=True)
    W = EV.EvalRootWriter(P, tdf, P.ck_run)
    K = EV.K
    cand = rng.normal(size=(n, K, 8, 3)).astype(np.float32)
    pred = {"ck_cand": cand, "ck_cand_idx": np.tile(np.arange(K), (n, 1)),
            "ck_v2_final": -np.sort(rng.random((n, K)), 1).astype(np.float32),
            "ck_v2_im": rng.dirichlet(np.ones(K), n).astype(np.float32),
            "ck_v2_sim": rng.random((n, K, 5)).astype(np.float32),
            "ck_score_logit": rng.normal(size=(n, K, 5)).astype(np.float32),
            "ck_z_lon": rng.normal(size=(n, K, 6)).astype(np.float32),
            "ck_w_lat": rng.normal(size=(n, K, 6)).astype(np.float32),
            "ck_c_lon": rng.normal(size=(n, K, 6)).astype(np.float32),
            "ck_e_lat": rng.normal(size=(n, K, 6)).astype(np.float32),
            "ck_corr_traj": cand + 0.1, "ck_corr_score_logit": rng.normal(size=(n, K, 5)).astype(np.float32)}
    ok = W.write(np.arange(n), pred, rng.random((n, 8)).astype(np.float32), cand[:, 0], cand[:, 0])
    assert ok.all()
    W.finalize_meta({"ckpt": "synthetic", "ckpt_sha16": "0" * 16, "limit": n, "n_done": n})
    for name in ("cand", f"corr_{P.name}"):
        d = P.labels / name
        d.mkdir(parents=True, exist_ok=True)
        lab = rng.random((n, K, len(Cn.LABEL_COLS))).astype(np.float32)
        lab[..., [Cn.LBL["nc"], Cn.LBL["dac"]]] = (lab[..., [Cn.LBL["nc"], Cn.LBL["dac"]]] > 0.2).astype(np.float32)
        np.save(d / "labels.npy", lab)
        np.save(d / "ok.npy", np.ones((n, K), bool))
        (d / "meta.json").write_text("{}")
    return run_dir, P, pred


def test_eval_root_runs_eval_ck_and_summary(tmp_path, monkeypatch):
    run_dir, P, pred = _synthetic_run(tmp_path)
    # format checks (pack_v2 / infer_ck layouts)
    tok = pd.read_parquet(P.packed / "tokens.parquet")
    assert list(tok.columns) == ["token", "log", "city", "row"] and len(tok) == 4
    assert np.load(P.packed / "cand.npy").shape == (4, 16, 8, 3) and np.load(P.packed / "cand_idx.npy").dtype == np.int16
    assert np.load(P.infer / "score_logit.npy").dtype == np.float16 and np.load(P.infer / "done.npy").all()
    assert np.array_equal(np.load(P.packed / "ok_rows.npy"), np.arange(4))
    np.testing.assert_allclose(np.load(P.infer / "corr_traj.npy"), pred["ck_corr_traj"])

    class A:
        fix_from = ""
        bootstrap = 50
    monkeypatch.setenv("CK_DATA_ROOT", str(P.root))
    m = EV.stage_eval(P, A())
    met = json.loads(m.read_text())
    variants = {r["variant"] for r in met["rows"]}
    assert {"v2", "oracle16", "a", "b", "c", "oracle32"} <= variants and met["n_eval"] == 4
    bv = met["best_variant"]                                  # variant chosen on navtrain_val (not on navtest)
    exp = max(("a", "b", "c"), key=lambda v: (EV._row(met["rows"], v, met["best_beta"][v])["pdms"],
                                              -("a", "b", "c").index(v)))
    assert bv["variant"] == exp and bv["beta"] == met["best_beta"][exp]
    tie = {"best_beta": {"a": 0.5, "b": 1.0}, "rows": [{"variant": "a", "beta": 0.5, "pdms": 0.9},
                                                        {"variant": "b", "beta": 1.0, "pdms": 0.9}]}
    assert EV.select_variant(tie)["variant"] == "a"           # ties -> a
    assert (P.root / "lead").is_symlink()
    # navtest: beta only from val -> fixed from the val metrics
    Pt = EV.Paths(run_dir, "navtest")
    with pytest.raises(SystemExit):
        A.fix_from = ""
        EV.stage_eval(EV.Paths(tmp_path / "other", "navtest"), A())
    s = EV.stage_summary(run_dir, n_boot=50)
    sp = s["splits"]["navtrain_val"]
    assert "v2-head (β=0)" in sp["rows"] and any(k.startswith("a (val β=") for k in sp["rows"])
    assert sp["variant_from_val"]["variant"] == bv["variant"] and sp["representative"].startswith(bv["variant"] + " (val")
    md = (run_dir / "eval/summary.md").read_text()
    assert md.startswith("# CK Phase 2 e2e") and "★ " + sp["representative"] in md and "(서술용)" in md
    ref = sp.get("r34_reference")
    if ref is not None:                                    # real val tokens have Phase 1 r34 labels
        assert ref["n"] == 4 and "d_pdms_e2e_minus_r34" in ref
    del Pt


def test_writer_refuses_other_token_list(tmp_path):
    run_dir, P, _ = _synthetic_run(tmp_path, n=4)
    from tools.ck.data import common as CM
    tdf = CM.split_tokens("navtrain_val").iloc[1:5].reset_index(drop=True)
    with pytest.raises(SystemExit):
        EV.EvalRootWriter(P, tdf)


def test_infer_check(tmp_path):
    run_dir, P, _ = _synthetic_run(tmp_path, n=2)
    net = _student()
    sd = {f"agent.ck_student.{k}": v for k, v in net.state_dict().items()}
    torch.save({"state_dict": sd}, tmp_path / "x.ckpt")
    EV.extract_ck(tmp_path / "x.ckpt", P.ck_run, "runA", {})
    rows = np.arange(2)
    bev = torch.randn(2, 256, 50, 100)
    cand = torch.from_numpy(np.load(P.packed / "cand.npy")[rows])
    status = torch.from_numpy(np.load(P.packed / "status.npy")[rows])
    status[:, :4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    np.save(P.packed / "status.npy", status.numpy())
    net.eval()
    with torch.no_grad():
        o = net(bev, cand, status, decode=True, slope=0.0, rescore_corr=True)
    vals = {"score_logit": o["score_logit"], "z_lon": o["z_lon"], "w_lat": o["w_lat"],
            "c_lon": o["corr"]["c_lon"][..., 2:], "e_lat": o["corr"]["e_lat"][..., 2:], "corr_traj": o["corr"]["traj"],
            "corr_score_logit": o["corr_score_logit"]}
    for k, v in vals.items():
        arr = np.load(P.infer / f"{k}.npy")
        arr[rows] = v.numpy().astype(arr.dtype)
        np.save(P.infer / f"{k}.npy", arr)
    np.savez(P.infer / "check_bev.npz", rows=rows, bev=bev.numpy())

    class A:
        gpus = ""
    res = EV.stage_infer_check(P, A())
    assert res["pass"]
    arr = np.load(P.infer / "corr_traj.npy")
    arr[0] += 0.5
    np.save(P.infer / "corr_traj.npy", arr)
    with pytest.raises(SystemExit):
        EV.stage_infer_check(P, A())


# ----------------------------------------------------------------------------------------------- status / smoke
def _steps(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _rec(epoch, i, phase=0, **kw):
    r = {"epoch": epoch, "batch": i, "mb": i, "epoch_frac": epoch + i / 100, "sec_step": 0.4, "mem_gb": 14.0,
         "skipped_steps": 0, "loss": 3.0, "loss_v2": 2.0, "ck/loss": 1.0, "ck/bce_cand": 0.3, "ck/bce_tkd": 0.2,
         "ck/kd_score": 0.4, "ck/sur": 0.2, "ck/kd_ctrl": 0.1, "ck/w_ema": 1.0, "ck/r": 1.0, "ck/phase": phase,
         "ck/kd_ok_frac": 1.0, "ck/G_src_phase1": 0.0, "ck/G_src_prev": 1.0, "ck/G_src_older": 0.0,
         "ck/gshare_ck": 0.3}
    r.update(kw)
    return r


def test_status_and_checks(tmp_path):
    R = tmp_path / "s4"
    _steps(R / "ck_e2e/steps_rank0.jsonl", [_rec(0, i, phase=2) for i in range(100)])
    (R / "launch").mkdir(parents=True)
    (R / "launch/train.exit").write_text("0\n")
    (R / "labeler").mkdir(parents=True)
    (R / "labeler/labeler.exit").write_text("0\n")
    (R / "launch/train.pid").write_text(str(os.getpid()))      # this pytest process: not run_training -> dead
    ch = R / "ck_e2e/lab/ep000/chunks"
    ch.mkdir(parents=True)
    for i in range(3):
        (ch / f"r0_c_{i}.json").write_text(json.dumps({"n": 64, "sec": 0.8, "tok_s": 80.0}))
    (R / "ck_e2e/lab/ep000/DONE.json").write_text("{}")
    st = LU.run_status(str(R))
    assert not st["train"]["alive"] and st["train"]["exit"] == "0" and st["generations_done"] == [0]
    assert st["steps"]["epoch"] == 0 and st["steps"]["sec_step_median"] == 0.4 and "eta" not in st   # not alive
    res = LU.check_stage("S4", str(R), ngpu=4)
    assert res["checks"]["sec_per_mb_le_0.55"] and res["checks"]["gshare_logged"]
    assert res["labeler_tok_s"] == 80.0 and res["checks"]["labeler_ge_1.5x_consume"]    # 80 >= 1.5 x 40
    assert res["pass"], res["checks"]
    drain = tmp_path / "drain"
    (drain / "lab").mkdir(parents=True)
    (drain / "lab/status.json").write_text(json.dumps({"total_tok_s": 55.0, "tokens_ok": 3200, "recent_tok_s": 70}))
    (R / "lab_snapshots.jsonl").write_text(json.dumps({"backlog_chunks": 3, "recent_tok_s": 66.0}) + "\n"
                                           + json.dumps({"backlog_chunks": 0, "recent_tok_s": 90.0}) + "\n")
    res = LU.check_stage("S4", str(R), ngpu=4, drain_dir=str(drain))
    assert res["labeler_tok_s"] == 55.0 and not res["checks"]["labeler_ge_1.5x_consume"]   # 55 < 60
    assert res["during_training_max_tok_s_with_backlog"] == 66.0
    _steps(R / "ck_e2e/steps_rank0.jsonl", [_rec(0, i, phase=2, sec_step=0.7) for i in range(100)])
    res = LU.check_stage("S4", str(R), ngpu=4)
    assert not res["pass"] and not res["checks"]["sec_per_mb_le_0.55"]
    R2 = tmp_path / "s2"
    _steps(R2 / "ck_e2e/steps_rank0.jsonl", [_rec(0, i) for i in range(10)] + [_rec(0, 10, **{"ck/sur": float("nan")})])
    (R2 / "launch").mkdir(parents=True)
    (R2 / "launch/train.exit").write_text("0")
    res = LU.check_stage("S2", str(R2))
    assert res["nonfinite"] == 1 and not res["pass"]


def test_s3_check_and_resume_state(tmp_path):
    R = tmp_path / "s3"
    recs = [_rec(e, i, phase=(0 if e < 1 else 1 if e < 2 else 2)) for e in range(4) for i in range(60)]
    _steps(R / "ck_e2e/steps_rank0.jsonl", recs)
    (R / "launch").mkdir(parents=True)
    (R / "launch/train.exit").write_text("0")
    from navsim.agents.para_ssr.ck import e2e_data as D
    rng = np.random.default_rng(0)
    io = R / "ck_e2e"
    for e in (1, 2):                      # recorded rows 3, 4, 7 -> generation with exactly those trajectories
        rows = np.array([3, 4, 7])
        cand = rng.normal(size=(3, 16, 8, 3)).astype(np.float32)
        kdc = rng.normal(size=(3, 16, 8, 3)).astype(np.float32)
        D.write_rec_chunk(D.rec_chunk_path(io, e, 0, "0000000a", 0), rows, cand, np.zeros((3, 16), np.int16), kdc,
                          np.ones((3, 16), bool), 0, e, 0)
        g = D.open_generation(io, e, 10, "w+")
        D.write_generation_rows(g, rows, np.concatenate([cand, kdc], 1), np.ones((3, 32, D.N_LAB), np.float32),
                                np.ones((3, 32), bool))
        (R / f"ck_e2e/lab/ep{e:03d}/DONE.json").write_text("{}")
    d0 = R / "train/lightning_logs/version_0/checkpoints"
    d1 = R / "train/lightning_logs/version_1/checkpoints"
    d0.mkdir(parents=True)
    d1.mkdir(parents=True)
    st = lambda mb, n: {"callbacks": {"CKE2ECallback": {"ema": {"sur": 1.0, "kd": 1.0, "n": n}, "mb": mb,  # noqa
                                                           "cum": {}, "skipped_steps": 0}}}
    torch.save(st(180, 179), d0 / "epoch=2-step=21.ckpt")
    torch.save(st(240, 239), d1 / "last.ckpt")
    res = LU.check_stage("S3", str(R), mb_per_epoch=60)
    assert res["pass"], res["checks"]
    assert res["gen_roundtrip"][1]["rows_labelled"] == 3 and res["gen_roundtrip"][2]["labelled_of_recorded"] == 1.0
    g = D.open_generation(io, 2, 10, "r+")                     # a labelled row whose trajectories differ -> fail
    g["traj"][4, 0, 0, 0] += 1.0
    g["traj"].flush()
    assert not LU.check_stage("S3", str(R), mb_per_epoch=60)["checks"]["gen_roundtrip_bitexact"]
    torch.save(st(230, 229), d1 / "last.ckpt")
    assert not LU.check_stage("S3", str(R), mb_per_epoch=60)["checks"]["resume_mb_continuous"]


def test_smoke_report_kd_table(tmp_path):
    _steps(tmp_path / "s4" / "ck_e2e/steps_rank0.jsonl",
           [_rec(e, i, phase=e, **{"ck/loss": 0.3 + 0.2 + 0.5 * 0.4 + 0.2 + 1.0 * 0.1, "sec_wait": 0.01})
            for e in range(3) for i in range(40)])
    _steps(tmp_path / "s4" / "ck_e2e/epochs.jsonl",
           [{"event": ev, "epoch": e, "time": 1000.0 + 30 * e + (0 if ev == "epoch_start" else 20)}
            for e in range(3) for ev in ("epoch_start", "epoch_end")])
    for s in ("S1", "S2", "S3", "S4", "S5"):
        (tmp_path / f"result_{s}.json").write_text(json.dumps({"stage": s, "pass": True}))
    s4 = LU.check_stage("S4", str(tmp_path / "s4"), ngpu=4)
    assert set(s4["phase_timing"]) == {"replay", "replay_record", "onpolicy"}
    assert s4["phase_timing"]["onpolicy"]["n"] == 20 and s4["phase_timing"]["onpolicy"]["wall_per_mb_mean"] == \
        pytest.approx(0.41)
    o = s4["epoch_overhead"]                                  # wall 20 s - 40 x 0.41 = 3.6 s inside, 10 s between
    assert o["inside_overhead_s_median"] == pytest.approx(3.6) and o["gap_s_median"] == pytest.approx(10.0)
    (tmp_path / "result_S4.json").write_text(json.dumps(dict(s4, **{"pass": True})))
    rep = LU.smoke_report(str(tmp_path))
    assert rep["all_pass"]
    est = rep["estimate_30ep"]["wall_per_mb_mean"]
    assert est["mb_per_epoch"] == 5320 == LU.mb_per_epoch(4)
    assert est["total_hours"] == pytest.approx(30 * (5320 * 0.41 + 13.6) / 3600)
    assert [r["epochs"] for r in est["rows"]] == [4, 1, 25]
    assert "30 epoch 예상" in (tmp_path / "smoke_summary.md").read_text()
    t = json.loads((tmp_path / "kd_loss_table.json").read_text())
    for name in ("replay(S4 ep0)", "replay+기록(S4 ep1)", "on-policy(S4 ep2)"):
        ph = t["phases"][name]
        assert ph["ck/kd_score"]["weighted"] == pytest.approx(0.2)
        assert ph["reconstruct"]["rel_diff"] < 1e-6
        assert ph["ck_over_v2"] == pytest.approx(0.5)
    md = (tmp_path / "kd_loss_table.md").read_text()
    assert "KD loss 표" in md and "ck/kd_ctrl" in md


def test_smoke_tokens_fixed():
    toks, logs = LU.smoke_tokens(240)
    assert len(toks) == 240 == len(set(toks))
    assert LU.smoke_tokens(240) == (toks, logs)
    df = pd.read_parquet(Path(LU.PACKED_TRAIN) / "tokens.parquet")
    assert set(toks) <= set(df.token) and set(df[df.token.isin(toks)].log) == set(logs)


def test_status_monitors_and_warnings(tmp_path):
    from navsim.agents.para_ssr.ck import online as O
    assert LU.RAW_FAIL_R34 == O.RAW_FAIL_R34
    R = tmp_path / "run"
    recs = [_rec(5, i, phase=2, **{"ck/raw_fail_nc": 0.03, "ck/raw_fail_dac": 0.034, "ck/raw_fail_ttc": 0.2})
            for i in range(100)]
    for i, r in enumerate(recs):
        if i % 10:
            r.pop("ck/gshare_ck")
        else:
            r["ck/gshare_ck"] = 0.6 if i >= 50 else 0.2
    _steps(R / "ck_e2e/steps_rank0.jsonl", recs)
    _steps(R / "ck_e2e/epochs.jsonl", [{"event": "epoch_start", "epoch": 6, "phase": "onpolicy", "world_size": 4,
                                        "label_warning": {"warn": True, "msg": "only 10/1000 rows"}}])
    (R / "launch").mkdir(parents=True)
    (R / "labeler").mkdir(parents=True)
    st = LU.run_status(str(R))
    mon = st["monitor"]
    assert mon["gshare_ck"]["n"] == 10 and mon["gshare_ck"]["mean"] == pytest.approx(0.4) and not mon["gshare_ck"]["warn"]
    assert mon["gshare_ck"]["max"] == pytest.approx(0.6)
    k = mon["raw_fail"]["keys"]
    assert k["nc"]["ratio"] == pytest.approx(1.25) and not k["nc"]["over"] and k["ttc"]["over"]
    assert not mon["raw_fail"]["within_1.5x"]
    w = " | ".join(st["warnings"])
    assert "raw fail" in w and "ttc" in w and "label supply" in w and "gradient" not in w
    assert st["watchdog"] == {"pid": None, "alive": False, "restarts": 0}
    for r in recs:
        if "ck/gshare_ck" in r:
            r["ck/gshare_ck"] = 0.7
    _steps(R / "ck_e2e/steps_rank0.jsonl", recs)
    st = LU.run_status(str(R))
    assert st["monitor"]["gshare_ck"]["warn"] and any("BEV gradient" in x for x in st["warnings"])
    LU.print_status(st)
    # training alive, labeler dead -> warning
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "run_training"])
    try:
        (R / "launch/train.pid").write_text(str(p.pid))
        (R / "labeler/labeler.exit").write_text("1\n")
        st = LU.run_status(str(R))
        assert any("labeler not running" in x for x in st["warnings"])
    finally:
        p.kill()


def test_labeler_health(tmp_path):
    R = tmp_path / "run"
    lab = R / "ck_e2e/lab"
    lab.mkdir(parents=True)
    (R / "labeler").mkdir()
    state = str(R / "labeler/watchdog.state")
    sp = lab / "status.json"
    assert LU.labeler_health(str(R), state, 100)[0]                                  # no status yet
    def put(t, **kw):
        d = {"pid": 11, "tokens_ok": 0, "tokens_err": 0, "tokens_skip": 0, "backlog_chunks": 2, "inflight": 8}
        d.update(kw)
        sp.write_text(json.dumps(d))
        os.utime(sp, (t, t))
    t0 = 1_000_000.0
    put(t0, tokens_ok=5)
    assert LU.labeler_health(str(R), state, 100, now=t0 + 1)[0]
    put(t0 + 50, tokens_ok=5)
    assert LU.labeler_health(str(R), state, 100, now=t0 + 60)[0]                     # stalled 59 s < 100
    put(t0 + 110, tokens_ok=5)
    ok, why = LU.labeler_health(str(R), state, 100, now=t0 + 120)
    assert not ok and "pool stuck" in why                                           # busy, no progress 119 s
    put(t0 + 130, tokens_ok=6)
    assert LU.labeler_health(str(R), state, 100, now=t0 + 135)[0]                    # progress resets
    put(t0 + 300, tokens_ok=6, backlog_chunks=0, inflight=0)
    assert LU.labeler_health(str(R), state, 100, now=t0 + 400)[0]                    # idle is fine
    ok, why = LU.labeler_health(str(R), state, 100, now=t0 + 500)                    # status 200 s old
    assert not ok and "hung" in why
    pf = R / "labeler/labeler.pid"                                                   # a new labeler just started
    pf.write_text("22\n")
    os.utime(pf, (t0 + 480, t0 + 480))
    assert LU.labeler_health(str(R), state, 100, now=t0 + 500)[0]
    os.utime(pf, (t0 + 300, t0 + 300))                                              # ... but long ago
    assert not LU.labeler_health(str(R), state, 100, now=t0 + 500)[0]
    r = subprocess.run([sys.executable, str(Path(LU.__file__)), "labeler-health", "--run-dir", str(R), "--state", state,
                        "--stall-s", "1"], capture_output=True, text=True)
    assert r.returncode == 5 and "hung" in r.stdout


def test_status_eta_alive(tmp_path):
    R = tmp_path / "run"
    _steps(R / "ck_e2e/steps_rank0.jsonl", [_rec(3, i) for i in range(1, 51)])      # frac 3.5 at batch 50 -> 100 mb
    _steps(R / "ck_e2e/epochs.jsonl", [{"event": "epoch_start", "epoch": 3, "world_size": 4}])
    (R / "launch").mkdir(parents=True)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "run_training"])
    try:
        (R / "launch/train.pid").write_text(str(p.pid))
        (R / "launch/ck_e2e_effective.json").write_text(json.dumps({"_trainer_max_epochs": 5}))
        st = LU.run_status(str(R))
        assert st["train"]["alive"]
        assert st["eta"]["hours_left"] == pytest.approx(1.5 * 100 * 0.4 / 3600, abs=0.01)
        assert "num_training_batches" in st["eta"]["basis"]
    finally:
        p.kill()
