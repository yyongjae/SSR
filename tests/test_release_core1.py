"""Release core patches: refiner KD off with BEV KD on, one-teacher KD, BN mode, online calibration as the default, launcher
records, run_training guards, silent-failure aborts.  CPU only:
  env CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_release_core1.py
"""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_KEYS  # noqa: E402
from navsim.agents.para_ssr.configs.default import ParaSSRConfig  # noqa: E402
from navsim.planning.script import run_training as RT  # noqa: E402
from tools.ck.e2e2 import launch_util2 as L2  # noqa: E402

need_smoke = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
SH = REPO / "tools/ck/e2e2/train_e2e2.sh"
V7 = Path("/home/external-user/ssd/yongjae_refiner/v7")
REFERENCE = V7 / "v2_v15_v7_main_overrides_reference.txt"
SMOKE_SETS = [f"teacher_det_run={TL.SMOKE_T}", f"teacher_map_run={TL.SMOKE_M}", "ep_target=official",
              "kd_calib.enabled=false"]
OFF_EXTRA = "++agent.config.ck_e2e2.enabled=false"


# ----------------------------------------------------------------------------------------------- KD calibration defaults
def test_calibration_default_is_online_with_no_teacher_path():
    c = O.CKE2E2Config()
    assert c.kd_calib["mode"] == "online" and c.kd_calib["det"] is None and c.kd_calib["map"] is None
    assert c.teacher_det_run == "" and c.teacher_map_run == "" and c.use_teachers is True
    y = L2.load_ck2_yaml()
    assert y["teacher_det_run"] is None and y["teacher_map_run"] is None and y["kd_calib"] == c.kd_calib
    assert y["kd_sources"] == ["det", "map"]
    # validate names the missing teacher instead of falling back to an old one
    with pytest.raises(ValueError, match="teacher_det_run is required"):
        O.CKE2E2Config.from_any({"enabled": True, "io_dir": "/x/io"}).validate()


def test_online_mode_with_calibration_off_is_valid():
    d = {"enabled": True, "io_dir": "/x/io", "use_teachers": False, "lambda_kd_score": 0.0,
         "lat_kd": {"balance": "fixed", "fixed": 0.0}, "kd_calib": {"enabled": False}}
    c = O.CKE2E2Config.from_any(d)
    c.validate()
    assert c.kd_calib["mode"] == "online" and O.CKE2E2(c).calib_online is None


@pytest.mark.parametrize("kc", [{"enabled": True, "det": "/old/T.json", "map": "/old/M.json"},
                                {"enabled": True, "det": "/old/T.json", "map": "null"},
                                {"enabled": True, "det": "/old/T.json", "map": "/old/M.json", "mode": "file"}])
def test_archived_file_mode_configs_still_load(kc):
    """configs written before the online mode have no mode key: a calibration path means file mode, no online state."""
    d = {"enabled": True, "io_dir": "/x/io", "teacher_det_run": "/none/T", "teacher_map_run": "/none/M", "kd_calib": kc}
    c = O.CKE2E2Config.from_any(d)
    c.validate()
    assert c.kd_calib["mode"] == "file" and c.kd_calib["det"] == "/old/T.json"
    ck = O.CKE2E2(c)
    assert ck.calib_online is None
    ck.load_state_dict({"ema": ck.ema.state_dict(), "mb": 1, "cum": {}})        # an old callback state resumes


@pytest.mark.skipif(not REFERENCE.is_file(), reason="reference overrides absent")
def test_reference_run_overrides_parse_as_file_mode():
    ck = {}
    for ln in REFERENCE.read_text().splitlines():
        if ln.startswith("++agent.config.ck_e2e2."):
            k, v = ln[len("++agent.config.ck_e2e2."):].split("=", 1)
            node = ck
            for p in k.split(".")[:-1]:
                node = node.setdefault(p, {})
            node[k.split(".")[-1]] = None if v == "null" else v
    assert "mode" not in ck["kd_calib"]
    c = O.CKE2E2Config.from_any(ck)
    c.validate()
    assert c.kd_calib["mode"] == "file" and c.use_teachers and c.kd_sources == ["det", "map"]


@need_smoke
def test_cli_refuses_file_mode_unless_allowed(tmp_path, monkeypatch):
    p = tmp_path / "file.set"
    p.write_text("\n".join(SMOKE_SETS + ["kd_calib.mode=file"]) + "\n")
    argv = ["overrides", "--run-dir", str(tmp_path / "ck2e2e_x"), "--gpus", "0,1", "--wandb", "offline", "--no-deep",
            "--ck-set-file", str(p), "--out", str(tmp_path / "launch/overrides.txt")]
    monkeypatch.delenv("SSR_ALLOW_FILE_CALIB", raising=False)
    assert L2.main(argv) == 3 and not (tmp_path / "launch/overrides.txt").exists()
    monkeypatch.setenv("SSR_ALLOW_FILE_CALIB", "1")
    assert L2.main(argv) == 0 and (tmp_path / "launch/overrides.txt").is_file()
    # the function level still accepts it (archived recipes, tests)
    assert L2.build_overrides2(str(tmp_path / "ck2e2e_y"), ["0", "1"], ck_sets=SMOKE_SETS + ["kd_calib.mode=file"],
                               wandb="offline", teacher_deep=False)[1]["kd_calib"]["mode"] == "file"


# ----------------------------------------------------------------------------------------------- launcher records
def test_refiner_off_from_the_extra_skips_the_refiner_checks(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not run with the refiner off")
    monkeypatch.setattr(L2, "teacher_check", boom)
    monkeypatch.setattr(L2, "warmup_check", boom)
    ov, ck, teachers = L2.build_overrides2(str(tmp_path / "noref"), ["0", "1"], wandb="offline", extra=[OFF_EXTRA])
    assert ck["enabled"] is False and L2.arm_of(ck) == "noref"
    assert teachers["mode"] == "none" and teachers["enabled"] is False and "warmup_sources" not in teachers
    assert OFF_EXTRA in ov and "wandb.tags=[ck2e2e,noref]" in ov


@need_smoke
def test_effective_record_includes_extra_lines(tmp_path):
    p = tmp_path / "x.extra"
    p.write_text("++agent.config.ck_e2e2.lambda_kd_score=0.25\n++agent.config.use_distill=false\n")
    s = tmp_path / "s.set"
    s.write_text("\n".join(SMOKE_SETS) + "\n")
    out = tmp_path / "launch/overrides.txt"
    assert L2.main(["overrides", "--run-dir", str(tmp_path / "ck2e2e_x"), "--gpus", "0,1", "--wandb", "offline",
                    "--no-deep", "--ck-set-file", str(s), "--extra-file", str(p), "--out", str(out)]) == 0
    eff = json.loads((out.parent / "ck_e2e2_effective.json").read_text())
    assert eff["lambda_kd_score"] == 0.25 and eff["_extra_ck_sets"] == ["lambda_kd_score=0.25"] and eff["_arm"] == "main"
    assert "++agent.config.ck_e2e2.lambda_kd_score=0.25" in out.read_text().splitlines()
    # a key the config does not have is refused here, not at agent build
    p.write_text("++agent.config.ck_e2e2.no_such_key=1\n")
    assert L2.main(["overrides", "--run-dir", str(tmp_path / "ck2e2e_x"), "--gpus", "0,1", "--wandb", "offline",
                    "--no-deep", "--ck-set-file", str(s), "--extra-file", str(p), "--out", str(out)]) == 3


@need_smoke
@pytest.mark.parametrize("n", [30, 7])
def test_max_epochs_sets_the_trainer_and_the_agent(tmp_path, n):
    ov, _, _ = L2.build_overrides2(str(tmp_path / "ck2e2e_x"), ["0", "1"], ck_sets=SMOKE_SETS, wandb="offline",
                                   max_epochs=n, teacher_deep=False)
    assert f"agent.config.max_epochs={n}" in ov and f"trainer.params.max_epochs={n}" in ov


@need_smoke
@pytest.mark.parametrize("key, value, msg", [("split", "navtest", "split"), ("split", None, "split"),
                                             ("init_from", "/some/run", "init_from")])
def test_teacher_check_needs_scratch_navtrain_teacher(tmp_path, key, value, msg):
    d = tmp_path / "T"
    d.mkdir()
    cfg = json.loads((TL.SMOKE_T / "config.json").read_text())
    cfg[key] = value
    (d / "config.json").write_text(json.dumps(cfg))
    shutil.copy(TL.SMOKE_T / "done.json", d / "done.json")
    os.symlink(TL.SMOKE_T / "ckpt_last.pt", d / "ckpt_last.pt")
    with pytest.raises(L2.LaunchError, match=msg):
        L2.teacher_check(str(d), str(TL.SMOKE_M), mode="smoke", deep=False, ep_target="official")
    rec = L2.teacher_check(str(TL.SMOKE_T), str(TL.SMOKE_M), mode="smoke", deep=False, ep_target="official")
    assert rec["det"]["split"] == rec["map"]["split"] == "navtrain_train" and rec["det"]["init_from"] == "none"


def _sh(env_extra, *args):
    env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               E2E_WANDB="disable", E2E_DRY_RUN="1", CK2_E2E_APPROVED="1", **env_extra)
    for k in ("E2E_SMOKE", "ARM", "E2E_CK_SET", "E2E_EXTRA", "E2E_NO_LABELER", "WANDB_ENTITY"):
        if k not in env_extra:
            env.pop(k, None)
    return subprocess.run(["bash", str(SH)] + list(args), env=env, capture_output=True, text=True, timeout=300)


@need_smoke
def test_shell_no_labeler_needs_refiner_off_and_prints_the_true_arm(tmp_path):
    s = tmp_path / "s.set"
    s.write_text("\n".join(SMOKE_SETS) + "\n")
    x = tmp_path / "off.extra"
    x.write_text(OFF_EXTRA + "\n")
    base = {"E2E_ROOT": str(tmp_path / "root"), "E2E_NO_DEEP_TEACHER_CHECK": "1"}
    r = _sh(dict(base, RUN="on", E2E_CK_SET=str(s), E2E_NO_LABELER="1"), "start", "--gpus", "0,1")
    assert r.returncode == 3 and "E2E_NO_LABELER=1 with the refiner on" in r.stderr, r.stderr[-800:]
    r = _sh(dict(base, RUN="off", E2E_EXTRA=str(x), E2E_NO_LABELER="1"), "start", "--gpus", "0,1")
    assert r.returncode == 0 and "arm noref" in r.stdout, r.stderr[-800:]
    eff = json.loads((tmp_path / "root/off/launch_dryrun/ck_e2e2_effective.json").read_text())
    assert eff["enabled"] is False and eff["_arm"] == "noref"
    assert json.loads((tmp_path / "root/off/launch_dryrun/teachers.json").read_text())["mode"] == "none"
    # no labeler flag needed either: the effective config decides
    r = _sh(dict(base, RUN="off2", E2E_EXTRA=str(x)), "start", "--gpus", "0,1")
    assert r.returncode == 0 and "arm noref" in r.stdout, r.stderr[-800:]


def test_shell_labeler_gets_the_tokens_of_the_run_and_the_data_env(tmp_path):
    L = tmp_path / "root/r/launch"
    L.mkdir(parents=True)
    (L / "ck_e2e2_effective.json").write_text(json.dumps({"warmup": {"packed": "/elsewhere/packed/navtrain_train"}}))
    base = {"E2E_ROOT": str(tmp_path / "root"), "RUN": "r"}
    r = _sh(base, "_labeler_cmd", "--gpus", "0,1,2,3")
    assert r.returncode == 0 and "--tokens /elsewhere/packed/navtrain_train/tokens.parquet" in r.stdout, r.stderr
    assert "--world-size 4" in r.stdout and "--workers 48" in r.stdout
    (L / "ck_e2e2_effective.json").unlink()
    assert "--tokens" not in _sh(base, "_labeler_cmd", "--gpus", "0,1").stdout    # a run from before this record
    env = _sh(base, "_print_env").stdout.splitlines()
    assert {e.split("=")[0] for e in env} == {"NUPLAN_MAP_VERSION", "NUPLAN_MAPS_ROOT", "OPENSCENE_DATA_ROOT"}
    text = SH.read_text()
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    env_lines = [ln for ln in code if '"${DATA_ENV[@]}"' in ln and "setsid nohup" in ln]
    assert len(env_lines) == 2 and any("labeler env" in ln for ln in env_lines)


def test_shell_writes_no_secret_and_has_no_default_entity():
    text = SH.read_text()
    assert ".wandb_netrc" not in text and "password" not in text
    assert "e2ekd" not in text and 'WANDB_ENTITY="$WANDB_ENTITY"' in text
    assert 'log "start RUN=$RUN arm=$TRUE_ARM ' in text           # launch.log carries the arm the run really is


@pytest.mark.parametrize("script", ["train_v2_v15_v7.sh", "smoke_v2_final.sh", "train_v2_nokd.sh", "smoke_v2_nokd.sh"])
def test_wrappers_state_they_are_a_direct_route(script):
    assert re.search(r"export .*SSR_ALLOW_DIRECT=1", (REPO / "scripts/training" / script).read_text())


def test_launcher_alone_grants_no_direct_route():
    assert "SSR_ALLOW_DIRECT" not in SH.read_text()


# ----------------------------------------------------------------------------------------------- run_training guards
def test_navsim_import_guard(tmp_path):
    RT.check_navsim_import(str(REPO / "navsim/__init__.py"), REPO)
    with pytest.raises(RuntimeError, match="not from this repo"):
        RT.check_navsim_import("/some/other/checkout/navsim/__init__.py", REPO)
    with pytest.raises(RuntimeError, match="not from this repo"):
        RT.check_navsim_import(None, REPO)


def test_launch_route_guard():
    RT.check_launch_route({"SSR_EXPERIMENT_SHA": "abc123"})
    RT.check_launch_route({"SSR_ALLOW_DIRECT": "1"})
    for env in ({}, {"SSR_EXPERIMENT_SHA": ""}, {"SSR_ALLOW_DIRECT": "0"}):
        with pytest.raises(RuntimeError, match="without a recipe"):
            RT.check_launch_route(env)


def test_log_splits_are_disjoint_in_the_repo_and_a_shared_log_is_refused():
    y = OmegaConf.load(REPO / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml")
    RT.check_log_splits(y.train_logs, y.val_logs, y.test_logs)
    assert len(y.test_logs) > 0
    for a, b, c, msg in ((["x"], ["x"], [], "train / val"), (["x"], [], ["x"], "train / test"),
                         ([], ["x"], ["x"], "val / test")):
        with pytest.raises(RuntimeError, match=msg):
            RT.check_log_splits(a, b, c)


@pytest.mark.parametrize("name", ["navtest", "navhard_two_stage", "navhard", "private_test_e2e", "warmup_test_e2e"])
def test_test_scene_filters_are_refused(name):
    with pytest.raises(RuntimeError, match="test set"):
        RT.check_scene_filter_name(name)


@pytest.mark.parametrize("name", ["navtrain", "navtrain_mini", "all_scenes", None])
def test_train_scene_filters_pass(name):
    RT.check_scene_filter_name(name)


def test_train_tokens_record_and_navtest_overlap(tmp_path):
    nav = RT.navtest_tokens()
    assert len(nav) == 12146 and "5798a6e25f2553e4" in nav
    rec = RT.check_and_record_train_tokens(["b", "a"], str(tmp_path / "out"))
    assert rec["count"] == 2
    disk = json.loads((tmp_path / "out/train_tokens.json").read_text())
    assert disk == rec and disk == RT.check_and_record_train_tokens(["a", "b"], str(tmp_path / "out"))    # sorted
    assert RT.check_and_record_train_tokens(["c"], str(tmp_path / "r1"), is_rank_zero=False)["count"] == 1
    assert not (tmp_path / "r1").exists()
    with pytest.raises(RuntimeError, match="navtest tokens"):
        RT.check_and_record_train_tokens(["5798a6e25f2553e4", "x"], str(tmp_path / "bad"))
    assert not (tmp_path / "bad").exists()


def test_navtrain_filter_has_no_navtest_token():
    y = OmegaConf.load(REPO / "navsim/planning/script/config/common/scene_filter/navtrain.yaml")
    RT.check_and_record_train_tokens(list(y.tokens), None)


def test_dataset_tokens_both_kinds():
    assert RT.dataset_tokens(SimpleNamespace(_scene_loader=SimpleNamespace(tokens=["a"]), tokens=["no"])) == ["a"]
    assert RT.dataset_tokens(SimpleNamespace(tokens=["b"])) == ["b"]


class _Stop(Exception):
    pass


@pytest.mark.parametrize("over, want", [({}, 0), ({"seed": 3}, 3)])
def test_seed_reads_the_config(monkeypatch, over, want):
    seen = {}
    monkeypatch.setenv("SSR_ALLOW_DIRECT", "1")
    monkeypatch.setattr(RT, "HydraConfig", SimpleNamespace(
        get=lambda: SimpleNamespace(runtime=SimpleNamespace(choices={"scene_filter": "navtrain"}))))
    monkeypatch.setattr(RT.pl, "seed_everything", lambda s, workers=True: seen.update(seed=s))

    def stop(_):
        raise _Stop

    monkeypatch.setattr(RT, "instantiate", stop)
    cfg = OmegaConf.create({"train_logs": ["a"], "val_logs": ["b"], "test_logs": ["c"], "agent": {}, "output_dir": "/x",
                            **over})
    with pytest.raises(_Stop):
        RT.main.__wrapped__(cfg)
    assert seen["seed"] == want
    assert OmegaConf.load(REPO / "navsim/planning/script/config/training/default_training.yaml").seed == 0


def test_guards_are_wired_into_main(monkeypatch, tmp_path):
    """main: the navtest check sees the final training tokens and train_tokens.json lands in output_dir."""
    monkeypatch.setenv("SSR_EXPERIMENT_SHA", "abc123")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(RT, "HydraConfig", SimpleNamespace(
        get=lambda: SimpleNamespace(runtime=SimpleNamespace(choices={"scene_filter": "navtrain"}))))
    monkeypatch.setattr(RT.pl, "seed_everything", lambda s, workers=True: None)
    monkeypatch.setattr(RT, "instantiate", lambda c: SimpleNamespace())
    monkeypatch.setattr(RT, "AgentLightningModule", lambda agent: None)

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(RT, "DataLoader", stop)

    class Data(list):
        tokens = property(list)

    monkeypatch.setattr(RT, "build_datasets", lambda cfg, agent: (Data(["t1", "t2"]), Data()))
    cfg = OmegaConf.create({"train_logs": ["a"], "val_logs": ["b"], "test_logs": ["c"], "agent": {},
                            "output_dir": str(tmp_path / "train"), "use_cache_without_dataset": False,
                            "dataloader": {"params": {}}})
    with pytest.raises(_Stop):
        RT.main.__wrapped__(cfg)
    assert json.loads((tmp_path / "train/train_tokens.json").read_text())["count"] == 2
    monkeypatch.setattr(RT, "build_datasets", lambda cfg, agent: (Data(["t1", "5798a6e25f2553e4"]), Data()))
    with pytest.raises(RuntimeError, match="navtest tokens"):
        RT.main.__wrapped__(cfg)
    bad = OmegaConf.merge(cfg, {"test_logs": ["a"]})
    with pytest.raises(RuntimeError, match="train / test"):
        RT.main.__wrapped__(bad)


def test_new_keys_reach_the_config_through_hydra():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/training"), version_base="1.2"):
        cfg = compose(config_name="default_training", overrides=[
            "agent=para_ssr_agent", "experiment_name=x", "split=trainval", "seed=5",
            "++agent.config.backbone_bn_mode=frozen", "++agent.config.distill_off_banks=[det]",
            "++agent.config.ck_e2e2.kd_sources=[map]", "++agent.config.ck_e2e2.kd_score_keys=[dac,ep]"])
    GlobalHydra.instance().clear()
    assert cfg.seed == 5
    pc = instantiate(cfg.agent.config)
    assert pc.backbone_bn_mode == "frozen" and list(pc.distill_off_banks) == ["det"]
    assert O.CKE2E2Config.from_any(pc.ck_e2e2).kd_sources == ["map"]


def _script(env_extra, *args, tmp):
    env = {k: v for k, v in os.environ.items() if k not in ("SSR_EXPERIMENT_SHA", "SSR_ALLOW_DIRECT", "PYTHONPATH")}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NAVSIM_EXP_ROOT=str(tmp), **env_extra)
    return subprocess.run([sys.executable, str(REPO / "navsim/planning/script/run_training.py"), "agent=para_ssr_agent",
                           "experiment_name=guard_test", "split=trainval", f"output_dir={tmp}/out/", *args], cwd=str(tmp),
                          env=env, capture_output=True, text=True, timeout=300)


def test_script_guards_end_to_end(tmp_path):
    good = {"PYTHONPATH": str(REPO)}
    r = _script(good, tmp=tmp_path)
    assert r.returncode != 0 and "without a recipe" in r.stderr, r.stderr[-800:]
    r = _script(dict(good, SSR_ALLOW_DIRECT="1"), "scene_filter=navtest", tmp=tmp_path)
    assert r.returncode != 0 and "is a test set" in r.stderr, r.stderr[-800:]
    # no PYTHONPATH: this venv's editable install may point at another checkout
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    where = subprocess.run([sys.executable, "-c", "import navsim; print(navsim.__file__)"], env=env, cwd=str(tmp_path),
                           capture_output=True, text=True).stdout.strip()
    if where and REPO not in Path(where).resolve().parents:
        r = _script({"SSR_ALLOW_DIRECT": "1"}, tmp=tmp_path)
        assert r.returncode != 0 and "not from this repo" in r.stderr, r.stderr[-800:]


# ----------------------------------------------------------------------------------------------- dataset order
def test_scene_order_does_not_depend_on_the_directory_order(tmp_path, monkeypatch):
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import filter_scenes

    for name, tok in (("log_b", "t_b"), ("log_a", "t_a"), ("log_c", "t_c")):
        with open(tmp_path / f"{name}.pkl", "wb") as f:
            pickle.dump([{"token": tok, "roadblock_ids": [1]}], f)
    real = Path.iterdir

    def scrambled(self):
        return reversed(sorted(real(self), key=lambda p: p.name[::-1]))

    sf = SceneFilter(num_history_frames=1, num_future_frames=0, frame_interval=1, has_route=False)
    want = ["t_a", "t_b", "t_c"]
    assert list(filter_scenes(tmp_path, sf)) == want
    monkeypatch.setattr(Path, "iterdir", scrambled)
    assert list(filter_scenes(tmp_path, sf)) == want


# ----------------------------------------------------------------------------------------------- map targets
@pytest.mark.parametrize("map_head, distill, want", [(True, False, True), (False, True, True), (False, False, False)])
def test_map_targets_also_for_the_bev_kd(map_head, distill, want):
    from navsim.agents.para_ssr.para_ssr_targets import ParaSSRTargetBuilder
    b = ParaSSRTargetBuilder.__new__(ParaSSRTargetBuilder)
    b._config = SimpleNamespace(use_det_motion_head=False, use_map_head=map_head, use_distill=distill)
    b._trajectory_sampling = SimpleNamespace(num_poses=8)
    b._compute_map_targets = lambda scene, idx: {"gt_map_pts": 1}
    frame = SimpleNamespace(ego_status=SimpleNamespace(driving_command=[0, 1, 0, 0]), token="t")
    scene = SimpleNamespace(scene_metadata=SimpleNamespace(num_history_frames=4), frames=[frame] * 4,
                            get_future_trajectory=lambda num_trajectory_frames: SimpleNamespace(poses=np.zeros((8, 3))))
    assert ("gt_map_pts" in b.compute_targets(scene)) is want


# ----------------------------------------------------------------------------------------------- BN mode
def test_bn_mode_default_and_real_model():
    from test_ck_e2e_model_agent import v2_config
    from navsim.agents.para_ssr.para_ssr_model import ParaSSRModel
    assert ParaSSRConfig().backbone_bn_mode == "moving"
    with pytest.raises(ValueError, match="backbone_bn_mode"):
        ParaSSRModel(v2_config(backbone_bn_mode="bogus"))
    for mode, want in (("moving", True), ("frozen", False)):
        torch.manual_seed(0)
        m = ParaSSRModel(v2_config(backbone_bn_mode=mode))
        m.train()
        assert m.image_encoder.bn1.training is want and m.image_encoder.layer4[0].bn1.training is want
        assert not m.image_encoder.bn1.weight.requires_grad and not m.image_encoder.conv1.weight.requires_grad
        m.eval()
        assert not m.image_encoder.bn1.training


# ----------------------------------------------------------------------------------------------- one-teacher refiner KD
def test_kd_sources_rules():
    base = {"enabled": True, "io_dir": "/x/io", "teacher_det_run": "/none/T", "teacher_map_run": "/none/M"}
    assert O.CKE2E2Config.from_any(dict(base, kd_sources="[map,det]")).kd_sources == ["det", "map"]
    for bad in ([], ["lidar"], ["det", "det"]):
        with pytest.raises(ValueError, match="kd_sources must be"):
            O.CKE2E2Config.from_any(dict(base, kd_sources=bad))
    for src, keys, ok in ((["map"], ["dac", "ep"], True), (["det"], ["nc", "ep", "ttc"], True),
                          (["map"], ["nc", "dac", "ep", "ttc"], False), (["det"], ["nc", "dac"], False),
                          (["map"], ["dac", "comfort"], False)):
        c = O.CKE2E2Config.from_any(dict(base, kd_sources=src, kd_score_keys=keys))
        if ok:
            c.validate()
        else:
            with pytest.raises(ValueError, match="not keys of that teacher"):
                c.validate()


@need_smoke
def test_one_source_combine_rule_and_masks():
    f, t = TL.fresh(2)
    pred = TL.fake_predictions(2, seed=4)
    cand = pred["trajectory_anchors"][None, :16].expand(2, -1, -1, -1).clone()
    status = f["status_feature"]
    args = (t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, status)
    both = O.TeacherPair2(str(TL.SMOKE_T), str(TL.SMOKE_M), "last", "cpu", amp=False).run(*args)
    jd, jm = CK_KEYS.index("dac"), CK_KEYS.index("nc")
    assert torch.equal(both["kd_e_lat"], 0.5 * (both["e_det"] + both["e_map"]))
    assert torch.equal(both["kd_score_prob"][..., jm], both["p_det"][..., jm])
    assert torch.equal(both["kd_score_prob"][..., jd], both["p_map"][..., jd])
    # a token without the OTHER teacher's BEV stays in the mask of a one-teacher run, its own missing token leaves it
    no_map = (t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], torch.zeros_like(t["kd_ok_1"]), cand, status)
    no_det = (t["kd_bev_0"], torch.zeros_like(t["kd_ok_0"]), t["kd_bev_1"], t["kd_ok_1"], cand, status)
    for src, p, e, drop_other, drop_own in (("det", "p_det", "e_det", no_map, no_det),
                                            ("map", "p_map", "e_map", no_det, no_map)):
        tp = O.TeacherPair2(str(TL.SMOKE_T), str(TL.SMOKE_M), "last", "cpu", amp=False, sources=[src])
        r = tp.run(*args)
        torch.testing.assert_close(r["kd_score_prob"], r[p], rtol=0, atol=1e-7)
        assert torch.equal(r["kd_e_lat"], r[e]) and bool(r["kd_ok"].all())
        assert torch.equal(r[e], both[e]) and torch.equal(r[p], both[p])     # the same teacher, only the targets differ
        assert bool(tp.run(*drop_other)["kd_ok"].all()) and not bool(tp.run(*drop_own)["kd_ok"].any())
    assert not bool(O.TeacherPair2(str(TL.SMOKE_T), str(TL.SMOKE_M), "last", "cpu", amp=False).run(*no_map)["kd_ok"].any())
    with pytest.raises(ValueError, match="sources"):
        O.TeacherPair2(str(TL.SMOKE_T), str(TL.SMOKE_M), "last", "cpu", amp=False, sources=[])


@need_smoke
@pytest.mark.parametrize("src, keys", [(["det"], ["nc", "ep", "ttc"]), (["map"], ["dac", "ep"])])
def test_one_source_student_gets_every_gradient(tmp_path, src, keys):
    c = TL.cfg2(tmp_path, kd_sources=src, kd_score_keys=keys)
    ck = O.CKE2E2(c, max_epochs=30)
    assert ck.teachers("cpu").sources == tuple(src)
    torch.manual_seed(0)
    st = O.build_student_ck2(c)
    f, t = TL.fresh(2)
    pred = TL.fake_predictions(2, seed=3)
    loss, logs = ck.loss(st, f, t, pred, v2_loss=(pred["bev_embed"] ** 2).mean())
    loss.backward()
    assert not [n for n, p in st.named_parameters() if p.requires_grad and p.grad is None]
    assert float(logs["ck2/kd_score"]) > 0 and float(logs["ck2/kd_ok_frac"]) == 1.0
    assert float(pred["bev_embed"].grad.abs().sum()) > 0


# ----------------------------------------------------------------------------------------------- BEV KD arms (own process)
@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    names = ["main", "nokd_bev_on", "det_off", "map_off", "det_off_bev", "det_off_ref", "noref_map_off", "noref_nobev"]
    if not (Path("/home/external-user/datasets/teacher_cache/refiner/det/done.json").is_file() and Path("/workspace/teacher_cache/bevfusion").is_dir()
            and TL.FIX2.is_file()):
        pytest.skip("64 m teachers / teacher caches / fixture absent")
    if REFERENCE.is_file() and (V7 / "e2e/v2_v15_v7_main/train/lightning_logs").is_dir():
        names.append("reference")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    r = subprocess.run([sys.executable, "-W", "ignore", str(REPO / "tests/release_agent_arms.py"),
                        str(tmp_path_factory.mktemp("arms")), ",".join(names)], env=env, capture_output=True, text=True,
                       timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_every_arm_runs_with_no_unused_parameter(arms):
    for name, a in arms.items():
        if name != "reference":
            assert a["no_grad"] == [] and np.isfinite(a["loss"]), name


def test_arms_share_the_init_outside_refiner_and_bev_kd(arms):
    shas = {a["init_sha"] for n, a in arms.items() if n != "reference"}
    assert len(shas) == 1, {n: a["init_sha"][:8] for n, a in arms.items() if n != "reference"}


def test_refiner_kd_off_with_bev_kd_on(arms):
    a = arms["nokd_bev_on"]
    assert a["builders"][-2:] == ["CK2E2ETargetBuilder", "DistillTeacherTargetBuilder"]
    assert a["teachers_built"] is False and a["calib_online"] is False and a["kd_score"] == 0.0
    assert "loss_distill_bevfusion" in a["loss_keys"] and "loss_distill_resmap" in a["loss_keys"]
    assert a["teacher_ok_frac"] == 1.0
    assert "DistillTeacherTargetBuilder" not in arms["main"]["builders"]


@pytest.mark.parametrize("arm, frozen, kept", [("det_off", 0, "resmap"), ("map_off", 1, "bevfusion"),
                                               ("det_off_bev", 0, "resmap"), ("noref_map_off", 1, "bevfusion")])
def test_off_bank_is_frozen_and_out_of_the_loss(arms, arm, frozen, kept):
    a = arms[arm]
    assert a["frozen"] == [f"anchor_distill.probes.{frozen}", f"anchor_distill.student_proj.{frozen}"]
    assert a["loss_keys"] == sorted([f"loss_distill_{kept}", "loss_distill_tok", f"loss_probe_{kept}"])


def test_off_paths_are_independent(arms):
    assert arms["det_off_ref"]["frozen"] == [] and "loss_distill_bevfusion" in arms["det_off_ref"]["loss_keys"]
    assert arms["det_off_bev"]["kd_score"] == arms["main"]["kd_score"]            # same step, refiner KD untouched
    assert arms["det_off_ref"]["kd_score"] == arms["det_off"]["kd_score"] != arms["main"]["kd_score"]
    assert arms["det_off"]["n_keys_state"] == arms["main"]["n_keys_state"]        # the off bank stays in the state dict


def test_old_main_run_checkpoint_loads_strictly(arms):
    if "reference" not in arms:
        pytest.skip("reference run absent")
    a = arms["reference"]
    assert a["missing"] == [] and a["unexpected"] == [] and a["n_keys"] > 0
    assert a["kd_calib_mode"] == "file" and a["calib_online"] is False and a["bn_mode"] == "moving"


def test_off_bank_validation():
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    cfg = dict(plan_anchor=True, plan_anchor_file=str(REPO / "resources/planning_vb/trajectory_anchors_256.npy"),
               distill_selector_v15=True, distill_teacher_source="targets")
    for off, msg in ((["lidar"], "must name one"), (["det", "map"], "must name one")):
        with pytest.raises(ValueError, match=msg):
            ParaSSRAgent._build_distill(SimpleNamespace(distill_off_banks=off, **cfg))
    with pytest.raises(ValueError, match="targets"):
        ParaSSRAgent._build_distill(SimpleNamespace(distill_off_banks=["det"], **dict(cfg, distill_teacher_source="store")))
    with pytest.raises(ValueError, match="probe path"):
        ParaSSRAgent._build_distill(SimpleNamespace(distill_off_banks=["det"], **dict(cfg, distill_selector_v15=False,
                                                                                      distill_selector_v8=True)))


def test_off_bank_builder_reads_nothing_of_that_cache(tmp_path):
    from navsim.agents.para_ssr.distill import teacher_store as TS
    opened = []

    class FakeStore:
        def __init__(self, root, name, **kw):
            opened.append(name)

        def load_one_token(self, token):
            return np.ones((256, 100, 100), np.float32)

    orig = TS.TeacherFeatureStore
    TS.TeacherFeatureStore = FakeStore
    try:
        b = TS.DistillTeacherTargetBuilder(SimpleNamespace(distill_feature_root=str(tmp_path), embed_dims=256, bev_h=100,
                                                           bev_w=100, distill_cache_size=(100, 100),
                                                           distill_off_banks=["det"]))
        scene = SimpleNamespace(scene_metadata=SimpleNamespace(num_history_frames=4),
                                frames=[SimpleNamespace(token="t")] * 4)
        out = b.compute_targets(scene)
    finally:
        TS.TeacherFeatureStore = orig
    assert opened == ["resmap"]
    assert not bool(out["kd_ok_0"]) and bool(out["kd_ok_1"]) and float(out["kd_bev_0"].abs().sum()) == 0.0


@pytest.mark.skipif(not Path("/workspace/teacher_cache/bevfusion/cache_train_100x100").is_dir(),
                    reason="teacher caches absent")
def test_the_two_teacher_readers_agree():
    """kd_bev_i of a run without refiner teachers (DistillTeacherTargetBuilder) = the refiner's reader, bit for bit."""
    env = dict(os.environ, CK_BEV64="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-W", "ignore", str(REPO / "tools/v2_v15_v7/check_teacher_inputs.py"), "--n", "6"],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "OK: identical teacher BEVs" in r.stdout, r.stdout[-500:] + r.stderr[-1500:]


# ----------------------------------------------------------------------------------------------- silent failures
@need_smoke
def test_input_fraction_aborts(tmp_path):
    ck = O.CKE2E2(TL.cfg2(tmp_path, strict_rows=0.999), max_epochs=30)
    yes, no = torch.ones(1000, dtype=torch.bool), torch.zeros(1000, dtype=torch.bool)
    ck._check_inputs(yes, None)
    ck._check_inputs(yes[:999], None)
    ck._check_inputs(yes[:1], None)
    assert ck._inputs_checked                                       # all present: passed once, not checked again
    ck._check_inputs(no, None)
    ck = O.CKE2E2(TL.cfg2(tmp_path, strict_rows=0.999), max_epochs=30)
    ck._check_inputs(yes, None)
    with pytest.raises(RuntimeError, match="ref_gt_ok"):
        ck._check_inputs(no, None)
    ck = O.CKE2E2(TL.cfg2(tmp_path, strict_rows=0.999), max_epochs=30)
    ck._check_inputs(yes, yes[None])
    with pytest.raises(RuntimeError, match="kd_ok_frac"):
        ck._check_inputs(yes, no[None])
    # a few missing tokens are tolerated up to strict_rows, other ranks do not check, 0 switches it off
    ck = O.CKE2E2(TL.cfg2(tmp_path, strict_rows=0.999), max_epochs=30)
    ck._check_inputs(torch.cat([yes[:998], no[:2]]), None)
    ck._check_inputs(yes, None)
    ck = O.CKE2E2(TL.cfg2(tmp_path, strict_rows=0.999), max_epochs=30)
    ck.rank = 1
    ck._check_inputs(no, None)
    ck._check_inputs(no, None)
    ck = O.CKE2E2(TL.cfg2(tmp_path, strict_rows=0.0), max_epochs=30)
    ck._check_inputs(no, None)
    ck._check_inputs(no, None)


@need_smoke
def test_loss_reaches_the_input_check(tmp_path):
    for name, cut, msg in (("gt", "ref", "ref_gt_ok"), ("kd", "kd", "kd_ok_frac")):
        c = TL.cfg2(tmp_path / name, strict_rows=0.999)
        ck = O.CKE2E2(c, max_epochs=30)
        ck._inputs.update(tokens=1998, ref_ok=1998, kd_ok=1, kd_all=1)
        torch.manual_seed(0)
        st = O.build_student_ck2(c)
        f, t = TL.fresh(2)
        if cut == "ref":
            t["ref_gt_ok"] = torch.zeros_like(t["ref_gt_ok"])
            ck._inputs.update(tokens=1998, ref_ok=0)
        else:
            t["kd_ok_0"] = torch.zeros_like(t["kd_ok_0"])
        with pytest.raises(RuntimeError, match=msg):
            ck.loss(st, f, t, TL.fake_predictions(2, seed=3))
    # no teachers: the KD fraction is not checked (always 0), the surrogate GT still is
    c = TL.cfg2(tmp_path / "nt", strict_rows=0.999, use_teachers=False, teacher_det_run=None, teacher_map_run=None,
                lambda_kd_score=0.0, lat_kd={"balance": "fixed", "fixed": 0.0, "start_mb": 0})
    ck = O.CKE2E2(c, max_epochs=30)
    ck._inputs.update(tokens=1998, ref_ok=1998)
    torch.manual_seed(0)
    st = O.build_student_ck2(c)
    f, t = TL.fresh(2)
    t = {k: v for k, v in t.items() if not k.startswith(("kd_bev_", "kd_ok_"))}
    ck.loss(st, f, t, TL.fake_predictions(2, seed=3))
    assert ck._inputs_checked


def test_distill_teacher_fraction_aborts():
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent

    def agent(strict=0.999):
        a = ParaSSRAgent.__new__(ParaSSRAgent)
        torch.nn.Module.__init__(a)
        a._config = SimpleNamespace(distill_strict_ok=strict)
        return a

    yes, no = torch.ones(1000, dtype=torch.bool), torch.zeros(1000, dtype=torch.bool)
    a = agent()
    a._check_distill_teachers(yes)
    with pytest.raises(RuntimeError, match="teacher_ok_frac"):
        a._check_distill_teachers(no)
    a = agent()
    a._check_distill_teachers(yes)
    a._check_distill_teachers(yes)
    a._check_distill_teachers(no)                                   # passed once: not checked again
    a = agent(0.0)
    a._check_distill_teachers(no)
    a._check_distill_teachers(no)
    a = agent()
    a.eval()
    a._check_distill_teachers(no)
    a._check_distill_teachers(no)
    assert ParaSSRConfig().distill_strict_ok == 0.999


@need_smoke
def test_label_fallback_marks_the_run_and_does_not_stop_it(tmp_path):
    ck = O.CKE2E2(TL.cfg2(tmp_path), max_epochs=30)
    p = Path(ck.cfg.io_dir) / "INVALID_LABELS.json"
    ck.epoch = 6
    ck.cum["G_src_prev"], ck.cum["G_src_fallback"] = 1000, 3
    ck.label_source_epoch_start()
    ck.cum["G_src_prev"] += 995
    ck.cum["G_src_fallback"] += 5
    rec = ck.label_source_epoch_end()
    assert rec["frac"] == 0.005 and rec["fallback"] == 5 and not p.exists()        # per epoch, not cumulative
    ck.epoch = 7
    ck.label_source_epoch_start()
    ck.cum["G_src_prev"] += 90
    ck.cum["G_src_fallback"] += 8
    ck.cum["G_src_nodata"] += 2
    assert ck.label_source_epoch_end()["frac"] == 0.1
    d = json.loads(p.read_text())
    assert [e["epoch"] for e in d["epochs"]] == [7] and d["epochs"][0]["nodata"] == 2
    ck.epoch = 8
    ck.label_source_epoch_start()
    ck.cum["G_src_nodata"] += 2
    ck.label_source_epoch_end()
    assert [e["epoch"] for e in json.loads(p.read_text())["epochs"]] == [7, 8]
    ck2 = O.CKE2E2(TL.cfg2(tmp_path / "wu"), max_epochs=30)
    ck2.epoch = 1
    ck2.label_source_epoch_start()
    assert ck2.label_source_epoch_end() is None


@need_smoke
def test_callback_writes_the_label_record_at_epoch_end(tmp_path):
    ck = O.CKE2E2(TL.cfg2(tmp_path), max_epochs=30)
    ck.refresh_labels = lambda: None
    cb = O.make_ck2_callback(SimpleNamespace(_ck_e2e2=ck))
    tr = SimpleNamespace(current_epoch=6, global_rank=0, world_size=1, is_global_zero=True, global_step=0)
    cb.on_train_epoch_start(tr, None)
    ck.cum["G_src_fallback"] += 4
    cb.on_train_epoch_end(tr, None)
    io = Path(ck.cfg.io_dir)
    assert json.loads((io / "INVALID_LABELS.json").read_text())["epochs"][0]["frac"] == 1.0
    end = [json.loads(ln) for ln in (io / "epochs.jsonl").read_text().splitlines()][-1]
    assert end["event"] == "epoch_end" and end["label_sources"]["fallback"] == 4


# ----------------------------------------------------------------------------------------------- teacher trainer
def test_teacher_r34_evals_flag(monkeypatch):
    from tools.ck import train_ck2 as T2
    from tools.ck.data import ck2_dataset, ck_dataset
    calls = {"cand": 0}

    class FakeCK2:
        def __init__(self, *a, **k):
            self.rows = np.arange(100)

    class FakeCK:
        def __init__(self, *a, labels=None, **k):
            assert labels == "cand"
            calls["cand"] += 1

    monkeypatch.setattr(ck2_dataset, "CK2Dataset", FakeCK2)
    monkeypatch.setattr(ck_dataset, "CKDataset", FakeCK)
    monkeypatch.setattr(T2.TC, "packed_tokens", lambda s: pd.DataFrame({"log": ["a"] * 100}))
    monkeypatch.setattr(T2.U, "log_stratified_rows", lambda logs, n, seed: np.arange(min(n, len(logs))))
    def args(argv):
        return T2.resolve_args(T2.get_parser().parse_args(argv))

    base = ["--arm", "T", "--run-name", "x", "--best-metric", "none", "--val-split", "navtrain_val"]
    kinds = [(n, k) for n, k, _ in T2.build_datasets(args(base))["evals"]]
    assert [k for _, k in kinds] == ["r34", "raw", "r34", "raw"] and calls["cand"] == 2
    calls["cand"] = 0
    kinds = [(n, k) for n, k, _ in T2.build_datasets(args(base + ["--r34-evals", "0"]))["evals"]]
    assert [k for _, k in kinds] == ["raw", "raw"] and calls["cand"] == 0
    # the default best metric of arm T reads an r34 set
    with pytest.raises(SystemExit, match="r34"):
        args(["--arm", "T", "--run-name", "x", "--val-split", "navtrain_val", "--r34-evals", "0"])
    args(["--arm", "T", "--run-name", "x", "--val-split", "navtrain_val", "--r34-evals", "0", "--best-metric",
          "navtrain_val_raw:pdms:max"])
