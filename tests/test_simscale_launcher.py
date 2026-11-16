"""trainval + SimScale recipe, launcher side: the compose settings DATA and TF32, scripts/experiment.sh (block,
MIN_FREE_GB by DATA, verify_assets --data), scripts/train_simscale.sh, the epoch length behind status and the smoke
estimate (launch_util2.train_tokens), the token checks of tools/release/smoke_check.py.  All CPU:
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD $SSR_PY -m pytest -q -p no:cacheprovider tests/test_simscale_launcher.py
The real dry run of the wrapper needs the trainval_simscale assets and the camera-teacher caches; it skips without them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import test_release_compose as TRC  # noqa: E402  (its arms, stubs and environment)
from tools.ck.e2e2 import launch_util2 as L2  # noqa: E402
from tools.release import compose as C  # noqa: E402
from tools.release import smoke_check as SC  # noqa: E402

EXP = REPO / "scripts/experiment.sh"
WRAP = REPO / "scripts/train_simscale.sh"
WRAPPER = {"DATA": "trainval_simscale", "BACKBONE": "dino_vits14_1148", "DET_TEACHER": "bevformer",
           "MAP_TEACHER": "mask2map", "TF32": "on"}
FIXED_ARGS = " ".join(f"{k}={v}" for k, v in WRAPPER.items())
TVSIM = {"DATA": "trainval_simscale", "DET_TEACHER": "bevformer", "MAP_TEACHER": "mask2map"}   # DATA alone: camera teachers
MAIN_NAME = "main_899596"     # the main recipe on every server (reference PASS); a new line in it changes this
CK = "agent.config.ck_e2e2."


def _env(**extra):
    env = {k: v for k, v in TRC._env().items() if k not in ("MIN_FREE_GB", "MEM_LIMIT_GB")}   # defaults under test
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run(script, *args, timeout=600, **extra):
    return subprocess.run(["bash", str(script), *args], env=_env(**extra), capture_output=True, text=True, timeout=timeout)


def _kv(c):
    return {C.LU._okey(o): o.split("=", 1)[1] for o in c.overrides if "=" in o}


def _sig(c):
    return (c.run, c.hash6, c.overrides, c.set_lines, c.extra_lines, c.env, c.pins, c.profile, c.default_recipe)


# ----------------------------------------------------------------------------------------------- names
@pytest.mark.parametrize("gpus, part", [("0,1,2,3,4,5,6,7", ""), ("0,1,2,3", "_g4"), ("4,5", "_g2")])
def test_the_recipe_name_on_8_4_2_gpus(gpus, part):
    n = C.compose(dict(WRAPPER, GPUS=gpus)).run
    assert re.fullmatch(rf"dino_vits14_1148_bevformer-mask2map_tvsim_tf32{part}_[0-9a-f]{{6}}", n), n
    TRC._clean_name(n)


def test_each_new_setting_has_its_part_and_its_hash():
    main, d, t = C.compose({}), C.compose(TVSIM), C.compose({"TF32": "on"})
    assert d.run == f"bevformer-mask2map_tvsim_{d.hash6}" and t.run == f"tf32_{t.hash6}"
    assert len({main.hash6, d.hash6, t.hash6, C.compose(WRAPPER).hash6}) == 4
    assert not d.default_recipe and not t.default_recipe and d.profile == t.profile == "full"


def test_simscale_data_takes_camera_only_teachers():
    for kw, what in (({"DATA": "trainval_simscale"}, "DET_TEACHER=bevfusion"),
                     ({"DATA": "trainval_simscale", "DET_TEACHER": "bevformer"}, "MAP_TEACHER=resmap"),
                     ({**TVSIM, "MAP_TEACHER": "resmap"}, "MAP_TEACHER=resmap")):
        with pytest.raises(C.Refused, match=f"{what}: SimScale scenes have no LiDAR"):
            C.compose(kw)
    off = C.compose({"DATA": "trainval_simscale", "REFINER_KD": "off", "BEV_KD": "off"})   # no KD: no teacher read
    assert "bevfusion" not in off.run and C.compose({"DET_TEACHER": "bevformer"}).run.startswith("bevformer-resmap_")


def test_the_recipe_has_one_name_on_any_server():
    code = ("from tools.release import compose as C; import json; c = C.compose(%r); "
            "print(json.dumps({'run': c.run, 'over': c.overrides, 'env': c.env}))" % WRAPPER)
    r = subprocess.run([sys.executable, "-c", code], env=_env(**TRC.OTHER_ROOTS), capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout.splitlines()[-1])
    assert d["run"] == C.compose(WRAPPER).run
    over = "\n".join(d["over"])
    for needle in ("agent.config.plan_score_file=/mnt/a/data/ck/raw256/pdm_score_256_officialEP_trainval_simscale",
                   CK + "warmup.packed=/mnt/a/data/ck/packed/trainval_simscale",
                   CK + "warmup.raw_labels=/mnt/a/data/ck/labels/trainval_simscale/raw256",
                   CK + "warmup.var_dir=/mnt/a/data/ck/ck2/labels/trainval_simscale/var_separate_sampler16_accstraight",
                   CK + "teacher_det_run=/mnt/b/tc/refiner/det_bevformer", "+train_data=trainval_simscale", "+tf32=true"):
        assert needle in over, needle
    assert "/home/external-user" not in over.replace(str(REPO), "") and "/workspace/" not in over
    assert d["env"]["SSR_TRAIN_SPLIT"] == "trainval_simscale"
    assert d["env"]["SSR_DET_CACHE_TRAIN"] == "/mnt/b/tc/bevformer/cache_train_100x100"


# ----------------------------------------------------------------------------------------------- the main recipe stays
def test_the_main_recipe_keeps_its_name_and_passes_the_reference():
    c = C.compose({})
    assert c.run == MAIN_NAME and c.default_recipe
    ok, lines = C.reference_diff(c.overrides, c.roots, c.run_dir)
    assert ok, lines


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("arm", [a for a, kw in TRC.ARMS.items() if "DATA" not in kw and "TF32" not in kw])
def test_the_default_data_and_tf32_add_nothing(arm, smoke):
    """DATA=navtrain TF32=off spelled out: the same name, hash, overrides, set / extra lines, environment, pins."""
    kw = TRC.ARMS[arm]
    assert _sig(C.compose(dict(kw, DATA="navtrain", TF32="off"), smoke=smoke)) == _sig(C.compose(kw, smoke=smoke))


def test_the_main_recipe_has_no_line_of_the_new_settings():
    c = C.compose({})
    keys = {C.LU._okey(o) for o in c.overrides}
    assert not keys & {"train_data", "tf32"} and "SSR_TRAIN_SPLIT" not in c.env
    d = c.roots["SSR_DATA_ROOT"]
    assert f"agent.config.plan_score_file={d}/ck/raw256/pdm_score_256_officialEP" in c.overrides
    assert c.set_lines[:3] == [f"warmup.packed={d}/ck/packed/navtrain_train", f"warmup.raw_labels={d}/ck/labels/navtrain_train/raw256",
                               f"warmup.var_dir={d}/ck/ck2/labels/navtrain_train/var_separate_sampler16_accstraight"]
    assert not any("trainval_simscale" in x for x in c.overrides + c.set_lines + list(c.env.values()))


def test_env_file_unsets_an_inherited_split(tmp_path):
    """A shell that still exports SSR_TRAIN_SPLIT must not hand it to a navtrain run's labeler."""
    for kw, sub in (({}, "main"), (WRAPPER, "tvsim")):
        C.write_composition(C.compose(dict(kw, RUN_ROOT=str(tmp_path))), str(tmp_path / sub))
    head, *rest = (tmp_path / "main/env.sh").read_text().splitlines()
    assert head.startswith("unset ") and "SSR_TRAIN_SPLIT" in head.split() and not any("SSR_TRAIN_SPLIT" in x for x in rest)
    assert "export SSR_TRAIN_SPLIT=trainval_simscale" in (tmp_path / "tvsim/env.sh").read_text()
    for sub, want in (("main", "<unset>"), ("tvsim", "trainval_simscale")):
        r = subprocess.run(["bash", "-c", f'. "{tmp_path / sub / "env.sh"}"; echo "${{SSR_TRAIN_SPLIT:-<unset>}}"'],
                           env=dict(os.environ, SSR_TRAIN_SPLIT="from_the_shell"), capture_output=True, text=True)
        assert r.stdout.strip() == want, r.stdout + r.stderr


# ----------------------------------------------------------------------------------------------- the lines of the recipe
def test_data_lines():
    c, m = C.compose(TVSIM), C.compose({"DET_TEACHER": "bevformer", "MAP_TEACHER": "mask2map"})
    d = c.roots["SSR_DATA_ROOT"]
    assert c.set_lines[:3] == [f"warmup.packed={d}/ck/packed/trainval_simscale",
                               f"warmup.raw_labels={d}/ck/labels/trainval_simscale/raw256",
                               f"warmup.var_dir={d}/ck/ck2/labels/trainval_simscale/var_separate_sampler16_accstraight"]
    k = _kv(c)
    assert k["train_data"] == "trainval_simscale" and "+train_data=trainval_simscale" in c.overrides
    assert k["agent.config.plan_score_file"] == f"{d}/ck/raw256/pdm_score_256_officialEP_trainval_simscale"
    assert k[CK + "warmup.packed"] == f"{d}/ck/packed/trainval_simscale" and "tf32" not in k
    assert c.env["SSR_TRAIN_SPLIT"] == "trainval_simscale"
    # the score file replaces the recipe's line in place; the only new key is train_data, the main keys keep their order
    keys, mkeys = [C.LU._okey(o) for o in c.overrides], [C.LU._okey(o) for o in m.overrides]
    assert keys.count("agent.config.plan_score_file") == 1 and [x for x in keys if x != "train_data"] == mkeys
    assert keys.index("agent.config.plan_score_file") == mkeys.index("agent.config.plan_score_file")


def test_tf32_line():
    c, m = C.compose({"TF32": "on"}), C.compose({})
    assert [o for o in c.overrides if o not in m.overrides and not o.startswith(("experiment_name", "wandb.name", "output_dir",
                                                                                "++" + CK + "io_dir"))] == ["+tf32=true"]
    assert "SSR_TRAIN_SPLIT" not in c.env and c.set_lines == m.set_lines


def test_the_launch_paths_are_the_ones_of_the_split_constants():
    """compose writes the score file and status / smoke_check read the table where the core constants put them."""
    from navsim.agents.para_ssr.ck import constants as K
    if "trainval_simscale" not in K.SPLITS:
        pytest.skip("constants.SPLITS has no trainval_simscale yet")
    sp = K.SPLITS["trainval_simscale"]
    assert sp["tokens"] == str(K.DATA_ROOT / L2.DATA_TABLE["trainval_simscale"])
    if "label_source" in sp:
        assert _kv(C.compose(TVSIM))["agent.config.plan_score_file"] == sp["label_source"]


def test_the_new_lines_reach_the_training_config():
    """+train_data / +tf32 compose against default_training (a key the yaml already had would make '+' fail) and reach
    the root of the config, where tools/release/check_build.py records them."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from tools.release import check_build as CB
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/training"), version_base="1.2"):
        cfg = compose(config_name="default_training", overrides=C.compose(WRAPPER).overrides)
        main = compose(config_name="default_training", overrides=C.compose({}).overrides)
    GlobalHydra.instance().clear()
    assert CB.run_keys(cfg) == {"train_data": "trainval_simscale", "tf32": True} and CB.run_keys(main) == {}
    assert cfg.agent.config.plan_score_file.endswith("pdm_score_256_officialEP_trainval_simscale")
    assert cfg.agent.config.ck_e2e2.warmup.packed.endswith("/ck/packed/trainval_simscale")
    assert list(cfg.agent.config.image_size) == [1148, 672] and cfg.agent.config.vit_lora_rank == 32


@pytest.mark.parametrize("kw, msg", [({"DATA": "navtest"}, "one of"), ({"TF32": "1"}, "one of"),
                                     ({"EXTRA": "+train_data=navtrain"}, "use DATA"), ({"EXTRA": "+tf32=false"}, "use TF32"),
                                     ({**TVSIM, "EXTRA": "agent.config.plan_score_file=/x"}, "use DATA"),
                                     ({**TVSIM, "EXTRA": "agent.config.ck_e2e2.warmup.packed=/x"},
                                      "locked")])
def test_refusals(kw, msg):
    with pytest.raises(C.Refused, match=msg):
        C.compose(kw)


# ----------------------------------------------------------------------------------------------- scripts/experiment.sh
@pytest.fixture()
def stubs(tmp_path):
    d = tmp_path / "stubs"
    d.mkdir()
    for name, text in (("train_e2e2.sh", TRC.STUB_E2E), ("verify_assets.py", TRC.STUB_ASSETS), ("eval.sh", TRC.STUB_EVAL),
                       ("smoke_check.py", TRC.STUB_SMOKE_CHECK)):
        (d / name).write_text(text)
    return dict(SSR_RELEASE_TEST_STUBS=str(d), REPO_FOR_STUB=str(REPO), STUB_LOG=str(tmp_path / "stub.log"), TMPDIR=str(tmp_path))


def test_scripts_parse():
    for s in (EXP, WRAP):
        r = subprocess.run(["bash", "-n", str(s)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_script_name_of_the_new_settings(tmp_path):
    r = run(EXP, "name", f"RUN_ROOT={tmp_path / 'root'}", *[f"{k}={v}" for k, v in WRAPPER.items()])
    assert r.returncode == 0 and r.stdout.strip() == C.compose(WRAPPER).run, r.stderr
    assert not (tmp_path / "root").exists()


def test_assets_passes_the_data_and_navtrain_calls_stay_as_they_were(tmp_path, stubs):
    r = run(EXP, "assets", f"RUN_ROOT={tmp_path}", **stubs)
    assert r.returncode == 0 and "stub verify_assets ['--profile', 'full', '--det-teacher', 'bevfusion', '--map-teacher', " \
                                 "'resmap']" in r.stdout, r.stdout + r.stderr
    r = run(EXP, "assets", f"RUN_ROOT={tmp_path}", "QUICK=1", "DATA=trainval_simscale", "DET_TEACHER=bevformer",
            "MAP_TEACHER=mask2map", **stubs)
    assert "['--profile', 'full', '--det-teacher', 'bevformer', '--map-teacher', 'mask2map', '--quick', '--data', " \
           "'trainval_simscale']" in r.stdout, r.stdout + r.stderr
    # assets RUN=<run>: profile, data and teachers of that run; a record from before DATA has none
    for name, settings, want in (("tv", C.canonicalize(WRAPPER), "'--det-teacher', 'bevformer', '--map-teacher', 'mask2map', "
                                                                 "'--data', 'trainval_simscale']"),
                                 ("old", {k: v for k, v in C.canonicalize({}).items() if k not in ("DATA", "TF32")},
                                  "'--det-teacher', 'bevfusion', '--map-teacher', 'resmap']")):
        (tmp_path / name / "launch").mkdir(parents=True)
        (tmp_path / name / "launch/settings.json").write_text(json.dumps({"profile": "full", "settings": settings}))
        r = run(EXP, "assets", f"RUN={name}", f"RUN_ROOT={tmp_path}", **stubs)
        assert r.returncode == 0 and want in r.stdout, r.stdout + r.stderr


def test_every_verify_assets_call_goes_through_the_data_aware_helper():
    calls = [ln.strip() for ln in EXP.read_text().splitlines() if '"$VERIFY_ASSETS"' in ln and "[ -f" not in ln]
    assert calls == ['py "$VERIFY_ASSETS" "$@" "${d[@]}"'], calls


def test_min_free_gb_follows_the_data(tmp_path, stubs):
    """df reports 250 GiB: navtrain (70) passes the disk guard, trainval_simscale (300) does not, unless MIN_FREE_GB."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "df").write_text("#!/usr/bin/env bash\necho Avail\necho ' 250G'\n")
    (bin_ / "df").chmod(0o755)
    root = tmp_path / "root"
    (root / "other/launch").mkdir(parents=True)
    env = dict(stubs, PATH=f"{bin_}:{os.environ['PATH']}")
    base = ["train", f"RUN_ROOT={root}", "ALLOW_DIRTY=1"]
    p = subprocess.Popen(["bash", "-c", "exec -a run_training sleep 60"])
    try:
        time.sleep(0.3)
        (root / "other/launch/train.pid").write_text(str(p.pid))      # the guard after the disk guard stops each run
        r = run(EXP, *base, **env)
        assert r.returncode == 3 and "already running" in r.stderr and "MIN_FREE_GB" not in r.stderr, r.stderr
        r = run(EXP, *base, *(f"{k}={v}" for k, v in TVSIM.items()), **env)
        assert r.returncode == 3 and "250 GiB free" in r.stderr and "< 300 GiB (MIN_FREE_GB)" in r.stderr, r.stderr
        assert "~200 GiB (estimate)" in r.stderr
        r = run(EXP, *base, *(f"{k}={v}" for k, v in TVSIM.items()), "MIN_FREE_GB=200", **env)
        assert r.returncode == 3 and "already running" in r.stderr, r.stderr
        r = run(EXP, *base, **dict(env, MIN_FREE_GB="300"))                # the environment, as before
        assert r.returncode == 3 and "< 300 GiB (MIN_FREE_GB); a 30-epoch run peaks at ~60 GiB" in r.stderr, r.stderr
    finally:
        p.kill()
        p.wait()
    assert not (root / "_check").exists() and not Path(stubs["STUB_LOG"]).exists()


# ----------------------------------------------------------------------------------------------- scripts/train_simscale.sh
STUB_EXP = r"""#!/usr/bin/env bash
# Stands in for scripts/experiment.sh: records each call; name is the real script's
echo "$*" >> "$STUB_LOG"
case "$1" in
  name) exec bash "$REAL_EXP" "$@";;
  check) exit "${STUB_CHECK_RC:-0}";;
  *) exit "${STUB_RC:-0}";;
esac
"""


@pytest.fixture()
def wstubs(tmp_path):
    d = tmp_path / "wstubs"
    d.mkdir()
    (d / "experiment.sh").write_text(STUB_EXP)
    return dict(SSR_RELEASE_TEST_STUBS=str(d), REAL_EXP=str(EXP), STUB_LOG=str(tmp_path / "wstub.log"))


def _calls(stubs):
    p = Path(stubs["STUB_LOG"])
    calls = p.read_text().splitlines() if p.is_file() else []
    p.unlink(missing_ok=True)
    return calls


def _ssr_run_root():
    r = subprocess.run(["bash", "-c", f'REPO={REPO}; source "$REPO/tools/release/load_paths.sh"; echo "$SSR_RUN_ROOT"'],
                       env=_env(), capture_output=True, text=True, timeout=120)
    return r.stdout.strip()


@pytest.mark.parametrize("args, gpus", [([], "0,1,2,3,4,5,6,7"), (["NGPU=8"], "0,1,2,3,4,5,6,7"), (["NGPU=4"], "0,1,2,3"),
                                        (["GPUS=0,1,2,3"], "0,1,2,3"), (["NGPU=2", "SEED=1", "TAG=t1", "EPOCHS=20"], "0,1")])
def test_wrapper_name_is_the_composed_name(args, gpus):
    r = run(WRAP, "name", *args)
    assert r.returncode == 0, r.stderr
    kw = dict(WRAPPER, GPUS=gpus, **dict(a.split("=", 1) for a in args if not a.startswith(("NGPU=", "GPUS="))))
    assert r.stdout.strip() == C.compose(kw).run


@pytest.mark.parametrize("args, code, msg", [
    (["DATA=navtrain"], 3, "DATA is fixed to trainval_simscale"), (["BACKBONE=r34"], 3, "BACKBONE is fixed to dino_vits14_1148"),
    (["DET_TEACHER=bevfusion"], 3, "DET_TEACHER is fixed to bevformer"), (["MAP_TEACHER=resmap"], 3, "fixed to mask2map"),
    (["TF32=off"], 3, "TF32 is fixed to on"), (["REFINER=off"], 3, "not a key of this script"),
    (["EXTRA=agent.lr=2e-4"], 3, "not a key of this script"), (["NGPU=6"], 3, "2, 4 or 8"), (["NGPU=1"], 3, "2, 4 or 8"),
    (["NGPU=4", "GPUS=0,1,2,3"], 3, "give one of them"), (["RUN=x"], 3, "only for status, stop and resume"),
    (["junk"], 3, "KEY=VALUE"), (["GPUS=0,1,2"], 3, "distinct GPU indices"),
])
def test_wrapper_refusals(args, code, msg, wstubs):
    r = run(WRAP, "train", *args, **wstubs)
    assert r.returncode == code and msg in r.stderr, r.stdout + r.stderr
    assert all(c.startswith("name ") for c in _calls(wstubs))       # nothing but the name was asked for


def test_wrapper_usage_and_the_same_values_of_the_fixed_settings(wstubs):
    r = run(WRAP, "eval")
    assert r.returncode == 2 and "train_simscale.sh [train|check|smoke|status|stop|resume|name]" in r.stdout
    r = run(WRAP, "name", "DATA=trainval_simscale", "TF32=on", "BACKBONE=dino_vits14_1148")
    assert r.returncode == 0 and r.stdout.strip() == C.compose(WRAPPER).run, r.stderr


def test_wrapper_train_runs_check_first_and_stops_when_it_fails(tmp_path, wstubs):
    root = tmp_path / "runs"
    name = C.compose(dict(WRAPPER, GPUS="0,1,2,3", TAG="t1")).run
    want = f"{FIXED_ARGS} GPUS=0,1,2,3 RUN_ROOT={root} TAG=t1"
    r = run(WRAP, "NGPU=4", f"RUN_ROOT={root}", "TAG=t1", **wstubs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _calls(wstubs) == [f"name {want}", f"check {want}", f"train {want}"]
    assert f"run {name} on 4 GPUs (0,1,2,3)" in r.stdout and f"{root}/{name}/launch/train.log" in r.stdout
    assert f"{root}/{name}/labeler/labeler.log" in r.stdout and f"status RUN={name} RUN_ROOT={root}" in r.stdout
    # a failing check: no start
    r = run(WRAP, "train", "NGPU=4", f"RUN_ROOT={root}", "TAG=t1", **wstubs, STUB_CHECK_RC="3")
    assert r.returncode == 3 and "check failed: nothing started" in r.stderr
    assert _calls(wstubs) == [f"name {want}", f"check {want}"]
    # a refused start: its code, no 'training' line
    r = run(WRAP, "train", "NGPU=4", f"RUN_ROOT={root}", "TAG=t1", **wstubs, STUB_RC="3")
    assert r.returncode == 3 and "training " not in r.stdout and len(_calls(wstubs)) == 3
    # DRY=1 goes through check and train DRY=1, starts nothing
    r = run(WRAP, "train", "DRY=1", "NGPU=4", f"RUN_ROOT={root}", "TAG=t1", **wstubs)
    assert r.returncode == 0 and f"(DRY) nothing started: {name}" in r.stdout
    assert [c.split()[0] for c in _calls(wstubs)] == ["name", "check", "train"]
    # an existing run folder: resume, before any check
    (root / name).mkdir(parents=True)
    r = run(WRAP, "NGPU=4", f"RUN_ROOT={root}", "TAG=t1", **wstubs)
    assert r.returncode == 3 and "train_simscale.sh resume" in r.stderr and [c.split()[0] for c in _calls(wstubs)] == ["name"]


def test_wrapper_default_root_is_ssr_run_root(wstubs):
    r = run(WRAP, **wstubs)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = _calls(wstubs)
    assert calls[-1] == f"train {FIXED_ARGS} GPUS=0,1,2,3,4,5,6,7 RUN_ROOT="      # experiment.sh: empty = $SSR_RUN_ROOT
    root = _ssr_run_root()
    assert root and f"folder  {root}/{C.compose(WRAPPER).run}" in r.stdout


def test_wrapper_smoke_memory_limit(tmp_path, wstubs):
    root = tmp_path / "runs"
    r = run(WRAP, "smoke", "GPUS=4,5", f"RUN_ROOT={root}", **wstubs)
    assert r.returncode == 0, r.stdout + r.stderr
    name = C.compose(dict(WRAPPER, GPUS="4,5")).run
    assert _calls(wstubs)[-1] == f"smoke {FIXED_ARGS} GPUS=4,5 RUN_ROOT={root} MEM_LIMIT_GB=30"
    assert f"{root}/_smoke/smoke_{name}/launch/train.log" in r.stdout
    r = run(WRAP, "smoke", "GPUS=4,5", f"RUN_ROOT={root}", "MEM_LIMIT_GB=40", **wstubs)
    assert _calls(wstubs)[-1] == f"smoke {FIXED_ARGS} GPUS=4,5 RUN_ROOT={root} MEM_LIMIT_GB=40"
    r = run(WRAP, "smoke", "GPUS=4,5", f"RUN_ROOT={root}", **wstubs, MEM_LIMIT_GB="35")       # from the environment
    assert _calls(wstubs)[-1].endswith("MEM_LIMIT_GB=35")


def test_wrapper_status_stop_resume(tmp_path, wstubs):
    root = f"RUN_ROOT={tmp_path}"
    for cmd in ("status", "stop", "check", "name"):
        assert run(WRAP, cmd, "NGPU=4", root, **wstubs).returncode == 0
        assert _calls(wstubs)[-1] == f"{cmd} {FIXED_ARGS} GPUS=0,1,2,3 {root}"
    assert run(WRAP, "status", "RUN=r1", root, **wstubs).returncode == 0
    assert _calls(wstubs) == [f"status {FIXED_ARGS} GPUS=0,1,2,3,4,5,6,7 RUN=r1 {root}"]
    # resume RUN=: the run's own settings, only how it is launched (experiment.sh refuses any other key)
    assert run(WRAP, "resume", "RUN=r1", root, **wstubs).returncode == 0
    assert _calls(wstubs) == [f"resume RUN=r1 {root}"]
    assert run(WRAP, "resume", "RUN=r1", "NGPU=4", "ALLOW_COMMIT_CHANGE=1", **wstubs).returncode == 0
    assert _calls(wstubs) == ["resume RUN=r1 ALLOW_COMMIT_CHANGE=1 RUN_ROOT= GPUS=0,1,2,3"]
    assert run(WRAP, "resume", root, **wstubs).returncode == 0
    assert _calls(wstubs) == [f"resume {FIXED_ARGS} GPUS=0,1,2,3,4,5,6,7 {root}"]


def test_wrapper_resume_by_name_reaches_experiment_sh_without_the_settings(tmp_path):
    """The real experiment.sh: resume RUN=... takes no setting, so the wrapper must not pass its fixed ones."""
    r = run(WRAP, "resume", "RUN=nope", f"RUN_ROOT={tmp_path}")
    assert r.returncode == 3 and "settings.json not found" in r.stderr, r.stdout + r.stderr


def _dev_assets_ready():
    """The real check of the recipe reads the warm-up arrays and the teacher runs and manifests of this server."""
    r = subprocess.run(["bash", "-c", f'REPO={REPO}; source "$REPO/tools/release/load_paths.sh"; '
                        'echo "$SSR_DATA_ROOT"; echo "$SSR_TEACHER_CACHE"'], env=_env(), capture_output=True, text=True)
    lines = r.stdout.split()
    if len(lines) != 2:
        return False
    data, tc = map(Path, lines)
    need = [data / "splits/trainval_simscale.parquet", data / "ck/packed/trainval_simscale/tokens.parquet",
            data / "ck/labels/trainval_simscale/raw256/labels.npy", tc / "refiner/det_bevformer/done.json",
            tc / "refiner/map_mask2map/done.json", tc / "bevformer/cache_train_100x100/manifest.json",
            tc / "mask2map/front64/meta.json"]
    return all(p.exists() for p in need) and any((data / "ck/ck2/labels/trainval_simscale").glob("*"))


@pytest.mark.skipif(not _dev_assets_ready(), reason="needs the trainval_simscale assets and the camera-teacher caches")
def test_wrapper_dry_run_with_the_real_launcher(tmp_path):
    r = run(WRAP, "train", "DRY=1", "GPUS=4,5", f"RUN_ROOT={tmp_path / 'runs'}", f"CHECK_ROOT={tmp_path / 'c'}",
            "ALLOW_DIRTY=1", timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "launcher overrides == composition" in r.stdout and "refusals: none" in r.stdout and "would run" in r.stdout
    assert "train_data trainval_simscale, tf32 True" in r.stdout and "nothing started" in r.stdout
    assert not (tmp_path / "runs").exists()


# ----------------------------------------------------------------------------------------------- epoch length
def _table(path, real=3, sim=2):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    toks = [f"{i:016x}" for i in range(real)] + [f"{i:016x}-004" for i in range(sim)]
    pd.DataFrame({"token": toks, "source": ["navtrain_train"] * real + ["simscale"] * sim}).to_parquet(path)
    return toks


def _run_dir(tmp_path, data=None, data_root=None, name="r"):
    R = tmp_path / name
    (R / "launch/gen").mkdir(parents=True)
    (R / "refiner").mkdir()
    s = C.canonicalize(dict(WRAPPER, DATA=data) if data else {})
    (R / "launch/settings.json").write_text(json.dumps({"run": name, "settings": s}))
    if data_root:
        (R / "launch/gen/composition.json").write_text(json.dumps({"roots": {"SSR_DATA_ROOT": str(data_root)}}))
    return R


def test_train_tokens_follow_the_data(tmp_path):
    assert L2.train_tokens(str(_run_dir(tmp_path, name="nav"))) == (85109, "navtrain_train")
    assert L2.train_tokens(str(tmp_path / "no_record")) == (85109, "navtrain_train")
    data = tmp_path / "data"
    R = _run_dir(tmp_path, "trainval_simscale", data, name="tv")
    _table(data / "splits/trainval_simscale.parquet", real=7, sim=4)
    n, src = L2.train_tokens(str(R))
    assert n == 11 and src == f"rows of {data}/splits/trainval_simscale.parquet"
    (data / "splits/trainval_simscale.parquet").unlink()
    assert L2.train_tokens(str(R))[0] is None
    (R / "train").mkdir()
    (R / "train/train_tokens.json").write_text(json.dumps({"count": 9, "train_token_sha256": "x"}))
    n, src = L2.train_tokens(str(R))
    assert n == 9 and src.startswith("count of ")


def _steps(R, ws=2):
    recs = [{"epoch": 1, "epoch_frac": 1.5, "batch": 10, "sec_step": 2.0, "ck2/phase": 0.0} for _ in range(5)]
    (R / "refiner/steps_rank0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (R / "refiner/epochs.jsonl").write_text(json.dumps({"event": "epoch_start", "epoch": 1, "world_size": ws}) + "\n")


def test_status_eta_counts_the_table_rows(tmp_path):
    data = tmp_path / "data"
    _table(data / "splits/trainval_simscale.parquet", real=600, sim=400)
    runs = {"tv": _run_dir(tmp_path, "trainval_simscale", data, name="tv"), "nav": _run_dir(tmp_path, name="nav")}
    p = subprocess.Popen(["bash", "-c", "exec -a run_training sleep 60"])
    try:
        time.sleep(0.3)
        for R in runs.values():
            _steps(R)
            (R / "launch/train.pid").write_text(str(p.pid))
        st = L2.run_status2(str(runs["tv"]))
        mb = L2.LU.mb_per_epoch(2, 1000)
        assert mb == 125 and st["epoch_tokens"] == {"n": 1000, "source": f"rows of {data}/splits/trainval_simscale.parquet"}
        assert st["eta"]["basis"] == (f"median sec_step of the last 300 mb x {mb} mb/epoch x 28.50 epochs left [추정]; "
                                      f"1000 tokens per epoch: rows of {data}/splits/trainval_simscale.parquet")
        assert st["eta"]["hours_left"] == round(28.5 * mb * 2.0 / 3600, 2)
        st = L2.run_status2(str(runs["nav"]))          # the main recipe's status as before
        assert "epoch_tokens" not in st and st["eta"]["basis"] == (
            f"median sec_step of the last 300 mb x {L2.LU.mb_per_epoch(2)} mb/epoch x 28.50 epochs left [추정]")
        (data / "splits/trainval_simscale.parquet").unlink()
        st = L2.run_status2(str(runs["tv"]))           # no count at all: no ETA, a warning instead of 85,109
        assert "eta" not in st and any(w.startswith("no ETA: no token count") for w in st["warnings"])
    finally:
        p.kill()
        p.wait()


def test_smoke_estimate_counts_the_table_rows():
    timing = {ph: {"wall_per_mb_mean": 1.0} for ph in ("warmup", "warmup_record", "onpolicy")}
    assert L2.estimate_hours2(timing, 2)["mb_per_epoch"] == L2.LU.mb_per_epoch(2) == 10639
    assert L2.estimate_hours2(timing, 8, n_tokens=340660)["mb_per_epoch"] == L2.LU.mb_per_epoch(8, 340660) == 10646


# ----------------------------------------------------------------------------------------------- smoke check
def _smoke_run(tmp_path, data="trainval_simscale", real=3, sim=2, trained=None):
    droot = tmp_path / "data"
    R = _run_dir(tmp_path, data, droot if data else None, name="smoke_x")
    toks = _table(droot / "splits/trainval_simscale.parquet", real, sim) if data else []
    trained = toks if trained is None else trained
    (R / "train").mkdir()
    (R / "train/train_tokens.json").write_text(json.dumps({
        "train_token_sha256": hashlib.sha256("\n".join(sorted(trained)).encode()).hexdigest(), "count": len(trained)}))
    return R, toks


NEW = ("train_tokens_eq_table_rows", "simscale_tokens_trained")


def test_smoke_check_the_run_trained_on_the_whole_table(tmp_path):
    R, toks = _smoke_run(tmp_path / "a")
    res = SC.check_run(str(R))
    assert all(res["checks"][k] for k in NEW + ("train_tokens_recorded",)), res
    assert res["info"]["table_rows"] == 5 and res["info"]["table_simscale_rows"] == 2
    # the 4 + 10 frame filter kept the real rows only
    R, toks = _smoke_run(tmp_path / "b", trained=[f"{i:016x}" for i in range(3)])
    c = SC.check_run(str(R))["checks"]
    assert c["train_tokens_recorded"] and not c["train_tokens_eq_table_rows"] and not c["simscale_tokens_trained"]
    # the right count, but a SimScale token replaced by a real one
    R, toks = _smoke_run(tmp_path / "c", trained=[f"{i:016x}" for i in range(4)] + ["0000000000000000-004"])
    c = SC.check_run(str(R))["checks"]
    assert c["train_tokens_eq_table_rows"] and not c["simscale_tokens_trained"]
    # a table without SimScale rows
    R, toks = _smoke_run(tmp_path / "d", real=5, sim=0)
    c = SC.check_run(str(R))["checks"]
    assert c["train_tokens_eq_table_rows"] and not c["simscale_tokens_trained"]
    # no table
    R, _ = _smoke_run(tmp_path / "e")
    (tmp_path / "e/data/splits/trainval_simscale.parquet").unlink()
    res = SC.check_run(str(R))
    assert not res["checks"]["train_tokens_eq_table_rows"] and "token_table_error" in res["info"]


def test_smoke_check_of_navtrain_has_no_new_check(tmp_path):
    R, _ = _smoke_run(tmp_path, data=None, trained=["a" * 16])
    res = SC.check_run(str(R))
    assert not set(NEW) & set(res["checks"]) and res["checks"]["train_tokens_recorded"] and "token_table" not in res["info"]
