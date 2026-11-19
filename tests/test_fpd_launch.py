"""REFINER_ARCH=fpd launch plumbing (scripts/experiment.sh, tools/release/{compose,smoke_check,check_build}.py,
tools/ck/e2e2/launch_util2.py, the eval hook of tools/ck/e2e2/eval_epdms.py), and that the default ck2 system is what it
was: every documented arm keeps its name, its settings record and its ck_e2e2 dict.  CPU only.  Run:
  set -a && . tools/release/paths.env && set +a && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /venv/ssr/bin/python -m pytest \
    -q -p no:cacheprovider tests/test_fpd_launch.py
Nothing here launches anything: the teachers / resume commands run against stub launchers in a copy of the script.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from navsim.common import ssr_paths as SP  # noqa: E402
from tools.ck.e2e import launch_util as LU  # noqa: E402
from tools.ck.e2e2 import launch_util2 as L2  # noqa: E402
from tools.release import compose as C  # noqa: E402
from tools.release import smoke_check as SC  # noqa: E402

SH = REPO / "scripts/experiment.sh"
REPO_REGISTRY = str(REPO / "tools/release/registry.yaml")
ENV_DROP = set(SP.NAMES) | set(C.CONTROL_ENV) | {"SSR_RELEASE_REGISTRY", "SSR_RELEASE_TEST_STUBS", "ALLOW_DIRTY", "CHECK_ROOT",
                                                 "ARCH", "REFINER_ARCH", "GPUS_DET", "GPUS_MAP", "WHICH"}
CK = "agent.config.ck_e2e2."
FPD = {"REFINER_ARCH": "fpd"}


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ENV_DROP}
    env.update(PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2")
    env.update(extra)
    return env


def _kv(c):
    return {LU._okey(o): o.split("=", 1)[1] for o in c.overrides if "=" in o}


def comp(kw, smoke=False, registry=REPO_REGISTRY):
    return C.compose(kw, registry=registry, smoke=smoke)


# ----------------------------------------------------------------------------------------------- the default system
# run names of v2_final (b902937) before REFINER_ARCH, with tools/release/registry.yaml (hash6 has no server root in it)
RECORDED = [
    ({}, "main_899596"), ({"REFINER": "off"}, "ref-off_49e633"), ({"REFINER_KD": "off"}, "rkd-off_fb5474"),
    ({"REFINER_KD": "lat"}, "rkd-lat_1f512b"), ({"REFINER_KD": "score"}, "rkd-score_f966c2"), ({"BEV_KD": "off"}, "bkd-off_1d842e"),
    ({"BEV_KD": "off", "REFINER_KD": "off"}, "rkd-off_bkd-off_4f2059"), ({"BEV_KD": "off", "REFINER": "off"}, "ref-off_bkd-off_86de14"),
    ({"DET_KD": "off"}, "detkd-off_098ddd"), ({"MAP_KD": "off"}, "mapkd-off_fee5d1"), ({"DET_KD": "off_bev"}, "detkd-offbev_cd0018"),
    ({"MAP_KD": "off_bev"}, "mapkd-offbev_b677a1"), ({"DET_KD": "off_ref"}, "detkd-offref_869030"),
    ({"MAP_KD": "off_ref"}, "mapkd-offref_3dba0d"), ({"DET_KD": "off", "REFINER": "off"}, "ref-off_detkd-offbev_6d9673"),
    ({"MAP_KD": "off", "REFINER": "off"}, "ref-off_mapkd-offbev_0a3ce7"),
    ({"DET_KD": "off_ref", "REFINER_KD": "score"}, "rkd-score_detkd-offref_404cad"),
    ({"MAP_KD": "off_ref", "REFINER_KD": "lat"}, "rkd-lat_mapkd-offref_094987"),
    ({"DET_KD": "off", "REFINER_KD": "off"}, "rkd-off_detkd-offbev_0cffce"), ({"BEV_KD": "off", "DET_KD": "off"}, "bkd-off_detkd-offref_aa2be5"),
    ({"DET_KD": "off_bev", "MAP_KD": "off_ref"}, "detkd-offbev_mapkd-offref_f2722d"), ({"BACKBONE": "r18"}, "r18_cc783f"),
    ({"BACKBONE": "r50"}, "r50_3658cb"), ({"BACKBONE": "r101"}, "r101_f23019"), ({"BACKBONE": "r152"}, "r152_a2ecb8"),
    ({"BACKBONE": "r50d"}, "r50d_867752"), ({"BACKBONE": "convnext_tiny"}, "convnext_tiny_63e520"),
    ({"BACKBONE": "regnety_032"}, "regnety_032_e6785b"), ({"BACKBONE": "efficientnet_b0"}, "efficientnet_b0_47b56a"),
    ({"BACKBONE": "dino_vits14_1148"}, "dino_vits14_1148_d6dceb"), ({"BACKBONE": "dino_vits14"}, "dino_vits14_446f94"),
    ({"UNDISTORT": "off"}, "dist_f139a1"), ({"PILLAR_Z": "-2,2"}, "z-2_2_059388"), ({"PILLAR_Z": "-1.5,2.5"}, "z-1p5_2p5_345b84"),
    ({"FRAMES": "3"}, "f1_df52c8"), ({"FRAMES": "0,1,2,3"}, "f4_bed639"), ({"HEADING": "free"}, "headfree_1a065c"),
    ({"BEV_KD": "off", "BEV_RANGE": "32", "REFINER": "off"}, "ref-off_bkd-off_r32_f21f97"), ({"BN": "frozen"}, "bn-frozen_ac002d"),
    ({"SEED": "1"}, "s1_005ae2"), ({"EPOCHS": "10"}, "e10_2f8302"), ({"GPUS": "0,1,2,3"}, "g4_7d652a"), ({"GPUS": "0,1"}, "g2_64835f"),
    ({"SAVE_EVERY": "5"}, "main_899596"), ({"TAG": "try1"}, "try1_899596"), ({"WANDB": "online"}, "main_899596"),
    ({"BEV_KD": "off", "EXTRA": "agent.config.plan_anchor=false", "REFINER": "off"}, "ref-off_bkd-off_x_065154"),
    ({"BACKBONE": "r50", "MAP_KD": "off_ref", "REFINER_KD": "score", "SEED": "1", "TAG": "try1"},
     "r50_rkd-score_mapkd-offref_s1_try1_4ff67a"),
    ({"EXTRA": "agent.config.ck_e2e2.lambda_sur=0.5"}, "x_5dbb2a"), ({"EXTRA": "agent.config.ck_e2e2.n_var_step=16"}, "x_183a76"),
]


@pytest.mark.parametrize("kw, want", RECORDED)
def test_default_names_are_the_recorded_ones(kw, want):
    a, b = comp(kw), comp(dict(kw, REFINER_ARCH="ck2"))
    assert a.run == b.run == want
    assert comp(kw, smoke=True).run == f"smoke_{want}"
    # nothing of the new setting in what a ck2 run emits or records
    for c in (a, b):
        assert "REFINER_ARCH" not in c.settings and "REFINER_ARCH" not in c.record()["settings"]
        assert not [o for o in c.overrides if LU._okey(o).startswith((CK + "arch", CK + "fpd"))]
        assert not [x for x in c.set_lines if x.split("=", 1)[0].split(".")[0] in ("arch", "fpd")]
    assert a.overrides == b.overrides and a.set_lines == b.set_lines and a.record() == b.record() and a.env == b.env


@pytest.mark.parametrize("kw, want", RECORDED[:8])
def test_old_records_still_compose_to_their_run(kw, want):
    """a settings.json written before the setting has no REFINER_ARCH: it is a ck2 run (absent = ck2)."""
    s = comp(kw).settings
    assert "REFINER_ARCH" not in s and C.canonicalize(s, C.load_registry(REPO_REGISTRY)) == s
    assert comp(s).run == want and C.arch_of(s) == "ck2"


def test_the_default_recipe_is_the_default_recipe():
    assert comp({}).default_recipe and comp({"REFINER_ARCH": "ck2"}).default_recipe and not comp(FPD).default_recipe
    assert C.is_default_recipe(comp({}).settings) and not C.is_default_recipe(comp(FPD).settings)


def _old_effective_ck(run_dir, ngpu, sets, extra):
    """launch_util2.effective_ck of v2_final, before the optional keys."""
    ck = L2.scale_mb_counters(L2.load_ck2_yaml(), ngpu)
    ck["io_dir"] = f"{run_dir}/{L2.IO_SUB}"
    return LU.apply_sets(ck, list(sets) + L2.extra_ck_sets(extra))


@pytest.mark.parametrize("kw", [kw for kw, _ in RECORDED[:21]] + [{"GPUS": "0,1,2,3"}, {"GPUS": "0,1"}])
def test_effective_ck_of_every_ck2_arm_is_the_old_dict(kw):
    for smoke in (False, True):
        c = comp(kw, smoke=smoke)
        ng = len(c.settings["GPUS"].split(","))
        new = L2.effective_ck(c.run_dir, ng, c.set_lines, c.extra_lines)
        old = _old_effective_ck(c.run_dir, ng, c.set_lines, c.extra_lines)
        assert json.dumps(new) == json.dumps(old)          # same keys, same order, same values
        assert "arch" not in new and "fpd" not in new and L2.arm_of(new) in ("main", "nokd", "noref")


def test_compose_of_fpd_imports_no_torch(tmp_path):
    code = ("import sys; from tools.release import compose as C; c = C.compose({'REFINER_ARCH': 'fpd', 'RUN_ROOT': %r}); "
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules; print(c.run)" % str(tmp_path / "nope"))
    r = subprocess.run([sys.executable, "-c", code], env=_env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().startswith("fpd_") and not (tmp_path / "nope").exists()


# ----------------------------------------------------------------------------------------------- the settings surface
def test_script_settings_are_the_compose_defaults():
    t = SH.read_text()
    arr = re.search(r"^SETTINGS=\(([^)]*)\)", t, flags=re.M).group(1).split()
    assert set(arr) == set(C.DEFAULTS) and len(arr) == len(C.DEFAULTS) and "REFINER_ARCH" in arr
    block, seen = t.split("end of settings")[0], {}
    for line in block.splitlines():
        m = re.match(r"^([A-Z_]+)=(\S*)\s*(#.*)?$", line)
        if m:
            seen[m.group(1)] = m.group(2)
            assert m.group(3), m.group(1)
    seen.pop("RUN_ROOT")
    assert seen == {k: v for k, v in C.DEFAULTS.items() if k != "RUN_ROOT"} and seen["REFINER_ARCH"] == "ck2"
    opts = re.search(r"^OPTIONS=\(([^)]*)\)", t, flags=re.M | re.S).group(1).split()
    assert "ARCH" in opts and "ARCH" not in C.DEFAULTS
    assert C.CHOICES["REFINER_ARCH"] == L2.ARCHS == ("ck2", "fpd") and C.RECORDED_ONLY_OFF_DEFAULT == ("REFINER_ARCH",)


def test_the_script_names_the_fpd_arm_like_compose(tmp_path):
    root = tmp_path / "root"
    for args, kw in ((["REFINER_ARCH=fpd"], FPD), (["REFINER_ARCH=fpd", "REFINER_KD=off"], dict(FPD, REFINER_KD="off")),
                     ([], {}), (["REFINER_ARCH=ck2"], {})):
        r = subprocess.run(["bash", str(SH), "name", f"RUN_ROOT={root}", *args], env=_env(), capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == comp(dict(kw, RUN_ROOT=str(root))).run
    for args, msg in ((["REFINER_ARCH=fpd", "REFINER=off"], "refiner is off"), (["REFINER_ARCH=fpd", "REFINER_KD=lat"], "no lateral head"),
                      (["REFINER_ARCH=x"], "one of ck2 | fpd"), (["ARCH=fpd"], "ARCH is an option of 'teachers' only")):
        r = subprocess.run(["bash", str(SH), "name", *args], env=_env(), capture_output=True, text=True, timeout=120)
        assert r.returncode == 3 and msg in r.stderr, r.stderr
    assert not root.exists()


# ----------------------------------------------------------------------------------------------- the fpd arm
FPD_NAMES = [({}, "fpd_fc49bc"), ({"REFINER_KD": "off"}, "fpd_rkd-off_5d372d"), ({"BEV_KD": "off"}, "fpd_bkd-off_0e51b8"),
             ({"REFINER_KD": "off", "BEV_KD": "off"}, "fpd_rkd-off_bkd-off_0ca983"), ({"GPUS": "0,1,2,3"}, "fpd_g4_ab9bd4")]


@pytest.mark.parametrize("kw", [{"DET_KD": "off_ref"}, {"MAP_KD": "off_ref"}, {"DET_KD": "off"}, {"MAP_KD": "off"}])
def test_fpd_refuses_a_single_refiner_kd_source(kw):
    """the DiT target bank needs both FPD teachers (DAC from the map teacher, NC / TTC from the det teacher): a single
    refiner-KD source is refused for fpd (REFINER_KD=off stays allowed)."""
    with pytest.raises(C.Refused, match="needs both FPD teachers"):
        comp(dict(FPD, **kw))


@pytest.mark.parametrize("kw, want", FPD_NAMES)
def test_fpd_names(kw, want):
    """'fpd' after the refiner part; the hash is fixed until the fpd recipe changes on purpose."""
    c = comp(dict(FPD, **kw))
    assert c.run == want and c.settings["REFINER_ARCH"] == "fpd" and c.record()["settings"]["REFINER_ARCH"] == "fpd"
    assert comp(dict(FPD, **kw), smoke=True).run == f"smoke_{want}"
    assert not set(re.split(r"[_.-]", c.run)) & {"v2", "ck", "ck2", "v7", "v15"}


def test_fpd_score_is_fpd_on_and_the_settings_are_canonical():
    a, b = comp(FPD), comp(dict(FPD, REFINER_KD="score"))
    assert a.run == b.run and a.overrides == b.overrides and a.settings == b.settings and a.settings["REFINER_KD"] == "on"
    reg = C.load_registry(REPO_REGISTRY)
    assert C.canonicalize(a.settings, reg) == a.settings and comp(a.settings).run == a.run
    assert a.hash6 != comp({}).hash6 and a.profile == "full" and comp(dict(FPD, REFINER_KD="off", BEV_KD="off")).profile == \
        "refiner_noteacher"


@pytest.mark.parametrize("kw, msg", [
    ({"REFINER": "off"}, "means nothing: the refiner is off"), ({"REFINER": "off", "BEV_KD": "off"}, "means nothing"),
    ({"REFINER_KD": "lat"}, "no lateral head"), ({"REFINER_KD": "lat", "MAP_KD": "off_ref"}, "no lateral head"),
    ({"REFINER_ARCH": "dit"}, "one of ck2 | fpd"),
    ({"EXTRA": "agent.config.ck_e2e2.fpd.n_draws=3"}, "use REFINER_ARCH"), ({"EXTRA": "++agent.config.ck_e2e2.fpd={}"}, "REFINER_ARCH"),
    ({"EXTRA": "agent.config.ck_e2e2.fpd.teacher_det_run=/x"}, "locked"),
])
def test_fpd_refusals(kw, msg):
    with pytest.raises(C.Refused, match=msg):
        comp(dict(FPD, **kw))


@pytest.mark.parametrize("line", ["agent.config.ck_e2e2.arch=fpd", "++agent.config.ck_e2e2.arch=ck2",
                                  "agent.config.ck_e2e2.fpd.mt_k=2", "~agent.config.ck_e2e2.fpd"])
def test_extra_cannot_set_the_arch(line):
    with pytest.raises(C.Refused, match="REFINER_ARCH"):
        comp({"EXTRA": line})


def test_fpd_needs_the_default_teachers_only_when_it_reads_them(tmp_path):
    import yaml
    reg = yaml.safe_load(Path(REPO_REGISTRY).read_text())
    reg["teachers"]["det"]["bevfusion_alias"] = dict(reg["teachers"]["det"]["bevfusion"], cache_name="bevfusion_alias")
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(reg))
    with pytest.raises(C.Refused, match="FPD teacher refiners"):
        comp(dict(FPD, DET_TEACHER="bevfusion_alias"), registry=str(p))
    # no refiner KD: no FPD teacher is read, the BEV KD may use another teacher
    c = comp(dict(FPD, REFINER_KD="off", DET_TEACHER="bevfusion_alias"), registry=str(p))
    assert c.run.startswith("bevfusion_alias-resmap_fpd_rkd-off_")


def test_fpd_set_lines_and_overrides():
    c = comp(FPD)
    roots = c.roots
    tc = roots["SSR_TEACHER_CACHE"].rstrip("/")
    fpd_block = ["arch=fpd", "n_var_step=80", "warmup.n_var=0", "onpolicy_label.n_var=80", "lambda_sur=0.0",
                 "lat_kd.balance=fixed", "lat_kd.fixed=0.0", f"fpd.teacher_det_run={tc}/refiner_fpd/det",
                 f"fpd.teacher_map_run={tc}/refiner_fpd/map"]
    base = comp({})
    # the ck2 lines first, the CK2 teacher lines kept, the fpd block after them
    assert c.set_lines == base.set_lines + fpd_block
    k = _kv(c)
    assert k[CK + "arch"] == "fpd" and k[CK + "n_var_step"] == "80" and k[CK + "warmup.n_var"] == "0"
    assert k[CK + "onpolicy_label.n_var"] == "80" and k[CK + "lambda_sur"] == "0.0"
    assert k[CK + "lat_kd.balance"] == "fixed" and k[CK + "lat_kd.fixed"] == "0.0" and k[CK + "use_teachers"] == "true"
    assert k[CK + "teacher_det_run"] == f"{tc}/refiner/det" and k[CK + "teacher_map_run"] == f"{tc}/refiner/map"
    assert k[CK + "fpd.teacher_det_run"] == f"{tc}/refiner_fpd/det" and k[CK + "fpd.teacher_map_run"] == f"{tc}/refiner_fpd/map"
    for leaf, v in L2.FPD_CK_DEFAULTS.items():          # every fpd leaf is explicit: a changed default changes the hash
        assert CK + f"fpd.{leaf}" in k, leaf
        if not leaf.startswith("teacher_") or leaf == "teacher_which":
            assert k[CK + f"fpd.{leaf}"] == LU.hydra_value(v), leaf
    assert k["wandb.tags"] == "[refiner,fpd,gpu8]" and c.env == dict(base.env, RUN=c.run, SSR_EXPERIMENT_SHA=c.hash6)
    # the ck2 lines of the recipe stay where they were; the fpd lines come after the last ck2 leaf
    keys = [LU._okey(o) for o in c.overrides]
    assert keys.index(CK + "arch") == keys.index(CK + "bev_kd.cap_map") + 1
    assert [x for x in keys if not x.startswith((CK + "arch", CK + "fpd."))] == [LU._okey(o) for o in base.overrides]
    # the hash sees the teacher roots as placeholders
    hl = C.hash_lines(c.overrides, roots)
    assert f"++{CK}fpd.teacher_det_run=${{SSR_TEACHER_CACHE}}/refiner_fpd/det" in hl


def test_fpd_without_teachers_names_none():
    c = comp(dict(FPD, REFINER_KD="off"))
    k = _kv(c)
    assert k[CK + "use_teachers"] == "false" and k[CK + "teacher_det_run"] == "null"
    assert k[CK + "fpd.teacher_det_run"] == "''" and k[CK + "fpd.teacher_map_run"] == "''" and k[CK + "arch"] == "fpd"
    assert c.set_lines.count("lat_kd.fixed=0.0") == 1 and not any(x.startswith("fpd.") for x in c.set_lines)
    assert k["wandb.tags"] == "[refiner,fpd_nokd,gpu8]"


def test_fpd_smoke_lines():
    s, m = comp(FPD, smoke=True), comp({}, smoke=True)
    assert "lat_kd.start_mb=30" in m.set_lines and "lat_kd.start_mb=30" not in s.set_lines   # no lateral KD to start
    assert "kd_calib.min_n=50" in s.set_lines and s.set_lines[-5:] == m.set_lines[-6:-2] + ["kd_calib.min_n=50"]
    k = _kv(s)
    assert k[CK + "onpolicy_from_epoch"] == "1" and k[CK + "lat_kd.start_mb"] == "250" and k[CK + "arch"] == "fpd"
    o = comp(dict(FPD, REFINER_KD="off"), smoke=True)
    assert "kd_calib.min_n=50" not in o.set_lines and "lat_kd.start_mb=30" not in o.set_lines


OTHER_ROOTS = {"SSR_DATA_ROOT": "/mnt/a/data", "SSR_TEACHER_CACHE": "/mnt/b/tc", "SSR_PLAN_DATA": "/mnt/c/plan",
               "SSR_RUN_ROOT": "/mnt/d/runs", "OPENSCENE_DATA_ROOT": "/mnt/e/ds"}


def test_fpd_has_one_name_on_any_server():
    code = ("from tools.release import compose as C; import json; c = C.compose({'REFINER_ARCH': 'fpd'}); "
            "print(json.dumps({'run': c.run, 'over': c.overrides}))")
    r = subprocess.run([sys.executable, "-c", code], env=_env(**OTHER_ROOTS), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout.splitlines()[-1])
    assert d["run"] == comp(FPD).run
    assert "++agent.config.ck_e2e2.fpd.teacher_det_run=/mnt/b/tc/refiner_fpd/det" in d["over"]


# ----------------------------------------------------------------------------------------------- effective_ck / arm_of
def test_effective_ck_has_the_fpd_keys_only_for_the_fpd_arm(tmp_path):
    rd = str(tmp_path / "r")
    y = L2.effective_ck(rd, 8)
    assert "arch" not in y and "fpd" not in y
    c = comp(FPD)
    ck = L2.effective_ck(c.run_dir, 8, c.set_lines, c.extra_lines)
    assert list(ck)[-2:] == ["arch", "fpd"] and ck["arch"] == "fpd" and list(ck["fpd"]) == list(L2.FPD_CK_DEFAULTS)
    assert {k: v for k, v in ck["fpd"].items() if not k.startswith("teacher_")} == \
        {k: v for k, v in L2.FPD_CK_DEFAULTS.items() if not k.startswith("teacher_")}
    assert [k for k in ck if k not in ("arch", "fpd")] == list(y)
    # arch alone: the whole fpd dict at its defaults; fpd lines alone: no arch key (a ck2 dict with an fpd block)
    a = L2.effective_ck(rd, 8, ["arch=fpd"])
    assert a["arch"] == "fpd" and a["fpd"] == L2.FPD_CK_DEFAULTS and a["fpd"] is not L2.FPD_CK_DEFAULTS
    b = L2.effective_ck(rd, 8, ["fpd.mt_k=2"])
    assert "arch" not in b and b["fpd"]["mt_k"] == 2 and L2.FPD_CK_DEFAULTS["mt_k"] == 4
    e = L2.effective_ck(rd, 8, ["arch=ck2"])
    assert e["arch"] == "ck2" and "fpd" not in e
    x = L2.effective_ck(rd, 8, [], ["++agent.config.ck_e2e2.arch=fpd", "++agent.config.ck_e2e2.fpd.steps=4"])
    assert x["arch"] == "fpd" and x["fpd"]["steps"] == 4
    with pytest.raises(L2.LaunchError, match="unknown key"):
        L2.effective_ck(rd, 8, ["fpd.nope=1"])
    with pytest.raises(L2.LaunchError, match="unknown key"):
        L2.effective_ck(rd, 8, ["archx=fpd"])


def test_arm_of():
    assert L2.arm_of({"enabled": True, "arch": "fpd"}) == "fpd"
    assert L2.arm_of({"enabled": True, "arch": "fpd", "use_teachers": False}) == "fpd_nokd"
    assert L2.arm_of({"enabled": False, "arch": "fpd"}) == "noref"
    for d, want in (({}, "main"), ({"use_teachers": False}, "nokd"), ({"enabled": "false"}, "noref"),
                    ({"bev_kd": {"enabled": True}}, "bevkd"), ({"arch": "ck2"}, "main")):
        assert L2.arm_of(d) == want
    for kw in ({}, {"REFINER_KD": "off"}):
        c = comp(dict(FPD, **kw))
        assert L2.arm_of(L2.effective_ck(c.run_dir, 8, c.set_lines, c.extra_lines)) != "noref"


def test_fpd_defaults_equal_the_config_defaults():
    O2 = pytest.importorskip("navsim.agents.para_ssr.ck.online2")
    if "fpd" not in {f.name for f in __import__("dataclasses").fields(O2.CKE2E2Config)}:
        pytest.skip("CKE2E2Config has no fpd field yet")
    assert O2.CKE2E2Config().fpd == L2.FPD_CK_DEFAULTS
    if hasattr(O2, "_d_fpd"):
        assert O2._d_fpd() == L2.FPD_CK_DEFAULTS and list(O2._d_fpd()) == list(L2.FPD_CK_DEFAULTS)
    if hasattr(O2, "ARCHS"):                            # the structures the agent knows = the REFINER_ARCH choices
        assert tuple(O2.ARCHS) == L2.ARCHS and O2.CKE2E2Config().arch == L2.ARCH_DEFAULT
    L2.fpd_defaults_check()
    # the dict the launcher emits is a valid config of the agent (the FPD teacher runs are checked by the launcher)
    c = comp(FPD)
    cfg = O2.CKE2E2Config.from_any(L2.effective_ck(c.run_dir, 8, c.set_lines, c.extra_lines))
    assert cfg.arch == "fpd" and cfg.fpd["teacher_det_run"].endswith("refiner_fpd/det")
    c = comp(dict(FPD, REFINER_KD="off"))
    cfg = O2.CKE2E2Config.from_any(L2.effective_ck(c.run_dir, 8, c.set_lines, c.extra_lines))
    cfg.validate(None)
    assert cfg.arch == "fpd" and not cfg.use_teachers and cfg.fpd["teacher_det_run"] == ""


# ----------------------------------------------------------------------------------------------- the FPD teacher guard
def _fpd_teacher(rd, arm, keys, over=None):
    """a teacher run in the format of tools/ck/fpd/train_fpd_teacher.py; over: config.json fields to replace, done False =
    no done.json"""
    over = dict(over or {})
    done = over.pop("done", True)
    rd.mkdir(parents=True)
    c = dict({"trainer": "train_fpd_teacher", "arm": arm, "score_keys": keys, "ep_target": "official", "seed": 0, "steps": 8,
              "grid": 8, "stage": "scorer"}, **over)
    (rd / "config.json").write_text(json.dumps(c))
    (rd / "ckpt_last.pt").write_bytes(b"not a real checkpoint: the guard reads its bytes only")
    if done:
        (rd / "done.json").write_text("{}")
    return rd


def _pair(tmp_path, **kw):
    det = _fpd_teacher(tmp_path / "det", "T", ["nc", "ttc", "ep", "comfort"], kw.get("det"))
    mp = _fpd_teacher(tmp_path / "map", "M", ["dac", "ep"], kw.get("map"))
    return str(det), str(mp)


KEYS = ["nc", "dac", "ep", "ttc"]


def test_fpd_teacher_check_accepts_a_finished_pair(tmp_path):
    det, mp = _pair(tmp_path)
    rec = L2.fpd_teacher_check(det, mp, "last", "strict", ep_target="official", kd_keys=KEYS)
    assert rec["det"]["arm"] == "T" and rec["map"]["arm"] == "M" and len(rec["det"]["ckpt_sha16"]) == 16
    assert rec["warnings"] == [] and rec["det"]["done"] == {} and rec["map"]["score_keys"] == ["dac", "ep"]


@pytest.mark.parametrize("what, kw, msg", [
    ("trainer", {"det": {"trainer": "train_ck2"}}, "trainer 'train_ck2'"), ("arm", {"map": {"arm": "T"}}, "arm 'T' != 'M'"),
    ("ep", {"det": {"ep_target": "decoupled"}}, "ep_target 'decoupled'"), ("keys", {"map": {"score_keys": ["ep"]}}, "lack \\['dac'\\]"),
    ("done", {"det": {"done": False}}, "no done.json"),
])
def test_fpd_teacher_check_refusals(tmp_path, what, kw, msg):
    det, mp = _pair(tmp_path, **kw)
    with pytest.raises(L2.LaunchError, match=msg):
        L2.fpd_teacher_check(det, mp, "last", "strict", ep_target="official", kd_keys=KEYS)


def test_fpd_teacher_check_smoke_mode_and_missing_runs(tmp_path, capsys):
    det, mp = _pair(tmp_path, det={"done": False})
    rec = L2.fpd_teacher_check(det, mp, "last", "smoke", ep_target="official", kd_keys=KEYS)
    assert rec["det"]["done"] is None and "accepted in smoke mode" in rec["warnings"][0]
    assert "WARNING" in capsys.readouterr().err
    with pytest.raises(L2.LaunchError, match="no config.json"):
        L2.fpd_teacher_check(str(tmp_path / "none"), mp, "last", "smoke", ep_target="official")
    with pytest.raises(L2.LaunchError, match="is not set"):
        L2.fpd_teacher_check("", mp, "last", "smoke", ep_target="official")
    (Path(mp) / "ckpt_last.pt").unlink()
    with pytest.raises(L2.LaunchError, match="ckpt_last.pt missing"):
        L2.fpd_teacher_check(det, mp, "last", "smoke", ep_target="official")
    # a teacher the student takes no KD key from needs none (single-source KD)
    det2, mp2 = _pair(tmp_path / "b", map={"score_keys": []})
    L2.fpd_teacher_check(det2, mp2, "last", "strict", ep_target="official", kd_keys=["nc", "ep", "ttc"], kd_sources=["det"])


def test_fpd_teacher_check_cli(tmp_path, capsys):
    """launch_util2.py fpd-teacher-check: the record (--out) of a usable pair, exit 3 with the reason otherwise."""
    pytest.importorskip("navsim.agents.para_ssr.ck.online2")
    det, mp = _pair(tmp_path)
    out = tmp_path / "rec.json"
    assert L2.main(["fpd-teacher-check", "--det", det, "--map", mp, "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["mode"] == "strict" and rec["det"]["arm"] == "T" and rec["map"]["arm"] == "M" and rec["warnings"] == []
    det3, mp3 = _pair(tmp_path / "c", det={"trainer": "train_ck2", "done": False})
    assert L2.main(["fpd-teacher-check", "--det", det3, "--map", mp3]) == 3
    assert "trainer 'train_ck2' != 'train_fpd_teacher'" in capsys.readouterr().err


def test_verify_prints_the_fpd_teachers(tmp_path, capsys):
    c = comp(FPD)
    C.write_composition(c, str(tmp_path / "gen"))
    (tmp_path / "ov.txt").write_text("\n".join(c.overrides) + "\n")
    tj = {r: {"ckpt_sha16": c.pins[r]["refiner_ckpt_sha16"]} for r in ("det", "map")}
    tj["fpd"] = {"det": {"run": "/x/refiner_fpd/det", "ckpt_sha16": "a" * 16, "done": {}},
                 "map": {"run": "/x/refiner_fpd/map", "ckpt_sha16": "b" * 16, "done": None}}
    (tmp_path / "t.json").write_text(json.dumps(tj))
    assert C.main(["verify", "--composition", str(tmp_path / "gen/composition.json"), "--overrides", str(tmp_path / "ov.txt"),
                   "--teachers-json", str(tmp_path / "t.json")]) == 0
    out = capsys.readouterr().out
    assert "det FPD teacher /x/refiner_fpd/det: ckpt sha16 aaaaaaaaaaaaaaaa, done.json yes" in out
    assert "map FPD teacher /x/refiner_fpd/map: ckpt sha16 bbbbbbbbbbbbbbbb, done.json NO" in out


# ----------------------------------------------------------------------------------------------- the script, with stubs
def _stub_tree(tmp_path, edit=None):
    """A copy of scripts/experiment.sh (optionally edited) beside links to the real tools/ and navsim/: REPO of the copy is
    this folder, the Python it runs is the repo's."""
    t = tmp_path / "tree"
    (t / "scripts").mkdir(parents=True)
    text = SH.read_text()
    if edit:
        for a, b in edit:
            assert a in text, a
            text = text.replace(a, b, 1)
    (t / "scripts/experiment.sh").write_text(text)
    for d in ("tools", "navsim", "resources"):
        os.symlink(REPO / d, t / d)
    return t


def test_teachers_arch_runs_the_fpd_script(tmp_path):
    """ARCH=fpd execs tools/release/setup/run_fpd_teachers.sh with the conventions of run_refiner_teachers.sh (here both
    are stubs of a copied tree: nothing trains)."""
    t = tmp_path / "tree"
    (t / "scripts").mkdir(parents=True)
    (t / "scripts/experiment.sh").write_text(SH.read_text())
    (t / "tools/release/setup").mkdir(parents=True)
    for f in ("load_paths.sh", "paths.env"):
        if (REPO / "tools/release" / f).is_file():
            os.symlink(REPO / "tools/release" / f, t / "tools/release" / f)
    (t / "navsim").mkdir()
    os.symlink(REPO / "navsim/common", t / "navsim/common")
    os.symlink(REPO / "tools/release/compose.py", t / "tools/release/compose.py")    # the default teachers (ARCH=fpd)
    os.symlink(REPO / "tools/ck", t / "tools/ck")
    for s in ("run_refiner_teachers", "run_fpd_teachers"):
        (t / f"tools/release/setup/{s}.sh").write_text(
            f'echo "{s} $DET_TEACHER $MAP_TEACHER ${{GPUS_DET:-}} ${{GPUS_MAP:-}} ${{WHICH:-}} $SSR_GPU_POLICY"\n')
    sh = t / "scripts/experiment.sh"

    def run(*args, cmd="teachers", **env):
        return subprocess.run(["bash", str(sh), cmd, *args], env=_env(**env), capture_output=True, text=True, timeout=60)

    assert run().stdout.strip() == "run_refiner_teachers bevfusion resmap    any"
    assert run("ARCH=ck2").stdout.strip() == "run_refiner_teachers bevfusion resmap    any"
    r = run("ARCH=fpd", "GPUS_DET=0,1", "GPUS_MAP=2,3", "WHICH=det")
    assert r.returncode == 0 and r.stdout.strip() == "run_fpd_teachers bevfusion resmap 0,1 2,3 det any", r.stderr
    assert run(ARCH="fpd").stdout.startswith("run_refiner_teachers")          # ARCH comes from the command line only
    r = run("ARCH=dit")
    assert r.returncode == 3 and "ARCH=dit: ck2 | fpd" in r.stderr
    r = run("REFINER_ARCH=fpd")
    assert r.returncode == 3 and "takes only" in r.stderr
    # the FPD teacher runs have one folder per role, read on the default teachers' BEV: no other teacher's BEV goes there
    # (the CK2 teachers of another registry entry have their own refiner_run)
    for args, who in ((["DET_TEACHER=bevfusion_x"], "DET_TEACHER=bevfusion_x"), (["MAP_TEACHER=resmap_x", "WHICH=map"],
                                                                                 "MAP_TEACHER=resmap_x")):
        r = run("ARCH=fpd", *args)
        assert r.returncode == 3 and f"teachers ARCH=fpd: {who}: the FPD teacher refiners are trained on the BEV of the " \
            "default teachers only" in r.stderr and not r.stdout.strip(), r.stdout + r.stderr
    r = run("ARCH=fpd", "MAP_TEACHER=resmap_x", "WHICH=det")                 # the map teacher is not trained
    assert r.returncode == 0 and r.stdout.strip() == "run_fpd_teachers bevfusion resmap_x   det any", r.stderr
    assert run("DET_TEACHER=bevfusion_x").stdout.strip() == "run_refiner_teachers bevfusion_x resmap    any"
    # ARCH belongs to teachers: a run's structure is REFINER_ARCH (train ARCH=fpd would launch the ck2 recipe)
    for cmd in ("name", "check", "train", "smoke", "assets", "status"):
        r = run("ARCH=fpd", cmd=cmd)
        assert r.returncode == 3 and "ARCH is an option of 'teachers' only" in r.stderr, (cmd, r.stdout + r.stderr)
    (t / "tools/release/setup/run_fpd_teachers.sh").unlink()
    r = run("ARCH=fpd")
    assert r.returncode == 3 and "run_fpd_teachers.sh is missing" in r.stderr


STUB_E2E = """#!/usr/bin/env bash
echo "$E2E_ROOT $RUN sha=$SSR_EXPERIMENT_SHA $*" >> "$STUB_LOG"
exit 0
"""


def _recorded_run(root, kw):
    """A run folder with launch/settings.json written as scripts/experiment.sh train writes it."""
    c = comp(dict(kw, RUN_ROOT=str(root)))
    gen = root / c.run / "launch/gen"
    C.write_composition(c, str(gen))
    assert C.main(["record", "--composition", str(gen / "composition.json"), "--out",
                   str(root / c.run / "launch/settings.json")]) == 0
    return c


@pytest.mark.parametrize("block, rec", [("fpd", {}), ("ck2", FPD), ("fpd", FPD), ("ck2", {})])
def test_resume_reads_a_record_without_the_key_as_ck2(tmp_path, block, rec):
    """A run is resumed with its recorded settings; a record without REFINER_ARCH (every ck2 run) is ck2 even when the
    settings block of the script says fpd."""
    t = _stub_tree(tmp_path, [("REFINER_ARCH=ck2            #", f"REFINER_ARCH={block}            #")])
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "train_e2e2.sh").write_text(STUB_E2E)
    root = tmp_path / "root"
    c = _recorded_run(root, rec)
    log = tmp_path / "stub.log"
    r = subprocess.run(["bash", str(t / "scripts/experiment.sh"), "resume", f"RUN={c.run}", f"RUN_ROOT={root}", "ALLOW_DIRTY=1",
                        "MIN_FREE_GB=0"], env=_env(SSR_RELEASE_TEST_STUBS=str(stubs), STUB_LOG=str(log)),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"{root} {c.run} sha={c.hash6} start --resume" in log.read_text()
    res = json.loads(next((root / c.run / "launch").glob("resume_*.json")).read_text())
    assert res["run"] == c.run and res["settings"] == c.settings


def test_resume_without_the_reset_would_take_the_block(tmp_path):
    """the control of the test above: with the reset line removed, a ck2 record under an fpd block composes another run"""
    t = _stub_tree(tmp_path, [("REFINER_ARCH=ck2            #", "REFINER_ARCH=fpd            #"),
                              ("  REFINER_ARCH=ck2\n  while IFS=", "  while IFS=")])
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "train_e2e2.sh").write_text(STUB_E2E)
    root = tmp_path / "root"
    c = _recorded_run(root, {})
    r = subprocess.run(["bash", str(t / "scripts/experiment.sh"), "resume", f"RUN={c.run}", f"RUN_ROOT={root}", "ALLOW_DIRTY=1",
                        "MIN_FREE_GB=0"], env=_env(SSR_RELEASE_TEST_STUBS=str(stubs), STUB_LOG=str(tmp_path / "log")),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 3 and f"this code composes {comp(dict(FPD, RUN_ROOT=str(root))).run}" in r.stderr, r.stderr


def test_settings_args_print_only_the_recorded_keys(tmp_path):
    for kw in ({}, FPD):
        c = _recorded_run(tmp_path / ("a" if kw else "b"), kw)
        out = subprocess.run([sys.executable, str(REPO / "tools/release/compose.py"), "settings-args", "--settings-json",
                              str(tmp_path / ("a" if kw else "b") / c.run / "launch/settings.json")], env=_env(),
                             capture_output=True, text=True, timeout=60).stdout.splitlines()
        assert ("REFINER_ARCH=fpd" in out) == bool(kw) and not any(x.startswith("REFINER_ARCH=ck2") for x in out)
        assert len(out) == len(c.settings)


# ----------------------------------------------------------------------------------------------- smoke check
def _settings(run, **kw):
    s = C.canonicalize(kw, C.load_registry(REPO_REGISTRY))
    (run / "launch").mkdir(parents=True, exist_ok=True)
    (run / "launch/settings.json").write_text(json.dumps({"run": run.name, "settings": s}))
    return s


def _common_files(run, ckpt_keys):
    import torch
    (run / "launch/train.exit").write_text("0\n")
    (run / "train/lightning_logs/version_0/checkpoints").mkdir(parents=True)
    (run / "train/train_tokens.json").write_text(json.dumps({"count": 480}))
    (run / "train/train_time.json").write_text(json.dumps({"sessions": [{"status": "completed", "epochs_completed": 2}]}))
    ck = run / "train/lightning_logs/version_0/checkpoints"
    torch.save({"state_dict": {k: torch.zeros(1) for k in ckpt_keys}}, ck / "last.ckpt")
    (ck / "epoch=1-step=30.ckpt").write_text("")


def _refiner_files(run, teachers, kd=0.3, lat=0.0, teacher_ms=80.0, cum=None, eff=None):
    io = run / "refiner"
    (io / "lab").mkdir(parents=True)
    ep = [{"event": "epoch_start", "epoch": 0, "phase": "warmup_record"}, {"event": "epoch_end", "epoch": 0, "cum": {}},
          {"event": "epoch_start", "epoch": 1, "phase": "onpolicy"},
          {"event": "epoch_end", "epoch": 1, "cum": {"nonfinite_loss": 0, "sur_nonfinite": 0} if cum is None else cum}]
    (io / "epochs.jsonl").write_text("\n".join(json.dumps(e) for e in ep) + "\n")
    rows = [{"epoch": 0, "mem_gb": 15.0, "skipped_steps": 0, "ck2/kd_ok_frac": 1.0, "ck2/wt_kd_score": kd, "ck2/wt_lat_kd": lat,
             "ck2/teacher_ms": teacher_ms, "ck2/cal_pairs": 10.0}]
    (io / "steps_rank0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (io / "lab/status.json").write_text(json.dumps({"tokens_ok": 10, "tokens_err": 0, "epochs": {"0": {"done": True, "rows_done": 10}}}))
    (run / "launch/teachers.json").write_text(json.dumps(teachers))
    eff = {"arch": "fpd", "fpd": dict(L2.FPD_CK_DEFAULTS)} if eff is None else eff
    (run / "launch/refiner_effective.json").write_text(json.dumps(eff))


def _calib_file(run, keys):
    ok = {k: {"fitted": True, "a": 1.0, "b": 0.0} for k in keys}
    lines = [{"window_phase": "warmup_record", "keys": ok} for _ in range(5)] + [{"window_phase": "onpolicy"}]
    (run / "refiner/kd_calib_online.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


ALL_KEYS = ["det/nc", "det/ep", "det/ttc", "map/dac", "map/ep"]
FPD_T = {"det": {}, "map": {}, "kd_calib": {"enabled": True, "mode": "online"},
         "fpd": {"det": {"run": "/t/refiner_fpd/det", "ckpt_sha16": "a" * 16}, "map": {"run": "/t/refiner_fpd/map", "ckpt_sha16": "b" * 16}}}
FPD_KEYS = ["agent.para_ssr_model.x", "agent._distill.x", "agent.ck_student.tok.proj.weight", "agent.ck_student.dit.blocks.0.ada.1.weight",
            "agent.ck_student.scorer.heads.0.0.weight"]


def _fpd_run(tmp_path, **kw):
    run = tmp_path / "r"
    _settings(run, REFINER_ARCH="fpd", **kw)
    _common_files(run, FPD_KEYS)
    return run


def test_smoke_check_fpd_passes_and_catches_what_is_not_fpd(tmp_path):
    run = _fpd_run(tmp_path)
    _refiner_files(run, FPD_T, cum={"nonfinite_loss": 0})       # no surrogate: sur_nonfinite may be absent
    _calib_file(run, ALL_KEYS)
    res = SC.check_run(str(run))
    assert res["pass"], res
    assert res["info"]["arch"] == "fpd" and res["checks"]["arch_as_set"] and res["checks"]["ckpt_fpd_student"]
    assert res["checks"]["fpd_teachers_recorded"] and res["checks"]["lateral_kd_as_set"]
    # lateral KD on: the FPD refiner has none
    steps = run / "refiner/steps_rank0.jsonl"
    good = steps.read_text()
    steps.write_text(good.replace('"ck2/wt_lat_kd": 0.0', '"ck2/wt_lat_kd": 0.2'))
    assert not SC.check_run(str(run))["checks"]["lateral_kd_as_set"]
    steps.write_text(good)
    # the launcher ran another structure, or recorded no FPD teacher
    (run / "launch/refiner_effective.json").write_text(json.dumps({}))
    assert not SC.check_run(str(run))["checks"]["arch_as_set"]
    (run / "launch/teachers.json").write_text(json.dumps(dict(FPD_T, fpd={})))
    assert not SC.check_run(str(run))["checks"]["fpd_teachers_recorded"]


def test_smoke_check_fpd_checkpoint_must_hold_the_fpd_student(tmp_path):
    run = tmp_path / "r"
    _settings(run, REFINER_ARCH="fpd")
    _common_files(run, ["agent.para_ssr_model.x", "agent._distill.x", "agent.ck_student.trunk.lon_head.0.weight"])
    _refiner_files(run, FPD_T)
    _calib_file(run, ALL_KEYS)
    res = SC.check_run(str(run))
    assert not res["checks"]["ckpt_fpd_student"] and res["checks"]["ckpt_refiner_as_set"]


def test_smoke_check_fpd_without_teachers(tmp_path):
    run = _fpd_run(tmp_path, REFINER_KD="off", BEV_KD="off")
    import torch                 # BEV KD off: the checkpoint of _fpd_run without its _distill tensor
    torch.save({"state_dict": {k: torch.zeros(1) for k in FPD_KEYS if "_distill" not in k}},
               run / "train/lightning_logs/version_0/checkpoints/last.ckpt")
    _refiner_files(run, {"mode": "none", "kd_calib": {"enabled": False}}, kd=0.0, teacher_ms=0.1)
    res = SC.check_run(str(run))
    assert res["pass"], res
    assert "fpd_teachers_recorded" not in res["checks"] and res["checks"]["no_teacher_record"]
    (run / "launch/teachers.json").write_text(json.dumps({"mode": "none", "fpd": {"det": {}}}))
    assert not SC.check_run(str(run))["checks"]["no_teacher_record"]


def test_smoke_check_of_a_ck2_run_has_no_arch(tmp_path):
    run = tmp_path / "r"
    _settings(run)
    _common_files(run, ["agent.para_ssr_model.x", "agent.ck_student.x", "agent._distill.x"])
    _refiner_files(run, {"det": {}, "map": {}, "kd_calib": {"enabled": True, "mode": "online"}}, lat=0.2, eff={})
    _calib_file(run, ALL_KEYS)
    res = SC.check_run(str(run))
    assert res["pass"], res
    assert "arch" not in res["info"] and not {"arch_as_set", "ckpt_fpd_student", "fpd_teachers_recorded"} & set(res["checks"])
    # ck2 still needs the surrogate record
    ep = run / "refiner/epochs.jsonl"
    ep.write_text(ep.read_text().replace(', "sur_nonfinite": 0', ""))
    assert not SC.check_run(str(run))["checks"]["no_nonfinite_loss"]


# ----------------------------------------------------------------------------------------------- check_build
def _check_build_module():
    spec = importlib.util.spec_from_file_location("ssr_check_build_under_test", REPO / "tools/release/check_build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_agent(arch):
    import torch
    cfg = SimpleNamespace(image_architecture="resnet34.tv_in1k", frame_indices=[2, 3], bev_h=100, bev_w=100,
                          pc_range=[-32.0, 0.0, -1.0, 32.0, 64.0, 3.0], use_task_interaction=True, use_det_motion_head=True,
                          use_map_head=True, map_head_type="seg", undistort=True, plan_heading_from_xy=True)
    ck_cfg = SimpleNamespace(use_teachers=True, teacher_det_run="/d", teacher_map_run="/m", kd_sources=["det", "map"],
                             kd_score_keys=KEYS, lambda_kd_score=0.5, lat_kd={"balance": "fixed", "fixed": 0.0},
                             kd_calib={"enabled": True, "mode": "online"}, infer_select=None, submit_selection=False, seed=0)
    if arch is not None:
        ck_cfg.arch, ck_cfg.fpd = arch, dict(L2.FPD_CK_DEFAULTS)
    student = torch.nn.Linear(2, 2)
    if arch == "fpd":
        student.param_counts = lambda: {"tok": 1, "dit": 2, "scorer": 3}
    lin = torch.nn.Linear(3, 1)
    return SimpleNamespace(_config=cfg, para_ssr_model=SimpleNamespace(_backbone_bn_mode="moving"), _distill=None,
                           named_parameters=lambda: list(lin.named_parameters()), get_target_builders=lambda: [],
                           _ck_e2e2=SimpleNamespace(cfg=ck_cfg), ck_student=student)


def test_check_build_records_the_arch_only_off_ck2():
    m = _check_build_module()
    for arch in (None, "ck2"):
        rec = m.summarize(_fake_agent(arch))
        assert not {"arch", "fpd", "student_params"} & set(rec["refiner"])
    rec = m.summarize(_fake_agent("fpd"))
    assert rec["refiner"]["arch"] == "fpd" and rec["refiner"]["fpd"] == L2.FPD_CK_DEFAULTS
    assert rec["refiner"]["student_params"] == {"tok": 1, "dit": 2, "scorer": 3}


# ----------------------------------------------------------------------------------------------- eval hook
@pytest.fixture(scope="module")
def EV():
    return pytest.importorskip("tools.ck.e2e2.eval_epdms")


def test_eval_gives_a_token_keyed_refiner_its_tokens(EV):
    class H:
        needs_tokens = True

    a = SimpleNamespace(_ck_e2e2=H())
    EV.set_infer_tokens(a, ["t1", "t2"])
    assert a._ck_e2e2.infer_tokens == ["t1", "t2"]
    b = SimpleNamespace(_ck_e2e2=SimpleNamespace())
    EV.set_infer_tokens(b, ["t1"])
    assert not hasattr(b._ck_e2e2, "infer_tokens")          # the CK2 refiner: nothing set
    EV.set_infer_tokens(SimpleNamespace(), ["t1"])           # no refiner at all
    EV.set_infer_tokens(SimpleNamespace(_ck_e2e2=None), ["t1"])
    src = (REPO / "tools/ck/e2e2/eval_epdms.py").read_text()
    loop = src.split("for btok, feats, _ in dl:")[1].split("cand96 = pr[")[0]
    assert loop.index("set_infer_tokens(agent, btok)") < loop.index("pr = agent(fg)")
    assert loop.count("set_infer_tokens(agent, btok)") == 2 and loop.index("set_infer_tokens(agent, btok)", loop.index(
        "pr = agent(fg)")) < loop.index("agent._ck_e2e2.infer(")


def test_eval_refuses_the_fallback_configs_for_an_fpd_run(tmp_path, EV):
    run = tmp_path / "run"
    ck = run / "train/lightning_logs/version_0/checkpoints/last.ckpt"
    ck.parent.mkdir(parents=True)
    ck.write_text("x")
    (run / "launch").mkdir()
    assert EV.recorded_arch(run, ck) == ("ck2", None)
    (run / "launch/settings.json").write_text(json.dumps({"settings": {"REFINER_ARCH": "fpd"}}))
    assert EV.recorded_arch(run, ck) == ("fpd", run / "launch/settings.json")
    assert EV.recorded_arch(ck.parent, ck)[0] == "fpd"                 # found from the checkpoint's ancestors
    with pytest.raises(SystemExit, match="REFINER_ARCH=fpd"):
        EV.resolve_hydra_cfg(run, ck)
    assert EV.try_import_eval_e2e2_dump(run, tmp_path / "pool", "navtest") is False
    (run / "launch/settings.json").unlink()
    (run / "launch/refiner_effective.json").write_text(json.dumps({"arch": "fpd"}))
    assert EV.recorded_arch(run, ck)[0] == "fpd"
    # its own config is used as before, and an explicit one always
    own = run / "train/code/hydra/config.yaml"
    own.parent.mkdir(parents=True)
    own.write_text("agent: {}\n")
    assert EV.resolve_hydra_cfg(run, ck) == own.resolve() and EV.resolve_hydra_cfg(run, ck, str(own)) == own.resolve()
    # a ck2 run without a config of its own still gets the fallback (not built here: the synthesized one writes a file)
    (run / "launch/refiner_effective.json").write_text(json.dumps({"enabled": True}))
    assert EV.recorded_arch(run, ck) == ("ck2", None)
