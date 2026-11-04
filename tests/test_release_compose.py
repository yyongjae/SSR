"""Release contract section 4: scripts/experiment.sh, tools/release/{compose,check_build,smoke_check,table}.py, registry.yaml.
All CPU.  Run:
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_release_compose.py
Tests that run the real launcher and build the real agent need this server's teachers and caches and skip elsewhere.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from navsim.common import ssr_paths as SP  # noqa: E402
from tools.ck.e2e2 import launch_util2 as L2  # noqa: E402
from tools.release import compose as C  # noqa: E402
from tools.release import smoke_check as SC  # noqa: E402
try:
    from tools.release import table as TB  # noqa: E402
except (ImportError, FileNotFoundError):  # the results table belongs to the eval side, committed separately
    TB = None
needs_table = pytest.mark.skipif(TB is None, reason="tools/release/table.py is not in this tree")

SH = REPO / "scripts/experiment.sh"
REFERENCE_RUN = Path("/home/external-user/ssd/yongjae_refiner/v7/v2_v15_v7_main_overrides_reference.txt")
SERVER = (Path("/home/external-user/datasets/teacher_cache/refiner/det/ckpt_last.pt").is_file()
          and Path("/home/external-user/datasets/teacher_cache/bevfusion/cache_train_100x100/manifest.json").is_file())
server_only = pytest.mark.skipif(not SERVER, reason="needs this server's teachers and caches")
NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
ENV_DROP = set(SP.NAMES) | set(C.CONTROL_ENV) | {"SSR_RELEASE_REGISTRY", "SSR_RELEASE_TEST_STUBS", "ALLOW_DIRTY", "CHECK_ROOT"}


def name(**kw):
    return C.compose(kw).run


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ENV_DROP}
    env.update(PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2")
    env.update(extra)
    return env


def sh(*args, timeout=600, **extra):
    return subprocess.run(["bash", str(SH), *args], env=_env(**extra), capture_output=True, text=True, timeout=timeout)


# ----------------------------------------------------------------------------------------------- settings surface
def test_script_block_equals_the_compose_defaults():
    block, seen = SH.read_text().split("end of settings")[0], {}
    for line in block.splitlines():
        m = re.match(r"^([A-Z_]+)=(\S*)\s*(#.*)?$", line)
        if m:
            seen[m.group(1)] = m.group(2)
            assert m.group(3), f"{m.group(1)}: the block says what each key does"
    seen.pop("RUN_ROOT")      # the block names this team's run folder; compose's default stays $SSR_RUN_ROOT
    assert seen == {k: v for k, v in C.DEFAULTS.items() if k != "RUN_ROOT"}


def test_defaults_match_the_contract_name():
    assert re.fullmatch(r"v2f_r34_bevfusion-resmap_ref1_rkd1_bkd1_det1_map1_stbase_bnm_e30_g8_s0_[0-9a-f]{6}", name())


ARMS = {
    "main": {}, "noref": {"REFINER": "off"}, "norefkd": {"REFINER_KD": "off"}, "noscorekd": {"REFINER_KD": "lat"},
    "nolatkd": {"REFINER_KD": "score"}, "nobevkd": {"BEV_KD": "off"}, "nokd": {"REFINER_KD": "off", "BEV_KD": "off"},
    "base": {"REFINER": "off", "BEV_KD": "off"}, "detoff": {"DET_KD": "off"}, "mapoff": {"MAP_KD": "off"},
    "bevkd_detoff": {"DET_KD": "off_bev"}, "bevkd_mapoff": {"MAP_KD": "off_bev"}, "refkd_detoff": {"DET_KD": "off_ref"},
    "refkd_mapoff": {"MAP_KD": "off_ref"}, "noref_detoff": {"REFINER": "off", "DET_KD": "off"},
    "norefkd_mapoff": {"REFINER_KD": "off", "MAP_KD": "off"}, "mix": {"DET_KD": "off_bev", "MAP_KD": "off_ref"},
    "r18": {"BACKBONE": "r18"}, "r50": {"BACKBONE": "r50"}, "r101": {"BACKBONE": "r101"}, "r152": {"BACKBONE": "r152"},
    "r50d": {"BACKBONE": "r50d"}, "cnxt": {"BACKBONE": "convnext_tiny"}, "regnet": {"BACKBONE": "regnety_032"},
    "effnet": {"BACKBONE": "efficientnet_b0"}, "noint": {"INTERACTION": "off"},
    "nodet": {"INTERACTION": "off", "DET_HEAD": "off"}, "nomap": {"INTERACTION": "off", "MAP_HEAD": "off"},
    "noheads": {"INTERACTION": "off", "DET_HEAD": "off", "MAP_HEAD": "off"}, "vecmap": {"MAP_HEAD": "vector"},
    "dist": {"UNDISTORT": "off"}, "z22": {"PILLAR_Z": "-2,2"}, "z15": {"PILLAR_Z": "-1.5,2.5"},
    "r32": {"BEV_RANGE": "32", "REFINER": "off", "BEV_KD": "off"}, "f1": {"FRAMES": "3"}, "f4": {"FRAMES": "0,1,2,3"},
    "headfree": {"HEADING": "free"}, "bnf": {"BN": "frozen"}, "s1": {"SEED": "1"}, "e20": {"EPOCHS": "20"},
    "g4": {"GPUS": "0,1,2,3"}, "g2": {"GPUS": "0,1"}, "tag": {"TAG": "a.1-b"}, "x": {"EXTRA": "agent.lr=2e-4"},
}


def test_arm_names_are_distinct_and_in_charset():
    names = {k: C.compose(v).run for k, v in ARMS.items()}
    assert len(ARMS) >= 15 and len(set(names.values())) == len(names), names
    for n in names.values():
        assert NAME_RE.fullmatch(n) and n.startswith("v2f_") and len(n) < 200
    # parts of the name
    assert "_ref0_rkd0_bkd0_det0_map0_" in names["base"]
    assert "_ref1_rkd0_bkd0_det0_map0_" in names["nokd"]
    assert "_rkdL_" in names["noscorekd"] and "_rkdS_" in names["nolatkd"]
    assert "_det0_map1_" in names["detoff"] and "_det0b_map1_" in names["bevkd_detoff"]
    assert "_det1_map0r_" in names["refkd_mapoff"] and "_det0b_map0r_" in names["mix"]
    assert names["cnxt"].startswith("v2f_convnext_tiny_") and "_stz-1p5_2p5_" in names["z15"]
    assert "_stnoint.nodet.nomap_" in names["noheads"] and "_str32_" in names["r32"]
    assert "_stf1_" in names["f1"] and "_stf4_" in names["f4"] and "_stdist_" in names["dist"]
    assert "_stvecmap_" in names["vecmap"] and "_stheadfree_" in names["headfree"]
    assert "_bnf_" in names["bnf"] and "_s1_" in names["s1"] and "_e20_" in names["e20"]
    assert "_g4_" in names["g4"] and "_g2_" in names["g2"]
    assert "_a.1-b_" in names["tag"] and re.search(r"_x_[0-9a-f]{6}$", names["x"])


@pytest.mark.parametrize("a, b", [
    ({"REFINER": "off"}, {"REFINER": "off", "REFINER_KD": "score"}),
    ({"REFINER_KD": "off", "BEV_KD": "off"}, {"REFINER_KD": "off", "BEV_KD": "off", "DET_KD": "off_bev", "MAP_KD": "off_ref"}),
    ({"REFINER": "off", "BEV_KD": "off"}, {"REFINER": "off", "REFINER_KD": "lat", "BEV_KD": "off", "DET_KD": "off"}),
    ({"REFINER_KD": "off", "DET_KD": "off"}, {"REFINER_KD": "off", "DET_KD": "off_bev"}),
    ({"REFINER": "off", "MAP_KD": "off"}, {"REFINER": "off", "MAP_KD": "off_bev"}),
    ({"BEV_KD": "off", "DET_KD": "off"}, {"BEV_KD": "off", "DET_KD": "off_ref"}),
    ({}, {"PILLAR_Z": "-1.0,3.0"}),
    ({"PILLAR_Z": "-2,2"}, {"PILLAR_Z": "-2.0,2"}),
])
def test_one_config_one_name(a, b):
    assert name(**a) == name(**b)
    # idempotent: the record of a run composes to the same run
    s = C.compose(a).settings
    assert C.canonicalize(s) == s and C.compose(s).run == name(**a)


@pytest.mark.parametrize("kw, msg", [
    ({"BACKBONE": "swin"}, "needs code"), ({"BACKBONE": "vit"}, "needs code"), ({"BACKBONE": "r200"}, "not in the registry"),
    ({"DET_TEACHER": "bevformer"}, "Missing: a BEVFormer"), ({"MAP_TEACHER": "mask2map"}, "Missing: a Mask2Map"),
    ({"MAP_TEACHER": "nope"}, "not in the registry"), ({"DET_KD": "off", "MAP_KD": "off"}, "use REFINER_KD=off BEV_KD=off"),
    ({"DET_KD": "off_ref", "REFINER_KD": "off"}, "means nothing"), ({"DET_KD": "off_ref", "REFINER": "off"}, "means nothing"),
    ({"MAP_KD": "off_bev", "BEV_KD": "off"}, "means nothing"), ({"DET_KD": "off_bev", "MAP_KD": "off_bev"}, "use BEV_KD=off"),
    ({"DET_KD": "off", "MAP_KD": "off_bev"}, "use BEV_KD=off"), ({"DET_KD": "off_ref", "MAP_KD": "off_ref"}, "use REFINER_KD=off"),
    ({"DET_KD": "off", "MAP_KD": "off_ref"}, "use REFINER_KD=off"), ({"DET_HEAD": "off"}, "need INTERACTION=off"),
    ({"MAP_HEAD": "off"}, "need INTERACTION=off"), ({"BEV_RANGE": "32"}, "only with REFINER=off BEV_KD=off"),
    ({"BEV_RANGE": "32", "REFINER": "off"}, "only with"), ({"BEV_RANGE": "32", "BEV_KD": "off"}, "only with"),
    ({"GPUS": "0,1,2"}, "distinct"), ({"GPUS": "0,0"}, "distinct"), ({"GPUS": "a,b"}, "distinct"), ({"GPUS": "0"}, "distinct"),
    ({"TAG": "a_b"}, "characters"), ({"TAG": "a/b"}, "characters"), ({"SEED": "-1"}, "integer"), ({"EPOCHS": "x"}, "integer"),
    ({"EPOCHS": "5"}, "needs more than 5"), ({"SAVE_EVERY": "7"}, "does not divide"), ({"PILLAR_Z": "3,1"}, "below"),
    ({"PILLAR_Z": "1"}, "a,b"), ({"PILLAR_Z": "a,b"}, "a,b"), ({"REFINER": "maybe"}, "one of"), ({"FRAMES": "1,3"}, "one of"),
    ({"BN": "x"}, "one of"), ({"WANDB": "x"}, "one of"), ({"RUN_ROOT": "rel/path"}, "absolute"), ({"RUN_ROOT": "/a=b"}, "absolute"),
    ({"NOPE": "1"}, "unknown setting"),
])
def test_refusals(kw, msg):
    with pytest.raises(C.Refused, match=msg):
        C.compose(kw)


@pytest.mark.parametrize("line", [
    "agent.config.ck_e2e2.infer_select={rule:wote}", "agent.config.ck_e2e2.submit_selection=true",
    "agent.config.ck_e2e2.kd_calib.mode=file", "agent.config.ck_e2e2.teacher_det_run=/x",
    "agent.config.ck_e2e2.warmup.sampler_seed=1", "agent.config.distill_lambda_gt=0.1", "agent.config.distill_tau_gt=1.0",
    "agent.config.plan_anchor_file=/x", "dataloader.params.batch_size=8", "agent.config.ck_e2e2.io_dir=/x", "output_dir=/x",
    "seed=3", "agent.config.distill_probe_seed=3", "++agent.config.ck_e2e2.seed=1", "~agent.config.ck_e2e2.kd_calib",
    "agent.lr=1e-4;seed=3",
])
def test_extra_refuses_the_locked_keys(line):
    with pytest.raises(C.Refused, match="locked"):
        C.compose({"EXTRA": line})


@pytest.mark.parametrize("line, setting", [
    ("agent.config.use_distill=false", "BEV_KD"), ("++agent.config.ck_e2e2.enabled=false", "REFINER"),
    ("agent.config.image_architecture=resnet50.a1_in1k", "BACKBONE"), ("agent.config.frame_indices=[3]", "FRAMES"),
    ("agent.config.ck_e2e2.lat_kd.fixed=0.0", "REFINER_KD"), ("trainer.params.max_epochs=3", "EPOCHS"),
])
def test_extra_cannot_contradict_the_run_name(line, setting):
    with pytest.raises(C.Refused, match=setting):
        C.compose({"EXTRA": line})


def test_extra_enters_the_hash_and_reaches_the_overrides():
    a, b = C.compose({"EXTRA": "agent.lr=2e-4"}), C.compose({"EXTRA": "agent.lr=3e-4"})
    assert a.hash6 != b.hash6 and a.hash6 != C.compose({}).hash6
    assert "agent.lr=2e-4" in a.overrides and "agent.lr=1e-4" not in a.overrides


# ----------------------------------------------------------------------------------------------- hash
@pytest.mark.parametrize("kw", [{"RUN_ROOT": "/tmp/other"}, {"WANDB": "disable"}, {"WANDB": "online"}, {"SAVE_EVERY": "5"},
                                {"GPUS": "7,6,5,4,3,2,1,0"}, {"TAG": "x"}, {"DRY": "1"}])
def test_hash_ignores_what_is_not_the_recipe(kw):
    assert C.compose(kw).hash6 == C.compose({}).hash6


@pytest.mark.parametrize("kw", [{"SEED": "1"}, {"EPOCHS": "20"}, {"GPUS": "0,1,2,3"}, {"BN": "frozen"}, {"BACKBONE": "r50"},
                                {"REFINER_KD": "score"}, {"DET_KD": "off_bev"}])
def test_hash_changes_with_the_recipe(kw):
    assert C.compose(kw).hash6 != C.compose({}).hash6


def test_hash_ignores_commit_and_is_stable_across_processes():
    code = "from tools.release import compose as C; print(C.compose({}).run)"
    r = subprocess.run([sys.executable, "-c", code], env=_env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == name()


OTHER_ROOTS = {"SSR_DATA_ROOT": "/mnt/a/data", "SSR_TEACHER_CACHE": "/mnt/b/tc", "SSR_PLAN_DATA": "/mnt/c/plan",
               "SSR_RUN_ROOT": "/mnt/d/runs", "OPENSCENE_DATA_ROOT": "/mnt/e/ds"}


def test_one_config_one_name_on_any_server():
    code = ("from tools.release import compose as C; c = C.compose({}); import json; "
            "print(json.dumps({'run': c.run, 'over': c.overrides, 'env': c.env}))")
    r = subprocess.run([sys.executable, "-c", code], env=_env(**OTHER_ROOTS), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout.splitlines()[-1])
    assert d["run"] == name()
    over = "\n".join(d["over"])
    # every root the recipe reads comes from the environment
    for needle in ("agent.config.plan_score_file=/mnt/c/plan/pdm_score_256",
                   "agent.config.ck_e2e2.warmup.packed=/mnt/a/data/ck/packed/navtrain_train",
                   "agent.config.ck_e2e2.teacher_det_run=/mnt/b/tc/refiner/det",
                   "agent.config.distill_feature_root=/mnt/b/tc", "agent.config.ref_data_root=/mnt/a/data",
                   "agent.config.distill_v7_root=/mnt/b/tc", "output_dir=/mnt/d/runs/"):
        assert needle in over, needle
    assert "/home/external-user" not in over and "/workspace/" not in over.replace(str(REPO), "")
    assert d["env"]["SSR_DET_CACHE_TRAIN"] == "/mnt/b/tc/bevfusion/cache_train_100x100"
    assert d["env"]["SSR_MAP_CACHE_ROOT"] == "/mnt/b/tc/resmap/stage1_front64"
    assert d["env"]["E2E_ROOT"] == "/mnt/d/runs"


def test_hash_follows_the_registry_pins(tmp_path):
    txt = (REPO / "tools/release/registry.yaml").read_text()
    new = tmp_path / "r.yaml"
    new.write_text(txt.replace('refiner_ckpt_sha16: "66c6a413dd866898"', 'refiner_ckpt_sha16: "ffffffffffffffff"'))
    assert C.compose({}, registry=str(new)).hash6 != C.compose({}).hash6
    # that teacher is read by nothing with the refiner KD off and its bank off: its pin is not part of the config
    a = {"REFINER_KD": "off", "DET_KD": "off"}
    assert C.compose(a, registry=str(new)).hash6 == C.compose(a).hash6


# ----------------------------------------------------------------------------------------------- what an arm emits
def _kv(c):
    return {C.LU._okey(o): o.split("=", 1)[1] for o in c.overrides if "=" in o}


CK = "agent.config.ck_e2e2."


def test_refiner_off_lines():
    c = C.compose({"REFINER": "off"})
    k = _kv(c)
    assert k[CK + "enabled"] == "false" and k["agent.config.use_distill"] == "true"
    assert k[CK + "teacher_det_run"] == "null" and c.env["E2E_NO_LABELER"] == "1"
    assert L2.arm_of(L2.effective_ck(c.run_dir, 8, c.set_lines, c.extra_lines)) == "noref"
    assert c.pins["det"]["name"] == "bevfusion" and c.profile == "full"


def test_refiner_kd_off_is_no_teacher_also_with_bev_kd_on():
    c = C.compose({"REFINER_KD": "off"})
    k = _kv(c)
    assert k[CK + "use_teachers"] == "false" and k[CK + "teacher_det_run"] == "null" and k[CK + "teacher_map_run"] == "null"
    assert k[CK + "lambda_kd_score"] == "0.0" and k[CK + "lat_kd.balance"] == "fixed" and k[CK + "lat_kd.fixed"] == "0.0"
    assert k[CK + "kd_calib.enabled"] == "false" and k[CK + "kd_calib.mode"] == "online"   # enabled=false alone (critic C7)
    assert k["agent.config.use_distill"] == "true"
    assert "E2E_NO_LABELER" not in c.env and "SSR_DET_CACHE_TRAIN" in c.env


def test_no_kd_needs_no_teacher_at_all():
    for kw, prof in (({"REFINER_KD": "off", "BEV_KD": "off"}, "refiner_noteacher"), ({"REFINER": "off", "BEV_KD": "off"}, "base")):
        c = C.compose(kw)
        assert c.pins == {} and c.profile == prof
        assert not any(k.startswith(("SSR_DET", "SSR_MAP")) for k in c.env)
        k = _kv(c)
        assert k["agent.config.use_distill"] == "false"
        assert not any(x.startswith("agent.config.distill_") for x in k)


@pytest.mark.parametrize("kw, off, src, keys", [
    ({"DET_KD": "off"}, "[det]", "[map]", "[dac,ep]"), ({"MAP_KD": "off"}, "[map]", "[det]", "[nc,ep,ttc]"),
    ({"DET_KD": "off_bev"}, "[det]", "[det,map]", "[nc,dac,ep,ttc]"), ({"MAP_KD": "off_ref"}, "[]", "[det]", "[nc,ep,ttc]"),
])
def test_single_teacher_lines(kw, off, src, keys):
    k = _kv(C.compose(kw))
    assert k["agent.config.distill_off_banks"] == off
    assert k[CK + "kd_sources"] == src and k[CK + "kd_score_keys"] == keys
    assert k[CK + "teacher_det_run"].endswith("refiner/det") and k[CK + "teacher_map_run"].endswith("refiner/map")


def test_a_bank_off_is_not_given_a_norm_file_and_keeps_the_other():
    k = _kv(C.compose({"DET_KD": "off_bev"}))
    assert k["agent.config.distill_norm_files"] == "{resmap:resources/km_corridor/norm_resmap_100x100.npz}"


def test_partial_refiner_kd_lines():
    k = _kv(C.compose({"REFINER_KD": "score"}))
    assert k[CK + "lat_kd.balance"] == "fixed" and k[CK + "lat_kd.fixed"] == "0.0" and k[CK + "lambda_kd_score"] == "0.5"
    assert k[CK + "kd_calib.enabled"] == "true"
    k = _kv(C.compose({"REFINER_KD": "lat"}))
    assert k[CK + "lat_kd.balance"] == "ema" and k[CK + "lambda_kd_score"] == "0.0"
    assert k[CK + "kd_calib.enabled"] == "false"     # calibration only acts on score KD logits


def test_seed_lines_and_the_sampler_seed_stays():
    k = _kv(C.compose({"SEED": "3"}))
    assert k["seed"] == "3" and k[CK + "seed"] == "3" and k["agent.config.distill_probe_seed"] == "3"
    assert k[CK + "warmup.sampler_seed"] == "0"
    assert "agent.config.distill_probe_seed" not in _kv(C.compose({}))    # contract 3.4: no new key for the default


def test_epochs_set_trainer_agent_and_the_label_schedule():
    c = C.compose({"EPOCHS": "20", "SAVE_EVERY": "5"})
    k = _kv(c)
    assert k["trainer.params.max_epochs"] == "20" and k["agent.config.max_epochs"] == "20"
    assert c.env["MAX_EPOCHS"] == "20" and k["checkpoint.every_n_epochs"] == "5"
    assert "checkpoint.every_n_epochs" not in _kv(C.compose({}))


@pytest.mark.parametrize("n, acc, warm, probe", [("8", "4", "2650", "2650"), ("4", "8", "5300", "5300"), ("2", "16", "10600", "10600")])
def test_counters_follow_the_gpu_count(n, acc, warm, probe):
    c = C.compose({"GPUS": ",".join(map(str, range(int(n))))})
    k = _kv(c)
    assert k["trainer.params.accumulate_grad_batches"] == acc and k["agent.config.grad_balance_warmup_iters"] == warm
    assert k["agent.config.distill_probe_warmup_steps"] == probe
    assert k["agent.config.distill_probe_ramp_steps"] == str(int(probe) // 2)


def test_student_lines():
    k = _kv(C.compose({"INTERACTION": "off", "DET_HEAD": "off", "MAP_HEAD": "off", "UNDISTORT": "off", "HEADING": "free",
                       "FRAMES": "0,1,2,3", "PILLAR_Z": "-2,2.5", "BN": "frozen"}))
    assert k["agent.config.use_task_interaction"] == "false" and k["agent.config.use_det_motion_head"] == "false"
    assert k["agent.config.use_map_head"] == "false" and k["agent.config.map_head_type"] == "vector"
    assert k["agent.config.bev_seg_weight"] == "0.0" and k["agent.config.undistort"] == "false"
    assert k["agent.config.plan_heading_from_xy"] == "false" and k["agent.config.frame_indices"] == "[0,1,2,3]"
    assert k["agent.config.pc_range"] == k["agent.config.map_pc_range"] == "[-32.0,0.0,-2.0,32.0,64.0,2.5]"
    assert k["agent.config.backbone_bn_mode"] == "frozen"


def test_32m_lines():
    c = C.compose({"BEV_RANGE": "32", "REFINER": "off", "BEV_KD": "off"})
    k = _kv(c)
    assert c.env["CK_BEV64"] == "0" and k["agent.config.bev_h"] == "50" and k["agent.config.bev_w"] == "100"
    assert k["agent.config.pc_range"] == "[-32.0,0.0,-1.0,32.0,32.0,3.0]"
    assert C.compose({}).env["CK_BEV64"] == "1"


@pytest.mark.parametrize("bb, arch, flags", [
    ("r50", "resnet50.tv_in1k", {}), ("r50d", "resnet50d.ra2_in1k", {}),
    ("convnext_tiny", "convnext_tiny.fb_in1k", {"frozen_stages": "-1", "backbone_out_indices": "[3]"}),
    ("efficientnet_b0", "efficientnet_b0.ra_in1k", {"frozen_stages": "-1"}),
])
def test_backbone_lines(bb, arch, flags):
    k = _kv(C.compose({"BACKBONE": bb}))
    assert k["agent.config.image_architecture"] == arch
    for f, v in flags.items():
        assert k[f"agent.config.{f}"] == v


def test_launch_environment_of_the_main_recipe():
    c = C.compose({})
    assert c.env["SSR_EXPERIMENT_SHA"] == c.hash6 and c.env["CK2_E2E_APPROVED"] == "1" and c.env["SSR_GPU_POLICY"] == "any"
    assert c.env["CK_BEV64"] == "1" and c.env["ARM"] == "main" and c.env["E2E_ROOT"] == SP.SSR_RUN_ROOT
    assert c.env["SSR_DET_SHA_HEAD"] == "71ea94f672c89521" and c.env["SSR_MAP_SHA256"].startswith("e67cc78d")
    assert "SSR_MAP_REQUIRE_SEG" not in c.env and "E2E_SMOKE" not in c.env


def test_smoke_composition():
    c, m = C.compose({"RUN_ROOT": "/tmp/rr"}, smoke=True), C.compose({"RUN_ROOT": "/tmp/rr"})
    assert c.run == "smoke_" + m.run and c.hash6 == m.hash6 and c.run_dir == f"/tmp/rr/_smoke/{c.run}"
    assert c.env["E2E_SMOKE"] == "1" and c.env["E2E_SMOKE_PREFIX"] == "/tmp/rr/_smoke/"
    assert c.env["MAX_EPOCHS"] == "2" and c.env["LABELER_MAX_EPOCH"] == "1"
    k = _kv(c)
    assert k["trainer.params.limit_train_batches"] == "60" and k["agent.config.distill_probe_warmup_steps"] == "20"
    assert k["agent.config.grad_balance_warmup_iters"] == "20" and k[CK + "onpolicy_from_epoch"] == "1"
    assert k[CK + "kd_calib.min_n"] == "50" and k[CK + "lat_kd.start_mb"] == "30"
    assert k["wandb.enable"] == "false" and c.env["E2E_WANDB"] == "disable" and m.env["E2E_WANDB"] == "offline"
    assert "checkpoint.every_n_epochs" not in _kv(C.compose({"SAVE_EVERY": "5"}, smoke=True))
    k = _kv(C.compose({"REFINER_KD": "off", "BEV_KD": "off"}, smoke=True))
    assert k[CK + "kd_calib.min_n"] == "2000.0" and k[CK + "lat_kd.start_mb"] == "250"
    assert "agent.config.distill_probe_warmup_steps" not in k


# ----------------------------------------------------------------------------------------------- reference diff
@pytest.mark.skipif(not REFERENCE_RUN.is_file(), reason="the reference run record is on this server only")
def test_reference_file_is_the_real_reference_with_paths_normalised():
    roots = {"SSR_DATA_ROOT": "/home/external-user/ssd/yongjae_refiner", "SSR_TEACHER_CACHE": "/workspace/teacher_cache",
             "SSR_PLAN_DATA": "/home/external-user/kyungmin/SSR-v2/data/planning_vb"}
    run_dir = roots["SSR_DATA_ROOT"] + "/v7/e2e/v2_v15_v7_main"
    want = [C.normalize_line(x, roots, run_dir) for x in REFERENCE_RUN.read_text().splitlines()]
    assert want == C.REFERENCE.read_text().splitlines() and len(want) == 153


def test_reference_file_has_no_absolute_path():
    t = C.REFERENCE.read_text()
    assert "/home/" not in t and "/workspace/" not in t


def test_reference_diff_passes_on_the_composed_main_recipe():
    for wandb in ("offline", "disable", "online"):
        c = C.compose({"WANDB": wandb})
        ok, lines = C.reference_diff(c.overrides, c.roots, c.run_dir)
        assert ok, lines
    assert C.compose({}).default_recipe and not C.compose({"GPUS": "0,1,2,3"}).default_recipe
    assert C.compose({"TAG": "a", "SAVE_EVERY": "5"}).default_recipe and not C.compose({"SEED": "1"}).default_recipe


def test_reference_diff_passes_with_other_roots():
    code = ("from tools.release import compose as C; c = C.compose({}); "
            "print(C.reference_diff(c.overrides, c.roots, c.run_dir))")
    r = subprocess.run([sys.executable, "-c", code], env=_env(**OTHER_ROOTS), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.splitlines()[-1].startswith("(True"), r.stdout + r.stderr


@pytest.mark.parametrize("old, new", [
    ("agent.lr=1e-4", "agent.lr=2e-4"), ("trainer.params.accumulate_grad_batches=4", "trainer.params.accumulate_grad_batches=8"),
    ("++agent.config.ck_e2e2.lambda_score=1.0", "++agent.config.ck_e2e2.lambda_score=0.9"),
    ("++agent.config.bev_h=100", "++agent.config.bev_h=50"), ("++agent.config.backbone_bn_mode=moving", "++agent.config.backbone_bn_mode=frozen"),
    ("seed=0", "seed=1"), ("++agent.config.ck_e2e2.kd_calib.mode=online", "++agent.config.ck_e2e2.kd_calib.mode=file"),
    ("++agent.config.ck_e2e2.kd_calib.det=null", "++agent.config.ck_e2e2.kd_calib.det=/x.json"),
    ("++agent.config.ck_e2e2.use_teachers=true", "++agent.config.ck_e2e2.use_teachers=false"),
    ("++agent.config.distill_off_banks=[]", "++agent.config.distill_off_banks=[det]"),
])
def test_reference_diff_catches_a_changed_line(old, new):
    c = C.compose({})
    assert old in c.overrides, old
    ok, lines = C.reference_diff([new if o == old else o for o in c.overrides], c.roots, c.run_dir)
    assert not ok and any(new.split("=", 1)[0] in x for x in lines), lines


def test_reference_diff_catches_new_missing_and_moved_lines():
    c = C.compose({})
    assert not C.reference_diff(c.overrides + ["agent.config.foo=1"], c.roots, c.run_dir)[0]
    assert not C.reference_diff([o for o in c.overrides if not o.startswith("++agent.config.undistort")], c.roots, c.run_dir)[0]
    i = c.overrides.index("agent.lr=1e-4")
    moved = c.overrides[:i] + c.overrides[i + 1:] + ["agent.lr=1e-4"]
    ok, lines = C.reference_diff(moved, c.roots, c.run_dir)
    assert not ok and "order" in lines[-1]


# ----------------------------------------------------------------------------------------------- record
def test_write_composition_and_name_from_settings_json(tmp_path):
    c = C.compose({"BACKBONE": "r50", "DET_KD": "off", "TAG": "t1", "SEED": "2", "GPUS": "0,1,2,3", "RUN_ROOT": str(tmp_path)})
    C.write_composition(c, str(tmp_path / "gen"))
    assert C.main(["record", "--composition", str(tmp_path / "gen/composition.json"), "--out", str(tmp_path / "settings.json"),
                   "--asset-profile", "full", "--asset-exit", "0", "--asset-log", "x.log"]) == 0
    rec = json.loads((tmp_path / "settings.json").read_text())
    assert C.compose(rec["settings"]).run == c.run == rec["run"]
    assert rec["overrides_sha256"] == C._sha(c.overrides) and rec["hash6"] == c.hash6
    for k in ("commit", "git_dirty", "gpus", "hostname", "python", "torch", "teacher_pins", "asset_check", "profile"):
        assert k in rec, k
    assert rec["gpus"] == ["0", "1", "2", "3"] and rec["asset_check"]["exit"] == 0 and len(rec["commit"]) == 40
    env = (tmp_path / "gen/env.sh").read_text()
    assert env.startswith("unset ") and f"export SSR_EXPERIMENT_SHA={c.hash6}" in env and "E2E_CK_SET=" in env
    assert L2.LU.read_lines(str(tmp_path / "gen/ck.set")) == c.set_lines
    assert L2.LU.read_lines(str(tmp_path / "gen/arm.extra")) == c.extra_lines


def test_compose_writes_nothing_and_imports_no_torch(tmp_path):
    code = ("import sys; from tools.release import compose as C; c = C.compose({'RUN_ROOT': %r}); "
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules; print(c.run)" % str(tmp_path / "nope"))
    r = subprocess.run([sys.executable, "-c", code], env=_env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "nope").exists()


def test_registry_entries_need_their_fields_and_the_right_layout(tmp_path):
    import yaml
    reg = yaml.safe_load((REPO / "tools/release/registry.yaml").read_text())
    good = dict(reg["teachers"]["det"]["bevfusion"])
    reg["teachers"]["det"]["no_norm"] = {k: v for k, v in good.items() if k != "norm_v15"}
    reg["teachers"]["det"]["wrong_layout"] = dict(good, layout="sharded")
    reg["teachers"]["det"]["number_sha"] = dict(good, sha_head=1234567890123456)
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(reg))
    with pytest.raises(C.Refused, match="lacks norm_v15"):
        C.compose({"DET_TEACHER": "no_norm"}, registry=str(p))
    with pytest.raises(C.Refused, match="layout sharded"):
        C.compose({"DET_TEACHER": "wrong_layout"}, registry=str(p))
    with pytest.raises(C.Refused, match="not a quoted hex sha"):
        C.compose({"DET_TEACHER": "number_sha"}, registry=str(p))


def test_a_new_teacher_needs_only_a_registry_entry(tmp_path):
    p = _alias_registry(tmp_path)
    c = C.compose({"DET_TEACHER": "bevfusion_alias", "MAP_TEACHER": "resmap_alias"}, registry=str(p))
    k = _kv(c)
    assert c.run.startswith("v2f_r34_bevfusion_alias-resmap_alias_")
    assert k["agent.config.distill_branches.bevfusion.cache_name"] == "bevfusion_alias"
    assert k["agent.config.distill_branches.resmap.cache_name"] == "resmap_alias"
    assert c.env["SSR_DET_SHA_HEAD"] == "71ea94f672c89521" and c.pins["map"]["name"] == "resmap_alias"
    # a teacher nothing reads does not change the run
    b = {"REFINER": "off", "BEV_KD": "off"}
    assert C.compose(dict(b, DET_TEACHER="bevfusion_alias"), registry=str(p)).run == C.compose(b).run


def _alias_registry(tmp_path):
    import yaml
    reg = yaml.safe_load((REPO / "tools/release/registry.yaml").read_text())
    reg["teachers"]["det"]["bevfusion_alias"] = dict(reg["teachers"]["det"]["bevfusion"], cache_name="bevfusion_alias")
    reg["teachers"]["map"]["resmap_alias"] = dict(reg["teachers"]["map"]["resmap"], cache_name="resmap_alias")
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(reg))
    return p


# ----------------------------------------------------------------------------------------------- GPU policy
def test_gpu_policy_any(monkeypatch):
    monkeypatch.delenv("CK_ALLOW_GPU45", raising=False)
    monkeypatch.delenv("CK_ALLOW_GPU4567", raising=False)
    monkeypatch.delenv("SSR_GPU_POLICY", raising=False)
    with pytest.raises(L2.LaunchError):
        L2.parse_gpus2("4,6")
    monkeypatch.setenv("SSR_GPU_POLICY", "any")
    assert L2.parse_gpus2("4,6") == ["4", "6"] and L2.parse_gpus2("9,12,13,14") == ["9", "12", "13", "14"]
    for bad in ("4,4", "a,b", "0,1,2", ""):
        with pytest.raises(L2.LaunchError):
            L2.parse_gpus2(bad)
    # each GPU still must be idle
    with pytest.raises(L2.LaunchError, match="busy"):
        L2.gpu_check2(["4", "6"], used={"4": 0, "6": 5000})
    assert L2.gpu_check2(["4", "6"], used={"4": 0, "6": 10}) == {"4": 0, "6": 10}


def test_gpu_policy_any_in_the_teacher_guards(monkeypatch):
    from tools.ck import ckutil as U
    for k in ("CK_ALLOW_GPU45", "CK_ALLOW_GPU4567", "CK_TEACHER_MAX_GPUS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    monkeypatch.delenv("SSR_GPU_POLICY", raising=False)
    with pytest.raises(SystemExit):
        U.gpu_guard_ddp("cuda", 2)
    monkeypatch.setenv("SSR_GPU_POLICY", "any")
    U.gpu_guard_ddp("cuda", 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,6")
    with pytest.raises(SystemExit):
        U.gpu_guard_ddp("cuda", 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    U.gpu_guard("cuda")


# ----------------------------------------------------------------------------------------------- the script
STUB_E2E = r"""#!/usr/bin/env bash
# Stands in for tools/ck/e2e2/train_e2e2.sh: the real launcher's dry run writes the records, nothing trains.
REAL="$REPO_FOR_STUB/tools/ck/e2e2/train_e2e2.sh"
R=$E2E_ROOT/$RUN
case "$1" in
  start)
    echo "$E2E_ROOT $RUN sha=$SSR_EXPERIMENT_SHA $*" >> "$STUB_LOG"
    env | grep -E '^(CK_BEV64|SSR_DET_|SSR_MAP_|SSR_GPU_POLICY|CK2_E2E_APPROVED|E2E_NO_LABELER|MAX_EPOCHS)' | sort >> "$STUB_LOG"
    [ -z "${STUB_REFUSE:-}" ] || { echo "stub refuses" >&2; exit 3; }
    shift
    args=(); for a in "$@"; do [ "$a" = --resume ] || args+=("$a"); done
    E2E_DRY_RUN=1 bash "$REAL" start "${args[@]}" > /dev/null || exit 3
    mkdir -p "$R/launch"; cp "$R/launch_dryrun/"* "$R/launch/"; rm -rf "$R/launch_dryrun"
    echo 1 > "$R/launch/train.pid";;
  wait) exit 0;;
  stop) echo stopped >> "$STUB_LOG";;
  *) exit 2;;
esac
"""
STUB_ASSETS = """import sys
print("stub verify_assets", sys.argv[1:])
sys.exit(int(__import__("os").environ.get("STUB_ASSETS_RC", "0")))
"""
STUB_EVAL = """#!/usr/bin/env bash
printf '[%s]' "$@"; echo " dirty=${ALLOW_DIRTY:-}"
"""
STUB_SMOKE_CHECK = """import os, sys
print("stub smoke_check", sys.argv[1:])
sys.exit(int(os.environ.get("STUB_SMOKE_RC", "0")))
"""


@pytest.fixture()
def stubs(tmp_path):
    d = tmp_path / "stubs"
    d.mkdir()
    (d / "train_e2e2.sh").write_text(STUB_E2E)
    (d / "verify_assets.py").write_text(STUB_ASSETS)
    (d / "eval.sh").write_text(STUB_EVAL)
    (d / "smoke_check.py").write_text(STUB_SMOKE_CHECK)
    return dict(SSR_RELEASE_TEST_STUBS=str(d), REPO_FOR_STUB=str(REPO), STUB_LOG=str(tmp_path / "stub.log"))


def test_script_name_touches_nothing(tmp_path):
    root = tmp_path / "root"
    r = sh("name", f"RUN_ROOT={root}", "BACKBONE=r50")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == C.compose({"RUN_ROOT": str(root), "BACKBONE": "r50"}).run
    assert not root.exists()


@pytest.mark.parametrize("args, msg", [
    (["name", "NOPE=1"], "unknown key"), (["name", "BACKBONE"], "KEY=VALUE"), (["name", "BACKBONE=swin"], "needs code"),
    (["eval", "GPUS=0,1"], "needs RUN"), (["eval", "RUN=x", "SEED=1"], "takes only"), (["eval", "RUN=../x"], "not a path"),
    (["eval", "RUN=does_not_exist", "RUN_ROOT=/tmp"], "not found"), (["purge"], "needs RUN"), (["purge", "RUN=../../x", "CONFIRM=1"], "not a path"),
    (["table", "SEED=1"], "takes only"), (["resume", "RUN=x", "SEED=1"], "only"), (["status", "RUN=a/b"], "not a path"),
])
def test_script_refusals(args, msg):
    r = sh(*args)
    assert r.returncode == 3 and msg in r.stderr, r.stdout + r.stderr


def test_status_by_run_name_goes_to_the_right_folder(tmp_path):
    _fake_run(tmp_path, "run1")
    (tmp_path / "_smoke").mkdir()
    _fake_run(tmp_path / "_smoke", "smoke_run2")
    r = sh("status", "RUN=run1", f"RUN_ROOT={tmp_path}")
    assert r.returncode == 0 and f"{tmp_path}/run1" in r.stdout and "not running" in r.stdout, r.stdout + r.stderr
    r = sh("status", "RUN=smoke_run2", f"RUN_ROOT={tmp_path}")
    assert r.returncode == 0 and f"{tmp_path}/_smoke/smoke_run2" in r.stdout, r.stdout + r.stderr


def test_resume_needs_a_record_of_this_script(tmp_path):
    _fake_run(tmp_path, "old_run")
    r = sh("resume", "RUN=old_run", f"RUN_ROOT={tmp_path}")
    assert r.returncode == 3 and "settings.json not found" in r.stderr, r.stderr


def test_script_unknown_command_prints_usage():
    r = sh("nonsense")
    assert r.returncode == 2 and "experiment.sh" in r.stdout


def test_eval_passes_exactly_the_four_positions(tmp_path, stubs):
    (tmp_path / "root/runA").mkdir(parents=True)
    root = f"RUN_ROOT={tmp_path / 'root'}"
    r = sh("eval", "RUN=runA", root, "GPUS=0,1", **stubs)
    assert r.returncode == 0 and f"[{tmp_path}/root/runA][0,1][][1]" in r.stdout, r.stdout + r.stderr
    r = sh("eval", "RUN=runA", root, "GPUS=0,1", "EPOCH=15", "EPDMS=0", "ALLOW_DIRTY=1", **stubs)
    assert r.returncode == 0 and "[0,1][15][0] dirty=1" in r.stdout, r.stdout + r.stderr


def _fake_run(root, name, done=True):
    r = Path(root) / name
    for d in ("ck_e2e2/lab/ep000", "ck_e2e2/rec/ep000", "launch", "train"):
        (r / d).mkdir(parents=True)
    (r / "ck_e2e2/steps_rank0.jsonl").write_text("{}\n")
    if done:
        (r / "ck_e2e2/TRAIN_DONE").write_text("")
    return r


def test_purge_removes_only_lab_and_rec_after_train_done(tmp_path):
    root = f"RUN_ROOT={tmp_path}"
    r = _fake_run(tmp_path, "run1")
    assert "CONFIRM=1" in sh("purge", "RUN=run1", root).stderr and (r / "ck_e2e2/lab").exists()
    r2 = _fake_run(tmp_path, "run2", done=False)
    x = sh("purge", "RUN=run2", root, "CONFIRM=1")
    assert x.returncode == 3 and "TRAIN_DONE" in x.stderr and (r2 / "ck_e2e2/rec").exists()
    x = sh("purge", "RUN=run1", root, "CONFIRM=1")
    assert x.returncode == 0, x.stderr
    assert not (r / "ck_e2e2/lab").exists() and not (r / "ck_e2e2/rec").exists()
    assert (r / "ck_e2e2/steps_rank0.jsonl").exists() and (r / "train").exists() and (r / "ck_e2e2/TRAIN_DONE").exists()


def test_purge_refuses_a_live_training(tmp_path):
    r = _fake_run(tmp_path, "run1")
    p = subprocess.Popen(["bash", "-c", "exec -a run_training sleep 60"])
    try:
        (r / "launch/train.pid").write_text(str(p.pid))
        import time
        time.sleep(0.3)
        x = sh("purge", "RUN=run1", f"RUN_ROOT={tmp_path}", "CONFIRM=1")
        assert x.returncode == 3 and "still running" in x.stderr, x.stderr
        assert (r / "ck_e2e2/lab").exists()
    finally:
        p.kill()
        p.wait()


def test_train_guards_stop_before_anything_is_written(tmp_path, stubs):
    root = tmp_path / "root"
    root.mkdir()
    base = [f"RUN_ROOT={root}", "ALLOW_DIRTY=1"]
    n = C.compose({"RUN_ROOT": str(root)}).run
    # disk
    r = sh("train", *base, "MIN_FREE_GB=100000000", **stubs)
    assert r.returncode == 3 and "MIN_FREE_GB" in r.stderr, r.stderr
    # a second training
    (root / "other/launch").mkdir(parents=True)
    p = subprocess.Popen(["bash", "-c", "exec -a run_training sleep 60"])
    try:
        import time
        time.sleep(0.3)
        (root / "other/launch/train.pid").write_text(str(p.pid))
        r = sh("train", *base, "MIN_FREE_GB=0", **stubs)
        assert r.returncode == 3 and "already running" in r.stderr, r.stderr
    finally:
        p.kill()
        p.wait()
    # the run exists
    (root / n).mkdir()
    r = sh("train", *base, "MIN_FREE_GB=0", **stubs)
    assert r.returncode == 3 and f"resume RUN={n}" in r.stderr, r.stderr
    assert not (root / "_check").exists() and sorted(x.name for x in root.iterdir()) == sorted(["other", n])
    assert not Path(stubs["STUB_LOG"]).exists()


def test_train_refuses_a_dirty_tree():
    if not subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
        pytest.skip("the tree is clean")
    r = sh("train", "RUN_ROOT=/tmp/ssr_release_test_never_created")
    assert r.returncode == 3 and "ALLOW_DIRTY" in r.stderr, r.stderr
    assert not Path("/tmp/ssr_release_test_never_created").exists()


@server_only
def test_train_flow_with_a_stub_launcher(tmp_path, stubs):
    root = tmp_path / "root"
    a = [f"RUN_ROOT={root}", "ALLOW_DIRTY=1", "MIN_FREE_GB=0", f"CHECK_ROOT={tmp_path / 'c'}"]
    n = C.compose({"RUN_ROOT": str(root)}).run
    R = root / n
    # DRY=1 launches nothing and leaves no run folder
    r = sh("train", *a, "DRY=1", **stubs)
    assert r.returncode == 0 and "would run" in r.stdout and not R.exists(), r.stdout + r.stderr
    assert "reference diff (tools/release/reference/main_overrides.txt): PASS" in r.stdout
    assert not Path(stubs["STUB_LOG"]).exists()
    # an asset failure stops it before the run folder exists
    r = sh("train", *a, **stubs, STUB_ASSETS_RC="1")
    assert r.returncode == 3 and "not ready" in r.stderr and not R.exists(), r.stdout + r.stderr
    # the launcher refuses: no run folder stays behind
    r = sh("train", *a, **stubs, STUB_REFUSE="1")
    assert r.returncode == 3 and "refused the start" in r.stderr and not R.exists(), r.stdout + r.stderr
    Path(stubs["STUB_LOG"]).unlink()
    # a start
    r = sh("train", *a, **stubs)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads((R / "launch/settings.json").read_text())
    assert rec["run"] == n and C.compose(rec["settings"]).run == n and rec["hostname"] and rec["torch"]
    assert rec["asset_check"] == {"profile": "full", "quick": True, "exit": 0, "log": "launch/verify_assets.log"}
    assert (R / "launch/verify_assets.log").is_file() and (R / "launch/gen/ck.set").is_file()
    assert rec["overrides_sha256"] == C._sha(C.LU.read_lines(str(R / "launch/overrides.txt")))
    log = Path(stubs["STUB_LOG"]).read_text()
    assert f"{root} {n} sha={rec['hash6']} start --gpus 0,1,2,3,4,5,6,7" in log and "SSR_GPU_POLICY=any" in log
    assert "CK_BEV64=1" in log and "SSR_DET_SHA_HEAD=71ea94f672c89521" in log and "E2E_NO_LABELER" not in log
    if rec["git_dirty"]:
        assert (R / "launch/git_dirty.diff").is_file() and (R / "launch/git_status.txt").is_file()
    # a second start refuses; status by RUN goes to the launcher
    r = sh("train", *a, **stubs)
    assert r.returncode == 3 and "resume" in r.stderr
    # resume: same settings by name; another GPU count or commit refuses
    ra = [f"RUN={n}", f"RUN_ROOT={root}", "ALLOW_DIRTY=1", "MIN_FREE_GB=0"]
    r = sh("resume", *ra, "GPUS=0,1,2,3", **stubs)
    assert r.returncode == 3 and "this code composes" in r.stderr, r.stdout + r.stderr
    old = (R / "launch/settings.json").read_text()
    (R / "launch/settings.json").write_text(old.replace(rec["commit"], "0" * 40))
    r = sh("resume", *ra, **stubs)
    assert r.returncode == 3 and "ALLOW_COMMIT_CHANGE" in r.stderr, r.stdout + r.stderr
    (R / "launch/settings.json").write_text(old)
    r = sh("resume", *ra, "GPUS=7,6,5,4,3,2,1,0", **stubs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "start --resume --gpus 7,6,5,4,3,2,1,0" in Path(stubs["STUB_LOG"]).read_text()
    assert list((R / "launch").glob("overrides.*.txt")) and list((R / "launch").glob("settings.*.json"))


@server_only
def test_smoke_flow_with_a_stub_launcher(tmp_path, stubs):
    root = tmp_path / "root"
    n = "smoke_" + C.compose({"RUN_ROOT": str(root), "REFINER_KD": "off", "BEV_KD": "off"}).run
    R = root / "_smoke" / n
    args = [f"RUN_ROOT={root}", "REFINER_KD=off", "BEV_KD=off"]
    # a failed check keeps everything and exits non-zero
    r = sh("smoke", *args, **stubs, STUB_SMOKE_RC="1")
    assert r.returncode == 3 and "smoke failed" in r.stderr, r.stdout + r.stderr
    rec = json.loads((R / "launch/settings.json").read_text())
    assert rec["smoke"] is True and rec["run"] == n
    log = Path(stubs["STUB_LOG"]).read_text()
    assert f"{root}/_smoke {n}" in log and "MAX_EPOCHS=2" in log and "CK_BEV64=1" in log
    # one smoke per recipe unless tagged
    r = sh("smoke", *args, **stubs)
    assert r.returncode == 3 and "exists" in r.stderr
    # SMOKE_KEEP=0 removes this smoke's checkpoints, rec, lab after a pass and keeps the records
    tag = args + ["TAG=b", "SMOKE_KEEP=0"]
    n2 = "smoke_" + C.compose({"RUN_ROOT": str(root), "REFINER_KD": "off", "BEV_KD": "off", "TAG": "b"}).run
    R = root / "_smoke" / n2
    stub_e2e = Path(stubs["SSR_RELEASE_TEST_STUBS"]) / "train_e2e2.sh"
    stub_e2e.write_text(stub_e2e.read_text().replace("  wait) exit 0;;", '  wait) mkdir -p "$R/ck_e2e2/rec" "$R/ck_e2e2/lab" '
                                                     '"$R/train/lightning_logs/version_0/checkpoints"; exit 0;;'))
    r = sh("smoke", *tag, **stubs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (R / "ck_e2e2/rec").exists() and not (R / "ck_e2e2/lab").exists()
    assert not (R / "train/lightning_logs/version_0/checkpoints").exists()
    assert (R / "launch/settings.json").is_file() and (R / "launch/overrides.txt").is_file()


@server_only
def test_smoke_dry_checks_the_smoke_overrides_and_launches_nothing(tmp_path):
    r = sh("smoke", f"RUN_ROOT={tmp_path / 'root'}", f"CHECK_ROOT={tmp_path / 'c'}", "DRY=1", "REFINER_KD=off", "BEV_KD=off")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "nothing launched" in r.stdout and not (tmp_path / "root").exists()
    assert "use_teachers=false" in r.stdout and "limit_train_batches=60" in r.stdout


@server_only
@pytest.mark.parametrize("kw, want", [
    ([], {"reference": "PASS", "builders": "CK2E2ETargetBuilder", "bev": True, "ref": True}),
    (["REFINER=off", "BEV_KD=off"], {"builders": "AnchorScoreTargetBuilder", "bev": False, "ref": False}),
    (["DET_KD=off", "REFINER_KD=off"], {"builders": "DistillTeacherTargetBuilder", "bev": True, "ref": True}),
])
def test_real_check(tmp_path, kw, want):
    r = sh("check", f"CHECK_ROOT={tmp_path}", *kw)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "launcher overrides == composition" in r.stdout and "refusals: none" in r.stdout
    assert want["builders"] in r.stdout
    if "reference" in want:
        assert f"reference diff (tools/release/reference/main_overrides.txt): {want['reference']}" in r.stdout
    else:
        assert "reference diff" not in r.stdout
    d = next((tmp_path / "_check").iterdir())
    rec = json.loads((d / "gen/build.json").read_text())
    assert (rec["bev_kd"] is not None) == want["bev"] and (rec["refiner"] is not None) == want["ref"]


@server_only
def test_real_check_refuses_a_wrong_teacher_pin(tmp_path):
    txt = (REPO / "tools/release/registry.yaml").read_text()
    new = tmp_path / "r.yaml"
    new.write_text(txt.replace('refiner_ckpt_sha16: "66c6a413dd866898"', 'refiner_ckpt_sha16: "ffffffffffffffff"'))
    r = sh("check", f"CHECK_ROOT={tmp_path}", f"REGISTRY={new}")
    assert r.returncode == 3 and "!= registry ffffffffffffffff" in r.stdout, r.stdout + r.stderr


@server_only
def test_real_check_of_a_new_teacher_name_reads_the_same_bytes(tmp_path):
    """The new name goes through the launcher and the real agent build; both readers give what the real name gives."""
    reg = _alias_registry(tmp_path)
    shas = {}
    for label, reg_kw, set_kw in (("real", [], []),
                                  ("alias", [f"REGISTRY={reg}"], ["DET_TEACHER=bevfusion_alias", "MAP_TEACHER=resmap_alias"])):
        root = tmp_path / label
        r = sh("check", f"CHECK_ROOT={root}", *reg_kw, *set_kw)
        assert r.returncode == 0, r.stdout + r.stderr
        d = next((root / "_check").iterdir())
        rec = json.loads((d / "gen/build.json").read_text())
        if label == "alias":
            assert rec["bev_kd"]["stores"]["bevfusion"]["cache"] == "bevfusion_alias"
        env = {}
        for ln in (d / "gen/env.sh").read_text().splitlines():
            if ln.startswith("export "):
                k, v = ln[len("export "):].split("=", 1)
                env[k] = v.strip("'")
        x = subprocess.run([sys.executable, str(REPO / "tools/release/check_build.py"), "--overrides",
                            str(d / "launch_dryrun/overrides.txt"), "--out", str(d / "gen/readers.json"), "--reader-tokens", "2"],
                           env=_env(**{k: env[k] for k in env if k.startswith(("SSR_DET", "SSR_MAP", "CK_BEV64"))}),
                           capture_output=True, text=True, timeout=600)
        assert x.returncode == 0, x.stdout + x.stderr
        shas[label] = json.loads((d / "gen/readers.json").read_text())["reader_sha16"]
    assert shas["real"] == shas["alias"] and set(shas["real"]) == {"det", "map"}


# ----------------------------------------------------------------------------------------------- smoke check
def _settings(run, **kw):
    s = C.canonicalize(kw)
    (run / "launch").mkdir(parents=True, exist_ok=True)
    (run / "launch/settings.json").write_text(json.dumps({"run": run.name, "settings": s}))
    return s


def _common_files(run, ckpt_keys):
    import torch
    (run / "launch/train.exit").write_text("0\n")
    (run / "train/lightning_logs/version_0/checkpoints").mkdir(parents=True)
    (run / "train/train_tokens.json").write_text(json.dumps({"train_token_sha256": "x", "count": 480}))
    (run / "train/train_time.json").write_text(json.dumps({"sessions": [{"status": "completed", "epochs_completed": 2}]}))
    ck = run / "train/lightning_logs/version_0/checkpoints"
    torch.save({"state_dict": {k: torch.zeros(1) for k in ckpt_keys}}, ck / "last.ckpt")
    (ck / "epoch=1-step=30.ckpt").write_text("")


def _refiner_files(run, teachers, mem=15.0, kd=0.3, lat=0.2, teacher_ms=80.0, nan=False):
    io = run / "ck_e2e2"
    (io / "lab").mkdir(parents=True)
    ep = [{"event": "epoch_start", "epoch": 0, "phase": "warmup_record"}, {"event": "epoch_end", "epoch": 0, "cum": {}},
          {"event": "epoch_start", "epoch": 1, "phase": "onpolicy"},
          {"event": "epoch_end", "epoch": 1, "cum": {"nonfinite_loss": 0, "sur_nonfinite": 0}}]
    (io / "epochs.jsonl").write_text("\n".join(json.dumps(e) for e in ep) + "\n")
    rows = [{"epoch": 0, "mem_gb": mem, "skipped_steps": 0, "ck2/kd_ok_frac": 1.0, "ck2/wt_kd_score": kd, "ck2/wt_lat_kd": lat,
             "ck2/teacher_ms": teacher_ms, "ck2/cal_det_nc_a": 1.0}]
    if nan:
        rows[0]["ck2/loss"] = float("nan")
    (io / "steps_rank0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (io / "lab/status.json").write_text(json.dumps({"tokens_ok": 10, "tokens_err": 0, "epochs": {"0": {"done": True, "rows_done": 10}}}))
    (run / "launch/teachers.json").write_text(json.dumps(teachers))


def _calib_file(run, keys):
    ok = {k: {"fitted": True, "a": 1.0, "b": 0.0} for k in keys}
    lines = [{"window_phase": "warmup_record", "keys": ok} for _ in range(5)] + [{"window_phase": "onpolicy"}]
    (run / "ck_e2e2/kd_calib_online.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


T_ON = {"det": {}, "map": {}, "kd_calib": {"enabled": True, "mode": "online"}}
ALL_KEYS = ["det/nc", "det/ep", "det/ttc", "map/dac", "map/ep"]
MAIN_KEYS = ["agent.para_ssr_model.x", "agent.ck_student.x", "agent._distill.x"]


def test_smoke_check_main_passes_and_catches_each_failure(tmp_path):
    run = tmp_path / "r"
    _settings(run)
    _common_files(run, MAIN_KEYS)
    _refiner_files(run, T_ON)
    _calib_file(run, ALL_KEYS)
    res = SC.check_run(str(run))
    assert res["pass"], res
    assert res["info"]["peak_mem_torch_gb"] == 15.0
    assert not SC.check_run(str(run), mem_limit_gb=10.0)["pass"]
    # non-finite step value
    good = (run / "ck_e2e2/steps_rank0.jsonl").read_text()
    (run / "ck_e2e2/steps_rank0.jsonl").write_text(good.replace('"mem_gb": 15.0', '"ck2/loss": NaN, "mem_gb": 15.0'))
    assert not SC.check_run(str(run))["checks"]["no_nonfinite_step_value"]
    (run / "ck_e2e2/steps_rank0.jsonl").write_text(good)
    # exit code
    (run / "launch/train.exit").write_text("1\n")
    assert not SC.check_run(str(run))["checks"]["train_exit_0"]
    (run / "launch/train.exit").write_text("0\n")
    # calibration: a missing key, a teacher pair not online
    _calib_file(run, ALL_KEYS[:-1])
    assert not SC.check_run(str(run))["checks"]["kd_calib_every_key_fitted"]
    _calib_file(run, ALL_KEYS)
    (run / "launch/teachers.json").write_text(json.dumps(dict(T_ON, kd_calib={"enabled": True, "mode": "file"})))
    assert not SC.check_run(str(run))["checks"]["kd_calib_mode_online"]
    assert SC.main(["--run-dir", str(run)]) == 1 and json.loads((run / "launch/smoke_check.json").read_text())["pass"] is False


def test_smoke_check_one_teacher_checks_its_keys_only(tmp_path):
    run = tmp_path / "r"
    _settings(run, DET_KD="off_ref")
    _common_files(run, MAIN_KEYS)
    _refiner_files(run, T_ON)
    _calib_file(run, ["det/ep", "map/dac", "map/ep"])
    res = SC.check_run(str(run))
    assert res["pass"], res
    assert res["info"]["kd_calib_keys"] == ["det/ep", "map/dac", "map/ep"]


@pytest.mark.parametrize("kw, teachers, need, ms, kd, lat, keys", [
    ({"REFINER_KD": "off", "BEV_KD": "off"}, {"mode": "none", "kd_calib": {"enabled": False}}, ["agent.ck_student.x"], 0.1, 0.0, 0.0, None),
    ({"REFINER_KD": "lat"}, {"det": {}, "map": {}, "kd_calib": {"enabled": False}}, MAIN_KEYS, 80.0, 0.0, 0.2, None),
    ({"REFINER_KD": "score"}, T_ON, MAIN_KEYS, 80.0, 0.3, 0.0, ALL_KEYS),
])
def test_smoke_check_follows_the_refiner_kd_setting(tmp_path, kw, teachers, need, ms, kd, lat, keys):
    run = tmp_path / "r"
    _settings(run, **kw)
    _common_files(run, need)
    _refiner_files(run, teachers, kd=kd, lat=lat, teacher_ms=ms)
    if keys:
        _calib_file(run, keys)
    res = SC.check_run(str(run))
    assert res["pass"], res
    # KD on where the settings say off
    steps = run / "ck_e2e2/steps_rank0.jsonl"
    txt = steps.read_text()
    steps.write_text(txt.replace(f'"ck2/wt_kd_score": {kd}', '"ck2/wt_kd_score": 0.7').replace(
        f'"ck2/wt_lat_kd": {lat}', '"ck2/wt_lat_kd": 0.7'))
    checks = SC.check_run(str(run))["checks"]
    if kd == 0.0:
        assert not checks["score_kd_as_set"]
    if lat == 0.0:
        assert not checks["lateral_kd_as_set"]


def test_smoke_check_refiner_off_uses_the_sampled_memory(tmp_path):
    run = tmp_path / "r"
    _settings(run, REFINER="off", BEV_KD="off")
    _common_files(run, ["agent.para_ssr_model.x"])
    res = SC.check_run(str(run))
    assert not res["checks"]["peak_memory_below_limit"]      # no memory record at all is a failure, not a pass
    (run / "launch/gpu_mem.csv").write_text("0, 9000\n1, 12288\n")
    res = SC.check_run(str(run))
    assert res["pass"], res
    assert res["info"]["peak_mem_nvidia_smi_gb"] == 12.0 and res["info"]["peak_mem_torch_gb"] is None
    assert not SC.check_run(str(run), mem_limit_gb=11.0)["pass"]


def test_smoke_check_checkpoint_must_match_the_settings(tmp_path):
    run = tmp_path / "r"
    _settings(run, REFINER="off", BEV_KD="off")
    _common_files(run, ["agent.para_ssr_model.x", "agent.ck_student.x"])
    (run / "launch/gpu_mem.csv").write_text("0, 9000\n")
    res = SC.check_run(str(run))
    assert not res["checks"]["ckpt_refiner_as_set"]


REAL_SMOKES = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e_smoke")


@pytest.mark.parametrize("name_, kw, may_fail", [
    # older code: no train_tokens.json yet; the second one ran file calibration
    ("v2nokd_smoke_1103_0459", {"REFINER_KD": "off", "BEV_KD": "off"}, {"train_tokens_recorded"}),
    ("v7smoke8_1102_0850", {}, {"train_tokens_recorded", "kd_calib_mode_online", "kd_calib_ge_5_warmup_refits",
                                "kd_calib_every_key_fitted", "kd_calib_onpolicy_window", "kd_calib_in_step_log"}),
])
def test_smoke_check_on_real_smoke_records(tmp_path, name_, kw, may_fail):
    real = REAL_SMOKES / name_
    if not (real / "ck_e2e2/epochs.jsonl").is_file():
        pytest.skip("that smoke record is not on this server")
    run = tmp_path / name_
    (run / "launch").mkdir(parents=True)
    for f in ("train.exit", "teachers.json"):
        (run / "launch" / f).write_bytes((real / "launch" / f).read_bytes())
    os.symlink(real / "ck_e2e2", run / "ck_e2e2")
    os.symlink(real / "train", run / "train")
    _settings(run, **kw)
    res = SC.check_run(str(run))
    assert {k for k, v in res["checks"].items() if not v} == may_fail, res


def test_smoke_check_bev_kd_banks_in_the_event_file(tmp_path, monkeypatch):
    run = tmp_path / "r"
    _settings(run, REFINER="off", DET_KD="off")
    _common_files(run, ["agent.para_ssr_model.x", "agent._distill.x"])
    (run / "launch/gpu_mem.csv").write_text("0, 9000\n")
    monkeypatch.setattr(SC, "_events", lambda r: {"train/loss_distill_resmap_epoch": [1.0, 0.9]})
    assert SC.check_run(str(run))["pass"]
    monkeypatch.setattr(SC, "_events", lambda r: {"train/loss_distill_resmap_epoch": [1.0], "train/loss_distill_bevfusion_epoch": [1.0]})
    assert not SC.check_run(str(run))["checks"]["bev_kd_det_as_set"]
    monkeypatch.setattr(SC, "_events", lambda r: {"train/loss_distill_resmap_epoch": [float("nan")]})
    assert not SC.check_run(str(run))["checks"]["no_nonfinite_event"]


# ----------------------------------------------------------------------------------------------- table
def _summary(root, run, ep=30, g=8, commit="c" * 40, proto="release-1", rule="last", pdms=90.0, delta=(0.5, 0.2, 0.8), selection="refiner_pick",
             scores=None, cache=None):
    d = Path(root) / run / "eval" / f"ep{ep:02d}_aaaa_bbbb"
    d.mkdir(parents=True)
    s = {"run": run, "ep": ep, "epoch_rule": rule, "protocol_version": proto, "git_sha": commit, "git_dirty": False, "train_gpus": g,
         "train_commit": commit, "selection": selection, "pdms_v11": pdms,
         "per_metric": {"nc": 98.0, "dac": 97.0, "ep": 80.0, "ttc": 96.0, "comfort": 99.0},
         "v2head": {"pdms_v11": pdms - 1.0} if delta else None,
         "delta_vs_v2head": dict(zip(("mean", "lo", "hi"), delta)) if delta else None,
         "epdms": {"post": {"epdms": 89.0}, "pre": {"epdms": 86.0}}}
    (d / "summary.json").write_text(json.dumps(s))
    if scores is not None:
        lines = [",token,valid,score"] + [f"{i},t{i},True,{v}" for i, v in enumerate(scores)] + [f"{len(scores)},average,True,0.5"]
        (d / "pdms_v11.csv").write_text("\n".join(lines) + "\n")
        (d / "preflight.json").write_text(json.dumps({"scorer": {"metric_cache_path": str(cache)}}))
    return d


@needs_table
def test_table_prints_rows_marks_manual_and_warns(tmp_path, capsys):
    _summary(tmp_path, "v2f_r50_bevfusion-resmap_ref1_rkd1_bkd1_det1_map1_stbase_bnm_e30_g8_s0_aaaaaa")
    _summary(tmp_path, "v2f_r34_bevfusion-resmap_ref1_rkd1_bkd0_det1_map1_stbase_bnm_e30_g8_s0_bbbbbb", ep=15, rule="manual",
             delta=(0.1, -0.2, 0.4))
    _summary(tmp_path, "v2f_r34_bevfusion-resmap_ref0_rkd0_bkd0_det0_map0_stbase_bnm_e30_g8_s0_cccccc", selection="none", delta=None)
    assert TB.main(["--run-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "| Run | ep |" in out and "| 15* | refiner_pick |" in out and "chosen by hand" in out
    assert "| 30 | none | 90.00 |" in out and "+0.50 [+0.20, +0.80]" in out
    assert "vs v2 head +0.10 [-0.20, +0.40]: interval contains 0" in out and out.count("WARNING") >= 1
    assert "vs base" not in out     # no run of the default recipe here


@pytest.mark.parametrize("kw", [{"g": 4}, {"commit": "d" * 40}, {"proto": "release-2"}])
@needs_table
def test_table_refuses_mixed_rows_unless_allowed(tmp_path, capsys, monkeypatch, kw):
    _summary(tmp_path, "run_a")
    _summary(tmp_path, "run_b", **kw)
    monkeypatch.delenv("ALLOW_MIXED", raising=False)
    assert TB.main(["--run-root", str(tmp_path)]) == 3
    assert "not comparable" in capsys.readouterr().err
    monkeypatch.setenv("ALLOW_MIXED", "1")
    assert TB.main(["--run-root", str(tmp_path)]) == 0
    assert "MIXED" in capsys.readouterr().out


@needs_table
def test_table_paired_delta_against_the_base_run(tmp_path, capsys):
    cache = tmp_path / "cache"
    (cache / "metadata").mkdir(parents=True)
    n = 40
    (cache / "metadata/x.csv").write_text("file_name\n" + "\n".join(f"{cache}/log{i % 8}/type/t{i}/metric_cache.pkl" for i in range(n)) + "\n")
    root = tmp_path / "runs"
    base = "v2f_r34_bevfusion-resmap_ref1_rkd1_bkd1_det1_map1_stbase_bnm_e30_g8_s0_aaaaaa"
    sc = [0.5 + 0.01 * (i % 7) for i in range(n)]
    _summary(root, base, scores=sc, cache=cache)
    _summary(root, "better", scores=[v + 0.02 for v in sc], cache=cache)
    _summary(root, "same", scores=[v + (0.01 if i % 2 else -0.01) for i, v in enumerate(sc)], cache=cache)
    assert TB.main(["--run-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "vs base [95% CI]" in out and "| base |" in out and "+2.00 [+2.00, +2.00]" in out
    assert "same ep30: vs base" in out and "interval contains 0" in out and "better ep30: vs base" not in out


@needs_table
def test_table_with_nothing_to_show(tmp_path, capsys):
    assert TB.main(["--run-root", str(tmp_path)]) == 1
