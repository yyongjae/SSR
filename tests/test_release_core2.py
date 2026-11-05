"""Release contract section 3: paths in one place, vendored scorer + metric-cache config, teacher hooks, distill_norm_files.
All CPU.  Run:
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_release_core2.py
Tests that need this server's data skip elsewhere (real caches, metric cache, the research tree's rescore_attr.py).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from navsim.common import ssr_paths as SP  # noqa: E402
from navsim.agents.para_ssr.distill.anchor_distill import TrajectoryAnchorDistillation as AD  # noqa: E402
from navsim.agents.para_ssr.distill.teacher_store import (DistillTeacherTargetBuilder, TeacherCacheMismatch,  # noqa: E402
                                                          TeacherFeatureStore, role_store_kwargs)

PY = sys.executable
TRAIN_SH = REPO / "tools/ck/e2e2/train_e2e2.sh"
DATASET_LITERAL = "/home/external-user/yongjae/SSR/data/dataset"
SERVER_DATASET = (Path(DATASET_LITERAL) / "navsim_logs/trainval").is_dir()
TC = Path("/home/external-user/datasets/teacher_cache")
REAL_CACHES = (TC / "bevfusion/cache_train_100x100/manifest.json").is_file() and (TC / "resmap/stage1_front64/index.json").is_file()
LIVE_RESCORE = Path("/home/external-user/yongjae/SSR/report/perception_reliability/pdm_attr/rescore_attr.py")
MC_TRAIN = Path("/home/external-user/ssd/yongjae_refiner/metric_cache")
MC_TEST = Path("/home/external-user/yongjae/SSR/data/exp/metric_cache")
ENV_NAMES = tuple(SP.NAMES) + ("SSR_DET_CACHE_TRAIN", "SSR_DET_CACHE_VAL", "SSR_DET_SHA_HEAD", "SSR_MAP_CACHE_ROOT",
                               "SSR_MAP_SHA256", "SSR_MAP_REQUIRE_SEG", "CK_BEV64", "E2E_SMOKE_PREFIX")


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in ENV_NAMES}
    env.update(PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    env.update(extra)
    return env


def _py(code, **extra):
    r = subprocess.run([PY, "-c", code], env=_env(**extra), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return r


def _dataset(tmp_path, name="ds"):
    (tmp_path / name / "navsim_logs/trainval").mkdir(parents=True)
    (tmp_path / name / "navsim_logs/trainval/a.pkl").write_bytes(b"")
    return str(tmp_path / name)


# ----------------------------------------------------------------------------------------------- ssr_paths
TODAY = {
    "SSR_DATA_ROOT": "/home/external-user/ssd/yongjae_refiner",
    "SSR_TEACHER_CACHE": "/home/external-user/datasets/teacher_cache",
    "OPENSCENE_DATA_ROOT": DATASET_LITERAL,
    "NUPLAN_MAPS_ROOT": DATASET_LITERAL + "/maps",
    "SSR_TEST_DATA_ROOT": "/workspace/navsim_workspace/dataset",
    "SSR_PLAN_DATA": "/home/external-user/kyungmin/SSR-v2/data/planning_vb",
    "SSR_NAVTEST_MC": "/home/external-user/yongjae/SSR/data/exp/metric_cache",
    "SSR_PY": "/venv/ssr/bin/python",
    "SSR_TORCHRUN": "/venv/ssr/bin/torchrun",
    "SSR_RUN_ROOT": "/home/external-user/ssd/yongjae_refiner/v7/e2e",
    "NAVSIM_V11": "/home/external-user/ssd/yongjae_refiner/v7/eval/official_v11/navsim_v1.1",
    "NAVSIM_V11_CACHE": "/home/external-user/ssd/yongjae_refiner/v7/eval/official_v11/exp/metric_cache_navtest_v11",
    "NAVSIM_V22_POST": "/workspace/navsim_workspace/navsim",
    "NAVSIM_V22_PRE": "/workspace/yongjae/perception_eval/epdms/navsim_prefix",
    "NAVSIM_V22_CACHE": "/workspace/yongjae/perception_eval/epdms/metric_cache_v22_navtest",
    "NAVSIM_V22_NAVHARD_CACHE": "/workspace/navsim_workspace/exp/metric_cache_navhard_two_stage_v22",
    "SSR_RESEARCH_TREE": "/home/external-user/yongjae/SSR",
}


def test_no_env_gives_this_servers_paths():
    assert SP.resolve({}) == TODAY
    assert set(SP.NAMES) == set(TODAY)


def test_env_wins_derived_defaults_follow_and_empty_counts_as_unset(tmp_path):
    ds = _dataset(tmp_path)
    r = SP.resolve({"SSR_DATA_ROOT": "/d", "OPENSCENE_DATA_ROOT": ds, "SSR_PY": "", "SSR_TEACHER_CACHE": "/t"})
    assert r["SSR_RUN_ROOT"] == "/d/v7/e2e" and r["NAVSIM_V11"] == "/d/v7/eval/official_v11/navsim_v1.1"
    assert r["NAVSIM_V11_CACHE"] == "/d/v7/eval/official_v11/exp/metric_cache_navtest_v11"
    assert r["NUPLAN_MAPS_ROOT"] == ds + "/maps" and r["OPENSCENE_DATA_ROOT"] == ds
    assert r["SSR_PY"] == TODAY["SSR_PY"] and r["SSR_TEACHER_CACHE"] == "/t"
    assert SP.resolve({"SSR_RUN_ROOT": "/r", "NUPLAN_MAPS_ROOT": "/m"})["SSR_RUN_ROOT"] == "/r"
    assert SP.resolve({"NUPLAN_MAPS_ROOT": "/m"})["NUPLAN_MAPS_ROOT"] == "/m"


def test_dataset_variable_with_the_wrong_layout_is_not_used_when_the_default_is_good(tmp_path, monkeypatch, capsys):
    good = _dataset(tmp_path, "good")
    (tmp_path / "nested/navsim_logs/trainval/trainval").mkdir(parents=True)
    (tmp_path / "nested/navsim_logs/trainval/trainval/a.pkl").write_bytes(b"")
    nested = str(tmp_path / "nested")
    monkeypatch.setattr(SP, "_DATASET_DEFAULT", good)
    # the shell of another checkout: both variables are left out, and the reason is printed
    r = SP.resolve({"OPENSCENE_DATA_ROOT": nested, "NUPLAN_MAPS_ROOT": nested + "/maps"})
    assert r["OPENSCENE_DATA_ROOT"] == good and r["NUPLAN_MAPS_ROOT"] == good + "/maps"
    assert "no navsim_logs/trainval/*.pkl" in capsys.readouterr().err
    # maps elsewhere stay; a good variable stays
    assert SP.resolve({"OPENSCENE_DATA_ROOT": nested, "NUPLAN_MAPS_ROOT": "/m"})["NUPLAN_MAPS_ROOT"] == "/m"
    other = _dataset(tmp_path, "other")
    assert SP.resolve({"OPENSCENE_DATA_ROOT": other})["OPENSCENE_DATA_ROOT"] == other
    # no good default (a new server): the variable is what the user asked for, the first scene load reports it
    monkeypatch.setattr(SP, "_DATASET_DEFAULT", str(tmp_path / "absent"))
    assert SP.resolve({"OPENSCENE_DATA_ROOT": nested})["OPENSCENE_DATA_ROOT"] == nested


def test_ssr_paths_imports_nothing_heavy_and_prints_shell_lines():
    r = _py("import sys; from navsim.common import ssr_paths; "
            "print(sorted(m for m in ('torch', 'numpy', 'hydra', 'navsim.agents') if m in sys.modules))")
    assert r.stdout.strip() == "[]"
    env = {"PATH": os.environ["PATH"], "SSR_DATA_ROOT": "/a b/c"}
    out = subprocess.run([PY, "-I", str(REPO / "navsim/common/ssr_paths.py"), "--shell"], env=env, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    assert [ln.split("=")[0] for ln in out] == [f"export {k}" for k in SP.NAMES]
    assert "export SSR_DATA_ROOT='/a b/c'" in out and "export SSR_RUN_ROOT='/a b/c/v7/e2e'" in out


def test_load_paths_sh_reads_paths_env_and_stops_without_python(tmp_path):
    for rel in ("tools/release/load_paths.sh", "navsim/common/ssr_paths.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text((REPO / rel).read_text())
    prog = f'REPO={tmp_path}; source "$REPO/tools/release/load_paths.sh"; echo "$SSR_RUN_ROOT|$SSR_PY"; echo after'
    r = subprocess.run(["bash", "-c", prog], env=_env(), capture_output=True, text=True)
    assert r.stdout.splitlines() == [TODAY["SSR_RUN_ROOT"] + "|" + TODAY["SSR_PY"], "after"]
    (tmp_path / "tools/release/paths.env").write_text("SSR_DATA_ROOT=/from/file\n")
    r = subprocess.run(["bash", "-c", prog], env=_env(), capture_output=True, text=True)
    assert r.stdout.splitlines()[0].startswith("/from/file/v7/e2e|")
    r = subprocess.run(["bash", "-c", prog], env=_env(SSR_PY="/nonexistent/python"), capture_output=True, text=True)
    assert r.returncode == 3 and "after" not in r.stdout and "load_paths" in r.stderr


def test_example_file_names_every_variable():
    txt = (REPO / "tools/release/paths.env.example").read_text()
    for k in SP.NAMES:
        assert re.search(rf"^#{k}=", txt, re.M), k
    assert all(ln.startswith("#") or not ln.strip() for ln in txt.splitlines()), "every line stays optional"


# ----------------------------------------------------------------------------------------------- every constant reads it
_DUMP = r"""
import json, os
from navsim.agents.para_ssr.ck import anchor_sampler as AS, constants as C, e2e_data as ED, e2e_data2 as ED2, online, online2
from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.refiner import data as RD, resmap_cache as RC, sdf as SDF
from tools.ck.e2e import launch_util as LU
from tools.ck.e2e2 import launch_util2 as L2
c = ParaSSRConfig()
out = dict(
    ck_data=C.CK_DATA, data_root=C.DATA_ROOT, anchors=C.ANCHORS, teacher_cache=C.TEACHER_CACHE, teacher_init=C.TEACHER_INIT,
    metric_cache=C.METRIC_CACHE, lead=C.LEAD_SRC, splits=C.SPLITS,
    phase1=ED.PHASE1_DEFAULT, det=ED.TEACHER_DET_RUN, map=ED.TEACHER_MAP_RUN, ref=ED.REF_DATA_ROOT,
    warmup2=ED2.WARMUP_DEFAULT, warmup_online2=online2._d_warmup(), phase1_online=online._default_phase1(),
    wote=AS.WOTE_LABELS, official=AS.OFFICIAL_LABELS, anchors_file=AS.ANCHORS_FILE,
    sdf=SDF.SDF_ROOT, rd_data=RD.DATA_ROOT, teacher_roots=RD.TEACHER_ROOTS, mc_roots=RD.MC_ROOTS,
    resmap=RC.RESMAP_ROOT, resmap_roots=RC.RESMAP_ROOTS,
    v7_root=c.distill_v7_root, ref_data_root=c.ref_data_root,
    py=LU.PY, lu_e2e=LU.E2E_ROOT, lu_smoke=LU.SMOKE_ROOT, lu_ck=LU.CK_DATA, packed=LU.PACKED_TRAIN, train_env=LU.TRAIN_ENV,
    anchor=LU.PLAN_ANCHOR_FILE, score=LU.PLAN_SCORE_FILE, l2_ck=L2.CK_DATA, l2_e2e=L2.E2E_ROOT, l2_smoke=L2.SMOKE_ROOT,
)
print("JSON" + json.dumps(out, default=str, sort_keys=True))
"""


def _dump(**extra):
    r = _py(_DUMP, **extra)
    return json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("JSON"))[4:])


def _strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def test_constants_with_no_env_are_todays_values():
    d = _dump()
    k = "/home/external-user/ssd/yongjae_refiner"
    assert d["ck_data"] == f"{k}/ck" and d["data_root"] == k and d["ref_data_root"] == k and d["ref"] == k
    assert d["anchors"] == str(REPO / "resources/planning_vb/trajectory_anchors_256.npy")
    assert d["teacher_cache"]["det"]["navtrain"] == f"{TODAY['SSR_TEACHER_CACHE']}/bevfusion/cache_train_50x100"
    assert d["metric_cache"] == {"navtrain": f"{k}/metric_cache", "navtest": TODAY["SSR_NAVTEST_MC"]}
    s = d["splits"]
    assert s["navtrain_train"]["navsim_logs"] == f"{DATASET_LITERAL}/navsim_logs/trainval"
    assert s["navtrain_val"]["sensor_blobs"] == f"{DATASET_LITERAL}/sensor_blobs/trainval"
    assert s["navtest"]["navsim_logs"] == "/workspace/navsim_workspace/dataset/navsim_logs/test/test"
    assert s["navtest"]["sensor_blobs"] == "/workspace/navsim_workspace/dataset/sensor_blobs/test/test"
    assert s["navtrain_train"]["tokens"] == f"{k}/splits/e2e_train_trainlogs.parquet"
    assert d["phase1"]["packed"] == f"{k}/ck/packed/navtrain_train" and d["det"] == f"{k}/ck/train/ckT_p1"
    assert d["warmup2"]["packed"] == d["warmup_online2"]["packed"] == f"{k}/ck/packed/navtrain_train"
    assert d["warmup2"]["var_dir"] == f"{k}/ck/ck2/labels/navtrain_train/var_separate_sampler16_accstraight"
    assert d["sdf"] == f"{k}/sdf" and d["rd_data"] == k
    assert d["teacher_roots"]["navtrain"] == f"{TODAY['SSR_TEACHER_CACHE']}/bevfusion/cache_train_50x100"
    assert d["mc_roots"][0] == f"{k}/metric_cache" and d["mc_roots"][-1] == TODAY["SSR_NAVTEST_MC"]
    assert d["resmap"] == f"{TODAY['SSR_TEACHER_CACHE']}/resmap"
    assert d["v7_root"] == TODAY["SSR_TEACHER_CACHE"]
    assert d["py"] == TODAY["SSR_PY"] and d["lu_e2e"] == f"{k}/ck/phase2/e2e" and d["l2_e2e"] == f"{k}/ck/ck2/e2e"
    assert d["train_env"]["OPENSCENE_DATA_ROOT"] == DATASET_LITERAL
    assert d["train_env"]["NUPLAN_MAPS_ROOT"] == DATASET_LITERAL + "/maps"
    assert d["anchor"] == d["anchors"] and d["score"] == f"{k}/ck/raw256/pdm_score_256_officialEP"
    assert d["wote"] == f"{TODAY['SSR_PLAN_DATA']}/pdm_score_256"
    if SERVER_DATASET:   # two spellings of one tree on this server: same file as before
        assert Path(d["lead"]["navtest"]).resolve() == Path(
            "/workspace/yongjae/ssd/yongjae_refiner/direction_analysis/uncertainty/out/lead_navtest.parquet").resolve()
        assert Path(d["wote"]).resolve() == Path(
            "/workspace/yongjae/ssd/yongjae_refiner/v2_planning_data/planning_vb/pdm_score_256").resolve()


def test_constants_follow_the_environment(tmp_path):
    ds = _dataset(tmp_path)
    d = _dump(SSR_DATA_ROOT="/new/data", SSR_TEACHER_CACHE="/new/tc", OPENSCENE_DATA_ROOT=ds, SSR_TEST_DATA_ROOT="/new/test",
              SSR_PLAN_DATA="/new/plan", SSR_NAVTEST_MC="/new/mc", SSR_PY="/new/py", SSR_RESEARCH_TREE="/new/research")
    for s in _strings(d):
        s = s.replace(str(REPO), "<repo>").replace(ds, "<ds>")
        for root in ("/home/", "/workspace/", "/venv/", "/data"):
            assert not s.startswith(root), s
    assert d["splits"]["navtest"]["navsim_logs"] == "/new/test/navsim_logs/test/test"
    assert d["train_env"]["NUPLAN_MAPS_ROOT"] == ds + "/maps" and d["py"] == "/new/py"
    assert d["score"] == "/new/data/ck/raw256/pdm_score_256_officialEP" and d["wote"] == "/new/plan/pdm_score_256"
    assert d["mc_roots"] == ["/new/data/metric_cache",
                             "/new/research/report/cause_and_correction_tests/E_train_split_feasibility/metric_cache",
                             "/new/mc"]


def _print_env(**extra):
    r = subprocess.run(["bash", str(TRAIN_SH), "_print_env"], env=_env(RUN="x", E2E_ROOT="/nonexistent/root", **extra),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_launcher_data_env_is_literal_with_no_env_and_follows_the_variables(tmp_path):
    assert _print_env() == ["NUPLAN_MAP_VERSION=nuplan-maps-v1.0", f"NUPLAN_MAPS_ROOT={DATASET_LITERAL}/maps",
                            f"OPENSCENE_DATA_ROOT={DATASET_LITERAL}"]
    ds = _dataset(tmp_path)
    assert _print_env(OPENSCENE_DATA_ROOT=ds)[1:] == [f"NUPLAN_MAPS_ROOT={ds}/maps", f"OPENSCENE_DATA_ROOT={ds}"]


@pytest.mark.skipif(not SERVER_DATASET, reason="needs this server's dataset")
def test_launcher_ignores_a_nested_dataset_from_the_login_shell():
    # the login shell here exports the raw download, whose navsim_logs/trainval is nested one level deeper
    assert _print_env(OPENSCENE_DATA_ROOT="/workspace/navsim_workspace/dataset",
                      NUPLAN_MAPS_ROOT="/workspace/navsim_workspace/dataset/maps") == _print_env()


def test_smoke_prefix_follows_the_data_root_and_its_own_variable():
    def smoke(**extra):
        return subprocess.run(["bash", str(TRAIN_SH), "start", "--gpus", "0,1"], capture_output=True, text=True, timeout=120,
                              env=_env(RUN="x", E2E_ROOT="/nonexistent/root", E2E_SMOKE="1", E2E_DRY_RUN="1", **extra))
    r = smoke(SSR_DATA_ROOT="/new/data")
    assert r.returncode == 3 and "only below /new/data/ck/ck2/e2e_smoke/" in r.stderr
    r = smoke(E2E_SMOKE_PREFIX="/elsewhere/")
    assert r.returncode == 3 and "only below /elsewhere/" in r.stderr


@pytest.mark.parametrize("script", ["tools/ck/e2e2/train_e2e2.sh", "tools/ck/e2e2/smoke_e2e2.sh", "tools/ck/e2e/train_e2e.sh",
                                    "tools/ck/data/run_data.sh", "tools/ck/data/build_val.sh", "tools/release/load_paths.sh",
                                    "scripts/training/train_v2_v15_v7.sh", "scripts/training/train_v2_nokd.sh",
                                    "scripts/training/smoke_v2_final.sh", "scripts/training/smoke_v2_nokd.sh"])
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(REPO / script)], check=True)


# ----------------------------------------------------------------------------------------------- nothing hard-coded left
FINAL_PATH = ("navsim/agents/para_ssr/ck/constants.py", "navsim/agents/para_ssr/ck/e2e_data.py",
              "navsim/agents/para_ssr/ck/e2e_data2.py", "navsim/agents/para_ssr/ck/online.py",
              "navsim/agents/para_ssr/ck/online2.py", "navsim/agents/para_ssr/ck/anchor_sampler.py",
              "navsim/agents/para_ssr/refiner/data.py", "navsim/agents/para_ssr/refiner/resmap_cache.py",
              "navsim/agents/para_ssr/refiner/sdf.py", "navsim/agents/para_ssr/distill/anchor_distill.py",
              "navsim/agents/para_ssr/distill/teacher_store.py", "navsim/agents/para_ssr/distill/distillation.py",
              "navsim/agents/para_ssr/configs/default.py", "tools/ck/ckutil.py", "tools/ck/e2e/launch_util.py",
              "tools/ck/e2e2/launch_util2.py", "tools/ck/e2e2/fit_kd_calib.py", "tools/ck/e2e2/eval_ck2.py",
              "tools/ck/e2e2/eval_e2e2.py", "tools/ck/e2e/eval_e2e.py", "tools/ck/data/dump_v2.py", "tools/ck/data/lead_split.py",
              "tools/refiner/score_trajectories.py", "tools/refiner/rescore_attr.py", "tools/refiner/build_metric_cache.py",
              "tools/refiner/build_sdf.py", "tools/refiner/build_e2e_side.py", "tools/refiner/build_future_objects.py",
              "tools/refiner/extract_human.py", "tools/refiner/make_splits.py")
FINAL_SH = ("tools/ck/e2e2/train_e2e2.sh", "tools/ck/e2e2/smoke_e2e2.sh", "tools/ck/e2e/train_e2e.sh", "tools/ck/data/run_data.sh",
            "tools/ck/data/build_val.sh", "scripts/training/train_v2_v15_v7.sh", "scripts/training/train_v2_nokd.sh",
            "scripts/training/smoke_v2_final.sh", "scripts/training/smoke_v2_nokd.sh")
ROOTS = re.compile(r"(?<![\w.}>)/-])/(home|workspace|venv|data[0-9]?|root|mnt)/\w")    # absolute, not <run>/data/...


@pytest.mark.parametrize("f", FINAL_PATH)
def test_no_hard_coded_root_in_python(f):
    tree = ast.parse((REPO / f).read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)}
    hits = [(n.lineno, n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs and ROOTS.search(n.value)]
    assert not hits, hits


@pytest.mark.parametrize("f", FINAL_SH)
def test_no_hard_coded_root_in_shell(f):
    hits = []
    for i, ln in enumerate((REPO / f).read_text().splitlines(), 1):
        code = ln.split(" #")[0]
        if ln.lstrip().startswith("#") or "byounggun" in code:   # the launcher's guard against another user's tree
            continue
        if ROOTS.search(code) and "/tmp" not in code:
            hits.append((i, ln))
    assert not hits, hits


_IMPORT_TOOLS = r"""
import importlib, json, os, sys
sys.path.insert(0, os.environ["PYTHONPATH"])
out = {}
for mod, names in (("tools.ck.e2e2.fit_kd_calib", ["CK_DATA"]), ("tools.ck.e2e2.eval_ck2", ["CK_DATA"]), ("tools.ck.e2e2.eval_e2e2", ["PY"]),
                   ("tools.ck.e2e.eval_e2e", ["PY", "CK_DATA"]), ("tools.ck.data.dump_v2", []), ("tools.ck.data.lead_split", ["EXTRACT"]),
                   ("tools.refiner.build_sdf", ["DATA", "DEFAULT_MC_ROOTS"]), ("tools.refiner.build_e2e_side", ["TOKENS", "OUT"]),
                   ("tools.refiner.build_future_objects", ["ROOT"]), ("tools.refiner.extract_human", ["TRAIN_LOGS", "TEST_LOGS", "DATA", "REPORT"]),
                   ("tools.refiner.make_splits", ["DATA", "OUT", "TEACHER", "POOL"])):
    m = importlib.import_module(mod)
    for n in names:
        out[f"{mod}.{n}"] = getattr(m, n)
out["maps_env"] = os.environ.get("NUPLAN_MAPS_ROOT")
print("JSON" + json.dumps(out, default=str, sort_keys=True))
"""


def test_tools_import_and_read_the_roots():
    def run(**extra):
        r = _py(_IMPORT_TOOLS, **extra)
        return json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("JSON"))[4:])
    d = run()
    k = TODAY["SSR_DATA_ROOT"]
    assert d["tools.ck.e2e2.fit_kd_calib.CK_DATA"] == d["tools.ck.e2e.eval_e2e.CK_DATA"] == f"{k}/ck"
    assert d["tools.ck.e2e2.eval_e2e2.PY"] == TODAY["SSR_PY"]
    assert d["tools.refiner.extract_human.TRAIN_LOGS"] == f"{DATASET_LITERAL}/navsim_logs/trainval"
    assert d["tools.refiner.extract_human.TEST_LOGS"] == "/workspace/navsim_workspace/dataset/navsim_logs/test/test"
    assert d["tools.refiner.build_e2e_side.OUT"] == f"{k}/e2e_side" and d["maps_env"] == TODAY["NUPLAN_MAPS_ROOT"]
    assert d["tools.refiner.build_sdf.DEFAULT_MC_ROOTS"]["navtest"] == [TODAY["SSR_NAVTEST_MC"]]
    assert d["tools.refiner.build_future_objects.ROOT"] == str(REPO)
    d = run(SSR_DATA_ROOT="/new/d", SSR_TEST_DATA_ROOT="/new/t", SSR_TEACHER_CACHE="/new/tc", SSR_RESEARCH_TREE="/new/r")
    assert d["tools.ck.data.lead_split.EXTRACT"].startswith("/new/d/") and d["tools.refiner.make_splits.OUT"] == "/new/d/splits"
    assert d["tools.refiner.make_splits.TEACHER"].startswith("/new/tc/") and d["tools.refiner.make_splits.POOL"].startswith("/new/r/")
    assert d["tools.refiner.extract_human.TEST_LOGS"] == "/new/t/navsim_logs/test/test"


# ----------------------------------------------------------------------------------------------- anchors
def test_anchors_have_one_canonical_file():
    canon = REPO / "resources/planning_vb/trajectory_anchors_256.npy"
    assert hashlib.sha256(canon.read_bytes()).hexdigest().startswith("44f64a763473c3a8")
    import inspect
    default = inspect.signature(AD.__init__).parameters["trajectory_anchors_file"].default
    assert Path(default) == canon
    from navsim.agents.para_ssr.ck import anchor_sampler as AS
    assert AS.ANCHORS_FILE == canon
    assert AS.load_anchors().shape == (256, 8, 3)     # checks the array sha pin
    m = AD(version="v8")
    assert m.spatial_bias is not None


# ----------------------------------------------------------------------------------------------- distill_norm_files
def _norm(path, value):
    np.savez(path, mean=np.full(256, value, np.float32), std=np.full(256, value + 1.0, np.float32))
    return str(path)


V15 = dict(version="v15", bev_h=100, bev_w=100, pc_range=(-32.0, 0.0, -1.0, 32.0, 64.0, 3.0))


def test_norm_files_empty_keeps_the_candidate_search():
    m = AD(**V15)
    for bank, name in ((0, "bevfusion"), (1, "resmap")):
        with np.load(REPO / f"resources/km_corridor/norm_{name}_100x100.npz") as z:
            want = torch.as_tensor(z["mean"].astype(np.float32)).view(1, 1, 256)
        assert torch.equal(getattr(m, f"z_mean_{bank}"), want)


def test_norm_files_named_file_is_used_absolute_or_repo_relative(tmp_path):
    m = AD(norm_files={"bevfusion": _norm(tmp_path / "d.npz", 2.0),
                       "resmap": "resources/km_corridor/norm_bevfusion_100x100.npz"}, **V15)
    assert m.z_mean_0.unique().tolist() == [2.0] and m.z_std_0.unique().tolist() == [3.0]
    with np.load(REPO / "resources/km_corridor/norm_bevfusion_100x100.npz") as z:
        assert torch.equal(m.z_mean_1, torch.as_tensor(z["mean"].astype(np.float32)).view(1, 1, 256))


@pytest.mark.parametrize("bank", ["bevfusion", "resmap"])
def test_norm_files_missing_file_is_refused(tmp_path, bank):
    with pytest.raises(FileNotFoundError, match=f"distill_norm_files.{bank}"):
        AD(norm_files={bank: str(tmp_path / "absent.npz")}, **V15)


def test_norm_files_unknown_bank_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown bank"):
        AD(norm_files={"bev": _norm(tmp_path / "d.npz", 1.0)}, **V15)


def test_norm_files_reach_the_module_through_hydra_and_the_builder(tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from navsim.agents.para_ssr.distill.distillation import build_planning_distillation
    d, m = _norm(tmp_path / "d.npz", 4.0), _norm(tmp_path / "m.npz", 5.0)
    root = _fake_det(tmp_path / "root" / "bevfusion" / "cache_train_100x100", ["aa00"])
    _fake_map(tmp_path / "root" / "resmap", ["aa00"])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/training"), version_base="1.2"):
        cfg = compose(config_name="default_training", overrides=[
            "agent=para_ssr_agent", "experiment_name=x", "split=trainval",
            f"++agent.config.distill_norm_files={{bevfusion:{d},resmap:{m}}}",
            "++agent.config.bev_h=100", "++agent.config.bev_w=100", "++agent.config.pc_range=[-32.0,0.0,-1.0,32.0,64.0,3.0]",
            "++agent.config.distill_cache_size=[100,100]", "++agent.config.use_distill=true",
            f"++agent.config.distill_feature_root={tmp_path / 'root'}", "++agent.config.distill_selector_v15=true",
            "++agent.config.distill_teacher_source=targets",
            f"++agent.config.plan_anchor_file={REPO / 'resources/planning_vb/trajectory_anchors_256.npy'}"])
    GlobalHydra.instance().clear()
    pc = instantiate(cfg.agent.config)
    assert dict(pc.distill_norm_files) == {"bevfusion": d, "resmap": m}
    a = build_planning_distillation(pc).anchor_distill
    assert a.z_mean_0.unique().tolist() == [4.0] and a.z_mean_1.unique().tolist() == [5.0]
    assert root.is_dir()


# ----------------------------------------------------------------------------------------------- teacher caches
def _fake_det(cache: Path, tokens, sha="dethead", fill=None):
    """Layout A, 100 x 100; bev_feature value = fill or the token's position + 1 in a recognisable ramp."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.json").write_text(json.dumps({
        "checkpoint_sha256_head": sha, "layout": "samples/<token[:2]>/<token>.npz", "target_bev_shape": [100, 100],
        "bev_channels": 256, "point_cloud_range": [0.0, -32.0, -3.0, 64.0, 32.0, 5.0],
        "to_student_transform": "student_bev = teacher_bev[:, :, ::-1]"}))
    for t in tokens:
        (cache / "samples" / t[:2]).mkdir(parents=True, exist_ok=True)
        x = np.arange(100, dtype=np.float16)[None, None, :] * np.ones((256, 100, 100), np.float16)
        np.savez(cache / "samples" / t[:2] / f"{t}.npz", bev_feature=x if fill is None else x + fill)
    return cache


def _fake_map(root: Path, tokens, sha="mapsha", seg=True):
    """Layout B, bev (256, lateral 100, forward 100)."""
    (root / "bev").mkdir(parents=True, exist_ok=True)
    tensors = {"bev": {"shape": [256, 100, 100], "dtype": "float16"}}
    if seg:
        tensors["seg"] = {"shape": [4, 200, 200], "dtype": "float16"}
    (root / "meta.json").write_text(json.dumps({
        "checkpoint_sha256": sha, "split": "none", "classes": ["road", "walkway", "centerline", "crosswalk"],
        "pc_range": [0, -32, -3, 64, 32, 5], "roi_size_m": [64, 64], "tensors": tensors, "num_frames": len(tokens)}))
    (root / "index.json").write_text(json.dumps({t: ["s0", i] for i, t in enumerate(tokens)}))
    x = np.arange(100, dtype=np.float16)[None, None, None, :] * np.ones((len(tokens), 256, 100, 100), np.float16)
    np.save(root / "bev" / "s0.npy", x)
    return root


def _scene(token):
    return SimpleNamespace(frames=[SimpleNamespace(token="old"), SimpleNamespace(token=token)],
                           scene_metadata=SimpleNamespace(num_history_frames=2))


def _builder_cfg(root, det="bevfusion", mp="resmap"):
    return SimpleNamespace(distill_feature_root=str(root), embed_dims=256, bev_h=100, bev_w=100,
                           distill_cache_size=(100, 100),
                           distill_branches={"bevfusion": {"cache_name": det}, "resmap": {"cache_name": mp, "flip_w": False}})


def _targets(root, token="aa00", **names):
    b = DistillTeacherTargetBuilder(_builder_cfg(root, **names))
    return b.compute_targets(_scene(token))


def _assert_reader_output(t):
    """DET flipped on the width, MAP transposed only: the fake caches ramp along the last raw axis."""
    assert bool(t["kd_ok_0"]) and bool(t["kd_ok_1"])
    assert t["kd_bev_0"].shape == t["kd_bev_1"].shape == (256, 100, 100)
    assert t["kd_bev_0"][0, 0, 0].item() == 99.0 and t["kd_bev_0"][0, 0, 99].item() == 0.0
    assert t["kd_bev_1"][0, 0, 0].item() == 0.0 and t["kd_bev_1"][0, 99, 0].item() == 99.0


def test_builder_follows_cache_name(tmp_path):
    _fake_det(tmp_path / "bevformer/cache_train_100x100", ["aa00"])
    _fake_map(tmp_path / "mask2map", ["aa00"])
    _assert_reader_output(_targets(tmp_path, det="bevformer", mp="mask2map"))
    assert not bool(_targets(tmp_path)["kd_ok_0"])        # the role names alone find nothing here


def test_builder_with_defaults_reads_the_role_names(tmp_path):
    _fake_det(tmp_path / "bevfusion/cache_train_100x100", ["aa00"])
    _fake_map(tmp_path / "resmap", ["aa00"])
    _assert_reader_output(_targets(tmp_path))
    plain = DistillTeacherTargetBuilder(SimpleNamespace(distill_feature_root=str(tmp_path), embed_dims=256, bev_h=100,
                                                        bev_w=100, distill_cache_size=(100, 100)))
    _assert_reader_output(plain.compute_targets(_scene("aa00")))      # config without distill_branches


def test_builder_follows_the_environment_over_the_root(tmp_path, monkeypatch):
    _fake_det(tmp_path / "x/any_train", ["aa00"])
    _fake_det(tmp_path / "x/any_val", ["bb00"])
    _fake_map(tmp_path / "y/map_cache", ["aa00"])
    monkeypatch.setenv("SSR_DET_CACHE_TRAIN", str(tmp_path / "x/any_train"))
    monkeypatch.setenv("SSR_DET_CACHE_VAL", str(tmp_path / "x/any_val"))
    monkeypatch.setenv("SSR_MAP_CACHE_ROOT", str(tmp_path / "y/map_cache"))
    _assert_reader_output(_targets(tmp_path / "empty", det="new_det", mp="new_map"))
    assert bool(_targets(tmp_path / "empty", token="bb00", det="new_det", mp="new_map")["kd_ok_0"])    # val directory too


def test_builder_50x100_config_switches_to_100_through_the_env_directory(tmp_path, monkeypatch):
    _fake_det(tmp_path / "x/any_train", ["aa00"])
    monkeypatch.setenv("SSR_DET_CACHE_TRAIN", str(tmp_path / "x/any_train"))
    cfg = _builder_cfg(tmp_path / "empty")
    cfg.distill_cache_size = (50, 100)
    assert DistillTeacherTargetBuilder(cfg)._cache_size == (100, 100)
    monkeypatch.delenv("SSR_DET_CACHE_TRAIN")
    assert DistillTeacherTargetBuilder(cfg)._cache_size == (50, 100)


def test_role_store_kwargs_empty_with_no_env(monkeypatch):
    for k in ENV_NAMES:
        monkeypatch.delenv(k, raising=False)
    assert role_store_kwargs("bevfusion") == {} and role_store_kwargs("resmap") == {} and role_store_kwargs("other") == {}


def _store_cfg():
    return SimpleNamespace(bev_h=100, bev_w=100, embed_dims=256, distill_cache_size=(100, 100),
                           pc_range=(-32.0, 0.0, -1.0, 32.0, 64.0, 3.0))


def test_store_sha_is_checked_only_when_named(tmp_path, monkeypatch):
    _fake_det(tmp_path / "dd", ["aa00"], sha="good")
    _fake_map(tmp_path / "mm", ["aa00"], sha="good")
    det = dict(cache_subdirs=(str(tmp_path / "dd"),))
    mp = dict(cache_dir=str(tmp_path / "mm"))
    TeacherFeatureStore(str(tmp_path), "n", **det).validate_manifest(_store_cfg())
    TeacherFeatureStore(str(tmp_path), "n", **mp).validate_manifest(_store_cfg())
    TeacherFeatureStore(str(tmp_path), "n", expect_sha="good", **det).validate_manifest(_store_cfg())
    TeacherFeatureStore(str(tmp_path), "n", expect_sha="good", **mp).validate_manifest(_store_cfg())
    for kw in (det, mp):
        with pytest.raises(TeacherCacheMismatch, match="checkpoint_sha256"):
            TeacherFeatureStore(str(tmp_path), "n", expect_sha="other", **kw).validate_manifest(_store_cfg())
    monkeypatch.setenv("SSR_DET_SHA_HEAD", "h")
    monkeypatch.setenv("SSR_MAP_SHA256", "s")
    assert role_store_kwargs("bevfusion") == {"expect_sha": "h"} and role_store_kwargs("resmap") == {"expect_sha": "s"}


_REFINER_READERS = r"""
import json, numpy as np
from navsim.agents.para_ssr.refiner import data as RD, resmap_cache as RC
out = dict(roots={k: str(v) for k, v in RD.TEACHER_ROOTS.items()}, sha=RD.TEACHER_SHA_HEAD, map_root=str(RC.RESMAP_ROOT),
           map_sha=RC.RESMAP_SHA256, map_head=RC.RESMAP_SHA_HEAD, req=list(RC.RESMAP_REQUIRED_FIELDS))
try:
    det = RD.TeacherCache.for_subset("navtrain")
    out["det"] = float(det.load_bev("aa00")[0, 0, 0])
except Exception as e:
    out["det_err"] = repr(e)
try:
    mp = RC.ResmapCache.for_subset("navtrain")
    out["map"] = float(mp.load_bev("aa00")[0, 99, 0])
except Exception as e:
    out["map_err"] = repr(e)
print("JSON" + json.dumps(out))
"""


def _readers(**extra):
    r = _py(_REFINER_READERS, CK_BEV64="1", **extra)
    return json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("JSON"))[4:])


def test_refiner_readers_defaults_are_todays_with_no_env():
    d = _readers()
    assert d["roots"] == {"navtrain": f"{TODAY['SSR_TEACHER_CACHE']}/bevfusion/cache_train_100x100",
                          "navtest": f"{TODAY['SSR_TEACHER_CACHE']}/bevfusion/cache_val_100x100"}
    assert d["sha"] == "71ea94f672c89521"
    assert d["map_root"] == f"{TODAY['SSR_TEACHER_CACHE']}/resmap/stage1_front64"
    assert d["map_sha"] == "e67cc78d0da6afe00f0396d9940e98024826d7288117f346b9f85a7ade699858" and d["req"] == ["bev", "seg"]


def test_refiner_readers_follow_the_environment(tmp_path):
    _fake_det(tmp_path / "d", ["aa00"], sha="dethead")
    _fake_map(tmp_path / "m", ["aa00"], sha="mapsha" + "0" * 58)
    env = dict(SSR_DET_CACHE_TRAIN=str(tmp_path / "d"), SSR_DET_CACHE_VAL=str(tmp_path / "dv"), SSR_DET_SHA_HEAD="dethead",
               SSR_MAP_CACHE_ROOT=str(tmp_path / "m"), SSR_MAP_SHA256="mapsha" + "0" * 58)
    d = _readers(**env)
    assert d["roots"] == {"navtrain": str(tmp_path / "d"), "navtest": str(tmp_path / "dv")}
    assert d["map_head"] == ("mapsha" + "0" * 58)[:16]
    assert d["det"] == 99.0 and d["map"] == 99.0, d          # S grid: DET flipped, MAP transposed


def test_refiner_readers_refuse_a_wrong_sha_and_a_missing_seg_unless_relaxed(tmp_path):
    _fake_det(tmp_path / "d", ["aa00"], sha="dethead")
    _fake_map(tmp_path / "m", ["aa00"], sha="mapsha", seg=False)
    d = _readers(SSR_DET_CACHE_TRAIN=str(tmp_path / "d"), SSR_DET_SHA_HEAD="another", SSR_MAP_CACHE_ROOT=str(tmp_path / "m"),
                 SSR_MAP_SHA256="another")
    assert "checkpoint_sha256_head" in d["det_err"] and "checkpoint_sha256" in d["map_err"]
    d = _readers(SSR_MAP_CACHE_ROOT=str(tmp_path / "m"), SSR_MAP_SHA256="mapsha")
    assert "seg" in d["map_err"]                             # seg metadata required by default
    d = _readers(SSR_MAP_CACHE_ROOT=str(tmp_path / "m"), SSR_MAP_SHA256="mapsha", SSR_MAP_REQUIRE_SEG="0")
    assert d["map"] == 99.0 and d["req"] == ["bev"]


@pytest.mark.skipif(not REAL_CACHES, reason="needs the real teacher caches")
def test_real_caches_read_the_same_through_a_new_name_and_the_environment(tmp_path, monkeypatch):
    with open(TC / "resmap/stage1_front64/index.json") as f:
        tok_map = next(iter(json.load(f)))
    det_root = TC / "bevfusion/cache_train_100x100"
    tok = None
    sub = sorted(os.listdir(det_root / "samples"))[0]
    for name in sorted(os.listdir(det_root / "samples" / sub)):
        tok = name[:-4]
        break
    plain = DistillTeacherTargetBuilder(_builder_cfg(TC))
    a, a_map = plain.compute_targets(_scene(tok)), plain.compute_targets(_scene(tok_map))
    monkeypatch.setenv("SSR_DET_CACHE_TRAIN", str(det_root))
    monkeypatch.setenv("SSR_DET_CACHE_VAL", str(TC / "bevfusion/cache_val_100x100"))
    monkeypatch.setenv("SSR_DET_SHA_HEAD", "71ea94f672c89521")
    monkeypatch.setenv("SSR_MAP_CACHE_ROOT", str(TC / "resmap/stage1_front64"))
    monkeypatch.setenv("SSR_MAP_SHA256", "e67cc78d0da6afe00f0396d9940e98024826d7288117f346b9f85a7ade699858")
    new = DistillTeacherTargetBuilder(_builder_cfg(tmp_path / "nothing_here", det="alias_det", mp="alias_map"))
    b, b_map = new.compute_targets(_scene(tok)), new.compute_targets(_scene(tok_map))
    assert bool(a["kd_ok_0"]) and bool(a_map["kd_ok_1"])
    assert torch.equal(a["kd_bev_0"], b["kd_bev_0"]) and torch.equal(a_map["kd_bev_1"], b_map["kd_bev_1"])
    cfg = _store_cfg()
    TeacherFeatureStore(str(tmp_path), "alias_det", **role_store_kwargs("bevfusion")).validate_manifest(cfg)
    TeacherFeatureStore(str(tmp_path), "alias_map", flip_w=False, **role_store_kwargs("resmap")).validate_manifest(cfg)


_REAL_BEV_SHA = r"""
import hashlib, json, sys
from navsim.agents.para_ssr.refiner import data as RD, resmap_cache as RC
det, mp = RD.TeacherCache.for_subset("navtrain"), RC.ResmapCache.for_subset("navtrain")
tok = next(t for t in mp.index if det.has(t))
print("JSON" + json.dumps({"tok": tok, "det": hashlib.sha256(det.load_bev(tok).tobytes()).hexdigest(),
                           "map": hashlib.sha256(mp.load_bev(tok).tobytes()).hexdigest(), "roots": str(det.root) + "|" + str(mp.root)}))
"""


@pytest.mark.skipif(not REAL_CACHES, reason="needs the real teacher caches")
def test_refiner_readers_same_bytes_with_the_environment_set_to_the_real_caches():
    def run(**extra):
        r = _py(_REAL_BEV_SHA, CK_BEV64="1", **extra)
        return json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("JSON"))[4:])
    a = run()
    b = run(SSR_DET_CACHE_TRAIN="/workspace/teacher_cache/bevfusion/cache_train_100x100",
            SSR_DET_CACHE_VAL="/workspace/teacher_cache/bevfusion/cache_val_100x100", SSR_DET_SHA_HEAD="71ea94f672c89521",
            SSR_MAP_CACHE_ROOT="/workspace/teacher_cache/resmap/stage1_front64",
            SSR_MAP_SHA256="e67cc78d0da6afe00f0396d9940e98024826d7288117f346b9f85a7ade699858")
    assert a["roots"] != b["roots"] and (a["tok"], a["det"], a["map"]) == (b["tok"], b["det"], b["map"])


# ----------------------------------------------------------------------------------------------- vendored scorer
_ST_IMPORT = r"""
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["PYTHONPATH"], "tools/refiner"))
import score_trajectories as ST, rescore_attr as RA
import navsim
print("JSON" + json.dumps(dict(root=str(ST.ROOT), yaml=str(ST.SCORING_YAML), sources=[str(p) for p in ST.SOURCES],
      ra=RA.__file__, navsim=navsim.__file__, maps=os.environ["NUPLAN_MAPS_ROOT"], ra_root=str(RA.ROOT), sys_path=sys.path,
      mc=[str(p) for p in ST.MC_ROOTS], arch=str(ST.ARCHIVED_EVAL_CFG))))
"""


def _st(**extra):
    r = _py(_ST_IMPORT, **extra)
    return json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("JSON"))[4:])


def test_scorer_reads_this_repo_only():
    d = _st()
    assert d["root"] == str(REPO) and d["ra_root"] == str(REPO)
    assert d["yaml"] == str(REPO / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    assert d["ra"] == str(REPO / "tools/refiner/rescore_attr.py") and d["navsim"] == str(REPO / "navsim/__init__.py")
    assert all(p.startswith(str(REPO)) and Path(p).is_file() for p in d["sources"]) and len(d["sources"]) == 5
    assert str(REPO / "tools/refiner/rescore_attr.py") in d["sources"]
    assert not [p for p in d["sys_path"] if "/report/" in p]      # no research tree on the path
    assert d["maps"] == TODAY["NUPLAN_MAPS_ROOT"]
    assert d["mc"][0] == f"{TODAY['SSR_DATA_ROOT']}/metric_cache" and d["mc"][-1] == TODAY["SSR_NAVTEST_MC"]
    assert d["arch"].startswith(TODAY["SSR_RESEARCH_TREE"])


def test_scorer_maps_root_comes_from_the_environment():
    assert _st(NUPLAN_MAPS_ROOT="/new/maps")["maps"] == "/new/maps"
    assert _st(SSR_RESEARCH_TREE="/new/r")["mc"][1].startswith("/new/r/report/")


@pytest.mark.skipif(not LIVE_RESCORE.is_file(), reason="needs the research tree's rescore_attr.py")
def test_vendored_rescore_differs_from_the_original_only_in_paths():
    assert hashlib.sha256(LIVE_RESCORE.read_bytes()).hexdigest().startswith("9dbd00cd606bb18c")
    live = LIVE_RESCORE.read_text().splitlines()
    new = (REPO / "tools/refiner/rescore_attr.py").read_text().splitlines()
    import difflib
    changed = [ln[2:] for ln in difflib.ndiff(live, new) if ln[:2] in ("- ", "+ ")]
    allowed = re.compile(r"^(ROOT|OUT|MC_ROOT|EVAL|TABLE|RESEARCH) = |^from navsim.common import ssr_paths|^$|^Run:|^Vendored|^file every"
                         r"|^_make_classes")
    assert all(allowed.search(ln) for ln in changed), [ln for ln in changed if not allowed.search(ln)]


_BITWISE = r"""
import importlib.util, lzma, os, pickle, sys
import navsim
import numpy as np
sys.path.insert(0, os.path.join(os.environ["PYTHONPATH"], "tools/refiner"))
import score_trajectories as ST
spec = importlib.util.spec_from_file_location("rescore_attr_live", sys.argv[1])
LIVE = importlib.util.module_from_spec(spec); spec.loader.exec_module(LIVE)
assert navsim.__file__.startswith(os.environ["PYTHONPATH"]), navsim.__file__
sim, scorer = ST.build_simulator_scorer(record=False)
S_live, C_live = LIVE._make_classes(type(sim), type(scorer))
sim_l, sc_l = S_live(sim.proposal_sampling), C_live(scorer.proposal_sampling, scorer._config, scorer._vehicle_parameters)
sim_n, sc_n = ST.build_simulator_scorer(record=True)
paths, n_each = [], int(sys.argv[2])
for root in sys.argv[3:]:
    logs = sorted(os.listdir(root)); rng = np.random.default_rng(0); got = 0
    for lg in [logs[i] for i in rng.permutation(len(logs))]:
        d = os.path.join(root, lg, "unknown")
        if os.path.isdir(d):
            ts = sorted(os.listdir(d)); p = os.path.join(d, ts[int(rng.integers(len(ts)))], "metric_cache.pkl")
            if os.path.isfile(p):
                paths.append(p); got += 1
        if got == n_each:
            break
t = np.linspace(0.5, 4.0, 8); nv = bad = fail = 0
for p in paths:
    with lzma.open(p, "rb") as f:
        mc = pickle.load(f)
    v = max(float(mc.ego_state.dynamic_car_state.speed), 3.0)
    tr = np.zeros((5, 8, 3), np.float32)
    tr[1, :, 0] = v * t; tr[2, :, 0] = v * t; tr[2, :, 1] = 0.25 * t ** 2; tr[3, :, 0] = 2.5 * v * t
    tr[4, :, 0] = v * t; tr[4, :, 1] = -1.0 * t ** 2
    a = ST.score_token(mc, tr, sim_n, sc_n, return_states=True)
    b = ST.score_token(mc, tr, sim_l, sc_l, return_states=True)
    bad += sim_n.__class__ is sim_l.__class__
    for x, y in zip(a, b):
        fail += x["pdms"] < 1.0
        assert set(x) == set(y)
        for k in x:
            nv += 1
            same = x[k].tobytes() == y[k].tobytes() if isinstance(x[k], np.ndarray) else (
                np.float64(x[k]).tobytes() == np.float64(y[k]).tobytes() if isinstance(x[k], float) else x[k] == y[k])
            bad += not same
print("RESULT", len(paths), nv, bad, fail)
"""


@pytest.mark.skipif(not (LIVE_RESCORE.is_file() and MC_TRAIN.is_dir() and MC_TEST.is_dir()),
                    reason="needs the research tree's rescore_attr.py and the metric caches")
def test_vendored_rescore_scores_real_tokens_bitwise_like_the_original():
    r = subprocess.run([PY, "-c", _BITWISE, str(LIVE_RESCORE), "12", str(MC_TRAIN), str(MC_TEST)], env=_env(),
                       capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, r.stderr[-3000:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")).split()
    tokens, values, bad, failing = (int(x) for x in line[1:])
    assert tokens >= 20 and values > 2000 and bad == 0, line
    assert failing > 0, "the probe trajectories must reach the failure paths"


# ----------------------------------------------------------------------------------------------- saved metric-cache config
def test_saved_metric_cache_config_is_vendored_and_pinned(tmp_path):
    r = _py(r"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("BMC", "tools/refiner/build_metric_cache.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
import json
print("JSON" + json.dumps(dict(cfg=str(B.SAVED_CFG), ok=B.sha256(B.SAVED_CFG) == B.SAVED_CFG_SHA256, train=str(B.TRAIN_LOGS),
      test=str(B.TEST_LOGS), mc=str(B.NAVTEST_MC), out=str(B.OUT), env=B.ENV)))
""".replace('"tools/', f'"{REPO}/tools/'))
    d = json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("JSON"))[4:])
    assert d["cfg"] == str(REPO / "resources/metric_cache_saved_config.yaml") and d["ok"]
    assert d["train"] == f"{DATASET_LITERAL}/navsim_logs/trainval"
    assert d["test"] == "/workspace/navsim_workspace/dataset/navsim_logs/test/test"
    assert d["out"] == f"{TODAY['SSR_DATA_ROOT']}/metric_cache" and d["mc"] == TODAY["SSR_NAVTEST_MC"]
    assert d["env"]["OPENSCENE_DATA_ROOT"] == DATASET_LITERAL and d["env"]["NUPLAN_MAPS_ROOT"] == DATASET_LITERAL + "/maps"
    assert hashlib.sha256((REPO / "resources/metric_cache_saved_config.yaml").read_bytes()).hexdigest() == \
        "92d42853253622b29eabf8e488a97741e350eb3d5a360cb45f69cb0a4b444f84"


def test_build_refuses_a_changed_saved_config(tmp_path):
    changed = tmp_path / "changed.yaml"
    changed.write_text((REPO / "resources/metric_cache_saved_config.yaml").read_text() + "\n# changed\n")
    code = f"""
import importlib.util
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location("BMC", "{REPO}/tools/refiner/build_metric_cache.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
B.SAVED_CFG = __import__("pathlib").Path("{changed}")
try:
    B.cmd_build(SimpleNamespace(out="{tmp_path / 'out'}"))
except AssertionError as e:
    print("REFUSED", e)
"""
    assert "REFUSED saved caching config changed" in _py(code).stdout
