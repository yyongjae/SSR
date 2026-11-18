"""Variant-set ablation arms (VARIANTS=base|v8|v4|spd4|lat4) on the release side: scripts/experiment.sh, tools/release/
{compose,verify_assets,smoke_check,check_build}.py, tools/release/setup/{50_warmup_labels,run_refiner_teachers}.sh and the
labeler worker cap.  All CPU, nothing is written outside tmp_path.  Run (with or without tools/release/paths.env exported):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_variant_arms_release.py
Base identity: the default (base) composition is checked against the pre-change behaviour by a recomputation of the old
set lines through the launcher helpers, by hard-coded old constants (the base var_dir line, the yaml variant table, the
main_053a64 name of this server's registry) and by base == an explicit VARIANTS=base.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from navsim.agents.para_ssr.ck import variants as V  # noqa: E402
from navsim.common import ssr_paths as SP  # noqa: E402
from tools.ck.e2e2 import launch_util2 as L2  # noqa: E402
from tools.release import compose as C  # noqa: E402
from tools.release import smoke_check as SC  # noqa: E402
from tools.release import verify_assets as VA  # noqa: E402

ARMS = ("v8", "v4", "spd4", "lat4", "none", "cross")
CK = "agent.config.ck_e2e2."
SH = REPO / "scripts/experiment.sh"
LOCAL_REGISTRY = Path("/workspace/data/registry.local.yaml")
# pre-change constants of the base recipe
OLD_VAR_DIR_TAIL = "/ck/ck2/labels/navtrain_train/var_separate_sampler16_accstraight"
OLD_SPEEDS_LINE = "++agent.config.ck_e2e2.variants.speeds=[-1.0,-0.5,0.5]"
OLD_LATS_LINE = "++agent.config.ck_e2e2.variants.lats=[-0.5,0.5]"
WANT = {"v8": dict(nv=9, speeds=[-1.0, -0.5, 0.5, 1.0], lats=[-1.0, -0.5, 0.5, 1.0]),
        "v4": dict(nv=5, speeds=[-0.5, 0.5], lats=[-0.5, 0.5]),
        "spd4": dict(nv=5, speeds=[-1.0, -0.5, 0.5, 1.0], lats=[]),
        "lat4": dict(nv=5, speeds=[], lats=[-1.0, -0.5, 0.5, 1.0]),
        "none": dict(nv=1, speeds=[], lats=[]),
        "cross": dict(nv=25, speeds=[-1.0, -0.5, 0.5, 1.0], lats=[-1.0, -0.5, 0.5, 1.0], combine="cross")}
for _w in WANT.values():
    _w.setdefault("combine", "separate")


def _kv(c):
    return {C.LU._okey(o): o.split("=", 1)[1] for o in c.overrides if "=" in o}


def _env(**extra):
    drop = set(SP.NAMES) | set(C.CONTROL_ENV) | {"SSR_RELEASE_REGISTRY", "SSR_RELEASE_TEST_STUBS", "ALLOW_DIRTY", "CHECK_ROOT"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


def sh(*args, **extra):
    return subprocess.run(["bash", str(SH), *args], env=_env(**extra), capture_output=True, text=True, timeout=600)


# ----------------------------------------------------------------------------------------------- the arm table
def test_compose_reads_the_arm_table_of_the_variants_module_without_torch():
    want = {a: {"speeds": tuple(map(float, v["speeds"])), "lats": tuple(map(float, v["lats"])),
                "combine": v.get("combine", "separate")} for a, v in V.VARIANT_ARMS.items()}
    assert C.variant_arms() == want and list(C.variant_arms())[0] == "base"
    assert C.CHOICES["VARIANTS"] == tuple(V.VARIANT_ARMS) == ("base",) + ARMS
    for a in V.VARIANT_ARMS:
        assert C.arm_var_name(a) == V.arm_var_name(a)
    for a in ARMS:
        lay = V.arm_layout(a)
        assert (lay.nv, list(lay.speeds), list(lay.lats), lay.combine) == (WANT[a]["nv"], WANT[a]["speeds"],
                                                                          WANT[a]["lats"], WANT[a]["combine"])
        assert C.variant_arms()[a]["combine"] == lay.combine
    code = "import sys; from tools.release import compose as C; C.compose({'VARIANTS': 'v8'}); print('torch' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], env=_env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr


# ----------------------------------------------------------------------------------------------- base identity
def _old_composition_overrides(s, roots, teachers, reg_x, run_dir):
    """The pre-change compose._resolved: set lines without any variant line and the base var_dir."""
    data = roots["SSR_DATA_ROOT"]
    sets = [f"warmup.packed={data}/ck/packed/navtrain_train", f"warmup.raw_labels={data}/ck/labels/navtrain_train/raw256",
            f"warmup.var_dir={data}{OLD_VAR_DIR_TAIL}", f"seed={s['SEED']}",
            f"teacher_det_run={teachers['det']['refiner_run']}", f"teacher_map_run={teachers['map']['refiner_run']}"]
    extra = C.extra_lines(s, roots, teachers, reg_x)
    ck = L2.effective_ck(run_dir, 8, sets, extra)
    return sets, L2.merge_overrides(run_dir, 8, ck, 6, 30, extra, None, s["WANDB"])


def test_base_composition_equals_the_pre_change_recomputation():
    c = C.compose({})
    reg = C.load_registry()
    reg_x = {"backbones": reg["backbones"],
             "teachers": {r: {n: C._expand(e, c.roots) for n, e in reg["teachers"][r].items()} for r in ("det", "map")}}
    used = {"det": reg_x["teachers"]["det"]["bevfusion"], "map": reg_x["teachers"]["map"]["resmap"]}
    sets, ov = _old_composition_overrides(c.settings, c.roots, used, reg_x, c.run_dir)
    assert c.set_lines == sets and c.overrides == ov
    assert OLD_SPEEDS_LINE in c.overrides and OLD_LATS_LINE in c.overrides
    assert _kv(c)[CK + "warmup.var_dir"].endswith(OLD_VAR_DIR_TAIL)
    assert not any(x.startswith("variants.") for x in c.set_lines)
    assert c.default_recipe and c.run.startswith("main_") and "VARIANTS" not in c.settings


def test_base_is_the_same_composition_with_or_without_the_key(tmp_path):
    a, b = C.compose({}), C.compose({"VARIANTS": "base"})
    for f in ("run", "hash6", "set_lines", "extra_lines", "overrides", "env", "pins", "profile", "default_recipe", "settings"):
        assert getattr(a, f) == getattr(b, f), f
    assert a.record() == b.record()
    # the written composition and the record name no new key (byte for byte as before the setting existed)
    for name, c in (("a", a), ("b", b)):
        C.write_composition(c, str(tmp_path / name))
    for f in ("refiner.set", "arm.extra", "env.sh", "composition.json"):
        assert (tmp_path / "a" / f).read_text() == (tmp_path / "b" / f).read_text().replace(str(tmp_path / "b"), str(tmp_path / "a"))
    comp = json.loads((tmp_path / "a/composition.json").read_text())
    assert "VARIANTS" not in comp["settings"] and list(comp["settings"]) == [k for k in C.DEFAULTS if k != "VARIANTS"]


@pytest.mark.skipif(not LOCAL_REGISTRY.is_file(), reason="this server's registry only")
def test_base_name_on_this_server_is_unchanged():
    assert C.compose({}, registry=str(LOCAL_REGISTRY)).run == "main_053a64"
    r = sh("name", "RUN_ROOT=/tmp/never_created_by_this_test", f"REGISTRY={LOCAL_REGISTRY}")
    assert r.returncode == 0 and r.stdout.strip() == "main_053a64", r.stderr
    r = sh("name", "RUN_ROOT=/tmp/never_created_by_this_test", f"REGISTRY={LOCAL_REGISTRY}", "VARIANTS=base")
    assert r.stdout.strip() == "main_053a64"
    assert not Path("/tmp/never_created_by_this_test").exists()


def test_every_existing_setting_composes_as_before():
    """Old settings records (no VARIANTS key) and the base value give the same composition for other settings."""
    for kw in ({"REFINER": "off"}, {"REFINER_KD": "off", "BEV_KD": "off"}, {"DET_KD": "off"}, {"BACKBONE": "r50", "SEED": "1"},
               {"REFINER": "off", "BEV_KD": "off"}, {"EXTRA": "agent.lr=2e-4"}):
        a, b = C.compose(kw), C.compose(dict(kw, VARIANTS="base"))
        assert (a.run, a.overrides, a.set_lines, a.record()) == (b.run, b.overrides, b.set_lines, b.record())
        assert "VARIANTS" not in a.settings and not any(x.startswith("variants.") for x in a.set_lines)
        assert C.compose(a.settings).run == a.run and C.canonicalize(a.settings) == a.settings


# ----------------------------------------------------------------------------------------------- arms
@pytest.mark.parametrize("arm", ARMS)
def test_arm_lines_name_and_layout(arm):
    base, c = C.compose({}), C.compose({"VARIANTS": arm})
    k, w = _kv(c), WANT[arm]
    assert re.fullmatch(rf"var-{arm}_[0-9a-f]{{6}}", c.run) and c.hash6 != base.hash6
    assert not c.default_recipe and c.profile == "full" and c.settings["VARIANTS"] == arm
    assert k[CK + "warmup.var_dir"].endswith(OLD_VAR_DIR_TAIL + "_" + arm)
    assert k[CK + "variants.speeds"] == C._float_list(w["speeds"]) and k[CK + "variants.lats"] == C._float_list(w["lats"])
    assert k[CK + "variants.combine"] == w["combine"]
    # only the three variant lines (and the run-name lines) differ from base (+ variants.combine for cross)
    hb, hc = set(C.hash_lines(base.overrides, base.roots)), set(C.hash_lines(c.overrides, c.roots))
    allowed = {CK + "variants.speeds", CK + "variants.lats", CK + "warmup.var_dir"}
    if w["combine"] != "separate":
        allowed.add(CK + "variants.combine")
        assert f"variants.combine={w['combine']}" in c.set_lines
    else:
        assert not any(x.startswith("variants.combine") for x in c.set_lines)
    if w["nv"] > 1:                  # the draw counts stay the yaml's (32 <= 16 x (V - 1)), also for 400 columns
        assert not ({CK + "n_var_step", CK + "warmup.n_var", CK + "onpolicy_label.n_var"} & {C.LU._okey(x) for x in hc ^ hb})
    if w["nv"] == 1:                 # V = 1: no variant columns, so the three variant draw counts are 0
        allowed |= {CK + "n_var_step", CK + "warmup.n_var", CK + "onpolicy_label.n_var"}
        assert all(k[CK + x] == "0" for x in ("n_var_step", "warmup.n_var", "onpolicy_label.n_var"))
    assert {C.LU._okey(x) for x in hc ^ hb} <= allowed
    ck = L2.effective_ck(c.run_dir, 8, c.set_lines, c.extra_lines)
    assert V.VariantLayout.from_cfg(ck["variants"]) == V.arm_layout(arm) and ck["warmup"]["var_dir"].endswith(V.arm_var_name(arm))
    assert L2.arm_of(ck) == "main"
    # one config, one spelling; the record composes to the same run
    assert C.canonicalize(c.settings) == c.settings and C.compose(c.settings).run == c.run
    assert C.compose({"VARIANTS": arm}, smoke=True).run == "smoke_" + c.run


def test_arms_are_distinct_also_with_the_same_column_count():
    runs = {a: C.compose({"VARIANTS": a}) for a in ("base",) + ARMS}
    assert len({c.hash6 for c in runs.values()}) == len(runs) and len({c.run for c in runs.values()}) == len(runs)
    assert len({tuple(c.overrides) for c in runs.values()}) == len(runs)


@pytest.mark.parametrize("kw, want", [
    ({"VARIANTS": "v8", "BACKBONE": "r50"}, "r50_var-v8"),
    ({"VARIANTS": "v4", "REFINER_KD": "off", "BEV_KD": "off"}, "rkd-off_bkd-off_var-v4"),
    ({"VARIANTS": "lat4", "MAP_KD": "off_ref", "FRAMES": "3", "SEED": "2", "TAG": "t"}, "mapkd-offref_var-lat4_f1_s2_t"),
    ({"VARIANTS": "spd4", "EXTRA": "agent.lr=2e-4"}, "var-spd4_x"),
])
def test_arm_name_part_is_in_the_fixed_order(kw, want):
    c = C.compose(kw)
    assert c.run.rsplit("_", 1)[0] == want
    assert not re.search(r"v2|ck|v7|v15", c.run)


def test_arm_refusals():
    with pytest.raises(C.Refused, match="means nothing"):
        C.compose({"VARIANTS": "v8", "REFINER": "off"})
    with pytest.raises(C.Refused, match="one of base"):
        C.compose({"VARIANTS": "v9"})
    for line in ("++agent.config.ck_e2e2.variants.speeds=[-0.5,0.5]", "agent.config.ck_e2e2.variants.lats=[]",
                 "++agent.config.ck_e2e2.variants.combine=cross", "++agent.config.ck_e2e2.variants.ext=const_curv"):
        with pytest.raises(C.Refused, match="use VARIANTS"):
            C.compose({"EXTRA": line})
    with pytest.raises(C.Refused, match="locked"):
        C.compose({"EXTRA": "++agent.config.ck_e2e2.warmup.var_dir=/x"})


def test_empty_variant_lists_survive_hydra():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    GlobalHydra.instance().clear()
    got = {}
    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/training"), version_base="1.2"):
        for arm in ("spd4", "lat4", "v8", "cross"):
            cfg = compose(config_name="default_training", overrides=C.compose({"VARIANTS": arm}).overrides)
            got[arm] = OmegaConf.to_container(cfg.agent.config.ck_e2e2, resolve=False)
    GlobalHydra.instance().clear()
    for arm, ck in got.items():
        assert ck["variants"]["speeds"] == WANT[arm]["speeds"] and ck["variants"]["lats"] == WANT[arm]["lats"]
        assert V.VariantLayout.from_cfg(ck["variants"]) == V.arm_layout(arm)
        assert ck["warmup"]["var_dir"].endswith(V.arm_var_name(arm))


def test_records_resume_and_variant_args(tmp_path, capsys):
    c = C.compose({"VARIANTS": "v8", "RUN_ROOT": str(tmp_path)})
    C.write_composition(c, str(tmp_path / "gen"))
    assert C.main(["record", "--composition", str(tmp_path / "gen/composition.json"), "--out", str(tmp_path / "s.json")]) == 0
    rec = json.loads((tmp_path / "s.json").read_text())
    assert rec["settings"]["VARIANTS"] == "v8" and C.compose(rec["settings"]).run == c.run
    capsys.readouterr()
    assert C.main(["settings-args", "--settings-json", str(tmp_path / "s.json")]) == 0
    assert "VARIANTS=v8" in capsys.readouterr().out.splitlines()
    # a record without the key (every run before it, and base): the recorded lines only, as before (experiment.sh
    # do_resume sets VARIANTS=base before it reads them)
    b = C.compose({"RUN_ROOT": str(tmp_path)})
    (tmp_path / "b.json").write_text(json.dumps({"settings": b.settings}))
    C.main(["settings-args", "--settings-json", str(tmp_path / "b.json")])
    lines = capsys.readouterr().out.splitlines()
    assert "VARIANTS" not in b.settings and lines == [f"{k}={v}" for k, v in b.settings.items()]
    sh = (REPO / "scripts/experiment.sh").read_text()
    body = sh[sh.index("do_resume() {"):]
    assert body.index("  VARIANTS=base\n") < body.index('settings-args --settings-json "$sj"')
    for arm in ARMS:
        assert C.main(["variant-args", arm]) == 0
        out = capsys.readouterr().out.split()
        w = WANT[arm]
        comb = [] if w["combine"] == "separate" else ["--combine", w["combine"]]
        assert out == [*comb, "--speeds", *map(repr, w["speeds"]), "--lats", *map(repr, w["lats"]), "--name",
                       V.arm_var_name(arm)]
    assert C.main(["variant-args", "base"]) == 3 and C.main(["variant-args", "v9"]) == 3


def test_variant_args_parse_into_the_arm_table(monkeypatch, capsys):
    from tools.ck.data import label_variants as LV
    got = {}
    monkeypatch.setattr(LV, "run", lambda *a, **k: got.update(speeds=a[7], lats=a[8], name=a[10], combine=a[1], ext=k["ext"]))
    flags = {}
    for arm in ARMS:
        C.main(["variant-args", arm])
        flags[arm] = capsys.readouterr().out.split()
    for arm in ARMS:
        LV.main(["--split", "navtrain_train", "--ext", "straight", *flags[arm], "--workers", "48"])   # combine: default
        assert V.VariantLayout(got["speeds"], got["lats"], got["combine"]) == V.arm_layout(arm), arm
        assert got["name"] == V.arm_var_name(arm) and got["ext"] == "straight"


# ----------------------------------------------------------------------------------------------- experiment.sh
def test_script_has_the_setting_in_its_block_and_array():
    t = SH.read_text()
    assert re.search(r"^VARIANTS=base\s+# base \| v8 \| v4 \| spd4 \| lat4", t, flags=re.M)
    arr = re.search(r"^SETTINGS=\(([^)]*)\)", t, flags=re.M).group(1).split()
    assert "VARIANTS" in arr and set(arr) == set(C.DEFAULTS)


def test_script_name_and_refusal():
    for arm in ARMS:
        r = sh("name", "RUN_ROOT=/tmp/never_created_by_this_test", f"VARIANTS={arm}")
        want = C.compose({"VARIANTS": arm, "RUN_ROOT": "/tmp/x"}, registry=str(C.REGISTRY)).run
        assert r.returncode == 0 and r.stdout.strip() == want, r.stderr
    r = sh("name", "VARIANTS=v8", "REFINER=off")
    assert r.returncode == 3 and "means nothing" in r.stderr


STUB_ASSETS = 'import sys\nprint("ARGS", " ".join(sys.argv[1:]))\n'


def _stubs(tmp_path):
    d = tmp_path / "stubs"
    d.mkdir(exist_ok=True)
    (d / "verify_assets.py").write_text(STUB_ASSETS)
    return dict(SSR_RELEASE_TEST_STUBS=str(d), TMPDIR=str(tmp_path))


def test_script_assets_passes_the_arm_to_verify_assets(tmp_path):
    st = _stubs(tmp_path)
    r = sh("assets", f"RUN_ROOT={tmp_path / 'root'}", **st)
    assert r.returncode == 0 and "ARGS --profile full --det-teacher bevfusion --map-teacher resmap" in r.stdout, r.stderr
    assert "--variants" not in r.stdout
    r = sh("assets", f"RUN_ROOT={tmp_path / 'root'}", "VARIANTS=v4", "QUICK=1", **st)
    assert r.returncode == 0 and "--quick --variants v4" in r.stdout, r.stderr
    # RUN=: the arm comes from the run's record; a record without the key is base
    for run, settings, want in (("ra", {"VARIANTS": "lat4"}, "--variants lat4"), ("rb", {}, None)):
        (tmp_path / "root" / run / "launch").mkdir(parents=True)
        (tmp_path / "root" / run / "launch/settings.json").write_text(json.dumps({"profile": "full", "settings": settings}))
        r = sh("assets", f"RUN={run}", f"RUN_ROOT={tmp_path / 'root'}", **st)
        assert r.returncode == 0, r.stderr
        assert (want in r.stdout) if want else ("--variants" not in r.stdout), r.stdout


def test_script_train_passes_the_arm_to_the_asset_check():
    t = SH.read_text()
    body = t.split("do_train() {")[1].split("\n}\n")[0]
    assert '[ "$VARIANTS" = base ] || va=(--variants "$VARIANTS")' in body and '"${va[@]}"' in body


# ----------------------------------------------------------------------------------------------- verify_assets
N = 3


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _label_dir(d: Path, lay, rng, base=None, split="navtrain_train"):
    """A label_variants-like dir of `lay` (N tokens); shared columns copied from `base` (dict of arrays)."""
    d.mkdir(parents=True)
    g = 16 * lay.nv
    arr = {"traj": rng.normal(size=(N, g, 8, 3)).astype(np.float32), "labels": rng.random((N, g, 9)).astype(np.float32),
           "ok": np.ones((N, g), bool)}
    valid = rng.random((N, 16, lay.nv)) > 0.2
    anchor = np.tile(np.arange(16, dtype=np.int16), (N, 1))
    if base is not None:
        bn = list(V.BASE_NAMES)
        for va, nm in enumerate(lay.names):
            if nm in bn:
                vb = bn.index(nm)
                for k in range(16):
                    for a in arr:
                        arr[a][:, k * lay.nv + va] = base[a][:, k * 6 + vb]
                valid[:, :, va] = base["valid"][:, :, vb]
    for a, x in arr.items():
        np.save(d / f"{a}.npy", x)
    cols = np.arange(g)
    np.savez(d / "index.npz", anchor_idx=anchor, valid=valid, names=np.array(lay.names), col_k=(cols // lay.nv).astype(np.int16),
             col_v=(cols % lay.nv).astype(np.int16))
    name = d.name
    (d / "build.json").write_text(json.dumps(dict(
        names=list(lay.names), n_variants=lay.nv, speeds=list(lay.speeds), lats=list(lay.lats), combine=lay.combine,
        ext="straight", subset="sampler", n_anchors=16, epoch=0, seed=0, limit=0, s_on_frac=V.S_ON_FRAC,
        token_sha16="t" * 16, anchors_sha16=VA.ANCHORS_SHA16, n_built=N, traj_sha16=_sha(arr["traj"]))))
    (d / "check.json").write_text(json.dumps(dict(
        name=name, n_rows=N, K=16, V=lay.nv, names=list(lay.names), n_ok=N * g, n_expected=N * g,
        identity_vs_raw256={"all_equal": True}, invalid_vs_parent={"all_equal": True})))
    return dict(arr, valid=valid, anchor=anchor)


@pytest.fixture
def labels_tree(tmp_path, monkeypatch):
    data = tmp_path / "data"
    rng = np.random.default_rng(0)
    root = data / "ck/ck2/labels"
    base = {s: _label_dir(root / s / VA.VAR_NAME, V.BASE_LAYOUT, rng, split=s) for s in VA.N_TOKENS}
    monkeypatch.setattr(VA, "N_TOKENS", {s: N for s in VA.N_TOKENS})
    monkeypatch.setattr(VA, "PINS", {**{f"packed/{s}": {"token_sha16": "t" * 16} for s in base},
                                     **{f"raw256/{s}": {} for s in base},
                                     **{f"var/{s}": {"anchor_idx": _sha(b["anchor"])} for s, b in base.items()}})
    arms = {a: _label_dir(root / "navtrain_train" / V.arm_var_name(a), V.arm_layout(a), rng, base["navtrain_train"]) for a in ARMS}
    return SimpleNamespace(data=data, root=root, base=base, arms=arms, rng=rng)


def _check(tree, arm, quick, split="navtrain_train", soft=False):
    rep = VA.Report()
    c = VA.Ctx({"SSR_DATA_ROOT": str(tree.data)}, quick, 1, rep, "bevfusion", "resmap", None, arm)
    VA.check_arm_labels(c, split, soft)
    return {r["name"]: r for r in rep.rows}


@pytest.mark.parametrize("arm", ARMS)
def test_verify_arm_label_dir_passes_quick_and_full(labels_tree, arm, capsys):
    rows = _check(labels_tree, arm, True)
    assert [r["status"] for r in rows.values()] == ["PASS"], rows
    assert f"V {V.arm_layout(arm).nv}" in rows[f"labels.var-{arm}.navtrain_train"]["detail"]
    rows = _check(labels_tree, arm, False)
    assert {k: r["status"] for k, r in rows.items()} == {f"labels.var-{arm}.navtrain_train": "PASS",
                                                          f"labels.var-{arm}.navtrain_train.shared_labels": "PASS"}, rows
    assert "sha16 labels" in rows[f"labels.var-{arm}.navtrain_train"]["detail"]
    # navtrain_val: only when present (no row), the train dir must be there
    assert _check(labels_tree, arm, True, "navtrain_val", soft=True) == {}
    shutil.rmtree(labels_tree.root / "navtrain_train" / V.arm_var_name(arm))
    rows = _check(labels_tree, arm, True)
    assert rows[f"labels.var-{arm}.navtrain_train"]["status"] == "FAIL" and "50_warmup_labels.sh" in rows[
        f"labels.var-{arm}.navtrain_train"]["fix"]


def _edit_json(p, **kw):
    j = json.loads(p.read_text())
    j.update(kw)
    p.write_text(json.dumps(j))


@pytest.mark.parametrize("break_it, msg", [
    (lambda d: _edit_json(d / "build.json", names=list(V.BASE_NAMES)), "build.json names"),
    (lambda d: _edit_json(d / "build.json", speeds=[-1.0, -0.5, 0.5]), "build.json speeds"),
    (lambda d: _edit_json(d / "build.json", lats=[]), "build.json lats"),
    (lambda d: _edit_json(d / "build.json", ext="const_curv"), "build.json ext"),
    (lambda d: _edit_json(d / "build.json", token_sha16="x" * 16), "build.json token_sha16"),
    (lambda d: _edit_json(d / "check.json", n_ok=1), "n_ok 1"),
    (lambda d: _edit_json(d / "check.json", invalid_vs_parent={"all_equal": False}), "invalid_vs_parent"),
    (lambda d: _edit_json(d / "check.json", identity_vs_raw256={}), "identity_vs_raw256"),
    (lambda d: np.save(d / "ok.npy", np.ones((N, 96), bool)), "ok.npy: shape"),
    (lambda d: (d / "check.json").unlink(), "check.json missing"),
])
def test_verify_arm_label_dir_catches(labels_tree, break_it, msg):
    """v4, spd4 and lat4 have the same 80 columns: only the recorded table tells them apart."""
    d = labels_tree.root / "navtrain_train" / V.arm_var_name("v4")
    break_it(d)
    r = _check(labels_tree, "v4", True)["labels.var-v4.navtrain_train"]
    assert r["status"] == "FAIL" and msg in r["detail"], r


def test_verify_cross_needs_a_cross_build(labels_tree):
    """cross = v8's speeds / lats + combos: a build.json saying 'separate' (a v8-like build) is refused."""
    d = labels_tree.root / "navtrain_train" / V.arm_var_name("cross")
    assert _check(labels_tree, "cross", True)["labels.var-cross.navtrain_train"]["status"] == "PASS"
    _edit_json(d / "build.json", combine="separate")
    r = _check(labels_tree, "cross", True)["labels.var-cross.navtrain_train"]
    assert r["status"] == "FAIL" and "build.json combine 'separate' != 'cross'" in r["detail"], r


def test_verify_another_80_column_arm_in_the_dir_fails(labels_tree):
    d = labels_tree.root / "navtrain_train"
    shutil.rmtree(d / V.arm_var_name("spd4"))
    shutil.copytree(d / V.arm_var_name("lat4"), d / V.arm_var_name("spd4"))
    r = _check(labels_tree, "spd4", True)["labels.var-spd4.navtrain_train"]
    assert r["status"] == "FAIL" and "names" in r["detail"]


def test_verify_full_check_compares_with_the_base_build(labels_tree):
    d = labels_tree.root / "navtrain_train" / V.arm_var_name("v8")
    t0 = np.load(d / "traj.npy")
    t = t0.copy()
    t[1, 9 * 3 + 0] += 1.0                          # identity column of parent 3: shared with base
    np.save(d / "traj.npy", t)
    _edit_json(d / "build.json", traj_sha16=_sha(t))
    r = _check(labels_tree, "v8", False)["labels.var-v8.navtrain_train"]
    assert r["status"] == "FAIL" and "traj: 1 rows of the shared columns" in r["detail"]
    t = t0.copy()
    t[2, 9 * 3 + 4] += 1.0                          # a+1.0: not in base, free; but traj.npy is no longer the build's
    np.save(d / "traj.npy", t)
    r = _check(labels_tree, "v8", False)["labels.var-v8.navtrain_train"]
    assert r["status"] == "FAIL" and "!= build.json traj_sha16" in r["detail"] and "shared columns" not in r["detail"]
    _edit_json(d / "build.json", traj_sha16=_sha(t))
    assert _check(labels_tree, "v8", False)["labels.var-v8.navtrain_train"]["status"] == "PASS"
    lab = np.load(d / "labels.npy")
    lab[0, 9 * 5 + 2] += 0.5                        # a-0.5 label of parent 5 differs from base: a WARN only
    np.save(d / "labels.npy", lab)
    rows = _check(labels_tree, "v8", False)
    assert rows["labels.var-v8.navtrain_train"]["status"] == "PASS"
    assert rows["labels.var-v8.navtrain_train.shared_labels"]["status"] == "WARN"
    with np.load(d / "index.npz") as z:
        x = dict(z)
    x["anchor_idx"] = x["anchor_idx"][:, ::-1].copy()
    np.savez(d / "index.npz", **x)
    r = _check(labels_tree, "v8", False)["labels.var-v8.navtrain_train"]
    assert r["status"] == "FAIL" and "anchor_idx" in r["detail"]


def test_verify_base_adds_no_row_and_the_cli_takes_an_arm(labels_tree, monkeypatch, capsys):
    rep = VA.Report()
    c = VA.Ctx({"SSR_DATA_ROOT": str(labels_tree.data)}, True, 1, rep, "bevfusion", "resmap", None)
    assert c.variants == "base"
    monkeypatch.setattr(VA, "check_arm_labels", lambda *a, **k: pytest.fail("base must not check an arm dir"))
    VA.check_labels(c, "navtrain_train")
    assert not any(r["name"].startswith("labels.var-") for r in rep.rows)
    with pytest.raises(SystemExit):
        VA.main(["--profile", "base", "--variants", "v9"])


REAL_LABELS = Path(SP.SSR_DATA_ROOT) / "ck/ck2/labels"


@pytest.mark.parametrize("arm, split", [(a, s) for a in ARMS for s in ("navtrain_train", "navtrain_val")])
def test_real_arm_label_dirs_pass_the_quick_check(arm, split):
    d = REAL_LABELS / split / V.arm_var_name(arm)
    if not (d / "check.json").is_file():
        pytest.skip(f"{d} not built (yet)")
    rep = VA.Report()
    c = VA.Ctx(SP.resolve(), True, 4, rep, "bevfusion", "resmap", None, arm)
    VA.check_arm_labels(c, split, split == "navtrain_val")
    assert [r["status"] for r in rep.rows] == ["PASS"], rep.rows


# ----------------------------------------------------------------------------------------------- smoke_check / check_build
def _smoke_run(tmp_path, eff=None, **kw):
    from test_release_compose import ALL_KEYS, MAIN_KEYS, T_ON, _calib_file, _common_files, _refiner_files, _settings
    run = tmp_path / "r"
    _settings(run, **kw)
    _common_files(run, MAIN_KEYS)
    _refiner_files(run, T_ON)
    _calib_file(run, ALL_KEYS)
    if eff is not None:
        (run / "launch/refiner_effective.json").write_text(json.dumps(eff))
    return run


def _eff(arm):
    c = C.compose({"VARIANTS": arm})
    return L2.effective_ck(c.run_dir, 8, c.set_lines, c.extra_lines)


def test_smoke_check_base_is_unchanged(tmp_path):
    res = SC.check_run(str(_smoke_run(tmp_path, eff=_eff("base"))))
    assert res["pass"] and "variants_as_set" not in res["checks"] and "variants" not in res["info"]


@pytest.mark.parametrize("arm", ARMS)
def test_smoke_check_arm_uses_its_table_and_label_dir(tmp_path, arm):
    res = SC.check_run(str(_smoke_run(tmp_path, eff=_eff(arm), VARIANTS=arm)))
    assert res["pass"] and res["checks"]["variants_as_set"], res
    assert res["info"]["variants"]["n_cand"] == 16 * WANT[arm]["nv"]
    other = "v8" if arm != "v8" else "v4"
    eff = _eff(arm)
    eff["variants"] = _eff(other)["variants"]
    (tmp_path / "r/launch/refiner_effective.json").write_text(json.dumps(eff))
    assert not SC.check_run(str(tmp_path / "r"))["checks"]["variants_as_set"]
    eff = _eff(arm)
    eff["warmup"] = _eff("base")["warmup"]
    (tmp_path / "r/launch/refiner_effective.json").write_text(json.dumps(eff))
    assert not SC.check_run(str(tmp_path / "r"))["checks"]["variants_as_set"]
    (tmp_path / "r/launch/refiner_effective.json").unlink()
    assert not SC.check_run(str(tmp_path / "r"))["checks"]["variants_as_set"]


def _fake_agent(variants):
    cfg = SimpleNamespace(image_architecture="resnet34.tv_in1k", frame_indices=[2, 3], bev_h=100, bev_w=100,
                          pc_range=[-32.0, 0.0, -1.0, 32.0, 64.0, 3.0], use_task_interaction=True, use_det_motion_head=True,
                          use_map_head=True, map_head_type="raster", undistort=True, plan_heading_from_xy=True)
    ck = SimpleNamespace(use_teachers=True, teacher_det_run="d", teacher_map_run="m", kd_sources=["det", "map"],
                         kd_score_keys=["nc"], lambda_kd_score=0.5, lat_kd={"balance": "ema", "fixed": 1.0},
                         kd_calib={"enabled": True, "mode": "online"}, infer_select=None, submit_selection=False, seed=0,
                         variants=variants, warmup={"var_dir": "/x/var"})
    return SimpleNamespace(_config=cfg, named_parameters=lambda: iter(()), get_target_builders=lambda: [],
                           para_ssr_model=SimpleNamespace(_backbone_bn_mode="moving"), _distill=None,
                           _ck_e2e2=SimpleNamespace(cfg=ck))


def test_check_build_records_the_arm_only_off_base():
    from tools.release import check_build as CB
    rec = CB.summarize(_fake_agent(dict(_eff("base")["variants"])))
    assert "variants" not in rec["refiner"]
    for arm in ARMS:
        rec = CB.summarize(_fake_agent(dict(_eff(arm)["variants"])))
        assert rec["refiner"]["variants"]["nv"] == WANT[arm]["nv"] and rec["refiner"]["variants"]["n_cand"] == 16 * WANT[arm]["nv"]


# ----------------------------------------------------------------------------------------------- labeler workers
def test_labeler_workers_default_stays_and_an_explicit_value_may_exceed_48():
    from tools.ck.e2e import labeler as L1
    t = (REPO / "tools/ck/e2e2/train_e2e2.sh").read_text()
    m = re.search(r'^  if \[ -n "\$\{LABELER_WORKERS:-\}" \]; then.*?fi$', t, flags=re.M | re.S)
    assert m, "the labeler worker rule moved"
    expr = "ng=$1; " + m.group(0) + "; echo $lw"
    env = {k: v for k, v in os.environ.items() if k != "LABELER_WORKERS"}

    def lw(ng, explicit=None):
        e = dict(env, LABELER_WORKERS=explicit) if explicit else env
        return int(subprocess.run(["bash", "-c", expr, "_", str(ng)], env=e, capture_output=True, text=True).stdout)
    assert (lw(2), lw(4), lw(8)) == (32, 48, 48)          # the recipe: 48 on 8 GPUs, as before
    assert (lw(8, "24"), lw(8, "72"), lw(8, "96"), lw(8, "192"), lw(8, "200")) == (24, 72, 96, 192, 192)
    assert L1.MAX_WORKERS == 192                          # VARIANTS=cross: 400 trajectories / token (256-CPU server)


def test_resume_restores_the_recorded_labeler_workers(tmp_path):
    """experiment.sh do_resume: LABELER_WORKERS unset -> the settings.json 'labeler_workers' of the start (e.g. 72 for
    a v8 run started with LABELER_WORKERS=72); null (every run started without it, the base default) -> stays unset,
    as before; given on the resume -> kept."""
    t = (REPO / "scripts/experiment.sh").read_text()
    m = re.search(r'^  if \[ -z "\$\{LABELER_WORKERS:-\}" \]; then\n.*?^  fi$', t, flags=re.M | re.S)
    assert m, "the resume labeler-worker rule moved"
    pre = ('py() { PYTHONDONTWRITEBYTECODE=1 "%s" "$@"; }; say() { echo "[experiment] $*"; }; '
           'die() { echo "REFUSED: $*" >&2; exit 3; }; f() { local sj=$1 RUN=r; ' % sys.executable)
    expr = pre + m.group(0) + '\n}; f "$1"; echo "lw=${LABELER_WORKERS:-}"'
    env = {k: v for k, v in os.environ.items() if k != "LABELER_WORKERS"}

    def run(rec, given=None):
        sj = tmp_path / "settings.json"
        sj.write_text(json.dumps(rec))
        e = dict(env, LABELER_WORKERS=given) if given else env
        r = subprocess.run(["bash", "-c", expr, "_", str(sj)], env=e, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()
    assert run({"settings": {}, "labeler_workers": None}) == ["lw="]
    assert run({"settings": {}}) == ["lw="]                                 # records written before the key
    assert run({"settings": {}, "labeler_workers": "72"}) == ["[experiment] LABELER_WORKERS=72 (recorded at the start of r)",
                                                              "lw=72"]
    assert run({"settings": {}, "labeler_workers": "72"}, given="24") == ["lw=24"]


# ----------------------------------------------------------------------------------------------- setup scripts
COPY = ("tools/release/setup/_common.sh", "tools/release/setup/50_warmup_labels.sh", "tools/release/setup/run_refiner_teachers.sh",
        "tools/release/setup/teacher_env.py", "tools/release/load_paths.sh", "tools/release/compose.py",
        "tools/release/verify_assets.py", "tools/release/registry.yaml", "tools/ck/__init__.py", "tools/ck/e2e/__init__.py",
        "tools/ck/e2e/launch_util.py", "tools/ck/e2e2/__init__.py", "tools/ck/e2e2/launch_util2.py", "navsim/__init__.py",
        "navsim/common/__init__.py", "navsim/common/ssr_paths.py", "navsim/agents/para_ssr/ck/variants.py")


@pytest.fixture
def setup_tree(tmp_path):
    """The setup scripts in a copy of the repo without tools/release/paths.env (the server's own file would win over the
    test roots), with roots under tmp_path."""
    repo = tmp_path / "repo"
    for rel in COPY:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    srv = tmp_path / "srv"
    (srv / "ds/navsim_logs/trainval").mkdir(parents=True)
    (srv / "ds/navsim_logs/trainval/x.pkl").write_bytes(b"")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SSR_", "NAVSIM_", "OPENSCENE", "NUPLAN", "SPLITS", "ARMS",
                                                                      "VAR_NAME", "N_VAR", "RESUME"))}
    env.update(SSR_DATA_ROOT=str(srv / "data"), SSR_TEACHER_CACHE=str(srv / "tc"), OPENSCENE_DATA_ROOT=str(srv / "ds"),
               SSR_TEST_DATA_ROOT=str(srv / "dst"), SSR_PLAN_DATA=str(srv / "plan"), SSR_PY=sys.executable,
               SSR_TORCHRUN="/bin/echo", SSR_RUN_ROOT=str(srv / "runs"), SETUP_DRY="1", PYTHONDONTWRITEBYTECODE="1")
    assert not (repo / "tools/release/paths.env").exists()
    return SimpleNamespace(repo=repo, srv=srv, env=env)


def _run(tree, name, **kw):
    r = subprocess.run(["bash", str(tree.repo / "tools/release/setup" / name)], env=dict(tree.env, **kw), capture_output=True,
                       text=True, timeout=300)
    return r, [x for x in r.stdout.splitlines() if x.startswith("+ ")]


def test_warmup_script_default_is_unchanged(setup_tree):
    r, g = _run(setup_tree, "50_warmup_labels.sh", SPLITS="navtrain_train")
    assert r.returncode == 0, r.stderr
    v = [x for x in g if "label_variants.py" in x]
    assert len(v) == 1 and v[0].endswith("--combine separate --ext straight --name var_separate_sampler16_accstraight --workers 48")
    assert [x for x in g if "verify_assets.py" in x] == [x for x in g if x.endswith("--profile refiner_noteacher --only labels")]
    assert any("make_raw256_traj.py --split navtrain_train" in x for x in g) and any("label_cands.py" in x for x in g)


def test_warmup_script_builds_the_arms(setup_tree):
    r, g = _run(setup_tree, "50_warmup_labels.sh", SPLITS="navtrain_train", ARMS=" ".join(ARMS))
    assert r.returncode == 0, r.stderr
    v = [x.split("label_variants.py ", 1)[1] for x in g if "label_variants.py" in x]
    assert len(v) == 1 + len(ARMS)
    for arm, line in zip(ARMS, v[1:]):
        w = WANT[arm]
        want = " ".join([f"--split navtrain_train --combine {w['combine']} --ext straight --speeds", *map(repr, w["speeds"]), "--lats",
                         *map(repr, w["lats"]), "--name", V.arm_var_name(arm), "--workers 48"])
        assert line == want, (line, want)
    va = [x for x in g if "verify_assets.py" in x]
    assert [x.rsplit(" ", 1)[1] for x in va[1:]] == list(ARMS) and all("--only labels --variants" in x for x in va[1:])


def test_warmup_script_keeps_finished_base_outputs_and_guards_the_arm_dirs(setup_tree):
    ck = setup_tree.srv / "data/ck"
    for p in (ck / "labels/navtrain_train/raw256/labels.npy", ck / "ck2/labels/navtrain_train/var_separate_sampler16_accstraight/labels.npy"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    r, _ = _run(setup_tree, "50_warmup_labels.sh", SPLITS="navtrain_train")
    assert r.returncode == 3 and "refusing to overwrite" in r.stderr           # as before
    r, g = _run(setup_tree, "50_warmup_labels.sh", SPLITS="navtrain_train", ARMS="lat4")
    assert r.returncode == 0 and "kept, only the arms" in r.stdout, r.stderr
    assert not any("make_raw256_traj" in x or "label_cands" in x for x in g)
    assert [x for x in g if "label_variants.py" in x][0].endswith("--name var_separate_sampler16_accstraight_lat4 --workers 48")
    ad = ck / "ck2/labels/navtrain_train" / V.arm_var_name("lat4")
    ad.mkdir()                                       # a build in progress: never a second writer without RESUME=1
    r, _ = _run(setup_tree, "50_warmup_labels.sh", SPLITS="navtrain_train", ARMS="lat4")
    assert r.returncode == 3 and "RESUME=1" in r.stderr
    r, _ = _run(setup_tree, "50_warmup_labels.sh", SPLITS="navtrain_train", ARMS="lat4", RESUME="1")
    assert r.returncode == 0, r.stderr
    (ad / "labels.npy").write_text("x")
    r, _ = _run(setup_tree, "50_warmup_labels.sh", SPLITS="navtrain_train", ARMS="lat4", RESUME="1")
    assert r.returncode == 3 and "refusing to overwrite" in r.stderr
    for bad in ("base", "v9"):
        r, _ = _run(setup_tree, "50_warmup_labels.sh", SPLITS="navtrain_train", ARMS=bad)
        assert r.returncode == 3 and "not a variant arm" in r.stderr


def _teacher_reg(tree):
    reg = tree.srv / "reg.yaml"
    reg.write_text("""teachers:
  det:
    bevfusion: {status: ready, layout: per_token_npz, cache_train: /c/t, cache_val: /c/v, sha_head: h, refiner_run: /o/det}
  map:
    resmap: {status: ready, layout: sharded, cache_root: /m, sha256: s, refiner_run: /o/map}
""")
    return str(reg)


def test_teacher_script_default_is_unchanged_and_takes_var_name_and_n_var(setup_tree):
    reg = _teacher_reg(setup_tree)
    r, g = _run(setup_tree, "run_refiner_teachers.sh", SSR_RELEASE_REGISTRY=reg)
    assert r.returncode == 0, r.stderr
    cmds = [x for x in g if "train_ck2.py" in x]
    assert len(cmds) == 2 and not any("--var-name" in x or "--n-var" in x for x in cmds)
    assert all("--r34-evals 0 --resume --score-keys" in x for x in cmds)
    assert "ck2/labels/navtrain_train/var_separate_sampler16_accstraight/labels.npy" in r.stdout
    r, g = _run(setup_tree, "run_refiner_teachers.sh", SSR_RELEASE_REGISTRY=reg, VAR_NAME=V.arm_var_name("v8"), N_VAR="128")
    assert r.returncode == 0, r.stderr
    cmds = [x for x in g if "train_ck2.py" in x]
    assert all(f"--resume --var-name {V.arm_var_name('v8')} --n-var 128 --score-keys" in x for x in cmds)
    assert f"ck2/labels/navtrain_val/{V.arm_var_name('v8')}/labels.npy" in r.stdout
    r, _ = _run(setup_tree, "run_refiner_teachers.sh", SSR_RELEASE_REGISTRY=reg, VAR_NAME=V.arm_var_name("v4"))
    assert r.returncode == 3 and "needs N_VAR" in r.stderr
