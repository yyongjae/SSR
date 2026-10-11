"""refiner-v3 tooling fixes 2026-10-11 (docs/refiner_v3/REVIEW_FIXES.md section 5), CPU only, temp files under /tmp.

  * F2 infer_val_v3 cache key: key fields, check_cache refusals (unknown / other key), and a real CPU inference on 8
    rows of ck2T: same-key rerun reuses rows, --which ep28 into the same --out and a dir without cache_key.json are
    refused without touching the outputs, decide's cache / ckpt checks accept the dump;
  * F3 pretrain_forecaster resume: an interrupted run resumed with the recorded settings equals the uninterrupted run
    bitwise; another --batch / --limit-tokens / --max-steps is refused; the pre-fix code (commit aa7a7b2) resumed with
    another --batch silently drops samples (the old failure, reproduced);
  * F4 build_obj_pack part key: an interrupted build resumed after MATCH_GATE_M changed rebuilds every chunk instead
    of mixing; an unchanged resume reuses them; existing complete packs (no fingerprint) still load;
  * review 2 (2026-10-12): default --out per --which, finished-run guard, post-load ckpt sha check, fc_metrics sums
    merged on a same-key resume, and decide_step1.run with arm checks on full-size a / b stand-ins + the real legacy
    arm (c): forward-code check between arms, non-pre-registered bootstrap = INVALID.

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider \
    tests/test_v3_tooling_fixes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
CK = str(Path(__file__).resolve().parents[1])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck.temporal import config as TC  # noqa: E402
from tools.ck.v3 import decide_step1 as DS  # noqa: E402
from tools.ck.v3 import infer_val_v3 as IV  # noqa: E402

PK = TC.PACK_ROOT / "bevfusion"
have_train_pack = pytest.mark.skipif(not (PK / "navtrain_train" / "build.json").is_file()
                                     or not Path(TC.TemporalCfg().ca_scales).is_file(),
                                     reason="K96 train pack / ca_scales absent")
CK2T_RUN = DS.CK / "ck2" / "train" / "ck2T"
have_ck2T = pytest.mark.skipif(not ((CK2T_RUN / "ckpt_last.pt").is_file() and (CK2T_RUN / "ckpt_ep28.pt").is_file()
                                    and (DS.PACK / "cand.npy").is_file() and DS.VARIANTS.is_file()),
                               reason="ck2T run / packed val / variants absent")
slow = pytest.mark.skipif(os.environ.get("V3_SKIP_SLOW") == "1", reason="slow skipped")
OLD_COMMIT = "aa7a7b2"           # refiner-v3 tooling before the 2026-10-11 fixes


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------- F2 infer cache key
def test_key_diff_and_check_cache(tmp_path):
    k1 = {"version": "v3_infer_cache_1", "which": "last", "ckpt_sha256": "a" * 64, "code": {"sha256": "c", "n": 1}}
    assert IV.key_diff(k1, json.loads(json.dumps(k1))) == []
    assert IV.key_diff(k1, dict(k1, which="ep7", code={"sha256": "d", "n": 1})) == ["code.sha256", "which"]
    out = tmp_path / "dump"
    assert IV.check_cache(out, k1) is False                                 # no dir yet: nothing to reuse
    out.mkdir()
    np.save(out / "done.npy", np.zeros(10, bool))
    assert IV.check_cache(out, k1) is False                                 # no done rows: any key is fine
    done = np.zeros(10, bool)
    done[:3] = True
    np.save(out / "done.npy", done)
    with pytest.raises(SystemExit, match="no cache_key.json"):              # pre-fix dump with rows: refused
        IV.check_cache(out, k1)
    (out / "cache_key.json").write_text(json.dumps(k1))
    assert IV.check_cache(out, k1) is True                                  # same key: resume allowed
    with pytest.raises(SystemExit, match=r"ckpt_sha256.*new --out"):
        IV.check_cache(out, dict(k1, ckpt_sha256="b" * 64))


def test_import_closure_static(tmp_path):
    """function-level and relative imports are followed, parent packages included, other roots ignored."""
    r = tmp_path / "repo"
    for f, src in {"tools/__init__.py": "", "tools/a/__init__.py": "", "navsim/__init__.py": "",
                   "navsim/m/__init__.py": "from . import k\n",
                   "tools/a/entry.py": "import os\nfrom navsim.m import x\ndef f():\n    from .lazy import g\n",
                   "tools/a/lazy.py": "from ..b.deep import h\n", "tools/b/__init__.py": "", "tools/b/deep.py": "",
                   "navsim/m/x.py": "", "navsim/m/k.py": "", "navsim/m/unused.py": "", "tests/t.py": ""}.items():
        (r / f).parent.mkdir(parents=True, exist_ok=True)
        (r / f).write_text(src)
    got = IV.import_closure("tools/a/entry.py", repo=r)
    assert got == sorted(["tools/__init__.py", "tools/a/__init__.py", "tools/a/entry.py", "tools/a/lazy.py",
                          "tools/b/__init__.py", "tools/b/deep.py", "navsim/__init__.py", "navsim/m/__init__.py",
                          "navsim/m/x.py", "navsim/m/k.py"])


@have_ck2T
def test_cache_key_fields(tmp_path):
    pdir = DS.PACK
    k = IV.cache_key(CK2T_RUN, "last", CK2T_RUN / "ckpt_last.pt", None, False, "fp32 (cpu)", {}, pdir)
    assert k["version"] == IV.CACHE_KEY_VERSION and k["version"].startswith(DS.CACHE_KEY_PREFIX)
    assert k["ckpt_sha256"][:16] == DS.sha16(CK2T_RUN / "ckpt_last.pt")
    assert k["packed"]["cand.npy"][:16] == DS.sha16(pdir / "cand.npy") and set(k["packed"]) == set(IV.PACKED_KEY_FILES)
    assert k["variants_sha256"][:16] == DS.sha16(DS.VARIANTS) and k["temporal"] == {"on": False}
    files = k["code"]["files"]
    for f in ("navsim/agents/para_ssr/ck/model.py", "navsim/agents/para_ssr/refiner/refiner_net.py",
              "navsim/agents/para_ssr/ck/temporal/branch.py", "navsim/agents/para_ssr/ck/temporal/forecast.py",
              "navsim/agents/para_ssr/ck/temporal/raster.py", "tools/ck/data/ck_dataset.py",
              "tools/ck/data/obj_pack.py", "tools/ck/ckutil.py", "tools/ck/v3/infer_val_v3.py",
              "navsim/agents/para_ssr/ck/__init__.py"):
        assert f in files, f
    assert set(IV.MODEL_FILES) <= set(files)
    for f in ("navsim/agents/para_ssr/ck/online2.py", "navsim/agents/para_ssr/ck/e2e_data2.py", "tools/ck/train_ck2.py"):
        assert f not in files, f                                            # unrelated code: no spurious refusals
    k28 = IV.cache_key(CK2T_RUN, "ep28", CK2T_RUN / "ckpt_ep28.pt", None, False, "fp32 (cpu)", {}, pdir)
    assert {"which", "ckpt_name", "ckpt_sha256"} <= set(IV.key_diff(k, k28))
    assert IV.key_diff(k, IV.cache_key(CK2T_RUN, "last", CK2T_RUN / "ckpt_last.pt", None, False, "fp16 autocast",
                                       {}, pdir)) == ["amp"]


@have_ck2T
@slow
def test_real_infer_cache_reuse_and_refusals(tmp_path):
    """8 ck2T rows on CPU into one --out: a same-key rerun reuses them (one more session, same key); --which ep28 into
    that --out is refused and leaves every output untouched (the STEP1_RUN ep7 / ep8 case); without cache_key.json
    the done rows are refused; decide's ckpt / cache checks accept the dump."""
    out = tmp_path / "ck2T_last"
    base = ["--run", str(CK2T_RUN), "--device", "cpu", "--limit", "8", "--workers", "0", "--out", str(out)]
    IV.main(base)
    meta = json.loads((out / "infer_meta.json").read_text())
    key = json.loads((out / "cache_key.json").read_text())
    assert meta["cache_key"] == key and meta["ckpt_sha16"] == key["ckpt_sha256"][:16] == DS.sha16(meta["ckpt"])
    assert Path(meta["ckpt"]).is_absolute() and np.load(out / "done.npy").sum() == 8
    IV.main(base)                                                           # same key: nothing recomputed
    meta2 = json.loads((out / "infer_meta.json").read_text())
    assert [s["n"] for s in meta2["sessions"]] == [8, 0] and meta2["cache_key"] == key
    snap = {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()}
    with pytest.raises(SystemExit, match=r"other cache key.*which"):
        IV.main(base[:2] + ["--which", "ep28"] + base[2:])
    assert {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()} == snap   # refused before any write
    ref = DS.reference_shas()
    assert DS.ckpt_problems("c", meta2) == [] and DS.cache_problems("c", meta2, ref, key) == []
    assert DS.cache_mode(meta2) == "cache_key"
    (out / "cache_key.json").unlink()                                      # a pre-fix dump with done rows
    with pytest.raises(SystemExit, match="no cache_key.json"):
        IV.main(base)
    assert any("cache_key.json missing" in x for x in DS.cache_problems("c", meta2, ref, None))


# ----------------------------------------------------------------------------------------------- review 2 (infer / decide)
A_RUN = TC.D3 / "train" / "v3T10_a_learned"
ARM_C = DS.D3 / "infer" / "ck2T10"
have_arm_a = pytest.mark.skipif(not ((A_RUN / "done.json").is_file() and (A_RUN / "ckpt_last.pt").is_file()
                                     and (ARM_C / "infer_meta.json").is_file() and (ARM_C / "prob80.npy").is_file()
                                     and (PK / "navtrain_val" / "build.json").is_file()),
                                reason="finished arm-(a) run / arm-(c) dump / K96 val pack absent")


def test_default_out_name():
    """review 2 finding 4: a non-last checkpoint never defaults to the decision dump's dir."""
    assert IV.default_out_name("v3T10_a_learned", "last") == "v3T10_a_learned"
    assert IV.default_out_name("v3T10_a_learned", "ep7") == "v3T10_a_learned_ep7"


class _Boom(Exception):
    pass


def test_infer_refuses_unfinished_run_and_changed_ckpt(tmp_path, monkeypatch):
    """--which last needs done.json (--allow-unfinished for checks; other checkpoints need none); a checkpoint whose
    sha after the load differs from the key's (replaced while loading, review 2 finding 6) is refused; nothing is
    written in either case."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"arm": "T", "run": "fake"}))
    (run / "ckpt_last.pt").write_bytes(b"w-last")
    (run / "ckpt_ep7.pt").write_bytes(b"w-ep7")
    out = tmp_path / "out"
    args = ["--run", str(run), "--device", "cpu", "--workers", "0", "--out", str(out)]

    def boom(*a, **k):
        raise _Boom()

    monkeypatch.setattr(IV, "load_model", boom)
    with pytest.raises(SystemExit, match="no done.json"):
        IV.main(args)
    with pytest.raises(_Boom):                       # passes the guard, reaches the load
        IV.main(args + ["--allow-unfinished"])
    with pytest.raises(_Boom):
        IV.main(args + ["--which", "ep7"])
    (run / "done.json").write_text("{}")

    class Net:
        init_report = {"ckpt_sha16": "0" * 16}

        def eval(self):
            return self

    monkeypatch.setattr(IV, "load_model", lambda *a, **k: (Net(), {"arm": "T"}, None))
    with pytest.raises(SystemExit, match="changed while it was loaded"):
        IV.main(args)
    Net.init_report = {"ckpt_sha16": IV.sha(run / "ckpt_last.pt")}
    with pytest.raises(SystemExit, match="arm"):     # same sha: passes the check (then fails on the fake config)
        monkeypatch.setattr(IV, "load_model", lambda *a, **k: (Net(), {"arm": "X"}, None))
        IV.main(args)
    assert not out.exists()


def test_merge_fc_sums():
    """review 2 finding 5: fc_metrics.json keeps the additive sums; a same-key resume adds them up."""
    from navsim.agents.para_ssr.ck.temporal.metrics import ForecastMetrics

    def fm(x):
        m = ForecastMetrics()
        m.sums = {k: float(x + i) for i, k in enumerate(m.KEYS)}
        return m

    old = json.loads(json.dumps(IV.fc_payload({"arm": fm(1), "ca_reference": fm(10)}, 4)))
    assert old["n_rows"] == 4 and set(old["sums"]) == {"arm", "ca_reference"}
    got, n, merged = IV.merge_fc(old, {"arm": fm(2), "ca_reference": fm(20)}, 3)
    assert merged and n == 7
    for g, a0, b0 in (("arm", 1, 2), ("ca_reference", 10, 20)):
        assert got[g].sums == {k: float(a0 + i) + float(b0 + i) for i, k in enumerate(ForecastMetrics.KEYS)}
    ref = ForecastMetrics()
    ref.sums = {k: float(1 + i) + float(2 + i) for i, k in enumerate(ForecastMetrics.KEYS)}
    assert IV.fc_payload(got, n)["arm"] == ref.result()
    new = {"arm": fm(2), "ca_reference": fm(20)}
    assert IV.merge_fc(None, new, 3)[1:] == (3, False)
    assert IV.merge_fc({"n_rows": 4, "arm": {}}, new, 3)[1:] == (3, False)            # file without sums
    bad = json.loads(json.dumps(old))
    bad["sums"]["arm"].pop("n_obj")
    assert IV.merge_fc(bad, new, 3)[1:] == (3, False)


@have_arm_a
@slow
def test_real_temporal_fc_merge_and_decide_code_check(tmp_path):
    """Real arm (a) (finished v3T10_a_learned) on CPU: 4 rows then 8 rows into one --out merge the forecast sums to
    those of a fresh 8-row run; then decide_step1.run WITH arm checks on full-size a / b dirs (arrays of the arm-(c)
    dump, copied to /tmp; meta + cache key of that CPU infer, b relabelled forecaster 'ca', check.ok set) and the real
    legacy arm (c): valid, final, legacy c, code check clean; another bootstrap -> INVALID; b scored by other code ->
    refused naming the file."""
    two, fresh = tmp_path / "a_two", tmp_path / "a_fresh"
    base = ["--run", str(A_RUN), "--device", "cpu", "--workers", "0"]
    IV.main(base + ["--limit", "4", "--out", str(two)])
    f4 = json.loads((two / "fc_metrics.json").read_text())
    assert f4["n_rows"] == 4 and f4["n_sessions_merged"] == 1
    IV.main(base + ["--limit", "8", "--out", str(two)])
    IV.main(base + ["--limit", "8", "--out", str(fresh)])
    f8, ff = (json.loads((d / "fc_metrics.json").read_text()) for d in (two, fresh))
    assert f8["n_rows"] == ff["n_rows"] == 8 and f8["n_sessions_merged"] == 2 and ff["n_sessions_merged"] == 1
    for g in ("arm", "ca_reference"):
        for k, v in ff["sums"][g].items():
            assert f8["sums"][g][k] == pytest.approx(v, rel=1e-4, abs=1e-6), (g, k)
    meta = json.loads((two / "infer_meta.json").read_text())
    key = json.loads((two / "cache_key.json").read_text())
    assert [x["n"] for x in meta["sessions"]] == [4, 4] and meta["run_done"] is True and meta["cache_key"] == key

    def arm_dir(name, forecaster, code=None):
        d = tmp_path / name
        d.mkdir()
        for f in ("prob16.npy", "prob80.npy", "done.npy", "tokens.parquet"):
            shutil.copyfile(ARM_C / f, d / f)
        m, k = json.loads(json.dumps(meta)), json.loads(json.dumps(key))
        m["temporal"]["forecaster"] = k["temporal"]["forecaster"] = forecaster
        if code is not None:
            k["code"] = code
        m.update(cache_key=k, teacher=name, n_rows=18179, n_done=18179,
                 check={"ok": True, "rule": "test stand-in for the GPU reproduce check"})
        (d / "infer_meta.json").write_text(json.dumps(m))
        (d / "cache_key.json").write_text(json.dumps(k))
        shutil.copyfile(two / "fc_metrics.json", d / "fc_metrics.json")
        return d

    da, db = arm_dir("syn_a", "learned"), arm_dir("syn_b", "ca")
    res = DS.run({"a": str(da), "b": str(db), "c": str(ARM_C)}, tmp_path / "step1.json")
    assert res["valid_for_decision"] and res["final"] and res["preregistered_bootstrap"]
    assert res["decision"] == "NO-GO"                       # a = b = c scores: delta 0
    assert res["legacy_dumps"] == ["c"] and res["inputs"]["a"]["cache"]["mode"] == "cache_key"
    cc = res["code_check"]
    assert cc["errors"] == [] and cc["ref"] == "a" and cc["sha256"]["a"] == cc["sha256"]["b"] == key["code"]["sha256"]
    assert set(cc["legacy_compared"]["c"]["files"]) == set(IV.MODEL_FILES)
    assert any("amp" in w for w in cc["warnings"])          # CPU fp32 stand-ins vs the fp16 GPU arm (c)
    assert any("arm a: forecast metrics" in x and "cover 8 rows" in x for x in res["notes"])
    print("legacy c differing model files:", cc["legacy_compared"]["c"]["differ"])
    r50 = DS.run({"a": str(da), "b": str(db), "c": str(ARM_C)}, None, n_boot=50)
    assert r50["decision"] == "INVALID (would be NO-GO)" and not r50["final"] and not r50["preregistered_bootstrap"]
    code = json.loads(json.dumps(key["code"]))
    f = "navsim/agents/para_ssr/ck/model.py"
    code["files"][f] = "0" * 16
    code["sha256"] = "f" * 64
    dbx = arm_dir("syn_b_other_code", "ca", code)
    with pytest.raises(SystemExit, match=rf"forward code differs.*{f}"):
        DS.run({"a": str(da), "b": str(dbx), "c": str(ARM_C)}, tmp_path / "x.json")
    assert not (tmp_path / "x.json").exists()


# ----------------------------------------------------------------------------------------------- F3 pretrain resume
class _Stop(Exception):
    pass


def _pf_args(out, batch=8, limit=40, epochs=3, extra=()):
    return ["--out", str(out), "--device", "cpu", "--epochs", str(epochs), "--batch", str(batch), "--limit-tokens",
            str(limit), "--eval-splits", "", "--train-eval-rows", "0", "--workers", "0", "--log-every", "1",
            "--warmup-steps", "2", *extra]


def _run_pf(mod, args, monkeypatch, stop_epoch=None):
    """run mod.main(args); -> {epoch: number of samples TRAINED in that epoch} (forward_fc calls; no evals here).
    stop_epoch: raise at the first step record of that epoch (= killed after the previous epoch's checkpoint)."""
    from tools.ck import ckutil as U
    trained, cur = {}, {"e": None}
    with monkeypatch.context() as mp:
        set_epoch, fwd = mod.PackRows.set_epoch, mod.forward_fc

        def se(self, e):
            cur["e"] = int(e)
            return set_epoch(self, e)

        def ff(model, b, dev):
            trained[cur["e"]] = trained.get(cur["e"], 0) + int(b["obj_in"].shape[0])
            return fwd(model, b, dev)

        mp.setattr(mod.PackRows, "set_epoch", se)
        mp.setattr(mod, "forward_fc", ff)
        if stop_epoch is not None:
            log = U.log_jsonl

            def lj(path, rec, echo=True):
                if rec.get("kind") == "step" and rec.get("epoch") == stop_epoch:
                    raise _Stop()
                return log(path, rec, echo)

            mp.setattr(U, "log_jsonl", lj)
        try:
            mod.main(args)
        except _Stop:
            pass
    return trained


def _old_pf(tmp_path):
    """tools/ck/v3/pretrain_forecaster.py as of OLD_COMMIT, loaded from a temp copy (repo root pinned)."""
    try:
        src = subprocess.run(["git", "-C", CK, "show", f"{OLD_COMMIT}:tools/ck/v3/pretrain_forecaster.py"],
                             capture_output=True, text=True, check=True).stdout
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"old pretrain_forecaster.py not available: {e}")
    line = "REPO = str(Path(__file__).resolve().parents[3])"
    assert src.count(line) == 1
    p = tmp_path / "pretrain_forecaster_old.py"
    p.write_text(src.replace(line, f"REPO = {CK!r}"))
    spec = importlib.util.spec_from_file_location("pretrain_forecaster_old_aa7a7b2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@have_train_pack
def test_pretrain_resume_geometry(tmp_path, monkeypatch):
    from tools.ck.v3 import pretrain_forecaster as PF
    # uninterrupted reference
    ref = tmp_path / "ref"
    t_ref = _run_pf(PF, _pf_args(ref), monkeypatch)
    conf = json.loads((ref / "config.json").read_text())
    n, spe = conf["n_train"], conf["steps_per_epoch"]
    assert conf["batch"] == 8 and spe == -(-n // 8) and conf["total_steps"] == 3 * spe and n > 32
    assert t_ref == {0: n, 1: n, 2: n}
    ck = torch.load(ref / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert (ck["batch"], ck["n_train"], ck["steps_per_epoch"], ck["total_steps"]) == (8, n, spe, 3 * spe)
    # killed after epoch 0's checkpoint
    run = tmp_path / "run"
    _run_pf(PF, _pf_args(run), monkeypatch, stop_epoch=1)
    ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert (ck["epoch"], ck["step"]) == (1, spe) and not (run / "done.json").exists()
    old_state = tmp_path / "run_for_old_code"
    shutil.copytree(run, old_state)
    # other step geometry: refused, nothing trained
    for args, msg in ((_pf_args(run, batch=16), "batch=16"), (_pf_args(run, limit=32), "n_train="),
                      (_pf_args(run, extra=("--max-steps", str(2 * spe))), "total_steps=")):
        with pytest.raises(SystemExit, match=msg):
            _run_pf(PF, args + ["--resume"], monkeypatch)
    assert not (run / "done.json").exists()
    # the recorded settings: every sample once per remaining epoch, bitwise the uninterrupted weights
    t_res = _run_pf(PF, _pf_args(run) + ["--resume"], monkeypatch)
    assert t_res == {1: n, 2: n}
    a = torch.load(ref / "forecaster.pt", map_location="cpu", weights_only=False)["forecaster"]
    b = torch.load(run / "forecaster.pt", map_location="cpu", weights_only=False)["forecaster"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert json.loads((run / "done.json").read_text())["epochs_done"] == 3
    # the old failure (pre-fix code, same interrupted state): --batch 16 resumes silently and drops samples
    old = _old_pf(tmp_path)
    t_old = _run_pf(old, _pf_args(old_state, batch=16) + ["--resume"], monkeypatch)
    assert t_old.get(1, 0) < n                                     # epoch 1 lost samples (skip counted in old steps)
    assert json.loads((old_state / "done.json").read_text())["epochs_done"] == 3          # ... and reports "done"
    assert json.loads((old_state / "config.json").read_text())["args"]["batch"] == 8      # config keeps the old batch


def test_resume_mismatch_unit():
    from tools.ck.v3 import pretrain_forecaster as PF
    conf = dict(n_modes=6, seed=0, epochs=8, aug={"x": 1}, batch=64, n_train=85109, steps_per_epoch=1330,
                total_steps=10640)
    pre_f3 = {k: v for k, v in conf.items() if k != "batch"}
    pre_f3["args"] = {"batch": 64}                                  # configs written before F3 (v3fc_pre_learned6)
    assert PF.resume_mismatch(pre_f3, conf) == [] and PF.resume_mismatch(conf, conf, dict(conf)) == []
    assert any("batch=128" in x for x in PF.resume_mismatch(pre_f3, dict(conf, batch=128)))
    assert any("ckpt_last.pt" in x for x in PF.resume_mismatch(conf, conf, dict(conf, steps_per_epoch=665)))
    assert PF.resume_mismatch(conf, conf, {"epoch": 4, "step": 5320}) == []              # pre-F3 checkpoint


# ----------------------------------------------------------------------------------------------- F4 part key
def _pack_arrays(P):
    """raw bytes of every pack array (frame_dt holds NaN: compare bytes, not values)."""
    from tools.ck.data import obj_pack as OP
    return {k: np.load(P / f"{k}.npy").tobytes() for k in list(OP.PACK_FILES) + list(OP.TARGET_FILES)}


def _drop_final(P):
    """simulate an interrupted build: every chunk part kept, no assembled arrays, no build.json."""
    for f in P.iterdir():
        if f.is_file():
            f.unlink()


@pytest.mark.skipif(not (TC.D3 / "gt_objects" / "navtrain_val" / "index.parquet").is_file(),
                    reason="val GT objects / BEVFusion caches absent")
def test_build_resume_rebuilds_on_changed_gate(tmp_path, monkeypatch, capsys):
    from navsim.agents.para_ssr.ck.temporal import objects as O
    from tools.ck.v3 import build_obj_pack as B
    root = tmp_path / "objpack"
    args = ["--split", "navtrain_val", "--workers", "2", "--limit", "32", "--chunk", "16", "--out-root", str(root),
            "--keep-parts"]
    P = root / "bevfusion" / "navtrain_val"
    assert O.MATCH_GATE_M == 2.0
    assert B.main(args) == 0
    g20 = _pack_arrays(P)
    b20 = json.loads((P / "build.json").read_text())
    assert b20["inputs_sha256"] and b20["inputs"]["match_gate_m"] == 2.0 and b20["inputs"]["N_FRAMES"] == TC.N_FRAMES
    assert b20["inputs"]["gt_manifest"]["path"].endswith("_manifests/navtrain_val.json")
    # unchanged inputs: the parts are reused
    _drop_final(P)
    capsys.readouterr()
    assert B.main(args) == 0
    assert "2 chunks, 0 to build" in capsys.readouterr().out
    assert _pack_arrays(P) == g20
    # the gate changed between the interrupted build and its resume: every chunk is rebuilt, nothing is mixed
    _drop_final(P)
    monkeypatch.setattr(O, "MATCH_GATE_M", 0.5)
    capsys.readouterr()
    assert B.main(args) == 0
    assert "2 chunks, 2 to build" in capsys.readouterr().out
    g05 = _pack_arrays(P)
    b05 = json.loads((P / "build.json").read_text())
    assert b05["match_gate_m"] == 0.5 and b05["inputs"]["match_gate_m"] == 0.5
    assert b05["inputs_sha256"] != b20["inputs_sha256"]
    for f in (P / "_parts").iterdir():
        with np.load(f) as z:
            assert json.loads(str(z["part_key"]))["inputs"] == b05["inputs_sha256"]
    fresh = tmp_path / "fresh"
    assert B.main(["--split", "navtrain_val", "--workers", "2", "--limit", "32", "--chunk", "16", "--out-root",
                   str(fresh)]) == 0
    g05_fresh = _pack_arrays(fresh / "bevfusion" / "navtrain_val")
    for k, v in g05_fresh.items():
        assert v == g05[k], k
    # the gate matters on these rows (else the test would not show mixing): the targets differ from gate 2.0
    assert any(g20[k] != g05[k] for k in ("fc_state", "fc_track", "fc_xyh"))


def test_part_key_includes_fingerprint():
    from tools.ck.v3 import build_obj_pack as B
    assert B.part_key(True, "a" * 64) != B.part_key(True, "b" * 64)
    assert json.loads(B.part_key(True, "a" * 64))["inputs"] == "a" * 64
    prov = {"caches": {"x": 1}, "code": {"objects.py": {"sha256": "s"}}, "git_head": "h1", "git_note": "n"}
    s1, d1 = B.input_fingerprint(prov, None)
    s2, _ = B.input_fingerprint(dict(prov, git_head="h2"), None)            # a commit alone does not rebuild
    s3, _ = B.input_fingerprint(dict(prov, code={"objects.py": {"sha256": "t"}}), None)
    assert s1 == s2 != s3 and d1["gt_manifest"] is None and "git_head" not in d1["provenance"]


@pytest.mark.skipif(not (PK / "navtrain_val" / "build.json").is_file(), reason="K96 val pack absent")
def test_existing_complete_pack_loads_unchanged():
    """packs built before F4 carry no fingerprint; ObjPack never reads it."""
    from tools.ck.data.obj_pack import ObjPack
    b = json.loads((PK / "navtrain_val" / "build.json").read_text())
    assert b["status"] == "complete" and "inputs_sha256" not in b
    pk = ObjPack("navtrain_val", root=TC.PACK_ROOT, targets=True)
    assert pk.n == 18179 and pk.has_targets
