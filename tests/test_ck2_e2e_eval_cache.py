"""CK2 e2e evaluation cache identity (report 48 F1 / F1-b / F1-c / F1-d; tools/ck/e2e2/eval_e2e2.py).  CPU, dummy
checkpoint files and tiny memmaps, no model:
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_eval_cache.py

  F1    done dump rows are reused only under the same cache key (ckpt file sha16, training-config sha16, dump-code
        sha16): another checkpoint / config / code -> SystemExit naming it (metadata and rows untouched), the same key
        -> resume (todo 0), --fresh -> rebuilt under the new key; done rows without a key are refused.
  F1-b  without --ckpt the pin must be the current 'last' with the pinned file sha16 (a newer version_N, a last.ckpt
        rewritten in place, an epoch:E pin, or no resolvable 'last' -> SystemExit); --fresh removes the pin.
  F1-c  summary refuses val / navtest results of different checkpoints (navtest --fix-from: test_ck2_e2e_select).
  labels are reused only when their meta.json traj_sha16 equals the current trajectories; infer_check refuses an
  extracted ck_run of another checkpoint than the dump.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.ck import ckutil as U  # noqa: E402
from tools.ck.data import common as CM  # noqa: E402
from tools.ck.e2e import eval_e2e as E1  # noqa: E402
from tools.ck.e2e2 import eval_e2e2 as E2  # noqa: E402

TDF = pd.DataFrame({"token": ["tok0000000000000", "tok0000000000001"], "log": ["L0", "L1"], "city": ["x", "x"]})


def _mk(p: Path, payload: bytes, age: float = 0.0) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    if age:
        t = time.time() - age
        os.utime(p, (t, t))
    return p


def _run(tmp: Path, name: str = "ck2e2e_fake") -> Path:
    R = tmp / name
    ck = R / "train/lightning_logs/version_0/checkpoints"
    _mk(ck / "epoch=9-step=100.ckpt", b"A" * 64)
    _mk(ck / "epoch=19-step=200.ckpt", b"B" * 64)
    _mk(ck / "last.ckpt", b"B" * 64)
    return R


def _main(R, *extra, split="navtrain_val"):
    return E2.main(["--run-dir", str(R), "--split", split, "--limit", "2", "--gpus", "", *extra])


@pytest.fixture
def fake_split(monkeypatch):
    monkeypatch.setattr(E1, "split_table", lambda split, limit: TDF.iloc[:limit or len(TDF)].reset_index(drop=True))


def _fill(value):
    """a run_dump_shard stand-in that 'dumps' every row (cand96 = value)"""
    calls = []

    def shard(P, ckpt, tdf, shard, nshard, batch_size, workers):
        calls.append(str(ckpt))
        W = E2.EvalRootWriter2(P, tdf, P.ck_run)
        W.A["cand96"][:] = value
        W.A["ok"][:] = True
        W.I["done"][:] = True
        W.flush()
        return {"shard": shard, "n_done": len(tdf)}
    return shard, calls


def _prefill_epoch9(R):
    P = E2.Paths(R, "navtrain_val")
    P.eval.mkdir(parents=True, exist_ok=True)
    p9, sha9 = E2.pinned_ckpt2(P, "epoch:9")
    E2.check_cache_key(P, E2.cache_key(P, p9, sha9, 2))
    W = E2.EvalRootWriter2(P, TDF, P.ck_run)
    W.A["cand96"][:] = 123.0
    W.I["score_logit"][:] = 7.0
    W.A["ok"][:] = True
    W.I["done"][:] = True
    W.flush()
    W.finalize_meta({"ckpt": str(p9), "ckpt_sha16": sha9, "limit": 2, "n_done": 2})
    return P, p9, sha9


# ----------------------------------------------------------------------------------------------- F1
def test_other_checkpoint_refused_same_resumes_fresh_rebuilds(tmp_path, fake_split, monkeypatch):
    R = _run(tmp_path)
    P, p9, sha9 = _prefill_epoch9(R)
    # (a) epoch:19 on the epoch-9 cache: refused before anything is rewritten
    with pytest.raises(SystemExit, match=r"different checkpoint.*--fresh"):
        _main(R, "--stage", "dump", "--ckpt", "epoch:19")
    assert U.read_json(P.packed / "meta.json")["ckpt"] == str(p9) == U.read_json(P.infer / "meta.json")["ckpt"]
    assert (np.load(P.packed / "cand96.npy") == 123.0).all() and (np.load(P.infer / "score_logit.npy") == 7.0).all()
    assert U.read_json(P.packed / "cache_key.json")["ckpt_sha16"] == sha9
    assert U.read_json(P.pin)["path"] == str(p9)                    # a refused attempt leaves the pin as it was
    # the same refusal when 'extract' comes first (checked before extract rewrites ck_run)
    with pytest.raises(SystemExit, match="different checkpoint"):
        _main(R, "--stage", "extract,dump", "--ckpt", "epoch:19")
    assert not P.ck_run.exists()
    # (b) the same checkpoint: plain resume (todo 0, rows untouched)
    assert _main(R, "--stage", "dump", "--ckpt", "epoch:9") == 0
    assert (np.load(P.packed / "cand96.npy") == 123.0).all()
    m = U.read_json(P.packed / "meta.json")
    assert m["ckpt_sha16"] == sha9 and m["code_sha16"] == U.read_json(P.packed / "cache_key.json")["code_sha16"]
    # (c) --fresh: rebuilt under epoch 19's key (the real file sha16, not the pin's)
    real_shard = E2.run_dump_shard
    shard, calls = _fill(456.0)
    monkeypatch.setattr(E2, "run_dump_shard", shard)
    assert _main(R, "--stage", "dump", "--ckpt", "epoch:19", "--fresh") == 0
    p19 = R / "train/lightning_logs/version_0/checkpoints/epoch=19-step=200.ckpt"
    k = U.read_json(P.packed / "cache_key.json")
    assert calls == [str(p19.resolve())] and k["ckpt_sha16"] == U.sha256_file(p19) != sha9
    assert U.read_json(P.infer / "meta.json")["ckpt_sha16"] == k["ckpt_sha16"]
    assert (np.load(P.packed / "cand96.npy") == 456.0).all()
    # (d) the same bytes under another path (Lightning: last.ckpt == the final epoch=E file): a resume, and the real
    # dump shard accepts the key (it checks the checkpoint path) -- not a 'names another checkpoint' refusal
    monkeypatch.setattr(E2, "run_dump_shard", real_shard)
    p_last = R / "train/lightning_logs/version_0/checkpoints/last.ckpt"
    assert U.sha256_file(p_last) == k["ckpt_sha16"]
    assert _main(R, "--stage", "dump", "--ckpt", "last") == 0
    k2 = U.read_json(P.packed / "cache_key.json")
    assert k2["ckpt"] == str(p_last.resolve()) and k2["ckpt_sha16"] == k["ckpt_sha16"]
    assert (np.load(P.packed / "cand96.npy") == 456.0).all()


def test_unkeyed_cache_with_done_rows_refused(tmp_path, fake_split):
    R = _run(tmp_path)
    P, _, _ = _prefill_epoch9(R)
    (P.packed / "cache_key.json").unlink()
    with pytest.raises(SystemExit, match="unkeyed"):
        _main(R, "--stage", "dump", "--ckpt", "epoch:9")
    assert (np.load(P.packed / "cand96.npy") == 123.0).all()


def test_dump_code_and_training_config_are_part_of_the_key(tmp_path, fake_split, monkeypatch):
    R = _run(tmp_path)
    P = E2.Paths(R, "navtrain_val")
    code = _mk(tmp_path / "src" / "online2_stub.py", b"x = 1\n")
    monkeypatch.setattr(E2, "dump_code_files", lambda root=None: [code])
    hyd = _mk(P.hydra_cfg, b"agent: {a: 1}\n")
    shard, calls = _fill(1.0)
    monkeypatch.setattr(E2, "run_dump_shard", shard)
    assert _main(R, "--stage", "dump", "--ckpt", "last") == 0
    k0 = U.read_json(P.packed / "cache_key.json")
    assert k0["code_files"] == {str(code): U.sha256_file(code)} and k0["hydra_sha16"] == U.sha256_file(hyd)
    code.write_bytes(b"x = 2\n")                                   # a dump-path code change
    with pytest.raises(SystemExit, match=r"dump code \(1 files: .*online2_stub.py"):
        _main(R, "--stage", "dump", "--ckpt", "last")
    code.write_bytes(b"x = 1\n")
    assert _main(R, "--stage", "dump", "--ckpt", "last") == 0       # bytes restored: same key, resume
    hyd.write_bytes(b"agent: {a: 2}\n")                           # another training config
    with pytest.raises(SystemExit, match="training config"):
        _main(R, "--stage", "dump", "--ckpt", "last")
    assert _main(R, "--stage", "dump", "--ckpt", "last", "--fresh") == 0
    assert U.read_json(P.packed / "cache_key.json")["hydra_sha16"] == U.sha256_file(hyd)


def test_real_dump_code_manifest():
    """the real manifest covers the v2 / CK2 inference code and the dump code (tests / __pycache__ excluded)"""
    files = {str(f.relative_to(REPO)) for f in E2.dump_code_files()}
    for need in ("navsim/agents/para_ssr/ck/online2.py", "navsim/agents/para_ssr/ck/cands2.py",
                 "navsim/agents/para_ssr/ck/variants.py", "navsim/agents/para_ssr/para_ssr_agent.py",
                 "navsim/agents/para_ssr/para_ssr_model.py", "tools/ck/e2e2/eval_e2e2.py", "tools/ck/data/dump_v2.py",
                 "navsim/planning/script/run_aux_evaluation.py"):
        assert need in files, need
    assert not any("__pycache__" in f or Path(f).name.startswith("test_") for f in files)
    man = E2.code_manifest()
    assert set(man) == files and all(len(v) == 16 for v in man.values())



# ----------------------------------------------------------------------------------------------- F1-b
def test_pin_must_be_current_last(tmp_path, fake_split):
    R = tmp_path / "ck2e2e_resume"
    v0 = _mk(R / "train/lightning_logs/version_0/checkpoints/last.ckpt", b"E13" * 30, age=100)
    P = E2.Paths(R, "navtrain_val")
    P.eval.mkdir(parents=True)
    p, sha = E2.pinned_ckpt2(P, None)                     # no pin yet: 'last' resolved and pinned
    assert p == v0.resolve() and sha == U.sha256_file(v0)
    assert E2.pinned_ckpt2(P, None) == (p, sha)           # consistent pin: accepted
    v1 = _mk(R / "train/lightning_logs/version_1/checkpoints/last.ckpt", b"E29" * 30)   # resumed run: version_1
    with pytest.raises(SystemExit, match="not the current 'last'"):
        E2.pinned_ckpt2(P, None)
    assert E2.pinned_ckpt2(P, "last")[0] == v1.resolve()  # explicit --ckpt last re-pins
    v1.write_bytes(b"E30" * 30)                           # last.ckpt rewritten in place (save_last)
    with pytest.raises(SystemExit, match="changed on disk"):
        E2.pinned_ckpt2(P, None)
    assert E2.pinned_ckpt2(P, "last")[1] == U.sha256_file(v1)
    _mk(R / "train/lightning_logs/version_1/checkpoints/epoch=14-step=1.ckpt", b"E14" * 30, age=50)
    E2.pinned_ckpt2(P, "epoch:14")
    with pytest.raises(SystemExit, match="not the current 'last'"):  # an epoch pin is not the default 'last'
        E2.pinned_ckpt2(P, None)
    # no lightning_logs (logger off, e.g. cpu_dryrun): only an explicit --ckpt
    R2 = tmp_path / "ck2e2e_nolog"
    c = _mk(R2 / "train/last.ckpt", b"Z" * 10)
    P2 = E2.Paths(R2, "navtrain_val")
    P2.eval.mkdir(parents=True)
    assert E2.pinned_ckpt2(P2, str(c))[0] == c.resolve()
    with pytest.raises(SystemExit, match="cannot resolve 'last'"):
        E2.pinned_ckpt2(P2, None)
    # --fresh clears the pin (and the split's outputs)
    E2.pinned_ckpt2(P, "epoch:14")
    (P.out.parent / "navtrain_val__selw_plugin").mkdir(parents=True)
    (P.out.parent / "navtest").mkdir(parents=True)
    assert _main(R, "--stage", "summary", "--fresh") == 0
    assert not P.pin.exists() and not (P.out.parent / "navtrain_val__selw_plugin").exists()
    assert (P.out.parent / "navtest").is_dir()                   # the other split is untouched


def test_extract_dump_without_ckpt_refuses_a_stale_pin(tmp_path, fake_split):
    """the documented command (no --ckpt) after an interim epoch:E evaluation: refused, not silently the old pin"""
    R = _run(tmp_path)
    P, _, _ = _prefill_epoch9(R)                          # interim eval pinned epoch:9
    with pytest.raises(SystemExit, match="not the current 'last'"):
        _main(R, "--stage", "dump")
    assert U.read_json(P.pin)["spec"] == "epoch:9"


# ----------------------------------------------------------------------------------------------- labels / infer_check
def test_labels_reused_only_on_matching_trajectories(tmp_path, monkeypatch):
    R = tmp_path / "ck2e2e_lab"
    P = E2.Paths(R, "navtrain_val")
    for d in (P.packed, P.infer, P.eval):
        d.mkdir(parents=True, exist_ok=True)
    cand = np.random.default_rng(0).normal(size=(2, 96, 8, 3)).astype(np.float32)
    lat = cand + 0.1
    np.save(P.packed / "cand96.npy", cand)
    np.save(P.infer / "lat_traj.npy", lat)
    np.save(P.packed / "ok_rows.npy", np.arange(2))
    for name, arr in ((f"pool96_{P.name}", cand), (f"lat96_{P.name}", lat)):
        d = P.labels / name
        d.mkdir(parents=True)
        np.save(d / "labels.npy", np.zeros((2, 96, 9), np.float32))
        (d / "meta.json").write_text(json.dumps({"traj_sha16": CM.sha16_array(arr)}))
    monkeypatch.setattr(E2.subprocess, "run", lambda *a, **k: pytest.fail("label_cands must not run"))
    E2.stage_label(P, SimpleNamespace(workers_label=2))          # both present and current: skipped
    np.save(P.packed / "cand96.npy", cand + 1.0)                 # a different dump under old labels
    with pytest.raises(SystemExit, match="stale labels"):
        E2.stage_label(P, SimpleNamespace(workers_label=2))


def test_infer_check_refuses_ck_run_of_another_checkpoint(tmp_path):
    P = E2.Paths(tmp_path / "ck2e2e_ic", "navtrain_val")
    for d in (P.infer, P.ck_run):
        d.mkdir(parents=True)
    np.savez(P.infer / "check_bev.npz", rows=np.arange(1), bev=np.zeros((1, 256, 50, 100), np.float32))
    (P.ck_run / "config.json").write_text(json.dumps({"source_ckpt_sha16": "a" * 16}))
    (P.infer / "meta.json").write_text(json.dumps({"ckpt_sha16": "b" * 16}))
    with pytest.raises(SystemExit, match="extracted ck_run"):
        E2.stage_infer_check(P, SimpleNamespace(gpus=""))


# ----------------------------------------------------------------------------------------------- F1-c summary
def _metrics(sha, split):
    row = {"key": "v2", "pdms": 0.5, "nc": 1.0, "dac": 1.0, "ep": 1.0, "ttc": 1.0, "comfort": 1.0, "frac_changed": 0.0,
           "frac_lat": 0.0}
    return {"split": split, "n_eval": 2, "n_total": 2, "n_logs": 1, "rows": [row], "representative": "v2",
            "best": {"beta": 1.0} if split == "navtrain_val" else None,
            "model": {"ckpt": f"/r/{sha[:1]}.ckpt", "ckpt_sha16": sha, "ep_target": "decoupled", "code_sha16": "c" * 16}}


def test_summary_refuses_val_and_navtest_of_different_checkpoints(tmp_path):
    R = tmp_path / "ck2e2e_sum"
    ev = R / "eval" / "root" / "eval" / R.name
    for split, sha in (("navtrain_val", "a" * 16), ("navtest", "b" * 16)):
        (ev / split).mkdir(parents=True)
        (ev / split / "metrics.json").write_text(json.dumps(_metrics(sha, split)))
    with pytest.raises(SystemExit, match="navtest on"):
        E2.stage_summary(R)
    (ev / "navtest" / "metrics.json").write_text(json.dumps(_metrics("a" * 16, "navtest")))
    out = E2.stage_summary(R)
    assert out["ckpt_by_split"]["navtrain_val"]["ckpt_sha16"] == out["ckpt_by_split"]["navtest"]["ckpt_sha16"] == "a" * 16
    assert out["code_sha16_same"] and json.loads((R / "eval" / "summary.json").read_text())["ckpt"]["ckpt_sha16"] == "a" * 16
