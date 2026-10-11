"""refiner-v3 OFF == CK2 @ 8ae8b1d, bitwise (docs/refiner_v3/SPEC.md 4.7; Integrate phase).

Runs tools/ck/v3/off_equiv_probe.py twice in fresh `python -B` interpreters (PYTHONPATH unset, CPU, 1 thread): once
against the reference tree D3/ref_8ae8b1d (`git archive 8ae8b1d` + the Phase-0 pristine tools/ck/data copies and
future_teacher.py) and once against this worktree, then tools/ck/v3/off_equiv_compare.py: every array (initial
state_dict, CPU forward with extra + decode + rescore_corr at init and with the ck2T10 checkpoint, synthetic
train_ck2.train runs of 1 and 3 steps: step stats, final weights and AdamW state, val metrics) and every fact (param
counts 3,516,178 / 3,391,243, state_dict keys, sha256 of every collated key of 64 navtrain_val tokens x 2 epochs for
CK2Dataset train style and CKDataset r34 style) must be identical; config.json may only gain the temporal keys.
Skipped when the reference tree is absent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
V3 = Path(__file__).resolve().parents[1]
D3 = Path("/home/external-user/ssd/yongjae_refiner/ck/v3")
REF = D3 / "ref_8ae8b1d"
PY = sys.executable


@pytest.mark.skipif(not (REF / "navsim").is_dir(), reason="reference tree D3/ref_8ae8b1d absent")
def test_off_bitwise_identical_to_8ae8b1d():
    work = D3 / "test_tmp" / f"off_equiv_{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    probe = str(V3 / "tools" / "ck" / "v3" / "off_equiv_probe.py")
    procs = {}
    for name, repo in (("ref", REF), ("v3", V3)):
        procs[name] = subprocess.Popen([PY, "-B", probe, "--repo", str(repo), "--out", str(work / f"{name}.npz"),
                                        "--work", str(work / f"{name}_work")], env=env, cwd="/tmp",
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for name, p in procs.items():
        out, _ = p.communicate(timeout=1500)
        assert p.returncode == 0, f"{name} probe failed:\n{out[-4000:]}"
    r = subprocess.run([PY, "-B", str(V3 / "tools" / "ck" / "v3" / "off_equiv_compare.py"), str(work / "ref.npz"),
                        str(work / "v3.npz")], env=env, cwd="/tmp", capture_output=True, text=True, timeout=600)
    rep = json.loads(r.stdout)
    (work / "compare.json").write_text(json.dumps(rep, indent=1))
    assert rep["identical"], json.dumps({k: rep[k] for k in ("only_ref", "only_new", "diff", "facts_diff")})[:4000]
    assert rep["n_arrays"] > 1000 and rep["n_data_hashes"] > 700
    assert rep["param_counts"]["total"] == 3516178 and rep["ck2T10"] == [3516178, 3391243]
    import shutil
    shutil.rmtree(work, ignore_errors=True)
