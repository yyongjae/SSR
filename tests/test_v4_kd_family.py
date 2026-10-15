"""refiner v4 I2: per-variant-family KD calibration (navsim ck/kd_calib.py params_by_family, online2.TeacherPair2
family gather, tools/ck/v4/fit_kd_calib_family.py) and the parent-only score KD option.  CPU, smoke teachers:
  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_kd_family.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import kd_calib as KC  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_KEYS  # noqa: E402

need_inputs = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
P_DET = {"nc": {"a": 0.7, "b": 0.9}, "ep": {"a": 1.3, "b": -0.4}, "ttc": {"a": 0.55, "b": 1.1}}


def _fam_params(arm, base, shift):
    """{family: {key: {a, b, status}}}: the identity family = base, family f: a * (1 + 0.1 f), b + shift f."""
    fk = KC.fit_keys(arm)
    out = {}
    for f, name in enumerate(KC.KD_FAMILIES):
        out[name] = {k: ({"a": base[k]["a"] * (1 + 0.1 * f), "b": base[k]["b"] + shift * f, "status": "fitted"}
                         if k in fk else {"a": 1.0, "b": 0.0, "status": "not_fitted"}) for k in CK_KEYS}
    return out


def _ckpt_sha(run: Path) -> str:
    from tools.ck import ckutil as U
    return U.sha256_file(run / "ckpt_last.pt")


def _write(tmp: Path, name: str, run: Path, arm: str, params, **over) -> Path:
    kw = dict(run=str(run), which="last", arm=arm, ckpt=str(run / "ckpt_last.pt"), ckpt_sha16=_ckpt_sha(run),
              ep_target="official", params=params, created="test")
    kw.update(over)
    p = Path(tmp) / name
    p.write_text(json.dumps(KC.build_record(**kw)))
    return p


def test_families_constant_matches_pool_layout():
    assert KC.KD_FAMILIES == S2.VNAMES


@need_inputs
def test_family_record_validation_and_matrix(tmp_path):
    T = TL.SMOKE_T
    fam = _fam_params("T", P_DET, 0.2)
    p = _write(tmp_path, "fam.json", T, "T", P_DET, params_by_family=fam)
    rec = KC.load_calib(p)
    a, b = KC.calib_matrix(rec)
    assert a.shape == (6, 5) and np.allclose(a[3, CK_KEYS.index("nc")], 0.7 * 1.3) and \
        np.allclose(b[5, CK_KEYS.index("ttc")], 1.1 + 1.0)
    # the identity row must equal 'params' (readers without family support apply 'params' to every column)
    bad = json.loads(json.dumps(fam))
    bad["id"]["nc"]["b"] = 0.0
    with pytest.raises(ValueError, match="params must equal"):
        _write(tmp_path, "bad.json", T, "T", P_DET, params_by_family=bad)
    # a v1 record: the matrix repeats the vector (old calibration on every column)
    p1 = _write(tmp_path, "v1.json", T, "T", P_DET)
    a1, b1 = KC.calib_matrix(KC.load_calib(p1))
    av, bv = KC.calib_vectors(KC.load_calib(p1))
    assert np.array_equal(a1, np.tile(av, (6, 1))) and np.array_equal(b1, np.tile(bv, (6, 1)))
    # loaders: family=False -> the vector (old API); family=True -> the matrix
    kw = dict(run=str(T), which="last", ckpt_sha16=_ckpt_sha(T), ep_target="official", arm="T")
    av2, _, info = KC.load_teacher_calib(p, **kw)
    am, _, info2 = KC.load_teacher_calib(p, family=True, **kw)
    assert av2.shape == (5,) and am.shape == (6, 5) and info2["families"] and "families" not in info


@need_inputs
def test_teacher_pair_family_gather(tmp_path):
    T, M = TL.SMOKE_T, TL.SMOKE_M
    fam = _fam_params("T", P_DET, 0.2)
    pF = _write(tmp_path, "famT.json", T, "T", P_DET, params_by_family=fam)
    p1 = _write(tmp_path, "v1T.json", T, "T", P_DET)
    f, t = TL.fresh(2)
    g = torch.Generator().manual_seed(1)
    anc = torch.from_numpy(np.load(TL.ANCHORS)).float()
    cand = anc[torch.randint(0, 256, (2, 12), generator=g)]
    family = (torch.arange(12) % 6).expand(2, 12)
    args = (t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, f["status_feature"])
    tp1 = O.TeacherPair2(str(T), str(M), "last", "cpu", amp=False, calib={"det": str(p1), "map": None})
    tpF = O.TeacherPair2(str(T), str(M), "last", "cpu", amp=False, calib={"det": str(pF), "map": None})
    o1, o1f = tp1.run(*args), tp1.run(*args, family=family)
    for k in o1:                                            # a v1 file: the family argument changes nothing
        assert torch.equal(o1[k], o1f[k]), k
    with pytest.raises(ValueError, match="family"):
        tpF.run(*args)
    oF = tpF.run(*args, family=family)
    with torch.no_grad():
        z = tpF.det(t["kd_bev_0"].float(), cand, f["status_feature"].float(), decode=False)["score_logit"].float()
    a, b = KC.calib_matrix(KC.load_calib(pF))
    want = torch.sigmoid(z * torch.from_numpy(a)[family] + torch.from_numpy(b)[family])
    torch.testing.assert_close(oF["p_det"], want, rtol=1e-5, atol=1e-6)
    idc = family == 0                                     # identity columns == the v1 calibration
    torch.testing.assert_close(oF["p_det"][idc], o1["p_det"][idc], rtol=1e-6, atol=1e-7)


@need_inputs
def test_fit_tool_cpu_dry(tmp_path):
    """tiny CPU run of the family fit on a real finished generation (skipped when absent)."""
    gen = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/e2e/ck2e2e_main/ck_e2e2/lab/ep028")
    if not (gen / "meta.json").is_file():
        pytest.skip("no finished student generation")
    from tools.ck.v4 import fit_kd_calib_family as FF
    out = tmp_path / "fam.json"
    rec = FF.main(["--run", str(TL.SMOKE_T), "--which", "last", "--gen", str(gen), "--n-tokens", "256", "--limit", "8",
                   "--device", "cpu", "--workers", "0", "--batch", "4", "--allow-unfinished", "--out", str(out)])
    assert KC.has_families(rec) and rec["fitted_keys"] == ["nc", "ep", "ttc"]
    KC.validate_record(json.loads(out.read_text()))
    assert rec["data"]["per_family_ok"]["id"] > 0 and (out.with_suffix(".infer.npz")).is_file()
