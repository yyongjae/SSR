"""CK EP target helper (navsim/agents/para_ssr/ck/ep_target.py; user decision 2026-10-08 ~20:10 KST).  CPU only.

  official   ck_targets(labels9, 'official') == labels9[..., CK_LABEL_IDX] bit for bit (numpy and torch, same dtype)
  decoupled  EP = clip(r / max(p, r), 0, 1) if max(p, r) > 5 else 1 (r raw_progress, p pdm_progress_eff); equals the
             official EP wherever M = NC * DAC * DDC == 1 (real raw256 labels, first 200 navtrain_val tokens, mmap);
             in [0, 1]; threshold rule; non-finite r / p -> NaN; torch == numpy; only the EP column changes.

  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck_ep_target.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck import ep_target as E  # noqa: E402

RAW = Path(Cn.CK_DATA) / "labels/navtrain_val/raw256/labels.npy"
L = Cn.LBL
CKI = list(Cn.CK_LABEL_IDX)
real = pytest.mark.skipif(not RAW.is_file(), reason="raw256 navtrain_val labels absent")


@pytest.fixture(scope="module")
def lab200():
    return np.array(np.load(RAW, mmap_mode="r")[:200])                       # [200, 256, 9] f32 (copy)


def _row(r, p, nc=1.0, dac=1.0, ddc=1.0, ep=0.5):
    x = np.zeros(9, np.float32)
    x[L["nc"]], x[L["dac"]], x[L["ddc"]], x[L["ep"]] = nc, dac, ddc, ep
    x[L["ttc"]], x[L["comfort"]], x[L["pdms"]] = 1.0, 1.0, 0.7
    x[L["raw_progress"]], x[L["pdm_progress_eff"]] = r, p
    return x


def test_constants():
    assert E.EP_TARGETS == ("official", "decoupled") and E.EP_DIST_THRESHOLD == 5.0
    assert Cn.CK_KEYS[E.EP_KEY_IDX] == "ep" and E.R_COL == 7 and E.P_COL == 8
    assert E.check_ep_target(None) == "official" and E.check_ep_target("decoupled") == "decoupled"
    with pytest.raises(ValueError):
        E.check_ep_target("bogus")
    with pytest.raises(ValueError):
        E.ck_targets(np.zeros((3, 8), np.float32), "official")              # not 9 label columns
    assert E.run_ep_target({}) == "official" and E.run_ep_target(None) == "official"
    assert E.run_ep_target({"ep_target": "decoupled"}) == "decoupled"


@real
def test_official_passthrough_identical(lab200):
    for x in (lab200, lab200[:, :3].astype(np.float64), lab200[0]):
        y = E.ck_targets(x, "official")
        old = x[..., CKI]
        assert y.dtype == x.dtype and y.shape == x.shape[:-1] + (5,)
        assert y.strides == old.strides and y.tobytes() == old.tobytes()        # bit for bit, same layout
        assert not np.shares_memory(y, x)
        yt = E.ck_targets(torch.from_numpy(x), "official")
        assert yt.dtype == torch.from_numpy(x).dtype and torch.equal(yt, torch.from_numpy(x)[..., CKI])
    assert E.ck_targets(lab200, None).tobytes() == lab200[..., CKI].tobytes()   # None = official


@real
def test_decoupled_equals_official_where_m_is_one(lab200):
    x = lab200
    dec = E.decoupled_ep(x)
    assert dec.dtype == np.float32 and dec.shape == x.shape[:-1]
    M = x[..., L["nc"]] * x[..., L["dac"]] * x[..., L["ddc"]]
    m1 = M == 1
    assert m1.sum() > 1000 and (~m1).sum() > 1000                            # both regimes present
    assert np.abs(dec[m1] - x[..., L["ep"]][m1]).max() <= 1e-6
    assert np.isfinite(dec).all() and dec.min() >= 0.0 and dec.max() <= 1.0
    # official EP is the decoupled formula with r * M (the definition this helper drops the factor from)
    r, p = x[..., L["raw_progress"]].astype(np.float64), x[..., L["pdm_progress_eff"]].astype(np.float64)
    rm = r * M
    mx = np.maximum(p, rm)
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where(mx > 5.0, np.clip(rm / mx, 0, 1), (M > 0).astype(np.float64))
    assert np.abs(off - x[..., L["ep"]]).max() < 1e-6
    # decoupled >= official everywhere (M in {0, 0.5, 1}); strictly larger mean
    assert (dec >= x[..., L["ep"]] - 1e-6).all() and dec.mean() > x[..., L["ep"]].mean()
    y = E.ck_targets(x, "decoupled")
    assert y.dtype == np.float32
    assert np.array_equal(np.delete(y, 2, -1), np.delete(x[..., CKI], 2, -1))   # only the EP column changes
    assert np.array_equal(y[..., 2], dec)


def test_threshold_rule_and_nan():
    rows = np.stack([
        _row(10.0, 20.0),           # p > r: 0.5
        _row(20.0, 10.0),           # r > p: clip(20 / 20) = 1
        _row(4.0, 4.9),             # max <= 5: 1
        _row(5.0, 5.0),             # max == 5 (not > 5): 1
        _row(5.0, 5.0001),          # just over: 5 / 5.0001
        _row(-3.0, 10.0),           # negative progress: clip -> 0
        _row(6.0, 12.0, nc=0.0, ep=0.0),     # NC fail: official 0, decoupled 0.5
        _row(np.nan, 10.0),         # non-finite -> NaN
        _row(10.0, np.inf),
        _row(0.0, 0.0),             # standing still on a 0-progress route: 1
    ]).astype(np.float32)
    want = np.array([0.5, 1.0, 1.0, 1.0, np.float32(5.0) / np.float32(5.0001), 0.0, 0.5, np.nan, np.nan, 1.0])
    got = E.decoupled_ep(rows)
    np.testing.assert_allclose(got, want.astype(np.float32), rtol=0, atol=1e-7, equal_nan=True)
    y = E.ck_targets(rows, "decoupled")
    assert np.isnan(y[7, 2]) and np.isnan(y[8, 2]) and np.isfinite(np.delete(y[7:9], 2, -1)).all()
    assert np.isfinite(E.ck_targets(rows, "official")).all()
    # float64 input -> float64 output; integer input -> float32
    assert E.decoupled_ep(rows.astype(np.float64)).dtype == np.float64
    assert E.ck_targets(np.ones((2, 9), np.int64), "decoupled").dtype == np.float32


@real
def test_torch_equals_numpy(lab200):
    x = lab200[:50].copy()
    x[0, :5, L["raw_progress"]] = np.nan
    x[1, :5, L["pdm_progress_eff"]] = -np.inf
    for ep in E.EP_TARGETS:
        yn = E.ck_targets(x, ep)
        yt = E.ck_targets(torch.from_numpy(x), ep)
        assert isinstance(yt, torch.Tensor) and yt.dtype == torch.float32
        assert np.array_equal(yt.numpy(), yn, equal_nan=True)
        y64 = E.ck_targets(torch.from_numpy(x.astype(np.float64)), ep)
        assert y64.dtype == torch.float64
        assert np.array_equal(y64.numpy(), E.ck_targets(x.astype(np.float64), ep), equal_nan=True)
    dn = E.decoupled_ep(x)
    assert np.isnan(dn[0, :5]).all() and np.isnan(dn[1, :5]).all() and np.isfinite(dn[2:]).all()
    assert np.array_equal(E.decoupled_ep(torch.from_numpy(x)).numpy(), dn, equal_nan=True)
    # the torch input is not modified
    xt = torch.from_numpy(x.copy())
    E.ck_targets(xt, "decoupled")
    assert np.array_equal(xt.numpy(), x, equal_nan=True)
