"""listwise head (ck/listwise.py), CPU, synthetic pool:

    L1  zero-initialised head: argmax(s0 + delta) == select2.select96 (CK2's selection, same ties / disallowed columns)
    L2  cross=True mixes columns (changing column j's input changes column i's delta); cross=False does not
    L3  the loss only sees allowed columns and its descent raises the expected PDMS

    cd ~/kyungmin/SSR-ck2km && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= \
      ~/miniconda3/envs/ssr/bin/python -m pytest -q -p no:cacheprovider tests/km/test_listwise.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from navsim.agents.para_ssr.ck import listwise as LW  # noqa: E402
from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402


def _pool(B=64, seed=0):
    g = np.random.default_rng(seed)
    v2f = g.normal(size=(B, 16)).astype(np.float32)
    v2f[:5, 3] = v2f[:5, 0]                                    # ties between parents
    im = g.dirichlet(np.ones(16), size=B).astype(np.float32)
    prob = (1 / (1 + np.exp(-g.normal(size=(B, 96, 5))))).astype(np.float32)
    valid = g.random((B, 96)) > 0.3
    return v2f, im, prob, valid


def test_zero_head_is_ck2():
    v2f, im, prob, valid = _pool()
    s0 = S2.pool_scores(v2f, im, prob, 0.5)
    allowed = S2.allowed_mask(valid, S2.VARIANT_SETS["all"])
    head = LW.ListwiseHead().eval()
    B = len(v2f)
    x = torch.randn(B, 96, LW.D_IN)
    with torch.no_grad():
        d = head(x, torch.as_tensor(allowed)).double().numpy()
    assert np.all(d == 0)
    got = LW.select(s0 + d, allowed)
    want = S2.select96(v2f, im, prob, valid, 0.5, S2.VARIANT_SETS["all"])
    assert np.array_equal(got, want)


def test_cross_mixes_columns():
    torch.manual_seed(0)
    for cross in (True, False):
        head = LW.ListwiseHead(cross=cross, dropout=0.0).eval()
        torch.nn.init.normal_(head.out.weight)
        x = torch.randn(2, 96, LW.D_IN)
        allowed = torch.ones(2, 96, dtype=torch.bool)
        y = x.clone()
        y[:, 50] = 5.0 * torch.randn(2, LW.D_IN)
        with torch.no_grad():
            d0, d1 = head(x, allowed), head(y, allowed)
        moved = float((d1[:, :50] - d0[:, :50]).abs().max())
        assert (moved > 1e-6) if cross else (moved == 0.0), (cross, moved)


def test_loss_masks_and_descends():
    torch.manual_seed(0)
    allowed = torch.ones(4, 96, dtype=torch.bool)
    allowed[:, 1::6] = False
    pdms = torch.rand(4, 96)
    pdms_nan = pdms.masked_fill(~allowed, float("nan"))
    r = torch.zeros(4, 96, requires_grad=True)
    L = LW.listwise_loss(r, pdms_nan, allowed, 0.1)
    assert torch.isfinite(L["loss"])
    exp0 = float(L["exp_pdms"])
    want = float((pdms * allowed).sum(-1).div(allowed.sum(-1)).mean())
    assert abs(exp0 - want) < 1e-6
    opt = torch.optim.SGD([r], lr=1.0)
    for _ in range(20):
        L = LW.listwise_loss(r, pdms_nan, allowed, 0.1)
        opt.zero_grad()
        L["loss"].backward()
        opt.step()
    assert float(L["exp_pdms"]) > exp0 + 0.05
    assert torch.all(r.grad[~allowed] == 0)
