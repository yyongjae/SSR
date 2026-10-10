"""CK2 e2e T4: per-token variant column draw (navsim/agents/para_ssr/ck/e2e_data2.py type_balanced_cols / uniform_cols /
sample_cols; SPEC ck2e2e §1-3 / NU10).  CPU, no data files.

  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_sampling.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.ck import e2e_data2 as D2
from navsim.agents.para_ssr.ck.anchor_sampler import token_rng


def _rng(i=0):
    return np.random.default_rng(i)


def _types(cols):
    return np.bincount(np.asarray(cols) % 6, minlength=6)


def test_constants():
    assert (D2.K16, D2.NV, D2.G_K) == (16, 6, 96)
    from navsim.agents.para_ssr.ck.variants import LATS, SPEEDS, variant_name, variant_table
    assert tuple(variant_name(a, d) for a, d in variant_table(SPEEDS, LATS, "separate")) == D2.VNAMES


@pytest.mark.parametrize("seed", range(20))
def test_all_valid_counts_6_or_7(seed):
    cols, ok = D2.type_balanced_cols(np.ones((16, 6), bool), 32, _rng(seed))
    assert cols.dtype == np.int64 and ok.dtype == bool and cols.shape == ok.shape == (32,)
    t = _types(cols)
    assert t[0] == 0                                           # identity never drawn
    assert sorted(t[1:].tolist()) == [6, 6, 6, 7, 7]
    assert ok.all() and len(np.unique(cols)) == 32             # without replacement
    assert cols.min() >= 0 and cols.max() < 96


def test_extra_quota_types_vary():
    seen = set()
    for s in range(40):
        t = _types(D2.type_balanced_cols(np.ones((16, 6), bool), 32, _rng(s))[0])
        seen.add(tuple(np.flatnonzero(t == 7).tolist()))
    assert len(seen) > 3                                        # the +1 types are drawn, not fixed


def test_invalid_never_drawn_and_shortfall_redistributed():
    v = np.ones((16, 6), bool)
    v[:, 4] = False                                             # type l-0.5 entirely invalid
    v[3:, 5] = False                                            # type l+0.5: only 3 valid
    for s in range(20):
        cols, ok = D2.type_balanced_cols(v, 32, _rng(s))
        assert ok.all() and len(np.unique(cols)) == 32
        assert v.reshape(-1)[cols].all()                        # every drawn column valid
        t = _types(cols)
        assert t[4] == 0 and t[5] == 3                          # short types give all they have
        assert t[1:4].sum() == 29 and t[1:4].min() >= 6         # shortfall filled from the other types


def test_padding_and_zero_valid():
    v = np.zeros((16, 6), bool)
    v[0, 1] = v[2, 3] = v[5, 5] = True                         # 3 valid columns
    cols, ok = D2.type_balanced_cols(v, 32, _rng(1))
    assert ok.sum() == 3 and set(cols[ok].tolist()) == {1, 15, 35}
    assert set(cols.tolist()) == {1, 15, 35}                    # padding repeats valid columns only
    cols0, ok0 = D2.type_balanced_cols(np.zeros((16, 6), bool), 32, _rng(1))
    assert (cols0 == 0).all() and not ok0.any()
    # the identity column is ignored even if marked valid
    v1 = np.zeros((16, 6), bool)
    v1[:, 0] = True
    c1, o1 = D2.type_balanced_cols(v1, 8, _rng(0))
    assert not o1.any() and (c1 == 0).all()
    e, eo = D2.type_balanced_cols(np.ones((16, 6), bool), 0, _rng(0))
    assert e.shape == (0,) and eo.shape == (0,)
    with pytest.raises(ValueError):
        D2.type_balanced_cols(np.ones((16, 6), bool), -1, _rng(0))


def test_n80_takes_every_valid_column():
    v = np.ones((16, 6), bool)
    cols, ok = D2.type_balanced_cols(v, 80, _rng(3))
    assert ok.all() and sorted(cols.tolist()) == [c for c in range(96) if c % 6]
    v[7, 4] = False
    cols, ok = D2.type_balanced_cols(v, 80, _rng(3))
    assert ok.sum() == 79 and (~ok).sum() == 1 and 7 * 6 + 4 not in cols.tolist()


def test_determinism_and_accepts_flat_valid():
    v = np.random.default_rng(5).random((16, 6)) > 0.2
    a = D2.type_balanced_cols(v, 32, _rng(11))
    b = D2.type_balanced_cols(v.reshape(96), 32, _rng(11))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = D2.type_balanced_cols(v, 32, _rng(12))
    assert not np.array_equal(a[0], c[0])


def test_uniform_rule_equals_ck2dataset_variant_cols():
    """'uniform' = CK2Dataset.variant_cols (ck2_dataset.py:190-204) given the same rng."""
    v = np.random.default_rng(2).random((16, 6)) > 0.3
    for n in (8, 32, 80):
        got = D2.uniform_cols(v, n, _rng(4))
        valid = v.copy()
        valid[:, 0] = False
        cols = np.flatnonzero(valid.reshape(-1))
        rng = _rng(4)
        cols = cols[rng.permutation(len(cols))]
        if len(cols) >= n:
            exp = (cols[:n], np.ones(n, bool))
        else:
            exp = (np.resize(cols, n), np.arange(n) < len(cols))
        assert np.array_equal(got[0], exp[0]) and np.array_equal(got[1], exp[1])
    with pytest.raises(ValueError):
        D2.draw_cols(v, 4, _rng(0), "bogus")


def test_sample_cols_streams_and_independence():
    g = np.random.default_rng(9)
    valid = g.random((4, 96)) > 0.1
    toks = ["a" * 16, "b" * 16, "c" * 16, "d" * 16]
    c1, o1 = D2.sample_cols(valid, toks, 5, 0, D2.STREAM_NOW, 32)
    assert c1.shape == (4, 32) and o1.shape == (4, 32)
    # per token == the single-token draw with token_rng (independent of batch composition / order)
    for b, t in enumerate(toks):
        exp = D2.type_balanced_cols(valid[b].reshape(16, 6), 32, token_rng(t, 5, 0, D2.STREAM_NOW))
        assert np.array_equal(c1[b], exp[0]) and np.array_equal(o1[b], exp[1])
    rev, _ = D2.sample_cols(valid[::-1], toks[::-1], 5, 0, D2.STREAM_NOW, 32)
    assert np.array_equal(rev[::-1], c1)
    # tensor input accepted
    ct, _ = D2.sample_cols(torch.from_numpy(valid), toks, 5, 0, D2.STREAM_NOW, 32)
    assert np.array_equal(ct, c1)
    # different epoch / stream / seed -> different draws
    for args in ((6, 0, D2.STREAM_NOW), (5, 1, D2.STREAM_NOW), (5, 0, D2.STREAM_LAB), (5, 0, D2.STREAM_WU)):
        c2, _ = D2.sample_cols(valid, toks, *args, 32)
        assert not np.array_equal(c2, c1), args
    assert len({D2.STREAM_WU, D2.STREAM_NOW, D2.STREAM_LAB}) == 3
