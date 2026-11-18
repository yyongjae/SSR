"""Variant-set ablation arms (v8 / v4 / spd4 / lat4) in selection + evaluation (group B-select-eval): ck/select2.py
with a VariantLayout, tools/ck/e2e2/eval_ck2.py (layout from the eval root, --tokens), eval_e2e2.py (writer / meta /
check_pool / summary), eval_epdms.py (pool layout, selection) and the base-only teacher navtest tools.  CPU, synthetic arrays, no
model, nothing under /workspace/data:
  cd /workspace/SSR && set -a && . tools/release/paths.env && set +a && PYTHONDONTWRITEBYTECODE=1 \
    /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_variant_arms_select.py

Base identity: the base layout (default) reproduces the v2_final select2 constants and functions exactly -- against
hard-coded constants, a frozen copy of the old functions below, and (when git has the ref) v2_final's select2.py itself.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.select import blend, ck_final, select_all  # noqa: E402
from navsim.agents.para_ssr.ck.variants import BASE_LAYOUT, VariantLayout, arm_layout  # noqa: E402

ARMS = ("v8", "v4", "spd4", "lat4", "cross", "x13")
ALL_ARMS = ("base",) + ARMS
PLANT = {"base": "a+0.5", "v8": "a+1.0", "v4": "a-0.5", "spd4": "a+1.0", "lat4": "l-1.0",   # a type the arm has
         "cross": "a-0.5|l+1.0", "x13": "a+0.5|l-0.5"}


# ----------------------------------------------------------------------------------------------- frozen v2_final copy
_OLD_NV, _OLD_G = 6, 96
_OLD_SETS = {"none": (0,), "decel": (0, 1, 2), "speed": (0, 1, 2, 3), "lat": (0, 4, 5),
             "all_noaccel": (0, 1, 2, 4, 5), "all": (0, 1, 2, 3, 4, 5)}
_OLD_LAT = (4, 5)
_OLD_TIE = np.concatenate([np.arange(0, 96, 6), np.array([c for c in range(96) if c % 6], np.int64)])


def _onp(x, dtype=None):
    a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return a if dtype is None else a.astype(dtype)


def _old_allowed(valid96, types=_OLD_SETS["all"]):
    v = _onp(valid96).astype(bool)
    t = np.arange(_OLD_G) % _OLD_NV
    in_set = np.isin(t, np.asarray(tuple(types), np.int64)) | (t == 0)
    return (v | (t == 0)[None]) & in_set[None]


def _old_select96(v2f16, im16, prob96, valid96, beta, types=_OLD_SETS["all"], offset=None, w=Cn.SEL_W):
    v2f = np.repeat(_onp(v2f16, np.float64), 6, 1)
    im = np.repeat(_onp(im16, np.float64), 6, 1)
    p = S2.apply_offset(_onp(prob96, np.float64), offset)
    s = blend(v2f, ck_final(p, im, w), beta)
    ok = _old_allowed(valid96, types) & np.isfinite(s)
    s = np.where(ok, s, -np.inf)
    return _OLD_TIE[np.argmax(s[:, _OLD_TIE], axis=1)].astype(np.int64)


def _old_select96_pdms(logit96, valid96, types=_OLD_SETS["all"]):
    z = _onp(logit96, np.float64)
    s = S2.exp_pdms96(1.0 / (1.0 + np.exp(-z)))
    ok = _old_allowed(valid96, types) & np.isfinite(s)
    s = np.where(ok, s, -np.inf)
    return _OLD_TIE[np.argmax(s[:, _OLD_TIE], axis=1)].astype(np.int64)


def _old_use_lat(idx, mode):
    i = _onp(idx).astype(np.int64)
    if mode == "on":
        return np.ones(i.shape, bool)
    if mode == "off":
        return np.zeros(i.shape, bool)
    return ~np.isin(i % 6, _OLD_LAT)


def _rand_base(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 30))
    v2f = np.round(rng.normal(size=(n, 16)), 1 if seed % 2 else 3)
    im = rng.dirichlet(np.ones(16), size=n)
    prob = rng.uniform(0, 1, size=(n, 96, 5))
    if seed % 3 == 0:
        prob = np.round(prob, 1)
        prob[rng.random((n, 96)) < 0.05] = np.nan
    logit = np.round(rng.normal(0, 3, size=(n, 96, 5)), 0 if seed % 2 else 4)
    valid = rng.random((n, 96)) > rng.uniform(0, 0.9)
    return rng, n, v2f, im, prob, logit, valid


# ----------------------------------------------------------------------------------------------- base identity
def test_base_constants_are_the_historical_ones():
    assert (S2.K16, S2.NV, S2.G_K) == (16, 6, 96)
    assert S2.VNAMES == ("id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5") == BASE_LAYOUT.names
    assert S2.VARIANT_SETS == _OLD_SETS and tuple(S2.VARIANT_SETS) == S2.SET_NAMES == VariantLayout.SET_NAMES
    assert S2.LAT_TYPES == _OLD_LAT == BASE_LAYOUT.lat_types
    assert np.array_equal(S2.TIE_ORDER, _OLD_TIE) and S2.TIE_ORDER.dtype == np.int64
    assert S2.tie_order() is S2.TIE_ORDER and S2.tie_order(VariantLayout.from_cfg({})) is S2.TIE_ORDER
    assert {k: tuple(v) for k, v in BASE_LAYOUT.sets.items()} == _OLD_SETS
    for name, types in _OLD_SETS.items():
        assert S2.set_types(name) == types == S2.set_types(types)
    assert S2.set_types(None) == _OLD_SETS["all"]
    assert np.array_equal(S2.parent_of(np.arange(32).reshape(2, 16)), np.repeat(np.arange(32).reshape(2, 16), 6, 1))


@pytest.mark.parametrize("seed", range(40))
def test_base_selection_identical_to_frozen_v2_final(seed):
    rng, n, v2f, im, prob, logit, valid = _rand_base(seed)
    beta = float(rng.choice([0.0, 0.5, 1.0, rng.uniform()]))
    off = None if seed % 2 else list(rng.normal(size=5))
    w = (Cn.SEL_W, "plugin", "noim", tuple(rng.uniform(0, 1, 4)))[seed % 4]
    lay_cfg = VariantLayout.from_cfg({"speeds": [-1.0, -0.5, 0.5], "lats": [-0.5, 0.5]})
    for kw in ({}, {"layout": BASE_LAYOUT}, {"layout": lay_cfg}):
        for name, types in _OLD_SETS.items():
            want = _old_select96(v2f, im, prob, valid, beta, types, off, w)
            for t in (types, name, S2.VARIANT_SETS[name]):
                got = S2.select96(v2f, im, prob, valid, beta, t, off, w=w, **kw)
                assert got.dtype == want.dtype and np.array_equal(got, want), (name, kw)
            wp = _old_select96_pdms(logit, valid, types)
            assert np.array_equal(S2.select96_pdms(logit, valid, name, **kw), wp)
            assert np.array_equal(S2.allowed_mask(valid, name, **kw), _old_allowed(valid, types))
        assert np.array_equal(S2.select96(v2f, im, prob, valid, beta, **kw), _old_select96(v2f, im, prob, valid, beta))
        assert np.array_equal(S2.select96_pdms(logit, valid, **kw), _old_select96_pdms(logit, valid))
        idx = rng.integers(0, 96, size=n)
        cand = rng.normal(size=(n, 96, 8, 3)).astype(np.float32)
        lat = rng.normal(size=(n, 96, 8, 3)).astype(np.float32)
        lp, ll = rng.random((n, 96, 9)), rng.random((n, 96, 9))
        ar = np.arange(n)
        for mode in S2.LAT_MODES:
            u = _old_use_lat(idx, mode)
            assert np.array_equal(S2.use_lat(idx, mode, **kw), u)
            t, src = S2.final_traj(cand, lat, idx, mode, **kw)
            assert t.dtype == np.float32 and np.array_equal(t, np.where(u[:, None, None], lat[ar, idx], cand[ar, idx]))
            assert src.tolist() == np.where(u, "lat", "pool").tolist()
            tt, _ = S2.final_traj(torch.tensor(cand), torch.tensor(lat), torch.tensor(idx), mode, **kw)
            assert torch.equal(tt, torch.tensor(t))
            assert np.array_equal(S2.chosen_labels(lp, ll, idx, mode, **kw),
                                  np.where(u[:, None], ll[ar, idx], lp[ar, idx]))


def test_base_selection_identical_to_git_v2_final(tmp_path):
    """the real v2_final select2.py (git show) on random inputs; skipped without git / the ref."""
    r = subprocess.run(["git", "-C", str(REPO), "show", "v2_final:navsim/agents/para_ssr/ck/select2.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git show v2_final: {r.stderr.strip()[:200]}")
    f = tmp_path / "old_select2.py"
    f.write_text(r.stdout)
    spec = importlib.util.spec_from_file_location("navsim.agents.para_ssr.ck._v2final_select2", f)
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    for k in ("K16", "NV", "G_K", "VNAMES", "VARIANT_SETS", "LAT_MODES", "LAT_TYPES"):
        assert getattr(OLD, k) == getattr(S2, k), k
    assert np.array_equal(OLD.TIE_ORDER, S2.TIE_ORDER)
    for seed in range(25):
        rng, n, v2f, im, prob, logit, valid = _rand_base(seed)
        beta = float(rng.uniform())
        for name, types in OLD.VARIANT_SETS.items():
            assert np.array_equal(OLD.select96(v2f, im, prob, valid, beta, types, w="plugin"),
                                  S2.select96(v2f, im, prob, valid, beta, name, w="plugin"))
            assert np.array_equal(OLD.select96_pdms(logit, valid, types), S2.select96_pdms(logit, valid, types))
        assert np.array_equal(OLD.pool_scores(v2f, im, prob, beta), S2.pool_scores(v2f, im, prob, beta), equal_nan=True)
        idx = rng.integers(0, 96, size=n)
        lp, ll = rng.random((n, 96, 9)), rng.random((n, 96, 9))
        cand, lat = rng.normal(size=(n, 96, 8, 3)), rng.normal(size=(n, 96, 8, 3))
        for mode in OLD.LAT_MODES:
            assert np.array_equal(OLD.chosen_labels(lp, ll, idx, mode), S2.chosen_labels(lp, ll, idx, mode))
            assert np.array_equal(OLD.final_traj(cand, lat, idx, mode)[0], S2.final_traj(cand, lat, idx, mode)[0])


# ----------------------------------------------------------------------------------------------- select2 per arm
EXPECT_SETS = {
    "v8": {"none": (0,), "decel": (0, 1, 2), "speed": (0, 1, 2, 3, 4), "lat": (0, 5, 6, 7, 8),
           "all_noaccel": (0, 1, 2, 5, 6, 7, 8), "all": tuple(range(9))},
    "v4": {"none": (0,), "decel": (0, 1), "speed": (0, 1, 2), "lat": (0, 3, 4), "all_noaccel": (0, 1, 3, 4),
           "all": tuple(range(5))},
    "spd4": {"none": (0,), "decel": (0, 1, 2), "speed": (0, 1, 2, 3, 4), "lat": (0,), "all_noaccel": (0, 1, 2),
             "all": tuple(range(5))},
    "lat4": {"none": (0,), "decel": (0,), "speed": (0,), "lat": (0, 1, 2, 3, 4), "all_noaccel": (0, 1, 2, 3, 4),
             "all": tuple(range(5))},
}
EXPECT_SETS["none"] = {s: (0,) for s in ("none", "decel", "speed", "lat", "all_noaccel", "all")}
# cross: 0 id, 1-4 a-1.0 a-0.5 a+0.5 a+1.0, 5-8 l-1.0 l-0.5 l+0.5 l+1.0, then the combos a-major: 9-12 a-1.0|l*,
# 13-16 a-0.5|l*, 17-20 a+0.5|l*, 21-24 a+1.0|l* (lat = pure lateral only; every combo is a lateral type)
EXPECT_SETS["cross"] = {"none": (0,), "decel": (0, 1, 2), "speed": (0, 1, 2, 3, 4), "lat": (0, 5, 6, 7, 8),
                        "all_noaccel": (0, 1, 2, 5, 6, 7, 8) + tuple(range(9, 17)), "all": tuple(range(25))}
EXPECT_LAT = {"v8": (5, 6, 7, 8), "v4": (3, 4), "spd4": (), "lat4": (1, 2, 3, 4), "none": (),
              "cross": tuple(range(5, 25))}
# x13: 0-8 as v8 / cross, then 9 a-0.5|l-0.5, 10 a-0.5|l+0.5, 11 a+0.5|l-0.5, 12 a+0.5|l+0.5 (combos are lateral types)
EXPECT_SETS["x13"] = {"none": (0,), "decel": (0, 1, 2), "speed": (0, 1, 2, 3, 4), "lat": (0, 5, 6, 7, 8),
                      "all_noaccel": (0, 1, 2, 5, 6, 7, 8, 9, 10), "all": tuple(range(13))}
EXPECT_LAT["x13"] = tuple(range(5, 13))


def _meaning_ok(layout: VariantLayout, name: str, t: int) -> bool:
    a, d = layout.table[t]
    return t == 0 or {"none": False, "decel": a < 0 and d == 0, "speed": a != 0 and d == 0, "lat": d != 0 and a == 0,
                      "all_noaccel": a <= 0, "all": True}[name]


def _rand_arm(layout, n=40, seed=0):
    rng = np.random.default_rng(seed)
    G = layout.g()
    v2f = np.sort(rng.normal(size=(n, 16)), 1)[:, ::-1].copy()
    im = rng.dirichlet(np.ones(16), size=n)
    prob = rng.uniform(0.05, 0.95, size=(n, G, 5))
    valid = rng.random((n, G)) > 0.2
    valid[:, ::layout.nv] = True
    return rng, v2f, im, prob, valid


@pytest.mark.parametrize("arm", ARMS)
def test_arm_layout_sets_lat_types_and_tie_order(arm):
    L = arm_layout(arm)
    assert {k: tuple(v) for k, v in L.sets.items()} == EXPECT_SETS[arm] and tuple(L.lat_types) == EXPECT_LAT[arm]
    assert L.g() == {"v8": 144, "cross": 400, "x13": 208}.get(arm, 80) and not L.is_base
    tie = S2.tie_order(L)
    assert sorted(tie.tolist()) == list(range(L.g())) and tie[:16].tolist() == list(range(0, L.g(), L.nv))
    assert not tie.flags.writeable and S2.tie_order(L) is tie                       # cached per layout
    for name in S2.SET_NAMES:
        assert S2.set_types(name, layout=L) == EXPECT_SETS[arm][name]
        assert all(_meaning_ok(L, name, t) for t in EXPECT_SETS[arm][name])
    # the base type-id tuples are refused with an arm layout where the set means other ids (wrong variants); names work
    for name in S2.SET_NAMES:
        for base_t in (S2.VARIANT_SETS[name], BASE_LAYOUT.sets[name]):
            if tuple(base_t) == EXPECT_SETS[arm][name]:
                assert S2.set_types(base_t, layout=L) == EXPECT_SETS[arm][name]          # e.g. 'none' = (0,)
            else:
                with pytest.raises(ValueError, match="variant-set name"):
                    S2.set_types(base_t, layout=L)
    for name in ("lat", "all"):
        with pytest.raises(ValueError, match="variant-set name"):
            S2.select96_pdms(np.zeros((1, L.g(), 5)), np.ones((1, L.g()), bool), S2.VARIANT_SETS[name], layout=L)
    with pytest.raises(ValueError, match="outside"):
        S2.set_types((0, L.nv), layout=L)
    with pytest.raises(ValueError, match="unknown variant set"):
        S2.set_types("lateral", layout=L)
    k, v, nm = S2.decode_cols(np.array([0, L.nv + 1, L.g() - 1]), layout=L)
    assert k.tolist() == [0, 1, 15] and v.tolist() == [0, 1, L.nv - 1]
    assert list(nm) == [L.names[0], L.names[1], L.names[-1]]


@pytest.mark.parametrize("arm", ARMS)
def test_arm_pool_scores_select_and_shape_guards(arm):
    L = arm_layout(arm)
    nv, G = L.nv, L.g()
    rng, v2f, im, prob, valid = _rand_arm(L, seed=hash(arm) % 1000)
    s0 = S2.pool_scores(v2f, im, prob, 0.0, layout=L)
    assert s0.shape == (40, G) and np.array_equal(s0, np.repeat(v2f, nv, 1))     # beta 0 = parent's v2 score
    np.testing.assert_allclose(S2.pool_scores(v2f, im, prob, 1.0, layout=L), ck_final(prob, np.repeat(im, nv, 1)),
                               rtol=0, atol=0)
    for beta in Cn.BETA_GRID:                              # identity-only set == old select_all 'a' over the originals
        a = select_all(v2f, im, prob[:, ::nv], beta=beta)["a"]
        assert np.array_equal(S2.select96(v2f, im, prob, valid, beta, "none", layout=L), nv * a)
    assert (S2.select96(v2f, im, prob, valid, 0.0, layout=L) == 0).all()          # ties -> identity -> column 0
    logit = rng.normal(size=prob.shape)
    for name in S2.SET_NAMES:
        for idx in (S2.select96(v2f, im, prob, valid, 1.0, name, layout=L),
                    S2.select96_pdms(logit, valid, name, layout=L)):
            assert idx.max() < G and valid[np.arange(40), idx].all()
            assert all(_meaning_ok(L, name, int(t)) for t in idx % nv), (arm, name)
    # a planted best variant column (k 1, PLANT type) wins for every set holding its type, never for the others
    c0 = nv + L.names.index(PLANT[arm])
    p2 = prob.copy()
    p2[:, c0] = 0.999
    v2 = valid.copy()
    v2[:, c0] = True
    imu = np.full_like(im, 1 / 16)                     # uniform im: the planted probabilities decide
    for name in S2.SET_NAMES:
        got = S2.select96(np.zeros_like(v2f), imu, p2, v2, 1.0, name, layout=L)
        assert ((got == c0).all()) == (L.names.index(PLANT[arm]) in L.sets[name]), name
    v2[:, c0] = False
    assert not (S2.select96(v2f, im, p2, v2, 1.0, layout=L) == c0).any()           # invalid -> never chosen
    # every function refuses a pool of the wrong width (base arrays with the arm layout and the reverse)
    b = _rand_arm(BASE_LAYOUT)
    with pytest.raises(ValueError, match="columns"):
        S2.select96(b[1], b[2], b[3], b[4], 1.0, layout=L)
    with pytest.raises(ValueError, match="columns"):
        S2.select96(v2f, im, prob, valid, 1.0)                                    # default base layout on an arm pool
    with pytest.raises(ValueError, match="columns"):
        S2.select96_pdms(logit, valid)
    with pytest.raises(ValueError, match="columns"):
        S2.allowed_mask(valid)
    cand = rng.normal(size=(40, G, 8, 3))
    with pytest.raises(ValueError, match="columns"):
        S2.final_traj(cand, cand, np.zeros(40, np.int64), "on")
    with pytest.raises(ValueError, match="columns"):
        S2.chosen_labels(cand[..., 0], cand[..., 0], np.zeros(40, np.int64), "on")
    with pytest.raises(ValueError, match="outside"):
        S2.use_lat(np.array([G]), "on_except_latvar", layout=L)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_lateral_rule_final_traj_and_labels(arm):
    L = arm_layout(arm)
    nv, G = L.nv, L.g()
    rng = np.random.default_rng(3)
    idx = np.arange(G)
    want = np.array([not (L.table[c % nv][1] != 0) for c in idx])                # lateral variants (d != 0) uncorrected
    assert np.array_equal(S2.use_lat(idx, "on_except_latvar", layout=L), want)
    assert S2.use_lat(idx, "on", layout=L).all() and not S2.use_lat(idx, "off", layout=L).any()
    if arm == "spd4":
        assert want.all()                                                         # no lateral variant at all
    if arm == "lat4":
        assert want.tolist() == [c % nv == 0 for c in idx]
    cand = rng.normal(size=(G, G, 8, 3)).astype(np.float32)
    lat = rng.normal(size=(G, G, 8, 3)).astype(np.float32)
    for mode in S2.LAT_MODES:
        t, src = S2.final_traj(cand, lat, idx, mode, layout=L)
        u = S2.use_lat(idx, mode, layout=L)
        assert np.array_equal(t, np.where(u[:, None, None], lat[idx, idx], cand[idx, idx]))
        tt, _ = S2.final_traj(torch.tensor(cand), torch.tensor(lat), torch.tensor(idx), mode, layout=L)
        assert torch.equal(tt, torch.tensor(t)) and (src == "lat").tolist() == u.tolist()
        lp, ll = rng.random((G, G, 9)), rng.random((G, G, 9))
        assert np.array_equal(S2.chosen_labels(lp, ll, idx, mode, layout=L),
                              np.where(u[:, None], ll[idx, idx], lp[idx, idx]))


def test_layout_from_cfg_and_meta():
    for arm in ALL_ARMS:
        L = arm_layout(arm)
        assert S2.layout_from_cfg({"speeds": list(L.speeds), "lats": list(L.lats), "combine": L.combine,
                                   **L.cross_kw()}) == L
        assert S2.layout_from_cfg({"speeds": "[" + ", ".join(map(str, L.speeds)) + "]", "lats": str(list(L.lats)),
                                   **({} if L.combine == "separate" else {"combine": L.combine}),
                                   **{k: str(v) for k, v in L.cross_kw().items()}}) == L  # hydra strings
        assert S2.layout_from_meta(L.meta()) == L
    assert S2.layout_from_cfg(None) == S2.layout_from_cfg({}) == BASE_LAYOUT
    assert S2.layout_from_cfg({"s_on_frac": 0.2, "ext": "straight"}) == BASE_LAYOUT
    assert S2.layout_from_meta(None) == S2.layout_from_meta({}) == BASE_LAYOUT
    bad = dict(arm_layout("v4").meta(), names=list(arm_layout("lat4").names))
    with pytest.raises(ValueError, match="names"):
        S2.layout_from_meta(bad)
    with pytest.raises(ValueError, match="not a variant table"):
        S2.layout_from_cfg({"speeds": "${oops}"})
    # the three 80-column arms are different layouts (never told apart by shape)
    v4, spd4, lat4 = (arm_layout(a) for a in ("v4", "spd4", "lat4"))
    assert v4.g() == spd4.g() == lat4.g() == 80 and len({v4, spd4, lat4}) == 3


# ----------------------------------------------------------------------------------------------- eval_ck2 per arm
def _arm_root(root: Path, run: str, split: str, layout: VariantLayout, n: int = 30, seed: int = 0, plant: str = None,
              meta: bool = True, extra_meta: dict = None):
    """eval root of one split in `layout`: column k 1 x `plant` labelled PDMS 1 with the highest CK probability,
    column 0 PDMS 0.5, laterally corrected labels 0.1 worse; token 0 packed not ok."""
    rng = np.random.default_rng(seed)
    G, nv = layout.g(), layout.nv
    pk, inf, lb = root / "packed" / split, root / "infer" / run / split, root / "labels" / split
    for d in (pk, inf, lb / f"pool96_{run}", lb / f"lat96_{run}"):
        d.mkdir(parents=True, exist_ok=True)
    toks = [f"t{split[:3]}{i:04d}" for i in range(n)]
    pd.DataFrame({"token": toks, "log": [f"log{i % 5}" for i in range(n)], "city": "x", "row": np.arange(n)}
                 ).to_parquet(pk / "tokens.parquet")
    c0 = nv + layout.names.index(plant or PLANT.get(_arm_of(layout), layout.names[1]))
    valid = rng.random((n, G)) > 0.1
    valid[:, ::nv] = True
    valid[:, c0] = True
    np.save(pk / "valid96.npy", valid)
    np.save(pk / "v2_final.npy", np.sort(rng.normal(size=(n, 16)), 1)[:, ::-1].astype(np.float32))
    np.save(pk / "v2_im.npy", rng.dirichlet(np.ones(16), size=n).astype(np.float32))
    ok = np.ones(n, bool)
    ok[0] = False
    np.save(pk / "ok.npy", ok)
    logit = rng.normal(-1.0, 0.5, size=(n, G, 5)).astype(np.float32)
    logit[:, c0] = 6.0
    np.save(inf / "score_logit.npy", logit)
    np.save(inf / "done.npy", np.ones(n, bool))
    L_ = Cn.LBL
    lab = np.full((n, G, 9), 0.5, np.float32)
    lab[..., L_["nc"]] = lab[..., L_["ttc"]] = 1.0
    lab[:, c0, L_["pdms"]] = 1.0
    labl = lab.copy()
    labl[..., L_["pdms"]] -= 0.1
    for name, arr in ((f"pool96_{run}", lab), (f"lat96_{run}", labl)):
        np.save(lb / name / "labels.npy", arr)
        np.save(lb / name / "ok.npy", np.ones((n, G), bool))
    if meta:
        m = {"ckpt": "/r/last.ckpt", "ckpt_sha16": "a" * 16, "limit": 0, "variant_layout": layout.meta(),
             **(extra_meta or {})}
        for d in (pk, inf):
            (d / "meta.json").write_text(json.dumps(m))
    return toks, c0


def _arm_of(layout: VariantLayout) -> str:
    return next((a for a in ALL_ARMS if arm_layout(a) == layout), "")


@pytest.mark.parametrize("arm", ARMS)
def test_eval_ck2_arm_val_choice_navtest_and_per_token(tmp_path, arm):
    from tools.ck.e2e2 import eval_ck2 as EV
    L = arm_layout(arm)
    root, run, lead = tmp_path / "root", f"ck2e2e_{arm}", tmp_path / "nolead"
    _, c0 = _arm_root(root, run, "navtrain_val", L, seed=0)
    _arm_root(root, run, "navtest", L, seed=1)
    m = EV.run_eval(root, run, "navtrain_val", n_boot=20, lead_root=lead)
    assert m["n_eval"] == 29 and m["n_total"] == 30
    assert m["variant_layout"] == L.meta() and m["variant_sets"] == {k: list(v) for k, v in L.sets.items()}
    assert sum(r["variant"] == "sel" for r in m["rows"]) == 5 * 6 * 3                # same grid for every arm
    rows = {r["key"]: r for r in m["rows"]}
    assert all(set(r["sel_share"]) == set(L.names) for r in m["rows"])
    assert rows["oracle96"]["pdms"] == pytest.approx(1.0) and rows["v2"]["pdms"] == pytest.approx(0.5)
    assert rows["oracle16"]["pdms"] == pytest.approx(0.5)                          # identities only (c % nv == 0)
    best = m["best"]
    assert best["set"] == "all" and best["lat_mode"] == "on" and best["pdms"] == pytest.approx(0.9)
    assert rows[best["key"]]["sel_share"][PLANT[arm]] == pytest.approx(1.0)
    pv = L.names.index(PLANT[arm])
    for name in S2.SET_NAMES:                       # a set without the planted type never submits it
        r = rows[f"sel b=1 set={name} lat=off"]
        assert (r["pdms"] == pytest.approx(1.0)) == (pv in L.sets[name]), name
    # lateral rule by meaning: lat4's planted l-1.0 is a lateral variant -> on_except_latvar submits it uncorrected
    r = rows["sel b=1 set=all lat=on_except_latvar"]
    assert r["frac_lat"] == (0.0 if L.table[pv][1] != 0 else 1.0)
    spec = m["eval_spec"]
    assert spec["variant_layout"] == L.meta() and spec["set_types"] == list(L.sets["all"])
    pt = pd.read_parquet(Path(m["out_dir"]) / "per_token.parquet")
    assert (pt.sel_col == c0).all() and (pt.sel_k == 1).all() and (pt.sel_v == pv).all()
    assert (pt.sel_name == PLANT[arm]).all() and pt.sel_lat.all()
    t = EV.run_eval(root, run, "navtest", fix_from=str(Path(m["out_dir"]) / "metrics.json"), n_boot=20, lead_root=lead)
    assert t["representative"] == best["key"] and t["variant_layout"] == L.meta()
    assert next(r for r in t["rows"] if r["key"] == best["key"])["pdms"] == pytest.approx(0.9)


def test_eval_ck2_refuses_confused_layouts(tmp_path):
    """v4 / spd4 / lat4 share the 80-column shape: the recorded layout decides, mismatches are refused."""
    from tools.ck.e2e2 import eval_ck2 as EV
    v4, lat4 = arm_layout("v4"), arm_layout("lat4")
    lead = tmp_path / "nolead"
    # a val spec chosen on v4 cannot fix a navtest dump recorded as lat4 (same shape)
    root, run = tmp_path / "r1", "runA"
    _arm_root(root, run, "navtrain_val", v4, plant="a-0.5")
    _arm_root(root, run, "navtest", lat4, plant="l-1.0", seed=1)
    m = EV.run_eval(root, run, "navtrain_val", n_boot=10, lead_root=lead)
    with pytest.raises(SystemExit, match="variant layout"):
        EV.run_eval(root, run, "navtest", fix_from=str(Path(m["out_dir"]) / "metrics.json"), n_boot=10, lead_root=lead)
    # an 80-column dump without a recorded layout is read as base and refused (never guessed from the shape)
    root2 = tmp_path / "r2"
    _arm_root(root2, run, "navtrain_val", v4, plant="a-0.5", meta=False)
    with pytest.raises(SystemExit, match="variant layout"):
        EV.run_eval(root2, run, "navtrain_val", n_boot=10, lead_root=lead)
    # packed vs infer meta, and packed vs the extracted ck_run config, must agree
    root3 = tmp_path / "r3" / "root"
    _arm_root(root3, run, "navtrain_val", v4, plant="a-0.5")
    (root3 / "infer" / run / "navtrain_val" / "meta.json").write_text(json.dumps({"variant_layout": lat4.meta()}))
    with pytest.raises(SystemExit, match="disagree"):
        EV.run_eval(root3, run, "navtrain_val", n_boot=10, lead_root=lead)
    (root3 / "infer" / run / "navtrain_val" / "meta.json").write_text(json.dumps({"variant_layout": v4.meta()}))
    (root3.parent / "ck_run").mkdir(parents=True)
    (root3.parent / "ck_run" / "config.json").write_text(json.dumps({"ck_e2e2": {"variants": {
        "speeds": [], "lats": [-1.0, -0.5, 0.5, 1.0], "combine": "separate"}}}))
    with pytest.raises(SystemExit, match="disagree"):
        EV.run_eval(root3, run, "navtrain_val", n_boot=10, lead_root=lead)
    (root3.parent / "ck_run" / "config.json").write_text(json.dumps({"ck_e2e2": {"variants": {
        "speeds": "[-0.5, 0.5]", "lats": "[-0.5, 0.5]", "combine": "separate"}}}))
    assert EV.layout_of_root(root3, run, "navtrain_val") == v4
    # base arrays with a recorded non-base layout are refused too
    root4 = tmp_path / "r4"
    _arm_root(root4, run, "navtrain_val", BASE_LAYOUT, plant="a+0.5", extra_meta={"variant_layout": v4.meta()})
    with pytest.raises(SystemExit, match="variant layout"):
        EV.run_eval(root4, run, "navtrain_val", n_boot=10, lead_root=lead)


def test_eval_ck2_base_outputs_keep_their_format(tmp_path, capsys):
    """base (no recorded layout = an old dump): no spec / per-token additions, only the 'variant_layout' key appended
    to metrics.json (no 'variant_sets'), and the v2_final stdout line."""
    from tools.ck.e2e2 import eval_ck2 as EV
    root, run = tmp_path / "root", "base_run"
    _arm_root(root, run, "navtrain_val", BASE_LAYOUT, meta=False)
    capsys.readouterr()
    m = EV.run_eval(root, run, "navtrain_val", n_boot=10, lead_root=tmp_path / "nolead")
    line = [x for x in capsys.readouterr().out.splitlines() if "eval_ck2 base_run/navtrain_val" in x]
    assert len(line) == 1 and line[0].split("] ", 1)[1] == (f"eval_ck2 base_run/navtrain_val: {m['n_eval']}/{m['n_total']} "
                                                            f"tokens, {len(m['rows'])} rows -> {m['out_dir']}")
    assert Path(m["out_dir"]).name == "navtrain_val"
    assert "variant_layout" not in m["eval_spec"] and "tokens" not in m["eval_spec"] and "tokens" not in m
    d = json.loads((Path(m["out_dir"]) / "metrics.json").read_text())
    assert list(d)[-1] == "variant_layout" and "variant_sets" not in d and d["variant_layout"] == BASE_LAYOUT.meta()
    assert list(pd.read_parquet(Path(m["out_dir"]) / "per_token.parquet").columns) == [
        "token", "log", "pdms_v2", "sel_col", "sel_lat", "pdms_sel"]
    assert "변형 배치" not in (Path(m["out_dir"]) / "table.md").read_text()


def test_eval_ck2_tokens_restriction(tmp_path):
    from tools.ck.e2e2 import eval_ck2 as EV
    root, run, lead = tmp_path / "root", "runT", tmp_path / "nolead"
    toks, _ = _arm_root(root, run, "navtrain_val", arm_layout("v8"))
    full = EV.run_eval(root, run, "navtrain_val", n_boot=10, lead_root=lead)
    assert full["n_eval"] == 29 and "tokens" not in full
    sub = toks[5:17]
    f = tmp_path / "common.txt"
    f.write_text("# common tokens\n" + "\n".join(sub + sub[:2]) + "\n")
    m = EV.run_eval(root, run, "navtrain_val", n_boot=10, lead_root=lead, tokens=str(f))
    assert m["n_eval"] == 12 and m["n_total"] == 30
    assert m["tokens"]["n_listed"] == 12 and m["tokens"]["n_evaluable"] == 12 and m["tokens"]["n_not_evaluable"] == 0
    assert Path(m["out_dir"]).name == "navtrain_val" + EV.tokens_tag(sub) and EV.tokens_tag(sub).startswith("__tok_")
    assert m["eval_spec"]["tokens"]["sha16"] == EV.tokens_sha16(sub) and m["eval_spec"]["n_eval"] == 12
    assert sorted(pd.read_parquet(Path(m["out_dir"]) / "per_token.parquet").token) == sorted(sub)
    assert json.loads((Path(full["out_dir"]) / "metrics.json").read_text())["n_eval"] == 29     # not overwritten
    # parquet / json token lists (e.g. another arm's per_token.parquet)
    pq = tmp_path / "pt.parquet"
    pd.DataFrame({"token": sub}).to_parquet(pq)
    js = tmp_path / "t.json"
    js.write_text(json.dumps({"tokens": sub}))
    for spec in (pq, js):
        assert EV.run_eval(root, run, "navtrain_val", n_boot=10, lead_root=lead, tokens=str(spec),
                           out=tmp_path / f"out_{spec.suffix[1:]}")["n_eval"] == 12
    # a token not in the split, or a listed token that is not evaluable here, is refused (unless allowed)
    bad = tmp_path / "bad.txt"
    bad.write_text("\n".join(sub + ["nope"]))
    with pytest.raises(SystemExit, match="not in navtrain_val"):
        EV.run_eval(root, run, "navtrain_val", n_boot=10, lead_root=lead, tokens=str(bad))
    ne = tmp_path / "ne.txt"
    ne.write_text("\n".join([toks[0]] + sub))                                       # token 0: packed not ok
    with pytest.raises(SystemExit, match="not evaluable"):
        EV.run_eval(root, run, "navtrain_val", n_boot=10, lead_root=lead, tokens=str(ne))
    ma = EV.run_eval(root, run, "navtrain_val", n_boot=10, lead_root=lead, tokens=str(ne), tokens_allow_missing=True)
    assert ma["n_eval"] == 12 and ma["tokens"]["n_not_evaluable"] == 1
    # CLI pass-through
    assert EV.main(["--root", str(root), "--run", run, "--split", "navtrain_val", "--bootstrap", "5",
                    "--tokens", str(f), "--out", str(tmp_path / "cli")]) == 0
    assert json.loads((tmp_path / "cli" / "metrics.json").read_text())["n_eval"] == 12


# ----------------------------------------------------------------------------------------------- eval_e2e2 per arm
def _write_hydra(P, layout: VariantLayout, string_form: bool = False, enabled: bool = True):
    P.hydra_cfg.parent.mkdir(parents=True, exist_ok=True)
    sp = "[" + ", ".join(map(str, layout.speeds)) + "]"
    la = "[" + ", ".join(map(str, layout.lats)) + "]"
    if string_form:
        sp, la = f"'{sp}'", f"'{la}'"
    P.hydra_cfg.write_text(f"agent:\n  config:\n    ck_e2e2:\n      enabled: {str(enabled).lower()}\n      variants:\n"
                           f"        speeds: {sp}\n        lats: {la}\n        combine: {layout.combine}\n"
                           + "".join(f"        {k}: {v}\n" for k, v in layout.cross_kw().items()))


@pytest.mark.parametrize("arm", ARMS)
def test_eval_e2e2_writer_meta_and_check_pool(tmp_path, arm):
    from tools.ck.e2e2 import eval_ck2 as EV
    from tools.ck.e2e2 import eval_e2e2 as E2
    L = arm_layout(arm)
    G = L.g()
    P = E2.Paths(tmp_path / f"run_{arm}", "navtrain_val")
    _write_hydra(P, L, string_form=(arm == "lat4"))
    assert E2.run_layout(P) == L
    tdf = pd.DataFrame({"token": ["a" * 16, "b" * 16], "log": ["L0", "L1"], "city": ["x", "x"]})
    W = E2.EvalRootWriter2(P, tdf, P.ck_run)                                          # layout from the run config
    assert W.layout == L and W.A["cand96"].shape == (2, G, 8, 3) and W.I["score_logit"].shape == (2, G, 5)
    assert W.I["w_lat"].shape == (2, G, 6) and W.A["cand_idx"].shape == (2, 16)
    rng = np.random.default_rng(0)
    pred = {"ck2_cand96": rng.normal(size=(2, G, 8, 3)), "ck2_valid96": np.ones((2, G), bool),
            "ck2_cand_idx": np.zeros((2, 16)), "ck2_v2_final": np.zeros((2, 16)), "ck2_v2_im": np.zeros((2, 16)),
            "ck2_v2_sim": np.zeros((2, 16, 5)), "ck2_score_logit": rng.normal(size=(2, G, 5)),
            "ck2_w_lat": np.zeros((2, G, 6)), "ck2_e_lat": np.zeros((2, G, 6)), "ck2_lat_traj": np.zeros((2, G, 8, 3))}
    assert W.write(np.arange(2), pred, np.zeros((2, 8)), np.zeros((2, 8, 3)), np.zeros((2, 8, 3))).all()
    bad = dict(pred, ck2_cand96=rng.normal(size=(2, 96, 8, 3)))
    with pytest.raises(SystemExit, match="pool columns"):
        W.write(np.arange(2), bad, np.zeros((2, 8)), np.zeros((2, 8, 3)), np.zeros((2, 8, 3)))
    W.finalize_meta({"ckpt": "/x", "ckpt_sha16": "a" * 16, "limit": 0, "n_done": 2})
    pm = json.loads((P.packed / "meta.json").read_text())
    im = json.loads((P.infer / "meta.json").read_text())
    assert pm["variant_layout"] == im["variant_layout"] == L.meta() and pm["k"] == im["k"] == G
    assert pm["layout"] == E2.pool_text(L) == f"c = k*{L.nv} + v (v2 top-16 x {', '.join(L.names)})"
    assert EV.layout_of_root(P.root, P.name, "navtrain_val") == L
    # the same eval root opened with another layout: same width (80) -> the recorded layout refuses, else the shapes
    other = arm_layout("lat4" if arm != "lat4" else "v4") if L.g() == 80 else BASE_LAYOUT
    with pytest.raises(SystemExit, match="variant layout|pool columns"):
        E2.EvalRootWriter2(P, tdf, P.ck_run, layout=other)
    # check_pool: identity columns at ::nv
    g = torch.Generator().manual_seed(1)
    anchors = torch.randn(256, 8, 3, generator=g)
    final = torch.randn(3, 256, generator=g)
    top = final.topk(16, dim=-1).indices
    off = torch.randn(3, 1, 8, 3, generator=g)
    cand = torch.randn(3, G, 8, 3, generator=g)
    cand[:, ::L.nv] = (anchors.reshape(1, -1, 8, 3) + off).gather(1, top[:, :, None, None].expand(-1, -1, 8, 3))
    pr = {"plan_final_rewards": final, "trajectory_anchors": anchors, "trajectory_offset": off, "ck2_cand96": cand,
          "ck2_cand_idx": top, "trajectory": cand[:, 0].clone()}
    assert E2.check_pool(pr, layout=L) == 0.0
    with pytest.raises(ValueError, match="columns"):
        E2.check_pool(pr)                                                           # base layout on an arm pool


def test_eval_e2e2_summary_refuses_mixed_layouts_and_base_defaults(tmp_path):
    from tools.ck.e2e2 import eval_e2e2 as E2
    R = tmp_path / "ck2e2e_sum"
    ev = R / "eval" / "root" / "eval" / R.name
    row = {"key": "v2", "pdms": 0.5, "nc": 1, "dac": 1, "ep": 1, "ttc": 1, "comfort": 1, "frac_changed": 0,
           "frac_lat": 0}

    def put(split, layout):
        (ev / split).mkdir(parents=True, exist_ok=True)
        m = {"n_eval": 9, "n_total": 10, "n_logs": 2, "representative": "v2", "rows": [row],
             "model": {"ckpt": "/r/a", "ckpt_sha16": "a" * 16, "code_sha16": "c"}}
        if layout is not None:
            m["variant_layout"] = layout.meta()
        (ev / split / "metrics.json").write_text(json.dumps(m))
    put("navtrain_val", arm_layout("v4"))
    put("navtest", arm_layout("spd4"))
    with pytest.raises(SystemExit, match="variant layouts"):
        E2.stage_summary(R)
    put("navtest", arm_layout("v4"))
    out = E2.stage_summary(R)
    assert out["variant_layout"] == arm_layout("v4").meta() and out["splits"]["navtest"]["n_eval"] == 9
    assert "변형 배치" in (R / "eval" / "summary.md").read_text()
    put("navtrain_val", None)                                                        # old metrics: base
    put("navtest", None)
    out = E2.stage_summary(R)
    assert out["variant_layout"] == BASE_LAYOUT.meta() and "변형 배치" not in (R / "eval" / "summary.md").read_text()
    # hydra config: ck_e2e2 without variants / no config = base; the base writer keeps the historical meta fields
    P = E2.Paths(tmp_path / "r_none", "navtest")
    assert E2.run_layout(P) == BASE_LAYOUT
    tdf = pd.DataFrame({"token": ["a" * 16], "log": ["L0"], "city": ["x"]})
    W = E2.EvalRootWriter2(P, tdf, P.ck_run)
    assert W.layout == BASE_LAYOUT and W.A["cand96"].shape == (1, 96, 8, 3) and W.I["e_lat"].shape == (1, 96, 6)
    W.finalize_meta({"ckpt": "/x", "ckpt_sha16": "a" * 16, "hydra_sha16": None, "code_sha16": "c", "limit": 0,
                     "n_done": 0})
    pm = json.loads((P.packed / "meta.json").read_text())
    assert list(pm) == ["split", "n", "k", "n_ok", "source", "layout", "created", "ckpt", "ckpt_sha16", "hydra_sha16",
                        "code_sha16", "limit", "n_done", "variant_layout"]
    assert pm["k"] == 96 and pm["layout"] == "c = k*6 + v (v2 top-16 x id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5)"
    assert pm["variant_layout"] == BASE_LAYOUT.meta()
    assert E2.pool_text(BASE_LAYOUT) == "c = k*6 + v (v2 top-16 x id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5)"
    p = E2.get_parser().parse_args(["--run-dir", "x", "--split", "navtest", "--tokens", "f.txt",
                                    "--tokens-allow-missing"])
    assert p.tokens == "f.txt" and p.tokens_allow_missing


# ----------------------------------------------------------------------------------------------- eval_epdms per arm
def _ep_pool(d: Path, layout: VariantLayout, N: int = 25, seed: int = 0, write_meta: bool = True):
    from tools.ck.e2e2 import eval_epdms as EP
    rng = np.random.default_rng(seed)
    G = layout.g()
    d.mkdir(parents=True, exist_ok=True)
    if write_meta:
        EP.write_pool_layout(d, layout)
    toks = [f"p{seed}_{i:04d}" for i in range(N)]
    json.dump(toks, open(d / "tokens.json", "w"))
    np.save(d / "cand96.npy", rng.normal(size=(N, G, 8, 3)).astype(np.float32))
    np.save(d / "lat_traj.npy", rng.normal(size=(N, G, 8, 3)).astype(np.float32))
    np.save(d / "v2_final.npy", rng.normal(size=(N, 16)).astype(np.float32))
    np.save(d / "v2_im.npy", rng.dirichlet(np.ones(16), size=N).astype(np.float32))
    np.save(d / "v2_sim.npy", rng.random((N, 16, 5)).astype(np.float32))
    np.save(d / "score_logit.npy", rng.normal(0, 2, size=(N, G, 5)).astype(np.float32))
    v = rng.random((N, G)) > 0.3
    v[:, ::layout.nv] = True
    np.save(d / "valid96.npy", v)
    kin = {"hc96": np.round(rng.random((N, G))), "cc96": np.round(rng.random((N, G))),
           "ec_self96": np.round(rng.random((N, G))), "f_now": rng.normal(size=(N, G, 4, 36)).astype(np.float32),
           "f_prev": rng.normal(size=(N, G, 4, 36)).astype(np.float32), "start_s": list(np.arange(N) * 0.5),
           "log_name": ["log0"] * N, "is_orig": [True] * N}
    return toks, kin


@pytest.fixture
def epdms_cpu(monkeypatch):
    """eval_epdms selection on CPU without the NAVSIM v2.2 checkout (thresholds injected; same for every layout)."""
    import navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics as PCM
    from tools.ck.e2e2 import eval_epdms as EP
    for k, v in (("acceleration_threshold", 4.0), ("jerk_threshold", 4.13), ("yaw_rate_threshold", 0.95),
                 ("yaw_accel_threshold", 1.93)):
        monkeypatch.setattr(PCM, k, v, raising=False)
    monkeypatch.setattr(EP, "_ensure_navsim22", lambda: None)
    return EP


@pytest.mark.parametrize("arm", ARMS)
def test_eval_epdms_arm_pool_layout_and_selection(tmp_path, epdms_cpu, arm):
    EP = epdms_cpu
    L = arm_layout(arm)
    nv, G = L.nv, L.g()
    d = tmp_path / f"pool96_T_navtest_{arm}"
    toks, kin = _ep_pool(d, L)
    assert (d / EP.POOL_META).is_file() and EP.layout_of_pool(d) == L
    out = EP.select_epdms_trajectories(d, "navtest", kin, None)
    lat = np.load(d / "lat_traj.npy")
    valid = np.load(d / "valid96.npy")
    for i, t in enumerate(toks):                       # every pick is an allowed column of this pool
        hits = [c for c in range(G) if np.array_equal(lat[i, c], out[t])]
        assert hits and (valid[i, hits[0]] or hits[0] % nv == 0)
    mp = {(toks[2], toks[1]): [(toks[4], toks[3])]}
    assert len(EP.select_epdms_trajectories(d, "navhard", kin, mp)) == len(toks)
    # a strongly best planted variant column is picked (layout-free weights over the whole pool)
    c0 = nv + L.names.index(PLANT[arm])
    lg = np.load(d / "score_logit.npy")
    lg[:, c0] = 12.0
    np.save(d / "score_logit.npy", lg)
    vv = np.load(d / "valid96.npy")
    vv[:, c0] = True
    np.save(d / "valid96.npy", vv)
    kin2 = dict(kin, hc96=np.ones((len(toks), G)), cc96=np.ones((len(toks), G)), ec_self96=np.ones((len(toks), G)))
    out = EP.select_epdms_trajectories(d, "navtest", kin2, None, beta=1.0)
    assert all(np.array_equal(out[t], lat[i, c0]) for i, t in enumerate(toks))
    cand = np.load(d / "cand96.npy")
    for name in S2.SET_NAMES:                          # PDMS selector: set names resolve by meaning
        for lm in S2.LAT_MODES:
            o = EP.select_pdms_trajectories(d, pdms_set=name, pdms_lat_mode=lm)
            idx = S2.select96_pdms(lg.astype(np.float64), vv, name, layout=L)
            want, _ = S2.final_traj(cand, lat, idx, lm, layout=L)
            assert all(np.array_equal(o[t], want[i]) for i, t in enumerate(toks)), (name, lm)
        planted = all(np.array_equal(o[t], lat[i, c0] if lm != "off" else cand[i, c0]) for i, t in enumerate(toks))
        assert planted == (L.names.index(PLANT[arm]) in L.sets[name]) or lm == "on_except_latvar"
    # tie order of this layout, refusing another length
    assert EP.pick_1row(np.zeros(G), np.ones(G, bool), S2.tie_order(L)) == 0
    with pytest.raises(ValueError, match="tie order"):
        EP.pick_1row(np.zeros(G), np.ones(G, bool))


def test_eval_epdms_refuses_layout_mixing(tmp_path, epdms_cpu):
    EP = epdms_cpu
    v4, lat4, v8 = arm_layout("v4"), arm_layout("lat4"), arm_layout("v8")
    # an 80-column pool without pool_meta.json reads as base -> refused
    d = tmp_path / "nometa"
    _ep_pool(d, v4, write_meta=False)
    with pytest.raises(SystemExit, match="variant layout"):
        EP.layout_of_pool(d)
    # a pool dir of another arm (same shape) cannot be reused / written as this arm, nor seed it
    d2 = tmp_path / "v4pool"
    _ep_pool(d2, v4)
    EP.write_pool_layout(d2, v4)                                                     # idempotent
    with pytest.raises(SystemExit, match="variant layout"):
        EP.write_pool_layout(d2, lat4)
    with pytest.raises(SystemExit, match="variant layout"):
        EP.write_pool_layout(d, v4)                  # rows without pool_meta.json = base rows: not an arm pool
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "pool96_part0.pkl").write_bytes(b"x")
    with pytest.raises(SystemExit, match="variant layout"):
        EP.write_pool_layout(tmp_path / "shards", v8)
    EP.write_pool_layout(tmp_path / "fresh_dir", v8)                                  # new dir: recorded
    assert EP.layout_of_pool(tmp_path / "fresh_dir") == v8
    with pytest.raises(SystemExit, match="cannot be mixed"):
        EP.expect_layout("seed", EP.layout_of_pool(d2), lat4)
    # base pools never get the file; a base pool cannot seed / be reused as an arm
    d3 = tmp_path / "base"
    _, kin_b = _ep_pool(d3, BASE_LAYOUT)
    assert not (d3 / EP.POOL_META).exists() and EP.layout_of_pool(d3) == BASE_LAYOUT
    with pytest.raises(SystemExit, match="cannot be mixed"):
        EP.expect_layout("seed", EP.layout_of_pool(d3), v8)
    # a kinematic cache of another width is refused
    kd = tmp_path / "kin"
    kd.mkdir()
    np.savez(kd / "kin_navtest_T.npz", **{k: kin_b[k].astype(np.float32) for k in ("hc96", "cc96", "ec_self96")},
             lk_traj96=kin_b["cc96"], f_now=kin_b["f_now"], f_prev=kin_b["f_prev"])
    (kd / "kin_navtest_T_meta.json").write_text(json.dumps({k: kin_b[k] for k in ("start_s", "log_name", "is_orig")}))
    assert EP.get_or_build_kin_cache(d3, kd, "T", "navtest", "", 1)["hc96"].shape[1] == 96
    with pytest.raises(SystemExit, match="kinematic cache"):
        EP.get_or_build_kin_cache(d2, kd, "T", "navtest", "", 1)
    # layout of a training config: enabled ck_e2e2 -> its variants, disabled / absent -> base
    for lay, en in ((v8, True), (lat4, True), (v8, False)):
        P = type("P", (), {"hydra_cfg": tmp_path / f"h_{lay.nv}_{en}" / "config.yaml"})
        _write_hydra(P, lay, enabled=en)
        assert EP.layout_of_hydra(P.hydra_cfg) == (lay if en else BASE_LAYOUT)
    # summary refuses results of different layouts
    od = tmp_path / "sum"
    (od / "results").mkdir(parents=True)
    m = {k: 90.0 for k in ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
                           "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
                           "lane_keeping", "history_comfort", "two_frame_extended_comfort", "score")}
    (od / "results" / "epdms_navtest_T.json").write_text(json.dumps({"after": m, "before": m,
                                                                     "variant_layout": v4.meta()}))
    (od / "results" / "epdms_navhard_T.json").write_text(json.dumps({
        "after": {"combined": m, "stage1": m, "stage2": m}, "before": {"combined": m}, "variant_layout": lat4.meta()}))
    with pytest.raises(SystemExit, match="different variant layouts"):
        EP.stage_summary(od, "T")
    (od / "results" / "epdms_navhard_T.json").unlink()
    EP.stage_summary(od, "T")
    assert json.loads((od / "summary_T.json").read_text())["variant_layout"] == v4.meta()
    assert "Variant layout: nv 5" in (od / "summary_T.md").read_text()
    a = EP.get_parser().parse_args(["--split", "navtest", "--pdms-set", "lat"])
    assert a.pdms_set == "lat"
    with pytest.raises(SystemExit):
        EP.get_parser().parse_args(["--split", "navtest", "--pdms-set", "lateral"])


def test_eval_epdms_recorded_arm_refuses_the_base_fallback_config(tmp_path, epdms_cpu):
    """a run recorded with a non-base VARIANTS arm (launch/settings.json) but without its own training config: the
    foreign / synthesized base config is refused (it would build a base agent and score the arm on a base pool), and
    stage_eval compares the pool with the recorded arm; base records (no key) and runs without a record are unchanged."""
    EP = epdms_cpu

    def run_dir(name, settings=None, own=None):
        d = tmp_path / name
        (d / "launch").mkdir(parents=True)
        if settings is not None:
            (d / "launch/settings.json").write_text(json.dumps({"run": name, "settings": settings}))
        if own is not None:
            P = type("P", (), {"hydra_cfg": d / "train/code/hydra/config.yaml"})
            _write_hydra(P, own)
        return d
    r8 = run_dir("var-v8_x", {"SEED": "0", "VARIANTS": "v8"})
    assert EP.recorded_arm(r8) == "v8" and EP.recorded_layout(r8) == arm_layout("v8")
    with pytest.raises(SystemExit, match="VARIANTS=v8 .*--hydra-cfg"):
        EP.resolve_hydra_cfg(r8, r8 / "train/x.ckpt")
    assert not (r8 / "eval_epdms").exists()                                 # nothing synthesized
    # its own config (or --hydra-cfg) is used as before
    r8own = run_dir("var-v8_own", {"VARIANTS": "v8"}, own=arm_layout("v8"))
    assert EP.resolve_hydra_cfg(r8own, r8own / "x.ckpt") == (r8own / "train/code/hydra/config.yaml").resolve()
    hc = tmp_path / "explicit/config.yaml"
    _write_hydra(type("P", (), {"hydra_cfg": hc}), arm_layout("v8"))
    assert EP.resolve_hydra_cfg(r8, r8 / "x.ckpt", str(hc)) == hc.resolve()
    # base: no key / no settings.json / no launch dir -> no refusal (the old fallback path)
    for d in (run_dir("main_x", {"SEED": "0"}), run_dir("old_run"), tmp_path / "bare"):
        EP.refuse_base_fallback(d, "x")
        assert EP.recorded_arm(d) in ("base", None)
    assert EP.recorded_layout(tmp_path / "main_x") == BASE_LAYOUT and EP.recorded_layout(tmp_path / "bare") is None
    with pytest.raises(SystemExit, match="unknown variant arm"):
        EP.recorded_layout(run_dir("odd", {"VARIANTS": "v9"}))
    # stage_eval without an own config: the pool must have the recorded arm's layout
    for arm, pool_lay in (("lat4", arm_layout("v4")), ("v4", BASE_LAYOUT), ("base", arm_layout("v8"))):
        d = run_dir(f"run_{arm}", {} if arm == "base" else {"VARIANTS": arm})
        _ep_pool(d / "eval_epdms" / "pool96_T_navtest", pool_lay)
        with pytest.raises(SystemExit, match="VARIANTS arm of .*cannot be mixed"):
            EP.stage_eval("navtest", d, d / "eval_epdms", "T", Path("x.ckpt"), 0.8, 2.0, 1.0, 1, "", "")


# ----------------------------------------------------------------------------------------------- teacher tools (base)
@pytest.mark.parametrize("arm", ARMS)
def test_teacher_navtest_tools_refuse_non_base(tmp_path, arm):
    from tools.ck.e2e2 import check_teacher_navtest_headline as H1
    from tools.ck.e2e2 import check_teacher_navtest_headline_ep as H2
    from tools.ck.e2e2 import eval_teacher_navtest_gtfree as GF
    from tools.ck.e2e2 import eval_teacher_navtest_r34 as R34
    from tools.ck.e2e2 import summarize_teacher_navtest as SUM
    names = list(arm_layout(arm).names)
    for M in (R34, GF, SUM):
        M.require_base()                                   # the tools' own tables are base
        M.require_base(list(BASE_LAYOUT.names), "base dir")
        with pytest.raises(SystemExit, match="base"):
            M.require_base(names, f"{arm} dir")
    assert H1.BASE_NAMES == H2.BASE_NAMES == BASE_LAYOUT.names
    d = tmp_path / "eval_root"
    (d / "r34_variants").mkdir(parents=True)
    (d / "ck2T" / "gtfree").mkdir(parents=True)
    (d / "r34_variants" / "meta.json").write_text(json.dumps({"names": list(BASE_LAYOUT.names)}))
    (d / "ck2T" / "gtfree" / "meta.json").write_text(json.dumps({"variants": {"names": list(BASE_LAYOUT.names)}}))
    for H in (H1, H2):
        H.require_base_layout(d)
    (d / "ck2T" / "gtfree" / "meta.json").write_text(json.dumps({"variants": {"names": names}}))
    for H in (H1, H2):
        with pytest.raises(SystemExit, match="base variant layout"):
            H.require_base_layout(d)


@pytest.mark.parametrize("arm", ("v4", "lat4"))
def test_eval_epdms_import_and_reuse_follow_the_run_layout(tmp_path, epdms_cpu, arm):
    """navtest pool imported from an eval_e2e2 dump (packed meta.json 'variant_layout') and stage_dump's reuse check:
    the run's training config decides, a same-shape arm of another layout is refused."""
    EP = epdms_cpu
    L = arm_layout(arm)
    other = arm_layout("lat4" if arm == "v4" else "v4")
    G = L.g()
    run = tmp_path / f"run_{arm}"
    pk, inf = run / "eval/root/packed/navtest", run / f"eval/root/infer/{run.name}/navtest"
    for d in (pk, inf):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pd.DataFrame({"token": ["t0", "t1"], "log": ["L", "L"], "city": ["x", "x"]}).to_parquet(pk / "tokens.parquet")
    for k, shp in (("cand96", (2, G, 8, 3)), ("v2_final", (2, 16)), ("v2_im", (2, 16)), ("v2_sim", (2, 16, 5))):
        np.save(pk / f"{k}.npy", rng.normal(size=shp).astype(np.float32))
    np.save(pk / "valid96.npy", np.ones((2, G), bool))
    np.save(inf / "lat_traj.npy", rng.normal(size=(2, G, 8, 3)).astype(np.float32))
    np.save(inf / "score_logit.npy", rng.normal(size=(2, G, 5)).astype(np.float32))
    np.save(inf / "done.npy", np.ones(2, bool))
    (pk / "meta.json").write_text(json.dumps({"variant_layout": L.meta()}))
    with pytest.raises(SystemExit, match="variant layout"):
        EP.try_import_eval_e2e2_dump(run, tmp_path / "pool_wrong", "navtest", other)
    pool = tmp_path / "out" / "pool96_T_navtest"
    assert EP.try_import_eval_e2e2_dump(run, pool, "navtest", L)
    assert EP.layout_of_pool(pool) == L and json.loads((pool / EP.POOL_META).read_text())["k"] == G
    assert len(EP.select_pdms_trajectories(pool, pdms_set="lat")) == 2
    # stage_dump reuses the pool only for a training config of the same layout
    P = type("P", (), {"hydra_cfg": run / "train/code/hydra/config.yaml"})
    _write_hydra(P, L)
    assert EP.stage_dump("navtest", run, tmp_path / "out", "T", Path("x.ckpt"), P.hydra_cfg, [], 1, 0) == pool
    _write_hydra(P, other)
    with pytest.raises(SystemExit, match="variant layout"):
        EP.stage_dump("navtest", run, tmp_path / "out", "T", Path("x.ckpt"), P.hydra_cfg, [], 1, 0)
    # a base dump (no recorded layout) is imported as base, without pool_meta.json
    (pk / "meta.json").write_text(json.dumps({"ckpt_sha16": "a"}))
    with pytest.raises(SystemExit, match="16 x 6"):
        EP.try_import_eval_e2e2_dump(run, tmp_path / "pool_b", "navtest", None)
