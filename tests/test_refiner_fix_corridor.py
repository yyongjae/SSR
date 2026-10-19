"""refiner-fix B1 (corridor loop): corridor_geometry(version=2) and the version plumbing, CPU only.

  v1 bit-identical : corridor_geometry(version=1) / DraftPath.eval == the code of c1c66b7 (loaded from git into a
                     temporary package) on random, near-stop, curved and real val r34 drafts; every geometry field
  v2 geometry      : no heading change and zero curvature past S_look; v_end < 3 m/s rows extend straight; rows with
                     v_end >= 3 m/s are unchanged up to S_look
  plumbing         : CKNet(corridor_version=1) forward == the same net with the c1c66b7 corridor (bit-exact);
                     version 2 reaches the trunk and changes the outputs; load_ck reads config.json corridor_version
                     (absent = 1); invalid versions are refused

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m pytest -q tests/test_refiner_fix_corridor.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.ck import model as M  # noqa: E402
from navsim.agents.para_ssr.refiner import corridor as C  # noqa: E402
from navsim.agents.para_ssr.refiner.adapters import save_norm  # noqa: E402

BASE_REV = "c1c66b7"           # last commit before refiner-fix
VAL_CAND = Path("/home/external-user/ssd/yongjae_refiner/ck/packed/navtrain_val/cand.npy")
GEOM_FIELDS = ("points", "geo", "s", "S", "S_look", "v_end", "near_stop", "ext_clipped", "kappa_knots")
EVAL_FIELDS = ("xy", "heading", "normal", "kappa", "tangent", "g", "g_s", "kappa_s", "beyond", "beyond_ext")


def _old_refiner(tmp_path_factory):
    """The c1c66b7 adapters / geometry / corridor as a temporary package 'refiner_old' (None if git lacks the rev)."""
    d = tmp_path_factory.mktemp("refiner_old") / "refiner_old"
    d.mkdir()
    for name in ("adapters", "geometry", "corridor"):
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_REV}:navsim/agents/para_ssr/refiner/{name}.py"],
                             capture_output=True, text=True)
        if src.returncode != 0:
            return None
        (d / f"{name}.py").write_text(src.stdout)
    (d / "__init__.py").write_text("")
    pkg = types.ModuleType("refiner_old")
    pkg.__path__ = [str(d)]
    sys.modules["refiner_old"] = pkg
    mods = {}
    for name in ("adapters", "geometry", "corridor"):
        spec = importlib.util.spec_from_file_location(f"refiner_old.{name}", d / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"refiner_old.{name}"] = m
        spec.loader.exec_module(m)
        mods[name] = m
    return mods


@pytest.fixture(scope="module")
def old(tmp_path_factory):
    mods = _old_refiner(tmp_path_factory)
    if mods is None:
        pytest.skip(f"git revision {BASE_REV} not available")
    return mods


def _drafts(n=600, seed=0):
    """Mixed drafts: near-stop jitter, slow curved, medium / fast curved, straight (x fwd, y left, heading)."""
    g = torch.Generator().manual_seed(seed)
    t = 0.5 * torch.arange(1, 9, dtype=torch.float32)
    out = []
    for i in range(n):
        v = float(torch.empty(1).uniform_(0.0, 15.0, generator=g)) if i % 3 else float(torch.empty(1).uniform_(0.0, 3.0, generator=g))
        a = float(torch.empty(1).uniform_(-2.0, 1.0, generator=g))
        k = float(torch.empty(1).uniform_(-0.15, 0.15, generator=g))
        s = torch.cummax(torch.clamp(v * t + 0.5 * a * t * t, min=0.0), 0).values
        th = k * s
        if abs(k) < 1e-4:
            x, y = s, torch.zeros_like(s)
        else:
            x, y = torch.sin(th) / k, (1 - torch.cos(th)) / k
        p = torch.stack([x, y, th], -1) + torch.randn(8, 3, generator=g) * torch.tensor([0.05, 0.05, 0.02])
        out.append(p)
    return torch.stack(out)


def _val_drafts(n_tok=400, seed=0):
    if not VAL_CAND.is_file():
        return None
    cand = np.load(VAL_CAND, mmap_mode="r")
    rows = np.sort(np.random.default_rng(seed).choice(len(cand), n_tok, replace=False))
    return torch.from_numpy(np.asarray(cand[rows], np.float32)).reshape(-1, 8, 3)


def _assert_geom_equal(a, b):
    for f in GEOM_FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        assert torch.equal(x, y), f
    s = torch.linspace(0.0, 90.0, 61)[None].expand(a.path.B, -1)
    ea, eb = a.path.eval(s), b.path.eval(s)
    for f in EVAL_FIELDS:
        assert torch.equal(getattr(ea, f), getattr(eb, f)), f"eval.{f}"


# ----------------------------------------------------------------------------------------------- v1 == c1c66b7
def test_v1_bit_identical_to_base_random(old):
    tau = _drafts()
    _assert_geom_equal(C.corridor_geometry(tau), old["corridor"].corridor_geometry(tau))
    _assert_geom_equal(C.corridor_geometry(tau, version=1), old["corridor"].corridor_geometry(tau))


def test_v1_bit_identical_to_base_val(old):
    tau = _val_drafts()
    if tau is None:
        pytest.skip(f"{VAL_CAND} missing")
    _assert_geom_equal(C.corridor_geometry(tau, version=1), old["corridor"].corridor_geometry(tau))


def test_draftpath_extend_without_arc_len_is_base(old):
    from navsim.agents.para_ssr.refiner.geometry import DraftPath
    tau = _drafts(200, seed=3)
    for pol in ("const_curv", "chord", "straight"):
        p, q = DraftPath(tau).extend(12.0, pol, v=3.0), old["geometry"].DraftPath(tau).extend(12.0, pol, v=3.0)
        assert p.ext_arc is None
        s = torch.linspace(0.0, 80.0, 41)[None].expand(p.B, -1)
        ep, eq = p.eval(s), q.eval(s)
        for f in EVAL_FIELDS:
            assert torch.equal(getattr(ep, f), getattr(eq, f)), (pol, f)


# ----------------------------------------------------------------------------------------------- v2 geometry
def _turn(h):
    d = torch.diff(h, dim=1)
    return (d - 2 * math.pi * torch.round(d / (2 * math.pi))).abs().sum(1)


@pytest.mark.parametrize("src", ["random", "val"])
def test_v2_no_turn_past_s_look_and_slow_rows_straight(src):
    tau = _drafts() if src == "random" else _val_drafts()
    if tau is None:
        pytest.skip(f"{VAL_CAND} missing")
    g = C.corridor_geometry(tau, version=2)
    S8, S_look = g.S[:, -1], g.S_look
    # past S_look: straight (constant tangent, zero curvature)
    s = S_look[:, None] + torch.linspace(0.5, 60.0, 31)[None]
    e = g.path.eval(s)
    assert torch.allclose(e.tangent, e.tangent[:, :1].expand_as(e.tangent), atol=1e-6)
    assert (e.kappa == 0).all()
    th = torch.atan2(e.tangent[..., 1], e.tangent[..., 0])
    assert float(_turn(th).max()) < 1e-5
    # the 48 stations: no heading change beyond S_look
    es = g.path.eval(g.s)
    hs = torch.atan2(es.tangent[..., 1], es.tangent[..., 0])
    past = g.s > S_look[:, None]
    hd = torch.where(past[:, 1:] & past[:, :-1], torch.diff(hs, dim=1), torch.zeros_like(hs[:, 1:]))
    assert float(hd.abs().max()) < 1e-5
    # slow rows: the whole extension (s > S_8) is straight
    slow = g.v_end < C.V_STRAIGHT_EXT
    assert bool(slow.any())
    assert (g.path.kappa_ext[slow] == 0).all()
    s2 = S8[slow][:, None] + torch.linspace(0.5, 50.0, 26)[None]          # strictly beyond S_8
    from navsim.agents.para_ssr.refiner.geometry import DraftPath
    sub = DraftPath(tau[slow]).extend((S_look - S8)[slow], "straight")
    ev, eref = _eval_rows(g.path, slow, s2), sub.eval(s2)
    torch.testing.assert_close(ev.xy, eref.xy, rtol=0, atol=1e-4)
    assert (ev.kappa == 0).all()


def _eval_rows(path, mask, s_rows):
    """path.eval on the masked rows (other rows evaluated at S_end and discarded)."""
    full = path.S_end[:, None].expand(path.B, s_rows.shape[1]).clone()
    full[mask] = s_rows
    e = path.eval(full)
    return types.SimpleNamespace(**{f: getattr(e, f)[mask] for f in ("xy", "tangent", "kappa")})


def test_v2_unchanged_up_to_s_look_for_fast_rows():
    tau = _drafts()
    g1, g2 = C.corridor_geometry(tau, version=1), C.corridor_geometry(tau, version=2)
    fast = g1.v_end >= C.V_STRAIGHT_EXT
    assert bool(fast.any())
    inside = (g1.s <= g1.S_look[:, None]) & fast[:, None]
    assert torch.equal(g1.points[inside], g2.points[inside])
    # geo channels 1-5 depend only on s / the draft; channel 0 (in_grid) on the points
    assert torch.equal(g1.geo[:, 1:], g2.geo[:, 1:])
    assert torch.equal(g1.geo[:, 0][inside], g2.geo[:, 0][inside])
    for f in ("s", "S", "S_look", "v_end", "near_stop", "ext_clipped", "kappa_knots"):
        assert torch.equal(getattr(g1, f), getattr(g2, f)), f
    # what changes: only stations past S_look (fast rows) or past S_8 (slow rows)
    changed = (g1.points != g2.points).any(-1).any(-1)                     # [B, 48]
    slow = ~fast
    assert not bool((changed & fast[:, None] & (g1.s <= g1.S_look[:, None])).any())
    assert not bool((changed & slow[:, None] & (g1.s <= g1.S[:, -1:])).any())
    assert g1.version == 1 and g2.version == 2


def test_invalid_version_refused():
    with pytest.raises(ValueError):
        C.corridor_geometry(_drafts(4), version=3)
    with pytest.raises(ValueError):
        M.CKNet("T", seed=0, norm=_norm(), corridor_version=0)


# ----------------------------------------------------------------------------------------------- CKNet plumbing
def _norm(seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.3, 256).astype(np.float32), rng.uniform(0.5, 2.0, 256).astype(np.float32)


def _bev(T=2, seed=1):
    return torch.randn(T, 256, 50, 100, generator=torch.Generator().manual_seed(seed)).half()


def _status(T=2, v=2.0):
    sf = torch.zeros(T, 8)
    sf[:, 1] = 1.0
    sf[:, 4] = v
    return sf


def test_cknet_v1_forward_bit_identical_to_base_corridor(old, monkeypatch):
    T, K = 2, 16
    tau = _drafts(T * K, seed=7).reshape(T, K, 8, 3)
    net = M.CKNet("T", seed=0, norm=_norm()).eval()
    assert net.trunk.corridor_version == 1
    with torch.no_grad():
        o_new = net(_bev(T), tau, _status(T), extra=_drafts(T * K, seed=8).reshape(T, K, 8, 3), decode=False)
    old_cg = old["corridor"].corridor_geometry
    monkeypatch.setattr(M, "corridor_geometry", lambda flat, version=1: old_cg(flat))
    with torch.no_grad():
        o_old = net(_bev(T), tau, _status(T), extra=_drafts(T * K, seed=8).reshape(T, K, 8, 3), decode=False)
    for k in ("score_logit", "extra_score_logit", "w_lat", "z_lon"):
        assert torch.equal(o_new[k], o_old[k]), k


def test_cknet_v2_reaches_trunk_and_load_ck_reads_config(tmp_path):
    T, K = 2, 16
    tau = _drafts(T * K, seed=11).reshape(T, K, 8, 3)
    n1 = M.CKNet("T", seed=0, norm=_norm()).eval()
    n2 = M.CKNet("T", seed=0, norm=_norm(), corridor_version=2).eval()
    assert n2.trunk.corridor_version == 2 and n2.corridor_version == 2
    assert all(torch.equal(a, b) for a, b in zip(n1.state_dict().values(), n2.state_dict().values()))
    with torch.no_grad():
        o1, o2 = n1(_bev(T), tau, _status(T)), n2(_bev(T), tau, _status(T))
    assert not torch.equal(o1["score_logit"], o2["score_logit"])
    b = M.build_ck("T", seed=0, norm=_norm(), corridor_version=2)
    assert b.trunk.corridor_version == 2
    for ver, cfg in ((1, {"arm": "T", "seed": 0}), (2, {"arm": "T", "seed": 0, "corridor_version": 2})):
        run = tmp_path / f"run{ver}"
        run.mkdir()
        (run / "config.json").write_text(json.dumps(cfg))
        save_norm(run / "norm.npz", *_norm())
        M.save_ck(run / "ckpt_last.pt", n1, cfg)
        got, _ = M.load_ck(run, "last")
        assert got.trunk.corridor_version == ver
