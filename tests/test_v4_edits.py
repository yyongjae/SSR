"""refiner v4_3 M5 computed edits (navsim/agents/para_ssr/ck/v4/edits.py, CKNet.computed_edits) on KNOWN predicted
fields (the range readout's scene replaced): rule family = the CK2 variants, the longitudinal edit is the minimal one
reaching arrival >= t_out_block + edit_dt and grows monotonically with the window / edit_dt, no edit inside the tube,
at v0 < 2 m/s, for short drafts or an empty tube, lateral edits on EITHER side (the smaller shift; a centred box
outside on the left or the right, a source offset toward the parent's larger-margin side) until the corner margin
reaches edit_dlat, edit_kind, the correct_topk keys (u = the 12 corrector controls) and scores, the parent-only
shortlist (one column per distinct parent, filler slots), the zero-init corrector residual, and the CPU runtime for
T = 16 x 96.  CPU, ~2 min.

  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider -s tests/test_v4_edits.py
"""
import math
import time

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.ck import variants as VAR
from navsim.agents.para_ssr.ck.cands2 import online_variants
from navsim.agents.para_ssr.ck.model import CKNet
from navsim.agents.para_ssr.ck.v4 import V4Cfg
from navsim.agents.para_ssr.ck.v4 import edits as ED
from navsim.agents.para_ssr.ck.v4 import lqr as LQR
from navsim.agents.para_ssr.ck.v4 import range_targets as RT
from navsim.agents.para_ssr.ck.v4.range_readout import grid_coords, read_cells
from navsim.agents.para_ssr.refiner.corridor import corridor_geometry, lateral_offsets
from navsim.agents.para_ssr.refiner.e2e import ego_inputs
from navsim.agents.para_ssr.refiner.sdf import ego_corners

M = 20
TB = torch.arange(M) * 0.25
KEYS_CT = ("src_idx", "u", "edit_traj", "edit_score_logit", "edit_w_lat", "edit_haz")      # correct_topk's
N_U = 12                                                                 # corrector controls (correct_topk's u)


def _draft(v, kap=0.0):
    t = torch.arange(1, 9, dtype=torch.float64) * 0.5
    s = v * t
    if abs(kap) < 1e-9:
        d = torch.stack([s, 0 * s, 0 * s], -1)
    else:
        d = torch.stack([torch.sin(kap * s) / kap, (1 - torch.cos(kap * s)) / kap, kap * s], -1)
    return d.float()


def _status(v0):
    v0 = torch.as_tensor(v0, dtype=torch.float32).reshape(-1)
    s = torch.zeros(v0.numel(), 8)
    s[:, 1] = 1
    s[:, 4] = v0
    return s


def _net(**kw):
    torch.manual_seed(0)
    return CKNet("S", seed=0, v4=V4Cfg.range_arm(edits="computed", **kw)).eval()


def _fields(par, block=None, U=None, curb=None):
    """known parent fields: block = (s_lo, s_hi, t_lo, t_hi) blocked B (and U unless U given) for |d| <= 2.4 m; curb =
    (left, right) drivable half widths (m, or callables of the station arc lengths s [48]) -> D = min(left - d,
    d + right)."""
    geom = corridor_geometry(par[None])
    s, d = geom.s[0], lateral_offsets()
    B = torch.zeros(48, 17, M, dtype=torch.bool)
    if block is not None:
        s0, s1, t0, t1 = block
        B = (((s >= s0) & (s < s1))[:, None, None] & (d.abs() <= 2.4)[None, :, None]
             & ((TB >= t0) & (TB < t1))[None, None])
    Ub = B if U is None else U
    blk = torch.cat([torch.where(B, 8.0, -8.0), torch.where(Ub, 8.0, -8.0)], -1)[None]
    left, right = curb if curb is not None else (8.0, 8.0)
    half = lambda w: (w(s) if callable(w) else torch.full_like(s, float(w)))[:, None]      # [48, 1]
    D = torch.minimum(half(left) - d[None], d[None] + half(right))[None].contiguous()       # [1, 48, 17]
    return blk, D, geom


def _run(net, pool, parents, status, blk, D, sel_score=None, valid=None, ctx=False, k=None):
    net.v4_readout.scene = lambda xs, Hc: {"blk": blk.expand(xs.shape[0], *blk.shape[1:]).contiguous(),
                                           "margin": D.expand(xs.shape[0], 48, 17).contiguous()}
    ego = ego_inputs(status)
    with torch.no_grad():
        feat, mem = net.trunk.scene(torch.randn(pool.shape[0], 256, 50, 100), pool.shape[0])
        pref = net.scene_pref(mem, ego)
        po = net.score_set(feat, mem, ego, pool, parents, pref=pref, return_pctx=True, return_ctx=ctx)
        ce = net.computed_edits(feat, mem, ego, pool, parents, po, pref=pref, sel_score=sel_score, valid=valid, k=k)
    return ce, po, (feat, mem, ego, pref)


def _arrivals(traj, v0, geom, cut):
    """executed (lqr) arrival of traj [n, 8, 3] at station coordinate cut: (rear axle, TTC rule min_delta)."""
    n = traj.shape[0]
    st = LQR.simulate(traj, torch.tensor([[float(v0), 0.0]]).expand(n, 2), dtype=torch.float32)
    fr = RT.gather_frame({k: v.float() for k, v in RT.parent_frame(geom).items()}, torch.zeros(n, dtype=torch.long))
    return ED.arrival_times(st, fr, torch.full((n,), float(cut)), torch.arange(41, dtype=torch.float64) * 0.1)


def _min_corner(traj, v0, geom, D):
    n = traj.shape[0]
    st = LQR.simulate(traj, torch.tensor([[float(v0), 0.0]]).expand(n, 2), dtype=torch.float32)
    fr = RT.gather_frame({k: v.float() for k, v in RT.parent_frame(geom).items()}, torch.zeros(n, dtype=torch.long))
    gc = grid_coords(fr, ego_corners(st[..., :3]).reshape(n, -1, 2), "field")
    v = read_cells(D[..., None].contiguous(), torch.zeros(n, dtype=torch.long), gc["us"], gc["ud"], 0)
    return torch.where(gc["ok"], v, torch.full_like(v, math.inf)).min(-1).values


# ----------------------------------------------------------------------------------------------- rule family
def test_rule_controls_and_decode_match_ck2_variants():
    for a in (-1.0, -0.5, -2.5):
        z, w = ED.rule_controls(torch.tensor([a], dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
        assert np.allclose(z[0].numpy(), VAR.z_speed(a), atol=1e-12) and float(w.abs().max()) == 0.0
    for d in (-0.5, 0.5, 1.7):
        _, w = ED.rule_controls(torch.zeros(1, dtype=torch.float64), torch.tensor([d], dtype=torch.float64))
        assert np.allclose(w[0].numpy(), VAR.w_rule(d), atol=1e-9)
    tau = torch.stack([_draft(7.0, 0.01), _draft(4.0, -0.03)])[None]
    V = VAR.make_variants(tau, torch.tensor([7.0]), speeds=(-1.0, -0.5), lats=(-0.5, 0.5), combine="cross")
    tab = VAR.variant_table((-1.0, -0.5), (-0.5, 0.5), "cross")
    for vi, (a, d) in enumerate(tab):
        e = ED.decode_rule(tau[0], torch.full((2,), a), torch.full((2,), d), torch.full((2,), 7.0))
        assert float((e - V["traj"][0, :, vi]).abs().max()) < 1e-5, (a, d)
    assert torch.equal(ED.decode_rule(tau[0], torch.zeros(2), torch.zeros(2), torch.full((2,), 7.0)), tau[0])


# ----------------------------------------------------------------------------------------------- longitudinal
def test_lon_edit_minimal_and_monotone():
    v = 10.0
    par = _draft(v)
    pool = torch.stack([par, _draft(9.5), _draft(10.5)])[None]
    parents = pool                                                       # three parents (sources are parents only)
    net = _net(edit_k=3)
    res = {}
    for t_end in (2.25, 2.5, 2.75):                                      # = t_out_block (ends of 0.25 s bins)
        blk, D, geom = _fields(par, block=(20.0, 30.0, 0.0, t_end))
        ce, _, _ = _run(net, pool, parents, _status(v), blk, D)
        assert ce["edit_kind"].tolist() == [[1, 1, 1]] and bool(ce["edit_clear"].all())
        assert float(ce["d_lat"].abs().max()) == 0.0 and torch.equal(ce["u_rule"][..., 0], ce["a_lon"])
        assert ce["u"].shape == (1, 3, N_U) and float(ce["u"].abs().max()) == 0.0   # no corrector: identity residual
        assert torch.equal(ce["edit_base"], ce["edit_traj"])
        cut = math.floor(20.0 / float(RT.parent_frame(geom)["ds"][0])) * float(RT.parent_frame(geom)["ds"][0])
        j = int(torch.nonzero(ce["src_idx"][0] == 0)[0])                  # the parent itself
        a = float(ce["a_lon"][0, j])
        _, t_ttc = _arrivals(ce["edit_traj"][0, j:j + 1], v, geom, cut)
        assert float(t_ttc[0]) >= t_end + 0.2 - 1e-6                      # target reached (TTC rule, U window)
        weaker = ED.decode_rule(par[None], torch.tensor([a + 0.016]), torch.zeros(1), torch.tensor([v]))
        assert float(_arrivals(weaker, v, geom, cut)[1][0]) < t_end + 0.2  # minimal up to the bisection step
        res[t_end] = a
    assert res[2.25] > res[2.5] > res[2.75] >= ED.A_MIN                   # a later window end needs more braking
    # a larger edit_dt needs more braking too
    blk, D, _ = _fields(par, block=(20.0, 30.0, 0.0, 2.5))
    ce6, _, _ = _run(_net(edit_k=3, edit_dt=0.6), pool, parents, _status(v), blk, D)
    j = int(torch.nonzero(ce6["src_idx"][0] == 0)[0])
    assert float(ce6["a_lon"][0, j]) < res[2.5]


def test_no_edit_inside_tube_slow_ego_short_draft_and_empty_tube():
    v = 10.0
    par = _draft(v)
    pool = torch.stack([par, _draft(9.0), _draft(11.0)])[None]
    parents = pool                                                       # three parents (sources are parents only)
    net = _net(edit_k=3)

    def none(ce, P=pool):
        src = torch.gather(P, 1, ce["src_idx"][..., None, None].expand(1, 3, 8, 3))
        return ((ce["edit_kind"] == 0).all() and torch.equal(ce["edit_traj"], src) and bool(ce["src_valid"].all())
                and float(ce["u_rule"].abs().max()) == 0 and float(ce["u"].abs().max()) == 0)

    # free fields, wide road: inside the tube
    blk, D, _ = _fields(par)
    assert none(_run(net, pool, parents, _status(v), blk, D)[0])
    # blocked window but the ego is slow (v0 < 2 m/s; its own plans at 1.5 m/s)
    slow = torch.stack([_draft(1.5), _draft(1.2), _draft(1.8)])[None]
    blk, D, _ = _fields(slow[0, 0], block=(0.0, 30.0, 0.0, 5.0))
    assert none(_run(net, slow, slow, _status(1.5), blk, D)[0], slow)
    # short drafts (S_8 < 3 m) at v0 = 4 m/s
    short = torch.stack([_draft(0.6), _draft(0.5), _draft(0.7)])[None]
    blk, D, _ = _fields(short[0, 0], block=(0.0, 30.0, 0.0, 5.0))
    assert none(_run(net, short, short, _status(4.0), blk, D)[0], short)
    # empty tube: blocked from 6 m for the whole horizon at 10 m/s -- no deceleration in the family stops before it
    blk, D, _ = _fields(par, block=(6.0, 60.0, 0.0, 5.0))
    assert none(_run(net, pool, parents, _status(v), blk, D)[0])
    # blocked from 40 m for the whole horizon: stopping before it is feasible -> lon edit that never arrives
    blk, D, geom = _fields(par, block=(40.0, 60.0, 0.0, 5.0))
    ce, _, _ = _run(net, pool, parents, _status(v), blk, D)
    assert (ce["edit_kind"] == 1).all()
    j = int(torch.nonzero(ce["src_idx"][0] == 0)[0])
    t_nc, t_ttc = _arrivals(ce["edit_traj"][0, j:j + 1], v, geom, math.floor(40.0 / 1.0416666) * 1.0416666)
    assert math.isinf(float(t_ttc[0])) and math.isinf(float(t_nc[0]))


# ----------------------------------------------------------------------------------------------- lateral / kinds
def _narrow(w_out, s0=24.0, w_in=3.0):
    """a drivable half width that narrows to w_out (m) from station s0 on (w_in before)."""
    return lambda s: torch.where(s >= s0, torch.full_like(s, w_out), torch.full_like(s, w_in))


def test_lateral_edit_either_side_minimal_and_kinds():
    """the parent's centred box leaves the road on the RIGHT (right half width 0.8 m from s = 24 m: corner margin
    0.8 - 1.1485 < 0, so both grid margins equal f0 there) -> shift LEFT; the mirror (left curb) -> shift RIGHT by the
    same |D| (the margin rule's tie used to resolve every such case to the left: no edit for a left-curb violation)."""
    v = 8.0
    par = _draft(v)
    pool, parents = par[None, None], par[None, None]
    net = _net(edit_k=1)
    d_out = {}
    for side, curb in (("right", (4.0, _narrow(0.8))), ("left", (_narrow(0.8), 4.0))):
        blk, D, geom = _fields(par, curb=curb)
        ce, po, _ = _run(net, pool, parents, _status(v), blk, D)
        assert float(po["haz"]["slack_dac"][0, 0]) < 0                    # the parent is outside the predicted tube
        assert ce["edit_kind"].tolist() == [[2]] and ce["edit_need"].tolist() == [[2]] and bool(ce["edit_clear"].all())
        dl = float(ce["d_lat"][0, 0])
        assert float(ce["a_lon"][0, 0]) == 0.0 and torch.equal(ce["u_rule"][..., 1], ce["d_lat"])
        m_edit = float(_min_corner(ce["edit_traj"][0], v, geom, D)[0])
        less = dl - math.copysign(0.009, dl)                              # one bisection step (0.25 / 32 m) less
        m_less = float(_min_corner(ED.decode_rule(par[None], torch.zeros(1), torch.tensor([less]), torch.tensor([v])),
                                   v, geom, D)[0])
        assert m_edit >= 0.3 - 1e-6 and m_less < 0.3                      # minimal up to the bisection step
        d_out[side] = dl
    assert d_out["right"] > 0 > d_out["left"]                             # away from the violated curb
    assert abs(d_out["right"] + d_out["left"]) < 1e-3                     # mirror-symmetric
    # both: a blocked window ahead + the right curb -> lon first (a mild brake: the braked path still reaches the narrow
    # part), then the lateral search on the edited profile (kind 3).  (Braking can clear DAC alone when the braked path
    # no longer reaches the violation, or the tracker follows a lateral offset more slowly at the lower speed.)
    blk, D, _ = _fields(par, block=(25.0, 35.0, 0.0, 3.25), curb=(4.0, _narrow(0.8)))
    ce, _, _ = _run(net, pool, parents, _status(v), blk, D)
    assert ce["edit_kind"].tolist() == [[3]] and ce["edit_need"].tolist() == [[3]]
    assert float(ce["a_lon"][0, 0]) < 0 and float(ce["d_lat"][0, 0]) > 0


def test_lateral_search_ignores_the_parent_margin_side():
    """the source drifts LEFT of its parent's centreline (0.2 .. 2.2 m) on a road whose parent-centred margins are
    larger on the LEFT (half widths 2.6 / 2.0 m) and violates on the left: the side 'toward the larger margin' (left)
    has no solution, the either-side search shifts it back right (lateral_search on lateral_ok, the computed_edits
    path); and when both sides work, the smaller |D| wins."""
    v = 8.0
    par = _draft(v)
    src = par.clone()
    src[:, 1] += torch.linspace(0.2, 2.2, 8)
    _, D, geom = _fields(par, curb=(2.6, 2.0))
    fr = {k: x.float() for k, x in RT.parent_frame(geom).items()}
    ego = torch.tensor([[v, 0.0]])
    ok = ED.lateral_ok(src[None], torch.zeros(1), torch.tensor([v]), ego, fr, D, 0.3)
    rows = torch.zeros(1, dtype=torch.long)
    assert float(_min_corner(src[None], v, geom, D)[0]) < 0                # outside on the left as planned
    left = torch.as_tensor(ED.D_SCAN)[None]
    _, f_left = ED.scan_bisect(ok, rows, torch.zeros(1), left)
    assert not bool(f_left[0])                                             # nothing toward the larger parent margin
    d, found = ED.lateral_search(ok, rows)
    assert bool(found[0]) and float(d[0]) < 0
    tr = ED.decode_rule(src[None], torch.zeros(1), d, torch.tensor([v]))
    assert float(_min_corner(tr, v, geom, D)[0]) >= 0.3 - 1e-6
    # both sides work (a non-drivable island under the left corners at s in [18, 26) m, a 4 m road each side): the
    # smaller |D| wins -- right for a wide island (0.9 .. 1.6 m), left for a narrow one (0.9 .. 1.3 m)
    s, dd = geom.s[0], lateral_offsets()
    road = torch.minimum(4.0 - dd, dd + 4.0)[None].expand(48, 17)
    for hi, sign in ((1.6, -1.0), (1.3, 1.0)):
        isl = torch.maximum(0.9 - dd, dd - hi)[None].expand(48, 17)
        D3 = torch.where(((s >= 18.0) & (s < 26.0))[:, None], torch.minimum(road, isl), road)[None].contiguous()
        ok3 = ED.lateral_ok(par[None], torch.zeros(1), torch.tensor([v]), ego, fr, D3, 0.3)
        dp, fp = ED.scan_bisect(ok3, rows, torch.zeros(1), left)
        dn, fn = ED.scan_bisect(ok3, rows, torch.zeros(1), -left)
        d3, f3 = ED.lateral_search(ok3, rows)
        assert bool(fp[0]) and bool(fn[0]) and bool(f3[0])
        assert float(d3[0]) == (float(dn[0]) if sign < 0 else float(dp[0])) and math.copysign(1.0, float(d3[0])) == sign
        assert abs(float(d3[0])) == min(abs(float(dp[0])), abs(float(dn[0])))


# ----------------------------------------------------------------------------------------------- interface
def test_keys_scores_shortlist_and_validity():
    v = 9.0
    top = torch.stack([_draft(v + 0.2 * k, 0.004 * (k - 4)) for k in range(8)])[None]
    V = online_variants(top.double(), torch.tensor([v], dtype=torch.float64), None)
    pool, par = V["traj96"][:, :48].float(), top.repeat_interleave(6, 1)
    net = _net(edit_k=4)
    blk, D, _ = _fields(top[0, 0], block=(18.0, 28.0, 0.0, 2.5))
    sel = torch.randn(1, 48, generator=torch.Generator().manual_seed(1))
    valid = torch.ones(1, 48, dtype=torch.bool)
    valid[0, sel[0].argmax()] = False
    valid[0, 6 * int(sel[0, ::6].argmax())] = False                       # and the best-scored PARENT column
    ce, po, (feat, mem, ego, pref) = _run(net, pool, par, _status(v), blk, D, sel_score=sel, valid=valid)
    assert all(k in ce for k in KEYS_CT + ("edit_kind", "edit_need", "a_lon", "d_lat", "u_rule", "edit_base",
                                           "src_valid", "edit_clear"))
    assert ce["src_idx"].shape == (1, 4) and ce["edit_traj"].shape == (1, 4, 8, 3)
    assert ce["u"].shape == (1, 4, N_U) and ce["u_rule"].shape == (1, 4, 2)          # correct_topk's u layout
    assert ce["edit_score_logit"].shape == (1, 4, 5) and ce["edit_w_lat"].shape == (1, 4, 6)
    assert ce["edit_haz"]["slack_ttc"].shape == (1, 4)
    # sources = the valid PARENT columns (c % 6 == 0: traj96[:, ::6] are the parents) ranked by sel_score
    is_par = (torch.arange(48) % 6 == 0)[None]
    assert torch.equal(po["haz"]["is_self"], is_par)
    sv = torch.where(valid & is_par, sel, torch.full_like(sel, -1e9))
    assert torch.equal(ce["src_idx"], torch.topk(sv, 4, 1).indices) and bool(ce["src_valid"].all())
    assert bool((ce["src_idx"] % 6 == 0).all())
    # re-scored on the source's parent corridor: the same as an explicit score_set with the gathered parents
    g = lambda a: torch.gather(a, 1, ce["src_idx"].reshape(1, 4, *([1] * (a.dim() - 2))).expand(1, 4, *a.shape[2:]))
    with torch.no_grad():
        ref = net.score_set(feat, mem, ego, ce["edit_traj"], g(par), pref=pref)
    assert torch.allclose(ce["edit_score_logit"], ref["score_logit"], atol=1e-5)
    # never a gate: the pool output is untouched by the call
    assert po["score_logit"].shape == (1, 48, 5)
    # critic-only shortlist without sel_score
    ce2, po2, _ = _run(net, pool, par, _status(v), blk, D)
    from navsim.agents.para_ssr.ck.v4.corrector import critic_select_score
    cs = torch.where(is_par, critic_select_score(po2["score_logit"]), torch.full((1, 48), -1e9))
    assert torch.equal(ce2["src_idx"], torch.topk(cs, 4, 1).indices)


def test_shortlist_parents_only_dedupe_and_fillers():
    """spec M5: sources are the top-k PARENTS -- a variant column is never edited (even when it scores best), a
    parent repeated in the pool takes one slot, and with fewer than k parents the extra slots are fillers (src_valid
    False, edit_kind 0, the column bitwise)."""
    v = 10.0
    par = _draft(v)
    var = ED.decode_rule(par[None], torch.tensor([-1.0]), torch.zeros(1), torch.tensor([v]))[0]   # a-1.0 variant
    pool = torch.stack([var, par, par, _draft(9.0)])[None]                # variant, parent, the parent again, parent 2
    parents = torch.stack([par, par, par, _draft(9.0)])[None]
    blk, D, _ = _fields(par, block=(20.0, 30.0, 0.0, 2.5))
    sel = torch.tensor([[9.0, 1.0, 2.0, 0.5]])                            # the variant and the repeat score highest
    net = _net(edit_k=3)
    ce, po, _ = _run(net, pool, parents, _status(v), blk, D, sel_score=sel)
    assert po["haz"]["is_self"].tolist() == [[False, True, True, True]]
    assert ED.parent_columns(pool, po["haz"]["is_self"]).tolist() == [[False, True, False, True]]
    assert ce["src_idx"][0, :2].tolist() == [1, 3] and ce["src_valid"].tolist() == [[True, True, False]]
    assert ce["edit_kind"][0, :2].tolist() == [1, 1] and int(ce["edit_kind"][0, 2]) == 0
    f = int(ce["src_idx"][0, 2])                                          # the filler: an excluded column, bitwise
    assert f in (0, 2) and torch.equal(ce["edit_traj"][0, 2], pool[0, f]) and int(ce["edit_need"][0, 2]) == 0


def test_corrector_residual_is_zero_at_init():
    v = 10.0
    par = _draft(v)
    pool = torch.stack([par, _draft(9.5), _draft(10.5)])[None]
    parents = pool                                                       # three parents (sources are parents only)
    blk, D, _ = _fields(par, block=(20.0, 30.0, 0.0, 2.5))
    ce0, _, _ = _run(_net(edit_k=3), pool, parents, _status(v), blk, D)
    assert ce0["edit_kind"].tolist() == [[1, 1, 1]]
    netc = _net(edit_k=3, corrector=True)
    with pytest.raises(ValueError, match="return_ctx"):
        _run(netc, pool, parents, _status(v), blk, D)
    ce1, _, _ = _run(netc, pool, parents, _status(v), blk, D, ctx=True)
    assert float(ce1["mu"].abs().max()) == 0.0 and ce1["u"].shape == (1, 3, N_U) and ce1["logp"].shape == (1, 3)
    assert torch.equal(ce1["src_idx"], ce0["src_idx"]) and torch.equal(ce1["a_lon"], ce0["a_lon"])
    assert torch.equal(ce1["edit_base"], ce0["edit_traj"])                # u applies on the computed edit
    assert float((ce1["edit_traj"] - ce0["edit_traj"]).abs().max()) < 1e-4
    with pytest.raises(ValueError, match="edits"):
        CKNet("S", seed=0, v4=V4Cfg.range_arm()).computed_edits(None, None, None, pool, parents, {})


def test_cpu_runtime_T16x96():
    """score_set of a 16 x 96 pool (16 parents per token, 7.0 .. 8.5 m/s drafts at v0 = 8 m/s) + computed_edits of the
    critic top-8 parents, the worst case: every source blocked ahead (128 feasible lon searches, rounds) and the right
    curb 1.1 m from every path, narrower than the box half width, so braking cannot clear it: 128 full two-sided lat
    searches on the braked profiles, all infeasible (the rule shape starts at 0.2 S_8, the box violates before that:
    an empty lateral tube)."""
    torch.manual_seed(0)
    T, v = 16, 8.0
    top = torch.stack([torch.stack([_draft(v - 1.0 + 0.1 * k, 0.003 * (k - 8)) for k in range(16)]) for _ in range(T)])
    V = online_variants(top.double(), torch.full((T,), v, dtype=torch.float64), None)
    pool, par = V["traj96"].float(), top.repeat_interleave(6, 1)
    net = _net(edit_k=8)
    blk, D, _ = _fields(top[0, 0], block=(14.0, 26.0, 0.0, 2.25), curb=(4.0, 1.1))   # right curb inside the box
    t0 = time.time()
    ce, po, _ = _run(net, pool, par, _status([v] * T), blk, D)
    dt = time.time() - t0
    net.v4_readout.scene = lambda xs, Hc: {"blk": blk.expand(xs.shape[0], *blk.shape[1:]).contiguous(),
                                           "margin": D.expand(xs.shape[0], 48, 17).contiguous()}
    ego = ego_inputs(_status([v] * T))
    with torch.no_grad():
        t1 = time.time()
        feat, mem = net.trunk.scene(torch.randn(T, 256, 50, 100), T)
        po = net.score_set(feat, mem, ego, pool, par, pref=net.scene_pref(mem, ego), return_pctx=True)
        t_pool = time.time() - t1
    kinds = torch.bincount(ce["edit_kind"].reshape(-1), minlength=4).tolist()
    print(f"\nT=16 x 96: score_set {t_pool:.2f} s, score_set + computed_edits {dt:.2f} s (edits {dt - t_pool:.2f} s), "
          f"kinds none/lon/lat/both {kinds}, threads {torch.get_num_threads()}")
    assert ce["edit_traj"].shape == (T, 8, 8, 3) and bool((ce["edit_need"] == 3).all()) and kinds[1] == T * 8
    assert dt < 120.0
