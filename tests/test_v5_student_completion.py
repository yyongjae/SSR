"""refiner v5 student completion, 'rel' group (CPU; design SSR/report/97_refiner_v5_design.md, refiner_v5_design_final.md
§3.2 / §3.7 / §3.9): a1 the det head's futures as relation modes m1..6 (rel_modes 'cv+fut', learned CV prior pi0), a2 the
futures at inference / in the eval dump, b1 L_rel (per-object BCE against the 2 m Hungarian-matched GT track at t0, KD
set's first 8 parents + their siblings), b2 L_rel,cv, c1 top-3 features 'v2' (station, side, key), c2 Delta r probes
(a-0.5 / l-0.5 / l+0.5 of tau0) via the relation re-read primitive, D3 det-box gradient on matched rows only."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.ck.v4 import lqr as LQR  # noqa: E402
from navsim.agents.para_ssr.ck.v5 import refine_loss as RL  # noqa: E402
from navsim.agents.para_ssr.ck.v5.config import V5Cfg  # noqa: E402
from navsim.agents.para_ssr.ck.v5.relation import RelationReader, fut_interp_matrix  # noqa: E402
from navsim.agents.para_ssr.ck.v5.student_objects import N_FEAT_S, rows_keep_graph, student_object_rows  # noqa: E402
from navsim.agents.para_ssr.refiner import gt_future as GF  # noqa: E402


# ----------------------------------------------------------------------------------------------- fixtures
def _straight_states(v: float = 6.0, lat: float = 0.0, n: int = 1):
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    traj = torch.stack([v * t, torch.full_like(t, lat), torch.zeros_like(t)], -1)[None].expand(n, 8, 3)
    st = LQR.simulate(traj, torch.tensor([[v, 0.0]]).expand(n, 2), dtype=torch.float32)
    tt = torch.arange(LQR.NT, dtype=torch.float64) * LQR.DT
    return st[:, ::2], tt[::2]


def _code(pos, vel=None, Q=6):
    """det-frame code [1, Q, 10]: queries at det (x_right, y_fwd) pos [k, 2] (cars 4.5 x 1.9 heading along y), others
    far behind (out of window)."""
    code = torch.zeros(1, Q, 10)
    code[0, :, 1] = -40.0
    code[0, :, 2], code[0, :, 3], code[0, :, 5], code[0, :, 7] = math.log(1.9), math.log(4.5), math.log(1.6), 1.0
    for i, p in enumerate(pos):
        code[0, i, 0:2] = torch.tensor(p, dtype=torch.float32)
        if vel is not None:
            code[0, i, 8:10] = torch.tensor(vel[i], dtype=torch.float32)
    return code


def _cls(Q=6, k=6, cls=0):
    c = torch.full((1, Q, 7), -6.0)
    c[0, :k, cls] = 4.0
    return c


def _gt_tracks(boxes, agent=None, n_kf=GF.N_KF, first_k=None):
    """GT of one token: boxes = list of (x0, y0, vx, vy) N frame (car 4.5 x 1.9, heading of the velocity)."""
    A = len(boxes)
    k = torch.arange(GF.N_KF, dtype=torch.float32)
    kf = torch.zeros(1, A, GF.N_KF, 6)
    first = torch.zeros(1, A, 6)
    meta = torch.zeros(1, A, 5, dtype=torch.long)
    for a, (x0, y0, vx, vy) in enumerate(boxes):
        kf[0, a, :, GF.KF_X] = x0 + vx * k * GF.KF_DT
        kf[0, a, :, GF.KF_Y] = y0 + vy * k * GF.KF_DT
        kf[0, a, :, GF.KF_H] = math.atan2(vy, vx) if (vx or vy) else 0.0
        kf[0, a, :, GF.KF_VX], kf[0, a, :, GF.KF_VY] = vx, vy
        kf[0, a, :, GF.KF_PRESENT] = 1.0
        first[0, a, GF.FI_L], first[0, a, GF.FI_W] = 4.5, 1.9
        first[0, a, GF.FI_VX], first[0, a, GF.FI_VY] = vx, vy
        meta[0, a, GF.MT_AGENT] = 1 if agent is None else int(agent[a])
        meta[0, a, GF.MT_LAST] = GF.N_KF - 1
        if first_k is not None:
            meta[0, a, GF.MT_FIRST] = int(first_k[a])
    return {"kf": kf, "first": first, "meta": meta, "valid": torch.ones(1, A, dtype=torch.bool),
            "n_kf": torch.tensor([n_kf]), "sdf": torch.ones(1, 320, 256) * 3.0, "p_pdm": torch.tensor([30.0])}


def _rows(n=4, **kw):
    """student rows dict of 1 token with n slots (all invalid; fill per test)."""
    O = {"c0": torch.zeros(1, n, 2), "v0": torch.zeros(1, n, 2), "h0": torch.zeros(1, n), "L": torch.ones(1, n) * 4.5,
         "W": torch.ones(1, n) * 1.9, "sp0": torch.zeros(1, n), "u0": torch.zeros(1, n, 2), "acc": torch.zeros(1, n),
         "has": torch.zeros(1, n, dtype=torch.bool), "group": torch.zeros(1, n, dtype=torch.long),
         "pi_s": torch.zeros(1, n), "valid": torch.zeros(1, n, dtype=torch.bool), "feats": torch.zeros(1, n, N_FEAT_S)}
    O.update(kw)
    return O


def _set_obj(O, i, c0, v0):
    O["c0"][0, i] = torch.tensor(c0, dtype=torch.float32)
    O["v0"][0, i] = torch.tensor(v0, dtype=torch.float32)
    sp = math.hypot(*v0)
    O["sp0"][0, i] = sp
    O["u0"][0, i] = torch.tensor(v0, dtype=torch.float32) / max(sp, 1e-6)
    O["h0"][0, i] = math.atan2(v0[1], v0[0]) if sp > 0 else 0.0
    O["pi_s"][0, i] = float(torch.sigmoid(torch.tensor((0.1 - sp) / 0.05)))
    O["valid"][0, i] = True


def _with_fut(O, fut, pi=None):
    n = O["c0"].shape[1]
    M = fut.shape[2]
    O = dict(O, fut=fut, pi_fut=torch.full((1, n, M), 1.0 / M) if pi is None else pi)
    return O


def _cv_fut(O, M=6):
    """futures equal to the CV path: cumulative v0 * 0.5 (k + 1) at the 8 keyframes, every mode."""
    k = torch.arange(1, 9, dtype=torch.float32) * 0.5
    f = O["v0"][:, :, None, None, :] * k[None, None, None, :, None]                        # [1, n, 1, 8, 2]
    return f.expand(1, O["c0"].shape[1], M, 8, 2).clone()


# ----------------------------------------------------------------------------------------------- a1: futures rows
def test_student_rows_futures_frame_cumsum_and_softmax():
    code = _code([(2.0, 10.0)]).requires_grad_(True)
    traj = torch.zeros(1, 6, 6, 8, 2)
    traj[0, 0, :, :, 0], traj[0, 0, :, :, 1] = 0.5, 1.0                                    # det per-step (x_r, y_f)
    tcls = torch.randn(1, 6, 6)
    O = student_object_rows(code, _cls(k=1), torch.zeros(1, 6, 256), 96, traj=traj, traj_cls=tcls)
    assert O["fut"].shape == (1, 1, 6, 8, 2) and O["pi_fut"].shape == (1, 1, 6)
    assert torch.allclose(O["fut"][0, 0, 0, 7], torch.tensor([8.0, -4.0]))                # N = (y_f, -x_r) cumsum
    assert torch.allclose(O["fut"][0, 0, 0, 0], torch.tensor([1.0, -0.5]))
    assert torch.allclose(O["pi_fut"][0, 0], torch.softmax(tcls[0, 0], -1))
    assert not O["fut"].requires_grad and not O["pi_fut"].requires_grad and O["c0"].requires_grad
    # padding / invalid slots: fut 0, pi uniform (finite)
    O2 = student_object_rows(_code([(2.0, 10.0), (1.0, 12.0)]), _cls(k=2), torch.zeros(1, 6, 256), 4, traj=traj,
                             traj_cls=tcls, score_thr=0.3)
    O3 = student_object_rows(_code([(2.0, 10.0)]), _cls(k=1), torch.zeros(1, 6, 256), 4, traj=traj, traj_cls=tcls)
    assert O3["fut"].shape[1] == 1
    assert bool(torch.isfinite(O2["pi_fut"]).all())
    O4 = student_object_rows(torch.cat([_code([(2.0, 10.0)])] * 2), torch.cat([_cls(k=1), _cls(k=0)]),
                             torch.zeros(2, 6, 256), 4, traj=torch.cat([traj] * 2), traj_cls=torch.cat([tcls] * 2))
    assert not bool(O4["valid"][1].any())
    assert float(O4["fut"][1].abs().sum()) == 0.0 and torch.allclose(O4["pi_fut"][1], torch.full((1, 6), 1 / 6))
    # old call: the dict without the futures (the 'cv' rows exactly)
    O5 = student_object_rows(_code([(2.0, 10.0)]), _cls(k=1), torch.zeros(1, 6, 256), 96)
    assert "fut" not in O5 and "pi_fut" not in O5
    with pytest.raises(ValueError):
        student_object_rows(_code([(2.0, 10.0)]), _cls(k=1), torch.zeros(1, 6, 256), 96, traj=traj[:, :3],
                            traj_cls=tcls)
    with pytest.raises(ValueError):
        student_object_rows(_code([(2.0, 10.0)]), _cls(k=1), torch.zeros(1, 6, 256), 96, traj=traj)


def test_future_interpolation_matrix():
    P = torch.arange(9, dtype=torch.float32)[:, None] * torch.tensor([[1.0, 0.5]])
    P[8] = P[7] + torch.tensor([3.0, 1.0])                                                 # last segment (3, 1)
    tt = torch.tensor([0.0, 0.25, 0.5, 3.75, 4.0, 4.6])
    W = fut_interp_matrix(tt, 8)
    pos = W @ P
    assert torch.allclose(pos[0], P[0]) and torch.allclose(pos[1], P[1] / 2) and torch.allclose(pos[2], P[1])
    assert torch.allclose(pos[3], (P[7] + P[8]) / 2) and torch.allclose(pos[4], P[8])
    assert torch.allclose(pos[5], P[8] + 1.2 * (P[8] - P[7]), atol=1e-5)                    # held CV beyond 4 s


def _reader(**kw):
    torch.manual_seed(0)
    return RelationReader(V5Cfg(relation="student", rel_ca=False, **kw))


def test_cv_mode_unchanged_and_no_new_parameter():
    rr = _reader()
    assert not hasattr(rr, "cv_prior") and sorted(rr.state_dict()) == ["log_eps", "rel_mlp.0.bias", "rel_mlp.0.weight",
                                                                        "rel_mlp.2.bias", "rel_mlp.2.weight"]
    rf = _reader(rel_modes="cv+fut")
    assert "cv_prior" in rf.state_dict() and float(rf.cv_prior) == 0.0
    for k, v in rr.state_dict().items():                                                    # same init otherwise
        assert torch.equal(v, rf.state_dict()[k]), k


def test_futures_equal_to_cv_reproduce_cv_and_prior_limit():
    st, t = _straight_states(6.0)
    O = _rows(4)
    _set_obj(O, 0, (14.0, 1.0), (-2.0, 0.0))
    _set_obj(O, 1, (6.0, 6.0), (0.0, -3.0))
    _set_obj(O, 2, (20.0, -3.0), (0.05, 0.0))
    _set_obj(O, 3, (-3.0, 0.5), (4.0, 0.0))                                                 # behind, faster
    tok = torch.zeros(1, dtype=torch.long)
    o_cv = _reader()(O, st, t, tok)
    rf = _reader(rel_modes="cv+fut")
    Of = _with_fut(O, _cv_fut(O), torch.softmax(torch.randn(1, 4, 6), -1))
    for prior in (0.0, -2.0, 3.0):
        with torch.no_grad():
            rf.cv_prior.fill_(prior)
        o_f = rf(Of, st, t, tok)
        for k in ("lp", "H", "r_nc", "r_ttc", "ev", "top"):
            assert float((o_f[k] - o_cv[k]).abs().max()) < 2e-5, (prior, k)
    with torch.no_grad():
        rf.cv_prior.fill_(30.0)
    Oz = _with_fut(O, torch.zeros(1, 4, 6, 8, 2))                                          # held futures, pi0 -> 1
    o_z = rf(Oz, st, t, tok)
    assert float((o_z["H"] - o_cv["H"]).abs().max()) < 1e-6


def test_future_mode_that_crosses_the_path_raises_r():
    st, t = _straight_states(6.0)
    O = _rows(1)
    _set_obj(O, 0, (14.0, 8.0), (3.0, 0.0))                                                 # CV: parallel, 8 m left
    fut = torch.zeros(1, 1, 6, 8, 2)
    fut[0, 0, :, :, 0] = torch.arange(1, 9).float() * 1.5
    fut[0, 0, 0, :, 1] = -torch.arange(1, 9).float() * 1.0                                 # mode 0 cuts in
    pi = torch.zeros(1, 1, 6)
    pi[0, 0, 0] = 1.0
    tok = torch.zeros(1, dtype=torch.long)
    r_cv = _reader()(O, st, t, tok)["r_nc"]
    r_f = _reader(rel_modes="cv+fut")(_with_fut(O, fut, pi), st, t, tok)["r_nc"]
    assert float(r_cv[0, 0]) < 0.05 and float(r_f[0, 0]) > float(r_cv[0, 0]) + 0.2


def test_futures_refusals():
    with pytest.raises(ValueError, match="student"):
        V5Cfg(relation="teacher", rel_modes="cv+fut").validate()
    with pytest.raises(ValueError, match="PT-11"):
        V5Cfg(relation="student", rel_modes="cv+fut+ca").validate()
    st, t = _straight_states()
    O = _rows(2)
    _set_obj(O, 0, (10.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="futures"):
        _reader(rel_modes="cv+fut")(O, st, t, torch.zeros(1, dtype=torch.long))
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, _d_v5, student_rel_rows
    pred = {"all_bbox_preds": [_code([(0.0, 10.0)])], "all_cls_scores": [_cls(k=1)], "det_hidden": torch.zeros(1, 6, 256)}
    with pytest.raises(KeyError, match="expose_det_outputs"):
        student_rel_rows(pred, dict(_d_v5(), enabled=True, relation="student", rel_modes="cv+fut"))
    with pytest.raises(ValueError, match="PT-11"):
        CKE2E2Config.from_any({"v4": {"v5": {"enabled": True, "relation": "student", "rel_modes": "cv+fut+ca"}}})
    with pytest.raises(ValueError):
        CKE2E2Config.from_any({"v4": {"v5": {"enabled": True, "relation": "off", "rel_modes": "cv+fut"}}})


def test_futures_gradients():
    st, t = _straight_states(6.0)
    code = _code([(-1.0, 12.0), (3.0, 20.0)], vel=[(0.0, -1.0), (-0.5, 0.0)]).requires_grad_(True)
    traj = (torch.randn(1, 6, 6, 8, 2) * 0.3).requires_grad_(True)
    tcls = torch.randn(1, 6, 6).requires_grad_(True)
    O = student_object_rows(code, _cls(k=2), torch.zeros(1, 6, 256), 96, traj=traj, traj_cls=tcls)
    rf = _reader(rel_modes="cv+fut")
    out = rf(O, st, t, torch.zeros(1, dtype=torch.long))
    out["H"].sum().backward()
    assert traj.grad is None and tcls.grad is None
    assert float(code.grad[0, :2, 0:2].abs().sum()) > 0 and float(code.grad[0, :2, 8:10].abs().sum()) > 0
    assert float(code.grad[0, :, 2:8].abs().sum()) == 0.0
    assert rf.cv_prior.grad is not None and float(rf.cv_prior.grad.abs()) > 0
    assert float(rf.log_eps.grad.abs()) > 0


# ----------------------------------------------------------------------------------------------- b1 / b2: L_rel
def test_match_t0_hungarian_duplicates_ghosts_and_rules():
    # GT: car A at (10, 0), car B at (11.5, 0.5), static cone C at (20, 3), future-first car D at (30, 0)
    gt = _gt_tracks([(10.0, 0.0, 0.0, 0.0), (11.5, 0.5, 0.0, 0.0), (20.0, 3.0, 0.0, 0.0), (30.0, 0.0, 0.0, 0.0)],
                    agent=[1, 1, 0, 1], first_k=[0, 0, 0, 2])
    c0 = torch.tensor([[[10.9, 0.3], [10.2, 0.1], [10.6, 0.2], [20.2, 3.1], [30.0, 0.1], [50.0, 0.0], [20.0, 2.8]]])
    valid = torch.tensor([[True, True, True, True, True, True, False]])
    agent = torch.tensor([[True, True, True, True, True, True, True]])
    g = RL.match_t0(c0, valid, agent, gt)
    # rows 0..2 compete for A / B: one-to-one (row 1 -> A, row 0 -> B: total distance), row 2 -> duplicate
    assert sorted(int(x) for x in g[0, :3]) == [RL.G_DUP, 0, 1]
    assert int(g[0, 1]) == 0
    # agent row 0.22 m from a static track: no pair (agent-ness differs), but a real object lies within the gate ->
    # MASKED, not a ghost (the class-blind reliability target gives it y = 1; y = 0 here would contradict it)
    assert int(g[0, 3]) == RL.G_DUP
    assert int(g[0, 4]) == RL.G_GHOST        # future-first track is never matched (and is not eligible at all)
    assert int(g[0, 5]) == RL.G_GHOST and int(g[0, 6]) == RL.G_GHOST
    g2 = RL.match_t0(c0, valid, torch.tensor([[True] * 3 + [False] * 4]), gt)
    assert int(g2[0, 3]) == 2                                                               # static row <-> cone


def test_match_t0_class_mismatch_near_real_object_is_masked_not_ghost():
    """a vehicle-class box 0.36 m from a static GT barrier that no row took: the class-blind reliability target says
    y = 1 (a real object is there), so L_rel / L_edit must not train its r towards 0 -> G_DUP (masked); only a box with
    no eligible GT of any class within 2 m is a ghost (y = 0)."""
    from navsim.agents.para_ssr.ck.v5.student_objects import student_reliability_targets
    gt = _gt_tracks([(10.0, 0.0, 0.0, 0.0), (20.0, 3.0, 0.0, 0.0)], agent=[0, 1])        # static barrier + a car
    c0 = torch.tensor([[[10.3, 0.2], [20.2, 3.1], [40.0, 0.0]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    g = RL.match_t0(c0, valid, torch.ones(1, 3, dtype=torch.bool), gt)                    # all rows vehicle-class
    assert g[0].tolist() == [RL.G_DUP, 1, RL.G_GHOST]
    y_a, _ = student_reliability_targets({"c0": c0, "v0": torch.zeros(1, 3, 2), "valid": valid},
                                         *RL.gt_t0_objects(gt))
    assert y_a[0].tolist() == [1.0, 1.0, 0.0]
    # every row the reliability target calls real (y_a = 1) is matched or masked -- never a y = 0 ghost
    assert not bool(((g == RL.G_GHOST) & (y_a > 0.5)).any())
    st, t = _straight_states(5.0)
    y, known = RL.rel_object_targets(st[None], t, gt, g)
    mk = RL.rel_object_mask(valid, g, known, y)
    assert not bool(mk[0, :, 0].any()) and bool(mk[0, :, 2].all()) and float(y[0, :, 2].sum()) == 0.0
    # a future-first SINGLETON (annotated once, at keyframe 2): never a pair (match_targets rule), but its held pose is
    # observed at t0 for the reliability target -> masked; a non-singleton future-first track is unobserved at t0 for
    # both -> the row near it stays a ghost
    gt2 = _gt_tracks([(10.0, 0.0, 0.0, 0.0), (30.0, 0.0, 0.0, 0.0)], first_k=[2, 2])
    gt2["meta"][0, 0, GF.MT_SINGLE] = 1
    gt2["meta"][0, 0, GF.MT_LAST] = 2
    c2 = torch.tensor([[[10.2, 0.1], [30.1, 0.0]]])
    v2 = torch.ones(1, 2, dtype=torch.bool)
    g2 = RL.match_t0(c2, v2, torch.ones(1, 2, dtype=torch.bool), gt2)
    y2, _ = student_reliability_targets({"c0": c2, "v0": torch.zeros(1, 2, 2), "valid": v2}, *RL.gt_t0_objects(gt2))
    assert g2[0].tolist() == [RL.G_DUP, RL.G_GHOST] and y2[0].tolist() == [1.0, 0.0]


def test_rel_targets_and_consistency_with_the_state_block():
    st, t = _straight_states(5.0, n=2)
    stl, _ = _straight_states(5.0, lat=6.0)
    states = torch.cat([st[:1], stl], 0)[None]                                             # [1, 2, Q, 6]
    gt = _gt_tracks([(21.0, 0.0, 0.0, 0.0), (-8.0, 0.0, -3.0, 0.0)])                     # stopped ahead; behind, away
    g = torch.tensor([[0, 1, RL.G_GHOST]])
    y, known = RL.rel_object_targets(states, t, gt, g)
    assert y.shape == (1, 2, 3, 2) and bool(known.all())
    assert float(y[0, 0, 0, 0]) == 1.0 and float(y[0, 0, 0, 1]) == 1.0                    # straight candidate hits
    assert float(y[0, 1, 0, 0]) == 0.0                                                      # 6 m lateral: clear
    assert float(y[0, :, 1].sum()) == 0.0 and float(y[0, :, 2].sum()) == 0.0               # behind / ghost: 0
    # OR over the matched rows == gt_state_block's any-over-states target of the same tracks (window / fov off)
    gm = RL.matched_tracks(gt, torch.tensor([[0, 1]]))
    yb, kn = RL.gt_state_block(states, t, dict(gm, valid=torch.ones(1, 2, dtype=torch.bool)))
    yr, _ = RL.rel_object_targets(states, t, gt, torch.tensor([[0, 1]]))
    assert torch.equal(yr.amax(2), yb.amax(2))


def test_rel_censoring_mask_and_cv_targets():
    st, t = _straight_states(5.0)
    states = st[None]
    gt = _gt_tracks([(30.0, 6.0, 0.0, -2.0), (10.0, 0.0, 0.0, 0.0)], n_kf=7)              # 3 s annotated
    g = torch.tensor([[0, 1, RL.G_GHOST, RL.G_DUP]])
    y, known = RL.rel_object_targets(states, t, gt, g)
    assert not bool(known.any())
    m = RL.rel_object_mask(torch.ones(1, 4, dtype=torch.bool), g, known, y)
    assert bool(m[0, 0, 1].all()) and float(y[0, 0, 1, 0]) == 1.0                          # a seen hit: kept
    assert not bool(m[0, 0, 0].any()) or float(y[0, 0, 0].min()) == 1.0                     # censored y = 0: masked
    assert bool(m[0, 0, 2].all()) and not bool(m[0, 0, 3].any())                            # ghost kept, duplicate masked
    # CV propagation: a track that turns across the path after 2 s hits, its CV propagation (parallel) does not
    gt2 = _gt_tracks([(16.0, 6.0, 0.0, 0.0)])
    kk = torch.arange(GF.N_KF, dtype=torch.float32) * GF.KF_DT
    gt2["kf"][0, 0, :, GF.KF_Y] = torch.where(kk <= 2.0, torch.full_like(kk, 6.0), 6.0 - 4.0 * (kk - 2.0))
    gt2["kf"][0, 0, :, GF.KF_X] = 16.0 + 0.0 * kk
    y2, _ = RL.rel_object_targets(states, t, gt2, torch.tensor([[0]]))
    y2c, kc = RL.rel_object_targets(states, t, gt2, torch.tensor([[0]]), motion="cv")
    assert float(y2[0, 0, 0, 1]) == 1.0 and float(y2c[0, 0, 0].sum()) == 0.0 and bool(kc.all())
    gt3 = _gt_tracks([(30.0, 0.0, 2.0, 0.0), (12.0, 7.0, 0.0, -2.5)])                     # straight movers
    y3, _ = RL.rel_object_targets(states, t, gt3, torch.tensor([[0, 1]]))
    y3c, _ = RL.rel_object_targets(states, t, gt3, torch.tensor([[0, 1]]), motion="cv")
    assert torch.equal(y3, y3c)
    # default motion: object_sep unchanged
    s1, c1 = RL.object_sep(states, t, gt3, (0.0, 0.3))
    s2, c2 = RL.object_sep(states, t, gt3, (0.0, 0.3), motion="gt")
    assert torch.equal(s1, s2) and torch.equal(c1, c2)


def test_parent_sibling_cols_first_8_parents():
    K = 48
    pidx = torch.cat([torch.arange(16), torch.arange(16).repeat_interleave(2)])[None]     # 16 parents + 32 variants
    is_self = torch.zeros(1, K, dtype=torch.bool)
    is_self[0, :16] = True
    cols = RL.parent_sibling_cols(pidx, is_self, 8)
    assert int(cols.sum()) == 8 + 16 and bool(cols[0, :8].all()) and not bool(cols[0, 8:16].any())
    assert bool(cols[0, 16:32].all()) and not bool(cols[0, 32:].any())


# ----------------------------------------------------------------------------------------------- D3
def test_matched_only_det_gradient():
    st, t = _straight_states(6.0)
    code = _code([(0.0, 10.0), (-1.0, 16.0), (2.5, 24.0)], vel=[(0.0, -1.0)] * 3).requires_grad_(True)
    O = student_object_rows(code, _cls(k=3), torch.zeros(1, 6, 256), 96)
    gt = _gt_tracks([(10.2, 0.1, 0.0, 0.0), (24.0, -2.4, 0.0, 0.0)])                       # rows 0 and 2 real
    g = RL.match_t0(O["c0"], O["valid"], O["group"] < 2, gt)
    q = O["q_idx"][0]
    assert sorted(int(x) for x in g[0]) == [RL.G_GHOST, 0, 1]
    Om = rows_keep_graph(O, g >= 0)
    for k in ("c0", "v0", "sp0", "u0", "pi_s", "valid", "feats"):
        assert torch.equal(Om[k].detach(), O[k].detach()), k                                # values unchanged
    out = _reader()(Om, st, t, torch.zeros(1, dtype=torch.long))
    (out["H"].sum() + out["lp"].sum()).backward()
    gq = code.grad[0].abs().sum(-1)
    ghost_q = int(q[(g[0] == RL.G_GHOST).nonzero()[0, 0]])
    real_q = [int(q[i]) for i in range(3) if int(g[0, i]) >= 0]
    assert float(gq[ghost_q]) == 0.0 and all(float(gq[j]) > 0 for j in real_q)


# ----------------------------------------------------------------------------------------------- c1: top 'v2'
def test_top_feats_v2_station_side_key_and_idx():
    st, t = _straight_states(6.0)
    O = _rows(5)
    _set_obj(O, 0, (12.0, 1.4), (0.0, 0.0))                                                 # static, slightly left
    _set_obj(O, 1, (18.0, -1.4), (0.0, 0.0))                                                # static, slightly right
    O["feats"][0, :, 0] = 1.0
    tok = torch.zeros(1, dtype=torch.long)
    o1 = _reader()(O, st, t, tok)
    o2 = _reader(top_feats="v2", corrector=True)(O, st, t, tok)
    assert o1["top"].shape == (1, 21) and o2["top"].shape == (1, 30) and "top_idx" not in o1
    top = o2["top"].reshape(1, 3, 10)
    assert torch.equal(top[..., :7], o1["top"].reshape(1, 3, 7))
    ti = o2["top_idx"][0]
    assert sorted(int(x) for x in ti[:2]) == [0, 1] and int(ti[2]) == -1                  # invalid slot (N >= 3): -1
    assert float(top[0, 2].abs().sum()) == 0.0
    for j in range(2):
        i = int(ti[j])
        t_star = float(top[0, j, 6]) * 4.0
        s_star = float(top[0, j, 7]) * 32.0
        q = int(round(t_star / 0.2))
        assert abs(s_star - float((st[0, 1:q + 1, :2] - st[0, :q, :2]).norm(dim=-1).sum())) < 1e-3
        assert (float(top[0, j, 8]) > 0) == (i == 0)                                         # side: left > 0
        assert float(top[0, j, 9]) in (0.0, 1.0)
    with pytest.raises(ValueError):
        V5Cfg(relation="student", top_feats="v2").validate()                               # needs the corrector
    assert V5Cfg(relation="student", top_feats="v2", corrector=True).n_top_feat == 13
    assert V5Cfg().n_top_feat == 7


# ----------------------------------------------------------------------------------------------- c2 + nets
def _student_net(**v5kw):
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, build_student_ck2
    v4 = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
          "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": True, "edits": "off",
          "corrector": False, "v5": dict({"enabled": True, "relation": "student", "unknown_ev": True,
                                          "corrector": True, "n_obj": 8}, **v5kw)}
    cfg = CKE2E2Config.from_any({"v4": v4})
    torch.manual_seed(0)
    return build_student_ck2(cfg, apply_prior=False), cfg


def _pool_inputs(n=2, seed=1):
    from navsim.agents.para_ssr.ck import anchor_sampler as AS
    from navsim.agents.para_ssr.ck import cands2 as C2
    from navsim.agents.para_ssr.ck.online2 import infer_pool
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    anc = torch.from_numpy(np.asarray(AS.load_anchors(), np.float32))
    top = torch.stack([anc[i * 16 + 3:(i + 1) * 16 + 3] for i in range(n)])
    status = torch.zeros(n, 8)
    status[:, 1], status[:, 4], status[:, 5] = 1.0, 6.0, 0.3
    V = C2.online_variants(top, C2.v0_from_status(status), None)
    cand, fin, par = infer_pool(V["traj96"], top)
    g = torch.Generator().manual_seed(seed)
    bev = torch.randn(n, 256, 50, 100, generator=g)
    Q = 12
    code = torch.zeros(n, Q, 10)
    code[..., 0] = torch.rand(n, Q, generator=g) * 8 - 4
    code[..., 1] = torch.rand(n, Q, generator=g) * 25 + 3
    code[..., 2:4] = 0.5
    code[..., 7] = 1.0
    code[..., 8:10] = torch.randn(n, Q, 2, generator=g)
    det = {"all_bbox_preds": [code], "all_cls_scores": [torch.randn(n, Q, 7, generator=g) + 1.0],
           "det_hidden": torch.randn(n, Q, 256, generator=g), "traj_preds": torch.randn(n, Q, 6, 8, 2, generator=g),
           "traj_cls_preds": torch.randn(n, Q, 6, generator=g)}
    return cand, par, V, ego_inputs(status), bev, det, status


def test_delta_r_probes_equal_the_pool_siblings_and_v2_width():
    from navsim.agents.para_ssr.ck.online2 import student_rel_rows
    net, cfg = _student_net(rel_modes="cv+fut", top_feats="v2")
    assert net.corrector_v5.n_top == 39 and net.relation.use_fut
    cand, par, V, ego, bev, det, _ = _pool_inputs()
    with torch.no_grad():
        feat, mem = net.trunk.scene(bev, cand.shape[0])
        rel = student_rel_rows(det, cfg.v5)
        assert int(rel["valid"].sum()) > 2
        o = net.score_set(feat, mem, ego, cand, par, return_ctx=True, rel_objs=rel)
        hz = o["haz"]
        cols = torch.tensor([[0, 6, 12], [18, 24, 30]])
        T, R = cols.shape
        g4 = lambda a: torch.gather(a, 1, cols.reshape(T, R, *([1] * (a.dim() - 2))).expand(T, R, *a.shape[2:]))  # noqa
        top = net._v5_top_v2(hz, g4, g4(cand.float()), ego, g4(hz["rel_top"]))
        dr = top.reshape(T, R, 3, 13)[..., 10:]
        a = hz["rel_a"]
        ar = torch.maximum((a[:, None] * hz["rel_r_nc"]).clamp(max=1 - 1e-4), (a[:, None] * hz["rel_r_ttc"]).clamp(
            max=1 - 1e-4))                                                                   # [T, 96, N]
        ti = g4(hz["rel_top_idx"])
        n_chk = 0
        for b in range(T):
            for r in range(R):
                c = int(cols[b, r])
                for j in range(3):
                    i = int(ti[b, r, j])
                    for p, off in enumerate((2, 4, 5)):                                    # a-0.5, l-0.5, l+0.5
                        if i < 0 or not bool(V["valid96"][b, c + off]):
                            assert float(dr[b, r, j, p]) == 0.0
                            continue
                        assert abs(float(dr[b, r, j, p]) - float(ar[b, c, i] - ar[b, c + off, i])) < 1e-5
                        n_chk += 1
        assert n_chk > 0


def test_policy_replay_equal_with_v2_inputs():
    """the recorded policy (_v5_act on the 96-pool score_set at the pick) and its AWR replay (v5_logp: score_set of
    the source alone) see the same inputs incl. Delta r -> equal log pi."""
    from navsim.agents.para_ssr.ck.online2 import student_rel_rows
    net, cfg = _student_net(rel_modes="cv+fut", top_feats="v2")
    with torch.no_grad():
        for p in net.corrector_v5.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    cand, par, V, ego, bev, det, _ = _pool_inputs()
    pick = torch.tensor([[7], [40]])
    with torch.no_grad():
        feat, mem = net.trunk.scene(bev, cand.shape[0])
        pref = net.scene_pref(mem, ego)
        rel = student_rel_rows(det, cfg.v5)
        po = net.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=True, rel_objs=rel)
        a, src, _, _ = net._v5_act(po, cand.float(), pick, ego)
        u = a["mu"] + 0.1
        lp_rec = net.corrector_v5.log_prob(a["mu"], a["log_sig"], u)
        src_par = torch.gather(par, 1, pick[..., None, None].expand(2, 1, 8, 3))
        lp_rep = net.v5_logp(feat, mem, ego, src, src_par, u, pref=pref, rel_objs=rel)
    assert float((lp_rec - lp_rep).abs().max()) < 1e-4


def test_relation_reread_matches_score_set_and_grad_switch():
    from navsim.agents.para_ssr.ck.online2 import student_rel_rows
    net, cfg = _student_net(rel_modes="cv+fut")
    cand, par, V, ego, bev, det, _ = _pool_inputs()
    with torch.no_grad():
        feat, mem = net.trunk.scene(bev, cand.shape[0])
        rel = student_rel_rows(det, cfg.v5)
        o = net.score_set(feat, mem, ego, cand, par, rel_objs=rel)
    rr = net.relation_reread(o["haz"]["rel_objs_tok"], cand[:, :10], ego)
    assert not rr["r_nc"].requires_grad and rr["r_nc"].shape == o["haz"]["rel_r_nc"][:, :10].shape
    assert float((rr["r_nc"] - o["haz"]["rel_r_nc"][:, :10]).abs().max()) < 1e-6
    assert float((rr["H"] - o["haz"]["zq"][:, :10, :, -2:]).abs().max()) < 1e-6
    rg = net.relation_reread(o["haz"]["rel_objs_tok"], cand[:, :10], ego, grad=True)
    rg["H"].sum().backward()
    assert net.relation.log_eps.grad is not None


def test_v2_mu0_identity_and_off_config_unchanged():
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, student_rel_rows
    net, cfg = _student_net(rel_modes="cv+fut", top_feats="v2")
    cand, par, V, ego, bev, det, _ = _pool_inputs()
    with torch.no_grad():
        feat, mem = net.trunk.scene(bev, cand.shape[0])
        o = net.score_set(feat, mem, ego, cand, par, return_ctx=True, rel_objs=student_rel_rows(det, cfg.v5))
        rf = net.refine_drafts(o, cand.float(), torch.tensor([[0, 13], [50, 91]]), ego)
    assert torch.equal(rf["tau1"], rf["tau_base"])                                          # zero-init mu
    # defaults: the new keys are off, V5Cfg / CKNet as before
    c0 = CKE2E2Config.from_any({"v4": {"v5": {"enabled": True, "relation": "student", "corrector": True}}})
    q = c0.v5
    assert (q["rel_modes"], q["top_feats"], q["lambda_rel"], q["lambda_rel_cv"], q["rel_max_parents"],
            q["rel_det_grad_matched"]) == ("cv", "v1", 0.0, 0.0, 8, False)
    assert c0.v5_net() == V5Cfg(relation="student", rel_ca=False, corrector=True, z_stride=2)
    assert "rel_modes" not in V5Cfg().to_dict() and "top_feats" not in V5Cfg().to_dict()
    assert V5Cfg.from_dict(V5Cfg(relation="student", rel_modes="cv+fut").to_dict()).rel_modes == "cv+fut"
    for bad in ({"lambda_rel": 0.5, "relation": "off"}, {"lambda_rel_cv": 0.25}, {"lambda_rel": -1.0},
                {"top_feats": "v2", "corrector": False}, {"rel_det_grad_matched": True},
                {"top_feats": "v3"}, {"rel_max_parents": 0}):
        with pytest.raises(ValueError):
            CKE2E2Config.from_any({"v4": {"v5": dict({"enabled": True, "relation": "student", "corrector": True},
                                                     **bad)}})


def test_teacher_checkpoints_load_strict():
    from navsim.agents.para_ssr.ck.model import load_ck
    base = Path("/home/external-user/ssd/yongjae_refiner/ck/v5/train")
    runs = [r for r in ("v5T_rel4", "v5M_unk3") if (base / r / "ckpt_last.pt").is_file()]
    if not runs:
        pytest.skip("no v5 teacher checkpoint")
    for r in runs:
        net, _ = load_ck(base / r, "last", "cpu")
        assert net.v5.rel_modes == "cv" and net.v5.top_feats == "v1"
        if net.corrector_v5 is not None and net.relation is not None:
            assert net.corrector_v5.n_top == 21


# ----------------------------------------------------------------------------------------------- _v5_terms (L_rel)
def _terms_inputs(net, cfg, lam_rel=0.5, lam_cv=0.25):
    """a KD-like set (16 parents + 32 variants per token) of 2 tokens, GT with tracks on the detections."""
    from navsim.agents.para_ssr.ck.online2 import student_forward2, student_rel_rows
    cand, par, V, ego, bev, det, status = _pool_inputs()
    keep = torch.cat([torch.arange(16) * 6, torch.arange(16) * 6 + 2, torch.arange(16) * 6 + 4])
    cand_kd, par_kd = cand[:, keep], par[:, keep]
    vtype = torch.cat([torch.zeros(16), torch.full((32,), 2.0)]).long()[None].expand(2, 48)
    rel = student_rel_rows(det, dict(cfg.v5, rel_det_grad=True))
    # GT: a track at each of the first 3 valid rows (+ 0.3 m), one far track
    tg = {}
    A = 4
    kf = torch.zeros(2, 8, GF.N_KF, 6)
    first = torch.zeros(2, 8, 6)
    meta = torch.zeros(2, 8, 5, dtype=torch.long)
    for b in range(2):
        vi = rel["valid"][b].nonzero()[:, 0][:3]
        for a in range(A):
            xy = rel["c0"][b, int(vi[a])].detach() + 0.3 if a < len(vi) else torch.tensor([60.0, 0.0])
            kf[b, a, :, GF.KF_X], kf[b, a, :, GF.KF_Y] = float(xy[0]), float(xy[1])
            kf[b, a, :, GF.KF_PRESENT] = 1.0
            first[b, a, GF.FI_L], first[b, a, GF.FI_W] = 4.5, 1.9
            meta[b, a, GF.MT_AGENT] = 1
            meta[b, a, GF.MT_LAST] = GF.N_KF - 1
    tg = {"ref_gt_ok": torch.ones(2, dtype=torch.bool), "ref_obj_kf": kf, "ref_obj_first": first, "ref_obj_meta": meta,
          "ref_obj_n": torch.tensor([A, A]), "ref_obj_n_kf": torch.tensor([GF.N_KF, GF.N_KF]),
          "ref_sdf": torch.ones(2, 320, 256) * 3.0, "ref_p_pdm": torch.tensor([30.0, 30.0])}
    return cand_kd, par_kd, vtype, rel, tg, status, bev


@pytest.mark.parametrize("lam", [(0.5, 0.25), (0.0, 0.0)])
def test_v5_terms_rel_loss_and_off_identity(lam):
    from navsim.agents.para_ssr.ck import online2 as O2
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, student_forward2, student_rel_match
    net, cfg0 = _student_net(rel_modes="cv+fut", top_feats="v2")
    q = dict(cfg0.v5, lambda_rel=lam[0], lambda_rel_cv=lam[1], lambda_ref=0.0, lambda_rel_a=0.0, lambda_rel_h=0.0,
             rel_det_grad=True, rel_det_grad_matched=True)
    cfg = SimpleNamespace(v5=q)
    cand_kd, par_kd, vtype, rel, tg, status, bev = _terms_inputs(net, cfg0)
    match = student_rel_match(rel, tg, torch.device("cpu"))
    assert int((match >= 0).sum()) >= 2
    from navsim.agents.para_ssr.ck.v5.student_objects import rows_keep_graph
    relm = rows_keep_graph(rel, match >= 0)
    out = student_forward2(net, bev, cand_kd, status, cand_parent=par_kd, rel_objs=relm)
    S = {"cand": cand_kd, "kd_tok_ok": torch.ones(2, dtype=torch.bool), "vtype_kd": vtype}
    logs = {}
    inner0 = torch.zeros(())
    fake = SimpleNamespace(cfg=cfg)
    fake._v5_rel_terms = lambda *a: O2.CKE2E2._v5_rel_terms(fake, *a)
    inner = O2.CKE2E2._v5_terms(fake, inner0, out, S, logs, tg, net, {}, relm, out["ego"], 1.0, rel_match=match)
    if lam[0] > 0:
        for k in ("v5/rel_bce", "v5/rel_cv_bce", "v5/rel_beyond_cv", "v5/rel_cv_disagree", "v5/rel_n_pairs",
                  "v5/rel_match_frac", "v5/rel_pos"):
            assert k in logs and math.isfinite(logs[k]), k
        assert logs["v5/rel_n_pairs"] > 0 and float(inner) > 0
        net.zero_grad()
        inner.backward()
        assert float(net.relation.log_eps.grad.abs()) > 0 and float(net.relation.cv_prior.grad.abs()) > 0
    else:
        assert not any(k.startswith("v5/rel_") and k not in ("v5/rel_a_bce",) for k in logs)
        assert float(inner) == 0.0


def test_arm_set_switches_the_rel_group_on():
    from tools.ck.e2e2 import launch_util2 as LU2
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    sets = [(REPO / f"tools/ck/e2e2/{f}").read_text().splitlines() for f in ("v5_arm.set", "v5kd_arm.set")]
    for arm in (sets[0], sets[0] + sets[1]):
        c = CKE2E2Config.from_any(LU2.apply_sets2(LU2.load_ck2_yaml(), arm))
        q = c.v5
        assert (q["rel_modes"], q["top_feats"], q["lambda_rel"], q["lambda_rel_cv"], q["rel_max_parents"],
                q["rel_det_grad_matched"]) == ("cv+fut", "v2", 0.5, 0.25, 8, True)
        n = c.v5_net()
        assert n.rel_modes == "cv+fut" and n.top_feats == "v2" and n.n_top_feat == 13
        assert n.probe_s_on_frac == float(c.variants["s_on_frac"])


def test_expose_det_outputs_carries_the_futures():
    from navsim.agents.para_ssr.para_ssr_model import EXPOSED_DET_KEYS
    assert "traj_preds" in EXPOSED_DET_KEYS and "traj_cls_preds" in EXPOSED_DET_KEYS


def test_online2_comment_names_the_critic_column_correctly():
    src = (REPO / "navsim/agents/para_ssr/ck/online2.py").read_text()
    assert "(L_label tau1 / L_edit-Bernoulli)" not in src


# ----------------------------------------------------------------------------------------------- a2: eval replay
def test_infer_check_replays_a_futures_v2_student(tmp_path):
    """eval_e2e2: a rel_modes cv+fut / top_feats v2 student's infer_check rebuilds the rows from check_bev.npz incl. the
    futures det_traj / det_traj_cls and replays v5_traj (Delta r probes included) exactly; a dump without the futures
    is refused."""
    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, build_student_ck2, lateral_decode, student_rel_rows
    from tools.ck.e2e import eval_e2e as E1
    from tools.ck.e2e2 import eval_e2e2 as E2
    v4 = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
          "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": True, "edits": "off",
          "corrector": False, "v5": {"enabled": True, "relation": "student", "unknown_ev": True, "corrector": True,
                                     "n_obj": 8, "rel_modes": "cv+fut", "top_feats": "v2"}}
    cfg = CKE2E2Config.from_any({"v4": v4})
    torch.manual_seed(0)
    net = build_student_ck2(cfg, apply_prior=False)
    with torch.no_grad():
        for p in net.corrector_v5.mu.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    R = tmp_path / "e2e_v5"
    ckpt = R / "train" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"state_dict": {E1.STUDENT_PREFIX + k: v for k, v in net.state_dict().items()}}, ckpt)
    P = E1.Paths(R, "navtrain_val")
    E2.extract_ck2(ckpt, P.ck_run, "e2e_v5", {"enabled": True, "seed": 0, "v4": v4})
    net2, _ = load_ck(P.ck_run, "last", "cpu")
    assert net2.v5.rel_modes == "cv+fut" and net2.v5.top_feats == "v2" and net2.corrector_v5.n_top == 39
    cand, par, V, ego, bev, det, status = _pool_inputs()
    n = cand.shape[0]
    raw = V["traj96"]
    P.packed.mkdir(parents=True, exist_ok=True)
    for k, v in (("cand96", raw), ("status", status), ("valid96", V["valid96"]),
                 ("v2_final", torch.linspace(1, 0, 16).expand(n, 16)), ("v2_im", torch.full((n, 16), 1 / 16))):
        np.save(P.packed / f"{k}.npy", v.numpy())
    rows = np.arange(n)
    P.infer.mkdir(parents=True, exist_ok=True)
    base = dict(rows=rows, bev=bev.numpy(), det_code=det["all_bbox_preds"][0].numpy(),
                det_cls=det["all_cls_scores"][0].numpy(), det_hidden=det["det_hidden"].numpy())
    np.savez(P.infer / "check_bev.npz", **base, det_traj=det["traj_preds"].numpy(),
             det_traj_cls=det["traj_cls_preds"].numpy())
    with torch.no_grad():
        feat, mem = net2.trunk.scene(bev, n)
        pref = net2.scene_pref(mem, ego)
        o = net2.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=True, rel_objs=student_rel_rows(det, cfg.v5))
        src = torch.tensor([3, 50])
        rf = net2.refine_drafts(o, cand.float(), src[:, None], ego)
        lat = lateral_decode(cand, o["w_lat"], ego[0], 0.0)
    for k, v in (("score_logit", o["score_logit"]), ("w_lat", o["w_lat"]), ("e_lat", lat["e_lat"][..., 2:]),
                 ("lat_traj", lat["traj"]), ("v5_traj", rf["tau1"]), ("v5_mu", rf["mu"][:, 0])):
        np.save(P.infer / f"{k}.npy", v.float().numpy())
    np.save(P.infer / "v5_src.npy", src.numpy().astype(np.int16))
    a = SimpleNamespace(gpus="")
    res = E2.stage_infer_check(P, a)
    assert res["pass"] and res["max_abs_err"]["v5_traj"] < 1e-5
    np.savez(P.infer / "check_bev.npz", **base)
    with pytest.raises(SystemExit, match="det_traj"):
        E2.stage_infer_check(P, a)


def test_extract_ck2_keeps_the_runs_variants_for_the_delta_r_probes(tmp_path):
    """a top_feats 'v2' student trained with non-default variants (s_on_frac 0.3, compute_dtype float32): extract_ck2
    passes the run's variants block on, so the extracted net (load_ck: eval / infer_check) builds its Delta r probes
    exactly as training / the eval agent (build_student_ck2 of the same config) do."""
    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, build_student_ck2
    from tools.ck.e2e import eval_e2e as E1
    from tools.ck.e2e2 import eval_e2e2 as E2
    v4 = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
          "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": True, "edits": "off",
          "corrector": False, "v5": {"enabled": True, "relation": "student", "unknown_ev": True, "corrector": True,
                                     "n_obj": 8, "rel_modes": "cv+fut", "top_feats": "v2"}}
    ck_cfg = {"enabled": True, "seed": 0, "v4": v4, "variants": {"s_on_frac": 0.3, "compute_dtype": "float32"}}
    cfg = CKE2E2Config.from_any(ck_cfg)
    torch.manual_seed(0)
    net = build_student_ck2(cfg, apply_prior=False)
    assert (net.v5.probe_s_on_frac, net.v5.probe_dtype) == (0.3, "float32")
    R = tmp_path / "e2e_v5"
    ckpt = R / "train" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"state_dict": {E1.STUDENT_PREFIX + k: v for k, v in net.state_dict().items()}}, ckpt)
    P = E1.Paths(R, "navtrain_val")
    E2.extract_ck2(ckpt, P.ck_run, "e2e_v5", ck_cfg)
    net2, _ = load_ck(P.ck_run, "last", "cpu")
    assert (net2.v5.probe_s_on_frac, net2.v5.probe_dtype) == (0.3, "float32")
    assert net2.v5.to_dict() == net.v5.to_dict()
    # the default-variants run still extracts with the defaults (and omits them from config.json 'v5')
    ck0 = {"enabled": True, "seed": 0, "v4": v4}
    E2.extract_ck2(ckpt, R / "ck0", "e2e_v5", ck0)
    net3, c3 = load_ck(R / "ck0", "last", "cpu")
    assert (net3.v5.probe_s_on_frac, net3.v5.probe_dtype) == (0.2, "float64")
    assert "probe_s_on_frac" not in c3["v5"] and "probe_dtype" not in c3["v5"]


# =============================================================================================== 'zkd' group
# (design §2.1 L82 / §3.7 item 3 / §3.8; main-session decisions D1 (a) / (a') / (b), D2, D6-D8): the student Z adapters
# P_det / P_map on teacher-normalised reads, tools/ck/v5/fit_z_stats.py, Z-KD in teacher-normalised units,
# L_edit-Bernoulli on the tau1 re-read.
from navsim.agents.para_ssr.ck.v5 import zadapt as ZA  # noqa: E402

ZSTATS_T = {"b_nc": (-0.9, 0.45), "b_ttc": (-0.6, 0.4), "H_nc": (0.01, 0.06), "H_ttc": (0.02, 0.1)}
ZSTATS_M = {"d_min": (0.55, 0.37)}


def _zrec(arm, chans=None, **kw):
    """a synthetic fit_z_stats record (zadapt.STATS_VERSION) of arm T / M."""
    chans = chans if chans is not None else (ZSTATS_T if arm == "T" else ZSTATS_M)
    rec = {"version": ZA.STATS_VERSION, "arm": arm, "run": "/x/run", "which": "last", "ckpt_sha16": "0" * 16,
           "grid": {"z_stride": 2}, "channels": {n: {"mean": m, "std": s, "n": 5000} for n, (m, s) in chans.items()},
           "unfinished": False, "dry_run": False, "data": {"n_tokens": 4096, "limit": 0}}
    rec.update(kw)
    return rec


def _zq(B=2, K=5, Q=21, C=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, K, Q, C, generator=g)
    z[..., 3] = (torch.rand(B, K, Q, generator=g) > 0.8).float()
    z[..., 4:] = torch.rand(B, K, Q, 2, generator=g)
    return z


def test_zadapter_identity_refusal_and_halves():
    za = ZA.ZAdapter(6, 32, "all")
    z = _zq()
    with pytest.raises(ValueError, match="statistics are not set"):
        za(z)
    with pytest.raises(ValueError, match="statistics are not set"):
        za.normalise(z)
    with pytest.raises(ValueError, match="required"):
        za.set_stats(_zrec("T"), None)
    with pytest.raises(ValueError, match="arm"):
        za.set_stats(_zrec("M"), _zrec("M"))
    with pytest.raises(ValueError, match="UNFINISHED"):
        za.set_stats(_zrec("T", unfinished=True), _zrec("M"))
    with pytest.raises(ValueError, match="H_ttc"):
        za.set_stats(_zrec("T", {k: v for k, v in ZSTATS_T.items() if k != "H_ttc"}), _zrec("M"))
    st = za.set_stats(_zrec("T"), _zrec("M"))
    assert set(st) == {"b_nc", "b_ttc", "H_nc", "H_ttc", "d_min"} and za.fitted()
    mu = torch.tensor([-0.9, -0.6, 0.55, 0.0, 0.01, 0.02])
    sd = torch.tensor([0.45, 0.4, 0.37, 1.0, 0.25, 0.25])          # H channels: sigma floored at SD_FLOOR_CH
    zn = (z - mu) / sd
    assert torch.allclose(za.normalise(z), zn, atol=1e-6)
    assert torch.equal(za(z), za.normalise(z))                                              # identity at init
    assert torch.equal(za(z)[..., 3], z[..., 3])                                            # unknown bit for bit
    zm = za.normalise(z[..., :4])                                                           # a MAP teacher's 4-ch zq
    assert torch.allclose(zm, zn[..., :4], atol=1e-6)
    # a trained-looking adapter changes only the stat channels, keeps unknown
    with torch.no_grad():
        for p in za.parameters():
            p.add_(0.1)
    y = za(z)
    assert torch.equal(y[..., 3], z[..., 3]) and float((y - zn).abs().max()) > 0.05
    # halves: 'det' builds P_det only (needs the DET record only; d_min passes unchanged), 'map' P_map only
    zd = ZA.ZAdapter(6, 16, "det")
    assert zd.p_map is None and zd.p_det is not None
    zd.set_stats(_zrec("T"), None)
    assert torch.equal(zd(z)[..., 2], z[..., 2])
    zmp = ZA.ZAdapter(6, 16, "map")
    assert zmp.p_det is None
    zmp.set_stats(None, _zrec("M"))
    assert torch.equal(zmp(z)[..., 0], z[..., 0]) and torch.allclose(zmp(z)[..., 2], zn[..., 2], atol=1e-6)
    # n_z 4 (no relation H): P_det on b_nc / b_ttc only
    z4 = ZA.ZAdapter(4)
    z4.set_stats(_zrec("T"), _zrec("M"))
    assert z4.det_ch == (0, 1) and z4(z[..., :4]).shape[-1] == 4
    with pytest.raises(ValueError):
        z4(z)
    # the buffers round-trip through a state dict (a checkpoint carries the statistics)
    zb = ZA.ZAdapter(6, 32, "all")
    zb.load_state_dict(za.state_dict())
    assert zb.fitted() and torch.equal(zb(z), za(z))


def test_check_z_stats_refusals():
    rec = _zrec("T", run="/x/run", which="last", ckpt_sha16="a" * 16)
    ZA.check_z_stats(rec, "T", run="/x/run", which="last", ckpt_sha16="a" * 16, z_stride=2)
    for kw, msg in (({"run": "/x/other"}, "run"), ({"which": "ep1"}, "which"), ({"ckpt_sha16": "b" * 16}, "sha16"),
                    ({"z_stride": 1}, "z_stride")):
        a = dict(run="/x/run", which="last", ckpt_sha16="a" * 16, z_stride=2)
        a.update(kw)
        with pytest.raises(ValueError, match=msg):
            ZA.check_z_stats(rec, "T", **a)
    with pytest.raises(ValueError, match="arm"):
        ZA.check_z_stats(rec, "M")
    dry = _zrec("T", dry_run=True)
    ZA.check_z_stats(dry, "T")                                                              # runtime: accepted
    with pytest.raises(ValueError, match="DRY-RUN"):
        ZA.check_z_stats(dry, "T", allow_dry_run=False)                                     # strict launch: refused
    with pytest.raises(ValueError, match="version"):
        ZA.check_z_stats(dict(rec, version="x"), "T")


def test_zstats_accumulator_matches_numpy_under_the_mask_rule():
    z = _zq(B=3, K=7, Q=21, C=6, seed=3)
    z[0, :2, :5, 0] = ZA.ZQ_FLOOR                                                           # not-read B states
    z[1, 3, :, 1] = ZA.ZQ_FLOOR
    z[2, 0, 0, 4] = float("nan")
    col_ok = torch.ones(3, 7, dtype=torch.bool)
    col_ok[1, 6] = False
    acc = ZA.ZStatsAcc(ZA.TEACHER_CHANNELS["T"])
    acc.update(z[:2], col_ok[:2])
    acc.update(z[2:], col_ok[2:])
    ch = acc.channels(n_min=10)
    zn, ok = z.double().numpy(), col_ok.numpy()
    for n in ZA.TEACHER_CHANNELS["T"]:
        c = ZA.CH[n]
        v = zn[..., c]
        m = ok[..., None] & np.isfinite(v) & ~(zn[..., 3] > 0.5)
        if n in ZA.B_CHANNELS:
            known = m.copy()
            m = m & (v > ZA.ZQ_FLOOR + 1e-6)
            assert abs(ch[n]["floor_share"] - (known & ~m).sum() / known.sum()) < 1e-12
        x = v[m]
        assert ch[n]["n"] == x.size
        assert abs(ch[n]["mean"] - x.mean()) < 1e-10 and abs(ch[n]["std"] - x.std(ddof=1)) < 1e-10
    assert ch["b_nc"]["floor_share"] > 0
    with pytest.raises(ValueError, match="N_MIN"):
        acc.channels(n_min=10 ** 9)
    accm = ZA.ZStatsAcc(ZA.TEACHER_CHANNELS["M"])
    accm.update(z, col_ok)
    assert set(accm.channels(n_min=10)) == {"d_min"}


def test_fit_z_stats_rows_out_and_parser():
    from tools.ck.v5 import fit_z_stats as FZ
    logs = np.repeat(np.arange(200), 5)                                                     # 200 logs x 5 rows
    for n in (8, 64, 300, 1000, 5000):
        sel = FZ.select_rows(logs, n, seed=0)
        assert len(sel) == min(n, len(logs)) and len(np.unique(sel)) == len(sel) and np.all(np.diff(sel) > 0)
    assert len(np.unique(logs[FZ.select_rows(logs, 64, 0)])) == 64                          # spread over logs
    assert np.array_equal(FZ.select_rows(logs, 64, 1), FZ.select_rows(logs, 64, 1))
    assert len(FZ.select_rows(logs, 64, 0, limit=4)) == 4
    for bad in ("/home/external-user/ssd/yongjae_refiner/ck/v5/train/v5T_rel4/z.json",
                "/home/external-user/ssd/yongjae_refiner/ck/ck2/train/x/z.json", "/tmp/a=b/z.json"):
        with pytest.raises(SystemExit):
            FZ.check_out(Path(bad))
    a = FZ.get_parser().parse_args(["--run", "v5T_rel4"])
    assert (a.tokens, a.which, a.device, a.limit) == (4096, "last", "cpu", 0)
    assert FZ.default_out(Path("/r/v5T_rel4"), "last") == FZ.CK_DATA / "v5/z_stats/v5T_rel4__last.json"


def _zkd_net(z_adapt=True, **kw):
    net, cfg = _student_net(rel_modes="cv+fut", top_feats="v2",
                            **({"z_adapt": True, "z_stats_det": "/x/d.json", "z_stats_map": "/x/m.json"}
                               if z_adapt else {}), **kw)
    if z_adapt:
        net.z_adapter.set_stats(_zrec("T"), _zrec("M"))
    return net, cfg


def test_z_adapt_off_identity_rng_and_config():
    net0, cfg0 = _zkd_net(False)
    net1, cfg1 = _zkd_net(True)
    a, b = net0.state_dict(), net1.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)                                          # RNG order: unchanged
    assert sorted(set(b) - set(a)) and all(k.startswith("z_adapter.") for k in set(b) - set(a))
    assert net0.z_adapter is None and "z_adapter" not in net0.param_counts()
    assert net1.param_counts()["z_adapter"] == sum(p.numel() for p in net1.z_adapter.parameters())
    assert "z_adapt" not in V5Cfg().to_dict() and "z_adapt_hidden" not in V5Cfg().to_dict()
    assert cfg1.v5_net().z_adapt and not cfg0.v5_net().z_adapt
    assert V5Cfg.from_dict(cfg1.v5_net().to_dict()).z_adapt
    with pytest.raises(ValueError, match="corrector"):
        V5Cfg(relation="student", z_adapt=True).validate()
    q0 = cfg0.v5
    assert (q0["z_adapt"], q0["z_stats_det"], q0["z_stats_map"], q0["lambda_edit"]) == (False, "", "", 0.0)
    base = {"enabled": True, "relation": "student", "corrector": True}
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    for bad in ({"z_adapt": True}, {"z_adapt": True, "z_stats_det": "/x/d.json"},
                {"z_adapt": True, "z_stats_det": "rel/d.json", "z_stats_map": "/x/m.json"},
                {"z_adapt": True, "z_stats_det": "/x/d=1.json", "z_stats_map": "/x/m.json"},
                {"z_adapt": True, "z_stats_det": "/x/d.json", "z_stats_map": "/x/m.json", "corrector": False},
                {"lambda_edit": -0.1}, {"lambda_edit": 0.25, "relation": "off"},
                {"lambda_edit": 0.25, "corrector": False}):
        with pytest.raises(ValueError):
            CKE2E2Config.from_any({"v4": {"v5": dict(base, **bad)}})
    # same drafts: tau1 at mu = 0 equal with and without the adapter (zero-init mu), corrector input = P(Z)
    cand, par, V, ego, bev, det, _ = _pool_inputs()
    from navsim.agents.para_ssr.ck.online2 import student_rel_rows
    seen = {}
    orig = net1.corrector_v5.forward

    def spy(ctx, logit, v0, zq, tq, top=None):
        seen["zq"] = zq
        return orig(ctx, logit, v0, zq, tq, top)
    net1.corrector_v5.forward = spy
    cols = torch.tensor([[0, 13], [50, 91]])
    with torch.no_grad():
        feat, mem = net1.trunk.scene(bev, cand.shape[0])
        rel = student_rel_rows(det, cfg1.v5)
        o1 = net1.score_set(feat, mem, ego, cand, par, return_ctx=True, rel_objs=rel)
        o0 = net0.score_set(feat, mem, ego, cand, par, return_ctx=True, rel_objs=rel)
        rf1 = net1.refine_drafts(o1, cand.float(), cols, ego)
        rf0 = net0.refine_drafts(o0, cand.float(), cols, ego)
    assert torch.equal(rf1["tau1"], rf0["tau1"]) and torch.equal(o1["haz"]["zq"], o0["haz"]["zq"])
    g = torch.gather(o1["haz"]["zq"], 1, cols[..., None, None].expand(2, 2, *o1["haz"]["zq"].shape[2:]))
    assert torch.equal(seen["zq"], net1.z_adapter(g))                                       # raw haz zq, adapted input


def test_policy_replay_and_infer_consistency_with_the_adapter():
    """recorded policy (_v5_act on the 96 pool) == AWR replay (v5_logp) with a trained-looking adapter."""
    from navsim.agents.para_ssr.ck.online2 import student_rel_rows
    net, cfg = _zkd_net(True)
    with torch.no_grad():
        for p in list(net.corrector_v5.parameters()) + list(net.z_adapter.parameters()):
            p.add_(torch.randn_like(p) * 0.05)
    cand, par, V, ego, bev, det, _ = _pool_inputs()
    pick = torch.tensor([[7], [40]])
    with torch.no_grad():
        feat, mem = net.trunk.scene(bev, cand.shape[0])
        pref = net.scene_pref(mem, ego)
        rel = student_rel_rows(det, cfg.v5)
        po = net.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=True, rel_objs=rel)
        a, src, _, _ = net._v5_act(po, cand.float(), pick, ego)
        u = a["mu"] + 0.1
        lp_rec = net.corrector_v5.log_prob(a["mu"], a["log_sig"], u)
        src_par = torch.gather(par, 1, pick[..., None, None].expand(2, 1, 8, 3))
        lp_rep = net.v5_logp(feat, mem, ego, src, src_par, u, pref=pref, rel_objs=rel)
    assert float((lp_rec - lp_rep).abs().max()) < 1e-4
    # an adapter without statistics refuses (no silent identity): e.g. a fresh build that never loaded them
    net2, _ = _student_net(z_adapt=True, z_stats_det="/x/d.json", z_stats_map="/x/m.json")
    with torch.no_grad(), pytest.raises(ValueError, match="statistics"):
        o2 = net2.score_set(feat, mem, ego, cand, par, return_ctx=True, rel_objs=rel)
        net2.refine_drafts(o2, cand.float(), pick, ego)


def _fake_e2e(q):
    from navsim.agents.para_ssr.ck import online2 as O2
    fake = SimpleNamespace(cfg=SimpleNamespace(v5=q))
    for m in ("_v5_rel_terms", "_v5_edit_terms", "_awr_v5"):
        fake.__dict__[m] = (lambda name: (lambda *a: getattr(O2.CKE2E2, name)(fake, *a)))(m)
    return O2, fake


def _zkd_terms(net, cfg0, q, T=None, match=True):
    from navsim.agents.para_ssr.ck.online2 import student_forward2, student_rel_match
    from navsim.agents.para_ssr.ck.v5.student_objects import rows_keep_graph
    cand_kd, par_kd, vtype, rel, tg, status, bev = _terms_inputs(net, cfg0)
    m = student_rel_match(rel, tg, torch.device("cpu"))
    relm = rows_keep_graph(rel, (m >= 0) & rel["valid"])
    out = student_forward2(net, bev, cand_kd, status, cand_parent=par_kd, rel_objs=relm)
    S = {"cand": cand_kd, "kd_tok_ok": torch.ones(2, dtype=torch.bool), "vtype_kd": vtype,
         "cand_ok": torch.ones(2, cand_kd.shape[1], dtype=torch.bool)}
    O2, fake = _fake_e2e(q)
    logs = {}
    inner = O2.CKE2E2._v5_terms(fake, torch.zeros(()), out, S, logs, tg, net, T or {}, relm, out["ego"], 1.0,
                                rel_match=m if match else None)
    return inner, logs, out, S, relm, m, tg


def _teacher_zq(out, seed=5):
    hk = out["haz"]
    B, K, Q, _ = hk["zq"].shape
    g = torch.Generator().manual_seed(seed)
    zd = hk["zq"].detach().float() + 0.3 * torch.randn(B, K, Q, 6, generator=g)
    zd[..., 3] = hk["zq"][..., 3].detach()
    zm = hk["zq"][..., :4].detach().float() + 0.2 * torch.randn(B, K, Q, 4, generator=g)
    zm[..., 3] = hk["zq"][..., 3].detach()
    return {"zq_det": zd, "zq_map": zm, "kd_ok": torch.ones(B, K, dtype=torch.bool)}


def test_zkd_teacher_normalised_units_and_per_channel_logs():
    import torch.nn.functional as F
    net, cfg0 = _zkd_net(True)
    q = dict(cfg0.v5, z_kd=True, lambda_ref=0.0, lambda_rel_a=0.0, lambda_rel_h=0.0, lambda_zkd_det=0.5,
             lambda_zkd_map=0.5)
    _, _, out, *_ = _zkd_terms(net, cfg0, dict(q, z_kd=False))
    T = _teacher_zq(out)
    inner, logs, out, S, *_ = _zkd_terms(net, cfg0, q, T)
    za = net.z_adapter
    zs = out["haz"]["zq"].detach().float()
    md = (~(zs[..., 3] > 0.5) & ~(T["zq_det"][..., 3] > 0.5)).float()
    ch = [0, 1, 4, 5]
    with torch.no_grad():
        want_d = (F.smooth_l1_loss(za(zs)[..., ch], za.normalise(T["zq_det"])[..., ch], reduction="none").mean(-1)
                  * md).sum() / md.sum()
        want_m = (F.smooth_l1_loss(za(zs)[..., 2], za.normalise(T["zq_map"])[..., 2], reduction="none") * md).sum() \
            / md.sum()
        raw_d = (F.smooth_l1_loss(zs[..., ch], T["zq_det"][..., ch], reduction="none").mean(-1) * md).sum() / md.sum()
    assert abs(logs["v5/zkd_det"] - float(want_d)) < 1e-5 and abs(logs["v5/zkd_map"] - float(want_m)) < 1e-5
    assert abs(logs["v5/zkd_raw_det"] - float(raw_d)) < 1e-5 and logs["v5/zkd_det"] != logs["v5/zkd_raw_det"]
    for n in ("b_nc", "b_ttc", "H_nc", "H_ttc", "d_min"):
        assert math.isfinite(logs[f"v5/zkd_{n}"]), n
    assert abs(np.mean([logs[f"v5/zkd_{n}"] for n in ("b_nc", "b_ttc", "H_nc", "H_ttc")]) - logs["v5/zkd_det"]) < 1e-5
    assert abs(float(inner) - 0.5 * (float(want_d) + float(want_m))) < 1e-4
    # DDP: every adapter parameter gets a (finite) gradient; the reads' fields too
    net.zero_grad()
    inner.backward()
    for n, p in net.z_adapter.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), n
    assert any(float(p.grad.abs().sum()) > 0 for p in net.z_adapter.parameters())
    # mu = 0 / sd = 1 statistics and an identity adapter: the normalised Z-KD equals today's raw-unit Z-KD exactly
    net_id, _ = _zkd_net(True)
    net_id.z_adapter.set_stats(_zrec("T", {k: (0.0, 1.0) for k in ZSTATS_T}), _zrec("M", {"d_min": (0.0, 1.0)}))
    net_off, _ = _zkd_net(False)
    _, l_id, *_ = _zkd_terms(net_id, cfg0, q, T)
    _, l_off, *_ = _zkd_terms(net_off, cfg0, q, T)
    assert l_id["v5/zkd_det"] == l_off["v5/zkd_det"] and l_id["v5/zkd_map"] == l_off["v5/zkd_map"]
    assert not any(k.startswith(("v5/zkd_b", "v5/zkd_H", "v5/zkd_raw", "v5/zadapt")) for k in l_off)   # off: keys


def test_arm_b_adapter_gets_gradients_without_zkd():
    """arm B (no Z-KD): the adapter is trained through the corrector's L_refine; every parameter has a gradient."""
    net, cfg0 = _zkd_net(True)
    q = dict(cfg0.v5, lambda_ref=1.0, lambda_rel_a=0.0, lambda_rel_h=0.0)
    with torch.no_grad():
        for p in net.corrector_v5.mu.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    inner, logs, *_ = _zkd_terms(net, cfg0, q)
    net.zero_grad()
    inner.backward()
    for n, p in net.z_adapter.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), n
    assert "v5/zadapt_gamma_dev" in logs and "v5/zkd_det" not in logs


def test_l_edit_bernoulli_targets_mask_and_gradients():
    from navsim.agents.para_ssr.ck.online2 import v5_needs_match
    net, cfg0 = _zkd_net(False)
    with torch.no_grad():
        for p in net.corrector_v5.mu.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    q = dict(cfg0.v5, lambda_edit=0.25, lambda_ref=0.0, lambda_rel_a=0.0, lambda_rel_h=0.0, rel_det_grad=True,
             rel_det_grad_matched=True)
    assert v5_needs_match(dict(q, rel_det_grad_matched=False)) and not v5_needs_match(dict(q, lambda_edit=0.0,
                                                                                         rel_det_grad_matched=False))
    inner, logs, out, S, relm, m, tg = _zkd_terms(net, cfg0, q)
    for k in ("v5/edit_bce", "v5/edit_pos", "v5/edit_n_pairs", "v5/edit_cols", "v5/edit_r_mean", "v5/edit_match_frac",
              "v5/edit_dup_masked"):
        assert k in logs and math.isfinite(logs[k]), k
    assert logs["v5/edit_n_pairs"] > 0 and abs(float(inner) - 0.25 * logs["v5/edit_bce"]) < 1e-6
    # = the hand-built term: the relation re-read of the detached tau1 vs the matched GT track
    from navsim.agents.para_ssr.ck.online2 import refine_cols_parents
    from navsim.agents.para_ssr.ck.v5 import refine_loss as RL2
    cols = refine_cols_parents(S["vtype_kd"], q["ref_cols"])
    hk = out["haz"]
    o_like = {"ctx": out["ctx_all"][:, :S["cand"].shape[1]], "score_logit": out["score_logit"], "haz": hk,
              "w_lat": out["w_lat"]}
    with torch.no_grad():
        rf = net.refine_drafts(o_like, S["cand"].float(), cols, out["ego"])
        assert not torch.equal(rf["tau1"], rf["tau_base"])
        rr = net.relation_reread(relm, rf["tau1"], out["ego"])
        gt = RL2.gt_of(tg, torch.arange(2))
        tz = torch.arange(LQR.NT, dtype=torch.float64)[::2] * LQR.DT
        y, known = RL2.rel_object_targets(rr["states"].reshape(2, cols.shape[1], -1, 6), tz, gt, m)
        mk = RL2.rel_object_mask(relm["valid"], m, known, y)
        want = RL2.rel_object_bce(rr["r_nc"], rr["r_ttc"], y, mk, rr.get("lr_nc"), rr.get("lr_ttc"))
    assert abs(float(want) - logs["v5/edit_bce"]) < 1e-6
    # duplicates (G_DUP) masked, ghosts kept with y = 0
    dup = (m == RL2.G_DUP)[:, None, :, None].expand_as(mk)
    assert not bool((mk & dup).any())
    gh = (m == RL2.G_GHOST) & relm["valid"]
    if bool(gh.any()):
        assert float(y[gh[:, None, :, None].expand_as(y)].abs().sum()) == 0.0
    # gradient: the reader (eps, CV prior) and the matched rows' boxes; never the corrector (tau1 detached)
    net.zero_grad()
    inner.backward()
    assert float(net.relation.log_eps.grad.abs()) > 0 and float(net.relation.cv_prior.grad.abs()) > 0
    for n, p in net.corrector_v5.named_parameters():
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0, n
    # off: no re-read, no keys
    inner0, logs0, *_ = _zkd_terms(net, cfg0, dict(q, lambda_edit=0.0))
    assert not any(k.startswith("v5/edit_") for k in logs0) and float(inner0) == 0.0


def test_l_edit_duplicate_row_masked_and_matched_only_box_gradient():
    """D8 + D3 on hand-made rows: two detections 0.3 / 0.6 m from one GT car (one-to-one match: the nearer wins, the
    other is a DUPLICATE -> masked), one ghost (y = 0, kept), the box graph only on the matched row."""
    from navsim.agents.para_ssr.ck.online2 import student_rel_match, student_rel_rows
    from navsim.agents.para_ssr.ck.v5 import refine_loss as RL2
    net, cfg0 = _zkd_net(False)
    code = _code([(0.3, 20.0), (0.6, 20.0), (-6.0, 12.0)], vel=[(0.0, 0.0)] * 3).requires_grad_(True)
    det = {"all_bbox_preds": [code], "all_cls_scores": [_cls(k=3)], "det_hidden": torch.zeros(1, 6, 256),
           "traj_preds": torch.zeros(1, 6, 6, 8, 2), "traj_cls_preds": torch.zeros(1, 6, 6)}
    rel = student_rel_rows(det, dict(cfg0.v5, rel_det_grad=True))
    gt = _gt_tracks([(20.0, 0.0, 0.0, 0.0)])                                                # a stopped car 20 m ahead
    tg = {"ref_gt_ok": torch.ones(1, dtype=torch.bool), "ref_obj_kf": gt["kf"], "ref_obj_first": gt["first"],
          "ref_obj_meta": gt["meta"], "ref_obj_n": torch.tensor([1]), "ref_obj_n_kf": gt["n_kf"],
          "ref_sdf": gt["sdf"], "ref_p_pdm": gt["p_pdm"]}
    m = student_rel_match(rel, tg, torch.device("cpu"))
    q_of = rel["q_idx"][0].tolist()
    g_of_q = {int(q): int(m[0, i]) for i, q in enumerate(q_of) if q >= 0}
    assert g_of_q[0] == 0 and g_of_q[1] == RL2.G_DUP and g_of_q[2] == RL2.G_GHOST
    from navsim.agents.para_ssr.ck.v5.student_objects import rows_keep_graph
    relm = rows_keep_graph(rel, (m >= 0) & rel["valid"])
    q = dict(cfg0.v5, lambda_edit=0.25)
    O2, fake = _fake_e2e(q)
    tau = torch.stack([6.0 * torch.arange(1, 9) * 0.5, torch.zeros(8), torch.zeros(8)], -1)[None, None].float()
    rf = {"tau1": tau.expand(1, 2, 8, 3).clone(), "tau0": tau.expand(1, 2, 8, 3).clone()}
    status = torch.zeros(1, 8)
    status[0, 1], status[0, 4] = 1.0, 6.0
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    logs = {}
    tz = torch.arange(LQR.NT, dtype=torch.float64)[::2] * LQR.DT
    inner = O2.CKE2E2._v5_edit_terms(fake, torch.zeros(()), net, rf, relm, m, torch.arange(1), RL2.gt_of(tg,
                                     torch.arange(1)), ego_inputs(status), {"cand_ok": None},
                                     torch.zeros(1, 2, dtype=torch.long), tz, logs)
    assert logs["v5/edit_pos"] > 0 and logs["v5/edit_dup_masked"] == 1.0
    assert logs["v5/edit_n_pairs"] == 2 * 2 * 2                                             # 2 rows (match, ghost) x 2 cols x 2 keys
    inner.backward()
    gq = code.grad[0].abs().sum(-1)
    assert float(gq[0]) > 0 and float(gq[1]) == 0.0 and float(gq[2]) == 0.0


def _write_teacher_run(root: Path, name: str, arm: str) -> Path:
    rd = root / name
    rd.mkdir(parents=True)
    (rd / "config.json").write_text('{"arm": "%s", "trainer": "train_ck2"}' % arm)
    (rd / "ckpt_last.pt").write_bytes(f"ckpt {name}".encode())
    return rd


def test_build_student_loads_and_checks_the_teacher_statistics(tmp_path):
    import json as _json
    from navsim.agents.para_ssr.ck.model import _sha16
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, build_student_ck2
    td, tm = _write_teacher_run(tmp_path, "tD", "T"), _write_teacher_run(tmp_path, "tM", "M")
    rd = _zrec("T", run=str(td), ckpt_sha16=_sha16(td / "ckpt_last.pt"))
    rm = _zrec("M", run=str(tm), ckpt_sha16=_sha16(tm / "ckpt_last.pt"))
    pd_, pm_ = tmp_path / "zd.json", tmp_path / "zm.json"
    pd_.write_text(_json.dumps(rd))
    pm_.write_text(_json.dumps(rm))
    v4 = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
          "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": True, "edits": "off",
          "corrector": False, "v5": {"enabled": True, "relation": "student", "unknown_ev": True, "corrector": True,
                                     "n_obj": 8, "z_adapt": True, "z_stats_det": str(pd_), "z_stats_map": str(pm_)}}

    def build(**over):
        c = CKE2E2Config.from_any({"v4": v4, "teacher_det_run": str(td), "teacher_map_run": str(tm),
                                   "score_prior": "none", **over})
        return build_student_ck2(c, apply_prior=True)
    net = build()
    zs = net.init_report["z_stats"]
    assert zs["det"]["sha16"] == ZA.file_sha16(pd_) and net.z_adapter.fitted()
    assert abs(float(net.z_adapter.z_mu[4]) - ZSTATS_T["H_nc"][0]) < 1e-7
    assert abs(float(net.z_adapter.z_sd[2]) - ZSTATS_M["d_min"][1]) < 1e-7
    # a checkpoint build (apply_prior False) opens no file: the buffers come from the state dict
    c = CKE2E2Config.from_any({"v4": v4, "teacher_det_run": str(td), "teacher_map_run": str(tm)})
    assert not build_student_ck2(c, apply_prior=False).z_adapter.fitted()
    # refusals: the record of another run / checkpoint, swapped files, an unfinished teacher, a missing file
    for bad in (dict(rd, run=str(tm)), dict(rd, ckpt_sha16="f" * 16), rm, dict(rd, unfinished=True)):
        pd_.write_text(_json.dumps(bad))
        with pytest.raises(ValueError, match="z_adapt"):
            build()
    pd_.write_text(_json.dumps(rd))
    (td / "ckpt_ep1.pt").write_bytes((td / "ckpt_last.pt").read_bytes())
    (tm / "ckpt_ep1.pt").write_bytes((tm / "ckpt_last.pt").read_bytes())
    with pytest.raises(ValueError, match="which"):
        build(teacher_which="ep1")
    with pytest.raises(ValueError, match="missing"):
        build(teacher_which="ep2")
    (td / "ckpt_last.pt").write_bytes(b"retrained")                                        # the teacher changed
    with pytest.raises(ValueError, match="sha16"):
        build()


def test_launcher_placeholder_and_z_stats_check(tmp_path):
    import json as _json
    from tools.ck.e2e2 import launch_util2 as LU2
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    sets = [(REPO / f"tools/ck/e2e2/{f}").read_text().splitlines() for f in ("v5_arm.set", "v5kd_arm.set")]
    for arm in (sets[0], sets[0] + sets[1]):
        ck = LU2.apply_sets2(LU2.load_ck2_yaml(), arm)
        q = ck["v4"]["v5"]
        assert q["z_adapt"] is True and q["lambda_edit"] == 0.25
        assert LU2.PLACEHOLDER in q["z_stats_det"] and LU2.PLACEHOLDER in q["z_stats_map"]
        c = CKE2E2Config.from_any(ck)
        assert c.v5_net().z_adapt and c.v5["lambda_edit"] == 0.25
        assert c.v5["lambda_zkd_det"] == 0.5 and c.v5["lambda_zkd_map"] == 0.5
        ck2 = LU2.apply_sets2(ck, ["kd_calib.enabled=false"])
        with pytest.raises(LU2.LaunchError, match="z_stats_det"):
            LU2.placeholder_check(ck2)                                                      # refused while z_adapt
        LU2.placeholder_check(LU2.apply_sets2(ck2, ["v4.v5.z_adapt=false"]))                 # unused paths: exempt
    # z_stats_check: off -> {}; strict refuses a dry run / a mismatch / a missing file (with the fit command)
    td, tm = _write_teacher_run(tmp_path, "tD", "T"), _write_teacher_run(tmp_path, "tM", "M")
    teachers = {"det": {"ckpt_sha16": "a" * 16}, "map": {"ckpt_sha16": "b" * 16}}
    pd_, pm_ = tmp_path / "zd.json", tmp_path / "zm.json"
    v5 = {"enabled": True, "relation": "student", "corrector": True, "z_adapt": True, "z_stats_det": str(pd_),
          "z_stats_map": str(pm_)}
    c = CKE2E2Config.from_any({"v4": {"v5": v5}, "teacher_det_run": str(td), "teacher_map_run": str(tm)})
    assert LU2.z_stats_check(CKE2E2Config.from_any({"v4": {"v5": dict(v5, z_adapt=False)}}), teachers) == {}
    with pytest.raises(LU2.LaunchError, match="fit_z_stats.py"):
        LU2.z_stats_check(c, teachers)
    pd_.write_text(_json.dumps(_zrec("T", run=str(td), ckpt_sha16="a" * 16)))
    pm_.write_text(_json.dumps(_zrec("M", run=str(tm), ckpt_sha16="b" * 16, dry_run=True)))
    with pytest.raises(LU2.LaunchError, match="DRY-RUN"):
        LU2.z_stats_check(c, teachers, "strict")
    rec = LU2.z_stats_check(c, teachers, "smoke")
    assert rec["det"]["sha16"] == ZA.file_sha16(pd_) and set(rec["det"]["channels"]) == set(ZSTATS_T)
    pm_.write_text(_json.dumps(_zrec("M", run=str(tm), ckpt_sha16="c" * 16)))
    with pytest.raises(LU2.LaunchError, match="sha16"):
        LU2.z_stats_check(c, teachers, "smoke")


def test_teacher_checkpoints_still_load_strict_without_adapter():
    from navsim.agents.para_ssr.ck.model import load_ck
    base = Path("/home/external-user/ssd/yongjae_refiner/ck/v5/train")
    runs = [r for r in ("v5T_rel4", "v5M_unk3") if (base / r / "ckpt_last.pt").is_file()]
    if not runs:
        pytest.skip("no v5 teacher checkpoint")
    for r in runs:
        net, cfg = load_ck(base / r, "last", "cpu")
        assert net.z_adapter is None and not net.v5.z_adapt and net.v5.version == "v5_1"
        assert "z_adapt" not in net.v5.to_dict()


def test_fit_z_stats_dry_run_on_the_teachers(tmp_path):
    """CPU dry run of tools/ck/v5/fit_z_stats.py (2 tokens) on the finished v5 teachers: record keys, channels per arm,
    finite statistics, the record checks against its teacher (a dry run: accepted at runtime, refused strict)."""
    from tools.ck.v5 import fit_z_stats as FZ
    base = Path("/home/external-user/ssd/yongjae_refiner/ck/v5/train")
    runs = [(r, a) for r, a in (("v5T_rel4", "T"), ("v5M_unk3", "M")) if (base / r / "done.json").is_file()]
    if not runs:
        pytest.skip("no finished v5 teacher")
    for r, arm in runs:
        out = tmp_path / f"{r}.json"
        rec = FZ.main(["--run", str(base / r), "--tokens", "2", "--out", str(out)])
        assert rec["arm"] == arm and set(rec["channels"]) == set(ZA.TEACHER_CHANNELS[arm]) and rec["dry_run"]
        assert all(math.isfinite(c["mean"]) and c["std"] > 0 for c in rec["channels"].values())
        r2, _ = ZA.load_z_stats(out)
        ZA.check_z_stats(r2, arm, run=base / r, which="last", ckpt_sha16=rec["ckpt_sha16"], z_stride=2)
        with pytest.raises(ValueError, match="DRY-RUN"):
            ZA.check_z_stats(r2, arm, allow_dry_run=False)


def test_infer_check_replays_a_z_adapt_student(tmp_path):
    """eval_e2e2: the extracted student carries the adapter's statistics (strict load round trip) and infer_check
    replays v5_traj through the adapted corrector input exactly."""
    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config, build_student_ck2, lateral_decode, student_rel_rows
    from tools.ck.e2e import eval_e2e as E1
    from tools.ck.e2e2 import eval_e2e2 as E2
    v4 = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
          "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": True, "edits": "off",
          "corrector": False, "v5": {"enabled": True, "relation": "student", "unknown_ev": True, "corrector": True,
                                     "n_obj": 8, "rel_modes": "cv+fut", "top_feats": "v2", "z_adapt": True,
                                     "z_stats_det": "/x/d.json", "z_stats_map": "/x/m.json", "lambda_edit": 0.25}}
    cfg = CKE2E2Config.from_any({"v4": v4})
    torch.manual_seed(0)
    net = build_student_ck2(cfg, apply_prior=False)
    net.z_adapter.set_stats(_zrec("T"), _zrec("M"))
    with torch.no_grad():
        for p in list(net.corrector_v5.mu.parameters()) + list(net.z_adapter.parameters()):
            p.add_(torch.randn_like(p) * 0.02)
    R = tmp_path / "e2e_v5"
    ckpt = R / "train" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"state_dict": {E1.STUDENT_PREFIX + k: v for k, v in net.state_dict().items()}}, ckpt)
    P = E1.Paths(R, "navtrain_val")
    E2.extract_ck2(ckpt, P.ck_run, "e2e_v5", {"enabled": True, "seed": 0, "v4": v4})
    net2, _ = load_ck(P.ck_run, "last", "cpu")
    assert net2.z_adapter is not None and net2.z_adapter.fitted()
    assert torch.equal(net2.z_adapter.z_mu, net.z_adapter.z_mu)
    cand, par, V, ego, bev, det, status = _pool_inputs()
    n = cand.shape[0]
    P.packed.mkdir(parents=True, exist_ok=True)
    for k, v in (("cand96", V["traj96"]), ("status", status), ("valid96", V["valid96"]),
                 ("v2_final", torch.linspace(1, 0, 16).expand(n, 16)), ("v2_im", torch.full((n, 16), 1 / 16))):
        np.save(P.packed / f"{k}.npy", v.numpy())
    rows = np.arange(n)
    P.infer.mkdir(parents=True, exist_ok=True)
    np.savez(P.infer / "check_bev.npz", rows=rows, bev=bev.numpy(), det_code=det["all_bbox_preds"][0].numpy(),
             det_cls=det["all_cls_scores"][0].numpy(), det_hidden=det["det_hidden"].numpy(),
             det_traj=det["traj_preds"].numpy(), det_traj_cls=det["traj_cls_preds"].numpy())
    with torch.no_grad():
        feat, mem = net.trunk.scene(bev, n)
        pref = net.scene_pref(mem, ego)
        o = net.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=True, rel_objs=student_rel_rows(det, cfg.v5))
        src = torch.tensor([3, 50])
        rf = net.refine_drafts(o, cand.float(), src[:, None], ego)
        lat = lateral_decode(cand, o["w_lat"], ego[0], 0.0)
    for k, v in (("score_logit", o["score_logit"]), ("w_lat", o["w_lat"]), ("e_lat", lat["e_lat"][..., 2:]),
                 ("lat_traj", lat["traj"]), ("v5_traj", rf["tau1"]), ("v5_mu", rf["mu"][:, 0])):
        np.save(P.infer / f"{k}.npy", v.float().numpy())
    np.save(P.infer / "v5_src.npy", src.numpy().astype(np.int16))
    res = E2.stage_infer_check(P, SimpleNamespace(gpus=""))
    assert res["pass"] and res["max_abs_err"]["v5_traj"] < 1e-5 and res["max_abs_err"]["v5_mu"] < 1e-5


def test_zadapter_floors_the_hazard_channels_sigma():
    """H_nc / H_ttc are mostly-zero probabilities: set_stats floors their sigma at SD_FLOOR_CH (0.25), the b / d_min
    channels keep the record's std (main session decision 2026-10-20, workflow review: 55-270x Z-KD gradient on H)."""
    import torch
    from navsim.agents.para_ssr.ck.v5 import zadapt as ZA
    za = ZA.ZAdapter(6, 32, "all")
    ch = lambda m, s: {"mean": m, "std": s, "std_raw": s, "n": 5000}                       # noqa: E731
    det = {"channels": {"b_nc": ch(-0.9, 0.44), "b_ttc": ch(-0.65, 0.40), "H_nc": ch(0.0075, 0.061),
                        "H_ttc": ch(0.018, 0.101)}}
    mp = {"channels": {"d_min": ch(0.56, 0.38)}}
    orig = ZA.check_z_stats
    ZA.check_z_stats = lambda *a, **k: None
    try:
        out = za.set_stats(det, mp)
    finally:
        ZA.check_z_stats = orig
    assert out["H_nc"][1] == 0.25 and out["H_ttc"][1] == 0.25
    assert abs(out["b_nc"][1] - 0.44) < 1e-9 and abs(out["d_min"][1] - 0.38) < 1e-9
    z = torch.tensor([[-0.9, -0.65, 0.56, 0.0, 1.0, 1.0]])
    zn = za.normalise(z)
    assert abs(float(zn[0, 4]) - (1.0 - 0.0075) / 0.25) < 1e-5 and float(zn[0, 3]) == 0.0


def test_report102_n1_relation_bce_keeps_the_gradient_of_a_deeply_missed_positive():
    """report 102 N1: a matched cut-in whose lateral velocity the student missed (r ~ 1e-7, GT y = 1): the relation BCE
    (L_rel / L_rel,cv / L_edit) must give a finite loss and a non-zero, correcting gradient on the box velocity code
    (the probability clamp at 1e-4 zeroed it); the training term _v5_rel_terms uses the reader's log r."""
    import math
    from types import SimpleNamespace
    import torch
    from navsim.agents.para_ssr.ck.v5 import refine_loss as R
    from navsim.agents.para_ssr.ck.v5.student_objects import student_object_rows
    from navsim.agents.para_ssr.ck.v5.relation import RelationReader
    from navsim.agents.para_ssr.ck.v5.config import V5Cfg
    from navsim.agents.para_ssr.ck.v4 import lqr as L
    from navsim.agents.para_ssr.refiner import gt_future as G
    from navsim.agents.para_ssr.ck.online2 import CKE2E2
    torch.manual_seed(23)
    c = torch.zeros(1, 1, 10)
    c[0, 0, :2] = torch.tensor([-8., 14.])
    c[0, 0, 2], c[0, 0, 3], c[0, 0, 5] = math.log(1.9), math.log(4.5), math.log(1.6)
    h = math.atan2(-2., 3.)
    yaw = -h - math.pi
    c[0, 0, 6:8] = torch.tensor([math.sin(yaw), math.cos(yaw)])
    c[0, 0, 8:10] = torch.tensor([0., 3.])
    c.requires_grad_(True)
    traj = torch.zeros(1, 1, 6, 8, 2)
    traj[..., 1] = 1.5
    O = student_object_rows(c, torch.tensor([[[4., -6, -6, -6, -6, -6, -6]]]), torch.zeros(1, 1, 256), 96,
                            traj=traj, traj_cls=torch.zeros(1, 1, 6))
    k = torch.arange(G.N_KF) * G.KF_DT
    kf = torch.zeros(1, 1, G.N_KF, 6)
    first = torch.zeros(1, 1, 6)
    meta = torch.zeros(1, 1, 5, dtype=torch.long)
    kf[0, 0, :, G.KF_X], kf[0, 0, :, G.KF_Y], kf[0, 0, :, G.KF_H] = 14 + 3 * k, 8 - 2 * k, h
    kf[0, 0, :, G.KF_VX], kf[0, 0, :, G.KF_VY], kf[0, 0, :, G.KF_PRESENT] = 3, -2, 1
    first[0, 0, G.FI_L], first[0, 0, G.FI_W], first[0, 0, G.FI_VX], first[0, 0, G.FI_VY] = 4.5, 1.9, 3, -2
    meta[0, 0, G.MT_AGENT], meta[0, 0, G.MT_LAST] = 1, G.N_KF - 1
    gt = dict(kf=kf, first=first, meta=meta, valid=torch.ones(1, 1, dtype=torch.bool), n_kf=torch.tensor([G.N_KF]))
    kt = torch.arange(1, 9) * .5
    draft = torch.stack([6 * kt, torch.zeros_like(kt), torch.zeros_like(kt)], -1)[None]
    full_states = L.simulate(draft, torch.tensor([[6., 0.]]), dtype=torch.float32)
    states = full_states[:, ::2]
    t = torch.arange(L.NT, dtype=torch.float64)[::2] * L.DT
    g = R.match_t0(O['c0'], O['valid'], O['group'] < 2, gt)
    y, known = R.rel_object_targets(states[None], t, gt, g)
    mask = R.rel_object_mask(O['valid'], g, known, y)
    assert bool(mask.any()) and float(y[mask].max()) == 1.0
    rr = RelationReader(V5Cfg(relation='student', rel_ca=False, rel_modes='cv+fut', top_feats='v2'))
    out = rr(O, states, t, torch.zeros(1, dtype=torch.long))
    assert float(out['r_nc'].max()) < 1e-4                                       # the missed positive
    torch.testing.assert_close(torch.exp(out['lr_nc']), out['r_nc'], rtol=1e-3, atol=1e-12)
    loss = R.rel_object_bce(out['r_nc'][None], out['r_ttc'][None], y, mask, out['lr_nc'][None], out['lr_ttc'][None])
    gc, ge = torch.autograd.grad(loss, (c, rr.log_eps), retain_graph=True)
    assert math.isfinite(float(loss)) and float(loss) > 9.21                     # beyond the old clamp's -log(1e-4)
    assert float(gc[0, 0, 8].abs()) > 0.1 and float(ge.abs()) > 0.1              # lateral velocity code + eps move
    # the old clamped path (no log r) is unchanged and still zero here: the training term must use the log path
    lc = R.rel_object_bce(out['r_nc'][None], out['r_ttc'][None], y, mask)
    assert float(torch.autograd.grad(lc, c, retain_graph=True)[0].abs().max()) == 0.0
    fake = SimpleNamespace(cfg=SimpleNamespace(v5=dict(lambda_rel=.5, lambda_rel_cv=.25, rel_max_parents=8)))
    hk = dict(rel_r_nc=out['r_nc'][None], rel_r_ttc=out['r_ttc'][None], rel_lr_nc=out['lr_nc'][None],
              rel_lr_ttc=out['lr_ttc'][None], exec_states=full_states[None],
              parent_index=torch.zeros(1, 1, dtype=torch.long), is_self=torch.ones(1, 1, dtype=torch.bool))
    tl = CKE2E2._v5_rel_terms(fake, torch.zeros(()), hk, O, g, torch.tensor([0]), gt, t, 2, {})
    tg = torch.autograd.grad(tl, c)[0]
    assert math.isfinite(float(tl)) and float(tg[0, 0, 8].abs()) > 0.05
    # negatives: a safe object far away gives a finite, ~0 loss
    yn = torch.zeros_like(y)
    ln = R.rel_object_bce(out['r_nc'][None], out['r_ttc'][None], yn, mask, out['lr_nc'][None], out['lr_ttc'][None])
    assert math.isfinite(float(ln)) and float(ln) < 1e-3
