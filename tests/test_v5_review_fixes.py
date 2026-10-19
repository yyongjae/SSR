"""refiner v5 regression tests for the review-98 findings (SSR/report/98_refiner_v5_code_review.md R1-R4)."""
import math

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.ck.v4 import lqr as LQR


def _straight(v: float, n: int = 1) -> torch.Tensor:
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    p = torch.stack([v * t, torch.zeros_like(t), torch.zeros_like(t)], -1)
    return p[None].expand(n, 8, 3).clone()


def test_r1_corrector_mu0_equals_mode_a_lateral_correction():
    """mu = 0 must reproduce today's laterally corrected draft (mode A, ck/correct.py) for real anchor shapes."""
    from navsim.agents.para_ssr.ck import anchor_sampler as AS
    from navsim.agents.para_ssr.ck.correct import correct
    from navsim.agents.para_ssr.ck.v5.corrector import decode_grad
    anc = torch.from_numpy(np.asarray(AS.load_anchors(), np.float32))
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(anc.shape[0], generator=g)[:48]
    src = anc[idx].reshape(3, 16, 8, 3)
    w = torch.randn(3, 16, 6, generator=g) * 0.8
    v0 = torch.tensor([3.0, 8.0, 14.0])
    ref = correct(src, torch.zeros_like(w), w, v0, 0.0)["traj"]
    tau1 = decode_grad(src, torch.zeros_like(w), w, v0)
    assert float((tau1 - ref).abs().max()) < 1e-4


def test_r2_unknown_beyond_bev_window():
    from navsim.agents.para_ssr.ck.v5.zread import bev_support
    st = LQR.simulate(_straight(10.0), torch.tensor([[10.0, 0.0]]), dtype=torch.float32)      # x reaches ~40 m
    sup = bev_support(st)[0]
    from navsim.agents.para_ssr.refiner.sdf import ego_corners
    from navsim.agents.para_ssr.refiner.corridor import in_s_grid
    pts = torch.cat([st[0, :, None, :2], ego_corners(st[0, :, :3])], 1)
    assert bool((sup == in_s_grid(pts).all(-1)).all())
    assert not bool(sup[-1]) and bool(sup[20])


def _bicycle_gt():
    """ego straight 10 m/s; bicycle starts at (6, 4) heading -pi/2, velocity (0, -5) m/s, 2 x 0.6 m (review 98 R3)."""
    from navsim.agents.para_ssr.refiner import gt_future as GF
    k = torch.arange(GF.N_KF, dtype=torch.float32)
    t = k * GF.KF_DT
    kf = torch.zeros(1, 1, GF.N_KF, 6)
    kf[0, 0, :, GF.KF_X] = 6.0
    kf[0, 0, :, GF.KF_Y] = 4.0 - 5.0 * t
    kf[0, 0, :, GF.KF_H] = -math.pi / 2
    kf[0, 0, :, GF.KF_VY] = -5.0
    kf[0, 0, :, GF.KF_PRESENT] = 1.0
    first = torch.tensor([[[2.0, 0.6, -math.pi / 2, 0.0, -5.0, 0.0]]])
    meta = torch.tensor([[[2, 1, 0, GF.N_KF - 1, 0]]], dtype=torch.long)
    return {"kf": kf, "first": first, "meta": meta, "valid": torch.ones(1, 1, dtype=torch.bool),
            "n_kf": torch.tensor([GF.N_KF]), "sdf": torch.zeros(1, 320, 256), "p_pdm": torch.tensor([40.0])}


def test_r3_ttc_counts_object_ahead_of_the_current_ego():
    """an object ahead of the CURRENT ego that the 0.6 s projection overlaps is a TTC hit (official rule), even though
    its centre lies behind the projected box."""
    from navsim.agents.para_ssr.ck.v5 import refine_loss as RL
    gt = _bicycle_gt()
    st = LQR.simulate(_straight(10.0), torch.tensor([[10.0, 0.0]]), dtype=torch.float32)[None]   # [1, 1, 41, 6]
    t = torch.arange(LQR.NT, dtype=torch.float64) * LQR.DT
    sep, cnt = RL.object_sep(st, t, gt, (0.0, 0.3, 0.6, 0.9))
    assert float(sep[0, 0, 0, 2, 0]) < 0                     # state 0, delta 0.6: the projected box overlaps
    assert bool(cnt[0, 0, 0, 2, 0])                          # ... and the object counts (ahead of the current ego)
    y, known = RL.gt_state_block(st, t, gt)
    assert float(y[0, 0, 0, 1]) == 1.0 and bool(known[0, 0, 0])


def test_r4_reliability_compares_t0_with_t0():
    """a correctly detected decelerating object (t0 velocity exact) is reliable."""
    from navsim.agents.para_ssr.ck.v5.student_objects import student_reliability_targets
    O = {"c0": torch.tensor([[[20.0, 0.0]]]), "v0": torch.tensor([[[5.0, 0.0]]]),
         "valid": torch.ones(1, 1, dtype=torch.bool)}
    y, m = student_reliability_targets(O, torch.tensor([[[20.3, 0.1]]]), torch.tensor([[[5.0, 0.0]]]),
                                       torch.ones(1, 1, dtype=torch.bool))
    assert float(y[0, 0]) == 1.0 and bool(m[0, 0])
    y2, _ = student_reliability_targets(O, torch.tensor([[[20.3, 0.1]]]), torch.tensor([[[2.0, 0.0]]]),
                                        torch.ones(1, 1, dtype=torch.bool))
    assert float(y2[0, 0]) == 0.0


def test_r3_relation_not_behind_uses_current_ego():
    """the relation channel's TTC overlap of the same bicycle is not suppressed by the projected-box not-behind test."""
    from navsim.agents.para_ssr.ck.v5.config import V5Cfg
    from navsim.agents.para_ssr.ck.v5.relation import RelationReader
    rr = RelationReader(V5Cfg(relation="student", rel_ca=False))
    st = LQR.simulate(_straight(10.0), torch.tensor([[10.0, 0.0]]), dtype=torch.float32)       # [1, 41, 6]
    from navsim.agents.para_ssr.ck.v5.student_objects import N_FEAT_S
    n = rr.n_obj
    O = {"c0": torch.zeros(1, n, 2), "v0": torch.zeros(1, n, 2), "h0": torch.zeros(1, n), "L": torch.ones(1, n),
         "W": torch.ones(1, n), "sp0": torch.zeros(1, n), "u0": torch.zeros(1, n, 2), "acc": torch.zeros(1, n),
         "has": torch.zeros(1, n, dtype=torch.bool), "group": torch.zeros(1, n, dtype=torch.long),
         "pi_s": torch.zeros(1, n), "valid": torch.zeros(1, n, dtype=torch.bool), "feats": torch.zeros(1, n, N_FEAT_S)}
    O["c0"][0, 0] = torch.tensor([6.0, 4.0])
    O["v0"][0, 0] = torch.tensor([0.0, -5.0])
    O["sp0"][0, 0] = 5.0
    O["u0"][0, 0] = torch.tensor([0.0, -1.0])
    O["h0"][0, 0] = -math.pi / 2
    O["L"][0, 0], O["W"][0, 0] = 2.0, 0.6
    O["group"][0, 0] = 1
    O["valid"][0, 0] = True
    t = torch.arange(LQR.NT, dtype=torch.float64) * LQR.DT
    out = rr(O, st[:, ::2], t[::2], torch.zeros(1, dtype=torch.long))
    assert float(out["r_ttc"][0, 0]) > 0.5
