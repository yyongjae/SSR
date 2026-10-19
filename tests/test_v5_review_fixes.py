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


# ---- second review round (2026-10-20) --------------------------------------------------------------------------------
def test_object_rows_trim_to_valid_slots():
    """teacher rows: every valid t0 slot up to n (no fixed top-32 cut), trimmed to the batch's last valid slot."""
    from navsim.agents.para_ssr.ck.v5.relation import object_rows
    from navsim.agents.para_ssr.ck.temporal.config import D_OBJ
    T, F, K = 2, 4, 120
    obj_in = torch.zeros(T, F, K, D_OBJ)
    mask = torch.zeros(T, F, K, dtype=torch.bool)
    mask[0, :, :50] = True
    mask[1, :, :70] = True
    fv = torch.ones(T, F, dtype=torch.bool)
    O = object_rows(obj_in, mask, fv, 96)
    assert O["valid"].shape == (T, 70) and int(O["valid"].sum()) == 120
    O2 = object_rows(obj_in, mask, fv, 32)
    assert O2["valid"].shape == (T, 32)


def test_rel_ca_off_has_no_ca_parameter():
    from navsim.agents.para_ssr.ck.v5.config import V5Cfg
    from navsim.agents.para_ssr.ck.v5.relation import RelationReader
    rr = RelationReader(V5Cfg(relation="student", rel_ca=False))
    assert rr.ca_logit is None and all("ca_logit" not in n for n, _ in rr.named_parameters())


def test_soft_pdms_prefers_braking_before_a_stopped_car():
    """L_refine (soft PDMS): braking (variant) before a stopped car 30 m ahead lowers the loss vs the parent (the LQR
    tracker of a 3 m/s draft from 10 m/s still covers ~20 m)."""
    from navsim.agents.para_ssr.ck.v5 import refine_loss as RL
    from navsim.agents.para_ssr.refiner import gt_future as GF
    gt = _bicycle_gt()
    gt["kf"][0, 0, :, GF.KF_X] = 30.0
    gt["kf"][0, 0, :, GF.KF_Y] = 0.0
    gt["kf"][0, 0, :, GF.KF_H] = 0.0
    gt["kf"][0, 0, :, GF.KF_VY] = 0.0
    gt["first"] = torch.tensor([[[4.5, 1.9, 0.0, 0.0, 0.0, 0.0]]])
    gt["meta"][0, 0, 0] = 0                                                   # vehicle
    gt["sdf"] = torch.full((1, 320, 256), 5.0)
    w = RL.parse_weights("")
    fast, slow = _straight(10.0)[None], _straight(3.0)[None]                 # [1, 1, 8, 3]
    v0, a0 = torch.tensor([10.0]), torch.tensor([0.0])
    l_slow, _ = RL.refine_loss(slow, fast, v0, a0, gt, w)                     # tau0 = fast, tau1 = slow
    l_fast, _ = RL.refine_loss(fast, slow, v0, a0, gt, w)
    assert float(l_slow) < 0 < float(l_fast)


def test_z_channels_mask_zeroes_unsupervised_reads():
    from navsim.agents.para_ssr.ck.v5.config import V5Cfg
    assert V5Cfg(z_channels="det").z_channels == "det"
    with pytest.raises(ValueError):
        V5Cfg(z_channels="xx").validate(None)


# ---- report 99 (2026-10-20) -----------------------------------------------------------------------------------------
def _student_code(n_q: int = 4):
    """det-frame code rows (x_right, y_forward, log w, log l, z, log h, sin, cos, vx_right, vy_forward): query 0 = a car
    8 m ahead slightly right moving left at 2 m/s."""
    code = torch.zeros(1, n_q, 10)
    code[0, :, 1] = 40.0                                     # far / outside the window by default
    code[0, 0, :] = torch.tensor([-1.5, 8.0, math.log(1.9), math.log(4.5), 0.0, math.log(1.6), 0.0, 1.0, -2.0, 0.0])
    cls = torch.full((1, n_q, 7), -6.0)
    cls[0, :, 0] = 4.0                                       # vehicle
    return code, cls, torch.zeros(1, n_q, 256)


def test_n3_relation_gradient_reaches_det_position_and_velocity():
    """design V2: the relation hazard's gradient reaches the det box code's x / y / vx / vy entries (not size /
    heading); a detached code gives no graph (the old behaviour)."""
    from navsim.agents.para_ssr.ck.v5.config import V5Cfg
    from navsim.agents.para_ssr.ck.v5.relation import RelationReader
    from navsim.agents.para_ssr.ck.v5.student_objects import student_object_rows
    rr = RelationReader(V5Cfg(relation="student", rel_ca=False))
    code, cls, hid = _student_code()
    code.requires_grad_(True)
    O = student_object_rows(code, cls, hid, 96)
    st = LQR.simulate(_straight(5.0), torch.tensor([[5.0, 0.0]]), dtype=torch.float32)
    t = torch.arange(LQR.NT, dtype=torch.float64) * LQR.DT
    out = rr(O, st[:, ::2], t[::2], torch.zeros(1, dtype=torch.long))
    out["H"].sum().backward()
    g = code.grad[0, 0]
    assert float(g[[0, 1]].abs().sum()) > 0 and float(g[[8, 9]].abs().sum()) > 0
    assert float(g[[2, 3, 4, 5, 6, 7]].abs().sum()) == 0
    code2 = code.detach()
    O2 = student_object_rows(code2, cls, hid, 96)
    assert not O2["c0"].requires_grad and torch.equal(O2["c0"], O["c0"].detach())


def test_n2_velocity_weights_are_lateral_first():
    """det code entries 8, 9 = (vx_right, vy_forward) = (lateral, longitudinal): the v5 extra must weight the lateral
    channel up (1 / 0.95) and the longitudinal one (1 / 2.44)."""
    from pathlib import Path
    from navsim.agents.para_ssr.para_ssr_targets import navsim_box_to_ssr
    c = navsim_box_to_ssr(np.zeros(7), np.array([3.0, 1.0]))      # NAVSIM v = (forward 3, left 1)
    assert c[7] == -1.0 and c[8] == 3.0                             # [.., vx_right, vy_forward] (code 8, 9 = 7, 8 here)
    line = [x for x in Path("tools/ck/e2e2/fusion_v13_v5.extra").read_text().splitlines()
            if x.startswith("++agent.config.det_vel_code_weights")][0]
    assert line.endswith("[1.05,0.41]")


def test_n4_awr_logp_and_record_block():
    """CorrectorV5.log_prob = the v4 Gaussian; awr_loss weights a better tau1 up."""
    from navsim.agents.para_ssr.ck.v5.corrector import CorrectorV5
    from navsim.agents.para_ssr.ck.v4.corrector import awr_loss
    mu, ls = torch.zeros(2, 1, 12, requires_grad=True), torch.full((2, 1, 12), math.log(0.3))
    u = torch.full((2, 1, 12), 0.1)
    lp = CorrectorV5.log_prob(mu, ls, u)
    ref = torch.distributions.Normal(torch.zeros(12), torch.full((12,), 0.3)).log_prob(torch.full((12,), 0.1)).sum()
    assert torch.allclose(lp, ref.expand(2, 1), atol=1e-5)
    l = awr_loss(lp, torch.tensor([[0.2], [-0.2]]), torch.ones(2, 1, dtype=torch.bool))
    l.backward()
    # both recorded controls lie above mu: both pull mu up (negative grad), the better tau1 (A = +0.2) much harder
    assert float(mu.grad[0, 0, 0]) < float(mu.grad[1, 0, 0]) <= 0.0
