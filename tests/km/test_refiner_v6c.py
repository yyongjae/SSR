"""refiner v6c (report 124): per-step labels, nets, losses, gradient contract.  CPU, synthetic tables only."""
import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.modules import exact_score as ES
from navsim.agents.para_ssr.modules import exact_verdict as EV
from navsim.agents.para_ssr.modules.torch_tracker import track
from navsim.agents.para_ssr.refiner_v6c import labels as L


def _rect_segs(x0, x1, y0, y1, pid=0):
    p = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return np.array([[*p[j], *p[(j + 1) % 4], pid] for j in range(4)], np.float32)


def _scene(lead_x=None, lead_y=0.0, road_half=6.0, moving_lead=0.0):
    """straight road x -20..120, |y| <= road_half; optional lead car (stopped or moving along +x) -> pack_full tables"""
    rows = []
    if lead_x is not None:
        for t in range(1, 50):
            x = lead_x + moving_lead * 0.1 * t
            rows.append([t, x, lead_y, 0.0, 2.4, 1.0, 1.0, 7, float(moving_lead == 0.0)])
    tab = ES._empty(ES.V1_TABLES)
    tab["agent_rows"] = np.array(rows, np.float32).reshape(-1, 9)
    tab["drv_seg"] = _rect_segs(-20, 120, -road_half, road_half)
    tab["lane_seg"] = _rect_segs(-20, 120, -road_half, road_half)
    tab["cl"] = np.stack([np.linspace(-20, 120, 141), np.zeros(141), np.linspace(0, 140, 141)], -1).astype(np.float32)
    tab["pdm"] = np.zeros((41, 9), np.float32)
    out = ES.pack_full(tab, "ex1_")
    out["ex1_valid"] = torch.tensor(1.0)
    return {k: v[None] for k, v in out.items()}


def _straight(v=10.0, y_end=0.0, n=1):
    tt = torch.arange(1, 9, dtype=torch.float64) * 0.5
    poses = torch.stack([v * tt, y_end * tt / 4.0, torch.zeros_like(tt)], -1)
    init = torch.tensor([v, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    st = track(poses[None].expand(n, -1, -1).contiguous(), init[None].expand(n, -1).contiguous(), full=True)
    return st.float()[None]                                                                   # [1, n, 41, 7]


def test_step_labels_lead_car_and_offroad():
    st = _straight(10.0)
    lab = L.exact_step_masks(st, _scene(lead_x=25.0))
    # ego box front reaches the stopped car (rear at 25 - 2.4) at x_rear ~ 18.3 m -> t ~ 1.8 s
    first = int(lab["ovl"][0, 0].nonzero()[0])
    assert 15 <= first <= 22, first
    # TTC fires earlier than the overlap (0.9 s look-ahead)
    assert int(lab["ttc"][0, 0].nonzero()[0]) < first
    # aggregated official verdict agrees: NC fails, TTC fails
    nc, ttc = EV.exact_nc_ttc(st, {k.replace("ex1_", "ex_"): v for k, v in _scene(lead_x=25.0).items()})
    assert float(nc[0, 0]) == 0.0 and float(ttc[0, 0]) == 0.0
    # the signed gap shrinks through 0 (negative = penetration), the object sits ahead on the centre line
    g = lab["gap"][0, 0]
    assert float(g[0]) > 5 and float(g[first]) < 0.0
    assert lab["obj"][0, 0, :first].all()
    pos = lab["pos"][0, 0, 5]
    assert pos[0] > 5 and abs(float(pos[1])) < 0.5
    assert lab["vapp"][0, 0, 5] > 8                                                         # closing at ~ 10 m/s
    assert lab["offroad"][0, 0].sum() == 0 and lab["valid"][0, 0].all()


def test_step_labels_offroad_and_no_object():
    st = _straight(10.0, y_end=12.0)                                                         # drifts left out of |y| <= 6
    lab = L.exact_step_masks(st, _scene(lead_x=None, road_half=6.0))
    off = lab["offroad"][0, 0]
    assert off[0] == 0 and off[-1] == 1
    assert lab["obj"][0, 0].sum() == 0 and lab["ovl"][0, 0].sum() == 0
    # no candidate object -> fixed values (no mask), off-road labels on every step of a present table
    assert (lab["gap"][0, 0] == L.NO_OBJ_M).all() and (lab["pos"][0, 0, :, 0] == L.NO_OBJ_M).all()
    assert (lab["pos"][0, 0, :, 1] == 0).all() and (lab["vapp"][0, 0] == 0).all()
    assert lab["valid_off"][0, 0].all()
    dac = EV.exact_dac(st, _scene(road_half=6.0)["ex1_seg"], _scene(road_half=6.0)["ex1_smask"])
    assert float(dac[0, 0]) == 0.0


def test_step_labels_lateral_position_sign():
    st = _straight(10.0)
    lab = L.exact_step_masks(st, _scene(lead_x=30.0, lead_y=2.5))                          # car ahead-left
    has = lab["obj"][0, 0] > 0
    assert has.any()
    assert (lab["pos"][0, 0, has][:, 1] > 0).all()                                          # y left positive


def test_step_labels_candidates_exclude_followers_and_out_of_window_tracks():
    st = _straight(10.0)
    # a car behind the ego at x = -12 m, driving faster (catches up): behind -> not a TTC-direction candidate
    lab = L.exact_step_masks(st, _scene(lead_x=-12.0, moving_lead=14.0))
    near = lab["obj"][0, 0, :10]
    assert near.sum() == 0
    # a lead first observed beyond the 64 m window: not a candidate even when it comes closer later
    lab2 = L.exact_step_masks(st, _scene(lead_x=70.0, moving_lead=-5.0))
    assert lab2["obj"][0, 0].sum() == 0
    lab3 = L.exact_step_masks(st, _scene(lead_x=70.0, moving_lead=-5.0), window=False)
    assert lab3["obj"][0, 0].sum() > 0


def test_stride2_keeps_odd_step_positives():
    v = torch.zeros(1, 1, 41)
    v[..., 7] = 1.0                                                                          # 0.7 s only
    s = L.stride2({"ovl": v, "gap": torch.arange(41.0).expand(1, 1, 41), "pos": torch.zeros(1, 1, 41, 2)})
    assert s["ovl"].shape[-1] == 21 and float(s["ovl"][0, 0, 4]) == 1.0                      # slot 4 = (0.6, 0.8]
    assert float(s["gap"][0, 0, 3]) == 6.0 and s["pos"].shape == (1, 1, 21, 2)


# ------------------------------------------------------------------------------------------------- nets / losses
from navsim.agents.para_ssr.refiner_v6c import losses as LS
from navsim.agents.para_ssr.refiner_v6c import nets as N


def _tau(v=10.0, n=2):
    tt = torch.arange(1, 9, dtype=torch.float32) * 0.5
    tau = torch.stack([v * tt, 0.02 * tt ** 2, 0.004 * tt], -1)[None].repeat(n, 1, 1)
    return tau, torch.full((n,), v)


def test_decoder_identity_at_zero_controls():
    tau, v0 = _tau()
    out = N.decode_v6(tau, torch.zeros(2, 12), v0)
    assert torch.equal(out["traj"], tau)                                                       # bitwise
    c = N.Corrector(hid=64, h_dim=64)
    E, M = torch.randn(2, 21, 12), torch.ones(2, 21, 5)
    S = track(tau.double(), torch.cat([torch.full((2, 1), 10.0), torch.zeros(2, 6)], -1).double(), full=True).float()
    ctrl = c(E, M, S, torch.zeros(2, 4), torch.eye(4)[:2])
    assert torch.equal(ctrl, torch.zeros_like(ctrl))                                           # zero-init last layer
    assert torch.equal(N.decode_v6(tau, ctrl, v0)["traj"], tau)


def test_decoder_bezier_heading_identity_and_consistency():
    """cf builds tau_base = anchors + (bezier_xyyaw(xy, anchor_h) - anchors) in float32; the v6c heading is
    h_base + wrap(bezier(xy_v6) - bezier(xy_base)): bitwise h_base at Delta = 0, the Bezier tangent change otherwise"""
    from navsim.agents.para_ssr.modules.kinematics import bezier_xyyaw
    anc = torch.tensor(np.load("resources/km_corridor/trajectory_anchors_256.npy"), dtype=torch.float32)[[10, 200]]
    xy = anc[..., :2] + 0.3 * torch.randn(2, 8, 2)
    tau = anc + (bezier_xyyaw(xy, anc[..., 2]) - anc)                                        # cf's arithmetic
    v0 = (tau[:, 0, :2].norm(dim=-1) / 0.5)
    out = N.decode_v6(tau, torch.zeros(2, 12), v0, anchor_h=anc[..., 2])
    assert torch.equal(out["traj"], tau)
    ctrl = torch.zeros(2, 12)
    ctrl[:, 6:] = 1.5
    out = N.decode_v6(tau, ctrl, v0, anchor_h=anc[..., 2])
    ref = bezier_xyyaw(out["traj"][..., :2], anc[..., 2])[..., 2]
    d = torch.atan2(torch.sin(out["traj"][..., 2] - ref), torch.cos(out["traj"][..., 2] - ref))
    d0 = torch.atan2(torch.sin(tau[..., 2] - bezier_xyyaw(tau[..., :2], anc[..., 2])[..., 2]),
                     torch.cos(tau[..., 2] - bezier_xyyaw(tau[..., :2], anc[..., 2])[..., 2]))
    assert torch.allclose(d, d0, atol=1e-5)                                                  # same offset as tau_base


def test_decoder_decel_only_and_lateral_bound():
    tau, v0 = _tau()
    ctrl = torch.cat([-3 * torch.ones(2, 6), 3 * torch.ones(2, 6)], -1)
    out = N.decode_v6(tau, ctrl, v0)
    assert (out["traj"][:, :, 0] <= tau[:, :, 0] + 1e-4).all()                                  # never ahead of tau_base
    assert (out["d"].abs() <= 0.5 + 1e-4).all()


def _occ_dist_with_box(x0, x1, y0, y1, planes=49):
    od = torch.full((1, planes, 200, 200), 8.0)
    xs = (torch.arange(200) + 0.5) * 0.32                                                       # row = forward
    ys = 32.0 - (torch.arange(200) + 0.5) * 0.32                                                # col = SSR x right -> y left
    inside = ((xs[:, None] >= x0) & (xs[:, None] <= x1) & (ys[None] >= y0) & (ys[None] <= y1))
    od[:, :, inside] = -0.5
    return od


def test_gt_guard_zero_at_base_and_fires_when_edit_hits_object():
    tau, v0 = _tau(n=1)
    init = torch.cat([torch.full((1, 1), 10.0), torch.zeros(1, 6)], -1)
    Sb = track(tau, init, full=True)
    tg = {"corridor_occ_dist": _occ_dist_with_box(10, 35, 1.6, 3.0), "corridor_sdf": torch.full((1, 200, 200), 5.0),
          "corridor_route_s": torch.zeros(1, 100, 100), "corridor_valid": torch.ones(1)}
    w = {"gt": 1.0, "road": 1.0, "comfort": 1.0, "ec": 0.1, "prog": 0.0}
    l0, _ = LS.corr_loss(Sb.clone().requires_grad_(), Sb, tg, None, torch.zeros(1), w)
    assert float(l0) == 0.0
    # a lateral edit to the left drives into the box at y 1..3 m -> GT guard > 0 with a gradient on the controls
    ctrl = torch.zeros(1, 12)
    ctrl[0, 6:] = 2.0
    ctrl.requires_grad_()
    tv = N.decode_v6(tau, ctrl, v0)["traj"]
    Sg = track(tv, init, full=True)
    l1, logs = LS.corr_loss(Sg, Sb, tg, None, torch.zeros(1), w)
    assert float(logs["v6c/corr_gt"]) > 0
    g, = torch.autograd.grad(l1, ctrl)
    assert torch.isfinite(g).all() and g[0, 6:].abs().sum() > 0


def test_safe_base_gives_zero_corrector_gradient():
    tau, v0 = _tau(n=1)
    init = torch.cat([torch.full((1, 1), 10.0), torch.zeros(1, 6)], -1)
    Sb = track(tau, init, full=True)
    tg = {"corridor_occ_dist": torch.full((1, 49, 200, 200), 8.0), "corridor_sdf": torch.full((1, 200, 200), 5.0),
          "corridor_route_s": torch.arange(100.0)[:, None].expand(100, 100)[None] * 0.64, "corridor_valid": torch.ones(1)}
    w = {"teacher": 1.0, "gt": 1.0, "road": 1.0, "comfort": 1.0, "ec": 0.1, "prog": 0.2}
    ctrl = torch.zeros(1, 12, requires_grad=True)
    Sg = track(N.decode_v6(tau, ctrl, v0)["traj"], init, full=True)
    l, _ = LS.corr_loss(Sg, Sb, tg, torch.zeros(1, 49, 200, 200), torch.zeros(1), w)
    g, = torch.autograd.grad(l, ctrl, allow_unused=True)
    assert g is None or float(g.abs().max()) == 0.0                                            # F.relu at the kink


def _hinge_grad(occ, gamma=1.0):
    tt = torch.arange(1, 9, dtype=torch.float32) * 0.5
    tau = torch.stack([10 * tt, 0.02 * tt ** 2, 0.004 * tt], -1)[None]
    init = torch.cat([torch.full((1, 1), 10.0), torch.zeros(1, 6)], -1)
    ctrl = torch.zeros(1, 12, requires_grad=True)
    Sg = track(N.decode_v6(tau, ctrl, torch.full((1,), 10.0))["traj"], init, full=True)
    h = LS.front_hinge(Sg, occ, "prob", 0.3)
    w = gamma ** torch.arange(40.0)
    return torch.autograd.grad((h * w).sum() / w.sum(), ctrl)[0][0]


def _bump_occ(t0, t1, x):
    occ = torch.zeros(1, 49, 200, 200)
    xs = (torch.arange(200) + 0.5) * 0.32
    occ[:, t0:t1, :, 90:110] = torch.exp(-((xs - x) / 2.0) ** 2)[None, None, :, None]      # a probability field
    return occ


def test_teacher_hinge_gradient_direction_at_zero():
    """report 124 §4-4 / 126 §7-3 (10-27): at Delta = 0 the sign of the time-mean occupancy hinge's gradient is set by
    WHERE the object is relative to tau_base's reach, not by its kind: near or beyond the 4 s end point (contact that a
    1 m/s^2 brake, ~8 m less travel in 4 s, can avoid or shorten) it asks for braking; well inside the reach (contact
    the brake cannot avoid, stopped or crossing) it asks to pass faster, which the decel-only decoder blocks (dead
    zone; v6c/lon_dead).  A strong time discount (gamma 0.93 per 0.1 s) turns this late crossing into braking."""
    assert float(_hinge_grad(_bump_occ(0, 49, 29.0))[:6].sum()) < 0                            # stopped but inside reach
    assert float(_hinge_grad(_bump_occ(0, 49, 40.0))[:6].sum()) > 0                            # lead: brake
    assert float(_hinge_grad(_bump_occ(20, 30, 29.0))[:6].sum()) < 0                           # crossing: faster
    assert float(_hinge_grad(_bump_occ(20, 30, 29.0), gamma=0.93)[:6].sum()) > 0               # discounted: brake


def test_risknet_and_judge_shapes_and_assembly():
    b, r = 2, 2
    reads = {"ro": torch.randn(b, r, 21, 3), "road": torch.randn(b, r, 21, 5), "M": torch.ones(b, r, 21, 5),
             "smp": torch.randn(b, r, 21, 5 * 32), "kin": torch.randn(b, r, 21, 6)}
    re = N.RiskNet(cin=32, hid=64, agent_dim=20)
    o = re(reads, agents=torch.randn(b, 12, 20))
    assert o["E"].shape == (b, r, 21, 12) and o["H"].shape == (b, r, 21, 64)
    te = N.RiskNet(cin=32, hid=64, agent_dim=0, n_area=3)
    assert te(reads, area=torch.zeros(b, r, 21, 3))["E"].shape == (b, r, 21, 12)
    j = N.RowJudge(hid=64)
    S = torch.zeros(b, r, 41, 7)
    out = j(o["E"].detach(), reads["M"], S, torch.zeros(b, r, 21, 2), torch.zeros(b, 4), torch.eye(4)[:b], S[:, 0])
    # assembly == candidate_scorer.expected_official with DDC = 1
    from navsim.agents.para_ssr.modules.candidate_scorer import expected_official
    pred = {"cand_prog": out["prog"], "cand_nc1": out["nc"], "cand_dac1": out["dac"], "cand_ttc1": out["ttc"],
            "cand_ddc1": torch.tensor([20.0, -20, -20]).expand(b, r, 3), "cand_eff1": out["eff"],
            "cand_nc2": out["nc"], "cand_dac2": out["dac"], "cand_ddc2": out["nc"], "cand_tlc2": out["dac"],
            "cand_eff2": out["eff"], "cand_ttc2": out["ttc"], "cand_lk2": out["dac"], "cand_hc2": out["dac"]}
    st = torch.zeros(b, r, 41, 7)
    ref = expected_official(pred, st)["pdms1"]
    from navsim.agents.para_ssr.modules.exact_verdict import exact_comfort
    mine = N.expected_pdms(out, exact_comfort(st))["pdms"]
    assert torch.allclose(ref, mine, atol=1e-5)


def test_step_labels_constant_objects_count_although_nobs_is_zero():
    """a stopped car stored once with t = 0 (build_exact constant rows): pack_token gives nobs = 0, so the baseline
    official NC ignores it; the v6c step labels use the last step that holds a row and see the overlap"""
    tab = ES._empty(ES.V1_TABLES)
    tab["agent_rows"] = np.array([[0, 25.0, 0.0, 0.0, 2.4, 1.0, 1.0, 3, 1.0]], np.float32)
    tab["drv_seg"] = _rect_segs(-20, 120, -6, 6)
    tab["lane_seg"] = _rect_segs(-20, 120, -6, 6)
    tab["cl"] = np.stack([np.linspace(-20, 120, 141), np.zeros(141), np.linspace(0, 140, 141)], -1).astype(np.float32)
    tab["pdm"] = np.zeros((41, 9), np.float32)
    t = ES.pack_full(tab, "ex1_")
    t["ex1_valid"] = torch.tensor(1.0)
    t = {k: v[None] for k, v in t.items()}
    assert int(t["ex1_nobs"][0]) == 0
    st = _straight(10.0)
    nc, _ = EV.exact_nc_ttc(st, {k.replace("ex1_", "ex_"): v for k, v in t.items()})
    assert float(nc[0, 0]) == 1.0                                                            # baseline: missed
    lab = L.exact_step_masks(st, t)
    assert lab["ovl"][0, 0].sum() > 0 and lab["valid"][0, 0].all()


def test_dead_flag_only_in_decel_only_mode():
    tau, v0 = _tau()
    ctrl = torch.zeros(2, 12)
    ctrl[:, :6] = 3.0                                                                          # push toward speeding up
    assert float(N.decode_v6(tau, ctrl, v0, mode="A")["lon_dead"].min()) == 1.0
    assert N.decode_v6(tau, ctrl, v0, mode="B")["lon_dead"] is None                           # a live speed-up in mode B
