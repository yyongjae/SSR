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
    WHERE the object sits relative to tau_base's front-footprint and TTC-shifted points over time, not by its kind.
    Braking pulls later steps back more; a point still before the object centre moves away from it (brake sign), a
    point already past it moves back toward it (faster sign).  The sum is NOT monotonic in distance (10 m/s straight,
    4 s end 40 m, stopped object: 27.5-32.5 m and 35.5-38.5 m faster, 26.5-27 m, 33-35 m and 39-55 m brake, >= 55.5 m
    zero).  a_max only scales it (dz'/dz = a_max / 4 at z = 0), so it is not a brake-budget rule.  The decel-only
    decoder blocks the faster sign (dead zone; v6c/lon_dead).  A time discount (gamma 0.93 per 0.1 s) turns this late
    crossing into braking."""
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


# ---------------------------------------------------------------- fixes after reports 127 and the 10-27 feedback
def _const_table(extra_rows=None):
    """one stopped car stored once with t = 0 at x = 25 m (+ optional moving rows), road and centerline straight"""
    tab = ES._empty(ES.V1_TABLES)
    rows = [[0, 25.0, 0.0, 0.0, 2.4, 1.0, 1.0, 3, 1.0]] + (extra_rows or [])
    tab["agent_rows"] = np.array(rows, np.float32)
    tab["drv_seg"] = _rect_segs(-20, 120, -6, 6)
    tab["lane_seg"] = _rect_segs(-20, 120, -6, 6)
    tab["cl"] = np.stack([np.linspace(-20, 120, 141), np.zeros(141), np.linspace(0, 140, 141)], -1).astype(np.float32)
    tab["pdm"] = np.zeros((41, 9), np.float32)
    t = ES.pack_full(tab, "ex1_")
    t["ex1_valid"] = torch.tensor(1.0)
    return {k: v[None] for k, v in t.items()}


@pytest.mark.parametrize("moving_until", [None, 15])
def test_official_row_labels_use_the_step_label_horizon(moving_until):
    """feedback 10-27 #1 / report 127 §3.1: the official row terms v6c trains R-J on (and the tau_base failure flag)
    must see the same constant object the R-E step labels see.  Two cases: only constant objects (nobs 0) and a
    moving object whose rows end at step 15 (nobs 15) next to the constant one."""
    extra = None if moving_until is None else [[ti, -30.0 + ti, 8.0, 0.0, 2.4, 1.0, 1.0, 3, 1.0]
                                               for ti in range(1, moving_until + 1)]
    t = _const_table(extra)
    assert int(t["ex1_nobs"][0]) == (0 if moving_until is None else moving_until)
    assert int(L.observed_horizon(t)[0]) == 49                                                # constants copied to 49
    t_obs = L.with_observed_horizon(t)
    assert int(t["ex1_nobs"][0]) == (0 if moving_until is None else moving_until)            # the input is untouched
    st7 = _straight(10.0)
    st9 = torch.cat([st7, torch.zeros(*st7.shape[:-1], 2)], -1)
    raw, fixed = ES.official_scores(st9, t, None), ES.official_scores(st9, t_obs, None)
    assert float(raw["nc1"][0, 0]) == 1.0                                                     # the baseline engine misses it
    assert float(fixed["nc1"][0, 0]) < 1.0                                                    # v6c's official rows see it
    lab = L.exact_step_masks(st7, t)
    assert lab["ovl"][0, 0].sum() > 0                                                         # ... like the R-E labels


def _random_teacher(tmp_path, phase="te", phases=("fields", "platt_occ", "te")):
    from navsim.agents.para_ssr.refiner_v6c.teacher import READ_RULE, TeacherSet
    torch.manual_seed(0)
    ts = TeacherSet()
    p = tmp_path / f"teachers_{phase}_{len(phases)}.pt"
    meta = {"folds": 2, "hid_e": 192, "planes": 49, "phase_done": phase, "phases": list(phases),
            "read_rule": READ_RULE}
    torch.save({"state_dict": ts.state_dict(), "meta": meta}, p)
    return p


def test_incomplete_teacher_checkpoint_is_refused(tmp_path):
    """report 127 §3.2: every stage-T phase saves the whole TeacherSet (untrained T-E included); only 'te' is complete"""
    from types import SimpleNamespace
    from navsim.agents.para_ssr.refiner_v6c.teacher import load_teachers
    cfg = SimpleNamespace(embed_dims=256, bev_h=100, bev_w=100)
    for phase, phases in (("fields", ("fields",)), ("platt_occ", ("fields", "platt_occ")),
                          ("te", ("fields", "te"))):                                       # resumed te, platt_occ skipped
        with pytest.raises(RuntimeError, match="incomplete"):
            load_teachers(str(_random_teacher(tmp_path, phase, phases)), cfg)
    ts = load_teachers(str(_random_teacher(tmp_path, "te")), cfg)
    assert ts.loaded_parts["te"] and not ts.training


def test_teacher_ttc_gate_needs_the_projected_box_in_the_window(tmp_path):
    """report 127 §3.4 / feedback 10-27: the TTC-based KD channels need the box shifted by speed * 0.9 s inside the
    teacher window too; the overlap and off-road gates need only the current box"""
    from types import SimpleNamespace
    from navsim.agents.para_ssr.refiner_v6c.teacher import load_teachers
    ts = load_teachers(str(_random_teacher(tmp_path)), SimpleNamespace(embed_dims=256, bev_h=100, bev_w=100))
    S = torch.zeros(1, 2, 41, 7)
    S[:, :, :, 0] = 52.0                                                                     # box front ~55 m, inside
    S[:, 0, :, 3] = 20.0                                                                     # fast: 0.9 s ahead is ~73 m
    S[:, 1, :, 3] = 1.0                                                                      # slow: ~56 m, inside
    flds = {"tok_det": torch.zeros(1, 100 * 100, 256), "sdf_m": torch.full((1, 200, 200), 5.0),
            "occ_prob": torch.zeros(1, 49, 200, 200)}
    po = ts.path_outputs(S, flds, torch.zeros(1, 2, 21, 3), torch.zeros(1, dtype=torch.long))
    assert bool((po["g_obj"] == 1).all()) and bool((po["g_off"] == 1).all())
    assert bool((po["g_ttc"][0, 0] == 0).all()) and bool((po["g_ttc"][0, 1] == 1).all())


def test_occupancy_platt_weighted_fit_recovers_the_real_rates():
    """report 127 §7.2 counterexample: per-batch class-balanced draws of a 1 % batch and a 9 % batch (each with its own
    constant logit).  The inverse-inclusion weighted fit gives back 1 % and 9 %; a bin with too few positives keeps
    its previous map and reports it (no shift is added any more, §7.3)"""
    from navsim.agents.para_ssr.refiner_v6c.teacher import Platt
    import tools.refiner_v6c.train_teachers as TT
    g = torch.Generator().manual_seed(0)
    xs, ys, ws = [[], [], []], [[], [], []], [[], [], []]
    for rate, logit, n in ((0.01, -1.0, 400_000), (0.09, 1.0, 400_000)):
        y = (torch.rand(n, generator=g) < rate).float()
        pos, neg = y.nonzero().flatten(), (y == 0).nonzero().flatten()
        tp, tn = pos[:4096], neg[:4096]
        for kk in range(2):
            xs[kk].append(torch.full((len(tp) + len(tn),), logit))
            ys[kk].append(torch.cat([y[tp], y[tn]]))
            ws[kk].append(torch.cat([torch.full((len(tp),), len(pos) / len(tp)), torch.full((len(tn),), len(neg) / len(tn))]))
    xs[2], ys[2], ws[2] = [torch.zeros(60)], [torch.cat([torch.ones(2), torch.zeros(58)])], [torch.ones(60)]
    pl = Platt()
    pl.b[2] = -0.7
    done = TT._fit(pl, xs, ys, lambda d: None, "occ", ws=ws)
    assert done == [True, True, False] and float(pl.b[2]) == pytest.approx(-0.7)
    p = torch.sigmoid(pl.a[0] * torch.tensor([-1.0, 1.0]) + pl.b[0])
    assert float(p[0]) == pytest.approx(0.01, abs=0.002) and float(p[1]) == pytest.approx(0.09, abs=0.005)


def test_resumed_teacher_set_starts_frozen():
    """report 127 §2.2: a stage-T run that restores a checkpoint must not run T-D / T-M BatchNorm in train mode"""
    import tools.refiner_v6c.train_teachers as TT
    ts = TT.frozen_teacher_set(torch.device("cpu"))
    assert not any(m.training for m in ts.modules())
    assert not any(p.requires_grad for p in ts.parameters())


def test_teacher_store_pread_row_equals_the_memmap_row(tmp_path):
    """report 128 §5: the sharded teacher store reads a row with one pread instead of faulting it in through the
    mapping; the bytes must be identical"""
    from navsim.agents.para_ssr.distill.teacher_store import TeacherFeatureStore
    a = (np.arange(5 * 3 * 4 * 6) % 251).astype(np.float16).reshape(5, 3, 4, 6)
    p = tmp_path / "shard.npy"
    np.save(p, a)
    mm = np.load(p, mmap_mode="r")
    st = TeacherFeatureStore.__new__(TeacherFeatureStore)
    for r in range(5):
        got = st._read_row("s", mm, r)
        assert got.dtype == a.dtype and got.shape == a.shape[1:] and np.array_equal(got, a[r])


def test_rear_mask_ignores_points_behind_the_ego():
    """10-28 review: CH._sample's in-range test had no x >= 0 side and its border padding repeats row 0, so a stopped
    ego's rear corners (x = -REAR) read the first row.  km's 10-29 fix (cf_v15_fix, |v| <= 1) masks them in the base
    (scorer, corridor loss), and since report 129 F01 every reader (student, T-E / T-M) uses km's field_readout as it
    is: the rear corners are out of range, a path entirely ahead reads finite values."""
    from navsim.agents.para_ssr.modules import corridor_head as CH
    from navsim.agents.para_ssr.modules.candidate_scorer import field_readout
    from navsim.agents.para_ssr.refiner_v6c import losses as LS
    sdf = torch.full((1, 200, 200), 3.0)
    sdf[:, 0] = -1.0                                                                          # row 0 (x 0..0.32 m) off-road
    occ = torch.zeros(1, 49, 200, 200)
    occ[:, :, 0] = 1.0
    stop = torch.zeros(1, 1, 41, 7)                                                           # stopped at the origin
    new = field_readout(stop, sdf, occ)
    assert float(new[0, 0, 5, 0]) == 3.0 and float(new[0, 0, 5, 1]) > 0.9                    # front points still read row 0
    ahead = stop.clone()
    ahead[..., 0] = 10.0                                                                      # every point at x > 0
    assert bool(torch.isfinite(field_readout(ahead, sdf, occ)).all())
    S = stop[:, 0]
    assert float(LS.road_hinge(S, sdf)) == 0.0 and float(LS.road_hinge(S, sdf, rear_mask=True)) == 0.0
    S_back = S.clone()
    S_back[..., 0] = -0.5
    S_back[..., 2] = torch.pi                                                                 # behind, facing backwards
    od = torch.full((1, 49, 200, 200), CH.CLIP)
    od[:, :, 0] = -1.0
    assert float(LS.front_hinge(S_back, od, "dist", 0.1).sum()) == 0.0
    assert float(LS.front_hinge(S_back, od, "dist", 0.1, rear_mask=True).sum()) == 0.0


def test_per_control_readout_separates_the_controls():
    """10-28 review: one read-out per control.  Zero at init (Delta = 0), and a loss on the lateral controls gives no
    gradient to the longitudinal read-out (with the shared Linear(hid, 1) it moved every control together)."""
    from navsim.agents.para_ssr.refiner_v6c import nets as N
    torch.manual_seed(0)
    rc = N.Corrector(32, h_dim=32, heads=4, n_cmd=4)
    E, M, S = torch.randn(3, 21, 12), torch.ones(3, 21, 5), _straight(8.0, n=3)[0]
    ego, cmd = torch.randn(3, 4), torch.eye(4)[:3]
    out = rc(E, M, S, ego, cmd)
    assert out.shape == (3, 12) and torch.equal(out, torch.zeros(3, 12))
    out.sum().backward()                                                                    # warm the read-out
    with torch.no_grad():
        rc.out.weight.normal_(0, 0.1)
    rc.zero_grad()
    rc(E, M, S, ego, cmd)[:, 6:].pow(2).sum().backward()
    assert torch.equal(rc.out.weight.grad[:6], torch.zeros(6, 32)) and float(rc.out.weight.grad[6:].abs().sum()) > 0
    assert torch.equal(rc.out.bias.grad[:6], torch.zeros(6))
    shared = N.Corrector(32, h_dim=32, heads=4, n_cmd=4, out_mode="shared")
    assert shared(E, M, S, ego, cmd).shape == (3, 12)
    with pytest.raises(ValueError):
        N.Corrector(32, out_mode="bad")


@pytest.mark.parametrize("lead_x,expect", [(80.0, "pass"), (50.0, "fix"), (30.0, "unfix")])
def test_official_decel_target(lead_x, expect):
    """10-28 review: the longitudinal target comes from the official scorer.  Straight tau_base at 10 m/s (40 m in 4 s)
    with a stopped car ahead: far (80 m) -> tau_base passes, target 0; 50 m -> TTC fails and only the -1.0 variant
    passes (a_max = 1 m/s^2 shortens the 4 s path by at most ~6.5 m); 30 m -> no variant avoids it, left out"""
    from types import SimpleNamespace
    from navsim.agents.para_ssr.refiner_v6c import hook as H
    t = _scene(lead_x=lead_x)
    tt = torch.arange(1, 9, dtype=torch.float32) * 0.5
    tau = torch.stack([10.0 * tt, torch.zeros_like(tt), torch.zeros_like(tt)], -1)[None]
    init = torch.tensor([[10.0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float64)
    st9 = H._trk(tau.double()[:, None].reshape(1, 8, 3), init, full=True, angular=True).reshape(1, 1, 41, 9).float()
    ex = ES.official_scores(st9, t, None)
    base_fail = ((ex["nc1"][:, 0] < 1) | (ex["ttc1"][:, 0] < 1)) & torch.isfinite(ex["nc1"][:, 0])
    cfg = SimpleNamespace(v6c_a_max=1.0, v6c_d_max=0.5, v6c_decoder_mode="A")
    out = H.official_decel_target(tau, torch.tensor([10.0]), torch.zeros(1, 8), init, t, ex, base_fail, cfg)
    if expect == "pass":
        assert not bool(base_fail[0]) and float(out["z"][0]) == 0.0 and bool(out["use"][0])
    elif expect == "fix":
        assert bool(base_fail[0]) and bool(out["fixable"][0]) and float(out["z"][0]) == -1.0 and bool(out["use"][0])
    else:
        assert bool(base_fail[0]) and bool(out["unfixable"][0]) and not bool(out["use"][0])


def test_fixing_term_on_lateral_only_states_leaves_lon_untouched():
    """10-28 review: with S_fix (tau_base edited by the lateral controls only) the fixing part of the teacher term
    gives the longitudinal controls no gradient; without it the time-mean hinge does (and asked for 'faster')"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from navsim.agents.para_ssr.refiner_v6c import hook as H
    from navsim.agents.para_ssr.refiner_v6c import losses as LS
    from navsim.agents.para_ssr.refiner_v6c import nets as N
    tt = torch.arange(1, 9, dtype=torch.float32) * 0.5
    tau = torch.stack([8.0 * tt, torch.zeros_like(tt), torch.zeros_like(tt)], -1)[None]
    init = torch.tensor([[8.0, 0, 0, 0, 0, 0, 0]])
    xs = (torch.arange(200) + 0.5) * 0.32
    ys = 32.0 - (torch.arange(200) + 0.5) * 0.32
    occ = ((xs - 20.0).abs()[:, None] < 2.3) & (ys.abs()[None] < 0.95)
    occ = occ.float()[None, None].expand(1, 49, 200, 200).contiguous() * 0.95
    tg = {"corridor_occ_dist": torch.full((1, 49, 200, 200), 8.0), "corridor_sdf": torch.full((1, 200, 200), 5.0),
          "corridor_route_s": torch.zeros(1, 100, 100), "corridor_valid": torch.ones(1)}
    w = {"teacher": 1.0, "gt": 0.0, "road": 0.0, "comfort": 0.0, "ec": 0.0, "prog": 0.0}
    S_base = H._trk(tau, init, full=True).float()

    def grads(use_fix):
        ctrl = torch.zeros(1, 12, requires_grad=True)
        Sg = H._trk(N.decode_v6(tau, ctrl, torch.tensor([8.0]), 1.0, 0.5, "A", 0.1, anchor_h=torch.zeros(1, 8))["traj"],
                    init, full=True).float()
        S_fix = None
        if use_fix:
            cl = torch.cat([torch.zeros_like(ctrl[:, :6]), ctrl[:, 6:]], -1)
            S_fix = H._trk(N.decode_v6(tau, cl, torch.tensor([8.0]), 1.0, 0.5, "A", 0.0,
                                       anchor_h=torch.zeros(1, 8))["traj"], init, full=True).float()
        l, _ = LS.corr_loss(Sg, S_base, tg, occ, torch.ones(1), w, 0.1, 0.3, S_fix=S_fix)
        return torch.autograd.grad(l, ctrl)[0][0]

    g_old, g_new = grads(False), grads(True)
    assert float(g_old[:6].abs().sum()) > 0
    assert torch.equal(g_new[:6], torch.zeros(6))


def test_rear_mask_rule_follows_the_checkpoint_generation():
    """review 10-28: v6c_rear_mask None = the new rule, except for the 10-27 checkpoints (v6c_rc_out 'shared')"""
    from types import SimpleNamespace
    from navsim.agents.para_ssr.refiner_v6c.hook import rear_mask_of
    assert rear_mask_of(SimpleNamespace()) is True
    assert rear_mask_of(SimpleNamespace(v6c_rear_mask=None, v6c_rc_out="per_control")) is True
    assert rear_mask_of(SimpleNamespace(v6c_rear_mask=None, v6c_rc_out="shared")) is False
    assert rear_mask_of(SimpleNamespace(v6c_rear_mask=True, v6c_rc_out="shared")) is True
    assert rear_mask_of(SimpleNamespace(v6c_rear_mask=False, v6c_rc_out="per_control")) is False


def test_official_decel_target_scores_only_failing_rows_in_a_batch():
    """the decel variants are scored only for failing tau_base rows; a 3-row batch (pass, fix, unfix) gives the same
    targets as the rows one by one"""
    from types import SimpleNamespace
    from navsim.agents.para_ssr.refiner_v6c import hook as H
    ts = [_scene(lead_x=x) for x in (80.0, 50.0, 30.0)]
    t = {k: torch.cat([x[k] for x in ts], 0) for k in ts[0]}
    tt = torch.arange(1, 9, dtype=torch.float32) * 0.5
    tau = torch.stack([10.0 * tt, torch.zeros_like(tt), torch.zeros_like(tt)], -1)[None].repeat(3, 1, 1)
    init = torch.tensor([[10.0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float64).repeat(3, 1)
    st9 = H._trk(tau.double(), init, full=True, angular=True).reshape(3, 1, 41, 9).float()
    ex = ES.official_scores(st9, t, None)
    base_fail = ((ex["nc1"][:, 0] < 1) | (ex["ttc1"][:, 0] < 1)) & torch.isfinite(ex["nc1"][:, 0])
    cfg = SimpleNamespace(v6c_a_max=1.0, v6c_d_max=0.5, v6c_decoder_mode="A")
    out = H.official_decel_target(tau, torch.full((3,), 10.0), torch.zeros(3, 8), init, t, ex, base_fail, cfg)
    assert out["z"].tolist() == [0.0, -1.0, 0.0]
    assert out["fixable"].tolist() == [False, True, False] and out["unfixable"].tolist() == [False, False, True]
    assert out["use"].tolist() == [True, True, False]
