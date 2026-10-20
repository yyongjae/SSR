"""km 10/18 (v3.2b): 49 occupancy planes (TTC to 4.9 s), occ / SDF loss weights, SDF sigma head, selection rules."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from navsim.agents.para_ssr.modules import corridor_head as CH
from navsim.agents.para_ssr.modules.corridor_select import corridor_select, corridor_verdict, parse_rule


def _preds(b=1, k=8, planes=49, seed=0, sigma=False):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.zeros(k, 8, 3)
    anchors[:, :, 0] = torch.linspace(1, 8, 8)[None] * torch.linspace(0.5, 2.0, k)[:, None]   # straight, k speeds
    p = {"trajectory_anchors": anchors, "trajectory_offset": torch.zeros(b, k, 8, 3),
         "plan_final_rewards": torch.randn(b, k, generator=g), "im_rewards": torch.softmax(torch.randn(b, k, generator=g), -1),
         "sim_rewards": torch.rand(b, 5, k, generator=g), "corridor_sdf": torch.full((b, 100, 200), 5.0),
         "corridor_occ": torch.full((b, planes, 100, 200), -20.0)}
    if sigma:
        p["corridor_sdf_sigma"] = torch.full((b, 100, 200), 0.5)
    return p


def test_plane_index():
    dev = torch.device("cpu")
    s = torch.arange(40)
    assert torch.equal(CH.occ_plane_index(40, dev), s) and torch.equal(CH.occ_plane_index(49, dev), s)
    assert torch.equal(CH.occ_plane_index(8, dev), s // 5)
    assert CH.ttc_plane_index(49, s + 9).max().item() == 48            # 4.9 s
    assert CH.ttc_plane_index(40, s + 9).max().item() == 39            # held 4.0 s (the v3.1 behaviour)
    assert torch.equal(CH.ttc_plane_index(8, s + 3), (s + 3).clamp(max=39) // 5)


def test_parse_rule_v32b():
    assert parse_rule("pred_multiply_t0.25") == ("multiply", 0.25)                  # unchanged 2-tuple
    assert parse_rule("pred_multiply_sigma_ttc", full=True) == ("multiply", "sigma", True)
    assert parse_rule("pred_multiply_t0.25_ttc", full=True) == ("multiply", 0.25, True)
    assert parse_rule("pred_replace_sigma", full=True) == ("replace", "sigma", False)
    for bad in ("pred_multiply_sigma_nc", "pred_multiply_ttc", "pred_multiply_s0.2_ttc"):
        with pytest.raises(ValueError):
            parse_rule(bad)


def test_sigma_rule_needs_sigma_head():
    with pytest.raises(AssertionError):
        corridor_select(_preds(), torch.tensor([[2.0, 0, 0, 0]]), "pred_multiply_sigma_ttc")


def test_free_corridor_keeps_v2_choice_49():
    p = _preds(b=2, sigma=True)
    for rule in ("pred_multiply_sigma_ttc", "pred_multiply_t0.25_ttc", "pred_multiply_sigma"):
        out = corridor_select(p, torch.tensor([[2.0, 0, 0, 0]] * 2), rule)
        assert torch.equal(out["index"], p["plan_final_rewards"].argmax(1)), rule


def test_ttc_sees_plane_past_4s_only_with_49_planes():
    # an agent appears only in the planes after 4.0 s (index 40..48), ahead of the candidates' end: NC (40 steps)
    # ignores it, the TTC projection (t + 0.9 s) of a moving candidate reaches it
    k = 8
    p = _preds(k=k)
    p["corridor_occ"][:, 40:, :, 85:115] = 20.0                                       # all rows, centre columns
    st = torch.tensor([[4.0, 0, 0, 0]])
    traj = p["trajectory_anchors"][None]
    _, occ_max, ttc_max = corridor_verdict(traj, st, p["corridor_sdf"], torch.sigmoid(p["corridor_occ"]), ttc=True)
    assert occ_max.max().item() < 1e-6
    assert ttc_max.min().item() > 0.99
    p40 = _preds(k=k, planes=40)                                                       # 40 planes: never sees it
    _, _, t40 = corridor_verdict(traj, st, p40["corridor_sdf"], torch.sigmoid(p40["corridor_occ"]), ttc=True)
    assert t40.max().item() < 1e-6


def test_sigma_verdict_is_z_units():
    p = _preds(sigma=True)
    p["corridor_sdf"][:] = 1.0
    traj = p["trajectory_anchors"][None]
    st = torch.tensor([[2.0, 0, 0, 0]])
    smin, _ = corridor_verdict(traj, st, p["corridor_sdf"], torch.sigmoid(p["corridor_occ"]))
    zmin, _ = corridor_verdict(traj, st, p["corridor_sdf"], torch.sigmoid(p["corridor_occ"]), sigma=p["corridor_sdf_sigma"])
    assert torch.allclose(zmin, smin / 0.5, atol=1e-4)


def _targets(b=2, planes=49, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"corridor_sdf": torch.randn(b, 100, 200, generator=g) * 3,
            "corridor_occ": (torch.rand(b, planes, 100, 200, generator=g) > 0.97).float(),
            "corridor_valid": torch.ones(b)}


def _old_corridor_loss(pred, targets):     # the v3.1 loss (corridor-v3-fwd 6689221), copied
    sp, op = pred["corridor_sdf"].float(), pred["corridor_occ"].float()
    s, o = targets["corridor_sdf"].float(), targets["corridor_occ"].float()
    v = targets["corridor_valid"].float()
    nv = v.sum().clamp(min=1.0)
    w = 1.0 + 4.0 * torch.exp(-s.abs())
    ls = ((w * F.smooth_l1_loss(sp, s, beta=0.5, reduction="none")).mean((1, 2)) * v).sum() / nv
    lo = (F.binary_cross_entropy_with_logits(op, o, pos_weight=torch.tensor(4.0), reduction="none").mean((1, 2, 3)) * v).sum() / nv
    return ls + lo


def test_corridor_loss_defaults_equal_v31():
    t = _targets(planes=40)
    g = torch.Generator().manual_seed(1)
    pred = {"corridor_sdf": torch.randn(2, 100, 200, generator=g), "corridor_occ": torch.randn(2, 40, 100, 200, generator=g)}
    new, _ = CH.corridor_loss(pred, t)
    assert torch.allclose(new, _old_corridor_loss(pred, t), rtol=1e-5)


def test_corridor_loss_time_weight_and_sigma():
    t = _targets()
    g = torch.Generator().manual_seed(2)
    sp = torch.randn(2, 100, 200, generator=g).requires_grad_()
    op = torch.randn(2, 49, 100, 200, generator=g).requires_grad_()
    sr = torch.zeros(2, 100, 200).requires_grad_()
    pred = {"corridor_sdf": sp, "corridor_occ": op, "corridor_sdf_sigma": F.softplus(sr) + CH.SIGMA_MIN}
    loss, logs = CH.corridor_loss(pred, t, occ_pos_weight=8.0, occ_time_weight=1.0, sdf_boundary_weight=8.0)
    assert {"corr/sdf_nll", "corr/sigma_band1m", "corr/sigma_cov1_band1m"} <= set(logs)
    loss.backward()
    assert sr.grad.abs().sum() > 0 and op.grad.abs().sum() > 0
    # the NLL does not move the SDF mean: its gradient equals the smooth-L1 term's alone
    sp2 = sp.detach().clone().requires_grad_()
    l2, _ = CH.corridor_loss({"corridor_sdf": sp2, "corridor_occ": op.detach()}, t, sdf_boundary_weight=8.0)
    l2.backward()
    assert torch.allclose(sp.grad, sp2.grad, atol=1e-7)
    # later planes weigh more: the same per-plane BCE, plane 48 gets 2x plane 0 (before the mean-1 normalisation)
    o1 = torch.zeros(1, 49, 100, 200).requires_grad_()
    t1 = {"corridor_sdf": torch.zeros(1, 100, 200), "corridor_occ": torch.zeros(1, 49, 100, 200), "corridor_valid": torch.ones(1)}
    l3, _ = CH.corridor_loss({"corridor_sdf": torch.zeros(1, 100, 200), "corridor_occ": o1}, t1, occ_time_weight=1.0)
    l3.backward()
    gpl = o1.grad.abs().sum((0, 2, 3))
    assert torch.allclose(gpl[-1] / gpl[0], torch.tensor(2.0), rtol=1e-4)


def test_verdict_loss_49_planes_sigma_runs_and_trains_sigma():
    b, k = 2, 16
    anchors = torch.zeros(k, 8, 3)
    anchors[:, :, 0] = torch.linspace(1, 8, 8)[None] * torch.linspace(0.5, 2.0, k)[:, None]
    anchors[:, :, 1] = torch.linspace(-1, 1, k)[:, None] * torch.linspace(0, 1, 8)[None] * 3
    t = _targets(b)
    occ = t["corridor_occ"].numpy().astype(bool)
    t["corridor_occ_dist"] = torch.from_numpy(np.stack([CH._occ_dist(o) for o in occ]).astype(np.float32))
    t["trajectory"] = anchors[3][None].repeat(b, 1, 1)
    sr = torch.zeros(b, 100, 200).requires_grad_()
    pred = {"trajectory_anchors": anchors, "corridor_sdf": torch.randn(b, 100, 200),
            "corridor_occ": torch.randn(b, 49, 100, 200).requires_grad_(),
            "corridor_sdf_sigma": F.softplus(sr) + CH.SIGMA_MIN}
    loss, logs = CH.corridor_verdict_loss(pred, t, torch.tensor([[4.0, 0, 0, 0]] * b), n_anchor=8, ttc=True)
    assert torch.isfinite(loss) and "corr_verd/ttc" in logs
    loss.backward()
    assert sr.grad.abs().sum() > 0


def test_plan_target_builder_name_and_planes():
    class C:
        corridor_dir, corridor_tokens_root, corridor_occ_planes = "x", "y", 49
    b = CH.CorridorPlanTargetBuilder(C())
    assert b.get_unique_name() == "para_ssr_corridor_dt01_p49_plan"
    C.corridor_occ_planes = 40
    assert CH.CorridorPlanTargetBuilder(C()).get_unique_name() == "para_ssr_corridor_dt01_plan"
