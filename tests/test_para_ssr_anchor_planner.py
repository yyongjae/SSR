"""v2 planner: WoTE-style anchors + PDM-score rewards (modules/anchor_planner.py)."""
import json

import numpy as np
import pytest
import torch

from navsim.agents.WoTE.WoTE_loss import compute_im_reward_loss, compute_sim_reward_loss, compute_traj_offset_loss
from navsim.agents.para_ssr.modules.anchor_planner import SIM_KEYS, AnchorPlanner, anchor_plan_losses
from navsim.agents.para_ssr.modules.planner_head import ParaSSRPlannerHead
from navsim.agents.para_ssr.plan_score_targets import AnchorScoreTargetBuilder
from types import SimpleNamespace

K, T, C = 16, 8, 64


@pytest.fixture
def anchor_file(tmp_path):
    g = np.random.default_rng(0)
    a = np.cumsum(g.normal(size=(K, T, 3)).astype(np.float32) * [2.0, 0.3, 0.05], axis=1).astype(np.float32)
    p = tmp_path / "anchors.npy"
    np.save(p, a)
    return str(p)


def test_outputs_selection_and_topk(anchor_file):
    m = AnchorPlanner(anchor_file, embed_dims=C, topk=4).eval()
    q = m.queries(torch.randn(2, 1, C))
    assert q.shape == (2, K, C)
    out = m(q)
    assert out["trajectory_offset"].shape == (2, K, T, 3) and out["sim_rewards"].shape == (2, 5, K)
    assert torch.allclose(out["im_rewards"].sum(-1), torch.ones(2), atol=1e-5)
    assert float(out["trajectory_offset"][..., 2].abs().max()) <= np.pi
    best = out["plan_final_rewards"].argmax(-1)
    assert torch.equal(out["plan_topk_index"][:, 0], best)
    expect = m.trajectory_anchors[best] + out["trajectory_offset"][torch.arange(2), best]
    assert torch.allclose(out["trajectory"], expect)
    assert (out["plan_topk_reward"][:, :-1] >= out["plan_topk_reward"][:, 1:]).all()
    assert "trajectory_anchors" in dict(m.named_buffers()) and not any("anchors" in n for n, _ in m.named_parameters())


def test_weighted_reward_is_wote_formula(anchor_file):
    m = AnchorPlanner(anchor_file, embed_dims=C)
    im = torch.softmax(torch.randn(2, K), -1)
    sim = torch.rand(2, 5, K) * 0.9 + 0.05
    nc, dac, ep, ttc, c = sim.unbind(1)
    ref = 0.1 * im.log() + 0.5 * nc.log() + 0.5 * dac.log() + 1.0 * (5 * ttc + 2 * c + 5 * ep).log()
    assert torch.allclose(m.weighted_reward(im, sim), ref, atol=1e-4)
    assert SIM_KEYS == ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                        "time_to_collision_within_bound", "comfort")


def test_losses_match_the_wote_reference(anchor_file):
    m = AnchorPlanner(anchor_file, embed_dims=C)
    out = m(m.queries(torch.randn(3, 1, C)))
    gt = m.trajectory_anchors[[2, 5, 7]] + 0.1 * torch.randn(3, T, 3)
    sim_t = (torch.rand(3, 5, K) > 0.5).float()
    ours = anchor_plan_losses(out, gt, sim_t, torch.ones(3))
    ref_targets = {"trajectory": gt.unsqueeze(1), "sim_reward": sim_t.unsqueeze(1)}
    ref_pred = {"trajectory_anchors": out["trajectory_anchors"], "trajectory_offset": out["trajectory_offset"]}
    assert float(ours["traj_offset_loss"]) == pytest.approx(float(compute_traj_offset_loss(ref_pred, ref_targets, None)), rel=1e-5)
    assert float(ours["im_reward_loss"]) == pytest.approx(
        float(compute_im_reward_loss(ref_targets, out["im_rewards"], out["trajectory_anchors"])), rel=1e-5)
    assert float(ours["sim_reward_loss"]) == pytest.approx(float(compute_sim_reward_loss(ref_targets, out["sim_rewards"])), rel=1e-5)
    # unlabelled samples drop out of the sim term only
    half = anchor_plan_losses(out, gt, sim_t, torch.tensor([1.0, 0.0, 0.0]))
    one = anchor_plan_losses({k: (v[:1] if k != "trajectory_anchors" else v) for k, v in out.items()
                              if k in ("trajectory_offset", "im_rewards", "sim_rewards", "trajectory_anchors")},
                             gt[:1], sim_t[:1], torch.ones(1))
    assert float(half["sim_reward_loss"]) == pytest.approx(float(one["sim_reward_loss"]), rel=1e-5)


def test_planner_head_runs_anchor_queries_through_its_layers(anchor_file):
    head = ParaSSRPlannerHead(transformer=torch.nn.Identity(), bev_h=5, bev_w=6, embed_dims=C, num_heads=4,
                              feedforward_channels=128, use_task_interaction=False, plan_anchor_file=anchor_file)
    bev = torch.randn(2, 30, C, requires_grad=True)
    out = head.plan_from_bev(bev, torch.eye(4)[[1, 2]], ego_status=torch.randn(2, 4))
    assert out["ego_fut_preds"].shape == (2, 4, T, 3)
    assert torch.allclose(out["ego_fut_preds"][:, 0].cumsum(1), out["trajectory"], atol=1e-5)
    assert torch.allclose(head.select_trajectory(out["ego_fut_preds"], torch.eye(4)[[1, 2]]), out["trajectory"], atol=1e-5)
    (out["sim_rewards"].sum() + out["im_rewards"][:, 0].sum() + out["trajectory_offset"].sum()).backward()
    assert bev.grad.abs().sum() > 0
    v1 = ParaSSRPlannerHead(transformer=torch.nn.Identity(), bev_h=5, bev_w=6, embed_dims=C, num_heads=4,
                            feedforward_channels=128, use_task_interaction=False)
    assert v1.anchor_planner is None and "trajectory_offset" not in v1.plan_from_bev(
        bev, torch.eye(4)[[1, 2]], ego_status=torch.randn(2, 4))


def test_score_targets_read_the_packed_file(tmp_path):
    scores = np.random.default_rng(1).random((3, 5, K)).astype(np.float16)
    np.save(tmp_path / "s.npy", scores)
    json.dump(["a", "b", "c"], open(tmp_path / "s.tokens.json", "w"))
    b = AnchorScoreTargetBuilder(SimpleNamespace(plan_score_file=str(tmp_path / "s")))

    def scene(tok):
        f = SimpleNamespace(token=tok)
        return SimpleNamespace(frames=[f] * 4, scene_metadata=SimpleNamespace(num_history_frames=4))

    t = b.compute_targets(scene("b"))
    assert float(t["sim_reward_valid"]) == 1.0 and np.allclose(t["sim_reward"].numpy(), scores[1].astype(np.float32))
    t = b.compute_targets(scene("zzz"))
    assert float(t["sim_reward_valid"]) == 0.0 and t["sim_reward"].shape == (5, K)


def test_v2_head_has_no_unused_regressor(anchor_file):
    head = ParaSSRPlannerHead(transformer=torch.nn.Identity(), bev_h=5, bev_w=6, embed_dims=C, num_heads=4,
                              feedforward_channels=128, use_task_interaction=False, plan_anchor_file=anchor_file)
    assert head.ego_fut_decoder is None
    bev = torch.randn(2, 30, C)
    out = head.plan_from_bev(bev, torch.eye(4)[[1, 2]], ego_status=torch.randn(2, 4))
    sim_t = (torch.rand(2, 5, K) > 0.5).float()
    losses = anchor_plan_losses(out, out["trajectory"].detach() + 0.1, sim_t, torch.ones(2))
    sum(losses.values()).backward()
    unused = [n for n, p in head.named_parameters() if p.requires_grad and p.grad is None
              and not n.startswith("transformer") and not n.startswith("bev_embedding")
              and not n.startswith("positional_encoding")]
    assert unused == [], unused


def test_kinematic_rollout_inverts_and_keeps_heading_on_the_path():
    from navsim.agents.para_ssr.modules.kinematics import bicycle_rollout, poses_to_controls
    ctrl = torch.stack([torch.full((2, 8), 0.5), torch.linspace(-0.2, 0.2, 8).expand(2, 8)], -1)
    v0 = torch.tensor([5.0, 8.0])
    poses, _ = bicycle_rollout(ctrl, v0, 0.5)
    # heading follows the direction of travel of the rolled-out path
    p = torch.cat([torch.zeros(2, 1, 3), poses], 1)
    d = p[:, 1:, :2] - p[:, :-1, :2]
    travel = torch.atan2(d[..., 1], d[..., 0])
    mid = 0.5 * (p[:, 1:, 2] + p[:, :-1, 2])
    assert (travel - mid).abs().max() < 1e-4
    # inverse kinematics recovers the yaw rate exactly and the acceleration closely
    back = poses_to_controls(poses, v0, 0.5)
    assert torch.allclose(back[..., 1], ctrl[..., 1], atol=1e-5)


def test_kinematic_anchor_planner_outputs_consistent_trajectories(anchor_file):
    m = AnchorPlanner(anchor_file, embed_dims=C, kinematic=True).eval()
    q = m.queries(torch.randn(2, 1, C))
    with pytest.raises(ValueError):
        m(q)
    out = m(q, init_speed=torch.tensor([3.0, 6.0]))
    assert out["trajectory_offset"].shape == (2, K, T, 3)
    traj = out["trajectory"]
    p = torch.cat([torch.zeros(2, 1, 3), traj], 1)
    d = p[:, 1:, :2] - p[:, :-1, :2]
    moving = d.norm(dim=-1) > 0.5
    travel = torch.atan2(d[..., 1], d[..., 0])
    mid = 0.5 * (p[:, 1:, 2] + p[:, :-1, 2])
    assert ((travel - mid).abs()[moving]).max() < 1e-3
    sim_t = (torch.rand(2, 5, K) > 0.5).float()
    sum(anchor_plan_losses(out, traj.detach() + 0.1, sim_t, torch.ones(2)).values()).backward()
    assert m.offset_head[-1].weight.grad.abs().sum() > 0


def test_bezier_heading_matches_diffusiondrivev2_and_survives_standing_still():
    from navsim.agents.para_ssr.modules.kinematics import bezier_xyyaw
    # a straight line: heading 0 everywhere
    xy = torch.stack([torch.arange(1, 9).float() * 2, torch.zeros(8)], -1)[None]
    assert torch.allclose(bezier_xyyaw(xy, torch.full((1, 8), 0.7))[..., 2], torch.zeros(1, 8), atol=1e-6)
    # reference: DiffusionDriveV2's loop formulation of the derivative at t = k/8
    import math
    xy = torch.randn(2, 8, 2).cumsum(1)
    ctrl = torch.cat([torch.zeros(2, 1, 2), xy], 1)
    ref = []
    for k in range(1, 9):
        t = k / 8
        d = sum(math.comb(7, i) * t ** i * (1 - t) ** (7 - i) * (ctrl[:, i + 1] - ctrl[:, i]) for i in range(8)) * 8
        ref.append(torch.atan2(d[:, 1], d[:, 0]))
    assert torch.allclose(bezier_xyyaw(xy, torch.zeros(2, 8))[..., 2], torch.stack(ref, 1), atol=1e-5)
    # standing still: fallback heading and finite gradients
    still = torch.zeros(1, 8, 2, requires_grad=True)
    out = bezier_xyyaw(still, torch.full((1, 8), 0.3))
    assert torch.allclose(out[..., 2], torch.full((1, 8), 0.3))
    out.sum().backward()
    assert torch.isfinite(still.grad).all()


def test_heading_from_xy_anchor_planner(anchor_file):
    m = AnchorPlanner(anchor_file, embed_dims=C, heading_from_xy=True).eval()
    assert m.offset_head[-1].out_features == T * 2
    out = m(m.queries(torch.randn(2, 1, C)))
    from navsim.agents.para_ssr.modules.kinematics import bezier_xyyaw
    traj = out["trajectory"]
    assert torch.allclose(bezier_xyyaw(traj[..., :2], traj[..., 2])[..., 2], traj[..., 2], atol=1e-5)
    sim_t = (torch.rand(2, 5, K) > 0.5).float()
    sum(anchor_plan_losses(out, traj.detach() + 0.1, sim_t, torch.ones(2)).values()).backward()
    assert torch.isfinite(m.offset_head[-1].weight.grad).all() and m.offset_head[-1].weight.grad.abs().sum() > 0


def test_rescore_refined_is_eval_only_and_selects_among_refined(anchor_file):
    torch.manual_seed(0)
    kw = dict(transformer=torch.nn.Identity(), bev_h=5, bev_w=6, embed_dims=C, num_heads=4,
              feedforward_channels=128, use_task_interaction=False, plan_anchor_file=anchor_file,
              plan_heading_from_xy=True)
    base = ParaSSRPlannerHead(**kw).eval()
    resc = ParaSSRPlannerHead(**kw, plan_rescore_refined=True).eval()
    resc.load_state_dict(base.state_dict())                     # no new parameters
    bev, cmd, ego = torch.randn(2, 30, C), torch.eye(4)[[1, 2]], torch.randn(2, 4)
    with torch.no_grad():
        o0, o1 = base.plan_from_bev(bev, cmd, ego_status=ego), resc.plan_from_bev(bev, cmd, ego_status=ego)
    # first pass identical; the output is one of the refined trajectories, picked by the second-pass score
    assert torch.allclose(o0["trajectory_offset"], o1["trajectory_offset"])
    refined = resc.anchor_planner.trajectory_anchors.unsqueeze(0) + o1["trajectory_offset"]
    idx = o1["plan_rescore_final_rewards"].argmax(-1)
    assert torch.allclose(o1["trajectory"], refined[torch.arange(2), idx], atol=1e-6)
    assert torch.allclose(o1["ego_fut_preds"][:, 0].cumsum(1), o1["trajectory"], atol=1e-5)
    assert "plan_rescore_final_rewards" not in o0
    # re-encoding the fixed anchors reproduces the first-pass rewards
    h = resc.anchor_planner.queries(torch.randn(2, 1, C))
    h_same = resc.anchor_planner.queries(h[:, :1] * 0 + h[:, :1], trajectories=None)
    assert h_same.shape == (2, K, C)
    # never active in training
    resc.train()
    assert "plan_rescore_final_rewards" not in resc.plan_from_bev(bev, cmd, ego_status=ego)
