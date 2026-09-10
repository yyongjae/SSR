"""Independent invariants for candidate planning and privileged metric labels."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch import nn

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.modules.candidate_planner import (
    CandidateMetricHead,
    candidate_imitation_loss,
    candidate_metric_loss,
    commanded_candidates,
    load_plan_anchors,
    offsets_to_poses,
    poses_to_offsets,
    rank_candidates,
)
from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
from navsim.agents.para_ssr.para_ssr_features import ParaSSRFeatureBuilder
from navsim.agents.para_ssr.para_ssr_loss import compute_plan_loss


def _anchors(count):
    times = torch.arange(1, 9, dtype=torch.float32) * 0.5
    anchors = torch.zeros(count, 8, 3)
    anchors[..., 0] = torch.arange(1, count + 1)[:, None] * times
    return anchors


def _archive(path, *, split="train", anchors=None, metadata_override=None, omit_metadata=()):
    if anchors is None:
        anchors = _anchors(3).numpy()
    metadata = {
        "source_split": split, "format_version": 1,
        "coordinates": "navsim_current_ego_x_forward_y_left_heading",
        "pose_representation": "absolute_in_current_ego_frame",
        "includes_current_pose": False,
    }
    metadata.update(metadata_override or {})
    for key in omit_metadata:
        metadata.pop(key)
    np.savez(path, anchors=anchors, interval_length=np.float64(0.5),
             metadata_json=np.array(json.dumps(metadata)))
    return str(path)


def test_circular_pose_roundtrip_preserves_xy_and_heading_across_pi():
    poses = torch.tensor([[[1.0, 2.0, 3.12], [2.0, 3.0, -3.12],
                           [4.0, 4.0, -3.0]]])
    offsets = poses_to_offsets(poses)
    assert offsets[0, 1, 2].abs() < 0.05
    torch.testing.assert_close(offsets[..., :2], torch.tensor([[[1., 2.], [1., 1.], [2., 1.]]]))
    torch.testing.assert_close(offsets_to_poses(offsets), poses, atol=1e-6, rtol=0)


def test_command_gating_and_metric_ranking_include_unknown_without_cross_command_selection():
    # Every branch/candidate has a different identifiable trajectory.
    all_candidates = torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3, 1, 1)
    all_candidates = all_candidates.expand(-1, -1, -1, 8, 3)
    commanded = commanded_candidates(all_candidates, torch.eye(4))
    expected_k = torch.tensor([2, 0, 1, 2])
    metric = torch.zeros(4, 3, 7)
    metric[..., -1] = -5
    metric[torch.arange(4), expected_k, -1] = 5
    chosen, actual_k, _ = rank_candidates(commanded, torch.zeros(4, 3), metric, 0.1, 1.0)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(chosen, all_candidates[torch.arange(4), torch.arange(4), expected_k])


@pytest.mark.parametrize("count", [1, 2, 5])
def test_winner_refinement_matches_legacy_loss_and_gradient_without_extra_candidate_divisor(count):
    torch.manual_seed(43)
    anchors = _anchors(count)
    winners = torch.arange(4) % count
    gt_offsets = poses_to_offsets(anchors[winners])
    mask = torch.ones(4, 8)
    mask[1, -2:] = 0
    command = torch.eye(4)
    offsets = torch.randn(4, 4, count, 8, 3, requires_grad=True)
    predictions = {
        "candidate_offsets": offsets,
        "candidate_logits": torch.zeros(4, count, requires_grad=True),
        "plan_anchors": anchors,
        "trajectory_candidates": offsets_to_poses(commanded_candidates(offsets, command)),
        "selected_candidate": torch.zeros(4, dtype=torch.long),
    }
    targets = {"trajectory_offsets": gt_offsets, "trajectory_mask": mask, "command": command}
    actual, _, _ = candidate_imitation_loss(predictions, targets, 0.5)

    # Independent explicit gathering fixes the intended winner before reading
    # the regressed candidates. Arbitrarily moving candidates must not reassign.
    expected_offsets = torch.stack([offsets[b, :, int(winners[b])] for b in range(4)])
    expected, _ = compute_plan_loss(expected_offsets, gt_offsets, mask, command, 0.5)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, offsets, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, offsets)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    for b in range(4):
        support = actual_grad[b].abs().sum(dim=(-1, -2)) > 0
        expected_support = torch.zeros(4, count, dtype=torch.bool)
        expected_support[b, winners[b]] = True
        torch.testing.assert_close(support, expected_support)


@pytest.mark.parametrize("detach_bev", [False, True])
def test_metric_bce_cannot_move_candidate_coordinates_and_obeys_bev_detach(detach_bev):
    torch.manual_seed(19)
    shared = nn.Linear(5, 16)
    generator = nn.Linear(5, 8 * 3)
    source = torch.randn(2, 6, 5)
    bev = shared(source)
    pos = torch.randn_like(bev, requires_grad=True)
    candidates = generator(torch.randn(2, 3, 5)).reshape(2, 3, 8, 3)
    status = torch.randn(2, 8, requires_grad=True)
    critic = CandidateMetricHead(16, 2, 8, detach_bev)
    logits = critic(bev, pos, candidates, status)
    labels = torch.full_like(logits, 0.5)
    loss, _ = candidate_metric_loss(logits, labels, (1.,) * 7)
    loss.backward()

    assert generator.weight.grad is None
    assert status.grad is None
    assert critic.output[-1].weight.grad.abs().sum() > 0
    if detach_bev:
        assert shared.weight.grad is None and pos.grad is None
    else:
        assert shared.weight.grad.abs().sum() > 0
        assert pos.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in critic.parameters())


def test_metric_loss_preserves_soft_half_labels_and_effective_per_metric_weights():
    logits = torch.linspace(-1, 1, 42).reshape(2, 3, 7).requires_grad_()
    labels = torch.tensor([0., .5, 1., .25, .75, .5, .8]).expand_as(logits).clone().requires_grad_()
    weights = (3., 3., 1., 2., 4., 1., 1.)
    loss, _ = candidate_metric_loss(logits, labels, weights)
    loss.backward()
    expected = (logits.detach().sigmoid() - labels.detach()) * torch.tensor(weights) / 6
    torch.testing.assert_close(logits.grad, expected)
    assert labels.grad is None


def test_anchor_loading_rejects_heldout_data_and_duplicate_modes(tmp_path):
    good = _archive(tmp_path / "good.npz")
    torch.testing.assert_close(load_plan_anchors(good, 3, 8), _anchors(3))
    heldout = _archive(tmp_path / "heldout.npz", split="val")
    with pytest.raises(ValueError, match="train split"):
        load_plan_anchors(heldout, 3, 8)
    duplicate = _archive(tmp_path / "duplicate.npz", anchors=np.zeros((3, 8, 3)))
    with pytest.raises(ValueError, match="distinct"):
        load_plan_anchors(duplicate, 3, 8)


@pytest.mark.parametrize("overrides,missing", [
    ({"coordinates": "ssr_x_right_y_forward"}, ()),
    ({}, ("coordinates",)),
    ({"pose_representation": "per_step_offsets"}, ()),
    ({"includes_current_pose": True}, ()),
    ({"format_version": 2}, ()),
])
def test_anchor_loading_rejects_wrong_or_missing_pose_convention(tmp_path, overrides, missing):
    archive = _archive(tmp_path / "wrong_convention.npz", metadata_override=overrides,
                       omit_metadata=missing)
    with pytest.raises(ValueError):
        load_plan_anchors(archive, 3, 8)


def _tiny_config(anchor_path="", **overrides):
    values = dict(
        image_architecture="resnet18", backbone_pretrained=False,
        image_scale=1.0, crop_top=0, bev_h=4, bev_w=4,
        embed_dims=16, num_heads=2, ffn_channels=32,
        encoder_num_layers=1, encoder_num_points_in_pillar=2,
        encoder_num_points_sca=2, encoder_attn_dropout=0., encoder_ffn_dropout=0.,
        latent_num_layers=1, num_scenes=2, num_query=3, max_agents=2,
        det_num_decoder_layers=1, map_num_vec=2, map_max_vec=2,
        map_num_pts_per_vec=4, map_num_orders=1, map_num_decoder_layers=1,
        use_grid_mask=False, grad_balance_target=None, grad_norm_log_interval=1,
        use_metric_planner=True, num_plan_candidates=3, plan_anchor_path=anchor_path,
        metric_cache_path="/nonexistent/cache/that/inference/must/not/read",
    )
    values.update(overrides)
    return ParaSSRConfig(**values)


def test_full_tiny_model_backward_validation_and_checkpoint_inference_without_external_files(tmp_path, monkeypatch):
    from test_para_ssr_front_cameras import _front_input

    torch.manual_seed(7)
    archive = tmp_path / "anchors.npz"
    config = _tiny_config(_archive(archive))
    agent = ParaSSRAgent(config, config.trajectory_sampling).train()
    features = {key: value.unsqueeze(0) for key, value in
                ParaSSRFeatureBuilder(config).compute_features(_front_input()).items()}

    class FakeSupervisor:
        def __init__(self):
            self.calls = 0

        def score(self, tokens, candidates):
            assert tokens == ["current_scene"]
            self.calls += 1
            return torch.full((*candidates.shape[:2], 7), 0.5, device=candidates.device)

    supervisor = FakeSupervisor()
    monkeypatch.setattr(agent, "_get_metric_supervisor", lambda: supervisor)
    prediction = agent(features)
    assert supervisor.calls == 0
    assert prediction["candidate_offsets"].shape == (1, 4, 3, 8, 3)
    assert prediction["trajectory_candidates"].shape == (1, 3, 8, 3)
    # Zero residual initialization must really start at the distinct anchors.
    torch.testing.assert_close(prediction["trajectory_candidates"][0], _anchors(3), atol=1e-6, rtol=0)
    targets = {
        "trajectory_offsets": poses_to_offsets(_anchors(3)[1:2]) + 0.01,
        "trajectory_mask": torch.ones(1, 8), "command": features["command"],
        "scene_token": ["current_scene"],
        "gt_boxes": torch.zeros(1, 2, 9), "gt_labels": torch.zeros(1, 2, dtype=torch.long),
        "gt_valid": torch.zeros(1, 2, dtype=torch.bool),
        "gt_fut_trajs": torch.zeros(1, 2, 8, 2), "gt_fut_masks": torch.zeros(1, 2, 8),
        "gt_map_pts": torch.zeros(1, 2, 1, 4, 2),
        "gt_map_labels": torch.zeros(1, 2, dtype=torch.long),
        "gt_map_valid": torch.zeros(1, 2, dtype=torch.bool),
    }
    optimizer = torch.optim.AdamW(agent.para_ssr_model.parameters(), lr=1e-4)
    loss = agent.compute_loss(features, targets, prediction)
    assert torch.isfinite(loss) and supervisor.calls == 1
    assert "candidate_metric_targets" not in targets
    torch.testing.assert_close(agent.latest_logs["loss_plan_total"],
                               sum(agent.latest_logs[key] for key in
                                   ("loss_plan_reg_weighted", "loss_plan_cls_weighted", "loss_plan_metric_weighted")))
    assert agent.latest_logs["gnorm/plan"] > 0
    loss.backward()
    model = agent.para_ssr_model
    assert model.metric_head.output[-1].weight.grad.abs().sum() > 0
    assert model.pts_bbox_head.candidate_cls.weight.grad.abs().sum() > 0
    assert model.pts_bbox_head.ego_fut_decoder[-1].weight.grad.abs().sum() > 0
    assert model.pts_bbox_head.transformer.cams_embeds.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.det_motion_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.map_head.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()

    agent.eval()
    with torch.no_grad():
        expected = agent(features)["trajectory"]
        val_prediction = agent(features)
        assert torch.isfinite(agent.compute_loss(features, targets, val_prediction))
    assert supervisor.calls == 2  # Validation also supervises the critic.
    assert agent._loss.iteration == 1

    checkpoint = tmp_path / "metric.ckpt"
    torch.save({"state_dict": {f"agent.{key}": value for key, value in agent.state_dict().items()}}, checkpoint)
    archive.unlink()
    # Keep the stale path from the old training config: checkpoint restoration
    # must avoid opening it, without requiring users to manually clear the key.
    restored_cfg = replace(config, metric_cache_path="", backbone_pretrained=True)
    import timm
    create_backbone = timm.create_model
    pretrained_requests = []

    def guarded_backbone(*args, **kwargs):
        pretrained_requests.append(kwargs.get("pretrained"))
        assert kwargs.get("pretrained") is False, "Checkpoint restore tried to fetch ImageNet weights"
        return create_backbone(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", guarded_backbone)
    restored = ParaSSRAgent(restored_cfg, restored_cfg.trajectory_sampling, checkpoint_path=str(checkpoint))
    restored.initialize()
    assert pretrained_requests == [False]
    monkeypatch.setattr(restored, "_get_metric_supervisor", lambda: pytest.fail("Inference accessed privileged cache"))
    restored.eval()
    with torch.no_grad():
        actual = restored(features)["trajectory"]
    torch.testing.assert_close(actual, expected)
    assert restored.para_ssr_model.pts_bbox_head.anchors_ready.item()


def test_same_candidate_imitation_ablation_needs_no_metric_cache_or_metric_targets(tmp_path, monkeypatch):
    from test_para_ssr_front_cameras import _front_input

    config = _tiny_config(_archive(tmp_path / "anchors.npz"), metric_loss_weight=0.,
                          metric_score_weight=0., candidate_score_weight=1.,
                          use_det_motion_head=False, use_map_head=False)
    agent = ParaSSRAgent(config, config.trajectory_sampling).train()
    monkeypatch.setattr(agent, "_get_metric_supervisor", lambda: pytest.fail("Imitation ablation accessed metric cache"))
    agent.validate_metric_cache((object(),))
    features = {key: value.unsqueeze(0) for key, value in
                ParaSSRFeatureBuilder(config).compute_features(_front_input()).items()}
    predictions = agent(features)
    predictions["metric_logits"].retain_grad()
    torch.testing.assert_close(predictions["selected_candidate"], predictions["candidate_logits"].argmax(-1))
    targets = {"trajectory_offsets": poses_to_offsets(_anchors(3)[1:2]),
               "trajectory_mask": torch.ones(1, 8), "command": features["command"]}
    loss = agent.compute_loss(features, targets, predictions)
    assert torch.isfinite(loss)
    assert agent.latest_logs["loss_plan_metric"].item() == 0.
    assert "metric/rollout_seconds" not in agent.latest_logs
    assert not any(key.startswith("metric_target/") for key in agent.latest_logs)
    loss.backward()
    torch.testing.assert_close(predictions["metric_logits"].grad, torch.zeros_like(predictions["metric_logits"]))
    assert all(p.grad is not None and p.grad.count_nonzero() == 0
               for p in agent.para_ssr_model.metric_head.parameters())
    assert agent.para_ssr_model.pts_bbox_head.candidate_cls.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("overrides,match", [
    ({"metric_loss_weight": 0., "metric_score_weight": 1.}, "untrained"),
    ({"metric_loss_weight": 0., "metric_score_weight": 0., "candidate_score_weight": 0.}, "ranking weight"),
    ({"metric_loss_weight": -1.}, "nonnegative"),
    ({"metric_score_weight": float("nan")}, "nonnegative"),
])
def test_invalid_metric_and_imitation_only_ranking_configs_fail_early(overrides, match):
    config = _tiny_config(**overrides)
    with pytest.raises(ValueError, match=match):
        ParaSSRAgent._validate_config(config, config.trajectory_sampling)


def test_disabled_metric_planner_preserves_baseline_state_layout():
    from navsim.agents.para_ssr.modules.planner_head import ParaSSRPlannerHead

    head = ParaSSRPlannerHead(nn.Identity(), bev_h=2, bev_w=2, embed_dims=8,
                             num_heads=2, feedforward_channels=16, use_metric_planner=False)
    assert head.way_point.weight.shape == (4 * 8, 16)
    assert not any("anchor" in key or "candidate" in key for key in head.state_dict())
