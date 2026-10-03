"""BEV selector distillation stays off the v4 adapter path."""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.distill.distillation import build_planning_distillation
from navsim.agents.para_ssr.distill.selector import (
    BEVRegisterSelector,
    anchor_log_attention,
    diversity_loss,
    late_cover_target,
    sharpen_rows,
    token_match_loss,
)
from navsim.agents.para_ssr.modules.planner_head import PlanTaskMemoryLayer


def _write_teacher(root, config, teacher):
    shape = [config.bev_h, config.bev_w]
    sx0, sy0, _, sx1, sy1, _ = config.pc_range
    pc_range = [sy0, -sx1, -3.0, sy1, -sx0, 5.0]
    base = os.path.join(root, teacher, "cache_train_50x100")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": "train",
                "target_bev_shape": shape,
                "bev_channels": config.embed_dims,
                "point_cloud_range": pc_range,
                "to_student_transform": "student_bev = teacher_bev[:, :, ::-1]",
            },
            handle,
        )
    token = "00aabbccddeeff01"
    shard = os.path.join(base, "samples", token[:2])
    os.makedirs(shard, exist_ok=True)
    rng = np.random.default_rng(1)
    np.savez(
        os.path.join(shard, token + ".npz"),
        bev_feature=rng.standard_normal((config.embed_dims, *shape)).astype(np.float16),
    )


def test_v4_config_does_not_enable_the_selector():
    config = ParaSSRConfig()
    assert config.distill_selector is False
    assert config.grad_balance_target == {"plan": 0.4, "det": 0.3, "map": 0.3}
    assert config.use_corridor_mask is True


def test_selector_build_skips_stage1_adapters(tmp_path):
    config = replace(
        ParaSSRConfig(),
        use_distill=True,
        distill_selector=True,
        distill_feature_root=str(tmp_path),
        distill_adapter_checkpoints={},
        embed_dims=8,
        bev_h=4,
        bev_w=4,
        distill_selector_registers=2,
    )
    # Manifest channels follow embed_dims. Rewrite after the small config.
    for teacher in ("bevfusion", "resmap"):
        _write_teacher(str(tmp_path), config, teacher)
    module = build_planning_distillation(config)
    assert module.selector_mode
    assert len(module.adapters) == 0
    assert module.selector is not None
    assert tuple(module.selector.registers.shape) == (2, 2, 8)


def test_token_match_does_not_train_the_registers():
    torch.manual_seed(0)
    selector = BEVRegisterSelector(
        channels=8, num_registers=2, bev_h=4, bev_w=4, tok_warmup_steps=1,
    )
    selector.steps.fill_(1)
    bev = torch.randn(2, 16, 8, requires_grad=True)
    teacher = torch.randn(2, 16, 8)
    command = torch.zeros(2, 4)
    command[:, 1] = 1.0
    plan_attn = torch.rand(2, 16)
    losses, _ = selector(
        bev, (teacher, teacher), plan_attn, command, torch.zeros(2, 4),
    )
    losses["loss_distill_tok"].backward(retain_graph=True)
    assert bev.grad is not None and bev.grad.abs().sum() > 0
    assert selector.registers.grad is None or selector.registers.grad.abs().sum() == 0
    assert selector.query_proj.weight.grad is None or selector.query_proj.weight.grad.abs().sum() == 0

    selector.zero_grad(set_to_none=True)
    bev.grad = None
    (losses["loss_distill_cover"] + losses["loss_distill_div"]).backward()
    assert selector.registers.grad is not None and selector.registers.grad.abs().sum() > 0
    assert bev.grad is None


def test_front_grid_anchors_split_the_registers_at_step_zero():
    selector = BEVRegisterSelector(
        channels=8, num_registers=16, bev_h=50, bev_w=100, tok_warmup_steps=10,
    )
    bev = torch.zeros(1, 5000, 8)
    command = torch.zeros(1, 4)
    command[:, 1] = 1.0
    attention = selector.attend(bev, command, torch.zeros(1, 4))
    means = torch.matmul(attention[0], selector.cell_xy)
    nearest = _nearest_metres(means.reshape(-1, 2).detach())
    # Collapsed init sits at ~1e-6 m and diversity loss 2. A few metres is split.
    assert float(nearest.min()) > 2.0
    assert float(diversity_loss(attention.reshape(1, -1, 5000), selector.cell_xy, 4.0)) < 1.5


def _nearest_metres(means: torch.Tensor) -> torch.Tensor:
    delta = means.unsqueeze(0) - means.unsqueeze(1)
    dist = delta.pow(2).sum(dim=-1).clamp_min(0).sqrt()
    eye = torch.eye(dist.size(0), dtype=torch.bool)
    return dist.masked_fill(eye, 1e6).amin(dim=-1)


def test_diversity_is_lower_when_peaks_are_apart():
    xy = torch.stack((
        torch.linspace(-8, 8, 16),
        torch.zeros(16),
    ), dim=-1)
    same = torch.zeros(1, 4, 16)
    same[..., 0] = 1.0
    apart = torch.zeros(1, 4, 16)
    apart[0, 0, 0] = 1.0
    apart[0, 1, 5] = 1.0
    apart[0, 2, 10] = 1.0
    apart[0, 3, 15] = 1.0
    assert diversity_loss(apart, xy, 4.0) < diversity_loss(same, xy, 4.0)


def test_cell_match_ignores_unselected_cells():
    student = torch.zeros(1, 4, 2, requires_grad=True)
    with torch.no_grad():
        student[0, 0] = torch.tensor([5.0, -5.0])
    teacher = torch.zeros(1, 4, 2)
    hit = torch.zeros(1, 1, 4)
    hit[0, 0, 0] = 1.0
    miss = torch.zeros(1, 1, 4)
    miss[0, 0, 1] = 1.0
    on_cell = token_match_loss(student, teacher, hit)
    off_cell = token_match_loss(student, teacher, miss)
    assert float(on_cell) > 0.1
    assert float(off_cell) < 1e-6
    on_cell.backward()
    assert student.grad[0, 0].abs().sum() > 0
    assert student.grad[0, 1].abs().sum() == 0


def test_late_cover_mixes_structure_and_keeps_an_empty_row():
    plan = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    sharp = sharpen_rows(plan, 0.3)
    assert float(sharp[0, 0]) > float(plan[0, 0])
    structure = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    late = late_cover_target(plan, structure, tau=0.3, mix=0.5)
    assert float(late[0, 3]) > float(sharp[0, 3])
    empty = late_cover_target(plan, torch.zeros(1, 4), tau=0.3, mix=0.5)
    assert torch.allclose(empty, sharp, atol=1e-5)


def test_step_zero_attention_is_the_six_metre_anchor():
    selector = BEVRegisterSelector(
        channels=4, num_registers=4, bev_h=8, bev_w=8, tok_warmup_steps=100,
    )
    bev = torch.randn(1, 64, 4)
    command = torch.zeros(1, 4)
    command[0, 1] = 1.0
    attention = selector.attend(bev, command, torch.zeros(1, 4))
    sigma = torch.tensor(6.0)
    anchor = (-selector.anchor_dist2 / (2.0 * sigma ** 2)).softmax(dim=-1)
    assert torch.allclose(attention, anchor.unsqueeze(0), atol=1e-5)
    assert float(selector.anchor_sigma()) == pytest.approx(6.0)
    assert float(selector.ramp()) == pytest.approx(0.0)


def test_warmup_end_is_the_two_metre_gaussian():
    selector = BEVRegisterSelector(
        channels=4, num_registers=16, bev_h=50, bev_w=100, tok_warmup_steps=10,
        anchor_sigma_m=6.0, anchor_sigma_end_m=2.0,
    )
    with torch.no_grad():
        for module in (
            selector.query_proj, selector.key_proj,
            selector.cmd_to_channel, selector.status_to_channel,
        ):
            module.weight.zero_()
            module.bias.zero_()
        selector.registers.zero_()
    selector.steps.fill_(selector.tok_warmup_steps)
    bev = torch.randn(1, 5000, 4)
    command = torch.zeros(1, 4)
    command[0, 1] = 1.0
    attention = selector.attend(bev, command, torch.zeros(1, 4))
    sigma = torch.tensor(2.0)
    anchor = (-selector.anchor_dist2 / (2.0 * sigma ** 2)).softmax(dim=-1)
    assert float(selector.anchor_sigma()) == pytest.approx(2.0)
    assert torch.allclose(attention, anchor.unsqueeze(0), atol=1e-5)
    means = torch.matmul(attention[0].reshape(-1, 5000), selector.cell_xy)
    nearest = _nearest_metres(means)
    # The v2 residue of a shared uniform map sat at 0.34 m and entropy 8.44.
    assert float(nearest.min()) > 4.0
    assert 4.5 < float(_entropy(attention)) < 5.5
    wide = (-selector.anchor_dist2 / (2.0 * 6.0 ** 2)).softmax(dim=-1)
    assert float(_entropy(anchor)) < float(_entropy(wide))


def test_uniform_mixture_has_no_diversity_gradient():
    """The v2 basin: 5% of a 2 m gaussian plus a shared uniform map."""
    selector = BEVRegisterSelector(
        channels=4, num_registers=16, bev_h=50, bev_w=100, tok_warmup_steps=10,
    )
    sigma = torch.tensor(2.0)
    anchor = (-selector.anchor_dist2 / (2.0 * sigma ** 2)).softmax(dim=-1)
    anchor = anchor.reshape(1, -1, 5000)
    logits = torch.zeros(1, 32, 5000, requires_grad=True)
    learned = logits.softmax(dim=-1)
    residue = 0.05 * anchor + 0.95 * learned
    loss = diversity_loss(residue, selector.cell_xy, 4.0)
    grad = torch.autograd.grad(loss, logits)[0]
    assert float(loss) == pytest.approx(1.757, abs=1e-2)
    assert float(grad.abs().mean()) < 1e-4


def test_a_logit_peak_can_leave_the_anchor():
    selector = BEVRegisterSelector(
        channels=4, num_registers=4, bev_h=50, bev_w=100, tok_warmup_steps=10,
    )
    sigma = torch.tensor(2.0)
    dist2 = selector.anchor_dist2
    anchor = (-dist2 / (2.0 * sigma ** 2)).softmax(dim=-1)
    mode = int(anchor[0, 0].argmax())
    distance = (selector.cell_xy - selector.cell_xy[mode]).pow(2).sum(-1).sqrt()
    far = int((distance - 8.0).abs().argmin())
    logits = torch.zeros(1, 2, 4, 5000)
    logits[0, 0, 0, far] = 40.0
    moved = anchor_log_attention(logits, dist2, sigma, torch.tensor(1.0))
    origin = torch.matmul(anchor[0, 0], selector.cell_xy)
    shifted = torch.matmul(moved[0, 0, 0], selector.cell_xy)
    assert float((shifted - origin).norm()) > 4.0
    flat = anchor_log_attention(torch.full_like(logits, 40.0), dist2, sigma, torch.tensor(1.0))
    assert torch.allclose(flat, anchor.unsqueeze(0), atol=1e-5)


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp_min(1e-8)
    return -(prob * prob.log()).sum(dim=-1).mean()


def test_selector_structure_is_agents_and_the_road_boundary(tmp_path):
    config = replace(
        ParaSSRConfig(),
        use_distill=True,
        distill_selector=True,
        distill_feature_root=str(tmp_path),
        distill_adapter_checkpoints={},
        embed_dims=8,
        bev_h=8,
        bev_w=8,
        distill_selector_registers=2,
        distill_map_sigma=0.6,
        distill_agent_inflate=1.0,
    )
    for teacher in ("bevfusion", "resmap"):
        _write_teacher(str(tmp_path), config, teacher)
    module = build_planning_distillation(config)
    boxes = torch.zeros(1, 1, 7)
    boxes[0, 0, 0] = 0.0
    boxes[0, 0, 1] = 16.0
    boxes[0, 0, 3] = 10.0
    boxes[0, 0, 4] = 10.0
    valid = torch.ones(1, 1)
    pts = torch.zeros(1, 1, 1, 8, 2)
    pts[0, 0, 0, :, 0] = torch.linspace(0.05, 0.95, 8)
    pts[0, 0, 0, :, 1] = 0.5
    labels = torch.zeros(1, 1)
    map_valid = torch.ones(1, 1)
    bev = torch.zeros(1, 64, 8)
    agent, boundary = module._selector_structure(
        bev, boxes, valid, pts, labels, map_valid,
    )
    assert agent is not None and boundary is not None
    assert agent.shape == (1, 64)
    assert boundary.shape == (1, 64)
    assert float(agent.sum()) > 0
    assert float(boundary.sum()) > 0
    assert not torch.allclose(agent, boundary)


def test_default_planner_layer_still_returns_a_tensor():
    layer = PlanTaskMemoryLayer(8, 2, 16, True)
    h = torch.randn(2, 1, 8)
    pos = torch.randn(2, 1, 8)
    bev = torch.randn(2, 4, 8)
    bev_pos = torch.randn(2, 4, 8)
    memory = torch.randn(2, 3, 8)
    meta = torch.zeros(2, 3, 8)
    out = layer(h, pos, bev, bev_pos, memory, meta, meta, memory, meta, meta)
    assert out.shape == h.shape
    hidden, attn = layer(
        h, pos, bev, bev_pos, memory, meta, meta, memory, meta, meta,
        return_bev_attn=True,
    )
    assert hidden.shape == h.shape
    assert attn.shape == (2, 4)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_selector_yaml_turns_grad_balance_off():
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "navsim", "planning", "script", "config", "common", "agent",
    )
    from hydra.utils import instantiate

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        selector = compose(config_name="para_ssr_selector_agent")
        v4 = compose(config_name="para_ssr_distill_agent")
    selector_cfg = instantiate(selector.config)
    v4_cfg = instantiate(v4.config)
    assert selector_cfg.distill_selector is True
    assert selector_cfg.image_architecture == "resnet34.tv_in1k"
    assert selector_cfg.grad_balance_target is None
    assert selector_cfg.use_corridor_mask is False
    assert selector_cfg.distill_selector_plan_tau == 0.3
    assert selector_cfg.distill_selector_anchor_sigma == 6.0
    assert selector_cfg.distill_selector_anchor_sigma_end == 2.0
    assert not hasattr(selector_cfg, "distill_selector_anchor_floor")
    assert selector_cfg.distill_selector_struct_mix == 0.5
    assert v4_cfg.distill_selector is False
    assert v4_cfg.image_architecture == "resnet50.tv_in1k"
    assert v4_cfg.grad_balance_target == {"plan": 0.4, "det": 0.3, "map": 0.3}
    assert v4_cfg.use_corridor_mask is True
