"""Student planner reads det/map latents in parallel and keeps ego status off BEV."""
import torch

from navsim.agents.para_ssr.modules.bevformer import SSRPerceptionTransformer
from navsim.agents.para_ssr.modules.grad_balance import balance_shared_gradients
from navsim.agents.para_ssr.modules.planner_head import (
    ParaSSRPlannerHead,
    PlanTaskMemoryLayer,
)


def test_ego_motion_mlp_is_absent_when_status_stays_off_bev():
    transformer = SSRPerceptionTransformer(
        embed_dims=8, num_cams=1, num_feature_levels=1, use_ego_motion=False,
    )
    assert transformer.ego_motion_mlp is None


def test_det_and_map_attention_read_the_same_post_bev_residual():
    torch.manual_seed(0)
    layer = PlanTaskMemoryLayer(8, 2, 16, True)
    queries = []

    def spy(module):
        original = module.forward

        def wrapped(query, key, value, need_weights=False):
            queries.append(query.detach().clone())
            return original(query, key, value, need_weights=need_weights)

        module.forward = wrapped

    spy(layer.plan_det_cross_attn)
    spy(layer.plan_map_cross_attn)
    h = torch.randn(2, 1, 8)
    pos = torch.randn(2, 1, 8)
    bev = torch.randn(2, 4, 8)
    bev_pos = torch.randn(2, 4, 8)
    memory = torch.randn(2, 3, 8)
    meta = torch.randn(2, 3, 8)
    layer(h, pos, bev, bev_pos, memory, meta, meta, memory, meta, meta)
    assert len(queries) == 2
    assert torch.equal(queries[0], queries[1])


def test_plan_loss_reaches_both_decoder_latents():
    torch.manual_seed(1)
    head = ParaSSRPlannerHead(
        transformer=None,
        bev_h=2,
        bev_w=2,
        embed_dims=8,
        num_heads=2,
        feedforward_channels=16,
        fut_ts=4,
        use_stl=False,
        plan_num_layers=1,
        use_task_interaction=True,
    )
    bev = torch.randn(2, 4, 8, requires_grad=True)
    cmd = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    status = torch.tensor([[1.0, 0.0, 0.2, -0.1], [0.0, 3.0, 0.0, 0.4]])
    det_hidden = torch.randn(2, 3, 8, requires_grad=True)
    motion_hidden = torch.randn(2, 3, 2, 8, requires_grad=True)
    map_hidden = torch.randn(2, 2, 4, 8, requires_grad=True)
    det_out = {
        "det_hidden": det_hidden,
        "motion_hidden": motion_hidden,
        "all_bbox_preds": torch.randn(1, 2, 3, 10),
        "all_cls_scores": torch.randn(1, 2, 3, 7),
    }
    map_out = {
        "map_point_hidden": map_hidden,
        "all_map_pts_preds": torch.rand(1, 2, 2, 4, 2),
        "all_map_cls_scores": torch.randn(1, 2, 2, 3),
    }
    preds = head.forward_from_bev(
        bev, cmd, ego_status=status, det_out=det_out, map_out=map_out,
    )
    assert preds["ego_fut_preds"].shape == (2, head.ego_fut_mode, 4, 3)
    preds["ego_fut_preds"].sum().backward()
    assert det_hidden.grad.abs().sum() > 0
    assert motion_hidden.grad.abs().sum() > 0
    assert map_hidden.grad.abs().sum() > 0
    assert bev.grad.abs().sum() > 0


def test_aux_bev_scale_does_not_scale_the_plan_path_through_the_same_features():
    torch.manual_seed(2)
    bev = torch.randn(1, 2, requires_grad=True)
    weight = torch.randn(2, 2, requires_grad=True)
    shared = bev @ weight
    plan = shared.sum()
    det = shared.sum()
    total = plan + det
    corrected, _ = balance_shared_gradients(
        total, bev, {"plan": plan, "det": det}, {"det": 0.25}, measure_norms=True,
    )
    plan_bev = torch.autograd.grad(plan, bev, retain_graph=True)[0]
    det_bev = torch.autograd.grad(det, bev, retain_graph=True)[0]
    full_weight = torch.autograd.grad(total, weight, retain_graph=True)[0]
    corrected.backward()
    assert torch.allclose(bev.grad, plan_bev + 0.25 * det_bev)
    assert torch.allclose(weight.grad, full_weight)
