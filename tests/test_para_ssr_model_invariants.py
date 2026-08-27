import timm
import torch
import torch.nn as nn

from navsim.agents.para_ssr.modules.bevformer import (
    BEVFormerEncoder,
    SSRPerceptionTransformer,
)
from navsim.agents.para_ssr.modules.ms_deform_attn import (
    CustomMSDeformableAttention,
    MSDeformableAttention3D,
    TemporalSelfAttention,
)
from navsim.agents.para_ssr.modules.det_motion_head import ParaDetMotionHead
from navsim.agents.para_ssr.modules.map_head import ParaMapHead
from navsim.agents.para_ssr.modules.planner_head import ParaSSRPlannerHead
from navsim.agents.para_ssr.para_ssr_model import GridMask, ParaSSRModel


def test_temporal_attention_batch_four_matches_single_sample_and_uses_own_history():
    torch.manual_seed(3)
    batch_size, num_query, embed_dims = 4, 4, 8
    attention = TemporalSelfAttention(
        embed_dims=embed_dims,
        num_heads=2,
        num_levels=1,
        num_points=2,
        dropout=0.0,
    ).eval()

    # Specialized initialization deliberately makes these predictors constant.
    # Give them deterministic non-zero weights so this regression also proves
    # that sample-specific history conditioning affects the full output.
    with torch.no_grad():
        for linear in (attention.sampling_offsets, attention.attention_weights):
            values = torch.linspace(-0.02, 0.02, linear.weight.numel())
            linear.weight.copy_(values.reshape_as(linear.weight))

    query = torch.randn(batch_size, num_query, embed_dims)
    temporal_queue = torch.randn(batch_size, 2, num_query, embed_dims)
    reference_points = 0.25 + 0.5 * torch.rand(
        batch_size, 2, num_query, 1, 2
    )
    spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)

    predictor_inputs = []
    hook = attention.sampling_offsets.register_forward_pre_hook(
        lambda _module, args: predictor_inputs.append(args[0].detach().clone())
    )
    batched_output = attention(
        query,
        value=temporal_queue.reshape(batch_size * 2, num_query, embed_dims),
        identity=query,
        reference_points=reference_points.reshape(
            batch_size * 2, num_query, 1, 2
        ),
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    hook.remove()

    expected_predictor_input = torch.cat([temporal_queue[:, 0], query], dim=-1)
    torch.testing.assert_close(predictor_inputs[0], expected_predictor_input)

    per_sample_output = torch.cat(
        [
            attention(
                query[index : index + 1],
                value=temporal_queue[index].reshape(2, num_query, embed_dims),
                identity=query[index : index + 1],
                reference_points=reference_points[index].reshape(
                    2, num_query, 1, 2
                ),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            for index in range(batch_size)
        ],
        dim=0,
    )
    torch.testing.assert_close(batched_output, per_sample_output, atol=1e-6, rtol=1e-6)


def test_temporal_attention_normalizes_each_queue_before_fusion():
    attention = TemporalSelfAttention(
        embed_dims=4,
        num_heads=1,
        num_levels=1,
        num_points=1,
        dropout=0.0,
        batch_first=True,
    ).eval()
    with torch.no_grad():
        attention.value_proj.weight.copy_(torch.eye(4))
        attention.value_proj.bias.zero_()
        attention.output_proj.weight.copy_(torch.eye(4))
        attention.output_proj.bias.zero_()
        attention.sampling_offsets.weight.zero_()
        attention.sampling_offsets.bias.zero_()
        attention.attention_weights.weight.zero_()
        attention.attention_weights.bias.zero_()

    # Each queue samples a constant value of one. Queue-local softmax gives
    # weight 1 to each queue, and the final queue mean must therefore remain 1.
    # A joint queue softmax would give 0.5 to each and incorrectly return 0.5.
    query = torch.zeros(1, 1, 4)
    value = torch.ones(2, 1, 4)
    output = attention(
        query,
        value=value,
        identity=torch.zeros_like(query),
        reference_points=torch.full((2, 1, 1, 2), 0.5),
        spatial_shapes=torch.tensor([[1, 1]], dtype=torch.long),
        level_start_index=torch.tensor([0], dtype=torch.long),
    )
    torch.testing.assert_close(output, torch.ones_like(output))


def test_transformer_restores_specialized_deformable_attention_initialization():
    encoder = BEVFormerEncoder(
        num_layers=1,
        embed_dims=8,
        num_heads=2,
        num_cams=1,
        num_points_in_pillar=2,
        num_points_sca=2,
        num_levels=1,
        feedforward_channels=16,
        ffn_dropout=0.0,
        attn_dropout=0.0,
    )
    # Decoder attention is not normally owned by this transformer, but adding
    # one verifies that the restoration covers all three SSR deformable types.
    encoder.custom_attention_probe = CustomMSDeformableAttention(
        embed_dims=8,
        num_heads=2,
        num_levels=1,
        num_points=2,
        dropout=0.0,
        batch_first=True,
    )
    transformer = SSRPerceptionTransformer(
        embed_dims=8,
        num_cams=1,
        num_feature_levels=1,
        ego_motion_dims=3,
        encoder=encoder,
    )
    assert isinstance(transformer.ego_motion_mlp[-1], nn.LayerNorm)

    deformable_types = (
        MSDeformableAttention3D,
        TemporalSelfAttention,
        CustomMSDeformableAttention,
    )
    deformable_modules = [
        module for module in transformer.modules() if isinstance(module, deformable_types)
    ]
    assert len(deformable_modules) == 3
    for module in deformable_modules:
        assert torch.count_nonzero(module.sampling_offsets.weight) == 0
        assert torch.count_nonzero(module.attention_weights.weight) == 0
        assert torch.count_nonzero(module.attention_weights.bias) == 0
        assert torch.isfinite(module.sampling_offsets.bias).all()
        assert module.sampling_offsets.bias.abs().sum() > 0


def test_grid_mask_mode_one_retained_fraction_and_determinism():
    images = torch.ones(2, 3, 128, 192)

    torch.manual_seed(0)
    mode_one_first = GridMask(ratio=0.5, prob=1.0, mode=1).train()(images)
    torch.manual_seed(0)
    mode_one_second = GridMask(ratio=0.5, prob=1.0, mode=1).train()(images)
    torch.testing.assert_close(mode_one_first, mode_one_second)

    # SSR expands one mask over every image/channel in the flattened batch.
    torch.testing.assert_close(
        mode_one_first,
        mode_one_first[0:1, 0:1].expand_as(mode_one_first),
    )
    retained_fraction = mode_one_first.mean().item()
    assert 0.70 < retained_fraction < 0.85

    # Resetting the RNG gives the same grid; mode=1 must invert mode=0.
    torch.manual_seed(0)
    mode_zero = GridMask(ratio=0.5, prob=1.0, mode=0).train()(images)
    torch.testing.assert_close(mode_one_first + mode_zero, images)


def test_timm_backbone_freeze_and_norm_eval_survive_train_call():
    model = ParaSSRModel.__new__(ParaSSRModel)
    nn.Module.__init__(model)
    model._backbone_frozen_stages = 1
    model._backbone_norm_requires_grad = False
    model._backbone_norm_eval = True
    model.image_encoder = timm.create_model(
        "resnet18", pretrained=False, features_only=True, out_indices=(4,)
    )

    model._apply_backbone_train_policy()
    model.train()

    for stage_name in ("conv1", "bn1", "act1", "maxpool", "layer1"):
        stage = getattr(model.image_encoder, stage_name)
        assert not stage.training
        assert all(not parameter.requires_grad for parameter in stage.parameters())

    assert model.image_encoder.layer2.training
    batch_norms = [
        module
        for module in model.image_encoder.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norms
    assert all(not module.training for module in batch_norms)
    assert all(
        not parameter.requires_grad
        for module in batch_norms
        for parameter in module.parameters()
    )
    # Unfrozen-stage convolutions remain trainable; only BN affine parameters
    # follow the original norm_cfg(requires_grad=False) policy.
    assert model.image_encoder.layer2[0].conv1.weight.requires_grad


def test_det_motion_head_uses_original_mode_tokens_and_box_center_pe():
    head = ParaDetMotionHead(
        num_query=3,
        num_classes=2,
        embed_dims=8,
        bev_h=2,
        bev_w=2,
        code_size=10,
        fut_ts=2,
        fut_mode=4,
        num_decoder_layers=1,
        num_heads=2,
        feedforward_channels=16,
        use_pe=True,
    ).eval()

    assert not hasattr(head, "bev_pos_embed")
    assert tuple(head.motion_mode_query.weight.shape) == (4, 8)
    assert isinstance(head.pos_mlp_sa, nn.Linear)
    assert any(isinstance(module, nn.LayerNorm) for module in head.cls_branches[0])
    assert any(isinstance(module, nn.LayerNorm) for module in head.traj_cls_branch)

    detection_states = []
    motion_inputs = []
    pe_inputs = []
    decoder_hook = head.decoder.register_forward_hook(
        lambda _module, _args, output: detection_states.append(output[0].detach())
    )
    motion_hook = head.motion_decoder.layers[0].attentions[0].register_forward_pre_hook(
        lambda _module, args, kwargs: motion_inputs.append(
            (args[0].detach(), kwargs["query_pos"].detach())
        ),
        with_kwargs=True,
    )
    pe_hook = head.pos_mlp_sa.register_forward_pre_hook(
        lambda _module, args: pe_inputs.append(args[0])
    )
    output = head(torch.randn(2, 4, 8, requires_grad=True))
    decoder_hook.remove()
    motion_hook.remove()
    pe_hook.remove()

    assert len(motion_inputs) == 1
    motion_query, motion_pos = motion_inputs[0]
    assert tuple(motion_query.shape) == (3 * 4, 2, 8)
    expected_query = (
        detection_states[0][-1][:, None]
        + head.motion_mode_query.weight.detach()[None, :, None]
    ).flatten(0, 1)
    torch.testing.assert_close(motion_query, expected_query)

    assert len(pe_inputs) == 1 and not pe_inputs[0].requires_grad
    expected_pos = (
        head.pos_mlp_sa(pe_inputs[0])
        .unsqueeze(2)
        .repeat(1, 1, head.fut_mode, 1)
        .flatten(1, 2)
        .permute(1, 0, 2)
    )
    torch.testing.assert_close(motion_pos, expected_pos)
    assert tuple(output["traj_preds"].shape) == (2, 3, 4, 2, 2)
    assert tuple(output["traj_cls_preds"].shape) == (2, 3, 4)

    deform_attn = head.decoder.layers[0].attentions[1]
    assert torch.count_nonzero(deform_attn.sampling_offsets.weight) == 0
    assert torch.count_nonzero(deform_attn.attention_weights.weight) == 0
    self_attn = head.decoder.layers[0].attentions[0]
    assert self_attn.attn.dropout == 0.1
    assert self_attn.proj_drop.p == 0.0
    assert self_attn.dropout_layer.p == 0.1
    prior_bias = -torch.log(torch.tensor((1 - 0.01) / 0.01))
    assert not torch.allclose(
        head.traj_cls_branch[-1].bias,
        torch.full_like(head.traj_cls_branch[-1].bias, prior_bias),
    )


def test_map_head_has_no_extra_bev_table_and_restores_decoder_init():
    head = ParaMapHead(
        map_num_vec=2,
        map_num_pts_per_vec=3,
        map_num_classes=2,
        embed_dims=8,
        bev_h=2,
        bev_w=2,
        num_decoder_layers=1,
        num_heads=2,
        feedforward_channels=16,
    ).eval()

    assert not hasattr(head, "bev_pos_embed")
    assert any(isinstance(module, nn.LayerNorm) for module in head.cls_branches[0])
    output = head(torch.randn(2, 4, 8))
    assert tuple(output["all_map_cls_scores"].shape) == (1, 2, 2, 2)
    assert tuple(output["all_map_pts_preds"].shape) == (1, 2, 2, 3, 2)

    deform_attn = head.decoder.layers[0].attentions[1]
    assert torch.count_nonzero(deform_attn.sampling_offsets.weight) == 0
    assert torch.count_nonzero(deform_attn.attention_weights.weight) == 0


def test_planner_selects_four_way_command_and_cumsums_navsim_offsets():
    head = ParaSSRPlannerHead.__new__(ParaSSRPlannerHead)
    nn.Module.__init__(head)
    head.num_navi_cmd = 4

    offsets = torch.zeros(2, 4, 2, 3)
    offsets[0, 2] = torch.tensor([[1.0, 2.0, 0.1], [3.0, 4.0, 0.2]])
    offsets[1, 3] = torch.tensor([[-1.0, 0.5, -0.1], [2.0, 1.5, 0.3]])
    command = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    trajectory = head.select_trajectory(offsets, command)

    expected = torch.tensor(
        [
            [[1.0, 2.0, 0.1], [4.0, 6.0, 0.3]],
            [[-1.0, 0.5, -0.1], [1.0, 2.0, 0.2]],
        ]
    )
    torch.testing.assert_close(trajectory, expected)
