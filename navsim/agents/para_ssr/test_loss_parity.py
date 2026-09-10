"""Focused parity tests for PARA-SSR's mmdet-free loss port."""

import torch
import torch.nn.functional as F

from navsim.agents.para_ssr.modules.det_motion_head import (
    ParaDetMotionHead,
    _last_valid_fde,
)
from navsim.agents.para_ssr.modules.losses import (
    denormalize_bbox,
    hungarian_assign_det,
    normalize_bbox,
    ordered_pts_l1_cost,
    sigmoid_focal_loss,
)
from navsim.agents.para_ssr.modules.map_head import ParaMapHead
from navsim.agents.para_ssr.para_ssr_loss import compute_plan_loss


def _det_head(
    *, num_query: int = 1, fut_mode: int = 1, fut_ts: int = 2
) -> ParaDetMotionHead:
    return ParaDetMotionHead(
        num_query=num_query,
        num_classes=1,
        embed_dims=8,
        bev_h=1,
        bev_w=1,
        code_size=10,
        num_reg_fcs=1,
        fut_ts=fut_ts,
        fut_mode=fut_mode,
        num_decoder_layers=1,
        num_heads=1,
        feedforward_channels=16,
        sync_cls_avg_factor=False,
        loss_bbox_weight=1.0,
        loss_traj_weight=1.0,
        loss_traj_cls_weight=1.0,
    )


def _det_predictions(
    raw_bbox: torch.Tensor, *, fut_mode: int = 1, fut_ts: int = 2
):
    raw_bbox = raw_bbox.reshape(-1, 10)
    num_query = raw_bbox.size(0)
    return {
        "all_cls_scores": torch.full(
            (1, 1, num_query, 1), 10.0, requires_grad=True
        ),
        "all_bbox_preds": raw_bbox.reshape(1, 1, num_query, 10)
        .clone()
        .detach()
        .requires_grad_(True),
        "traj_preds": torch.zeros(
            1, num_query, fut_mode, fut_ts, 2, requires_grad=True
        ),
        "traj_cls_preds": torch.zeros(
            1, num_query, fut_mode, requires_grad=True
        ),
    }


def test_bbox_canonical_encode_decode_roundtrip():
    physical = torch.tensor(
        [
            [1.5, -2.0, 0.4, 2.2, 4.7, 1.6, 0.35, 3.0, -1.0],
            [-3.0, 5.0, -0.2, 0.8, 1.4, 1.1, -2.4, -0.5, 0.75],
        ],
        dtype=torch.float64,
    )
    code = normalize_bbox(physical)

    expected = torch.stack(
        [
            physical[:, 0],
            physical[:, 1],
            physical[:, 3].log(),
            physical[:, 4].log(),
            physical[:, 2],
            physical[:, 5].log(),
            physical[:, 6].sin(),
            physical[:, 6].cos(),
            physical[:, 7],
            physical[:, 8],
        ],
        dim=-1,
    )
    torch.testing.assert_close(code, expected)
    torch.testing.assert_close(denormalize_bbox(code), physical)


def test_perfect_raw_bbox_prediction_has_zero_regression_loss():
    head = _det_head()
    gt_box = torch.tensor(
        [[[2.0, 3.0, 0.5, 1.8, 4.2, 1.5, 0.4, 0.2, -0.1]]]
    )
    raw_code = normalize_bbox(gt_box[0, 0])
    predictions = _det_predictions(raw_code)

    q_idx, g_idx = hungarian_assign_det(
        predictions["all_cls_scores"][0, 0],
        predictions["all_bbox_preds"][0, 0],
        raw_code[None],
        torch.tensor([0]),
    )
    assert q_idx.tolist() == [0]
    assert g_idx.tolist() == [0]

    losses = head.loss(
        predictions,
        gt_box,
        torch.tensor([[0]]),
        torch.tensor([[True]]),
    )
    torch.testing.assert_close(losses["loss_bbox"], torch.tensor(0.0))


def test_motion_wta_uses_cumulative_last_valid_fde():
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [50.0, 0.0]]])
    predictions = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 0.0], [-50.0, 0.0]],
                [[0.0, 0.0], [2.0, 0.0], [50.0, 0.0]],
            ]
        ]
    )
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    fde, has_future = _last_valid_fde(predictions, target, mask)
    assert has_future.tolist() == [True]
    # Per-offset aggregate error would choose mode 0; cumulative FDE at t=1
    # correctly chooses mode 1, which reaches the target position exactly.
    assert fde.argmin(dim=-1).tolist() == [1]
    torch.testing.assert_close(fde, torch.tensor([[1.0, 0.0]]))


def test_motion_regression_reduces_by_positive_agent_count():
    head = _det_head(fut_mode=1, fut_ts=2)
    gt_box = torch.tensor(
        [[[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0]]]
    )
    predictions = _det_predictions(normalize_bbox(gt_box[0, 0]))
    gt_future = torch.ones(1, 1, 2, 2)
    gt_mask = torch.ones(1, 1, 2)

    losses = head.loss(
        predictions,
        gt_box,
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        gt_future,
        gt_mask,
    )
    # Four coordinate errors / one positive agent.  Dividing by the number of
    # valid coordinates instead would incorrectly produce 1.0.
    torch.testing.assert_close(losses["loss_traj"], torch.tensor(4.0))


def test_motion_regression_denominator_includes_zero_future_detection_positive():
    head = _det_head(num_query=2, fut_mode=1, fut_ts=2)
    gt_boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0],
            ]
        ]
    )
    predictions = _det_predictions(normalize_bbox(gt_boxes[0]))
    gt_future = torch.zeros(1, 2, 2, 2)
    gt_future[:, 0] = 1.0
    gt_mask = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])

    losses = head.loss(
        predictions,
        gt_boxes,
        torch.tensor([[0, 0]]),
        torch.tensor([[True, True]]),
        gt_future,
        gt_mask,
    )
    # Numerator 4, divided by both detection positives.  The zero-future
    # positive has a zero numerator but remains in the original denominator.
    torch.testing.assert_close(losses["loss_traj"], torch.tensor(2.0))


def test_unmatched_query_is_motion_background():
    head = _det_head(num_query=2, fut_mode=2, fut_ts=2)
    gt_box = torch.tensor(
        [[[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0]]]
    )
    matched_code = normalize_bbox(gt_box[0, 0])
    unmatched_code = matched_code.clone()
    unmatched_code[0] = 12.0
    predictions = _det_predictions(
        torch.stack([matched_code, unmatched_code]), fut_mode=2, fut_ts=2
    )
    losses = head.loss(
        predictions,
        gt_box,
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        torch.zeros(1, 1, 2, 2),
        torch.ones(1, 1, 2),
    )

    expected = sigmoid_focal_loss(
        predictions["traj_cls_preds"].reshape(-1, 2),
        torch.tensor([0, 2]),  # matched mode 0, unmatched background
        torch.ones(2),
        avg_factor=1.0,  # one detection positive; background weight is zero
    )
    torch.testing.assert_close(losses["loss_traj_cls"], expected)
    losses["loss_traj_cls"].backward()
    assert torch.count_nonzero(predictions["traj_cls_preds"].grad[0, 1]) > 0


def test_zero_future_and_empty_gt_keep_motion_branches_in_graph():
    head = _det_head(fut_mode=2, fut_ts=2)
    gt_box = torch.tensor(
        [[[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0]]]
    )

    for valid in (True, False):
        predictions = _det_predictions(
            normalize_bbox(gt_box[0, 0]), fut_mode=2, fut_ts=2
        )
        losses = head.loss(
            predictions,
            gt_box,
            torch.tensor([[0]]),
            torch.tensor([[valid]]),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2),
        )
        torch.testing.assert_close(losses["loss_traj"], torch.tensor(0.0))
        if valid:
            # Matched zero-future query is explicitly ignored.
            torch.testing.assert_close(
                losses["loss_traj_cls"], torch.tensor(0.0)
            )
        else:
            # With no detection GT, the query is supervised as motion
            # background and must produce a real classifier gradient.
            assert losses["loss_traj_cls"] > 0

        sum(losses.values()).backward()
        for key in (
            "all_cls_scores",
            "all_bbox_preds",
            "traj_preds",
            "traj_cls_preds",
        ):
            assert predictions[key].grad is not None, key
        assert torch.count_nonzero(predictions["traj_preds"].grad) == 0
        if valid:
            assert torch.count_nonzero(predictions["traj_cls_preds"].grad) == 0
        else:
            assert torch.count_nonzero(predictions["traj_cls_preds"].grad) > 0


def test_empty_gt_full_head_backward_has_no_unused_parameters():
    torch.manual_seed(7)
    head = _det_head(fut_mode=2, fut_ts=2)
    predictions = head(torch.randn(1, 1, 8, requires_grad=True))
    losses = head.loss(
        predictions,
        torch.zeros(1, 1, 9),
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 2, 2),
        torch.zeros(1, 1, 2),
    )
    sum(losses.values()).backward()
    missing = [
        name
        for name, parameter in head.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    assert torch.count_nonzero(head.traj_cls_branch[-1].bias.grad) > 0


def test_detection_and_motion_losses_contain_nonfinite_predictions():
    head = _det_head(fut_mode=2, fut_ts=2)
    predictions = {
        "all_cls_scores": torch.full(
            (1, 1, 1, 1), float("nan"), requires_grad=True
        ),
        "all_bbox_preds": torch.full(
            (1, 1, 1, 10), float("nan"), requires_grad=True
        ),
        "traj_preds": torch.full(
            (1, 1, 2, 2, 2), float("nan"), requires_grad=True
        ),
        "traj_cls_preds": torch.full(
            (1, 1, 2), float("nan"), requires_grad=True
        ),
    }
    losses = head.loss(
        predictions,
        torch.zeros(1, 1, 9),
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 2, 2),
        torch.zeros(1, 1, 2),
    )
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    assert all(
        prediction.grad is not None and torch.isfinite(prediction.grad).all()
        for prediction in predictions.values()
    )


def test_planning_matches_weighted_l1_full_tensor_mean():
    predictions = torch.zeros(1, 4, 2, 3)
    predictions[:, 1] = 1.0
    loss, _ = compute_plan_loss(
        predictions,
        torch.zeros(1, 2, 3),
        torch.ones(1, 2),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        heading_weight=0.5,
    )
    # Selected branch numerator: 2 steps * (1 + 1 + 0.5) = 5.
    # Original weighted L1 denominator: all 1*4*2*3 prediction elements.
    torch.testing.assert_close(loss, torch.tensor(5.0 / 24.0))


def test_planning_heading_is_periodic_across_two_pi():
    predictions = torch.zeros(1, 4, 2, 3)
    predictions[:, 1, :, 2] = 2.0 * torch.pi
    predictions.requires_grad_()
    loss, _ = compute_plan_loss(
        predictions,
        torch.zeros(1, 2, 3),
        torch.ones(1, 2),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        heading_weight=0.5,
    )
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0.0)
    loss.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_ordered_map_cost_is_sum_not_per_point_mean():
    pred = torch.zeros(1, 2, 2)
    target = torch.ones(1, 1, 2, 2)
    torch.testing.assert_close(
        ordered_pts_l1_cost(pred, target), torch.tensor([[[4.0]]])
    )


def test_map_points_and_metric_direction_reduce_by_num_pos():
    head = ParaMapHead(
        map_num_vec=1,
        map_num_pts_per_vec=3,
        map_num_classes=1,
        embed_dims=8,
        bev_h=1,
        bev_w=1,
        pc_range=(-1.0, -5.0, -2.0, 1.0, 5.0, 2.0),
        num_reg_fcs=1,
        num_decoder_layers=1,
        num_heads=1,
        feedforward_channels=16,
        sync_cls_avg_factor=False,
        loss_map_pts_weight=1.0,
        loss_map_dir_weight=1.0,
        assigner_cls_weight=0.0,
        assigner_pts_weight=1.0,
    )
    target = torch.tensor(
        [[[[[0.10, 0.10], [0.20, 0.20], [0.30, 0.30]]]]]
    )
    pred = torch.tensor(
        [[[[[0.10, 0.10], [0.15, 0.20], [0.20, 0.30]]]]]
    )
    losses = head.loss(
        {
            "all_map_cls_scores": torch.zeros(1, 1, 1, 1),
            "all_map_pts_preds": pred,
        },
        target,
        torch.tensor([[0]]),
        torch.tensor([[True]]),
    )

    # PtsL1Loss sums all point-coordinate errors and divides by one positive.
    torch.testing.assert_close(losses["loss_map_pts"], (pred - target).abs().sum())

    scale = torch.tensor([2.0, 10.0])
    pred_dir = (pred[0, 0, 0, 1:] - pred[0, 0, 0, :-1]) * scale
    target_dir = (target[0, 0, 0, 1:] - target[0, 0, 0, :-1]) * scale
    expected_dir = (1.0 - F.cosine_similarity(pred_dir, target_dir, dim=-1)).sum()
    torch.testing.assert_close(losses["loss_map_dir"], expected_dir)


def test_map_losses_contain_nonfinite_empty_gt_predictions():
    head = ParaMapHead(
        map_num_vec=1,
        map_num_pts_per_vec=3,
        map_num_classes=1,
        embed_dims=8,
        bev_h=1,
        bev_w=1,
        num_reg_fcs=1,
        num_decoder_layers=1,
        num_heads=1,
        feedforward_channels=16,
        sync_cls_avg_factor=False,
    )
    cls_prediction = torch.full((1, 1, 1, 1), float("nan"), requires_grad=True)
    pts_prediction = torch.full((1, 1, 1, 3, 2), float("nan"), requires_grad=True)
    losses = head.loss(
        {
            "all_map_cls_scores": cls_prediction,
            "all_map_pts_preds": pts_prediction,
        },
        torch.zeros(1, 1, 1, 3, 2),
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    assert cls_prediction.grad is not None and torch.isfinite(cls_prediction.grad).all()
    assert pts_prediction.grad is not None and torch.isfinite(pts_prediction.grad).all()
