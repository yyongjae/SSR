import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.modules.losses import normalize_bbox
from navsim.agents.para_ssr.para_ssr_targets import navsim_box_to_ssr
from navsim.evaluate.aux_metrics import (
    decode_detection_predictions,
    decode_map_predictions,
    evaluate_auxiliary_records,
    resample_polyline,
)


PC_RANGE = (-15.0, -30.0, -2.0, 15.0, 30.0, 2.0)


def _line(y=0.0):
    return np.asarray([[0.0, y], [1.0, y]], dtype=np.float32)


def _record(
    token="a",
    *,
    det_pred_boxes=None,
    det_pred_scores=None,
    det_pred_labels=None,
    det_gt_boxes=None,
    det_gt_labels=None,
    map_pred_points=None,
    map_pred_scores=None,
    map_pred_labels=None,
    map_gt_points=None,
    map_gt_labels=None,
):
    return {
        "token": token,
        "det_pred_boxes": np.asarray(
            det_pred_boxes if det_pred_boxes is not None else [], dtype=np.float32
        ).reshape(-1, 9),
        "det_pred_scores": np.asarray(
            det_pred_scores if det_pred_scores is not None else [], dtype=np.float32
        ),
        "det_pred_labels": np.asarray(
            det_pred_labels if det_pred_labels is not None else [], dtype=np.int64
        ),
        "det_gt_boxes": np.asarray(
            det_gt_boxes if det_gt_boxes is not None else [], dtype=np.float32
        ).reshape(-1, 9),
        "det_gt_labels": np.asarray(
            det_gt_labels if det_gt_labels is not None else [], dtype=np.int64
        ),
        "map_pred_points": np.asarray(
            map_pred_points if map_pred_points is not None else [], dtype=np.float32
        ).reshape(-1, 2, 2),
        "map_pred_scores": np.asarray(
            map_pred_scores if map_pred_scores is not None else [], dtype=np.float32
        ),
        "map_pred_labels": np.asarray(
            map_pred_labels if map_pred_labels is not None else [], dtype=np.int64
        ),
        "map_gt_points": np.asarray(
            map_gt_points if map_gt_points is not None else [], dtype=np.float32
        ).reshape(-1, 2, 2),
        "map_gt_labels": np.asarray(
            map_gt_labels if map_gt_labels is not None else [], dtype=np.int64
        ),
    }


def _box(x=0.0, y=0.0):
    return np.asarray([x, y, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0], dtype=np.float32)


def test_detection_decoder_uses_last_layer_sigmoid_flattened_topk():
    logits = torch.full((2, 1, 2, 7), -10.0)
    logits[0, 0, 0, 0] = 100.0  # must be ignored: not the final decoder layer
    logits[1, 0, 0, 3] = 6.0
    logits[1, 0, 1, 2] = 5.0

    physical = torch.stack(
        [
            torch.tensor(_box(1.0, 2.0)),
            torch.tensor(_box(3.0, 4.0)),
        ]
    )
    codes = normalize_bbox(physical)
    all_codes = torch.zeros((2, 1, 2, 10))
    all_codes[-1, 0] = codes

    decoded = decode_detection_predictions(
        {"all_cls_scores": logits, "all_bbox_preds": all_codes}, max_predictions=2
    )[0]
    assert decoded["labels"].tolist() == [3, 2]
    np.testing.assert_allclose(decoded["boxes"][:, :2], [[1, 2], [3, 4]])
    assert decoded["scores"][0] > decoded["scores"][1]


def test_map_decoder_uses_last_layer_flattened_topk_and_metric_axes():
    logits = torch.full((2, 1, 2, 3), -10.0)
    logits[0, 0, 0, 0] = 100.0
    logits[1, 0, 1, 2] = 6.0
    logits[1, 0, 0, 1] = 5.0
    points = torch.zeros((2, 1, 2, 2, 2))
    points[-1, 0, 1] = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    points[-1, 0, 0] = 0.5

    decoded = decode_map_predictions(
        {"all_map_cls_scores": logits, "all_map_pts_preds": points},
        pc_range=PC_RANGE,
        max_predictions=2,
    )[0]
    assert decoded["labels"].tolist() == [2, 1]
    np.testing.assert_allclose(decoded["points"][0], [[-15, -30], [15, 30]])
    np.testing.assert_allclose(decoded["points"][1], [[0, 0], [0, 0]])


def test_navsim_box_conversion_matches_ssr_axes_and_dimensions():
    converted = navsim_box_to_ssr(
        np.asarray([10.0, 2.0, 1.0, 4.0, 2.0, 1.5, 0.25]),
        np.asarray([3.0, 1.0, 0.0]),
    )
    np.testing.assert_allclose(converted[:6], [-2.0, 10.0, 1.0, 2.0, 4.0, 1.5])
    np.testing.assert_allclose(converted[7:], [-1.0, 3.0])
    expected_yaw = np.arctan2(np.sin(-0.25 - np.pi), np.cos(-0.25 - np.pi))
    assert converted[6] == pytest.approx(expected_yaw)


def test_oracle_detection_and_map_have_map_one():
    record = _record(
        det_pred_boxes=[_box()],
        det_pred_scores=[0.9],
        det_pred_labels=[0],
        det_gt_boxes=[_box()],
        det_gt_labels=[0],
        map_pred_points=[_line()],
        map_pred_scores=[0.9],
        map_pred_labels=[0],
        map_gt_points=[_line()],
        map_gt_labels=[0],
    )
    result = evaluate_auxiliary_records([record])
    assert result["detection"]["mAP"] == pytest.approx(1.0)
    assert result["map"]["mAP"] == pytest.approx(1.0)


def test_map_chamfer_thresholds_use_metric_distance_and_inclusive_boundary():
    record = _record(
        map_pred_points=[_line(0.75)],
        map_pred_scores=[0.9],
        map_pred_labels=[0],
        map_gt_points=[_line(0.0)],
        map_gt_labels=[0],
    )
    result = evaluate_auxiliary_records([record])
    thresholds = result["map"]["classes"]["divider"]["thresholds"]
    assert thresholds["0.5"]["AP"] == pytest.approx(0.0)
    assert thresholds["1"]["AP"] == pytest.approx(1.0)
    assert thresholds["1.5"]["AP"] == pytest.approx(1.0)

    boundary = _record(
        map_pred_points=[_line(0.5)],
        map_pred_scores=[0.9],
        map_pred_labels=[0],
        map_gt_points=[_line(0.0)],
        map_gt_labels=[0],
    )
    boundary_result = evaluate_auxiliary_records([boundary])
    assert boundary_result["map"]["classes"]["divider"]["thresholds"]["0.5"][
        "AP"
    ] == pytest.approx(1.0)


def test_detection_threshold_is_strict_and_wrong_class_does_not_match():
    exactly_half_metre = _record(
        det_pred_boxes=[_box(0.5, 0.0)],
        det_pred_scores=[0.9],
        det_pred_labels=[0],
        det_gt_boxes=[_box(0.0, 0.0)],
        det_gt_labels=[0],
    )
    result = evaluate_auxiliary_records([exactly_half_metre])
    vehicle = result["detection"]["classes"]["vehicle"]["thresholds"]
    assert vehicle["0.5"]["AP"] == pytest.approx(0.0)
    assert vehicle["1"]["AP"] == pytest.approx(1.0)

    wrong_class = _record(
        det_pred_boxes=[_box()],
        det_pred_scores=[0.9],
        det_pred_labels=[1],
        det_gt_boxes=[_box()],
        det_gt_labels=[0],
    )
    wrong_result = evaluate_auxiliary_records([wrong_class])
    assert wrong_result["detection"]["mAP"] == pytest.approx(0.0)


def test_high_confidence_false_positive_reduces_both_protocols():
    record = _record(
        det_pred_boxes=[_box(10, 10), _box()],
        det_pred_scores=[0.99, 0.9],
        det_pred_labels=[0, 0],
        det_gt_boxes=[_box()],
        det_gt_labels=[0],
        map_pred_points=[_line(10.0), _line()],
        map_pred_scores=[0.99, 0.9],
        map_pred_labels=[0, 0],
        map_gt_points=[_line()],
        map_gt_labels=[0],
    )
    result = evaluate_auxiliary_records([record])
    # Numeric fixture for the nuScenes-style 101-bin/min-recall/min-precision
    # integration: one high-score FP followed by the only TP is exactly 0.2.
    assert result["detection"]["mAP"] == pytest.approx(0.2)
    assert result["map"]["mAP"] == pytest.approx(0.5)


def test_map_duplicate_does_not_fall_back_to_second_closest_gt():
    # Both predictions are closest to y=0.  The first claims it.  Legacy
    # VAD/MapTR matching makes the second an FP even though the y=1 GT is also
    # within the 1.5 m threshold; it never falls back to a second-best GT.
    record = _record(
        map_pred_points=[_line(0.0), _line(0.1)],
        map_pred_scores=[0.9, 0.8],
        map_pred_labels=[0, 0],
        map_gt_points=[_line(0.0), _line(1.0)],
        map_gt_labels=[0, 0],
    )
    result = evaluate_auxiliary_records([record])
    divider = result["map"]["classes"]["divider"]["thresholds"]
    assert divider["1.5"]["max_recall"] == pytest.approx(0.5)


def test_records_must_be_unique_and_sorted_and_nonfinite_fails_closed():
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_auxiliary_records([_record("b"), _record("a")])

    bad = _record(
        det_pred_boxes=[_box()],
        det_pred_scores=[np.nan],
        det_pred_labels=[0],
    )
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_auxiliary_records([bad])


def test_record_cardinality_score_and_threshold_contracts_fail_closed():
    mismatched = _record(det_gt_boxes=[_box()], det_gt_labels=[])
    with pytest.raises(ValueError, match="GT boxes/labels count mismatch"):
        evaluate_auxiliary_records([mismatched])

    invalid_score = _record(
        det_pred_boxes=[_box()], det_pred_scores=[1.1], det_pred_labels=[0]
    )
    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        evaluate_auxiliary_records([invalid_score])

    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_auxiliary_records([_record()], det_thresholds=(0.5, 0.5, 1.0))


def test_degenerate_polyline_resampling_is_finite():
    result = resample_polyline(np.zeros((2, 2), dtype=np.float32), 100)
    assert result.shape == (100, 2)
    assert np.isfinite(result).all()
    assert np.count_nonzero(result) == 0
