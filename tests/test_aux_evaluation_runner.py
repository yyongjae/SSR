from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.planning.script import run_aux_evaluation as runner


def _record(token: str = "token_a", identity: str = "a" * 64):
    det_boxes = np.zeros((runner.PROTOCOL_TOP_K, 9), dtype=np.float32)
    det_boxes[:, 3:6] = 1.0
    return {
        "token": np.asarray(token),
        "manifest_identity_sha256": np.asarray(identity),
        "det_pred_boxes": det_boxes,
        "det_pred_scores": np.linspace(0.0, 1.0, runner.PROTOCOL_TOP_K).astype(
            np.float32
        ),
        "det_pred_labels": np.zeros(runner.PROTOCOL_TOP_K, dtype=np.int64),
        "det_gt_boxes": np.zeros((0, 9), dtype=np.float32),
        "det_gt_labels": np.zeros(0, dtype=np.int64),
        "map_pred_points": np.zeros(
            (runner.PROTOCOL_TOP_K, runner.PROTOCOL_MAP_POINTS, 2), dtype=np.float32
        ),
        "map_pred_scores": np.linspace(0.0, 1.0, runner.PROTOCOL_TOP_K).astype(
            np.float32
        ),
        "map_pred_labels": np.zeros(runner.PROTOCOL_TOP_K, dtype=np.int64),
        "map_gt_points": np.zeros(
            (0, runner.PROTOCOL_MAP_POINTS, 2), dtype=np.float32
        ),
        "map_gt_labels": np.zeros(0, dtype=np.int64),
    }


def test_record_schema_fixes_top100_and_twenty_raw_points():
    record = _record()
    runner._validate_record(record, "token_a", "a" * 64)

    wrong_topk = {**record, "det_pred_boxes": record["det_pred_boxes"][:-1]}
    wrong_topk["det_pred_scores"] = record["det_pred_scores"][:-1]
    wrong_topk["det_pred_labels"] = record["det_pred_labels"][:-1]
    with pytest.raises(ValueError, match="exactly 100 detection"):
        runner._validate_record(wrong_topk, "token_a", "a" * 64)

    wrong_points = {
        **record,
        "map_pred_points": record["map_pred_points"][:, :-1],
    }
    with pytest.raises(ValueError, match="20 raw points"):
        runner._validate_record(wrong_points, "token_a", "a" * 64)

    outside_range = _record()
    outside_range["det_pred_boxes"][0, 0] = runner.PROTOCOL_PC_RANGE[3] + 1.0
    with pytest.raises(ValueError, match="outside pc_range"):
        runner._validate_record(outside_range, "token_a", "a" * 64)


def test_resume_record_rejects_another_manifest_identity(tmp_path):
    path = tmp_path / "token_a.npz"
    runner._atomic_write_record(path, _record())
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        runner._load_record(path, "token_a", "b" * 64)


def test_collate_stacks_features_but_keeps_variable_length_targets():
    batch = []
    for index, gt_count in enumerate((1, 3)):
        batch.append(
            (
                f"token_{index}",
                {"image": torch.full((2, 3), float(index))},
                {
                    "gt_boxes": torch.zeros((gt_count, 9)),
                    "gt_map_pts": torch.zeros(
                        (gt_count + 1, 20, 20, 2), dtype=torch.float32
                    ),
                },
            )
        )

    tokens, features, targets = runner._collate_auxiliary_batch(batch)
    assert tokens == ["token_0", "token_1"]
    assert features["image"].shape == (2, 2, 3)
    assert [len(target["gt_boxes"]) for target in targets] == [1, 3]
    assert [len(target["gt_map_pts"]) for target in targets] == [2, 4]


def test_record_builder_keeps_uncapped_variable_map_ground_truth():
    record_template = _record()
    targets = {
        "gt_boxes": torch.zeros((18, 9), dtype=torch.float32),
        "gt_labels": torch.zeros(18, dtype=torch.long),
        "gt_map_pts": torch.zeros((112, 20, 20, 2), dtype=torch.float32),
        "gt_map_labels": torch.zeros(112, dtype=torch.long),
    }
    targets["gt_boxes"][:, 3:6] = 1.0
    record = runner._record_from_batch_item(
        "token_a",
        {
            "boxes": record_template["det_pred_boxes"],
            "scores": record_template["det_pred_scores"],
            "labels": record_template["det_pred_labels"],
        },
        {
            "points": record_template["map_pred_points"],
            "scores": record_template["map_pred_scores"],
            "labels": record_template["map_pred_labels"],
        },
        targets,
        runner.PROTOCOL_PC_RANGE,
        "a" * 64,
    )
    assert record["map_gt_points"].shape == (112, 20, 2)
    assert record["det_gt_boxes"].shape == (18, 9)


def test_production_token_count_cannot_be_overridden_through_config():
    cfg = SimpleNamespace(
        production_guard=True,
        split="test",
        scene_filter_name="navtest",
        # Deliberately present legacy-looking values: the guard must ignore them.
        expected_token_count=1,
        expected_token_sha256=runner._token_sha256(["token_a"]),
    )
    scene_filter = SimpleNamespace(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=None,
    )
    with pytest.raises(RuntimeError, match="expected 12146"):
        runner._production_guard(cfg, ["token_a"], scene_filter)


def test_existing_final_receipt_rejects_changed_record_digest(tmp_path):
    identity = {
        "checkpoint": {"sha256": "c" * 64},
        "training_config": {"sha256": "d" * 64},
        "dataset": {"token_sha256": "e" * 64, "num_tokens": 1},
    }
    metrics = {
        "detection": {"mAP": 0.0, "classes": {}},
        "map": {"mAP": 0.0, "classes": {}},
    }
    runner._publish_final_results(
        tmp_path, "a" * 64, identity, metrics, records_sha256="1" * 64
    )
    with pytest.raises(RuntimeError, match="artifact receipt"):
        runner._publish_final_results(
            tmp_path, "a" * 64, identity, metrics, records_sha256="2" * 64
        )
