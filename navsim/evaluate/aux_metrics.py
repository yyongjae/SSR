"""Held-out auxiliary-head metrics for the NAVSIM PARA-SSR port.

These are deliberately named ``NAVSIMAux*`` rather than official NAVSIM
metrics.  NAVSIM's benchmark consumes only an ego trajectory; this module
scores PARA-SSR's private detection and vector-map heads against the NAVSIM
annotations/map API so their quality can be analysed alongside planning.

Detection follows the nuScenes-style *AP calculation* (not its taxonomy): sigmoid,
flattened query/class top-k decoding, class-aware 2-D centre matching at
0.5/1/2/4 m, 101 recall bins, min recall 0.1 and min precision 0.1.

Map follows VAD/MapTR's Chamfer protocol: sigmoid flattened top-k decoding,
100-point arclength resampling, symmetric mean Chamfer distance, greedy 1:1
matching at 0.5/1.0/1.5 m, and precision-envelope area AP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from navsim.agents.para_ssr.modules.losses import denormalize_bbox
from navsim.agents.para_ssr.para_ssr_targets import (
    DET_CLASS_NAMES,
    MAP_CLASS_NAMES,
)


AUX_METRIC_PROTOCOL_VERSION = 1
DEFAULT_DET_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_MAP_THRESHOLDS = (0.5, 1.0, 1.5)


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        bad = int(value.size - np.isfinite(value).sum())
        raise ValueError(f"{name} contains {bad} non-finite value(s)")


def _stable_flat_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Descending top-k with the flattened index as deterministic tie-break."""

    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError(f"top-k scores must be flat, got {scores.shape}")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"top-k must be positive, got {k}")
    _require_finite("classification scores", scores)
    flat_index = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort((flat_index, -scores))
    return order[: min(k, scores.size)]


def decode_detection_predictions(
    predictions: Mapping[str, torch.Tensor],
    *,
    max_predictions: int = 100,
) -> List[Dict[str, np.ndarray]]:
    """Decode final-layer detection output without mmdet/mmcv dependencies."""

    logits = predictions.get("all_cls_scores")
    box_codes = predictions.get("all_bbox_preds")
    if logits is None or box_codes is None:
        raise KeyError("predictions are missing detection auxiliary outputs")
    if logits.ndim != 4 or box_codes.ndim != 4:
        raise ValueError(
            "detection outputs must be [layers,batch,queries,channels], got "
            f"{tuple(logits.shape)} and {tuple(box_codes.shape)}"
        )
    if logits.shape[:3] != box_codes.shape[:3]:
        raise ValueError("detection class and box output prefixes do not match")
    if logits.shape[-1] != len(DET_CLASS_NAMES) or box_codes.shape[-1] != 10:
        raise ValueError(
            "unexpected detection output dimensions: "
            f"classes={logits.shape[-1]}, code={box_codes.shape[-1]}"
        )
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(box_codes).all()):
        raise ValueError("detection output contains NaN or Inf")

    final_logits = logits[-1].detach().float().cpu()
    final_codes = box_codes[-1].detach().float().cpu()
    decoded: List[Dict[str, np.ndarray]] = []
    num_classes = final_logits.shape[-1]
    for batch_idx in range(final_logits.shape[0]):
        probabilities = final_logits[batch_idx].sigmoid().numpy()
        flat_scores = probabilities.reshape(-1)
        selected = _stable_flat_topk(flat_scores, max_predictions)
        labels = selected % num_classes
        query_indices = selected // num_classes
        physical_boxes = denormalize_bbox(final_codes[batch_idx, query_indices]).numpy()
        _require_finite("decoded detection boxes", physical_boxes)
        if np.any(physical_boxes[:, 3:6] <= 0):
            raise ValueError("decoded detection box has a non-positive size")
        decoded.append(
            {
                "boxes": physical_boxes.astype(np.float32, copy=False),
                "scores": flat_scores[selected].astype(np.float32, copy=False),
                "labels": labels.astype(np.int64, copy=False),
            }
        )
    return decoded


def decode_map_predictions(
    predictions: Mapping[str, torch.Tensor],
    *,
    pc_range: Sequence[float],
    max_predictions: int = 100,
) -> List[Dict[str, np.ndarray]]:
    """Decode final-layer vector-map output into metric PARA-SSR BEV points."""

    logits = predictions.get("all_map_cls_scores")
    normalized_points = predictions.get("all_map_pts_preds")
    if logits is None or normalized_points is None:
        raise KeyError("predictions are missing vector-map auxiliary outputs")
    if logits.ndim != 4 or normalized_points.ndim != 5:
        raise ValueError(
            "map outputs must be [layers,batch,vectors,classes] and "
            f"[layers,batch,vectors,points,2], got {tuple(logits.shape)} and "
            f"{tuple(normalized_points.shape)}"
        )
    if logits.shape[:3] != normalized_points.shape[:3]:
        raise ValueError("map class and point output prefixes do not match")
    if logits.shape[-1] != len(MAP_CLASS_NAMES) or normalized_points.shape[-1] != 2:
        raise ValueError(
            "unexpected map output dimensions: "
            f"classes={logits.shape[-1]}, point_dim={normalized_points.shape[-1]}"
        )
    _validate_pc_range(pc_range)
    if not bool(torch.isfinite(logits).all()) or not bool(
        torch.isfinite(normalized_points).all()
    ):
        raise ValueError("map output contains NaN or Inf")

    final_logits = logits[-1].detach().float().cpu()
    final_points = normalized_points[-1].detach().float().cpu().numpy()
    x0, y0, x1, y1 = (
        float(pc_range[0]),
        float(pc_range[1]),
        float(pc_range[3]),
        float(pc_range[4]),
    )
    decoded: List[Dict[str, np.ndarray]] = []
    num_classes = final_logits.shape[-1]
    for batch_idx in range(final_logits.shape[0]):
        probabilities = final_logits[batch_idx].sigmoid().numpy()
        flat_scores = probabilities.reshape(-1)
        selected = _stable_flat_topk(flat_scores, max_predictions)
        labels = selected % num_classes
        query_indices = selected // num_classes
        points = final_points[batch_idx, query_indices].copy()
        points[..., 0] = points[..., 0] * (x1 - x0) + x0
        points[..., 1] = points[..., 1] * (y1 - y0) + y0
        _require_finite("decoded map points", points)
        decoded.append(
            {
                "points": points.astype(np.float32, copy=False),
                "scores": flat_scores[selected].astype(np.float32, copy=False),
                "labels": labels.astype(np.int64, copy=False),
            }
        )
    return decoded


def denormalize_map_ground_truth(
    normalized_points: np.ndarray, pc_range: Sequence[float]
) -> np.ndarray:
    """Take the canonical GT ordering and convert it to metric BEV points."""

    points = np.asarray(normalized_points, dtype=np.float32)
    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError(
            "map GT must be [instances,orders,points,2], got " f"{points.shape}"
        )
    _require_finite("normalized map GT", points)
    _validate_pc_range(pc_range)
    canonical = points[:, 0].copy()
    x0, y0, x1, y1 = (
        float(pc_range[0]),
        float(pc_range[1]),
        float(pc_range[3]),
        float(pc_range[4]),
    )
    canonical[..., 0] = canonical[..., 0] * (x1 - x0) + x0
    canonical[..., 1] = canonical[..., 1] * (y1 - y0) + y0
    return canonical


def _validate_pc_range(pc_range: Sequence[float]) -> None:
    if len(pc_range) < 5:
        raise ValueError(f"pc_range must contain at least five values, got {pc_range}")
    values = np.asarray(pc_range, dtype=np.float64)
    _require_finite("pc_range", values)
    if float(values[3]) <= float(values[0]) or float(values[4]) <= float(values[1]):
        raise ValueError(f"pc_range must have positive x/y spans, got {pc_range}")


def _validate_thresholds(name: str, thresholds: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in thresholds)
    if not values or not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise ValueError(f"{name} thresholds must be finite and positive, got {thresholds}")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(
            f"{name} thresholds must be unique and strictly increasing, got {thresholds}"
        )
    return values


def resample_polyline(points: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Uniform arclength samples, with a finite zero-length fallback."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 1:
        raise ValueError(f"polyline must be non-empty [points,2], got {points.shape}")
    if num_points < 2:
        raise ValueError(f"num_points must be at least two, got {num_points}")
    _require_finite("polyline", points)
    if len(points) == 1:
        return np.repeat(points, num_points, axis=0).astype(np.float32)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 1e-12:
        return np.repeat(points[:1], num_points, axis=0).astype(np.float32)
    samples = np.linspace(0.0, cumulative[-1], num_points)
    result = np.empty((num_points, 2), dtype=np.float64)
    result[:, 0] = np.interp(samples, cumulative, points[:, 0])
    result[:, 1] = np.interp(samples, cumulative, points[:, 1])
    return result.astype(np.float32)


def _resample_polylines(lines: np.ndarray, num_points: int) -> np.ndarray:
    lines = np.asarray(lines)
    if lines.ndim != 3 or lines.shape[-1] != 2:
        raise ValueError(f"polylines must be [instances,points,2], got {lines.shape}")
    if len(lines) == 0:
        return np.zeros((0, num_points, 2), dtype=np.float32)
    return np.stack([resample_polyline(line, num_points) for line in lines])


def center_distance_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    if pred_boxes.ndim != 2 or pred_boxes.shape[1] < 2:
        raise ValueError(f"pred_boxes must be [N,>=2], got {pred_boxes.shape}")
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] < 2:
        raise ValueError(f"gt_boxes must be [N,>=2], got {gt_boxes.shape}")
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.empty((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    delta = pred_boxes[:, None, :2] - gt_boxes[None, :, :2]
    return np.linalg.norm(delta, axis=-1).astype(np.float32)


def chamfer_distance_matrix(
    pred_lines: np.ndarray,
    gt_lines: np.ndarray,
    *,
    max_relevant_distance: float,
) -> np.ndarray:
    """Symmetric mean nearest-point distance with safe AABB pruning."""

    pred_lines = np.asarray(pred_lines, dtype=np.float32)
    gt_lines = np.asarray(gt_lines, dtype=np.float32)
    if pred_lines.ndim != 3 or pred_lines.shape[-1] != 2:
        raise ValueError(f"pred_lines must be [N,P,2], got {pred_lines.shape}")
    if gt_lines.ndim != 3 or gt_lines.shape[-1] != 2:
        raise ValueError(f"gt_lines must be [N,P,2], got {gt_lines.shape}")
    _require_finite("prediction polylines", pred_lines)
    _require_finite("GT polylines", gt_lines)
    result = np.full((len(pred_lines), len(gt_lines)), np.inf, dtype=np.float32)
    if len(pred_lines) == 0 or len(gt_lines) == 0:
        return result

    gt_min = gt_lines.min(axis=1)
    gt_max = gt_lines.max(axis=1)
    for pred_idx, pred in enumerate(pred_lines):
        pred_min = pred.min(axis=0)
        pred_max = pred.max(axis=0)
        dx = np.maximum.reduce(
            (gt_min[:, 0] - pred_max[0], pred_min[0] - gt_max[:, 0], np.zeros(len(gt_lines)))
        )
        dy = np.maximum.reduce(
            (gt_min[:, 1] - pred_max[1], pred_min[1] - gt_max[:, 1], np.zeros(len(gt_lines)))
        )
        candidates = np.flatnonzero(np.hypot(dx, dy) <= max_relevant_distance)
        if not len(candidates):
            continue
        delta = (
            pred[None, :, None, :]
            - gt_lines[candidates, None, :, :]
        )
        pair_distance = np.linalg.norm(delta, axis=-1)
        pred_to_gt = pair_distance.min(axis=2).mean(axis=1)
        gt_to_pred = pair_distance.min(axis=1).mean(axis=1)
        result[pred_idx, candidates] = 0.5 * (pred_to_gt + gt_to_pred)
    return result


def _match_one_scene(
    scores: np.ndarray,
    distances: np.ndarray,
    thresholds: Sequence[float],
    *,
    inclusive: bool,
    fallback_to_unmatched: bool,
) -> np.ndarray:
    """Greedy confidence-ordered 1:1 matching for all thresholds."""

    scores = np.asarray(scores, dtype=np.float32)
    distances = np.asarray(distances, dtype=np.float32)
    if distances.ndim != 2 or distances.shape[0] != len(scores):
        raise ValueError(
            f"distance matrix must be [predictions,GT], got {distances.shape} "
            f"for {len(scores)} prediction score(s)"
        )
    _require_finite("prediction scores", scores)
    # Inf is the intended sentinel for safely pruned map pairs.
    if np.isnan(distances).any():
        raise ValueError("distance matrix contains NaN")
    num_gt = distances.shape[1]
    tp = np.zeros((len(scores), len(thresholds)), dtype=np.uint8)
    prediction_order = np.lexsort((np.arange(len(scores)), -scores))
    for threshold_idx, threshold in enumerate(thresholds):
        matched = np.zeros(num_gt, dtype=bool)
        for pred_idx in prediction_order:
            if not num_gt:
                break
            if fallback_to_unmatched:
                # nuScenes detection searches for the closest *untaken* GT,
                # allowing a later prediction to fall back to its second-best
                # object after the nearest one was claimed.
                candidates = np.flatnonzero(~matched)
                if not len(candidates):
                    break
                local = int(np.argmin(distances[pred_idx, candidates]))
                gt_idx = int(candidates[local])
            else:
                # VAD/MapTR's custom_tpfp_gen fixes each prediction to its
                # closest GT first.  A duplicate is an FP; it does not fall
                # back to another, farther unmatched vector.
                gt_idx = int(np.argmin(distances[pred_idx]))
            distance = float(distances[pred_idx, gt_idx])
            is_match = distance <= threshold if inclusive else distance < threshold
            if is_match and not matched[gt_idx]:
                matched[gt_idx] = True
                tp[pred_idx, threshold_idx] = 1
    return tp


def _nuscenes_ap(
    scores: np.ndarray,
    tp: np.ndarray,
    num_gt: int,
    *,
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> tuple[float, float]:
    """nuScenes-style 101-bin detection AP with a deterministic tie order."""

    if num_gt <= 0:
        raise ValueError("nuScenes AP requires at least one GT instance")
    if len(scores) == 0 or int(np.asarray(tp).sum()) == 0:
        return 0.0, 0.0
    scores = np.asarray(scores, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.uint8)
    order = np.lexsort((np.arange(len(scores)), -scores))
    tp_cumulative = np.cumsum(tp[order], dtype=np.float64)
    fp_cumulative = np.cumsum(1 - tp[order], dtype=np.float64)
    recall = tp_cumulative / float(num_gt)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1.0)
    recall_grid = np.linspace(0.0, 1.0, 101)
    interpolated = np.interp(recall_grid, recall, precision, right=0.0)
    clipped = interpolated[round(100 * min_recall) + 1 :].copy()
    clipped -= min_precision
    clipped[clipped < 0] = 0
    ap = float(clipped.mean() / (1.0 - min_precision))
    return ap, float(recall[-1])


def _area_ap(scores: np.ndarray, tp: np.ndarray, num_gt: int) -> tuple[float, float]:
    """All-point precision-envelope area AP used by the legacy map evaluator."""

    if num_gt <= 0:
        raise ValueError("area AP requires at least one GT instance")
    if len(scores) == 0:
        return 0.0, 0.0
    scores = np.asarray(scores, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.uint8)
    order = np.lexsort((np.arange(len(scores)), -scores))
    tp_cumulative = np.cumsum(tp[order], dtype=np.float64)
    fp_cumulative = np.cumsum(1 - tp[order], dtype=np.float64)
    recall = tp_cumulative / float(num_gt)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1.0)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    changing = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))
    return ap, float(recall[-1]) if len(recall) else 0.0


@dataclass
class _TaskAccumulator:
    class_names: Sequence[str]
    thresholds: Sequence[float]
    ap_kind: str
    inclusive: bool
    fallback_to_unmatched: bool

    def __post_init__(self) -> None:
        self.thresholds = _validate_thresholds(self.ap_kind, self.thresholds)
        self.scores: List[List[np.ndarray]] = [[] for _ in self.class_names]
        self.true_positives: List[List[np.ndarray]] = [[] for _ in self.class_names]
        self.num_gt = np.zeros(len(self.class_names), dtype=np.int64)
        self.num_predictions = np.zeros(len(self.class_names), dtype=np.int64)

    def update(
        self,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        distance_builder,
    ) -> None:
        pred_scores = np.asarray(pred_scores, dtype=np.float32)
        pred_labels = np.asarray(pred_labels, dtype=np.int64)
        gt_labels = np.asarray(gt_labels, dtype=np.int64)
        if pred_scores.shape != pred_labels.shape:
            raise ValueError("prediction scores and labels must have equal shape")
        if pred_scores.ndim != 1:
            raise ValueError(
                f"prediction scores and labels must be one-dimensional, got {pred_scores.shape}"
            )
        if gt_labels.ndim != 1:
            raise ValueError(f"GT labels must be one-dimensional, got {gt_labels.shape}")
        if np.any((pred_labels < 0) | (pred_labels >= len(self.class_names))):
            raise ValueError("prediction label is outside the configured class range")
        if np.any((gt_labels < 0) | (gt_labels >= len(self.class_names))):
            raise ValueError("GT label is outside the configured class range")
        _require_finite("prediction scores", pred_scores)
        if np.any((pred_scores < 0.0) | (pred_scores > 1.0)):
            raise ValueError("prediction scores must be sigmoid probabilities in [0, 1]")

        for class_idx in range(len(self.class_names)):
            pred_indices = np.flatnonzero(pred_labels == class_idx)
            gt_indices = np.flatnonzero(gt_labels == class_idx)
            class_scores = pred_scores[pred_indices]
            distances = distance_builder(pred_indices, gt_indices)
            if distances.shape != (len(pred_indices), len(gt_indices)):
                raise ValueError(
                    f"distance builder returned {distances.shape}, expected "
                    f"{(len(pred_indices), len(gt_indices))}"
                )
            tp = _match_one_scene(
                class_scores,
                distances,
                self.thresholds,
                inclusive=self.inclusive,
                fallback_to_unmatched=self.fallback_to_unmatched,
            )
            self.scores[class_idx].append(class_scores)
            self.true_positives[class_idx].append(tp)
            self.num_gt[class_idx] += len(gt_indices)
            self.num_predictions[class_idx] += len(pred_indices)

    def compute(self) -> Dict[str, object]:
        class_results: Dict[str, Dict[str, object]] = {}
        all_aps: List[float] = []
        aps_by_threshold: List[List[float]] = [list() for _ in self.thresholds]
        for class_idx, class_name in enumerate(self.class_names):
            scores = (
                np.concatenate(self.scores[class_idx])
                if self.scores[class_idx]
                else np.zeros(0, dtype=np.float32)
            )
            tp = (
                np.concatenate(self.true_positives[class_idx], axis=0)
                if self.true_positives[class_idx]
                else np.zeros((0, len(self.thresholds)), dtype=np.uint8)
            )
            threshold_results: Dict[str, Dict[str, float]] = {}
            class_aps: List[float] = []
            for threshold_idx, threshold in enumerate(self.thresholds):
                key = f"{float(threshold):g}"
                if int(self.num_gt[class_idx]) == 0:
                    threshold_results[key] = {"AP": None, "max_recall": None}
                    continue
                if self.ap_kind == "nuscenes_101":
                    ap, max_recall = _nuscenes_ap(
                        scores, tp[:, threshold_idx], int(self.num_gt[class_idx])
                    )
                elif self.ap_kind == "precision_envelope_area":
                    ap, max_recall = _area_ap(
                        scores, tp[:, threshold_idx], int(self.num_gt[class_idx])
                    )
                else:
                    raise ValueError(f"unsupported AP kind: {self.ap_kind}")
                threshold_results[key] = {"AP": ap, "max_recall": max_recall}
                class_aps.append(ap)
                aps_by_threshold[threshold_idx].append(ap)
                all_aps.append(ap)
            class_results[class_name] = {
                "num_gt": int(self.num_gt[class_idx]),
                "num_predictions": int(self.num_predictions[class_idx]),
                "AP": float(np.mean(class_aps)) if class_aps else None,
                "thresholds": threshold_results,
            }

        return {
            "mAP": float(np.mean(all_aps)) if all_aps else None,
            "mAP_by_threshold": {
                f"{float(threshold):g}": (
                    float(np.mean(aps_by_threshold[idx]))
                    if aps_by_threshold[idx]
                    else None
                )
                for idx, threshold in enumerate(self.thresholds)
            },
            "classes": class_results,
        }


def evaluate_auxiliary_records(
    records: Iterable[Mapping[str, np.ndarray]],
    *,
    det_thresholds: Sequence[float] = DEFAULT_DET_THRESHOLDS,
    map_thresholds: Sequence[float] = DEFAULT_MAP_THRESHOLDS,
    map_resample_points: int = 100,
) -> Dict[str, object]:
    """Aggregate exact token-keyed records into detection and map mAP.

    Records must be supplied in strictly increasing token order.  This makes
    tied confidence scores deterministic without retaining millions of token
    strings in memory and lets the full evaluation stream compressed shards.
    """

    if (
        isinstance(map_resample_points, bool)
        or not isinstance(map_resample_points, (int, np.integer))
        or map_resample_points < 2
    ):
        raise ValueError(
            f"map_resample_points must be an integer >= 2, got {map_resample_points}"
        )
    det_thresholds = _validate_thresholds("detection", det_thresholds)
    map_thresholds = _validate_thresholds("map", map_thresholds)

    det_accumulator = _TaskAccumulator(
        DET_CLASS_NAMES,
        tuple(det_thresholds),
        "nuscenes_101",
        inclusive=False,
        fallback_to_unmatched=True,
    )
    map_accumulator = _TaskAccumulator(
        MAP_CLASS_NAMES,
        tuple(map_thresholds),
        "precision_envelope_area",
        inclusive=True,
        fallback_to_unmatched=False,
    )
    previous_token = None
    num_tokens = 0
    scenes_over_map_training_cap = 0
    for record in records:
        token = str(record["token"])
        if not token:
            raise ValueError("record token must be non-empty")
        if previous_token is not None and token <= previous_token:
            raise ValueError(
                "records must have unique, strictly increasing tokens; got "
                f"{token!r} after {previous_token!r}"
            )
        previous_token = token
        num_tokens += 1

        det_pred_boxes = np.asarray(record["det_pred_boxes"], dtype=np.float32)
        det_pred_scores = np.asarray(record["det_pred_scores"], dtype=np.float32)
        det_pred_labels = np.asarray(record["det_pred_labels"], dtype=np.int64)
        det_gt_boxes = np.asarray(record["det_gt_boxes"], dtype=np.float32)
        det_gt_labels = np.asarray(record["det_gt_labels"], dtype=np.int64)
        if det_pred_boxes.ndim != 2 or det_pred_boxes.shape[1] != 9:
            raise ValueError(f"det_pred_boxes must be [N,9], got {det_pred_boxes.shape}")
        if det_gt_boxes.ndim != 2 or det_gt_boxes.shape[1] != 9:
            raise ValueError(f"det_gt_boxes must be [N,9], got {det_gt_boxes.shape}")
        if det_pred_scores.ndim != 1 or det_pred_labels.ndim != 1:
            raise ValueError("detection prediction scores/labels must be one-dimensional")
        if det_gt_labels.ndim != 1:
            raise ValueError("detection GT labels must be one-dimensional")
        if not (
            len(det_pred_boxes) == len(det_pred_scores) == len(det_pred_labels)
        ):
            raise ValueError("detection prediction boxes/scores/labels count mismatch")
        if len(det_gt_boxes) != len(det_gt_labels):
            raise ValueError("detection GT boxes/labels count mismatch")
        if len(det_pred_boxes) > 100:
            raise ValueError(
                f"detection record exceeds the top-100 protocol: {len(det_pred_boxes)}"
            )
        _require_finite("detection prediction boxes", det_pred_boxes)
        _require_finite("detection GT boxes", det_gt_boxes)
        if np.any(det_pred_boxes[:, 3:6] <= 0) or np.any(det_gt_boxes[:, 3:6] <= 0):
            raise ValueError("detection boxes must have positive width/length/height")

        det_accumulator.update(
            det_pred_scores,
            det_pred_labels,
            det_gt_labels,
            lambda pred_idx, gt_idx: center_distance_matrix(
                det_pred_boxes[pred_idx], det_gt_boxes[gt_idx]
            ),
        )

        map_pred_points = np.asarray(record["map_pred_points"], dtype=np.float32)
        map_pred_scores = np.asarray(record["map_pred_scores"], dtype=np.float32)
        map_pred_labels = np.asarray(record["map_pred_labels"], dtype=np.int64)
        map_gt_points = np.asarray(record["map_gt_points"], dtype=np.float32)
        map_gt_labels = np.asarray(record["map_gt_labels"], dtype=np.int64)
        if (
            map_pred_points.ndim != 3
            or map_pred_points.shape[-1] != 2
            or map_pred_points.shape[1] < 1
        ):
            raise ValueError(
                f"map_pred_points must be [N,points,2] with points >= 1, got "
                f"{map_pred_points.shape}"
            )
        if (
            map_gt_points.ndim != 3
            or map_gt_points.shape[-1] != 2
            or map_gt_points.shape[1] < 1
        ):
            raise ValueError(
                f"map_gt_points must be [N,points,2] with points >= 1, got "
                f"{map_gt_points.shape}"
            )
        if map_pred_scores.ndim != 1 or map_pred_labels.ndim != 1:
            raise ValueError("map prediction scores/labels must be one-dimensional")
        if map_gt_labels.ndim != 1:
            raise ValueError("map GT labels must be one-dimensional")
        if not (
            len(map_pred_points) == len(map_pred_scores) == len(map_pred_labels)
        ):
            raise ValueError("map prediction points/scores/labels count mismatch")
        if len(map_gt_points) != len(map_gt_labels):
            raise ValueError("map GT points/labels count mismatch")
        if len(map_pred_points) > 100:
            raise ValueError(
                f"map record exceeds the top-100 protocol: {len(map_pred_points)}"
            )
        _require_finite("map prediction points", map_pred_points)
        _require_finite("map GT points", map_gt_points)
        if len(map_gt_points) > 100:
            scenes_over_map_training_cap += 1
        resampled_predictions = _resample_polylines(
            map_pred_points, map_resample_points
        )
        resampled_gt = _resample_polylines(map_gt_points, map_resample_points)
        max_map_threshold = float(max(map_thresholds))
        map_accumulator.update(
            map_pred_scores,
            map_pred_labels,
            map_gt_labels,
            lambda pred_idx, gt_idx: chamfer_distance_matrix(
                resampled_predictions[pred_idx],
                resampled_gt[gt_idx],
                max_relevant_distance=max_map_threshold,
            ),
        )

    if num_tokens == 0:
        raise ValueError("cannot evaluate an empty record set")
    return {
        "protocol_version": AUX_METRIC_PROTOCOL_VERSION,
        "num_tokens": num_tokens,
        "detection": {
            "name": "NAVSIMAuxDet/center_mAP",
            "official_navsim_metric": False,
            "decode": "sigmoid_flattened_query_class_top100",
            "matching": "class-aware_2d_center_greedy_strict_lt",
            "ap": "nuscenes_101_min_recall_0.1_min_precision_0.1",
            "thresholds_m": [float(value) for value in det_thresholds],
            **det_accumulator.compute(),
        },
        "map": {
            "name": "NAVSIMAuxMap/chamfer_mAP",
            "official_navsim_metric": False,
            "decode": "sigmoid_flattened_query_class_top100",
            "matching": "class-aware_symmetric_chamfer_greedy_lte",
            "ap": "precision_envelope_area",
            "thresholds_m": [float(value) for value in map_thresholds],
            "resample_points": int(map_resample_points),
            "scenes_over_training_gt_cap_100": scenes_over_map_training_cap,
            **map_accumulator.compute(),
        },
    }
