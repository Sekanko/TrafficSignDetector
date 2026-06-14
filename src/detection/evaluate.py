"""Class-agnostic localization metrics for one-class detector comparisons."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.detection.dataset import load_ground_truth, split_image_paths
from src.detection.predict import predict_boxes
from src.detection.types import Box, EvaluationMetrics, Prediction


def box_iou(a: Box, b: Box) -> float:
    intersection_x1 = max(a.x1, b.x1)
    intersection_y1 = max(a.y1, b.y1)
    intersection_x2 = min(a.x2, b.x2)
    intersection_y2 = min(a.y2, b.y2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def average_precision(precision: list[float], recall: list[float]) -> float:
    if not precision:
        return 0.0

    mrec = [0.0, *recall, 1.0]
    mpre = [0.0, *precision, 0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def precision_recall_ap(
    predictions: dict[Path, list[Prediction]],
    ground_truths: dict[Path, list[Box]],
    *,
    iou_threshold: float,
) -> tuple[float, float, float]:
    total_ground_truths = sum(len(boxes) for boxes in ground_truths.values())
    if total_ground_truths == 0:
        return 0.0, 0.0, 0.0

    ordered_predictions = sorted(
        (
            (image_path, prediction)
            for image_path, image_predictions in predictions.items()
            for prediction in image_predictions
        ),
        key=lambda item: item[1].confidence,
        reverse=True,
    )
    matched: dict[Path, set[int]] = {image_path: set() for image_path in ground_truths}
    true_positive = []
    false_positive = []

    for image_path, prediction in ordered_predictions:
        image_ground_truths = ground_truths.get(image_path, [])
        best_iou = 0.0
        best_index = -1
        for index, ground_truth in enumerate(image_ground_truths):
            if index in matched[image_path]:
                continue
            iou = box_iou(prediction.box, ground_truth)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_iou >= iou_threshold and best_index >= 0:
            matched[image_path].add(best_index)
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)

    cumulative_tp = 0
    cumulative_fp = 0
    precision_curve = []
    recall_curve = []
    for tp, fp in zip(true_positive, false_positive):
        cumulative_tp += tp
        cumulative_fp += fp
        precision_curve.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1))
        recall_curve.append(cumulative_tp / total_ground_truths)

    precision = precision_curve[-1] if precision_curve else 0.0
    recall = recall_curve[-1] if recall_curve else 0.0
    return precision, recall, average_precision(precision_curve, recall_curve)


def evaluate_predictions(
    predictions: dict[Path, list[Prediction]],
    ground_truths: dict[Path, list[Box]],
) -> EvaluationMetrics:
    precision50, recall50, map50 = precision_recall_ap(predictions, ground_truths, iou_threshold=0.5)
    f1 = 0.0 if precision50 + recall50 == 0 else 2 * precision50 * recall50 / (precision50 + recall50)
    thresholds = [threshold / 100 for threshold in range(50, 100, 5)]
    aps = [
        precision_recall_ap(predictions, ground_truths, iou_threshold=threshold)[2]
        for threshold in thresholds
    ]
    return EvaluationMetrics(
        images=len(ground_truths),
        ground_truth_boxes=sum(len(boxes) for boxes in ground_truths.values()),
        predicted_boxes=sum(len(boxes) for boxes in predictions.values()),
        precision_at_50=precision50,
        recall_at_50=recall50,
        f1_at_50=f1,
        map50=map50,
        map50_95=sum(aps) / len(aps),
    )


def evaluate_model(
    model: YOLO,
    *,
    split: str,
    img_size: int,
    conf: float,
    max_det: int,
    device: str | None,
    batch_size: int,
) -> tuple[EvaluationMetrics, dict[Path, list[Prediction]], dict[Path, list[Box]]]:
    image_paths = split_image_paths(split)
    ground_truths = {path: load_ground_truth(path, split) for path in image_paths}
    predictions = predict_boxes(
        model,
        image_paths,
        img_size=img_size,
        conf=conf,
        max_det=max_det,
        device=device,
        batch_size=batch_size,
    )
    return evaluate_predictions(predictions, ground_truths), predictions, ground_truths


def metric_delta(fine_tuned: EvaluationMetrics, baseline: EvaluationMetrics) -> dict[str, float]:
    return {
        "precision_at_50": fine_tuned.precision_at_50 - baseline.precision_at_50,
        "recall_at_50": fine_tuned.recall_at_50 - baseline.recall_at_50,
        "f1_at_50": fine_tuned.f1_at_50 - baseline.f1_at_50,
        "map50": fine_tuned.map50 - baseline.map50,
        "map50_95": fine_tuned.map50_95 - baseline.map50_95,
    }
