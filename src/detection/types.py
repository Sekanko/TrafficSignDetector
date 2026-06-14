"""Small value objects used by detection evaluation and visualization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Prediction:
    box: Box
    confidence: float


@dataclass(frozen=True)
class EvaluationMetrics:
    images: int
    ground_truth_boxes: int
    predicted_boxes: int
    precision_at_50: float
    recall_at_50: float
    f1_at_50: float
    map50: float
    map50_95: float
