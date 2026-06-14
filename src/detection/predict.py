"""Prediction helpers for Ultralytics detection models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

from src.detection.types import Box, Prediction


def batched(items: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def predict_boxes(
    model: YOLO,
    image_paths: list[Path],
    *,
    img_size: int,
    conf: float,
    max_det: int,
    device: str | None,
    batch_size: int,
) -> dict[Path, list[Prediction]]:
    predictions: dict[Path, list[Prediction]] = {path: [] for path in image_paths}
    for batch in batched(image_paths, max(1, batch_size)):
        predict_kwargs = {
            "source": [str(path) for path in batch],
            "imgsz": img_size,
            "conf": conf,
            "iou": 0.7,
            "max_det": max_det,
            "verbose": False,
        }
        if device is not None:
            predict_kwargs["device"] = device
        results = model.predict(**predict_kwargs)

        for image_path, result in zip(batch, results):
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            predictions[image_path] = [
                Prediction(
                    box=Box(x1=float(coords[0]), y1=float(coords[1]), x2=float(coords[2]), y2=float(coords[3])),
                    confidence=float(confidence),
                )
                for coords, confidence in zip(xyxy, confidences)
            ]
    return predictions
