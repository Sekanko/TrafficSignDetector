"""Command-line options for YOLO detector fine-tuning."""

from __future__ import annotations

import argparse

from src.detection.constants import SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model for traffic sign detection")
    parser.add_argument("--base-model", default="yolo11n.pt", help="YOLO checkpoint used as the fine-tuning start point")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. '0', 'cpu'")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--compare-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for visualizations")
    parser.add_argument("--eval-conf", type=float, default=0.001, help="Low confidence threshold used for AP calculation")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument("--tflite-int8", action="store_true")
    return parser.parse_args()
