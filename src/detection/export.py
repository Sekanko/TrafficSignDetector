"""Model export helpers for mobile deployment."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from ultralytics import YOLO

from src.detection.utils import project_relative


def export_tflite(model: YOLO, args: Namespace, dataset_yaml: Path) -> str | None:
    if not args.export_tflite:
        return None

    export_kwargs = {
        "format": "tflite",
        "imgsz": args.img_size,
        "int8": args.tflite_int8,
    }
    if args.tflite_int8:
        export_kwargs["data"] = str(dataset_yaml)

    exported_path = model.export(**export_kwargs)
    return project_relative(exported_path)
