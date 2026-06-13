"""Helpers for the generated YOLO detection dataset."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data_prep.paths import DETECTION_DIR
from src.detection.constants import IMAGE_EXTENSIONS
from src.detection.types import Box


def _split_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        image_dir = DETECTION_DIR / "images" / split
        if not image_dir.exists():
            counts[split] = 0
            continue
        counts[split] = sum(
            1 for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts


def ensure_dataset_exists() -> Path:
    dataset_yaml = DETECTION_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Detection dataset not found: {dataset_yaml}. "
            "Run `python -m src.data_prep.build_detection` first, "
            "or `python -m src.data_prep.build_smoke_detection` for a local smoke test."
        )

    counts = _split_counts()
    if sum(counts.values()) == 0:
        raise FileNotFoundError(
            f"Detection dataset at {DETECTION_DIR} has dataset.yaml but no images. "
            "The processed folder looks incomplete. "
            "Run `python -m src.data_prep.build_detection` after downloading raw data, "
            "or `python -m src.data_prep.build_smoke_detection` for a local smoke test."
        )
    if counts.get("test", 0) == 0:
        missing = ", ".join(f"{split}={count}" for split, count in counts.items())
        raise FileNotFoundError(
            f"Detection test split is empty ({missing}). "
            "Rebuild the dataset with `python -m src.data_prep.build_detection` "
            "or `python -m src.data_prep.build_smoke_detection`."
        )
    return dataset_yaml


def split_image_paths(split: str) -> list[Path]:
    image_dir = DETECTION_DIR / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Detection split directory not found: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for(image_path: Path, split: str) -> Path:
    return DETECTION_DIR / "labels" / split / f"{image_path.stem}.txt"


def load_ground_truth(image_path: Path, split: str) -> list[Box]:
    label_path = label_path_for(image_path, split)
    if not label_path.exists():
        return []

    with Image.open(image_path) as image:
        width, height = image.size

    boxes = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = parts
        center_x = float(cx) * width
        center_y = float(cy) * height
        box_width = float(w) * width
        box_height = float(h) * height
        boxes.append(
            Box(
                x1=center_x - box_width / 2,
                y1=center_y - box_height / 2,
                x2=center_x + box_width / 2,
                y2=center_y + box_height / 2,
            )
        )
    return boxes
