"""Adapter for the Polish Traffic Signs Dataset.

https://www.kaggle.com/datasets/chriskjm/polish-traffic-signs-dataset

Layout (see configs/datasets/polish.yaml for details):
    classification/<CODE>/*.jpg  - ImageFolder-style crops, <CODE> is the
                                    official Polish road sign code (A1, B2, ...)
    detection/imgs/*.jpg + detection/labels/*.txt - YOLO format, single
                                    class 0 ("sign present", no fine-grained
                                    type).

Classification folders are mapped to taxonomy ids via configs/datasets/
polish.yaml's `id_map`; folders not listed there are skipped with a warning.
Detection boxes have no fine-grained class, so their `taxonomy_id` is `None`
(see adapters/base.py).
"""

from __future__ import annotations

import logging
from typing import Iterator

from PIL import Image

from src.data_prep.adapters.base import ClassificationItem, DatasetAdapter, DetectionItem
from src.data_prep.bbox import BBox
from src.data_prep.config import load_dataset_config
from src.data_prep.paths import DATA_RAW
from src.data_prep.splits import assign_split
from src.data_prep.taxonomy import Taxonomy, load_taxonomy

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class PolishAdapter(DatasetAdapter):
    name = "polish"

    def __init__(self, taxonomy: Taxonomy | None = None):
        self.config = load_dataset_config(self.name)
        self.taxonomy = taxonomy or load_taxonomy()
        self.root = DATA_RAW / self.config["raw_dir"]
        self.id_map: dict[str, int] = dict(self.config.get("id_map", {}))

    def classification_items(self) -> Iterator[ClassificationItem]:
        classification_dir = self.root / "classification"
        if not classification_dir.is_dir():
            logger.warning("%s: %s not found, skipping", self.name, classification_dir)
            return

        ratios = self.config["split_ratios"]
        seed = self.config["split_seed"]
        skipped: list[str] = []

        for folder in sorted(classification_dir.iterdir()):
            if not folder.is_dir():
                continue
            taxonomy_id = self.id_map.get(folder.name)
            if taxonomy_id is None:
                skipped.append(folder.name)
                continue

            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                key = f"{folder.name}/{image_path.name}"
                yield ClassificationItem(
                    image_path=image_path,
                    taxonomy_id=taxonomy_id,
                    split=assign_split(key, ratios, seed),
                    source=self.name,
                    bbox=None,
                )

        if skipped:
            logger.warning(
                "%s: skipped unmapped classification folders (see id_map in "
                "configs/datasets/polish.yaml): %s",
                self.name, sorted(skipped),
            )

    def detection_items(self) -> Iterator[DetectionItem]:
        detection_dir = self.root / "detection"
        images_dir = detection_dir / "imgs"
        labels_dir = detection_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            logger.warning("%s: %s not found, skipping", self.name, detection_dir)
            return

        ratios = self.config["split_ratios"]
        seed = self.config["split_seed"]

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = images_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                logger.warning("%s: no image found for label %s", self.name, label_path)
                continue

            with Image.open(image_path) as img:
                width, height = img.size

            boxes: list[tuple[BBox, int | None]] = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                _class_id, cx, cy, w, h = line.split()
                bbox = BBox.from_yolo(float(cx), float(cy), float(w), float(h), width, height)
                boxes.append((bbox, None))

            yield DetectionItem(
                image_path=image_path,
                width=width,
                height=height,
                boxes=boxes,
                split=assign_split(label_path.stem, ratios, seed),
                source=self.name,
            )
