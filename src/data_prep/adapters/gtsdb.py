"""Adapter for GTSDB - German Traffic Sign Detection Benchmark (detection
only - see build_detection.py).

Its bounding-box crops cover the same 43 classes as GTSRB and largely
duplicate images already in GTSRB (both come from the same underlying
recordings), so GTSDB is not used as a classification source.

Kaggle mirrors of GTSDB vary in layout, so this adapter auto-detects between:

- Raw INI layout (the original benchmark format): a `gt.txt` file
  (";"-separated: `file;x1;y1;x2;y2;classid`) next to `*.ppm` images.
- Pre-converted YOLO layout: `images/*` + `labels/*.txt` (normalized
  `class cx cy w h`), where `class` is the original GTSRB/GTSDB class id
  (0-42).

If a different mirror's layout doesn't match either of these, only this
adapter needs to change - everything downstream (taxonomy, build scripts)
stays the same.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from PIL import Image

from src.data_prep.adapters.base import DatasetAdapter, DetectionItem
from src.data_prep.bbox import BBox
from src.data_prep.config import load_dataset_config
from src.data_prep.paths import DATA_RAW
from src.data_prep.splits import assign_split
from src.data_prep.taxonomy import Taxonomy, load_taxonomy

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm")


class GtsdbAdapter(DatasetAdapter):
    name = "gtsdb"

    def __init__(self, taxonomy: Taxonomy | None = None):
        self.config = load_dataset_config(self.name)
        self.taxonomy = taxonomy or load_taxonomy()
        self.root = DATA_RAW / self.config["raw_dir"]

    def detection_items(self) -> Iterator[DetectionItem]:
        gt_file = self._find_gt_file()
        if gt_file is not None:
            yield from self._iter_raw(gt_file)
            return

        if (self.root / "images").is_dir() and (self.root / "labels").is_dir():
            yield from self._iter_yolo()
            return

        logger.warning("%s: no recognised layout found under %s", self.name, self.root)

    def _find_gt_file(self) -> Path | None:
        if not self.root.exists():
            return None
        candidates = sorted(self.root.rglob("gt.txt"))
        if not candidates:
            return None

        # Kaggle mirrors sometimes ship gt.txt both at the dataset root and
        # alongside the images it describes - pick the one whose first
        # referenced image actually exists next to it.
        for candidate in candidates:
            lines = candidate.read_text(encoding="utf-8").splitlines()
            if lines and (candidate.parent / lines[0].split(";")[0]).exists():
                return candidate

        logger.warning("%s: multiple gt.txt found, using %s", self.name, candidates[0])
        return candidates[0]

    def _iter_raw(self, gt_file: Path) -> Iterator[DetectionItem]:
        image_dir = gt_file.parent
        boxes_by_image: dict[str, list[tuple[BBox, int]]] = {}

        for line in gt_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            filename, x1, y1, x2, y2, class_id = line.split(";")
            taxonomy_class = self.taxonomy.by_gtsrb_id(int(class_id))
            bbox = BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
            boxes_by_image.setdefault(filename, []).append((bbox, taxonomy_class.id))

        ratios = self.config["split_ratios"]
        seed = self.config["split_seed"]
        for filename, boxes in boxes_by_image.items():
            image_path = image_dir / filename
            with Image.open(image_path) as img:
                width, height = img.size
            yield DetectionItem(
                image_path=image_path,
                width=width,
                height=height,
                boxes=boxes,
                split=assign_split(filename, ratios, seed),
                source=self.name,
            )

    def _iter_yolo(self) -> Iterator[DetectionItem]:
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"
        ratios = self.config["split_ratios"]
        seed = self.config["split_seed"]

        for label_path in sorted(labels_dir.rglob("*.txt")):
            image_path = self._find_image(images_dir, label_path.stem)
            if image_path is None:
                logger.warning("%s: no image found for label %s", self.name, label_path)
                continue

            with Image.open(image_path) as img:
                width, height = img.size

            boxes: list[tuple[BBox, int]] = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                class_id, cx, cy, w, h = line.split()
                taxonomy_class = self.taxonomy.by_gtsrb_id(int(class_id))
                bbox = BBox.from_yolo(float(cx), float(cy), float(w), float(h), width, height)
                boxes.append((bbox, taxonomy_class.id))

            yield DetectionItem(
                image_path=image_path,
                width=width,
                height=height,
                boxes=boxes,
                split=assign_split(label_path.stem, ratios, seed),
                source=self.name,
            )

    @staticmethod
    def _find_image(images_dir: Path, stem: str) -> Path | None:
        for ext in _IMAGE_EXTENSIONS:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        matches = list(images_dir.rglob(f"{stem}.*"))
        return matches[0] if matches else None
