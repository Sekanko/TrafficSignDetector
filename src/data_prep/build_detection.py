"""Builds data/processed/detection/ (YOLO format) from GTSDB and Polish.

Copies each detection image to data/processed/detection/images/<split>/ (as
JPEG, regardless of source format) and writes a matching YOLO label file to
data/processed/detection/labels/<split>/ - one line per box:
`0 cx cy w h` (normalized). The detector is single-class ("traffic_sign"):
it only localizes signs, while the separate classifier (see
build_classification.py) determines the fine-grained type. This lets
sources with different label granularity (GTSDB's per-box class vs Polish's
generic "sign present" boxes) combine into one dataset.

Also writes detection/dataset.yaml in Ultralytics YOLO format.

Usage:
    python -m src.data_prep.build_detection
"""

from __future__ import annotations

import itertools
import logging
import shutil

import yaml
from PIL import Image

from src.data_prep.adapters.gtsdb import GtsdbAdapter
from src.data_prep.adapters.polish import PolishAdapter
from src.data_prep.paths import DETECTION_DIR
from src.data_prep.taxonomy import load_taxonomy

logger = logging.getLogger(__name__)


def build_detection() -> None:
    taxonomy = load_taxonomy()
    gtsdb = GtsdbAdapter(taxonomy)
    polish = PolishAdapter(taxonomy)

    if DETECTION_DIR.exists():
        shutil.rmtree(DETECTION_DIR)
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for item in itertools.chain(gtsdb.detection_items(), polish.detection_items()):
        image_dir = DETECTION_DIR / "images" / item.split
        label_dir = DETECTION_DIR / "labels" / item.split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{item.source}_{item.image_path.stem}"

        with Image.open(item.image_path) as img:
            img.convert("RGB").save(image_dir / f"{stem}.jpg", quality=95)

        lines = []
        for bbox, _taxonomy_id in item.boxes:
            cx, cy, w, h = bbox.to_yolo(item.width, item.height)
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[item.split] = counts.get(item.split, 0) + 1

    dataset_yaml = {
        "path": str(DETECTION_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "traffic_sign"},
    }
    with open(DETECTION_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

    total_images = sum(counts.values())
    if total_images == 0:
        raise FileNotFoundError(
            f"No detection images were written to {DETECTION_DIR}. "
            "Download raw datasets first with "
            "`python -m src.data_prep.download --dataset gtsdb polish`, "
            "or generate a tiny smoke dataset with "
            "`python -m src.data_prep.build_smoke_detection`."
        )

    logger.info("Wrote %d detection images to %s", total_images, DETECTION_DIR)
    for split, count in sorted(counts.items()):
        logger.info("  %-5s %d images", split, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_detection()
