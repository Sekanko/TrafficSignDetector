"""Build a tiny synthetic YOLO detection dataset for local smoke tests.

Use this when Kaggle credentials or raw downloads are not available yet, but
you still want to verify the detection training pipeline end-to-end.

Usage:
    python -m src.data_prep.build_smoke_detection
"""

from __future__ import annotations

import logging
import random
import shutil

import yaml
from PIL import Image, ImageDraw

from src.data_prep.paths import DETECTION_DIR

logger = logging.getLogger(__name__)

SPLITS = {
    "train": 6,
    "val": 2,
    "test": 2,
}
IMAGE_SIZE = (640, 480)
SEED = 67


def _write_sample(image_path, label_path, *, seed: int) -> None:
    rng = random.Random(seed)
    image = Image.new("RGB", IMAGE_SIZE, (rng.randint(40, 90), rng.randint(40, 90), rng.randint(40, 90)))
    draw = ImageDraw.Draw(image)

    sign_width = rng.randint(48, 96)
    sign_height = rng.randint(48, 96)
    x1 = rng.randint(40, IMAGE_SIZE[0] - sign_width - 40)
    y1 = rng.randint(40, IMAGE_SIZE[1] - sign_height - 40)
    x2 = x1 + sign_width
    y2 = y1 + sign_height
    color = (rng.randint(180, 255), rng.randint(20, 80), rng.randint(20, 80))
    draw.rectangle((x1, y1, x2, y2), fill=color, outline="white", width=3)

    cx = ((x1 + x2) / 2) / IMAGE_SIZE[0]
    cy = ((y1 + y2) / 2) / IMAGE_SIZE[1]
    width = sign_width / IMAGE_SIZE[0]
    height = sign_height / IMAGE_SIZE[1]
    image.save(image_path, quality=95)
    label_path.write_text(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n", encoding="utf-8")


def build_smoke_detection() -> None:
    if DETECTION_DIR.exists():
        shutil.rmtree(DETECTION_DIR)
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)

    sample_index = 0
    for split, count in SPLITS.items():
        image_dir = DETECTION_DIR / "images" / split
        label_dir = DETECTION_DIR / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(count):
            stem = f"smoke_{sample_index:03d}"
            _write_sample(
                image_dir / f"{stem}.jpg",
                label_dir / f"{stem}.txt",
                seed=SEED + sample_index,
            )
            sample_index += 1

    dataset_yaml = {
        "path": str(DETECTION_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "traffic_sign"},
    }
    with open(DETECTION_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

    logger.info("Wrote smoke detection dataset to %s", DETECTION_DIR)
    for split, count in SPLITS.items():
        logger.info("  %-5s %d images", split, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_smoke_detection()
