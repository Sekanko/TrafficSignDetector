"""Builds data/processed/classification/ from all dataset adapters.

For every (image, optional bbox, taxonomy class, split) item produced by the
adapters - including one extra item per GTSDB bounding box, so detection
images also feed the classifier - crops the image to its bbox (expanded by
the source dataset's configured margin), or keeps the full image if no bbox
is given, and saves it under
data/processed/classification/<split>/<taxonomy_name>/.

Also writes manifest.csv listing every output file with its labels, for
traceability and for building a PyTorch dataset without relying on folder
names alone.

Usage:
    python -m src.data_prep.build_classification
"""

from __future__ import annotations

import csv
import itertools
import logging
from pathlib import Path
from typing import Iterator

from PIL import Image

from src.data_prep.adapters.base import ClassificationItem
from src.data_prep.adapters.gtsdb import GtsdbAdapter
from src.data_prep.adapters.gtsrb import GtsrbAdapter
from src.data_prep.adapters.polish import PolishAdapter
from src.data_prep.config import load_dataset_config
from src.data_prep.paths import CLASSIFICATION_DIR
from src.data_prep.taxonomy import Taxonomy, load_taxonomy

logger = logging.getLogger(__name__)


def _gtsdb_crops(adapter: GtsdbAdapter) -> Iterator[ClassificationItem]:
    """Turn each GTSDB detection box into a classification crop."""
    for item in adapter.detection_items():
        for bbox, taxonomy_id in item.boxes:
            yield ClassificationItem(
                image_path=item.image_path,
                taxonomy_id=taxonomy_id,
                split=item.split,
                source=adapter.name,
                bbox=bbox,
            )


def _save_item(item: ClassificationItem, taxonomy: Taxonomy, index: int) -> Path | None:
    taxonomy_class = taxonomy.by_id(item.taxonomy_id)
    out_dir = CLASSIFICATION_DIR / item.split / taxonomy_class.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{item.source}_{item.image_path.stem}_{index}.png"

    try:
        with Image.open(item.image_path) as img:
            if item.bbox is not None:
                margin = load_dataset_config(item.source)["crop_margin"]
                bbox = item.bbox.expand(margin, img.width, img.height)
                img = img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            img.convert("RGB").save(out_path)
    except (OSError, ValueError) as exc:
        logger.warning("Skipping %s: %s", item.image_path, exc)
        return None

    return out_path


def build_classification() -> None:
    taxonomy = load_taxonomy()
    gtsrb = GtsrbAdapter(taxonomy)
    polish = PolishAdapter(taxonomy)
    gtsdb = GtsdbAdapter(taxonomy)

    all_items = itertools.chain(
        gtsrb.classification_items(),
        polish.classification_items(),
        _gtsdb_crops(gtsdb),
    )

    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CLASSIFICATION_DIR / "manifest.csv"

    counts: dict[tuple[str, str], int] = {}
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "taxonomy_id", "class_name", "split", "source"])

        for index, item in enumerate(all_items):
            out_path = _save_item(item, taxonomy, index)
            if out_path is None:
                continue

            taxonomy_class = taxonomy.by_id(item.taxonomy_id)
            writer.writerow([
                out_path.relative_to(CLASSIFICATION_DIR),
                item.taxonomy_id,
                taxonomy_class.name,
                item.split,
                item.source,
            ])
            key = (item.split, taxonomy_class.name)
            counts[key] = counts.get(key, 0) + 1

    logger.info("Wrote %d classification images to %s", sum(counts.values()), CLASSIFICATION_DIR)
    for (split, name), count in sorted(counts.items()):
        logger.info("  %-5s %-40s %d", split, name, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_classification()
