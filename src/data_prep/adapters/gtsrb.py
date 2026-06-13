"""Adapter for GTSRB - German Traffic Sign Recognition Benchmark
(classification only).

https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

import pandas as pd

from src.data_prep.adapters.base import ClassificationItem, DatasetAdapter
from src.data_prep.bbox import BBox
from src.data_prep.config import load_dataset_config
from src.data_prep.paths import DATA_RAW
from src.data_prep.splits import assign_split
from src.data_prep.taxonomy import Taxonomy, load_taxonomy

logger = logging.getLogger(__name__)

# Train.csv / Test.csv use "Roi.X1" etc., which aren't valid Python
# identifiers - rename to access via itertuples().
_ROI_COLUMNS = {"Roi.X1": "roi_x1", "Roi.Y1": "roi_y1", "Roi.X2": "roi_x2", "Roi.Y2": "roi_y2"}

# Train.csv paths look like "Train/<ClassId>/<ClassId>_<TrackId>_<FrameId>.png".
# Each track is ~30 near-duplicate consecutive video frames of the same
# physical sign, so splitting must key on (ClassId, TrackId) - otherwise
# near-identical frames of the same sign end up in both train and val.
_TRACK_PATTERN = re.compile(r"(\d+_\d+)_\d+\.png$")


def _track_key(path: str) -> str:
    match = _TRACK_PATTERN.search(path)
    if match is None:
        raise ValueError(f"Unrecognized GTSRB train path format: {path!r}")
    return match.group(1)


class GtsrbAdapter(DatasetAdapter):
    name = "gtsrb"

    def __init__(self, taxonomy: Taxonomy | None = None):
        self.config = load_dataset_config(self.name)
        self.taxonomy = taxonomy or load_taxonomy()
        self.root = DATA_RAW / self.config["raw_dir"]

    def classification_items(self) -> Iterator[ClassificationItem]:
        yield from self._read_csv("Train.csv", fixed_split=None)
        yield from self._read_csv("Test.csv", fixed_split="test")

    def _read_csv(self, csv_name: str, fixed_split: str | None) -> Iterator[ClassificationItem]:
        csv_path = self.root / csv_name
        if not csv_path.exists():
            logger.warning("%s: %s not found, skipping", self.name, csv_path)
            return

        df = pd.read_csv(csv_path).rename(columns=_ROI_COLUMNS)
        ratios = {"val": self.config["val_fraction"], "train": 1 - self.config["val_fraction"]}
        seed = self.config["split_seed"]

        for row in df.itertuples(index=False):
            taxonomy_class = self.taxonomy.by_gtsrb_id(int(row.ClassId))
            split = fixed_split or assign_split(_track_key(row.Path), ratios, seed)
            bbox = BBox(x1=row.roi_x1, y1=row.roi_y1, x2=row.roi_x2, y2=row.roi_y2)
            yield ClassificationItem(
                image_path=self.root / row.Path,
                taxonomy_id=taxonomy_class.id,
                split=split,
                source=self.name,
                bbox=bbox,
            )
