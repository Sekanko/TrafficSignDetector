"""Shared types implemented by each dataset adapter."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.data_prep.bbox import BBox


@dataclass(frozen=True)
class ClassificationItem:
    """A single image (or region of an image) for the classification dataset."""

    image_path: Path
    taxonomy_id: int
    split: str  # "train" | "val" | "test"
    source: str  # dataset name, used as the output filename prefix
    bbox: BBox | None = None  # region to crop; None = use the full image


@dataclass(frozen=True)
class DetectionItem:
    """A single image with one or more annotated sign bounding boxes."""

    image_path: Path
    width: int
    height: int
    # (bbox, taxonomy_id); taxonomy_id is None if the source dataset only
    # marks "sign present" without a fine-grained class (e.g. Polish
    # detection labels).
    boxes: list[tuple[BBox, int | None]]
    split: str
    source: str


class DatasetAdapter(ABC):
    """Common interface implemented by each dataset's adapter.

    Adapters that don't contribute to one of the two output sets simply
    leave the corresponding method as the default empty iterator.
    """

    name: str

    def classification_items(self) -> Iterator[ClassificationItem]:
        return iter(())

    def detection_items(self) -> Iterator[DetectionItem]:
        return iter(())
