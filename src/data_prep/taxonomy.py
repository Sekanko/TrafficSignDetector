"""Loads and exposes the unified label taxonomy (configs/taxonomy.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from src.data_prep.paths import TAXONOMY_PATH


@dataclass(frozen=True)
class TaxonomyClass:
    id: int
    name: str
    category: str
    gtsrb_id: int | None
    en: str
    de: str
    pl: str


class Taxonomy:
    """Unified label taxonomy for the classifier's classes."""

    def __init__(self, classes: list[TaxonomyClass]):
        self.classes = classes
        self._by_id = {c.id: c for c in classes}
        self._by_name = {c.name: c for c in classes}
        self._by_gtsrb_id = {c.gtsrb_id: c for c in classes if c.gtsrb_id is not None}

    def by_id(self, taxonomy_id: int) -> TaxonomyClass:
        return self._by_id[taxonomy_id]

    def by_name(self, name: str) -> TaxonomyClass:
        return self._by_name[name]

    def by_gtsrb_id(self, gtsrb_id: int) -> TaxonomyClass:
        return self._by_gtsrb_id[gtsrb_id]

    def __len__(self) -> int:
        return len(self.classes)


@lru_cache(maxsize=1)
def load_taxonomy() -> Taxonomy:
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    classes = [
        TaxonomyClass(
            id=c["id"],
            name=c["name"],
            category=c["category"],
            gtsrb_id=c.get("gtsrb_id"),
            en=c["en"],
            de=c["de"],
            pl=c["pl"],
        )
        for c in data["classes"]
    ]
    return Taxonomy(classes=classes)
