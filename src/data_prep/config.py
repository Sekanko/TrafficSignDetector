"""Loads per-dataset YAML configs from configs/datasets/."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from src.data_prep.paths import DATASET_CONFIGS_DIR


@lru_cache(maxsize=None)
def load_dataset_config(name: str) -> dict[str, Any]:
    path = DATASET_CONFIGS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
