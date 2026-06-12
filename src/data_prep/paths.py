"""Central path constants used across the data-prep pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_CONFIGS_DIR = CONFIGS_DIR / "datasets"
TAXONOMY_PATH = CONFIGS_DIR / "taxonomy.yaml"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

CLASSIFICATION_DIR = DATA_PROCESSED / "classification"
DETECTION_DIR = DATA_PROCESSED / "detection"
