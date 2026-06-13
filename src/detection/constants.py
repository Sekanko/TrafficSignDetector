"""Constants shared by the detection pipeline."""

from __future__ import annotations

from src.data_prep.paths import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
DETECTION_MODELS_DIR = MODELS_DIR / "detection"
SEED = 67
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
