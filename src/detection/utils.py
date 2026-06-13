"""General utilities for detection training scripts."""

from __future__ import annotations

from pathlib import Path

from src.data_prep.paths import PROJECT_ROOT


def project_relative(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
