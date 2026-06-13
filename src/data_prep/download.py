"""Downloads the configured Kaggle datasets into data/raw/<name>/.

Requires Kaggle API credentials - see docs/kaggle_setup.md.

Usage:
    python -m src.data_prep.download                  # all datasets
    python -m src.data_prep.download --dataset gtsrb gtsdb
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

from src.data_prep.config import load_dataset_config
from src.data_prep.paths import DATA_RAW, PROJECT_ROOT

logger = logging.getLogger(__name__)

DATASETS = ["gtsrb", "gtsdb", "polish"]


def _check_credentials() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    if not kaggle_json.exists() and not has_env:
        raise SystemExit(
            "Kaggle credentials not found.\n"
            f"Expected {kaggle_json} or KAGGLE_USERNAME/KAGGLE_KEY env vars.\n"
            f"See {PROJECT_ROOT / 'docs' / 'kaggle_setup.md'} for setup steps."
        )


def download_dataset(name: str) -> Path:
    import kagglehub

    config = load_dataset_config(name)
    slug = config["kaggle"]["slug"]
    dest = DATA_RAW / config["raw_dir"]

    logger.info("Downloading %s (%s) ...", name, slug)
    cache_path = Path(kagglehub.dataset_download(slug))

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("%s already exists, removing before re-copy", dest)
        shutil.rmtree(dest)
    shutil.copytree(cache_path, dest)
    logger.info("%s -> %s", name, dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASETS,
        help="Dataset(s) to download (default: all).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _check_credentials()

    for name in args.dataset or DATASETS:
        download_dataset(name)


if __name__ == "__main__":
    main()
