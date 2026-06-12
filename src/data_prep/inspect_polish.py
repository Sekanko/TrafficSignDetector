"""Discovery script for the Polish Traffic Signs Dataset.

Run after `python -m src.data_prep.download --dataset polish`. Prints the
real folder layout and (if present) Train.csv/Test.csv columns and class id
distribution. Its output is used to fill in configs/datasets/polish.yaml's
`id_map` and to append any Polish-only classes to configs/taxonomy.yaml
(see the TODO at the top of polish.yaml).

Usage:
    python -m src.data_prep.inspect_polish
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_prep.config import load_dataset_config
from src.data_prep.paths import DATA_RAW

_TREE_MAX_DEPTH = 3
_TREE_ENTRY_LIMIT = 20


def _print_tree(root: Path, depth: int = 0) -> None:
    if depth > _TREE_MAX_DEPTH:
        return
    entries = sorted(root.iterdir())
    for entry in entries[:_TREE_ENTRY_LIMIT]:
        print("  " * depth + entry.name + ("/" if entry.is_dir() else ""))
        if entry.is_dir():
            _print_tree(entry, depth + 1)
    if len(entries) > _TREE_ENTRY_LIMIT:
        print("  " * depth + f"... ({len(entries) - _TREE_ENTRY_LIMIT} more)")


def _print_csv_summary(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    print(f"\n{csv_path.name}: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
    if "ClassId" in df.columns:
        counts = df["ClassId"].value_counts().sort_index()
        print(f"\nClassId value counts ({len(counts)} classes):")
        print(counts)


def main() -> None:
    config = load_dataset_config("polish")
    root = DATA_RAW / config["raw_dir"]

    if not root.exists():
        print(f"{root} does not exist yet - run the download script first:")
        print("  python -m src.data_prep.download --dataset polish")
        return

    print(f"Contents of {root}:")
    _print_tree(root)

    for csv_name in ("Train.csv", "Test.csv"):
        csv_path = root / csv_name
        if csv_path.exists():
            _print_csv_summary(csv_path)


if __name__ == "__main__":
    main()
