"""Train every model with and without augmentation, logging results to a CSV.

For each model in the registry, runs `src.classification.train` twice (once
with `--no-augment`, once with `--augment`) as a subprocess so training
output/progress bars render normally. After each run, the JSON summary it
writes under models/ is read and appended as a row to
models/experiments_log.csv.

Usage:
    python -m src.classification.run_experiments
    python -m src.classification.run_experiments --epochs 3 --models custom_cnn mobilenet_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys

from src.classification.models import MODEL_NAMES
from src.data_prep.paths import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
LOG_PATH = MODELS_DIR / "experiments_log.csv"

LOG_FIELDS = [
    "run_name", "model", "augment", "epochs", "learning_rate", "optimizer",
    "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train every model with and without augmentation")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def run_training(model: str, epochs: int, batch_size: int, augment: bool):
    cmd = [
        sys.executable, "-m", "src.classification.train",
        "--model", model,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--augment" if augment else "--no-augment",
    ]
    print(f"\n=== {' '.join(cmd)} ===")

    before = set(MODELS_DIR.glob("*.json"))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for model={model} augment={augment} (exit code {result.returncode})")

    new_files = set(MODELS_DIR.glob("*.json")) - before
    if len(new_files) != 1:
        raise RuntimeError(f"Expected exactly one new summary file for model={model} augment={augment}, found {len(new_files)}")
    return new_files.pop()


def append_log_row(summary_path) -> None:
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    last_epoch = summary["history"][-1]
    row = {
        "run_name": summary["run_name"],
        "model": summary["model"],
        "augment": summary["augment"],
        "epochs": summary["epochs"],
        "learning_rate": summary["learning_rate"],
        "optimizer": summary["optimizer"],
        "train_loss": last_epoch["train_loss"],
        "train_acc": last_epoch["train_acc"],
        "val_loss": last_epoch["val_loss"],
        "val_acc": last_epoch["val_acc"],
        "test_loss": summary["test_loss"],
        "test_acc": summary["test_acc"],
    }

    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        for augment in (False, True):
            summary_path = run_training(model, args.epochs, args.batch_size, augment)
            append_log_row(summary_path)

    print(f"\nAll runs done. Results logged to {LOG_PATH}")


if __name__ == "__main__":
    main()
