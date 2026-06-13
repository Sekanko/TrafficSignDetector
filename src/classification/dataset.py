"""Classification dataset backed by data/processed/classification/manifest.csv.

Loading via the manifest (rather than torchvision.datasets.ImageFolder
scanning <split>/<class_name>/ directories) means the label is always the
taxonomy id from configs/taxonomy.yaml, so class indices stay consistent
across splits even if some class has zero images in a given split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["taxonomy_id"])


def load_splits(root: Path, transform_train, transform_eval) -> dict[str, ManifestDataset]:
    manifest = pd.read_csv(root / "manifest.csv")
    return {
        "train": ManifestDataset(manifest[manifest["split"] == "train"], root, transform_train),
        "val": ManifestDataset(manifest[manifest["split"] == "val"], root, transform_eval),
        "test": ManifestDataset(manifest[manifest["split"] == "test"], root, transform_eval),
    }
