"""Train a sign classifier on data/processed/classification/.

Pick a model and hyperparameters via CLI flags; the trained weights and a
JSON run summary are saved under models/<model>_<timestamp>.{pt,json}, so
every run gets its own files regardless of what changed between runs.

Usage:
    python -m src.classification.train --model mobilenet_v2 --epochs 10 --lr 1e-3
    python -m src.classification.train --model custom_cnn --epochs 5 --optimizer sgd --lr 0.01
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.classification.dataset import load_splits
from src.classification.models import HAS_PRETRAINED, MODEL_NAMES, build_model
from src.data_prep.paths import CLASSIFICATION_DIR, PROJECT_ROOT
from src.data_prep.taxonomy import load_taxonomy

MODELS_DIR = PROJECT_ROOT / "models"
SEED = 67

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, augment: bool = True) -> tuple[transforms.Compose, transforms.Compose]:
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if not augment:
        return eval_tf, eval_tf

    # No random flips: many signs (turn arrows, etc.) are not left/right symmetric.
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])
    return train_tf, eval_tf


def build_optimizer(name: str, params, lr: float, weight_decay: float, momentum: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unknown optimizer '{name}'")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    train: bool,
    desc: str,
) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=desc, leave=False, mininterval=1.0)
    with torch.set_grad_enabled(train):
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
            progress.set_postfix(loss=total_loss / total, acc=correct / total, refresh=False)

    return total_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a traffic sign classifier")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9, help="used by --optimizer sgd")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)

    taxonomy = load_taxonomy()
    num_classes = len(taxonomy)

    train_tf, eval_tf = build_transforms(args.img_size, augment=args.augment)
    datasets = load_splits(CLASSIFICATION_DIR, train_tf, eval_tf)
    loaders = {
        split: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )
        for split, ds in datasets.items()
    }

    pretrained = args.pretrained and HAS_PRETRAINED[args.model]
    model = build_model(args.model, num_classes, pretrained=pretrained).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.momentum)

    history = []
    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_desc = f"epoch {epoch:3d}/{args.epochs} [train]"
        val_desc = f"epoch {epoch:3d}/{args.epochs} [val]"
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, args.device, train=True, desc=train_desc)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, args.device, train=False, desc=val_desc)
        best_val_acc = max(best_val_acc, val_acc)
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
    training_time = time.time() - start_time

    test_loss, test_acc = run_epoch(model, loaders["test"], criterion, optimizer, args.device, train=False, desc="test")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{run_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    summary = {
        "run_name": run_name,
        "model": args.model,
        "pretrained": pretrained,
        "num_classes": num_classes,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "augment": args.augment,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum if args.optimizer == "sgd" else None,
        "seed": SEED,
        "device": args.device,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_time_sec": training_time,
        "train_acc": history[-1]["train_acc"],
        "val_acc": history[-1]["val_acc"],
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
    }
    summary_path = MODELS_DIR / f"{run_name}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    optimizer_desc = f"{args.optimizer} (weight_decay={args.weight_decay}"
    if args.optimizer == "sgd":
        optimizer_desc += f", momentum={args.momentum}"
    optimizer_desc += ")"

    print()
    print("=" * 60)
    print(f"Model:          {args.model} (pretrained={pretrained}, augment={args.augment})")
    print(f"Epochs:         {args.epochs}")
    print(f"Learning rate:  {args.lr}")
    print(f"Optimizer:      {optimizer_desc}")
    print(f"Train:          loss={history[-1]['train_loss']:.4f}  acc={history[-1]['train_acc']:.4f}")
    print(f"Val:            loss={history[-1]['val_loss']:.4f}  acc={history[-1]['val_acc']:.4f} (best acc: {best_val_acc:.4f})")
    print(f"Test:           loss={test_loss:.4f}  acc={test_acc:.4f}")
    print(f"Saved model:    {checkpoint_path}")
    print(f"Saved summary:  {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
