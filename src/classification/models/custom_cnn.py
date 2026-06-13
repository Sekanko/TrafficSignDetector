"""Custom CNN classifier - placeholder.

A small from-scratch baseline (no pretrained weights) so the training script
runs end-to-end. Replace `CustomCNN` with the project's own architecture once
designed; `build()` is the only thing train.py relies on.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build(num_classes: int, pretrained: bool = False) -> nn.Module:
    # No pretrained weights exist for this architecture; `pretrained` is
    # accepted (and ignored) so it has the same signature as the other
    # builders and train.py's --pretrained flag doesn't need special-casing.
    return CustomCNN(num_classes)
