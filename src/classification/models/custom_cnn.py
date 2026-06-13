"""Custom CNN classifier.

A small from-scratch network (no pretrained weights) built from
depthwise-separable convolutions (MobileNet-style), kept well under
SqueezeNet's parameter count for this taxonomy (~168K vs ~748K) while having
meaningfully more capacity than a plain few-layer baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _depthwise_separable(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(3, 32, stride=2),
            _depthwise_separable(32, 64, stride=2),
            _depthwise_separable(64, 128, stride=2),
            _depthwise_separable(128, 256, stride=2),
            _depthwise_separable(256, 384, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(384, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build(num_classes: int, pretrained: bool = False) -> nn.Module:
    # No pretrained weights exist for this architecture; `pretrained` is
    # accepted (and ignored) so it has the same signature as the other
    # builders and train.py's --pretrained flag doesn't need special-casing.
    return CustomCNN(num_classes)
