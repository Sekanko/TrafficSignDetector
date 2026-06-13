"""SqueezeNet 1.1 classifier (torchvision, ImageNet-pretrained backbone)."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1


def build(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
    model = squeezenet1_1(weights=weights)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    return model
