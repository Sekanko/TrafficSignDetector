"""MobileNetV2 classifier (torchvision, ImageNet-pretrained backbone)."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[-1] = nn.Linear(model.last_channel, num_classes)
    return model
