"""EfficientNet-Lite0 classifier (timm's tf_efficientnet_lite0, TFLite-friendly).

torchvision doesn't ship the "Lite" variants (no squeeze-excite, ReLU6,
fixed-size pooling), which are specifically designed for easy TFLite export -
matching this project's eventual PyTorch -> TFLite goal. timm provides
ImageNet-pretrained weights for it.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build(num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model("tf_efficientnet_lite0", pretrained=pretrained, num_classes=num_classes)
