"""Model registry for the sign classifier.

Each model module exposes a `build(num_classes, pretrained) -> nn.Module`
function. `build_model()` is the single entry point used by train.py.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from src.classification.models import custom_cnn, efficientnet_lite0, mobilenet_v2, squeezenet

_BUILDERS: dict[str, Callable[[int, bool], nn.Module]] = {
    "mobilenet_v2": mobilenet_v2.build,
    "efficientnet_lite0": efficientnet_lite0.build,
    "squeezenet": squeezenet.build,
    "custom_cnn": custom_cnn.build,
}

# Whether `pretrained=True` actually loads ImageNet weights for this model -
# custom_cnn has none, so train.py can report this accurately.
HAS_PRETRAINED: dict[str, bool] = {
    "mobilenet_v2": True,
    "efficientnet_lite0": True,
    "squeezenet": True,
    "custom_cnn": False,
}

MODEL_NAMES = sorted(_BUILDERS)


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {MODEL_NAMES}") from None
    return builder(num_classes, pretrained)
