from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_densenet121_backbone(pretrained: bool = True) -> nn.Module:
    """
    Returns DenseNet121 feature extractor.
    Output feature map channel: 1024
    """
    if pretrained:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    else:
        model = models.densenet121(weights=None)

    return model.features