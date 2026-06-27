"""
ResNet50 transfer learning model.

CA1 Model 2 — traditional transfer learning with ImageNet-pretrained backbone.
The original 1000-class fc layer is replaced with Dropout + Linear(10).
Training uses freeze-then-fine-tune (see train.py).
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """ResNet50 with ImageNet weights and replaced classification head."""
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=pretrained)

    in_features = model.fc.in_features
    # Replace ImageNet 1000-class head with 10-class urban/animal head
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the final fully connected head."""
    for name, param in model.named_parameters():
        # Only fc.* layers get gradients — backbone weights stay at ImageNet values
        param.requires_grad = name.startswith("fc")


def unfreeze_top_layers(model: nn.Module, num_layers: int = 2) -> None:
    """Unfreeze the last residual blocks for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = False
    children = list(model.children())
    # Unfreeze last 2 residual blocks (layer3, layer4) plus the fc head
    for block in children[-num_layers - 1 :]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
