"""
MobileNetV2 efficient transfer learning model.

CA1 Model 3 — customised transfer-learning model (best urban performer).
ImageNet backbone + replaced classifier head; head-size ablation tested
64/128/256 hidden units (see run_ca1_ablation_studies.py).
Deployed in Streamlit for urban and animal modes.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_mobilenetv2(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """MobileNetV2 with ImageNet weights and replaced classifier head."""
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except AttributeError:
        model = models.mobilenet_v2(pretrained=pretrained)

    in_features = model.classifier[1].in_features
    # MobileNetV2 classifier is Sequential(Dropout, Linear) — replace the Linear
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze feature extractor; train classifier only."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")


def unfreeze_top_layers(model: nn.Module) -> None:
    """Unfreeze last inverted residual blocks and classifier."""
    for param in model.parameters():
        param.requires_grad = False
    features = list(model.features.children())
    # Last 4 inverted-residual blocks capture high-level patterns
    for block in features[-4:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
