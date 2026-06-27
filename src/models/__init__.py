"""
Model factory and registry for the three CA1 models.

Models:
    custom_cnn    — Model 1: conventional CNN baseline (from scratch)
    resnet50      — Model 2: ImageNet transfer learning
    mobilenetv2   — Model 3: customised transfer-learning head + fine-tune

uses_pretrained_norm() returns True for transfer models that need ImageNet
mean/std normalisation at input (see dataset.py).
"""

from __future__ import annotations

import torch.nn as nn

from src.models.custom_cnn import build_custom_cnn
from src.models.mobilenetv2_model import build_mobilenetv2
from src.models.resnet50_model import build_resnet50

MODEL_NAMES = ("custom_cnn", "resnet50", "mobilenetv2")


def build_model(name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "custom_cnn":
        return build_custom_cnn(num_classes=num_classes)
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    if name == "mobilenetv2":
        return build_mobilenetv2(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}")


def uses_pretrained_norm(name: str) -> bool:
    return name.lower() in {"resnet50", "mobilenetv2"}
