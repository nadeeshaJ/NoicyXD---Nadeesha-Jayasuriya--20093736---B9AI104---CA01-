"""
Custom CNN baseline for Mel-spectrogram image classification.

CA1 Model 1 — conventional CNN from scratch (NOT the customised transfer model).
Architecture: 4 conv blocks → flatten → FC(256) → dropout → FC(10).
Trained without ImageNet weights; uses Mel-spec [0,1] input directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    Baseline CNN designed from scratch for 224x224 RGB Mel-spectrogram input.

    Architecture:
        Conv(32) -> ReLU -> MaxPool
        Conv(64) -> ReLU -> MaxPool
        Conv(128) -> ReLU -> MaxPool
        Flatten -> FC(256) -> ReLU -> Dropout -> FC(num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        # Four conv blocks: each MaxPool halves spatial size (224→112→56→28→14)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 128 channels × 14×14 spatial = 25,088 features after flatten
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # regularisation — 50% neurons dropped during training
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)      # extract spatial features
        return self.classifier(x)   # logits for each class (no softmax — CrossEntropyLoss applies it)


def build_custom_cnn(num_classes: int = 10, dropout: float = 0.5) -> CustomCNN:
    return CustomCNN(num_classes=num_classes, dropout=dropout)
