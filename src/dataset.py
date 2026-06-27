"""
PyTorch Dataset for preprocessed Mel-spectrogram PNG images.

CA1 role:
    Loads PNG paths from *_processed.csv split files. Transfer-learning models
    (ResNet50, MobileNetV2) use ImageNet mean/std; Custom CNN uses [0,1] only.

Used by:
    src/train.py, src/evaluate.py, src/predict.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet statistics — required so pretrained ResNet/MobileNet see familiar input scale
IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def default_transform(pretrained: bool = False) -> transforms.Compose:
    """Standard eval transform; ImageNet stats for transfer-learning models."""
    steps: list = [
        transforms.ToTensor(),  # uint8 [0,255] → float [0,1], shape C×H×W
    ]
    if pretrained:
        # Custom CNN skips this — it was trained on raw [0,1] Mel images
        steps.append(IMAGENET_NORMALIZE)
    return transforms.Compose(steps)


class MelSpectrogramImageDataset(Dataset):
    """Load PNG images listed in a *_processed.csv split file."""

    def __init__(
        self,
        processed_csv: str | Path,
        transform: transforms.Compose | None = None,
        pretrained_norm: bool = False,
    ):
        self.df = pd.read_csv(processed_csv)
        # Drop rows that failed preprocessing; keep skipped (already on disk)
        self.df = self.df[self.df["status"].isin(["processed", "skipped"])].reset_index(drop=True)
        self.transform = transform or default_transform(pretrained=pretrained_norm)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["class_idx"])  # integer class for CrossEntropyLoss
        if self.transform:
            image = self.transform(image)
        return image, label
