"""
SpecAugment and spectrogram image augmentation transforms.

CA1 role:
    Applied on training Mel-spectrograms only (never val/test) to reduce
    overfitting. Time masking hides short time regions; frequency masking
    hides Mel bands — see report Section 4.7.

Used by:
    dataset training path and run_step2_preprocess.py (demo figure).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def spec_augment(
    mel: np.ndarray,
    aug_cfg: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply time and frequency masking on a 2D Mel-spectrogram (before RGB conversion)."""
    rng = rng or np.random.default_rng()
    augmented = mel.copy()

    n_mels, n_steps = augmented.shape
    time_mask_max = aug_cfg.get("time_mask_max", 24)
    freq_mask_max = aug_cfg.get("freq_mask_max", 16)
    num_time_masks = aug_cfg.get("num_time_masks", 1)
    num_freq_masks = aug_cfg.get("num_freq_masks", 1)

    # Frequency masking: zero out random horizontal bands (Mel bins)
    for _ in range(num_freq_masks):
        f = int(rng.integers(0, freq_mask_max + 1))  # mask width in Mel bins
        if f == 0:
            continue
        f0 = int(rng.integers(0, max(1, n_mels - f)))  # random start row
        augmented[f0:f0 + f, :] = 0.0

    # Time masking: zero out random vertical strips (time frames)
    for _ in range(num_time_masks):
        t = int(rng.integers(0, time_mask_max + 1))
        if t == 0:
            continue
        t0 = int(rng.integers(0, max(1, n_steps - t)))
        augmented[:, t0:t0 + t] = 0.0

    return augmented


def add_gaussian_noise(
    mel: np.ndarray,
    std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add small Gaussian noise to a normalized Mel-spectrogram."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, std, size=mel.shape)
    return np.clip(mel + noise, 0.0, 1.0)
