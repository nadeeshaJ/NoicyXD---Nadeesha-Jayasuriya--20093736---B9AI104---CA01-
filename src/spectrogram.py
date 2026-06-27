"""
Mel-spectrogram generation and conversion to RGB images.

CA1 role:
    Converts fixed-length waveforms into normalised 224×224×3 PNG images that
    CNN models (Custom CNN, ResNet50, MobileNetV2) can classify.

Pipeline:
    waveform → STFT/Mel power → dB scale → min-max [0,1] → resize → RGB stack

Used by:
    preprocess.py, predict.py, error_analysis.py.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
from PIL import Image


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    spec_cfg: dict[str, Any],
) -> np.ndarray:
    """Return power Mel-spectrogram (2D float array)."""
    # STFT → Mel filterbank → power spectrogram (shape: n_mels × time_frames)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=spec_cfg["n_fft"],           # window size for frequency bins
        hop_length=spec_cfg["hop_length"], # hop between STFT frames
        n_mels=spec_cfg["n_mels"],         # number of Mel bands (128)
        fmin=spec_cfg.get("fmin", 0),
        fmax=spec_cfg.get("fmax", sample_rate // 2),  # Nyquist limit
        power=2.0,                         # magnitude squared → power
    )
    if spec_cfg.get("power_to_db", True):
        # Log scale compresses dynamic range; ref=np.max normalises to peak 0 dB
        return librosa.power_to_db(mel, ref=np.max)
    return mel


def normalize_spectrogram(mel: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize spectrogram to [0, 1]."""
    min_val = float(mel.min())
    max_val = float(mel.max())
    if max_val - min_val < eps:
        return np.zeros_like(mel, dtype=np.float32)  # silent / flat clip
    # Per-clip normalisation — each image uses its own min/max
    return ((mel - min_val) / (max_val - min_val)).astype(np.float32)


def mel_to_rgb_image(
    mel: np.ndarray,
    height: int = 224,
    width: int = 224,
) -> np.ndarray:
    """Convert a 2D Mel-spectrogram to an H×W×3 uint8 RGB image."""
    gray = (np.clip(mel, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(gray, mode="L")
    # Bilinear resize to 224×224 — matches ImageNet input size for transfer models
    pil_img = pil_img.resize((width, height), Image.Resampling.BILINEAR)
    # Grayscale replicated to 3 channels so torchvision models accept RGB input
    rgb = np.stack([pil_img, pil_img, pil_img], axis=-1)
    return rgb


def waveform_to_rgb_mel_image(
    waveform: np.ndarray,
    sample_rate: int,
    spec_cfg: dict[str, Any],
    image_cfg: dict[str, Any],
) -> np.ndarray:
    """Full pipeline: waveform → normalized Mel-spectrogram → RGB image."""
    mel = compute_mel_spectrogram(waveform, sample_rate, spec_cfg)
    mel_norm = normalize_spectrogram(mel)
    return mel_to_rgb_image(
        mel_norm,
        height=image_cfg["height"],
        width=image_cfg["width"],
    )


def save_rgb_image(image: np.ndarray, path: str | Path) -> None:
    """Save H×W×3 uint8 RGB image as PNG."""
    Image.fromarray(image, mode="RGB").save(path)
