"""
Load, resample, mono conversion, and duration fixing for audio clips.

CA1 role:
    First stage of the audio-to-image pipeline. Every clip is standardised to
    22,050 Hz mono and exactly 4 seconds before Mel-spectrogram conversion.

Used by:
    preprocess.py, run_step2_preprocess.py, predict.py, Streamlit app.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(
    path: str | Path,
    sample_rate: int = 22050,
    duration_sec: float = 4.0,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load a wav file and return a fixed-length mono waveform."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # librosa resamples to target rate and optionally mixes stereo → mono
    waveform, _ = librosa.load(path, sr=sample_rate, mono=mono)
    # Every clip must be the same length so batches stack into a tensor
    return fix_duration(waveform, sample_rate, duration_sec), sample_rate


def fix_duration(
    waveform: np.ndarray,
    sample_rate: int,
    duration_sec: float,
) -> np.ndarray:
    """Pad with silence or trim to the target duration."""
    target_len = int(sample_rate * duration_sec)  # e.g. 22050 * 4 = 88,200 samples
    if len(waveform) < target_len:
        # Short clips: append zeros (silence) on the right
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        # Long clips: keep only the first N seconds
        waveform = waveform[:target_len]
    return waveform.astype(np.float32)


def peak_normalize(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Optional peak normalization to [-1, 1]."""
    peak = np.max(np.abs(waveform))
    if peak > eps:
        # Scale so loudest sample hits ±1.0 (used in some EDA plots, not training)
        return waveform / peak
    return waveform
