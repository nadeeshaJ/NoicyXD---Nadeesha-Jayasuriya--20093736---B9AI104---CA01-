"""
Single-file inference for deployment and Streamlit app.

CA1 role:
    Loads best_model.pt checkpoints, runs the same preprocessing as training,
    and returns top-k class probabilities with confidence scores.

Used by:
    app/streamlit_app.py, scripts/verify_step5_deployment.py
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.audio_utils import fix_duration, load_audio
from src.gradcam import compute_gradcam, gradcam_summary, plot_gradcam_overlay
from src.models import MODEL_NAMES, build_model, uses_pretrained_norm
from src.spectrogram import compute_mel_spectrogram, normalize_spectrogram, waveform_to_rgb_mel_image
from src.utils import load_config, project_path


IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_class_names(cfg: dict, dataset_key: str) -> list[str]:
    return cfg["datasets"][dataset_key]["classes"]


def load_checkpoint_model(
    model_name: str,
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    model = build_model(model_name, num_classes=num_classes)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def resolve_model_checkpoint(mode: str, model_name: str, cfg: dict) -> tuple[str, Path]:
    """Return canonical model name and checkpoint path for a mode/model pair."""
    deploy_cfg = cfg["deployment"][mode]
    model_name = model_name.lower()
    models_cfg = deploy_cfg.get("models", {})
    if model_name in models_cfg:
        checkpoint = project_path(models_cfg[model_name]["checkpoint"])
    else:
        # Fall back to the default deployed model for this mode
        model_name = deploy_cfg["model_name"]
        checkpoint = project_path(deploy_cfg["checkpoint"])
    return model_name, checkpoint


def load_mode_model(
    mode: str,
    cfg: dict | None = None,
    device: torch.device | None = None,
    model_name: str | None = None,
) -> tuple[nn.Module, list[str], dict, str, Path]:
    """Load a trained model for 'urban' or 'animal' mode."""
    cfg = cfg or load_config()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deploy_cfg = cfg["deployment"][mode]
    dataset_key = deploy_cfg["dataset_key"]
    class_names = get_class_names(cfg, dataset_key)
    chosen_name, checkpoint = resolve_model_checkpoint(mode, model_name or deploy_cfg["model_name"], cfg)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    model = load_checkpoint_model(
        chosen_name,
        checkpoint,
        len(class_names),
        device,
    )
    return model, class_names, deploy_cfg, chosen_name, checkpoint


def load_benchmark_table(cfg: dict | None = None) -> dict[str, dict[str, Any]]:
    cfg = cfg or load_config()
    path = project_path(cfg["app"]["benchmarks_json"])
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return {row["model"]: row for row in payload.get("benchmarks", [])}


def available_models_for_mode(mode: str, cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    models_cfg = cfg["deployment"][mode].get("models", {})
    available: list[str] = []
    for name in MODEL_NAMES:
        if name not in models_cfg:
            continue
        if project_path(models_cfg[name]["checkpoint"]).exists():
            available.append(name)
    if not available:
        deploy_cfg = cfg["deployment"][mode]
        fallback = project_path(deploy_cfg["checkpoint"])
        if fallback.exists():
            available.append(deploy_cfg["model_name"])
    return available


def preprocess_uploaded_audio(
    audio_source: str | Path | bytes,
    cfg: dict | None = None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Return waveform, sample rate, normalized mel-spec, and RGB preview image."""
    cfg = cfg or load_config()
    audio_cfg = cfg["audio"]
    spec_cfg = cfg["spectrogram"]
    image_cfg = cfg["image"]

    if isinstance(audio_source, bytes):
        # Streamlit uploads arrive as raw bytes — load from memory buffer
        y, sr = librosa.load(io.BytesIO(audio_source), sr=audio_cfg["sample_rate"], mono=audio_cfg.get("mono", True))
        y = fix_duration(y, audio_cfg["sample_rate"], audio_cfg["duration_sec"])
    else:
        y, sr = load_audio(
            audio_source,
            sample_rate=audio_cfg["sample_rate"],
            duration_sec=audio_cfg["duration_sec"],
            mono=audio_cfg.get("mono", True),
        )

    mel = compute_mel_spectrogram(y, sr, spec_cfg)
    mel_norm = normalize_spectrogram(mel)
    rgb = waveform_to_rgb_mel_image(y, sr, spec_cfg, image_cfg)
    return y, sr, mel_norm, rgb


def rgb_to_model_tensor(rgb: np.ndarray, model_name: str) -> torch.Tensor:
    pil = Image.fromarray(rgb, mode="RGB")
    tensor = transforms.ToTensor()(pil)  # shape (3, 224, 224), values [0,1]
    if uses_pretrained_norm(model_name):
        tensor = IMAGENET_NORMALIZE(tensor)
    return tensor.unsqueeze(0)  # add batch dimension → (1, 3, 224, 224)


def _format_prediction_result(
    class_names: list[str],
    probs: np.ndarray,
    y: np.ndarray,
    sr: int,
    mel_norm: np.ndarray,
    rgb: np.ndarray,
    top_k: int,
    inference_ms: float | None = None,
) -> dict[str, Any]:
    top_indices = probs.argsort()[::-1][:top_k]  # highest probability first
    predictions = [
        {"label": class_names[i], "confidence": float(probs[i])}
        for i in top_indices
    ]
    result = {
        "predictions": predictions,
        "top_label": predictions[0]["label"],
        "top_confidence": predictions[0]["confidence"],
        "waveform": y,
        "sample_rate": sr,
        "mel_spectrogram": mel_norm,
        "rgb_image": rgb,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }
    if inference_ms is not None:
        result["inference_ms"] = inference_ms
    return result


@torch.no_grad()
def predict_audio(
    model: nn.Module,
    class_names: list[str],
    model_name: str,
    audio_source: str | Path | bytes,
    device: torch.device | None = None,
    cfg: dict | None = None,
    top_k: int = 3,
    measure_latency: bool = False,
) -> dict[str, Any]:
    """Run full inference pipeline on an uploaded audio clip."""
    cfg = cfg or load_config()
    device = device or next(model.parameters()).device

    y, sr, mel_norm, rgb = preprocess_uploaded_audio(audio_source, cfg)
    tensor = rgb_to_model_tensor(rgb, model_name).to(device)

    inference_ms = None
    if measure_latency:
        if device.type == "cuda":
            torch.cuda.synchronize()  # wait for GPU kernels before timing
        start = time.perf_counter()
        logits = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000.0
    else:
        logits = model(tensor)

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return _format_prediction_result(
        class_names,
        probs,
        y,
        sr,
        mel_norm,
        rgb,
        top_k,
        inference_ms=inference_ms,
    )


def predict_with_gradcam(
    model: nn.Module,
    class_names: list[str],
    model_name: str,
    audio_source: str | Path | bytes,
    device: torch.device | None = None,
    cfg: dict | None = None,
    top_k: int = 3,
    measure_latency: bool = False,
) -> dict[str, Any]:
    """Run inference and attach Grad-CAM for the predicted top class."""
    cfg = cfg or load_config()
    device = device or next(model.parameters()).device

    y, sr, mel_norm, rgb = preprocess_uploaded_audio(audio_source, cfg)
    tensor = rgb_to_model_tensor(rgb, model_name).to(device)
    tensor = tensor.clone().requires_grad_(False)

    inference_ms = None
    if measure_latency:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    top_idx = int(probs.argmax())

    # Grad-CAM requires gradients — create a fresh tensor with grad enabled
    grad_input = rgb_to_model_tensor(rgb, model_name).to(device)
    cam = compute_gradcam(model, model_name, grad_input, top_idx)

    if measure_latency:
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000.0

    result = _format_prediction_result(
        class_names,
        probs,
        y,
        sr,
        mel_norm,
        rgb,
        top_k,
        inference_ms=inference_ms,
    )
    top_label = class_names[top_idx]
    result["gradcam"] = cam
    result["gradcam_figure"] = plot_gradcam_overlay(rgb, cam, top_label, float(probs[top_idx]))
    result["gradcam_summary"] = gradcam_summary(model_name, top_label, float(probs[top_idx]))
    return result


def plot_waveform(waveform: np.ndarray, sample_rate: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.5))
    times = np.linspace(0, len(waveform) / sample_rate, len(waveform))
    ax.plot(times, waveform, color="navy", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Input Waveform")
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(mel: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Mel-Spectrogram (normalized)")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
