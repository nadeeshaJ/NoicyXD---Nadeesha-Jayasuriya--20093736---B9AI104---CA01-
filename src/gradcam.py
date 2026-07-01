"""Grad-CAM visual explanations for CNN models on Mel-spectrogram images.

Used by: app/streamlit_app.py (explainability toggle), run_step6_error_analysis.py
Shows which time-frequency regions the model attends to for a given prediction.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping for a single target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        # Hooks capture forward activations and backward gradients at target layer
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        logits = self.model(input_tensor)
        if class_idx < 0 or class_idx >= logits.shape[1]:
            raise ValueError(f"class_idx {class_idx} out of range for {logits.shape[1]} classes")

        score = logits[0, class_idx]
        score.backward()  # backprop only the target class score

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

        # Global-average-pool gradients → importance weight per feature map channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # keep only positive contributions
        # Upsample heatmap to match input image size (224×224)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # scale to [0,1]

        if was_training:
            self.model.train()
        return cam


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    name = model_name.lower()
    if name == "custom_cnn":
        return model.features[11]  # final MaxPool2d in feature extractor (128 ch, 14×14)
    if name == "resnet50":
        return model.layer4[-1]     # deepest residual block
    if name == "mobilenetv2":
        return model.features[-1]   # last inverted-residual block
    raise ValueError(f"No Grad-CAM target layer mapping for model: {model_name}")


def compute_gradcam(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    class_idx: int,
) -> np.ndarray:
    target_layer = get_gradcam_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)
    try:
        return gradcam.generate(input_tensor, class_idx)
    finally:
        gradcam.close()


def overlay_gradcam_on_image(
    rgb_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    base = np.asarray(rgb_image, dtype=np.float32) / 255.0
    heat = plt.cm.jet(cam)[..., :3]
    overlay = np.clip((1 - alpha) * base + alpha * heat, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def plot_gradcam_overlay(
    rgb_image: np.ndarray,
    cam: np.ndarray,
    class_label: str,
    confidence: float,
) -> plt.Figure:
    overlay = overlay_gradcam_on_image(rgb_image, cam)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Model Input (224×224 RGB Mel)")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM — {class_label.replace('_', ' ').title()} ({confidence:.1%})")
    axes[1].axis("off")

    fig.tight_layout()
    return fig


def gradcam_summary(model_name: str, class_label: str, confidence: float) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "class_label": class_label,
        "confidence": confidence,
        "method": "Grad-CAM on final convolutional layer",
    }
