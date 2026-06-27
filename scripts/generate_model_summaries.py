"""Generate torchinfo model.summary() figures for the final report.

Run:
    python scripts/generate_model_summaries.py

Outputs (report Figures 2.3, 2.4, 2.6):
    reports/figures/final/model_summary_custom_cnn.png
    reports/figures/final/model_summary_resnet50.png
    reports/figures/final/model_summary_mobilenetv2.png
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchinfo import summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "reports" / "figures" / "final"
INPUT_SIZE = (1, 3, 224, 224)


def summary_to_figure(model_name: str, model: torch.nn.Module, title: str) -> Path:
    buf = StringIO()
    stats = summary(
        model,
        input_size=INPUT_SIZE,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        verbose=0,
        depth=4,
    )
    text = str(stats)
  # torchinfo prints to stdout when verbose>0; capture via redirect
    import contextlib

    with contextlib.redirect_stdout(buf):
        summary(
            model,
            input_size=INPUT_SIZE,
            col_names=("input_size", "output_size", "num_params", "trainable"),
            verbose=1,
            depth=4,
        )
    text = buf.getvalue()

    lines = text.splitlines()
    fig_height = max(6, len(lines) * 0.22)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.text(
        0.01,
        0.99,
        text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    out_path = OUT_DIR / f"model_summary_{model_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("custom_cnn", "Model 1 — Custom CNN (conventional baseline from scratch)"),
        ("resnet50", "Model 2 — ResNet50 (ImageNet backbone + 10-class head)"),
        ("mobilenetv2", "Model 3 — MobileNetV2 (custom classifier head + fine-tune)"),
    ]
    for name, title in specs:
        model = build_model(name, num_classes=10, pretrained=True)
        model.eval()
        path = summary_to_figure(name, model, title)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: {total:,} params ({trainable:,} trainable) -> {path}")


if __name__ == "__main__":
    main()
