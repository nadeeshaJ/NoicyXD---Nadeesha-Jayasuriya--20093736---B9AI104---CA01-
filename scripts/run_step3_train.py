"""Step 3: Model design, UrbanSound8K training, and architecture figures.

Run:
    python scripts/run_step3_train.py

Trains all three CA1 models on UrbanSound8K fold-10 test evaluation:
    Model 1 — custom_cnn (baseline)
    Model 2 — resnet50 (transfer)
    Model 3 — mobilenetv2 (customised transfer, best model)

Outputs:
    experiments/urbansound8k/{model}/best_model.pt, test_metrics.json
    reports/figures/step3/architecture_*.png, model_comparison_urbansound8k.png
    reports/figures/step3/urbansound8k/{model}/training_history.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from src.models import MODEL_NAMES, build_model
from src.train import train_model
from src.utils import load_config, project_path, save_json, set_seed

FIG_DIR = project_path("reports", "figures", "step3")
REPORTS_DIR = project_path("reports", "step3")


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def draw_custom_cnn_diagram() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.5, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    layers = [
        "Input 224×224×3",
        "Conv2d(3→32) + ReLU + MaxPool",
        "Conv2d(32→64) + ReLU + MaxPool",
        "Conv2d(64→128) + ReLU + MaxPool",
        "Conv2d(128→128) + ReLU + MaxPool",
        "Flatten",
        "FC(256) + ReLU + Dropout(0.5)",
        "FC(10) + Softmax",
    ]

    box_w, box_h = 7.2, 0.72
    gap = 0.55
    x_center = 5.0
    x_left = x_center - box_w / 2
    y_top_start = 10.5

    box_bounds: list[tuple[float, float, float, float]] = []
    for i, label in enumerate(layers):
        y_top = y_top_start - i * (box_h + gap)
        y_bottom = y_top - box_h
        box = FancyBboxPatch(
            (x_left, y_bottom),
            box_w,
            box_h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor="#fef3e2",
            edgecolor="#e67e22",
            linewidth=2.0,
        )
        ax.add_patch(box)
        ax.text(x_center, y_bottom + box_h / 2, label, ha="center", va="center", fontsize=10.5)
        box_bounds.append((x_left, y_bottom, x_left + box_w, y_top))

    for i in range(len(box_bounds) - 1):
        _, y_bot_cur, _, y_top_cur = box_bounds[i]
        _, y_bot_next, _, y_top_next = box_bounds[i + 1]
        arrow = FancyArrowPatch(
            (x_center, y_bot_cur),
            (x_center, y_top_next),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.8,
            color="#c0392b",
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)

    ax.text(
        x_center,
        0.55,
        "Data flow: Mel-spec RGB image → convolutional feature extraction → classification head",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        style="italic",
    )
    ax.set_title(
        "Custom CNN Architecture (Model 1 — Conventional Baseline)",
        fontsize=14,
        pad=16,
        fontweight="bold",
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "architecture_custom_cnn.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def draw_transfer_learning_diagram() -> None:
    """Horizontal transfer-learning strategy diagram for ResNet50 / MobileNetV2."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    main_steps = [
        ("Mel-spec RGB\n224×224×3", "#f3f4f6", "#6b7280"),
        ("ImageNet\npretrained backbone", "#dbeafe", "#2563eb"),
        ("Feature extractor\n(frozen → fine-tuned)", "#ffedd5", "#ea580c"),
        ("Custom classifier head\n(10 classes)", "#dcfce7", "#16a34a"),
    ]

    n = len(main_steps)
    box_w, box_h = 2.35, 1.15
    gap = 0.55
    total_w = n * box_w + (n - 1) * gap
    x_start = (15 - total_w) / 2
    y_main = 3.15

    bounds: list[tuple[float, float, float, float]] = []
    for i, (label, face, edge) in enumerate(main_steps):
        x_left = x_start + i * (box_w + gap)
        y_bottom = y_main - box_h / 2
        box = FancyBboxPatch(
            (x_left, y_bottom),
            box_w,
            box_h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=face,
            edgecolor=edge,
            linewidth=2.0,
        )
        ax.add_patch(box)
        ax.text(
            x_left + box_w / 2,
            y_main,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            linespacing=1.2,
        )
        bounds.append((x_left, y_bottom, x_left + box_w, y_bottom + box_h))

    for i in range(len(bounds) - 1):
        arrow = FancyArrowPatch(
            (bounds[i][2], y_main),
            (bounds[i + 1][0], y_main),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.4,
            color="#4b5563",
            shrinkA=3,
            shrinkB=3,
        )
        ax.add_patch(arrow)

    phase_steps = [
        ("Phase 1\nFreeze backbone\nTrain head only", "#ede9fe", "#7c3aed"),
        ("Phase 2\nUnfreeze top layers\nFine-tune + early stop", "#fce7f3", "#db2777"),
    ]
    phase_w, phase_h = 3.6, 0.95
    phase_gap = 1.2
    phase_total = 2 * phase_w + phase_gap
    phase_x_start = (15 - phase_total) / 2
    y_phase = 1.05

    for i, (label, face, edge) in enumerate(phase_steps):
        x_left = phase_x_start + i * (phase_w + phase_gap)
        box = FancyBboxPatch(
            (x_left, y_phase - phase_h / 2),
            phase_w,
            phase_h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.8,
            linestyle="--",
        )
        ax.add_patch(box)
        ax.text(x_left + phase_w / 2, y_phase, label, ha="center", va="center", fontsize=8.5, linespacing=1.15)

    ax.annotate(
        "",
        xy=(phase_x_start + phase_w + phase_gap - 0.15, y_phase),
        xytext=(phase_x_start + phase_w + 0.15, y_phase),
        arrowprops=dict(arrowstyle="-|>", color="#9ca3af", lw=2.0),
    )
    ax.text(
        7.5,
        1.95,
        "Two-phase training strategy (ResNet50 & MobileNetV2)",
        ha="center",
        va="center",
        fontsize=9,
        color="#6b7280",
        style="italic",
    )

    ax.set_title(
        "Transfer-Learning Strategy for ResNet50 and MobileNetV2",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    legend_items = [
        ("Input", "#f3f4f6", "#6b7280"),
        ("Pretrained backbone", "#dbeafe", "#2563eb"),
        ("Feature learning", "#ffedd5", "#ea580c"),
        ("Custom head", "#dcfce7", "#16a34a"),
    ]
    lx = x_start
    for label, face, edge in legend_items:
        ax.add_patch(
            FancyBboxPatch(
                (lx, 0.12),
                0.25,
                0.18,
                boxstyle="round,pad=0.02",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.4,
            )
        )
        ax.text(lx + 0.32, 0.21, label, va="center", fontsize=7.5)
        lx += 1.75

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "architecture_transfer_learning.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def build_model_specs(cfg: dict) -> list[dict]:
    num_classes = cfg["datasets"]["urbansound8k"]["num_classes"]
    specs = []
    for name in MODEL_NAMES:
        model = build_model(name, num_classes=num_classes, pretrained=False)
        total, trainable = count_params(model)
        specs.append({
            "model": name,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "role": {
                "custom_cnn": "Baseline — designed from scratch",
                "resnet50": "Transfer learning — strong accuracy",
                "mobilenetv2": "Efficient transfer learning — speed/size",
            }[name],
        })
    return specs


def plot_model_comparison(summaries: list[dict]) -> None:
    df = pd.DataFrame(
        [
            {
                "model": s["model_name"],
                "accuracy": s["test_metrics"]["accuracy"],
                "macro_f1": s["test_metrics"]["macro_f1"],
                "train_time_min": s["train_time_sec"] / 60,
            }
            for s in summaries
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df_melt = df.melt(id_vars="model", value_vars=["accuracy", "macro_f1"], var_name="metric", value_name="score")
    import seaborn as sns
    sns.barplot(data=df_melt, x="model", y="score", hue="metric", ax=axes[0])
    axes[0].set_title("Test Set Performance Comparison")
    axes[0].set_ylim(0, 1)

    sns.barplot(data=df, x="model", y="train_time_min", ax=axes[1], color="steelblue")
    axes[1].set_title("Training Time (minutes)")
    axes[1].set_ylabel("Minutes")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison_urbansound8k.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    set_seed(42)
    cfg = load_config()

    # Architecture diagrams for report Figures 2.2 and 2.5
    print("Step 3.1 — Generating architecture diagrams...")
    draw_custom_cnn_diagram()
    draw_transfer_learning_diagram()
    model_specs = build_model_specs(cfg)
    save_json({"models": model_specs}, REPORTS_DIR / "model_specifications.json")

    # Train all three CA1 models; each saves best_model.pt + test metrics
    print("Step 3.2 — Training all models on UrbanSound8K...")
    summaries = []
    for model_name in MODEL_NAMES:
        print(f"\n=== Training {model_name} ===")
        summary = train_model(model_name, dataset_key="urbansound8k", cfg=cfg)
        summaries.append(summary)

    print("\nStep 3.3 — Model comparison plot...")
    plot_model_comparison(summaries)

    # Pick best model by macro F1 (handles class imbalance better than accuracy)
    best = max(summaries, key=lambda s: s["test_metrics"]["macro_f1"])
    step3_summary = {
        "step": 3,
        "title": "Model Design and UrbanSound8K Training",
        "model_specs": model_specs,
        "training_summaries": summaries,
        "best_model": best["model_name"],
        "best_macro_f1": best["test_metrics"]["macro_f1"],
        "figures": [
            "architecture_custom_cnn.png",
            "architecture_transfer_learning.png",
            "model_comparison_urbansound8k.png",
        ],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(step3_summary, REPORTS_DIR / "step3_summary.json")

    comparison_rows = []
    for s in summaries:
        comparison_rows.append({
            "model": s["model_name"],
            **s["test_metrics"],
            "train_time_sec": s["train_time_sec"],
            "epochs_run": s["total_epochs_run"],
        })
    pd.DataFrame(comparison_rows).to_csv(REPORTS_DIR / "model_comparison_urbansound8k.csv", index=False)

    print("\nStep 3 complete.")
    print(f"  Best model: {best['model_name']} (macro F1={best['test_metrics']['macro_f1']:.4f})")
    print(f"  Reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
