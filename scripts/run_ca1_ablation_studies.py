"""CA1 compliance ablations: Model 3 head sizes and Custom CNN dropout comparison.

Run:
    python scripts/run_ca1_ablation_studies.py

Experiments:
    MobileNetV2 head ablation — 64 / 128 / 256 hidden units (frozen backbone)
    Custom CNN dropout ablation — 0.0 vs 0.5

Outputs:
    reports/final/ca1_ablation_summary.json
    reports/figures/final/ablation_model3_head_sizes.png
    reports/figures/final/ablation_custom_cnn_dropout.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset import MelSpectrogramImageDataset
from src.evaluate import evaluate_model
from src.models.custom_cnn import build_custom_cnn
from src.models.mobilenetv2_model import freeze_backbone
from src.utils import load_config, project_path, save_json, set_seed

OUT_DIR = project_path("reports", "final")
FIG_DIR = project_path("reports", "figures", "final")


def build_mobilenet_with_head(num_classes: int, hidden_units: int | None) -> nn.Module:
    try:
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except AttributeError:
        backbone = models.mobilenet_v2(pretrained=True)

    in_features = backbone.classifier[1].in_features
    if hidden_units and hidden_units > 0:
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, num_classes),
        )
    else:
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )
    return backbone


def quick_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    history = {"val_acc": [], "val_loss": []}
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        val_loss /= total
        val_acc = correct / total
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return history


def run_head_ablation(cfg: dict, device: torch.device) -> list[dict]:
    num_classes = cfg["datasets"]["urbansound8k"]["num_classes"]
    class_names = cfg["datasets"]["urbansound8k"]["classes"]
    splits = project_path(cfg["datasets"]["urbansound8k"]["splits_dir"])
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]

    train_loader = DataLoader(
        MelSpectrogramImageDataset(splits / "train_processed.csv", pretrained_norm=True),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MelSpectrogramImageDataset(splits / "val_processed.csv", pretrained_norm=True),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        MelSpectrogramImageDataset(splits / "test_processed.csv", pretrained_norm=True),
        batch_size=batch_size,
        shuffle=False,
    )

    results = []
    for hidden in (0, 64, 128, 256):
        label = "direct_linear" if hidden == 0 else f"dense_{hidden}"
        print(f"Head ablation: {label} ...", flush=True)
        model = build_mobilenet_with_head(num_classes, hidden if hidden > 0 else None).to(device)
        freeze_backbone(model)
        start = time.time()
        history = quick_train(model, train_loader, val_loader, device, epochs=8, lr=lr)
        test_results = evaluate_model(model, test_loader, device, class_names)
        macro_recall = test_results["classification_report"]["macro avg"]["recall"]
        results.append(
            {
                "experiment": "model3_head_ablation",
                "head_config": label,
                "hidden_units": hidden if hidden > 0 else "none",
                "epochs": 8,
                "phase": "frozen_backbone_head_only",
                "best_val_acc": max(history["val_acc"]),
                "test_accuracy": test_results["metrics"]["accuracy"],
                "test_macro_recall": macro_recall,
                "test_macro_f1": test_results["metrics"]["macro_f1"],
                "train_time_sec": round(time.time() - start, 1),
            }
        )
        print(
            f"  {label}: acc={test_results['metrics']['accuracy']:.4f} "
            f"recall={macro_recall:.4f} f1={test_results['metrics']['macro_f1']:.4f}",
            flush=True,
        )
    return results


def run_dropout_ablation(cfg: dict, device: torch.device) -> list[dict]:
    num_classes = cfg["datasets"]["urbansound8k"]["num_classes"]
    class_names = cfg["datasets"]["urbansound8k"]["classes"]
    splits = project_path(cfg["datasets"]["urbansound8k"]["splits_dir"])
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]

    train_loader = DataLoader(
        MelSpectrogramImageDataset(splits / "train_processed.csv", pretrained_norm=False),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MelSpectrogramImageDataset(splits / "val_processed.csv", pretrained_norm=False),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        MelSpectrogramImageDataset(splits / "test_processed.csv", pretrained_norm=False),
        batch_size=batch_size,
        shuffle=False,
    )

    results = []
    for dropout in (0.0, 0.5):
        print(f"Dropout ablation: {dropout} ...", flush=True)
        model = build_custom_cnn(num_classes=num_classes, dropout=dropout).to(device)
        start = time.time()
        history = quick_train(model, train_loader, val_loader, device, epochs=12, lr=lr)
        test_results = evaluate_model(model, test_loader, device, class_names)
        macro_recall = test_results["classification_report"]["macro avg"]["recall"]
        results.append(
            {
                "experiment": "custom_cnn_dropout_ablation",
                "dropout": dropout,
                "epochs": 12,
                "best_val_acc": max(history["val_acc"]),
                "test_accuracy": test_results["metrics"]["accuracy"],
                "test_macro_recall": macro_recall,
                "test_macro_f1": test_results["metrics"]["macro_f1"],
                "train_time_sec": round(time.time() - start, 1),
            }
        )
        print(
            f"  dropout={dropout}: acc={test_results['metrics']['accuracy']:.4f} "
            f"recall={macro_recall:.4f} f1={test_results['metrics']['macro_f1']:.4f}",
            flush=True,
        )
    return results


def plot_ablation_charts(head_rows: list[dict], dropout_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if head_rows:
        labels = [r["head_config"].replace("_", " ") for r in head_rows]
        accs = [r["test_accuracy"] * 100 for r in head_rows]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, accs, color="#2563eb")
        plt.ylabel("Test accuracy (%)")
        plt.title("Model 3 — MobileNetV2 custom head size ablation (8 epochs, frozen backbone)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "ablation_model3_head_sizes.png", dpi=150, bbox_inches="tight")
        plt.close()

    if dropout_rows:
        labels = [f"dropout={r['dropout']}" for r in dropout_rows]
        accs = [r["test_accuracy"] * 100 for r in dropout_rows]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, accs, color="#059669")
        plt.ylabel("Test accuracy (%)")
        plt.title("Custom CNN — dropout comparison (12 epochs)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "ablation_custom_cnn_dropout.png", dpi=150, bbox_inches="tight")
        plt.close()


def main() -> None:
    cfg = load_config()
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    head_rows = run_head_ablation(cfg, device)
    dropout_rows = run_dropout_ablation(cfg, device)
    plot_ablation_charts(head_rows, dropout_rows)

    summary = {
        "title": "CA1 Hyperparameter Ablation Studies",
        "model3_head_ablation": head_rows,
        "custom_cnn_dropout_ablation": dropout_rows,
        "notes": [
            "Model 3 ablation: MobileNetV2 frozen backbone, train custom Dense head only for 8 epochs.",
            "Head sizes tested: direct linear, 64, 128, 256 nodes (course guideline: 64 x 2^i).",
            "Dropout ablation: Custom CNN trained 12 epochs with dropout 0.0 vs 0.5.",
            "Full production models used longer training and early stopping; ablations isolate hyperparameter effect.",
        ],
    }
    save_json(summary, OUT_DIR / "ca1_ablation_summary.json")
    print(f"Saved: {OUT_DIR / 'ca1_ablation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
