"""
Training loop for all three CA1 model architectures.

CA1 role:
    Model 1 (custom_cnn)  — train all layers from scratch.
    Model 2 (resnet50)    — Phase 1: frozen backbone + train head;
                              Phase 2: fine-tune top blocks + head.
    Model 3 (mobilenetv2)   — same two-phase transfer-learning strategy.

Features:
    - Early stopping on validation loss (patience from config.yaml)
    - Best checkpoint saved to experiments/{dataset}/{model}/best_model.pt
    - Training curves + test metrics + confusion matrix figures

Used by:
    scripts/run_step3_train.py, run_step4_esc50.py, run_ca1_ablation_studies.py
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import MelSpectrogramImageDataset
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_training_history, save_evaluation_results
from src.models import MODEL_NAMES, build_model, uses_pretrained_norm
from src.models.mobilenetv2_model import freeze_backbone as freeze_mobilenet
from src.models.mobilenetv2_model import unfreeze_top_layers as unfreeze_mobilenet
from src.models.resnet50_model import freeze_backbone as freeze_resnet
from src.models.resnet50_model import unfreeze_top_layers as unfreeze_resnet
from src.utils import load_config, project_path, save_json, set_seed


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(
    dataset_key: str,
    cfg: dict,
    model_name: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    splits_dir = project_path(cfg["datasets"][dataset_key]["splits_dir"])
    # ResNet/MobileNet need ImageNet normalisation; Custom CNN does not
    pretrained_norm = uses_pretrained_norm(model_name)
    use_cuda = torch.cuda.is_available()

    train_ds = MelSpectrogramImageDataset(
        splits_dir / "train_processed.csv",
        pretrained_norm=pretrained_norm,
    )
    val_ds = MelSpectrogramImageDataset(
        splits_dir / "val_processed.csv",
        pretrained_norm=pretrained_norm,
    )
    test_ds = MelSpectrogramImageDataset(
        splits_dir / "test_processed.csv",
        pretrained_norm=pretrained_norm,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,       # 0 avoids Windows multiprocessing issues
        "pin_memory": use_cuda,   # faster CPU→GPU transfer when CUDA available
    }
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),   # shuffle train only
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )


def load_backbone_weights(model: nn.Module, checkpoint_path: Path, model_name: str) -> int:
    """Load matching backbone weights from a checkpoint; skip classifier head."""
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")

    current = model.state_dict()
    loaded = 0
    for key, value in state.items():
        # Skip layers with mismatched shapes or the final classification head
        if key not in current or value.shape != current[key].shape:
            continue
        if model_name == "mobilenetv2" and key.startswith("classifier"):
            continue
        if model_name == "resnet50" and key.startswith("fc"):
            continue
        if model_name == "custom_cnn" and key.startswith("classifier"):
            continue
        current[key] = value
        loaded += 1
    model.load_state_dict(current)
    return loaded


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train and optimizer is not None:
            optimizer.zero_grad()  # clear gradients from previous batch

        # torch.no_grad() during validation — no backprop, saves memory
        with torch.set_grad_enabled(train and optimizer is not None):
            outputs = model(images)          # logits shape: (batch, num_classes)
            loss = criterion(outputs, labels)

        if train and optimizer is not None:
            loss.backward()                  # compute gradients
            optimizer.step()                 # update weights

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)        # class with highest logit
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model_name: str,
    dataset_key: str = "urbansound8k",
    cfg: dict | None = None,
    epochs: int | None = None,
    run_name: str | None = None,
    backbone_checkpoint: Path | str | None = None,
    figure_step: str = "step3",
) -> dict[str, Any]:
    cfg = cfg or load_config()
    set_seed(cfg["training"]["seed"])
    device = get_device()

    train_cfg = cfg["training"]
    num_classes = cfg["datasets"][dataset_key]["num_classes"]
    class_names = cfg["datasets"][dataset_key]["classes"]
    exp_subdir = run_name or model_name
    exp_dir = project_path(cfg["experiments"][f"{dataset_key}_dir"], exp_subdir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        dataset_key,
        cfg,
        model_name,
        batch_size=train_cfg["batch_size"],
    )

    model = build_model(model_name, num_classes=num_classes).to(device)
    if backbone_checkpoint:
        loaded = load_backbone_weights(model, Path(backbone_checkpoint), model_name)
        print(f"[{exp_subdir}] Loaded {loaded} backbone tensors from {backbone_checkpoint}", flush=True)
    criterion = nn.CrossEntropyLoss()

    max_epochs = epochs or train_cfg["epochs"]
    patience = train_cfg["early_stopping_patience"]
    base_lr = train_cfg["learning_rate"]

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    phase_log: list[dict] = []

    # --- Phase 1: train classifier head only (ImageNet backbone frozen) ---
    if model_name == "resnet50":
        freeze_resnet(model)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
        phase_epochs = min(3, max_epochs // 4)
        phase_log.append({"phase": "head_only", "epochs": phase_epochs})
        for epoch in range(1, phase_epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, False)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            print(f"[{model_name}] P1 Epoch {epoch}: val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        unfreeze_resnet(model)
        # Lower LR for backbone, higher for head — standard transfer-learning recipe
        optimizer = Adam(
            [
                {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")], "lr": base_lr * 0.1},
                {"params": model.fc.parameters(), "lr": base_lr},
            ]
        )
        remaining = max_epochs - phase_epochs
    elif model_name == "mobilenetv2":
        freeze_mobilenet(model)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
        phase_epochs = min(3, max_epochs // 4)
        phase_log.append({"phase": "head_only", "epochs": phase_epochs})
        for epoch in range(1, phase_epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, False)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            print(f"[{model_name}] P1 Epoch {epoch}: val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        unfreeze_mobilenet(model)
        optimizer = Adam(
            [
                {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier")], "lr": base_lr * 0.1},
                {"params": model.classifier.parameters(), "lr": base_lr},
            ]
        )
        remaining = max_epochs - phase_epochs
    else:
        # Model 1 — Custom CNN: no freeze/unfreeze; all layers trainable from epoch 1
        optimizer = Adam(model.parameters(), lr=base_lr)
        remaining = max_epochs
        phase_log.append({"phase": "full_training", "epochs": remaining})

    start_time = time.time()
    # --- Phase 2 (or full training for Custom CNN) with early stopping ---
    for epoch in range(1, remaining + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, False)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(
            f"[{model_name}] Epoch {len(history['train_loss'])}: "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}",
            flush=True,
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())  # save weights from best epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Stop if validation loss hasn't improved for `patience` epochs
                print(f"[{model_name}] Early stopping at epoch {len(history['train_loss'])}")
                break

    train_time = time.time() - start_time
    model.load_state_dict(best_state)  # restore best weights (not last epoch)
    torch.save(model.state_dict(), exp_dir / "best_model.pt")

    fig_dir = project_path("reports", "figures", figure_step, dataset_key, exp_subdir)
    plot_training_history(history, fig_dir / "training_history.png")

    test_results = evaluate_model(model, test_loader, device, class_names)
    save_evaluation_results(test_results, exp_dir)
    plot_confusion_matrix(
        __import__("numpy").array(test_results["confusion_matrix"]),
        class_names,
        f"{exp_subdir} — Test Confusion Matrix",
        fig_dir / "confusion_matrix_test.png",
    )

    summary = {
        "model_name": model_name,
        "run_name": exp_subdir,
        "dataset": dataset_key,
        "device": str(device),
        "backbone_checkpoint": str(backbone_checkpoint) if backbone_checkpoint else None,
        "trainable_parameters": count_parameters(model),
        "total_epochs_run": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "train_time_sec": round(train_time, 1),
        "test_metrics": test_results["metrics"],
        "phases": phase_log,
        "history": history,
    }
    save_json(summary, exp_dir / "training_summary.json")
    return summary
