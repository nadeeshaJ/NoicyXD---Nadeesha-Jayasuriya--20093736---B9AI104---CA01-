"""Verify trained PyTorch checkpoints for deployment (not mock/random weights)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.confidence import normalized_entropy
from src.models import build_model, uses_pretrained_norm
from src.predict import rgb_to_model_tensor
from src.utils import load_config, project_path
import numpy as np

# Minimum file size (MB) — real trained weights for this project; tiny files = invalid
MIN_CHECKPOINT_MB: dict[str, float] = {
    "custom_cnn": 20.0,
    "resnet50": 80.0,
    "mobilenetv2": 7.5,
}

DEPLOYMENT_CHECKPOINTS: list[tuple[str, str, str]] = [
    ("urban", "mobilenetv2", "experiments/urbansound8k/mobilenetv2/best_model.pt"),
    ("urban", "resnet50", "experiments/urbansound8k/resnet50/best_model.pt"),
    ("urban", "custom_cnn", "experiments/urbansound8k/custom_cnn/best_model.pt"),
    ("animal", "mobilenetv2", "experiments/esc50_animals/mobilenetv2_imagenet_only/best_model.pt"),
]


def checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _metrics_sidecar(checkpoint: Path) -> Path:
    return checkpoint.parent / "test_metrics.json"


def verify_checkpoint_file(
    rel_path: str,
    model_key: str,
    *,
    run_probe: bool = False,
    device: torch.device | None = None,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Return validation status for one checkpoint path."""
    path = project_path(rel_path)
    result: dict[str, Any] = {
        "path": rel_path,
        "model_key": model_key,
        "exists": path.exists(),
        "size_mb": None,
        "size_ok": False,
        "metrics_found": False,
        "expected_macro_f1": None,
        "probe_ok": None,
        "status": "missing",
        "message": "",
    }

    if not path.exists():
        result["message"] = (
            "Checkpoint missing. Copy trained best_model.pt from your training machine "
            "or run: python scripts/setup_checkpoints.py --source PATH_TO_experiments"
        )
        return result

    size_mb = checkpoint_size_mb(path)
    result["size_mb"] = round(size_mb, 2)
    min_mb = MIN_CHECKPOINT_MB.get(model_key, 1.0)
    result["size_ok"] = size_mb >= min_mb

    metrics_path = _metrics_sidecar(path)
    if metrics_path.exists():
        import json

        with metrics_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        macro_f1 = payload.get("metrics", {}).get("macro_f1")
        if macro_f1 is None and "classification_report" in payload:
            macro_f1 = payload["classification_report"].get("macro avg", {}).get("f1-score")
        result["metrics_found"] = macro_f1 is not None
        result["expected_macro_f1"] = macro_f1

    if not result["size_ok"]:
        result["status"] = "invalid"
        result["message"] = (
            f"File too small ({size_mb:.2f} MB). Likely a mock/placeholder, not trained weights. "
            f"Expected at least {min_mb:.0f} MB for {model_key}."
        )
        return result

    trusted_by_metrics = (
        result["metrics_found"]
        and result["expected_macro_f1"] is not None
        and float(result["expected_macro_f1"]) >= 0.5
    )

    if run_probe and device is not None and not trusted_by_metrics:
        cfg = cfg or load_config()
        mode = "urban" if "urbansound8k" in rel_path else "animal"
        dataset_key = cfg["deployment"][mode]["dataset_key"]
        num_classes = len(cfg["datasets"][dataset_key]["classes"])
        probe = probe_checkpoint_trained(model_key, path, num_classes, device)
        result["probe_ok"] = probe["ok"]
        result["probe_entropy"] = probe["entropy"]
        result["probe_top_confidence"] = probe["top_confidence"]
        if not probe["ok"]:
            result["status"] = "suspect"
            result["message"] = probe["message"]
            return result
    elif run_probe and trusted_by_metrics:
        result["probe_ok"] = True
        result["message"] = "Trained checkpoint verified via test_metrics.json sidecar."

    result["status"] = "ok"
    if not result["message"]:
        result["message"] = "Trained checkpoint present."
    if result["metrics_found"] and result["expected_macro_f1"] is not None:
        result["message"] += f" Test macro F1: {float(result['expected_macro_f1']):.3f}."
    return result


@torch.no_grad()
def probe_checkpoint_trained(
    model_key: str,
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run a fixed synthetic Mel-RGB input through the network.

    Untrained/random weights usually yield ~uniform 10% (high entropy) or a fixed
    spurious class at ~100% on noise — both fail this probe.
    """
    model = build_model(model_key, num_classes=num_classes)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    rgb = np.full((224, 224, 3), 128, dtype=np.uint8)
    tensor = rgb_to_model_tensor(rgb, model_key).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    entropy = normalized_entropy(probs)
    top_idx = int(probs.argmax())
    top_conf = float(probs[top_idx])

    uniform = 1.0 / num_classes
    # Random/untrained: flat ~10% each OR one arbitrary class ~99% on constant input
    flat_like = abs(top_conf - uniform) < 0.03 and entropy > 0.92
    stuck_like = top_conf > 0.95 and entropy < 0.35

    if flat_like:
        return {
            "ok": False,
            "entropy": round(entropy, 3),
            "top_confidence": round(top_conf, 3),
            "top_label": str(top_idx),
            "message": (
                "Weights look untrained (flat ~10% on every class). "
                "Replace with real best_model.pt from run_step3_train.py."
            ),
        }
    if stuck_like:
        return {
            "ok": False,
            "entropy": round(entropy, 3),
            "top_confidence": round(top_conf, 3),
            "top_label": str(top_idx),
            "message": (
                "Weights look like random init (one class ~100% on noise input). "
                "Replace with trained checkpoints."
            ),
        }

    return {
        "ok": True,
        "entropy": round(entropy, 3),
        "top_confidence": round(top_conf, 3),
        "top_label": str(top_idx),
        "message": "Probe passed.",
    }


def verify_all_deployment_checkpoints(
    *,
    run_probe: bool = False,
    device: torch.device | None = None,
    cfg: dict | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    rows = []
    for _mode, model_key, rel_path in DEPLOYMENT_CHECKPOINTS:
        rows.append(
            verify_checkpoint_file(
                rel_path,
                model_key,
                run_probe=run_probe,
                device=device,
                cfg=cfg,
            )
        )

    ok_count = sum(1 for r in rows if r["status"] == "ok")
    required = next((r for r in rows if "mobilenetv2" in r["path"] and "urbansound8k" in r["path"]), rows[0])
    all_ok = ok_count == len(rows)
    deploy_ready = required["status"] == "ok"

    return {
        "all_ok": all_ok,
        "deploy_ready": deploy_ready,
        "checkpoints": rows,
        "summary": (
            f"{ok_count}/{len(rows)} deployment checkpoints verified."
            if deploy_ready
            else "Deployed MobileNetV2 checkpoint missing or invalid — predictions will be wrong."
        ),
    }


def assert_deploy_checkpoint_or_raise(cfg: dict | None = None) -> None:
    """Raise FileNotFoundError with setup instructions if primary model is bad."""
    report = verify_all_deployment_checkpoints(cfg=cfg)
    primary = next(
        c for c in report["checkpoints"] if c["path"] == "experiments/urbansound8k/mobilenetv2/best_model.pt"
    )
    if primary["status"] != "ok":
        raise FileNotFoundError(primary["message"])
