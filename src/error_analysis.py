"""
Error analysis: misclassified examples, confusion insights, and benchmarking.

CA1 role:
    Produces confusion matrices, top confused class pairs, case-study figures,
    and GPU inference latency benchmarks for all three urban models.

Outputs:
    reports/step6/, reports/figures/step6/

Used by:
    scripts/run_step6_error_analysis.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from src.audio_utils import load_audio
from src.dataset import MelSpectrogramImageDataset, default_transform
from src.evaluate import collect_predictions, plot_confusion_matrix
from src.models import build_model, uses_pretrained_norm
from src.spectrogram import compute_mel_spectrogram, normalize_spectrogram
from src.utils import project_path, save_json

PAIR_EXPLANATIONS: dict[tuple[str, str], str] = {
    ("siren", "car_horn"): (
        "Both classes contain short, high-energy tonal bursts. Car horns and sirens "
        "share similar frequency content in Mel images, especially when the siren is "
        "distant or partially masked by background noise."
    ),
    ("car_horn", "siren"): (
        "A brief horn blast can resemble the onset of a siren cycle in a 4-second "
        "spectrogram window, leading the model to confuse transient tonal events."
    ),
    ("drilling", "jackhammer"): (
        "Drilling and jackhammer sounds are both rhythmic mechanical impacts with "
        "overlapping low-mid frequency energy. The CNN sees similar vertical striation "
        "patterns in the Mel-spectrogram."
    ),
    ("jackhammer", "drilling"): (
        "Jackhammer pulses can look like sustained drilling texture when only a short "
        "segment is visible in the fixed 4 s clip."
    ),
    ("dog_bark", "children_playing"): (
        "Outdoor recordings of children playing may contain dog barks or similar "
        "transient broadband events. Ambient human activity adds noise that obscures "
        "class-specific patterns."
    ),
    ("children_playing", "dog_bark"): (
        "Isolated sharp transients in a children-playing clip (shouts, claps) can "
        "resemble bark-like bursts in the Mel representation."
    ),
    ("children_playing", "street_music"): (
        "Both classes often appear in outdoor urban scenes with mixed background "
        "activity. Mel-spectrogram energy can spread across similar frequency bands."
    ),
    ("street_music", "children_playing"): (
        "Music with percussive elements and human vocalisations can overlap with the "
        "spectral texture of children playing in public spaces."
    ),
    ("siren", "street_music"): (
        "Music with strong high-frequency content or horn-like instruments may overlap "
        "with siren harmonics in the Mel image."
    ),
    ("siren", "children_playing"): (
        "Distant or intermittent sirens mixed with outdoor background activity can "
        "produce broadband energy similar to children playing. The 4 s window may "
        "capture more ambient noise than clear siren cycles."
    ),
    ("dog_bark", "street_music"): (
        "Urban street-music recordings often include percussive beats and vocal "
        "bursts that resemble isolated bark transients after Mel conversion."
    ),
    ("engine_idling", "air_conditioner"): (
        "Steady low-frequency hums from idling engines and air-conditioning units "
        "produce similar sustained horizontal bands in Mel-spectrograms."
    ),
    ("air_conditioner", "engine_idling"): (
        "Continuous mechanical hum without clear transients is difficult to separate "
        "when recording conditions compress dynamic range."
    ),
    ("dog", "hen"): (
        "Animal vocalisations in ESC-50 can share harmonic structure. Short clips "
        "with limited context increase confusion between bird and mammal calls."
    ),
    ("hen", "rooster"): (
        "Hen and rooster calls occupy similar frequency ranges and temporal patterns "
        "in small animal-sound datasets."
    ),
    ("cow", "sheep"): (
        "Low-frequency animal calls recorded in similar farm environments can appear "
        "nearly identical after Mel conversion and resizing."
    ),
}


def load_trained_model(
    model_name: str,
    num_classes: int,
    checkpoint_path: Path,
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


def count_model_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    model_name: str,
    checkpoint_path: Path,
    warmup: int = 20,
    runs: int = 100,
) -> dict[str, float]:
    pretrained = uses_pretrained_norm(model_name)
    dummy = default_transform(pretrained=pretrained)(Image.new("RGB", (224, 224))).unsqueeze(0)
    dummy = dummy.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    timings: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    params = count_model_parameters(model)
    return {
        **params,
        "model_file_size_mb": round(file_size_mb, 2),
        "inference_ms_mean": round(float(np.mean(timings)), 3),
        "inference_ms_std": round(float(np.std(timings)), 3),
        "inference_runs": runs,
        "device": str(device),
    }


def top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_k: int = 5,
) -> list[dict]:
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                    }
                )
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def build_misclassification_records(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
) -> list[dict]:
    records: list[dict] = []
    for idx, (_, row) in enumerate(test_df.iterrows()):
        true_idx = int(y_true[idx])
        pred_idx = int(y_pred[idx])
        if true_idx == pred_idx:
            continue
        records.append(
            {
                "index": idx,
                "image_path": row["image_path"],
                "audio_path": row.get("audio_path", ""),
                "true_label": class_names[true_idx],
                "predicted_label": class_names[pred_idx],
                "true_idx": true_idx,
                "predicted_idx": pred_idx,
                "confidence": float(y_probs[idx][pred_idx]),
                "true_class_prob": float(y_probs[idx][true_idx]),
            }
        )
    return records


def explain_misclassification(true_label: str, predicted_label: str) -> str:
    key = (true_label, predicted_label)
    if key in PAIR_EXPLANATIONS:
        return PAIR_EXPLANATIONS[key]
    return (
        f"The model predicted `{predicted_label}` instead of `{true_label}`. "
        "These classes may share overlapping spectral patterns in the 4 s Mel-spectrogram "
        "window, or the recording may contain ambiguous or mixed environmental cues."
    )


def select_case_studies(
    misclassified: list[dict],
    confused_pairs: list[dict],
    max_examples: int = 5,
) -> list[dict]:
    selected: list[dict] = []
    used_indices: set[int] = set()

    for pair in confused_pairs:
        for ex in misclassified:
            if ex["index"] in used_indices:
                continue
            if (
                ex["true_label"] == pair["true_class"]
                and ex["predicted_label"] == pair["predicted_class"]
            ):
                enriched = dict(ex)
                enriched["explanation"] = explain_misclassification(
                    ex["true_label"], ex["predicted_label"]
                )
                enriched["confusion_pair_count"] = pair["count"]
                selected.append(enriched)
                used_indices.add(ex["index"])
                break
        if len(selected) >= max_examples:
            break

    for ex in misclassified:
        if len(selected) >= max_examples:
            break
        if ex["index"] in used_indices:
            continue
        enriched = dict(ex)
        enriched["explanation"] = explain_misclassification(
            ex["true_label"], ex["predicted_label"]
        )
        enriched["confusion_pair_count"] = 0
        selected.append(enriched)
        used_indices.add(ex["index"])

    return selected


def plot_error_examples(
    examples: list[dict],
    out_path: Path,
    title: str,
) -> None:
    n = len(examples)
    if n == 0:
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ex in enumerate(examples):
        img = np.array(Image.open(ex["image_path"]))
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {ex['true_label']}\nPred: {ex['predicted_label']}",
            fontsize=10,
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confused_pairs_bar(
    pairs: list[dict],
    out_path: Path,
    title: str,
    top_k: int = 8,
) -> None:
    if not pairs:
        return
    subset = pairs[:top_k]
    labels = [f"{p['true_class']}\n-> {p['predicted_class']}" for p in subset]
    counts = [p["count"] for p in subset]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts, y=labels, orient="h", color="#b91c1c")
    plt.xlabel("Misclassification count")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_case_study(
    example: dict,
    cfg: dict,
    out_path: Path,
    case_number: int,
) -> None:
    audio_path = Path(example.get("audio_path", ""))
    if not audio_path.exists():
        plot_error_examples([example], out_path, f"Case Study {case_number}")
        return

    waveform, sr = load_audio(
        audio_path,
        sample_rate=cfg["audio"]["sample_rate"],
        duration_sec=cfg["audio"]["duration_sec"],
        mono=cfg["audio"]["mono"],
    )
    mel = compute_mel_spectrogram(waveform, sr, cfg["spectrogram"])
    if cfg["spectrogram"].get("normalize", True):
        mel_norm = normalize_spectrogram(mel)
    else:
        mel_norm = mel

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    times = np.linspace(0, len(waveform) / sr, num=len(waveform))
    axes[0].plot(times, waveform, color="#2563eb", linewidth=0.8)
    axes[0].set_title("Waveform (4 s)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    librosa.display.specshow(
        mel,
        sr=sr,
        hop_length=cfg["spectrogram"]["hop_length"],
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title("Mel-spectrogram")

    rgb = np.array(Image.open(example["image_path"]))
    axes[2].imshow(rgb)
    axes[2].set_title("Model input (224x224 RGB)")
    axes[2].axis("off")

    fig.suptitle(
        f"Case {case_number}: True={example['true_label']} | "
        f"Pred={example['predicted_label']} | "
        f"Confidence={example.get('confidence', 0):.1%}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mel_vs_mfcc(
    audio_path: Path,
    cfg: dict,
    out_path: Path,
) -> None:
    waveform, sr = load_audio(
        audio_path,
        sample_rate=cfg["audio"]["sample_rate"],
        duration_sec=cfg["audio"]["duration_sec"],
        mono=cfg["audio"]["mono"],
    )
    mel = compute_mel_spectrogram(waveform, sr, cfg["spectrogram"])
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=13,
        n_fft=cfg["spectrogram"]["n_fft"],
        hop_length=cfg["spectrogram"]["hop_length"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    librosa.display.specshow(
        mel,
        sr=sr,
        hop_length=cfg["spectrogram"]["hop_length"],
        x_axis="time",
        y_axis="mel",
        ax=axes[0],
        cmap="magma",
    )
    axes[0].set_title("Mel-spectrogram (CNN input representation)")

    librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=cfg["spectrogram"]["hop_length"],
        x_axis="time",
        ax=axes[1],
        cmap="coolwarm",
    )
    axes[1].set_title("MFCC coefficients (13 x time)")
    axes[1].set_ylabel("MFCC index")

    plt.suptitle("Mel-spectrogram vs MFCC for the same clip", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_error_analysis_for_run(
    cfg: dict,
    dataset_key: str,
    model_name: str,
    run_name: str,
    figure_dir: Path,
    report_dir: Path,
    max_examples: int = 6,
    case_studies: int = 0,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["datasets"][dataset_key]["classes"]
    exp_dir = project_path(cfg["experiments"][f"{dataset_key}_dir"], run_name)
    checkpoint = exp_dir / "best_model.pt"
    test_csv = project_path(cfg["datasets"][dataset_key]["splits_dir"]) / "test_processed.csv"

    model = load_trained_model(model_name, len(class_names), checkpoint, device)
    test_df = pd.read_csv(test_csv)
    test_df = test_df[test_df["status"].isin(["processed", "skipped"])].reset_index(drop=True)

    loader = DataLoader(
        MelSpectrogramImageDataset(test_csv, pretrained_norm=uses_pretrained_norm(model_name)),
        batch_size=32,
        shuffle=False,
    )
    y_true, y_pred, y_probs = collect_predictions(model, loader, device)
    confused = top_confused_pairs(y_true, y_pred, class_names)

    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plot_confusion_matrix(
        cm,
        class_names,
        f"Confusion Matrix — {run_name}",
        figure_dir / f"confusion_matrix_{run_name}.png",
    )
    plot_confused_pairs_bar(
        confused,
        figure_dir / f"confused_pairs_{run_name}.png",
        f"Top Confused Pairs — {run_name}",
    )

    misclassified = build_misclassification_records(
        test_df, y_true, y_pred, y_probs, class_names
    )
    examples = misclassified[:max_examples]
    plot_error_examples(
        examples,
        figure_dir / f"error_examples_{run_name}.png",
        f"Misclassified Examples — {run_name}",
    )

    selected_cases: list[dict] = []
    if case_studies > 0:
        selected_cases = select_case_studies(misclassified, confused, max_examples=case_studies)
        case_dir = figure_dir / "case_studies"
        for i, case in enumerate(selected_cases, start=1):
            plot_case_study(case, cfg, case_dir / f"case_study_{i:02d}.png", i)

    result = {
        "run_name": run_name,
        "dataset_key": dataset_key,
        "model_name": model_name,
        "top_confused_pairs": confused,
        "misclassified_count": len(misclassified),
        "misclassified_examples": examples,
        "case_studies": selected_cases,
        "test_accuracy": float((y_true == y_pred).mean()),
    }
    save_json(result, report_dir / f"error_analysis_{run_name}.json")
    return result


def run_urban_benchmarks(cfg: dict, report_dir: Path) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["datasets"]["urbansound8k"]["num_classes"]
    urban_dir = project_path(cfg["experiments"]["urbansound8k_dir"])
    rows: list[dict] = []

    for model_name in ("custom_cnn", "resnet50", "mobilenetv2"):
        checkpoint = urban_dir / model_name / "best_model.pt"
        summary_path = urban_dir / model_name / "training_summary.json"
        metrics_path = urban_dir / model_name / "test_metrics.json"
        training_summary = {}
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as f:
                training_summary = json.load(f)
        test_macro_recall = None
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as f:
                test_macro_recall = (
                    json.load(f).get("classification_report", {}).get("macro avg", {}).get("recall")
                )

        model = load_trained_model(model_name, num_classes, checkpoint, device)
        bench = benchmark_inference(model, device, model_name, checkpoint)
        epochs = training_summary.get("total_epochs_run", 0)
        train_time_sec = training_summary.get("train_time_sec", 0.0)
        epoch_time_sec = round(train_time_sec / epochs, 1) if epochs else 0.0

        row = {
            "model": model_name,
            **bench,
            "training_epochs": epochs,
            "training_time_sec": train_time_sec,
            "training_time_per_epoch_sec": epoch_time_sec,
            "test_accuracy": training_summary.get("test_metrics", {}).get("accuracy"),
            "test_macro_recall": test_macro_recall,
            "test_macro_f1": training_summary.get("test_metrics", {}).get("macro_f1"),
        }
        rows.append(row)

    save_json({"benchmarks": rows}, report_dir / "inference_benchmarks.json")
    return rows
