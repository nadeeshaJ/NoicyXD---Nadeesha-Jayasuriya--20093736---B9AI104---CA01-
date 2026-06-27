"""Step 2: Preprocessing pipeline, splits, validation, and demo figures.

Run:
    python scripts/run_step2_preprocess.py

Outputs:
    data/splits/{urbansound8k,esc50_animals}/train|val|test.csv
    data/processed/{dataset}/images/*.png
    reports/figures/step2/preprocessing_pipeline_diagram.png  (Figure 2.1)
    reports/figures/step2/preprocessing_examples_urbansound8k.png
    reports/step2/step2_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.audio_utils import load_audio
from src.augmentation import add_gaussian_noise, spec_augment
from src.preprocess import load_class_mapping, preprocess_split
from src.spectrogram import (
    compute_mel_spectrogram,
    mel_to_rgb_image,
    normalize_spectrogram,
    waveform_to_rgb_mel_image,
)
from src.splits import create_esc50_splits, create_urbansound8k_splits
from src.utils import load_config, project_path, save_json, set_seed

FIG_DIR = project_path("reports", "figures", "step2")
REPORTS_DIR = project_path("reports", "step2")


def draw_pipeline_diagram() -> None:
    """Create horizontal preprocessing flow diagram for the report."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(18, 4.2))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    steps = [
        ("Audio\n(.wav)", "#dbeafe", "#2563eb"),
        ("Load +\nmono", "#dbeafe", "#2563eb"),
        ("Resample\n22,050 Hz", "#dbeafe", "#2563eb"),
        ("Pad/trim\n4 s", "#dbeafe", "#2563eb"),
        ("Mel-spec\n128 bins", "#ffedd5", "#ea580c"),
        ("Power\n→ dB", "#ffedd5", "#ea580c"),
        ("Normalise\n[0, 1]", "#ffedd5", "#ea580c"),
        ("Resize\n224×224", "#dcfce7", "#16a34a"),
        ("RGB\n×3", "#dcfce7", "#16a34a"),
        ("PNG →\nCNN input", "#dcfce7", "#16a34a"),
    ]

    n = len(steps)
    box_w, box_h = 1.45, 1.05
    gap = 0.28
    total_w = n * box_w + (n - 1) * gap
    x_start = (18 - total_w) / 2
    y_center = 2.05

    box_bounds: list[tuple[float, float, float, float]] = []
    for i, (label, face, edge) in enumerate(steps):
        x_left = x_start + i * (box_w + gap)
        y_bottom = y_center - box_h / 2
        box = FancyBboxPatch(
            (x_left, y_bottom),
            box_w,
            box_h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(
            x_left + box_w / 2,
            y_center,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.15,
        )
        box_bounds.append((x_left, y_bottom, x_left + box_w, y_bottom + box_h))

    for i in range(len(box_bounds) - 1):
        x_right = box_bounds[i][2]
        x_left_next = box_bounds[i + 1][0]
        arrow = FancyArrowPatch(
            (x_right, y_center),
            (x_left_next, y_center),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.2,
            color="#4b5563",
            shrinkA=2,
            shrinkB=2,
        )
        ax.add_patch(arrow)

    ax.set_title(
        "Preprocessing Pipeline — Audio to Mel-Spectrogram RGB Image",
        fontsize=13,
        fontweight="bold",
        pad=12,
        y=0.98,
    )

    legend_items = [
        ("Audio preparation", "#dbeafe", "#2563eb"),
        ("Time-frequency transform", "#ffedd5", "#ea580c"),
        ("Image formatting", "#dcfce7", "#16a34a"),
    ]
    lx = x_start
    for label, face, edge in legend_items:
        ax.add_patch(
            FancyBboxPatch(
                (lx, 0.28),
                0.28,
                0.22,
                boxstyle="round,pad=0.02",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.4,
            )
        )
        ax.text(lx + 0.36, 0.39, label, va="center", fontsize=8)
        lx += 2.35

    ax.text(
        9,
        0.08,
        "Output: 224×224×3 Mel-spectrogram RGB image for Custom CNN / ResNet50 / MobileNetV2",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#6b7280",
        style="italic",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "preprocessing_pipeline_diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def create_demo_figures(cfg: dict) -> None:
    """Waveform → Mel → RGB examples and SpecAugment demo."""
    audio_cfg = cfg["audio"]
    spec_cfg = cfg["spectrogram"]
    image_cfg = cfg["image"]
    aug_cfg = cfg["augmentation"]

    urban_train = pd.read_csv(
        project_path(cfg["datasets"]["urbansound8k"]["splits_dir"]) / "train.csv"
    )
    sample_classes = urban_train["class"].unique()[:4]

    fig, axes = plt.subplots(len(sample_classes), 3, figsize=(12, 10))
    for row_idx, cls in enumerate(sample_classes):
        row = urban_train[urban_train["class"] == cls].iloc[0]
        waveform, sr = load_audio(
            row["audio_path"],
            sample_rate=audio_cfg["sample_rate"],
            duration_sec=audio_cfg["duration_sec"],
        )
        mel = compute_mel_spectrogram(waveform, sr, spec_cfg)
        mel_norm = normalize_spectrogram(mel)
        rgb = waveform_to_rgb_mel_image(waveform, sr, spec_cfg, image_cfg)

        axes[row_idx, 0].plot(np.linspace(0, len(waveform) / sr, len(waveform)), waveform, color="navy")
        axes[row_idx, 0].set_title(f"{cls}\nWaveform")
        axes[row_idx, 0].set_xlabel("Time (s)")

        im = axes[row_idx, 1].imshow(mel_norm, origin="lower", aspect="auto", cmap="magma")
        axes[row_idx, 1].set_title("Mel-Spectrogram (norm)")
        fig.colorbar(im, ax=axes[row_idx, 1], fraction=0.046)

        axes[row_idx, 2].imshow(rgb)
        axes[row_idx, 2].set_title("RGB Image 224×224")
        axes[row_idx, 2].axis("off")

    plt.suptitle("UrbanSound8K — Preprocessing Examples", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "preprocessing_examples_urbansound8k.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SpecAugment demo
    row = urban_train.iloc[0]
    waveform, sr = load_audio(row["audio_path"], audio_cfg["sample_rate"], audio_cfg["duration_sec"])
    mel = normalize_spectrogram(compute_mel_spectrogram(waveform, sr, spec_cfg))
    mel_aug = spec_augment(mel, aug_cfg, rng=np.random.default_rng(42))
    mel_aug = add_gaussian_noise(mel_aug, std=0.02, rng=np.random.default_rng(42))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Original Mel-Spectrogram")
    axes[1].imshow(mel_aug, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("After SpecAugment + Noise")
    axes[2].imshow(mel_to_rgb_image(mel_aug, image_cfg["height"], image_cfg["width"]))
    axes[2].set_title("Augmented RGB Image")
    axes[2].axis("off")
    plt.suptitle("Training Augmentation Demo (SpecAugment)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "specaugment_demo.png", dpi=150, bbox_inches="tight")
    plt.close()


def validate_processed(dataset_key: str, cfg: dict) -> dict:
    """Validate PNG outputs: shape, errors, label integrity."""
    dataset_cfg = cfg["datasets"][dataset_key]
    splits_dir = project_path(dataset_cfg["splits_dir"])
    results = {"dataset": dataset_key, "splits": {}}

    for split_name in ("train", "val", "test"):
        proc_path = splits_dir / f"{split_name}_processed.csv"
        if not proc_path.exists():
            continue
        df = pd.read_csv(proc_path)
        errors = df[df["status"].str.startswith("error")].shape[0]
        skipped = (df["status"] == "skipped").sum()
        processed = (df["status"] == "processed").sum()

        shapes = []
        bad_shape = 0
        for img_path in df[df["image_path"] != ""]["image_path"]:
            if not Path(img_path).exists():
                bad_shape += 1
                continue
            img = np.array(Image.open(img_path))
            shapes.append(img.shape)
            if img.shape != (224, 224, 3):
                bad_shape += 1

        results["splits"][split_name] = {
            "total": len(df),
            "processed": int(processed),
            "skipped": int(skipped),
            "errors": int(errors),
            "bad_shape_or_missing": int(bad_shape),
            "unique_labels": int(df["label"].nunique()),
        }

    return results


def write_validation_report(validation: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json({"validation": validation}, REPORTS_DIR / "step2_validation.json")

    rows = []
    for ds in validation:
        for split_name, stats in ds["splits"].items():
            rows.append(
                {
                    "dataset": ds["dataset"],
                    "split": split_name,
                    **stats,
                }
            )
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "preprocessing_validation.csv", index=False)


def run_preprocessing(cfg: dict, overwrite: bool = False) -> None:
    for dataset_key in ("urbansound8k", "esc50_animals"):
        splits_dir = project_path(cfg["datasets"][dataset_key]["splits_dir"])
        for split_name in ("train", "val", "test"):
            split_path = splits_dir / f"{split_name}.csv"
            if not split_path.exists():
                continue
            split_df = pd.read_csv(split_path)
            # Convert each wav in the split to a 224×224 Mel PNG
            preprocess_split(split_name, split_df, cfg, dataset_key, overwrite=overwrite)


def main() -> None:
    set_seed(42)
    cfg = load_config()
    sns.set_theme(style="whitegrid")

    # Step 2.1 — create train/val/test CSVs with audio paths and class indices
    print("Step 2.1 — Creating dataset splits...")
    urban_map = load_class_mapping(project_path(cfg["datasets"]["urbansound8k"]["splits_dir"]))
    esc_map = load_class_mapping(project_path(cfg["datasets"]["esc50_animals"]["splits_dir"]))
    create_urbansound8k_splits(cfg, urban_map)
    create_esc50_splits(cfg, esc_map)

    # Step 2.2 — batch convert all audio clips to Mel-spectrogram PNG images
    print("Step 2.2 — Running preprocessing (audio to Mel PNG)...")
    run_preprocessing(cfg, overwrite=False)

    # Step 2.3 — report figures (pipeline diagram, examples, SpecAugment demo)
    print("Step 2.3 — Generating figures...")
    draw_pipeline_diagram()
    create_demo_figures(cfg)

    # Step 2.4 — verify every PNG exists and has correct 224×224×3 shape
    print("Step 2.4 — Validating outputs...")
    validation = [
        validate_processed("urbansound8k", cfg),
        validate_processed("esc50_animals", cfg),
    ]
    write_validation_report(validation)

    summary = {
        "step": 2,
        "title": "Preprocessing Pipeline",
        "audio": cfg["audio"],
        "spectrogram": cfg["spectrogram"],
        "image": cfg["image"],
        "augmentation": cfg["augmentation"],
        "validation": validation,
        "figures": [
            "preprocessing_pipeline_diagram.png",
            "preprocessing_examples_urbansound8k.png",
            "specaugment_demo.png",
        ],
    }
    save_json(summary, REPORTS_DIR / "step2_summary.json")

    print("Step 2 complete.")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Validation: {REPORTS_DIR / 'preprocessing_validation.csv'}")


if __name__ == "__main__":
    main()
