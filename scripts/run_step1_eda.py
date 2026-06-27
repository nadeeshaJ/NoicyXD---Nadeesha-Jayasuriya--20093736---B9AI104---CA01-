"""Step 1: Dataset acquisition verification, EDA, class mappings, and inventory.

Run:
    python scripts/run_step1_eda.py

Outputs:
    reports/step1/step1_summary.json, dataset_inventory.csv
    reports/figures/step1/*.png  (class distribution, duration, waveforms)
    data/splits/*/class_mapping.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as: python scripts/run_step1_eda.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import load_config, project_path, save_json, set_seed

FIG_DIR = project_path("reports", "figures", "step1")
URBAN_SPLITS = project_path("data", "splits", "urbansound8k")
ESC50_SPLITS = project_path("data", "splits", "esc50_animals")
REPORTS_DIR = project_path("reports", "step1")


def build_class_mapping(classes: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(classes)}


def urban_audio_path(row: pd.Series, raw_dir: Path) -> Path:
    return raw_dir / "audio" / f"fold{row['fold']}" / row["slice_file_name"]


def run_urbansound8k_eda(cfg: dict) -> pd.DataFrame:
    urban_cfg = cfg["datasets"]["urbansound8k"]
    raw_dir = project_path(urban_cfg["raw_dir"])
    meta_path = project_path(urban_cfg["metadata"])
    df = pd.read_csv(meta_path)

    df["duration_sec"] = df["end"] - df["start"]
    df["audio_path"] = df.apply(lambda r: str(urban_audio_path(r, raw_dir)), axis=1)
    df["file_exists"] = df["audio_path"].apply(lambda p: Path(p).exists())

    missing = (~df["file_exists"]).sum()
    if missing:
        print(f"Warning: {missing} UrbanSound8K files missing from disk")

    # Class mapping
    classes = urban_cfg["classes"]
    mapping = build_class_mapping(classes)
    save_json(
        {"classes": classes, "class_to_idx": mapping, "idx_to_class": {i: c for c, i in mapping.items()}},
        URBAN_SPLITS / "class_mapping.json",
    )

    # Inventory row
    inventory = {
        "dataset": "UrbanSound8K",
        "source": "Salamon & Bello (2014)",
        "total_clips": len(df),
        "num_classes": len(classes),
        "clips_per_class_min": int(df["class"].value_counts().min()),
        "clips_per_class_max": int(df["class"].value_counts().max()),
        "duration_min_sec": round(df["duration_sec"].min(), 3),
        "duration_max_sec": round(df["duration_sec"].max(), 3),
        "duration_mean_sec": round(df["duration_sec"].mean(), 3),
        "num_folds": int(df["fold"].nunique()),
        "test_fold": urban_cfg["test_fold"],
        "train_folds": "1-9",
        "raw_path": str(raw_dir),
        "metadata_file": str(meta_path),
        "missing_files": int(missing),
    }

    # Plots
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    order = df["class"].value_counts().index
    sns.countplot(data=df, x="class", order=order, palette="viridis")
    plt.title("UrbanSound8K — Clips per Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "urban_class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["duration_sec"], bins=40, kde=True, color="steelblue")
    plt.title("UrbanSound8K — Clip Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "urban_duration_distribution.png", dpi=150)
    plt.close()

    fold_class = pd.crosstab(df["class"], df["fold"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(fold_class, annot=True, fmt="d", cmap="Blues")
    plt.title("UrbanSound8K — Class Counts per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "urban_fold_heatmap.png", dpi=150)
    plt.close()

    # Sample waveforms (one per class)
    fig, axes = plt.subplots(5, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, cls in enumerate(classes):
        sample = df[df["class"] == cls].iloc[0]
        y, sr = librosa.load(sample["audio_path"], sr=None, mono=True)
        axes[i].plot(np.linspace(0, len(y) / sr, len(y)), y, color="navy", linewidth=0.6)
        axes[i].set_title(cls.replace("_", " "))
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude")
    plt.suptitle("UrbanSound8K — Example Waveforms per Class", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "urban_sample_waveforms.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Sample rate check on subset
    sample_rates = []
    for path in df["audio_path"].sample(min(200, len(df)), random_state=42):
        _, sr = librosa.load(path, sr=None, mono=True, duration=0.5)
        sample_rates.append(sr)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=sample_rates, palette="muted")
    plt.title("UrbanSound8K — Sample Rate (200 random clips)")
    plt.xlabel("Sample rate (Hz)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "urban_sample_rate.png", dpi=150)
    plt.close()
    inventory["sample_rate_hz"] = int(pd.Series(sample_rates).mode().iloc[0])

    df.to_csv(URBAN_SPLITS / "metadata_enriched.csv", index=False)
    return pd.DataFrame([inventory])


def run_esc50_eda(cfg: dict) -> pd.DataFrame:
    esc_cfg = cfg["datasets"]["esc50_animals"]
    raw_dir = project_path(esc_cfg["raw_dir"])
    meta_path = project_path(esc_cfg["metadata"])
    df = pd.read_csv(meta_path)

    audio_dir = raw_dir / "audio"
    df["audio_path"] = df["filename"].apply(lambda f: str(audio_dir / f))
    df["file_exists"] = df["audio_path"].apply(lambda p: Path(p).exists())

    classes = esc_cfg["classes"]
    mapping = build_class_mapping(classes)
    save_json(
        {"classes": classes, "class_to_idx": mapping, "idx_to_class": {i: c for c, i in mapping.items()}},
        ESC50_SPLITS / "class_mapping.json",
    )

    missing = (~df["file_exists"]).sum()
    inventory = {
        "dataset": "ESC-50 Animals (HF subset)",
        "source": "DynamicSuperb/EnvironmentalSoundClassification_ESC50-Animals",
        "total_clips": len(df),
        "num_classes": len(classes),
        "clips_per_class_min": int(df["label"].value_counts().min()),
        "clips_per_class_max": int(df["label"].value_counts().max()),
        "duration_min_sec": round(df["duration_sec"].min(), 3),
        "duration_max_sec": round(df["duration_sec"].max(), 3),
        "duration_mean_sec": round(df["duration_sec"].mean(), 3),
        "num_folds": "N/A (stratified split planned)",
        "test_fold": "N/A",
        "train_folds": "80/20 stratified split (Step 2)",
        "raw_path": str(raw_dir),
        "metadata_file": str(meta_path),
        "missing_files": int(missing),
        "sample_rate_hz": int(df["sampling_rate"].mode().iloc[0]),
    }

    plt.figure(figsize=(10, 5))
    order = df["label"].value_counts().index
    sns.countplot(data=df, x="label", order=order, palette="magma")
    plt.title("ESC-50 Animals — Clips per Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "esc50_class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["duration_sec"], bins=20, kde=True, color="darkorange")
    plt.title("ESC-50 Animals — Clip Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "esc50_duration_distribution.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(5, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, cls in enumerate(classes):
        subset = df[df["label"] == cls]
        if subset.empty:
            axes[i].set_title(f"{cls} (no samples)")
            continue
        sample = subset.iloc[0]
        y, sr = librosa.load(sample["audio_path"], sr=None, mono=True)
        axes[i].plot(np.linspace(0, len(y) / sr, len(y)), y, color="darkred", linewidth=0.6)
        axes[i].set_title(cls)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude")
    plt.suptitle("ESC-50 Animals — Example Waveforms per Class", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "esc50_sample_waveforms.png", dpi=150, bbox_inches="tight")
    plt.close()

    df.to_csv(ESC50_SPLITS / "metadata_enriched.csv", index=False)
    return pd.DataFrame([inventory])


def write_eda_summary(inventory_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(REPORTS_DIR / "dataset_inventory.csv", index=False)

    summary = {
        "step": 1,
        "title": "Dataset Acquisition & EDA",
        "urbansound8k": inventory_df[inventory_df["dataset"].str.contains("Urban")].iloc[0].to_dict(),
        "esc50_animals": inventory_df[inventory_df["dataset"].str.contains("ESC")].iloc[0].to_dict(),
        "figures": [
            "urban_class_distribution.png",
            "urban_duration_distribution.png",
            "urban_fold_heatmap.png",
            "urban_sample_waveforms.png",
            "urban_sample_rate.png",
            "esc50_class_distribution.png",
            "esc50_duration_distribution.png",
            "esc50_sample_waveforms.png",
        ],
    }
    with (REPORTS_DIR / "step1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    set_seed(42)
    cfg = load_config()
    sns.set_theme(style="whitegrid")

    # Explore both datasets: class counts, duration stats, example waveforms
    urban_inventory = run_urbansound8k_eda(cfg)
    esc_inventory = run_esc50_eda(cfg)
    inventory_df = pd.concat([urban_inventory, esc_inventory], ignore_index=True)
    write_eda_summary(inventory_df)

    print("Step 1 EDA complete.")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Inventory: {REPORTS_DIR / 'dataset_inventory.csv'}")
    print(f"  Urban mapping: {URBAN_SPLITS / 'class_mapping.json'}")
    print(f"  ESC-50 mapping: {ESC50_SPLITS / 'class_mapping.json'}")


if __name__ == "__main__":
    main()
