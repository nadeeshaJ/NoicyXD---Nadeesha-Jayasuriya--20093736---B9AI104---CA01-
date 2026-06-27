"""
Dataset split creation for UrbanSound8K and ESC-50 Animals.

CA1 role:
    UrbanSound8K — official fold 10 as test; folds 1–9 split 90/10 train/val.
    ESC-50 Animals — stratified 70/15/15 train/val/test (small dataset).

Outputs:
    data/splits/{dataset}/train.csv, val.csv, test.csv, class_mapping.json

Used by:
    scripts/run_step2_preprocess.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import project_path, save_json


def urban_audio_path(row: pd.Series, raw_dir: Path) -> Path:
    # UrbanSound8K stores wavs in fold1/ … fold10/ subfolders
    return raw_dir / "audio" / f"fold{int(row['fold'])}" / row["slice_file_name"]


def esc50_audio_path(row: pd.Series, raw_dir: Path) -> Path:
    return raw_dir / "audio" / row["filename"]


def create_urbansound8k_splits(cfg: dict, class_to_idx: dict[str, int]) -> dict[str, pd.DataFrame]:
    """Official fold-10 test split with stratified validation from train folds."""
    urban_cfg = cfg["datasets"]["urbansound8k"]
    raw_dir = project_path(urban_cfg["raw_dir"])
    splits_dir = project_path(urban_cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(project_path(urban_cfg["metadata"]))
    df["audio_path"] = df.apply(lambda r: str(urban_audio_path(r, raw_dir)), axis=1)
    df["label"] = df["class"]
    df["class_idx"] = df["class"].map(class_to_idx)  # string label → integer 0–9

    test_fold = int(urban_cfg["test_fold"])
    test_df = df[df["fold"] == test_fold].copy()       # official CA1 test set: fold 10
    train_pool = df[df["fold"] != test_fold].copy()    # folds 1–9 for train + val

    # 90/10 split of folds 1–9; stratify keeps class balance in train and val
    train_df, val_df = train_test_split(
        train_pool,
        test_size=0.1,
        random_state=cfg["training"]["seed"],
        stratify=train_pool["class"],
    )

    for name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items():
        out = splits_dir / f"{name}.csv"
        split_df.to_csv(out, index=False)

    summary = {
        "dataset": "UrbanSound8K",
        "test_fold": test_fold,
        "train_clips": len(train_df),
        "val_clips": len(val_df),
        "test_clips": len(test_df),
        "train_folds": sorted(train_pool["fold"].unique().tolist()),
    }
    save_json(summary, splits_dir / "split_summary.json")
    return {"train": train_df, "val": val_df, "test": test_df}


def create_esc50_splits(cfg: dict, class_to_idx: dict[str, int]) -> dict[str, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test split for ESC-50 Animals."""
    esc_cfg = cfg["datasets"]["esc50_animals"]
    raw_dir = project_path(esc_cfg["raw_dir"])
    splits_dir = project_path(esc_cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(project_path(esc_cfg["metadata"]))
    df["audio_path"] = df.apply(lambda r: str(esc50_audio_path(r, raw_dir)), axis=1)
    df["label"] = df["label"]
    df["class_idx"] = df["label"].map(class_to_idx)

    seed = cfg["training"]["seed"]
    # 70% train, 30% held out for val+test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=seed,
        stratify=df["label"],
    )
    # Split the 30% evenly → 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )

    for name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items():
        out = splits_dir / f"{name}.csv"
        split_df.to_csv(out, index=False)

    summary = {
        "dataset": "ESC-50 Animals",
        "train_clips": len(train_df),
        "val_clips": len(val_df),
        "test_clips": len(test_df),
        "split_ratio": "70/15/15 stratified",
    }
    save_json(summary, splits_dir / "split_summary.json")
    return {"train": train_df, "val": val_df, "test": test_df}
