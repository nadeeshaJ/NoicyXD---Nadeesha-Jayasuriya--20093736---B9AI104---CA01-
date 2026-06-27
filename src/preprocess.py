"""
Batch preprocessing: audio → Mel-spectrogram RGB PNG images.

CA1 role:
    Reads train/val/test split CSVs and writes one PNG per clip under
    data/processed/{dataset}/images/. Skips files that already exist unless
    overwrite=True.

Outputs:
    *_processed.csv files listing image_path, label, class_idx, status.

Used by:
    scripts/run_step2_preprocess.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio_utils import load_audio
from src.spectrogram import save_rgb_image, waveform_to_rgb_mel_image
from src.utils import load_config, project_path


def image_output_path(processed_dir: Path, row: pd.Series, dataset_key: str) -> Path:
    # UrbanSound8K uses slice_file_name; ESC-50 uses filename — different CSV columns
    if dataset_key == "urbansound8k":
        stem = Path(row["slice_file_name"]).stem
    else:
        stem = Path(row["filename"]).stem
    return processed_dir / "images" / f"{stem}.png"


def preprocess_row(
    row: pd.Series,
    cfg: dict,
    dataset_key: str,
    processed_dir: Path,
    overwrite: bool = False,
) -> dict:
    urban_cfg = cfg["datasets"]["urbansound8k"]
    esc_cfg = cfg["datasets"]["esc50_animals"]
    audio_cfg = cfg["audio"]
    spec_cfg = cfg["spectrogram"]
    image_cfg = cfg["image"]

    out_path = image_output_path(processed_dir, row, dataset_key)
    # Skip re-processing if PNG already exists (saves time on re-runs)
    if out_path.exists() and not overwrite:
        return {
            "image_path": str(out_path),
            "audio_path": row["audio_path"],
            "label": row["label"],
            "class_idx": int(row["class_idx"]),
            "status": "skipped",
        }

    waveform, sr = load_audio(
        row["audio_path"],
        sample_rate=audio_cfg["sample_rate"],
        duration_sec=audio_cfg["duration_sec"],
        mono=audio_cfg.get("mono", True),
    )
    # Full pipeline: waveform → Mel-spec → normalise → 224×224 RGB
    image = waveform_to_rgb_mel_image(waveform, sr, spec_cfg, image_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb_image(image, out_path)

    return {
        "image_path": str(out_path),
        "audio_path": row["audio_path"],
        "label": row["label"],
        "class_idx": int(row["class_idx"]),
        "status": "processed",
    }


def preprocess_split(
    split_name: str,
    split_df: pd.DataFrame,
    cfg: dict,
    dataset_key: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    dataset_cfg = cfg["datasets"][dataset_key]
    processed_dir = project_path(dataset_cfg["processed_dir"])

    records = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"{dataset_key}:{split_name}"):
        try:
            records.append(preprocess_row(row, cfg, dataset_key, processed_dir, overwrite))
        except Exception as exc:
            # Log failed clips instead of stopping the whole batch
            records.append(
                {
                    "image_path": "",
                    "audio_path": row["audio_path"],
                    "label": row["label"],
                    "class_idx": int(row["class_idx"]),
                    "status": f"error: {exc}",
                }
            )

    out_df = pd.DataFrame(records)
    # e.g. train_processed.csv — lists PNG path + label for PyTorch Dataset
    out_path = project_path(dataset_cfg["splits_dir"]) / f"{split_name}_processed.csv"
    out_df.to_csv(out_path, index=False)
    return out_df


def load_class_mapping(splits_dir: Path) -> dict[str, int]:
    with (splits_dir / "class_mapping.json").open(encoding="utf-8") as f:
        data = json.load(f)
    return data["class_to_idx"]
