"""Rewrite split CSV audio/image paths as portable project-relative POSIX paths.

Run after clone or before deploy:
    python scripts/normalize_split_paths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, project_path


def basename_from_any(path_str: str) -> str:
    return str(path_str).strip().replace("\\", "/").rsplit("/", 1)[-1]


def normalize_urbansound8k(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "slice_file_name" in out.columns:
        filenames = out["slice_file_name"].astype(str)
    else:
        filenames = out["audio_path"].astype(str).map(basename_from_any)
    if "fold" in out.columns:
        folds = out["fold"].astype(int)
    else:
        folds = pd.Series([10] * len(out), index=out.index)
    out["audio_path"] = [
        f"data/raw/urbansound8k/audio/fold{int(fold)}/{fname}"
        for fold, fname in zip(folds, filenames, strict=True)
    ]
    if "image_path" in out.columns:
        out["image_path"] = [
            f"data/processed/urbansound8k/images/{Path(fname).stem}.png"
            for fname in filenames
        ]
    return out


def normalize_esc50(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "filename" in out.columns:
        filenames = out["filename"].astype(str)
    else:
        filenames = out["audio_path"].astype(str).map(basename_from_any)
    out["audio_path"] = [f"data/raw/esc50/audio/{fname}" for fname in filenames]
    if "image_path" in out.columns:
        out["image_path"] = [
            f"data/processed/esc50_animals/images/{Path(fname).stem}.png"
            for fname in filenames
        ]
    return out


def normalize_dataset(key: str, normalizer) -> int:
    cfg = load_config()
    splits_dir = project_path(cfg["datasets"][key]["splits_dir"])
    count = 0
    for csv_path in sorted(splits_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "audio_path" not in df.columns:
            continue
        normalized = normalizer(df)
        normalized.to_csv(csv_path, index=False)
        count += 1
        print(f"  updated {csv_path.relative_to(PROJECT_ROOT)}")
    return count


def main() -> None:
    print("Normalizing split CSV paths to portable data/... references")
    n = 0
    n += normalize_dataset("urbansound8k", normalize_urbansound8k)
    n += normalize_dataset("esc50_animals", normalize_esc50)
    print(f"Done — {n} files updated.")


if __name__ == "__main__":
    main()
