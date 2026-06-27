"""Clean UrbanSound8K layout and download ESC-50 Animals from Hugging Face.

Step 0 — Dataset setup

Run:
    python scripts/setup_datasets.py

Outputs:
    data/raw/urbansound8k/  (audio + UrbanSound8K.csv)
    data/raw/esc50/         (200 animal clips + esc50_animals.csv)

Notes:
    UrbanSound8K must be downloaded manually from the official site/Kaggle first.
    ESC-50 Animals subset is pulled from Hugging Face automatically.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URBAN_RAW = PROJECT_ROOT / "data" / "raw" / "urbansound8k"
ESC50_RAW = PROJECT_ROOT / "data" / "raw" / "esc50"
HF_DATASET = "DynamicSuperb/EnvironmentalSoundClassification_ESC50-Animals"


def clean_urbansound8k() -> None:
    nested = URBAN_RAW / "UrbanSound8K"
    if nested.exists():
        for name in ("audio", "metadata"):
            src = nested / name
            dst = URBAN_RAW / name
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
        shutil.rmtree(nested)

    for path in URBAN_RAW.rglob(".DS_Store"):
        path.unlink(missing_ok=True)
    for path in URBAN_RAW.rglob("Thumbs.db"):
        path.unlink(missing_ok=True)

    for fname in ("FREESOUNDCREDITS.txt", "UrbanSound8K_README.txt", ".gitkeep"):
        (URBAN_RAW / fname).unlink(missing_ok=True)

    csv_path = URBAN_RAW / "metadata" / "UrbanSound8K.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing UrbanSound8K metadata: {csv_path}")

    wav_count = len(list((URBAN_RAW / "audio").rglob("*.wav")))
    print(f"UrbanSound8K ready: {wav_count} wav files, metadata at {csv_path}")


def read_audio_entry(audio_entry: dict) -> tuple[object, int]:
    """Read audio from HF dataset entry without torchcodec."""
    if audio_entry.get("bytes"):
        import io

        data, sr = sf.read(io.BytesIO(audio_entry["bytes"]))
        return data, sr

    path = audio_entry.get("path")
    if path:
        data, sr = sf.read(path)
        return data, sr

    raise ValueError("Audio entry has no bytes or path")


def download_esc50_animals() -> None:
    audio_dir = ESC50_RAW / "audio"
    meta_dir = ESC50_RAW / "meta"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    (ESC50_RAW / ".gitkeep").unlink(missing_ok=True)

    print(f"Downloading {HF_DATASET} from Hugging Face...")
    dataset = load_dataset(HF_DATASET, split="test")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    rows: list[dict] = []
    for idx, item in enumerate(dataset):
        filename = item["file"]
        label = item["label"]
        array, sr = read_audio_entry(item["audio"])

        out_path = audio_dir / filename
        sf.write(out_path, array, sr)

        rows.append(
            {
                "filename": filename,
                "label": label,
                "instruction": item.get("instruction", ""),
                "sampling_rate": sr,
                "duration_sec": round(len(array) / sr, 3),
            }
        )
        if (idx + 1) % 50 == 0:
            print(f"  saved {idx + 1}/{len(dataset)} clips")

    csv_path = meta_dir / "esc50_animals.csv"
    fieldnames = ["filename", "label", "instruction", "sampling_rate", "duration_sec"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labels = sorted({r["label"] for r in rows})
    print(f"ESC-50 Animals ready: {len(rows)} clips, {len(labels)} classes")
    print(f"  audio: {audio_dir}")
    print(f"  metadata: {csv_path}")


if __name__ == "__main__":
    clean_urbansound8k()
    download_esc50_animals()
