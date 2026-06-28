"""Load dataset summaries and test samples from the parent ML project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.supabase_client import ensure_ml_path, get_ml_project_root


def _load_step1_summary() -> dict[str, Any]:
    path = get_ml_project_root() / "reports" / "step1" / "step1_summary.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _split_csv(domain: str) -> Path:
    ensure_ml_path()
    from src.utils import load_config, project_path

    cfg = load_config()
    key = "urbansound8k" if domain == "urban" else "esc50_animals"
    return project_path(cfg["datasets"][key]["splits_dir"]) / "test_processed.csv"


def get_dataset_overview() -> list[dict[str, Any]]:
    ensure_ml_path()
    from src.utils import load_config

    cfg = load_config()
    summary = _load_step1_summary()

    datasets = []
    for domain, key, title in [
        ("urban", "urbansound8k", "UrbanSound8K"),
        ("animal", "esc50_animals", "ESC-50 Animals"),
    ]:
        split_path = _split_csv(domain)
        test_count = len(pd.read_csv(split_path)) if split_path.exists() else 0
        meta = summary.get(key, {})
        datasets.append(
            {
                "domain": domain,
                "dataset_key": key,
                "title": title,
                "total_clips": meta.get("total_clips"),
                "num_classes": meta.get("num_classes", len(cfg["datasets"][key]["classes"])),
                "test_clips": test_count,
                "source": meta.get("source"),
                "classes": cfg["datasets"][key]["classes"],
                "raw_path": str(get_ml_project_root() / cfg["datasets"][key]["raw_dir"]),
                "processed_path": str(get_ml_project_root() / cfg["datasets"][key]["processed_dir"]),
            }
        )
    return datasets


def list_test_samples(
    domain: str,
    label: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if domain not in {"urban", "animal"}:
        raise ValueError("Domain must be urban or animal.")

    split_path = _split_csv(domain)
    if not split_path.exists():
        return []

    df = pd.read_csv(split_path)
    if label:
        df = df[df["label"] == label]
    df = df.head(limit)

    samples = []
    for _, row in df.iterrows():
        audio_path = Path(str(row["audio_path"]))
        samples.append(
            {
                "sample_id": audio_path.name,
                "filename": audio_path.name,
                "label": row["label"],
                "class_idx": int(row["class_idx"]),
                "audio_path": str(audio_path),
                "image_path": str(row["image_path"]),
                "domain": domain,
            }
        )
    return samples


def get_curated_samples(domain: str) -> list[dict[str, Any]]:
    preferred_labels = {
        "urban": ["siren", "dog_bark", "jackhammer", "car_horn", "street_music"],
        "animal": ["dog", "cow", "rooster", "frog", "cat"],
    }
    notes = {
        "siren": "Classic urban siren from UrbanSound8K test fold",
        "dog_bark": "Urban dog bark — overlaps with animal domain",
        "jackhammer": "Construction noise sample",
        "car_horn": "Traffic horn sample",
        "street_music": "Street performance sample",
        "dog": "ESC-50 dog vocalization",
        "cow": "ESC-50 cow moo",
        "rooster": "ESC-50 rooster crow",
        "frog": "ESC-50 frog croak",
        "cat": "ESC-50 cat meow",
    }

    split_path = _split_csv(domain)
    if not split_path.exists():
        return []

    df = pd.read_csv(split_path)
    results = []
    for label in preferred_labels[domain]:
        rows = df[df["label"] == label]
        if rows.empty:
            continue
        row = rows.iloc[0]
        audio_path = Path(str(row["audio_path"]))
        results.append(
            {
                "sample_id": audio_path.name,
                "filename": audio_path.name,
                "label": label,
                "note": notes.get(label, f"{label} from project test split"),
                "audio_path": str(audio_path),
                "image_path": str(row["image_path"]),
                "domain": domain,
                "curated": True,
            }
        )
    return results


def resolve_sample_audio(domain: str, sample_id: str) -> tuple[Path, str, str]:
    split_path = _split_csv(domain)
    if split_path.exists():
        df = pd.read_csv(split_path)
        match = df[df["audio_path"].astype(str).str.endswith(sample_id)]
        if not match.empty:
            row = match.iloc[0]
            return Path(str(row["audio_path"])), str(row["label"]), sample_id

    curated = get_curated_samples(domain)
    for item in curated:
        if item["sample_id"] == sample_id:
            return Path(item["audio_path"]), item["label"], sample_id

    raise FileNotFoundError(f"Sample not found: {sample_id}")
