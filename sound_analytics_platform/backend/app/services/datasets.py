"""Load dataset summaries and test samples from the parent ML project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.supabase_client import ensure_ml_path, get_ml_project_root


def _basename_from_any_path(path_str: str) -> str:
    """Extract WAV filename from Windows, POSIX, or bare filename strings."""
    normalized = str(path_str).strip().replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


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


def _raw_audio_root(domain: str) -> Path:
    ensure_ml_path()
    from src.utils import load_config, project_path

    cfg = load_config()
    key = "urbansound8k" if domain == "urban" else "esc50_animals"
    return project_path(cfg["datasets"][key]["raw_dir"]) / "audio"


def _lookup_split_row(df: pd.DataFrame, sample_id: str) -> pd.Series | None:
    filename = _basename_from_any_path(sample_id)
    names = df["audio_path"].astype(str).map(_basename_from_any_path)
    match = df[names == filename]
    if match.empty:
        return None
    return match.iloc[0]


def _resolve_raw_audio_path(domain: str, sample_id: str) -> Path:
    """Resolve a dataset WAV on any OS — never trust absolute paths baked into CSVs."""
    filename = _basename_from_any_path(sample_id)
    audio_root = _raw_audio_root(domain)

    if domain == "urban":
        candidates = [audio_root / "fold10" / filename]
        candidates.extend(sorted(audio_root.glob(f"fold*/{filename}")))
    else:
        candidates = [audio_root / filename]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Audio file not found: {filename} (domain={domain}). "
        f"Looked under {audio_root}. Ensure data/raw is mounted on the server."
    )


def _portable_audio_ref(domain: str, filename: str) -> str:
    if domain == "urban":
        return f"data/raw/urbansound8k/audio/fold10/{filename}"
    return f"data/raw/esc50/audio/{filename}"


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
        filename = _basename_from_any_path(row["audio_path"])
        samples.append(
            {
                "sample_id": filename,
                "filename": filename,
                "label": row["label"],
                "class_idx": int(row["class_idx"]),
                "audio_path": _portable_audio_ref(domain, filename),
                "image_path": str(row.get("image_path", "")),
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
        filename = _basename_from_any_path(row["audio_path"])
        results.append(
            {
                "sample_id": filename,
                "filename": filename,
                "label": label,
                "note": notes.get(label, f"{label} from project test split"),
                "audio_path": _portable_audio_ref(domain, filename),
                "image_path": str(row.get("image_path", "")),
                "domain": domain,
                "curated": True,
            }
        )
    return results


def resolve_sample_audio(domain: str, sample_id: str) -> tuple[Path, str, str]:
    filename = _basename_from_any_path(sample_id)
    split_path = _split_csv(domain)
    label: str | None = None

    if split_path.exists():
        df = pd.read_csv(split_path)
        row = _lookup_split_row(df, filename)
        if row is not None:
            label = str(row["label"])

    if label is None:
        raise FileNotFoundError(f"Sample not found: {filename}")

    resolved_path = _resolve_raw_audio_path(domain, filename)
    return resolved_path, label, filename
