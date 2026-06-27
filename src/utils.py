"""
Shared utilities: config loading, paths, and reproducibility helpers.

CA1 role:
    All scripts read hyperparameters from config/config.yaml via load_config().
    project_path() resolves paths relative to the repository root.

Used by:
    Every module in src/ and scripts/.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of src/)
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(*parts: str) -> Path:
    # Build absolute path relative to repo root — works from any working directory
    return PROJECT_ROOT.joinpath(*parts)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)  # numpy used in augmentation and splits
    except ImportError:
        pass


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

