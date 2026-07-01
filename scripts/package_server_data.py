"""Package raw WAV files for uploading to the production server.

Git does not store data/raw/audio. After deploy, copy this zip to the server
and extract into the repo's data/raw/ folder.

Run:
    python scripts/package_server_data.py

Output:
    deploy/server_data_raw.zip
    deploy/SERVER_DATA_README.txt
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = PROJECT_ROOT / "deploy"
ZIP_PATH = DEPLOY_DIR / "server_data_raw.zip"

# Minimum for Datasets tab + demo (fold-10 urban test + full ESC-50 animals)
INCLUDE = [
    ("data/raw/urbansound8k/audio/fold10", "data/raw/urbansound8k/audio/fold10"),
    ("data/raw/esc50/audio", "data/raw/esc50/audio"),
]


def count_wavs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.wav"))


def main() -> None:
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    missing: list[str] = []
    staged = 0

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_rel, arc_prefix in INCLUDE:
            src = PROJECT_ROOT / src_rel
            if not src.exists():
                missing.append(src_rel)
                continue
            for wav in sorted(src.rglob("*.wav")):
                rel = wav.relative_to(src)
                arcname = f"{arc_prefix}/{rel.as_posix()}"
                zf.write(wav, arcname)
                staged += 1

    readme = DEPLOY_DIR / "SERVER_DATA_README.txt"
    readme.write_text(
        "\n".join(
            [
                "Server raw audio install — noicyXD",
                "================================",
                "",
                f"Package: {ZIP_PATH.name} ({staged} WAV files)",
                "",
                "WHY: data/raw is not in git. Without these files the Datasets tab",
                "and Play Sound return: Audio file not found ... under /app/data/raw/...",
                "",
                "ON THE SERVER (repo root, same folder as docker-compose.yml):",
                "",
                "  1. Upload server_data_raw.zip to the server",
                "  2. unzip -o deploy/server_data_raw.zip",
                "     (creates data/raw/urbansound8k/audio/fold10/ and data/raw/esc50/audio/)",
                "  3. Verify:",
                "       ls data/raw/urbansound8k/audio/fold10/159742-8-0-0.wav",
                "       ls data/raw/esc50/audio/*.wav | head",
                "  4. Restart:",
                "       docker compose up -d",
                "",
                "Docker mounts ./data -> /app/data (see docker-compose.yml).",
                "",
                "Minimum contents:",
                "  - urbansound8k/audio/fold10/*.wav  (837 files — test set)",
                "  - esc50/audio/*.wav                (200 files — animal mode)",
                "",
                "Also required separately (not in this zip):",
                "  - experiments/**/best_model.pt     (run setup_checkpoints.py on server)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Created: {ZIP_PATH}")
    print(f"  WAV files: {staged}")
    print(f"  Urban fold10: {count_wavs(PROJECT_ROOT / 'data/raw/urbansound8k/audio/fold10')}")
    print(f"  ESC-50:       {count_wavs(PROJECT_ROOT / 'data/raw/esc50/audio')}")
    print(f"  Guide:        {readme}")
    if missing:
        print("\nMISSING locally (copy from Raw_Data first):")
        for m in missing:
            print(f"  - {m}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
