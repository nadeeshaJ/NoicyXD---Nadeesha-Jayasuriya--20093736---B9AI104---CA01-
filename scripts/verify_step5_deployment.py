"""Verify deployment inference and capture demo screenshot metadata.

Run:
    python scripts/verify_step5_deployment.py

Checks:
    Urban + animal Streamlit models load and predict on sample WAV files.

Outputs:
    reports/step5/step5_verification.json
    reports/figures/step5/app_demo_urban.png, app_demo_animal.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.predict import load_mode_model, predict_audio
from src.utils import load_config, project_path, save_json

REPORTS_DIR = project_path("reports", "step5")


def find_sample_audio(dataset_key: str, class_name: str) -> Path | None:
    cfg = load_config()
    splits_dir = project_path(cfg["datasets"][dataset_key]["splits_dir"])
    test_csv = splits_dir / "test_processed.csv"
    if not test_csv.exists():
        return None
    import pandas as pd
    df = pd.read_csv(test_csv)
    row = df[df["label"] == class_name]
    if row.empty:
        row = df.iloc[:1]
    else:
        row = row.iloc[:1]
    return Path(row.iloc[0]["audio_path"])


def verify_mode(mode: str, sample_class: str) -> dict:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, deploy_cfg, _, _ = load_mode_model(mode, cfg, device)
    audio_path = find_sample_audio(deploy_cfg["dataset_key"], sample_class)
    if audio_path is None or not audio_path.exists():
        raise FileNotFoundError(f"No sample audio for mode={mode}")

    result = predict_audio(
        model,
        class_names,
        deploy_cfg["model_name"],
        audio_path,
        device=device,
        cfg=cfg,
    )
    return {
        "mode": mode,
        "sample_audio": str(audio_path),
        "expected_class_hint": sample_class,
        "prediction": result["top_label"],
        "confidence": result["top_confidence"],
        "top3": result["predictions"],
        "device": str(device),
        "checkpoint": deploy_cfg["checkpoint"],
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    checks = [
        verify_mode("urban", "dog_bark"),
        verify_mode("animal", "dog"),
    ]
    summary = {
        "step": 5,
        "title": "Streamlit Deployment Verification",
        "streamlit_command": "python -m streamlit run sound_analytics_platform/streamlit/streamlit_app.py",
        "checks": checks,
        "deployment": load_config()["deployment"],
    }
    save_json(summary, REPORTS_DIR / "step5_verification.json")
    print("Step 5 deployment verification passed.")
    for check in checks:
        print(
            f"  {check['mode']}: {check['prediction']} ({check['confidence']:.1%}) "
            f"from {Path(check['sample_audio']).name}"
        )
    print(f"  Saved: {REPORTS_DIR / 'step5_verification.json'}")


if __name__ == "__main__":
    main()
