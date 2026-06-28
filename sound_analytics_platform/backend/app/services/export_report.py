"""Build downloadable ZIP reports from prediction payloads."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Any


def _write_png(zip_file: zipfile.ZipFile, name: str, base64_png: str | None) -> None:
    if not base64_png:
        return
    zip_file.writestr(name, base64.b64decode(base64_png))


def build_prediction_report_zip(payload: dict[str, Any]) -> bytes:
    """Return ZIP bytes containing JSON summary and PNG artefacts."""
    buffer = io.BytesIO()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    assessment = payload.get("assessment") or {}
    router = payload.get("router") or {}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processing_mode": payload.get("processing_mode"),
        "effective_mode": payload.get("effective_mode"),
        "model_key": payload.get("model_key"),
        "input_source": payload.get("input_source"),
        "top_label": payload.get("top_label"),
        "top_confidence": payload.get("top_confidence"),
        "display_label": assessment.get("display_name") or payload.get("top_label"),
        "reliability_level": assessment.get("reliability_level"),
        "reliability_message": assessment.get("reliability_message"),
        "is_unknown": assessment.get("is_unknown", False),
        "entropy_normalized": assessment.get("entropy_normalized"),
        "inference_ms": payload.get("inference_ms"),
        "device_used": payload.get("device_used"),
        "predictions_top3": payload.get("predictions"),
        "probabilities": payload.get("probabilities"),
        "router": router,
        "gradcam_summary": payload.get("gradcam_summary"),
        "saved_prediction_id": payload.get("saved_prediction_id"),
    }

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report_summary.json", json.dumps(summary, indent=2))
        zf.writestr(
            "report_summary.csv",
            "\n".join(
                [
                    "field,value",
                    f"display_label,{summary['display_label']}",
                    f"top_confidence,{summary['top_confidence']}",
                    f"reliability_level,{summary.get('reliability_level') or ''}",
                    f"is_unknown,{summary.get('is_unknown')}",
                    f"model_key,{summary.get('model_key') or ''}",
                    f"effective_mode,{summary.get('effective_mode') or ''}",
                    f"inference_ms,{summary.get('inference_ms') or ''}",
                ]
            ),
        )
        _write_png(zf, "waveform.png", payload.get("waveform_png"))
        _write_png(zf, "mel_spectrogram.png", payload.get("mel_png"))
        _write_png(zf, "model_input_rgb.png", payload.get("rgb_png"))
        _write_png(zf, "gradcam_overlay.png", payload.get("gradcam_png"))

    buffer.seek(0)
    return buffer.getvalue()
