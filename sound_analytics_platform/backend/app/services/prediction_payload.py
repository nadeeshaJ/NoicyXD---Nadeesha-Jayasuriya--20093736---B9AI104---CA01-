from __future__ import annotations

from typing import Any


def build_save_row(
    *,
    session_id: str,
    processing_mode: str,
    model_key: str,
    input_source: str,
    top_label: str,
    top_confidence: float,
    probabilities: dict[str, float],
    top_predictions: list[dict[str, Any]],
    inference_ms: float | None,
    assessment: dict[str, Any] | None = None,
    router: dict[str, Any] | None = None,
    routed_domain: str | None = None,
    gradcam_enabled: bool = False,
    original_filename: str | None = None,
    device_used: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    assessment = assessment or {}
    row = {
        "session_id": session_id,
        "user_id": user_id,
        "processing_mode": processing_mode,
        "routed_domain": routed_domain,
        "model_key": model_key,
        "input_source": input_source,
        "original_filename": original_filename,
        "top_label": top_label,
        "top_confidence": top_confidence,
        "probabilities": probabilities,
        "top_predictions": top_predictions,
        "inference_ms": inference_ms,
        "router_reason": (router or {}).get("primary_reason") or (router or {}).get("reason"),
        "gradcam_enabled": gradcam_enabled,
        "device_used": device_used,
        "reliability_level": assessment.get("reliability_level"),
        "is_unknown": bool(assessment.get("is_unknown", False)),
        "display_label": assessment.get("display_name") or top_label,
        "entropy_normalized": assessment.get("entropy_normalized"),
        "router_metrics": router,
    }
    return row
