from __future__ import annotations

from typing import Any

from postgrest.exceptions import APIError

from app.services.label_matching import labels_match
from app.services.prediction_payload import build_save_row
from app.services.supabase_client import get_supabase_client

EXTENDED_PREDICTION_FIELDS = frozenset(
    {
        "reliability_level",
        "is_unknown",
        "display_label",
        "entropy_normalized",
        "router_metrics",
        "ground_truth_label",
        "sample_id",
        "dataset_domain",
    }
)


def _is_missing_column_error(exc: Exception) -> bool:
    if isinstance(exc, APIError):
        code = getattr(exc, "code", None)
        if code == "PGRST204":
            return True
        message = str(getattr(exc, "message", "") or exc).lower()
        return "could not find" in message and "column" in message
    message = str(exc).lower()
    return "could not find" in message and "column" in message


def _insert_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    client = get_supabase_client()
    response = client.table("predictions").insert(row).execute()
    return response.data[0] if response.data else row


def save_prediction_record(
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
    router_reason: str | None = None,
    gradcam_enabled: bool = False,
    original_filename: str | None = None,
    device_used: str | None = None,
    user_id: str | None = None,
    ground_truth_label: str | None = None,
    sample_id: str | None = None,
    dataset_domain: str | None = None,
) -> dict[str, Any]:
    row = build_save_row(
        session_id=session_id,
        processing_mode=processing_mode,
        model_key=model_key,
        input_source=input_source,
        top_label=top_label,
        top_confidence=top_confidence,
        probabilities=probabilities,
        top_predictions=top_predictions,
        inference_ms=inference_ms,
        assessment=assessment,
        router=router,
        routed_domain=routed_domain,
        gradcam_enabled=gradcam_enabled,
        original_filename=original_filename,
        device_used=device_used,
        user_id=user_id,
        ground_truth_label=ground_truth_label,
        sample_id=sample_id,
        dataset_domain=dataset_domain,
    )
    if router_reason and not row.get("router_reason"):
        row["router_reason"] = router_reason
    try:
        return _insert_prediction_row(row)
    except Exception as exc:
        if not _is_missing_column_error(exc):
            raise
        base_row = {key: value for key, value in row.items() if key not in EXTENDED_PREDICTION_FIELDS}
        return _insert_prediction_row(base_row)


def _lookup_sample_ground_truth(sample_id: str) -> dict[str, str] | None:
    from app.services.datasets import resolve_sample_audio

    for domain in ("urban", "animal"):
        try:
            _, ground_truth, resolved_id = resolve_sample_audio(domain, sample_id)
            return {
                "ground_truth_label": ground_truth,
                "sample_id": resolved_id,
                "dataset_domain": domain,
            }
        except FileNotFoundError:
            continue
    return None


def enrich_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    ground_truth = enriched.get("ground_truth_label")

    if not ground_truth:
        sample_key = enriched.get("sample_id") or enriched.get("original_filename")
        if enriched.get("input_source") == "dataset" and sample_key:
            lookup = _lookup_sample_ground_truth(str(sample_key))
            if lookup:
                enriched.update(lookup)
                ground_truth = enriched.get("ground_truth_label")

    if ground_truth:
        enriched["has_ground_truth"] = True
        enriched["audit_match"] = labels_match(str(enriched.get("top_label", "")), str(ground_truth))
    else:
        enriched["has_ground_truth"] = False
        enriched["audit_match"] = None

    return enriched


def fetch_recent_predictions(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    client = get_supabase_client()
    response = (
        client.table("predictions")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [enrich_prediction_row(row) for row in (response.data or [])]


def fetch_model_benchmarks() -> list[dict[str, Any]]:
    client = get_supabase_client()
    response = (
        client.table("model_benchmarks")
        .select("*")
        .order("test_accuracy", desc=True)
        .execute()
    )
    return response.data or []


def fetch_sound_classes(domain: str | None = None) -> list[dict[str, Any]]:
    client = get_supabase_client()
    query = client.table("sound_classes").select("*").order("sort_order")
    if domain:
        query = query.eq("domain", domain)
    response = query.execute()
    return response.data or []
