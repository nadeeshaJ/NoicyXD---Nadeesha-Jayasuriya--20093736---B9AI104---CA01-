"""Aggregate telemetry metrics from Supabase prediction logs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.predictions_repo import fetch_recent_predictions

URBAN_MONITOR_CLASSES = {
    "siren",
    "car_horn",
    "drilling",
    "jackhammer",
    "gun_shot",
    "engine_idling",
    "street_music",
}

ANIMAL_MONITOR_CLASSES = {
    "dog",
    "cat",
    "cow",
    "sheep",
    "pig",
    "rooster",
    "hen",
    "crow",
    "frog",
    "insects",
}


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_dashboard_metrics(session_id: str, limit: int = 100) -> dict[str, Any]:
    rows = fetch_recent_predictions(session_id, limit=limit)
    if not rows:
        return {
            "total_predictions": 0,
            "avg_latency_ms": None,
            "predictions_last_hour": 0,
            "low_confidence_count": 0,
            "unknown_count": 0,
            "latency_trend": [],
            "class_distribution": [],
            "mode_distribution": [],
            "source_distribution": [],
            "model_distribution": [],
            "urban_event_summary": [],
            "animal_event_summary": [],
        }

    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)

    latencies: list[float] = []
    latency_trend: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    mode_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    urban_counter: Counter[str] = Counter()
    animal_counter: Counter[str] = Counter()
    recent_hour = 0
    low_confidence_count = 0
    unknown_count = 0

    for row in reversed(rows):
        created_at = row.get("created_at")
        if created_at:
            ts = _parse_ts(created_at)
            if ts >= one_hour_ago:
                recent_hour += 1

        if row.get("inference_ms") is not None:
            latency = float(row["inference_ms"])
            latencies.append(latency)
            latency_trend.append(
                {
                    "timestamp": created_at,
                    "latency_ms": round(latency, 2),
                    "label": row.get("top_label"),
                }
            )

        label = str(row.get("top_label") or "")
        if label:
            class_counter[label] += 1
            domain = row.get("routed_domain") or row.get("processing_mode")
            if label in URBAN_MONITOR_CLASSES or domain == "urban":
                if label in URBAN_MONITOR_CLASSES:
                    urban_counter[label] += 1
            if label in ANIMAL_MONITOR_CLASSES or domain == "animal":
                if label in ANIMAL_MONITOR_CLASSES:
                    animal_counter[label] += 1

        if row.get("processing_mode"):
            mode_counter[str(row["processing_mode"])] += 1
        if row.get("input_source"):
            source_counter[str(row["input_source"])] += 1
        if row.get("model_key"):
            model_counter[str(row["model_key"])] += 1

        reliability = row.get("reliability_level")
        if reliability == "Low" or float(row.get("top_confidence") or 0) < 0.40:
            low_confidence_count += 1
        if row.get("is_unknown"):
            unknown_count += 1

    def counter_to_list(counter: Counter[str]) -> list[dict[str, Any]]:
        return [{"name": name, "count": count} for name, count in counter.most_common()]

    return {
        "total_predictions": len(rows),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "predictions_last_hour": recent_hour,
        "low_confidence_count": low_confidence_count,
        "unknown_count": unknown_count,
        "latency_trend": latency_trend[-30:],
        "class_distribution": counter_to_list(class_counter),
        "mode_distribution": counter_to_list(mode_counter),
        "source_distribution": counter_to_list(source_counter),
        "model_distribution": counter_to_list(model_counter),
        "urban_event_summary": counter_to_list(urban_counter),
        "animal_event_summary": counter_to_list(animal_counter),
    }
