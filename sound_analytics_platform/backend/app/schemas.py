from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    label: str
    confidence: float


class RouterProbe(BaseModel):
    top_label: str
    top_confidence: float


class RouterExpertMetrics(BaseModel):
    top_label: str
    top_confidence: float
    entropy_normalized: float
    uncertainty_level: str
    strength_score: float


class RouterInfo(BaseModel):
    domain: str
    reason: str
    primary_reason: str | None = None
    hint_note: str | None = None
    urban_score: float
    animal_score: float
    confidence_gap: float
    selected_uncertainty: str
    urban_metrics: RouterExpertMetrics
    animal_metrics: RouterExpertMetrics
    urban_probe: RouterProbe
    animal_probe: RouterProbe


class AssessmentInfo(BaseModel):
    confidence: float
    entropy_normalized: float
    uncertainty_level: str
    reliability_level: str
    reliability_message: str
    is_unknown: bool
    display_label: str
    display_name: str
    best_guess_label: str
    calibration_note: str


class PredictResponse(BaseModel):
    processing_mode: str
    effective_mode: str
    model_key: str
    top_label: str
    top_confidence: float
    predictions: list[PredictionItem]
    probabilities: dict[str, float]
    inference_ms: float | None = None
    device_used: str | None = None
    assessment: AssessmentInfo
    waveform_png: str
    mel_png: str
    rgb_png: str
    gradcam_png: str | None = None
    gradcam_summary: dict[str, Any] | None = None
    router: RouterInfo | None = None
    benchmark: dict[str, Any] | None = None
    saved_prediction_id: str | None = None
    ground_truth_label: str | None = None
    sample_id: str | None = None
    input_source: str | None = None


class CheckpointStatus(BaseModel):
    path: str
    model_key: str
    status: str
    size_mb: float | None = None
    message: str


class HealthResponse(BaseModel):
    status: str
    device: str
    supabase_configured: bool
    ml_project_root: str
    checkpoints_ready: bool = False
    checkpoint_summary: str = ""
    checkpoints: list[CheckpointStatus] = Field(default_factory=list)


class ModelBenchmark(BaseModel):
    model_key: str
    display_name: str
    total_parameters: int
    model_file_size_mb: float
    inference_ms_mean: float
    test_accuracy: float | None = None
    test_macro_f1: float | None = None
    is_deployed: bool = False
    notes: str | None = None


class ValidationCheck(BaseModel):
    name: str
    target: str
    actual: str
    passed: bool


class AudioPreviewResponse(BaseModel):
    valid: bool
    filename: str | None = None
    input_source: str
    original_duration_sec: float
    processed_duration_sec: float
    sample_rate: int
    channels: str
    waveform_png: str
    mel_png: str
    validation_checks: list[ValidationCheck]


class ModelCompareItem(BaseModel):
    model_key: str
    display_name: str
    top_label: str
    top_confidence: float
    inference_ms: float | None
    checkpoint_size_mb: float | None = None
    benchmark_latency_ms: float | None = None


class ModelCompareResponse(BaseModel):
    effective_mode: str
    comparisons: list[ModelCompareItem]


class AnalyticsDashboardResponse(BaseModel):
    total_predictions: int
    avg_latency_ms: float | None
    predictions_last_hour: int
    low_confidence_count: int = 0
    unknown_count: int = 0
    latency_trend: list[dict[str, Any]]
    class_distribution: list[dict[str, Any]]
    mode_distribution: list[dict[str, Any]]
    source_distribution: list[dict[str, Any]]
    model_distribution: list[dict[str, Any]]
    urban_event_summary: list[dict[str, Any]] = Field(default_factory=list)
    animal_event_summary: list[dict[str, Any]] = Field(default_factory=list)
