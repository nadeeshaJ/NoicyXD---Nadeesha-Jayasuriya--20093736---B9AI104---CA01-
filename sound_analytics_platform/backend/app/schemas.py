from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    label: str = Field(description="Sound class key, e.g. `siren` or `dog_bark`.")
    confidence: float = Field(description="Softmax probability in [0, 1].", ge=0.0, le=1.0)


class RouterProbe(BaseModel):
    top_label: str = Field(description="Top class from the expert probe.")
    top_confidence: float = Field(description="Probe softmax confidence.", ge=0.0, le=1.0)


class RouterExpertMetrics(BaseModel):
    top_label: str
    top_confidence: float = Field(ge=0.0, le=1.0)
    entropy_normalized: float = Field(description="Normalized Shannon entropy in [0, 1].", ge=0.0, le=1.0)
    uncertainty_level: str = Field(description="Low, Medium, or High uncertainty band.")
    strength_score: float = Field(description="Domain-adjusted routing strength score.")


class RouterInfo(BaseModel):
    domain: str = Field(description="Selected domain after routing: `urban` or `animal`.")
    reason: str = Field(description="Human-readable routing explanation.")
    primary_reason: str | None = None
    hint_note: str | None = None
    urban_score: float = Field(description="Calibrated urban expert strength.")
    animal_score: float = Field(description="Calibrated animal expert strength.")
    confidence_gap: float = Field(description="Absolute gap between urban and animal scores.")
    selected_uncertainty: str
    urban_metrics: RouterExpertMetrics
    animal_metrics: RouterExpertMetrics
    urban_probe: RouterProbe
    animal_probe: RouterProbe


class AssessmentInfo(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    entropy_normalized: float = Field(ge=0.0, le=1.0)
    uncertainty_level: str
    reliability_level: str = Field(description="High, Medium, or Low reliability rating.")
    reliability_message: str
    is_unknown: bool = Field(description="True when confidence is below the unknown threshold (40%).")
    display_label: str
    display_name: str
    best_guess_label: str
    calibration_note: str


class PredictResponse(BaseModel):
    processing_mode: str = Field(description="Mode requested by the client: urban, animal, or auto.")
    effective_mode: str = Field(description="Domain used after routing: urban or animal.")
    model_key: str = Field(description="CNN architecture that produced the prediction.")
    top_label: str
    top_confidence: float = Field(ge=0.0, le=1.0)
    predictions: list[PredictionItem] = Field(description="Top-k class predictions (default k=3).")
    probabilities: dict[str, float] = Field(description="Full softmax distribution over all classes.")
    inference_ms: float | None = Field(default=None, description="End-to-end inference latency in milliseconds.")
    device_used: str | None = Field(default=None, description="Compute device, e.g. `cuda` or `cpu`.")
    assessment: AssessmentInfo
    waveform_png: str = Field(description="Base64-encoded waveform PNG.")
    mel_png: str = Field(description="Base64-encoded normalized Mel-spectrogram PNG.")
    rgb_png: str = Field(description="Base64-encoded 224×224 RGB model input PNG.")
    gradcam_png: str | None = Field(default=None, description="Base64-encoded Grad-CAM overlay PNG.")
    gradcam_summary: dict[str, Any] | None = None
    router: RouterInfo | None = Field(default=None, description="Present when `processing_mode=auto`.")
    benchmark: dict[str, Any] | None = Field(default=None, description="Static benchmark row for the model used.")
    saved_prediction_id: str | None = Field(default=None, description="Supabase row ID if saved successfully.")
    ground_truth_label: str | None = Field(default=None, description="Dataset label when predicting from a sample.")
    sample_id: str | None = Field(default=None, description="Dataset WAV filename when applicable.")
    input_source: str | None = Field(default=None, description="upload, microphone, or dataset.")


class CheckpointStatus(BaseModel):
    path: str
    model_key: str
    status: str = Field(description="ok, missing, invalid, or suspect.")
    size_mb: float | None = None
    message: str


class HealthResponse(BaseModel):
    status: str = Field(description="`ok` when primary MobileNetV2 checkpoint is valid, else `degraded`.")
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
    is_deployed: bool = Field(default=False, description="True for the production MobileNetV2 model.")
    notes: str | None = None


class ValidationCheck(BaseModel):
    name: str
    target: str
    actual: str
    passed: bool


class AudioPreviewResponse(BaseModel):
    valid: bool = Field(description="True when all preprocessing validation checks pass.")
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
