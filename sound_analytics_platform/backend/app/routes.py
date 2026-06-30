from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.config import settings
from app.schemas import (
    AnalyticsDashboardResponse,
    AudioPreviewResponse,
    CheckpointStatus,
    HealthResponse,
    ModelBenchmark,
    ModelCompareResponse,
    PredictResponse,
)
from app.services.analytics import build_dashboard_metrics
from app.services.datasets import (
    get_curated_samples,
    get_dataset_overview,
    list_test_samples,
    resolve_sample_audio,
)
from app.services.export_report import build_prediction_report_zip
from app.services.inference import inference_service
from app.services.predictions_repo import (
    fetch_model_benchmarks,
    fetch_recent_predictions,
    fetch_sound_classes,
    save_prediction_record,
)
from app.services.supabase_client import get_ml_project_root

router = APIRouter()

ProcessingMode = Annotated[
    str,
    Form(
        description="Processing pipeline: `urban` (UrbanSound8K), `animal` (ESC-50), or `auto` (Smart Auto-Router).",
        examples=["urban"],
    ),
]
ModelName = Annotated[
    str,
    Form(
        description="CNN architecture key: `mobilenetv2` (deployed), `resnet50`, or `custom_cnn`.",
        examples=["mobilenetv2"],
    ),
]
DatasetDomain = Annotated[
    str,
    Form(description="Dataset domain: `urban` or `animal`.", examples=["urban"]),
]
SampleId = Annotated[
    str,
    Form(description="WAV filename from the test split, e.g. `115241-9-0-9.wav`.", examples=["115241-9-0-9.wav"]),
]
GradCamFlag = Annotated[
    bool,
    Form(description="Generate Grad-CAM explainability heatmap overlay.", examples=[True]),
]
SaveToDb = Annotated[
    bool,
    Form(description="Persist prediction to Supabase when configured.", examples=[True]),
]
InputSource = Annotated[
    str,
    Form(description="Origin of audio: `upload` or `microphone`.", examples=["upload"]),
]
SessionHeader = Annotated[
    str,
    Header(
        alias="X-Session-Id",
        description="Browser session ID for grouping prediction history and analytics.",
        examples=["anonymous"],
    ),
]


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service health and checkpoint status",
    response_description="API status, compute device, and trained weight verification.",
)
def health_check() -> HealthResponse:
    """Check API readiness, GPU/CPU device, Supabase configuration, and deployment checkpoints."""
    from src.checkpoint_utils import verify_all_deployment_checkpoints

    report = verify_all_deployment_checkpoints(run_probe=False)
    checkpoint_rows = [
        CheckpointStatus(
            path=row["path"],
            model_key=row["model_key"],
            status=row["status"],
            size_mb=row.get("size_mb"),
            message=row["message"],
        )
        for row in report["checkpoints"]
    ]
    return HealthResponse(
        status="ok" if report["deploy_ready"] else "degraded",
        device=str(inference_service.device),
        supabase_configured=bool(settings.supabase_anon_key or settings.supabase_service_role_key),
        ml_project_root=str(get_ml_project_root()),
        checkpoints_ready=report["deploy_ready"],
        checkpoint_summary=report["summary"],
        checkpoints=checkpoint_rows,
    )


@router.get(
    "/models",
    response_model=list[ModelBenchmark],
    tags=["System"],
    summary="Model benchmark table",
    response_description="Accuracy, F1, latency, and file size for each trained CNN.",
)
def list_models() -> list[ModelBenchmark]:
    """Return benchmark metrics for Custom CNN, ResNet50, and MobileNetV2 (deployed)."""
    try:
        rows = fetch_model_benchmarks()
        if rows:
            return [ModelBenchmark(**row) for row in rows]
    except Exception:
        pass

    benchmarks = inference_service.load_benchmark_table(inference_service.cfg)
    labels = inference_service.cfg["app"]["model_labels"]
    return [
        ModelBenchmark(
            model_key=key,
            display_name=labels.get(key, key),
            total_parameters=row["total_parameters"],
            model_file_size_mb=row["model_file_size_mb"],
            inference_ms_mean=row["inference_ms_mean"],
            test_accuracy=row.get("test_accuracy"),
            test_macro_f1=row.get("test_macro_f1"),
            is_deployed=key == "mobilenetv2",
            notes=None,
        )
        for key, row in benchmarks.items()
    ]


@router.get(
    "/classes/{domain}",
    tags=["System"],
    summary="List sound classes for a domain",
    response_description="Class keys and display names for urban or animal datasets.",
)
def list_classes(domain: str):
    """
  Return the label vocabulary for a domain.

  - **urban** — 10 UrbanSound8K classes (e.g. `siren`, `dog_bark`)
  - **animal** — 10 ESC-50 animal classes (e.g. `dog`, `cow`)
    """
    if domain not in {"urban", "animal"}:
        raise HTTPException(status_code=400, detail="Domain must be urban or animal.")
    try:
        rows = fetch_sound_classes(domain)
        if rows:
            return rows
    except Exception:
        pass
    dataset_key = "urbansound8k" if domain == "urban" else "esc50_animals"
    classes = inference_service.cfg["datasets"][dataset_key]["classes"]
    return [{"domain": domain, "class_key": name, "display_name": name.replace("_", " ").title()} for name in classes]


@router.get(
    "/datasets",
    tags=["Datasets"],
    summary="Dataset overview",
    response_description="Summary cards for UrbanSound8K and ESC-50 Animals test splits.",
)
def dataset_overview():
    """Return clip counts, class counts, and paths for both project datasets."""
    try:
        return get_dataset_overview()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/datasets/{domain}/samples",
    tags=["Datasets"],
    summary="Browse test-split samples",
    response_description="Paginated list of test WAV files with ground-truth labels.",
)
def dataset_samples(domain: str, label: str | None = None, limit: int = 50):
    """
  List WAV samples from the official test split.

  - **domain** — `urban` or `animal`
  - **label** — optional class filter (e.g. `siren`)
  - **limit** — max rows returned (capped at 100)
    """
    if domain not in {"urban", "animal"}:
        raise HTTPException(status_code=400, detail="Domain must be urban or animal.")
    try:
        return list_test_samples(domain, label=label, limit=min(limit, 100))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/datasets/{domain}/curated",
    tags=["Datasets"],
    summary="Curated demo samples",
    response_description="Hand-picked demo clips with descriptive notes for presentations.",
)
def curated_samples(domain: str):
    """Return recommended demo samples (siren, dog bark, cow, etc.) for quick testing."""
    if domain not in {"urban", "animal"}:
        raise HTTPException(status_code=400, detail="Domain must be urban or animal.")
    return get_curated_samples(domain)


@router.get(
    "/datasets/{domain}/samples/{sample_id}/audio",
    tags=["Datasets"],
    summary="Stream sample audio",
    response_description="Raw WAV file bytes for browser playback.",
)
def get_sample_audio_file(domain: str, sample_id: str):
    """Stream a test-split WAV file by filename (e.g. `115241-9-0-9.wav`)."""
    if domain not in {"urban", "animal"}:
        raise HTTPException(status_code=400, detail="Domain must be urban or animal.")
    try:
        path, _, _ = resolve_sample_audio(domain, sample_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found on disk.")
        return FileResponse(path, media_type="audio/wav")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _finalize_prediction(
    result: dict,
    *,
    mode: str,
    model_name: str,
    input_source: str,
    gradcam: bool,
    save_to_db: bool,
    x_session_id: str,
    original_filename: str | None,
    ground_truth_label: str | None = None,
    sample_id: str | None = None,
) -> PredictResponse:
    saved_id = None
    if save_to_db:
        try:
            saved = save_prediction_record(
                session_id=x_session_id,
                processing_mode=mode,
                routed_domain=result.get("effective_mode"),
                model_key=result["model_key"],
                input_source=input_source,
                original_filename=original_filename,
                top_label=result["top_label"],
                top_confidence=float(result["top_confidence"]),
                probabilities=result["probabilities"],
                top_predictions=result["predictions"],
                inference_ms=result.get("inference_ms"),
                assessment=result.get("assessment"),
                router=result.get("router"),
                gradcam_enabled=gradcam,
                device_used=result.get("device_used"),
            )
            saved_id = saved.get("id")
        except Exception:
            saved_id = None

    result["saved_prediction_id"] = saved_id
    result["ground_truth_label"] = ground_truth_label
    result["sample_id"] = sample_id
    result["input_source"] = input_source
    return PredictResponse(**result)


@router.post(
    "/predict/sample",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Classify a dataset sample",
    response_description="Full prediction payload including ground-truth label for auditing.",
)
async def predict_from_sample(
    domain: DatasetDomain,
    sample_id: SampleId,
    mode: ProcessingMode = "urban",
    model_name: ModelName = "mobilenetv2",
    gradcam: GradCamFlag = True,
    save_to_db: SaveToDb = True,
    x_session_id: SessionHeader = "anonymous",
):
    """
  Run inference on a WAV from the project test split.

  Returns `ground_truth_label` and `sample_id` so the UI can show match/mismatch auditing.
    """
    if domain not in {"urban", "animal"}:
        raise HTTPException(status_code=400, detail="Invalid domain.")
    if mode not in {"urban", "animal", "auto"}:
        raise HTTPException(status_code=400, detail="Invalid mode.")

    try:
        audio_path, ground_truth, resolved_id = resolve_sample_audio(domain, sample_id)
        audio_bytes = audio_path.read_bytes()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        result = inference_service.run_prediction(
            audio_bytes=audio_bytes,
            mode=mode,
            model_name=model_name,
            gradcam=gradcam,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return _finalize_prediction(
        result,
        mode=mode,
        model_name=model_name,
        input_source="dataset",
        gradcam=gradcam,
        save_to_db=save_to_db,
        x_session_id=x_session_id,
        original_filename=resolved_id,
        ground_truth_label=ground_truth,
        sample_id=resolved_id,
    )


@router.get(
    "/analytics/dashboard",
    response_model=AnalyticsDashboardResponse,
    tags=["Analytics"],
    summary="Session analytics dashboard",
    response_description="Aggregated telemetry: latency trends, class/mode distributions, event summaries.",
)
def analytics_dashboard(
    x_session_id: SessionHeader = "anonymous",
    limit: int = 100,
):
    """Return monitoring metrics for the current `X-Session-Id` from Supabase prediction logs."""
    try:
        return build_dashboard_metrics(x_session_id, limit=min(limit, 200))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Analytics query failed: {exc}") from exc


@router.post(
    "/audio/preview",
    response_model=AudioPreviewResponse,
    tags=["Inference"],
    summary="Validate and preview uploaded audio",
    response_description="Waveform and Mel-spectrogram PNGs plus preprocessing validation checks.",
)
async def preview_audio(
    file: Annotated[UploadFile, File(description="WAV audio file (mono recommended, ~4 seconds).")],
    input_source: InputSource = "upload",
):
    """
  Preprocess audio without running the CNN.

  Validates sample rate (22050 Hz), mono channel, duration (4.0 s), and Mel normalization.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        return inference_service.preview_audio(
            audio_bytes,
            input_source=input_source,
            filename=file.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview failed: {exc}") from exc


@router.post(
    "/predict/compare",
    response_model=ModelCompareResponse,
    tags=["Comparison"],
    summary="Compare all models on uploaded audio",
    response_description="Side-by-side top label, confidence, and latency for each available model.",
)
async def compare_models(
    file: Annotated[UploadFile, File(description="WAV audio file to classify with every available model.")],
    mode: ProcessingMode = "urban",
):
    """Run the same clip through Custom CNN, ResNet50, and MobileNetV2 (where checkpoints exist)."""
    if mode not in {"urban", "animal", "auto"}:
        raise HTTPException(status_code=400, detail="Invalid mode.")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        return inference_service.compare_models(audio_bytes, mode=mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {exc}") from exc


@router.post(
    "/predict/sample/compare",
    response_model=ModelCompareResponse,
    tags=["Comparison"],
    summary="Compare all models on a dataset sample",
    response_description="Model comparison using a test-split WAV by filename.",
)
async def compare_sample_models(
    domain: DatasetDomain,
    sample_id: SampleId,
    mode: ProcessingMode = "urban",
):
    """Compare all available models on a curated or browsed dataset sample."""
    if domain not in {"urban", "animal"}:
        raise HTTPException(status_code=400, detail="Invalid domain.")
    try:
        audio_path, _, _ = resolve_sample_audio(domain, sample_id)
        return inference_service.compare_models(audio_path.read_bytes(), mode=mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sample comparison failed: {exc}") from exc


@router.get(
    "/predictions",
    tags=["Analytics"],
    summary="Prediction history",
    response_description="Recent predictions for the session, newest first.",
)
def prediction_history(
    x_session_id: SessionHeader,
    limit: int = 20,
):
    """Fetch recent predictions from Supabase for the given `X-Session-Id` header."""
    try:
        return fetch_recent_predictions(x_session_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase query failed: {exc}") from exc


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Classify uploaded audio",
    response_description="Top-3 predictions, reliability assessment, visualizations, and optional Grad-CAM.",
)
async def predict_audio(
    file: Annotated[UploadFile, File(description="WAV audio file to classify.")],
    mode: ProcessingMode = "urban",
    model_name: ModelName = "mobilenetv2",
    input_source: InputSource = "upload",
    gradcam: GradCamFlag = True,
    save_to_db: SaveToDb = True,
    x_session_id: SessionHeader = "anonymous",
):
    """
  Main inference endpoint.

  Pipeline: WAV → Mel-spectrogram → 224×224 RGB → CNN → softmax probabilities.

  When `mode=auto`, both urban and animal MobileNetV2 experts probe the clip and the
  stronger domain is selected. The response includes a `router` object with probe details.

  PNG fields (`waveform_png`, `mel_png`, `rgb_png`, `gradcam_png`) are base64-encoded.
    """
    if mode not in {"urban", "animal", "auto"}:
        raise HTTPException(status_code=400, detail="Invalid mode.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        result = inference_service.run_prediction(
            audio_bytes=audio_bytes,
            mode=mode,
            model_name=model_name,
            gradcam=gradcam,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    saved_id = None
    if save_to_db:
        try:
            saved = save_prediction_record(
                session_id=x_session_id,
                processing_mode=mode,
                routed_domain=result.get("effective_mode"),
                model_key=result["model_key"],
                input_source=input_source,
                original_filename=file.filename,
                top_label=result["top_label"],
                top_confidence=float(result["top_confidence"]),
                probabilities=result["probabilities"],
                top_predictions=result["predictions"],
                inference_ms=result.get("inference_ms"),
                assessment=result.get("assessment"),
                router=result.get("router"),
                gradcam_enabled=gradcam,
                device_used=result.get("device_used"),
            )
            saved_id = saved.get("id")
        except Exception:
            saved_id = None

    result["saved_prediction_id"] = saved_id
    result["input_source"] = input_source
    return PredictResponse(**result)


@router.post(
    "/reports/export",
    tags=["Reports"],
    summary="Export analysis ZIP report",
    response_description="ZIP archive with JSON/CSV summary and PNG visualizations.",
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": "ZIP file: report_summary.json, report_summary.csv, waveform.png, mel_spectrogram.png, model_input_rgb.png, gradcam_overlay.png",
        }
    },
)
async def export_prediction_report(payload: PredictResponse):
    """
  Build a downloadable ZIP from a `/api/predict` or `/api/predict/sample` response body.

  Pass the full JSON response as the request body. No file upload required.
    """
    try:
        zip_bytes = build_prediction_report_zip(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report export failed: {exc}") from exc
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="sound_analysis_report.zip"'},
    )
