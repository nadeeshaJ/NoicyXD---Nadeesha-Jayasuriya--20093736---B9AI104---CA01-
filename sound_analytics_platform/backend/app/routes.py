from __future__ import annotations

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.schemas import (
    AnalyticsDashboardResponse,
    AudioPreviewResponse,
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


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(inference_service.device),
        supabase_configured=bool(settings.supabase_anon_key or settings.supabase_service_role_key),
        ml_project_root=str(get_ml_project_root()),
    )


@router.get("/models", response_model=list[ModelBenchmark])
def list_models() -> list[ModelBenchmark]:
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


@router.get("/classes/{domain}")
def list_classes(domain: str):
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


@router.get("/datasets")
def dataset_overview():
    try:
        return get_dataset_overview()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/datasets/{domain}/samples")
def dataset_samples(domain: str, label: str | None = None, limit: int = 50):
    if domain not in {"urban", "animal"}:
        raise HTTPException(status_code=400, detail="Domain must be urban or animal.")
    try:
        return list_test_samples(domain, label=label, limit=min(limit, 100))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/datasets/{domain}/curated")
def curated_samples(domain: str):
    if domain not in {"urban", "animal"}:
        raise HTTPException(status_code=400, detail="Domain must be urban or animal.")
    return get_curated_samples(domain)


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


@router.post("/predict/sample", response_model=PredictResponse)
async def predict_from_sample(
    domain: str = Form(...),
    sample_id: str = Form(...),
    mode: str = Form("urban"),
    model_name: str = Form("mobilenetv2"),
    gradcam: bool = Form(True),
    save_to_db: bool = Form(True),
    x_session_id: str = Header(default="anonymous", alias="X-Session-Id"),
):
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


@router.get("/analytics/dashboard", response_model=AnalyticsDashboardResponse)
def analytics_dashboard(
    x_session_id: str = Header(default="anonymous", alias="X-Session-Id"),
    limit: int = 100,
):
    try:
        return build_dashboard_metrics(x_session_id, limit=min(limit, 200))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Analytics query failed: {exc}") from exc


@router.post("/audio/preview", response_model=AudioPreviewResponse)
async def preview_audio(
    file: UploadFile = File(...),
    input_source: str = Form("upload"),
):
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


@router.post("/predict/compare", response_model=ModelCompareResponse)
async def compare_models(
    file: UploadFile = File(...),
    mode: str = Form("urban"),
):
    if mode not in {"urban", "animal", "auto"}:
        raise HTTPException(status_code=400, detail="Invalid mode.")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        return inference_service.compare_models(audio_bytes, mode=mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {exc}") from exc


@router.post("/predict/sample/compare", response_model=ModelCompareResponse)
async def compare_sample_models(
    domain: str = Form(...),
    sample_id: str = Form(...),
    mode: str = Form("urban"),
):
    if domain not in {"urban", "animal"}:
        raise HTTPException(status_code=400, detail="Invalid domain.")
    try:
        audio_path, _, _ = resolve_sample_audio(domain, sample_id)
        return inference_service.compare_models(audio_path.read_bytes(), mode=mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sample comparison failed: {exc}") from exc


@router.get("/predictions")
def prediction_history(
    x_session_id: str = Header(..., alias="X-Session-Id"),
    limit: int = 20,
):
    try:
        return fetch_recent_predictions(x_session_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase query failed: {exc}") from exc


@router.post("/predict", response_model=PredictResponse)
async def predict_audio(
    file: UploadFile = File(...),
    mode: str = Form("urban"),
    model_name: str = Form("mobilenetv2"),
    input_source: str = Form("upload"),
    gradcam: bool = Form(True),
    save_to_db: bool = Form(True),
    x_session_id: str = Header(default="anonymous", alias="X-Session-Id"),
):
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


@router.post("/reports/export")
async def export_prediction_report(payload: PredictResponse):
    """Download ZIP report with summary JSON/CSV and PNG artefacts."""
    try:
        zip_bytes = build_prediction_report_zip(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report export failed: {exc}") from exc
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="sound_analysis_report.zip"'},
    )
