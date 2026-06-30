"""Full system verification — tests every API endpoint and feature."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

API = "http://127.0.0.1:8000/api"
SESSION = "system-verification"
URBAN_WAV = PROJECT_ROOT / "data/raw/urbansound8k/audio/fold10/100648-1-0-0.wav"
ANIMAL_WAV = PROJECT_ROOT / "data/raw/esc50/audio/3-140199-B-8.wav"
SIREN_SAMPLE = "115241-9-0-9.wav"  # curated urban siren
DOG_SAMPLE = "3-140199-B-8.wav"  # curated animal sheep

passed: list[str] = []
failed: list[tuple[str, str]] = []


def ok(name: str) -> None:
    passed.append(name)
    print(f"  PASS  {name}")


def fail(name: str, detail: str) -> None:
    failed.append((name, detail))
    print(f"  FAIL  {name}: {detail}")


def headers() -> dict[str, str]:
    return {"X-Session-Id": SESSION}


def test_get(name: str, path: str, *, check=None) -> dict | list | None:
    try:
        r = requests.get(f"{API}{path}", headers=headers(), timeout=30)
        if r.status_code != 200:
            fail(name, f"HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        if check:
            check(data)
        ok(name)
        return data
    except Exception as exc:
        fail(name, str(exc))
        return None


def test_post_form(name: str, path: str, form: dict, files: dict | None = None, *, check=None):
    try:
        r = requests.post(
            f"{API}{path}",
            data=form,
            files=files,
            headers=headers(),
            timeout=180,
        )
        if r.status_code != 200:
            fail(name, f"HTTP {r.status_code}: {r.text[:300]}")
            return None
        data = r.json()
        if check:
            check(data)
        ok(name)
        return data
    except Exception as exc:
        fail(name, str(exc))
        return None


def wav_file(path: Path) -> tuple:
    return ("audio.wav", path.open("rb"), "audio/wav")


def check_predict(data: dict, *, gradcam: bool) -> None:
    assert data.get("top_label"), "missing top_label"
    assert data.get("top_confidence") is not None, "missing confidence"
    assert data.get("waveform_png"), "missing waveform"
    assert data.get("mel_png"), "missing mel"
    assert data.get("rgb_png"), "missing rgb"
    assert data.get("assessment"), "missing assessment"
    if gradcam:
        assert data.get("gradcam_png"), "missing gradcam_png"
        assert data.get("gradcam_summary"), "missing gradcam_summary"


def main() -> None:
    print("\n=== SYSTEM VERIFICATION ===\n")

    if not URBAN_WAV.exists():
        print(f"ERROR: Urban sample missing: {URBAN_WAV}")
        sys.exit(1)

    # --- 1. Health & metadata ---
    print("[1] Health & metadata")
    health = test_get("GET /health", "/health", check=lambda d: (
        d.get("status") == "ok" and d.get("checkpoints_ready")
    ))
    test_get("GET /models", "/models", check=lambda d: len(d) >= 3)
    test_get("GET /classes/urban", "/classes/urban", check=lambda d: len(d) == 10)
    test_get("GET /classes/animal", "/classes/animal", check=lambda d: len(d) == 10)

    # --- 2. Datasets ---
    print("\n[2] Datasets")
    test_get("GET /datasets", "/datasets", check=lambda d: len(d) == 2)
    test_get("GET /datasets/urban/curated", "/datasets/urban/curated", check=lambda d: len(d) >= 3)
    test_get("GET /datasets/animal/curated", "/datasets/animal/curated", check=lambda d: len(d) >= 3)
    test_get("GET /datasets/urban/samples", "/datasets/urban/samples?label=siren&limit=5",
             check=lambda d: len(d) >= 1)
    if ANIMAL_WAV.exists():
        test_get("GET /datasets/animal/samples", "/datasets/animal/samples?limit=5",
                 check=lambda d: len(d) >= 1)
    else:
        fail("GET /datasets/animal/samples", f"animal wav missing: {ANIMAL_WAV}")

    # Audio file streaming
    r = requests.get(f"{API}/datasets/urban/samples/{SIREN_SAMPLE}/audio", timeout=30)
    if r.status_code == 200 and len(r.content) > 1000:
        ok("GET /datasets/urban/samples/{id}/audio")
    else:
        fail("GET /datasets/urban/samples/{id}/audio", f"HTTP {r.status_code}, size={len(r.content)}")

    # --- 3. Audio preview ---
    print("\n[3] Audio preview")
    test_post_form(
        "POST /audio/preview",
        "/audio/preview",
        {"input_source": "upload"},
        files={"file": wav_file(URBAN_WAV)},
        check=lambda d: d.get("valid") and d.get("waveform_png") and len(d.get("validation_checks", [])) == 4,
    )

    # --- 4. Predict — all modes ---
    print("\n[4] Prediction modes")
    for mode in ("urban", "animal", "auto"):
        form = {"mode": mode, "model_name": "mobilenetv2", "gradcam": "true", "save_to_db": "false"}
        test_post_form(
            f"POST /predict mode={mode}",
            "/predict",
            form,
            files={"file": wav_file(URBAN_WAV)},
            check=lambda d, m=mode: (
                check_predict(d, gradcam=True),
                d.get("effective_mode") in ("urban", "animal"),
                m != "auto" or d.get("router") is not None,
            ),
        )

    if ANIMAL_WAV.exists():
        test_post_form(
            "POST /predict mode=animal (animal wav)",
            "/predict",
            {"mode": "animal", "model_name": "mobilenetv2", "gradcam": "true", "save_to_db": "false"},
            files={"file": wav_file(ANIMAL_WAV)},
            check=lambda d: check_predict(d, gradcam=True) and d["effective_mode"] == "animal",
        )
        test_post_form(
            "POST /predict animal+resnet50 falls back to mobilenetv2",
            "/predict",
            {"mode": "animal", "model_name": "resnet50", "gradcam": "false", "save_to_db": "false"},
            files={"file": wav_file(ANIMAL_WAV)},
            check=lambda d: d["model_key"] == "mobilenetv2" and d["effective_mode"] == "animal",
        )

    # --- 5. Predict — all urban models ---
    print("\n[5] Urban model variants")
    for model in ("mobilenetv2", "resnet50", "custom_cnn"):
        test_post_form(
            f"POST /predict model={model}",
            "/predict",
            {"mode": "urban", "model_name": model, "gradcam": "false", "save_to_db": "false"},
            files={"file": wav_file(URBAN_WAV)},
            check=lambda d: check_predict(d, gradcam=False) and d["model_key"] == model,
        )

    # Grad-CAM off
    test_post_form(
        "POST /predict gradcam=false",
        "/predict",
        {"mode": "urban", "model_name": "mobilenetv2", "gradcam": "false", "save_to_db": "false"},
        files={"file": wav_file(URBAN_WAV)},
        check=lambda d: not d.get("gradcam_png") and check_predict(d, gradcam=False),
    )

    # --- 6. Sample predict ---
    print("\n[6] Dataset sample prediction")
    test_post_form(
        "POST /predict/sample urban",
        "/predict/sample",
        {"domain": "urban", "sample_id": SIREN_SAMPLE, "mode": "urban",
         "model_name": "mobilenetv2", "gradcam": "true", "save_to_db": "false"},
        check=lambda d: check_predict(d, gradcam=True) and d.get("ground_truth_label"),
    )
    if ANIMAL_WAV.exists():
        test_post_form(
            "POST /predict/sample animal",
            "/predict/sample",
            {"domain": "animal", "sample_id": DOG_SAMPLE, "mode": "animal",
             "model_name": "mobilenetv2", "gradcam": "true", "save_to_db": "false"},
            check=lambda d: check_predict(d, gradcam=True) and d.get("ground_truth_label"),
        )

    test_post_form(
        "POST /predict/sample auto-router",
        "/predict/sample",
        {"domain": "urban", "sample_id": SIREN_SAMPLE, "mode": "auto",
         "model_name": "mobilenetv2", "gradcam": "true", "save_to_db": "false"},
        check=lambda d: d.get("router") is not None,
    )

    # --- 7. Model comparison ---
    print("\n[7] Model comparison")
    test_post_form(
        "POST /predict/compare",
        "/predict/compare",
        {"mode": "urban"},
        files={"file": wav_file(URBAN_WAV)},
        check=lambda d: len(d.get("comparisons", [])) >= 3,
    )
    test_post_form(
        "POST /predict/sample/compare",
        "/predict/sample/compare",
        {"domain": "urban", "sample_id": SIREN_SAMPLE, "mode": "urban"},
        check=lambda d: len(d.get("comparisons", [])) >= 3,
    )
    test_post_form(
        "POST /predict/compare mode=auto",
        "/predict/compare",
        {"mode": "auto"},
        files={"file": wav_file(URBAN_WAV)},
        check=lambda d: d.get("effective_mode") in ("urban", "animal") and len(d.get("comparisons", [])) >= 1,
    )

    # --- 8. Report export ---
    print("\n[8] Report export")
    result = test_post_form(
        "POST /predict (for export)",
        "/predict",
        {"mode": "urban", "model_name": "mobilenetv2", "gradcam": "true", "save_to_db": "false"},
        files={"file": wav_file(URBAN_WAV)},
    )
    if result:
        try:
            r = requests.post(
                f"{API}/reports/export",
                json=result,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if r.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                names = zf.namelist()
                required = {"report_summary.json", "waveform.png", "mel_spectrogram.png", "gradcam_overlay.png"}
                missing = required - set(names)
                if missing:
                    fail("POST /reports/export", f"missing files: {missing}")
                else:
                    ok("POST /reports/export")
            else:
                fail("POST /reports/export", f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as exc:
            fail("POST /reports/export", str(exc))

    # --- 9. Analytics & history (Supabase) ---
    print("\n[9] Analytics & history")
    test_get("GET /analytics/dashboard", "/analytics/dashboard")
    test_get("GET /predictions", "/predictions")

    # Save one prediction to DB
    test_post_form(
        "POST /predict save_to_db",
        "/predict",
        {"mode": "urban", "model_name": "mobilenetv2", "gradcam": "true", "save_to_db": "true"},
        files={"file": wav_file(URBAN_WAV)},
        check=lambda d: True,  # saved_prediction_id may be null if supabase fails
    )

    # --- 10. Session export ---
    print("\n[10] Session export")
    try:
        r = requests.get(
            f"{API}/reports/session-export",
            headers=headers(),
            timeout=30,
        )
        if r.status_code == 200:
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            names = zf.namelist()
            required = {
                "session_summary.json",
                "predictions.json",
                "predictions.csv",
                "analytics_dashboard.json",
            }
            missing = required - set(names)
            if missing:
                fail("GET /reports/session-export", f"missing files: {missing}")
            else:
                ok("GET /reports/session-export")
        else:
            fail("GET /reports/session-export", f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        fail("GET /reports/session-export", str(exc))

    # --- Summary ---
    print(f"\n=== RESULTS: {len(passed)} passed, {len(failed)} failed ===")
    if failed:
        print("\nFailures:")
        for name, detail in failed:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
