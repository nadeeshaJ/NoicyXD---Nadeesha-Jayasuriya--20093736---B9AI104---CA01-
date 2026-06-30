"""OpenAPI / Swagger metadata for the Sound Analytics API."""

from __future__ import annotations

API_DESCRIPTION = """
Environmental sound classification API for the **noicyXD** CA1 platform.

Audio clips are converted to 224×224 RGB Mel-spectrogram images and classified with trained
PyTorch CNNs (Custom CNN, ResNet50, MobileNetV2).

## Processing modes

| Mode | Value | Behaviour |
|------|-------|-----------|
| Urban Sound | `urban` | UrbanSound8K expert — 10 urban classes |
| Animal Vocalization | `animal` | ESC-50 animals expert — 10 animal classes |
| Smart Auto-Router | `auto` | Probes both experts, routes to stronger domain |

## Models

| Key | Urban | Animal | Notes |
|-----|-------|--------|-------|
| `mobilenetv2` | Yes | Yes | **Deployed** default (82.7% acc) |
| `resnet50` | Yes | No | Falls back to MobileNetV2 in animal mode |
| `custom_cnn` | Yes | No | From-scratch baseline |

## Authentication

No API key required for local development. Optional header:

- **`X-Session-Id`** — groups prediction history and analytics per browser session (default: `anonymous`)

## Interactive docs

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **OpenAPI JSON:** `/openapi.json`

## Typical workflow

1. `GET /api/health` — confirm checkpoints and device
2. `POST /api/audio/preview` — validate uploaded WAV before inference
3. `POST /api/predict` — classify with Grad-CAM and optional Supabase logging
4. `POST /api/reports/export` — download ZIP report from a prediction response
"""

OPENAPI_TAGS = [
    {
        "name": "System",
        "description": "Health checks, model benchmarks, and class label metadata.",
    },
    {
        "name": "Datasets",
        "description": "UrbanSound8K and ESC-50 test-split browsing, curated demo samples, and audio streaming.",
    },
    {
        "name": "Inference",
        "description": "Audio classification with optional Grad-CAM explainability and Smart Auto-Router.",
    },
    {
        "name": "Comparison",
        "description": "Run the same clip through all available models and compare latency and confidence.",
    },
    {
        "name": "Analytics",
        "description": "Session-scoped prediction history and dashboard telemetry (Supabase).",
    },
    {
        "name": "Reports",
        "description": "Export downloadable ZIP archives with JSON summary and PNG artefacts.",
    },
]

CONTACT = {
    "name": "Nadeesha Jayasuriya",
    "url": "https://github.com/NoicyXD",
}

LICENSE_INFO = {
    "name": "Academic — B9AI104 CA1",
}

SERVERS = [
    {"url": "http://localhost:8000", "description": "Local development"},
    {"url": "http://127.0.0.1:8000", "description": "Local development (loopback)"},
]
