# Sound Analytics Platform — Current State

**Updated:** June 2026  
**Student:** Nadeesha Jayasuriya (20093736) · B9AI104 Deep Learning CA1

Snapshot of what the platform includes today. Setup: `README.md`. Usage: `USER_GUIDE.md`.

---

## 1. Summary

Web application for classifying environmental audio with trained CNN models on Mel-spectrogram images.
- Live upload and microphone recording
- Dataset browsing (UrbanSound8K + ESC-50 animals)
- Multi-model comparison and Grad-CAM
- Auto-routing between urban and animal experts
- Prediction logging, history, and analytics via Supabase
```
Browser (React + TypeScript)  →  FastAPI API  →  PyTorch (parent repo src/)
         │                              │
         └──────── Supabase ◄───────────┘
                  (predictions, benchmarks, classes)
```

| Layer | Stack | Location |
|-------|-------|----------|
| Frontend | React 18, TypeScript, Vite, Tailwind, Recharts | `frontend/` |
| Backend | FastAPI, Uvicorn, PyTorch | `backend/` |
| Database | Supabase PostgreSQL + RLS | `supabase/migrations/` |
| ML core | Training + inference pipeline | Parent repo `src/`, `config/`, `experiments/` |
| Alt UI | Streamlit demo | `streamlit/streamlit_app.py` |
---

## 2. UI structure (current)

### 2.1 Navigation tabs

| Tab | Purpose | Inference controls in header? |
|-----|---------|--------------------------------|
| **Analyze Live** | Upload WAV or record ~4s → preview → classify / compare | Yes — Processing Mode, Backend Model, Grad-CAM |
| **Project Datasets** | Browse test-split clips; analyze or compare per sample | Partial — Backend Model + Grad-CAM only (domain from dataset) |
| **Analytics Dashboard** | Session metrics and charts | No |
| **Prediction History** | Table of saved predictions for this browser session | No |
| **CNN Models** | Static offline benchmark cards (urban models) | No |

### 2.2 Header control card (context-sensitive)

Controls appear in the **top header**, not the sidebar.

| Control | Analyze Live | Project Datasets | Analytics / History / CNN Models |
|---------|:------------:|:----------------:|:--------------------------------:|
| Processing Mode | ✓ | ✗ | ✗ |
| Backend Model | ✓ | ✓ | ✗ |
| Grad-CAM | ✓ | ✓ | ✗ |

**Processing Mode options (Analyze Live only):**

| Mode | Behaviour |
|------|-----------|
| Urban Sound | UrbanSound8K expert (10 urban classes) |
| Animal Vocalization | ESC-50 animal expert (10 classes); Backend Model locked to MobileNetV2 |
| Smart Auto-Router | Picks urban or animal expert from dual-probe scores |
**Backend Model options:**

| Model | Urban | Animal |
|-------|:-----:|:------:|
| MobileNetV2 (Deployed) | ✓ | ✓ (only option) |
| ResNet50 | ✓ | ✗ |
| Custom CNN | ✓ | ✗ |

### 2.3 Global UI elements

- **Sidebar:** branding, tab navigation, system status (API online/offline, Supabase connected/local)
- **WaveLoader:** shown during inference, dataset analyze/compare, analytics load, history load, audio preview validation
- **Result modal:** full-screen overlay for classification report or multi-model comparison report
- **Error banner:** API / inference failures surfaced above main content

---

## 3. Feature breakdown by tab

### 3.1 Analyze Live

**Components:** `AudioInputPanel`, `AudioPreviewPanel`, `AnalysisResults`, `ModelComparisonPanel`

**Flow:**

1. User uploads `.wav` or records from microphone (WebM converted to WAV client-side via `lib/audio.ts`)
2. `POST /api/audio/preview` — validation (sample rate 22050, mono, ~4s, Mel normalization) + waveform/Mel PNGs
3. **Run Classifier** → `POST /api/predict` with `save_to_db=true` and `X-Session-Id` header
4. **Compare All Models** → `POST /api/predict/compare` (urban: all 3 models; auto: effective expert models)
5. Modal opens with full report

**Analysis report includes:**

- Top prediction + top-3 probability bars
- Reliability assessment (High / Medium / Low) and unknown/uncertain flag (< 40% threshold)
- Router explanation panel (auto mode)
- Waveform, Mel-spectrogram, Grad-CAM overlay images
- Model benchmark comparison cards
- **Play Sound** — replay uploaded/recording audio in browser
- **Export Report** — ZIP with JSON summary, CSV, PNG assets

### 3.2 Project Datasets

**Component:** `DatasetsPanel`

**Flow:**

1. Overview cards for UrbanSound8K and ESC-50 animals (clip counts, classes)
2. Switch domain (urban / animal); filter by class label
3. Grid or list view of test-split samples with **Listen**, **Analyze**, **Compare**
4. Inference uses **sample's domain** automatically (no Processing Mode selector)
5. WaveLoader shown while analyze/compare runs; panel hidden during load
6. Dataset sample audio playable via `GET /api/datasets/{domain}/samples/{id}/audio`
7. Analyze saves to Supabase with `input_source=dataset`; **Play Sound** in report uses dataset stream URL

**Ground truth:** sample predictions return `ground_truth_label` and `sample_id` for match/mismatch in results.
### 3.3 Analytics Dashboard

**Component:** `AnalyticsDashboardPanel`

**Data source:** `GET /api/analytics/dashboard` with `X-Session-Id` (backend aggregates from Supabase)

**Metrics:**

| Metric | Description |
|--------|-------------|
| Session Predictions | Total logged events for session |
| Average Latency | Mean `inference_ms` |
| Last Hour Activity | Predictions in trailing 60 minutes |
| Low Confidence Detections | Reliability Low or confidence < 40% |
| Out-of-Distribution Events | `is_unknown=true` |

**Charts:**

- Inference latency trend (last 30 runs)
- Urban monitoring events (siren, car horn, drilling, etc.)
- Animal monitoring events (dog, cow, rooster, etc.)
- Predictions by class, model, mode, input source

**UI:** loader on load, empty state, Sync Logs button, error message with retry.
### 3.4 Prediction History

**Component:** `PredictionHistoryPanel`

**Data source:** `GET /api/predictions?limit=50` with `X-Session-Id`

**Table columns:** Timestamp, Source, Mode, Model, Top Guess, Confidence, Reliability, Inference ms

**Behaviour:**

- Always loads via backend API (consistent with how predictions are saved)
- Auto-refreshes when a new `saved_prediction_id` is returned from analyze
- Derives reliability from confidence when DB field is null (matches `config.yaml` thresholds)
- WaveLoader, empty state, error + retry

### 3.5 CNN Models

**Location:** inline in `App.tsx` (benchmark cards)

**Purpose:** Offline benchmark table for the three urban models from Step 4 training. Does not run inference.
| Model | Test accuracy | Macro F1 | Deployed? |
|-------|---------------|----------|-----------|
| Custom CNN | 75.0% | 0.767 | No |
| ResNet50 | 81.2% | 0.811 | No |
| **MobileNetV2** | **82.7%** | **0.831** | **Yes (deployed)** |
Also shows mean latency and checkpoint file size. Data from Supabase `model_benchmarks` table (seeded in migration 001), with API fallback to `config.yaml` / `reports/step6/inference_benchmarks.json`.

**Note:** Animal ESC-50 MobileNetV2 is not listed here (urban benchmarks only). Animal expert is used at runtime in animal/auto modes.

---

## 4. Machine learning pipeline

### 4.1 Audio → image → CNN

Configured in parent `config/config.yaml`:

| Setting | Value |
|---------|-------|
| Sample rate | 22050 Hz |
| Duration | 4.0 s (pad/trim) |
| Channels | Mono |
| Mel bins | 128 |
| STFT | n_fft=2048, hop=512 |
| CNN input | 224×224 RGB (Mel replicated to 3 channels) |

### 4.2 Deployed checkpoints

Installed via `python scripts/setup_checkpoints.py` from repo root (not in Git).

| Domain | Model | Checkpoint path |
|--------|-------|-----------------|
| Urban | MobileNetV2 (default) | `experiments/urbansound8k/mobilenetv2/best_model.pt` |
| Urban | ResNet50 | `experiments/urbansound8k/resnet50/best_model.pt` |
| Urban | Custom CNN | `experiments/urbansound8k/custom_cnn/best_model.pt` |
| Animal | MobileNetV2 | `experiments/esc50_animals/mobilenetv2_imagenet_only/best_model.pt` |

### 4.3 Class vocabularies

**Urban (10):** air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music

**Animal (10):** dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow

### 4.4 Confidence & routing

From `src/confidence.py` and `src/domain_router.py`:

| Setting | Default |
|---------|---------|
| Unknown threshold | top confidence < 0.40 |
| High reliability | ≥ 0.70 |
| Medium reliability | ≥ 0.40 |
| Auto-router | Dual-expert probe; routes to urban or animal MobileNetV2 |

### 4.5 Grad-CAM

Generated server-side when `gradcam=true`. Overlays CNN attention on Mel-spectrogram; returned as base64 PNG in API response.

---

## 5. Backend API reference

**Base URL:** `http://localhost:8000/api`  
**Interactive docs:** `/docs` (Swagger), `/redoc` (ReDoc)

**Session header:** `X-Session-Id` — browser UUID from `localStorage` (`sound-analytics-session-id`). Groups history and analytics per browser.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API status, device, checkpoint verification, Supabase flag |
| GET | `/models` | Model benchmark table |
| GET | `/classes/{domain}` | Class registry (`urban` \| `animal`) |
| GET | `/datasets` | Dataset overview cards |
| GET | `/datasets/{domain}/samples` | Test-split sample list (`label`, `limit` query params) |
| GET | `/datasets/{domain}/curated` | Curated demo samples |
| GET | `/datasets/{domain}/samples/{sample_id}/audio` | Stream WAV for playback |
| POST | `/audio/preview` | Validate audio + waveform/Mel PNGs (no inference) |
| POST | `/predict` | Upload audio → inference + optional DB save |
| POST | `/predict/compare` | Upload audio → all models compared |
| POST | `/predict/sample` | Dataset sample inference + ground truth |
| POST | `/predict/sample/compare` | Dataset sample multi-model compare |
| GET | `/predictions` | Session prediction history (`limit` param) |
| GET | `/analytics/dashboard` | Session metrics aggregated from prediction logs |
| POST | `/reports/export` | ZIP export from prediction JSON payload |

---

## 6. Supabase schema

**Migrations (run in order in SQL Editor):**

1. `001_initial_schema.sql` — tables, RLS, seed data
2. `002_prediction_metadata.sql` — reliability, unknown, display_label, router_metrics
3. `002_dataset_input_source.sql` — allow `input_source='dataset'`

| Table | Purpose |
|-------|---------|
| `predictions` | Per-inference logs (session, mode, model, label, confidence, latency, assessment) |
| `model_benchmarks` | Static urban model evaluation stats |
| `sound_classes` | Urban + animal class registry |
| `profiles` | Optional auth extension |
| Storage `audio-clips` | Optional audio retention bucket |

**Save behaviour:** Predictions insert via backend service role / anon key. If extended metadata columns are missing, backend falls back to base schema insert (`predictions_repo.py`).

---

## 7. Frontend component map

| Component | Role |
|-----------|------|
| `App.tsx` | Shell, routing, header controls, modals, benchmarks load |
| `AudioInputPanel.tsx` | File upload + microphone recording |
| `AudioPreviewPanel.tsx` | Preview validation UI, Run Classifier, Compare All |
| `DatasetsPanel.tsx` | Dataset browser, analyze/compare per sample |
| `AnalysisResults.tsx` | Full classification report modal content |
| `ModelComparisonPanel.tsx` | Multi-model comparison report |
| `RouterExplanationPanel.tsx` | Auto-router scores and routing reason |
| `AnalyticsDashboardPanel.tsx` | Analytics charts and summary cards |
| `PredictionHistoryPanel.tsx` | Session history table |
| `WaveLoader.tsx` | Animated loading state |
| `MetricCard.tsx` | Reusable KPI card |

**Libraries:** `lib/api.ts` (API client), `lib/session.ts` (session ID), `lib/supabase.ts` (optional direct Supabase for benchmarks), `lib/audio.ts` (recording → WAV)

---

## 8. Environment & running

### Prerequisites

- Python 3.10+ with PyTorch (CUDA optional)
- Node.js 18+
- Trained checkpoints in `../experiments/`
- Supabase project with migrations applied

### Quick start (Windows)

```powershell
# From repo root — install checkpoints first
python scripts/setup_checkpoints.py

# From sound_analytics_platform/
powershell -ExecutionPolicy Bypass -File .\start_platform.ps1
```

| URL | Purpose |
|-----|---------|
| http://localhost:5173 | Web app |
| http://localhost:8000/docs | API documentation |
| http://localhost:8000/api/health | Health check |

### Environment files

| File | Keys |
|------|------|
| `backend/.env` | `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `ML_PROJECT_ROOT`, `CORS_ORIGINS` |
| `frontend/.env` | `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`, `VITE_API_BASE_URL` |

See `.env.example` for templates.

### System verification

From repo root (backend must be running):

```powershell
python scripts/verify_system.py
```

Runs 31 automated checks across health, datasets, inference modes, Grad-CAM, compare, export, analytics, and history.

---

## 9. Data flow diagrams

### Live analyze

```
Upload/Record → preview API → validation UI
                    ↓
              predict API (+ X-Session-Id, save_to_db)
                    ↓
         PyTorch inference + Grad-CAM + assessment
                    ↓
         Supabase insert → modal report → optional ZIP export
```

### Dataset analyze

```
Pick sample → predict/sample API (domain from sample)
                    ↓
         Same pipeline + ground_truth_label
                    ↓
         Modal report + Play Sound via audio stream URL
```

### Analytics / history

```
Browser session ID → GET /predictions or /analytics/dashboard
                    ↓
         Backend queries Supabase predictions table
                    ↓
         Aggregated metrics or history table in UI
```

---

## 10. Change log

| Area | Change |
|------|--------|
| Checkpoints | `setup_checkpoints.py` copies 4 `.pt` files from experiments |
| Auto-router | Fixed `urban_probe` / `urban_score` serialization mismatch |
| Microphone | WebM recordings converted to WAV before upload |
| Datasets tab | Processing Mode hidden; loader on analyze/compare; domain from sample |
| Analytics | Fixed Supabase save failure; loader; empty and error states |
| History | Dedicated panel; API load path; reliability fallback; model column |
| Header | Controls hidden on analytics, history, and CNN Models tabs |
| Play Sound | Fixed blob URL lifecycle in analysis report |
| API | OpenAPI/Swagger descriptions added |
---

## 11. Known limitations (current)

| Limitation | Notes |
|------------|-------|
| Animal model selection | Only MobileNetV2 trained/deployed for animal domain |
| CNN Models tab | Urban benchmarks only; no animal expert card |
| Compare on datasets | Uses sample domain; does not save comparison to history |
| Session scoped | History/analytics per browser `localStorage` session ID |
| Audio format | Live input expects WAV-compatible pipeline (mic converted client-side) |
| No user auth | Optional Supabase auth schema exists; app uses anonymous sessions |
| Streamlit | Separate demo app; React UI is the main interface |

---

## 12. Repository layout
```
sound_analytics_platform/
├── PLATFORM_CURRENT_STATE.md    ← this document
├── README.md                    ← setup & run
├── USER_GUIDE.md                ← usage walkthrough
├── start_platform.ps1
├── .env.example
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   ├── openapi.py
│   │   └── services/
│   │       ├── inference.py
│   │       ├── analytics.py
│   │       ├── datasets.py
│   │       ├── predictions_repo.py
│   │       ├── prediction_payload.py
│   │       └── export_report.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── components/
│       └── lib/
├── supabase/migrations/
└── streamlit/streamlit_app.py

../  (parent CA1 repo)
├── src/                   predict.py, train.py, models/, gradcam, domain_router
├── config/config.yaml
├── experiments/           best_model.pt checkpoints
├── data/                  raw audio + splits
├── reports/               figures, benchmarks JSON
└── scripts/
    ├── setup_checkpoints.py
    └── verify_system.py
```

---

## 13. CA1 feature map

| Requirement | Implementation |
|-------------|----------------|
| Mel-spectrogram CNN classification | Parent `src/` pipeline |
| Multiple model comparison | Compare endpoints and UI |
| Explainability | Grad-CAM overlays |
| Deployment / inference | FastAPI + React |
| Urban + animal sounds | Dual experts and dataset tabs |
| Evaluation metrics | Benchmark cards and ZIP export |
| Logging and monitoring | Supabase predictions, history, analytics dashboard |