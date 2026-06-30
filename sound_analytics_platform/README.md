# Environmental Sound Analytics Platform

**Module:** B9AI104 Deep Learning — CA1  
**Student:** Nadeesha Jayasuriya (20093736)  
**Project:** Mel-spectrogram CNN classification for urban and animal environmental sounds

Web application for uploading or recording audio, running trained PyTorch models, viewing Grad-CAM outputs, logging predictions to Supabase, and viewing session history and analytics.

---

## Overview

This folder contains the **web app** (React + FastAPI + Supabase). It sits inside the full CA1 repository and uses the parent project for:

- trained model checkpoints (`experiments/`)
- preprocessing and inference code (`src/`)
- shared settings (`config/config.yaml`)

**Technical reference:** [`PLATFORM_CURRENT_STATE.md`](PLATFORM_CURRENT_STATE.md) · **Changes:** [`CHANGELOG.md`](CHANGELOG.md)

| Layer | Technology | Folder |
|-------|------------|--------|
| Frontend | React, TypeScript, Tailwind, Vite | `frontend/` |
| Backend API | FastAPI, PyTorch | `backend/` |
| Database | Supabase (PostgreSQL + RLS) | `supabase/migrations/` |
| ML pipeline | Parent repo (`../`) | `src/`, `experiments/`, `config/` |

---

## Features

- **Urban / Animal / Auto-Router** processing modes  
- **Showcase tab** — one-click curated demo scenarios  
- **Router panel** — dual-expert scores and routing reason (auto mode)  
- **Reliability labels** — High / Medium / Low  
- **Confidence calibration** — thresholds and entropy in analysis reports  
- **Comparison winner summary** — fastest, confidence, agreement on compare reports  
- **Unknown / uncertain** when top probability is below 40%  
- **Model selection** — Custom CNN, ResNet50, MobileNetV2 (urban)  
- **Grad-CAM** overlays on Mel-spectrograms  
- **Inference latency** and benchmark comparison in results  
- **Prediction history** in Supabase (per browser session)  
- **Analytics dashboard** — class/model distributions, latency trend  
- **CNN Models tab** — urban + animal benchmarks, deployment profiles  
- **ZIP report export** — JSON, CSV, waveform, Mel-spec, Grad-CAM PNGs  
- **Dataset samples** — inference on UrbanSound8K / ESC-50 test clips  
### Deployed models (UrbanSound8K fold-10)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Custom CNN | 75.0% | 0.767 |
| ResNet50 | 81.2% | 0.811 |
| **MobileNetV2** | **82.7%** | **0.831** |

MobileNetV2 is the default deployed model for both urban and cross-domain routing.

---

## Repository layout

```
sound_analytics_platform/
├── README.md
├── USER_GUIDE.md
├── PLATFORM_CURRENT_STATE.md
├── CHANGELOG.md
├── start_platform.ps1
├── backend/
├── frontend/
├── streamlit/streamlit_app.py
├── supabase/
```

**Parent project (required for inference):**

```
../
├── src/                      ← predict.py, train.py, models/, gradcam, router
├── config/config.yaml
├── experiments/              ← best_model.pt checkpoints
├── reports/                    ← figures + inference_benchmarks.json
└── scripts/                  ← training pipeline
```

### Streamlit demo

From the **repository root**:

```powershell
python -m streamlit run sound_analytics_platform/streamlit/streamlit_app.py
```

---

## Prerequisites

- **Python 3.10+** with PyTorch (CUDA optional but recommended)
- **Node.js 18+** and npm
- **Supabase project** (schema applied — see below)
- **Trained checkpoints** under `../experiments/` (see below — not in Git)

### Install trained weights (required)

After `git clone`, run from the **repository root**:

```powershell
python scripts/setup_checkpoints.py --source "D:\path\to\experiments"
python scripts/setup_checkpoints.py --verify-only
```

Without real `.pt` files, the API may load random/mock weights and predictions will be wrong (flat ~10% or stuck on one class). Expected sizes: MobileNetV2 ~8.8 MB, Custom CNN ~25 MB, ResNet50 ~90 MB.

---

## Supabase setup

1. Open [Supabase Dashboard](https://supabase.com/dashboard) → **SQL Editor**
2. Run migrations **in order**:
   - `supabase/migrations/001_initial_schema.sql`
   - `supabase/migrations/002_prediction_metadata.sql`
   - `supabase/migrations/002_dataset_input_source.sql`
3. Confirm tables exist: `predictions`, `model_benchmarks`, `sound_classes`
Project URL used in this submission: `https://fhpcrtnhqrmjsdcrpqzm.supabase.co`

---

## Environment configuration

### Backend — `backend/.env`

Copy from `.env.example` or set:

```env
SUPABASE_URL=https://fhpcrtnhqrmjsdcrpqzm.supabase.co
SUPABASE_ANON_KEY=your_publishable_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
ML_PROJECT_ROOT=../../
API_PORT=8000
CORS_ORIGINS=http://localhost:5173
```

`ML_PROJECT_ROOT=../../` points to the parent CA1 folder (`src/`, `experiments/`).

### Frontend — `frontend/.env`

```env
VITE_SUPABASE_URL=https://fhpcrtnhqrmjsdcrpqzm.supabase.co
VITE_SUPABASE_ANON_KEY=your_publishable_key
VITE_API_BASE_URL=http://localhost:8000
```

Never commit real keys to Git. Use `.env` locally only.

---

## Run locally

### Option A — start script (Windows)

From this folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_platform.ps1
```

This opens two terminals (API + UI) and launches **http://localhost:5173**.

### Option B — manual start

**Terminal 1 — API**

```powershell
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — UI**

```powershell
cd frontend
npm install
npm run dev
```

Then open:

| URL | Purpose |
|-----|---------|
| http://localhost:5173 | Web application |
| http://localhost:8000/docs | **Swagger UI** — interactive API docs |
| http://localhost:8000/redoc | **ReDoc** — readable API reference |
| http://localhost:8000/openapi.json | OpenAPI 3 schema (import into Postman) |
| http://localhost:8000/api/health | Health check |

**Important:** start the frontend with `npm run dev` only. Do not pass host/port as bare arguments to Vite (that breaks the dev server).

### Verify backend

```powershell
curl http://localhost:8000/api/health
```

Expected: `"status":"ok"` and `"supabase_configured":true`.

---

## Using the application

1. **Showcase** or **Analyze Live** — run a classification  
2. Review validation preview (upload) or scenario card (Showcase)  
3. **Run Analysis** — view prediction, calibration panel, Grad-CAM, benchmarks  
4. **Compare All Models** — winner summary + per-model table  
5. **History** — past predictions from Supabase (per browser session)  
6. **Analytics** — class counts and latency trends  
7. **Export Report** — download ZIP with summary + images  

---

## API endpoints (summary)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service and Supabase status |
| POST | `/api/predict` | Upload audio → inference + DB log |
| POST | `/api/predict/sample` | Inference on dataset sample |
| POST | `/api/reports/export` | Download analysis ZIP |
| GET | `/api/predictions` | Session prediction history |
| GET | `/api/analytics/dashboard` | Monitoring metrics |
| GET | `/api/models` | Model benchmark table |

Full interactive docs:

- **Swagger UI:** http://localhost:8000/docs — try endpoints in the browser
- **ReDoc:** http://localhost:8000/redoc — printable reference layout

---


## Production deployment (outline)

1. Apply Supabase migrations on your project  
2. Deploy FastAPI backend with access to parent `experiments/` checkpoints  
3. Build frontend: `cd frontend && npm run build`  
4. Serve `frontend/dist` via Nginx/Caddy; proxy `/api` → backend  
5. Set production env vars for Supabase URL, API URL, and CORS origins  

---

## Security notes

- **Anon/publishable key** — safe in the browser with Row Level Security enabled  
- **Service role key** — backend only; never expose in frontend code  
- Audio clips can optionally be stored in Supabase Storage bucket `audio-clips`  

---


## Author

**Nadeesha Jayasuriya** — B9AI104 Deep Learning CA1  
Environmental Sound Classification via Mel-Spectrogram Images
