# Environmental Sound Analytics Platform

**Module:** B9AI104 Deep Learning — CA1  
**Student:** Nadeesha Jayasuriya (20093736)  
**Project:** Mel-spectrogram CNN classification for urban and animal environmental sounds

Production web application for uploading or recording audio, running trained PyTorch models, viewing explainability outputs, logging predictions to Supabase, and monitoring sound events over time.

---

## Overview

This folder contains the **deployable web app** (React + FastAPI + Supabase). It sits inside the full CA1 repository and uses the parent project for:

- trained model checkpoints (`experiments/`)
- preprocessing and inference code (`src/`)
- shared settings (`config/config.yaml`)

| Layer | Technology | Folder |
|-------|------------|--------|
| Frontend | React, TypeScript, Tailwind, Vite | `frontend/` |
| Backend API | FastAPI, PyTorch | `backend/` |
| Database | Supabase (PostgreSQL + RLS) | `supabase/migrations/` |
| ML pipeline | Parent repo (`../`) | `src/`, `experiments/`, `config/` |

---

## Features

- **Urban / Animal / Smart Auto-Router** processing modes  
- **Scientific router panel** — dual-expert confidence, entropy, calibrated strength, routing reason  
- **Reliability warnings** — High / Medium / Low confidence messaging  
- **Unknown / uncertain** classification when top probability is below 40%  
- **Model selection** — Custom CNN, ResNet50, MobileNetV2 (urban)  
- **Grad-CAM** explainability overlays on Mel-spectrograms  
- **Live inference latency** + static benchmark comparison cards  
- **Supabase prediction history** with reliability and router metadata  
- **Analytics dashboards** — urban/animal event summaries, latency trends  
- **Exportable ZIP report** — JSON, CSV, waveform, Mel-spec, Grad-CAM PNGs  
- **Dataset sample testing** — run inference on curated UrbanSound8K / ESC-50 clips  

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

### CA1 Streamlit demo (optional)

From the **repository root**:

```powershell
python -m streamlit run sound_analytics_platform/streamlit/streamlit_app.py
```

---

## Prerequisites

- **Python 3.10+** with PyTorch (CUDA optional but recommended)
- **Node.js 18+** and npm
- **Supabase project** (schema applied — see below)
- Trained checkpoints present under `../experiments/` (from CA1 training runs)

---

## Supabase setup

1. Open [Supabase Dashboard](https://supabase.com/dashboard) → **SQL Editor**
2. Run migrations **in order**:
   - `supabase/migrations/001_initial_schema.sql`
   - `supabase/migrations/002_prediction_metadata.sql`
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

### Option A — start script (recommended on Windows)

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
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:8000/api/health | Health check |

**Important:** start the frontend with `npm run dev` only. Do not pass host/port as bare arguments to Vite (that breaks the dev server).

### Verify backend

```powershell
curl http://localhost:8000/api/health
```

Expected: `"status":"ok"` and `"supabase_configured":true`.

---

## Using the application

1. **Analyze Live** — upload a `.wav` file or record ~4 seconds from the microphone  
2. Review validation preview (sample rate, duration, Mel normalization)  
3. **Run Analysis** — view prediction, reliability, Grad-CAM, and benchmarks  
4. **History** — past predictions from Supabase (per browser session)  
5. **Analytics** — urban/animal event counts and latency trends  
6. **Export Report** — download ZIP with summary + images  

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

Full interactive docs: **http://localhost:8000/docs**

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

## Related deliverables

| Item | Location |
|------|----------|
| CA1 Word report | Moodle submission |
| CA1 Streamlit demo | `streamlit/streamlit_app.py` (run from repo root — see below) |
| Training / evaluation code | `../scripts/`, `../src/` |
| User guide (extended) | `USER_GUIDE.md` (if present) |

---

## Author

**Nadeesha Jayasuriya** — B9AI104 Deep Learning CA1  
Environmental Sound Classification via Mel-Spectrogram Images
