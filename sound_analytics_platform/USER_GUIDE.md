# Sound Analytics Platform — User Guide

Complete guide for the **Sound Analytics Platform** (`sound_analytics_platform/`).  
Setup: `README.md`. Technical reference: `PLATFORM_CURRENT_STATE.md`. Changes: `CHANGELOG.md`.

---

## Table of contents

1. [What this system does](#1-what-this-system-does)
2. [How it works (architecture)](#2-how-it-works-architecture)
3. [Getting started](#3-getting-started)
4. [Header controls](#4-header-controls)
5. [Tab: Analyze Live](#5-tab-analyze-live)
6. [Tab: Project Datasets](#6-tab-project-datasets)
7. [Tab: Showcase](#7-tab-showcase)
8. [Tab: Analytics](#8-tab-analytics)
9. [Tab: History](#9-tab-history)
10. [Tab: Models](#10-tab-models)
11. [Understanding results](#11-understanding-results)
12. [Processing modes explained](#12-processing-modes-explained)
13. [Supabase database](#13-supabase-database)
14. [Troubleshooting](#14-troubleshooting)
15. [Demo walkthrough](#15-demo-walkthrough)

---

## 1. What this system does

The platform classifies **environmental sounds** using deep learning. It supports two workflows:

| Workflow | What you do |
|----------|-------------|
| **Live analysis** | Upload a WAV file or record from your microphone |
| **Dataset analysis** | Pick clips from your project's UrbanSound8K or ESC-50 test data and compare predictions to known labels |

Both workflows run the same pipeline:

```
Audio → Mel-spectrogram image (224×224 RGB) → CNN → prediction + Grad-CAM → saved to Supabase
```

### Supported sound domains

| Domain | Dataset | Classes | Clips |
|--------|---------|---------|-------|
| **Urban** | UrbanSound8K | siren, car horn, dog bark, jackhammer, street music, etc. (10) | 8,732 |
| **Animal** | ESC-50 animals | dog, cow, rooster, frog, cat, crow, etc. (10) | 200 |

---

## 2. How it works (architecture)

```
Browser (React UI)  ──►  FastAPI backend  ──►  PyTorch models (parent project)
       │                        │
       └──── Supabase ◄─────────┘
              (predictions, benchmarks, classes)
```

| Part | URL | Role |
|------|-----|------|
| Frontend | http://localhost:5173 | Web interface |
| Backend API | http://localhost:8000 | Inference + charts + DB writes |
| Supabase | Cloud | Stores history and benchmark data |
| ML models | `../experiments/` | Trained CNN checkpoints |

The platform does **not** replace your CA1 training pipeline — it **uses** those trained models and datasets at runtime.

---

## 3. Getting started

### Prerequisites

- Python environment with PyTorch (parent project)
- Node.js (for frontend)
- Supabase migrations applied
- Trained model checkpoints in `experiments/`

### Start the system

**Terminal 1 — Backend:**
```powershell
cd sound_analytics_platform/backend
python run.py
```

**Terminal 2 — Frontend:**
```powershell
cd sound_analytics_platform/frontend
npm run dev
```

**Open:** http://localhost:5173

**Verify API:** http://localhost:8000/api/health

### Supabase migrations (one-time)

If not done yet, run all scripts in **Supabase Dashboard → SQL Editor** (in order):

1. `supabase/migrations/001_initial_schema.sql`
2. `supabase/migrations/002_prediction_metadata.sql`
3. `supabase/migrations/002_dataset_input_source.sql`

---

## 4. Header controls (context-sensitive)

These settings appear in the **top header** on inference tabs only.

| Tab | Processing Mode | Backend Model | Grad-CAM |
|-----|:---------------:|:-------------:|:--------:|
| Analyze Live | ✓ | ✓ | ✓ |
| Project Datasets | ✗ | ✓ | ✓ |
| Showcase / Analytics / History / CNN Models | ✗ | ✗ | ✗ |

| Control | Options | Purpose |
|---------|---------|---------|
| **Processing Mode** | Urban Sound / Animal Vocalization / Smart Auto-Router | Which expert model(s) to use *(Analyze Live only)* |
| **Backend Model** | MobileNetV2 / ResNet50 / Custom CNN | Which CNN runs inference |
| **Grad-CAM** | On / Off | Show visual explainability heatmap |
| **System Status** (sidebar) | Online/Offline, Supabase | Health check |

**Default settings:**
- Mode: Urban Sound or Smart Auto-Router
- Model: MobileNetV2 (Deployed)
- Grad-CAM: On

---

## 5. Tab: Analyze Live

Use this tab for **new audio** — files from your computer or live microphone.

### Step-by-step

1. Set mode and model in the header
2. Upload a WAV file or record 4 seconds from mic
3. **Validation preview appears automatically** with:
   - Pass/fail checklist (sample rate, mono, duration, normalization)
   - Raw waveform plot **before inference**
   - Mel-spectrogram preview
4. Click **Run Analysis** for full CNN inference + Grad-CAM
5. Or click **Compare All Models** to run Custom CNN, ResNet50, and MobileNetV2 on the **same clip**

### Tips

- Best input: **mono WAV**, ~4 seconds, clear single sound
- The pipeline auto-resamples to 22,050 Hz and pads/trims to 4 s
- Noisy mixtures are harder to classify correctly

---

## 6. Tab: Project Datasets

Use this tab to run inference on clips from the project test splits (UrbanSound8K and ESC-50 animals).

### What you see

**Dataset overview cards**
- UrbanSound8K: 8,732 clips, 10 classes, test fold 10
- ESC-50 Animals: 200 clips, 10 classes

**Class filter chips**
- Click a class name (e.g. `Siren`, `Cow`) to filter the sample table

**Recommended samples**
- Pre-selected clips from the test split
- **Analyze** runs inference on the selected row

**Browse test split samples**
- Table of real test clips with filename and ground-truth label
- Click **Analyze** on any row

### Ground truth comparison

After analyzing a dataset clip, the results show:

- **Known label** (dataset metadata)
- **Predicted label** (model output)
- **Match / mismatch** indicator

Example: Analyze a siren clip → ground truth `siren`, prediction `siren` at 94% → **correct**.

### When to use each tab

| Situation | Tab |
|-----------|-----|
| Clip with known ground-truth label | Project Datasets |
| Your own recording or upload | Analyze Live |
| Test auto-routing on fixed samples | Project Datasets + Smart Auto-Router |

---

## 7. Tab: Showcase

One-click runs on curated test clips. No header controls on this tab — each scenario sets mode and model internally (MobileNetV2).

| Scenario | What it tests |
|----------|----------------|
| Urban siren | Urban mode on a siren clip |
| Construction noise | Urban mode on jackhammer |
| Animal dog bark | Animal mode on ESC-50 dog |
| Auto-router · urban dog bark | Auto mode routes urban `dog_bark` |
| Auto-router · animal dog | Auto mode routes animal `dog` |

Click **Run scenario** → full analysis report modal opens (same as Analyze / Datasets).

---

## 8. Tab: Analytics

Session metrics from Supabase prediction logs.

### Metrics shown

| Metric | Description |
|--------|-------------|
| Total predictions | Count for your browser session |
| Avg latency | Mean inference time (ms) |
| Last hour activity | Predictions in the last 60 minutes |

### Charts

- **Latency trend** — line chart of inference times over recent runs
- **Predictions by class** — bar chart of top labels
- **Predictions by model** — which CNN was used most
- **Predictions by mode** — urban / animal / auto split
- **Predictions by input source** — upload / microphone / dataset

Click **Refresh** after running new analyses to update charts.

---

## 9. Tab: History

Predictions saved to Supabase for the current browser session.

Loads via the backend API. Shows a loader while fetching; empty state if nothing is logged yet.

| Column | Meaning |
|--------|---------|
| Time | When analysis ran |
| Source | `upload`, `microphone`, or `dataset` |
| Mode | urban / animal / auto |
| Model | CNN used |
| Top Guess | Predicted class |
| Confidence | Model probability |
| Reliability | High / Medium / Low |
| Inference | Latency in ms |

Click **Refresh Logs** after new analyses.

Session ID is stored in browser `localStorage`. Clearing site data starts a new session.

---

## 10. Tab: Models

Benchmark stats from training/evaluation. Read-only — does not run inference.

**Urban (UrbanSound8K fold-10):** Custom CNN, ResNet50, MobileNetV2 — accuracy, F1, latency, checkpoint size.

**Animal (ESC-50):** MobileNetV2 animal expert (~60% accuracy, 0.607 macro F1).

**Deployment profiles:** how each architecture maps to mobile edge, GPU server, or baseline reference use.

---

## 11. Understanding results

After any analysis (live or dataset), you see:

### Metric cards (top row)

| Card | Meaning |
|------|---------|
| Top Prediction | Winning class + confidence % |
| Live Latency | Actual inference time (ms) vs benchmark |
| Model Size | Checkpoint file size |
| Saved To DB | Whether result was stored in Supabase |

### Visual panels (three columns)

| Panel | What it shows |
|-------|---------------|
| **Raw Waveform** | Amplitude over time |
| **Mel-Spectrogram** | Frequency content (what the model "sees" before RGB conversion) |
| **Grad-CAM Overlay** | Where the CNN focused (warm = important regions) |

### Ground truth box (dataset only)

Shows known label vs prediction and whether they match.

### Confidence calibration

Bar showing top-1 confidence with 40% (unknown) and 70% (high reliability) markers. Also shows top-1 vs top-2 gap, normalized entropy, and reliability label.

### Smart Auto-Router box (auto mode only)

Shows urban vs animal probe scores and routing reason.

### Efficiency vs accuracy

Side-by-side benchmark cards for all three urban models.

### Top 3 predictions

Progress bars for the three highest-confidence classes.

### All class probabilities

Full softmax output for every class.

### Same-clip multi-model comparison

**Winner summary** (four cards): fastest model, highest confidence, label agreement %, suggested pick.

Table below compares all models on one clip:
- Prediction and confidence per model
- Live latency vs benchmark latency
- Checkpoint size

Available via **Compare All Models** (Analyze Live) or **Compare** (Project Datasets).

### Help banner

Dismissible bar under the header with a short tab map. Reappears only if you clear site data (`sap-help-dismissed` in localStorage).

---

## 12. Processing modes explained

### Urban Sound
- Routes to UrbanSound8K-trained model
- 10 urban classes
- All 3 models available (Custom CNN, ResNet50, MobileNetV2)

### Animal Vocalization
- Routes to ESC-50 animal model
- 10 animal classes
- MobileNetV2 is the trained expert; others fall back if selected

### Smart Auto-Router
- Runs **both** MobileNetV2 experts as probes
- Scores which domain fits better (with class-aware logic)
- Automatically picks urban or animal expert
- Best for clips where domain is unknown (e.g. dog bark could be urban or animal)

---

## 13. Supabase database

**Project ID:** `fhpcrtnhqrmjsdcrpqzm`  
**URL:** https://fhpcrtnhqrmjsdcrpqzm.supabase.co

### Tables

| Table | Purpose |
|-------|---------|
| `predictions` | Every analysis — label, confidence, mode, model, latency |
| `model_benchmarks` | Static stats for Custom CNN, ResNet50, MobileNetV2 |
| `sound_classes` | Urban + animal class registry |
| `profiles` | User profiles (optional auth) |

### What gets saved per prediction

- Processing mode and routed domain
- Model used
- Input source (upload / microphone / dataset)
- Top label and full probability JSON
- Inference latency
- Router reason (if auto mode)
- Timestamp

View data in **Supabase Dashboard → Table Editor → predictions**.

---

## 14. Troubleshooting

| Problem | Fix |
|---------|-----|
| API Offline | Start backend: `python run.py` in `backend/` |
| Blank page | Start frontend: `npm run dev` in `frontend/` |
| Empty History | Run a prediction; check Supabase migration applied |
| Dataset tab empty | Ensure `data/splits/*/test_processed.csv` exists in parent project |
| Missing checkpoint error | Verify `experiments/urbansound8k/mobilenetv2/best_model.pt` exists |
| Mic not working | Allow microphone in browser permissions |
| Port 8000 in use | Close old backend terminal or change `API_PORT` in `.env` |
| DB save fails on dataset | Run migration `002_dataset_input_source.sql` |

---

## 15. Demo walkthrough

Example order for a short walkthrough:

1. Dismiss or read the help banner under the header
2. **Showcase** → Run **Urban siren** scenario
3. In the report: confidence calibration, Mel-spec, Grad-CAM
4. **Project Datasets** → analyze a clip; check ground truth
5. **Analyze Live** → Compare All Models → winner summary cards
6. **CNN Models** → urban + animal benchmarks and deployment profiles
7. **History** and **Analytics** after a few runs

---

## Quick reference

| Action | Where |
|--------|-------|
| One-click demo | Showcase tab |
| Upload new audio | Analyze Live |
| Pre-inference validation | Automatic after upload/record |
| Compare all 3 models on same clip | Analyze Live → Compare All Models |
| Test on project data | Project Datasets |
| Session charts | Analytics |
| Compare models on dataset clip | Project Datasets → Compare |
| View past runs | History |
| Model benchmarks | CNN Models |
| Change CNN | Header → Backend Model (Analyze or Datasets tab) |
| Grad-CAM | Header checkbox (Analyze or Datasets tab) |
| API docs | http://localhost:8000/docs |

---

*B9AI104 Deep Learning CA1 — `sound_analytics_platform/`*
