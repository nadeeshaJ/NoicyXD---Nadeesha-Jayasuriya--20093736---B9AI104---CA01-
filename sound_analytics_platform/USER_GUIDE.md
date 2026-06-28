# Sound Analytics Platform — User Guide

Complete guide for using the **Sound Analytics Platform** (`sound_analytics_platform/`).  
For setup and deployment, see `README.md`.

---

## Table of contents

1. [What this system does](#1-what-this-system-does)
2. [How it works (architecture)](#2-how-it-works-architecture)
3. [Getting started](#3-getting-started)
4. [Sidebar controls](#4-sidebar-controls)
5. [Tab: Analyze Live](#5-tab-analyze-live)
6. [Tab: Project Datasets](#6-tab-project-datasets)
7. [Tab: Analytics](#7-tab-analytics)
8. [Tab: History](#8-tab-history)
9. [Tab: Models](#9-tab-models)
10. [Understanding results](#10-understanding-results)
11. [Processing modes explained](#11-processing-modes-explained)
12. [Supabase database](#12-supabase-database)
13. [Troubleshooting](#13-troubleshooting)
14. [Presentation demo script](#14-presentation-demo-script)

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
| ML models | `noicy_XD/experiments/` | Trained CNN checkpoints |

The platform does **not** replace your CA1 training pipeline — it **uses** those trained models and datasets at runtime.

---

## 3. Getting started

### Prerequisites

- Python environment with PyTorch (parent project)
- Node.js (for frontend)
- Supabase migrations applied (you've done this)
- Trained model checkpoints in `experiments/`

### Start the system

**Terminal 1 — Backend:**
```powershell
cd "d:\DBS - Sem 2\Deep Learning\CA01\noicy_XD\sound_analytics_platform\backend"
python run.py
```

**Terminal 2 — Frontend:**
```powershell
cd "d:\DBS - Sem 2\Deep Learning\CA01\noicy_XD\sound_analytics_platform\frontend"
npm run dev
```

**Open:** http://localhost:5173

**Verify API:** http://localhost:8000/api/health

### Supabase migrations (one-time)

If not done yet, run both scripts in **Supabase Dashboard → SQL Editor**:

1. `supabase/migrations/001_initial_schema.sql`
2. `supabase/migrations/002_dataset_input_source.sql`

---

## 4. Sidebar controls

These settings apply to **both** live analysis and dataset analysis.

| Control | Options | Purpose |
|---------|---------|---------|
| **Processing Mode** | Urban Sound / Animal Vocalization / Smart Auto-Router | Which expert model(s) to use |
| **Backend Model** | MobileNetV2 / ResNet50 / Custom CNN | Which CNN runs inference |
| **Grad-CAM** | On / Off | Show visual explainability heatmap |
| **System Status** | Online/Offline, Supabase | Health check |

**Recommended defaults for demos:**
- Mode: **Urban Sound** or **Smart Auto-Router**
- Model: **MobileNetV2 (Deployed)**
- Grad-CAM: **On**

---

## 5. Tab: Analyze Live

Use this tab for **new audio** — files from your computer or live microphone.

### Step-by-step

1. Select mode and model in the sidebar
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

Use this tab to analyze clips from your **existing CA1 datasets** — the same test splits used in evaluation.

### What you see

**Dataset overview cards**
- UrbanSound8K: 8,732 clips, 10 classes, test fold 10
- ESC-50 Animals: 200 clips, 10 classes

**Class filter chips**
- Click a class name (e.g. `Siren`, `Cow`) to filter the sample table

**Recommended demo samples**
- Pre-selected clips from your test split (siren, dog bark, cow, rooster, etc.)
- Click **Analyze sample** for a one-click demo

**Browse test split samples**
- Table of real test clips with filename and ground-truth label
- Click **Analyze** on any row

### Ground truth comparison

After analyzing a dataset clip, the results show:

- **Known label** (from dataset metadata)
- **Predicted label** (from model)
- **Match / mismatch** — useful for error analysis demos

Example: Analyze a siren clip → ground truth `siren`, prediction `siren` at 94% → **correct**.

### When to use this tab vs Analyze Live

| Situation | Use |
|-----------|-----|
| Presentation with known correct answers | **Project Datasets** |
| Testing your own recording | **Analyze Live** |
| Showing cross-domain routing (dog bark urban vs animal) | **Project Datasets** + Smart Auto-Router |
| Live audience interaction | **Analyze Live** (mic or upload) |

---

## 7. Tab: Analytics

Live **MLOps telemetry dashboard** powered by Supabase prediction logs.

### Metrics shown

| Metric | Description |
|--------|-------------|
| Total predictions | Count for your browser session |
| Avg latency | Mean inference time (ms) |
| Last hour activity | Recent utilization proxy |

### Charts

- **Latency trend** — line chart of inference times over recent runs
- **Predictions by class** — bar chart of top labels
- **Predictions by model** — which CNN was used most
- **Predictions by mode** — urban / animal / auto split
- **Predictions by input source** — upload / microphone / dataset

Click **Refresh** after running new analyses to update charts.

---

## 8. Tab: History

Shows all predictions saved to **Supabase** for your browser session.

| Column | Meaning |
|--------|---------|
| Time | When analysis ran |
| Source | `upload`, `microphone`, or `dataset` |
| Mode | urban / animal / auto |
| Prediction | Top predicted class |
| Confidence | Model certainty |

Click **Refresh** after new analyses.

Predictions are tied to a session ID stored in your browser (localStorage). Clearing browser data starts a new session.

---

## 9. Tab: Models

Compares all three CNN architectures using benchmark data from training/evaluation:

| Model | Urban accuracy | Checkpoint | Role |
|-------|----------------|------------|------|
| Custom CNN | ~75% | 25 MB | Baseline (from scratch) |
| ResNet50 | ~81% | 90 MB | Transfer learning |
| **MobileNetV2** | **~83%** | **9 MB** | **Deployed model** |

Each card shows accuracy, macro F1, latency, parameter count, and deployment badge.

Use this tab to explain **why MobileNetV2 was chosen** for production deployment.

---

## 10. Understanding results

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

### Smart Auto-Router box (auto mode only)

Shows urban vs animal probe scores and routing reason.

### Efficiency vs accuracy trade-off

Side-by-side comparison of all three models — supports deployment justification.

### Top 3 predictions

Progress bars for the three highest-confidence classes.

### All class probabilities

Full softmax output for every class — useful for showing uncertainty (e.g. dog bark vs gun shot confusion).

### Same-clip multi-model comparison

Table comparing all models on one audio clip:
- Prediction and confidence per model
- Live latency vs benchmark latency
- Checkpoint size

Available via **Compare All Models** (Analyze Live) or **Compare** (Project Datasets).

---

## 11. Processing modes explained

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

## 12. Supabase database

**Project ID:** `fhpcrtnhqrmjsdcrpqzm`  
**URL:** https://fhpcrtnhqrmjsdcrpqzm.supabase.co

### Tables

| Table | Purpose |
|-------|---------|
| `predictions` | Every analysis — label, confidence, mode, model, latency |
| `model_benchmarks` | Static stats for Custom CNN, ResNet50, MobileNetV2 |
| `sound_classes` | Urban + animal class registry |
| `profiles` | User profiles (for future auth) |

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

## 13. Troubleshooting

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

## 14. Presentation demo script

Suggested 5-minute live demo flow:

1. **Open app** → show sidebar (mode, model, Grad-CAM)
2. **Project Datasets tab** → pick UrbanSound8K → analyze **siren** demo sample
3. Point to **ground truth match**, **Mel-spec**, **Grad-CAM** heatmap
4. Switch to **Smart Auto-Router** → analyze **dog bark** (urban) vs **dog** (animal) to show routing
5. **Models tab** → explain MobileNetV2 deployment choice (83% acc, 9 MB)
6. **Analyze Live** → upload or record a clip from audience (optional)
7. **History tab** → show Supabase persistence

### Key sentences for Q&A

- *"We convert audio to Mel-spectrogram images so we can use standard CNN image classifiers."*
- *"Three models compared: Custom CNN baseline, ResNet50, MobileNetV2 — MobileNetV2 won on accuracy and efficiency."*
- *"Grad-CAM shows which frequency bands and time regions drove the prediction."*
- *"The Smart Auto-Router runs dual experts and picks urban vs animal automatically."*
- *"Every prediction is persisted to Supabase for audit and history."*

---

## Quick reference

| Action | Where |
|--------|-------|
| Upload new audio | Analyze Live tab |
| Pre-inference validation | Automatic after upload/record |
| Compare all 3 models on same clip | Analyze Live → Compare All Models |
| Test on project data | Project Datasets tab |
| Live telemetry charts | Analytics tab |
| Compare models on dataset clip | Project Datasets → Compare |
| View past runs | History tab |
| Change CNN | Sidebar → Backend Model |
| Enable XAI | Sidebar → Grad-CAM checkbox |
| API docs | http://localhost:8000/docs |

---

*Sound Analytics Platform — Deep Learning CA1 (B9AI104)*  
*Location: `noicy_XD/sound_analytics_platform/`*
