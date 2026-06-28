# Environmental Sound Classification — B9AI104 CA1

**Student:** Nadeesha Jayasuriya (20093736)  
**Module:** B9AI104 Deep Learning

Mel-spectrogram CNN classification on **UrbanSound8K** and **ESC-50 Animals**.

---

## Repository layout

| Folder | Purpose |
|--------|---------|
| `src/` | ML core — preprocess, models, train, evaluate, router, Grad-CAM |
| `config/` | Shared `config.yaml` |
| `scripts/` | Training pipeline and report generators |
| `notebooks/` | EDA and analysis notebooks |
| `experiments/` | Model checkpoints (`.pt`) and JSON metrics |
| `reports/` | Generated figures and step summaries |
| `data/` | Splits metadata; raw/processed audio (not in Git) |
| **`sound_analytics_platform/`** | **All web app code** (React + FastAPI + Supabase + Streamlit) |

**Web app only lives in `sound_analytics_platform/`** — nothing duplicated at repo root.

---

## Key results (UrbanSound8K fold-10)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Custom CNN | 75.0% | 0.767 |
| ResNet50 | 81.2% | 0.811 |
| **MobileNetV2** | **82.7%** | **0.831** |

---

## ML pipeline (train & evaluate)

```bash
pip install -r requirements.txt
python scripts/setup_datasets.py
python scripts/run_step1_eda.py
python scripts/run_step2_preprocess.py
python scripts/run_step3_train.py
python scripts/run_step4_esc50.py
python scripts/run_step6_error_analysis.py
```

---

## Web applications

All UI/deployment code: **`sound_analytics_platform/`**

| App | How to run |
|-----|------------|
| **Production platform** (React + API + Supabase) | `cd sound_analytics_platform` → `powershell -ExecutionPolicy Bypass -File .\start_platform.ps1` |
| **CA1 Streamlit demo** | `python -m streamlit run sound_analytics_platform/streamlit/streamlit_app.py` |

See **`sound_analytics_platform/README.md`** for Supabase setup, env vars, and features.

---

## Not in Git (too large)

- `data/raw/` — download with `scripts/setup_datasets.py`
- `data/processed/` — regenerate with `run_step2_preprocess.py`
- `experiments/**/*.pt` — train with `run_step3_train.py`
