# Environmental Sound Classification — B9AI104 CA1

**Student:** Nadeesha Jayasuriya  
**Module:** B9AI104 Deep Learning  
**Updated:** 28 June 2026

Mel-spectrogram image classification on **UrbanSound8K** (10 urban classes) and **ESC-50 Animals** (cross-domain transfer).

## What is in this repo

| Folder | Contents |
|--------|----------|
| `src/` | Core Python modules (preprocess, models, train, evaluate, Grad-CAM) |
| `scripts/` | Runnable pipeline + figure/report generators |
| `notebooks/` | EDA, preprocessing demo, results analysis, training notebook |
| `app/` | Streamlit deployment demo |
| `config/` | Shared `config.yaml` settings |
| `outputs/figures/` | Generated result plots and report diagrams (54 PNGs) |
| `outputs/metrics/` | JSON/CSV summaries and experiment metrics |
| `data/splits/` | Train/val/test split CSVs and class mappings |
| `experiments/` | Test metrics + training summaries (JSON only) |

## Not included (too large for GitHub)

- Raw audio datasets → run `python scripts/setup_datasets.py`
- Processed Mel-spec PNG images → run `python scripts/run_step2_preprocess.py`
- Model checkpoints (`.pt`) → run `python scripts/run_step3_train.py`

## Key results (UrbanSound8K fold-10 test)

| Model | Accuracy | Macro F1 | Train time |
|-------|----------|----------|------------|
| Custom CNN | 75.0% | 0.767 | 4252s |
| ResNet50 | 81.2% | 0.811 | 8293s |
| MobileNetV2 | 82.7% | 0.831 | 1974s |

**Deployed model:** MobileNetV2 (`experiments/urbansound8k/mobilenetv2/`)

## Quick start

```bash
pip install -r requirements.txt
python scripts/setup_datasets.py
python scripts/run_step1_eda.py
python scripts/run_step2_preprocess.py
python scripts/run_step3_train.py
python scripts/run_step4_esc50.py
python scripts/run_step6_error_analysis.py
python -m streamlit run app/streamlit_app.py
```


---
*Synced from `noicy_XD` via `python scripts/build_github_repo.py`*
