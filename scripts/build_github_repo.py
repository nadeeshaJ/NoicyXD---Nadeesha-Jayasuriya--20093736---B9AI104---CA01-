"""Sync CA1 project code + result outputs into the GitHub folder.

Run from project root:
    python scripts/build_github_repo.py

Copies:
  - Source code, config, scripts, notebooks, Streamlit app
  - outputs/figures  (report plots)
  - outputs/metrics  (JSON/CSV summaries + experiment metrics)
  - data/splits      (train/val/test metadata only)
  - experiments/**   (JSON metrics only — not .pt checkpoints)

Does NOT copy:
  - Raw audio (data/raw/)
  - Processed Mel-spec PNGs (data/processed/)
  - Model checkpoints (*.pt)
  - __pycache__, .env, Word documents
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITHUB_ROOT = PROJECT_ROOT.parent / "gitHub"

PIPELINE_SCRIPTS = [
    "setup_datasets.py",
    "run_step1_eda.py",
    "run_step2_preprocess.py",
    "run_step3_train.py",
    "run_step4_esc50.py",
    "run_step6_error_analysis.py",
    "run_ca1_ablation_studies.py",
    "verify_step5_deployment.py",
    "capture_app_demo.py",
    "build_github_repo.py",
    "build_submission_package.py",
]

REPORT_SCRIPTS = [
    "generate_model_summaries.py",
    "generate_final_report.py",
    "generate_phase9_presentation.py",
    "generate_ca1_checklist.py",
    "generate_step1_report.py",
    "generate_step2_report.py",
    "generate_step3_report.py",
    "generate_step4_report.py",
    "generate_step5_report.py",
    "generate_step6_report.py",
    "build_code_submission.py",
    "_submission_meta.py",
]

REPORT_JSON = [
    "reports/step1/step1_summary.json",
    "reports/step1/dataset_inventory.csv",
    "reports/step2/step2_summary.json",
    "reports/step2/step2_validation.json",
    "reports/step2/preprocessing_validation.csv",
    "reports/step3/step3_summary.json",
    "reports/step3/model_specifications.json",
    "reports/step3/model_comparison_urbansound8k.csv",
    "reports/step4/step4_summary.json",
    "reports/step4/esc50_model_comparison.csv",
    "reports/step4/cross_domain_comparison.csv",
    "reports/step5/step5_verification.json",
    "reports/step6/step6_summary.json",
    "reports/step6/inference_benchmarks.json",
    "reports/step6/cross_domain_summary.csv",
    "reports/final/ca1_ablation_summary.json",
    "reports/final/final_report_manifest.json",
    "reports/final/phase9_manifest.json",
]

EXPERIMENT_METRICS = [
    "experiments/urbansound8k/custom_cnn/test_metrics.json",
    "experiments/urbansound8k/custom_cnn/training_summary.json",
    "experiments/urbansound8k/resnet50/test_metrics.json",
    "experiments/urbansound8k/resnet50/training_summary.json",
    "experiments/urbansound8k/mobilenetv2/test_metrics.json",
    "experiments/urbansound8k/mobilenetv2/training_summary.json",
    "experiments/esc50_animals/custom_cnn_from_scratch/test_metrics.json",
    "experiments/esc50_animals/custom_cnn_from_scratch/training_summary.json",
    "experiments/esc50_animals/mobilenetv2_urbansound_transfer/test_metrics.json",
    "experiments/esc50_animals/mobilenetv2_urbansound_transfer/training_summary.json",
    "experiments/esc50_animals/mobilenetv2_imagenet_only/test_metrics.json",
    "experiments/esc50_animals/mobilenetv2_imagenet_only/training_summary.json",
]

GITIGNORE = """# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# Secrets / local env
.env
.env.local

# Large data — download with scripts/setup_datasets.py
data/raw/**
!data/raw/.gitkeep
data/processed/**
!data/processed/.gitkeep

# Model weights — retrain with scripts/run_step3_train.py
experiments/**/*.pt
experiments/**/*.pth
experiments/**/*.ckpt

# IDE / OS
.idea/
.vscode/
.DS_Store
Thumbs.db

# Jupyter checkpoints
.ipynb_checkpoints/
"""


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree_py(src: Path, dst: Path) -> int:
    count = 0
    if not src.exists():
        return count
    for path in src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(src)
        if copy_file(path, dst / rel):
            count += 1
    return count


def copy_tree_pattern(src: Path, dst: Path, pattern: str) -> int:
    count = 0
    if not src.exists():
        return count
    for path in src.rglob(pattern):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = path.relative_to(src)
            if copy_file(path, dst / rel):
                count += 1
    return count


def remove_pycache(root: Path) -> int:
    removed = 0
    for cache in root.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
            removed += 1
    return removed


def write_github_readme(manifest: dict) -> None:
    text = f"""# Environmental Sound Classification — B9AI104 CA1

**Student:** Nadeesha Jayasuriya  
**Module:** B9AI104 Deep Learning  
**Updated:** {date.today().strftime("%d %B %Y")}

Mel-spectrogram image classification on **UrbanSound8K** (10 urban classes) and **ESC-50 Animals** (cross-domain transfer).

## What is in this repo

| Folder | Contents |
|--------|----------|
| `src/` | Core Python modules (preprocess, models, train, evaluate, Grad-CAM) |
| `scripts/` | Runnable pipeline + figure/report generators |
| `notebooks/` | EDA, preprocessing demo, results analysis, training notebook |
| `app/` | Streamlit deployment demo |
| `config/` | Shared `config.yaml` settings |
| `outputs/figures/` | Generated result plots and report diagrams ({manifest["figure_count"]} PNGs) |
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

## Moodle submission

Submit **Word documents separately** on Moodle (report, dev notes, cover sheet, slides).  
This GitHub repo holds **code + reproducible outputs** only.

---
*Synced from `noicy_XD` via `python scripts/build_github_repo.py`*
"""
    (GITHUB_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    GITHUB_ROOT.mkdir(parents=True, exist_ok=True)

    file_count = 0

    # Root files
    for name in ("requirements.txt",):
        if copy_file(PROJECT_ROOT / name, GITHUB_ROOT / name):
            file_count += 1

    (GITHUB_ROOT / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    file_count += 1

    # Code trees
    file_count += copy_tree_py(PROJECT_ROOT / "src", GITHUB_ROOT / "src")
    file_count += copy_tree_py(PROJECT_ROOT / "app", GITHUB_ROOT / "app")
    file_count += copy_tree_pattern(PROJECT_ROOT / "notebooks", GITHUB_ROOT / "notebooks", "*.ipynb")
    file_count += copy_tree_pattern(PROJECT_ROOT / "config", GITHUB_ROOT / "config", "*")

    scripts_dst = GITHUB_ROOT / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    for name in PIPELINE_SCRIPTS + REPORT_SCRIPTS:
        if copy_file(PROJECT_ROOT / "scripts" / name, scripts_dst / name):
            file_count += 1

    # Figures
    figure_count = copy_tree_pattern(
        PROJECT_ROOT / "reports" / "figures",
        GITHUB_ROOT / "outputs" / "figures",
        "*.png",
    )
    file_count += figure_count

    # Metrics
    metrics_dst = GITHUB_ROOT / "outputs" / "metrics"
    for rel in REPORT_JSON:
        if copy_file(PROJECT_ROOT / rel, metrics_dst / Path(rel).name):
            file_count += 1

    for rel in EXPERIMENT_METRICS:
        flat_name = rel.replace("/", "__").replace("\\", "__")
        if copy_file(PROJECT_ROOT / rel, metrics_dst / flat_name):
            file_count += 1

    # Experiment JSON (structured paths for clarity)
    exp_count = 0
    for rel in EXPERIMENT_METRICS:
        if copy_file(PROJECT_ROOT / rel, GITHUB_ROOT / rel):
            exp_count += 1
            file_count += 1

    # Data splits only
    split_count = copy_tree_pattern(
        PROJECT_ROOT / "data" / "splits",
        GITHUB_ROOT / "data" / "splits",
        "*",
    )
    file_count += split_count

    # Placeholders for large folders
    for placeholder in (
        GITHUB_ROOT / "data" / "raw" / ".gitkeep",
        GITHUB_ROOT / "data" / "processed" / ".gitkeep",
    ):
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        if not placeholder.exists():
            placeholder.write_text("", encoding="utf-8")
            file_count += 1

    manifest = {
        "synced": date.today().isoformat(),
        "source": str(PROJECT_ROOT),
        "destination": str(GITHUB_ROOT),
        "file_count": file_count,
        "figure_count": figure_count,
        "experiment_json_files": exp_count,
        "split_files": split_count,
    }
    manifest_path = GITHUB_ROOT / "outputs" / "github_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    file_count += 1

    write_github_readme(manifest)
    file_count += 1

    cache_removed = remove_pycache(GITHUB_ROOT)

    print(f"GitHub folder synced: {GITHUB_ROOT}")
    print(f"  Files copied/updated: {file_count}")
    print(f"  Figures: {figure_count}")
    print(f"  Experiment JSON: {exp_count}")
    print(f"  Split metadata files: {split_count}")
    print(f"  __pycache__ dirs removed: {cache_removed}")
    print(f"  Manifest: {manifest_path}")
    print()
    print("Excluded: data/raw audio, data/processed PNGs, experiments/*.pt")


if __name__ == "__main__":
    main()
